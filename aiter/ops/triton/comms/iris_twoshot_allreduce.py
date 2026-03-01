# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Iris two-shot fused AllReduce + RMSNorm + per-tensor FP8 Quant.

Single persistent kernel:
  Step 1 (Two-shot AllReduce): Each rank reduces its assigned rows by
    gathering from all ranks, then broadcasts reduced rows to all peers.
    Reduce and broadcast are interleaved per-row (data stays in registers).
  Step 2 (Inlined device barrier): CTA 0 performs cross-rank synchronization
    via atomic ops on the symmetric heap (same protocol as iris device_barrier).
    Other CTAs spin on a local flag until CTA 0 signals completion.
  Step 3 (RMSNorm + amax): All CTAs process ALL rows (not just their assigned
    subset). Residual addition, RMSNorm, and per-row amax accumulation.
  Step 4 (Cross-CTA barrier): Spin-wait until all CTAs finish step 3.
  Step 5 (FP8 Quantize): Read global scale from amax, quantize all rows.

Compared to one-shot:
- Each rank only reads its 1/world_size subset of rows from all ranks
  (vs one-shot where every rank reads ALL rows from all ranks).
- Extra cost: broadcast phase (iris.store to all peers) + inlined barrier.

CUDA graph compatibility:
- Buffer pre-allocation with view pattern (same as oneshot)
- Device barriers are inlined (no separate kernel launch)
- Pre-barrier uses shmem.device_barrier() only during buffer allocation
"""

import os
from typing import Any, Optional, Tuple

import iris
import torch
import triton
import triton.language as tl

import logging

__all__ = ["fused_allreduce_add_rms_quant_gemm_iris_twoshot"]

logger = logging.getLogger(__name__)

AUTOTUNE = False
if AUTOTUNE:
    os.environ["TRITON_PRINT_AUTOTUNING"] = "1"


# ============================================================================
# Inlined from iris.ccl.utils and iris._distributed_helpers
# ============================================================================


@triton.jit()
def chiplet_transform_chunked(
        pid,
        num_workgroups: tl.constexpr,
        num_xcds: tl.constexpr,
        chunk_size: tl.constexpr,
    ):
        """Redistribute workgroups across XCDs in chunks."""
        if pid > (num_workgroups // (num_xcds * chunk_size)) * (
            num_xcds * chunk_size
        ):
            return pid

        local_pid = pid // num_xcds
        chunk_idx = local_pid // chunk_size
        pos_in_chunk = local_pid % chunk_size

        xcd = pid % num_xcds
        new_pid = (
            chunk_idx * num_xcds * chunk_size + xcd * chunk_size + pos_in_chunk
        )
        return new_pid


@triton.jit
def _translate_ptr(ptr, from_rank, to_rank, heap_bases):
    """Translate a pointer from one rank's address space to another's."""
    from_base = tl.load(heap_bases + from_rank)
    to_base = tl.load(heap_bases + to_rank)
    offset = tl.cast(ptr, tl.uint64) - from_base
    translated_ptr = tl.cast(
        tl.cast(to_base, tl.pointer_type(tl.int8)) + offset, ptr.dtype
    )
    return translated_ptr


@triton.jit
def _inlined_device_barrier(
    raw_pid,
    barrier_flags_ptr,
    barrier_epoch,
    barrier_done_ptr,
    heap_bases: tl.tensor,
    iris_rank: tl.constexpr,
    world_size: tl.constexpr,
    rank_start: tl.constexpr,
    rank_stride: tl.constexpr,
):
    """Cross-rank device barrier for use inside persistent kernels.

    Same protocol as iris._distributed_helpers._device_barrier_kernel:
    CTA 0 signals own readiness, polls remote ranks, then signals
    other CTAs via barrier_done_ptr.
    """
    target_epoch = barrier_epoch + 1

    if raw_pid == 0:
        own_flag_ptr = barrier_flags_ptr + iris_rank
        own_translated = _translate_ptr(
            own_flag_ptr, iris_rank, iris_rank, heap_bases
        )
        tl.atomic_xchg(own_translated, target_epoch, sem="release", scope="sys")

        for i in range(world_size):
            remote_rank = rank_start + i * rank_stride
            if remote_rank != iris_rank:
                remote_flag_ptr = barrier_flags_ptr + remote_rank
                remote_translated = _translate_ptr(
                    remote_flag_ptr, iris_rank, remote_rank, heap_bases
                )
                while (
                    tl.atomic_cas(
                        remote_translated,
                        target_epoch,
                        target_epoch,
                        sem="acquire",
                        scope="sys",
                    )
                    != target_epoch
                ):
                    pass

        tl.atomic_add(barrier_done_ptr, 1, sem="release")
    else:
        while tl.atomic_add(barrier_done_ptr, 0, sem="acquire") < 1:
            pass


def extract_group_info(
    shmem: Any,
) -> Tuple[int, int, int, int, int]:
    """Extract rank/group info from iris shmem context.

    Returns: (rank_in_group, rank_global, world_size, rank_start, rank_stride)
    """
    rank_global = shmem.get_rank()
    rank_in_group = rank_global
    world_size = shmem.get_num_ranks()
    rank_start = 0
    rank_stride = 1
    return rank_in_group, rank_global, world_size, rank_start, rank_stride


# ============================================================================
# Autotune
# ============================================================================

IRIS_TWOSHOT_AUTOTUNE_KEYS = [
    "M",
    "N",
    "HAS_RESIDUAL",
]


def _compute_chunk_size(comm_sms: int, num_xcds: int, swizzle_size: int = 4) -> int:
    chunk = swizzle_size * swizzle_size
    return min(chunk, comm_sms // num_xcds)


def get_iris_twoshot_configs(autotune: bool):
    num_xcds = iris.hip.get_num_xcc()

    if not autotune:
        comm_sms = 128
        return [
            triton.Config(
                {
                    "COMM_SMS": comm_sms,
                    "NUM_XCDS": num_xcds,
                    "CHUNK_SIZE": _compute_chunk_size(comm_sms, num_xcds),
                    "waves_per_eu": 4,
                },
                num_warps=16,
                num_stages=2,
            )
        ]

    configs = []
    COMM_SMS_OPTIONS = [32, 64, 128]
    NUM_WARPS_OPTIONS = [4, 8, 16]
    NUM_STAGES_OPTIONS = [1, 2]
    WAVES_PER_EU_OPTIONS = [1, 2, 4]
    for sms in COMM_SMS_OPTIONS:
        chunk = _compute_chunk_size(sms, num_xcds)
        for nw in NUM_WARPS_OPTIONS:
            for ns in NUM_STAGES_OPTIONS:
                for waves in WAVES_PER_EU_OPTIONS:
                    configs.append(
                        triton.Config(
                            {
                                "COMM_SMS": sms,
                                "NUM_XCDS": num_xcds,
                                "CHUNK_SIZE": chunk,
                                "waves_per_eu": waves,
                            },
                            num_stages=ns,
                            num_warps=nw,
                        )
                    )
    return configs


iris_twoshot_autotune_configs = get_iris_twoshot_configs(AUTOTUNE)


# ============================================================================
# Fused two-shot AllReduce + RMSNorm + per-tensor FP8 Quant kernel
# ============================================================================


@triton.autotune(
    configs=iris_twoshot_autotune_configs,
    key=IRIS_TWOSHOT_AUTOTUNE_KEYS,
    use_cuda_graph=True,
)
@triton.jit
def fused_twoshot_allreduce_rmsnorm_quant_kernel(
    # Input (in symmetric heap for iris.load)
    input_ptr,
    # Allreduce output (in symmetric heap for iris.store broadcast)
    allreduce_out_ptr,
    # Outputs (regular GPU memory)
    rms_out_ptr,
    quant_out_ptr,
    scale_out_ptr,
    # Cross-CTA sync for per-tensor scale
    global_amax_ptr,
    cta_arrival_ptr,
    # Inlined device barrier state (symmetric heap)
    barrier_flags_ptr,
    barrier_epoch,
    # Inlined device barrier: CTA 0 -> other CTAs signal (local GPU memory)
    barrier_done_ptr,
    # Residual (regular GPU memory; dummy ptr when HAS_RESIDUAL=False)
    residual_in_ptr,
    residual_out_ptr,
    # RMSNorm weight
    rms_weight_ptr,
    # Scalar params
    rms_eps,
    fp8_max,
    # Dimensions
    M,
    N,
    # Input strides (iris heap buffer)
    stride_in_m,
    stride_in_n,
    # Allreduce output strides (iris heap buffer)
    stride_ar_m,
    stride_ar_n,
    # Iris params
    heap_bases: tl.tensor,
    group_rank: tl.constexpr,
    iris_rank: tl.constexpr,
    world_size: tl.constexpr,
    rank_start: tl.constexpr,
    rank_stride: tl.constexpr,
    # Explicit kernel params
    BLOCK_SIZE_N: tl.constexpr,
    HAS_RESIDUAL: tl.constexpr,
    # Autotuned params
    COMM_SMS: tl.constexpr,
    NUM_XCDS: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
):
    """Fused two-shot AllReduce + RMSNorm + per-tensor FP8 quant.

    Single persistent kernel combining two-shot allreduce communication
    with an inlined cross-rank barrier and fused normalization/quantization.
    """
    raw_pid = tl.program_id(0)
    pid = raw_pid

    if NUM_XCDS != 1:
        pid = chiplet_transform_chunked(raw_pid, COMM_SMS, NUM_XCDS, CHUNK_SIZE)

    cols = tl.arange(0, BLOCK_SIZE_N)
    col_mask = cols < N

    # ================================================================
    # Step 1: Two-shot AllReduce (reduce assigned rows + broadcast)
    # ================================================================

    # Block distribution of rows across ranks
    rows_per_rank = tl.cdiv(M, world_size)
    my_start = group_rank * rows_per_rank
    my_end = tl.minimum(my_start + rows_per_rank, M)

    # Reduce + broadcast interleaved per row (like iris CCL two_shot)
    for row in range(my_start + pid, my_end, COMM_SMS):
        in_offset = row * stride_in_m + cols * stride_in_n
        ar_offset = row * stride_ar_m + cols * stride_ar_n

        # Gather from all ranks and reduce
        acc = tl.zeros((BLOCK_SIZE_N,), dtype=tl.float32)
        for i in tl.static_range(0, world_size):
            remote_rank = rank_start + i * rank_stride
            partial = iris.load(
                input_ptr + in_offset,
                iris_rank,
                remote_rank,
                heap_bases,
                mask=col_mask,
            )
            acc += partial.to(tl.float32)

        reduced = acc.to(allreduce_out_ptr.type.element_ty)

        # Store locally with write-through
        tl.store(
            allreduce_out_ptr + ar_offset,
            reduced,
            mask=col_mask,
        )

        # Broadcast to all peers
        for i in tl.static_range(0, world_size):
            remote_rank = rank_start + i * rank_stride
            if remote_rank != iris_rank:
                iris.store(
                    allreduce_out_ptr + ar_offset,
                    reduced,
                    iris_rank,
                    remote_rank,
                    heap_bases,
                    mask=col_mask,
                )

    # ================================================================
    # Step 2: Cross-rank device barrier
    # ================================================================

    _inlined_device_barrier(
        raw_pid,
        barrier_flags_ptr,
        barrier_epoch,
        barrier_done_ptr,
        heap_bases,
        iris_rank,
        world_size,
        rank_start,
        rank_stride,
    )

    # ================================================================
    # Step 3: Residual Add + RMSNorm + amax (all rows, all CTAs)
    # ================================================================

    rms_w = tl.load(
        rms_weight_ptr + cols, mask=col_mask, other=0.0
    ).to(tl.float32)

    for row in range(pid, M, COMM_SMS):
        ar_offset = row * stride_ar_m + cols * stride_ar_n
        out_offset = row * N + cols

        # Load allreduce result (now complete on all ranks)
        ar_val = tl.load(
            allreduce_out_ptr + ar_offset, mask=col_mask, other=0.0
        ).to(tl.float32)

        # Optional residual addition
        if HAS_RESIDUAL:
            res_in = tl.load(
                residual_in_ptr + out_offset, mask=col_mask, other=0.0
            ).to(tl.float32)
            rms_in = ar_val + res_in
            tl.store(
                residual_out_ptr + out_offset,
                rms_in.to(residual_out_ptr.type.element_ty),
                mask=col_mask,
            )
        else:
            rms_in = ar_val

        # RMSNorm
        sq_sum = tl.sum(rms_in * rms_in, axis=0)
        variance = sq_sum / N
        rrms = tl.rsqrt(variance + rms_eps)
        normed = rms_in * rrms * rms_w

        tl.store(
            rms_out_ptr + out_offset,
            normed.to(rms_out_ptr.type.element_ty),
            mask=col_mask,
        )

        # Per-row amax -> atomic max to global
        row_amax = tl.max(tl.abs(normed), axis=0)
        tl.atomic_max(global_amax_ptr, row_amax, sem="relaxed")

    # ================================================================
    # Step 4: Cross-CTA barrier (all CTAs finished RMSNorm + amax)
    # ================================================================

    tl.atomic_add(cta_arrival_ptr, 1, sem="release")
    while tl.atomic_add(cta_arrival_ptr, 0, sem="acquire") < COMM_SMS:
        pass

    # ================================================================
    # Step 5: Quantize with per-tensor scale
    # ================================================================

    global_amax = tl.load(global_amax_ptr)
    global_amax = tl.maximum(global_amax, 1e-12)
    scale = global_amax / fp8_max

    if pid == 0:
        tl.store(scale_out_ptr, scale)

    for row in range(pid, M, COMM_SMS):
        out_offset = row * N + cols

        normed = tl.load(
            rms_out_ptr + out_offset, mask=col_mask, other=0.0
        ).to(tl.float32)

        quantized = (normed / scale).to(quant_out_ptr.type.element_ty)
        tl.store(
            quant_out_ptr + out_offset, quantized, mask=col_mask
        )


# ============================================================================
# Manager and public API
# ============================================================================


class IrisTwoshotManager:
    """Singleton manager for two-shot AllReduce+RMSNorm+Quant."""

    _instance: Optional["IrisTwoshotManager"] = None
    _initialized: bool = False

    def __new__(cls) -> "IrisTwoshotManager":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if IrisTwoshotManager._initialized:
            return
        IrisTwoshotManager._initialized = True

        self._shmem: Any = None
        self._heap_size: int = 2**33  # 8GB default

        # Pre-allocated input buffer (iris symmetric heap)
        self._input_buf: Any = None
        self._input_M: int = 0
        self._input_N: int = 0
        self._input_dtype: Optional[torch.dtype] = None

        # Pre-allocated allreduce output buffer (iris symmetric heap for broadcast)
        self._ar_output_buf: Any = None
        self._ar_output_M: int = 0

        # Pre-allocated output buffers (regular GPU memory)
        self._out_rms: Optional[torch.Tensor] = None
        self._out_quant: Optional[torch.Tensor] = None
        self._out_scale: Optional[torch.Tensor] = None
        self._out_residual: Optional[torch.Tensor] = None
        self._out_quant_dtype: Optional[torch.dtype] = None

        # Cross-CTA sync buffers for per-tensor scale reduction
        self._global_amax: Optional[torch.Tensor] = None
        self._cta_arrival: Optional[torch.Tensor] = None

        # Inlined device barrier state
        self._barrier_flags: Any = None  # on symmetric heap
        self._barrier_epoch: int = 0
        self._barrier_done: Optional[torch.Tensor] = None  # local GPU memory

    def initialize(self, heap_size: Optional[int] = None) -> None:
        """Initialize Iris symmetric heap (call once at startup)."""
        if self._shmem is not None:
            logger.debug("Iris (twoshot) already initialized, skipping")
            return

        if heap_size is not None:
            self._heap_size = heap_size

        cur_rank = (
            torch.distributed.get_rank()
            if torch.distributed.is_initialized()
            else 0
        )
        logger.info(
            "Initializing Iris (twoshot) symmetric heap: "
            f"rank={cur_rank}, heap_size={self._heap_size / 2**30:.1f}GB"
        )

        self._shmem = iris.iris(self._heap_size)

        # Allocate barrier flags on symmetric heap (one int32 per rank)
        num_ranks = self._shmem.get_num_ranks()
        self._barrier_flags = self._shmem.zeros(
            (num_ranks,), dtype=torch.int32
        )
        self._shmem.device_barrier()

        logger.info(
            f"Iris (twoshot) initialized successfully on rank {cur_rank}"
        )

    @property
    def shmem(self) -> Any:
        """Get the Iris symmetric memory instance (auto-initializes)."""
        if self._shmem is None:
            self.initialize()
        return self._shmem

    def _get_input_buffer(
        self,
        M: int,
        N: int,
        dtype: torch.dtype,
    ) -> Any:
        """Get a view into the pre-allocated iris input buffer."""
        if (self._input_buf is not None
                and M <= self._input_M
                and N == self._input_N
                and dtype == self._input_dtype):
            return self._input_buf[:M]

        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                f"Iris (twoshot): input buffer too small for M={M} "
                f"(allocated {self._input_M}). Cannot allocate during "
                f"CUDA graph capture."
            )

        shmem = self.shmem
        self._input_buf = shmem.zeros((M, N), dtype=dtype)
        shmem.device_barrier()
        self._input_M = M
        self._input_N = N
        self._input_dtype = dtype

        cur_rank = (
            torch.distributed.get_rank()
            if torch.distributed.is_initialized()
            else 0
        )
        logger.info(
            f"Iris (twoshot): allocated input buffer ({M}, {N}), "
            f"dtype={dtype}, rank={cur_rank}"
        )

        return self._input_buf

    def _get_ar_output_buffer(
        self,
        M: int,
        N: int,
        dtype: torch.dtype,
    ) -> Any:
        """Get a view into the pre-allocated iris allreduce output buffer.

        This buffer lives on the symmetric heap so that the broadcast phase
        can write reduced rows to remote ranks via iris.store.
        """
        if (self._ar_output_buf is not None
                and M <= self._ar_output_M
                and N == self._input_N
                and dtype == self._input_dtype):
            return self._ar_output_buf[:M]

        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                f"Iris (twoshot): allreduce output buffer too small for M={M} "
                f"(allocated {self._ar_output_M}). Cannot allocate during "
                f"CUDA graph capture."
            )

        shmem = self.shmem
        self._ar_output_buf = shmem.zeros((M, N), dtype=dtype)
        shmem.device_barrier()
        self._ar_output_M = M

        cur_rank = (
            torch.distributed.get_rank()
            if torch.distributed.is_initialized()
            else 0
        )
        logger.info(
            f"Iris (twoshot): allocated allreduce output buffer ({M}, {N}), "
            f"dtype={dtype}, rank={cur_rank}"
        )

        return self._ar_output_buf

    def _get_output_buffers(
        self,
        M: int,
        N: int,
        dtype: torch.dtype,
        quant_dtype: torch.dtype,
        device: torch.device,
        has_residual: bool,
    ) -> Tuple[
        torch.Tensor,
        Optional[torch.Tensor],
        torch.Tensor,
        torch.Tensor,
    ]:
        """Get views into pre-allocated output buffers (regular GPU memory)."""
        need_alloc = (
            self._out_rms is None
            or M > self._out_rms.shape[0]
            or N != self._out_rms.shape[1]
            or quant_dtype != self._out_quant_dtype
        )

        if need_alloc:
            if torch.cuda.is_current_stream_capturing():
                existing = self._out_rms.shape[0] if self._out_rms is not None else 0
                raise RuntimeError(
                    f"Iris (twoshot): output buffers too small for M={M} "
                    f"(allocated {existing}). Cannot allocate during "
                    f"CUDA graph capture."
                )

            self._out_rms = torch.empty((M, N), dtype=dtype, device=device)
            self._out_quant = torch.empty((M, N), dtype=quant_dtype, device=device)
            self._out_scale = torch.empty(1, dtype=torch.float32, device=device)
            self._out_residual = torch.empty((M, N), dtype=dtype, device=device)
            self._out_quant_dtype = quant_dtype

            # Cross-CTA sync buffers
            self._global_amax = torch.zeros(1, dtype=torch.float32, device=device)
            self._cta_arrival = torch.zeros(1, dtype=torch.int32, device=device)

            # Inlined barrier: CTA 0 -> other CTAs signal
            self._barrier_done = torch.zeros(1, dtype=torch.int32, device=device)

            cur_rank = (
                torch.distributed.get_rank()
                if torch.distributed.is_initialized()
                else 0
            )
            logger.info(
                f"Iris (twoshot): allocated output buffers ({M}, {N}), "
                f"dtype={dtype}, rank={cur_rank}"
            )

        assert self._out_rms is not None
        assert self._out_quant is not None
        assert self._out_scale is not None
        assert self._out_residual is not None

        return (
            self._out_rms[:M],
            self._out_residual[:M] if has_residual else None,
            self._out_quant[:M],
            self._out_scale,
        )

    def fused_allreduce_rmsnorm_quant_gemm(
        self,
        input_tensor: torch.Tensor,
        rms_weight: torch.Tensor,
        rms_eps: float,
        quant_dtype: torch.dtype,
        gemm_weight: torch.Tensor,
        weight_scale: torch.Tensor,
        out_dtype: torch.dtype,
        residual: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Two-shot AllReduce + RMSNorm + per-tensor FP8 quant + GEMM.

        Args:
            input_tensor: Input tensor (M, N) on GPU
            rms_weight: RMSNorm weight (N,)
            rms_eps: RMSNorm epsilon
            quant_dtype: FP8 dtype (e.g. torch.float8_e4m3fn)
            gemm_weight: FP8 weight matrix for scaled GEMM
            weight_scale: Scale for gemm_weight
            out_dtype: Output dtype for GEMM (e.g. torch.bfloat16)
            residual: Optional residual tensor (M, N)
            bias: Optional bias for GEMM

        Returns:
            (gemm_out, residual_out). residual_out is None when
            residual is None.
        """
        shmem = self.shmem
        M, N = input_tensor.shape
        device = input_tensor.device

        # Get pre-allocated buffers
        iris_input = self._get_input_buffer(M, N, input_tensor.dtype)
        ar_output = self._get_ar_output_buffer(M, N, input_tensor.dtype)
        rms_out, residual_out, quant_out, scale_out = (
            self._get_output_buffers(
                M, N, input_tensor.dtype, quant_dtype, device,
                has_residual=residual is not None,
            )
        )

        # Copy input to symmetric heap
        iris_input.copy_(input_tensor)

        # FP8 max value
        fp8_max = torch.finfo(quant_dtype).max

        # Iris group info
        rank_in_group, rank_global, world_size, rank_start, rank_stride = (
            extract_group_info(shmem)
        )
        heap_bases = shmem.get_heap_bases()

        BLOCK_SIZE_N = triton.next_power_of_2(N)

        # Reset sync buffers before kernel launch
        assert self._global_amax is not None
        assert self._cta_arrival is not None
        assert self._barrier_done is not None
        self._global_amax.zero_()
        self._cta_arrival.zero_()
        self._barrier_done.zero_()

        def grid(META):
            return (META["COMM_SMS"],)

        # Single fused kernel
        fused_twoshot_allreduce_rmsnorm_quant_kernel[grid](
            iris_input,
            ar_output,
            rms_out,
            quant_out,
            scale_out,
            self._global_amax,
            self._cta_arrival,
            self._barrier_flags,
            self._barrier_epoch,
            self._barrier_done,
            residual if residual is not None else input_tensor,
            residual_out if residual_out is not None else ar_output,
            rms_weight,
            rms_eps,
            fp8_max,
            M,
            N,
            iris_input.stride(0),
            iris_input.stride(1),
            ar_output.stride(0),
            ar_output.stride(1),
            heap_bases,
            rank_in_group,
            rank_global,
            world_size,
            rank_start,
            rank_stride,
            BLOCK_SIZE_N,
            residual is not None,
        )

        # Advance epoch for the inlined barrier consumed by the kernel
        self._barrier_epoch += 1

        # Scaled GEMM
        gemm_out = torch._scaled_mm(
            quant_out, gemm_weight, out_dtype=out_dtype,
            scale_a=scale_out, scale_b=weight_scale, bias=bias,
        )

        return gemm_out, residual_out


_iris_twoshot_manager: Optional[IrisTwoshotManager] = None


def get_iris_twoshot_manager() -> IrisTwoshotManager:
    """Get the global Iris twoshot manager instance."""
    global _iris_twoshot_manager
    if _iris_twoshot_manager is None:
        _iris_twoshot_manager = IrisTwoshotManager()
    return _iris_twoshot_manager


def initialize_iris_twoshot(heap_size: Optional[int] = None) -> None:
    """Initialize Iris for two-shot fused all-reduce operations.

    Call this once at model load time before any forward passes.

    Args:
        heap_size: Size of symmetric heap in bytes (default: 8GB)
    """
    get_iris_twoshot_manager().initialize(heap_size)


def fused_allreduce_add_rms_quant_gemm_iris_twoshot(
    input: torch.Tensor,
    rms_weight: torch.Tensor,
    rms_eps: float,
    quant_dtype: torch.dtype,
    group_name: str,
    gemm_weight: torch.Tensor,
    weight_scale: torch.Tensor,
    out_dtype: torch.dtype,
    residual: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Fused AllReduce + RMSNorm + per-tensor FP8 Quant + GEMM (two-shot).

    Two-shot allreduce with inlined device barrier and fused
    RMSNorm + per-tensor FP8 quantization, followed by scaled GEMM.
    All FP8 is internal.

    Returns (gemm_out, residual_out).
    """
    return get_iris_twoshot_manager().fused_allreduce_rmsnorm_quant_gemm(
        input, rms_weight, rms_eps, quant_dtype,
        gemm_weight, weight_scale, out_dtype, residual, bias,
    )
