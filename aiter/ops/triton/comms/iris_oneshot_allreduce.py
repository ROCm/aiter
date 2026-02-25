# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Iris fused AllReduce + RMSNorm + per-tensor FP8 Quant in a single kernel.

Single Triton kernel that fuses:
1. One-shot all-reduce (gather from all ranks via iris.load)
2. Optional residual addition
3. RMSNorm (row-level variance + normalize)
4. Per-tensor FP8 quantization (global scale across all rows)

Two-pass persistent kernel:
  Pass 1: AllReduce + RMSNorm + compute per-row amax (atomic max to global)
  Barrier: Spin-wait until all CTAs finish pass 1
  Pass 2: Read global per-tensor scale, reload rms_out from L2, quantize

CUDA graph compatibility:
- Buffer pre-allocation: The first forward pass (warmup) sees the largest M
  (max_num_batched_tokens). We allocate iris input and output buffers at that
  size, then return views for smaller M values during CUDA graph capture.
  This ensures fixed GPU memory addresses across graph capture and replay.
- Device barriers: All barriers use shmem.device_barrier() (Triton device-side
  atomics) instead of shmem.barrier() (host-side NCCL). This avoids
  hipErrorStreamCaptureUnsupported on ROCm during CUDA graph capture.
"""

import os
from typing import Any, Optional, Tuple

import iris
import torch
import triton
import triton.language as tl

import logging

__all__ = ["fused_allreduce_add_rms_quant_iris_oneshot"]

logger = logging.getLogger(__name__)

AUTOTUNE = False
if AUTOTUNE:
    os.environ["TRITON_PRINT_AUTOTUNING"] = "1"


# ============================================================================
# Inlined from iris.ccl.utils
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

IRIS_ONESHOT_AUTOTUNE_KEYS = [
    "M",
    "N",
    "HAS_RESIDUAL",
]


def _compute_chunk_size(comm_sms: int, num_xcds: int, swizzle_size: int = 4) -> int:
    chunk = swizzle_size * swizzle_size
    return min(chunk, comm_sms // num_xcds)


def get_iris_oneshot_configs(autotune: bool):
    num_xcds = iris.hip.get_num_xcc()

    if not autotune:
        # Best config from autotuning on gfx950 (8x MI355, M=512, N=8192)
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


iris_oneshot_autotune_configs = get_iris_oneshot_configs(AUTOTUNE)


# ============================================================================
# Fused AllReduce + RMSNorm + Per-Tensor FP8 Quant kernel
# ============================================================================


@triton.autotune(
    configs=iris_oneshot_autotune_configs,
    key=IRIS_ONESHOT_AUTOTUNE_KEYS,
    use_cuda_graph=True,
)
@triton.jit
def fused_allreduce_rmsnorm_quant_kernel(
    # Input (in symmetric heap for iris.load)
    input_ptr,
    # Outputs (regular GPU memory)
    allreduce_out_ptr,
    rms_out_ptr,
    quant_out_ptr,
    scale_out_ptr,
    # Cross-CTA synchronization for per-tensor scale
    global_amax_ptr,
    arrival_counter_ptr,
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
    """Fused one-shot AllReduce + RMSNorm + per-tensor FP8 quant.

    Two-pass persistent kernel:
      Pass 1: AllReduce + RMSNorm + compute per-row amax (atomic max to global)
      Barrier: Spin-wait until all CTAs finish pass 1
      Pass 2: Read global per-tensor scale, reload rms_out, quantize to FP8

    Each CTA persistently iterates over rows.
    """
    pid = tl.program_id(0)

    if NUM_XCDS != 1:
        pid = chiplet_transform_chunked(pid, COMM_SMS, NUM_XCDS, CHUNK_SIZE)

    cols = tl.arange(0, BLOCK_SIZE_N)
    col_mask = cols < N

    # Load RMSNorm weight once (shared across all rows)
    rms_w = tl.load(
        rms_weight_ptr + cols, mask=col_mask, other=0.0
    ).to(tl.float32)

    # === Pass 1: AllReduce + RMSNorm + accumulate global amax ===
    for row in range(pid, M, COMM_SMS):
        in_offset = row * stride_in_m + cols * stride_in_n
        out_offset = row * N + cols

        # --- Step 1: Gather from all ranks and reduce ---
        acc = tl.zeros((BLOCK_SIZE_N,), dtype=tl.float32)
        for i in range(world_size):
            remote_rank = rank_start + i * rank_stride
            partial = iris.load(
                input_ptr + in_offset,
                iris_rank,
                remote_rank,
                heap_bases,
                mask=col_mask,
            )
            acc += partial.to(tl.float32)

        # Store allreduce output
        tl.store(
            allreduce_out_ptr + out_offset,
            acc.to(allreduce_out_ptr.type.element_ty),
            mask=col_mask,
        )

        # --- Step 2: Optional residual add ---
        if HAS_RESIDUAL:
            res_in = tl.load(
                residual_in_ptr + out_offset, mask=col_mask, other=0.0
            ).to(tl.float32)
            rms_in = acc + res_in
            tl.store(
                residual_out_ptr + out_offset,
                rms_in.to(residual_out_ptr.type.element_ty),
                mask=col_mask,
            )
        else:
            rms_in = acc

        # --- Step 3: RMSNorm (in float32) ---
        sq_sum = tl.sum(rms_in * rms_in, axis=0)
        variance = sq_sum / N
        rrms = tl.rsqrt(variance + rms_eps)
        normed = rms_in * rrms * rms_w

        tl.store(
            rms_out_ptr + out_offset,
            normed.to(rms_out_ptr.type.element_ty),
            mask=col_mask,
        )

        # --- Step 4: Per-row amax → atomic max to global ---
        row_amax = tl.max(tl.abs(normed), axis=0)
        tl.atomic_max(global_amax_ptr, row_amax, sem="relaxed")

    # === Cross-CTA barrier: spin until all CTAs finish pass 1 ===
    # Each CTA increments the counter. Spin until counter == COMM_SMS.
    # atomic_add(ptr, 0) is a non-destructive atomic read.
    tl.atomic_add(arrival_counter_ptr, 1, sem="release")
    while tl.atomic_add(arrival_counter_ptr, 0, sem="acquire") < COMM_SMS:
        pass

    # === Pass 2: Quantize with per-tensor scale ===
    global_amax = tl.load(global_amax_ptr)
    global_amax = tl.maximum(global_amax, 1e-12)
    scale = global_amax / fp8_max

    # Store per-tensor scale (only one CTA needs to do this)
    if pid == 0:
        tl.store(scale_out_ptr, scale)

    for row in range(pid, M, COMM_SMS):
        out_offset = row * N + cols

        # Reload rms_out (should be warm in L2)
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


class IrisOneshotManager:
    """Singleton manager for fused one-shot AllReduce+RMSNorm+Quant."""

    _instance: Optional["IrisOneshotManager"] = None
    _initialized: bool = False

    def __new__(cls) -> "IrisOneshotManager":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if IrisOneshotManager._initialized:
            return
        IrisOneshotManager._initialized = True

        self._shmem: Any = None
        self._heap_size: int = 2**33  # 8GB default

        # Pre-allocated input buffer (iris symmetric heap)
        self._input_buf: Any = None
        self._input_M: int = 0
        self._input_N: int = 0
        self._input_dtype: Optional[torch.dtype] = None

        # Pre-allocated output buffers (regular GPU memory)
        self._out_allreduce: Optional[torch.Tensor] = None
        self._out_rms: Optional[torch.Tensor] = None
        self._out_quant: Optional[torch.Tensor] = None
        self._out_scale: Optional[torch.Tensor] = None
        self._out_residual: Optional[torch.Tensor] = None
        self._out_quant_dtype: Optional[torch.dtype] = None

        # Cross-CTA sync buffers for per-tensor scale reduction
        self._global_amax: Optional[torch.Tensor] = None
        self._arrival_counter: Optional[torch.Tensor] = None

    def initialize(self, heap_size: Optional[int] = None) -> None:
        """Initialize Iris symmetric heap (call once at startup)."""
        if self._shmem is not None:
            logger.debug("Iris (oneshot) already initialized, skipping")
            return

        if heap_size is not None:
            self._heap_size = heap_size

        cur_rank = (
            torch.distributed.get_rank()
            if torch.distributed.is_initialized()
            else 0
        )
        logger.info(
            "Initializing Iris (oneshot) symmetric heap: "
            f"rank={cur_rank}, heap_size={self._heap_size / 2**30:.1f}GB"
        )

        self._shmem = iris.iris(self._heap_size)

        logger.info(
            f"Iris (oneshot) initialized successfully on rank {cur_rank}"
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
        """Get a view into the pre-allocated iris input buffer.

        On the first call (warmup), allocates for (M, N). Subsequent calls
        with smaller M return a view into the same allocation. This avoids
        shmem.zeros() and shmem.barrier() during CUDA graph capture.
        """
        # Return view if existing buffer is large enough
        if (self._input_buf is not None
                and M <= self._input_M
                and N == self._input_N
                and dtype == self._input_dtype):
            return self._input_buf[:M]

        # Need a (larger) buffer. Not safe during CUDA graph capture.
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                f"Iris (oneshot): input buffer too small for M={M} "
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
            f"Iris (oneshot): allocated input buffer ({M}, {N}), "
            f"dtype={dtype}, rank={cur_rank}"
        )

        return self._input_buf

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
        torch.Tensor,
        Optional[torch.Tensor],
        torch.Tensor,
        torch.Tensor,
    ]:
        """Get views into pre-allocated output buffers.

        On the first call (warmup), allocates for (M, N). Subsequent calls
        with smaller M return views into the same allocation.
        """
        need_alloc = (
            self._out_allreduce is None
            or M > self._out_allreduce.shape[0]
            or N != self._out_allreduce.shape[1]
            or quant_dtype != self._out_quant_dtype
        )

        if need_alloc:
            if torch.cuda.is_current_stream_capturing():
                existing = self._out_allreduce.shape[0] if self._out_allreduce is not None else 0
                raise RuntimeError(
                    f"Iris (oneshot): output buffers too small for M={M} "
                    f"(allocated {existing}). Cannot allocate during "
                    f"CUDA graph capture."
                )

            self._out_allreduce = torch.empty((M, N), dtype=dtype, device=device)
            self._out_rms = torch.empty((M, N), dtype=dtype, device=device)
            self._out_quant = torch.empty((M, N), dtype=quant_dtype, device=device)
            # Per-tensor scale: single scalar
            self._out_scale = torch.empty(1, dtype=torch.float32, device=device)
            self._out_residual = torch.empty((M, N), dtype=dtype, device=device)
            self._out_quant_dtype = quant_dtype

            # Cross-CTA sync buffers (fixed size, not M-dependent)
            self._global_amax = torch.zeros(1, dtype=torch.float32, device=device)
            self._arrival_counter = torch.zeros(1, dtype=torch.int32, device=device)

            cur_rank = (
                torch.distributed.get_rank()
                if torch.distributed.is_initialized()
                else 0
            )
            logger.info(
                f"Iris (oneshot): allocated output buffers ({M}, {N}), "
                f"dtype={dtype}, rank={cur_rank}"
            )

        assert self._out_allreduce is not None
        assert self._out_rms is not None
        assert self._out_quant is not None
        assert self._out_scale is not None
        assert self._out_residual is not None

        return (
            self._out_allreduce[:M],
            self._out_rms[:M],
            self._out_residual[:M] if has_residual else None,
            self._out_quant[:M],
            self._out_scale,
        )

    def fused_allreduce_rmsnorm_quant(
        self,
        input_tensor: torch.Tensor,
        rms_weight: torch.Tensor,
        rms_eps: float,
        quant_dtype: torch.dtype,
        residual: Optional[torch.Tensor] = None,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        Optional[torch.Tensor],
        torch.Tensor,
        torch.Tensor,
    ]:
        """Fused AllReduce + RMSNorm + per-tensor FP8 quant in one kernel.

        Args:
            input_tensor: Input tensor (M, N) on GPU
            rms_weight: RMSNorm weight (N,)
            rms_eps: RMSNorm epsilon
            quant_dtype: FP8 dtype (e.g. torch.float8_e4m3fn)
            residual: Optional residual tensor (M, N)

        Returns:
            (allreduce_out, rms_out, residual_out, quant_out, scale_out)
            residual_out is None when residual is None.
            scale_out is per-tensor: shape (1,).
        """
        shmem = self.shmem
        M, N = input_tensor.shape
        device = input_tensor.device

        # Get pre-allocated buffers (views into max-size allocations)
        iris_input = self._get_input_buffer(M, N, input_tensor.dtype)
        allreduce_out, rms_out, residual_out, quant_out, scale_out = (
            self._get_output_buffers(
                M, N, input_tensor.dtype, quant_dtype, device,
                has_residual=residual is not None,
            )
        )

        # Copy input to symmetric heap
        iris_input.copy_(input_tensor)
        shmem.device_barrier()

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
        assert self._arrival_counter is not None
        self._global_amax.zero_()
        self._arrival_counter.zero_()

        def grid(META):
            return (META["COMM_SMS"],)

        # Launch fused kernel (two-pass: allreduce+rmsnorm+amax, then quant)
        fused_allreduce_rmsnorm_quant_kernel[grid](
            iris_input,
            allreduce_out,
            rms_out,
            quant_out,
            scale_out,
            self._global_amax,
            self._arrival_counter,
            # Dummy ptrs when no residual (HAS_RESIDUAL=False skips access)
            residual if residual is not None else input_tensor,
            residual_out if residual_out is not None else allreduce_out,
            rms_weight,
            rms_eps,
            fp8_max,
            M,
            N,
            iris_input.stride(0),
            iris_input.stride(1),
            heap_bases,
            rank_in_group,
            rank_global,
            world_size,
            rank_start,
            rank_stride,
            BLOCK_SIZE_N,
            residual is not None,
        )

        shmem.device_barrier()

        return allreduce_out, rms_out, residual_out, quant_out, scale_out


_iris_oneshot_manager: Optional[IrisOneshotManager] = None


def get_iris_oneshot_manager() -> IrisOneshotManager:
    """Get the global Iris oneshot manager instance."""
    global _iris_oneshot_manager
    if _iris_oneshot_manager is None:
        _iris_oneshot_manager = IrisOneshotManager()
    return _iris_oneshot_manager


def initialize_iris_oneshot(heap_size: Optional[int] = None) -> None:
    """Initialize Iris for fused all-reduce operations.

    Call this once at model load time before any forward passes.

    Args:
        heap_size: Size of symmetric heap in bytes (default: 8GB)
    """
    get_iris_oneshot_manager().initialize(heap_size)


def fused_allreduce_add_rms_quant_iris_oneshot(
    input: torch.Tensor,
    rms_weight: torch.Tensor,
    rms_eps: float,
    quant_scale: torch.Tensor,
    quant_dtype: torch.dtype,
    group_name: str,
    residual: Optional[torch.Tensor] = None,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    Optional[torch.Tensor],
    torch.Tensor,
    torch.Tensor,
]:
    """Fused AllReduce + Add + RMSNorm + per-tensor FP8 Quant.

    Single Triton kernel: one-shot all-reduce with RMSNorm and per-tensor
    FP8 quantization. Two-pass persistent kernel computes the global amax
    across all rows, then quantizes with a single per-tensor scale.
    """
    iris_mgr = get_iris_oneshot_manager()
    return iris_mgr.fused_allreduce_rmsnorm_quant(
        input, rms_weight, rms_eps, quant_dtype, residual,
    )
