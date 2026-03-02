# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Iris two-shot fused AllReduce + RMSNorm + per-row FP8 Quant, with
hipBLASLt GEMM (torch._scaled_mm).

Hybrid approach: the communication-bound part (allreduce + rmsnorm +
per-row FP8 quant + FP8 broadcast) runs as a single persistent Triton
kernel on the iris symmetric heap. The compute-bound GEMM is delegated
to torch._scaled_mm (hipBLASLt) which is vendor-tuned and significantly
faster than a Triton GEMM for production matrix shapes.

After the Triton kernel completes:
  - quant_heap contains the complete (M, N) FP8 matrix for ALL rows
  - scale_heap contains the per-row float32 scales for ALL rows
  - result_out contains the BF16 allreduce result (only owned rows)

Then torch._scaled_mm is called:
  gemm_out = torch._scaled_mm(
      quant_heap, gemm_weight,
      out_dtype=out_dtype,
      scale_a=scale_heap.unsqueeze(1),
      scale_b=weight_scale,
      bias=bias,
  )

This eliminates the 3.7x slowdown observed with the inlined Triton GEMM
while keeping the iris communication benefits (FP8 broadcast halves
cross-rank traffic).

CUDA graph compatibility:
- Buffer pre-allocation with view pattern (same as twoshot)
- Device barriers are inlined (no separate kernel launch)
"""

import logging
import os
from typing import Any, Optional, Tuple

import iris
import torch
import triton
import triton.language as tl

from .iris_twoshot_allreduce import (
    _compute_chunk_size,
    _inlined_device_barrier,
    chiplet_transform_chunked,
    extract_group_info,
)

__all__ = ["fused_allreduce_add_rms_row_quant_gemm_iris_twoshot_hipblaslt"]

logger = logging.getLogger(__name__)

AUTOTUNE = False
if AUTOTUNE:
    os.environ["TRITON_PRINT_AUTOTUNING"] = "1"


IRIS_TWOSHOT_ROW_HIPBLASLT_AUTOTUNE_KEYS = [
    "M",
    "N",
    "HAS_RESIDUAL",
]


def _get_iris_twoshot_row_hipblaslt_configs(autotune: bool):
    """Generate autotune configs for the comm-only kernel."""
    num_xcds = iris.hip.get_num_xcc()

    if not autotune:
        comm_sms = 128
        return [
            triton.Config(
                {
                    "COMM_SMS": comm_sms,
                    "NUM_XCDS": num_xcds,
                    "CHUNK_SIZE": _compute_chunk_size(comm_sms, num_xcds),
                    "waves_per_eu": 1,
                },
                num_warps=16,
                num_stages=2,
            )
        ]

    configs = []
    COMM_SMS_OPTIONS = [64, 128]
    NUM_WARPS_OPTIONS = [4, 16]
    NUM_STAGES_OPTIONS = [2]
    WAVES_PER_EU_OPTIONS = [1]
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


iris_twoshot_row_hipblaslt_autotune_configs = (
    _get_iris_twoshot_row_hipblaslt_configs(AUTOTUNE)
)


# ============================================================================
# Fused two-shot AllReduce + RMSNorm + per-row FP8 Quant kernel
# (Communication only — no GEMM. GEMM is done via torch._scaled_mm.)
# ============================================================================


@triton.autotune(
    configs=iris_twoshot_row_hipblaslt_autotune_configs,
    key=IRIS_TWOSHOT_ROW_HIPBLASLT_AUTOTUNE_KEYS,
    use_cuda_graph=True,
)
@triton.jit
def fused_twoshot_allreduce_rmsnorm_row_quant_kernel(
    # Input (in symmetric heap for iris.load)
    input_ptr,
    # FP8 quant heap (in symmetric heap for FP8 broadcast)
    quant_heap_ptr,
    # Scale heap (in symmetric heap for per-row scale broadcast)
    scale_heap_ptr,
    # Outputs (regular GPU memory)
    result_out_ptr,
    # Inlined device barrier state (symmetric heap)
    barrier_flags_ptr,
    barrier_epoch_ptr,
    # Inlined device barrier: CTA 0 -> other CTAs signal (local GPU memory)
    barrier_done_ptr,
    # Residual (regular GPU memory; dummy ptr when HAS_RESIDUAL=False)
    residual_in_ptr,
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
    # Quant heap strides (iris heap buffer, FP8)
    stride_qh_m,
    stride_qh_n,
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
    # Autotuned params (from config)
    COMM_SMS: tl.constexpr,
    NUM_XCDS: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
):
    """Fused two-shot AllReduce + RMSNorm + per-row FP8 quant.

    Communication-only kernel: performs allreduce, rmsnorm, per-row FP8
    quantization, and FP8 broadcast via the iris symmetric heap. No GEMM.
    After this kernel completes, quant_heap contains the complete (M, N)
    FP8 matrix and scale_heap contains per-row scales for ALL rows.
    """
    raw_pid = tl.program_id(0)
    pid = raw_pid

    if NUM_XCDS != 1:
        pid = chiplet_transform_chunked(raw_pid, COMM_SMS, NUM_XCDS, CHUNK_SIZE)

    cols = tl.arange(0, BLOCK_SIZE_N)
    col_mask = cols < N

    rms_w = tl.load(
        rms_weight_ptr + cols, mask=col_mask, other=0.0
    ).to(tl.float32)

    # ================================================================
    # Step 1: Two-shot AllReduce + fused RMSNorm + quant + FP8 broadcast
    # ================================================================

    # Block distribution of rows across ranks
    rows_per_rank = tl.cdiv(M, world_size)
    my_start = group_rank * rows_per_rank
    my_end = tl.minimum(my_start + rows_per_rank, M)

    for row in range(my_start + pid, my_end, COMM_SMS):
        in_offset = row * stride_in_m + cols * stride_in_n
        qh_offset = row * stride_qh_m + cols * stride_qh_n
        out_offset = row * N + cols

        # Gather from all ranks and reduce, with inline residual pre-add
        acc = tl.zeros((BLOCK_SIZE_N,), dtype=tl.float32)
        for i in tl.static_range(0, world_size):
            remote_rank = rank_start + i * rank_stride
            val = iris.load(
                input_ptr + in_offset,
                iris_rank,
                remote_rank,
                heap_bases,
                mask=col_mask,
            ).to(tl.float32)
            # Pre-add residual for this rank's own partial
            if HAS_RESIDUAL and remote_rank == iris_rank:
                res_in = tl.load(
                    residual_in_ptr + out_offset, mask=col_mask, other=0.0
                ).to(tl.float32)
                val = val + res_in
            acc += val

        # Store BF16 result residual_out for downstream
        tl.store(
            result_out_ptr + out_offset,
            acc.to(result_out_ptr.type.element_ty),
            mask=col_mask,
        )

        # RMSNorm
        sq_sum = tl.sum(acc * acc, axis=0)
        rrms = tl.rsqrt(sq_sum / N + rms_eps)
        normed = acc * rrms * rms_w

        # Per-row scale and quantize
        row_amax = tl.max(tl.abs(normed), axis=0)
        row_amax = tl.maximum(row_amax, 1e-12)
        scale = row_amax / fp8_max

        quantized = (normed / scale).to(quant_heap_ptr.type.element_ty)

        # Store FP8 to heap and broadcast to all peers
        tl.store(
            quant_heap_ptr + qh_offset,
            quantized,
            mask=col_mask,
        )
        # Store scale to heap
        tl.store(scale_heap_ptr + row, scale)

        for i in tl.static_range(0, world_size):
            remote_rank = rank_start + i * rank_stride
            if remote_rank != iris_rank:
                iris.store(
                    quant_heap_ptr + qh_offset,
                    quantized,
                    iris_rank,
                    remote_rank,
                    heap_bases,
                    mask=col_mask,
                )
                iris.store(
                    scale_heap_ptr + row,
                    scale,
                    iris_rank,
                    remote_rank,
                    heap_bases,
                )

    # ================================================================
    # Step 2: Cross-rank device barrier
    # ================================================================

    _inlined_device_barrier(
        raw_pid,
        barrier_flags_ptr,
        barrier_epoch_ptr,
        barrier_done_ptr,
        heap_bases,
        iris_rank,
        world_size,
        rank_start,
        rank_stride,
    )

    # No Step 3 — GEMM is done after kernel returns via torch._scaled_mm


# ============================================================================
# Manager and public API
# ============================================================================


class IrisTwoshotRowHipblasltManager:
    """Singleton manager for two-shot AllReduce+RMSNorm+per-row Quant + hipBLASLt GEMM.

    FP8 broadcast variant: broadcasts FP8 and per-row scales on the
    heap instead of BF16. Only owned rows get BF16 result_out and
    residual addition. After the Triton kernel completes, GEMM is
    performed via torch._scaled_mm (hipBLASLt).
    """

    _instance: Optional["IrisTwoshotRowHipblasltManager"] = None
    _initialized: bool = False

    def __new__(cls) -> "IrisTwoshotRowHipblasltManager":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if IrisTwoshotRowHipblasltManager._initialized:
            return
        IrisTwoshotRowHipblasltManager._initialized = True

        self._shmem: Any = None
        self._heap_size: int = 2**33  # 8GB default

        # Pre-allocated input buffer (iris symmetric heap)
        self._input_buf: Any = None
        self._input_M: int = 0
        self._input_N: int = 0
        self._input_dtype: Optional[torch.dtype] = None

        # Pre-allocated FP8 quant heap buffer (iris symmetric heap)
        self._quant_heap_buf: Any = None
        self._quant_heap_M: int = 0
        self._quant_heap_N: int = 0
        self._quant_heap_dtype: Optional[torch.dtype] = None

        # Pre-allocated per-row scale heap buffer (iris symmetric heap)
        self._scale_heap_buf: Any = None
        self._scale_heap_M: int = 0

        # Pre-allocated output buffers (regular GPU memory)
        self._out_result: Optional[torch.Tensor] = None

        # Inlined device barrier state
        self._barrier_flags: Any = None  # on symmetric heap
        self._barrier_epoch: Optional[torch.Tensor] = None
        self._barrier_done: Optional[torch.Tensor] = None  # local GPU memory

    def initialize(self, heap_size: Optional[int] = None) -> None:
        """Initialize Iris symmetric heap (call once at startup)."""
        if self._shmem is not None:
            logger.debug(
                "Iris (twoshot-row-hipblaslt) already initialized, skipping"
            )
            return

        if heap_size is not None:
            self._heap_size = heap_size

        cur_rank = (
            torch.distributed.get_rank()
            if torch.distributed.is_initialized()
            else 0
        )
        logger.info(
            "Initializing Iris (twoshot-row-hipblaslt) symmetric heap: "
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
            f"Iris (twoshot-row-hipblaslt) initialized successfully "
            f"on rank {cur_rank}"
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
                f"Iris (twoshot-row-hipblaslt): input buffer too small for "
                f"M={M} (allocated {self._input_M}). Cannot allocate during "
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
            f"Iris (twoshot-row-hipblaslt): allocated input buffer "
            f"({M}, {N}), dtype={dtype}, rank={cur_rank}"
        )

        return self._input_buf

    def _get_quant_heap_buffer(
        self,
        M: int,
        N: int,
        quant_dtype: torch.dtype,
    ) -> Any:
        """Get a view into the pre-allocated iris FP8 quant heap buffer."""
        if (self._quant_heap_buf is not None
                and M <= self._quant_heap_M
                and N == self._quant_heap_N
                and quant_dtype == self._quant_heap_dtype):
            return self._quant_heap_buf[:M]

        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                f"Iris (twoshot-row-hipblaslt): quant heap buffer too small "
                f"for M={M} (allocated {self._quant_heap_M}). Cannot "
                f"allocate during CUDA graph capture."
            )

        shmem = self.shmem
        self._quant_heap_buf = shmem.zeros((M, N), dtype=quant_dtype)
        shmem.device_barrier()
        self._quant_heap_M = M
        self._quant_heap_N = N
        self._quant_heap_dtype = quant_dtype

        cur_rank = (
            torch.distributed.get_rank()
            if torch.distributed.is_initialized()
            else 0
        )
        logger.info(
            f"Iris (twoshot-row-hipblaslt): allocated quant heap buffer "
            f"({M}, {N}), dtype={quant_dtype}, rank={cur_rank}"
        )

        return self._quant_heap_buf

    def _get_scale_heap_buffer(
        self,
        M: int,
    ) -> Any:
        """Get a view into the pre-allocated iris per-row scale heap buffer."""
        if (self._scale_heap_buf is not None
                and M <= self._scale_heap_M):
            return self._scale_heap_buf[:M]

        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                f"Iris (twoshot-row-hipblaslt): scale heap buffer too small "
                f"for M={M} (allocated {self._scale_heap_M}). Cannot "
                f"allocate during CUDA graph capture."
            )

        shmem = self.shmem
        self._scale_heap_buf = shmem.zeros((M,), dtype=torch.float32)
        shmem.device_barrier()
        self._scale_heap_M = M

        cur_rank = (
            torch.distributed.get_rank()
            if torch.distributed.is_initialized()
            else 0
        )
        logger.info(
            f"Iris (twoshot-row-hipblaslt): allocated scale heap buffer "
            f"({M},), dtype=float32, rank={cur_rank}"
        )

        return self._scale_heap_buf

    def _get_output_buffers(
        self,
        M: int,
        N: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        """Get view into pre-allocated result_out buffer (regular GPU memory).

        Returns result_out only. GEMM output comes from torch._scaled_mm.
        """
        need_result = (
            self._out_result is None
            or M > self._out_result.shape[0]
            or N != self._out_result.shape[1]
        )
        need_barrier = self._barrier_done is None

        if need_result or need_barrier:
            if torch.cuda.is_current_stream_capturing():
                existing_M = (
                    self._out_result.shape[0]
                    if self._out_result is not None
                    else 0
                )
                raise RuntimeError(
                    f"Iris (twoshot-row-hipblaslt): output buffers too small "
                    f"for M={M} (allocated M={existing_M}). Cannot allocate "
                    f"during CUDA graph capture."
                )

            if need_result:
                self._out_result = torch.empty(
                    (M, N), dtype=dtype, device=device,
                )

            if need_barrier:
                self._barrier_done = torch.zeros(
                    1, dtype=torch.int32, device=device
                )
                self._barrier_epoch = torch.zeros(
                    1, dtype=torch.int32, device=device
                )

            cur_rank = (
                torch.distributed.get_rank()
                if torch.distributed.is_initialized()
                else 0
            )
            logger.info(
                f"Iris (twoshot-row-hipblaslt): allocated output buffer "
                f"result=({M}, {N}), dtype={dtype}, rank={cur_rank}"
            )

        assert self._out_result is not None
        return self._out_result[:M]

    def fused_allreduce_rmsnorm_row_quant_gemm(
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
        """Two-shot AllReduce + RMSNorm + per-row FP8 quant + hipBLASLt GEMM.

        Hybrid approach:
        1. Triton kernel: allreduce + rmsnorm + per-row FP8 quant + broadcast
        2. torch._scaled_mm (hipBLASLt): FP8 GEMM with per-row scaling

        Args:
            input_tensor: Input tensor (M, N) on GPU
            rms_weight: RMSNorm weight (N,)
            rms_eps: RMSNorm epsilon
            quant_dtype: FP8 dtype (e.g. torch.float8_e4m3fn)
            gemm_weight: FP8 weight matrix (N, K_GEMM)
            weight_scale: Per-tensor scale for gemm_weight
            out_dtype: Output dtype for GEMM (e.g. torch.bfloat16)
            residual: Optional residual tensor (M, N)
            bias: Optional bias for GEMM (K_GEMM,)

        Returns:
            (gemm_out, residual_out). residual_out is None when
            residual is None.
        """
        shmem = self.shmem
        M, N = input_tensor.shape
        device = input_tensor.device

        # Get pre-allocated buffers
        iris_input = self._get_input_buffer(M, N, input_tensor.dtype)
        quant_heap = self._get_quant_heap_buffer(M, N, quant_dtype)
        scale_heap = self._get_scale_heap_buffer(M)
        result_out = self._get_output_buffers(M, N, input_tensor.dtype, device)

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

        # Reset sync buffer before kernel launch
        assert self._barrier_done is not None
        self._barrier_done.zero_()

        def grid(META):
            return (META["COMM_SMS"],)

        # Step 1+2: Triton kernel for allreduce + rmsnorm + quant + barrier
        fused_twoshot_allreduce_rmsnorm_row_quant_kernel[grid](
            iris_input,
            quant_heap,
            scale_heap,
            result_out,
            # Barrier
            self._barrier_flags,
            self._barrier_epoch,
            self._barrier_done,
            # Residual
            residual if residual is not None else input_tensor,
            # RMSNorm
            rms_weight,
            rms_eps,
            fp8_max,
            # Dimensions
            M,
            N,
            # Strides
            iris_input.stride(0),
            iris_input.stride(1),
            quant_heap.stride(0),
            quant_heap.stride(1),
            # Iris
            heap_bases,
            rank_in_group,
            rank_global,
            world_size,
            rank_start,
            rank_stride,
            # Explicit constexprs
            BLOCK_SIZE_N,
            residual is not None,
        )

        # Step 3: hipBLASLt GEMM via torch._scaled_mm
        # quant_heap is (M, N) FP8, scale_heap is (M,) float32
        # gemm_weight is (N, K_GEMM) FP8, weight_scale is float32 scalar
        # torch._scaled_mm with row-wise scale_a (M, 1) requires
        # scale_b to be (1, K_GEMM). weight_scale is per-tensor so
        # we expand it to match.
        K_GEMM = gemm_weight.shape[1]
        scale_b = weight_scale.reshape(1, 1).expand(1, K_GEMM).contiguous()
        gemm_out = torch._scaled_mm(
            quant_heap,
            gemm_weight,
            out_dtype=out_dtype,
            scale_a=scale_heap.unsqueeze(1),
            scale_b=scale_b,
            bias=bias,
        )

        residual_out = result_out if residual is not None else None

        return gemm_out, residual_out


_iris_twoshot_row_hipblaslt_manager: Optional[
    IrisTwoshotRowHipblasltManager
] = None


def get_iris_twoshot_row_hipblaslt_manager() -> IrisTwoshotRowHipblasltManager:
    """Get the global Iris twoshot-row-hipblaslt manager instance."""
    global _iris_twoshot_row_hipblaslt_manager
    if _iris_twoshot_row_hipblaslt_manager is None:
        _iris_twoshot_row_hipblaslt_manager = IrisTwoshotRowHipblasltManager()
    return _iris_twoshot_row_hipblaslt_manager


def initialize_iris_twoshot_row_hipblaslt(
    heap_size: Optional[int] = None,
) -> None:
    """Initialize Iris for two-shot fused all-reduce with per-row quant + hipBLASLt.

    Call this once at model load time before any forward passes.

    Args:
        heap_size: Size of symmetric heap in bytes (default: 8GB)
    """
    get_iris_twoshot_row_hipblaslt_manager().initialize(heap_size)


def fused_allreduce_add_rms_row_quant_gemm_iris_twoshot_hipblaslt(
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
    """Fused AllReduce + RMSNorm + per-row FP8 Quant + hipBLASLt GEMM (two-shot).

    FP8 broadcast variant: broadcasts FP8 and per-row scales on the
    heap instead of BF16. Communication via iris Triton kernel,
    GEMM via torch._scaled_mm (hipBLASLt).

    Returns (gemm_out, residual_out).
    """
    return (
        get_iris_twoshot_row_hipblaslt_manager()
        .fused_allreduce_rmsnorm_row_quant_gemm(
            input, rms_weight, rms_eps, quant_dtype,
            gemm_weight, weight_scale, out_dtype, residual, bias,
        )
    )
