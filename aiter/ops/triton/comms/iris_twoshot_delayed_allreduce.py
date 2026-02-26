# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Iris two-shot fused AllReduce + RMSNorm + delayed-scaling FP8 Quant.

Single persistent kernel:
  Step 1 (Two-shot AllReduce + fused RMSNorm + quant):
    Each rank reduces its assigned rows by gathering from all ranks,
    then broadcasts reduced rows to all peers. For assigned rows, the
    data stays in registers after reduction, so RMSNorm + FP8
    quantization happen immediately using the PREVIOUS call's scale.
  Step 2 (Inlined device barrier): CTA 0 performs cross-rank
    synchronization via atomic ops on the symmetric heap. Other CTAs
    spin on a local flag until CTA 0 signals completion.
  Step 3 (RMSNorm + quant for remaining rows):
    Process rows NOT assigned to this rank (received via broadcast),
    again quantizing with the previous scale.

Delayed scaling:
  Quantization uses the scale from the PREVIOUS call (prev_scale).
  The kernel computes the current iteration's amax as a side-effect
  (via per-CTA local max + atomicMax to a global accumulator). After
  the kernel, the Python wrapper derives the new scale for the NEXT
  call: new_scale = amax / fp8_max.

  This eliminates BOTH the cross-CTA barrier for global amax AND the
  separate quant pass, while outputting a per-tensor scale (1,) that
  is compatible with downstream scaled GEMMs.

  The first call uses prev_scale=1.0, so the first iteration's quant
  may clip. In practice, CUDA graph warmup runs absorb this.

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
    IRIS_TWOSHOT_AUTOTUNE_KEYS,
    _inlined_device_barrier,
    chiplet_transform_chunked,
    extract_group_info,
    get_iris_twoshot_configs,
)

__all__ = ["fused_allreduce_add_rms_delayed_quant_iris_twoshot"]

logger = logging.getLogger(__name__)

AUTOTUNE = False
if AUTOTUNE:
    os.environ["TRITON_PRINT_AUTOTUNING"] = "1"


iris_twoshot_delayed_autotune_configs = get_iris_twoshot_configs(AUTOTUNE)


# ============================================================================
# Helpers
# ============================================================================


@triton.jit
def _rmsnorm_delayed_quant(
    rms_in,
    rms_w,
    rms_eps,
    prev_scale,
    rms_out_ptr,
    quant_out_ptr,
    residual_in_ptr,
    residual_out_ptr,
    out_offset,
    N,
    col_mask,
    rms_out_dtype: tl.constexpr,
    quant_out_dtype: tl.constexpr,
    residual_out_dtype: tl.constexpr,
    HAS_RESIDUAL: tl.constexpr,
):
    """RMSNorm + delayed-scaling FP8 quantization for a single row.

    Quantizes using prev_scale (from previous call). Returns the row's
    amax so the caller can accumulate it for the next call's scale.
    """
    # Optional residual addition
    if HAS_RESIDUAL:
        res_in = tl.load(
            residual_in_ptr + out_offset, mask=col_mask, other=0.0
        ).to(tl.float32)
        rms_in = rms_in + res_in
        tl.store(
            residual_out_ptr + out_offset,
            rms_in.to(residual_out_dtype),
            mask=col_mask,
        )

    # RMSNorm
    sq_sum = tl.sum(rms_in * rms_in, axis=0)
    variance = sq_sum / N
    rrms = tl.rsqrt(variance + rms_eps)
    normed = rms_in * rrms * rms_w

    tl.store(
        rms_out_ptr + out_offset,
        normed.to(rms_out_dtype),
        mask=col_mask,
    )

    # Delayed-scaling quantize: use prev_scale, track amax
    row_amax = tl.max(tl.abs(normed), axis=0)
    quantized = (normed / prev_scale).to(quant_out_dtype)
    tl.store(quant_out_ptr + out_offset, quantized, mask=col_mask)

    return row_amax


# ============================================================================
# Fused two-shot AllReduce + RMSNorm + delayed-scaling FP8 Quant kernel
# ============================================================================


@triton.autotune(
    configs=iris_twoshot_delayed_autotune_configs,
    key=IRIS_TWOSHOT_AUTOTUNE_KEYS,
    use_cuda_graph=True,
)
@triton.jit
def fused_twoshot_allreduce_rmsnorm_delayed_quant_kernel(
    # Input (in symmetric heap for iris.load)
    input_ptr,
    # Allreduce output (in symmetric heap for iris.store broadcast)
    allreduce_out_ptr,
    # Outputs (regular GPU memory)
    rms_out_ptr,
    quant_out_ptr,
    # Delayed scaling state
    prev_scale_ptr,   # (1,) -- scale from previous call, used for quant
    amax_out_ptr,     # (1,) -- accumulated amax, zeroed before launch
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
    """Fused two-shot AllReduce + RMSNorm + delayed-scaling FP8 quant.

    Delayed scaling uses prev_scale for quantization and accumulates
    the current amax for the next call. Outputs per-tensor scale (1,).
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

    # Load previous scale (scalar, same for all rows)
    prev_scale = tl.load(prev_scale_ptr)
    prev_scale = tl.maximum(prev_scale, 1e-12)

    # Track max amax across all rows processed by this CTA
    cta_amax = 0.0

    # ================================================================
    # Step 1: Two-shot AllReduce (reduce + broadcast + fused norm+quant)
    # ================================================================

    # Block distribution of rows across ranks
    rows_per_rank = tl.cdiv(M, world_size)
    my_start = group_rank * rows_per_rank
    my_end = tl.minimum(my_start + rows_per_rank, M)

    for row in range(my_start + pid, my_end, COMM_SMS):
        in_offset = row * stride_in_m + cols * stride_in_n
        ar_offset = row * stride_ar_m + cols * stride_ar_n
        out_offset = row * N + cols

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

        # Store locally
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

        # Fused RMSNorm + delayed-scaling quant (data already in acc)
        row_amax = _rmsnorm_delayed_quant(
            acc, rms_w, rms_eps, prev_scale,
            rms_out_ptr, quant_out_ptr,
            residual_in_ptr, residual_out_ptr,
            out_offset, N, col_mask,
            rms_out_ptr.type.element_ty,
            quant_out_ptr.type.element_ty,
            residual_out_ptr.type.element_ty,
            HAS_RESIDUAL,
        )
        cta_amax = tl.maximum(cta_amax, row_amax)

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
    # Step 3: RMSNorm + delayed-scaling quant for remaining rows
    # ================================================================

    for row in range(pid, M, COMM_SMS):
        # Only process rows NOT assigned to this rank
        if row < my_start or row >= my_end:
            ar_offset = row * stride_ar_m + cols * stride_ar_n
            out_offset = row * N + cols

            ar_val = tl.load(
                allreduce_out_ptr + ar_offset, mask=col_mask, other=0.0
            ).to(tl.float32)

            row_amax = _rmsnorm_delayed_quant(
                ar_val, rms_w, rms_eps, prev_scale,
                rms_out_ptr, quant_out_ptr,
                residual_in_ptr, residual_out_ptr,
                out_offset, N, col_mask,
                rms_out_ptr.type.element_ty,
                quant_out_ptr.type.element_ty,
                residual_out_ptr.type.element_ty,
                HAS_RESIDUAL,
            )
            cta_amax = tl.maximum(cta_amax, row_amax)

    # ================================================================
    # Step 4: Accumulate this CTA's amax into the global accumulator
    # ================================================================

    tl.atomic_max(amax_out_ptr, cta_amax)


# ============================================================================
# Manager and public API
# ============================================================================


class IrisTwoshotDelayedManager:
    """Singleton manager for two-shot AllReduce+RMSNorm+delayed-scaling Quant.

    Maintains prev_scale (1,) across calls. The first call uses
    prev_scale=1.0. Each subsequent call uses the scale derived from
    the previous call's amax.
    """

    _instance: Optional["IrisTwoshotDelayedManager"] = None
    _initialized: bool = False

    def __new__(cls) -> "IrisTwoshotDelayedManager":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if IrisTwoshotDelayedManager._initialized:
            return
        IrisTwoshotDelayedManager._initialized = True

        self._shmem: Any = None
        self._heap_size: int = 2**33  # 8GB default

        # Pre-allocated input buffer (iris symmetric heap)
        self._input_buf: Any = None
        self._input_M: int = 0
        self._input_N: int = 0
        self._input_dtype: Optional[torch.dtype] = None

        # Pre-allocated allreduce output buffer (iris symmetric heap)
        self._ar_output_buf: Any = None
        self._ar_output_M: int = 0

        # Pre-allocated output buffers (regular GPU memory)
        self._out_rms: Optional[torch.Tensor] = None
        self._out_quant: Optional[torch.Tensor] = None
        self._out_residual: Optional[torch.Tensor] = None
        self._out_quant_dtype: Optional[torch.dtype] = None

        # Delayed scaling state
        self._prev_scale: Optional[torch.Tensor] = None   # (1,), persistent
        self._current_amax: Optional[torch.Tensor] = None  # (1,), zeroed each call

        # Inlined device barrier state
        self._barrier_flags: Any = None  # on symmetric heap
        self._barrier_epoch: int = 0
        self._barrier_done: Optional[torch.Tensor] = None  # local GPU memory

    def initialize(self, heap_size: Optional[int] = None) -> None:
        """Initialize Iris symmetric heap (call once at startup)."""
        if self._shmem is not None:
            logger.debug("Iris (twoshot-delayed) already initialized, skipping")
            return

        if heap_size is not None:
            self._heap_size = heap_size

        cur_rank = (
            torch.distributed.get_rank()
            if torch.distributed.is_initialized()
            else 0
        )
        logger.info(
            "Initializing Iris (twoshot-delayed) symmetric heap: "
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
            f"Iris (twoshot-delayed) initialized successfully on rank {cur_rank}"
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
                f"Iris (twoshot-delayed): input buffer too small for M={M} "
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
            f"Iris (twoshot-delayed): allocated input buffer ({M}, {N}), "
            f"dtype={dtype}, rank={cur_rank}"
        )

        return self._input_buf

    def _get_ar_output_buffer(
        self,
        M: int,
        N: int,
        dtype: torch.dtype,
    ) -> Any:
        """Get a view into the pre-allocated iris allreduce output buffer."""
        if (self._ar_output_buf is not None
                and M <= self._ar_output_M
                and N == self._input_N
                and dtype == self._input_dtype):
            return self._ar_output_buf[:M]

        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                f"Iris (twoshot-delayed): allreduce output buffer too small for "
                f"M={M} (allocated {self._ar_output_M}). Cannot allocate "
                f"during CUDA graph capture."
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
            f"Iris (twoshot-delayed): allocated allreduce output buffer "
            f"({M}, {N}), dtype={dtype}, rank={cur_rank}"
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
                    f"Iris (twoshot-delayed): output buffers too small for M={M} "
                    f"(allocated {existing}). Cannot allocate during "
                    f"CUDA graph capture."
                )

            self._out_rms = torch.empty((M, N), dtype=dtype, device=device)
            self._out_quant = torch.empty((M, N), dtype=quant_dtype, device=device)
            self._out_residual = torch.empty((M, N), dtype=dtype, device=device)
            self._out_quant_dtype = quant_dtype

            # Delayed scaling: prev_scale starts at 1.0, current_amax zeroed each call
            if self._prev_scale is None:
                self._prev_scale = torch.ones(1, dtype=torch.float32, device=device)
            if self._current_amax is None:
                self._current_amax = torch.zeros(1, dtype=torch.float32, device=device)

            # Inlined barrier: CTA 0 -> other CTAs signal
            self._barrier_done = torch.zeros(1, dtype=torch.int32, device=device)

            cur_rank = (
                torch.distributed.get_rank()
                if torch.distributed.is_initialized()
                else 0
            )
            logger.info(
                f"Iris (twoshot-delayed): allocated output buffers ({M}, {N}), "
                f"dtype={dtype}, rank={cur_rank}"
            )

        assert self._out_rms is not None
        assert self._out_quant is not None
        assert self._out_residual is not None

        return (
            self._out_rms[:M],
            self._out_residual[:M] if has_residual else None,
            self._out_quant[:M],
        )

    def fused_allreduce_rmsnorm_delayed_quant(
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
        """Two-shot AllReduce + RMSNorm + delayed-scaling FP8 quant.

        Args:
            input_tensor: Input tensor (M, N) on GPU
            rms_weight: RMSNorm weight (N,)
            rms_eps: RMSNorm epsilon
            quant_dtype: FP8 dtype (e.g. torch.float8_e4m3fn)
            residual: Optional residual tensor (M, N)

        Returns:
            (allreduce_out, rms_out, residual_out, quant_out, scale_out)
            residual_out is None when residual is None.
            scale_out shape is (1,) -- per-tensor, compatible with scaled GEMMs.
        """
        shmem = self.shmem
        M, N = input_tensor.shape
        device = input_tensor.device

        # Get pre-allocated buffers
        iris_input = self._get_input_buffer(M, N, input_tensor.dtype)
        ar_output = self._get_ar_output_buffer(M, N, input_tensor.dtype)
        rms_out, residual_out, quant_out = (
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

        # Reset sync buffer and amax accumulator before kernel launch
        assert self._barrier_done is not None
        assert self._prev_scale is not None
        assert self._current_amax is not None
        self._barrier_done.zero_()
        self._current_amax.zero_()

        def grid(META):
            return (META["COMM_SMS"],)

        # Single fused kernel
        fused_twoshot_allreduce_rmsnorm_delayed_quant_kernel[grid](
            iris_input,
            ar_output,
            rms_out,
            quant_out,
            self._prev_scale,
            self._current_amax,
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

        # Derive scale_out from accumulated amax (for downstream GEMM)
        # and update prev_scale for the next call
        scale_out = self._current_amax.clamp(min=1e-12) / fp8_max
        self._prev_scale.copy_(scale_out)

        # Return allreduce_out as a view from the heap buffer
        allreduce_out = ar_output[:M]

        return allreduce_out, rms_out, residual_out, quant_out, scale_out


_iris_twoshot_delayed_manager: Optional[IrisTwoshotDelayedManager] = None


def get_iris_twoshot_delayed_manager() -> IrisTwoshotDelayedManager:
    """Get the global Iris twoshot-delayed manager instance."""
    global _iris_twoshot_delayed_manager
    if _iris_twoshot_delayed_manager is None:
        _iris_twoshot_delayed_manager = IrisTwoshotDelayedManager()
    return _iris_twoshot_delayed_manager


def initialize_iris_twoshot_delayed(heap_size: Optional[int] = None) -> None:
    """Initialize Iris for two-shot fused all-reduce with delayed scaling.

    Call this once at model load time before any forward passes.

    Args:
        heap_size: Size of symmetric heap in bytes (default: 8GB)
    """
    get_iris_twoshot_delayed_manager().initialize(heap_size)


def fused_allreduce_add_rms_delayed_quant_iris_twoshot(
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
    """Fused AllReduce + Add + RMSNorm + delayed-scaling FP8 Quant (two-shot).

    Two-shot allreduce with inlined device barrier and fused
    RMSNorm + delayed-scaling FP8 quantization in a single kernel.
    scale_out shape is (1,) -- per-tensor, compatible with scaled GEMMs.
    """
    iris_mgr = get_iris_twoshot_delayed_manager()
    return iris_mgr.fused_allreduce_rmsnorm_delayed_quant(
        input, rms_weight, rms_eps, quant_dtype, residual,
    )
