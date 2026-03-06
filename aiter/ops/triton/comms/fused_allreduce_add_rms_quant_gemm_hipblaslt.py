# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Iris two-shot fused AllReduce + RMSNorm + per-row FP8 Quant, with
hipBLASLt GEMM (torch._scaled_mm).

2D-tiled variant: processes BLOCK_SIZE_M rows at a time (default 2).
RMSNorm requires a full-row reduction. Since BLOCK_SIZE_N covers the
entire row, tl.sum(..., axis=1) gives the correct per-row variance.

After the Triton kernel completes:
  - quant_heap: (M, N) FP8 matrix on symmetric heap (all rows, all ranks)
  - scale_heap: (M,) per-row float32 scales on symmetric heap
  - result_out: (M, N) BF16 allreduced values (only owned rows)

Then torch._scaled_mm is called for the GEMM via hipBLASLt.

CUDA graph compatibility:
- Buffer pre-allocation with view pattern
- device_barrier() before and after kernel (graph-capturable,
  pure device-side atomics on the symmetric heap)
"""

import logging
from typing import Any, Optional, Tuple

import iris
import torch
import triton
import triton.language as tl

from aiter.ops.triton.comms import is_graph_capturing

__all__ = ["fused_allreduce_add_rms_quant_gemm_hipblaslt"]

logger = logging.getLogger(__name__)


# ============================================================================
# Utilities
# ============================================================================


def _compute_chunk_size(
    comm_sms: int, num_xcds: int, swizzle_size: int = 4
) -> int:
    chunk = swizzle_size * swizzle_size
    return min(chunk, comm_sms // num_xcds)


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
# Fused two-shot AllReduce + RMSNorm + per-row FP8 Quant kernel (2D tiling)
# ============================================================================


@triton.jit
def persistent_fused_allreduce_rmsnorm_2d_quant_two_shot(
    # Symmetric heap buffers
    input_ptr,
    quant_heap_ptr,
    scale_heap_ptr,
    # Regular GPU memory
    result_out_ptr,
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
    # Input strides
    stride_in_m,
    stride_in_n,
    # Quant heap strides
    stride_qh_m,
    stride_qh_n,
    # Result out strides
    stride_out_m,
    stride_out_n,
    # Scale heap stride
    stride_sh,
    # Iris params
    heap_bases: tl.tensor,
    group_rank: tl.constexpr,
    iris_rank: tl.constexpr,
    world_size: tl.constexpr,
    rank_start: tl.constexpr,
    rank_stride: tl.constexpr,
    # Tile params
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    ACTUAL_N: tl.constexpr,
    HAS_RESIDUAL: tl.constexpr,
    PADDED_N: tl.constexpr,
    COMM_SMS: tl.constexpr,
    NUM_XCDS: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    DISTRIBUTION: tl.constexpr,
):
    """Two-shot fused AllReduce + RMSNorm + per-row FP8 quant (2D).

    Each rank reduces its assigned tiles, applies RMSNorm + per-row FP8
    quant, then broadcasts the FP8 result and scales to all peers.
    Barriers are external (called by the manager before/after kernel).
    """
    pid = tl.program_id(0)

    # Row distribution across ranks
    rows_per_rank = tl.cdiv(M, world_size)
    if DISTRIBUTION == 0:
        my_start = group_rank
        stride = world_size
        remaining = M - my_start
        remaining = tl.maximum(remaining, 0)
        max_row_offset = tl.cdiv(remaining, stride)
    else:
        my_start = group_rank * rows_per_rank
        stride = 1
        remaining = M - my_start
        remaining = tl.maximum(remaining, 0)
        max_row_offset = tl.minimum(rows_per_rank, remaining)

    # Number of tile iterations (each tile covers BLOCK_SIZE_M rows)
    num_tiles = tl.cdiv(max_row_offset, BLOCK_SIZE_M)

    # Load RMSNorm weights
    rn = tl.arange(0, BLOCK_SIZE_N)
    if PADDED_N:
        rms_w = tl.load(
            rms_weight_ptr + rn, mask=rn < ACTUAL_N, other=0.0
        ).to(tl.float32)
    else:
        rms_w = tl.load(rms_weight_ptr + rn).to(tl.float32)

    # Persistent traversal over this rank's assigned tiles
    for tile_offset in range(pid, num_tiles, COMM_SMS):
        row_base = my_start + tile_offset * BLOCK_SIZE_M * stride

        # Build 2D indices
        rm = row_base + tl.arange(0, BLOCK_SIZE_M) * stride
        row_mask = rm < M

        input_offset = rm[:, None] * stride_in_m + rn[None, :] * stride_in_n
        qh_offset = rm[:, None] * stride_qh_m + rn[None, :] * stride_qh_n
        out_offset = rm[:, None] * stride_out_m + rn[None, :] * stride_out_n
        scale_offset = rm * stride_sh

        is_full = (row_base + BLOCK_SIZE_M * stride <= M)

        if PADDED_N:
            col_mask = rn < ACTUAL_N
            mask = row_mask[:, None] & col_mask[None, :]
        else:
            mask = row_mask[:, None]

        if is_full and not PADDED_N:
            # ---- Fast path: no masks ----
            acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
            for i in tl.static_range(0, world_size):
                remote_rank = rank_start + i * rank_stride
                partial = iris.load(
                    input_ptr + input_offset,
                    iris_rank, remote_rank, heap_bases,
                )
                acc += partial.to(tl.float32)
                if HAS_RESIDUAL and remote_rank == iris_rank:
                    res_in = tl.load(
                        residual_in_ptr + out_offset
                    ).to(tl.float32)
                    acc += res_in

            tl.store(
                result_out_ptr + out_offset,
                acc.to(result_out_ptr.type.element_ty),
            )

            sq_sum = tl.sum(acc * acc, axis=1)
            rrms = tl.rsqrt(sq_sum / N + rms_eps)
            normed = acc * rrms[:, None] * rms_w[None, :]

            row_amax = tl.max(tl.abs(normed), axis=1)
            row_amax = tl.maximum(row_amax, 1e-12)
            scale = row_amax / fp8_max

            quantized = (normed / scale[:, None]).to(
                quant_heap_ptr.type.element_ty
            )

            tl.store(quant_heap_ptr + qh_offset, quantized)
            tl.store(scale_heap_ptr + scale_offset, scale)

            for i in tl.static_range(0, world_size):
                remote_rank = rank_start + i * rank_stride
                if remote_rank != iris_rank:
                    iris.store(
                        quant_heap_ptr + qh_offset, quantized,
                        iris_rank, remote_rank, heap_bases,
                    )
                    iris.store(
                        scale_heap_ptr + scale_offset, scale,
                        iris_rank, remote_rank, heap_bases,
                    )
        else:
            # ---- Slow path: masked (boundary tiles or padded N) ----
            acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
            for i in tl.static_range(0, world_size):
                remote_rank = rank_start + i * rank_stride
                partial = iris.load(
                    input_ptr + input_offset,
                    iris_rank, remote_rank, heap_bases,
                    mask=mask,
                )
                acc += partial.to(tl.float32)
                if HAS_RESIDUAL and remote_rank == iris_rank:
                    res_in = tl.load(
                        residual_in_ptr + out_offset, mask=mask, other=0.0
                    ).to(tl.float32)
                    acc += res_in

            tl.store(
                result_out_ptr + out_offset,
                acc.to(result_out_ptr.type.element_ty),
                mask=mask,
            )

            sq_sum = tl.sum(acc * acc, axis=1)
            rrms = tl.rsqrt(sq_sum / N + rms_eps)
            normed = acc * rrms[:, None] * rms_w[None, :]

            row_amax = tl.max(tl.abs(normed), axis=1)
            row_amax = tl.maximum(row_amax, 1e-12)
            scale = row_amax / fp8_max

            quantized = (normed / scale[:, None]).to(
                quant_heap_ptr.type.element_ty
            )

            tl.store(quant_heap_ptr + qh_offset, quantized, mask=mask)
            tl.store(scale_heap_ptr + scale_offset, scale, mask=row_mask)

            for i in tl.static_range(0, world_size):
                remote_rank = rank_start + i * rank_stride
                if remote_rank != iris_rank:
                    iris.store(
                        quant_heap_ptr + qh_offset, quantized,
                        iris_rank, remote_rank, heap_bases,
                        mask=mask,
                    )
                    iris.store(
                        scale_heap_ptr + scale_offset, scale,
                        iris_rank, remote_rank, heap_bases,
                        mask=row_mask,
                    )

    # Post-kernel barrier is external (called by the manager).


# ============================================================================
# Manager and public API
# ============================================================================


class IrisTwoshot2dHipblasltManager:
    """Singleton manager for 2D-tiled two-shot AllReduce+RMSNorm+per-row Quant
    + hipBLASLt GEMM.

    Manages symmetric heap buffers (input, quant, scale) and GPU memory
    (result_out). GEMM is delegated to torch._scaled_mm.
    """

    _instance: Optional["IrisTwoshot2dHipblasltManager"] = None
    _initialized: bool = False

    def __new__(cls) -> "IrisTwoshot2dHipblasltManager":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if IrisTwoshot2dHipblasltManager._initialized:
            return
        IrisTwoshot2dHipblasltManager._initialized = True

        self._shmem: Any = None
        self._heap_size: int = 2**33  # 8GB default

        # Symmetric heap buffers
        self._input_buf: Any = None
        self._input_M: int = 0
        self._input_N: int = 0
        self._input_dtype: Optional[torch.dtype] = None

        self._quant_heap_buf: Any = None
        self._quant_heap_M: int = 0
        self._quant_heap_N: int = 0
        self._quant_heap_dtype: Optional[torch.dtype] = None

        self._scale_heap_buf: Any = None
        self._scale_heap_M: int = 0

        # Regular GPU memory
        self._out_result: Optional[torch.Tensor] = None

    def initialize(self, heap_size: Optional[int] = None) -> None:
        """Initialize Iris symmetric heap (call once at startup)."""
        if self._shmem is not None:
            logger.debug(
                "Iris (twoshot-2d-hipblaslt) already initialized, skipping"
            )
            return

        if heap_size is not None:
            self._heap_size = heap_size

        cur_rank = (
            torch.distributed.get_rank()
            if torch.distributed.is_initialized()
            else 0
        )
        self._shmem = iris.iris(self._heap_size)

        logger.info(
            f"Iris (twoshot-2d-hipblaslt) initialized: "
            f"rank={cur_rank}, heap_size={self._heap_size / 2**30:.1f}GB"
        )

    @property
    def shmem(self) -> Any:
        if self._shmem is None:
            self.initialize()
        return self._shmem

    def _ensure_heap_buffers(
        self,
        M: int,
        N: int,
        input_dtype: torch.dtype,
        quant_dtype: torch.dtype,
    ) -> Tuple[Any, Any, Any]:
        """Allocate or reuse symmetric heap buffers.

        Returns (input_buf, quant_heap, scale_heap).
        """
        need_input = (
            self._input_buf is None
            or M > self._input_M
            or N != self._input_N
            or input_dtype != self._input_dtype
        )
        need_quant = (
            self._quant_heap_buf is None
            or M > self._quant_heap_M
            or N != self._quant_heap_N
            or quant_dtype != self._quant_heap_dtype
        )
        need_scale = (
            self._scale_heap_buf is None
            or M > self._scale_heap_M
        )

        if not (need_input or need_quant or need_scale):
            return (
                self._input_buf[:M],
                self._quant_heap_buf[:M],
                self._scale_heap_buf[:M],
            )

        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                f"Iris (twoshot-2d-hipblaslt): heap buffers too small for "
                f"M={M}. Cannot allocate during CUDA graph capture."
            )

        shmem = self.shmem
        cur_rank = (
            torch.distributed.get_rank()
            if torch.distributed.is_initialized()
            else 0
        )

        if need_input:
            self._input_buf = shmem.zeros((M, N), dtype=input_dtype)
            self._input_M = M
            self._input_N = N
            self._input_dtype = input_dtype
            logger.debug(
                f"Iris (twoshot-2d-hipblaslt): allocated input buffer "
                f"({M}, {N}), dtype={input_dtype}, rank={cur_rank}"
            )

        if need_quant:
            self._quant_heap_buf = shmem.zeros((M, N), dtype=quant_dtype)
            self._quant_heap_M = M
            self._quant_heap_N = N
            self._quant_heap_dtype = quant_dtype
            logger.debug(
                f"Iris (twoshot-2d-hipblaslt): allocated quant heap buffer "
                f"({M}, {N}), dtype={quant_dtype}, rank={cur_rank}"
            )

        if need_scale:
            self._scale_heap_buf = shmem.zeros((M,), dtype=torch.float32)
            self._scale_heap_M = M
            logger.debug(
                f"Iris (twoshot-2d-hipblaslt): allocated scale heap buffer "
                f"({M},), dtype=float32, rank={cur_rank}"
            )

        return (
            self._input_buf[:M],
            self._quant_heap_buf[:M],
            self._scale_heap_buf[:M],
        )

    def _ensure_gpu_buffers(
        self,
        M: int,
        N: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        """Allocate or reuse result_out buffer on GPU memory."""
        need_result = (
            self._out_result is None
            or M > self._out_result.shape[0]
            or N != self._out_result.shape[1]
        )

        if not need_result:
            assert self._out_result is not None
            return self._out_result[:M]

        if torch.cuda.is_current_stream_capturing():
            existing_M = (
                self._out_result.shape[0]
                if self._out_result is not None
                else 0
            )
            raise RuntimeError(
                f"Iris (twoshot-2d-hipblaslt): GPU buffers too small for "
                f"M={M} (allocated M={existing_M}). Cannot allocate during "
                f"CUDA graph capture."
            )

        cur_rank = (
            torch.distributed.get_rank()
            if torch.distributed.is_initialized()
            else 0
        )

        self._out_result = torch.empty(
            (M, N), dtype=dtype, device=device,
        )
        logger.debug(
            f"Iris (twoshot-2d-hipblaslt): allocated result_out "
            f"({M}, {N}), dtype={dtype}, rank={cur_rank}"
        )

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

        Returns (gemm_out, residual_out). residual_out is None when
        residual is None.
        """
        shmem = self.shmem
        M, N = input_tensor.shape
        device = input_tensor.device

        # Allocate buffers
        iris_input, quant_heap, scale_heap = self._ensure_heap_buffers(
            M, N, input_tensor.dtype, quant_dtype,
        )
        result_out = self._ensure_gpu_buffers(
            M, N, input_tensor.dtype, device,
        )

        # Copy input to symmetric heap (captured in graph)
        iris_input.copy_(input_tensor)

        # Pre-kernel barrier: ensure all ranks have copied input to heap
        # before any rank's kernel starts reading from peers.
        if not is_graph_capturing():
            shmem.device_barrier()

        # FP8 max value
        fp8_max = torch.finfo(quant_dtype).max

        # Iris group info
        rank_in_group, rank_global, world_size, rank_start, rank_stride = (
            extract_group_info(shmem)
        )
        heap_bases = shmem.get_heap_bases()

        # ---- Tunable parameters ----
        num_xcds = iris.hip.get_num_xcc()
        BLOCK_SIZE_M = 1
        BLOCK_SIZE_N = triton.next_power_of_2(N)
        ACTUAL_N = N
        PADDED_N = (BLOCK_SIZE_N != N)
        DISTRIBUTION = 1       # 0=striding, 1=block
        COMM_SMS = 128
        CHUNK_SIZE = _compute_chunk_size(COMM_SMS, num_xcds)
        NUM_WARPS = 16
        NUM_STAGES = 2
        WAVES_PER_EU = 1
        # ---- End tunable parameters ----

        # Launch kernel
        persistent_fused_allreduce_rmsnorm_2d_quant_two_shot[(COMM_SMS,)](
            iris_input,
            quant_heap,
            scale_heap,
            result_out,
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
            result_out.stride(0),
            result_out.stride(1),
            scale_heap.stride(0),
            # Iris
            heap_bases,
            rank_in_group,
            rank_global,
            world_size,
            rank_start,
            rank_stride,
            # Tile params
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
            ACTUAL_N,
            residual is not None,  # HAS_RESIDUAL
            PADDED_N,
            COMM_SMS,
            num_xcds,
            CHUNK_SIZE,
            DISTRIBUTION,
            num_warps=NUM_WARPS,
            num_stages=NUM_STAGES,
            waves_per_eu=WAVES_PER_EU,
        )

        # Post-kernel barrier: ensure all ranks have finished writing
        # FP8 data + scales to the heap before any rank overwrites its
        # input buffer on the next iteration.
        if not is_graph_capturing():
            shmem.device_barrier()

        # hipBLASLt GEMM
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


_manager: Optional[IrisTwoshot2dHipblasltManager] = None


def _get_manager() -> IrisTwoshot2dHipblasltManager:
    global _manager
    if _manager is None:
        _manager = IrisTwoshot2dHipblasltManager()
    return _manager


def initialize_iris_twoshot_2d_hipblaslt(
    heap_size: Optional[int] = None,
) -> None:
    """Initialize Iris for 2D-tiled two-shot fused all-reduce.

    Call once at model load time before any forward passes.
    """
    _get_manager().initialize(heap_size)


def fused_allreduce_add_rms_quant_gemm_hipblaslt(
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
    """Fused AllReduce + RMSNorm + per-row FP8 Quant + hipBLASLt GEMM.

    2D-tiled two-shot variant. Returns (gemm_out, residual_out).
    """
    return (
        _get_manager()
        .fused_allreduce_rmsnorm_row_quant_gemm(
            input, rms_weight, rms_eps, quant_dtype,
            gemm_weight, weight_scale, out_dtype, residual, bias,
        )
    )
