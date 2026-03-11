# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
"""
Fused AllReduce + RMSNorm + FP8 Quant + Scaled GEMM for ROCm.

Combines allreduce + residual add + rmsnorm + FP8 per-row quant into a single
Triton kernel launch using iris symmetric heap, followed by hipBLASLt GEMM
via torch._scaled_mm.

2D-tiled variant: processes BLOCK_SIZE_M rows at a time (default 1).
RMSNorm requires a full-row reduction. Since BLOCK_SIZE_N covers the
entire row, tl.sum(..., axis=1) gives the correct per-row variance.

After the Triton kernel completes:
  - quant_heap: (M, N) FP8 matrix on symmetric heap (all rows, all ranks)
  - scale_heap: (M,) per-row float32 scales on symmetric heap
  - result_out: (M, N) BF16 allreduced values (only owned rows)

Then torch._scaled_mm is called for the GEMM via hipBLASLt.

CUDA graph compatibility:
- Buffer pre-allocation with view pattern
- device_barrier() before and after kernel
- Caller must capture on a dedicated stream (not the default stream)

Also provides a reference (NCCL) implementation for correctness testing.
"""

import logging
from typing import Any, Literal, Optional, Tuple

import iris
import torch
import torch.distributed as dist
import triton
import triton.language as tl
from torch.distributed import ProcessGroup

__all__ = ["fused_allreduce_add_rms_quant_gemm"]

logger = logging.getLogger(__name__)


# ============================================================================
# Utilities
# ============================================================================


def _compute_chunk_size(comm_sms: int, num_xcds: int, swizzle_size: int = 4) -> int:
    chunk = swizzle_size * swizzle_size
    return min(chunk, comm_sms // num_xcds)


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
    # Residual strides
    stride_res_m,
    stride_res_n,
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
        rms_w = tl.load(rms_weight_ptr + rn, mask=rn < ACTUAL_N, other=0.0).to(
            tl.float32
        )
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
        res_offset = rm[:, None] * stride_res_m + rn[None, :] * stride_res_n
        scale_offset = rm * stride_sh

        is_full = row_base + BLOCK_SIZE_M * stride <= M

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
                    iris_rank,
                    remote_rank,
                    heap_bases,
                )
                acc += partial.to(tl.float32)
                if HAS_RESIDUAL and remote_rank == iris_rank:
                    res_in = tl.load(residual_in_ptr + res_offset).to(tl.float32)
                    acc += res_in

            if HAS_RESIDUAL:
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

            quantized = normed / scale[:, None]
            quantized = tl.maximum(tl.minimum(quantized, fp8_max), -fp8_max)
            quantized = quantized.to(quant_heap_ptr.type.element_ty)

            tl.store(quant_heap_ptr + qh_offset, quantized)
            tl.store(scale_heap_ptr + scale_offset, scale)

            for i in tl.static_range(0, world_size):
                remote_rank = rank_start + i * rank_stride
                if remote_rank != iris_rank:
                    iris.store(
                        quant_heap_ptr + qh_offset,
                        quantized,
                        iris_rank,
                        remote_rank,
                        heap_bases,
                    )
                    iris.store(
                        scale_heap_ptr + scale_offset,
                        scale,
                        iris_rank,
                        remote_rank,
                        heap_bases,
                    )
                    if HAS_RESIDUAL:
                        iris.store(
                            result_out_ptr + out_offset,
                            acc.to(result_out_ptr.type.element_ty),
                            iris_rank,
                            remote_rank,
                            heap_bases,
                        )
        else:
            # ---- Slow path: masked (boundary tiles or padded N) ----
            acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
            for i in tl.static_range(0, world_size):
                remote_rank = rank_start + i * rank_stride
                partial = iris.load(
                    input_ptr + input_offset,
                    iris_rank,
                    remote_rank,
                    heap_bases,
                    mask=mask,
                )
                acc += partial.to(tl.float32)
                if HAS_RESIDUAL and remote_rank == iris_rank:
                    res_in = tl.load(
                        residual_in_ptr + res_offset, mask=mask, other=0.0
                    ).to(tl.float32)
                    acc += res_in

            if HAS_RESIDUAL:
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

            quantized = normed / scale[:, None]
            quantized = tl.maximum(tl.minimum(quantized, fp8_max), -fp8_max)
            quantized = quantized.to(quant_heap_ptr.type.element_ty)

            tl.store(quant_heap_ptr + qh_offset, quantized, mask=mask)
            tl.store(scale_heap_ptr + scale_offset, scale, mask=row_mask)

            for i in tl.static_range(0, world_size):
                remote_rank = rank_start + i * rank_stride
                if remote_rank != iris_rank:
                    iris.store(
                        quant_heap_ptr + qh_offset,
                        quantized,
                        iris_rank,
                        remote_rank,
                        heap_bases,
                        mask=mask,
                    )
                    iris.store(
                        scale_heap_ptr + scale_offset,
                        scale,
                        iris_rank,
                        remote_rank,
                        heap_bases,
                        mask=row_mask,
                    )
                    if HAS_RESIDUAL:
                        iris.store(
                            result_out_ptr + out_offset,
                            acc.to(result_out_ptr.type.element_ty),
                            iris_rank,
                            remote_rank,
                            heap_bases,
                            mask=mask,
                        )

    # Post-kernel barrier is external (called by the manager).


# ============================================================================
# Manager and public API
# ============================================================================


class _AllReduceManager:
    """Singleton manager for fused AllReduce+RMSNorm+per-row Quant + hipBLASLt GEMM.

    Manages symmetric heap buffers (input, quant, scale, result_out).
    GEMM is delegated to torch._scaled_mm.
    """

    _instance: Optional["_AllReduceManager"] = None
    _initialized: bool = False

    def __new__(cls) -> "_AllReduceManager":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if _AllReduceManager._initialized:
            return
        _AllReduceManager._initialized = True

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

        # Result heap buffer (residual output, on heap for broadcast)
        self._out_result: Optional[torch.Tensor] = None
        self._out_result_M: int = 0
        self._out_result_N: int = 0
        self._out_result_dtype: Optional[torch.dtype] = None

    def initialize(self, heap_size: Optional[int] = None) -> None:
        """Initialize Iris symmetric heap (call once at startup)."""
        if self._shmem is not None:
            logger.debug("Iris (fused allreduce) already initialized, skipping")
            return

        if heap_size is not None:
            self._heap_size = heap_size

        cur_rank = (
            torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        )
        self._shmem = iris.iris(self._heap_size)

        logger.info(
            f"Iris (fused allreduce) initialized: "
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
    ) -> Tuple[Any, Any, Any, Any]:
        """Allocate or reuse symmetric heap buffers.

        All buffers live on the iris symmetric heap so the kernel can
        broadcast them to all peers via iris.store.

        Returns (input_buf, quant_heap, scale_heap, result_out).
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
        need_scale = self._scale_heap_buf is None or M > self._scale_heap_M
        need_result = (
            self._out_result is None
            or M > self._out_result_M
            or N != self._out_result_N
            or input_dtype != self._out_result_dtype
        )

        if not (need_input or need_quant or need_scale or need_result):
            return (
                self._input_buf[:M],
                self._quant_heap_buf[:M],
                self._scale_heap_buf[:M],
                self._out_result[:M],
            )

        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                f"Iris (fused allreduce): heap buffers too small for "
                f"M={M}. Cannot allocate during CUDA graph capture."
            )

        shmem = self.shmem

        if need_input:
            self._input_buf = shmem.zeros((M, N), dtype=input_dtype)
            self._input_M = M
            self._input_N = N
            self._input_dtype = input_dtype

        if need_quant:
            self._quant_heap_buf = shmem.zeros((M, N), dtype=quant_dtype)
            self._quant_heap_M = M
            self._quant_heap_N = N
            self._quant_heap_dtype = quant_dtype

        if need_scale:
            self._scale_heap_buf = shmem.zeros((M,), dtype=torch.float32)
            self._scale_heap_M = M

        if need_result:
            self._out_result = shmem.zeros((M, N), dtype=input_dtype)
            self._out_result_M = M
            self._out_result_N = N
            self._out_result_dtype = input_dtype

        return (
            self._input_buf[:M],
            self._quant_heap_buf[:M],
            self._scale_heap_buf[:M],
            self._out_result[:M],
        )

    def run(
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

        # Allocate heap buffers
        iris_input, quant_heap, scale_heap, result_out = self._ensure_heap_buffers(
            M, N, input_tensor.dtype, quant_dtype
        )

        # Copy input to symmetric heap (captured in graph)
        iris_input.copy_(input_tensor)

        # Pre-kernel barrier
        shmem.device_barrier()

        # FP8 max value
        fp8_max = torch.finfo(quant_dtype).max

        # Iris group info
        rank_global = shmem.get_rank()
        rank_in_group = rank_global
        world_size = shmem.get_num_ranks()
        rank_start = 0
        rank_stride = 1
        heap_bases = shmem.get_heap_bases()

        # ---- Tunable parameters ----
        num_xcds = iris.hip.get_num_xcc()
        BLOCK_SIZE_M = 1
        BLOCK_SIZE_N = triton.next_power_of_2(N)
        ACTUAL_N = N
        PADDED_N = BLOCK_SIZE_N != N
        DISTRIBUTION = 1  # 0=striding, 1=block
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
            residual.stride(0) if residual is not None else input_tensor.stride(0),
            residual.stride(1) if residual is not None else input_tensor.stride(1),
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

        # Post-kernel barrier
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


_manager: Optional[_AllReduceManager] = None


def _get_manager() -> _AllReduceManager:
    global _manager
    if _manager is None:
        _manager = _AllReduceManager()
    return _manager


def fused_allreduce_add_rms_quant_gemm(
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
    impl: Literal["iris", "ref"] = "iris",
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """AllReduce + RMSNorm + per-row FP8 Quant + hipBLASLt GEMM.

    Args:
        impl: "iris" (default) uses iris symmetric heap for GPU-initiated
            allreduce. "ref" uses NCCL allreduce + standard PyTorch ops.

    Returns (gemm_out, residual_out). residual_out is None when residual
    is None.
    """
    if impl == "iris":
        return _get_manager().run(
            input,
            rms_weight,
            rms_eps,
            quant_dtype,
            gemm_weight,
            weight_scale,
            out_dtype,
            residual,
            bias,
        )
    else:
        return _run_ref(
            input,
            rms_weight,
            rms_eps,
            quant_dtype,
            group_name,
            gemm_weight,
            weight_scale,
            out_dtype,
            residual=residual,
            bias=bias,
        )


# ============================================================================
# Reference implementation (NCCL baseline for correctness testing)
# ============================================================================


def _run_ref(
    input: torch.Tensor,
    rms_weight: torch.Tensor,
    rms_eps: float,
    quant_dtype: torch.dtype,
    group_name: str,
    gemm_weight: torch.Tensor,
    weight_scale: torch.Tensor,
    out_dtype: torch.dtype,
    group: Optional[ProcessGroup] = None,
    residual: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """NCCL AllReduce + Add + RMSNorm + FP8 Quant + hipBLASLt GEMM.

    Reference implementation using standard PyTorch ops. No iris or custom
    kernels. Useful for correctness testing against the fused variant.

    Returns (gemm_out, residual_out). residual_out is None when residual
    is None.
    """
    # Step 1: All-reduce in fp32 (matches fused kernel which accumulates in fp32)
    allreduce_out = input.to(torch.float32)
    dist.all_reduce(allreduce_out, group=group)

    # Step 2: Optional residual add + RMSNorm (all in fp32)
    if residual is not None:
        residual_out = (allreduce_out + residual.to(torch.float32)).to(input.dtype)
        rms_input_f32 = residual_out.to(torch.float32)
    else:
        residual_out = None
        rms_input_f32 = allreduce_out
    variance = rms_input_f32.pow(2).mean(dim=-1, keepdim=True)
    rms_input_normed = rms_input_f32 * torch.rsqrt(variance + rms_eps)
    rms_out_f32 = rms_input_normed * rms_weight.to(torch.float32)

    # Step 3: Per-row FP8 quantization (matches fused kernel: fp32 → FP8)
    fp8_max = torch.finfo(quant_dtype).max
    row_amax = rms_out_f32.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
    quant_scale_out = (row_amax / fp8_max).to(torch.float32)
    quant_out = (rms_out_f32 / quant_scale_out).clamp(-fp8_max, fp8_max).to(quant_dtype)

    # Step 4: hipBLASLt GEMM via torch._scaled_mm
    K_GEMM = gemm_weight.shape[1]
    scale_b = weight_scale.reshape(1, 1).expand(1, K_GEMM).contiguous()
    gemm_out = torch._scaled_mm(
        quant_out,
        gemm_weight,
        out_dtype=out_dtype,
        scale_a=quant_scale_out,
        scale_b=scale_b,
        bias=bias,
    )

    return gemm_out, residual_out
