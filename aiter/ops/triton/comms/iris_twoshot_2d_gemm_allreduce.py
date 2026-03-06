# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Iris two-shot fused AllReduce + RMSNorm + per-row FP8 Quant + inlined GEMM.
2D-tiled variant with external pre/post barriers and inlined mid-barrier.

Based on iris_twoshot_2d_hipblaslt but replaces the external hipBLASLt GEMM
with an inlined Triton GEMM. This eliminates the separate GEMM kernel launch,
the rocclr_copyBuffer overhead from scale_b expansion, and reads FP8 data
directly from the heap without an intermediate round-trip.

Single Triton kernel with 3 phases:
  Phase 1 (Comm): Two-shot allreduce + RMSNorm + per-row FP8 quant +
    FP8 broadcast to all peers via symmetric heap.
  Phase 2 (Barrier): Inlined cross-rank device barrier via atomic ops.
  Phase 3 (GEMM): Persistent tiled matmul reading FP8 + per-row scales
    directly from heap. Produces BF16 output.

External barriers (pre/post kernel) are skipped during CUDA graph capture.
The inlined mid-barrier (between comm and GEMM) is always active.

CUDA graph compatibility:
- Buffer pre-allocation with view pattern
- External device_barrier() before and after kernel (skipped during capture)
- Inlined barrier between comm and GEMM phases
"""

import logging
from typing import Any, Optional, Tuple

import iris
import torch
import triton
import triton.language as tl

from aiter.ops.triton.comms import is_graph_capturing

__all__ = ["fused_allreduce_add_rms_row_quant_gemm_iris_twoshot_2d_gemm"]

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
    """Extract rank/group info from iris shmem context."""
    rank_global = shmem.get_rank()
    rank_in_group = rank_global
    world_size = shmem.get_num_ranks()
    rank_start = 0
    rank_stride = 1
    return rank_in_group, rank_global, world_size, rank_start, rank_stride


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
    barrier_epoch_ptr,
    barrier_done_ptr,
    heap_bases: tl.tensor,
    iris_rank: tl.constexpr,
    world_size: tl.constexpr,
    rank_start: tl.constexpr,
    rank_stride: tl.constexpr,
):
    """Cross-rank device barrier for use inside persistent kernels.

    CTA 0 signals own readiness, polls remote ranks, then signals
    other CTAs via barrier_done_ptr. Epoch is self-advancing so it
    works correctly in autotuning and CUDA graph replay.
    """
    barrier_epoch = tl.load(barrier_epoch_ptr)
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

        tl.store(barrier_epoch_ptr, target_epoch)
        tl.atomic_add(barrier_done_ptr, 1, sem="release")
    else:
        while tl.atomic_add(barrier_done_ptr, 0, sem="acquire") < 1:
            pass


# ============================================================================
# Fused kernel: AllReduce + RMSNorm + per-row FP8 Quant + inlined GEMM
# ============================================================================


@triton.jit
def persistent_fused_allreduce_rmsnorm_2d_quant_gemm(
    # Symmetric heap buffers
    input_ptr,
    quant_heap_ptr,
    scale_heap_ptr,
    # Regular GPU memory
    result_out_ptr,
    # GEMM parameters
    gemm_weight_ptr,
    gemm_out_ptr,
    bias_ptr,
    weight_scale_ptr,
    K_GEMM,
    stride_gw_k,
    stride_gw_n,
    stride_go_m,
    stride_go_k,
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
    HAS_BIAS: tl.constexpr,
    PADDED_N: tl.constexpr,
    COMM_SMS: tl.constexpr,
    NUM_XCDS: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    DISTRIBUTION: tl.constexpr,
    # GEMM tile params
    GEMM_BLOCK_M: tl.constexpr,
    GEMM_BLOCK_N: tl.constexpr,
    GEMM_BLOCK_K: tl.constexpr,
    GEMM_GROUP_SIZE_M: tl.constexpr,
    EVEN_GEMM_K: tl.constexpr,
):
    """Fused AllReduce + RMSNorm + per-row FP8 quant + inlined GEMM (2D).

    Phase 1: Each rank reduces its assigned tiles, applies RMSNorm + per-row
    FP8 quant, broadcasts FP8 result and scales to all peers.
    Phase 2: Inlined cross-rank barrier.
    Phase 3: Persistent tiled GEMM reading FP8 + scales from heap.
    """
    pid = tl.program_id(0)

    # ================================================================
    # Phase 1: Two-shot AllReduce + RMSNorm + Quant + FP8 Broadcast
    # ================================================================

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
            # ---- Slow path: masked ----
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

    # ================================================================
    # Phase 2: Inlined cross-rank device barrier
    # ================================================================

    _inlined_device_barrier(
        pid,
        barrier_flags_ptr,
        barrier_epoch_ptr,
        barrier_done_ptr,
        heap_bases,
        iris_rank,
        world_size,
        rank_start,
        rank_stride,
    )

    # ================================================================
    # Phase 3: Inlined GEMM — read FP8 + scales from heap
    # ================================================================
    # After barrier, quant_heap_ptr has complete (M, N) FP8 matrix and
    # scale_heap_ptr has per-row scales for ALL rows on every rank.

    num_pid_m = tl.cdiv(M, GEMM_BLOCK_M)
    num_pid_n = tl.cdiv(K_GEMM, GEMM_BLOCK_N)
    total_tiles = num_pid_m * num_pid_n

    for tile_id in range(pid, total_tiles, COMM_SMS):
        # Map tile_id -> (pid_m, pid_n) with GROUP_SIZE_M swizzle
        if GEMM_GROUP_SIZE_M == 1:
            pid_m = tile_id // num_pid_n
            pid_n = tile_id % num_pid_n
        else:
            num_pid_in_group = GEMM_GROUP_SIZE_M * num_pid_n
            group_id = tile_id // num_pid_in_group
            first_pid_m = group_id * GEMM_GROUP_SIZE_M
            group_size_m = tl.minimum(
                num_pid_m - first_pid_m, GEMM_GROUP_SIZE_M
            )
            pid_m = first_pid_m + (tile_id % group_size_m)
            pid_n = (tile_id % num_pid_in_group) // group_size_m

        offs_am = pid_m * GEMM_BLOCK_M + tl.arange(0, GEMM_BLOCK_M)
        offs_bn = pid_n * GEMM_BLOCK_N + tl.arange(0, GEMM_BLOCK_N)
        offs_k = tl.arange(0, GEMM_BLOCK_K)

        # A pointers: quant_heap (M, N) FP8
        a_ptrs = (
            quant_heap_ptr
            + (offs_am[:, None] % M) * stride_qh_m
            + offs_k[None, :] * stride_qh_n
        )
        # B pointers: gemm_weight (N, K_GEMM) FP8
        b_ptrs = (
            gemm_weight_ptr
            + offs_k[:, None] * stride_gw_k
            + (offs_bn[None, :] % K_GEMM) * stride_gw_n
        )

        # Per-row activation scales
        a_scale = tl.load(
            scale_heap_ptr + (offs_am % M) * stride_sh
        )

        # Tiled GEMM accumulation
        accumulator = tl.zeros(
            (GEMM_BLOCK_M, GEMM_BLOCK_N), dtype=tl.float32
        )

        for k in range(0, tl.cdiv(N, GEMM_BLOCK_K)):
            if EVEN_GEMM_K:
                a = tl.load(a_ptrs)
                b = tl.load(b_ptrs)
            else:
                k_remaining = N - k * GEMM_BLOCK_K
                a = tl.load(
                    a_ptrs,
                    mask=offs_k[None, :] < k_remaining,
                    other=0.0,
                )
                b = tl.load(
                    b_ptrs,
                    mask=offs_k[:, None] < k_remaining,
                    other=0.0,
                )

            accumulator += tl.dot(a, b, input_precision="ieee")

            a_ptrs += GEMM_BLOCK_K * stride_qh_n
            b_ptrs += GEMM_BLOCK_K * stride_gw_k

        # Apply scales: per-row activation scale * per-tensor weight scale
        ws = tl.load(weight_scale_ptr)
        accumulator *= a_scale[:, None] * ws

        # Optional bias
        if HAS_BIAS:
            bias_offs = (pid_n * GEMM_BLOCK_N
                         + tl.arange(0, GEMM_BLOCK_N)) % K_GEMM
            bias_val = tl.load(bias_ptr + bias_offs)
            accumulator = (
                accumulator.to(bias_ptr.type.element_ty) + bias_val[None, :]
            )

        # Store GEMM output
        c = accumulator.to(gemm_out_ptr.type.element_ty)

        offs_cm = pid_m * GEMM_BLOCK_M + tl.arange(0, GEMM_BLOCK_M)
        offs_cn = pid_n * GEMM_BLOCK_N + tl.arange(0, GEMM_BLOCK_N)
        c_ptrs = (
            gemm_out_ptr
            + offs_cm[:, None] * stride_go_m
            + offs_cn[None, :] * stride_go_k
        )
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < K_GEMM)
        tl.store(c_ptrs, c, mask=c_mask)


# ============================================================================
# Manager and public API
# ============================================================================


class IrisTwoshot2dGemmManager:
    """Singleton manager for 2D-tiled AllReduce+RMSNorm+Quant+inlined GEMM.

    Combines 2D comm tiling from 2d_hipblaslt with inlined Triton GEMM.
    External pre/post barriers, inlined mid-barrier between comm and GEMM.
    """

    _instance: Optional["IrisTwoshot2dGemmManager"] = None
    _initialized: bool = False

    def __new__(cls) -> "IrisTwoshot2dGemmManager":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if IrisTwoshot2dGemmManager._initialized:
            return
        IrisTwoshot2dGemmManager._initialized = True

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
        # GEMM output buffers keyed by K_GEMM (different layers have
        # different output dims, e.g. 1280, 3584, 8192 in Llama 70B)
        self._out_gemm_bufs: dict[int, torch.Tensor] = {}

        # Inlined device barrier state
        self._barrier_flags: Any = None  # on symmetric heap
        self._barrier_epoch: Optional[torch.Tensor] = None
        self._barrier_done: Optional[torch.Tensor] = None

    def initialize(self, heap_size: Optional[int] = None) -> None:
        """Initialize Iris symmetric heap (call once at startup)."""
        if self._shmem is not None:
            logger.debug(
                "Iris (twoshot-2d-gemm) already initialized, skipping"
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

        # Allocate barrier flags on symmetric heap (one int32 per rank)
        num_ranks = self._shmem.get_num_ranks()
        self._barrier_flags = self._shmem.zeros(
            (num_ranks,), dtype=torch.int32
        )
        self._shmem.device_barrier()

        logger.info(
            f"Iris (twoshot-2d-gemm) initialized: "
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
        """Allocate or reuse symmetric heap buffers."""
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
                f"Iris (twoshot-2d-gemm): heap buffers too small for "
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
                f"Iris (twoshot-2d-gemm): allocated input buffer "
                f"({M}, {N}), dtype={input_dtype}, rank={cur_rank}"
            )

        if need_quant:
            self._quant_heap_buf = shmem.zeros((M, N), dtype=quant_dtype)
            self._quant_heap_M = M
            self._quant_heap_N = N
            self._quant_heap_dtype = quant_dtype
            logger.debug(
                f"Iris (twoshot-2d-gemm): allocated quant heap buffer "
                f"({M}, {N}), dtype={quant_dtype}, rank={cur_rank}"
            )

        if need_scale:
            self._scale_heap_buf = shmem.zeros((M,), dtype=torch.float32)
            self._scale_heap_M = M
            logger.debug(
                f"Iris (twoshot-2d-gemm): allocated scale heap buffer "
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
        K_GEMM: int,
        dtype: torch.dtype,
        out_dtype: torch.dtype,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Allocate or reuse result_out and gemm_out on GPU memory."""
        need_result = (
            self._out_result is None
            or M > self._out_result.shape[0]
            or N != self._out_result.shape[1]
        )
        need_gemm = (
            K_GEMM not in self._out_gemm_bufs
            or M > self._out_gemm_bufs[K_GEMM].shape[0]
        )
        need_barrier = self._barrier_done is None

        if not (need_result or need_gemm or need_barrier):
            assert self._out_result is not None
            return self._out_result[:M], self._out_gemm_bufs[K_GEMM][:M]

        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                f"Iris (twoshot-2d-gemm): GPU buffers too small for "
                f"M={M}, K_GEMM={K_GEMM}. Cannot allocate during "
                f"CUDA graph capture."
            )

        cur_rank = (
            torch.distributed.get_rank()
            if torch.distributed.is_initialized()
            else 0
        )

        if need_result:
            self._out_result = torch.empty(
                (M, N), dtype=dtype, device=device,
            )

        if need_gemm:
            self._out_gemm_bufs[K_GEMM] = torch.empty(
                (M, K_GEMM), dtype=out_dtype, device=device,
            )

        if need_barrier:
            self._barrier_done = torch.zeros(
                1, dtype=torch.int32, device=device
            )
            self._barrier_epoch = torch.zeros(
                1, dtype=torch.int32, device=device
            )

        logger.debug(
            f"Iris (twoshot-2d-gemm): allocated GPU buffers "
            f"result=({M}, {N}), gemm=({M}, {K_GEMM}), rank={cur_rank}"
        )

        assert self._out_result is not None
        return self._out_result[:M], self._out_gemm_bufs[K_GEMM][:M]

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
        """Fused AllReduce + RMSNorm + per-row FP8 quant + inlined GEMM.

        Returns (gemm_out, residual_out). residual_out is None when
        residual is None.
        """
        shmem = self.shmem
        M, N = input_tensor.shape
        K_GEMM = gemm_weight.shape[1]
        device = input_tensor.device

        # Allocate buffers
        iris_input, quant_heap, scale_heap = self._ensure_heap_buffers(
            M, N, input_tensor.dtype, quant_dtype,
        )
        result_out, gemm_out = self._ensure_gpu_buffers(
            M, N, K_GEMM, input_tensor.dtype, out_dtype, device,
        )

        # Copy input to symmetric heap (captured in graph)
        iris_input.copy_(input_tensor)

        # Pre-kernel barrier
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
        DISTRIBUTION = 1
        COMM_SMS = 128
        CHUNK_SIZE = _compute_chunk_size(COMM_SMS, num_xcds)
        NUM_WARPS = 16
        NUM_STAGES = 2
        WAVES_PER_EU = 1
        # GEMM tile sizes (from twoshot_row autotuning on MI350X)
        GEMM_BLOCK_M = 256
        GEMM_BLOCK_N = 128
        GEMM_BLOCK_K = 128
        GEMM_GROUP_SIZE_M = 4
        EVEN_GEMM_K = (N % GEMM_BLOCK_K == 0)
        # ---- End tunable parameters ----

        # Reset barrier sync buffer
        assert self._barrier_done is not None
        self._barrier_done.zero_()

        # Dummy bias pointer
        bias_ptr = bias if bias is not None else input_tensor

        # Single fused kernel: comm + barrier + GEMM
        persistent_fused_allreduce_rmsnorm_2d_quant_gemm[(COMM_SMS,)](
            iris_input,
            quant_heap,
            scale_heap,
            result_out,
            # GEMM params
            gemm_weight,
            gemm_out,
            bias_ptr,
            weight_scale,
            K_GEMM,
            gemm_weight.stride(0),
            gemm_weight.stride(1),
            gemm_out.stride(0),
            gemm_out.stride(1),
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
            bias is not None,  # HAS_BIAS
            PADDED_N,
            COMM_SMS,
            num_xcds,
            CHUNK_SIZE,
            DISTRIBUTION,
            # GEMM tile params
            GEMM_BLOCK_M,
            GEMM_BLOCK_N,
            GEMM_BLOCK_K,
            GEMM_GROUP_SIZE_M,
            EVEN_GEMM_K,
            num_warps=NUM_WARPS,
            num_stages=NUM_STAGES,
            waves_per_eu=WAVES_PER_EU,
        )

        # Post-kernel barrier
        if not is_graph_capturing():
            shmem.device_barrier()

        residual_out = result_out if residual is not None else None

        return gemm_out, residual_out


_manager: Optional[IrisTwoshot2dGemmManager] = None


def _get_manager() -> IrisTwoshot2dGemmManager:
    global _manager
    if _manager is None:
        _manager = IrisTwoshot2dGemmManager()
    return _manager


def initialize_iris_twoshot_2d_gemm(
    heap_size: Optional[int] = None,
) -> None:
    """Initialize Iris for 2D-tiled fused all-reduce with inlined GEMM."""
    _get_manager().initialize(heap_size)


def fused_allreduce_add_rms_row_quant_gemm_iris_twoshot_2d_gemm(
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
    """Fused AllReduce + RMSNorm + per-row FP8 Quant + inlined GEMM.

    2D-tiled two-shot variant with Triton GEMM. Returns (gemm_out, residual_out).
    """
    return (
        _get_manager()
        .fused_allreduce_rmsnorm_row_quant_gemm(
            input, rms_weight, rms_eps, quant_dtype,
            gemm_weight, weight_scale, out_dtype, residual, bias,
        )
    )
