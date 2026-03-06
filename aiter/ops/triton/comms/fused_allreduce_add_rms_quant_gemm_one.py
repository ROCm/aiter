# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Iris fused AllReduce + RMSNorm + per-row FP8 Quant + partial GEMM + allgather.

Single Triton kernel with 2 phases:
  Phase 1 (Comm + GEMM): Each rank reduces its assigned rows from the
    symmetric heap, applies RMSNorm + per-row FP8 quant, then immediately
    does a small GEMM on just its owned rows. The BF16 GEMM output is
    broadcast to all peers via iris.store.
  Phase 2 (Allgather): Each rank reads other ranks' GEMM output rows
    from the symmetric heap via iris.load, assembling the full output.

No cross-rank barrier needed between quant and GEMM because each rank
only GEMMs its own rows. No hipBLASLt needed because the per-rank GEMM
is small (M/world_size rows).

CUDA graph compatibility:
- Buffer pre-allocation with view pattern
- External device_barrier() before and after kernel (skipped during capture)
"""

import logging
from typing import Any, Optional, Tuple

import iris
import torch
import triton
import triton.language as tl

from aiter.ops.triton.comms import is_graph_capturing

__all__ = ["fused_allreduce_add_rms_quant_gemm_one"]

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


# ============================================================================
# Fused kernel: AllReduce + RMSNorm + FP8 Quant + partial GEMM + allgather
# ============================================================================


@triton.jit
def persistent_fused_allreduce_rmsnorm_quant_partial_gemm(
    # Symmetric heap buffers
    input_ptr,
    quant_local_ptr,
    scale_local_ptr,
    # GEMM output on symmetric heap (for allgather)
    gemm_heap_ptr,
    # Regular GPU memory
    result_out_ptr,
    gemm_out_ptr,
    # GEMM parameters
    gemm_weight_ptr,
    bias_ptr,
    weight_scale_ptr,
    K_GEMM,
    stride_gw_k,
    stride_gw_n,
    stride_go_m,
    stride_go_n,
    stride_gh_m,
    stride_gh_n,
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
    # Quant local strides
    stride_ql_m,
    stride_ql_n,
    # Result out strides
    stride_out_m,
    stride_out_n,
    # Scale local stride
    stride_sl,
    # Iris params
    heap_bases: tl.tensor,
    group_rank: tl.constexpr,
    iris_rank: tl.constexpr,
    world_size: tl.constexpr,
    rank_start: tl.constexpr,
    rank_stride: tl.constexpr,
    # Comm tile params
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
    """Fused AllReduce + RMSNorm + per-row FP8 quant + partial GEMM + allgather.

    Phase 1: Each rank reduces its assigned rows, applies RMSNorm + per-row
    FP8 quant, does GEMM on owned rows, broadcasts BF16 GEMM output to peers.
    Phase 2: Each rank reads all other ranks' GEMM output via iris.load.
    """
    pid = tl.program_id(0)

    # ================================================================
    # Phase 1: Reduce + RMSNorm + Quant + partial GEMM + broadcast
    # ================================================================

    # Row distribution across ranks (block distribution)
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

    num_comm_tiles = tl.cdiv(max_row_offset, BLOCK_SIZE_M)

    # Load RMSNorm weights
    rn = tl.arange(0, BLOCK_SIZE_N)
    if PADDED_N:
        rms_w = tl.load(
            rms_weight_ptr + rn, mask=rn < ACTUAL_N, other=0.0
        ).to(tl.float32)
    else:
        rms_w = tl.load(rms_weight_ptr + rn).to(tl.float32)

    # Phase 1a: Allreduce + RMSNorm + FP8 quant on owned rows
    for tile_offset in range(pid, num_comm_tiles, COMM_SMS):
        row_base = my_start + tile_offset * BLOCK_SIZE_M * stride

        rm = row_base + tl.arange(0, BLOCK_SIZE_M) * stride
        row_mask = rm < M

        input_offset = rm[:, None] * stride_in_m + rn[None, :] * stride_in_n
        out_offset = rm[:, None] * stride_out_m + rn[None, :] * stride_out_n

        # Local quant buffer offsets (indexed by local row position)
        local_row = tile_offset * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        local_row_mask = local_row < max_row_offset
        ql_offset = (
            local_row[:, None] * stride_ql_m + rn[None, :] * stride_ql_n
        )
        sl_offset = local_row * stride_sl

        is_full = (row_base + BLOCK_SIZE_M * stride <= M)

        if PADDED_N:
            col_mask = rn < ACTUAL_N
            mask = row_mask[:, None] & col_mask[None, :]
            local_mask = local_row_mask[:, None] & col_mask[None, :]
        else:
            mask = row_mask[:, None]
            local_mask = local_row_mask[:, None]

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
                quant_local_ptr.type.element_ty
            )

            tl.store(quant_local_ptr + ql_offset, quantized)
            tl.store(scale_local_ptr + sl_offset, scale)
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
                quant_local_ptr.type.element_ty
            )

            tl.store(
                quant_local_ptr + ql_offset, quantized, mask=local_mask
            )
            tl.store(
                scale_local_ptr + sl_offset, scale, mask=local_row_mask
            )

    # ================================================================
    # Phase 1b: Partial GEMM on owned rows + broadcast output
    # ================================================================
    # Each rank GEMMs its own rows: (max_row_offset, N) x (N, K_GEMM)
    # No barrier needed -- we just wrote the quant data ourselves.

    num_pid_m = tl.cdiv(max_row_offset, GEMM_BLOCK_M)
    num_pid_n = tl.cdiv(K_GEMM, GEMM_BLOCK_N)
    total_gemm_tiles = num_pid_m * num_pid_n

    for tile_id in range(pid, total_gemm_tiles, COMM_SMS):
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

        # A pointers: local quant buffer (max_row_offset, N)
        a_ptrs = (
            quant_local_ptr
            + (offs_am[:, None] % max_row_offset) * stride_ql_m
            + offs_k[None, :] * stride_ql_n
        )
        # B pointers: gemm_weight (N, K_GEMM)
        b_ptrs = (
            gemm_weight_ptr
            + offs_k[:, None] * stride_gw_k
            + (offs_bn[None, :] % K_GEMM) * stride_gw_n
        )

        # Per-row activation scales (local indexing)
        a_scale = tl.load(
            scale_local_ptr + (offs_am % max_row_offset) * stride_sl
        )

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

            a_ptrs += GEMM_BLOCK_K * stride_ql_n
            b_ptrs += GEMM_BLOCK_K * stride_gw_k

        # Apply scales
        ws = tl.load(weight_scale_ptr)
        accumulator *= a_scale[:, None] * ws

        if HAS_BIAS:
            bias_offs = (pid_n * GEMM_BLOCK_N
                         + tl.arange(0, GEMM_BLOCK_N)) % K_GEMM
            bias_val = tl.load(bias_ptr + bias_offs)
            accumulator = (
                accumulator.to(bias_ptr.type.element_ty) + bias_val[None, :]
            )

        c = accumulator.to(gemm_heap_ptr.type.element_ty)

        # Global row indices for this rank's owned rows
        offs_cm_global = my_start + offs_am * stride
        offs_cn = pid_n * GEMM_BLOCK_N + tl.arange(0, GEMM_BLOCK_N)
        c_mask = (offs_am[:, None] < max_row_offset) & (
            offs_cn[None, :] < K_GEMM
        )

        # Store to GEMM heap (symmetric, for allgather)
        gh_ptrs = (
            gemm_heap_ptr
            + offs_cm_global[:, None] * stride_gh_m
            + offs_cn[None, :] * stride_gh_n
        )
        tl.store(gh_ptrs, c, mask=c_mask)

        # Broadcast GEMM output to all peers
        for i in tl.static_range(0, world_size):
            remote_rank = rank_start + i * rank_stride
            if remote_rank != iris_rank:
                iris.store(
                    gh_ptrs, c,
                    iris_rank, remote_rank, heap_bases,
                    mask=c_mask,
                )

    # ================================================================
    # Phase 2: Copy from heap to output
    # ================================================================
    # After all ranks have broadcast, gemm_heap has the full (M, K_GEMM)
    # output. Copy our assigned rows to gemm_out. The external post-barrier
    # ensures all ranks have finished writing before the next iteration
    # overwrites the input buffer.

    # Copy all rows (not just owned) from heap to output
    num_copy_m = tl.cdiv(M, GEMM_BLOCK_M)
    num_copy_n = tl.cdiv(K_GEMM, GEMM_BLOCK_N)
    total_copy_tiles = num_copy_m * num_copy_n

    for tile_id in range(pid, total_copy_tiles, COMM_SMS):
        cm = tile_id // num_copy_n
        cn = tile_id % num_copy_n

        offs_m = cm * GEMM_BLOCK_M + tl.arange(0, GEMM_BLOCK_M)
        offs_n = cn * GEMM_BLOCK_N + tl.arange(0, GEMM_BLOCK_N)
        cp_mask = (offs_m[:, None] < M) & (offs_n[None, :] < K_GEMM)

        src_ptrs = (
            gemm_heap_ptr
            + offs_m[:, None] * stride_gh_m
            + offs_n[None, :] * stride_gh_n
        )
        dst_ptrs = (
            gemm_out_ptr
            + offs_m[:, None] * stride_go_m
            + offs_n[None, :] * stride_go_n
        )
        vals = tl.load(src_ptrs, mask=cp_mask)
        tl.store(dst_ptrs, vals, mask=cp_mask)


# ============================================================================
# Manager and public API
# ============================================================================


class IrisOneManager:
    """Singleton manager for fused allreduce + partial GEMM + allgather.

    Single kernel: allreduce+rmsnorm+quant on owned rows, GEMM on owned
    rows, broadcast BF16 GEMM output, copy full output from heap.
    """

    _instance: Optional["IrisOneManager"] = None
    _initialized: bool = False

    def __new__(cls) -> "IrisOneManager":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if IrisOneManager._initialized:
            return
        IrisOneManager._initialized = True

        self._shmem: Any = None
        self._heap_size: int = 2**33  # 8GB default

        # Symmetric heap buffers
        self._input_buf: Any = None
        self._input_M: int = 0
        self._input_N: int = 0
        self._input_dtype: Optional[torch.dtype] = None

        # GEMM output on symmetric heap (for allgather broadcast)
        self._gemm_heap_bufs: dict[int, Any] = {}  # keyed by K_GEMM

        # Local GPU memory (not on heap)
        self._quant_local_buf: Optional[torch.Tensor] = None
        self._quant_local_M: int = 0
        self._quant_local_N: int = 0
        self._quant_local_dtype: Optional[torch.dtype] = None

        self._scale_local_buf: Optional[torch.Tensor] = None
        self._scale_local_M: int = 0

        self._out_result: Optional[torch.Tensor] = None
        self._out_gemm_bufs: dict[int, torch.Tensor] = {}

    def initialize(self, heap_size: Optional[int] = None) -> None:
        """Initialize Iris symmetric heap (call once at startup)."""
        if self._shmem is not None:
            logger.debug("Iris (one) already initialized, skipping")
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
            f"Iris (one) initialized: "
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
        K_GEMM: int,
        input_dtype: torch.dtype,
        out_dtype: torch.dtype,
    ) -> Tuple[Any, Any]:
        """Allocate or reuse symmetric heap buffers.

        Returns (input_buf, gemm_heap).
        """
        need_input = (
            self._input_buf is None
            or M > self._input_M
            or N != self._input_N
            or input_dtype != self._input_dtype
        )
        need_gemm_heap = (
            K_GEMM not in self._gemm_heap_bufs
            or M > self._gemm_heap_bufs[K_GEMM].shape[0]
        )

        if not (need_input or need_gemm_heap):
            return (
                self._input_buf[:M],
                self._gemm_heap_bufs[K_GEMM][:M],
            )

        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                f"Iris (one): heap buffers too small for "
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
                f"Iris (one): allocated input buffer "
                f"({M}, {N}), dtype={input_dtype}, rank={cur_rank}"
            )

        if need_gemm_heap:
            self._gemm_heap_bufs[K_GEMM] = shmem.zeros(
                (M, K_GEMM), dtype=out_dtype
            )
            logger.debug(
                f"Iris (one): allocated GEMM heap buffer "
                f"({M}, {K_GEMM}), dtype={out_dtype}, rank={cur_rank}"
            )

        return (
            self._input_buf[:M],
            self._gemm_heap_bufs[K_GEMM][:M],
        )

    def _ensure_local_buffers(
        self,
        M: int,
        N: int,
        K_GEMM: int,
        world_size: int,
        quant_dtype: torch.dtype,
        out_dtype: torch.dtype,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Allocate local GPU buffers (not on heap).

        Returns (quant_local, scale_local, result_out, gemm_out).
        """
        # Max rows this rank will own
        rows_per_rank = (M + world_size - 1) // world_size

        need_quant = (
            self._quant_local_buf is None
            or rows_per_rank > self._quant_local_M
            or N != self._quant_local_N
            or quant_dtype != self._quant_local_dtype
        )
        need_scale = (
            self._scale_local_buf is None
            or rows_per_rank > self._scale_local_M
        )
        need_result = (
            self._out_result is None
            or M > self._out_result.shape[0]
            or N != self._out_result.shape[1]
        )
        need_gemm = (
            K_GEMM not in self._out_gemm_bufs
            or M > self._out_gemm_bufs[K_GEMM].shape[0]
        )

        if not (need_quant or need_scale or need_result or need_gemm):
            assert self._quant_local_buf is not None
            assert self._scale_local_buf is not None
            assert self._out_result is not None
            return (
                self._quant_local_buf[:rows_per_rank],
                self._scale_local_buf[:rows_per_rank],
                self._out_result[:M],
                self._out_gemm_bufs[K_GEMM][:M],
            )

        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                f"Iris (one): local buffers too small for "
                f"M={M}. Cannot allocate during CUDA graph capture."
            )

        if need_quant:
            self._quant_local_buf = torch.empty(
                (rows_per_rank, N), dtype=quant_dtype, device=device,
            )
            self._quant_local_M = rows_per_rank
            self._quant_local_N = N
            self._quant_local_dtype = quant_dtype

        if need_scale:
            self._scale_local_buf = torch.empty(
                (rows_per_rank,), dtype=torch.float32, device=device,
            )
            self._scale_local_M = rows_per_rank

        if need_result:
            self._out_result = torch.empty(
                (M, N), dtype=torch.bfloat16, device=device,
            )

        if need_gemm:
            self._out_gemm_bufs[K_GEMM] = torch.empty(
                (M, K_GEMM), dtype=out_dtype, device=device,
            )

        assert self._quant_local_buf is not None
        assert self._scale_local_buf is not None
        assert self._out_result is not None
        return (
            self._quant_local_buf[:rows_per_rank],
            self._scale_local_buf[:rows_per_rank],
            self._out_result[:M],
            self._out_gemm_bufs[K_GEMM][:M],
        )

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
        """Fused AllReduce + RMSNorm + FP8 quant + partial GEMM + allgather.

        Returns (gemm_out, residual_out). residual_out is None when
        residual is None.
        """
        shmem = self.shmem
        M, N = input_tensor.shape
        K_GEMM = gemm_weight.shape[1]
        device = input_tensor.device

        # Iris group info
        rank_in_group, rank_global, world_size, rank_start, rank_stride = (
            extract_group_info(shmem)
        )

        # Allocate buffers
        iris_input, gemm_heap = self._ensure_heap_buffers(
            M, N, K_GEMM, input_tensor.dtype, out_dtype,
        )
        quant_local, scale_local, result_out, gemm_out = (
            self._ensure_local_buffers(
                M, N, K_GEMM, world_size,
                quant_dtype, out_dtype, device,
            )
        )

        # Copy input to symmetric heap (captured in graph)
        iris_input.copy_(input_tensor)

        # Pre-kernel barrier
        if not is_graph_capturing():
            shmem.device_barrier()

        fp8_max = torch.finfo(quant_dtype).max
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
        GEMM_BLOCK_M = 128
        GEMM_BLOCK_N = 128
        GEMM_BLOCK_K = 128
        GEMM_GROUP_SIZE_M = 4
        EVEN_GEMM_K = (N % GEMM_BLOCK_K == 0)
        # ---- End tunable parameters ----

        bias_ptr = bias if bias is not None else input_tensor

        persistent_fused_allreduce_rmsnorm_quant_partial_gemm[(COMM_SMS,)](
            iris_input,
            quant_local,
            scale_local,
            gemm_heap,
            result_out,
            gemm_out,
            # GEMM params
            gemm_weight,
            bias_ptr,
            weight_scale,
            K_GEMM,
            gemm_weight.stride(0),
            gemm_weight.stride(1),
            gemm_out.stride(0),
            gemm_out.stride(1),
            gemm_heap.stride(0),
            gemm_heap.stride(1),
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
            quant_local.stride(0),
            quant_local.stride(1),
            result_out.stride(0),
            result_out.stride(1),
            scale_local.stride(0),
            # Iris
            heap_bases,
            rank_in_group,
            rank_global,
            world_size,
            rank_start,
            rank_stride,
            # Comm tile params
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


_manager: Optional[IrisOneManager] = None


def _get_manager() -> IrisOneManager:
    global _manager
    if _manager is None:
        _manager = IrisOneManager()
    return _manager


def initialize_iris_one(
    heap_size: Optional[int] = None,
) -> None:
    """Initialize Iris for fused allreduce + partial GEMM."""
    _get_manager().initialize(heap_size)


def fused_allreduce_add_rms_quant_gemm_one(
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
    """Fused AllReduce + RMSNorm + per-row FP8 Quant + partial GEMM.

    Single kernel variant. Returns (gemm_out, residual_out).
    """
    return (
        _get_manager()
        .fused_allreduce_rmsnorm_row_quant_gemm(
            input, rms_weight, rms_eps, quant_dtype,
            gemm_weight, weight_scale, out_dtype, residual, bias,
        )
    )
