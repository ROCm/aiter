# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Iris fused AllReduce + RMSNorm + per-row FP8 Quant + partial GEMM + allgather.

Single Triton kernel, two passes per block of BLOCK_SIZE_M owned rows:

  Pass 1 (Allreduce + scalars): Tile over K in GEMM_BLOCK_K chunks.
    For each tile: allreduce via iris.load, store to result_out,
    accumulate sq_sum (for RMSNorm variance) and weighted_amax
    (for quant scale). After the loop, compute rrms and scale.

  Pass 2 (GEMM): Tile over K and output columns. For each tile:
    load allreduced BF16 from result_out (L2 cache hit), apply
    inline RMSNorm + FP8 quant using rrms/scale from pass 1,
    tl.dot with weight. After all K tiles, broadcast BF16 GEMM
    output to all peers via iris.store.

After the kernel, a device_barrier() ensures all ranks have finished
broadcasting, then the caller copies gemm_heap -> gemm_out.

Key insight: max(|x * rrms * rms_w|) = rrms * max(|x * rms_w|).
Since rrms is per-row, we accumulate max(|x * rms_w|) during pass 1
before rrms is known, then multiply afterwards. This avoids a third
pass for the quant scale.

No scratch buffers — result_out (needed anyway for residual chain)
serves as the intermediate between passes.

No cross-rank barrier between passes because each rank only GEMMs
its own rows.

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
    input_ptr,        # (M, N) on heap — input partials from each rank
    gemm_heap_ptr,    # (M, K_GEMM) on heap — GEMM output, broadcast target
    # Regular GPU memory outputs
    result_out_ptr,   # (M, N) BF16 — allreduced result (residual chain + GEMM input)
    # GEMM weight and parameters
    gemm_weight_ptr,  # (N, K_GEMM) FP8 — weight matrix
    bias_ptr,         # (K_GEMM,) or dummy when HAS_BIAS=False
    weight_scale_ptr, # scalar f32 — per-tensor weight scale
    K_GEMM,           # output columns of GEMM
    stride_gw_k,      # weight stride dim 0 (K/N direction)
    stride_gw_n,      # weight stride dim 1 (output direction)
    stride_gh_m,      # gemm_heap stride dim 0
    stride_gh_n,      # gemm_heap stride dim 1
    # Residual (dummy ptr when HAS_RESIDUAL=False)
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
    # Result out strides
    stride_out_m,
    stride_out_n,
    # Iris params
    heap_bases: tl.tensor,
    group_rank: tl.constexpr,
    iris_rank: tl.constexpr,
    world_size: tl.constexpr,
    rank_start: tl.constexpr,
    rank_stride: tl.constexpr,
    # Tile params
    BLOCK_SIZE_M: tl.constexpr,   # rows per block (>= 16 for tl.dot)
    HAS_RESIDUAL: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    COMM_SMS: tl.constexpr,       # number of persistent CTAs
    # GEMM tile params
    GEMM_BLOCK_N: tl.constexpr,   # output column tile size
    GEMM_BLOCK_K: tl.constexpr,   # reduction tile size (also comm tile width)
    EVEN_GEMM_K: tl.constexpr,    # True when N % GEMM_BLOCK_K == 0
):
    """Fused AllReduce + RMSNorm + FP8 quant + partial GEMM + allgather.

    Two passes per block:
      Pass 1: tiled allreduce, accumulate sq_sum + weighted_amax
      Pass 2: tiled GEMM with inline rmsnorm + quant, broadcast via iris.store
    """
    pid = tl.program_id(0)

    # Row distribution: contiguous block of rows per rank
    rows_per_rank = tl.cdiv(M, world_size)
    my_start = group_rank * rows_per_rank
    remaining = tl.maximum(M - my_start, 0)
    max_row_offset = tl.minimum(rows_per_rank, remaining)

    num_row_blocks = tl.cdiv(max_row_offset, BLOCK_SIZE_M)

    # Scalar loaded once
    ws = tl.load(weight_scale_ptr)

    num_k_tiles = tl.cdiv(N, GEMM_BLOCK_K)
    num_n_tiles = tl.cdiv(K_GEMM, GEMM_BLOCK_N)

    # ================================================================
    # Main loop: one block of BLOCK_SIZE_M rows per iteration
    # ================================================================
    for block_id in range(pid, num_row_blocks, COMM_SMS):
        block_start = block_id * BLOCK_SIZE_M

        rm = tl.arange(0, BLOCK_SIZE_M)
        row_offsets = block_start + rm
        m_mask = row_offsets < max_row_offset
        global_rows = my_start + row_offsets

        # ---- Pass 1: Tiled allreduce + accumulate scalars ----
        sq_sum = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
        weighted_amax = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)

        for k_tile in range(num_k_tiles):
            offs_k = k_tile * GEMM_BLOCK_K + tl.arange(0, GEMM_BLOCK_K)

            in_offset = (
                global_rows[:, None] * stride_in_m
                + offs_k[None, :] * stride_in_n
            )
            out_offset = (
                global_rows[:, None] * stride_out_m
                + offs_k[None, :] * stride_out_n
            )

            if EVEN_GEMM_K:
                tile_mask = m_mask[:, None]
            else:
                k_valid = offs_k < N
                tile_mask = m_mask[:, None] & k_valid[None, :]

            # Allreduce: sum this K slice from all ranks
            tile_acc = tl.zeros(
                (BLOCK_SIZE_M, GEMM_BLOCK_K), dtype=tl.float32
            )
            for i in tl.static_range(0, world_size):
                remote_rank = rank_start + i * rank_stride
                partial = iris.load(
                    input_ptr + in_offset,
                    iris_rank, remote_rank, heap_bases,
                    mask=tile_mask,
                )
                tile_acc += partial.to(tl.float32)
                if HAS_RESIDUAL and remote_rank == iris_rank:
                    res_tile = tl.load(
                        residual_in_ptr + out_offset,
                        mask=tile_mask, other=0.0,
                    ).to(tl.float32)
                    tile_acc += res_tile

            # Store allreduced tile to result_out
            tl.store(
                result_out_ptr + out_offset,
                tile_acc.to(result_out_ptr.type.element_ty),
                mask=tile_mask,
            )

            # Accumulate RMSNorm variance
            sq_sum += tl.sum(tile_acc * tile_acc, axis=1)

            # Accumulate weighted amax for quant scale
            # max(|x * rrms * rms_w|) = rrms * max(|x * rms_w|)
            if EVEN_GEMM_K:
                rms_w_k = tl.load(rms_weight_ptr + offs_k).to(tl.float32)
            else:
                rms_w_k = tl.load(
                    rms_weight_ptr + offs_k,
                    mask=k_valid, other=0.0,
                ).to(tl.float32)
            tile_amax = tl.max(
                tl.abs(tile_acc * rms_w_k[None, :]), axis=1
            )
            weighted_amax = tl.maximum(weighted_amax, tile_amax)

        # Compute rrms and quant scale from accumulated values
        rrms = tl.rsqrt(sq_sum / N + rms_eps)          # (BLOCK_SIZE_M,)
        row_amax = rrms * weighted_amax
        row_amax = tl.maximum(row_amax, 1e-12)
        scales = row_amax / fp8_max                     # (BLOCK_SIZE_M,)

        # ---- Pass 2: GEMM with inline RMSNorm + FP8 quant ----
        for n_tile in range(num_n_tiles):
            offs_n = n_tile * GEMM_BLOCK_N + tl.arange(0, GEMM_BLOCK_N)
            n_mask = offs_n < K_GEMM

            gemm_acc = tl.zeros(
                (BLOCK_SIZE_M, GEMM_BLOCK_N), dtype=tl.float32
            )

            for k_tile in range(num_k_tiles):
                offs_k = (
                    k_tile * GEMM_BLOCK_K + tl.arange(0, GEMM_BLOCK_K)
                )

                # Load allreduced BF16 from result_out (L2 cache hit)
                out_offset_k = (
                    global_rows[:, None] * stride_out_m
                    + offs_k[None, :] * stride_out_n
                )
                if EVEN_GEMM_K:
                    a_raw = tl.load(
                        result_out_ptr + out_offset_k,
                        mask=m_mask[:, None], other=0.0,
                    ).to(tl.float32)
                    rms_w_k = tl.load(
                        rms_weight_ptr + offs_k
                    ).to(tl.float32)
                else:
                    k_valid = offs_k < N
                    a_raw = tl.load(
                        result_out_ptr + out_offset_k,
                        mask=m_mask[:, None] & k_valid[None, :],
                        other=0.0,
                    ).to(tl.float32)
                    rms_w_k = tl.load(
                        rms_weight_ptr + offs_k,
                        mask=k_valid, other=0.0,
                    ).to(tl.float32)

                # Inline RMSNorm + FP8 quant
                a_normed = a_raw * rrms[:, None] * rms_w_k[None, :]
                a_quant = (a_normed / scales[:, None]).to(
                    gemm_weight_ptr.type.element_ty
                )

                # Load weight tile: (GEMM_BLOCK_K, GEMM_BLOCK_N)
                if EVEN_GEMM_K:
                    b = tl.load(
                        gemm_weight_ptr
                        + offs_k[:, None] * stride_gw_k
                        + offs_n[None, :] * stride_gw_n,
                        mask=n_mask[None, :], other=0.0,
                    )
                else:
                    b = tl.load(
                        gemm_weight_ptr
                        + offs_k[:, None] * stride_gw_k
                        + offs_n[None, :] * stride_gw_n,
                        mask=k_valid[:, None] & n_mask[None, :],
                        other=0.0,
                    )

                # (BLOCK_SIZE_M, GEMM_BLOCK_K) @ (GEMM_BLOCK_K, GEMM_BLOCK_N)
                gemm_acc += tl.dot(a_quant, b, input_precision="ieee")

            # Apply per-row activation scales and per-tensor weight scale
            gemm_acc *= scales[:, None] * ws

            # Optional bias
            if HAS_BIAS:
                bias_val = tl.load(
                    bias_ptr + offs_n, mask=n_mask, other=0.0
                )
                gemm_acc += bias_val[None, :].to(tl.float32)

            c = gemm_acc.to(gemm_heap_ptr.type.element_ty)

            # Store to GEMM heap + broadcast to all peers
            gh_offsets = (
                global_rows[:, None] * stride_gh_m
                + offs_n[None, :] * stride_gh_n
            )
            store_mask = m_mask[:, None] & n_mask[None, :]
            tl.store(gemm_heap_ptr + gh_offsets, c, mask=store_mask)

            for i in tl.static_range(0, world_size):
                remote_rank = rank_start + i * rank_stride
                if remote_rank != iris_rank:
                    iris.store(
                        gemm_heap_ptr + gh_offsets, c,
                        iris_rank, remote_rank, heap_bases,
                        mask=store_mask,
                    )


# ============================================================================
# Manager and public API
# ============================================================================


class IrisOneManager:
    """Singleton manager for fused allreduce + partial GEMM + allgather."""

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

        self._gemm_heap_bufs: dict[int, Any] = {}

        # Local GPU memory
        self._out_result: Optional[torch.Tensor] = None
        self._out_gemm_bufs: dict[int, torch.Tensor] = {}

    def initialize(self, heap_size: Optional[int] = None) -> None:
        if self._shmem is not None:
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
        """Allocate or reuse symmetric heap buffers for input and GEMM output."""
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

        if need_input:
            self._input_buf = shmem.zeros((M, N), dtype=input_dtype)
            self._input_M = M
            self._input_N = N
            self._input_dtype = input_dtype

        if need_gemm_heap:
            self._gemm_heap_bufs[K_GEMM] = shmem.zeros(
                (M, K_GEMM), dtype=out_dtype
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
        out_dtype: torch.dtype,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Allocate or reuse result_out and gemm_out on local GPU memory."""
        need_result = (
            self._out_result is None
            or M > self._out_result.shape[0]
            or N != self._out_result.shape[1]
        )
        need_gemm = (
            K_GEMM not in self._out_gemm_bufs
            or M > self._out_gemm_bufs[K_GEMM].shape[0]
        )

        if not (need_result or need_gemm):
            assert self._out_result is not None
            return (
                self._out_result[:M],
                self._out_gemm_bufs[K_GEMM][:M],
            )

        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                f"Iris (one): local buffers too small for "
                f"M={M}. Cannot allocate during CUDA graph capture."
            )

        if need_result:
            self._out_result = torch.empty(
                (M, N), dtype=torch.bfloat16, device=device,
            )

        if need_gemm:
            self._out_gemm_bufs[K_GEMM] = torch.empty(
                (M, K_GEMM), dtype=out_dtype, device=device,
            )

        assert self._out_result is not None
        return (
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
        """Fused AllReduce + RMSNorm + per-row FP8 quant + partial GEMM.

        Returns (gemm_out, residual_out). residual_out is None when
        residual is None.
        """
        shmem = self.shmem
        M, N = input_tensor.shape
        K_GEMM = gemm_weight.shape[1]
        device = input_tensor.device

        rank_in_group, rank_global, world_size, rank_start, rank_stride = (
            extract_group_info(shmem)
        )

        iris_input, gemm_heap = self._ensure_heap_buffers(
            M, N, K_GEMM, input_tensor.dtype, out_dtype,
        )
        result_out, gemm_out = self._ensure_local_buffers(
            M, N, K_GEMM, out_dtype, device,
        )

        # Copy input to symmetric heap (captured in graph)
        iris_input.copy_(input_tensor)

        # Pre-kernel barrier
        if not is_graph_capturing():
            shmem.device_barrier()

        fp8_max = torch.finfo(quant_dtype).max
        heap_bases = shmem.get_heap_bases()

        # ---- Tunable parameters ----
        BLOCK_SIZE_M = 16        # rows per block (>= 16 for tl.dot matrix cores)
        COMM_SMS = 128           # number of persistent CTAs
        GEMM_BLOCK_N = 128       # output column tile size
        GEMM_BLOCK_K = 128       # reduction tile size (also comm tile width)
        EVEN_GEMM_K = (N % GEMM_BLOCK_K == 0)
        NUM_WARPS = 16
        NUM_STAGES = 2
        WAVES_PER_EU = 1
        # ---- End tunable parameters ----

        bias_ptr = bias if bias is not None else input_tensor

        persistent_fused_allreduce_rmsnorm_quant_partial_gemm[(COMM_SMS,)](
            # Heap buffers
            iris_input,
            gemm_heap,
            # Outputs
            result_out,
            # GEMM weight
            gemm_weight,
            bias_ptr,
            weight_scale,
            K_GEMM,
            gemm_weight.stride(0),
            gemm_weight.stride(1),
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
            result_out.stride(0),
            result_out.stride(1),
            # Iris
            heap_bases,
            rank_in_group,
            rank_global,
            world_size,
            rank_start,
            rank_stride,
            # Tile params
            BLOCK_SIZE_M,
            residual is not None,  # HAS_RESIDUAL
            bias is not None,      # HAS_BIAS
            COMM_SMS,
            # GEMM tile params
            GEMM_BLOCK_N,
            GEMM_BLOCK_K,
            EVEN_GEMM_K,
            num_warps=NUM_WARPS,
            num_stages=NUM_STAGES,
            waves_per_eu=WAVES_PER_EU,
        )

        # Post-kernel barrier ensures all ranks finished iris.store
        if not is_graph_capturing():
            shmem.device_barrier()

        # Copy assembled GEMM output from heap to local memory
        gemm_out.copy_(gemm_heap)

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
