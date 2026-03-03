# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Iris fused AllReduce + RMSNorm + per-row FP8 Quant + partial GEMM + allgather.

Key idea: each rank only GEMMs its own rows. No FP8 broadcast, no
internal cross-rank barrier. The GEMM reads from local quant data only.
After all ranks finish, an allgather assembles the full GEMM output.

Single persistent kernel (two phases + local grid barrier):
  Phase 1 (Comm): Each rank reduces its assigned rows by gathering from
    all ranks via iris.load, applies RMSNorm + per-row FP8 quant, stores
    FP8 + scales to LOCAL GPU memory (not broadcast).
  Local grid barrier: cheap atomic counter (~1-2us, no fabric traffic).
  Phase 2 (Partial GEMM): Standard tiled GEMM on this rank's rows only.
    Reads local FP8 quant + scales, writes BF16 output to gemm_heap on
    the iris symmetric heap (for allgather).

Manager flow:
  copy_ -> device_barrier -> fused_kernel -> device_barrier -> allgather_kernel

The allgather kernel reads each rank's partial GEMM output from the
heap via iris.load and assembles the full (M, K_GEMM) output.

Compared to iris_twoshot_row_hipblaslt:
  - No FP8 broadcast (saves iris.store traffic)
  - No cross-rank barrier between comm and GEMM (only local grid sync)
  - GEMM reads local memory only (better access patterns)
  - Extra allgather step for BF16 GEMM output

CUDA graph compatibility:
  - Buffer pre-allocation with view pattern
  - device_barrier() before and after kernel (graph-capturable)
  - Local grid barrier is graph-capturable (pure device-side atomics)
"""

import logging
from typing import Any, Optional, Tuple

import iris
import torch
import triton
import triton.language as tl

__all__ = ["fused_allreduce_add_rms_row_quant_gemm_iris_partial"]

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
# Fused AllReduce + RMSNorm + per-row FP8 Quant + partial GEMM kernel
# ============================================================================


@triton.jit
def fused_allreduce_partial_gemm_kernel(
    # Symmetric heap buffers (for allreduce iris.load)
    input_ptr,
    # Local GPU memory (FP8 quant + scales, not on heap)
    quant_local_ptr,
    scale_local_ptr,
    # BF16 allreduce result (regular GPU memory, owned rows only)
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
    # Quant local strides
    stride_ql_m,
    stride_ql_n,
    # GEMM params
    gemm_weight_ptr,  # (N, K_GEMM) FP8 weight
    gemm_heap_ptr,    # (M, K_GEMM) BF16 output on iris heap
    weight_scale_ptr, # per-tensor weight scale
    bias_ptr,         # optional bias (K_GEMM,); dummy when HAS_BIAS=False
    K_GEMM,
    stride_gw_k,      # gemm_weight stride dim 0 (N dim)
    stride_gw_n,      # gemm_weight stride dim 1 (K_GEMM dim)
    stride_gh_m,       # gemm_heap stride dim 0
    stride_gh_k,       # gemm_heap stride dim 1
    # Local grid barrier
    grid_sync_ptr,
    # Iris params
    heap_bases: tl.tensor,
    group_rank: tl.constexpr,
    iris_rank: tl.constexpr,
    world_size: tl.constexpr,
    rank_start: tl.constexpr,
    rank_stride: tl.constexpr,
    # Comm tile params
    BLOCK_SIZE_N: tl.constexpr,
    ACTUAL_N: tl.constexpr,
    HAS_RESIDUAL: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    PADDED_N: tl.constexpr,
    COMM_SMS: tl.constexpr,
    # GEMM tile params
    GEMM_BLOCK_M: tl.constexpr,
    GEMM_BLOCK_N: tl.constexpr,
    GEMM_BLOCK_K: tl.constexpr,
    EVEN_GEMM_K: tl.constexpr,
):
    """Fused allreduce + RMSNorm + FP8 quant + partial GEMM.

    Phase 1: allreduce own rows, RMSNorm, FP8 quant (local stores).
    Local grid barrier.
    Phase 2: tiled GEMM on own rows only, output to gemm_heap.
    """
    pid = tl.program_id(0)

    # Row distribution across ranks (block distribution)
    rows_per_rank = tl.cdiv(M, world_size)
    my_start = group_rank * rows_per_rank
    my_end = tl.minimum(my_start + rows_per_rank, M)
    my_M = my_end - my_start

    # ================================================================
    # Phase 1: AllReduce + RMSNorm + per-row FP8 quant (local only)
    # ================================================================

    cols = tl.arange(0, BLOCK_SIZE_N)

    if PADDED_N:
        col_mask = cols < ACTUAL_N
        rms_w = tl.load(
            rms_weight_ptr + cols, mask=col_mask, other=0.0
        ).to(tl.float32)
    else:
        rms_w = tl.load(rms_weight_ptr + cols).to(tl.float32)

    for row in range(my_start + pid, my_end, COMM_SMS):
        in_offset = row * stride_in_m + cols * stride_in_n
        ql_offset = row * stride_ql_m + cols * stride_ql_n
        out_offset = row * N + cols

        if PADDED_N:
            # ---- Slow path: masked ----
            acc = tl.zeros((BLOCK_SIZE_N,), dtype=tl.float32)
            for i in tl.static_range(0, world_size):
                remote_rank = rank_start + i * rank_stride
                val = iris.load(
                    input_ptr + in_offset,
                    iris_rank, remote_rank, heap_bases,
                    mask=col_mask,
                )
                acc += val.to(tl.float32)
                if HAS_RESIDUAL and remote_rank == iris_rank:
                    res_in = tl.load(
                        residual_in_ptr + out_offset, mask=col_mask, other=0.0
                    ).to(tl.float32)
                    acc += res_in

            tl.store(
                result_out_ptr + out_offset,
                acc.to(result_out_ptr.type.element_ty),
                mask=col_mask,
            )

            sq_sum = tl.sum(acc * acc, axis=0)
            rrms = tl.rsqrt(sq_sum / N + rms_eps)
            normed = acc * rrms * rms_w

            row_amax = tl.max(tl.abs(normed), axis=0)
            row_amax = tl.maximum(row_amax, 1e-12)
            scale = row_amax / fp8_max

            quantized = (normed / scale).to(quant_local_ptr.type.element_ty)

            tl.store(quant_local_ptr + ql_offset, quantized, mask=col_mask)
            tl.store(scale_local_ptr + row, scale)
        else:
            # ---- Fast path: no masks ----
            acc = tl.zeros((BLOCK_SIZE_N,), dtype=tl.float32)
            for i in tl.static_range(0, world_size):
                remote_rank = rank_start + i * rank_stride
                val = iris.load(
                    input_ptr + in_offset,
                    iris_rank, remote_rank, heap_bases,
                )
                acc += val.to(tl.float32)
                if HAS_RESIDUAL and remote_rank == iris_rank:
                    res_in = tl.load(
                        residual_in_ptr + out_offset
                    ).to(tl.float32)
                    acc += res_in

            tl.store(
                result_out_ptr + out_offset,
                acc.to(result_out_ptr.type.element_ty),
            )

            sq_sum = tl.sum(acc * acc, axis=0)
            rrms = tl.rsqrt(sq_sum / N + rms_eps)
            normed = acc * rrms * rms_w

            row_amax = tl.max(tl.abs(normed), axis=0)
            row_amax = tl.maximum(row_amax, 1e-12)
            scale = row_amax / fp8_max

            quantized = (normed / scale).to(quant_local_ptr.type.element_ty)

            tl.store(quant_local_ptr + ql_offset, quantized)
            tl.store(scale_local_ptr + row, scale)

    # ================================================================
    # Local grid barrier (all CTAs on this GPU finished phase 1)
    # ================================================================
    tl.atomic_add(grid_sync_ptr, 1, sem="release")
    while tl.atomic_add(grid_sync_ptr, 0, sem="acquire") < COMM_SMS:
        pass

    # ================================================================
    # Phase 2: Partial GEMM on own rows only
    # ================================================================
    # quant_local[my_start:my_end, :] @ gemm_weight[:, :] → gemm_heap[my_start:my_end, :]

    num_pid_m = tl.cdiv(my_M, GEMM_BLOCK_M)
    num_pid_n = tl.cdiv(K_GEMM, GEMM_BLOCK_N)
    total_tiles = num_pid_m * num_pid_n

    for tile_id in range(pid, total_tiles, COMM_SMS):
        pid_m = tile_id // num_pid_n
        pid_n = tile_id % num_pid_n

        # Row offsets relative to this rank's start
        offs_am = my_start + pid_m * GEMM_BLOCK_M + tl.arange(0, GEMM_BLOCK_M)
        offs_bn = pid_n * GEMM_BLOCK_N + tl.arange(0, GEMM_BLOCK_N)
        offs_k = tl.arange(0, GEMM_BLOCK_K)

        # A pointers: read from quant_local (local GPU memory)
        a_ptrs = (
            quant_local_ptr
            + offs_am[:, None] * stride_ql_m
            + offs_k[None, :] * stride_ql_n
        )
        # B pointers: gemm_weight (N, K_GEMM)
        b_ptrs = (
            gemm_weight_ptr
            + offs_k[:, None] * stride_gw_k
            + offs_bn[None, :] * stride_gw_n
        )

        # Per-row activation scales
        a_scale = tl.load(scale_local_ptr + offs_am, mask=offs_am < my_end)

        # Tiled GEMM accumulation
        accumulator = tl.zeros((GEMM_BLOCK_M, GEMM_BLOCK_N), dtype=tl.float32)

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

        # Optional bias
        if HAS_BIAS:
            bias_offs = pid_n * GEMM_BLOCK_N + tl.arange(0, GEMM_BLOCK_N)
            bias_val = tl.load(bias_ptr + bias_offs, mask=bias_offs < K_GEMM)
            accumulator = accumulator.to(bias_ptr.type.element_ty) + bias_val[None, :]

        # Store to gemm_heap (iris symmetric heap for allgather)
        c = accumulator.to(gemm_heap_ptr.type.element_ty)
        offs_cm = my_start + pid_m * GEMM_BLOCK_M + tl.arange(0, GEMM_BLOCK_M)
        offs_cn = pid_n * GEMM_BLOCK_N + tl.arange(0, GEMM_BLOCK_N)
        c_ptrs = (
            gemm_heap_ptr
            + offs_cm[:, None] * stride_gh_m
            + offs_cn[None, :] * stride_gh_k
        )
        c_mask = (offs_cm[:, None] < my_end) & (offs_cn[None, :] < K_GEMM)
        tl.store(c_ptrs, c, mask=c_mask)


# ============================================================================
# Allgather kernel: read each rank's partial GEMM output from heap
# ============================================================================


@triton.jit
def allgather_gemm_output_kernel(
    gemm_heap_ptr,
    gemm_out_ptr,
    M,
    K_GEMM,
    stride_gh_m,
    stride_gh_k,
    stride_go_m,
    stride_go_k,
    heap_bases: tl.tensor,
    iris_rank: tl.constexpr,
    world_size: tl.constexpr,
    rank_start: tl.constexpr,
    rank_stride: tl.constexpr,
    rows_per_rank,
    BLOCK_K: tl.constexpr,
    NUM_SMS: tl.constexpr,
):
    """Allgather: iris.load each rank's partial GEMM output into full output.

    Each CTA handles a slice of rows. For each row, determines which
    rank produced it and iris.loads from that rank's gemm_heap.
    Local rank's rows are copied directly (tl.load).
    """
    pid = tl.program_id(0)
    cols = tl.arange(0, BLOCK_K)
    num_col_blocks = tl.cdiv(K_GEMM, BLOCK_K)
    total_work = M * num_col_blocks

    for work_id in range(pid, total_work, NUM_SMS):
        row = work_id // num_col_blocks
        col_block = work_id % num_col_blocks
        col_offset = col_block * BLOCK_K + cols
        col_mask = col_offset < K_GEMM

        # Determine which rank owns this row
        owner_rank = row // rows_per_rank
        owner_rank = tl.minimum(owner_rank, world_size - 1)

        heap_offset = row * stride_gh_m + col_offset * stride_gh_k
        out_offset = row * stride_go_m + col_offset * stride_go_k

        remote_rank = rank_start + owner_rank * rank_stride
        if remote_rank == iris_rank:
            val = tl.load(gemm_heap_ptr + heap_offset, mask=col_mask)
        else:
            val = iris.load(
                gemm_heap_ptr + heap_offset,
                iris_rank, remote_rank, heap_bases,
                mask=col_mask,
            )

        tl.store(gemm_out_ptr + out_offset, val, mask=col_mask)


# ============================================================================
# Manager and public API
# ============================================================================


class IrisPartialGemmManager:
    """Singleton manager for fused AllReduce + partial GEMM + allgather.

    No FP8 broadcast, no internal cross-rank barrier. GEMM operates
    on local data only. Allgather assembles the full output via iris.load.
    """

    _instance: Optional["IrisPartialGemmManager"] = None
    _initialized: bool = False

    def __new__(cls) -> "IrisPartialGemmManager":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if IrisPartialGemmManager._initialized:
            return
        IrisPartialGemmManager._initialized = True

        self._shmem: Any = None
        self._heap_size: int = 2**33  # 8GB default

        # Symmetric heap: input buffer (for allreduce iris.load)
        self._input_buf: Any = None
        self._input_M: int = 0
        self._input_N: int = 0
        self._input_dtype: Optional[torch.dtype] = None

        # Symmetric heap: GEMM output buffer (for allgather iris.load)
        self._gemm_heap_buf: Any = None
        self._gemm_heap_M: int = 0
        self._gemm_heap_K: int = 0

        # Regular GPU memory: FP8 quant + scales (local only, not on heap)
        self._quant_local: Optional[torch.Tensor] = None
        self._quant_M: int = 0
        self._quant_N: int = 0
        self._quant_dtype: Optional[torch.dtype] = None

        self._scale_local: Optional[torch.Tensor] = None
        self._scale_M: int = 0

        # Regular GPU memory: BF16 allreduce result (owned rows)
        self._out_result: Optional[torch.Tensor] = None

        # Regular GPU memory: assembled GEMM output
        self._out_gemm_bufs: dict[int, torch.Tensor] = {}

        # Local grid barrier (regular GPU memory)
        self._grid_sync: Optional[torch.Tensor] = None

    def initialize(self, heap_size: Optional[int] = None) -> None:
        """Initialize Iris symmetric heap (call once at startup)."""
        if self._shmem is not None:
            logger.debug(
                "Iris (partial-gemm) already initialized, skipping"
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
            "Initializing Iris (partial-gemm) symmetric heap: "
            f"rank={cur_rank}, heap_size={self._heap_size / 2**30:.1f}GB"
        )

        self._shmem = iris.iris(self._heap_size)

        logger.info(
            f"Iris (partial-gemm) initialized successfully on rank {cur_rank}"
        )

    @property
    def shmem(self) -> Any:
        if self._shmem is None:
            self.initialize()
        return self._shmem

    def _ensure_buffers(
        self,
        M: int,
        N: int,
        K_GEMM: int,
        input_dtype: torch.dtype,
        quant_dtype: torch.dtype,
        out_dtype: torch.dtype,
        device: torch.device,
    ) -> Tuple[Any, torch.Tensor, torch.Tensor, torch.Tensor, Any, torch.Tensor]:
        """Allocate or reuse all buffers.

        Returns: (input_buf, quant_local, scale_local, result_out,
                  gemm_heap, gemm_out)
        """
        shmem = self.shmem
        capturing = torch.cuda.is_current_stream_capturing()

        # Input buffer (iris heap)
        if (self._input_buf is None or M > self._input_M
                or N != self._input_N or input_dtype != self._input_dtype):
            if capturing:
                raise RuntimeError(
                    f"Iris (partial-gemm): input buffer too small. "
                    f"Cannot allocate during CUDA graph capture."
                )
            self._input_buf = shmem.zeros((M, N), dtype=input_dtype)
            self._input_M = M
            self._input_N = N
            self._input_dtype = input_dtype

        # GEMM output on heap (iris heap, for allgather)
        if (self._gemm_heap_buf is None or M > self._gemm_heap_M
                or K_GEMM > self._gemm_heap_K):
            if capturing:
                raise RuntimeError(
                    f"Iris (partial-gemm): gemm heap buffer too small. "
                    f"Cannot allocate during CUDA graph capture."
                )
            self._gemm_heap_buf = shmem.zeros((M, K_GEMM), dtype=out_dtype)
            self._gemm_heap_M = M
            self._gemm_heap_K = K_GEMM

        # Quant local (regular GPU memory)
        if (self._quant_local is None or M > self._quant_M
                or N != self._quant_N or quant_dtype != self._quant_dtype):
            if capturing:
                raise RuntimeError(
                    f"Iris (partial-gemm): quant buffer too small. "
                    f"Cannot allocate during CUDA graph capture."
                )
            self._quant_local = torch.empty(
                (M, N), dtype=quant_dtype, device=device
            )
            self._quant_M = M
            self._quant_N = N
            self._quant_dtype = quant_dtype

        # Scale local (regular GPU memory)
        if self._scale_local is None or M > self._scale_M:
            if capturing:
                raise RuntimeError(
                    f"Iris (partial-gemm): scale buffer too small. "
                    f"Cannot allocate during CUDA graph capture."
                )
            self._scale_local = torch.empty(
                (M,), dtype=torch.float32, device=device
            )
            self._scale_M = M

        # Result out (regular GPU memory)
        if (self._out_result is None or M > self._out_result.shape[0]
                or N != self._out_result.shape[1]):
            if capturing:
                raise RuntimeError(
                    f"Iris (partial-gemm): result buffer too small. "
                    f"Cannot allocate during CUDA graph capture."
                )
            self._out_result = torch.empty(
                (M, N), dtype=input_dtype, device=device
            )

        # GEMM output (regular GPU memory, assembled)
        if (K_GEMM not in self._out_gemm_bufs
                or M > self._out_gemm_bufs[K_GEMM].shape[0]):
            if capturing:
                raise RuntimeError(
                    f"Iris (partial-gemm): gemm output buffer too small. "
                    f"Cannot allocate during CUDA graph capture."
                )
            self._out_gemm_bufs[K_GEMM] = torch.empty(
                (M, K_GEMM), dtype=out_dtype, device=device
            )

        # Grid sync counter
        if self._grid_sync is None:
            self._grid_sync = torch.zeros(
                1, dtype=torch.int32, device=device
            )

        return (
            self._input_buf[:M],
            self._quant_local[:M],
            self._scale_local[:M],
            self._out_result[:M],
            self._gemm_heap_buf[:M],
            self._out_gemm_bufs[K_GEMM][:M],
        )

    def fused_allreduce_partial_gemm(
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

        # Allocate buffers
        (iris_input, quant_local, scale_local,
         result_out, gemm_heap, gemm_out) = self._ensure_buffers(
            M, N, K_GEMM, input_tensor.dtype, quant_dtype,
            out_dtype, device,
        )

        # Copy input to symmetric heap
        iris_input.copy_(input_tensor)

        # Pre-kernel barrier
        shmem.device_barrier()

        # FP8 max value
        fp8_max = torch.finfo(quant_dtype).max

        # Iris group info
        rank_in_group, rank_global, world_size, rank_start, rank_stride = (
            extract_group_info(shmem)
        )
        heap_bases = shmem.get_heap_bases()

        # ---- Tunable parameters ----
        BLOCK_SIZE_N = triton.next_power_of_2(N)
        ACTUAL_N = N
        PADDED_N = (BLOCK_SIZE_N != N)
        COMM_SMS = 128
        GEMM_BLOCK_M = 32
        GEMM_BLOCK_N = 128
        GEMM_BLOCK_K = 128
        EVEN_GEMM_K = (N % GEMM_BLOCK_K == 0)
        NUM_WARPS = 16
        NUM_STAGES = 2
        WAVES_PER_EU = 1
        # ---- End tunable parameters ----

        # Reset grid sync counter
        assert self._grid_sync is not None
        self._grid_sync.zero_()

        # Launch fused kernel
        fused_allreduce_partial_gemm_kernel[(COMM_SMS,)](
            iris_input,
            quant_local,
            scale_local,
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
            quant_local.stride(0),
            quant_local.stride(1),
            # GEMM params
            gemm_weight,
            gemm_heap,
            weight_scale,
            bias if bias is not None else input_tensor,
            K_GEMM,
            gemm_weight.stride(0),
            gemm_weight.stride(1),
            gemm_heap.stride(0),
            gemm_heap.stride(1),
            # Grid sync
            self._grid_sync,
            # Iris
            heap_bases,
            rank_in_group,
            rank_global,
            world_size,
            rank_start,
            rank_stride,
            # Tile params
            BLOCK_SIZE_N,
            ACTUAL_N,
            residual is not None,  # HAS_RESIDUAL
            bias is not None,      # HAS_BIAS
            PADDED_N,
            COMM_SMS,
            GEMM_BLOCK_M,
            GEMM_BLOCK_N,
            GEMM_BLOCK_K,
            EVEN_GEMM_K,
            num_warps=NUM_WARPS,
            num_stages=NUM_STAGES,
            waves_per_eu=WAVES_PER_EU,
        )

        # Post-kernel barrier: all ranks finished writing to gemm_heap
        shmem.device_barrier()

        # Allgather: read each rank's partial GEMM output into full output
        rows_per_rank = (M + world_size - 1) // world_size
        ALLGATHER_BLOCK_K = 128
        ALLGATHER_SMS = 128

        allgather_gemm_output_kernel[(ALLGATHER_SMS,)](
            gemm_heap,
            gemm_out,
            M,
            K_GEMM,
            gemm_heap.stride(0),
            gemm_heap.stride(1),
            gemm_out.stride(0),
            gemm_out.stride(1),
            heap_bases,
            rank_global,
            world_size,
            rank_start,
            rank_stride,
            rows_per_rank,
            ALLGATHER_BLOCK_K,
            ALLGATHER_SMS,
            num_warps=4,
        )

        residual_out = result_out if residual is not None else None

        return gemm_out, residual_out


_manager: Optional[IrisPartialGemmManager] = None


def _get_manager() -> IrisPartialGemmManager:
    global _manager
    if _manager is None:
        _manager = IrisPartialGemmManager()
    return _manager


def initialize_iris_partial_gemm(
    heap_size: Optional[int] = None,
) -> None:
    """Initialize Iris for partial GEMM + allgather fused all-reduce.

    Call once at model load time before any forward passes.
    """
    _get_manager().initialize(heap_size)


def fused_allreduce_add_rms_row_quant_gemm_iris_partial(
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
    """Fused AllReduce + RMSNorm + per-row FP8 Quant + partial GEMM + allgather.

    Returns (gemm_out, residual_out).
    """
    return (
        _get_manager()
        .fused_allreduce_partial_gemm(
            input, rms_weight, rms_eps, quant_dtype,
            gemm_weight, weight_scale, out_dtype, residual, bias,
        )
    )
