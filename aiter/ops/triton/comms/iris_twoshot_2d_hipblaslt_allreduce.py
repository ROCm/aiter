# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Iris two-shot fused AllReduce + RMSNorm + per-row FP8 Quant, with
hipBLASLt GEMM (torch._scaled_mm).

Closely mirrors the iris CCL `persistent_all_reduce_two_shot` kernel
structure with 2D tiling (BLOCK_SIZE_M x BLOCK_SIZE_N). The only
difference from the plain two-shot allreduce is an RMSNorm + per-row
FP8 quantization step inserted between the reduce and broadcast phases.

Tile assignment:
  - Block distribution: each rank owns a contiguous range of tiles.
  - BLOCK_SIZE_M rows are batched per tile iteration. BLOCK_SIZE_N
    spans the full hidden dimension (next_power_of_2(N)).
  - RMSNorm requires a full-row reduction. Since BLOCK_SIZE_N covers
    the entire row, `tl.sum(..., axis=1)` gives the correct per-row
    variance within each 2D tile.

After the Triton kernel completes:
  - quant_heap: (M, N) FP8 matrix on symmetric heap (all rows, all ranks)
  - scale_heap: (M,) per-row float32 scales on symmetric heap
  - result_out: (M, N) BF16 allreduced values (only owned rows)

Then torch._scaled_mm is called for the GEMM via hipBLASLt.

CUDA graph compatibility:
- Buffer pre-allocation with view pattern
- Device barriers are inlined (no separate kernel launch)
"""

import logging
from typing import Any, Optional, Tuple

import iris
import torch
import triton
import triton.language as tl

__all__ = ["fused_allreduce_add_rms_row_quant_gemm_iris_twoshot_2d_hipblaslt"]

logger = logging.getLogger(__name__)


# ============================================================================
# Utilities (inlined from iris.ccl.utils and iris._distributed_helpers)
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
    other CTAs via barrier_done_ptr. Epoch is self-advancing for
    CUDA graph compatibility.
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
# Fused two-shot AllReduce + RMSNorm + per-row FP8 Quant kernel (2D tiling)
# ============================================================================


@triton.jit
def persistent_fused_allreduce_rmsnorm_row_quant_two_shot(
    # Symmetric heap buffers
    input_ptr,
    quant_heap_ptr,
    scale_heap_ptr,
    # Regular GPU memory
    result_out_ptr,
    # Inlined device barrier state
    barrier_flags_ptr,
    barrier_epoch_ptr,
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
    GROUP_SIZE_M: tl.constexpr,
    HAS_RESIDUAL: tl.constexpr,
    PADDED_N: tl.constexpr,
    COMM_SMS: tl.constexpr,
    NUM_XCDS: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    DISTRIBUTION: tl.constexpr,
):
    """Two-shot fused AllReduce + RMSNorm + per-row FP8 quant.

    Mirrors persistent_all_reduce_two_shot from iris CCL. Each rank
    reduces its assigned tiles, applies RMSNorm + per-row FP8 quant,
    then broadcasts the FP8 result and scales to all peers.

    Uses fast/slow path pattern from flash attention:
    - PADDED_N: constexpr flag, True when BLOCK_SIZE_N != ACTUAL_N.
      When False, column masking is eliminated entirely.
    - is_full: runtime check per tile. When True (rm_base + BLOCK_SIZE_M <= M),
      row masking is also eliminated.
    """
    pid = tl.program_id(0)

    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    total_tiles = num_pid_m * num_pid_n

    # Tile distribution across ranks (same as CCL two_shot)
    tiles_per_rank = tl.cdiv(total_tiles, world_size)
    if DISTRIBUTION == 0:
        start_tile = group_rank
        stride = world_size
        remaining = total_tiles - start_tile
        remaining = tl.maximum(remaining, 0)
        max_tile_offset = tl.cdiv(remaining, stride)
    else:
        start_tile = group_rank * tiles_per_rank
        stride = 1
        remaining = total_tiles - start_tile
        remaining = tl.maximum(remaining, 0)
        max_tile_offset = tl.minimum(tiles_per_rank, remaining)

    # Load RMSNorm weights (shared across all tiles)
    rn_w = tl.arange(0, BLOCK_SIZE_N)
    if PADDED_N:
        rms_w = tl.load(
            rms_weight_ptr + rn_w, mask=rn_w < ACTUAL_N, other=0.0
        ).to(tl.float32)
    else:
        rms_w = tl.load(rms_weight_ptr + rn_w).to(tl.float32)

    # Persistent traversal over this rank's assigned tiles
    for tile_offset in range(pid, max_tile_offset, COMM_SMS):
        tile_id = start_tile + tile_offset * stride

        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = tile_id // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((tile_id % num_pid_in_group) % group_size_m)
        pid_n = (tile_id % num_pid_in_group) // group_size_m

        rm_base = pid_m * BLOCK_SIZE_M
        rn_base = pid_n * BLOCK_SIZE_N

        is_full = (rm_base + BLOCK_SIZE_M <= M) & (rn_base + BLOCK_SIZE_N <= N)

        # Build 2D indices
        rm = rm_base + tl.arange(0, BLOCK_SIZE_M)
        rn = rn_base + tl.arange(0, BLOCK_SIZE_N)
        rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
        rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)

        input_offset = rm[:, None] * stride_in_m + rn[None, :] * stride_in_n
        qh_offset = rm[:, None] * stride_qh_m + rn[None, :] * stride_qh_n
        out_offset = rm[:, None] * stride_out_m + rn[None, :] * stride_out_n
        scale_offset = rm * stride_sh

        # ---- Fast path: no masks (full tiles) ----
        if is_full:
            row_mask = rm < M
            mask = row_mask[:, None] & (rn < N)[None, :]

            acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
            for i in tl.static_range(0, world_size):
                remote_rank = rank_start + i * rank_stride
                partial = iris.load(
                    input_ptr + input_offset,
                    iris_rank, remote_rank, heap_bases,
                )
                acc += partial.to(tl.float32)
                if HAS_RESIDUAL and remote_rank == iris_rank:
                    res_in = tl.load(residual_in_ptr + out_offset).to(tl.float32)
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

        # ---- Slow path: masked (boundary tiles) ----
        else:
            row_mask = rm < M
            if PADDED_N:
                col_mask = rn < ACTUAL_N
                mask = row_mask[:, None] & col_mask[None, :]
            else:
                mask = row_mask[:, None] & (rn < N)[None, :]

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

    # ---- Cross-rank device barrier ----
    _inlined_device_barrier(
        tl.program_id(0),
        barrier_flags_ptr,
        barrier_epoch_ptr,
        barrier_done_ptr,
        heap_bases,
        iris_rank,
        world_size,
        rank_start,
        rank_stride,
    )


# ============================================================================
# Manager and public API
# ============================================================================


class IrisTwoshot2dHipblasltManager:
    """Singleton manager for 2D-tiled two-shot AllReduce+RMSNorm+per-row Quant
    + hipBLASLt GEMM.

    Manages symmetric heap buffers (input, quant, scale) and GPU memory
    (result_out, barrier state). GEMM is delegated to torch._scaled_mm.
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

        # Inlined device barrier state
        self._barrier_flags: Any = None  # symmetric heap
        self._barrier_epoch: Optional[torch.Tensor] = None
        self._barrier_done: Optional[torch.Tensor] = None

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
        logger.info(
            "Initializing Iris (twoshot-2d-hipblaslt) symmetric heap: "
            f"rank={cur_rank}, heap_size={self._heap_size / 2**30:.1f}GB"
        )

        self._shmem = iris.iris(self._heap_size)

        num_ranks = self._shmem.get_num_ranks()
        self._barrier_flags = self._shmem.zeros(
            (num_ranks,), dtype=torch.int32
        )
        self._shmem.device_barrier()

        logger.info(
            f"Iris (twoshot-2d-hipblaslt) initialized successfully "
            f"on rank {cur_rank}"
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
            logger.info(
                f"Iris (twoshot-2d-hipblaslt): allocated input buffer "
                f"({M}, {N}), dtype={input_dtype}, rank={cur_rank}"
            )

        if need_quant:
            self._quant_heap_buf = shmem.zeros((M, N), dtype=quant_dtype)
            self._quant_heap_M = M
            self._quant_heap_N = N
            self._quant_heap_dtype = quant_dtype
            logger.info(
                f"Iris (twoshot-2d-hipblaslt): allocated quant heap buffer "
                f"({M}, {N}), dtype={quant_dtype}, rank={cur_rank}"
            )

        if need_scale:
            self._scale_heap_buf = shmem.zeros((M,), dtype=torch.float32)
            self._scale_heap_M = M
            logger.info(
                f"Iris (twoshot-2d-hipblaslt): allocated scale heap buffer "
                f"({M},), dtype=float32, rank={cur_rank}"
            )

        shmem.device_barrier()

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
        """Allocate or reuse result_out and barrier buffers on GPU memory.

        Returns result_out.
        """
        need_result = (
            self._out_result is None
            or M > self._out_result.shape[0]
            or N != self._out_result.shape[1]
        )
        need_barrier = self._barrier_done is None

        if not (need_result or need_barrier):
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

        if need_result:
            self._out_result = torch.empty(
                (M, N), dtype=dtype, device=device,
            )
            logger.info(
                f"Iris (twoshot-2d-hipblaslt): allocated result_out "
                f"({M}, {N}), dtype={dtype}, rank={cur_rank}"
            )

        if need_barrier:
            self._barrier_done = torch.zeros(
                1, dtype=torch.int32, device=device
            )
            self._barrier_epoch = torch.zeros(
                1, dtype=torch.int32, device=device
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

        Args:
            input_tensor: (M, N) BF16 on GPU
            rms_weight: (N,) RMSNorm weight
            rms_eps: RMSNorm epsilon
            quant_dtype: FP8 dtype (e.g. torch.float8_e4m3fn)
            gemm_weight: (N, K_GEMM) FP8 weight
            weight_scale: per-tensor scale for gemm_weight
            out_dtype: output dtype for GEMM (e.g. torch.bfloat16)
            residual: optional (M, N) residual tensor
            bias: optional (K_GEMM,) bias

        Returns:
            (gemm_out, residual_out). residual_out is None when
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

        # Copy input to symmetric heap
        iris_input.copy_(input_tensor)

        # FP8 max value
        fp8_max = torch.finfo(quant_dtype).max

        # Iris group info
        rank_in_group, rank_global, world_size, rank_start, rank_stride = (
            extract_group_info(shmem)
        )
        heap_bases = shmem.get_heap_bases()

        # ---- Tunable parameters ----
        num_xcds = iris.hip.get_num_xcc()
        BLOCK_SIZE_M = 4
        BLOCK_SIZE_N = triton.next_power_of_2(N)
        ACTUAL_N = N                       # real hidden dim (for PADDED_N flag)
        PADDED_N = (BLOCK_SIZE_N != N)     # True when N is not a power of 2
        GROUP_SIZE_M = 4
        DISTRIBUTION = 1       # 0=striding, 1=block
        COMM_SMS = 128
        CHUNK_SIZE = _compute_chunk_size(COMM_SMS, num_xcds)
        NUM_WARPS = 8
        NUM_STAGES = 1
        WAVES_PER_EU = 1
        # ---- End tunable parameters ----

        # Reset barrier
        assert self._barrier_done is not None
        self._barrier_done.zero_()

        # Launch kernel
        persistent_fused_allreduce_rmsnorm_row_quant_two_shot[(COMM_SMS,)](
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
            GROUP_SIZE_M,
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


def fused_allreduce_add_rms_row_quant_gemm_iris_twoshot_2d_hipblaslt(
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
