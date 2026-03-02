# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Iris two-shot fused AllReduce + RMSNorm + per-row FP8 Quant + inlined GEMM.

FP8 broadcast variant: the symmetric heap carries FP8 data AND per-row
scales instead of BF16, halving cross-rank traffic for the dominant
data payload.  The GEMM reads directly from the heap buffer after the
barrier, eliminating the FP8 round-trip through global memory and the
separate GEMM kernel launch.

Single persistent kernel:
  Step 1 (Two-shot AllReduce + fused RMSNorm + per-row quant + FP8 broadcast):
    Each rank reduces its assigned rows by gathering from all ranks,
    optionally pre-adds the residual for its own partial, stores the
    BF16 result (owned rows only), then RMSNorm + per-row FP8 quant.
    The FP8 result and per-row scale are broadcast to all peers via
    the symmetric heap.  No separate quant_out buffer is written.
  Step 2 (Inlined device barrier): CTA 0 performs cross-rank
    synchronization via atomic ops on the symmetric heap. Other CTAs
    spin on a local flag until CTA 0 signals completion.
  Step 3 (Inlined GEMM): After the barrier, all CTAs transition to
    GEMM mode. The complete (M, N) FP8 matrix and per-row scales are
    read directly from the heap buffers. A persistent tiled GEMM
    computes C = scale_a * (A_fp8 @ W_fp8) * weight_scale,
    producing BF16 output.

Compared to delayed-scaling twoshot (iris_twoshot_delayed_allreduce.py):
- No delayed scaling bookkeeping (no cross-rank amax sync, no
  prev_scale/current_amax state)
- Per-row scales: shape (M, 1) instead of (1,), computed independently
  per row with no cross-CTA or cross-rank coordination
- Broadcasts both FP8 data and per-row scales (scale is one float32
  per row, tiny compared to the FP8 data)

Compared to BF16-broadcast per-row (original iris_twoshot_row):
- Broadcasts FP8 + scale instead of BF16, halving the dominant traffic
- GEMM is inlined: no separate kernel launch, no quant_out buffer

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

__all__ = ["fused_allreduce_add_rms_row_quant_gemm_iris_twoshot"]

logger = logging.getLogger(__name__)

AUTOTUNE = False
if AUTOTUNE:
    os.environ["TRITON_PRINT_AUTOTUNING"] = "1"


# Autotune keys — includes K_GEMM since GEMM tiling depends on it
IRIS_TWOSHOT_ROW_GEMM_AUTOTUNE_KEYS = [
    "M",
    "N",
    "K_GEMM",
    "HAS_RESIDUAL",
]

def _get_iris_twoshot_row_gemm_configs(autotune: bool):
    """Generate autotune configs including GEMM block size variations."""
    num_xcds = iris.hip.get_num_xcc()

    if not autotune:
        # Best config from autotuning on MI350X 8-GPU (512 tokens, 8192 hidden):
        # COMM_SMS=128, GEMM_BLOCK_M=256, num_warps=16, waves_per_eu=1
        comm_sms = 128
        return [
            triton.Config(
                {
                    "COMM_SMS": comm_sms,
                    "NUM_XCDS": num_xcds,
                    "CHUNK_SIZE": _compute_chunk_size(comm_sms, num_xcds),
                    "waves_per_eu": 1,
                    "GEMM_BLOCK_M": 256,
                    "GEMM_BLOCK_N": 128,
                    "GEMM_BLOCK_K": 128,
                    "GEMM_GROUP_SIZE_M": 4,
                },
                num_warps=16,
                num_stages=2,
            )
        ]

    # ===================== Autotune Sweep =====================
    # Best known config: COMM_SMS=128, GEMM_BLOCK_M=256, GEMM_BLOCK_N=128,
    # GEMM_BLOCK_K=128, GROUP_SIZE_M=4, num_warps=16, num_stages=2,
    # waves_per_eu=1.  Add neighboring values to any list to search around it.
    configs = []
    # Comm parameters
    COMM_SMS_OPTIONS = [64, 128]
    # GEMM tile parameters
    GEMM_BLOCK_M_OPTIONS = [128, 256]
    GEMM_BLOCK_N_OPTIONS = [128]
    GEMM_BLOCK_K_OPTIONS = [128]
    GEMM_GROUP_SIZE_M_OPTIONS = [4]
    # Shared parameters
    NUM_WARPS_OPTIONS = [4, 16]
    NUM_STAGES_OPTIONS = [2]
    WAVES_PER_EU_OPTIONS = [1]
    for sms in COMM_SMS_OPTIONS:
        chunk = _compute_chunk_size(sms, num_xcds)
        for gm in GEMM_BLOCK_M_OPTIONS:
            for gn in GEMM_BLOCK_N_OPTIONS:
                for gk in GEMM_BLOCK_K_OPTIONS:
                    for gsm in GEMM_GROUP_SIZE_M_OPTIONS:
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
                                                "GEMM_BLOCK_M": gm,
                                                "GEMM_BLOCK_N": gn,
                                                "GEMM_BLOCK_K": gk,
                                                "GEMM_GROUP_SIZE_M": gsm,
                                            },
                                            num_stages=ns,
                                            num_warps=nw,
                                        )
                                    )
    return configs


iris_twoshot_row_autotune_configs = _get_iris_twoshot_row_gemm_configs(
    AUTOTUNE
)


# ============================================================================
# Fused two-shot AllReduce + RMSNorm + per-row FP8 Quant + GEMM kernel
# (FP8 broadcast variant with inlined GEMM)
# ============================================================================


@triton.autotune(
    configs=iris_twoshot_row_autotune_configs,
    key=IRIS_TWOSHOT_ROW_GEMM_AUTOTUNE_KEYS,
    use_cuda_graph=True,
)
@triton.heuristics(
    {"EVEN_GEMM_K": lambda args: args["N"] % args["GEMM_BLOCK_K"] == 0}
)
@triton.jit
def fused_twoshot_allreduce_rmsnorm_row_quant_gemm_kernel(
    # Input (in symmetric heap for iris.load)
    input_ptr,
    # FP8 quant heap (in symmetric heap for FP8 broadcast)
    quant_heap_ptr,
    # Scale heap (in symmetric heap for per-row scale broadcast)
    scale_heap_ptr,
    # Outputs (regular GPU memory)
    result_out_ptr,
    # GEMM parameters
    gemm_weight_ptr,  # (N, K_GEMM) fp8 weight matrix
    gemm_out_ptr,  # (M, K_GEMM) output buffer
    bias_ptr,  # optional bias (K_GEMM,); dummy ptr when HAS_BIAS=False
    weight_scale_ptr,  # pointer to float32 scalar (per-tensor weight scale)
    K_GEMM,  # GEMM output dimension
    stride_gw_k,  # gemm_weight stride dim 0
    stride_gw_n,  # gemm_weight stride dim 1
    stride_go_m,  # gemm_out stride dim 0
    stride_go_k,  # gemm_out stride dim 1
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
    HAS_BIAS: tl.constexpr,
    # Autotuned params (from config)
    COMM_SMS: tl.constexpr,
    NUM_XCDS: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    GEMM_BLOCK_M: tl.constexpr,
    GEMM_BLOCK_N: tl.constexpr,
    GEMM_BLOCK_K: tl.constexpr,
    GEMM_GROUP_SIZE_M: tl.constexpr,
    # Heuristic params
    EVEN_GEMM_K: tl.constexpr,
):
    """Fused two-shot AllReduce + RMSNorm + per-row FP8 quant + inlined GEMM.

    FP8 broadcast: broadcasts FP8 quant results and per-row scales
    instead of BF16 allreduce results, halving cross-rank traffic.
    Only owned rows get BF16 result_out and residual addition.
    After barrier, all CTAs perform a persistent tiled GEMM reading
    FP8 data and per-row scales directly from the heap buffers.
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

        # Store BF16 result (owned rows only) -- serves as both
        # allreduce_out and residual_out for downstream
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

    # ================================================================
    # Step 3: Inlined GEMM — read FP8 + scales from heap, produce BF16
    # ================================================================
    # After barrier, quant_heap_ptr contains the complete (M, N) FP8
    # matrix and scale_heap_ptr contains per-row scales for ALL rows.
    # All COMM_SMS CTAs cooperatively tile the (M, K_GEMM) output.

    num_pid_m = tl.cdiv(M, GEMM_BLOCK_M)
    num_pid_n = tl.cdiv(K_GEMM, GEMM_BLOCK_N)
    total_tiles = num_pid_m * num_pid_n

    for tile_id in range(pid, total_tiles, COMM_SMS):
        # Map tile_id → (pid_m, pid_n) with GROUP_SIZE_M swizzle
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

        # Offsets for this tile
        offs_am = pid_m * GEMM_BLOCK_M + tl.arange(0, GEMM_BLOCK_M)
        offs_bn = pid_n * GEMM_BLOCK_N + tl.arange(0, GEMM_BLOCK_N)
        offs_k = tl.arange(0, GEMM_BLOCK_K)

        # A pointers: read from quant_heap (M, N) — A is (M, N), B is (N, K_GEMM)
        # stride_qh_m, stride_qh_n are heap strides
        a_ptrs = (
            quant_heap_ptr
            + (offs_am[:, None] % M) * stride_qh_m
            + offs_k[None, :] * stride_qh_n
        )
        # B pointers: gemm_weight is (N, K_GEMM) with strides (stride_gw_k, stride_gw_n)
        # Note: gemm_weight layout is (K=N, N=K_GEMM) in the GEMM sense
        b_ptrs = (
            gemm_weight_ptr
            + offs_k[:, None] * stride_gw_k
            + (offs_bn[None, :] % K_GEMM) * stride_gw_n
        )

        # Load per-row activation scales for this tile's rows
        a_scale_offs = pid_m * GEMM_BLOCK_M + tl.arange(0, GEMM_BLOCK_M)
        a_scale = tl.load(
            scale_heap_ptr + (a_scale_offs % M)
        )

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

            # Advance pointers along K (= N) dimension
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
            accumulator = accumulator.to(bias_ptr.type.element_ty) + bias_val[None, :]

        # Convert to output dtype and store
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


class IrisTwoshotRowManager:
    """Singleton manager for two-shot AllReduce+RMSNorm+per-row Quant+GEMM.

    FP8 broadcast variant: broadcasts FP8 and per-row scales on the
    heap instead of BF16. Only owned rows get BF16 result_out and
    residual addition. GEMM is inlined in the kernel — reads FP8 and
    scales directly from the heap after the barrier.
    """

    _instance: Optional["IrisTwoshotRowManager"] = None
    _initialized: bool = False

    def __new__(cls) -> "IrisTwoshotRowManager":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if IrisTwoshotRowManager._initialized:
            return
        IrisTwoshotRowManager._initialized = True

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
        self._out_gemm: Optional[torch.Tensor] = None  # (M, K_GEMM)
        self._out_gemm_K: int = 0

        # Inlined device barrier state
        self._barrier_flags: Any = None  # on symmetric heap
        self._barrier_epoch: Optional[torch.Tensor] = None  # GPU tensor for graph compat
        self._barrier_done: Optional[torch.Tensor] = None  # local GPU memory

    def initialize(self, heap_size: Optional[int] = None) -> None:
        """Initialize Iris symmetric heap (call once at startup)."""
        if self._shmem is not None:
            logger.debug("Iris (twoshot-row) already initialized, skipping")
            return

        if heap_size is not None:
            self._heap_size = heap_size

        cur_rank = (
            torch.distributed.get_rank()
            if torch.distributed.is_initialized()
            else 0
        )
        logger.info(
            "Initializing Iris (twoshot-row) symmetric heap: "
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
            f"Iris (twoshot-row) initialized successfully on rank {cur_rank}"
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
                f"Iris (twoshot-row): input buffer too small for M={M} "
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
            f"Iris (twoshot-row): allocated input buffer ({M}, {N}), "
            f"dtype={dtype}, rank={cur_rank}"
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
                f"Iris (twoshot-row): quant heap buffer too small for "
                f"M={M} (allocated {self._quant_heap_M}). Cannot allocate "
                f"during CUDA graph capture."
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
            f"Iris (twoshot-row): allocated quant heap buffer "
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
                f"Iris (twoshot-row): scale heap buffer too small for "
                f"M={M} (allocated {self._scale_heap_M}). Cannot allocate "
                f"during CUDA graph capture."
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
            f"Iris (twoshot-row): allocated scale heap buffer "
            f"({M},), dtype=float32, rank={cur_rank}"
        )

        return self._scale_heap_buf

    def _get_output_buffers(
        self,
        M: int,
        N: int,
        K_GEMM: int,
        dtype: torch.dtype,
        out_dtype: torch.dtype,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get views into pre-allocated output buffers (regular GPU memory).

        Returns (result_out, gemm_out).
        result_out stores BF16 allreduce result (only owned rows are valid).
        gemm_out stores the GEMM output (M, K_GEMM) in out_dtype.
        """
        need_alloc = (
            self._out_result is None
            or M > self._out_result.shape[0]
            or N != self._out_result.shape[1]
            or self._out_gemm is None
            or K_GEMM > self._out_gemm_K
        )

        if need_alloc:
            if torch.cuda.is_current_stream_capturing():
                existing_M = (
                    self._out_result.shape[0]
                    if self._out_result is not None
                    else 0
                )
                existing_K = self._out_gemm_K or 0
                raise RuntimeError(
                    f"Iris (twoshot-row): output buffers too small for "
                    f"M={M}, K_GEMM={K_GEMM} "
                    f"(allocated M={existing_M}, K={existing_K}). "
                    f"Cannot allocate during CUDA graph capture."
                )

            alloc_K = max(K_GEMM, self._out_gemm_K or 0)
            self._out_result = torch.empty((M, N), dtype=dtype, device=device)
            self._out_gemm = torch.empty(
                (M, alloc_K), dtype=out_dtype, device=device
            )
            self._out_gemm_K = alloc_K

            # Inlined barrier: CTA 0 -> other CTAs signal
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
                f"Iris (twoshot-row): allocated output buffers "
                f"result=({M}, {N}), gemm=({M}, {K_GEMM}), "
                f"dtype={dtype}, rank={cur_rank}"
            )

        assert self._out_result is not None
        assert self._out_gemm is not None

        return self._out_result[:M], self._out_gemm[:M, :K_GEMM]

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
        """Two-shot AllReduce + RMSNorm + per-row FP8 quant + inlined GEMM.

        FP8 broadcast variant: broadcasts FP8 and per-row scales on
        the heap. Only owned rows get BF16 result_out. GEMM reads
        directly from the heap after the barrier — no separate kernel
        launch or quant_out buffer.

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
        K_GEMM = gemm_weight.shape[1]
        device = input_tensor.device

        # Get pre-allocated buffers
        iris_input = self._get_input_buffer(M, N, input_tensor.dtype)
        quant_heap = self._get_quant_heap_buffer(M, N, quant_dtype)
        scale_heap = self._get_scale_heap_buffer(M)
        result_out, gemm_out = self._get_output_buffers(
            M, N, K_GEMM, input_tensor.dtype, out_dtype, device,
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

        # Reset sync buffer before kernel launch
        assert self._barrier_done is not None
        self._barrier_done.zero_()

        def grid(META):
            return (META["COMM_SMS"],)

        # Dummy bias pointer when no bias
        bias_ptr = bias if bias is not None else input_tensor

        # Single fused kernel: allreduce + rmsnorm + quant + gemm
        # GEMM block sizes and EVEN_GEMM_K come from autotune/heuristics
        fused_twoshot_allreduce_rmsnorm_row_quant_gemm_kernel[grid](
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
            bias is not None,
        )



        # result_out and residual_out are the same buffer -- only owned
        # rows are valid, which is all that downstream reads.
        residual_out = result_out if residual is not None else None

        return gemm_out, residual_out


_iris_twoshot_row_manager: Optional[IrisTwoshotRowManager] = None


def get_iris_twoshot_row_manager() -> IrisTwoshotRowManager:
    """Get the global Iris twoshot-row manager instance."""
    global _iris_twoshot_row_manager
    if _iris_twoshot_row_manager is None:
        _iris_twoshot_row_manager = IrisTwoshotRowManager()
    return _iris_twoshot_row_manager


def initialize_iris_twoshot_row(heap_size: Optional[int] = None) -> None:
    """Initialize Iris for two-shot fused all-reduce with per-row quant.

    Call this once at model load time before any forward passes.

    Args:
        heap_size: Size of symmetric heap in bytes (default: 8GB)
    """
    get_iris_twoshot_row_manager().initialize(heap_size)


def fused_allreduce_add_rms_row_quant_gemm_iris_twoshot(
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
    """Fused AllReduce + RMSNorm + per-row FP8 Quant + GEMM (two-shot).

    FP8 broadcast variant: broadcasts FP8 and per-row scales on the
    heap instead of BF16. All FP8 is internal -- takes BF16 in,
    produces BF16 GEMM output.

    Returns (gemm_out, residual_out).
    """
    return get_iris_twoshot_row_manager().fused_allreduce_rmsnorm_row_quant_gemm(
        input, rms_weight, rms_eps, quant_dtype,
        gemm_weight, weight_scale, out_dtype, residual, bias,
    )
