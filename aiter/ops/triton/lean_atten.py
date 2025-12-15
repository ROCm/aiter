# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""
Lean Attention
===============
This is a Triton implementation of the Lean Attention algorithm from https://arxiv.org/abs/2405.10480
Lean Attention adopts streamK style tiling strategy, which efficiently utilize all available CUs in the system.
Lean Attention is for both decode and prefill attention of transformer based models.

It currently supports ragged batching decode and prefill attention with causal=1

TO be added features:
- Add GQA support
- Misc
    - N_CTX with non-integer number of BLOCK_SIZE_N (pad zeros or add mask)
    -
"""

import torch
from typing import Optional
from bisect import bisect_right
import math
import triton
from aiter.ops.triton._triton_kernels.lean_atten import la_persistent, _get_config
from aiter.ops.triton.utils.logger import AiterTritonLogger
from aiter.ops.triton.utils.device_info import get_num_xcds
from aiter.ops.triton.utils._triton import arch_info

_LOGGER = AiterTritonLogger()

# Support tensor in [B, Seqlen, H, d] format. Taking tensors in [B*Seqlen, H, d] as inputs


def persistent_lean_attention(
    q: torch.Tensor,  # (B * seq_len_q, H, d)
    k: torch.Tensor,  # (total_seq_len_k, H, d) -> supports ragged batching
    v: torch.Tensor,  # (total_seq_len_k, H, d)
    Mp: torch.Tensor,  # temp buffer to store partial max during sm
    Lp: torch.Tensor,  # temp buffer to store partial se during sm
    Op: torch.Tensor,  # (total_programs, n_ctx_q, d)
    locks: torch.Tensor,  # (H, seq_len_q) -> used to synchronize blocks
    batch_num_block_n: torch.Tensor,  # (B) -> cumulative sum of BLOCK_N
    batch_size: int,
    sm_scale: torch.float16,
    causal: bool = True,  # causal masking
    RAGGED_BATCH: bool = False,
    config: Optional[dict] = None,
    program_count: Optional[int] = None,
):
    """
    Lean Attention using stream-K tiling for efficient CU utilization.
    Supports both prefill and decode with ragged batching and causal masking.

    Args:
        q (torch.Tensor): Query tensor with shape (batch_size * seq_len_q, num_heads, head_dim).
        k (torch.Tensor): Key tensor with shape (total_seq_len_k, num_heads, head_dim).
            For ragged batching, total_seq_len_k is sum of all K sequence lengths.
        v (torch.Tensor): Value tensor with shape (total_seq_len_k, num_heads, head_dim).
        Mp (torch.Tensor): Partial max buffer for softmax with shape (total_programs, BLOCK_M).
        Lp (torch.Tensor): Partial sum buffer for softmax with shape (total_programs, BLOCK_M).
        Op (torch.Tensor): Partial output buffer with shape (total_programs, seq_len_q, head_dim).
        locks (torch.Tensor): Synchronization locks with shape (num_heads, seq_len_q).
        batch_num_block_n (torch.Tensor): Cumulative BLOCK_N counts per batch with shape (batch_size,).
        batch_size (int): Number of sequences in batch.
        sm_scale (torch.float16): Softmax scale, typically 1/sqrt(head_dim).
        causal (bool): Apply causal masking.
        RAGGED_BATCH (bool): Enable ragged batching mode for variable-length sequences.
        config (Optional[dict]): Kernel tuning parameters (BLOCK_SIZE_M, BLOCK_SIZE_N, SM_CNT_FACTOR,
            XCD_REMAP, num_warps, waves_per_eu).
        program_count (Optional[int]): Override number of thread blocks (CTAs). Defaults to
            SM_count * SM_CNT_FACTOR.

    Returns:
        Tuple[torch.Tensor, float]: Output tensor with shape (batch_size * seq_len_q, num_heads, head_dim)
            and kernel execution time in milliseconds.
    """
    _LOGGER.info(
        f"LEAN_ATTEN: q={tuple(q.shape)}  k={tuple(k.shape)}  v={tuple(v.shape)} Mp={tuple(Mp.shape)} Lp={tuple(Lp.shape)}  Op={tuple(Op.shape)}"
    )
    if config is None:
        config = _get_config(causal=causal, batch_size=batch_size)
    sm_count = arch_info.get_num_sms()
    total_programs = (
        program_count
        if program_count is not None
        else sm_count * config["SM_CNT_FACTOR"]
    )

    return _persistent_lean_attention(
        q=q,
        k=k,
        v=v,
        Mp=Mp,
        Lp=Lp,
        Op=Op,
        locks=locks,
        batch_num_block_n=batch_num_block_n,
        total_programs=total_programs,
        BLOCK_M=config["BLOCK_SIZE_M"],
        BLOCK_N=config["BLOCK_SIZE_N"],
        XCD_REMAP=config["XCD_REMAP"],
        causal=causal,
        batch_size=batch_size,
        RAGGED_BATCH=RAGGED_BATCH,
        num_warps=config["num_warps"],
        waves_per_eu=config["waves_per_eu"],
        config=config,
    )


# Support tensor in [B, Seqlen, H, d] format. Taking tensors in [B*Seqlen, H, d] as inputs
def _persistent_lean_attention(
    q: torch.Tensor,  # (B * seq_len_q, H, d)
    k: torch.Tensor,  # (total_seq_len_k, H, d) -> supports ragged batching
    v: torch.Tensor,  # (total_seq_len_k, H, d)
    Mp: torch.Tensor,  # temp buffer to store partial max during sm
    Lp: torch.Tensor,  # temp buffer to store partial se during sm
    Op: torch.Tensor,  # (total_programs, n_ctx_q, d) -> stores partial output values
    locks: torch.Tensor,  # (H, seq_len_q) -> used to synchronize blocks
    batch_num_block_n: torch.Tensor,  # (B) -> cumulative sum of BLOCK_N for each item in the batch
    total_programs: int,  # number of thread blocks (CTAs) to launch -> eq to num SMs
    BLOCK_M: int,  # seq_q tile size
    BLOCK_N: int,  # seq_k tile size
    XCD_REMAP: bool,  # xcd_remap for spatial
    causal: bool,  # causal masking
    batch_size: int,
    RAGGED_BATCH: bool,
    num_warps: int,
    waves_per_eu: int,
    config: dict = {},
):
    """
    Internal implementation of Lean Attention with workload scheduling and buffer allocation.
    Performs validation and launches the la_persistent Triton kernel.

    Args:
        q (torch.Tensor): Query tensor with shape (batch_size * seq_len_q, num_heads, head_dim).
        k (torch.Tensor): Key tensor with shape (total_seq_len_k, num_heads, head_dim).
        v (torch.Tensor): Value tensor with shape (total_seq_len_k, num_heads, head_dim).
        Mp (torch.Tensor): Partial max buffer with shape (total_programs, BLOCK_M).
        Lp (torch.Tensor): Partial sum buffer with shape (total_programs, BLOCK_M).
        Op (torch.Tensor): Partial output buffer with shape (total_programs, n_ctx_q, head_dim).
        locks (torch.Tensor): Synchronization locks with shape (num_heads, seq_len_q).
        batch_num_block_n (torch.Tensor): Cumulative BLOCK_N counts per batch.
        total_programs (int): Number of thread blocks (CTAs) to launch.
        BLOCK_M (int): Query tile size.
        BLOCK_N (int): Key tile size.
        XCD_REMAP (bool): Enable XCD remapping for spatial distribution across compute dies.
        causal (bool): Apply causal masking.
        batch_size (int): Batch size.
        RAGGED_BATCH (bool): Enable ragged batching mode.
        num_warps (int): Number of warps per CTA.
        waves_per_eu (int): Number of waves per execution unit.
        config (dict): Additional kernel configuration parameters.

    Returns:
        Tuple[torch.Tensor, float]: Output tensor and kernel execution time (currently 0).
    """
    DEBUG = False

    NUM_XCDS = get_num_xcds()

    # shape constraints
    HEAD_DIM_Q, HEAD_DIM_K, HEAD_DIM_V = q.shape[-1], k.shape[-1], v.shape[-1]
    assert (
        HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V
    ), "Incompatible Q/K/V Hidden Dimensions"
    # Allow irregular head dims by padding compute width and masking I/O
    HEAD_DIM_PADDED = triton.next_power_of_2(HEAD_DIM_K)
    if HEAD_DIM_PADDED < 16:
        HEAD_DIM_PADDED = 16

    # MASKED_BLOCKS is used for prefill/causal for BLOCK_M > BLOCK_N
    # For gfx942, BLOCK_M=128, BLOCK_N=64 is better for performance
    MASKED_BLOCKS = BLOCK_M // BLOCK_N

    if causal:
        # Only support BLOCK_M is multiple of BLOCK_N
        # TODO: add other scenarios
        assert BLOCK_M % BLOCK_N == 0

    N_CTX_Q = q.shape[0] // batch_size
    N_CTX_K = k.shape[0]  # This is the sum of all ctx_n in a batch
    H = q.shape[1]
    H_K = k.shape[1]
    assert H % H_K == 0, "For GQA, the number of Q heads must be divisible by K/V heads"
    GQA_GROUP_SIZE = H // H_K
    HEADS_PER_XCD = H // NUM_XCDS

    sm_scale = q.shape[-1] ** (-0.5)

    (
        num_m_blocks,
        num_n_blocks,
        high_load_wgs,
        max_tiles_per_wg,
        tiles_per_head,
        total_programs,
        num_splits,
        even_split,
    ) = get_num_splits_and_buffer_sizes(
        causal,
        batch_size,
        N_CTX_Q,
        N_CTX_K,
        H,
        BLOCK_M,
        BLOCK_N,
        total_programs,
        XCD_REMAP,
        NUM_XCDS,
    )
    if DEBUG:
        print(
            f"high_load_wgs={high_load_wgs}, max_tiles_per_wg={max_tiles_per_wg}, tiles_per_head={tiles_per_head}"
        )
        print(
            f"total_programs={total_programs}, num_splits={num_splits}, even_split={even_split}"
        )
        print(f"num_m_blocks={num_m_blocks}, num_n_blocks={num_n_blocks}")
        print(
            f"HEADS_PER_XCD={HEADS_PER_XCD}, NUM_XCDS={NUM_XCDS}. XCD_REMAP={XCD_REMAP}"
        )

    CAUSAL_MODE = 0  # 0:ping-pong, 1: sequential
    max_output_tile_cnt = calculate_max_output_tiles_analytically(
        tiles_per_head=tiles_per_head,
        num_m_blocks=num_m_blocks,
        num_wgs=total_programs,
        high_load_wgs=high_load_wgs,
        max_tiles_per_wg=max_tiles_per_wg,
        causal=causal,
        MASKED_BLOCKS=MASKED_BLOCKS,
        MODE=CAUSAL_MODE,
    )
    if not causal:
        max_output_tile_cnt = math.ceil((H * batch_size) / total_programs) + 4

    if DEBUG:
        print(f"max_output_tile_cnt={max_output_tile_cnt}")

    # Clamp to buffer capacity to avoid deadlocks
    max_supported = min(
        int(Mp.shape[0]), int(Lp.shape[0]), int(Op.shape[0]), int(locks.numel())
    )
    total_programs = min(total_programs, max_supported)

    # Recompute schedule with clamped total_programs to keep splits consistent
    (
        num_m_blocks,
        num_n_blocks,
        high_load_wgs,
        max_tiles_per_wg,
        tiles_per_head,
        total_programs,
        num_splits,
        even_split,
    ) = get_num_splits_and_buffer_sizes(
        causal,
        batch_size,
        N_CTX_Q,
        N_CTX_K,
        H,
        BLOCK_M,
        BLOCK_N,
        total_programs,
        XCD_REMAP,
        NUM_XCDS,
    )

    # Runtime safety checks
    if not (Mp.dim() == 2 and Mp.shape[0] >= total_programs and Mp.shape[1] >= BLOCK_M):
        raise ValueError(
            f"Mp must have at least [total_programs, BLOCK_M] >= [{total_programs}, {BLOCK_M}], got {tuple(Mp.shape)}"
        )
    if not (Lp.dim() == 2 and Lp.shape[0] >= total_programs and Lp.shape[1] >= BLOCK_M):
        raise ValueError(
            f"Lp must have at least [total_programs, BLOCK_M] >= [{total_programs}, {BLOCK_M}], got {tuple(Lp.shape)}"
        )
    if not (
        Op.dim() == 3
        and Op.shape[0] >= total_programs
        and Op.shape[1] >= N_CTX_Q
        and Op.shape[2] >= HEAD_DIM_K
    ):
        raise ValueError(
            f"Op must have shape[0] >= total_programs, rows >= N_CTX_Q, cols >= HEAD_DIM_K; got {tuple(Op.shape)} while required first dim >= {total_programs}, rows >= {N_CTX_Q}, cols >= {HEAD_DIM_K}"
        )
    if not (locks.numel() >= total_programs):
        raise ValueError(
            f"locks must have length >= total_programs ({total_programs}), got {locks.numel()}"
        )

    grid = (total_programs, 1, 1)

    o = torch.empty_like(q, dtype=v.dtype)

    """
    kernel_timing = {
        "attn_fwd": {
            "start_event": torch.cuda.Event(enable_timing=True),
            "end_event": torch.cuda.Event(enable_timing=True),
            "ms": 0,
            "experiments": 0,
        },
    }
    kernel_timing["attn_fwd"]["start_event"].record()
    """

    la_kernel = la_persistent[grid](
        False,
        0,
        q,
        k,
        v,
        Mp,
        Lp,
        Op,
        o,
        batch_num_block_n,
        locks,
        q.stride(0),  # N_CTX_Q
        q.stride(1),  # H
        q.stride(2),  # Head_Dim
        k.stride(0),
        k.stride(1),
        k.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        o.stride(0),
        o.stride(1),
        o.stride(2),
        N_CTX_Q,
        Op.stride(0),  # total_programs
        Op.stride(1),  # n_ctx_q
        Op.stride(2),  # head_dim
        sm_scale,
        HEADS_PER_XCD=HEADS_PER_XCD,
        HEAD_DIM_ORIG=HEAD_DIM_K,
        HEAD_DIM=HEAD_DIM_K,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        MASKED_BLOCKS=MASKED_BLOCKS,
        XCD_REMAP=XCD_REMAP,
        NUM_XCDS=NUM_XCDS,
        batch_size=batch_size,
        causal=causal,
        num_m_blocks=num_m_blocks,
        num_n_blocks=num_n_blocks,
        total_programs=total_programs,
        # leanAttention params
        high_load_wgs=high_load_wgs,
        max_tiles_per_wg=max_tiles_per_wg,
        tiles_per_head=tiles_per_head,
        num_splits=num_splits,
        max_output_tile_cnt=max_output_tile_cnt,
        waves_per_eu=waves_per_eu,
        num_warps=num_warps,
        num_stages=1,
        num_ctas=1,
        gqa_group_size=GQA_GROUP_SIZE,
        use_64_indexing=(
            (k.stride(0) * N_CTX_K) >= (1 << 31)
            or (v.stride(0) * N_CTX_K) >= (1 << 31)
            or (Op.stride(0) * total_programs) >= (1 << 31)
            or (Op.stride(1) * N_CTX_Q) >= (1 << 31)
            or (o.stride(0) * N_CTX_Q) >= (1 << 31)
            or (q.stride(0) * N_CTX_Q) >= (1 << 31)
        ),
        RAGGED_BATCH=RAGGED_BATCH,
        **config,
    )

    """
    kernel_timing["attn_fwd"]["end_event"].record()
    torch.cuda.synchronize()
    for k in ["attn_fwd"]:
        ms = kernel_timing[k]["start_event"].elapsed_time(kernel_timing[k]["end_event"])
        kernel_timing[k]["ms"] += ms
    total_ms = kernel_timing["attn_fwd"]["ms"]
    """
    # print(f"la kernel {la_kernel.n_regs} registers used, {la_kernel.n_spills} spills")
    ms = 0
    return (o, ms)


def get_num_splits_and_buffer_sizes(
    causal,
    batch_size,
    max_seqlen_q,
    max_seqlen_k,
    num_heads,
    BLOCK_M,
    BLOCK_N,
    num_SMs,
    XCD_REMAP,
    NUM_XCDS,
):
    """
    Calculates workload distribution parameters for Lean Attention stream-K scheduling.

    Args:
        causal (bool): Causal masking mode.
        batch_size (int): Batch size.
        max_seqlen_q (int): Maximum query sequence length.
        max_seqlen_k (int): Maximum key sequence length.
        num_heads (int): Number of query heads.
        BLOCK_M (int): Query tile size.
        BLOCK_N (int): Key tile size.
        num_SMs (int): Number of streaming multiprocessors (CTAs to launch).
        XCD_REMAP (bool): Enable XCD remapping for spatial distribution.
        NUM_XCDS (int): Number of XCDs (compute dies).

    Returns:
        Tuple: (num_m_blocks, num_n_blocks, high_load_wgs, max_tiles_per_wg,
            tiles_per_head, total_programs, num_splits, even_split).
    """
    ##### Lean Attention: Calculate Splits and Tile Sizes #####
    ## based on onnxruntime/contrib_ops/cuda/bert/lean_attention
    num_m_blocks = (max_seqlen_q + BLOCK_M - 1) // BLOCK_M
    num_n_blocks = (max_seqlen_k + BLOCK_N - 1) // BLOCK_N

    # Schedule over Q heads; K/V heads are mapped inside the kernel via gqa_group_size

    # print(f"block_m: {BLOCK_M}, block_n: {BLOCK_N} ")
    # print(f"num_m_block: {num_m_blocks}, num_n_block: {num_n_blocks} ")
    # print(f"max_seqlen_q: {max_seqlen_q}, max_seqlen_k: {max_seqlen_k}")

    if max_seqlen_q == 1:
        causal = False

    tiles_per_head = 0
    if causal:
        # Prefill - Causal
        for i in range(0, num_m_blocks):
            tiles_per_head += (((i + 1) * BLOCK_M) + BLOCK_N - 1) // BLOCK_N
        # Does not support ragged batch for causal.
        tiles_per_head = tiles_per_head * batch_size
    else:
        # Decode or Not Causal
        tiles_per_head = num_m_blocks * num_n_blocks

    # Total tiles across all Q heads
    if XCD_REMAP:
        total_tiles = tiles_per_head * (num_heads // NUM_XCDS)
    else:
        total_tiles = tiles_per_head * num_heads

    # StreamK Lean has as many threadblocks as SMs
    # This should be a function of tile size and number of scratchpad space
    # LeanAttention assign 3 CTAs per SM (bounded by LDS size)
    if total_tiles <= num_SMs:
        lean_griddimz = min(
            (total_tiles + 1) // 2,
            (32 * total_tiles + num_n_blocks - 1) // num_n_blocks,
        )
    else:
        lean_griddimz = num_SMs  # CTA launch grid

    # Max number lean tiles per task block (CTA)
    if XCD_REMAP:
        xcd_programs = lean_griddimz // NUM_XCDS
    else:
        xcd_programs = lean_griddimz

    max_tiles_per_tb = (total_tiles + xcd_programs - 1) // xcd_programs

    # Find max number of splits
    num_splits = 0
    even_split = False
    if (total_tiles % xcd_programs) == 0:
        even_split = True
        num_splits = 1 + ((num_n_blocks + max_tiles_per_tb - 2) // (max_tiles_per_tb))
    else:
        even_split = False
        num_splits = 1 + (
            (num_n_blocks + max_tiles_per_tb - 3) // (max_tiles_per_tb - 1)
        )

    # high_load_tbs is the remainder of total_tile / num_cta
    # When XCD_REMAP, total_tiles, max_tiles_per_tb, high_load_tbs are all relative to 1 XCD
    high_load_tbs = total_tiles - ((max_tiles_per_tb - 1) * xcd_programs)

    # Needed for causal. This is (per batch n_ctx) // BLOCK_N
    num_n_blocks = num_n_blocks // batch_size

    return (
        num_m_blocks,
        num_n_blocks,
        high_load_tbs,
        max_tiles_per_tb,
        tiles_per_head,
        lean_griddimz,
        num_splits,
        even_split,
    )


def calculate_max_output_tiles_analytically(
    tiles_per_head: int,
    num_m_blocks: int,
    num_wgs: int,
    high_load_wgs: int,
    max_tiles_per_wg: int,
    causal: bool,
    MASKED_BLOCKS: int,
    MODE: int,  # 0-ping-pong, 1-sequential
):
    """
    Calculates maximum output tiles per workgroup for buffer allocation.
    Uses binary search for efficient causal workload analysis.

    Args:
        tiles_per_head (int): Total tiles per attention head.
        num_m_blocks (int): Number of M-dimension blocks.
        num_wgs (int): Number of workgroups (CTAs).
        high_load_wgs (int): Number of workgroups with extra tile.
        max_tiles_per_wg (int): Maximum tiles assigned to any workgroup.
        causal (bool): Causal masking mode.
        MASKED_BLOCKS (int): BLOCK_M / BLOCK_N ratio for causal tiling.
        MODE (int): Scheduling mode (0: ping-pong, 1: sequential).

    Returns:
        int: Maximum number of output tiles any workgroup will produce.
    """
    if num_wgs == 0:
        return 0

    m_block_boundaries = []
    if causal:
        # Pre-compute the boundaries of each M-block's workload for a single head.
        # This list will be used for binary searches.
        total_blocks = 0
        for i in range(num_m_blocks):
            if MODE == 0:  # ping-pong selection of output tile
                pair_idx = i // 2
                q_block_idx = pair_idx if (i % 2) == 0 else num_m_blocks - 1 - pair_idx
            else:
                q_block_idx = i  # sequential
            task_size = (q_block_idx + 1) * MASKED_BLOCKS
            total_blocks += task_size
            m_block_boundaries.append(total_blocks)

    max_total_output_tiles = 0
    # Loop through each workgroup to find the one that spans the most output tiles.
    for wg_id in range(num_wgs):
        total_output_tiles_for_wg = 0

        # Determine the range of global lean tile indices for this WG
        if wg_id < high_load_wgs:
            start_iter = max_tiles_per_wg * wg_id
            end_iter = start_iter + max_tiles_per_wg
        else:
            start_iter = (max_tiles_per_wg - 1) * (
                wg_id - high_load_wgs
            ) + high_load_wgs * max_tiles_per_wg
            end_iter = start_iter + (max_tiles_per_wg - 1)

        start_head = start_iter // tiles_per_head
        end_head = (end_iter - 1) // tiles_per_head

        # Loop through each head this workgroup touches
        for head_idx in range(start_head, end_head + 1):
            head_start_iter = head_idx * tiles_per_head

            # Find the intersection of the WG's range and the current head's range
            wg_start_in_head = max(start_iter, head_start_iter)
            wg_end_in_head = min(end_iter, head_start_iter + tiles_per_head)

            if not causal:
                # For non-causal, each head is one output tile.
                total_output_tiles_for_wg += 1
                continue

            # --- Causal Logic using Binary Search ---
            # Convert to indices relative to the start of the head's workload
            relative_start = wg_start_in_head - head_start_iter
            relative_end = wg_end_in_head - head_start_iter

            # Use binary search to find which M-block the start and end tiles fall into
            start_m_idx = bisect_right(m_block_boundaries, relative_start)
            end_m_idx = bisect_right(m_block_boundaries, relative_end - 1)

            # The number of output tiles is the number of boundaries crossed
            tiles_in_this_head = (end_m_idx - start_m_idx) + 1
            total_output_tiles_for_wg += tiles_in_this_head

        max_total_output_tiles = max(max_total_output_tiles, total_output_tiles_for_wg)

    return max_total_output_tiles
