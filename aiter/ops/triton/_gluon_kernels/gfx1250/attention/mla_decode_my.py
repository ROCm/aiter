# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""
Scope: SHUFFLED_KV_CACHE only, NUM_BLOCKS_GATHER_PER_TILE == 1 (so
TILE_SIZE == page size), num_stages == 2, and QUERY_DTYPE / KV_CACHE_DTYPE in
{"bf16", "fp8"}. The nvfp4 (A8W4 / A4W4) paths of the original are NOT
reproduced -- see the assert below.
"""

import torch
import triton.experimental.gluon.language as gl
import triton.language as tl
from triton.experimental import gluon

from aiter.ops.triton.utils.types import e4m3_dtype

float8_info = torch.finfo(e4m3_dtype)


@gluon.jit
def _mla_decode_fwd_kernel_my(
    segm_output_ptr,  # [total_tokens, num_query_heads, NUM_SEGMENTS, KV_LORA_RANK]
    segm_lse_ptr,  # [total_tokens, num_query_heads, NUM_SEGMENTS]
    query_ptr,  # [total_tokens, num_query_heads, KV_LORA_RANK + QK_ROPE_HEAD_DIM]
    kv_buffer_ptr,  # shuffled: [num_blocks, num_kv_heads, TILE_SIZE * head_size]
    block_tables_ptr,  # [num_seqs, max_num_blocks_per_seq]
    seq_lens_ptr,  # [num_seqs]
    SCALE: gl.constexpr,
    q_scale_ptr,
    kv_scale_ptr,
    out_scale_ptr,
    num_query_heads: gl.constexpr,
    num_kv_heads: gl.constexpr,
    block_tables_stride: gl.int64,
    query_stride_0: gl.int64,
    query_stride_1: gl.int64,
    KV_LORA_RANK: gl.constexpr,
    QK_ROPE_HEAD_DIM: gl.constexpr,
    stride_kv_buffer_1: gl.int32,
    query_start_len_ptr,  # [num_seqs + 1]
    num_tokens_per_seq: gl.int32,
    num_blocks: gl.int32,
    TILE_SIZE: gl.constexpr,
    BLOCK_Q: gl.constexpr,
    BLOCK_M: gl.constexpr,
    NUM_SEGMENTS_PER_SEQ: gl.constexpr,
    WARP_SIZE: gl.constexpr,
    num_warps: gl.constexpr,
    num_stages: gl.constexpr,
    ALL_DECODE: gl.constexpr = False,
    K_WIDTH: gl.constexpr = 16,
    QUERY_DTYPE: gl.constexpr = "fp8",
    KV_CACHE_DTYPE: gl.constexpr = "fp8",
    NUM_HEAD_BLOCKS: gl.constexpr = 1,
    # Cache modifier for the epilogue split-K stores. "" is the default (scope=CU,
    # TH=Regular). ".cg" raises them to scope=DEV -- the correct scope here, since
    # the partials are read back by a separate reduce dispatch on other CUs and
    # never by the storing CU. ".cs" (CU + non-temporal) and ".wt" (SYS, bypasses
    # all caches on gfx1250) are the other legal values. See
    # getCtrlBitsForCacheModifierOn_GFX12 in the AMD backend for the bit mapping.
    STORE_CACHE: gl.constexpr = "",
    FP8_MIN: tl.constexpr = float8_info.min,
    FP8_MAX: tl.constexpr = float8_info.max,
):
    # ------------------------------------------------------------------
    # 0. compile-time config  (was MLAConfig.__init__)
    # ------------------------------------------------------------------
    tl.static_assert(num_stages == 2)
    tl.static_assert(WARP_SIZE == 32)
    tl.static_assert(BLOCK_Q == 1)
    tl.static_assert(QUERY_DTYPE == "bf16" or QUERY_DTYPE == "fp8")
    tl.static_assert(KV_CACHE_DTYPE == "bf16" or KV_CACHE_DTYPE == "fp8")

    NUM_QUERIES_PER_KV: gl.constexpr = num_query_heads // num_kv_heads
    RCP_LN2: gl.constexpr = 1.4426950408889634
    QK_SCALE: gl.constexpr = SCALE * RCP_LN2

    # warp tiling: QK and PV both split along M
    if num_warps == 1:
        WARP_BASES_QK: gl.constexpr = []
    elif num_warps == 2:
        WARP_BASES_QK: gl.constexpr = [(1, 0)]
    elif num_warps == 4:
        WARP_BASES_QK: gl.constexpr = [(1, 0), (2, 0)]
    else:
        WARP_BASES_QK: gl.constexpr = [(1, 0), (2, 0), (4, 0)]

    # A16W16 / A16W8 -> bf16 WMMA (k=32);  A8W8 -> fp8 WMMA (k=64)
    if QUERY_DTYPE == "fp8":
        INSTR_K: gl.constexpr = 64
    else:
        INSTR_K: gl.constexpr = 32

    QK_WMMA: gl.constexpr = gl.amd.AMDWMMALayout(
        version=3,
        transposed=True,
        warp_bases=WARP_BASES_QK,
        reg_bases=[],
        instr_shape=[16, 16, INSTR_K],
    )
    # PV shares QK's warp tiling
    PV_WMMA: gl.constexpr = QK_WMMA

    Q_DOT: gl.constexpr = gl.DotOperandLayout(0, QK_WMMA, K_WIDTH)
    K_DOT: gl.constexpr = gl.DotOperandLayout(1, QK_WMMA, K_WIDTH)
    P_DOT: gl.constexpr = gl.DotOperandLayout(0, PV_WMMA, K_WIDTH)
    V_DOT: gl.constexpr = gl.DotOperandLayout(1, PV_WMMA, K_WIDTH)

    Q_LORA_SMEM: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[KV_LORA_RANK, 16]], [BLOCK_M, KV_LORA_RANK], [1, 0]
    )
    Q_ROPE_SMEM: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[QK_ROPE_HEAD_DIM, 16]], [BLOCK_M, QK_ROPE_HEAD_DIM], [1, 0]
    )
    # shuffled KV lands in LDS already in WMMA order -> no swizzle needed
    KV_SMEM: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [1, 0])

    # Q global-load layouts: 8 elements/thread along the head dim
    if KV_LORA_RANK // 8 > WARP_SIZE:
        TPW_LORA: gl.constexpr = WARP_SIZE
    elif KV_LORA_RANK // 8 < 1:
        TPW_LORA: gl.constexpr = 1
    else:
        TPW_LORA: gl.constexpr = KV_LORA_RANK // 8
    if QK_ROPE_HEAD_DIM // 8 > WARP_SIZE:
        TPW_ROPE: gl.constexpr = WARP_SIZE
    elif QK_ROPE_HEAD_DIM // 8 < 1:
        TPW_ROPE: gl.constexpr = 1
    else:
        TPW_ROPE: gl.constexpr = QK_ROPE_HEAD_DIM // 8

    Q_LORA_LOAD: gl.constexpr = gl.BlockedLayout(
        [1, 8], [WARP_SIZE // TPW_LORA, TPW_LORA], [num_warps, 1], [1, 0]
    )
    Q_ROPE_LOAD: gl.constexpr = gl.BlockedLayout(
        [1, 8], [WARP_SIZE // TPW_ROPE, TPW_ROPE], [num_warps, 1], [1, 0]
    )

    # ------------------------------------------------------------------
    # 1. which (sequence, head block, segment) is this workgroup?
    # ------------------------------------------------------------------
    q_block_global_idx = gl.program_id(0)
    kv_head_idx = gl.program_id(1)
    segm_idx = gl.program_id(2)

    num_token_blocks_per_seq = (num_tokens_per_seq + BLOCK_Q - 1) // BLOCK_Q
    num_q_blocks_per_seq = num_token_blocks_per_seq * NUM_HEAD_BLOCKS

    if ALL_DECODE:
        seq_idx = q_block_global_idx // NUM_HEAD_BLOCKS
    else:
        seq_idx = q_block_global_idx // num_q_blocks_per_seq
    q_block_local_idx = q_block_global_idx - seq_idx * num_q_blocks_per_seq

    q_start_idx = gl.load(query_start_len_ptr + seq_idx)
    token_q_block_local_idx = q_block_local_idx // NUM_HEAD_BLOCKS
    head_block_idx = q_block_local_idx % NUM_HEAD_BLOCKS
    head_offset = head_block_idx * BLOCK_M

    seq_len = gl.load(seq_lens_ptr + seq_idx)
    tiles_per_segment = (
        seq_len + NUM_SEGMENTS_PER_SEQ * TILE_SIZE - 1
    ) // (NUM_SEGMENTS_PER_SEQ * TILE_SIZE)

    # this segment starts past the end of the sequence -> nothing to do
    if segm_idx * tiles_per_segment * TILE_SIZE >= seq_len:
        return

    # ------------------------------------------------------------------
    # 2. fold the fp8 descales into the QK scale and the output scale
    # ------------------------------------------------------------------
    qk_factor: gl.float32 = QK_SCALE
    if q_scale_ptr is not None:
        qk_factor = qk_factor * gl.load(q_scale_ptr)

    out_factor: gl.float32 = 1.0
    if kv_scale_ptr is not None:
        kv_scale = gl.load(kv_scale_ptr)
        qk_factor = qk_factor * kv_scale
        out_factor = kv_scale
    if out_scale_ptr is not None:
        out_factor = out_factor / tl.load(out_scale_ptr)

    context_len = seq_len - num_tokens_per_seq
    block_tables_row = block_tables_ptr + seq_idx * block_tables_stride

    # ------------------------------------------------------------------
    # 3. load Q (lora + rope halves) HBM -> LDS via TDM, once.
    #    BLOCK_Q == 1 => every one of the BLOCK_M rows is the same token and a
    #    distinct, consecutive head, so the block is a plain (BLOCK_M, D) tile
    #    strided by query_stride_1 -- expressible as one TDM descriptor, no
    #    register staging and no ds_store.
    # ------------------------------------------------------------------
    q_lora_shared = gl.allocate_shared_memory(
        query_ptr.type.element_ty, [BLOCK_M, KV_LORA_RANK], Q_LORA_SMEM
    )
    q_rope_shared = gl.allocate_shared_memory(
        query_ptr.type.element_ty, [BLOCK_M, QK_ROPE_HEAD_DIM], Q_ROPE_SMEM
    )

    QK_HEAD_DIM: gl.constexpr = KV_LORA_RANK + QK_ROPE_HEAD_DIM
    # BLOCK_Q == 1 => offs_m // NUM_QUERIES_PER_KV == 0, so the token index is
    # uniform across the block and folds into the descriptor base.
    q_base = query_ptr + (q_start_idx + token_q_block_local_idx).to(
        gl.int64
    ) * query_stride_0
    q_head_base = kv_head_idx * NUM_QUERIES_PER_KV + head_offset

    q_lora_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        base=q_base,
        shape=(num_query_heads, QK_HEAD_DIM),
        strides=(query_stride_1, 1),
        block_shape=(BLOCK_M, KV_LORA_RANK),
        layout=Q_LORA_SMEM,
    )
    q_rope_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        base=q_base,
        shape=(num_query_heads, QK_HEAD_DIM),
        strides=(query_stride_1, 1),
        block_shape=(BLOCK_M, QK_ROPE_HEAD_DIM),
        layout=Q_ROPE_SMEM,
    )
    gl.amd.gfx1250.tdm.async_load(q_lora_desc, [q_head_base, 0], q_lora_shared)
    gl.amd.gfx1250.tdm.async_load(
        q_rope_desc, [q_head_base, KV_LORA_RANK], q_rope_shared
    )

    # ------------------------------------------------------------------
    # 4. the same query coordinates, in the two accumulator layouts
    # ------------------------------------------------------------------
    offs_m_qk = gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, QK_WMMA))
    q_pos_qk = token_q_block_local_idx * BLOCK_Q + offs_m_qk // NUM_QUERIES_PER_KV
    q_off0_qk = q_start_idx + q_pos_qk
    q_off1_qk = (
        kv_head_idx * NUM_QUERIES_PER_KV + head_offset + offs_m_qk % NUM_QUERIES_PER_KV
    )
    q_mask0_qk = q_pos_qk < num_tokens_per_seq
    q_mask1_qk = q_off1_qk < num_query_heads

    q_off0_pv = gl.convert_layout(q_off0_qk, layout=gl.SliceLayout(1, PV_WMMA))
    q_off1_pv = gl.convert_layout(q_off1_qk, layout=gl.SliceLayout(1, PV_WMMA))
    q_mask0_pv = gl.convert_layout(q_mask0_qk, layout=gl.SliceLayout(1, PV_WMMA))
    q_mask1_pv = gl.convert_layout(q_mask1_qk, layout=gl.SliceLayout(1, PV_WMMA))

    q_pos_qk_2d = gl.convert_layout(q_pos_qk, gl.SliceLayout(1, QK_WMMA))[:, None]

    # longest prefix any query row in this block attends to
    max_seq_prefix_len = (
        context_len
        + token_q_block_local_idx * BLOCK_Q
        + (BLOCK_M - 1) // NUM_QUERIES_PER_KV
        + 1
    )
    max_seq_prefix_len = gl.minimum(max_seq_prefix_len, seq_len)

    # ------------------------------------------------------------------
    # 5. TDM descriptors over the shuffled KV cache + double-buffered LDS
    # ------------------------------------------------------------------
    kv_lora_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        base=kv_buffer_ptr,
        shape=(num_blocks * num_kv_heads, TILE_SIZE * KV_LORA_RANK),
        strides=(stride_kv_buffer_1, 1),
        block_shape=(gl.constexpr(1), TILE_SIZE * KV_LORA_RANK),
        layout=KV_SMEM,
    )
    k_rope_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        base=kv_buffer_ptr + TILE_SIZE * KV_LORA_RANK,
        shape=(num_blocks * num_kv_heads, TILE_SIZE * QK_ROPE_HEAD_DIM),
        strides=(stride_kv_buffer_1, 1),
        block_shape=(gl.constexpr(1), TILE_SIZE * QK_ROPE_HEAD_DIM),
        layout=KV_SMEM,
    )
    kv_lora_shared = gl.allocate_shared_memory(
        kv_lora_desc.dtype, [num_stages, 1, TILE_SIZE * KV_LORA_RANK], KV_SMEM
    )
    k_rope_shared = gl.allocate_shared_memory(
        k_rope_desc.dtype, [num_stages, 1, TILE_SIZE * QK_ROPE_HEAD_DIM], KV_SMEM
    )

    num_tiles = (max_seq_prefix_len + TILE_SIZE - 1) // TILE_SIZE
    tile_start = segm_idx * tiles_per_segment
    tile_end = min((segm_idx + 1) * tiles_per_segment, num_tiles)

    # ------------------------------------------------------------------
    # 6. accumulators for the online (flash) softmax
    # ------------------------------------------------------------------
    M = gl.full([BLOCK_M], float("-inf"), dtype=gl.float32, layout=gl.SliceLayout(1, QK_WMMA))
    L = gl.full([BLOCK_M], 1.0, dtype=gl.float32, layout=gl.SliceLayout(1, QK_WMMA))
    acc = gl.zeros([BLOCK_M, KV_LORA_RANK], dtype=gl.float32, layout=PV_WMMA)

    # ------------------------------------------------------------------
    # 7. prologue: kick off the page-0 loads before entering the loop
    # ------------------------------------------------------------------
    j_hbm_start: gl.int32 = segm_idx * tiles_per_segment
    max_num_tiles_this_seg: gl.int32 = tile_end - tile_start
    j_hbm: gl.int32 = 0
    buffer_id: gl.int32 = 0

    physical_block_idx = gl.load(block_tables_row + j_hbm_start + j_hbm)
    j_hbm += 1
    safe_j = gl.minimum(j_hbm, max_num_tiles_this_seg - 1)
    next_physical_block_idx = gl.load(block_tables_row + j_hbm_start + safe_j)
    j_hbm += 1

    row_offsets = (physical_block_idx * num_kv_heads + kv_head_idx).to(gl.int32)
    gl.amd.gfx1250.tdm.async_load(kv_lora_desc, [row_offsets, 0], kv_lora_shared.index(0))
    gl.amd.gfx1250.tdm.async_load(k_rope_desc, [row_offsets, 0], k_rope_shared.index(0))

    # move here to hide latency and make q k separate buffer
    gl.amd.gfx1250.tdm.async_wait(2)
    Q_lora = q_lora_shared.load(layout=Q_DOT)
    Q_rope = q_rope_shared.load(layout=Q_DOT)
    # ------------------------------------------------------------------
    # 8. main loop over KV pages: every tile but the last (no causal mask
    #    needed there), software-pipelined one page ahead
    # ------------------------------------------------------------------
    for _ in range(tile_start, tile_end - 1):
        # one drain per iteration: the previous iteration's prefetch into
        # this buffer must have landed before anything reads it.
        gl.amd.gfx1250.tdm.async_wait(0)
        physical_block_idx = next_physical_block_idx
        safe_j = gl.minimum(j_hbm, max_num_tiles_this_seg - 1)
        next_physical_block_idx = gl.load(block_tables_row + j_hbm_start + safe_j)
        j_hbm += 1

        S = gl.zeros([BLOCK_M, TILE_SIZE], dtype=tl.float32, layout=QK_WMMA)

        # --- prefetch the next page before the QK MACs issue -------------------
        next_buffer_id = 1 - buffer_id
        row_offsets = (physical_block_idx * num_kv_heads + kv_head_idx).to(gl.int32)
        gl.amd.gfx1250.tdm.async_load(
            kv_lora_desc, [row_offsets, 0], kv_lora_shared.index(next_buffer_id)
        )
        gl.amd.gfx1250.tdm.async_load(
            k_rope_desc, [row_offsets, 0], k_rope_shared.index(next_buffer_id)
        )

        # --- Q_lora @ K_lora^T -------------------------------------------------
        # un-shuffle in LDS: the (16 lanes, K_WIDTH elems) tiling written by
        # shuffle_kv_buffer is undone by this reshape/permute, for free.
        k_lora = (
            kv_lora_shared.index(buffer_id)
            .reshape((1, TILE_SIZE // 16, KV_LORA_RANK // (2 * K_WIDTH), 2, 16, K_WIDTH))
            .permute((0, 1, 4, 2, 3, 5))
            .reshape((TILE_SIZE, KV_LORA_RANK))
            .permute((1, 0))
            .load(layout=K_DOT)
        )
        S = gl.amd.gfx1250.wmma(Q_lora, k_lora.to(Q_lora.dtype), S)

        # --- Q_rope @ K_rope^T, accumulated into the same S --------------------
        k_rope = (
            k_rope_shared.index(buffer_id)
            .reshape((1, TILE_SIZE // 16, QK_ROPE_HEAD_DIM // (2 * K_WIDTH), 2, 16, K_WIDTH))
            .permute((0, 1, 4, 2, 3, 5))
            .reshape((TILE_SIZE, QK_ROPE_HEAD_DIM))
            .permute((1, 0))
            .load(layout=K_DOT)
        )
        S = gl.amd.gfx1250.wmma(Q_rope, k_rope.to(Q_rope.dtype), S)
        S = S * qk_factor

        # --- online softmax ----------------------------------------------------
        m_ij = gl.maximum(M, gl.max(S, axis=1))
        p = gl.exp2(S - m_ij[:, None])
        alpha = gl.exp2(M - m_ij)
        M = m_ij
        acc = acc * gl.convert_layout(alpha[:, None], layout=PV_WMMA)
        L = L * alpha + gl.sum(p, 1)

        # --- P @ V (V is the lora half again, un-transposed) -------------------
        v_lora = (
            kv_lora_shared.index(buffer_id)
            .reshape((1, TILE_SIZE // 16, KV_LORA_RANK // (2 * K_WIDTH), 2, 16, K_WIDTH))
            .permute((0, 1, 4, 2, 3, 5))
            .reshape((TILE_SIZE, KV_LORA_RANK))
            .load(layout=V_DOT)
        )
        if QUERY_DTYPE == "fp8":
            p = p.to(v_lora.dtype)
        elif KV_CACHE_DTYPE == "fp8":
            p = p.to(gl.bfloat16, fp_downcast_rounding="rtz")
            v_lora = v_lora.to(gl.bfloat16)
        else:
            p = p.to(gl.bfloat16, fp_downcast_rounding="rtz")
        acc = gl.amd.gfx1250.wmma(gl.convert_layout(p, P_DOT), v_lora, acc)

        buffer_id = next_buffer_id

    # ------------------------------------------------------------------
    # 9. epilogue tile: same body, plus the causal mask
    # ------------------------------------------------------------------
    S = gl.zeros([BLOCK_M, TILE_SIZE], dtype=tl.float32, layout=QK_WMMA)

    gl.amd.gfx1250.tdm.async_wait(0)
    k_lora = (
        kv_lora_shared.index(buffer_id)
        .reshape((1, TILE_SIZE // 16, KV_LORA_RANK // (2 * K_WIDTH), 2, 16, K_WIDTH))
        .permute((0, 1, 4, 2, 3, 5))
        .reshape((TILE_SIZE, KV_LORA_RANK))
        .permute((1, 0))
        .load(layout=K_DOT)
    )
    S = gl.amd.gfx1250.wmma(Q_lora, k_lora.to(Q_lora.dtype), S)

    # gl.amd.gfx1250.tdm.async_wait(0)
    k_rope = (
        k_rope_shared.index(buffer_id)
        .reshape((1, TILE_SIZE // 16, QK_ROPE_HEAD_DIM // (2 * K_WIDTH), 2, 16, K_WIDTH))
        .permute((0, 1, 4, 2, 3, 5))
        .reshape((TILE_SIZE, QK_ROPE_HEAD_DIM))
        .permute((1, 0))
        .load(layout=K_DOT)
    )
    S = gl.amd.gfx1250.wmma(Q_rope, k_rope.to(Q_rope.dtype), S)
    S = S * qk_factor

    seq_offset = (tile_end - 1) * TILE_SIZE + gl.arange(
        0, TILE_SIZE, layout=gl.SliceLayout(0, QK_WMMA)
    )
    S = gl.where(seq_offset[None, :] < context_len + q_pos_qk_2d + 1, S, float("-inf"))

    m_ij = gl.maximum(M, gl.max(S, axis=1))
    p = gl.exp2(S - m_ij[:, None])
    alpha = gl.exp2(M - m_ij)
    M = m_ij
    acc = acc * gl.convert_layout(alpha[:, None], layout=PV_WMMA)
    L = L * alpha + gl.sum(p, 1)

    # gl.amd.gfx1250.tdm.async_wait(0)
    v_lora = (
        kv_lora_shared.index(buffer_id)
        .reshape((1, TILE_SIZE // 16, KV_LORA_RANK // (2 * K_WIDTH), 2, 16, K_WIDTH))
        .permute((0, 1, 4, 2, 3, 5))
        .reshape((TILE_SIZE, KV_LORA_RANK))
        .load(layout=V_DOT)
    )
    if QUERY_DTYPE == "fp8":
        p = p.to(v_lora.dtype)
    elif KV_CACHE_DTYPE == "fp8":
        p = p.to(gl.bfloat16, fp_downcast_rounding="rtz")
        v_lora = v_lora.to(gl.bfloat16)
    else:
        p = p.to(gl.bfloat16, fp_downcast_rounding="rtz")
    acc = gl.amd.gfx1250.wmma(gl.convert_layout(p, P_DOT), v_lora, acc)

    acc = acc * out_factor
    acc = acc * gl.convert_layout(1.0 / L[:, None], layout=PV_WMMA)
    if segm_output_ptr.type.element_ty.is_fp8():
        acc = tl.clamp(acc, FP8_MIN, FP8_MAX)

    # TDM store: acc -> LDS -> HBM, instead of 64 buffer_store_b128/wave straight
    # from registers.
    OUT_SMEM: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[KV_LORA_RANK, 8]], [BLOCK_M, KV_LORA_RANK], [1, 0]
    )
    out_shared = gl.allocate_shared_memory(
        segm_output_ptr.type.element_ty, [BLOCK_M, KV_LORA_RANK], OUT_SMEM
    )
    out_shared.store(acc.to(segm_output_ptr.type.element_ty))
    # every wave's ds_store must land before the TDM engine reads LDS
    gl.barrier()

    out_base = segm_output_ptr + (q_start_idx + token_q_block_local_idx).to(
        gl.int64
    ) * (num_query_heads * NUM_SEGMENTS_PER_SEQ * KV_LORA_RANK)
    out_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        base=out_base,
        shape=(num_query_heads, NUM_SEGMENTS_PER_SEQ * KV_LORA_RANK),
        strides=(NUM_SEGMENTS_PER_SEQ * KV_LORA_RANK, 1),
        block_shape=(BLOCK_M, KV_LORA_RANK),
        layout=OUT_SMEM,
    )
    gl.amd.gfx1250.tdm.async_store(
        out_desc,
        [q_head_base, segm_idx * KV_LORA_RANK],
        out_shared,
        cache_modifier=STORE_CACHE,
    )

    if NUM_SEGMENTS_PER_SEQ > 1:
        segm_offset = (
            q_off0_qk * (num_query_heads * NUM_SEGMENTS_PER_SEQ)
            + q_off1_qk * NUM_SEGMENTS_PER_SEQ
            + segm_idx
        )
        # (max, expsum) packed into a single fp32 lse. Softmax is base-2
        # throughout (log2(e) is folded into SCALE), so the units are base-2:
        # lse = M + log2(L), and the reduce weights split i by exp2(lse_i - max).
        # L > 0 is guaranteed here -- empty segments early-returned at the top.
        lse = M + gl.log2(L)
        gl.amd.cdna4.buffer_store(
            stored_value=lse,
            ptr=segm_lse_ptr,
            offsets=segm_offset,
            mask=q_mask0_qk & q_mask1_qk,
            cache=STORE_CACHE,
        )

    gl.amd.gfx1250.tdm.async_wait(0)
