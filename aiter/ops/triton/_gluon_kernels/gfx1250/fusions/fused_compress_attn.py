# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

# Compress attention: softmax pool of a token window, RMSNorm, RoPE and MXFP4 quant

import functools

from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from aiter.ops.triton._gluon_kernels.common.utils import (
    exp_scaled,
    interleaved_rope,
    load_interleaved_cos_sin,
)
from aiter.ops.triton._gluon_kernels.gfx1250.quant.quant import _mxfp4_downcast_op
from aiter.ops.triton._triton_kernels.common.rmsnorm import _rmsnorm_op
from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr

HEAD_DIM = gl.constexpr(128)
ROPE_DIM = gl.constexpr(64)
RATIO = gl.constexpr(4)
# The e2m1 convert takes 8 values from one lane. Give each lane 8 columns.
COLS_PER_LANE = 8


@functools.cache
def get_fused_compress_attn_fp4_layouts(num_warps):
    # One row needs 16 lanes, so the 32 lanes of a wave make 2 lane groups. The lane
    # groups and the waves split the 8 window tokens.
    lanes = 32
    lanes_per_row = HEAD_DIM // COLS_PER_LANE
    lane_groups = lanes // lanes_per_row
    window_layout = gl.BlockedLayout(
        size_per_thread=[2 * RATIO // (lane_groups * num_warps), COLS_PER_LANE],
        threads_per_warp=[lane_groups, lanes_per_row],
        warps_per_cta=[num_warps, 1],
        order=[1, 0],
    )
    return {"window_layout": window_layout}


# ======================================
# Kernel
# ======================================

_fused_compress_attn_fp4_repr = make_kernel_repr(
    "fused_compress_attn_fp4_gfx1250",
    ["STATE_SIZE", "K_PER_BLOCK", "cache_modifier"],
)


@gluon.jit(repr=_fused_compress_attn_fp4_repr)
def fused_compress_attn_fp4_gfx1250(
    plan_ptr,
    state_slots_ptr,
    block_tables_ptr,
    kv_state_ptr,
    score_state_ptr,
    kv_in_ptr,
    score_in_ptr,
    ape_ptr,
    rms_weight_ptr,
    cos_ptr,
    sin_ptr,
    kv_cache_ptr,
    cache_scale_ptr,
    num_seqs,
    stride_block_tables_seq,
    stride_in_token,
    RMS_EPS: gl.constexpr,
    STATE_SIZE: gl.constexpr,
    K_PER_BLOCK: gl.constexpr,
    STATE_SLOT_STRIDE: gl.constexpr,
    cache_modifier: gl.constexpr,
    window_layout: gl.constexpr,
):
    # Arguments after the first 30 dwords cost a memory load. Keep the runtime
    # arguments in 30 dwords.

    # Compile-time constants
    WINDOW: gl.constexpr = 2 * RATIO
    # A state, input or ape row has the previous group, then the current group.
    ROW_VALUES: gl.constexpr = 2 * HEAD_DIM
    # Two e2m1 values fit in one byte. One e8m0 scale covers 32 values.
    ROW_BYTES: gl.constexpr = HEAD_DIM // 2
    SCALE_GROUP: gl.constexpr = 32
    NUM_SCALES: gl.constexpr = HEAD_DIM // SCALE_GROUP

    gl.static_assert(STATE_SIZE >= WINDOW)

    # =====================================================================
    # Read the plan row
    # =====================================================================
    plan_row = gl.program_id(axis=0)
    # Each plan row is [last_token_row, seq, last_pos, num_from_state].
    plan_row_ptr = plan_ptr + plan_row * 4
    last_token_row = gl.load(plan_row_ptr + 0)
    seq = gl.load(plan_row_ptr + 1)
    last_pos = gl.load(plan_row_ptr + 2)
    num_from_state = gl.load(plan_row_ptr + 3)
    # Row i of a decode plan is usually sequence i, so load that slot together with
    # the plan row. Clamp, because a plan can have more rows than sequences.
    state_slot = gl.load(state_slots_ptr + gl.minimum(plan_row, num_seqs - 1))

    # An empty plan row has a negative position. The slot test stops the compiler
    # from moving the slot load after the plan-row load.
    if (last_pos >= 0) & ((state_slot >= 0) | (seq != plan_row)):
        # A row of another sequence loads its own slot.
        if seq != plan_row:
            state_slot = gl.load(state_slots_ptr + seq)
        entry = last_pos // RATIO
        block_table_row = block_tables_ptr + seq * stride_block_tables_seq
        block = gl.load(block_table_row + entry // K_PER_BLOCK)
        cache_row = block * K_PER_BLOCK + entry % K_PER_BLOCK

        # =================================================================
        # 1. Load the window, the RoPE values and the RMSNorm weight
        # =================================================================
        window_token = gl.arange(0, WINDOW, layout=gl.SliceLayout(1, window_layout))
        window_token = window_token[:, None]
        col = gl.arange(0, HEAD_DIM, layout=gl.SliceLayout(0, window_layout))
        token_pos = last_pos - (WINDOW - 1) + window_token
        is_real = token_pos >= 0
        from_state = is_real & (window_token < num_from_state)
        from_input = is_real & (window_token >= num_from_state)
        # The previous group uses the first half of a row, the current group the
        # second half.
        col_offset = gl.where(window_token >= RATIO, HEAD_DIM, 0)

        # The older tokens come from the state ring. Slots can be far apart, so use a
        # 64-bit slot offset.
        slot = state_slot.to(gl.int64)
        kv_ring = kv_state_ptr + slot * STATE_SLOT_STRIDE
        score_ring = score_state_ptr + slot * STATE_SLOT_STRIDE
        state_start = (token_pos % STATE_SIZE) * ROW_VALUES
        # The compiler does not know the alignment. The hint gives 16-byte loads.
        state_start = gl.multiple_of(state_start + col_offset, [8, 8])
        # A masked gl.load adds a branch for each load. A buffer load sets masked lanes
        # out of range, so it needs no branch.
        state_values = gl.amd.gfx1250.buffer_load(
            kv_ring, state_start + col[None, :], from_state, 0.0
        )
        state_scores = gl.amd.gfx1250.buffer_load(
            score_ring, state_start + col[None, :], from_state, float("-inf")
        )

        # The newer tokens come from the input rows.
        input_start = (last_token_row - (WINDOW - 1) + window_token) * stride_in_token
        input_start = gl.multiple_of(input_start + col_offset, [8, 8])
        input_values = gl.amd.gfx1250.buffer_load(
            kv_in_ptr, input_start + col[None, :], from_input, 0.0
        )
        input_scores = gl.amd.gfx1250.buffer_load(
            score_in_ptr, input_start + col[None, :], from_input, float("-inf")
        )
        # The state ring stores scores with ape added. Add ape to the input scores only.
        ape_start = (window_token % RATIO) * ROW_VALUES
        ape_start = gl.multiple_of(ape_start + col_offset, [8, 8])
        ape = gl.amd.gfx1250.buffer_load(
            ape_ptr, ape_start + col[None, :], from_input, 0.0
        )

        # The cache row uses the RoPE angle of the first position in its group.
        cos, sin = load_interleaved_cos_sin(
            cos_ptr, sin_ptr, entry * RATIO, HEAD_DIM, ROPE_DIM, window_layout
        )
        weight = gl.load(rms_weight_ptr + col)

        # --- 2. Softmax pool ---
        # Each token comes from at most one source. A masked source gives 0 and -inf,
        # so add the values and take the max of the scores.
        values = state_values + input_values
        scores = gl.maximum(state_scores, input_scores + ape)
        max_score = gl.max(scores, axis=0)[None, :]
        # gl.exp adds a denormal fix-up to each value. exp_scaled(1.0, x) is exp(x) with
        # one exp2.
        weights = exp_scaled(1.0, scores - max_score)
        # The IEEE divide is slow. fast_dividef uses v_rcp_f32 (1 ulp).
        pooled = gl.extra.libdevice.fast_dividef(
            gl.sum(weights * values, axis=0), gl.sum(weights, axis=0)
        )

        # --- 3. RMSNorm ---
        normed = _rmsnorm_op(pooled, weight, HEAD_DIM, RMS_EPS)[None, :]

        # --- 4. RoPE ---
        # RoPE is on the last ROPE_DIM columns.
        is_rope = col[None, :] >= HEAD_DIM - ROPE_DIM
        rotated = gl.where(is_rope, interleaved_rope(normed, cos, sin), normed)

        # --- 5. MXFP4 ---
        # SCALING_MODE=1 rounds each scale up, as the torch reference does.
        packed, scales = _mxfp4_downcast_op(
            rotated,
            BLOCK_SIZE_N=HEAD_DIM,
            BLOCK_SIZE_M=1,
            MXFP4_QUANT_BLOCK_SIZE=SCALE_GROUP,
            SCALING_MODE=1,
        )

        # =================================================================
        # 6. Write the cache row
        # =================================================================
        # A cache can have more than 2^31 bytes. Use a 64-bit row offset.
        row = cache_row.to(gl.int64)
        byte = gl.arange(0, ROW_BYTES, layout=gl.SliceLayout(0, packed.type.layout))
        gl.store(
            kv_cache_ptr + row * ROW_BYTES + byte[None, :],
            packed,
            cache_modifier=cache_modifier,
        )
        group = gl.arange(0, NUM_SCALES, layout=gl.SliceLayout(0, scales.type.layout))
        gl.store(
            cache_scale_ptr + row * NUM_SCALES + group[None, :],
            scales,
            cache_modifier=cache_modifier,
        )
