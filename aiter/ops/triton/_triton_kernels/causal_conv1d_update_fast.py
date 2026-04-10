# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2024, Tri Dao.
# Adapted from https://github.com/Dao-AILab/causal-conv1d/blob/main/causal_conv1d/causal_conv1d_interface.py
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#
# Kernels for causal_conv1d **update** fast paths: ``conv_state`` is updated in place.

import triton
import triton.language as tl

@triton.jit()
def _causal_conv1d_update_fast_kernel(
    # Pointers to matrices
    x_ptr,  # (batch, dim, seqlen)
    w_ptr,  # (dim, width)
    bias_ptr,
    conv_state_ptr,
    conv_state_indices_ptr,
    block_idx_last_scheduled_token,  # (batch,)
    initial_state_idx,  # (batch,)
    o_ptr,  # (batch, dim, seqlen)
    # Matrix dimensions
    batch: int,
    dim: tl.constexpr,
    seqlen: tl.constexpr,
    state_len: tl.constexpr,
    num_cache_lines: tl.constexpr,  # added to support vLLM larger cache lines
    # Strides
    stride_x_seq: tl.constexpr,
    stride_x_dim: tl.constexpr,
    stride_x_token: tl.constexpr,
    stride_w_dim: tl.constexpr,
    stride_w_width: tl.constexpr,
    stride_conv_state_seq: tl.constexpr,
    stride_conv_state_dim: tl.constexpr,
    stride_conv_state_tok: tl.constexpr,
    stride_state_indices: tl.constexpr,
    stride_o_seq: tl.constexpr,
    stride_o_dim: tl.constexpr,
    stride_o_token: tl.constexpr,
    # others
    pad_slot_id: tl.constexpr,
    # Meta-parameters
    HAS_BIAS: tl.constexpr,
    KERNEL_WIDTH: tl.constexpr,
    SILU_ACTIVATION: tl.constexpr,
    IS_APC_ENABLED: tl.constexpr,
    NP2_STATELEN: tl.constexpr,
    USE_PAD_SLOT: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # ruff: noqa: E501
    idx_seq = tl.program_id(0)
    if idx_seq >= batch:
        return

    # [BLOCK_N,] elements along the feature-dimension (channel)
    idx_feats = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)

    if IS_APC_ENABLED:
        # Get the state from the initial_state_idx
        conv_state_init = tl.load(initial_state_idx + idx_seq)
        current_last_index = tl.load(block_idx_last_scheduled_token + idx_seq)
    else:
        conv_state_init = 0
        current_last_index = 0

    # cache_idx
    conv_states_input_coord = tl.load(
        conv_state_indices_ptr + idx_seq * stride_state_indices + conv_state_init
    ).to(tl.int64)

    if USE_PAD_SLOT:  # noqa
        if conv_states_input_coord == pad_slot_id:
            # not processing as this is not the actual sequence
            return

    # IS_VARLEN is False
    query_start_index = idx_seq * seqlen
    query_end_index = query_start_index + seqlen
    x_offset = idx_seq * stride_x_seq
    o_offset = idx_seq * stride_o_seq

    if query_start_index == query_end_index:
        return

    # IS_SPEC_DECODING is False
    conv_state_token_offset = 0

    # STEP 1: READ init_state data
    # note: NP2_STATELEN = triton.next_power_of_2(KERNEL_WIDTH - 1)
    idx_cols = tl.arange(0, NP2_STATELEN)
    conv_state_ptrs_cols = (
        conv_state_ptr
        + (conv_states_input_coord * stride_conv_state_seq)
        + conv_state_token_offset * stride_conv_state_tok
        + (idx_feats * stride_conv_state_dim)[:, None]
        + (idx_cols * stride_conv_state_tok)[None, :]
    )  # [BLOCK_N, NP2_STATELEN]
    mask_cols = (
        (conv_states_input_coord < num_cache_lines)
        & (idx_feats < dim)[:, None]
        & (idx_cols < KERNEL_WIDTH - 1)[None, :]
    )
    cols = tl.load(conv_state_ptrs_cols, mask_cols, other=0.0)

    # STEP 2: assume state_len > seqlen
    idx_tokens = tl.arange(0, NP2_STATELEN)  # [BLOCK_M]

    # With speculative decoding, the conv_state updates works in a sliding
    # window manner, at each forward pass, the tokens are shift by 1, so we
    # load since idx_tokens + 1.
    conv_state_ptrs_source = (
        conv_state_ptr
        + (conv_states_input_coord * stride_conv_state_seq)
        + conv_state_token_offset * stride_conv_state_tok
        + (idx_feats * stride_conv_state_dim)[None, :]
        + ((idx_tokens + seqlen) * stride_conv_state_tok)[
            :, None
        ]
    )  # [BLOCK_M, BLOCK_N]
    mask = (
        (conv_states_input_coord < num_cache_lines)
        & ((idx_tokens + seqlen) < state_len)[:, None]
        & (idx_feats < dim)[None, :]
    )
    conv_state = tl.load(conv_state_ptrs_source, mask, other=0.0)

    VAL = state_len - seqlen
    x_base = x_ptr + x_offset + (idx_feats * stride_x_dim)  # [BLOCK_N]

    x_ptrs = (
        x_base[None, :] + ((idx_tokens - VAL) * stride_x_token)[:, None]
    )  # [BLOCK_M, BLOCK_N]

    mask_x = (
        (idx_tokens - VAL >= 0)[:, None]
        & (idx_tokens - VAL < seqlen)[:, None]
        & (idx_feats < dim)[None, :]
    )  # token-index  # token-index  # feature-index
    loaded_x = tl.load(x_ptrs, mask_x, 0.0)
    tl.debug_barrier()

    new_conv_state = tl.where(mask, conv_state, loaded_x)

    # Get the state from the initial_state_idx
    # cache_idx
    conv_states_offset = tl.load(
        conv_state_indices_ptr + idx_seq * stride_state_indices + current_last_index
    ).to(tl.int64)
    conv_state_ptrs_target = (
        conv_state_ptr
        + (conv_states_offset * stride_conv_state_seq)  # Offset from seq
        + (idx_feats * stride_conv_state_dim)
    )[None, :] + (  # [BLOCK_N,]
        idx_tokens * stride_conv_state_tok
    )[:, None]
    mask = (idx_tokens < state_len)[:, None] & (idx_feats < dim)[None, :]
    tl.store(conv_state_ptrs_target, new_conv_state, mask)

    # STEP 3: init accumulator, not necessary
    #if HAS_BIAS:
    #    bias = bias_ptr + idx_feats
    #    mask_bias = idx_feats < dim
    #    acc_preload = tl.load(bias, mask=mask_bias, other=0.0).to(
    #        tl.float32
    #    )  # [BLOCK_N]
    #else:
    #    acc_preload = tl.zeros((BLOCK_N,), dtype=tl.float32)

    # STEP 4:
    # LOAD WEIGHTS and compute
    w_cols_ptrs = w_ptr + (idx_feats * stride_w_dim)[:, None] + (idx_cols * stride_w_width)[None, :]
    mask_w_cols = (idx_feats < dim)[:, None] & (idx_cols < KERNEL_WIDTH - 1)[None, :]
    w_cols = tl.load(w_cols_ptrs, mask_w_cols, other=0.0)  # [BLOCK_N, NP2_STATELEN]

    w_last_ptrs = w_ptr + (idx_feats * stride_w_dim) + (KERNEL_WIDTH - 1) * stride_w_width
    w_last = tl.load(w_last_ptrs, idx_feats < dim, other=0.0) # [BLOCK_N]

    # For the convolution output: dot(weights, [state_cols | x])
    # cols is [BLOCK_N, NP2_STATELEN] = conv_state history
    # We need x as 1D [BLOCK_N] for the last weight column
    x_1d = tl.load(x_base, mask=(idx_feats < dim), other=0.0)  # [BLOCK_N], reload as 1D
    acc = tl.sum((w_cols * cols).to(tl.float32), axis=1) + (w_last * x_1d).to(tl.float32)

    if HAS_BIAS:
        bias = bias_ptr + idx_feats
        acc += tl.load(bias, idx_feats < dim, other=0.0).to(
            tl.float32
        )  # [BLOCK_N]

    if SILU_ACTIVATION:
        acc = acc / (1 + tl.exp(-acc))
    mask_1d = idx_feats < dim
    o_ptrs = (
        o_ptr + o_offset + (idx_feats * stride_o_dim)
    )

    tl.store(o_ptrs, acc, mask=mask_1d)
@triton.jit()
def _reshape_causal_conv1d_update_fast_kernel(
    # Pointers to matrices
    x_ptr,  # (num_tokens, dim+z_dim, seqlen) where seqlen=1
    ba_ptr,
    z_ptr, # (num_tokens, num_v_heads, head_v_dim)
    core_attn_out_ptr, # (num_tokens, num_v_heads, head_v_dim)
    b_ptr, # (num_accepted_tokens, num_v_heads)
    a_ptr, # (num_accepted_tokens, num_v_heads)
    w_ptr,  # (dim, width)
    bias_ptr,
    conv_state_ptr,
    conv_state_indices_ptr,
    block_idx_last_scheduled_token,  # (batch,)
    initial_state_idx,  # (batch,)
    o_ptr,  # (num_accepted_tokens, dim, seqlen)
    # Matrix dimensions
    batch: int,
    num_tokens: int,
    num_k_heads: tl.constexpr,
    num_v_heads: tl.constexpr,
    head_k_dim: tl.constexpr,
    head_v_dim: tl.constexpr,
    dim: tl.constexpr,
    head_qkvz_dim: tl.constexpr,
    seqlen: tl.constexpr,
    state_len: tl.constexpr,
    num_cache_lines: tl.constexpr,  # added to support vLLM larger cache lines
    # Strides
    stride_x_seq: tl.constexpr,
    stride_x_dim: tl.constexpr,
    stride_x_token: tl.constexpr,
    stride_w_dim: tl.constexpr,
    stride_w_width: tl.constexpr,
    stride_conv_state_seq: tl.constexpr,
    stride_conv_state_dim: tl.constexpr,
    stride_conv_state_tok: tl.constexpr,
    stride_state_indices: tl.constexpr,
    stride_o_seq: tl.constexpr,
    stride_o_dim: tl.constexpr,
    stride_o_token: tl.constexpr,
    stride_z_seq: tl.constexpr,
    stride_ba_seq: tl.constexpr,
    stride_ba_token: tl.constexpr,
    stride_b_seq: tl.constexpr,
    # others
    pad_slot_id: tl.constexpr,
    num_program_write_z: tl.constexpr,
    BLOCK_Z: tl.constexpr,
    HV: tl.constexpr,
    # Meta-parameters
    HAS_BIAS: tl.constexpr,
    KERNEL_WIDTH: tl.constexpr,
    SILU_ACTIVATION: tl.constexpr,
    IS_APC_ENABLED: tl.constexpr,
    NP2_STATELEN: tl.constexpr,
    USE_PAD_SLOT: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # ruff: noqa: E501
    idx_seq = tl.program_id(0)
    if idx_seq >= batch:
        return

    ## write b, a
    if tl.program_id(1) == 0:
        ## HV = triton.next_power_of_2(num_v_heads)
        idx_hv = tl.arange(0, HV)
        ## map idx_hv to source idx
        idx_h = idx_hv // (num_v_heads // num_k_heads)
        idx_v = idx_hv % (num_v_heads // num_k_heads)
        b_source_offset = idx_h * (2 * num_v_heads // num_k_heads) + idx_v
        a_source_offset = idx_h * (2 * num_v_heads // num_k_heads) + num_v_heads // num_k_heads + idx_v

        b_source_ptrs = ba_ptr + idx_seq * stride_ba_seq + b_source_offset * stride_ba_token
        a_source_ptrs = ba_ptr + idx_seq * stride_ba_seq + a_source_offset * stride_ba_token
        mask_ba = idx_hv < num_v_heads
        b = tl.load(b_source_ptrs, mask=mask_ba, other=0.0)
        a = tl.load(a_source_ptrs, mask=mask_ba, other=0.0)
        ## b, a should be contiguous so the last stride is 1
        b_ptrs = b_ptr + idx_seq * stride_b_seq + idx_hv
        a_ptrs = a_ptr + idx_seq * stride_b_seq + idx_hv
        tl.store(b_ptrs, b, mask_ba)
        tl.store(a_ptrs, a, mask_ba)
    ## write z
    elif tl.program_id(1) < 1 + num_program_write_z:
        idx_z = (tl.program_id(1) -  1) * BLOCK_Z + tl.arange(0, BLOCK_Z)
        ## map idx_z to source idx
        idx_z_x = idx_z // (num_v_heads // num_k_heads * head_v_dim) * head_qkvz_dim + 2 * head_k_dim + num_v_heads // num_k_heads * head_v_dim + idx_z % (num_v_heads // num_k_heads * head_v_dim)
        z_source_ptrs = x_ptr + idx_seq * stride_x_seq + idx_z_x * stride_x_dim
        mask_z = idx_z < num_v_heads * head_v_dim
        z = tl.load(z_source_ptrs, mask=mask_z, other=0.0)
        z_ptrs = z_ptr + idx_seq * stride_z_seq + idx_z
        tl.store(z_ptrs, z, mask=mask_z)

        ## zero-fill core_attn_out
        # first, zero_fill [0, batch) for core_attn_out
        core_attn_out_ptrs = core_attn_out_ptr + idx_seq * stride_z_seq + idx_z
        tl.store(core_attn_out_ptrs, 0.0, mask=mask_z)
        # second, zero_fill [batch, num_tokens) for both z and core_attn_out
        n_repeat = (num_tokens - 1) // batch
        for idx_repeat in tl.range(n_repeat):
            idx_seq_remain = batch * (1 + idx_repeat) + idx_seq
            z_ptrs = z_ptr + idx_seq_remain * stride_z_seq + idx_z
            core_attn_out_ptrs = core_attn_out_ptr + idx_seq_remain * stride_z_seq + idx_z
            mask_remain = (idx_seq_remain < num_tokens) & mask_z
            tl.store(z_ptrs, 0.0, mask=mask_remain)
            tl.store(core_attn_out_ptrs, 0.0, mask=mask_remain)
    ## do regular causal conv1d udpate
    else:
        # [BLOCK_N,] elements along the feature-dimension (channel)
        idx_feats = (tl.program_id(1) - 1 - num_program_write_z) * BLOCK_N + tl.arange(0, BLOCK_N)
        ## map idx_feats to idx_feats_x
        idx_feats_x = (
            (idx_feats < num_k_heads * head_k_dim).to(tl.int64) * (idx_feats // head_k_dim * head_qkvz_dim + idx_feats % head_k_dim)
            + ((idx_feats >= num_k_heads * head_k_dim) & (idx_feats < num_k_heads * head_k_dim * 2)).to(tl.int64) * ((idx_feats - num_k_heads * head_k_dim) // head_k_dim * head_qkvz_dim + head_k_dim + (idx_feats - num_k_heads * head_k_dim) % head_k_dim)
            + (idx_feats >= num_k_heads * head_k_dim * 2).to(tl.int64) * ((idx_feats - num_k_heads * head_k_dim * 2) // (num_v_heads // num_k_heads * head_v_dim) * head_qkvz_dim + 2 * head_k_dim + (idx_feats - num_k_heads * head_k_dim * 2) % (num_v_heads // num_k_heads * head_v_dim))
        )

        if IS_APC_ENABLED:
            # Get the state from the initial_state_idx
            conv_state_init = tl.load(initial_state_idx + idx_seq)
            current_last_index = tl.load(block_idx_last_scheduled_token + idx_seq)
        else:
            conv_state_init = 0
            current_last_index = 0

        # cache_idx
        conv_states_input_coord = tl.load(
            conv_state_indices_ptr + idx_seq * stride_state_indices + conv_state_init
        ).to(tl.int64)

        if USE_PAD_SLOT:  # noqa
            if conv_states_input_coord == pad_slot_id:
                # not processing as this is not the actual sequence
                return

        # IS_VARLEN is False
        query_start_index = idx_seq * seqlen
        query_end_index = query_start_index + seqlen
        x_offset = idx_seq * stride_x_seq
        o_offset = idx_seq * stride_o_seq

        if query_start_index == query_end_index:
            return

        # STEP 1: READ init_state data
        # note: NP2_STATELEN = triton.next_power_of_2(KERNEL_WIDTH - 1)
        idx_cols = tl.arange(0, NP2_STATELEN)
        conv_state_ptrs_cols = (
            conv_state_ptr
            + (conv_states_input_coord * stride_conv_state_seq)
            + (idx_feats * stride_conv_state_dim)[:, None]
            + (idx_cols * stride_conv_state_tok)[None, :]
        )  # [BLOCK_N, NP2_STATELEN]
        mask_cols = (
            (conv_states_input_coord < num_cache_lines)
            & (idx_feats < dim)[:, None]
            & (idx_cols < KERNEL_WIDTH - 1)[None, :]
        )
        cols = tl.load(conv_state_ptrs_cols, mask_cols, other=0.0)

        # STEP 2: assume state_len > seqlen
        idx_tokens = tl.arange(0, NP2_STATELEN)  # [BLOCK_M]

        # With speculative decoding, the conv_state updates works in a sliding
        # window manner, at each forward pass, the tokens are shift by 1, so we
        # load since idx_tokens + 1.
        conv_state_ptrs_source = (
            conv_state_ptr
            + (conv_states_input_coord * stride_conv_state_seq)
            + (idx_feats * stride_conv_state_dim)[None, :]
            + ((idx_tokens + seqlen) * stride_conv_state_tok)[
                :, None
            ]
        )  # [BLOCK_M, BLOCK_N]
        mask = (
            (conv_states_input_coord < num_cache_lines)
            & ((idx_tokens + seqlen) < state_len)[:, None]
            & (idx_feats < dim)[None, :]
        )
        conv_state = tl.load(conv_state_ptrs_source, mask, other=0.0)

        VAL = state_len - seqlen
        x_base = x_ptr + x_offset + (idx_feats_x * stride_x_dim)  # [BLOCK_N]

        x_ptrs = (
            x_base[None, :] + ((idx_tokens - VAL) * stride_x_token)[:, None]
        )  # [BLOCK_M, BLOCK_N]

        mask_x = (
            (idx_tokens - VAL >= 0)[:, None]
            & (idx_tokens - VAL < seqlen)[:, None]
            & (idx_feats < dim)[None, :]
        )  # token-index  # token-index  # feature-index
        loaded_x = tl.load(x_ptrs, mask_x, 0.0)
        tl.debug_barrier()

        new_conv_state = tl.where(mask, conv_state, loaded_x)

        # Get the state from the initial_state_idx
        # cache_idx
        conv_states_offset = tl.load(
            conv_state_indices_ptr + idx_seq * stride_state_indices + current_last_index
        ).to(tl.int64)
        conv_state_ptrs_target = (
            conv_state_ptr
            + (conv_states_offset * stride_conv_state_seq)  # Offset from seq
            + (idx_feats * stride_conv_state_dim)
        )[None, :] + (  # [BLOCK_N,]
            idx_tokens * stride_conv_state_tok
        )[:, None]
        mask = (idx_tokens < state_len)[:, None] & (idx_feats < dim)[None, :]
        tl.store(conv_state_ptrs_target, new_conv_state, mask)

        # STEP 3: init accumulator, not necessary
        #if HAS_BIAS:
        #    bias = bias_ptr + idx_feats
        #    mask_bias = idx_feats < dim
        #    acc_preload = tl.load(bias, mask=mask_bias, other=0.0).to(
        #        tl.float32
        #    )  # [BLOCK_N]
        #else:
        #    acc_preload = tl.zeros((BLOCK_N,), dtype=tl.float32)

        # STEP 4:
        # LOAD WEIGHTS and compute
        w_cols_ptrs = w_ptr + (idx_feats * stride_w_dim)[:, None] + (idx_cols * stride_w_width)[None, :]
        mask_w_cols = (idx_feats < dim)[:, None] & (idx_cols < KERNEL_WIDTH - 1)[None, :]
        w_cols = tl.load(w_cols_ptrs, mask_w_cols, other=0.0)  # [BLOCK_N, NP2_STATELEN]

        w_last_ptrs = w_ptr + (idx_feats * stride_w_dim) + (KERNEL_WIDTH - 1) * stride_w_width
        w_last = tl.load(w_last_ptrs, idx_feats < dim, other=0.0) # [BLOCK_N]

        # For the convolution output: dot(weights, [state_cols | x])
        # cols is [BLOCK_N, NP2_STATELEN] = conv_state history
        # We need x as 1D [BLOCK_N] for the last weight column
        x_1d = tl.load(x_base, mask=(idx_feats < dim), other=0.0)  # [BLOCK_N], reload as 1D
        acc = tl.sum((w_cols * cols).to(tl.float32), axis=1) + (w_last * x_1d).to(tl.float32)

        if HAS_BIAS:
            bias = bias_ptr + idx_feats
            acc += tl.load(bias, idx_feats < dim, other=0.0).to(
                tl.float32
            )  # [BLOCK_N]

        if SILU_ACTIVATION:
            acc = acc / (1 + tl.exp(-acc))
        mask_1d = idx_feats < dim
        o_ptrs = (
            o_ptr + o_offset + (idx_feats * stride_o_dim)
        )

        tl.store(o_ptrs, acc, mask=mask_1d)
