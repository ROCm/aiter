"""Prefill causal conv1d (fused split q/k/v) — Triton ``@triton.jit`` kernels.

Three kernels: a 1D per-token kernel (serial token loop, any conv
width), a 2D-tiled kernel (vectorized over feature/token, conv width in
{2, 3, 4}) and a sliding-window kernel for x with unit channel stride (conv width 4).
Host-side launch wrappers live in the public namespace
``aiter.ops.triton.gated_delta_net.causal_conv1d_prefill``.
"""

import triton
import triton.language as tl

from aiter.ops.triton._triton_kernels.common.activation import (
    _silu_exp2,
    _silu_exp2_fast,
)
from aiter.ops.triton._triton_kernels.common.pid_preprocessing import remap_xcd
from aiter.ops.triton._triton_kernels.common.segment_tile import _lower_bound
from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr

PAD_SLOT_ID = -1

_causal_conv1d_fwd_split_qkv_repr = make_kernel_repr(
    "_causal_conv1d_fwd_split_qkv_kernel",
    ["KERNEL_WIDTH", "BLOCK_M", "BLOCK_N", "HAS_BIAS", "SILU_ACTIVATION"],
)
_causal_conv1d_fwd_split_qkv_tile_repr = make_kernel_repr(
    "_causal_conv1d_fwd_split_qkv_tile_kernel",
    [
        "KERNEL_WIDTH",
        "BLOCK_M",
        "BLOCK_N",
        "HAS_BIAS",
        "SILU_ACTIVATION",
        "IS_APC_ENABLED",
        "QKV_ALIGNED",
    ],
)


@triton.jit(repr=_causal_conv1d_fwd_split_qkv_repr)
def _causal_conv1d_fwd_split_qkv_kernel(
    x_ptr,
    w_ptr,
    bias_ptr,
    initial_states_ptr,
    cache_indices_ptr,
    has_initial_states_ptr,
    query_start_loc_ptr,
    q_ptr,
    k_ptr,
    v_ptr,
    key_dim: tl.constexpr,
    value_dim: tl.constexpr,
    dim: tl.constexpr,
    seqlen: tl.int32,
    num_cache_lines: tl.constexpr,
    stride_x_dim: tl.constexpr,
    stride_x_token: tl.constexpr,
    stride_w_dim: tl.constexpr,
    stride_w_width: tl.constexpr,
    stride_istate_seq: tl.constexpr,
    stride_istate_dim: tl.constexpr,
    stride_istate_token: tl.constexpr,
    stride_q_token: tl.constexpr,
    stride_q_dim: tl.constexpr,
    stride_k_token: tl.constexpr,
    stride_k_dim: tl.constexpr,
    stride_v_token: tl.constexpr,
    stride_v_dim: tl.constexpr,
    pad_slot_id: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    KERNEL_WIDTH: tl.constexpr,
    SILU_ACTIVATION: tl.constexpr,
    HAS_INITIAL_STATES: tl.constexpr,
    HAS_CACHE: tl.constexpr,
    IS_CONTINUOUS_BATCHING: tl.constexpr,
    USE_PAD_SLOT: tl.constexpr,
    NP2_STATELEN: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Fused causal conv1d + split q/k/v output for prefill."""
    conv_states_ptr = initial_states_ptr
    conv_state_indices_ptr = cache_indices_ptr
    stride_conv_state_seq = stride_istate_seq
    stride_conv_state_dim = stride_istate_dim
    stride_conv_state_tok = stride_istate_token
    state_len = KERNEL_WIDTH - 1

    idx_seq = tl.program_id(0)
    chunk_offset = tl.program_id(1)
    idx_feats = tl.program_id(2) * BLOCK_N + tl.arange(0, BLOCK_N)

    if idx_seq == pad_slot_id:
        return

    sequence_start_index = tl.load(query_start_loc_ptr + idx_seq)
    sequence_end_index = tl.load(query_start_loc_ptr + idx_seq + 1)
    seqlen = sequence_end_index - sequence_start_index

    token_offset = BLOCK_M * chunk_offset
    segment_len = min(BLOCK_M, seqlen - token_offset)

    if segment_len <= 0:
        return

    x_base = x_ptr + sequence_start_index * stride_x_token + idx_feats * stride_x_dim

    if IS_CONTINUOUS_BATCHING:
        conv_state_batch_coord = tl.load(conv_state_indices_ptr + idx_seq).to(tl.int64)
    else:
        conv_state_batch_coord = idx_seq

    if USE_PAD_SLOT and conv_state_batch_coord == pad_slot_id:
        return

    conv_states_base = (
        conv_states_ptr
        + (conv_state_batch_coord * stride_conv_state_seq)
        + (idx_feats * stride_conv_state_dim)
    )

    w_base = w_ptr + (idx_feats * stride_w_dim)

    if chunk_offset == 0:
        load_init_state = False
        if HAS_INITIAL_STATES:
            load_init_state = tl.load(has_initial_states_ptr + idx_seq).to(tl.int1)
        if load_init_state:
            prior_tokens = conv_states_base + (state_len - 1) * stride_conv_state_tok
            mask_w = idx_feats < dim
            if KERNEL_WIDTH == 2:
                col0 = tl.load(prior_tokens, mask_w, 0.0)
            if KERNEL_WIDTH == 3:
                col1 = tl.load(prior_tokens, mask_w, 0.0)
                col0 = tl.load(prior_tokens - 1 * stride_conv_state_tok, mask_w, 0.0)
            if KERNEL_WIDTH == 4:
                col2 = tl.load(prior_tokens, mask_w, 0.0)
                col1 = tl.load(prior_tokens - 1 * stride_conv_state_tok, mask_w, 0.0)
                col0 = tl.load(prior_tokens - 2 * stride_conv_state_tok, mask_w, 0.0)
        else:
            if KERNEL_WIDTH >= 2:
                col0 = tl.zeros((BLOCK_N,), dtype=x_ptr.dtype.element_ty)
            if KERNEL_WIDTH >= 3:
                col1 = tl.zeros((BLOCK_N,), dtype=x_ptr.dtype.element_ty)
            if KERNEL_WIDTH >= 4:
                col2 = tl.zeros((BLOCK_N,), dtype=x_ptr.dtype.element_ty)

        if state_len <= seqlen:
            idx_tokens_last = (seqlen - state_len) + tl.arange(0, NP2_STATELEN)
            x_ptrs = (
                x_ptr
                + ((sequence_start_index + idx_tokens_last) * stride_x_token)[:, None]
                + (idx_feats * stride_x_dim)[None, :]
            )
            mask_x = (
                (idx_tokens_last >= 0)[:, None]
                & (idx_tokens_last < seqlen)[:, None]
                & (idx_feats < dim)[None, :]
            )
            new_conv_state = tl.load(x_ptrs, mask_x, 0.0)
            idx_tokens_conv = tl.arange(0, NP2_STATELEN)
            conv_states_ptrs_target = (
                conv_states_base[None, :]
                + (idx_tokens_conv * stride_conv_state_tok)[:, None]
            )
            mask = (idx_tokens_conv < state_len)[:, None] & (idx_feats < dim)[None, :]
            tl.debug_barrier()
            tl.store(conv_states_ptrs_target, new_conv_state, mask)
        else:
            if load_init_state:
                idx_tokens_conv = tl.arange(0, NP2_STATELEN)
                conv_states_ptrs_source = (
                    conv_states_ptr
                    + (conv_state_batch_coord * stride_conv_state_seq)
                    + (idx_feats * stride_conv_state_dim)[None, :]
                    + ((idx_tokens_conv + seqlen) * stride_conv_state_tok)[:, None]
                )
                mask = (
                    (conv_state_batch_coord < num_cache_lines)
                    & ((idx_tokens_conv + seqlen) < state_len)[:, None]
                    & (idx_feats < dim)[None, :]
                )
                conv_state = tl.load(conv_states_ptrs_source, mask, other=0.0)
                VAL = state_len - seqlen
                x_ptrs = (
                    x_base[None, :]
                    + ((idx_tokens_conv - VAL) * stride_x_token)[:, None]
                )
                mask_x = (
                    (idx_tokens_conv - VAL >= 0)[:, None]
                    & (idx_tokens_conv - VAL < seqlen)[:, None]
                    & (idx_feats < dim)[None, :]
                )
                loaded_x = tl.load(x_ptrs, mask_x, 0.0)
                tl.debug_barrier()
                new_conv_state = tl.where(mask, conv_state, loaded_x)
                conv_states_ptrs_target = (
                    conv_states_base
                    + (idx_tokens_conv * stride_conv_state_tok)[:, None]
                )
                mask = (idx_tokens_conv < state_len)[:, None] & (idx_feats < dim)[
                    None, :
                ]
                tl.store(conv_states_ptrs_target, new_conv_state, mask)
            else:
                idx_tokens_conv = tl.arange(0, NP2_STATELEN)
                VAL = state_len - seqlen
                x_ptrs = (
                    x_base[None, :]
                    + ((idx_tokens_conv - VAL) * stride_x_token)[:, None]
                )
                mask_x = (
                    (idx_tokens_conv - VAL >= 0)[:, None]
                    & (idx_tokens_conv - VAL < seqlen)[:, None]
                    & (idx_feats < dim)[None, :]
                )
                new_conv_state = tl.load(x_ptrs, mask_x, 0.0)
                conv_states_ptrs_target = (
                    conv_states_base
                    + (idx_tokens_conv * stride_conv_state_tok)[:, None]
                )
                mask = (idx_tokens_conv < state_len)[:, None] & (idx_feats < dim)[
                    None, :
                ]
                tl.store(conv_states_ptrs_target, new_conv_state, mask)
    else:
        prior_tokens = x_base + (token_offset - 1) * stride_x_token
        mask_w = idx_feats < dim
        if KERNEL_WIDTH == 2:
            col0 = tl.load(prior_tokens, mask_w, 0.0, cache_modifier=".ca")
        if KERNEL_WIDTH == 3:
            col1 = tl.load(prior_tokens, mask_w, 0.0, cache_modifier=".ca")
            col0 = tl.load(
                prior_tokens - 1 * stride_x_token, mask_w, 0.0, cache_modifier=".ca"
            )
        if KERNEL_WIDTH == 4:
            col2 = tl.load(prior_tokens, mask_w, 0.0, cache_modifier=".ca")
            col1 = tl.load(
                prior_tokens - 1 * stride_x_token, mask_w, 0.0, cache_modifier=".ca"
            )
            col0 = tl.load(
                prior_tokens - 2 * stride_x_token, mask_w, 0.0, cache_modifier=".ca"
            )

    if HAS_BIAS:
        bias = bias_ptr + idx_feats
        mask_bias = idx_feats < dim
        acc_preload = tl.load(bias, mask=mask_bias, other=0.0).to(tl.float32)
    else:
        acc_preload = tl.zeros((BLOCK_N,), dtype=tl.float32)

    x_base_1d = x_base + token_offset * stride_x_token

    mask_w = idx_feats < dim
    if KERNEL_WIDTH >= 2:
        w_col0 = tl.load(w_base + 0 * stride_w_width, mask_w, other=0.0)
        w_col1 = tl.load(w_base + 1 * stride_w_width, mask_w, other=0.0)
    if KERNEL_WIDTH >= 3:
        w_col2 = tl.load(w_base + 2 * stride_w_width, mask_w, other=0.0)
    if KERNEL_WIDTH >= 4:
        w_col3 = tl.load(w_base + 3 * stride_w_width, mask_w, other=0.0)

    mask_x_1d = idx_feats < dim

    for idx_token in range(segment_len):
        acc = acc_preload
        matrix_w = w_col0
        matrix_x = col0

        for j in tl.static_range(KERNEL_WIDTH):
            if KERNEL_WIDTH == 2:
                if j == 1:
                    matrix_w = w_col1
                    x_ptrs_1d = x_base_1d + idx_token * stride_x_token
                    matrix_x = tl.load(x_ptrs_1d, mask=mask_x_1d)
            elif KERNEL_WIDTH == 3:
                if j == 1:
                    matrix_w = w_col1
                    matrix_x = col1
                elif j == 2:
                    matrix_w = w_col2
                    x_ptrs_1d = x_base_1d + idx_token * stride_x_token
                    matrix_x = tl.load(x_ptrs_1d, mask=mask_x_1d)
            elif KERNEL_WIDTH == 4:
                if j == 1:
                    matrix_w = w_col1
                    matrix_x = col1
                elif j == 2:
                    matrix_w = w_col2
                    matrix_x = col2
                elif j == 3:
                    matrix_w = w_col3
                    x_ptrs_1d = x_base_1d + idx_token * stride_x_token
                    matrix_x = tl.load(x_ptrs_1d, mask=mask_x_1d)

            acc += matrix_x * matrix_w

        if KERNEL_WIDTH == 2:
            col0 = matrix_x
        elif KERNEL_WIDTH == 3:
            col0 = col1
            col1 = matrix_x
        elif KERNEL_WIDTH == 4:
            col0 = col1
            col1 = col2
            col2 = matrix_x

        if SILU_ACTIVATION:
            acc = _silu_exp2(acc)

        global_token_idx = sequence_start_index + token_offset + idx_token
        mask_feat = (idx_token < segment_len) & (idx_feats < dim)

        is_query = idx_feats < key_dim
        q_ptrs = q_ptr + global_token_idx * stride_q_token + idx_feats * stride_q_dim
        tl.store(q_ptrs, acc, mask=mask_feat & is_query)

        is_key = (idx_feats >= key_dim) & (idx_feats < 2 * key_dim)
        k_ptrs = (
            k_ptr
            + global_token_idx * stride_k_token
            + (idx_feats - key_dim) * stride_k_dim
        )
        tl.store(k_ptrs, acc, mask=mask_feat & is_key)

        is_value = (idx_feats >= 2 * key_dim) & (idx_feats < 2 * key_dim + value_dim)
        v_ptrs = (
            v_ptr
            + global_token_idx * stride_v_token
            + (idx_feats - 2 * key_dim) * stride_v_dim
        )
        tl.store(v_ptrs, acc, mask=mask_feat & is_value)


# 2D-tiled variant: loads a [feature, token] tile and vectorizes the conv over
# both axes. grid.x decodes a flattened (sequence, chunk) schedule through
# batch_ptr / token_chunk_offset_ptr; grid.y is the feature block.


@triton.jit(repr=_causal_conv1d_fwd_split_qkv_tile_repr)
def _causal_conv1d_fwd_split_qkv_tile_kernel(
    x_ptr,
    w_ptr,
    bias_ptr,
    initial_states_ptr,
    cache_indices_ptr,
    has_initial_states_ptr,
    query_start_loc_ptr,
    batch_ptr,
    token_chunk_offset_ptr,
    block_idx_first_scheduled_token,
    block_idx_last_scheduled_token,
    initial_state_idx,
    num_computed_tokens,
    query_ptr,
    key_ptr,
    value_ptr,
    dim,
    k_dim_size,
    v_dim_size,
    cu_seqlen,
    num_cache_lines,
    stride_x_dim,
    stride_x_token: tl.constexpr,
    stride_w_dim: tl.constexpr,
    stride_w_width: tl.constexpr,
    stride_istate_seq,
    stride_istate_dim,
    stride_istate_token,
    stride_cache_indices,
    query_dim_stride,
    query_token_stride: tl.constexpr,
    key_dim_stride,
    key_token_stride: tl.constexpr,
    value_dim_stride,
    value_token_stride: tl.constexpr,
    stride_block_m: tl.constexpr,
    pad_slot_id,
    HAS_BIAS: tl.constexpr,
    KERNEL_WIDTH: tl.constexpr,
    SILU_ACTIVATION: tl.constexpr,
    IS_APC_ENABLED: tl.constexpr,
    USE_PAD_SLOT: tl.constexpr,
    NP2_STATELEN: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    QKV_ALIGNED: tl.constexpr,
):
    """2D vectorized causal conv1d with conv state update and APC support."""
    k_start_dim = k_dim_size
    v_start_dim = k_dim_size * 2
    conv_states_ptr = initial_states_ptr
    conv_state_indices_ptr = cache_indices_ptr
    stride_conv_state_seq = stride_istate_seq
    stride_conv_state_dim = stride_istate_dim
    stride_conv_state_tok = stride_istate_token
    state_len = KERNEL_WIDTH - 1

    idx_seq = tl.load(batch_ptr + tl.program_id(0)).to(tl.int64)
    chunk_offset = tl.load(token_chunk_offset_ptr + tl.program_id(0))

    if idx_seq == pad_slot_id:
        return

    idx_feats = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)

    sequence_start_index = tl.load(query_start_loc_ptr + idx_seq)
    sequence_end_index = tl.load(query_start_loc_ptr + idx_seq + 1)
    seqlen = sequence_end_index - sequence_start_index

    B_size: tl.constexpr = stride_block_m * BLOCK_M

    if IS_APC_ENABLED:
        current_first_index = tl.load(block_idx_first_scheduled_token + idx_seq)
        current_last_index = tl.load(block_idx_last_scheduled_token + idx_seq)
        sequence_completed_index = tl.load(num_computed_tokens + idx_seq)

        sequence_completed_offset_token = sequence_completed_index % B_size
        seq_completed_offset = B_size - sequence_completed_offset_token
        seq_end_offset = (seqlen - seq_completed_offset) % B_size
        last_full_block_token_index = sequence_end_index - seq_end_offset
        if seq_end_offset == 0:
            last_full_block_token_index = last_full_block_token_index - B_size

        n_block_to_fill = current_last_index - current_first_index
        conv_state_init_index = tl.load(initial_state_idx + idx_seq)
    else:
        n_block_to_fill = 0
        current_last_index = 0
        conv_state_init_index = 0
        current_first_index = 0
        last_full_block_token_index = 0

    token_offset = BLOCK_M * chunk_offset
    segment_len = tl.minimum(BLOCK_M, seqlen - token_offset)

    if segment_len <= 0:
        return

    valid_feat = idx_feats < dim

    x_feat_base = (
        x_ptr + sequence_start_index * stride_x_token + idx_feats * stride_x_dim
    )

    conv_states_input_coord = tl.load(
        conv_state_indices_ptr + idx_seq * stride_cache_indices + conv_state_init_index
    ).to(tl.int64)

    if USE_PAD_SLOT and conv_states_input_coord == pad_slot_id:
        return

    conv_states_base = (
        conv_states_ptr
        + (conv_states_input_coord * stride_conv_state_seq)
        + (idx_feats * stride_conv_state_dim)
    )

    # Pre-define initial state columns as zeros (SSA: both paths define cols)
    if KERNEL_WIDTH >= 2:
        col0 = tl.zeros((BLOCK_N,), dtype=x_ptr.dtype.element_ty)
    if KERNEL_WIDTH >= 3:
        col1 = tl.zeros((BLOCK_N,), dtype=x_ptr.dtype.element_ty)
    if KERNEL_WIDTH >= 4:
        col2 = tl.zeros((BLOCK_N,), dtype=x_ptr.dtype.element_ty)
    has_prior_state = False

    if chunk_offset == 0:
        load_init_state = tl.load(has_initial_states_ptr + idx_seq).to(tl.int1)
        if load_init_state:
            has_prior_state = True
            prior_tokens = conv_states_base + (state_len - 1) * stride_conv_state_tok
            mask_w = valid_feat
            if KERNEL_WIDTH == 4:
                col2 = tl.load(prior_tokens, mask_w, 0.0)
                col1 = tl.load(prior_tokens - 1 * stride_conv_state_tok, mask_w, 0.0)
                col0 = tl.load(prior_tokens - 2 * stride_conv_state_tok, mask_w, 0.0)
            elif KERNEL_WIDTH == 3:
                col1 = tl.load(prior_tokens, mask_w, 0.0)
                col0 = tl.load(prior_tokens - 1 * stride_conv_state_tok, mask_w, 0.0)
            elif KERNEL_WIDTH == 2:
                col0 = tl.load(prior_tokens, mask_w, 0.0)

        # Update conv_state cache
        if state_len <= seqlen:
            idx_tokens_last = (seqlen - state_len) + tl.arange(0, NP2_STATELEN)
            x_ptrs = (
                x_ptr
                + ((sequence_start_index + idx_tokens_last) * stride_x_token)[:, None]
                + (idx_feats * stride_x_dim)[None, :]
            )
            mask_x = (
                (idx_tokens_last >= 0)[:, None]
                & (idx_tokens_last < seqlen)[:, None]
                & (idx_feats < dim)[None, :]
            )
            loaded_x = tl.load(x_ptrs, mask_x, 0.0)
            idx_tokens_conv = tl.arange(0, NP2_STATELEN)

            conv_states_output_coord = tl.load(
                conv_state_indices_ptr
                + idx_seq * stride_cache_indices
                + current_last_index
            ).to(tl.int64)

            conv_states_ptrs_target = (
                conv_states_ptr
                + (conv_states_output_coord * stride_conv_state_seq)
                + (idx_feats * stride_conv_state_dim)
            )[None, :] + (idx_tokens_conv * stride_conv_state_tok)[:, None]

            mask = (idx_tokens_conv < state_len)[:, None] & (idx_feats < dim)[None, :]
            tl.debug_barrier()
            tl.store(conv_states_ptrs_target, loaded_x, mask)

        else:
            if load_init_state:
                idx_tokens_conv = tl.arange(0, NP2_STATELEN)
                conv_states_ptrs_source = (
                    conv_states_ptr
                    + (conv_states_input_coord * stride_conv_state_seq)
                    + (idx_feats * stride_conv_state_dim)[None, :]
                    + ((idx_tokens_conv + seqlen) * stride_conv_state_tok)[:, None]
                )
                mask = (
                    (conv_states_input_coord < num_cache_lines)
                    & ((idx_tokens_conv + seqlen) < state_len)[:, None]
                    & (idx_feats < dim)[None, :]
                )
                conv_state = tl.load(conv_states_ptrs_source, mask, other=0.0)

                VAL = state_len - seqlen
                x_ptrs = (
                    x_feat_base[None, :]
                    + ((idx_tokens_conv - VAL) * stride_x_token)[:, None]
                )
                mask_x = (
                    (idx_tokens_conv - VAL >= 0)[:, None]
                    & (idx_tokens_conv - VAL < seqlen)[:, None]
                    & (idx_feats < dim)[None, :]
                )
                loaded_x = tl.load(x_ptrs, mask_x, 0.0)

                tl.debug_barrier()
                new_conv_state = tl.where(mask, conv_state, loaded_x)
                conv_states_ptrs_target = (
                    conv_states_base
                    + (idx_tokens_conv * stride_conv_state_tok)[:, None]
                )
                mask = (idx_tokens_conv < state_len)[:, None] & (idx_feats < dim)[
                    None, :
                ]
                tl.store(conv_states_ptrs_target, new_conv_state, mask)
            else:
                idx_tokens_conv = tl.arange(0, NP2_STATELEN)
                VAL = state_len - seqlen
                x_ptrs = (
                    x_feat_base[None, :]
                    + ((idx_tokens_conv - VAL) * stride_x_token)[:, None]
                )
                mask_x = (
                    (idx_tokens_conv - VAL >= 0)[:, None]
                    & (idx_tokens_conv - VAL < seqlen)[:, None]
                    & (idx_feats < dim)[None, :]
                )
                new_conv_state = tl.load(x_ptrs, mask_x, 0.0)
                conv_states_ptrs_target = (
                    conv_states_base
                    + (idx_tokens_conv * stride_conv_state_tok)[:, None]
                )
                mask = (idx_tokens_conv < state_len)[:, None] & (idx_feats < dim)[
                    None, :
                ]
                tl.store(conv_states_ptrs_target, new_conv_state, mask)

    else:  # chunk_offset > 0
        # APC: store intermediate conv states at chunk boundaries
        if (chunk_offset - 1) < n_block_to_fill:
            idx_tokens_last = (
                last_full_block_token_index
                - (n_block_to_fill - chunk_offset) * B_size
                - state_len
            ) + tl.arange(0, NP2_STATELEN)
            x_ptrs = (
                x_ptr
                + (idx_tokens_last * stride_x_token)[:, None]
                + (idx_feats * stride_x_dim)[None, :]
            )
            mask_x = (idx_tokens_last >= 0)[:, None] & (idx_feats < dim)[None, :]
            loaded_x = tl.load(x_ptrs, mask_x, 0.0)
            idx_tokens_conv = tl.arange(0, NP2_STATELEN)

            conv_states_output_coord = tl.load(
                conv_state_indices_ptr
                + idx_seq * stride_cache_indices
                + current_first_index
                + (chunk_offset - 1)
            ).to(tl.int64)

            conv_states_ptrs_target = (
                conv_states_ptr
                + (conv_states_output_coord * stride_conv_state_seq)
                + (idx_feats * stride_conv_state_dim)
            )[None, :] + (idx_tokens_conv * stride_conv_state_tok)[:, None]

            mask = (idx_tokens_conv < state_len)[:, None] & (idx_feats < dim)[None, :]
            tl.debug_barrier()
            tl.store(conv_states_ptrs_target, loaded_x, mask)

    idx_tokens = tl.arange(0, BLOCK_M)
    token_global = token_offset + idx_tokens
    valid_token = idx_tokens < segment_len
    valid_2d = valid_feat[:, None] & valid_token[None, :]

    x_t = tl.load(
        x_feat_base[:, None] + token_global[None, :] * stride_x_token,
        mask=valid_2d,
        other=0.0,
    ).to(tl.float32)

    if KERNEL_WIDTH >= 2:
        x_s1 = tl.load(
            x_feat_base[:, None] + (token_global[None, :] - 1) * stride_x_token,
            mask=valid_2d & (token_global[None, :] >= 1),
            other=0.0,
        ).to(tl.float32)
    if KERNEL_WIDTH >= 3:
        x_s2 = tl.load(
            x_feat_base[:, None] + (token_global[None, :] - 2) * stride_x_token,
            mask=valid_2d & (token_global[None, :] >= 2),
            other=0.0,
        ).to(tl.float32)
    if KERNEL_WIDTH >= 4:
        x_s3 = tl.load(
            x_feat_base[:, None] + (token_global[None, :] - 3) * stride_x_token,
            mask=valid_2d & (token_global[None, :] >= 3),
            other=0.0,
        ).to(tl.float32)

    # Blend initial state into boundary tokens
    if has_prior_state:
        if KERNEL_WIDTH == 2:
            x_s1 = tl.where(
                (token_global[None, :] == 0) & valid_feat[:, None],
                col0.to(tl.float32)[:, None],
                x_s1,
            )
        if KERNEL_WIDTH == 3:
            x_s1 = tl.where(
                (token_global[None, :] == 0) & valid_feat[:, None],
                col1.to(tl.float32)[:, None],
                x_s1,
            )
            x_s2 = tl.where(
                (token_global[None, :] == 0) & valid_feat[:, None],
                col0.to(tl.float32)[:, None],
                x_s2,
            )
            x_s2 = tl.where(
                (token_global[None, :] == 1) & valid_feat[:, None],
                col1.to(tl.float32)[:, None],
                x_s2,
            )
        if KERNEL_WIDTH == 4:
            x_s1 = tl.where(
                (token_global[None, :] == 0) & valid_feat[:, None],
                col2.to(tl.float32)[:, None],
                x_s1,
            )
            x_s2 = tl.where(
                (token_global[None, :] == 0) & valid_feat[:, None],
                col1.to(tl.float32)[:, None],
                x_s2,
            )
            x_s2 = tl.where(
                (token_global[None, :] == 1) & valid_feat[:, None],
                col2.to(tl.float32)[:, None],
                x_s2,
            )
            x_s3 = tl.where(
                (token_global[None, :] == 0) & valid_feat[:, None],
                col0.to(tl.float32)[:, None],
                x_s3,
            )
            x_s3 = tl.where(
                (token_global[None, :] == 1) & valid_feat[:, None],
                col1.to(tl.float32)[:, None],
                x_s3,
            )
            x_s3 = tl.where(
                (token_global[None, :] == 2) & valid_feat[:, None],
                col2.to(tl.float32)[:, None],
                x_s3,
            )

    w_base = w_ptr + idx_feats * stride_w_dim
    mask_w = valid_feat
    if KERNEL_WIDTH == 4:
        w0 = tl.load(w_base + 0 * stride_w_width, mask=mask_w, other=0.0).to(tl.float32)
        w1 = tl.load(w_base + 1 * stride_w_width, mask=mask_w, other=0.0).to(tl.float32)
        w2 = tl.load(w_base + 2 * stride_w_width, mask=mask_w, other=0.0).to(tl.float32)
        w3 = tl.load(w_base + 3 * stride_w_width, mask=mask_w, other=0.0).to(tl.float32)
    elif KERNEL_WIDTH == 3:
        w0 = tl.load(w_base + 0 * stride_w_width, mask=mask_w, other=0.0).to(tl.float32)
        w1 = tl.load(w_base + 1 * stride_w_width, mask=mask_w, other=0.0).to(tl.float32)
        w2 = tl.load(w_base + 2 * stride_w_width, mask=mask_w, other=0.0).to(tl.float32)
    elif KERNEL_WIDTH == 2:
        w0 = tl.load(w_base + 0 * stride_w_width, mask=mask_w, other=0.0).to(tl.float32)
        w1 = tl.load(w_base + 1 * stride_w_width, mask=mask_w, other=0.0).to(tl.float32)

    if HAS_BIAS:
        acc = tl.load(bias_ptr + idx_feats, mask=valid_feat, other=0.0).to(tl.float32)
        acc = acc[:, None] + tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
    else:
        acc = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)

    if KERNEL_WIDTH == 4:
        acc += (
            w0[:, None] * x_s3
            + w1[:, None] * x_s2
            + w2[:, None] * x_s1
            + w3[:, None] * x_t
        )
    elif KERNEL_WIDTH == 3:
        acc += w0[:, None] * x_s2 + w1[:, None] * x_s1 + w2[:, None] * x_t
    elif KERNEL_WIDTH == 2:
        acc += w0[:, None] * x_s1 + w1[:, None] * x_t

    if SILU_ACTIVATION:
        acc = _silu_exp2_fast(acc)

    token_pos = sequence_start_index + token_global
    valid_token_2d = valid_token[None, :]

    if QKV_ALIGNED:
        # Block is inside one of q, k or v: one store
        block_start = tl.program_id(1) * BLOCK_N
        if block_start < k_start_dim:
            tl.store(
                query_ptr
                + token_pos[None, :] * query_token_stride
                + idx_feats[:, None] * query_dim_stride,
                acc,
                mask=valid_token_2d,
            )
        elif block_start < v_start_dim:
            tl.store(
                key_ptr
                + token_pos[None, :] * key_token_stride
                + (idx_feats - k_start_dim)[:, None] * key_dim_stride,
                acc,
                mask=valid_token_2d,
            )
        else:
            tl.store(
                value_ptr
                + token_pos[None, :] * value_token_stride
                + (idx_feats - v_start_dim)[:, None] * value_dim_stride,
                acc,
                mask=valid_token_2d,
            )
        return

    q_feat_idx = idx_feats
    is_q_block = idx_feats < k_start_dim
    q_ptrs = (
        query_ptr
        + token_pos[None, :] * query_token_stride
        + q_feat_idx[:, None] * query_dim_stride
    )
    tl.store(q_ptrs, acc, mask=valid_token_2d & is_q_block[:, None])

    k_feat_idx = idx_feats - k_start_dim
    is_k_block = (idx_feats >= k_start_dim) & (idx_feats < v_start_dim)
    k_ptrs = (
        key_ptr
        + token_pos[None, :] * key_token_stride
        + k_feat_idx[:, None] * key_dim_stride
    )
    tl.store(k_ptrs, acc, mask=valid_token_2d & is_k_block[:, None])

    v_feat_idx = idx_feats - v_start_dim
    is_v_block = (idx_feats >= v_start_dim) & (idx_feats < dim)
    v_ptrs = (
        value_ptr
        + token_pos[None, :] * value_token_stride
        + v_feat_idx[:, None] * value_dim_stride
    )
    tl.store(v_ptrs, acc, mask=valid_token_2d & is_v_block[:, None])


@triton.jit
def _conv_taps(w_ptr, c0, BD: tl.constexpr):
    # 4 taps of channels c0 .. c0 + BD - 1 as [1, BD] fp32 rows
    wt = tl.load(
        w_ptr + (c0 + tl.arange(0, BD))[:, None] * 4 + tl.arange(0, 4)[None, :]
    ).to(tl.float32)
    even, odd = tl.split(tl.reshape(wt, [BD, 2, 2]))
    w0, w2 = tl.split(even)
    w1, w3 = tl.split(odd)
    return w0[None, :], w1[None, :], w2[None, :], w3[None, :]


@triton.jit
def _window_row(
    xs,
    i: tl.constexpr,
    w0,
    w1,
    w2,
    w3,
    bias,
    HAS_BIAS: tl.constexpr,
    SILU: tl.constexpr,
):
    # Output i of each run from window rows i .. i + 3
    y = tl.fma(
        w3,
        xs[i + 3].to(tl.float32),
        tl.fma(
            w2,
            xs[i + 2].to(tl.float32),
            tl.fma(w1, xs[i + 1].to(tl.float32), w0 * xs[i].to(tl.float32)),
        ),
    )
    if HAS_BIAS:
        y += bias
    if SILU:
        y = _silu_exp2_fast(y)
    return y


@triton.jit
def _window_sequence_heads(
    x_ptr,
    w_ptr,
    b_ptr,
    st_ptr,
    ci_ptr,
    hi_ptr,
    qsl_ptr,
    out_ptr,
    s0,
    n_seq,
    stride_x,
    s_seq,
    s_dim,
    s_tok,
    stride_ci,
    pad_slot_id,
    c0,
    o_seg,
    W,
    cs,
    SB: tl.constexpr,
    BD: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    SILU: tl.constexpr,
):
    """First 3 outputs and new conv state of sequences s0 .. s0 + SB - 1"""
    seq = s0 + tl.arange(0, SB)
    in_range = seq < n_seq
    # Load sequence scalars first, then state and x rows
    a = tl.load(qsl_ptr + seq, mask=in_range, other=0)
    L = tl.load(qsl_ptr + seq + 1, mask=in_range, other=0) - a
    slot = tl.load(ci_ptr + seq * stride_ci, mask=in_range, other=pad_slot_id)
    init = tl.load(hi_ptr + seq, mask=in_range, other=0).to(tl.int1)
    live = in_range & (L > 0) & (slot != pad_slot_id)
    cl = tl.arange(0, BD)
    cv = (c0 + cl)[None, :]
    st_b = st_ptr + slot.to(tl.int64)[:, None] * s_seq + cv * s_dim
    # Read old state before any store: short sequences shift it
    mh = (init & live)[:, None]
    h0 = tl.load(st_b, mask=mh, other=0.0)
    h1 = tl.load(st_b + s_tok, mask=mh, other=0.0)
    h2 = tl.load(st_b + 2 * s_tok, mask=mh, other=0.0)
    xr = x_ptr + a.to(tl.int64)[:, None] * stride_x + cv
    m0 = live[:, None]
    m1 = (live & (L > 1))[:, None]
    m2 = (live & (L > 2))[:, None]
    m3 = (live & (L >= 3))[:, None]
    x0 = tl.load(xr, mask=m0, other=0.0)
    x1 = tl.load(xr + stride_x, mask=m1, other=0.0)
    x2 = tl.load(xr + 2 * stride_x, mask=m2, other=0.0)
    tail = xr + (L - 3)[:, None] * stride_x
    t0 = tl.load(tail, mask=m3, other=0.0)
    t1 = tl.load(tail + stride_x, mask=m3, other=0.0)
    t2 = tl.load(tail + 2 * stride_x, mask=m3, other=0.0)
    a0, a1, a2, a3 = _conv_taps(w_ptr, c0, BD)
    f0 = h0.to(tl.float32)
    f1 = h1.to(tl.float32)
    f2 = h2.to(tl.float32)
    g0 = x0.to(tl.float32)
    g1 = x1.to(tl.float32)
    g2 = x2.to(tl.float32)
    o0 = a0 * f0 + a1 * f1 + a2 * f2 + a3 * g0
    o1 = a0 * f1 + a1 * f2 + a2 * g0 + a3 * g1
    o2 = a0 * f2 + a1 * g0 + a2 * g1 + a3 * g2
    if HAS_BIAS:
        bb = tl.load(b_ptr + c0 + cl).to(tl.float32)[None, :]
        o0 += bb
        o1 += bb
        o2 += bb
    if SILU:
        o0 = _silu_exp2_fast(o0)
        o1 = _silu_exp2_fast(o1)
        o2 = _silu_exp2_fast(o2)
    odt = out_ptr.dtype.element_ty
    ob = out_ptr + o_seg + a.to(tl.int64)[:, None] * W + cs + cl[None, :]
    tl.store(ob, o0.to(odt), mask=m0)
    tl.store(ob + W, o1.to(odt), mask=m1)
    tl.store(ob + 2 * W, o2.to(odt), mask=m2)
    # New state: last 3 entries of [old state | x]
    l2 = (L == 2)[:, None]
    n0 = tl.where(m3, t0, tl.where(l2, h2, h1))
    n1 = tl.where(m3, t1, tl.where(l2, x0, h2))
    n2 = tl.where(m3, t2, tl.where(l2, x1, x0))
    tl.debug_barrier()
    sdt = st_ptr.dtype.element_ty
    tl.store(st_b, n0.to(sdt), mask=m0)
    tl.store(st_b + s_tok, n1.to(sdt), mask=m0)
    tl.store(st_b + 2 * s_tok, n2.to(sdt), mask=m0)


_causal_conv1d_fwd_split_qkv_window_repr = make_kernel_repr(
    "_causal_conv1d_fwd_split_qkv_window_kernel",
    ["KD", "VD", "R", "BC", "BD", "ORDER", "SEQ_VEC", "NUM_XCDS", "HAS_BIAS", "SILU"],
)


@triton.jit(repr=_causal_conv1d_fwd_split_qkv_window_repr)
def _causal_conv1d_fwd_split_qkv_window_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    st_ptr,
    ci_ptr,
    hi_ptr,
    qsl_ptr,
    out_ptr,
    n_seq,
    T,
    stride_x,
    s_seq,
    s_dim,
    s_tok,
    stride_ci,
    pad_slot_id,
    n_tiles,
    n_head_programs,
    KD: tl.constexpr,
    VD: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    SILU: tl.constexpr,
    R: tl.constexpr,
    BC: tl.constexpr,
    BD: tl.constexpr,
    ORDER: tl.constexpr,
    NUM_XCDS: tl.constexpr,
    SEQ_VEC: tl.constexpr,
):
    """Width-4 causal conv1d with split q/k/v for channel-contiguous x, window kept in registers"""
    BT: tl.constexpr = R * BC
    NCB: tl.constexpr = (2 * KD + VD) // BD
    pid = tl.program_id(0)
    if pid < n_head_programs:
        # Head programs: first 3 tokens and conv state of R sequences
        if pid < tl.cdiv(n_seq, R) * NCB:
            c0 = (pid % NCB) * BD
            seg = tl.minimum(c0 // KD, 2)
            _window_sequence_heads(
                x_ptr,
                w_ptr,
                b_ptr,
                st_ptr,
                ci_ptr,
                hi_ptr,
                qsl_ptr,
                out_ptr,
                (pid // NCB) * R,
                n_seq,
                stride_x,
                s_seq,
                s_dim,
                s_tok,
                stride_ci,
                pad_slot_id,
                c0,
                (seg * T).to(tl.int64) * KD,
                KD + (seg // 2) * (VD - KD),
                c0 - seg * KD,
                R,
                BD,
                HAS_BIAS,
                SILU,
            )
        return
    pid -= n_head_programs
    if ORDER == 2:
        # Contiguous tiles per XCD so halo rows stay in one L2
        pid = remap_xcd(pid, n_tiles * NCB, NUM_XCDS)
    pid_c = pid % NCB
    pid_t = pid // NCB
    tl.assume(stride_x > 0)
    tl.assume(T > 0)

    t0 = pid_t * BT
    c0 = pid_c * BD
    cl = tl.arange(0, BD)
    rows = tl.arange(0, R) * BC

    sidx = tl.arange(0, SEQ_VEC)
    starts = tl.load(qsl_ptr + sidx, mask=sidx < n_seq, other=0x3FFFFFFF)

    # Rows clamped to [0, T) so x loads are unmasked
    xb = x_ptr + c0
    roff = (t0 + rows - 3) * stride_x
    last = (T - 1) * stride_x
    xs = ()
    for i in tl.static_range(BC + 3):
        off = tl.minimum(tl.maximum(roff + i * stride_x, 0), last)
        xs += (tl.load(xb + off[:, None] + cl[None, :]),)

    w0, w1, w2, w3 = _conv_taps(w_ptr, c0, BD)
    if HAS_BIAS:
        bias = tl.load(b_ptr + c0 + cl).to(tl.float32)[None, :]
    else:
        bias = 0.0

    # Sequences starting in [t0 - 2, t0 + BT) touch this tile
    if n_seq <= SEQ_VEC:
        s_lm = tl.sum((starts < t0 - 2).to(tl.int32), 0)
        s_hi = tl.sum((starts < t0 + BT).to(tl.int32), 0)
    else:
        s_lm = _lower_bound(qsl_ptr, n_seq, t0 - 2)
        s_hi = _lower_bound(qsl_ptr, n_seq, t0 + BT)

    # q, k, v are consecutive [T, KD], [T, KD], [T, VD] blocks of one buffer
    seg = tl.minimum(c0 // KD, 2)
    W = KD + (seg // 2) * (VD - KD)
    o_base = (
        out_ptr + (seg * T).to(tl.int64) * KD + t0.to(tl.int64) * W + (c0 - seg * KD)
    )
    odt = out_ptr.dtype.element_ty

    if (s_hi == s_lm) & (t0 + BT <= T):
        for i in tl.static_range(BC):
            y = _window_row(xs, i, w0, w1, w2, w3, bias, HAS_BIAS, SILU)
            tl.store(o_base + ((rows + i) * W)[:, None] + cl[None, :], y.to(odt))
    else:
        # Bit i of run r marks a head token (done by the head programs)
        bits = tl.zeros([R], dtype=tl.int32)
        for s in range(s_lm, s_hi):
            d = tl.load(qsl_ptr + s) - (t0 + rows)
            m = tl.where(d >= 0, 7 << tl.minimum(d, 31), 7 >> tl.minimum(-d, 31))
            bits = bits | tl.where((d > -3) & (d < BC), m, 0)
        for i in tl.static_range(BC):
            y = _window_row(xs, i, w0, w1, w2, w3, bias, HAS_BIAS, SILU)
            ok = ((t0 + rows + i) < T) & (((bits >> i) & 1) == 0)
            tl.store(
                o_base + ((rows + i) * W)[:, None] + cl[None, :],
                y.to(odt),
                mask=ok[:, None],
            )
