"""Prefill causal conv1d (fused split q/k/v) — Triton ``@triton.jit`` kernels.

Two interchangeable kernels: a 1D per-token kernel (serial token loop, any conv
width) and a 2D-tiled kernel (vectorized over feature/token, conv width in
{2, 3, 4}). Host-side launch wrappers live in the public namespace
``aiter.ops.triton.gated_delta_net.causal_conv1d_prefill``.
"""

import triton
import triton.language as tl

PAD_SLOT_ID = -1


@triton.jit()
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

    if USE_PAD_SLOT:
        if conv_state_batch_coord == pad_slot_id:
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
            acc = acc / (1 + tl.exp(-acc))

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


@triton.jit()
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

    if USE_PAD_SLOT:
        if conv_states_input_coord == pad_slot_id:
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
        exp_neg = tl.math.exp2(acc * (-1.4426950408889634))
        rcp = tl.inline_asm_elementwise(
            "v_rcp_f32 $0, $1",
            "=v,v",
            args=[1.0 + exp_neg],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )
        acc = acc * rcp

    token_pos = sequence_start_index + token_global
    valid_token_2d = valid_token[None, :]

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
