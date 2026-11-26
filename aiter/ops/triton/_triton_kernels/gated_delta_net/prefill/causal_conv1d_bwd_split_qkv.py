# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Backward of the varlen causal conv1d with split q/k/v output.

Forward (per sequence, per channel ``c``, per token ``t``):
``pre[t] = bias[c] + sum_k w[c, k] * xe[t + k]`` and ``out[t] = act(pre[t])``,
where ``xe`` is the sequence prefixed by its ``width - 1`` initial-state columns
(zeros when the sequence has no initial state).
"""

import triton
import triton.language as tl

from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr


@triton.jit
def _bwd_load_x(
    x_ptr,
    state_ptr,
    seq_start,
    pos,
    feats,
    feat_mask,
    seqlen,
    stride_x_dim,
    stride_x_token,
    state_base,
    stride_s_dim,
    stride_s_tok,
    n_state,
    KERNEL_WIDTH: tl.constexpr,
    HAS_INITIAL_STATES: tl.constexpr,
):
    """fp32 [BLOCK_M, BLOCK_N] tile of ``xe`` at sequence positions ``pos`` (may be negative).

    ``n_state`` is the number of initial-state columns to read: ``width - 1`` or 0.
    """
    in_seq = (pos >= 0) & (pos < seqlen)
    ptrs = (
        x_ptr
        + ((seq_start + pos).to(tl.int64) * stride_x_token)[:, None]
        + (feats * stride_x_dim)[None, :]
    )
    val = tl.load(ptrs, mask=in_seq[:, None] & feat_mask[None, :], other=0.0).to(
        tl.float32
    )
    if HAS_INITIAL_STATES:
        in_state = (pos < 0) & (pos >= -n_state)
        s_ptrs = (
            state_ptr
            + state_base
            + (feats * stride_s_dim)[None, :]
            + ((pos + KERNEL_WIDTH - 1) * stride_s_tok)[:, None]
        )
        val += tl.load(
            s_ptrs, mask=in_state[:, None] & feat_mask[None, :], other=0.0
        ).to(tl.float32)
    return val


@triton.jit
def _bwd_load_dout(
    dq_ptr,
    dk_ptr,
    dv_ptr,
    rows,
    row_mask,
    feats,
    k_dim,
    v_dim,
    stride_dq_tok,
    stride_dq_dim,
    stride_dk_tok,
    stride_dk_dim,
    stride_dv_tok,
    stride_dv_dim,
):
    """fp32 [BLOCK_M, BLOCK_N] tile of the output gradient for channels ``feats``."""
    r = rows.to(tl.int64)
    f = feats[None, :]
    m = row_mask[:, None]
    is_q = f < k_dim
    is_k = (f >= k_dim) & (f < 2 * k_dim)
    is_v = (f >= 2 * k_dim) & (f < 2 * k_dim + v_dim)
    g = tl.load(
        dq_ptr + (r * stride_dq_tok)[:, None] + f * stride_dq_dim,
        mask=m & is_q,
        other=0.0,
    ).to(tl.float32)
    g += tl.load(
        dk_ptr + (r * stride_dk_tok)[:, None] + (f - k_dim) * stride_dk_dim,
        mask=m & is_k,
        other=0.0,
    ).to(tl.float32)
    g += tl.load(
        dv_ptr + (r * stride_dv_tok)[:, None] + (f - 2 * k_dim) * stride_dv_dim,
        mask=m & is_v,
        other=0.0,
    ).to(tl.float32)
    return g


_causal_conv1d_bwd_split_qkv_kernel_repr = make_kernel_repr(
    "_causal_conv1d_bwd_split_qkv_kernel",
    [
        "BLOCK_M",
        "BLOCK_N",
        "KERNEL_WIDTH",
        "HAS_BIAS",
        "SILU_ACTIVATION",
        "HAS_INITIAL_STATES",
        "USE_PAD_SLOT",
    ],
)


@triton.jit(repr=_causal_conv1d_bwd_split_qkv_kernel_repr)
def _causal_conv1d_bwd_split_qkv_kernel(
    x_ptr,  # [dim, cu_seqlen]
    w_ptr,  # [dim, width]
    bias_ptr,  # [dim]
    state_ptr,  # [lines, dim, >= width - 1], pre-forward initial states
    cache_indices_ptr,  # [n_seqs]
    has_initial_state_ptr,  # [n_seqs]
    query_start_loc_ptr,  # [n_seqs + 1]
    batch_ptr,  # [n_chunks] sequence of each program
    chunk_offset_ptr,  # [n_chunks] chunk index inside the sequence
    dq_ptr,  # [cu_seqlen, k_dim]
    dk_ptr,  # [cu_seqlen, k_dim]
    dv_ptr,  # [cu_seqlen, v_dim]
    dx_ptr,  # [dim, cu_seqlen]
    dw_partial_ptr,  # [width + 1, n_chunks, dim] fp32: per-chunk dweight taps, then dbias
    dinit_ptr,  # [n_seqs, dim, width - 1]
    dim,
    k_dim,
    v_dim,
    stride_part,
    stride_x_dim,
    stride_x_token,
    stride_w_dim,
    stride_w_width,
    stride_s_seq,
    stride_s_dim,
    stride_s_tok,
    stride_cache_indices,
    stride_dq_tok,
    stride_dq_dim,
    stride_dk_tok,
    stride_dk_dim,
    stride_dv_tok,
    stride_dv_dim,
    stride_dx_dim,
    stride_dx_token,
    stride_di_seq,
    stride_di_dim,
    stride_di_tok,
    pad_slot_id,
    HAS_BIAS: tl.constexpr,
    KERNEL_WIDTH: tl.constexpr,
    SILU_ACTIVATION: tl.constexpr,
    HAS_INITIAL_STATES: tl.constexpr,
    USE_PAD_SLOT: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_n = tl.program_id(1)
    seq = tl.load(batch_ptr + pid)
    chunk = tl.load(chunk_offset_ptr + pid)
    seq_start = tl.load(query_start_loc_ptr + seq)
    seqlen = tl.load(query_start_loc_ptr + seq + 1) - seq_start

    feats = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    feat_mask = feats < dim
    tok = chunk * BLOCK_M + tl.arange(0, BLOCK_M)

    slot = tl.load(cache_indices_ptr + seq * stride_cache_indices).to(tl.int64)
    # A padded sequence gets zero gradients: its gradient rows are masked off.
    live_len = seqlen
    if USE_PAD_SLOT:
        live_len = tl.where(slot == pad_slot_id, 0, seqlen)
    n_state = 0
    if HAS_INITIAL_STATES:
        has_state = tl.load(has_initial_state_ptr + seq).to(tl.int32)
        n_state = tl.where((has_state != 0) & (live_len == seqlen), KERNEL_WIDTH - 1, 0)
    state_base = slot * stride_s_seq

    w_base = w_ptr + feats * stride_w_dim
    if HAS_BIAS:
        bias = tl.load(bias_ptr + feats, mask=feat_mask, other=0.0).to(tl.float32)
    else:
        bias = tl.zeros((BLOCK_N,), dtype=tl.float32)

    # dpre at u = tok + s; dx[t] = sum_j w[j] * dpre[t + width - 1 - j]
    dx = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    d0 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for s in tl.static_range(KERNEL_WIDTH):
        u = tok + s
        pre = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32) + bias[None, :]
        for k in tl.static_range(KERNEL_WIDTH):
            wk = tl.load(w_base + k * stride_w_width, mask=feat_mask, other=0.0).to(
                tl.float32
            )
            xk = _bwd_load_x(
                x_ptr,
                state_ptr,
                seq_start,
                u + k - (KERNEL_WIDTH - 1),
                feats,
                feat_mask,
                seqlen,
                stride_x_dim,
                stride_x_token,
                state_base,
                stride_s_dim,
                stride_s_tok,
                n_state,
                KERNEL_WIDTH,
                HAS_INITIAL_STATES,
            )
            pre += xk * wk[None, :]
        g = _bwd_load_dout(
            dq_ptr,
            dk_ptr,
            dv_ptr,
            seq_start + u,
            u < live_len,
            feats,
            k_dim,
            v_dim,
            stride_dq_tok,
            stride_dq_dim,
            stride_dk_tok,
            stride_dk_dim,
            stride_dv_tok,
            stride_dv_dim,
        )
        if SILU_ACTIVATION:
            sig = 1.0 / (1.0 + tl.exp(-pre))
            g = g * sig * (1.0 + pre * (1.0 - sig))
        wj = tl.load(
            w_base + (KERNEL_WIDTH - 1 - s) * stride_w_width, mask=feat_mask, other=0.0
        ).to(tl.float32)
        dx += g * wj[None, :]
        if s == 0:
            d0 = g

    tok_mask = tok < seqlen
    dx_ptrs = (
        dx_ptr
        + ((seq_start + tok).to(tl.int64) * stride_dx_token)[:, None]
        + (feats * stride_dx_dim)[None, :]
    )
    tl.store(
        dx_ptrs,
        dx.to(dx_ptr.dtype.element_ty),
        mask=tok_mask[:, None] & feat_mask[None, :],
    )

    # Per-chunk partial sums; the host reduces them over chunks (deterministic).
    part_base = dw_partial_ptr + pid.to(tl.int64) * dim + feats
    for j in tl.static_range(KERNEL_WIDTH):
        xj = _bwd_load_x(
            x_ptr,
            state_ptr,
            seq_start,
            tok + j - (KERNEL_WIDTH - 1),
            feats,
            feat_mask,
            seqlen,
            stride_x_dim,
            stride_x_token,
            state_base,
            stride_s_dim,
            stride_s_tok,
            n_state,
            KERNEL_WIDTH,
            HAS_INITIAL_STATES,
        )
        tl.store(part_base + j * stride_part, tl.sum(d0 * xj, axis=0), mask=feat_mask)
    if HAS_BIAS:
        tl.store(
            part_base + KERNEL_WIDTH * stride_part, tl.sum(d0, axis=0), mask=feat_mask
        )

    # dinit[k] = sum_{j <= k} w[j] * dpre[k - j], from the first chunk only
    # A constexpr and a runtime test stay as two nested ifs
    if HAS_INITIAL_STATES:  # noqa: SIM102
        if chunk == 0:
            for k in tl.static_range(KERNEL_WIDTH - 1):
                sel = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
                for j in tl.static_range(k + 1):
                    wj = tl.load(
                        w_base + j * stride_w_width, mask=feat_mask, other=0.0
                    ).to(tl.float32)
                    sel += tl.where((tok == k - j)[:, None], wj[None, :], 0.0)
                di = tl.sum(d0 * sel, axis=0)
                tl.store(
                    dinit_ptr
                    + seq.to(tl.int64) * stride_di_seq
                    + feats * stride_di_dim
                    + k * stride_di_tok,
                    di.to(dinit_ptr.dtype.element_ty),
                    mask=feats < tl.where(n_state > 0, dim, 0),
                )
