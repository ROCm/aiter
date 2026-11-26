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
def _bwd_load_x_row(
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
    """fp32 row of ``xe`` at the sequence position ``pos`` (may be negative).

    Positions at or past ``seqlen`` read as zero. ``n_state`` is the number of
    initial-state columns to read: ``width - 1`` or 0.
    """
    in_seq = (pos >= 0) & (pos < seqlen)
    val = tl.load(
        x_ptr
        + (seq_start + pos).to(tl.int64) * stride_x_token
        + feats.to(tl.int64) * stride_x_dim,
        mask=feat_mask & in_seq,
        other=0.0,
    ).to(tl.float32)
    if HAS_INITIAL_STATES:
        in_state = (pos < 0) & (pos >= -n_state)
        val += tl.load(
            state_ptr
            + state_base
            + feats.to(tl.int64) * stride_s_dim
            + (pos + KERNEL_WIDTH - 1) * stride_s_tok,
            mask=feat_mask & in_state,
            other=0.0,
        ).to(tl.float32)
    return val


@triton.jit
def _bwd_load_dout_row(
    dq_ptr,
    dk_ptr,
    dv_ptr,
    row,
    mask,
    feats,
    blk_lo,
    k_dim,
    v_dim,
    stride_dq_tok,
    stride_dq_dim,
    stride_dk_tok,
    stride_dk_dim,
    stride_dv_tok,
    stride_dv_dim,
    BLOCK_N: tl.constexpr,
    G_UNIT_DIM: tl.constexpr,
):
    """One token row of the output gradient for channels ``feats``, in its own dtype.

    ``row`` is the global token index (int64). dq, dk and dv share one dtype.
    A channel block inside one of q, k, v reads that tensor once; a block that
    straddles a boundary (or runs past the end) reads all three under masks.
    """
    blk_hi = blk_lo + BLOCK_N
    if G_UNIT_DIM:
        cq = feats
        ck = feats - k_dim
        cv = feats - 2 * k_dim
    else:
        f64 = feats.to(tl.int64)
        cq = f64 * stride_dq_dim
        ck = (f64 - k_dim) * stride_dk_dim
        cv = (f64 - 2 * k_dim) * stride_dv_dim
    if blk_hi <= k_dim:
        g = tl.load(dq_ptr + row * stride_dq_tok + cq, mask=mask, other=0.0)
    elif (blk_lo >= k_dim) & (blk_hi <= 2 * k_dim):
        g = tl.load(dk_ptr + row * stride_dk_tok + ck, mask=mask, other=0.0)
    elif (blk_lo >= 2 * k_dim) & (blk_hi <= 2 * k_dim + v_dim):
        g = tl.load(dv_ptr + row * stride_dv_tok + cv, mask=mask, other=0.0)
    else:
        is_q = feats < k_dim
        is_k = (feats >= k_dim) & (feats < 2 * k_dim)
        is_v = (feats >= 2 * k_dim) & (feats < 2 * k_dim + v_dim)
        gq = tl.load(dq_ptr + row * stride_dq_tok + cq, mask=mask & is_q, other=0.0)
        gk = tl.load(dk_ptr + row * stride_dk_tok + ck, mask=mask & is_k, other=0.0)
        gv = tl.load(dv_ptr + row * stride_dv_tok + cv, mask=mask & is_v, other=0.0)
        g = tl.where(is_q, gq, tl.where(is_k, gk, gv))
    return g


@triton.jit
def _pick(k: tl.constexpr, v0, v1, v2, v3):
    if k == 0:
        return v0
    elif k == 1:
        return v1
    elif k == 2:
        return v2
    else:
        return v3


_causal_conv1d_bwd_split_qkv_kernel_repr = make_kernel_repr(
    "_causal_conv1d_bwd_split_qkv_kernel",
    [
        "BLOCK_M",
        "BLOCK_N",
        "KERNEL_WIDTH",
        "HAS_BIAS",
        "SILU_ACTIVATION",
        "HAS_INITIAL_STATES",
        "HAS_CACHE_INDICES",
        "USE_PAD_SLOT",
        "X_UNIT_DIM",
        "G_UNIT_DIM",
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
    dw_partial_ptr,  # [n_chunks, dim, n_part_rows] fp32: per-chunk dweight taps, then dbias
    dinit_ptr,  # [n_seqs, dim, width - 1]
    dim,
    k_dim,
    v_dim,
    n_seqs,
    cu_seqlen,
    n_part_rows,  # width, or width + 1 with a bias
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
    HAS_CACHE_INDICES: tl.constexpr,  # else the state line of a sequence is its index
    USE_PAD_SLOT: tl.constexpr,
    X_UNIT_DIM: tl.constexpr,  # x and dx have unit channel stride
    G_UNIT_DIM: tl.constexpr,  # dq, dk and dv have unit channel stride
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Each program walks the ``BLOCK_M`` tokens of its chunk, plus ``width - 1``
    halo rows, one token row of ``BLOCK_N`` channels at a time. The last
    ``width - 1`` rows of ``xe`` and of the pre-activation gradient stay in
    registers, so every element of x and of the output gradient is read once and
    dx is written once. The next row is loaded before the current one is used.
    """
    pid = tl.program_id(0)
    pid_n = tl.program_id(1)
    seq = tl.load(batch_ptr + pid)
    chunk = tl.load(chunk_offset_ptr + pid)
    seq_start = tl.load(query_start_loc_ptr + seq)
    seqlen = tl.load(query_start_loc_ptr + seq + 1) - seq_start

    blk_lo = pid_n * BLOCK_N
    feats = blk_lo + tl.arange(0, BLOCK_N)
    feat_mask = feats < dim

    if HAS_CACHE_INDICES:
        slot = tl.load(cache_indices_ptr + seq * stride_cache_indices).to(tl.int64)
    else:
        slot = seq.to(tl.int64)
    # A padded sequence gets zero gradients: x and the output gradient are read
    # under live_len (0 for it), so NaN or Inf in its rows cannot leak in.
    live_len = seqlen
    if USE_PAD_SLOT:
        live_len = tl.where(slot == pad_slot_id, 0, seqlen)
    n_state = 0
    if HAS_INITIAL_STATES:
        has_state = tl.load(has_initial_state_ptr + seq).to(tl.int32)
        n_state = tl.where((has_state != 0) & (live_len == seqlen), KERNEL_WIDTH - 1, 0)
    state_base = slot * stride_s_seq

    w_base = w_ptr + feats * stride_w_dim
    w0 = tl.load(w_base, mask=feat_mask, other=0.0).to(tl.float32)
    w1 = tl.load(w_base + stride_w_width, mask=feat_mask, other=0.0).to(tl.float32)
    w2 = w1
    w3 = w1
    if KERNEL_WIDTH >= 3:
        w2 = tl.load(w_base + 2 * stride_w_width, mask=feat_mask, other=0.0).to(
            tl.float32
        )
    if KERNEL_WIDTH >= 4:
        w3 = tl.load(w_base + 3 * stride_w_width, mask=feat_mask, other=0.0).to(
            tl.float32
        )
    if HAS_BIAS:
        bias = tl.load(bias_ptr + feats, mask=feat_mask, other=0.0).to(tl.float32)
    else:
        bias = tl.zeros((BLOCK_N,), dtype=tl.float32)

    c0 = chunk * BLOCK_M
    n_tok = tl.minimum(seqlen - c0, BLOCK_M)

    # Loop-carried values need distinct initial values: the frontend carries a
    # name only when the loop body rebinds it to a different value.
    # xd<d> = xe[u - d]; before the first row it may come from the initial state.
    xd1 = _bwd_load_x_row(
        x_ptr, state_ptr, seq_start, c0 - 1, feats, feat_mask, live_len, stride_x_dim,
        stride_x_token, state_base, stride_s_dim, stride_s_tok, n_state,
        KERNEL_WIDTH, HAS_INITIAL_STATES,
    )  # fmt: skip
    xd2 = tl.zeros((BLOCK_N,), dtype=tl.float32)
    xd3 = tl.zeros((BLOCK_N,), dtype=tl.float32)
    if KERNEL_WIDTH >= 3:
        xd2 = _bwd_load_x_row(
            x_ptr, state_ptr, seq_start, c0 - 2, feats, feat_mask, live_len,
            stride_x_dim, stride_x_token, state_base, stride_s_dim, stride_s_tok,
            n_state, KERNEL_WIDTH, HAS_INITIAL_STATES,
        )  # fmt: skip
    if KERNEL_WIDTH >= 4:
        xd3 = _bwd_load_x_row(
            x_ptr, state_ptr, seq_start, c0 - 3, feats, feat_mask, live_len,
            stride_x_dim, stride_x_token, state_base, stride_s_dim, stride_s_tok,
            n_state, KERNEL_WIDTH, HAS_INITIAL_STATES,
        )  # fmt: skip
    # gd<d> = dpre[u - d], only used once u - d is inside the chunk.
    gd1 = tl.zeros((BLOCK_N,), dtype=tl.float32)
    gd2 = tl.zeros((BLOCK_N,), dtype=tl.float32)
    gd3 = tl.zeros((BLOCK_N,), dtype=tl.float32)
    # dw<d> sums dpre[u] * xe[u - d] over the chunk rows: the partial of tap width - 1 - d.
    dw0 = tl.zeros((BLOCK_N,), dtype=tl.float32)
    dw1 = tl.zeros((BLOCK_N,), dtype=tl.float32)
    dw2 = tl.zeros((BLOCK_N,), dtype=tl.float32)
    dw3 = tl.zeros((BLOCK_N,), dtype=tl.float32)
    db = tl.zeros((BLOCK_N,), dtype=tl.float32)
    # dpre of the first width - 1 rows, for dinitial_states.
    gf0 = tl.zeros((BLOCK_N,), dtype=tl.float32)
    gf1 = tl.zeros((BLOCK_N,), dtype=tl.float32)
    gf2 = tl.zeros((BLOCK_N,), dtype=tl.float32)

    if X_UNIT_DIM:
        x_feat = feats
        dx_feat = feats
    else:
        x_feat = feats.to(tl.int64) * stride_x_dim
        dx_feat = feats.to(tl.int64) * stride_dx_dim
    row0 = (seq_start + c0).to(tl.int64)
    x_ptrs = x_ptr + row0 * stride_x_token + x_feat
    dx_ptrs = dx_ptr + (row0 - (KERNEL_WIDTH - 1)) * stride_dx_token + dx_feat

    xn = tl.load(x_ptrs, mask=feat_mask & (c0 < live_len), other=0.0)
    gn = _bwd_load_dout_row(
        dq_ptr, dk_ptr, dv_ptr, row0, feat_mask & (c0 < live_len), feats, blk_lo,
        k_dim, v_dim, stride_dq_tok, stride_dq_dim, stride_dk_tok, stride_dk_dim,
        stride_dv_tok, stride_dv_dim, BLOCK_N, G_UNIT_DIM,
    )  # fmt: skip
    for i in range(n_tok + KERNEL_WIDTH - 1):
        u = c0 + i
        xu = xn.to(tl.float32)
        gu = gn.to(tl.float32)
        x_ptrs += stride_x_token
        xn = tl.load(x_ptrs, mask=feat_mask & (u + 1 < live_len), other=0.0)
        gn = _bwd_load_dout_row(
            dq_ptr, dk_ptr, dv_ptr, row0 + i + 1, feat_mask & (u + 1 < live_len),
            feats, blk_lo, k_dim, v_dim, stride_dq_tok, stride_dq_dim, stride_dk_tok,
            stride_dk_dim, stride_dv_tok, stride_dv_dim, BLOCK_N, G_UNIT_DIM,
        )  # fmt: skip

        pre = (
            bias
            + _pick(KERNEL_WIDTH - 1, w0, w1, w2, w3) * xu
            + _pick(KERNEL_WIDTH - 2, w0, w1, w2, w3) * xd1
        )
        if KERNEL_WIDTH >= 3:
            pre += _pick(KERNEL_WIDTH - 3, w0, w1, w2, w3) * xd2
        if KERNEL_WIDTH >= 4:
            pre += w0 * xd3
        g = gu
        if SILU_ACTIVATION:
            sig = 1.0 / (1.0 + tl.exp(-pre))
            g = gu * sig * (1.0 + pre * (1.0 - sig))

        gin = tl.where(i < n_tok, g, 0.0)
        dw0 += gin * xu
        dw1 += gin * xd1
        if KERNEL_WIDTH >= 3:
            dw2 += gin * xd2
        if KERNEL_WIDTH >= 4:
            dw3 += gin * xd3
        if HAS_BIAS:
            db += gin
        if HAS_INITIAL_STATES:
            gf0 = tl.where(i == 0, g, gf0)
            gf1 = tl.where(i == 1, g, gf1)
            gf2 = tl.where(i == 2, g, gf2)

        # dx[u - (width - 1)] = sum_j w[j] * dpre[u - j]
        dxv = w0 * g + w1 * gd1
        if KERNEL_WIDTH >= 3:
            dxv += w2 * gd2
        if KERNEL_WIDTH >= 4:
            dxv += w3 * gd3
        tl.store(
            dx_ptrs,
            dxv.to(dx_ptr.dtype.element_ty),
            mask=feat_mask & (i >= KERNEL_WIDTH - 1),
        )
        dx_ptrs += stride_dx_token

        xd3 = xd2
        xd2 = xd1
        xd1 = xu
        gd3 = gd2
        gd2 = gd1
        gd1 = g

    # Per-chunk partial sums; the host reduces them over chunks (deterministic).
    part_base = dw_partial_ptr + (pid.to(tl.int64) * dim + feats) * n_part_rows
    tl.store(part_base + (KERNEL_WIDTH - 1), dw0, mask=feat_mask)
    tl.store(part_base + (KERNEL_WIDTH - 2), dw1, mask=feat_mask)
    if KERNEL_WIDTH >= 3:
        tl.store(part_base + (KERNEL_WIDTH - 3), dw2, mask=feat_mask)
    if KERNEL_WIDTH >= 4:
        tl.store(part_base, dw3, mask=feat_mask)
    if HAS_BIAS:
        tl.store(part_base + KERNEL_WIDTH, db, mask=feat_mask)

    # dinit[k] = sum_{j <= k} w[j] * dpre[k - j], from the first chunk only
    # A constexpr and a runtime test stay as two nested ifs
    if HAS_INITIAL_STATES:  # noqa: SIM102
        if chunk == 0:
            for k in tl.static_range(KERNEL_WIDTH - 1):
                di = tl.zeros((BLOCK_N,), dtype=tl.float32)
                for j in tl.static_range(k + 1):
                    di += _pick(j, w0, w1, w2, w3) * _pick(k - j, gf0, gf1, gf2, gf2)
                tl.store(
                    dinit_ptr
                    + seq.to(tl.int64) * stride_di_seq
                    + feats.to(tl.int64) * stride_di_dim
                    + k * stride_di_tok,
                    di.to(dinit_ptr.dtype.element_ty),
                    mask=feats < tl.where(n_state > 0, dim, 0),
                )

    # dx of the tokens after the last sequence is zero. The end is read on the
    # device, so a reused schedule needs no host sync; the programs share the rows.
    end = tl.load(query_start_loc_ptr + n_seqs)
    for r in range(end + pid, cu_seqlen, tl.num_programs(0)):
        tl.store(
            dx_ptr + r.to(tl.int64) * stride_dx_token + dx_feat,
            tl.zeros((BLOCK_N,), dtype=dx_ptr.dtype.element_ty),
            mask=feat_mask,
        )
