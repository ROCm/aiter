# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Kernels of the top-k / top-p draw from bf16 logits (see the wrapper in
aiter/ops/triton/fusions/topk_topp_sample.py).

Pass 1, one program a (row, chunk): the chunk's max and sum of exp of
``logits / t``, and the max of each 32-column group as a 16-bit ordered key.
Pass 2, one program a row: the row's softmax statistics, the top-k columns
(the largest keys, the lowest column first on a tie), then the draw of the
top-k-first sampler over them.
"""

import triton
import triton.language as tl
from triton.language.extra import libdevice

from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr

GROUP = tl.constexpr(32)
KEY_NONE = tl.constexpr(-32769)  # below the key of every bf16 value

_stats_repr = make_kernel_repr(
    "_topk_topp_sample_stats_kernel",
    ["GROUPS", "CHUNKS_PER_PROGRAM", "ROW_MAP", "PENALTY", "num_warps"],
)
_select_repr = make_kernel_repr(
    "_topk_topp_sample_select_kernel",
    ["KB", "CAPG", "ROW_MAP", "PENALTY", "num_warps"],
)


@triton.jit
def _ordered_key(bits):
    """int16 bf16 bits -> int32 key whose order is the value order (-0 and +0
    get the same key, as they compare equal)."""
    b = bits.to(tl.int32)
    return tl.where(b < 0, (b ^ 0x7FFF) + 1, b)


@triton.jit
def _key_value(key):
    """The inverse of _ordered_key, as fp32."""
    b = tl.where(key < 0, (key - 1) ^ 0x7FFF, key).to(tl.int16)
    return b.to(tl.bfloat16, bitcast=True).to(tl.float32)


@triton.jit
def _row_source(
    r, verify_idx_ptr, bonus_idx_ptr, cu_ptr, n_verify, n_cu, ROW_MAP, BLOCK_BS
):
    """(logits row, request) of output row r. ROW_MAP 0: both are r.
    ROW_MAP 1: verify rows then bonus rows; a verify row's request is
    searchsorted(cu, r, right=True)."""
    if ROW_MAP:
        is_v = r < n_verify
        src_v = tl.load(verify_idx_ptr + r, mask=is_v, other=0)
        src_b = tl.load(bonus_idx_ptr + (r - n_verify), mask=r >= n_verify, other=0)
        src = tl.where(is_v, src_v, src_b).to(tl.int64)
        offs = tl.arange(0, BLOCK_BS)
        cu = tl.load(cu_ptr + offs, mask=offs < n_cu, other=0)
        req_v = tl.sum(((cu <= r) & (offs < n_cu)).to(tl.int32), axis=0)
        req = tl.where(is_v, req_v, r - n_verify)
    else:
        src = r.to(tl.int64)
        req = r
    return src, req


@triton.jit
def _load_logits(
    logits_ptr,
    stride_row,
    src,
    cols,
    valid,
    other,
    col_lo,
    col_hi,
    pen_ids_ptr,
    pen_row_seq_ptr,
    pen_row_pos_ptr,
    pen_seq_start_ptr,
    pen_slot_ptr,
    pen_val_ptr,
    pen_mask_ptr,
    pen_mask_stride,
    PENALTY: tl.constexpr,
    MAX_EXTRA: tl.constexpr,
):
    """bf16 logits of row src at cols. With PENALTY: minus the request's
    presence penalty on its generated tokens (a byte mask a slot) and on the
    drafts before the row, rounded to bf16 as an in-place pass would store
    them (atom presence_penalty)."""
    x = tl.load(logits_ptr + src * stride_row + cols, mask=valid, other=other)
    if PENALTY:
        s = tl.load(pen_row_seq_ptr + src)
        p = tl.load(pen_val_ptr + s)
        slot = tl.load(pen_slot_ptr + s).to(tl.int64)
        pos = tl.load(pen_row_pos_ptr + src)
        start = tl.load(pen_seq_start_ptr + s)
        mask_row = pen_mask_ptr + slot * pen_mask_stride
        present = tl.load(mask_row + cols, mask=valid, other=0) != 0
        for k in tl.static_range(1, MAX_EXTRA + 1):
            tok = tl.load(pen_ids_ptr + start + k, mask=k <= pos, other=-1)
            # a draft outside [col_lo, col_hi] cannot be in cols
            if (tok >= col_lo) & (tok <= col_hi):
                present = present | (cols == tok)
        x = (x.to(tl.float32) - tl.where(present, p, 0.0)).to(tl.bfloat16)
    return x


@triton.jit(repr=_stats_repr)
def _topk_topp_sample_stats_kernel(
    logits_ptr,
    stride_row,
    verify_idx_ptr,
    bonus_idx_ptr,
    cu_ptr,
    temp_ptr,
    m_ptr,
    z_ptr,
    g_ptr,
    pen_ids_ptr,
    pen_row_seq_ptr,
    pen_row_pos_ptr,
    pen_seq_start_ptr,
    pen_slot_ptr,
    pen_val_ptr,
    pen_mask_ptr,
    pen_mask_stride,
    n_verify,
    n_cu,
    V,
    NG,
    NCH,
    ROW_MAP: tl.constexpr,
    BLOCK_BS: tl.constexpr,
    PENALTY: tl.constexpr,
    MAX_EXTRA: tl.constexpr,
    GROUPS: tl.constexpr,
    CHUNKS_PER_PROGRAM: tl.constexpr,
):
    r = tl.program_id(0)
    src, req = _row_source(
        r, verify_idx_ptr, bonus_idx_ptr, cu_ptr, n_verify, n_cu, ROW_MAP, BLOCK_BS
    )
    t = tl.load(temp_ptr + req)
    L2E: tl.constexpr = 1.4426950408889634
    for i in range(CHUNKS_PER_PROGRAM):
        c = tl.program_id(1) * CHUNKS_PER_PROGRAM + i
        g = c * GROUPS + tl.arange(0, GROUPS)
        cols = g[:, None] * 32 + tl.arange(0, 32)[None, :]
        mask = cols < V
        x = _load_logits(
            logits_ptr,
            stride_row,
            src,
            cols,
            mask,
            float("-inf"),
            c * GROUPS * 32,
            c * GROUPS * 32 + GROUPS * 32 - 1,
            pen_ids_ptr,
            pen_row_seq_ptr,
            pen_row_pos_ptr,
            pen_seq_start_ptr,
            pen_slot_ptr,
            pen_val_ptr,
            pen_mask_ptr,
            pen_mask_stride,
            PENALTY,
            MAX_EXTRA,
        )
        xf = x.to(tl.float32)
        # group max as a key (the max of a bf16 group is a bf16 value)
        gmax = tl.max(xf, axis=1)
        gkey = _ordered_key(gmax.to(tl.bfloat16).to(tl.int16, bitcast=True))
        tl.store(g_ptr + r * NG + g, gkey.to(tl.int16), mask=g < NG)
        # max(x / t) is max(x) / t: the division rounds monotonically. The sum
        # takes exp2((x - max(x)) * log2(e) / t): no division a column, and the
        # max column gives exp2(0) for every t.
        xmax = tl.max(gmax, axis=0)
        m = xmax / t
        ref = tl.where(xmax == float("-inf"), 0.0, xmax)
        z = tl.sum(tl.sum(tl.math.exp2((xf - ref) * (L2E / t)), axis=1), axis=0)
        tl.store(m_ptr + r * NCH + c, m, mask=c < NCH)
        tl.store(z_ptr + r * NCH + c, z, mask=c < NCH)


@triton.jit
def _count_ge(key, mid):
    return tl.sum(tl.sum((key >= mid).to(tl.int32), axis=1), axis=0)


@triton.jit(repr=_select_repr)
def _topk_topp_sample_select_kernel(
    logits_ptr,
    stride_row,
    verify_idx_ptr,
    bonus_idx_ptr,
    cu_ptr,
    temp_ptr,
    top_p_ptr,
    top_p_val,
    u_ptr,
    m_ptr,
    z_ptr,
    g_ptr,
    ws_ptr,
    out_ptr,
    dbg_ptr,
    pen_ids_ptr,
    pen_row_seq_ptr,
    pen_row_pos_ptr,
    pen_seq_start_ptr,
    pen_slot_ptr,
    pen_val_ptr,
    pen_mask_ptr,
    pen_mask_stride,
    n_verify,
    n_cu,
    V,
    NG,
    NCH,
    K,
    eps,
    ROW_MAP: tl.constexpr,
    BLOCK_BS: tl.constexpr,
    NCH_P2: tl.constexpr,
    NG_P2: tl.constexpr,
    NB: tl.constexpr,
    CAPG: tl.constexpr,
    KB: tl.constexpr,
    TOP_P_PER_REQ: tl.constexpr,
    DEBUG: tl.constexpr,
    PENALTY: tl.constexpr,
    MAX_EXTRA: tl.constexpr,
):
    r = tl.program_id(0)
    src, req = _row_source(
        r, verify_idx_ptr, bonus_idx_ptr, cu_ptr, n_verify, n_cu, ROW_MAP, BLOCK_BS
    )
    t = tl.load(temp_ptr + req)
    # workspace a row: NG group ids, then KB (column, key) pairs
    ws = ws_ptr + r.to(tl.int64) * (NG_P2 + 2 * KB)
    sel_ws = ws + NG_P2

    # softmax statistics of the row
    ch = tl.arange(0, NCH_P2)
    ms = tl.load(m_ptr + r * NCH + ch, mask=ch < NCH, other=float("-inf"))
    zs = tl.load(z_ptr + r * NCH + ch, mask=ch < NCH, other=0.0)
    M = tl.max(ms, axis=0)
    zc = tl.where(ms == float("-inf"), 0.0, zs * tl.exp(ms - M))
    Z = tl.sum(zc, axis=0)

    # T: the k-th largest group max. The top-k columns are in the groups whose
    # max is >= T.
    BS: tl.constexpr = NG_P2 // NB
    gi = tl.arange(0, NB)[:, None] * BS + tl.arange(0, BS)[None, :]
    G = tl.load(g_ptr + r * NG + gi, mask=gi < NG, other=0).to(tl.int32)
    G = tl.where(gi < NG, G, KEY_NONE)
    # Bounds: T <= the max key; T >= the smallest max of NB blocks of groups
    # when K blocks or more hold groups (each of those maxima is a group max).
    bmax = tl.max(G, axis=1)
    bvalid = tl.arange(0, NB) * BS < NG
    top = tl.max(bmax, axis=0)
    lo = tl.min(tl.where(bvalid, bmax, 32767), axis=0)
    lo = tl.where(tl.sum(bvalid.to(tl.int32), axis=0) >= K, lo, KEY_NONE)
    hi = top + 1
    while hi - lo > 1:
        mid = (lo + hi) >> 1
        ok = _count_ge(G, mid) >= K
        lo = tl.where(ok, mid, lo)
        hi = tl.where(ok, hi, mid)
    T = lo
    cand = tl.reshape((G >= T).to(tl.int32), (NG_P2,))
    ng = tl.sum(cand, axis=0)
    pos = tl.cumsum(cand, axis=0) - 1
    tl.store(ws + pos, tl.arange(0, NG_P2), mask=cand > 0)
    tl.debug_barrier()

    # vK: the k-th largest key of the candidate columns; select the columns
    # above it and the lowest columns equal to it.
    lane = tl.arange(0, 32)
    if ng <= CAPG:
        gs = tl.arange(0, CAPG)
        gid = tl.load(ws + gs, mask=gs < ng, other=0)
        cols = gid[:, None] * 32 + lane[None, :]
        valid = (gs[:, None] < ng) & (cols < V)
        x = _load_logits(
            logits_ptr,
            stride_row,
            src,
            cols,
            valid,
            0.0,
            0,
            V,
            pen_ids_ptr,
            pen_row_seq_ptr,
            pen_row_pos_ptr,
            pen_seq_start_ptr,
            pen_slot_ptr,
            pen_val_ptr,
            pen_mask_ptr,
            pen_mask_stride,
            PENALTY,
            MAX_EXTRA,
        )
        key = tl.where(valid, _ordered_key(x.to(tl.int16, bitcast=True)), KEY_NONE)
        lo = T
        hi = top + 1
        while hi - lo > 1:
            mid = (lo + hi) >> 1
            ok = _count_ge(key, mid) >= K
            lo = tl.where(ok, mid, lo)
            hi = tl.where(ok, hi, mid)
        vK = lo
        n_gt = tl.sum(tl.sum((key > vK).to(tl.int32), axis=1), axis=0)
        need_eq = K - n_gt
        flat_key = tl.reshape(key, (CAPG * 32,))
        flat_col = tl.reshape(cols, (CAPG * 32,))
        eq = (flat_key == vK).to(tl.int32)
        eq_rank = tl.cumsum(eq, axis=0) - eq
        sel = (flat_key > vK) | ((eq > 0) & (eq_rank < need_eq))
        spos = tl.cumsum(sel.to(tl.int32), axis=0) - 1
        tl.store(sel_ws + spos, flat_col, mask=sel)
        tl.store(sel_ws + KB + spos, flat_key, mask=sel)
    else:
        # many groups tie at T: the same steps over batches of CAPG groups
        lo = T
        hi = top + 1
        while hi - lo > 1:
            mid = (lo + hi) >> 1
            cnt = 0
            for b in range(0, ng, CAPG):
                gs = b + tl.arange(0, CAPG)
                gid = tl.load(ws + gs, mask=gs < ng, other=0)
                cols = gid[:, None] * 32 + lane[None, :]
                valid = (gs[:, None] < ng) & (cols < V)
                x = _load_logits(
                    logits_ptr,
                    stride_row,
                    src,
                    cols,
                    valid,
                    0.0,
                    0,
                    V,
                    pen_ids_ptr,
                    pen_row_seq_ptr,
                    pen_row_pos_ptr,
                    pen_seq_start_ptr,
                    pen_slot_ptr,
                    pen_val_ptr,
                    pen_mask_ptr,
                    pen_mask_stride,
                    PENALTY,
                    MAX_EXTRA,
                )
                key = tl.where(
                    valid, _ordered_key(x.to(tl.int16, bitcast=True)), KEY_NONE
                )
                cnt += _count_ge(key, mid)
            ok = cnt >= K
            lo = tl.where(ok, mid, lo)
            hi = tl.where(ok, hi, mid)
        vK = lo
        n_gt = 0
        for b in range(0, ng, CAPG):
            gs = b + tl.arange(0, CAPG)
            gid = tl.load(ws + gs, mask=gs < ng, other=0)
            cols = gid[:, None] * 32 + lane[None, :]
            valid = (gs[:, None] < ng) & (cols < V)
            x = _load_logits(
                logits_ptr,
                stride_row,
                src,
                cols,
                valid,
                0.0,
                0,
                V,
                pen_ids_ptr,
                pen_row_seq_ptr,
                pen_row_pos_ptr,
                pen_seq_start_ptr,
                pen_slot_ptr,
                pen_val_ptr,
                pen_mask_ptr,
                pen_mask_stride,
                PENALTY,
                MAX_EXTRA,
            )
            key = tl.where(valid, _ordered_key(x.to(tl.int16, bitcast=True)), KEY_NONE)
            n_gt += tl.sum(tl.sum((key > vK).to(tl.int32), axis=1), axis=0)
        need_eq = K - n_gt
        eq_seen = 0
        sel_seen = 0
        for b in range(0, ng, CAPG):
            gs = b + tl.arange(0, CAPG)
            gid = tl.load(ws + gs, mask=gs < ng, other=0)
            cols = gid[:, None] * 32 + lane[None, :]
            valid = (gs[:, None] < ng) & (cols < V)
            x = _load_logits(
                logits_ptr,
                stride_row,
                src,
                cols,
                valid,
                0.0,
                0,
                V,
                pen_ids_ptr,
                pen_row_seq_ptr,
                pen_row_pos_ptr,
                pen_seq_start_ptr,
                pen_slot_ptr,
                pen_val_ptr,
                pen_mask_ptr,
                pen_mask_stride,
                PENALTY,
                MAX_EXTRA,
            )
            key = tl.where(valid, _ordered_key(x.to(tl.int16, bitcast=True)), KEY_NONE)
            flat_key = tl.reshape(key, (CAPG * 32,))
            flat_col = tl.reshape(cols, (CAPG * 32,))
            eq = (flat_key == vK).to(tl.int32)
            eq_rank = eq_seen + tl.cumsum(eq, axis=0) - eq
            sel = (flat_key > vK) | ((eq > 0) & (eq_rank < need_eq))
            spos = sel_seen + tl.cumsum(sel.to(tl.int32), axis=0) - 1
            tl.store(sel_ws + spos, flat_col, mask=sel)
            tl.store(sel_ws + KB + spos, flat_key, mask=sel)
            eq_seen += tl.sum(eq, axis=0)
            sel_seen += tl.sum(sel.to(tl.int32), axis=0)
    tl.debug_barrier()

    # the k selected columns by value, then column
    offs = tl.arange(0, KB)
    kmask = offs < K
    s_col = tl.load(sel_ws + offs, mask=kmask, other=0)
    s_key = tl.load(sel_ws + KB + offs, mask=kmask, other=KEY_NONE)
    s_col_j = tl.load(sel_ws + offs[None, :], mask=kmask[None, :], other=0)
    s_key_j = tl.load(sel_ws + KB + offs[None, :], mask=kmask[None, :], other=KEY_NONE)
    before = (s_key_j > s_key[:, None]) | (
        (s_key_j == s_key[:, None]) & (s_col_j < s_col[:, None])
    )
    rank = tl.sum((before & kmask[None, :]).to(tl.int32), axis=1)
    rank = tl.where(kmask, rank, KB)
    at = rank[:, None] == offs[None, :]
    col = tl.sum(tl.where(at, s_col[:, None], 0), axis=0)
    xk = _key_value(tl.sum(tl.where(at, s_key[:, None], 0), axis=0))

    # The top-k-first sampler (aiter.ops.sampling): softmax probabilities of
    # the k columns, renormalised, top-p threshold p / (their mass).
    e = tl.exp(xk / t - M)
    pk = tl.where(kmask, e * (1.0 / Z), 0.0)
    # clamps that keep a NaN, as torch's clamp does
    denom = tl.maximum(tl.sum(pk, axis=0), 1e-8, propagate_nan=tl.PropagateNan.ALL)
    p = tl.where(kmask, pk / denom, 0.0)
    if TOP_P_PER_REQ:
        tp = tl.load(top_p_ptr + req).to(tl.float32)
    else:
        tp = top_p_val
    top_p = tl.minimum(tp / denom, 1.0, propagate_nan=tl.PropagateNan.ALL)
    u = tl.load(u_ptr + r)
    # top_p_sampling_small_k
    gt = tl.sum(tl.where(p[None, :] > p[:, None], p[None, :], 0.0), axis=1)
    keep = (gt < top_p) & kmask
    q = tl.where(keep, p, 0.0)
    cdf = tl.cumsum(q, axis=0)
    total = tl.sum(q, axis=0)
    hit = keep & (cdf > u * total)
    idx = tl.min(tl.where(hit, offs, KB), axis=0)
    last = tl.max(tl.where(keep, offs, -1), axis=0)
    idx = tl.where(idx == KB, last, idx)
    idx = tl.where(idx < 0, 0, idx)
    # a greedy row (t at the clamp floor) takes the argmax: the first column
    idx = tl.where(t <= eps, 0, idx)
    token = tl.sum(tl.where(offs == idx, col, 0), axis=0)
    # A row whose softmax is NaN (a NaN or +inf logit, or every logit -inf):
    # the argmax of the NaN probabilities is column 0.
    nan_row = libdevice.isnan(Z) | (M == float("-inf"))
    token = tl.where((t <= eps) & nan_row, 0, token)
    tl.store(out_ptr + r, token)
    if DEBUG:
        dbg = dbg_ptr + r * (3 * KB + 2)
        tl.store(dbg + offs, col, mask=kmask)
        tl.store(dbg + KB + offs, p.to(tl.int32, bitcast=True), mask=kmask)
        tl.store(dbg + 2 * KB + offs, pk.to(tl.int32, bitcast=True), mask=kmask)
        tl.store(dbg + 3 * KB, top_p.to(tl.int32, bitcast=True))
        tl.store(dbg + 3 * KB + 1, Z.to(tl.int32, bitcast=True))
