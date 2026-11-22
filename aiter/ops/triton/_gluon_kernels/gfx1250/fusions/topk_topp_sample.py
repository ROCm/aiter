# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""The top-k / top-p draw of
aiter/ops/triton/_triton_kernels/fusions/topk_topp_sample.py as two Gluon
kernels (gfx1250, wave32), with the same token for the same uniform.

Stats: each GROUPS-group chunk is reduced in the Triton stats kernel's layout
and order, also with the presence penalty, so the softmax partials are the
same bits. Select: the same k columns in the same order and the same top-p
arithmetic, in fewer dependent steps. L, the k-th
largest chunk max, bounds the k-th largest group max from below, so the k
first groups by (key desc, group asc) are among the few groups >= L. Their
columns hold the k first columns by (key desc, column asc), and those columns
are >= T, the k-th group max. Candidates are compacted in shared memory and
ranked pairwise. When a bound leaves more than CAP candidates, exact
bisections pick the k first.

FAST (aiter AITER_TOPK_SELECT_FAST): the group rank takes up to CAP1_ (128)
candidates, the bisections carry their counts and skip the index bisection
when the ties at the k-th key are all taken, and up to 1024 group candidates
are compacted before their bisection: the same k columns, far fewer dependent
steps when the chunk bound is loose (realistic rows: 30-200 groups >= L).
NP_P2 (atom ATOM_MTP_DRAFT_LMHEAD_STATS): the softmax partials and group
keys come from the draft LM head GEMM's epilogue, one per 256-column tile.
RNG_STATE / OUT2 / REC_CTX (atom ATOM_MTP_DRAFT_FUSED_SELECT): the draft
draw's uniform (tl.rand at a device counter the stats kernel advances), a
second copy of the tokens and the draft presence record, in the kernels.
"""

import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr

KEY_NONE = gl.constexpr(-32769)  # below the key of every bf16 value

_stats_repr = make_kernel_repr(
    "_topk_topp_sample_stats_gluon_kernel",
    ["GROUPS", "CPP", "ROW_MAP", "PENALTY", "ANCHOR", "NW", "RNG_ADV"],
)
_select_repr = make_kernel_repr(
    "_topk_topp_sample_select_gluon_kernel",
    ["GROUPS", "KB", "ROW_MAP", "PENALTY", "ANCHOR", "PHILOX", "NW", "FAST", "RNG_STATE", "REC_CTX"],
)


@gluon.jit
def _ordered_key(bits):
    """int16 bf16 bits -> int32 key whose order is the value order."""
    b = bits.to(gl.int32)
    return gl.where(b < 0, (b ^ 0x7FFF) + 1, b)


@gluon.jit
def _key_value(key):
    """The inverse of _ordered_key, as fp32."""
    b = gl.where(key < 0, (key - 1) ^ 0x7FFF, key).to(gl.int16)
    return b.to(gl.bfloat16, bitcast=True).to(gl.float32)


@gluon.jit
def _count(m):
    return gl.sum(gl.sum(m.to(gl.int32), axis=1), axis=0)


@gluon.jit
def _first_k(key, idx, valid, K, lo, IDX_HI):
    """Mask of the K first valid elements by (key desc, idx asc), given that
    count(valid & key >= lo) >= K: the K-th largest key by bisection, then the
    largest index of the ties at it that is still taken."""
    hi = gl.max(gl.max(gl.where(valid, key, KEY_NONE), axis=1), axis=0) + 1
    while hi - lo > 1:
        mid = (lo + hi) >> 1
        ok = _count(valid & (key >= mid)) >= K
        lo = gl.where(ok, mid, lo)
        hi = gl.where(ok, hi, mid)
    need = K - _count(valid & (key > lo))
    eq = valid & (key == lo)
    ilo = -1
    ihi = IDX_HI
    while ihi - ilo > 1:
        mid = (ilo + ihi) >> 1
        ok = _count(eq & (idx <= mid)) >= need
        ihi = gl.where(ok, mid, ihi)
        ilo = gl.where(ok, ilo, mid)
    return valid & ((key > lo) | (eq & (idx <= ihi)))


@gluon.jit
def _first_k2(key, idx, valid, K, lo, c_lo, IDX_HI):
    """_first_k's mask (the same bits) with its counts carried: the K-th key
    by bisection from (lo, c_lo = count(valid & key >= lo)), the count above
    it known at the end, and the index bisection only when the ties at the
    K-th key are cut."""
    hi = gl.max(gl.max(gl.where(valid, key, KEY_NONE), axis=1), axis=0) + 1
    c_hi = c_lo * 0
    while hi - lo > 1:
        mid = (lo + hi) >> 1
        c = _count(valid & (key >= mid))
        ok = c >= K
        lo = gl.where(ok, mid, lo)
        c_lo = gl.where(ok, c, c_lo)
        hi = gl.where(ok, hi, mid)
        c_hi = gl.where(ok, c_hi, c)
    need = K - c_hi
    eq = valid & (key == lo)
    ihi = IDX_HI
    if c_lo - c_hi > need:
        ilo = -1
        while ihi - ilo > 1:
            mid = (ilo + ihi) >> 1
            ok = _count(eq & (idx <= mid)) >= need
            ihi = gl.where(ok, mid, ihi)
            ilo = gl.where(ok, ilo, mid)
    return valid & ((key > lo) | (eq & (idx <= ihi)))


@gluon.jit
def _rand_at(seed_lo, seed_hi, off, like):
    """tl.rand(seed, off) (triton.language.random) for an int64 scalar
    counter: Philox-4x32-10 at counter (lo(off), hi(off), 0, 0), key
    (lo(seed), hi(seed)); shaped like `like`."""
    c0 = (gl.zeros_like(like).to(gl.int64) + off).to(gl.uint32)
    c1 = ((gl.zeros_like(like).to(gl.int64) + off) >> 32).to(gl.uint32)
    c2 = gl.zeros_like(like).to(gl.uint32)
    c3 = gl.zeros_like(like).to(gl.uint32)
    k0 = (gl.zeros_like(like) + seed_lo).to(gl.uint32)
    k1 = (gl.zeros_like(like) + seed_hi).to(gl.uint32)
    for _ in gl.static_range(10):
        _c0 = c0
        _c2 = c2
        c0 = gl.umulhi(_c2, 0xCD9E8D57) ^ c1 ^ k0
        c2 = gl.umulhi(_c0, 0xD2511F53) ^ c3 ^ k1
        c1 = _c2 * 0xCD9E8D57
        c3 = _c0 * 0xD2511F53
        k0 = k0 + 0x9E3779B9
        k1 = k1 + 0xBB67AE85
    x = c0.to(gl.int32, bitcast=True)
    x = gl.where(x < 0, ~x, x)
    return x.to(gl.float32) * 4.6566127342e-10


@gluon.jit
def _slots(cand, spare, L1: gl.constexpr):
    """Row-major exclusive prefix count of the 2-D mask cand: the slot of each
    candidate. Others get 1024 + spare, past the candidates."""
    c = cand.to(gl.int32)
    rc = gl.sum(c, axis=1)
    rc1 = gl.convert_layout(rc, L1)
    roff = gl.convert_layout(gl.cumsum(rc1, axis=0) - rc1, rc.type.layout)
    wpre = gl.cumsum(c, axis=1) - c
    return gl.where(cand, roff[:, None] + wpre, 1024 + spare)


@gluon.jit
def _penalize(
    x,
    cols,
    valid,
    src,
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
    MAX_EXTRA: gl.constexpr,
    ANCHOR: gl.constexpr,
):
    """x minus the request's presence penalty on its marked tokens and on the
    drafts before the row (ANCHOR: and on the query's first token), rounded to
    bf16 as an in-place pass stores them (atom presence_penalty)."""
    s = gl.load(pen_row_seq_ptr + src)
    p = gl.load(pen_val_ptr + s)
    slot = gl.load(pen_slot_ptr + s).to(gl.int64)
    pos = gl.load(pen_row_pos_ptr + src)
    start = gl.load(pen_seq_start_ptr + s)
    present = (
        gl.load(pen_mask_ptr + slot * pen_mask_stride + cols, mask=valid, other=0) != 0
    )
    K0: gl.constexpr = 0 if ANCHOR else 1
    for k in gl.static_range(K0, MAX_EXTRA + 1):
        tok = gl.load(pen_ids_ptr + start + k, mask=k <= pos, other=-1)
        # a token outside [col_lo, col_hi] cannot be in cols
        if (tok >= col_lo) & (tok <= col_hi):
            present = present | (cols == tok)
    return (x.to(gl.float32) - gl.where(present, p, 0.0)).to(gl.bfloat16)


@gluon.jit
def _philox_uniform(r, seed_lo, seed_hi, off_lo, off_hi, like):
    """U[0, 1) from Philox-4x32-10 at counter (r, off_lo, off_hi, 0) and key
    (seed_lo, seed_hi), as triton.language.random.rand; shaped like `like`."""
    c0 = (gl.zeros_like(like) + r).to(gl.uint32)
    c1 = (gl.zeros_like(like) + off_lo).to(gl.uint32)
    c2 = (gl.zeros_like(like) + off_hi).to(gl.uint32)
    c3 = gl.zeros_like(like).to(gl.uint32)
    k0 = (gl.zeros_like(like) + seed_lo).to(gl.uint32)
    k1 = (gl.zeros_like(like) + seed_hi).to(gl.uint32)
    for _ in gl.static_range(10):
        _c0 = c0
        _c2 = c2
        c0 = gl.umulhi(_c2, 0xCD9E8D57) ^ c1 ^ k0
        c2 = gl.umulhi(_c0, 0xD2511F53) ^ c3 ^ k1
        c1 = _c2 * 0xCD9E8D57
        c3 = _c0 * 0xD2511F53
        k0 = k0 + 0x9E3779B9
        k1 = k1 + 0xBB67AE85
    x = c0.to(gl.int32, bitcast=True)
    x = gl.where(x < 0, -x - 1, x)
    return x.to(gl.float32) * 4.6566127342e-10


@gluon.jit
def _row_source(
    r,
    verify_idx_ptr,
    bonus_idx_ptr,
    cu_ptr,
    n_verify,
    n_cu,
    ROW_MAP: gl.constexpr,
    BLOCK_BS: gl.constexpr,
    L1: gl.constexpr,
):
    """(logits row, request) of output row r: verify rows then bonus rows."""
    if ROW_MAP:
        is_v = r < n_verify
        src_v = gl.load(verify_idx_ptr + r, mask=is_v, other=0)
        src_b = gl.load(bonus_idx_ptr + (r - n_verify), mask=r >= n_verify, other=0)
        src = gl.where(is_v, src_v, src_b).to(gl.int64)
        bs = gl.arange(0, BLOCK_BS, layout=L1)
        cu = gl.load(cu_ptr + bs, mask=bs < n_cu, other=0)
        req_v = gl.sum(((cu <= r) & (bs < n_cu)).to(gl.int32), axis=0)
        req = gl.where(is_v, req_v, r - n_verify)
    else:
        src = r.to(gl.int64)
        req = r
    return src, req


@gluon.jit(repr=_stats_repr)
def _topk_topp_sample_stats_gluon_kernel(
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
    rng_state_ptr,
    ROW_MAP: gl.constexpr,
    BLOCK_BS: gl.constexpr,
    PENALTY: gl.constexpr,
    MAX_EXTRA: gl.constexpr,
    ANCHOR: gl.constexpr,
    GROUPS: gl.constexpr,
    CPP: gl.constexpr,
    NW: gl.constexpr,
    RNG_ADV: gl.constexpr = False,
):
    """A program per (row, CPP chunks): each chunk's max and sum of exp of
    logits / t, and each 32-column group's max as a 16-bit ordered key."""
    gl.static_assert(NW == 4 * CPP, "4 waves a chunk")
    L1: gl.constexpr = gl.BlockedLayout([1], [32], [NW], [0])
    # each chunk is [GROUPS, 32] in the Triton stats kernel's layout
    B3: gl.constexpr = gl.BlockedLayout([1, 1, 8], [1, 8, 4], [CPP, 4, 1], [2, 1, 0])
    L2E: gl.constexpr = 1.4426950408889634
    r = gl.program_id(0)
    pc = gl.program_id(1)
    src, req = _row_source(
        r, verify_idx_ptr, bonus_idx_ptr, cu_ptr, n_verify, n_cu, ROW_MAP, BLOCK_BS, L1
    )
    t = gl.load(temp_ptr + req)
    ci = gl.arange(0, CPP, layout=gl.SliceLayout(1, gl.SliceLayout(2, B3)))
    gi = gl.arange(0, GROUPS, layout=gl.SliceLayout(0, gl.SliceLayout(2, B3)))
    li = gl.arange(0, 32, layout=gl.SliceLayout(0, gl.SliceLayout(1, B3)))
    c = pc * CPP + ci
    g = c[:, None] * GROUPS + gi[None, :]
    cols = g[:, :, None] * 32 + li[None, None, :]
    valid = cols < V
    x = gl.load(logits_ptr + src * stride_row + cols, mask=valid, other=float("-inf"))
    if PENALTY:
        lo_col = pc * (CPP * GROUPS * 32)
        x = _penalize(
            x,
            cols,
            valid,
            src,
            lo_col,
            lo_col + CPP * GROUPS * 32 - 1,
            pen_ids_ptr,
            pen_row_seq_ptr,
            pen_row_pos_ptr,
            pen_seq_start_ptr,
            pen_slot_ptr,
            pen_val_ptr,
            pen_mask_ptr,
            pen_mask_stride,
            MAX_EXTRA,
            ANCHOR,
        )
    xf = x.to(gl.float32)
    # group max as a key (the max of a bf16 group is a bf16 value)
    gmax = gl.max(xf, axis=2)
    gkey = _ordered_key(gmax.to(gl.bfloat16).to(gl.int16, bitcast=True))
    gl.store(g_ptr + r * NG + g, gkey.to(gl.int16), mask=g < NG)
    # max(x / t) is max(x) / t; the sum takes exp2((x - max(x)) * log2(e) / t)
    xmax = gl.max(gmax, axis=1)
    ref = gl.where(xmax == float("-inf"), 0.0, xmax)
    z = gl.sum(gl.sum(gl.exp2((xf - ref[:, None, None]) * (L2E / t)), axis=2), axis=1)
    gl.store(m_ptr + r * NCH + c, xmax / t, mask=c < NCH)
    gl.store(z_ptr + r * NCH + c, z, mask=c < NCH)
    if RNG_ADV:
        # the draft draws' Philox counter += rows (the select reads it back
        # as state - rows, after this kernel)
        if (r == 0) & (pc == 0):
            gl.store(rng_state_ptr, gl.load(rng_state_ptr) + gl.num_programs(0))


@gluon.jit(repr=_select_repr)
def _topk_topp_sample_select_gluon_kernel(
    logits_ptr,
    stride_row,
    verify_idx_ptr,
    bonus_idx_ptr,
    cu_ptr,
    temp_ptr,
    top_p_ptr,
    top_p_val,
    u_ptr,
    seed_lo,
    seed_hi,
    off_lo,
    off_hi,
    m_ptr,
    z_ptr,
    g_ptr,
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
    pen_mask_width,
    n_verify,
    n_cu,
    V,
    NG,
    NCH,
    K,
    eps,
    rng_state_ptr,
    out2_ptr,
    rec_ctx_ptr,
    rec_pos_ptr,
    NP,
    ROW_MAP: gl.constexpr,
    BLOCK_BS: gl.constexpr,
    PENALTY: gl.constexpr,
    MAX_EXTRA: gl.constexpr,
    ANCHOR: gl.constexpr,
    GROUPS: gl.constexpr,
    NCH_P2: gl.constexpr,
    KB: gl.constexpr,
    CAP: gl.constexpr,
    TOP_P_PER_REQ: gl.constexpr,
    PHILOX: gl.constexpr,
    DEBUG: gl.constexpr,
    NW: gl.constexpr,
    FAST: gl.constexpr = False,
    RNG_STATE: gl.constexpr = False,
    OUT2: gl.constexpr = False,
    REC_CTX: gl.constexpr = 0,
    CAP1_: gl.constexpr = 0,
    CAP2_: gl.constexpr = 0,
    NP_P2: gl.constexpr = 0,
):
    # FAST: the pairwise ranks take up to CAP1_ groups / CAP2_ columns before
    # the exact first-k searches (realistic rows often bound 64-200 groups)
    CAP1: gl.constexpr = CAP1_ if CAP1_ > 0 else CAP
    CAP2: gl.constexpr = CAP2_ if CAP2_ > 0 else CAP
    L1: gl.constexpr = gl.BlockedLayout([1], [32], [NW], [0])
    TPW1: gl.constexpr = GROUPS // 8
    GT: gl.constexpr = gl.BlockedLayout([1, 8], [32 // TPW1, TPW1], [NW, 1], [1, 0])
    RL: gl.constexpr = gl.BlockedLayout([1, 4], [4, 8], [NW, 1], [1, 0])
    CT: gl.constexpr = gl.BlockedLayout([1, 8], [8, 4], [NW, 1], [1, 0])
    FL: gl.constexpr = gl.BlockedLayout([1, 1], [1, 32], [NW, 1], [1, 0])
    SH: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [0])

    r = gl.program_id(0)
    src, req = _row_source(
        r, verify_idx_ptr, bonus_idx_ptr, cu_ptr, n_verify, n_cu, ROW_MAP, BLOCK_BS, L1
    )
    t = gl.load(temp_ptr + req)

    lane = gl.arange(0, 32 * NW, layout=L1)
    # softmax statistics of the row
    # NP partials a row: the stats kernel's NCH chunks, or (NP_P2 > 0) the
    # LM head GEMM's per-tile records (aiter gemm_a8w8_lmhead_stats)
    if NP_P2 > 0:
        ch = gl.arange(0, NP_P2, layout=L1)
        # a 32-byte record a (row, tile): 8 keys, max / t (word 4), sum (5)
        rec = (r * NP + ch) * 8
        ms = gl.load(m_ptr + rec + 4, mask=ch < NP, other=float("-inf"))
        zs = gl.load(z_ptr + rec + 5, mask=ch < NP, other=0.0)
    else:
        ch = gl.arange(0, NCH_P2, layout=L1)
        ms = gl.load(m_ptr + r * NCH + ch, mask=ch < NCH, other=float("-inf"))
        zs = gl.load(z_ptr + r * NCH + ch, mask=ch < NCH, other=0.0)
    M = gl.max(ms, axis=0)
    zc = gl.where(ms == float("-inf"), 0.0, zs * gl.exp(ms - M))
    Z = gl.sum(zc, axis=0)

    # group keys as [chunk, group of the chunk]
    cr = gl.arange(0, NCH_P2, layout=gl.SliceLayout(1, GT))
    gj = gl.arange(0, GROUPS, layout=gl.SliceLayout(0, GT))
    gid = cr[:, None] * GROUPS + gj[None, :]
    gv = (cr[:, None] < NCH) & (gid < NG)
    if NP_P2 > 0:
        # the records' keys: 8 contiguous int16 of each 16
        goff = (r * NP + cr[:, None] * (GROUPS // 8)) * 16 + (gj[None, :] >> 3) * 16 + (gj[None, :] & 7)
        goff = gl.max_contiguous(gl.multiple_of(goff, [1, 8]), [1, 8])
        G = gl.load(g_ptr + goff, mask=gv, other=0)
    else:
        G = gl.load(g_ptr + r * NG + gid, mask=gv, other=0)
    G = G.to(gl.int32)
    G = gl.where(gv, G, KEY_NONE)

    # L: the K-th largest chunk max (K chunk maxima are K group maxima >= L)
    cm = gl.max(G, axis=1)
    cpk = gl.where(cr < NCH, ((cm + 32768) << 10) | (1023 - cr), -1)
    ca = gl.convert_layout(cpk, gl.SliceLayout(1, RL))
    cb = gl.convert_layout(cpk, gl.SliceLayout(0, RL))
    crank = gl.sum((cb[None, :] > ca[:, None]).to(gl.int32), axis=1)
    Lp = gl.max(gl.where(crank == K - 1, ca, -1), axis=0)
    L = gl.where(Lp >= 0, (Lp >> 10) - 32768, KEY_NONE)

    # the groups >= L, compacted in group order
    cand = gv & (G >= L)
    n1 = _count(cand)
    gpk = ((G + 32768) << 13) | (8191 - gid)
    s1 = gl.allocate_shared_memory(gl.int32, [2048], SH)
    if n1 > CAP1:
        if FAST & (n1 <= 1024):
            # the candidates compacted (gpk: key, then group order) in a
            # [32, 32] tile, 4 a thread, the first-k search over them
            s1.scatter(
                gl.reshape(gpk, [NCH_P2 * GROUPS]),
                gl.reshape(_slots(cand, gid & 1023, L1), [NCH_P2 * GROUPS]),
                axis=0,
            )
            gl.barrier()
            fi = gl.arange(0, 32, layout=gl.SliceLayout(1, FL))
            fj = gl.arange(0, 32, layout=gl.SliceLayout(0, FL))
            fv = (fi[:, None] * 32 + fj[None, :]) < n1
            P1 = gl.convert_layout(gl.reshape(s1.slice(0, 1024).load(L1), [32, 32]), FL)
            ck = gl.where(fv, (P1 >> 13) - 32768, KEY_NONE)
            cg = 8191 - (P1 & 8191)
            m1 = _first_k2(ck, cg, fv, K, L, n1, NG)
            # the K-th by (key desc, group asc): its gpk is the smallest taken
            Tp1 = gl.min(gl.min(gl.where(m1, P1, 2147483647), axis=1), axis=0)
            gl.barrier()
            cand = gv & (gpk >= Tp1)
        else:
            cand = _first_k(G, gid, gv, K, L, NG)
        n1 = K
    slot = _slots(cand, gid & 1023, L1)
    s1.scatter(
        gl.reshape(gpk, [NCH_P2 * GROUPS]),
        gl.reshape(slot, [NCH_P2 * GROUPS]),
        axis=0,
    )
    gl.barrier()

    # their order by (key desc, group asc); T is the K-th group max
    sa = gl.arange(0, CAP1, layout=gl.SliceLayout(1, RL))
    sb = gl.arange(0, CAP1, layout=gl.SliceLayout(0, RL))
    A = gl.where(sa < n1, s1.slice(0, CAP1).load(gl.SliceLayout(1, RL)), -1)
    B = gl.where(sb < n1, s1.slice(0, CAP1).load(gl.SliceLayout(0, RL)), -1)
    before = (B[None, :] > A[:, None]) | (
        (B[None, :] == A[:, None]) & (sb[None, :] < sa[:, None])
    )
    grank = gl.sum(before.to(gl.int32), axis=1)
    Tp = gl.max(gl.where(grank == K - 1, A, -1), axis=0)
    T = (Tp >> 13) - 32768
    s2 = gl.allocate_shared_memory(gl.int32, [CAP1], SH)
    s2.scatter(A, grank, axis=0)
    gl.barrier()

    # the columns >= T of the K first groups
    kk = gl.arange(0, KB, layout=gl.SliceLayout(1, CT))
    ln = gl.arange(0, 32, layout=gl.SliceLayout(0, CT))
    sg = 8191 - (s2.slice(0, KB).load(gl.SliceLayout(1, CT)) & 8191)
    ccols = sg[:, None] * 32 + ln[None, :]
    cv = (kk[:, None] < K) & (ccols < V)
    xc = gl.load(logits_ptr + src * stride_row + ccols, mask=cv, other=0.0)
    if PENALTY:
        xc = _penalize(
            xc,
            ccols,
            cv,
            src,
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
            MAX_EXTRA,
            ANCHOR,
        )
    key = gl.where(cv, _ordered_key(xc.to(gl.int16, bitcast=True)), KEY_NONE)
    cand2 = cv & (key >= T)
    n2 = _count(cand2)
    if n2 > CAP2:
        if FAST:
            cand2 = _first_k2(key, ccols, cv, K, T, n2, V)
        else:
            cand2 = _first_k(key, ccols, cv, K, T, V)
        n2 = K
    slot2 = _slots(cand2, (kk[:, None] * 32 + ln[None, :]) & 1023, L1)
    s3k = gl.allocate_shared_memory(gl.int32, [2048], SH)
    s3c = gl.allocate_shared_memory(gl.int32, [2048], SH)
    s3k.scatter(gl.reshape(key, [KB * 32]), gl.reshape(slot2, [KB * 32]), axis=0)
    s3c.scatter(gl.reshape(ccols, [KB * 32]), gl.reshape(slot2, [KB * 32]), axis=0)
    gl.barrier()

    # the K first columns by (key desc, column asc), in that order
    sa2 = gl.arange(0, CAP2, layout=gl.SliceLayout(1, RL))
    sb2 = gl.arange(0, CAP2, layout=gl.SliceLayout(0, RL))
    Ak = gl.where(sa2 < n2, s3k.slice(0, CAP2).load(gl.SliceLayout(1, RL)), KEY_NONE)
    Ac = gl.where(sa2 < n2, s3c.slice(0, CAP2).load(gl.SliceLayout(1, RL)), V + sa2)
    Bk = gl.where(sb2 < n2, s3k.slice(0, CAP2).load(gl.SliceLayout(0, RL)), KEY_NONE)
    Bc = gl.where(sb2 < n2, s3c.slice(0, CAP2).load(gl.SliceLayout(0, RL)), V + sb2)
    before2 = (Bk[None, :] > Ak[:, None]) | (
        (Bk[None, :] == Ak[:, None]) & (Bc[None, :] < Ac[:, None])
    )
    rank2 = gl.sum(before2.to(gl.int32), axis=1)
    s4k = gl.allocate_shared_memory(gl.int32, [CAP2], SH)
    s4c = gl.allocate_shared_memory(gl.int32, [CAP2], SH)
    s4k.scatter(Ak, rank2, axis=0)
    s4c.scatter(Ac, rank2, axis=0)
    gl.barrier()

    # The top-k-first sampler (aiter.ops.sampling): softmax probabilities of
    # the k columns, renormalised, top-p threshold p / (their mass).
    offs = gl.arange(0, KB, layout=L1)
    kmask = offs < K
    col = s4c.slice(0, KB).load(L1)
    xk = _key_value(s4k.slice(0, KB).load(L1))
    e = gl.exp(xk / t - M)
    pk = gl.where(kmask, e * (1.0 / Z), 0.0)
    # clamps that keep a NaN, as torch's clamp does
    denom = gl.maximum(gl.sum(pk, axis=0), 1e-8, propagate_nan=tl.PropagateNan.ALL)
    p = gl.where(kmask, pk / denom, 0.0)
    if TOP_P_PER_REQ:
        tp = gl.load(top_p_ptr + req).to(gl.float32)
    else:
        tp = top_p_val
    top_p = gl.minimum(tp / denom, 1.0, propagate_nan=tl.PropagateNan.ALL)
    if RNG_STATE:
        # _draft_uniform_kernel's draw: tl.rand(seed, state + r), the state
        # already advanced by the rows (the stats kernel)
        st = gl.load(rng_state_ptr) - gl.num_programs(0)
        u = _rand_at(seed_lo, seed_hi, st + r, offs)
    elif PHILOX:
        u = _philox_uniform(r, seed_lo, seed_hi, off_lo, off_hi, offs)
    else:
        u = gl.load(u_ptr + r)
    # top_p_sampling_small_k
    pi = gl.convert_layout(p, gl.SliceLayout(1, FL))
    pj = gl.convert_layout(p, gl.SliceLayout(0, FL))
    gt = gl.convert_layout(
        gl.sum(gl.where(pj[None, :] > pi[:, None], pj[None, :], 0.0), axis=1), L1
    )
    keep = (gt < top_p) & kmask
    q = gl.where(keep, p, 0.0)
    cdf = gl.cumsum(q, axis=0)
    qsum = gl.sum(q, axis=0)
    hit = keep & (cdf > u * qsum)
    idx = gl.min(gl.where(hit, offs, KB), axis=0)
    lastk = gl.max(gl.where(keep, offs, -1), axis=0)
    idx = gl.where(idx == KB, lastk, idx)
    idx = gl.where(idx < 0, 0, idx)
    # a greedy row (t at the clamp floor) takes the argmax: the first column
    idx = gl.where(t <= eps, 0, idx)
    token = gl.sum(gl.where(offs == idx, col, 0), axis=0)
    # A row whose softmax is NaN (a NaN or +inf logit, or every logit -inf):
    # the argmax of the NaN probabilities is column 0.
    nan_row = (Z != Z) | (M == float("-inf"))  # noqa: PLR0124 (isnan)
    token = gl.where((t <= eps) & nan_row, 0, token)
    gl.store(out_ptr + r, token)
    if OUT2:
        gl.store(out2_ptr + r, token)
    if REC_CTX > 0:
        # DraftPresencePenalty.record: the row's draft into its context at
        # 1 + pos, then pos + 1 (row r is request r)
        rpos = gl.load(rec_pos_ptr + r)
        if rpos < REC_CTX - 1:
            gl.store(rec_ctx_ptr + r * REC_CTX + 1 + rpos, token)
        gl.store(rec_pos_ptr + r, rpos + 1)
    if PENALTY and ANCHOR:
        # the query's first token joins the request's marked tokens
        s = gl.load(pen_row_seq_ptr + src)
        if gl.load(pen_row_pos_ptr + src) == 0:
            tok = gl.load(pen_ids_ptr + gl.load(pen_seq_start_ptr + s))
            mslot = gl.load(pen_slot_ptr + s).to(gl.int64)
            ok = (tok >= 0) & (tok < pen_mask_width)
            gl.store(
                pen_mask_ptr + mslot * pen_mask_stride + tok + lane * 0,
                gl.full([32 * NW], 1, gl.uint8, L1),
                mask=(lane == 0) & ok,
            )
    if DEBUG:
        dbg = dbg_ptr + r * (3 * KB + 2)
        gl.store(dbg + offs, col, mask=kmask)
        gl.store(dbg + KB + offs, p.to(gl.int32, bitcast=True), mask=kmask)
        gl.store(dbg + 2 * KB + offs, pk.to(gl.int32, bitcast=True), mask=kmask)
        gl.store(dbg + 3 * KB, top_p.to(gl.int32, bitcast=True))
        gl.store(dbg + 3 * KB + 1, Z.to(gl.int32, bitcast=True))
