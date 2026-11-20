# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Gluon kernels of the batch-invariant KDA decode (gfx950): one new row per request,
computed with kda_bi's per-chunk helpers. Row p of a chunk depends only on rows 0..p, so
every per-row result of the rows before p is final and kept in the slot cache; a step
computes row p only (16-row MFMA tiles where kda_bi uses MFMA)."""

from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.runtime.jit import constexpr_function

from aiter.ops.triton._gluon_kernels.gfx950.kimi_delta_attn.batch_invariant.kda_bi import (
    BC,
    RCP_LN2,
    _a_block,
    _a_rows,
    _f32_dot,
    _gate_cumsum,
    _inverses,
    _l2norm,
    _load_rows,
    _load_rows_t,
    _solve,
    _state_ptrs,
    _state_step,
    _store_rows,
    _store_vn,
    _u_rhs,
    _vnew,
    _w_rhs,
    _w_rhs_block,
    a_cols,
    blk_layout,
    col_major_layout,
    f32_mfma_layout,
    key_t_layout,
    mfma_layout,
    row_major_layout,
    tile_layout,
)
from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr

_scatter_repr = make_kernel_repr("_kda_bi_decode_scatter_kernel", ["D", "BT"])
_intra_repr = make_kernel_repr("_kda_bi_decode_intra_kernel", ["BT"])
_out_repr = make_kernel_repr("_kda_bi_decode_out_kernel", ["BV", "BT"])
_step_repr = make_kernel_repr("_kda_bi_decode_step_kernel", ["BV", "BT"])
_fill_repr = make_kernel_repr("_kda_bi_cache_fill_kernel", ["D", "BT"])
_derive_repr = make_kernel_repr("_kda_bi_cache_derive_kernel", ["BT"])
_fill_vn_repr = make_kernel_repr("_kda_bi_cache_vnew_kernel", ["BV", "BT"])


@constexpr_function
def _row_layout(num_warps):
    """[16, D] rows: 8 columns per thread over 16 lanes, as tile_layout, so a row sum has the
    same order as in kda_bi's [64, D] tile."""
    return gl.BlockedLayout([1, 8], [4, 16], [num_warps, 1], [1, 0])


@gluon.jit
def _gate(x, a_log, lower_bound):
    """kda_bi's gate (elementwise): lower / (1 + exp(-exp(A_log) x))."""
    return lower_bound / (1.0 + gl.exp(-(gl.exp(a_log) * x)))


@gluon.jit
def _scatter_row(Q, K, V, G, BETA, A_LOG, DT_BIAS, sq_r, sq_h, sk_r, sk_h, sv_r, sv_h,
                 sg_r, sg_h, sb_r, sb_h, CQ, CK, CV, CG, CB, CGC, CQG, lower_bound, H, r, h,
                 s, p, lr, D: gl.constexpr, BT: gl.constexpr, num_warps: gl.constexpr,
                 WITH_G: gl.constexpr = False):  # fmt: skip
    """Cache row p of (slot s, head h) <- the new row of request r: q^, k^ (bf16 L2-normed),
    v, the gate (fp32, before the cumsum) and beta, computed as kda_bi computes them. With
    WITH_G, also G_p and qg_p (as _intra_row computes them). The row is row lr of a [16, D]
    tile (other rows 0); returns that tile of qg_p (WITH_G) or q^."""
    L: gl.constexpr = _row_layout(num_warps)
    rows = gl.arange(0, 16, layout=gl.SliceLayout(1, L))
    cols = gl.arange(0, D, layout=gl.SliceLayout(0, L))
    m = gl.expand_dims(rows == lr, 1) & gl.expand_dims(cols < D, 0)
    c2 = gl.expand_dims(cols, 0) + gl.expand_dims(rows * 0, 1)
    dst = ((s * H + h) * BT + p) * D + c2
    qn = _l2norm(
        gl.load(Q + r * sq_r + h * sq_h + c2, mask=m, other=0.0).to(gl.float32)
    )
    kn = _l2norm(
        gl.load(K + r * sk_r + h * sk_h + c2, mask=m, other=0.0).to(gl.float32)
    )
    x = gl.load(G + r * sg_r + h * sg_h + c2, mask=m, other=0.0).to(gl.float32)
    x = x + gl.expand_dims(gl.load(DT_BIAS + h * D + cols).to(gl.float32), 0)
    gate = _gate(x, gl.load(A_LOG + h).to(gl.float32), lower_bound)
    gl.store(CQ + dst, qn, mask=m)
    gl.store(CK + dst, kn, mask=m)
    gl.store(CG + dst, gate, mask=m)
    res = qn
    if WITH_G:
        # G_p = G_(p-1) + gate_p / ln 2; G_(p-1) is final in the cache
        g_p = gate * RCP_LN2
        if p > 0:
            g_p = gl.load(CGC + dst - D, mask=m, other=0.0) + g_p
        gl.store(CGC + dst, g_p, mask=m)
        qg = qn.to(gl.float32) * gl.exp2(g_p)
        gl.store(CQG + dst, qg.to(gl.bfloat16), mask=m)
        res = gl.where(m, qg.to(gl.bfloat16), 0.0)
    gl.store(CV + dst, gl.load(V + r * sv_r + h * sv_h + c2, mask=m), mask=m)
    gl.store(
        CB + (s * H + h) * BT + p, gl.load(BETA + r * sb_r + h * sb_h).to(gl.float32)
    )
    return res


@gluon.jit(repr=_scatter_repr)
def _kda_bi_decode_scatter_kernel(
    Q, K, V, G, BETA, A_LOG, DT_BIAS,
    sq_r, sq_h, sk_r, sk_h, sv_r, sv_h, sg_r, sg_h, sb_r, sb_h,
    SLOTS, POS, PCUR, CQ, CK, CV, CG, CB, lower_bound, H,
    D: gl.constexpr, BT: gl.constexpr, num_warps: gl.constexpr,
):  # fmt: skip
    """Program (r, h): _scatter_row at p = pos[slot, h]; records p in PCUR[r, h]."""
    r = gl.program_id(0).to(gl.int64)
    h = gl.program_id(1).to(gl.int64)
    s = gl.load(SLOTS + r).to(gl.int64)
    p32 = gl.load(POS + s * H + h)
    gl.store(PCUR + r * H + h, p32)
    _scatter_row(Q, K, V, G, BETA, A_LOG, DT_BIAS, sq_r, sq_h, sk_r, sk_h, sv_r, sv_h,
                 sg_r, sg_h, sb_r, sb_h, CQ, CK, CV, CG, CB, CQ, CQ, lower_bound, H, r, h,
                 s, p32.to(gl.int64), 0, D, BT, num_warps)  # fmt: skip


@gluon.jit
def _prelude(Q, K, G, sq, sk, sg, DT_BIAS, a_log, lower_bound, n, D: gl.constexpr,
             BT: gl.constexpr, num_warps: gl.constexpr):  # fmt: skip
    """kda_bi _intra_tile's first stage on raw rows [0, n) (row strides sq, sk, sg): bf16
    q^, k^, the gate and its in-chunk cumsum G in log2 units; rows >= n are zero."""
    L: gl.constexpr = tile_layout(num_warps)
    rows = gl.arange(0, BT, layout=gl.SliceLayout(1, L))
    cols = gl.arange(0, D, layout=gl.SliceLayout(0, L))
    m = gl.expand_dims(rows < n, 1) & gl.expand_dims(cols < D, 0)
    r64 = gl.expand_dims(rows.to(gl.int64), 1)
    c2 = gl.expand_dims(cols, 0)
    qn = _l2norm(gl.load(Q + r64 * sq + c2, mask=m, other=0.0).to(gl.float32))
    kn = _l2norm(gl.load(K + r64 * sk + c2, mask=m, other=0.0).to(gl.float32))
    x = gl.load(G + r64 * sg + c2, mask=m, other=0.0).to(gl.float32)
    x = x + gl.expand_dims(gl.load(DT_BIAS + cols).to(gl.float32), 0)
    gate = gl.where(m, _gate(x, a_log, lower_bound), 0.0)
    gcum = _gate_cumsum(gate, num_warps)
    return qn, kn, gate, gcum, rows, cols, r64 * D + c2, m


@gluon.jit
def _load_row(P, stride, p, NR: gl.constexpr, NC: gl.constexpr, LAY: gl.constexpr):
    """[NR, NC] tile holding only row p of P (other rows 0)."""
    r = gl.arange(0, NR, layout=gl.SliceLayout(1, LAY))
    c = gl.arange(0, NC, layout=gl.SliceLayout(0, LAY))
    off = gl.expand_dims(r.to(gl.int64) * stride, 1) + gl.expand_dims(c, 0)
    m = gl.expand_dims(r == p, 1) & gl.expand_dims(c < NC, 0)
    return gl.load(P + off, mask=m, other=0.0)


@gluon.jit
def _a_block_cached(CQ, CK, CGC, CB, s_l, AQK, n, scale, i: gl.constexpr,
                    D: gl.constexpr, BT: gl.constexpr, num_warps: gl.constexpr):  # fmt: skip
    """kda_bi's _a_block for row block i, with its operands read from the slot's row caches
    (q^, k^, G of rows < n) instead of the shared 64-row tiles. Rows >= n get the values of
    prefill's tile (q^ = k^ = 0, G = G[n - 1]). Stores the Aqk row block and puts the Akk
    block row in s_l ([16, 64]); returns k^, G and beta of block i for the w right-hand side.
    """
    MF: gl.constexpr = f32_mfma_layout(num_warps)
    KT: gl.constexpr = key_t_layout(num_warps)
    r = gl.arange(0, BC, layout=gl.SliceLayout(1, MF)) + i * BC
    cm = gl.arange(0, D, layout=gl.SliceLayout(0, MF))
    off_i = gl.expand_dims(r.to(gl.int64), 1) * D + gl.expand_dims(cm, 0)
    m_i = gl.expand_dims(r < n, 1) & gl.expand_dims(cm < D, 0)
    g_last = gl.expand_dims(gl.load(CGC + (n - 1).to(gl.int64) * D + cm), 0)
    g_i = gl.where(m_i, gl.load(CGC + off_i, mask=m_i, other=0.0), g_last)
    k_i = gl.load(CK + off_i, mask=m_i, other=0.0)
    beta_i = gl.load(CB + r.to(gl.int64), mask=r < n, other=0.0).to(gl.float32)
    ch = gl.arange(0, D, layout=gl.SliceLayout(1, KT))
    NC: gl.constexpr = a_cols(i)
    col = gl.arange(0, NC, layout=gl.SliceLayout(0, KT))
    off_t = gl.expand_dims(col.to(gl.int64), 0) * D + gl.expand_dims(ch, 1)
    m_t = gl.expand_dims(col < n, 0) & gl.expand_dims(ch < D, 1)
    g_last_t = gl.expand_dims(gl.load(CGC + (n - 1).to(gl.int64) * D + ch), 1)
    aqk, akk = _a_block(
        gl.load(CQ + off_i, mask=m_i, other=0.0),
        k_i,
        g_i,
        gl.expand_dims(gl.load(CGC + i * BC * D + cm), 0),
        gl.load(CK + off_t, mask=m_t, other=0.0),
        gl.where(m_t, gl.load(CGC + off_t, mask=m_t, other=0.0), g_last_t),
        gl.expand_dims(gl.load(CGC + i * BC * D + ch), 1),
        beta_i,
        scale,
        i,
        num_warps,
    )
    r2 = gl.expand_dims(r, 1)
    c = gl.expand_dims(gl.arange(0, NC, layout=gl.SliceLayout(0, MF)), 0)
    row = AQK + r2.to(gl.int64) * BT + c
    m = (r2 == n - 1) & (c < NC)  # rows < p are already in the cache
    gl.store(row, aqk.to(gl.bfloat16), mask=m)
    for j in gl.static_range(NC, BT, NC):  # the columns past NC are 0, as in _a_rows
        gl.store(row + j, gl.zeros([BC, NC], gl.bfloat16, MF), mask=m)
    s_l.slice(0, NC, 1).store(akk)
    return k_i, g_i, beta_i, gl.where(m, aqk.to(gl.bfloat16), 0.0)


@gluon.jit
def _inv_step(akk, ai, i, num_warps: gl.constexpr):
    """One row iteration of kda_bi's _inv_block (row i of the forward substitution, same ops
    and layout): ai holds the finished rows < i and -akk below."""
    PB: gl.constexpr = blk_layout(num_warps)
    r = gl.expand_dims(gl.arange(0, BC, layout=gl.SliceLayout(1, PB)), 1)
    a = -gl.sum(gl.where(r == i, akk, 0.0), axis=0)
    a_c = gl.convert_layout(a, gl.SliceLayout(1, PB))
    a = a + gl.sum(gl.expand_dims(a_c, 1) * ai, axis=0)
    return gl.where(r == i, gl.expand_dims(a, 0), ai)


@gluon.jit
def _ai_offs(num_warps: gl.constexpr):
    PB: gl.constexpr = blk_layout(num_warps)
    r = gl.expand_dims(gl.arange(0, BC, layout=gl.SliceLayout(1, PB)), 1)
    c = gl.expand_dims(gl.arange(0, BC, layout=gl.SliceLayout(0, PB)), 0)
    return r, c, r.to(gl.int64) * BC + c


@gluon.jit
def _x_offs(i: gl.constexpr, D: gl.constexpr, num_warps: gl.constexpr):
    MF: gl.constexpr = f32_mfma_layout(num_warps)
    r = gl.arange(0, BC, layout=gl.SliceLayout(1, MF)) + i * BC
    c = gl.arange(0, D, layout=gl.SliceLayout(0, MF))
    return r, gl.expand_dims(r.to(gl.int64), 1) * D + gl.expand_dims(c, 0), c


@gluon.jit
def _load_x(P, i: gl.constexpr, D: gl.constexpr, num_warps: gl.constexpr,
            OP: gl.constexpr = True):  # fmt: skip
    """fp32 solve rows of block i ([16, D]): the B operand layout of _f32_dot (OP) or the
    f32 MFMA layout (coalesced)."""
    MF: gl.constexpr = f32_mfma_layout(num_warps)
    B: gl.constexpr = gl.DotOperandLayout(1, MF, 1) if OP else MF
    r = gl.arange(0, BC, layout=gl.SliceLayout(1, B)) + i * BC
    c = gl.arange(0, D, layout=gl.SliceLayout(0, B))
    return gl.load(P + gl.expand_dims(r.to(gl.int64), 1) * D + gl.expand_dims(c, 0))


@gluon.jit
def _store_row_p(P, x, i: gl.constexpr, n, D: gl.constexpr, num_warps: gl.constexpr):
    """Row n - 1 of a [16, D] row block i (f32 MFMA layout)."""
    r, off, c = _x_offs(i, D, num_warps)
    gl.store(P + off, x, mask=gl.expand_dims(r == n - 1, 1) & gl.expand_dims(c < D, 0))


@gluon.jit
def _store_x(P, x, i: gl.constexpr, n, D: gl.constexpr, num_warps: gl.constexpr):
    r, off, c = _x_offs(i, D, num_warps)
    gl.store(P + off, x, mask=gl.expand_dims(r < n, 1) & gl.expand_dims(c < D, 0))


@gluon.jit
def _intra_row(CQ, CK, CV, CG, CB, CQG, CW, CU, CXW, CXU, CAQK, CGC, CAI, KG, GLAST, scale, H,
               r, h, s, n, D: gl.constexpr, BT: gl.constexpr, num_warps: gl.constexpr,
               HAVE_G: gl.constexpr = False, s_a=None):  # fmt: skip
    """Request r, head h: row block i = pos // 16 of kda_bi's intra pass on the slot's open
    chunk, rows [0, pos + 1). Blocks j < i come from the slot's fp32 solve rows (final by
    row closure); the order is _solve's: a = r_i, a += -L_ij x_j for j ascending, x_i =
    T_ii a. When the row closes the chunk, kg and G_last go to KG / GLAST. Returns the bf16
    w and u rows ([16, D], row p only); with s_a, also puts the Aqk row ([16, 64]) in s_a.
    """
    c = (s * H + h) * BT
    L: gl.constexpr = tile_layout(num_warps)
    cols = gl.arange(0, D, layout=gl.SliceLayout(0, L))
    p64 = (n - 1).to(gl.int64)
    if HAVE_G:
        g_p = gl.load(CGC + c * D + p64 * D + cols)
    else:
        # G row p by kda_bi's row-order cumsum: G_p = G_(p-1) + gate_p / ln 2
        g_p = gl.load(CG + c * D + p64 * D + cols) * RCP_LN2
        if n > 1:
            g_p = gl.load(CGC + c * D + (p64 - 1) * D + cols) + g_p
        gl.store(CGC + c * D + p64 * D + cols, g_p)
        q_p = gl.load(CQ + c * D + p64 * D + cols).to(gl.float32)
        gl.store(CQG + c * D + p64 * D + cols, (q_p * gl.exp2(g_p)).to(gl.bfloat16))
    if n == BT:
        rows = gl.arange(0, BT, layout=gl.SliceLayout(1, L))
        off = gl.expand_dims(rows.to(gl.int64), 1) * D + gl.expand_dims(cols, 0)
        gl.store(GLAST + (r * H + h) * D + cols, g_p)
        gl.barrier()  # G row p is read back in the tile
        gcum = gl.load(CGC + c * D + off)
        kn = gl.load(CK + c * D + off)
        kg = kn.to(gl.float32) * gl.exp2(gl.expand_dims(g_p, 0) - gcum)
        gl.store(KG + (r * H + h) * BT * D + off, kg.to(gl.bfloat16))
    if not HAVE_G:
        gl.barrier()  # G row p is read back below

    SH: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, order=[1, 0])
    PB: gl.constexpr = blk_layout(num_warps)
    s_l = gl.allocate_shared_memory(gl.float32, [BC, BT], SH)
    MFL: gl.constexpr = f32_mfma_layout(num_warps)
    z = gl.zeros([BC, D], gl.float32, MFL)
    w16 = gl.zeros([BC, D], gl.bfloat16, MFL)
    u16 = gl.zeros([BC, D], gl.bfloat16, MFL)
    blk = (n - 1) // BC
    for i in gl.static_range(BT // BC):
        if blk == i:
            # solve rows of the earlier blocks (final); the step kernel reads them first
            if (i > 0) and (s_a is not None):
                xw0 = _load_x(CXW + c * D, 0, D, num_warps)
                xu0 = _load_x(CXU + c * D, 0, D, num_warps)
            if (i > 1) and (s_a is not None):
                xw1 = _load_x(CXW + c * D, 1, D, num_warps)
                xu1 = _load_x(CXU + c * D, 1, D, num_warps)
            if (i > 2) and (s_a is not None):
                xw2 = _load_x(CXW + c * D, 2, D, num_warps)
                xu2 = _load_x(CXU + c * D, 2, D, num_warps)
            k_i, g_i, beta_i, a16 = _a_block_cached(
                CQ + c * D, CK + c * D, CGC + c * D, CB + c, s_l, CAQK + c * BT, n,
                scale, i, D, BT, num_warps,
            )  # fmt: skip
            # (I + L_ii)^-1 rows < p are final: cached (before the + I); only row p runs
            akk = s_l.slice(i * BC, BC, 1).load(PB)
            pl = n - 1 - i * BC
            ar, ac, aoff = _ai_offs(num_warps)
            cai = CAI + (c + i * BC) * BC + aoff
            ai = gl.where(ar < pl, gl.load(cai, mask=ar < pl, other=0.0), -akk)
            if pl >= 2:
                ai = _inv_step(akk, ai, pl, num_warps)
            gl.store(cai, ai, mask=(ar == pl) & (ac < BC))
            t = ai + gl.where(ar == ac, 1.0, 0.0)
            aw = _w_rhs_block(k_i, g_i, gl.expand_dims(beta_i, 1))
            au = _u_rhs(CV + c * D, D, CB + c, 1, i, n, num_warps)
            LA: gl.constexpr = gl.DotOperandLayout(0, MFL, 1)
            if s_a is None:
                if i > 0:
                    xw0 = _load_x(CXW + c * D, 0, D, num_warps, False)
                    xu0 = _load_x(CXU + c * D, 0, D, num_warps, False)
                if i > 1:
                    xw1 = _load_x(CXW + c * D, 1, D, num_warps, False)
                    xu1 = _load_x(CXU + c * D, 1, D, num_warps, False)
                if i > 2:
                    xw2 = _load_x(CXW + c * D, 2, D, num_warps, False)
                    xu2 = _load_x(CXU + c * D, 2, D, num_warps, False)
            if i > 0:
                lij = -s_l.slice(0, BC, 1).load(LA)
                aw = _f32_dot(lij, xw0, aw, num_warps)
                au = _f32_dot(lij, xu0, au, num_warps)
            if i > 1:
                lij = -s_l.slice(BC, BC, 1).load(LA)
                aw = _f32_dot(lij, xw1, aw, num_warps)
                au = _f32_dot(lij, xu1, au, num_warps)
            if i > 2:
                lij = -s_l.slice(2 * BC, BC, 1).load(LA)
                aw = _f32_dot(lij, xw2, aw, num_warps)
                au = _f32_dot(lij, xu2, au, num_warps)
            xw = _f32_dot(t, aw, z, num_warps)
            xu = _f32_dot(t, au, z, num_warps)
            # rows < p are already in the cache (final); store row p only
            _store_row_p(CW + c * D, xw.to(gl.bfloat16), i, n, D, num_warps)
            _store_row_p(CU + c * D, xu.to(gl.bfloat16), i, n, D, num_warps)
            _store_row_p(CXW + c * D, xw, i, n, D, num_warps)
            _store_row_p(CXU + c * D, xu, i, n, D, num_warps)
            sel = gl.expand_dims(gl.arange(0, BC, layout=gl.SliceLayout(1, MFL)) + i * BC == n - 1, 1)  # fmt: skip
            w16 = gl.where(sel, xw.to(gl.bfloat16), 0.0)
            u16 = gl.where(sel, xu.to(gl.bfloat16), 0.0)
            if s_a is not None:
                NC: gl.constexpr = a_cols(i)
                s_a.slice(0, NC, 1).store(a16)
                for j in gl.static_range(NC, BT, NC):  # the columns past NC are 0
                    s_a.slice(j, NC, 1).store(gl.zeros([BC, NC], gl.bfloat16, MFL))
    return w16, u16


@gluon.jit(repr=_intra_repr)
def _kda_bi_decode_intra_kernel(
    CQ, CK, CV, CG, CB, SLOTS, POS, PCUR,
    CQG, CW, CU, CXW, CXU, CAQK, CGC, CAI, KG, GLAST, scale, H,
    D: gl.constexpr, BT: gl.constexpr, num_warps: gl.constexpr,
):  # fmt: skip
    """Program (r, h): _intra_row for p = PCUR[r, h]; advances pos[slot, h]."""
    r = gl.program_id(0).to(gl.int64)
    h = gl.program_id(1).to(gl.int64)
    s = gl.load(SLOTS + r).to(gl.int64)
    n = gl.load(PCUR + r * H + h) + 1
    gl.store(POS + s * H + h, n % BT)  # pos is read only by the scatter kernel
    _intra_row(CQ, CK, CV, CG, CB, CQG, CW, CU, CXW, CXU, CAQK, CGC, CAI, KG, GLAST, scale, H,
               r, h, s, n, D, BT, num_warps)  # fmt: skip


@gluon.jit(repr=_derive_repr)
def _kda_bi_cache_derive_kernel(
    Q,
    K,
    G,
    sq,
    sk,
    sg,
    sqh,
    skh,
    sgh,
    CQ,
    CK,
    CG,
    CV,
    CB,
    A_LOG,
    DT_BIAS,
    CU_SEQ,
    SLOTS,
    CQG,
    CW,
    CU,
    CXW,
    CXU,
    CAQK,
    CGC,
    CAI,
    lower_bound,
    scale,
    H,
    D: gl.constexpr,
    BT: gl.constexpr,
    num_warps: gl.constexpr,
):
    """Program (n, h): kda_bi's full intra pass on the open chunk of prompt n (its last
    T mod BT rows, read from the prompt q, k, g; v and beta already in the cache). Keeps
    q^, k^, the gate, qg, w, u, Aqk and the fp32 solve rows in the slot cache."""
    seq = gl.program_id(0)
    h = gl.program_id(1).to(gl.int64)
    t1 = gl.load(CU_SEQ + seq + 1).to(gl.int64)
    n = ((t1 - gl.load(CU_SEQ + seq).to(gl.int64)) % BT).to(gl.int32)
    t0 = t1 - n
    s = gl.load(SLOTS + seq).to(gl.int64)
    c = (s * H + h) * BT
    qn, kn, gate, gcum, _rows, _cols, off, m = _prelude(
        Q + t0 * sq + h * sqh, K + t0 * sk + h * skh, G + t0 * sg + h * sgh, sq, sk, sg,
        DT_BIAS + h * D, gl.load(A_LOG + h).to(gl.float32), lower_bound, n, D, BT,
        num_warps,
    )  # fmt: skip
    gl.store(CQ + c * D + off, qn, mask=m)
    gl.store(CK + c * D + off, kn, mask=m)
    gl.store(CG + c * D + off, gate, mask=m)
    gl.store(CGC + c * D + off, gcum, mask=m)
    gl.store(
        CQG + c * D + off, (qn.to(gl.float32) * gl.exp2(gcum)).to(gl.bfloat16), mask=m
    )
    SH: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, order=[1, 0])
    s_q = gl.allocate_shared_memory(gl.bfloat16, [BT, D], SH, qn)
    s_k = gl.allocate_shared_memory(gl.bfloat16, [BT, D], SH, kn)
    s_g = gl.allocate_shared_memory(gl.float32, [BT, D], SH, gcum)
    s_l = gl.allocate_shared_memory(gl.float32, [BT, BT], SH)
    for i in gl.static_range(BT // BC):
        _a_rows(
            s_q, s_k, s_g, s_l, CB + c, 1, CAQK + c * BT, BT, n, scale, i, num_warps
        )
    t0, t1, t2, t3 = _inverses(s_l, num_warps)
    x0, x1, x2, x3 = _solve(
        s_l, t0, t1, t2, t3,
        _w_rhs(s_k, s_g, CB + c, 1, 0, n, num_warps),
        _w_rhs(s_k, s_g, CB + c, 1, 1, n, num_warps),
        _w_rhs(s_k, s_g, CB + c, 1, 2, n, num_warps),
        _w_rhs(s_k, s_g, CB + c, 1, 3, n, num_warps),
        num_warps,
    )  # fmt: skip
    _store_rows(CW + c * D, D, x0, 0, n, num_warps)
    _store_rows(CW + c * D, D, x1, 1, n, num_warps)
    _store_rows(CW + c * D, D, x2, 2, n, num_warps)
    _store_rows(CW + c * D, D, x3, 3, n, num_warps)
    _store_x(CXW + c * D, x0, 0, n, D, num_warps)
    _store_x(CXW + c * D, x1, 1, n, D, num_warps)
    _store_x(CXW + c * D, x2, 2, n, D, num_warps)
    _store_x(CXW + c * D, x3, 3, n, D, num_warps)
    v = CV + c * D
    x0, x1, x2, x3 = _solve(
        s_l, t0, t1, t2, t3,
        _u_rhs(v, D, CB + c, 1, 0, n, num_warps),
        _u_rhs(v, D, CB + c, 1, 1, n, num_warps),
        _u_rhs(v, D, CB + c, 1, 2, n, num_warps),
        _u_rhs(v, D, CB + c, 1, 3, n, num_warps),
        num_warps,
    )  # fmt: skip
    _store_rows(CU + c * D, D, x0, 0, n, num_warps)
    _store_rows(CU + c * D, D, x1, 1, n, num_warps)
    _store_rows(CU + c * D, D, x2, 2, n, num_warps)
    _store_rows(CU + c * D, D, x3, 3, n, num_warps)
    _store_x(CXU + c * D, x0, 0, n, D, num_warps)
    _store_x(CXU + c * D, x1, 1, n, D, num_warps)
    _store_x(CXU + c * D, x2, 2, n, D, num_warps)
    _store_x(CXU + c * D, x3, 3, n, D, num_warps)
    # the open block's inverse rows (before the + I) for the decode steps
    PB: gl.constexpr = blk_layout(num_warps)
    ar, ac, aoff = _ai_offs(num_warps)
    for i in gl.static_range(BT // BC):
        if n // BC == i:
            akk = s_l.slice(i * BC, BC).slice(i * BC, BC, 1).load(PB)
            ai = -akk
            for j in gl.static_range(2, BC):
                ai = _inv_step(akk, ai, j, num_warps)
            gl.store(
                CAI + (c + i * BC) * BC + aoff, ai, mask=(ar + i * BC < n) & (ac < BC)
            )


@constexpr_function
def _row_mfma_layout(num_warps):
    """kda_bi's bf16 16x16x32 MFMA (mfma_layout) on a 16-row tile: same instruction and
    K order per element, warps along the V columns instead of the rows."""
    return gl.amd.AMDMFMALayout(
        version=4,
        instr_shape=[16, 16, 32],
        transposed=True,
        warps_per_cta=[1, num_warps],
    )


@gluon.jit
def _rows16(P, stride, p, i: gl.constexpr, NC: gl.constexpr, LAY: gl.constexpr):
    """[16, NC] tile of row block i holding only row p of P (other rows 0)."""
    r = gl.arange(0, BC, layout=gl.SliceLayout(1, LAY)) + i * BC
    c = gl.arange(0, NC, layout=gl.SliceLayout(0, LAY))
    off = gl.expand_dims(r.to(gl.int64) * stride, 1) + gl.expand_dims(c, 0)
    m = gl.expand_dims(r == p, 1) & gl.expand_dims(c < NC, 0)
    return gl.load(P + off, mask=m, other=0.0), off, m


@gluon.jit
def _hb_offs(v0, D: gl.constexpr, BV: gl.constexpr, LAY: gl.constexpr):
    """Offsets of V block v0 of the bf16(h_b) cache, which is stored transposed ([V, K]),
    as a [K, BV] tile: a B operand reads 8 consecutive k with one load."""
    kk = gl.arange(0, D, layout=gl.SliceLayout(1, LAY))
    vv = gl.arange(0, BV, layout=gl.SliceLayout(0, LAY))
    return gl.expand_dims((vv + v0).to(gl.int64) * D, 0) + gl.expand_dims(kk, 1)


@gluon.jit
def _load_hb(CHB, STATE, s_slot, s_head, h, s, rs, p, v0, H, D: gl.constexpr,
             BV: gl.constexpr, num_warps: gl.constexpr):  # fmt: skip
    """bf16(h_b) of V block v0 as the B operand of the row MFMA; copied from the state at p == 0."""
    B: gl.constexpr = gl.DotOperandLayout(1, _row_mfma_layout(num_warps), 8)
    chb = CHB + (s * H + h) * D * D + _hb_offs(v0, D, BV, B)
    if p == 0:
        kk = gl.arange(0, D, layout=gl.SliceLayout(1, B))
        vv = gl.arange(0, BV, layout=gl.SliceLayout(0, B))
        st = STATE + rs * s_slot + h * s_head + v0
        hb = gl.load(
            st + gl.expand_dims(kk.to(gl.int64) * D, 1) + gl.expand_dims(vv, 0)
        )
        hb = hb.to(gl.bfloat16)
        gl.store(chb, hb)
    else:
        hb = gl.load(chb)
    return hb


@gluon.jit
def _out_row(CQG, CW, CU, CAQK, CVN, hb_op, KG, GLAST, STATE, s_slot, s_head, O, scale, H,
             r, h, s, rs, n, v0, D: gl.constexpr, BT: gl.constexpr, BV: gl.constexpr,
             num_warps: gl.constexpr):  # fmt: skip
    """Request r, head h, V block v0: kda_bi's state and output passes (the MFMA sequences of
    _vnew and _out_tile) for row p only, on 16-row tiles. v_new of rows < p and bf16(h_b)
    come from the slot cache (h_b is copied at p == 0). When p closes the chunk, the next
    boundary state goes to STATE (_state_step)."""
    MF: gl.constexpr = mfma_layout(num_warps)
    M16: gl.constexpr = _row_mfma_layout(num_warps)
    RM: gl.constexpr = row_major_layout(num_warps)
    p = n - 1
    c = (s * H + h) * BT
    st0 = STATE + rs * s_slot + h * s_head + v0
    z = gl.zeros([BC, BV], gl.float32, M16)
    for i in gl.static_range(BT // BC):
        if p // BC == i:
            w, _, _ = _rows16(CW + c * D, D, p, i, D, RM)
            d = gl.amd.cdna4.mfma(gl.convert_layout(w, gl.DotOperandLayout(0, M16, 8)), hb_op, z)  # fmt: skip
            u, off_u, m_u = _rows16(CU + c * D + v0, D, p, i, BV, M16)
            gl.store(CVN + c * D + v0 + off_u, (u.to(gl.float32) - d).to(gl.bfloat16), mask=m_u)  # fmt: skip
            qg, _, _ = _rows16(CQG + c * D, D, p, i, D, RM)
            acc = gl.amd.cdna4.mfma(gl.convert_layout(qg, gl.DotOperandLayout(0, M16, 8)), hb_op, z)  # fmt: skip
            acc = acc * scale
            a, _, _ = _rows16(CAQK + c * BT, BT, p, i, BT, RM)
            gl.barrier()  # v_new row p is read back below
            vn = _load_rows(CVN + c * D + v0, D, n, BT, BV, RM)
            acc = gl.amd.cdna4.mfma(
                gl.convert_layout(a, gl.DotOperandLayout(0, M16, 8)),
                gl.convert_layout(vn, gl.DotOperandLayout(1, M16, 8)),
                acc,
            )
            rr = gl.arange(0, BC, layout=gl.SliceLayout(1, M16)) + i * BC
            cc = gl.arange(0, BV, layout=gl.SliceLayout(0, M16))
            off = (
                (r * H + h) * D + v0 + gl.expand_dims(rr * 0, 1) + gl.expand_dims(cc, 0)
            )
            gl.store(O + off, acc.to(gl.bfloat16), mask=gl.expand_dims(rr == p, 1) & gl.expand_dims(cc < BV, 0))  # fmt: skip
    if n == BT:
        # _load_chunk's kg^T and G_last, then _state_step on the FP32 state
        st = _state_ptrs(st0, D, BV, num_warps)
        kg = _load_rows_t(KG + (r * H + h) * BT * D, D, n, BT, D, col_major_layout(num_warps))  # fmt: skip
        g_last = gl.load(GLAST + (r * H + h) * D + gl.arange(0, D, layout=gl.SliceLayout(1, MF)))  # fmt: skip
        vn = _load_rows(CVN + c * D + v0, D, n, BT, BV, MF)
        gl.store(st, _state_step(gl.load(st), kg, g_last, vn, num_warps))


@gluon.jit
def _out_row_regs(w16, u16, qg16, s_a, s_vn, CVN, hb_op, KG, GLAST, STATE, s_slot, s_head,
                  O, scale, H, r, h, s, rs, n, D: gl.constexpr, BT: gl.constexpr,
                  BV: gl.constexpr, num_warps: gl.constexpr):  # fmt: skip
    """_out_row (BV == D) with the row-p operands from registers: w16, u16, qg16 ([16, D]
    bf16, row p only), the Aqk row in s_a and v_new rows < p in s_vn ([64, BV] bf16)."""
    MF: gl.constexpr = mfma_layout(num_warps)
    M16: gl.constexpr = _row_mfma_layout(num_warps)
    p = n - 1
    c = (s * H + h) * BT
    z = gl.zeros([BC, BV], gl.float32, M16)
    d = gl.amd.cdna4.mfma(gl.convert_layout(w16, gl.DotOperandLayout(0, M16, 8)), hb_op, z)  # fmt: skip
    vn16 = (gl.convert_layout(u16, M16).to(gl.float32) - d).to(gl.bfloat16)
    rr = gl.arange(0, BC, layout=gl.SliceLayout(1, M16))
    cc = gl.arange(0, BV, layout=gl.SliceLayout(0, M16))
    off = gl.expand_dims(rr * 0, 1) + gl.expand_dims(cc, 0)
    m = gl.expand_dims(rr == p % BC, 1) & gl.expand_dims(cc < BV, 0)
    gl.store(CVN + (c + p) * D + off, vn16, mask=m)
    acc = gl.amd.cdna4.mfma(gl.convert_layout(qg16, gl.DotOperandLayout(0, M16, 8)), hb_op, z)  # fmt: skip
    acc = acc * scale
    for i in gl.static_range(BT // BC):
        if p // BC == i:
            # each thread rewrites the elements it read
            t = s_vn.slice(i * BC, BC).load(M16)
            s_vn.slice(i * BC, BC).store(gl.where(m, vn16, t))
    gl.barrier()  # the v_new tile and the Aqk row are read below
    acc = gl.amd.cdna4.mfma(
        s_a.load(gl.DotOperandLayout(0, M16, 8)),
        s_vn.load(gl.DotOperandLayout(1, M16, 8)),
        acc,
    )
    gl.store(O + (r * H + h) * D + off, acc.to(gl.bfloat16), mask=m)
    if n == BT:
        gl.barrier()  # kg and G_last are read back below
        st = _state_ptrs(STATE + rs * s_slot + h * s_head, D, BV, num_warps)
        kg = _load_rows_t(KG + (r * H + h) * BT * D, D, n, BT, D, col_major_layout(num_warps))  # fmt: skip
        g_last = gl.load(GLAST + (r * H + h) * D + gl.arange(0, D, layout=gl.SliceLayout(1, MF)))  # fmt: skip
        gl.store(st, _state_step(gl.load(st), kg, g_last, s_vn.load(MF), num_warps))


@gluon.jit(repr=_out_repr)
def _kda_bi_decode_out_kernel(
    CQG, CW, CU, CAQK, CVN, CHB, KG, GLAST, SLOTS, RSLOTS, PCUR, STATE, s_slot, s_head,
    O, scale, H, D: gl.constexpr, BT: gl.constexpr, BV: gl.constexpr,
    num_warps: gl.constexpr,
):  # fmt: skip
    """Program (r, h, V block): _out_row for p = PCUR[r, h]."""
    r = gl.program_id(0).to(gl.int64)
    h = gl.program_id(1).to(gl.int64)
    s = gl.load(SLOTS + r).to(gl.int64)
    n = gl.load(PCUR + r * H + h) + 1
    rs = gl.load(RSLOTS + r).to(gl.int64)
    v0 = gl.program_id(2) * BV
    hb_op = _load_hb(
        CHB, STATE, s_slot, s_head, h, s, rs, n - 1, v0, H, D, BV, num_warps
    )
    _out_row(CQG, CW, CU, CAQK, CVN, hb_op, KG, GLAST, STATE, s_slot, s_head, O, scale, H,
             r, h, s, rs, n, v0, D, BT, BV, num_warps)  # fmt: skip


@gluon.jit(repr=_step_repr)
def _kda_bi_decode_step_kernel(
    Q, K, V, G, BETA, A_LOG, DT_BIAS,
    sq_r, sq_h, sk_r, sk_h, sv_r, sv_h, sg_r, sg_h, sb_r, sb_h,
    SLOTS, RSLOTS, POS, CQ, CK, CV, CG, CB,
    CQG, CW, CU, CXW, CXU, CAQK, CGC, CAI, CVN, CHB, KG, GLAST, STATE, s_slot, s_head, O,
    lower_bound, scale, H,
    D: gl.constexpr, BT: gl.constexpr, BV: gl.constexpr, num_warps: gl.constexpr,
):  # fmt: skip
    """Program (r, h): the whole decode step of one head in one launch (_scatter_row with
    G_p, _intra_row, _out_row_regs); the row-p results pass in registers; advances pos.
    """
    r = gl.program_id(0).to(gl.int64)
    h = gl.program_id(1).to(gl.int64)
    s = gl.load(SLOTS + r).to(gl.int64)
    p32 = gl.load(POS + s * H + h)
    rs = gl.load(RSLOTS + r).to(gl.int64)
    gl.static_assert(BV == D)
    hb_op = _load_hb(CHB, STATE, s_slot, s_head, h, s, rs, p32, 0, H, D, BV, num_warps)
    SH: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, order=[1, 0])
    s_vn = gl.allocate_shared_memory(gl.bfloat16, [BT, BV], SH)
    s_a = gl.allocate_shared_memory(gl.bfloat16, [BC, BT], SH)
    c = (s * H + h) * BT
    # v_new rows < p are final: read them before the step's chain
    s_vn.store(_load_rows(CVN + c * D, D, p32, BT, BV, row_major_layout(num_warps)))
    qg16 = _scatter_row(Q, K, V, G, BETA, A_LOG, DT_BIAS, sq_r, sq_h, sk_r, sk_h, sv_r,
                        sv_h, sg_r, sg_h, sb_r, sb_h, CQ, CK, CV, CG, CB, CGC, CQG,
                        lower_bound, H, r, h, s, p32.to(gl.int64), p32 % BC, D, BT,
                        num_warps, True)  # fmt: skip
    gl.barrier()  # the new row is read back below
    n = p32 + 1
    w16, u16 = _intra_row(CQ, CK, CV, CG, CB, CQG, CW, CU, CXW, CXU, CAQK, CGC, CAI, KG,
                          GLAST, scale, H, r, h, s, n, D, BT, num_warps, True, s_a)  # fmt: skip
    _out_row_regs(w16, u16, qg16, s_a, s_vn, CVN, hb_op, KG, GLAST, STATE, s_slot, s_head,
                  O, scale, H, r, h, s, rs, n, D, BT, BV, num_warps)  # fmt: skip
    gl.store(POS + s * H + h, n % BT)


@gluon.jit(repr=_fill_repr)
def _kda_bi_cache_fill_kernel(
    V,
    BETA,
    sv_t,
    sv_h,
    sb_t,
    sb_h,
    CU,
    SLOTS,
    POS,
    CV,
    CB,
    H,
    D: gl.constexpr,
    BT: gl.constexpr,
):
    """v and beta of the open chunk of prompt n (its last T mod BT rows) -> cache rows 0..,
    pos <- T mod BT."""
    L2: gl.constexpr = gl.BlockedLayout([1, 8], [4, 16], [gl.num_warps(), 1], [1, 0])
    n = gl.program_id(0)
    h = gl.program_id(1).to(gl.int64)
    t1 = gl.load(CU + n + 1).to(gl.int64)
    nt = (t1 - gl.load(CU + n).to(gl.int64)) % BT
    t0 = t1 - nt
    s = gl.load(SLOTS + n).to(gl.int64)
    i = gl.arange(0, BT, layout=gl.SliceLayout(1, L2)).to(gl.int64)
    d = gl.arange(0, D, layout=gl.SliceLayout(0, L2))
    m = gl.expand_dims(i < nt, 1) & gl.expand_dims(d >= 0, 0)
    t = gl.expand_dims(t0 + i, 1)
    dst = ((s * H + h) * BT + gl.expand_dims(i, 1)) * D + gl.expand_dims(d, 0)
    gl.store(
        CV + dst,
        gl.load(V + t * sv_t + h * sv_h + gl.expand_dims(d, 0), mask=m),
        mask=m,
    )
    L1: gl.constexpr = gl.BlockedLayout([1], [64], [gl.num_warps()], [0])
    i1 = gl.arange(0, BT, layout=L1).to(gl.int64)
    m1 = i1 < nt
    b = gl.load(BETA + (t0 + i1) * sb_t + h * sb_h, mask=m1)
    gl.store(CB + (s * H + h) * BT + i1, b.to(gl.float32), mask=m1)
    gl.store(POS + s * H + h, nt.to(gl.int32))


@gluon.jit(repr=_fill_vn_repr)
def _kda_bi_cache_vnew_kernel(
    CW,
    CU,
    CVN,
    CHB,
    CU_SEQ,
    SLOTS,
    RSLOTS,
    STATE,
    s_slot,
    s_head,
    H,
    D: gl.constexpr,
    BT: gl.constexpr,
    BV: gl.constexpr,
    num_warps: gl.constexpr,
):
    """Program (n, h, V block): bf16(h_b) and v_new = bf16(u - w bf16(h_b)) (kda_bi's
    _vnew) of a filled open chunk, kept in the slot cache for the decode steps."""
    MF: gl.constexpr = mfma_layout(num_warps)
    seq = gl.program_id(0)
    h = gl.program_id(1).to(gl.int64)
    v0 = gl.program_id(2) * BV
    t1 = gl.load(CU_SEQ + seq + 1).to(gl.int64)
    n = ((t1 - gl.load(CU_SEQ + seq).to(gl.int64)) % BT).to(gl.int32)
    s = gl.load(SLOTS + seq).to(gl.int64)
    c = (s * H + h) * BT
    st = STATE + gl.load(RSLOTS + seq).to(gl.int64) * s_slot + h * s_head + v0
    hb = gl.load(_state_ptrs(st, D, BV, num_warps)).to(gl.bfloat16)
    gl.store(CHB + (s * H + h) * D * D + _hb_offs(v0, D, BV, MF), hb)
    w = _load_rows(CW + c * D, D, n, BT, D, row_major_layout(num_warps))
    u = _load_rows(CU + c * D + v0, D, n, BT, BV, MF)
    _store_vn(CVN + c * D + v0, _vnew(w, u, hb, n, BV, num_warps), n, D, BV, num_warps)
