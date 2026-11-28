# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
# Adapted from flash-linear-attention (MIT): Copyright (c) 2023-2026, Songlin Yang, Yu Zhang,
# Zhiyuan Li. The chunk decomposition (gate cumsum, sub-chunk A blocks with a reference row,
# forward substitution, state and output passes) follows fla.ops.kda.
"""Gluon kernels of the batch-invariant KDA chunk forward (gfx950).

One 64-row chunk tile per program; every reduction has a fixed order and row p of a chunk
depends only on rows 0..p and the chunk's boundary state. The @gluon.jit helpers work on one
chunk tile so a decode kernel can reuse them."""

from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.runtime.jit import constexpr_function

from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr

BT = gl.constexpr(64)  # chunk rows
BC = gl.constexpr(16)  # sub-chunk rows
D = gl.constexpr(128)  # head dim (K = V)
RCP_LN2 = gl.constexpr(1.4426950408889634)
L2_EPS = gl.constexpr(1e-6)


@constexpr_function
def tile_layout(num_warps):
    """[64, 128] rows x channels: L2 norm, gate and its scan."""
    return gl.BlockedLayout([4, 8], [4, 16], [num_warps, 1], [1, 0])


@constexpr_function
def blk_layout(num_warps):
    """[16, 16] diagonal blocks of the solve."""
    return gl.BlockedLayout([1, 1], [16, 4], [1, num_warps], [1, 0])


@constexpr_function
def f32_mfma_layout(num_warps):
    """fp32 16x16x4 MFMA: A blocks ([16, 64] per row block) and the solve ([16, 128])."""
    return gl.amd.AMDMFMALayout(
        version=4,
        instr_shape=[16, 16, 4],
        transposed=True,
        warps_per_cta=[1, num_warps],
    )


@constexpr_function
def key_t_layout(num_warps):
    """[128, 64] channels x chunk rows (transposed key operand of the A blocks)."""
    return gl.BlockedLayout([8, 1], [16, 4], [1, num_warps], [0, 1])


@constexpr_function
def a_cols(i):
    """Key columns of row block i's A blocks: a power of two >= 16 (i + 1)."""
    return 16 if i == 0 else (32 if i == 1 else 64)


@constexpr_function
def mfma_layout(num_warps):
    """bf16 16x16x32 MFMA for the state and output passes."""
    return gl.amd.AMDMFMALayout(
        version=4,
        instr_shape=[16, 16, 32],
        transposed=True,
        warps_per_cta=[num_warps, 1],
    )


@constexpr_function
def row_major_layout(num_warps):
    """Loads of [rows, 128] bf16 operands before the MFMA layout conversion."""
    return gl.BlockedLayout([1, 8], [8, 8], [num_warps, 1], [1, 0])


@constexpr_function
def col_major_layout(num_warps):
    """Loads of transposed [128, rows] bf16 operands."""
    return gl.BlockedLayout([8, 1], [8, 8], [1, num_warps], [0, 1])


@gluon.jit
def _add(a, b):
    return a + b


@constexpr_function
def scan_layout(num_warps):
    """[64, 128] with all 64 rows of a column in one thread: the in-chunk cumsum is a
    sequential sum in row order."""
    return gl.BlockedLayout([BT, 1], [1, 64], [1, num_warps], [0, 1])


@gluon.jit
def _gate_cumsum(gate, num_warps: gl.constexpr):
    """In-chunk gate cumsum in log2 units, row by row: G_p = G_(p-1) + gate_p / ln 2 with
    G_(-1) = 0, in fp32 (a decode step adds one row to the cached G_(p-1)). gate is a
    [64, 128] tile in tile_layout; so is the result."""
    g = gl.convert_layout(gate * RCP_LN2, scan_layout(num_warps))
    return gl.convert_layout(gl.associative_scan(g, 0, _add), tile_layout(num_warps))


@gluon.jit
def _l2norm(x):
    """x / sqrt(sum(x^2) + eps) per row, rounded to bf16."""
    r = 1.0 / gl.sqrt(gl.sum(x * x, axis=1) + L2_EPS)
    return (x * gl.expand_dims(r, 1)).to(gl.bfloat16)


@gluon.jit
def _f32_dot(a, b, acc, num_warps: gl.constexpr):
    """acc + a @ b on the fp32 16x16x4 MFMA (fixed K order)."""
    MF: gl.constexpr = f32_mfma_layout(num_warps)
    return gl.amd.cdna4.mfma(
        gl.convert_layout(a, gl.DotOperandLayout(0, MF, 1)),
        gl.convert_layout(b, gl.DotOperandLayout(1, MF, 1)),
        acc,
    )


@gluon.jit
def _a_block(
    q_i,
    k_i,
    g_i,
    g_ref,
    k_t,
    g_t,
    g_ref_t,
    beta_i,
    scale,
    i: gl.constexpr,
    num_warps: gl.constexpr,
):
    """Row block i of Aqk and of beta-scaled Akk, fp32 [16, 64] in f32_mfma_layout, from
    register operands:
    - f32_mfma_layout: q_i, k_i (bf16 [16, 128]), g_i (fp32 [16, 128]), g_ref (fp32
      [1, 128], the gate cumsum of row 16 i) and beta_i (fp32 [16], SliceLayout(1, .));
    - key_t_layout: k_t (bf16 [128, NC]), g_t (fp32 [128, NC]) and g_ref_t ([128, 1]),
      NC = a_cols(i), the key columns [0, NC); the result is [16, NC].
    Key columns >= 16 (i + 1) only need finite values: they are zeroed in the operand.
    Exponents are relative to row 16 i. Aqk keeps c <= r, Akk c < r."""
    MF: gl.constexpr = f32_mfma_layout(num_warps)
    KT: gl.constexpr = key_t_layout(num_warps)
    e = gl.exp2(g_i - g_ref)
    qg = q_i.to(gl.float32) * e
    kg = k_i.to(gl.float32) * e
    NC: gl.constexpr = a_cols(i)
    col_t = gl.expand_dims(gl.arange(0, NC, layout=gl.SliceLayout(0, KT)), 0)
    kn = gl.where(
        col_t < (i + 1) * BC, k_t.to(gl.float32) * gl.exp2(g_ref_t - g_t), 0.0
    )
    z = gl.zeros([BC, NC], gl.float32, MF)
    aqk = _f32_dot(qg, kn, z, num_warps) * scale
    akk = _f32_dot(kg, kn, z, num_warps)
    r2 = gl.expand_dims(gl.arange(0, BC, layout=gl.SliceLayout(1, MF)) + i * BC, 1)
    c = gl.expand_dims(gl.arange(0, NC, layout=gl.SliceLayout(0, MF)), 0)
    aqk = gl.where(r2 >= c, aqk, 0.0)
    akk = gl.where(r2 > c, akk * gl.expand_dims(beta_i, 1), 0.0)
    return aqk, akk


@gluon.jit
def _a_rows(
    s_q,
    s_k,
    s_g,
    s_l,
    BETA,
    sb,
    AQK,
    s_a,
    n,
    scale,
    i: gl.constexpr,
    num_warps: gl.constexpr,
):
    """_a_block for row block i with its operands from the chunk's shared tiles; stores the
    Aqk row block (bf16) and the Akk row block to s_l (fp32)."""
    MF: gl.constexpr = f32_mfma_layout(num_warps)
    KT: gl.constexpr = key_t_layout(num_warps)
    NC: gl.constexpr = a_cols(i)
    r = gl.arange(0, BC, layout=gl.SliceLayout(1, MF)) + i * BC
    aqk, akk = _a_block(
        s_q.slice(i * BC, BC).load(MF),
        s_k.slice(i * BC, BC).load(MF),
        s_g.slice(i * BC, BC).load(MF),
        s_g.slice(i * BC, 1).load(MF),
        s_k.slice(0, NC).permute([1, 0]).load(KT),
        s_g.slice(0, NC).permute([1, 0]).load(KT),
        s_g.slice(i * BC, 1).permute([1, 0]).load(KT),
        gl.load(BETA + r.to(gl.int64) * sb, mask=r < n, other=0.0).to(gl.float32),
        scale,
        i,
        num_warps,
    )
    r2 = gl.expand_dims(r, 1)
    c = gl.expand_dims(gl.arange(0, NC, layout=gl.SliceLayout(0, MF)), 0)
    row = AQK + r2.to(gl.int64) * s_a + c
    m = (r2 < n) & (c < NC)
    gl.store(row, aqk.to(gl.bfloat16), mask=m)
    for j in gl.static_range(NC, BT, NC):  # the columns past NC are 0
        gl.store(row + j, gl.zeros([BC, NC], gl.bfloat16, MF), mask=m)
    s_l.slice(i * BC, BC).slice(0, NC, 1).store(akk)


@gluon.jit
def _l_block(s_l, i: gl.constexpr, j: gl.constexpr, num_warps: gl.constexpr):
    return s_l.slice(i * BC, BC).slice(j * BC, BC, 1).load(blk_layout(num_warps))


@gluon.jit
def _inv_block(akk, num_warps: gl.constexpr):
    """(I + akk)^-1 of a strictly lower 16x16 block, by forward substitution over rows."""
    PB: gl.constexpr = blk_layout(num_warps)
    r = gl.expand_dims(gl.arange(0, BC, layout=gl.SliceLayout(1, PB)), 1)
    c = gl.expand_dims(gl.arange(0, BC, layout=gl.SliceLayout(0, PB)), 0)
    ai = -akk
    for i in gl.static_range(2, BC):
        a = -gl.sum(gl.where(r == i, akk, 0.0), axis=0)
        a_c = gl.convert_layout(a, gl.SliceLayout(1, PB))
        a = a + gl.sum(gl.expand_dims(a_c, 1) * ai, axis=0)
        ai = gl.where(r == i, gl.expand_dims(a, 0), ai)
    return ai + gl.where(r == c, 1.0, 0.0)


@gluon.jit
def _inverses(s_l, num_warps: gl.constexpr):
    """T_ii = (I + L_ii)^-1 of the four diagonal blocks."""
    t0 = _inv_block(_l_block(s_l, 0, 0, num_warps), num_warps)
    t1 = _inv_block(_l_block(s_l, 1, 1, num_warps), num_warps)
    t2 = _inv_block(_l_block(s_l, 2, 2, num_warps), num_warps)
    t3 = _inv_block(_l_block(s_l, 3, 3, num_warps), num_warps)
    return t0, t1, t2, t3


@gluon.jit
def _solve(s_l, t0, t1, t2, t3, r0, r1, r2, r3, num_warps: gl.constexpr):
    """x = (I + L)^-1 r by block forward substitution: x_i = T_ii (r_i - sum_j<i L_ij x_j),
    the j terms in ascending order."""
    z = gl.zeros([BC, D], gl.float32, f32_mfma_layout(num_warps))
    x0 = _f32_dot(t0, r0, z, num_warps)
    a = _f32_dot(-_l_block(s_l, 1, 0, num_warps), x0, r1, num_warps)
    x1 = _f32_dot(t1, a, z, num_warps)
    a = _f32_dot(-_l_block(s_l, 2, 0, num_warps), x0, r2, num_warps)
    a = _f32_dot(-_l_block(s_l, 2, 1, num_warps), x1, a, num_warps)
    x2 = _f32_dot(t2, a, z, num_warps)
    a = _f32_dot(-_l_block(s_l, 3, 0, num_warps), x0, r3, num_warps)
    a = _f32_dot(-_l_block(s_l, 3, 1, num_warps), x1, a, num_warps)
    a = _f32_dot(-_l_block(s_l, 3, 2, num_warps), x2, a, num_warps)
    x3 = _f32_dot(t3, a, z, num_warps)
    return x0, x1, x2, x3


@gluon.jit
def _solve2(
    s_l, t0, t1, t2, t3, w0, w1, w2, w3, u0, u1, u2, u3, num_warps: gl.constexpr
):
    """_solve for two right-hand sides (w and u) with their steps interleaved, so the two
    chains overlap; each output gets the same operations as from _solve."""
    z = gl.zeros([BC, D], gl.float32, f32_mfma_layout(num_warps))
    l10 = -_l_block(s_l, 1, 0, num_warps)
    l20 = -_l_block(s_l, 2, 0, num_warps)
    l21 = -_l_block(s_l, 2, 1, num_warps)
    l30 = -_l_block(s_l, 3, 0, num_warps)
    l31 = -_l_block(s_l, 3, 1, num_warps)
    l32 = -_l_block(s_l, 3, 2, num_warps)
    xw0 = _f32_dot(t0, w0, z, num_warps)
    xu0 = _f32_dot(t0, u0, z, num_warps)
    aw = _f32_dot(l10, xw0, w1, num_warps)
    au = _f32_dot(l10, xu0, u1, num_warps)
    xw1 = _f32_dot(t1, aw, z, num_warps)
    xu1 = _f32_dot(t1, au, z, num_warps)
    aw = _f32_dot(l20, xw0, w2, num_warps)
    au = _f32_dot(l20, xu0, u2, num_warps)
    aw = _f32_dot(l21, xw1, aw, num_warps)
    au = _f32_dot(l21, xu1, au, num_warps)
    xw2 = _f32_dot(t2, aw, z, num_warps)
    xu2 = _f32_dot(t2, au, z, num_warps)
    aw = _f32_dot(l30, xw0, w3, num_warps)
    au = _f32_dot(l30, xu0, u3, num_warps)
    aw = _f32_dot(l31, xw1, aw, num_warps)
    au = _f32_dot(l31, xu1, au, num_warps)
    aw = _f32_dot(l32, xw2, aw, num_warps)
    au = _f32_dot(l32, xu2, au, num_warps)
    xw3 = _f32_dot(t3, aw, z, num_warps)
    xu3 = _f32_dot(t3, au, z, num_warps)
    return xw0, xw1, xw2, xw3, xu0, xu1, xu2, xu3


@gluon.jit
def _store_rows(P, stride, x, i: gl.constexpr, n, num_warps: gl.constexpr):
    """bf16 rows [16 i, 16 i + 16) of a [*, 128] output; rows >= n are not written."""
    MF: gl.constexpr = f32_mfma_layout(num_warps)
    r = gl.arange(0, BC, layout=gl.SliceLayout(1, MF)) + i * BC
    c = gl.arange(0, D, layout=gl.SliceLayout(0, MF))
    off = gl.expand_dims(r.to(gl.int64) * stride, 1) + gl.expand_dims(c, 0)
    m = gl.expand_dims(r < n, 1) & gl.expand_dims(c < D, 0)
    gl.store(P + off, x.to(gl.bfloat16), mask=m)


@gluon.jit
def _beta_rows(BETA, sb, i: gl.constexpr, n, num_warps: gl.constexpr):
    r = gl.arange(0, BC, layout=gl.SliceLayout(1, f32_mfma_layout(num_warps))) + i * BC
    return gl.expand_dims(
        gl.load(BETA + r.to(gl.int64) * sb, mask=r < n, other=0.0).to(gl.float32), 1
    )


@gluon.jit
def _w_rhs_block(k_i, g_i, beta_i):
    """w right-hand side of one row block, k^ beta 2^G, from register operands (k_i bf16,
    g_i fp32 [16, 128] and beta_i [16, 1] fp32, all in f32_mfma_layout)."""
    return k_i.to(gl.float32) * beta_i * gl.exp2(g_i)


@gluon.jit
def _w_rhs(s_k, s_g, BETA, sb, i: gl.constexpr, n, num_warps: gl.constexpr):
    MF: gl.constexpr = f32_mfma_layout(num_warps)
    return _w_rhs_block(
        s_k.slice(i * BC, BC).load(MF),
        s_g.slice(i * BC, BC).load(MF),
        _beta_rows(BETA, sb, i, n, num_warps),
    )


@gluon.jit
def _u_rhs(V, sv, BETA, sb, i: gl.constexpr, n, num_warps: gl.constexpr):
    MF: gl.constexpr = f32_mfma_layout(num_warps)
    r = gl.arange(0, BC, layout=gl.SliceLayout(1, MF)) + i * BC
    c = gl.arange(0, D, layout=gl.SliceLayout(0, MF))
    off = gl.expand_dims(r.to(gl.int64) * sv, 1) + gl.expand_dims(c, 0)
    m = gl.expand_dims(r < n, 1) & gl.expand_dims(c < D, 0)
    v = gl.load(V + off, mask=m, other=0.0).to(gl.float32)
    return v * _beta_rows(BETA, sb, i, n, num_warps)


@gluon.jit
def _intra_tile(
    Q,
    K,
    V,
    G,
    BETA,
    DT_BIAS,
    a_log,
    lower_bound,
    scale,
    n,
    sq,
    sk,
    sv,
    sg,
    sb,
    QG,
    KG,
    W,
    U,
    AQK,
    GLAST,
    s_out,
    s_a,
    num_warps: gl.constexpr,
):
    """One chunk of one head: rows [0, n) at Q/K/V/G + r * s*, BETA + r * sb.

    Stores qg = q^ 2^G, kg = k^ 2^(G_last - G), w, u (bf16, row stride s_out), Aqk (bf16
    [rows, 64], row stride s_a) and G_last (fp32 [128]). q^, k^ are the bf16 L2-normed rows,
    G the in-chunk inclusive gate cumsum in log2 units. Strides may be int64."""
    L: gl.constexpr = tile_layout(num_warps)
    SH: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, order=[1, 0])
    rows = gl.arange(0, BT, layout=gl.SliceLayout(1, L))
    cols = gl.arange(0, D, layout=gl.SliceLayout(0, L))
    m = gl.expand_dims(rows < n, 1) & gl.expand_dims(cols < D, 0)
    r64 = gl.expand_dims(rows.to(gl.int64), 1)
    c2 = gl.expand_dims(cols, 0)

    qn = _l2norm(gl.load(Q + r64 * sq + c2, mask=m, other=0.0).to(gl.float32))
    kn = _l2norm(gl.load(K + r64 * sk + c2, mask=m, other=0.0).to(gl.float32))
    x = gl.load(G + r64 * sg + c2, mask=m, other=0.0).to(gl.float32)
    x = x + gl.expand_dims(gl.load(DT_BIAS + cols).to(gl.float32), 0)
    gate = lower_bound / (1.0 + gl.exp(-(gl.exp(a_log) * x)))
    gate = gl.where(m, gate, 0.0)
    gcum = _gate_cumsum(gate, num_warps)
    g_last = gl.sum(gl.where(gl.expand_dims(rows == n - 1, 1), gcum, 0.0), axis=0)
    gl.store(GLAST + cols, g_last)
    off_o = r64 * s_out + c2
    gl.store(QG + off_o, (qn.to(gl.float32) * gl.exp2(gcum)).to(gl.bfloat16), mask=m)
    kg = kn.to(gl.float32) * gl.exp2(gl.expand_dims(g_last, 0) - gcum)
    gl.store(KG + off_o, kg.to(gl.bfloat16), mask=m)

    s_q = gl.allocate_shared_memory(gl.bfloat16, [BT, D], SH, qn)
    s_k = gl.allocate_shared_memory(gl.bfloat16, [BT, D], SH, kn)
    s_g = gl.allocate_shared_memory(gl.float32, [BT, D], SH, gcum)
    s_l = gl.allocate_shared_memory(gl.float32, [BT, BT], SH)
    for i in gl.static_range(4):
        _a_rows(s_q, s_k, s_g, s_l, BETA, sb, AQK, s_a, n, scale, i, num_warps)

    t0, t1, t2, t3 = _inverses(s_l, num_warps)
    xw0, xw1, xw2, xw3, xu0, xu1, xu2, xu3 = _solve2(
        s_l,
        t0,
        t1,
        t2,
        t3,
        _w_rhs(s_k, s_g, BETA, sb, 0, n, num_warps),
        _w_rhs(s_k, s_g, BETA, sb, 1, n, num_warps),
        _w_rhs(s_k, s_g, BETA, sb, 2, n, num_warps),
        _w_rhs(s_k, s_g, BETA, sb, 3, n, num_warps),
        _u_rhs(V, sv, BETA, sb, 0, n, num_warps),
        _u_rhs(V, sv, BETA, sb, 1, n, num_warps),
        _u_rhs(V, sv, BETA, sb, 2, n, num_warps),
        _u_rhs(V, sv, BETA, sb, 3, n, num_warps),
        num_warps,
    )
    _store_rows(W, s_out, xw0, 0, n, num_warps)
    _store_rows(W, s_out, xw1, 1, n, num_warps)
    _store_rows(W, s_out, xw2, 2, n, num_warps)
    _store_rows(W, s_out, xw3, 3, n, num_warps)
    _store_rows(U, s_out, xu0, 0, n, num_warps)
    _store_rows(U, s_out, xu1, 1, n, num_warps)
    _store_rows(U, s_out, xu2, 2, n, num_warps)
    _store_rows(U, s_out, xu3, 3, n, num_warps)


@gluon.jit
def _load_rows(P, stride, n, NR: gl.constexpr, NC: gl.constexpr, LAY: gl.constexpr):
    """[NR, NC] tile of rows [0, n) at P + r * stride (rows >= n are 0)."""
    r = gl.arange(0, NR, layout=gl.SliceLayout(1, LAY))
    c = gl.arange(0, NC, layout=gl.SliceLayout(0, LAY))
    off = gl.expand_dims(r.to(gl.int64) * stride, 1) + gl.expand_dims(c, 0)
    m = gl.expand_dims(r < n, 1) & gl.expand_dims(c < NC, 0)
    return gl.load(P + off, mask=m, other=0.0)


@gluon.jit
def _load_rows_t(P, stride, n, NR: gl.constexpr, NC: gl.constexpr, LAY: gl.constexpr):
    """[NC, NR] transpose of _load_rows."""
    c = gl.arange(0, NC, layout=gl.SliceLayout(1, LAY))
    r = gl.arange(0, NR, layout=gl.SliceLayout(0, LAY))
    off = gl.expand_dims(r.to(gl.int64) * stride, 0) + gl.expand_dims(c, 1)
    m = gl.expand_dims(r < n, 0) & gl.expand_dims(c < NC, 1)
    return gl.load(P + off, mask=m, other=0.0)


@gluon.jit
def _load_chunk(W, U, KG, GLAST, n, s_out, BV: gl.constexpr, num_warps: gl.constexpr):
    """Operands of one chunk's state pass (U pre-offset to the V block): w [64, 128],
    u [64, BV], kg^T [128, 64] (bf16, rows >= n are 0) and G_last [128] (fp32)."""
    MF: gl.constexpr = mfma_layout(num_warps)
    w = _load_rows(W, s_out, n, BT, D, row_major_layout(num_warps))
    u = _load_rows(U, s_out, n, BT, BV, MF)
    kg = _load_rows_t(KG, s_out, n, BT, D, col_major_layout(num_warps))
    g_last = gl.load(GLAST + gl.arange(0, D, layout=gl.SliceLayout(1, MF)))
    return w, u, kg, g_last


@gluon.jit
def _vnew(w, u, hb, n, BV: gl.constexpr, num_warps: gl.constexpr):
    """bf16(u - w @ hb) in the MFMA layout, rows >= n set to 0; hb = bf16(h) [128, BV]."""
    MF: gl.constexpr = mfma_layout(num_warps)
    d = gl.amd.cdna4.mfma(
        gl.convert_layout(w, gl.DotOperandLayout(0, MF, 8)),
        gl.convert_layout(hb, gl.DotOperandLayout(1, MF, 8)),
        gl.zeros([BT, BV], gl.float32, MF),
    )
    vn = (u.to(gl.float32) - d).to(gl.bfloat16)
    r = gl.arange(0, BT, layout=gl.SliceLayout(1, MF))
    return gl.where(gl.expand_dims(r < n, 1), vn, 0.0)


@gluon.jit
def _store_vn(VN, vn, n, s_vn, BV: gl.constexpr, num_warps: gl.constexpr):
    MF: gl.constexpr = mfma_layout(num_warps)
    r = gl.arange(0, BT, layout=gl.SliceLayout(1, MF))
    c = gl.arange(0, BV, layout=gl.SliceLayout(0, MF))
    off = gl.expand_dims(r.to(gl.int64) * s_vn, 1) + gl.expand_dims(c, 0)
    gl.store(VN + off, vn, mask=gl.expand_dims(r < n, 1) & gl.expand_dims(c < BV, 0))


@gluon.jit
def _state_step(h, kg, g_last, vn, num_warps: gl.constexpr):
    """h 2^G_last + kg^T @ v_new: the fp32 [128, BV] state after the chunk."""
    MF: gl.constexpr = mfma_layout(num_warps)
    h = h * gl.expand_dims(gl.exp2(g_last), 1)
    return gl.amd.cdna4.mfma(
        gl.convert_layout(kg, gl.DotOperandLayout(0, MF, 8)),
        gl.convert_layout(vn, gl.DotOperandLayout(1, MF, 8)),
        h,
    )


@gluon.jit
def _out_tile(
    QG,
    AQK,
    VN,
    hb,
    O,
    n,
    scale,
    s_out,
    s_a,
    s_vn,
    s_o,
    BV: gl.constexpr,
    num_warps: gl.constexpr,
):
    """o = scale (qg @ bf16(h)) + Aqk @ v_new for one V block; stores bf16 rows [0, n)."""
    MF: gl.constexpr = mfma_layout(num_warps)
    RM: gl.constexpr = row_major_layout(num_warps)
    qg = gl.convert_layout(
        _load_rows(QG, s_out, n, BT, D, RM), gl.DotOperandLayout(0, MF, 8)
    )
    acc = gl.amd.cdna4.mfma(
        qg,
        gl.convert_layout(hb, gl.DotOperandLayout(1, MF, 8)),
        gl.zeros([BT, BV], gl.float32, MF),
    )
    acc = acc * scale
    a = gl.convert_layout(
        _load_rows(AQK, s_a, n, BT, BT, RM), gl.DotOperandLayout(0, MF, 8)
    )
    vn = _load_rows(VN, s_vn, n, BT, BV, RM)
    acc = gl.amd.cdna4.mfma(
        a, gl.convert_layout(vn, gl.DotOperandLayout(1, MF, 8)), acc
    )
    r = gl.arange(0, BT, layout=gl.SliceLayout(1, MF))
    c = gl.arange(0, BV, layout=gl.SliceLayout(0, MF))
    off = gl.expand_dims(r.to(gl.int64) * s_o, 1) + gl.expand_dims(c, 0)
    gl.store(
        O + off,
        acc.to(gl.bfloat16),
        mask=gl.expand_dims(r < n, 1) & gl.expand_dims(c < BV, 0),
    )


@gluon.jit
def _state_ptrs(P, s_k, BV: gl.constexpr, num_warps: gl.constexpr):
    """Offsets of a [128, BV] state block (row stride s_k, unit column stride)."""
    MF: gl.constexpr = mfma_layout(num_warps)
    k = gl.arange(0, D, layout=gl.SliceLayout(1, MF))
    v = gl.arange(0, BV, layout=gl.SliceLayout(0, MF))
    return P + gl.expand_dims(k.to(gl.int64) * s_k, 1) + gl.expand_dims(v, 0)


_intra_repr = make_kernel_repr("_kda_bi_intra_kernel", [])


@gluon.jit(repr=_intra_repr)
def _kda_bi_intra_kernel(
    Q,
    K,
    V,
    G,
    BETA,
    A_LOG,
    DT_BIAS,
    CHUNKS,
    QG,
    KG,
    W,
    U,
    AQK,
    GLAST,
    lower_bound,
    scale,
    H,
    num_warps: gl.constexpr,
):
    """Program (chunk c, head h): _intra_tile on rows CHUNKS[c] = [r0, r1)."""
    c = gl.program_id(0)
    h = gl.program_id(1)
    r0 = gl.load(CHUNKS + 2 * c)
    n = gl.load(CHUNKS + 2 * c + 1) - r0
    sx = H * D
    base = (r0.to(gl.int64) * H + h) * D
    _intra_tile(
        Q + base,
        K + base,
        V + base,
        G + base,
        BETA + r0.to(gl.int64) * H + h,
        DT_BIAS + h * D,
        gl.load(A_LOG + h).to(gl.float32),
        lower_bound,
        scale,
        n,
        sx,
        sx,
        sx,
        sx,
        H,
        QG + base,
        KG + base,
        W + base,
        U + base,
        AQK + (r0.to(gl.int64) * H + h) * BT,
        GLAST + (c.to(gl.int64) * H + h) * D,
        sx,
        H * BT,
        num_warps,
    )


@gluon.jit
def _out_vn(
    QG,
    AQK,
    vn,
    hb,
    O,
    n,
    scale,
    s_out,
    s_a,
    s_o,
    BV: gl.constexpr,
    num_warps: gl.constexpr,
):
    """_out_tile with v_new given in registers (bf16 [64, BV] in the MFMA layout): the
    same operations, so the same bits."""
    MF: gl.constexpr = mfma_layout(num_warps)
    RM: gl.constexpr = row_major_layout(num_warps)
    qg = gl.convert_layout(
        _load_rows(QG, s_out, n, BT, D, RM), gl.DotOperandLayout(0, MF, 8)
    )
    acc = gl.amd.cdna4.mfma(
        qg,
        gl.convert_layout(hb, gl.DotOperandLayout(1, MF, 8)),
        gl.zeros([BT, BV], gl.float32, MF),
    )
    acc = acc * scale
    a = gl.convert_layout(
        _load_rows(AQK, s_a, n, BT, BT, RM), gl.DotOperandLayout(0, MF, 8)
    )
    acc = gl.amd.cdna4.mfma(
        a, gl.convert_layout(vn, gl.DotOperandLayout(1, MF, 8)), acc
    )
    r = gl.arange(0, BT, layout=gl.SliceLayout(1, MF))
    c = gl.arange(0, BV, layout=gl.SliceLayout(0, MF))
    off = gl.expand_dims(r.to(gl.int64) * s_o, 1) + gl.expand_dims(c, 0)
    gl.store(
        O + off,
        acc.to(gl.bfloat16),
        mask=gl.expand_dims(r < n, 1) & gl.expand_dims(c < BV, 0),
    )


_state_repr = make_kernel_repr("_kda_bi_state_kernel", ["BV"])


@gluon.jit
def _chunk_at(CHUNKS, c, c_end, n_chunks):
    """(first row, rows) of chunk c, or (row of a valid chunk, 0) when c >= c_end."""
    cc = gl.minimum(c, n_chunks - 1)
    r0 = gl.load(CHUNKS + 2 * cc)
    n = gl.load(CHUNKS + 2 * cc + 1) - r0
    return cc, r0, gl.where(c < c_end, n, 0)


@gluon.jit(repr=_state_repr)
def _kda_bi_state_kernel(
    W,
    U,
    KG,
    GLAST,
    CHUNKS,
    SEQ_CHUNKS,
    SEQ_ORDER,
    H0,
    HT,
    HB,
    HS,
    VN,
    QG,
    AQK,
    O,
    scale,
    H,
    n_chunks,
    HAS_H0: gl.constexpr,
    STORE_HT: gl.constexpr,
    STORE_HB: gl.constexpr,
    FUSE_OUT: gl.constexpr,
    ORDERED: gl.constexpr,
    BV: gl.constexpr,
    num_warps: gl.constexpr,
):
    """Program (sequence, head h, V block): walks the sequence's chunks in order, loading
    chunk c + 1 while chunk c computes. Per chunk: bf16(h) to HS, v_new, the FP32 state step.
    HT gets the final state, HB the state after the last complete chunk. With FUSE_OUT the
    output of each chunk is computed here too (_out_vn), instead of HS + the out kernel.
    """
    MF: gl.constexpr = mfma_layout(num_warps)
    NV: gl.constexpr = D // BV
    # flat grid, heads fastest: the programs of SEQ_ORDER[0] first, then SEQ_ORDER[1], ...
    # (longest sequences first, so the longest serial chains start first)
    pid = gl.program_id(0)
    h = pid % H
    v0 = ((pid // H) % NV) * BV
    if ORDERED:
        s = gl.load(SEQ_ORDER + pid // (NV * H))
    else:
        s = pid // (NV * H)
    st = (s.to(gl.int64) * H + h) * D * D + v0
    if HAS_H0:
        hs = gl.load(_state_ptrs(H0 + st, D, BV, num_warps))
    else:
        hs = gl.zeros([D, BV], gl.float32, MF)
    c0 = gl.load(SEQ_CHUNKS + s)
    c1 = gl.load(SEQ_CHUNKS + s + 1)
    sx = H * D
    cc, r0, n = _chunk_at(CHUNKS, c0, c1, n_chunks)
    base = (r0.to(gl.int64) * H + h) * D
    abase = (r0.to(gl.int64) * H + h) * BT
    w, u, kg, g_last = _load_chunk(
        W + base,
        U + base + v0,
        KG + base,
        GLAST + (cc.to(gl.int64) * H + h) * D,
        n,
        sx,
        BV,
        num_warps,
    )
    for c in range(c0, c1):
        cn, r0n, nn = _chunk_at(CHUNKS, c + 1, c1, n_chunks)
        bn = (r0n.to(gl.int64) * H + h) * D
        w2, u2, kg2, g2 = _load_chunk(
            W + bn,
            U + bn + v0,
            KG + bn,
            GLAST + (cn.to(gl.int64) * H + h) * D,
            nn,
            sx,
            BV,
            num_warps,
        )
        hb = hs.to(gl.bfloat16)
        if not FUSE_OUT:
            gl.store(
                _state_ptrs(
                    HS + (c.to(gl.int64) * H + h) * D * D + v0, D, BV, num_warps
                ),
                hb,
            )
        if STORE_HB:
            # an open chunk: the state so far is the boundary state
            gl.store(_state_ptrs(HB + st, D, BV, num_warps), hs, mask=n < BT)
        vn = _vnew(w, u, hb, n, BV, num_warps)
        if FUSE_OUT:
            _out_vn(
                QG + base,
                AQK + abase,
                vn,
                hb,
                O + base + v0,
                n,
                scale,
                sx,
                H * BT,
                sx,
                BV,
                num_warps,
            )
        else:
            _store_vn(VN + base + v0, vn, n, sx, BV, num_warps)
        hs = _state_step(hs, kg, g_last, vn, num_warps)
        w, u, kg, g_last, n, base = w2, u2, kg2, g2, nn, bn
        abase = (r0n.to(gl.int64) * H + h) * BT
    if STORE_HT:
        gl.store(_state_ptrs(HT + st, D, BV, num_warps), hs)
    if STORE_HB:
        if c1 == c0:
            gl.store(_state_ptrs(HB + st, D, BV, num_warps), hs)
        else:
            last = gl.load(CHUNKS + 2 * c1 - 1) - gl.load(CHUNKS + 2 * c1 - 2)
            if last == BT:
                gl.store(_state_ptrs(HB + st, D, BV, num_warps), hs)


_out_repr = make_kernel_repr("_kda_bi_out_kernel", ["BV"])


@gluon.jit(repr=_out_repr)
def _kda_bi_out_kernel(
    QG,
    AQK,
    VN,
    HS,
    O,
    CHUNKS,
    scale,
    H,
    BV: gl.constexpr,
    num_warps: gl.constexpr,
):
    """Program (chunk c, head h, V block): _out_tile with the chunk's boundary state HS[c]."""
    c = gl.program_id(0)
    h = gl.program_id(1)
    v0 = gl.program_id(2) * BV
    r0 = gl.load(CHUNKS + 2 * c)
    n = gl.load(CHUNKS + 2 * c + 1) - r0
    hb = gl.load(
        _state_ptrs(HS + (c.to(gl.int64) * H + h) * D * D + v0, D, BV, num_warps)
    )
    sx = H * D
    base = (r0.to(gl.int64) * H + h) * D
    _out_tile(
        QG + base,
        AQK + (r0.to(gl.int64) * H + h) * BT,
        VN + base + v0,
        hb,
        O + base + v0,
        n,
        scale,
        sx,
        H * BT,
        sx,
        sx,
        BV,
        num_warps,
    )


_seq_chunks_repr = make_kernel_repr("_kda_bi_seq_chunks_kernel", ["BLOCK"])


@gluon.jit(repr=_seq_chunks_repr)
def _kda_bi_seq_chunks_kernel(
    CU, SEQ_CHUNKS, n_seq, BLOCK: gl.constexpr, num_warps: gl.constexpr
):
    """SEQ_CHUNKS[s + 1] = 64-row chunks of sequences 0..s (SEQ_CHUNKS[0] = 0), one program."""
    LAY: gl.constexpr = gl.BlockedLayout(
        [BLOCK // (64 * num_warps)], [64], [num_warps], [0]
    )
    gl.store(SEQ_CHUNKS, 0)
    carry = gl.to_tensor(0)
    for t in range(0, n_seq, BLOCK):
        s = t + gl.arange(0, BLOCK, layout=LAY)
        m = s < n_seq
        lens = gl.load(CU + s + 1, mask=m, other=0) - gl.load(CU + s, mask=m, other=0)
        nch = ((lens + BT - 1) // BT).to(gl.int32)
        gl.store(SEQ_CHUNKS + s + 1, gl.associative_scan(nch, 0, _add) + carry, mask=m)
        carry = carry + gl.sum(nch, axis=0)


_chunk_table_repr = make_kernel_repr("_kda_bi_chunk_table_kernel", ["BLOCK"])


@gluon.jit(repr=_chunk_table_repr)
def _kda_bi_chunk_table_kernel(
    CU,
    SEQ_CHUNKS,
    CHUNKS,
    n_seq,
    n_chunks,
    n_steps,
    BLOCK: gl.constexpr,
    num_warps: gl.constexpr,
):
    """CHUNKS[c] = (first row, end row) of chunk c: the sequence is found by binary search in
    SEQ_CHUNKS (n_steps >= log2(n_seq + 1))."""
    LAY: gl.constexpr = gl.BlockedLayout(
        [BLOCK // (64 * num_warps)], [64], [num_warps], [0]
    )
    c = gl.program_id(0) * BLOCK + gl.arange(0, BLOCK, layout=LAY)
    m = c < n_chunks
    lo = gl.zeros([BLOCK], gl.int32, LAY)
    hi = gl.zeros([BLOCK], gl.int32, LAY) + n_seq
    for _ in range(n_steps):
        live = hi - lo > 1
        mid = (lo + hi) // 2
        go = gl.load(SEQ_CHUNKS + mid, mask=m & live, other=0) <= c
        lo = gl.where(live & go, mid, lo)
        hi = gl.where(live & (~go), mid, hi)
    start = gl.load(CU + lo, mask=m, other=0) + BT * (
        c - gl.load(SEQ_CHUNKS + lo, mask=m, other=0)
    )
    end = gl.minimum(start + BT, gl.load(CU + lo + 1, mask=m, other=0))
    gl.store(CHUNKS + 2 * c, start.to(gl.int32), mask=m)
    gl.store(CHUNKS + 2 * c + 1, end.to(gl.int32), mask=m)


_seq_order_repr = make_kernel_repr("_kda_bi_seq_order_kernel", ["BLOCK"])


@gluon.jit(repr=_seq_order_repr)
def _kda_bi_seq_order_kernel(
    SEQ_CHUNKS, SEQ_ORDER, n_seq, BLOCK: gl.constexpr, num_warps: gl.constexpr
):
    """SEQ_ORDER = sequences by chunk count, most first, ties by index: the rank of s is
    the number of sequences before it in that order."""
    L2: gl.constexpr = gl.BlockedLayout([1, 4], [8, 8], [num_warps, 1], [1, 0])
    s = gl.program_id(0) * BLOCK + gl.arange(0, BLOCK, layout=gl.SliceLayout(1, L2))
    ms = s < n_seq
    ns = gl.load(SEQ_CHUNKS + s + 1, mask=ms, other=0) - gl.load(
        SEQ_CHUNKS + s, mask=ms, other=0
    )
    rank = gl.zeros([BLOCK], gl.int32, gl.SliceLayout(1, L2))
    for t0 in range(0, n_seq, BLOCK):
        t = t0 + gl.arange(0, BLOCK, layout=gl.SliceLayout(0, L2))
        mt = t < n_seq
        nt = gl.load(SEQ_CHUNKS + t + 1, mask=mt, other=0) - gl.load(
            SEQ_CHUNKS + t, mask=mt, other=0
        )
        ns2 = gl.expand_dims(ns, 1)
        nt2 = gl.expand_dims(nt, 0)
        before = (nt2 > ns2) | (
            (nt2 == ns2) & (gl.expand_dims(t, 0) < gl.expand_dims(s, 1))
        )
        before = before & gl.expand_dims(mt, 0)
        rank += gl.sum(before.to(gl.int32), axis=1)
    gl.store(SEQ_ORDER + rank, s, mask=ms)
