# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL kernels of the KDA chunk forward (K = V = 128, chunk 64, no GVA).

Stage ports of ``chunk_delta_attn_fwd`` in
``aiter/ops/triton/_triton_kernels/kimi_delta_attn/chunk_fwd.py``; the
inter-chunk recurrence between ``intra`` and ``output`` is the GDN K5 kernel.

* ``gate_cumsum``: chunk-local cumsum of the gate in log2 space, optionally
  through ``-exp(A_log) * softplus(g + dt_bias)`` or
  ``lower_bound * sigmoid(exp(A_log) * (g + dt_bias))``.
* ``intra``: optional q/k l2norm and beta sigmoid, then per chunk
  ``Aqk = tril(q k^T * decay) * scale``,
  ``Akk^-1 = (I + strict_tril(beta k k^T * decay))^-1``,
  ``w = Akk^-1 (beta k exp2(g))``, ``u = Akk^-1 (beta v)``,
  ``qg = q exp2(g)`` and ``kg = k exp2(g_last - g)``.
* ``output``: ``o = scale * qg h + Aqk v_new``.

The intra off-diagonal ``Aqk`` / ``Akk`` tiles, ``w`` / ``u`` and both
``output`` GEMMs run on the 16x16x16 bf16 MFMA, one 16-row tile per wave.

Every program owns one 64-token chunk of one head with 256 threads (the gate
kernel: 128). Operand tiles are staged in LDS and the contractions run as fp32
FMAs: thread ``tid`` owns row ``tid // 4`` and the columns ``tid % 4 + 4 * e``
of each ``[64, *]`` tile, so a row reduction is two lane shuffles. ``intra``
stages ~128 KB of LDS, which needs gfx950.

Layouts are token-major ``[B, T, H, *]`` except ``h`` (the K5 snapshot,
``[B, NT, H, V, K]``) and ``v_new`` (head-major ``[B, H, T, V]``), which are
read as the K5 kernel writes them.
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl.expr import const_expr, gpu, range_constexpr

BT = 64
K = 128
V = 128
NUM_THREADS = 256
NE = K // 4  # row columns per thread
_LOG2E = 1.4426950408889634
_RCP_LN2 = 1.4426950408889634
_EXTENT = 1 << 30
_L2_EPS = 1e-6


def _num(dtype: torch.dtype):
    if dtype == torch.bfloat16:
        return fx.BFloat16
    if dtype == torch.float32:
        return fx.Float32
    raise ValueError(f"KDA FlyDSL forward: unsupported dtype {dtype}")


def _flat(tensor, base):
    """1-D element view of ``tensor`` starting at element ``base``."""
    it = fx.get_iter(fx.rocdl.make_buffer_tensor(tensor, max_size=True))
    if base is not None:
        it = fx.add_offset(it, base)
    return fx.Tensor(fx.make_view(it, fx.make_layout((_EXTENT, 1), (1, 1))))


def _lds(arr, n):
    return arr.view(fx.make_layout((n, 1), (1, 1)))


def _g_atom(num):
    if num is fx.BFloat16:
        return fx.make_copy_atom(fx.rocdl.BufferCopy16b(), fx.BFloat16)
    return fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Float32)


def _s_atom(num):
    return fx.make_copy_atom(fx.UniversalCopy(16 if num is fx.BFloat16 else 32), num)


def _rd(atom, view, i, num):
    frag = fx.make_rmem_tensor(1, num)
    fx.copy(atom, fx.slice(view, (i, None)), frag)
    val = frag.load()[0]
    return val if num is fx.Float32 else val.to(fx.Float32)


def _wr(atom, view, i, val, num):
    frag = fx.make_rmem_tensor(1, num)
    val = fx.Float32(val)
    if num is not fx.Float32:
        val = val.to(num)
    frag.store(fx.Vector.from_elements([val], dtype=num))
    fx.copy(atom, frag, fx.slice(view, (i, None)))


def ldg(view, i, num):
    return _rd(_g_atom(num), view, i, num)


def stg(view, i, val, num):
    _wr(_g_atom(num), view, i, val, num)


def lds_ld(view, i, num=fx.Float32):
    return _rd(_s_atom(num), view, i, num)


def lds_st(view, i, val, num=fx.Float32):
    _wr(_s_atom(num), view, i, val, num)


def _results(res, n):
    return [res] if n == 1 and not isinstance(res, (list, tuple)) else list(res)


def _zero():
    return fx.Float32(0.0)


def _sel(cond, a, b):
    return cond.select(a, b)


def _min(a, b):
    return (a < b).select(a, b)


def _exp2(x):
    return fx.exp2(x, fastmath="afn")


def _sigmoid(x):
    return fx.Float32(1.0) / (fx.Float32(1.0) + _exp2(-x * _LOG2E))


def _row_sum(x):
    """Sum over the 4 lanes that share a row."""
    x = x + x.shuffle_xor(fx.Int32(1), fx.Int32(64))
    return x + x.shuffle_xor(fx.Int32(2), fx.Int32(64))


def _vflat(tensor, base, n):
    """Element view of ``tensor`` from ``base`` addressed in groups of ``n``."""
    it = fx.add_offset(
        fx.get_iter(fx.rocdl.make_buffer_tensor(tensor, max_size=True)), base
    )
    return fx.Tensor(fx.make_view(it, fx.make_layout((_EXTENT // n, n), (n, 1))))


def _bits(num):
    return 16 if num is fx.BFloat16 else 32


def ldg_n(view_n, i, n, num):
    """``n`` contiguous elements at element ``i`` (a multiple of ``n``), as fp32."""
    frag = fx.make_rmem_tensor(n, num)
    fx.copy(
        fx.make_copy_atom(fx.rocdl.BufferCopy(n * _bits(num)), num),
        fx.slice(view_n, (i // n, None)),
        frag,
    )
    vec = fx.Vector(frag.load())
    return [vec[e] if num is fx.Float32 else vec[e].to(fx.Float32) for e in range(n)]


def stg_n(view_n, i, vals, num):
    frag = fx.make_rmem_tensor(len(vals), num)
    frag.store(
        fx.Vector.from_elements(
            [
                fx.Float32(x) if num is fx.Float32 else fx.Float32(x).to(num)
                for x in vals
            ],
            dtype=num,
        )
    )
    fx.copy(
        fx.make_copy_atom(fx.rocdl.BufferCopy(len(vals) * _bits(num)), num),
        frag,
        fx.slice(view_n, (i // len(vals), None)),
    )


def _row_ld(view_n, row, c0, rv, num):
    """Columns ``c0 .. c0 + NE`` of ``row`` in 128-bit loads, zero when ``not rv``."""
    n = 128 // _bits(num)
    out = []
    for m in range_constexpr(NE // n):
        out += ldg_n(view_n, row + c0 + m * n, n, num)
    return [_sel(rv, x, _zero()) for x in out]


def _row_st(view_n, row, c0, vals, num):
    n = 128 // _bits(num)
    for m in range_constexpr(NE // n):
        stg_n(view_n, row + c0 + m * n, vals[m * n : (m + 1) * n], num)


def _lds4(arr, n):
    """LDS view addressed in aligned groups of 4 elements."""
    return arr.view(fx.make_layout((n // 4, 4), (4, 1)))


def lds_ld4(view4, i, num=fx.Float32):
    """4 contiguous elements starting at element ``i`` (a multiple of 4)."""
    frag = fx.make_rmem_tensor(4, num)
    bits = 64 if num is fx.BFloat16 else 128
    fx.copy(
        fx.make_copy_atom(fx.UniversalCopy(bits), num),
        fx.slice(view4, (i // 4, None)),
        frag,
    )
    return fx.Vector(frag.load())


def _bf4(vals):
    return fx.Vector.from_elements(
        [fx.Float32(x).to(fx.BFloat16) for x in vals], dtype=fx.BFloat16
    )


def _mfma(a, b, acc):
    """One ``v_mfma_f32_16x16x16_bf16_1k`` on bare v4 register values.

    Lane ``l`` holds ``A[l % 16][4 * (l // 16) + v]``,
    ``B[4 * (l // 16) + v][l % 16]`` and returns
    ``C[4 * (l // 16) + v][l % 16]``.
    """
    mma = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 16, fx.BFloat16))
    fa = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.BFloat16)
    fb = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.BFloat16)
    fc = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Float32)
    fx.memref_store_vec(a, fa)
    fx.memref_store_vec(b, fb)
    fx.memref_store_vec(acc, fc)
    fx.mma_atom_call(mma, fc, fa, fb, fc)
    return fx.Vector(fx.memref_load_vec(fc))


def _mfma_f32(a, b, acc):
    """A 16x16x16 fp32 product as four ``v_mfma_f32_16x16x4_f32``.

    Same lane layout as ``_mfma``: step ``p`` pairs ``A[.][4 * (l // 16) + p]``
    with ``B[4 * (l // 16) + p][.]``, so the four steps cover all 16 of K.
    """
    mma = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 4, fx.Float32))
    fa = fx.make_rmem_tensor(1, fx.Float32)
    fb = fx.make_rmem_tensor(1, fx.Float32)
    fc = fx.make_rmem_tensor(4, fx.Float32)
    fc.store(acc)
    for p in range_constexpr(4):
        fa[0] = a[p]
        fb[0] = b[p]
        fx.gemm(mma, fc, fa, fb, fc)
    return fx.Vector(fc.load())


def lds_st4(view4, i, vals):
    frag = fx.make_rmem_tensor(4, fx.Float32)
    frag.store(fx.Vector.from_elements([fx.Float32(x) for x in vals], dtype=fx.Float32))
    fx.copy(
        fx.make_copy_atom(fx.UniversalCopy(128), fx.Float32),
        frag,
        fx.slice(view4, (i // 4, None)),
    )


def _zero4():
    return fx.Vector.filled(4, 0.0, fx.Float32)


def _l2norm(x):
    ss = _zero()
    for e in range_constexpr(NE):
        ss = ss + x[e] * x[e]
    rstd = fx.rsqrt(_row_sum(ss) + _L2_EPS)
    return [x[e] * rstd for e in range_constexpr(NE)]


def _load_tile(view, ld_stride, lds_view, rows, cols, nvalid, tid, num, lds_num):
    """Stage a ``[rows, cols]`` token tile into LDS, zeroing rows ``>= nvalid``."""
    for e in range_constexpr(rows * cols // NUM_THREADS):
        i = tid + e * NUM_THREADS
        t = i // cols
        c = i % cols
        ok = t < nvalid
        x = ldg(view, _sel(ok, t, 0) * ld_stride + c, num)
        lds_st(lds_view, i, _sel(ok, x, _zero()), lds_num)


def _chunk(i_t, T):
    t0 = i_t * BT
    return t0, _min(T - t0, fx.Int32(BT))


def ldg_i32(view, i):
    frag = fx.make_rmem_tensor(1, fx.Int32)
    fx.copy(
        fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Int32),
        fx.slice(view, (i, None)),
        frag,
    )
    return frag.load()[0]


def _chunk_of(i_t, T, varlen, cbos, clen):
    """First token and valid length of chunk ``i_t``.

    Varlen (B = 1): ``i_t`` is the global chunk in K5's ``chunk_offsets``
    order, and the host tables ``cbos`` / ``clen`` give its first packed
    token and its length, so no chunk crosses a sequence boundary.
    """
    if const_expr(varlen):
        return ldg_i32(_flat(cbos, None), i_t), ldg_i32(_flat(clen, None), i_t)
    return _chunk(i_t, T)


# ---------------------------------------------------------------------------
# gate_cumsum
# ---------------------------------------------------------------------------
@functools.cache
def compile_gate_cumsum(
    H: int,
    g_dtype: torch.dtype,
    use_gate: bool,
    has_bias: bool,
    use_lower_bound: bool,
    varlen: bool = False,
):
    GN = _num(g_dtype)

    @flyc.kernel(name="kda_fwd_gate_cumsum_flydsl")
    def kernel(
        g: fx.Tensor,
        A_log: fx.Tensor,
        dt_bias: fx.Tensor,
        gc: fx.Tensor,
        cbos: fx.Tensor,
        clen: fx.Tensor,
        T: fx.Int32,
        lower_bound: fx.Float32,
    ):
        i_t = fx.block_idx.x
        i_bh = fx.block_idx.y
        b = i_bh // H
        h = i_bh % H
        kk = fx.thread_idx.x
        t0, nvalid = _chunk_of(i_t, T, varlen, cbos, clen)
        hk = (fx.Int64(b * T + t0) * H + fx.Int64(h)) * K
        g_v = _flat(g, hk)
        o_v = _flat(gc, hk)
        a = _zero()
        bias = _zero()
        if const_expr(use_gate):
            a = fx.exp(ldg(_flat(A_log, None), h, fx.Float32))
            if const_expr(has_bias):
                bias = ldg(_flat(dt_bias, None), h * K + kk, fx.Float32)

        for i, st in range(0, BT, 1, init=[_zero()]):
            t = fx.Int32(i)
            ok = t < nvalid
            x = ldg(g_v, _sel(ok, t, 0) * (H * K) + kk, GN)
            if const_expr(use_gate):
                z = x + bias
                if const_expr(use_lower_bound):
                    x = lower_bound * _sigmoid(a * z)
                else:
                    sp = _sel(
                        z > fx.Float32(20.0),
                        z,
                        fx.log(fx.Float32(1.0) + fx.exp(z)),
                    )
                    x = -a * sp
            run = st[0] + _sel(ok, x, _zero())
            if ok:
                stg(o_v, t * (H * K) + kk, run * _RCP_LN2, fx.Float32)
            res = yield [run]  # noqa: F841 (loop-carried form)

    @flyc.jit
    def launch(
        g: fx.Tensor,
        A_log: fx.Tensor,
        dt_bias: fx.Tensor,
        gc: fx.Tensor,
        cbos: fx.Tensor,
        clen: fx.Tensor,
        T: fx.Int32,
        lower_bound: fx.Float32,
        NT: fx.Int32,
        NBH: fx.Int32,
        stream: fx.Stream,
    ):
        kernel(g, A_log, dt_bias, gc, cbos, clen, T, lower_bound).launch(
            grid=(NT, NBH, 1), block=(K, 1, 1), stream=stream
        )

    return launch


# ---------------------------------------------------------------------------
# intra
# ---------------------------------------------------------------------------
@functools.cache
def compile_intra(
    H: int,
    qk_dtype: torch.dtype,
    v_dtype: torch.dtype,
    beta_dtype: torch.dtype,
    l2norm_qk: bool,
    beta_sigmoid: bool,
    store_qkb: bool,
    varlen: bool = False,
    diag_mfma: bool = False,
):
    """``diag_mfma``: the diagonal tiles also run on MFMA, with the decay split
    at the block's middle row. Needs a per-token log2 decay bounded well under
    127 / 8 (the ``lower_bound`` gate: at most 5 / ln 2), else fp32."""
    QN = _num(qk_dtype)
    VN = _num(v_dtype)
    BN = _num(beta_dtype)
    BF = fx.BFloat16

    @fx.struct
    class Smem:
        q: fx.Array[fx.Float32, BT * K, 16]  # q, later beta * k * exp2(g)
        k: fx.Array[fx.Float32, BT * K, 16]  # k, later beta * v
        g: fx.Array[fx.Float32, BT * K, 16]
        l: fx.Array[fx.Float32, BT * BT, 16]
        x: fx.Array[fx.Float32, BT * BT, 16]
        beta: fx.Array[fx.Float32, BT, 16]
        t: fx.Array[fx.Float32, 4 * 16 * 16, 16]  # per-wave 16x16 solve scratch

    @flyc.kernel(name="kda_fwd_intra_flydsl")
    def kernel(
        q: fx.Tensor,
        k: fx.Tensor,
        v: fx.Tensor,
        g: fx.Tensor,
        beta: fx.Tensor,
        qn: fx.Tensor,
        kn: fx.Tensor,
        bn: fx.Tensor,
        qg: fx.Tensor,
        kg: fx.Tensor,
        w: fx.Tensor,
        u: fx.Tensor,
        Aqk: fx.Tensor,
        Akk: fx.Tensor,
        cbos: fx.Tensor,
        clen: fx.Tensor,
        T: fx.Int32,
        scale: fx.Float32,
    ):
        i_t = fx.block_idx.x
        i_bh = fx.block_idx.y
        b = i_bh // H
        h = i_bh % H
        tid = fx.thread_idx.x
        r = tid // 4
        seg = tid % 4
        t0, nvalid = _chunk_of(i_t, T, varlen, cbos, clen)
        rv = r < nvalid
        rc = _sel(rv, r, 0)
        hb = fx.Int64(b * T + t0) * H + fx.Int64(h)
        hk = hb * K
        row = rc * (H * K)

        lds = fx.SharedAllocator().allocate(Smem).peek()
        s_q = _lds(lds.q, BT * K)
        s_k = _lds(lds.k, BT * K)
        s_g = _lds(lds.g, BT * K)
        s_l = _lds(lds.l, BT * BT)
        s_x = _lds(lds.x, BT * BT)
        s_b = _lds(lds.beta, BT)

        # Thread (r, seg) owns the contiguous columns seg * NE .. seg * NE + NE.
        c0 = seg * NE
        cols = [c0 + e for e in range_constexpr(NE)]
        q_r = _row_ld(_vflat(q, hk, 128 // _bits(QN)), row, c0, rv, QN)
        k_r = _row_ld(_vflat(k, hk, 128 // _bits(QN)), row, c0, rv, QN)
        g_r = _row_ld(_vflat(g, hk, 4), row, c0, rv, fx.Float32)
        if const_expr(l2norm_qk):
            q_r = _l2norm(q_r)
            k_r = _l2norm(k_r)
        beta_r = ldg(_flat(beta, hb), rc * H, BN)
        if const_expr(beta_sigmoid):
            beta_r = _sigmoid(beta_r)
        beta_r = _sel(rv, beta_r, _zero())

        s_q4 = _lds4(lds.q, BT * K)
        s_k4 = _lds4(lds.k, BT * K)
        s_g4 = _lds4(lds.g, BT * K)
        for m in range_constexpr(NE // 4):
            o4 = r * K + c0 + 4 * m
            lds_st4(s_q4, o4, q_r[4 * m : 4 * m + 4])
            lds_st4(s_k4, o4, k_r[4 * m : 4 * m + 4])
            lds_st4(s_g4, o4, g_r[4 * m : 4 * m + 4])
        if seg == 0:
            lds_st(s_b, r, beta_r)
        if const_expr(store_qkb):  # noqa: SIM102 (compile-time vs runtime branch)
            if rv:
                _row_st(_vflat(qn, hk, 128 // _bits(QN)), r * (H * K), c0, q_r, QN)
                _row_st(_vflat(kn, hk, 128 // _bits(QN)), r * (H * K), c0, k_r, QN)
                if seg == 0:
                    stg(_flat(bn, hb), r * H, beta_r, fx.Float32)
        gpu.barrier()

        lane = tid % 64
        wave = tid // 64
        lr = lane % 16
        lk = (lane // 16) * 4
        aqk_v = _flat(Aqk, hb * BT)
        s_x4 = _lds4(lds.x, BT * BT)

        # Off-diagonal 16x16 tiles (I = wave > J) on MFMA. The decay splits at
        # gn = g[end of block J]: exp2(g_i - gn) and exp2(gn - g_j) are both
        # <= 1 there, so the bf16 operands neither overflow nor need masking.
        ia = wave * 16 + lr  # A-operand row
        for J in range_constexpr(BT // 16 - 1):
            jb = J * 16 + lr  # B-operand row (column j of the tile)
            gn_row = _min(fx.Int32(J * 16 + 15), nvalid - 1)
            acc_q = _zero4()
            acc_k = _zero4()
            for kt in range_constexpr(K // 16):
                kc = kt * 16 + lk
                gn = lds_ld4(s_g4, gn_row * K + kc)
                gi = lds_ld4(s_g4, ia * K + kc)
                qi = lds_ld4(s_q4, ia * K + kc)
                ki = lds_ld4(s_k4, ia * K + kc)
                gj = lds_ld4(s_g4, jb * K + kc)
                kj = lds_ld4(s_k4, jb * K + kc)
                # min(., 0) only bites on padded rows (g = 0), where it keeps inf * 0 out.
                di = [
                    _exp2(_min(gi[vi] - gn[vi], _zero())) for vi in range_constexpr(4)
                ]
                bj = _bf4(
                    [kj[vi] * _exp2(gn[vi] - gj[vi]) for vi in range_constexpr(4)]
                )
                acc_q = _mfma(
                    _bf4([qi[vi] * di[vi] for vi in range_constexpr(4)]), bj, acc_q
                )
                acc_k = _mfma(
                    _bf4([ki[vi] * di[vi] for vi in range_constexpr(4)]), bj, acc_k
                )
            if wave > J:
                for vi in range_constexpr(4):
                    i = wave * 16 + lk + vi
                    j = J * 16 + lr
                    lds_st(s_l, i * BT + j, acc_k[vi] * lds_ld(s_b, i))
                    if i < nvalid:
                        stg(aqk_v, i * (H * BT) + j, acc_q[vi] * scale, fx.Float32)

        if const_expr(diag_mfma):
            # Diagonal tiles on MFMA, decay split at the block's middle row:
            # |g_i - g_ref| <= 8 tokens of bounded decay, so no bf16 overflow.
            # min(., 64) only bites on padded rows (g = 0), which are dropped.
            jb = wave * 16 + lr
            ref = _min(wave * 16 + 7, nvalid - 1)
            cap = fx.Float32(64.0)
            acc_q = _zero4()
            acc_k = _zero4()
            for kt in range_constexpr(K // 16):
                kc = kt * 16 + lk
                gr = lds_ld4(s_g4, ref * K + kc)
                gi = lds_ld4(s_g4, ia * K + kc)
                qi = lds_ld4(s_q4, ia * K + kc)
                ki = lds_ld4(s_k4, ia * K + kc)
                gj = lds_ld4(s_g4, jb * K + kc)
                kj = lds_ld4(s_k4, jb * K + kc)
                di = [_exp2(_min(gi[vi] - gr[vi], cap)) for vi in range_constexpr(4)]
                bj = _bf4(
                    [
                        kj[vi] * _exp2(_min(gr[vi] - gj[vi], cap))
                        for vi in range_constexpr(4)
                    ]
                )
                acc_q = _mfma(
                    _bf4([qi[vi] * di[vi] for vi in range_constexpr(4)]), bj, acc_q
                )
                acc_k = _mfma(
                    _bf4([ki[vi] * di[vi] for vi in range_constexpr(4)]), bj, acc_k
                )
            for vi in range_constexpr(4):
                i = wave * 16 + lk + vi
                j = wave * 16 + lr
                lds_st(
                    s_l, i * BT + j, _sel(j < i, acc_k[vi] * lds_ld(s_b, i), _zero())
                )
                if i < nvalid:
                    stg(
                        aqk_v,
                        i * (H * BT) + j,
                        _sel(j <= i, acc_q[vi] * scale, _zero()),
                        fx.Float32,
                    )
        else:
            # Diagonal tiles in fp32 with the exact pairwise decay: wave I, row
            # 16I + lane // 4, columns 16I + lane % 4 + 4p.
            di_r = wave * 16 + lane // 4
            for kk, st in range(0, K, 1, init=[_zero()] * 8):
                kq = fx.Int32(kk)
                qx = lds_ld(s_q, di_r * K + kq)
                kx = lds_ld(s_k, di_r * K + kq)
                gx = lds_ld(s_g, di_r * K + kq)
                acc_q, acc_k = [], []
                for p in range_constexpr(4):
                    j = wave * 16 + lane % 4 + 4 * p
                    ex = _exp2(
                        _sel(
                            j <= di_r,
                            _min(gx - lds_ld(s_g, j * K + kq), _zero()),
                            _zero(),
                        )
                    )
                    kj = lds_ld(s_k, j * K + kq) * ex
                    acc_q.append(st[p] + qx * kj)
                    acc_k.append(st[4 + p] + kx * kj)
                res = yield acc_q + acc_k
            acc = _results(res, 8)
            beta_d = lds_ld(s_b, di_r)
            for p in range_constexpr(4):
                j = wave * 16 + lane % 4 + 4 * p
                lds_st(s_l, di_r * BT + j, _sel(j < di_r, acc[4 + p] * beta_d, _zero()))
                if di_r < nvalid:
                    stg(
                        aqk_v,
                        di_r * (H * BT) + j,
                        _sel(j <= di_r, acc[p] * scale, _zero()),
                        fx.Float32,
                    )
        # Aqk above the diagonal tiles is zero.
        if rv:
            for e in range_constexpr(16):
                j = seg + 4 * e
                if j // 16 > r // 16:
                    stg(aqk_v, r * (H * BT) + j, _zero(), fx.Float32)

        kb_r, qg_r, kg_r = [], [], []
        for m in range_constexpr(NE // 4):
            gl4 = lds_ld4(s_g4, (nvalid - 1) * K + c0 + 4 * m)
            for vi in range_constexpr(4):
                e = 4 * m + vi
                ge = _exp2(g_r[e])
                kb_r.append(k_r[e] * beta_r * ge)
                qg_r.append(q_r[e] * ge)
                kg_r.append(k_r[e] * _exp2(gl4[vi] - g_r[e]))
        if rv:
            _row_st(_vflat(qg, hk, 8), r * (H * K), c0, qg_r, BF)
            _row_st(_vflat(kg, hk, 8), r * (H * K), c0, kg_r, BF)
        vb_r = [
            x * beta_r
            for x in _row_ld(_vflat(v, hk, 128 // _bits(VN)), row, c0, rv, VN)
        ]
        gpu.barrier()

        # Transposed ([col][token]) so each MFMA B operand is 4 contiguous tokens.
        for e in range_constexpr(NE):
            lds_st(s_q, cols[e] * BT + r, kb_r[e])
            lds_st(s_k, cols[e] * BT + r, vb_r[e])

        # X = (I + L)^-1 blockwise. Each wave inverts its 16x16 diagonal block
        # by forward substitution (the four in lockstep), then block row I is
        # X_IJ = -X_II sum_{K < I} L_IK X_KJ on fp32 MFMA (X's upper blocks
        # are zero, so the sum can start at K = 0).
        for e in range_constexpr(BT * BT // NUM_THREADS):
            lds_st(s_x, tid + e * NUM_THREADS, _zero())
        gpu.barrier()
        dbase = wave * 16
        dc = lane // 4
        dseg = lane % 4
        for i, st in range(0, 16, 1, init=[_zero()]):
            iq = fx.Int32(i)
            part = _zero()
            for m in range_constexpr(4):
                j = dseg + 4 * m
                part = part + _sel(
                    j < iq,
                    lds_ld(s_l, (dbase + iq) * BT + dbase + j)
                    * lds_ld(s_x, (dbase + j) * BT + dbase + dc),
                    _zero(),
                )
            part = _row_sum(part)
            if dseg == 0:
                lds_st(
                    s_x,
                    (dbase + iq) * BT + dbase + dc,
                    _sel(iq == dc, fx.Float32(1.0), -part),
                )
            gpu.barrier()
            res = yield [st[0]]

        s_l4 = _lds4(lds.l, BT * BT)
        s_t = _lds(lds.t, 4 * 256)
        tbase = wave * 256
        for I in range_constexpr(1, BT // 16):
            acc = _zero4()
            for Kb in range_constexpr(I):
                a = lds_ld4(s_l4, (I * 16 + lr) * BT + Kb * 16 + lk)
                bx = fx.Vector.from_elements(
                    [
                        lds_ld(s_x, (Kb * 16 + lk + p) * BT + dbase + lr)
                        for p in range_constexpr(4)
                    ],
                    dtype=fx.Float32,
                )
                acc = _mfma_f32(a, bx, acc)
            for vi in range_constexpr(4):
                lds_st(s_t, tbase + (lk + vi) * 16 + lr, acc[vi])
            gpu.barrier()
            a = lds_ld4(s_x4, (I * 16 + lr) * BT + I * 16 + lk)
            bt = fx.Vector.from_elements(
                [lds_ld(s_t, tbase + (lk + p) * 16 + lr) for p in range_constexpr(4)],
                dtype=fx.Float32,
            )
            acc = _mfma_f32(a, bt, _zero4())
            if wave < I:
                for vi in range_constexpr(4):
                    lds_st(s_x, (I * 16 + lk + vi) * BT + dbase + lr, -acc[vi])
            gpu.barrier()

        akk_v = _flat(Akk, hb * BT)
        if rv:
            for e in range_constexpr(16):
                j = seg + 4 * e
                stg(akk_v, r * (H * BT) + j, lds_ld(s_x, r * BT + j), fx.Float32)

        # w = X (beta k exp2(g)), u = X (beta v) on MFMA: wave I owns rows
        # 16I..16I+15; X is lower triangular, so its upper tiles add zeros.
        xa = [
            _bf4(list(lds_ld4(s_x4, ia * BT + jt * 16 + lk)))
            for jt in range_constexpr(BT // 16)
        ]
        w_v = _flat(w, hk)
        u_v = _flat(u, hk)
        for nt in range_constexpr(K // 16):
            c = nt * 16 + lr
            acc_w = _zero4()
            acc_u = _zero4()
            for jt in range_constexpr(BT // 16):
                tj = jt * 16 + lk
                acc_w = _mfma(xa[jt], _bf4(list(lds_ld4(s_q4, c * BT + tj))), acc_w)
                acc_u = _mfma(xa[jt], _bf4(list(lds_ld4(s_k4, c * BT + tj))), acc_u)
            for vi in range_constexpr(4):
                i = wave * 16 + lk + vi
                if i < nvalid:
                    stg(w_v, i * (H * K) + nt * 16 + lr, acc_w[vi], BF)
                    stg(u_v, i * (H * V) + nt * 16 + lr, acc_u[vi], BF)

    @flyc.jit
    def launch(
        q: fx.Tensor,
        k: fx.Tensor,
        v: fx.Tensor,
        g: fx.Tensor,
        beta: fx.Tensor,
        qn: fx.Tensor,
        kn: fx.Tensor,
        bn: fx.Tensor,
        qg: fx.Tensor,
        kg: fx.Tensor,
        w: fx.Tensor,
        u: fx.Tensor,
        Aqk: fx.Tensor,
        Akk: fx.Tensor,
        cbos: fx.Tensor,
        clen: fx.Tensor,
        T: fx.Int32,
        scale: fx.Float32,
        NT: fx.Int32,
        NBH: fx.Int32,
        stream: fx.Stream,
    ):
        kernel(
            q, k, v, g, beta, qn, kn, bn, qg, kg, w, u, Aqk, Akk, cbos, clen, T, scale
        ).launch(grid=(NT, NBH, 1), block=(NUM_THREADS, 1, 1), stream=stream)

    return launch


# ---------------------------------------------------------------------------
# output
# ---------------------------------------------------------------------------
@functools.cache
def compile_output(
    H: int,
    h_dtype: torch.dtype,
    vn_dtype: torch.dtype,
    o_dtype: torch.dtype,
    varlen: bool = False,
):
    HN = _num(h_dtype)
    VN = _num(vn_dtype)
    ON = _num(o_dtype)
    BF = fx.BFloat16

    @fx.struct
    class Smem:
        h: fx.Array[fx.BFloat16, V * K, 16]  # K5 snapshot h[v][k]
        qg: fx.Array[fx.BFloat16, BT * K, 16]
        vn: fx.Array[fx.BFloat16, V * BT, 16]  # v_new^T [v][t]
        a: fx.Array[fx.BFloat16, BT * BT, 16]

    @flyc.kernel(name="kda_fwd_output_flydsl")
    def kernel(
        qg: fx.Tensor,
        hs: fx.Tensor,
        vnew: fx.Tensor,
        Aqk: fx.Tensor,
        o: fx.Tensor,
        cbos: fx.Tensor,
        clen: fx.Tensor,
        T: fx.Int32,
        scale: fx.Float32,
    ):
        i_t = fx.block_idx.x
        i_bh = fx.block_idx.y
        b = i_bh // H
        h = i_bh % H
        tid = fx.thread_idx.x
        NT = (T + (BT - 1)) // BT
        t0, nvalid = _chunk_of(i_t, T, varlen, cbos, clen)
        hb = fx.Int64(b * T + t0) * H + fx.Int64(h)
        # Varlen has B = 1, so b * NT + i_t is the global chunk either way.
        state = (fx.Int64(b * NT + i_t) * H + fx.Int64(h)) * (K * V)

        lds = fx.SharedAllocator().allocate(Smem).peek()
        s_h = _lds(lds.h, V * K)
        s_qg = _lds(lds.qg, BT * K)
        s_vn = _lds(lds.vn, V * BT)
        s_a = _lds(lds.a, BT * BT)
        h_v = _flat(hs, state)
        for e in range_constexpr(V * K // NUM_THREADS):
            i = tid + e * NUM_THREADS
            lds_st(s_h, i, ldg(h_v, i, HN), BF)
        _load_tile(_flat(qg, hb * K), H * K, s_qg, BT, K, nvalid, tid, BF, BF)
        vn_v = _flat(vnew, (fx.Int64(i_bh) * fx.Int64(T) + fx.Int64(t0)) * V)
        for e in range_constexpr(BT * V // NUM_THREADS):
            i = tid + e * NUM_THREADS
            t = i // V
            c = i % V
            ok = t < nvalid
            x = ldg(vn_v, _sel(ok, t, 0) * V + c, VN)
            lds_st(s_vn, c * BT + t, _sel(ok, x, _zero()), BF)
        _load_tile(
            _flat(Aqk, hb * BT), H * BT, s_a, BT, BT, nvalid, tid, fx.Float32, BF
        )
        gpu.barrier()

        # o = scale * qg h + Aqk v_new on MFMA; wave I owns rows 16I..16I+15.
        lane = tid % 64
        wave = tid // 64
        lr = lane % 16
        lk = (lane // 16) * 4
        ia = wave * 16 + lr
        s_h4 = _lds4(lds.h, V * K)
        s_qg4 = _lds4(lds.qg, BT * K)
        s_vn4 = _lds4(lds.vn, V * BT)
        s_a4 = _lds4(lds.a, BT * BT)
        qa = [
            lds_ld4(s_qg4, ia * K + kt * 16 + lk, BF) for kt in range_constexpr(K // 16)
        ]
        aa = [
            lds_ld4(s_a4, ia * BT + jt * 16 + lk, BF)
            for jt in range_constexpr(BT // 16)
        ]
        o_v = _flat(o, hb * V)
        for nt in range_constexpr(V // 16):
            c = nt * 16 + lr
            acc_h = _zero4()
            for kt in range_constexpr(K // 16):
                acc_h = _mfma(qa[kt], lds_ld4(s_h4, c * K + kt * 16 + lk, BF), acc_h)
            acc_v = _zero4()
            for jt in range_constexpr(BT // 16):
                acc_v = _mfma(aa[jt], lds_ld4(s_vn4, c * BT + jt * 16 + lk, BF), acc_v)
            for vi in range_constexpr(4):
                i = wave * 16 + lk + vi
                if i < nvalid:
                    stg(o_v, i * (H * V) + c, acc_h[vi] * scale + acc_v[vi], ON)

    @flyc.jit
    def launch(
        qg: fx.Tensor,
        hs: fx.Tensor,
        vnew: fx.Tensor,
        Aqk: fx.Tensor,
        o: fx.Tensor,
        cbos: fx.Tensor,
        clen: fx.Tensor,
        T: fx.Int32,
        scale: fx.Float32,
        NT: fx.Int32,
        NBH: fx.Int32,
        stream: fx.Stream,
    ):
        kernel(qg, hs, vnew, Aqk, o, cbos, clen, T, scale).launch(
            grid=(NT, NBH, 1), block=(NUM_THREADS, 1, 1), stream=stream
        )

    return launch
