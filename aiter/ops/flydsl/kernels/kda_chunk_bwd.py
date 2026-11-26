# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL kernels of the KDA chunk backward (K = V = 128, chunk 64, no GVA).

Stage-for-stage ports of ``chunk_delta_attn_bwd`` in
``aiter/ops/triton/_triton_kernels/kimi_delta_attn/chunk_bwd.py``:

* ``dav``: ``dAqk = tril(do @ v_new^T) * scale``, ``dv = tril(Aqk)^T @ do``.
* ``dhu``: reverse-time state gradient ``dh`` per chunk, ``dh0`` and
  ``dv2 = dv + kg @ dh``.
* ``wy_dqkg``: inter-chunk ``dq``/``dk``/``dg``, the WY gradients
  ``dv``/``dbeta`` and ``dAkk``.
* ``intra``: intra-chunk ``dq``/``dk``/``dg``/``dbeta``; optionally the l2norm
  backward of q/k and the beta-sigmoid backward.
* ``gate_cumsum``: reverse chunk cumsum of ``dg``; optionally the gate
  activation backward with per-program ``dA_log`` / ``dt_bias`` partials.

Every program owns one 64-token chunk of one head (``dhu``: one 32-column V
block of one state) with 256 threads. Operand tiles are staged in LDS and the
contractions run as fp32 FMAs: thread ``tid`` owns row ``tid // 4`` and the
columns ``tid % 4 + 4 * e`` of each ``[64, *]`` output, so a row reduction is
two lane shuffles. The LDS budget (up to ~147 KB in ``wy_dqkg``) needs gfx950.

Layouts are token-major ``[B, T, H, *]`` throughout, except ``h`` (the FlyDSL
K5 snapshot, ``[B, NT, H, V, K]``) and ``v_new`` (head-major ``[B, H, T, V]``),
which are read as the K5 kernel writes them; ``dh`` is ``[B, NT, H, K, V]``.
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl.expr import const_expr, gpu, range_constexpr

BT = 64
K = 128
V = 128
BV = 32
NUM_THREADS = 256
_LOG2E = 1.4426950408889634
_EXTENT = 1 << 30


def _num(dtype: torch.dtype):
    if dtype == torch.bfloat16:
        return fx.BFloat16
    if dtype == torch.float32:
        return fx.Float32
    raise ValueError(f"KDA FlyDSL backward: unsupported dtype {dtype}")


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


# ---------------------------------------------------------------------------
# dav
# ---------------------------------------------------------------------------
@functools.cache
def compile_dav(H: int, v_dtype: torch.dtype, a_dtype: torch.dtype):
    VN = _num(v_dtype)
    AN = _num(a_dtype)

    @fx.struct
    class Smem:
        do: fx.Array[fx.Float32, BT * V, 16]
        vn: fx.Array[fx.Float32, BT * V, 16]
        a: fx.Array[fx.Float32, BT * BT, 16]

    @flyc.kernel(name="kda_bwd_dav_flydsl")
    def kernel(
        vnew: fx.Tensor,
        A: fx.Tensor,
        do: fx.Tensor,
        dA: fx.Tensor,
        dv: fx.Tensor,
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
        t0, nvalid = _chunk(i_t, T)
        tok = fx.Int64(b * T + t0)
        do_v = _flat(do, (tok * H + fx.Int64(h)) * V)
        dv_v = _flat(dv, (tok * H + fx.Int64(h)) * V)
        vn_v = _flat(vnew, (fx.Int64(i_bh) * fx.Int64(T) + fx.Int64(t0)) * V)
        a_v = _flat(A, (tok * H + fx.Int64(h)) * BT)
        da_v = _flat(dA, (tok * H + fx.Int64(h)) * BT)

        lds = fx.SharedAllocator().allocate(Smem).peek()
        s_do = _lds(lds.do, BT * V)
        s_vn = _lds(lds.vn, BT * V)
        s_a = _lds(lds.a, BT * BT)
        _load_tile(do_v, H * V, s_do, BT, V, nvalid, tid, VN, fx.Float32)
        _load_tile(vn_v, V, s_vn, BT, V, nvalid, tid, VN, fx.Float32)
        _load_tile(a_v, H * BT, s_a, BT, BT, nvalid, tid, AN, fx.Float32)
        gpu.barrier()

        # dA[r, j] = scale * sum_v do[r, v] v_new[j, v], j <= r
        for c, st in range(0, V, 1, init=[_zero()] * 16):
            cc = fx.Int32(c)
            d = lds_ld(s_do, r * V + cc)
            acc = [
                st[e] + d * lds_ld(s_vn, (seg + 4 * e) * V + cc)
                for e in range_constexpr(16)
            ]
            res = yield acc
        acc_da = _results(res, 16)
        # dv[r, v] = sum_{t >= r} A[t, r] do[t, v]
        for t, st in range(0, BT, 1, init=[_zero()] * 32):
            tt = fx.Int32(t)
            a = _sel(tt >= r, lds_ld(s_a, tt * BT + r), _zero())
            acc = [
                st[e] + a * lds_ld(s_do, tt * V + seg + 4 * e)
                for e in range_constexpr(32)
            ]
            res = yield acc
        acc_dv = _results(res, 32)
        if r < nvalid:
            for e in range_constexpr(16):
                j = seg + 4 * e
                stg(
                    da_v,
                    r * (H * BT) + j,
                    _sel(j <= r, acc_da[e] * scale, _zero()),
                    fx.Float32,
                )
            for e in range_constexpr(32):
                stg(dv_v, r * (H * V) + seg + 4 * e, acc_dv[e], VN)

    @flyc.jit
    def launch(
        vnew: fx.Tensor,
        A: fx.Tensor,
        do: fx.Tensor,
        dA: fx.Tensor,
        dv: fx.Tensor,
        T: fx.Int32,
        scale: fx.Float32,
        NT: fx.Int32,
        NBH: fx.Int32,
        stream: fx.Stream,
    ):
        kernel(vnew, A, do, dA, dv, T, scale).launch(
            grid=(NT, NBH, 1), block=(NUM_THREADS, 1, 1), stream=stream
        )

    return launch


# ---------------------------------------------------------------------------
# dhu
# ---------------------------------------------------------------------------
@functools.cache
def compile_dhu(
    H: int,
    k_dtype: torch.dtype,
    v_dtype: torch.dtype,
    use_initial_state: bool,
    use_final_state_gradient: bool,
    transpose_state: bool,
):
    KN = _num(k_dtype)
    VN = _num(v_dtype)
    NE = BV * K // NUM_THREADS  # 16 state cells per thread

    @fx.struct
    class Smem:
        kg: fx.Array[fx.Float32, BT * K, 16]
        qg: fx.Array[fx.Float32, BT * K, 16]
        w: fx.Array[fx.Float32, BT * K, 16]
        dh: fx.Array[fx.Float32, K * BV, 16]
        do: fx.Array[fx.Float32, BT * BV, 16]
        dv: fx.Array[fx.Float32, BT * BV, 16]
        dv2: fx.Array[fx.Float32, BT * BV, 16]
        gl: fx.Array[fx.Float32, K, 16]

    @flyc.kernel(name="kda_bwd_dhu_flydsl")
    def kernel(
        qg: fx.Tensor,
        kg: fx.Tensor,
        w: fx.Tensor,
        g: fx.Tensor,
        do: fx.Tensor,
        dv: fx.Tensor,
        dht: fx.Tensor,
        dh: fx.Tensor,
        dh0: fx.Tensor,
        dv2: fx.Tensor,
        T: fx.Int32,
        scale: fx.Float32,
    ):
        i_v = fx.block_idx.x
        i_bh = fx.block_idx.y
        b = i_bh // H
        h = i_bh % H
        tid = fx.thread_idx.x
        kr = tid // 2  # state row (K)
        sc = tid % 2
        r = tid // 4  # chunk row for dv2
        seg = tid % 4
        NT = (T + (BT - 1)) // BT
        v0 = i_v * BV

        lds = fx.SharedAllocator().allocate(Smem).peek()
        s_kg = _lds(lds.kg, BT * K)
        s_qg = _lds(lds.qg, BT * K)
        s_w = _lds(lds.w, BT * K)
        s_dh = _lds(lds.dh, K * BV)
        s_do = _lds(lds.do, BT * BV)
        s_dv = _lds(lds.dv, BT * BV)
        s_dv2 = _lds(lds.dv2, BT * BV)
        s_gl = _lds(lds.gl, K)

        state_v = _flat(dht, fx.Int64(i_bh) * (K * V))
        h0_v = _flat(dh0, fx.Int64(i_bh) * (K * V))

        def state_idx(kk, vv):
            return vv * K + kk if transpose_state else kk * V + vv

        init = []
        for e in range_constexpr(NE):
            if const_expr(use_final_state_gradient):
                init.append(ldg(state_v, state_idx(kr, v0 + sc + 2 * e), fx.Float32))
            else:
                init.append(_zero())

        for i, st in range(0, NT, 1, init=init):
            it = NT - 1 - fx.Int32(i)
            t0, nvalid = _chunk(it, T)
            tok = fx.Int64(b * T + t0)
            hk = (tok * H + fx.Int64(h)) * K
            hv = (tok * H + fx.Int64(h)) * V
            gpu.barrier()
            dh_v = _flat(dh, ((fx.Int64(b * NT + it)) * H + fx.Int64(h)) * (K * V))
            for e in range_constexpr(NE):
                c = sc + 2 * e
                stg(dh_v, kr * V + v0 + c, st[e], KN)
                lds_st(s_dh, kr * BV + c, st[e])
            _load_tile(_flat(kg, hk), H * K, s_kg, BT, K, nvalid, tid, KN, fx.Float32)
            _load_tile(_flat(qg, hk), H * K, s_qg, BT, K, nvalid, tid, KN, fx.Float32)
            _load_tile(_flat(w, hk), H * K, s_w, BT, K, nvalid, tid, KN, fx.Float32)
            _load_tile(
                _flat(do, hv + fx.Int64(v0)),
                H * V,
                s_do,
                BT,
                BV,
                nvalid,
                tid,
                VN,
                fx.Float32,
            )
            _load_tile(
                _flat(dv, hv + fx.Int64(v0)),
                H * V,
                s_dv,
                BT,
                BV,
                nvalid,
                tid,
                VN,
                fx.Float32,
            )
            if tid < K:
                gl = ldg(_flat(g, hk), (nvalid - 1) * (H * K) + tid, fx.Float32)
                lds_st(s_gl, tid, gl)
            gpu.barrier()

            # dv2 = dv + kg @ dh
            for kk, st2 in range(0, K, 1, init=[_zero()] * 8):
                kq = fx.Int32(kk)
                a = lds_ld(s_kg, r * K + kq)
                acc = [
                    st2[e] + a * lds_ld(s_dh, kq * BV + seg + 4 * e)
                    for e in range_constexpr(8)
                ]
                res2 = yield acc
            acc2 = _results(res2, 8)
            dv2_v = _flat(dv2, hv + fx.Int64(v0))
            for e in range_constexpr(8):
                c = seg + 4 * e
                val = acc2[e] + lds_ld(s_dv, r * BV + c)
                lds_st(s_dv2, r * BV + c, val)
                if r < nvalid:
                    stg(dv2_v, r * (H * V) + c, val, VN)
            gpu.barrier()

            # dh = dh * exp2(g_last) + scale * qg^T do - w^T dv2
            decay = _exp2(lds_ld(s_gl, kr))
            for t, st3 in range(0, BT, 1, init=[_zero()] * NE):
                tt = fx.Int32(t)
                q_ = lds_ld(s_qg, tt * K + kr) * scale
                w_ = lds_ld(s_w, tt * K + kr)
                acc = [
                    st3[e]
                    + q_ * lds_ld(s_do, tt * BV + sc + 2 * e)
                    - w_ * lds_ld(s_dv2, tt * BV + sc + 2 * e)
                    for e in range_constexpr(NE)
                ]
                res3 = yield acc
            acc3 = _results(res3, NE)
            res = yield [st[e] * decay + acc3[e] for e in range_constexpr(NE)]
        final = _results(res, NE)
        if const_expr(use_initial_state):
            for e in range_constexpr(NE):
                stg(h0_v, state_idx(kr, v0 + sc + 2 * e), final[e], fx.Float32)

    @flyc.jit
    def launch(
        qg: fx.Tensor,
        kg: fx.Tensor,
        w: fx.Tensor,
        g: fx.Tensor,
        do: fx.Tensor,
        dv: fx.Tensor,
        dht: fx.Tensor,
        dh: fx.Tensor,
        dh0: fx.Tensor,
        dv2: fx.Tensor,
        T: fx.Int32,
        scale: fx.Float32,
        NBH: fx.Int32,
        stream: fx.Stream,
    ):
        kernel(qg, kg, w, g, do, dv, dht, dh, dh0, dv2, T, scale).launch(
            grid=(V // BV, NBH, 1), block=(NUM_THREADS, 1, 1), stream=stream
        )

    return launch


# ---------------------------------------------------------------------------
# wy_dqkg
# ---------------------------------------------------------------------------
@functools.cache
def compile_wy_dqkg(
    H: int,
    k_dtype: torch.dtype,
    v_dtype: torch.dtype,
    beta_dtype: torch.dtype,
    a_dtype: torch.dtype,
    h_dtype: torch.dtype,
):
    KN = _num(k_dtype)
    VN = _num(v_dtype)
    BN = _num(beta_dtype)
    AN = _num(a_dtype)
    HN = _num(h_dtype)

    @fx.struct
    class Smem:
        h: fx.Array[fx.BFloat16, K * V, 16]  # h[k][v], later kg[t][k]
        dh: fx.Array[fx.BFloat16, K * V, 16]  # dh[k][v], later dw[t][k]
        do: fx.Array[fx.BFloat16, BT * V, 16]
        vn: fx.Array[fx.BFloat16, BT * V, 16]  # v_new, later v
        dv: fx.Array[fx.BFloat16, BT * V, 16]
        a: fx.Array[fx.Float32, BT * BT, 16]
        m: fx.Array[fx.Float32, BT * BT, 16]
        beta: fx.Array[fx.Float32, BT, 16]
        dgk: fx.Array[fx.Float32, K, 16]
        col: fx.Array[fx.Float32, 4 * K, 16]

    BF = fx.BFloat16

    @flyc.kernel(name="kda_bwd_wy_dqkg_flydsl")
    def kernel(
        q: fx.Tensor,
        k: fx.Tensor,
        v: fx.Tensor,
        vnew: fx.Tensor,
        g: fx.Tensor,
        beta: fx.Tensor,
        A: fx.Tensor,
        hs: fx.Tensor,
        do: fx.Tensor,
        dh: fx.Tensor,
        dv: fx.Tensor,
        dq: fx.Tensor,
        dk: fx.Tensor,
        dv_out: fx.Tensor,
        dg: fx.Tensor,
        db: fx.Tensor,
        dA: fx.Tensor,
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
        lane = tid % 64
        wave = tid // 64
        NT = (T + (BT - 1)) // BT
        t0, nvalid = _chunk(i_t, T)
        last = nvalid - 1
        rv = r < nvalid
        tok = fx.Int64(b * T + t0)
        hk = (tok * H + fx.Int64(h)) * K
        hv = (tok * H + fx.Int64(h)) * V
        state = (fx.Int64(b * NT + i_t) * H + fx.Int64(h)) * (K * V)

        lds = fx.SharedAllocator().allocate(Smem).peek()
        s_h = _lds(lds.h, K * V)
        s_dh = _lds(lds.dh, K * V)
        s_do = _lds(lds.do, BT * V)
        s_vn = _lds(lds.vn, BT * V)
        s_dv = _lds(lds.dv, BT * V)
        s_a = _lds(lds.a, BT * BT)
        s_m = _lds(lds.m, BT * BT)
        s_beta = _lds(lds.beta, BT)
        s_dgk = _lds(lds.dgk, K)
        s_col = _lds(lds.col, 4 * K)

        # h is the K5 snapshot [V][K]; dh is [K][V].
        h_v = _flat(hs, state)
        dh_v = _flat(dh, state)
        for e in range_constexpr(K * V // NUM_THREADS):
            i = tid + e * NUM_THREADS
            kk = i // V
            vv = i % V
            lds_st(s_h, i, ldg(h_v, vv * K + kk, HN), BF)
            lds_st(s_dh, i, ldg(dh_v, i, HN), BF)
        _load_tile(_flat(do, hv), H * V, s_do, BT, V, nvalid, tid, VN, BF)
        _load_tile(
            _flat(vnew, (fx.Int64(i_bh) * fx.Int64(T) + fx.Int64(t0)) * V),
            V,
            s_vn,
            BT,
            V,
            nvalid,
            tid,
            VN,
            BF,
        )
        _load_tile(_flat(dv, hv), H * V, s_dv, BT, V, nvalid, tid, VN, BF)
        a_off = (tok * H + fx.Int64(h)) * BT
        _load_tile(_flat(A, a_off), H * BT, s_a, BT, BT, nvalid, tid, AN, fx.Float32)
        beta_v = _flat(beta, tok * H + fx.Int64(h))
        if tid < BT:
            bb = ldg(beta_v, _sel(tid < nvalid, tid, 0) * H, BN)
            lds_st(s_beta, tid, _sel(tid < nvalid, bb, _zero()))
        gpu.barrier()

        # dgk[k] = sum_v h[k, v] dh[k, v]
        if tid < K:
            for vv, st in range(0, V, 1, init=[_zero()]):
                vq = fx.Int32(vv)
                res = yield [
                    st[0]
                    + lds_ld(s_h, tid * V + vq, BF) * lds_ld(s_dh, tid * V + vq, BF)
                ]
            lds_st(s_dgk, tid, _results(res, 1)[0])

        # dq = do h^T, dk = v_new dh^T, dw = dv h^T  (per row r, columns k_e)
        for vv, st in range(0, V, 1, init=[_zero()] * 96):
            vq = fx.Int32(vv)
            d_o = lds_ld(s_do, r * V + vq, BF)
            d_n = lds_ld(s_vn, r * V + vq, BF)
            d_v = lds_ld(s_dv, r * V + vq, BF)
            acc = []
            for e in range_constexpr(32):
                kk = seg + 4 * e
                hh = lds_ld(s_h, kk * V + vq, BF)
                acc.append(st[e] + d_o * hh)
            for e in range_constexpr(32):
                kk = seg + 4 * e
                acc.append(st[32 + e] + d_n * lds_ld(s_dh, kk * V + vq, BF))
            for e in range_constexpr(32):
                kk = seg + 4 * e
                acc.append(st[64 + e] + d_v * lds_ld(s_h, kk * V + vq, BF))
            res = yield acc
        acc = _results(res, 96)
        b_dq, b_dk, b_dw = acc[:32], acc[32:64], acc[64:]
        gpu.barrier()

        # v replaces v_new in LDS.
        _load_tile(_flat(v, hv), H * V, s_vn, BT, V, nvalid, tid, VN, BF)
        gpu.barrier()

        # dA = dv v^T ; dvb = A^T dv
        for vv, st in range(0, V, 1, init=[_zero()] * 16):
            vq = fx.Int32(vv)
            d_v = lds_ld(s_dv, r * V + vq, BF)
            res = yield [
                st[e] + d_v * lds_ld(s_vn, (seg + 4 * e) * V + vq, BF)
                for e in range_constexpr(16)
            ]
        b_da = _results(res, 16)
        for s, st in range(0, BT, 1, init=[_zero()] * 32):
            sq = fx.Int32(s)
            a = lds_ld(s_a, sq * BT + r)
            res = yield [
                st[e] + a * lds_ld(s_dv, sq * V + seg + 4 * e, BF)
                for e in range_constexpr(32)
            ]
        dvb = _results(res, 32)
        beta_r = lds_ld(s_beta, r)
        dvo_v = _flat(dv_out, hv)
        db_r = _zero()
        for e in range_constexpr(32):
            c = seg + 4 * e
            db_r = db_r + dvb[e] * lds_ld(s_vn, r * V + c, BF)
            if rv:
                stg(dvo_v, r * (H * V) + c, dvb[e] * beta_r, VN)

        q_v = _flat(q, hk)
        k_v = _flat(k, hk)
        g_v = _flat(g, hk)
        rc = _sel(rv, r, 0)
        gexp = []
        kg_r = []
        k_r = []
        gn = []
        for e in range_constexpr(32):
            kk = seg + 4 * e
            gg = ldg(g_v, rc * (H * K) + kk, fx.Float32)
            gl = ldg(g_v, last * (H * K) + kk, fx.Float32)
            kx = _sel(rv, ldg(k_v, rc * (H * K) + kk, KN), _zero())
            ge = _exp2(gg)
            gexp.append(ge)
            gn.append(gl)
            k_r.append(kx)
            kg_r.append(kx * ge)
            b_dq[e] = b_dq[e] * ge * scale
            b_dk[e] = _sel(rv, b_dk[e] * _exp2(gl - gg), _zero())
            b_dw[e] = -b_dw[e]
        gpu.barrier()
        for e in range_constexpr(32):
            kk = seg + 4 * e
            lds_st(s_h, r * K + kk, kg_r[e], BF)
            lds_st(s_dh, r * K + kk, b_dw[e], BF)
        gpu.barrier()

        # dA += dw kg^T ; dkgb = A^T dw
        for kk, st in range(0, K, 1, init=b_da):
            kq = fx.Int32(kk)
            w_ = lds_ld(s_dh, r * K + kq, BF)
            res = yield [
                st[e] + w_ * lds_ld(s_h, (seg + 4 * e) * K + kq, BF)
                for e in range_constexpr(16)
            ]
        b_da = _results(res, 16)
        for s, st in range(0, BT, 1, init=[_zero()] * 32):
            sq = fx.Int32(s)
            a = lds_ld(s_a, sq * BT + r)
            res = yield [
                st[e] + a * lds_ld(s_dh, sq * K + seg + 4 * e, BF)
                for e in range_constexpr(32)
            ]
        dkgb = _results(res, 32)

        kdk = []
        for e in range_constexpr(32):
            db_r = db_r + dkgb[e] * kg_r[e]
            kdk.append(k_r[e] * b_dk[e])
        # Column sums of k * dk: lanes sharing seg in a wave, then the 4 waves.
        for e in range_constexpr(32):
            x = kdk[e]
            for sh in (4, 8, 16, 32):
                x = x + x.shuffle_xor(fx.Int32(sh), fx.Int32(64))
            if lane < 4:
                lds_st(s_col, wave * K + seg + 4 * e, x)
        gpu.barrier()

        dq_v = _flat(dq, hk)
        dk_v = _flat(dk, hk)
        dg_v = _flat(dg, hk)
        is_last = r == last
        for e in range_constexpr(32):
            kk = seg + 4 * e
            col = (
                lds_ld(s_col, kk)
                + lds_ld(s_col, K + kk)
                + lds_ld(s_col, 2 * K + kk)
                + lds_ld(s_col, 3 * K + kk)
            )
            dgk = lds_ld(s_dgk, kk) * _exp2(gn[e]) + col
            qx = _sel(rv, ldg(q_v, rc * (H * K) + kk, KN), _zero())
            dgv = (
                qx * b_dq[e]
                - kdk[e]
                + _sel(is_last, dgk, _zero())
                + kg_r[e] * dkgb[e] * beta_r
            )
            dkv = b_dk[e] + dkgb[e] * gexp[e] * beta_r
            if rv:
                stg(dq_v, r * (H * K) + kk, b_dq[e], fx.Float32)
                stg(dk_v, r * (H * K) + kk, dkv, fx.Float32)
                stg(dg_v, r * (H * K) + kk, dgv, fx.Float32)
        db_r = _row_sum(db_r)
        if rv & (seg == 0):
            stg(_flat(db, tok * H + fx.Int64(h)), r * H, db_r, fx.Float32)

        # dAkk = -mask(A^T (mask(dA * beta) A^T)), strictly lower.
        for e in range_constexpr(16):
            s = seg + 4 * e
            m_a = (s < r) & rv
            lds_st(s_m, r * BT + s, _sel(m_a, b_da[e] * lds_ld(s_beta, s), _zero()))
        gpu.barrier()
        for j, st in range(0, BT, 1, init=[_zero()] * 16):
            jq = fx.Int32(j)
            m_ = lds_ld(s_m, r * BT + jq)
            res = yield [
                st[e] + m_ * lds_ld(s_a, (seg + 4 * e) * BT + jq)
                for e in range_constexpr(16)
            ]
        m1 = _results(res, 16)
        gpu.barrier()
        for e in range_constexpr(16):
            lds_st(s_m, r * BT + seg + 4 * e, m1[e])
        gpu.barrier()
        for j, st in range(0, BT, 1, init=[_zero()] * 16):
            jq = fx.Int32(j)
            a = lds_ld(s_a, jq * BT + r)
            res = yield [
                st[e] + a * lds_ld(s_m, jq * BT + seg + 4 * e)
                for e in range_constexpr(16)
            ]
        m2 = _results(res, 16)
        da_v = _flat(dA, a_off)
        if rv:
            for e in range_constexpr(16):
                s = seg + 4 * e
                stg(da_v, r * (H * BT) + s, _sel(s < r, -m2[e], _zero()), fx.Float32)

    @flyc.jit
    def launch(
        q: fx.Tensor,
        k: fx.Tensor,
        v: fx.Tensor,
        vnew: fx.Tensor,
        g: fx.Tensor,
        beta: fx.Tensor,
        A: fx.Tensor,
        hs: fx.Tensor,
        do: fx.Tensor,
        dh: fx.Tensor,
        dv: fx.Tensor,
        dq: fx.Tensor,
        dk: fx.Tensor,
        dv_out: fx.Tensor,
        dg: fx.Tensor,
        db: fx.Tensor,
        dA: fx.Tensor,
        T: fx.Int32,
        scale: fx.Float32,
        NT: fx.Int32,
        NBH: fx.Int32,
        stream: fx.Stream,
    ):
        kernel(
            q,
            k,
            v,
            vnew,
            g,
            beta,
            A,
            hs,
            do,
            dh,
            dv,
            dq,
            dk,
            dv_out,
            dg,
            db,
            dA,
            T,
            scale,
        ).launch(grid=(NT, NBH, 1), block=(NUM_THREADS, 1, 1), stream=stream)

    return launch


# ---------------------------------------------------------------------------
# intra
# ---------------------------------------------------------------------------
@functools.cache
def compile_intra(
    H: int,
    k_dtype: torch.dtype,
    beta_dtype: torch.dtype,
    raw_k_dtype: torch.dtype,
    raw_beta_dtype: torch.dtype,
    out_k_dtype: torch.dtype,
    out_beta_dtype: torch.dtype,
    l2norm_qk: bool,
    beta_sigmoid: bool,
):
    KN = _num(k_dtype)
    BN = _num(beta_dtype)
    RKN = _num(raw_k_dtype)
    RBN = _num(raw_beta_dtype)
    OKN = _num(out_k_dtype)
    OBN = _num(out_beta_dtype)
    EPS = 1e-6

    @fx.struct
    class Smem:
        q: fx.Array[fx.Float32, BT * K, 16]
        k: fx.Array[fx.Float32, BT * K, 16]
        g: fx.Array[fx.Float32, BT * K, 16]
        aq: fx.Array[fx.Float32, BT * BT, 16]
        ak: fx.Array[fx.Float32, BT * BT, 16]
        beta: fx.Array[fx.Float32, BT, 16]

    def l2norm_bwd(x, dy):
        """``dx`` for ``y = x / sqrt(sum(x^2) + eps)`` along this thread's row."""
        ss = _zero()
        for e in range_constexpr(32):
            ss = ss + x[e] * x[e]
        rstd = fx.rsqrt(_row_sum(ss) + EPS)
        y = [x[e] * rstd for e in range_constexpr(32)]
        sdy = _zero()
        for e in range_constexpr(32):
            sdy = sdy + dy[e] * y[e]
        sdy = _row_sum(sdy)
        return [dy[e] * rstd - sdy * y[e] * rstd for e in range_constexpr(32)]

    @flyc.kernel(name="kda_bwd_intra_flydsl")
    def kernel(
        q: fx.Tensor,
        k: fx.Tensor,
        g: fx.Tensor,
        beta: fx.Tensor,
        dAqk: fx.Tensor,
        dAkk: fx.Tensor,
        dq: fx.Tensor,
        dk: fx.Tensor,
        dg: fx.Tensor,
        db: fx.Tensor,
        q_raw: fx.Tensor,
        k_raw: fx.Tensor,
        beta_raw: fx.Tensor,
        dq2: fx.Tensor,
        dk2: fx.Tensor,
        dg2: fx.Tensor,
        db2: fx.Tensor,
        T: fx.Int32,
    ):
        i_t = fx.block_idx.x
        i_bh = fx.block_idx.y
        b = i_bh // H
        h = i_bh % H
        tid = fx.thread_idx.x
        r = tid // 4
        seg = tid % 4
        t0, nvalid = _chunk(i_t, T)
        rv = r < nvalid
        rc = _sel(rv, r, 0)
        tok = fx.Int64(b * T + t0)
        hk = (tok * H + fx.Int64(h)) * K
        hb = tok * H + fx.Int64(h)

        lds = fx.SharedAllocator().allocate(Smem).peek()
        s_q = _lds(lds.q, BT * K)
        s_k = _lds(lds.k, BT * K)
        s_g = _lds(lds.g, BT * K)
        s_aq = _lds(lds.aq, BT * BT)
        s_ak = _lds(lds.ak, BT * BT)
        s_b = _lds(lds.beta, BT)
        _load_tile(_flat(q, hk), H * K, s_q, BT, K, nvalid, tid, KN, fx.Float32)
        _load_tile(_flat(k, hk), H * K, s_k, BT, K, nvalid, tid, KN, fx.Float32)
        _load_tile(_flat(g, hk), H * K, s_g, BT, K, nvalid, tid, fx.Float32, fx.Float32)
        _load_tile(
            _flat(dAqk, hb * BT),
            H * BT,
            s_aq,
            BT,
            BT,
            nvalid,
            tid,
            fx.Float32,
            fx.Float32,
        )
        _load_tile(
            _flat(dAkk, hb * BT),
            H * BT,
            s_ak,
            BT,
            BT,
            nvalid,
            tid,
            fx.Float32,
            fx.Float32,
        )
        if tid < BT:
            bb = ldg(_flat(beta, hb), _sel(tid < nvalid, tid, 0) * H, BN)
            lds_st(s_b, tid, _sel(tid < nvalid, bb, _zero()))
        gpu.barrier()

        g_r = [lds_ld(s_g, r * K + seg + 4 * e) for e in range_constexpr(32)]

        # dq2 / dk2 = sum_{j <= r} dA[r, j] k[j] exp2(g[r] - g[j])
        for j, st in range(0, BT, 1, init=[_zero()] * 64):
            jq = fx.Int32(j)
            ok = jq <= r
            aq = _sel(ok, lds_ld(s_aq, r * BT + jq), _zero())
            ak = _sel(ok, lds_ld(s_ak, r * BT + jq), _zero())
            acc_q, acc_k = [], []
            for e in range_constexpr(32):
                kk = seg + 4 * e
                ex = _exp2(_sel(ok, g_r[e] - lds_ld(s_g, jq * K + kk), _zero()))
                kx = lds_ld(s_k, jq * K + kk) * ex
                acc_q.append(st[e] + aq * kx)
                acc_k.append(st[32 + e] + ak * kx)
            res = yield acc_q + acc_k
        acc = _results(res, 64)
        b_dq2, b_dk2 = acc[:32], acc[32:]

        k_r = [lds_ld(s_k, r * K + seg + 4 * e) for e in range_constexpr(32)]
        q_r = [lds_ld(s_q, r * K + seg + 4 * e) for e in range_constexpr(32)]
        beta_r = lds_ld(s_b, r)
        db_r = _zero()
        for e in range_constexpr(32):
            db_r = db_r + b_dk2[e] * k_r[e]
        db_r = _row_sum(db_r)
        b_dk2 = [b_dk2[e] * beta_r for e in range_constexpr(32)]

        dq_v = _flat(dq, hk)
        dk_v = _flat(dk, hk)
        dg_v = _flat(dg, hk)
        b_dg2 = [q_r[e] * b_dq2[e] for e in range_constexpr(32)]
        b_dq2 = [
            b_dq2[e] + ldg(dq_v, rc * (H * K) + seg + 4 * e, fx.Float32)
            for e in range_constexpr(32)
        ]
        if const_expr(l2norm_qk):
            qraw_v = _flat(q_raw, hk)
            xr = [
                ldg(qraw_v, rc * (H * K) + seg + 4 * e, RKN)
                for e in range_constexpr(32)
            ]
            b_dq2 = l2norm_bwd(xr, b_dq2)
        dq2_v = _flat(dq2, hk)
        if rv:
            for e in range_constexpr(32):
                stg(dq2_v, r * (H * K) + seg + 4 * e, b_dq2[e], OKN)

        db_r = db_r + ldg(_flat(db, hb), rc * H, fx.Float32)
        if const_expr(beta_sigmoid):
            sgm = _sigmoid(ldg(_flat(beta_raw, hb), rc * H, RBN))
            db_r = db_r * sgm * (fx.Float32(1.0) - sgm)
        if rv & (seg == 0):
            stg(_flat(db2, hb), r * H, db_r, OBN)

        # dkt = sum_{j >= r} (dAqk[j, r] q[j] + dAkk[j, r] beta[j] k[j]) exp2(g[j] - g[r])
        for j, st in range(0, BT, 1, init=[_zero()] * 32):
            jq = fx.Int32(j)
            ok = jq >= r
            aq = _sel(ok, lds_ld(s_aq, jq * BT + r), _zero())
            ak = _sel(ok, lds_ld(s_ak, jq * BT + r) * lds_ld(s_b, jq), _zero())
            acc = []
            for e in range_constexpr(32):
                kk = seg + 4 * e
                ex = _exp2(_sel(ok, lds_ld(s_g, jq * K + kk) - g_r[e], _zero()))
                acc.append(
                    st[e]
                    + (aq * lds_ld(s_q, jq * K + kk) + ak * lds_ld(s_k, jq * K + kk))
                    * ex
                )
            res = yield acc
        dkt = _results(res, 32)

        dk_out = []
        for e in range_constexpr(32):
            kk = seg + 4 * e
            b_dg2[e] = (
                b_dg2[e]
                + (b_dk2[e] - dkt[e]) * k_r[e]
                + ldg(dg_v, rc * (H * K) + kk, fx.Float32)
            )
            dk_out.append(b_dk2[e] + ldg(dk_v, rc * (H * K) + kk, fx.Float32) + dkt[e])
        if const_expr(l2norm_qk):
            kraw_v = _flat(k_raw, hk)
            xr = [
                ldg(kraw_v, rc * (H * K) + seg + 4 * e, RKN)
                for e in range_constexpr(32)
            ]
            dk_out = l2norm_bwd(xr, dk_out)
        dk2_v = _flat(dk2, hk)
        dg2_v = _flat(dg2, hk)
        if rv:
            for e in range_constexpr(32):
                kk = seg + 4 * e
                stg(dk2_v, r * (H * K) + kk, dk_out[e], OKN)
                stg(dg2_v, r * (H * K) + kk, b_dg2[e], fx.Float32)

    @flyc.jit
    def launch(
        q: fx.Tensor,
        k: fx.Tensor,
        g: fx.Tensor,
        beta: fx.Tensor,
        dAqk: fx.Tensor,
        dAkk: fx.Tensor,
        dq: fx.Tensor,
        dk: fx.Tensor,
        dg: fx.Tensor,
        db: fx.Tensor,
        q_raw: fx.Tensor,
        k_raw: fx.Tensor,
        beta_raw: fx.Tensor,
        dq2: fx.Tensor,
        dk2: fx.Tensor,
        dg2: fx.Tensor,
        db2: fx.Tensor,
        T: fx.Int32,
        NT: fx.Int32,
        NBH: fx.Int32,
        stream: fx.Stream,
    ):
        kernel(
            q,
            k,
            g,
            beta,
            dAqk,
            dAkk,
            dq,
            dk,
            dg,
            db,
            q_raw,
            k_raw,
            beta_raw,
            dq2,
            dk2,
            dg2,
            db2,
            T,
        ).launch(grid=(NT, NBH, 1), block=(NUM_THREADS, 1, 1), stream=stream)

    return launch


# ---------------------------------------------------------------------------
# gate_cumsum
# ---------------------------------------------------------------------------
@functools.cache
def compile_gate_cumsum(
    H: int,
    g_dtype: torch.dtype,
    out_dtype: torch.dtype,
    use_gate: bool,
    has_bias: bool,
    use_lower_bound: bool,
):
    GN = _num(g_dtype)
    ON = _num(out_dtype)

    @flyc.kernel(name="kda_bwd_gate_cumsum_flydsl")
    def kernel(
        s: fx.Tensor,
        g: fx.Tensor,
        A_log: fx.Tensor,
        dt_bias: fx.Tensor,
        o: fx.Tensor,
        dA_part: fx.Tensor,
        db_part: fx.Tensor,
        T: fx.Int32,
        lower_bound: fx.Float32,
        NBH: fx.Int32,
    ):
        i_t = fx.block_idx.x
        i_bh = fx.block_idx.y
        b = i_bh // H
        h = i_bh % H
        kk = fx.thread_idx.x
        t0, nvalid = _chunk(i_t, T)
        tok = fx.Int64(b * T + t0)
        hk = (tok * H + fx.Int64(h)) * K
        s_v = _flat(s, hk)
        g_v = _flat(g, hk)
        o_v = _flat(o, hk)
        part = (fx.Int64(i_t) * fx.Int64(NBH) + fx.Int64(i_bh)) * K
        if const_expr(use_gate):
            a_raw = ldg(_flat(A_log, None), h, fx.Float32)
            a = fx.exp(a_raw)
            bias = _zero()
            if const_expr(has_bias):
                bias = ldg(_flat(dt_bias, None), h * K + kk, fx.Float32)

        # Reverse chunk-local cumsum, then the gate activation backward.
        for i, st in range(0, BT, 1, init=[_zero(), _zero(), _zero()]):
            t = fx.Int32(BT - 1) - fx.Int32(i)
            ok = t < nvalid
            tc = _sel(ok, t, 0)
            run = st[0] + _sel(ok, ldg(s_v, tc * (H * K) + kk, fx.Float32), _zero())
            out = run
            da = st[1]
            dbias = st[2]
            if const_expr(use_gate):
                z = ldg(g_v, tc * (H * K) + kk, GN) + bias
                if const_expr(use_lower_bound):
                    sgm = _sigmoid(a * z)
                    dgz = run * (lower_bound * sgm * (fx.Float32(1.0) - sgm)) * a
                    da = da + _sel(ok, dgz * z, _zero())
                else:
                    sp = _sel(
                        z > fx.Float32(20.0),
                        z,
                        fx.log(fx.Float32(1.0) + fx.exp(z)),
                    )
                    dgz = -a * (run * _sigmoid(z))
                    da = da + _sel(ok, run * (-a * sp), _zero())
                dbias = dbias + _sel(ok, dgz, _zero())
                out = dgz
            if ok:
                stg(o_v, t * (H * K) + kk, out, ON)
            res = yield [run, da, dbias]
        fin = _results(res, 3)
        if const_expr(use_gate):
            stg(_flat(dA_part, part), kk, fin[1], fx.Float32)
            if const_expr(has_bias):
                stg(_flat(db_part, part), kk, fin[2], fx.Float32)

    @flyc.jit
    def launch(
        s: fx.Tensor,
        g: fx.Tensor,
        A_log: fx.Tensor,
        dt_bias: fx.Tensor,
        o: fx.Tensor,
        dA_part: fx.Tensor,
        db_part: fx.Tensor,
        T: fx.Int32,
        lower_bound: fx.Float32,
        NT: fx.Int32,
        NBH: fx.Int32,
        stream: fx.Stream,
    ):
        kernel(s, g, A_log, dt_bias, o, dA_part, db_part, T, lower_bound, NBH).launch(
            grid=(NT, NBH, 1), block=(K, 1, 1), stream=stream
        )

    return launch
