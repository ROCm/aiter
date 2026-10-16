# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Single-kernel Gated DeltaNet chunked prefill (gfx950).

One workgroup owns one (sequence, value head, BV-wide slice of V) and walks the
sequence's 64-token chunks in order, keeping its slice of the state on chip.
Per chunk, with gc the in-chunk cumulative log-decay:

    A    = strict_tril(beta_t * k_t.k_j * exp(gc_t - gc_j))      [64, 64]
    Tinv = (I + A)^-1
    vnew = Tinv @ (beta * (v - exp(gc) * k @ S^T))                [64, BV]
    o    = scale * exp(gc) * q @ S^T + (tril(q k^T * exp(gc_t - gc_j))) @ vnew
    S    = exp(gc_last) * S + vnew^T @ (k * exp(gc_last - gc))

The state slice S^T [K, BV] lives in fp32 MFMA accumulators; a bf16 copy in
LDS feeds the next chunk's k @ S^T and q @ S^T. The V-independent work (q/k
l2norm, k k^T, the triangular inverse, the masked q k^T) is recomputed by every
V slice of a head; that is the price of not round-tripping the per-chunk state
through HBM.

Lane conventions (wave of 64, c = lane % 16, gq = lane // 16), for the MFMAs:
  16x16x32 bf16: A[c][8gq + e], B[8gq + e][c]
  16x16x16 bf16 / 16x16x4 f32 (x4): A[c][4gq + e], B[4gq + e][c]
  accumulator C[4gq + i][c]
A 16x16 tile stored "fragment-native" keeps each lane's 4 accumulator values
contiguous at ``lane * 4``; read back by the same lane it is directly the B
operand (and the A operand of the transpose) of the 16x16x16 MFMA.
"""

import functools
import math

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T

from ..kernels_common import LOG2E

BT = 64
KD = 128
VD = 128
# Two wave groups of four (one wave of each per SIMD): group 0 owns the state
# slice and the triangular inverse, group 1 the q-side tiles and the output.
NUM_WAVES = 8
GROUP = 4
WAVE = 64
BLOCK = NUM_WAVES * WAVE
# Row pitch of the [token][K] and [V][K] bf16 LDS tiles; the 8-element pad
# shifts consecutive rows by 4 banks so a 16-row b128 operand read is
# conflict-free.
ROW = KD + 8
FRAG = 256
L2NORM_EPS = 1e-6

N_A_TILES = 6


def _ld(ptr, off, n, dt, atom):
    view = fx.make_view(ptr + off, fx.make_layout(n, 1))
    frag = fx.make_rmem_tensor(n, dt)
    fx.copy(fx.make_copy_atom(atom, dt), view, frag)
    return frag.load()


def _st(ptr, off, vec, n, dt, atom):
    view = fx.make_view(ptr + off, fx.make_layout(n, 1))
    frag = fx.make_rmem_tensor(n, dt)
    frag.store(vec)
    fx.copy(fx.make_copy_atom(atom, dt), frag, view)


def _lds_ld(ptr, off, n, dt):
    return _ld(ptr, off, n, dt, fx.UniversalCopy(n * dt.width))


def _lds_st(ptr, off, vec, n, dt):
    _st(ptr, off, vec, n, dt, fx.UniversalCopy(n * dt.width))


def _g_ld(it, off, n, dt):
    return _ld(it, off, n, dt, fx.rocdl.BufferCopy(n * dt.width))


def _g_st(it, off, vec, n, dt):
    _st(it, off, vec, n, dt, fx.rocdl.BufferCopy(n * dt.width))


def _vec(elems, dt=fx.Float32):
    return fx.Vector.from_elements(list(elems), dtype=dt)


def _zeros4():
    return fx.Vector.filled(4, 0.0, fx.Float32)


def _exp(x):
    """exp via v_exp_f32; callers keep the argument <= 0 or masked."""
    return fx.Float32(rocdl.exp2(T.f32, (x * LOG2E).ir_value()))


def _mfma32(a8, b8, c4):
    return fx.Vector(rocdl.mfma_f32_16x16x32_bf16(T.vec(4, T.f32), [a8, b8, c4]))


def _mfma16(a4, b4, c4):
    return fx.Vector(
        rocdl.mfma_f32_16x16x16bf16_1k(
            T.vec(4, T.f32),
            [fx.Vector(a4).bitcast(fx.Int16), fx.Vector(b4).bitcast(fx.Int16), c4],
        )
    )


def _split_bf16(v4):
    v = fx.Vector(v4)
    hi = v.to(fx.BFloat16)
    lo = (v - hi.to(fx.Float32)).to(fx.BFloat16)
    return hi, lo


def _mm16_x3(a, b, c4):
    """16x16 product to ~2^-16 relative error from three bf16 MFMAs; ``a`` and
    ``b`` are (hi, lo) splits."""
    acc = _mfma16(a[1], b[0], c4)
    acc = _mfma16(a[0], b[1], acc)
    return _mfma16(a[0], b[0], acc)


def _dpp_i32(src, ctrl, row_mask, bank_mask, bound_ctrl=False, old=0):
    from flydsl._mlir.dialects import llvm as _llvm

    return fx.Int32(
        _llvm.call_intrinsic(
            fx.Int32.ir_type,
            "llvm.amdgcn.update.dpp.i32",
            [
                fx.Int32(old).ir_value(),
                fx.Int32(src).ir_value(),
                fx.Int32(ctrl).ir_value(),
                fx.Int32(row_mask).ir_value(),
                fx.Int32(bank_mask).ir_value(),
                fx.Boolean(bound_ctrl).ir_value(),
            ],
            [],
            [],
        )
    )


def _dpp_f32(x, ctrl, row_mask=0xF, bank_mask=0xF):
    """DPP move of ``x``; lanes with no valid source read 0."""
    return _dpp_i32(fx.Float32(x).bitcast(fx.Int32), ctrl, row_mask, bank_mask).bitcast(
        fx.Float32
    )


def _bf16(v4):
    return fx.Vector(v4).to(fx.BFloat16)


def _diag_inverse(aa, ab, ident):
    """(I + A)^-1 of a strictly lower 16x16 block via (I-A)(I+A^2)(I+A^4)(I+A^8).

    ``aa`` / ``ab`` hold A in the A- / B-operand layout; the result is the
    accumulator layout. Products are split-bf16, close to fp32.
    """
    naa = _split_bf16(-aa)
    nab = _split_bf16(-ab)
    z = _zeros4()
    b2 = _split_bf16(_mm16_x3(naa, nab, z))
    b2t = _split_bf16(_mm16_x3(nab, naa, z))
    b4 = _split_bf16(_mm16_x3(b2t, b2, z))
    b4t = _split_bf16(_mm16_x3(b2, b2t, z))
    acc = _mm16_x3(b4t, b4, ident)
    acc = _mm16_x3(b4t, _split_bf16(acc), acc)
    acc = _mm16_x3(b2t, _split_bf16(acc), acc)
    return _mm16_x3(naa, _split_bf16(acc), acc)


def _exp2(x):
    return fx.Float32(rocdl.exp2(T.f32, fx.Float32(x).ir_value()))


def _log2(x):
    return fx.Float32(rocdl.log(T.f32, fx.Float32(x).ir_value()))


def _rsq(x):
    return fx.Float32(rocdl.rsq(T.f32, fx.Float32(x).ir_value()))


def _fdot2(a2, b2, c):
    return fx.Float32(
        rocdl.fdot2_f32_bf16(T.f32, a2.ir_value(), b2.ir_value(), c.ir_value()).result
    )


def _sumsq_bf16x8(v8, acc):
    """acc + sum(v8 ** 2) for a bf16x8 vector, via v_dot2_f32_bf16."""
    for p in range(4):
        pair = v8.shuffle(v8, [2 * p, 2 * p + 1])
        acc = _fdot2(pair, pair, acc)
    return acc


def _row_sum8(x):
    """Sum over each aligned group of 8 lanes (DPP quad_perm, then row_shr/shl 4)."""
    s = x + _dpp_f32(x, 0xB1)  # quad_perm [1, 0, 3, 2]
    s = s + _dpp_f32(s, 0x4E)  # quad_perm [2, 3, 0, 1]
    si = fx.Float32(s).bitcast(fx.Int32)
    x4 = _dpp_i32(si, 0x114, 0xF, 0xA, old=si)  # row_shr:4 into banks 1, 3
    x4 = _dpp_i32(si, 0x104, 0xF, 0x5, old=x4)  # row_shl:4 into banks 0, 2
    return s + x4.bitcast(fx.Float32)


def _wave_scan(x, lane):
    """Inclusive wave64 prefix sum (gfx9 DPP row_shr + row_bcast)."""
    s = x
    for i in range(4):
        s = s + _dpp_f32(s, 0x110 + (1 << i))  # row_shr
    s = s + _dpp_f32(s, 0x142, row_mask=0xA)  # row_bcast:15
    return s + _dpp_f32(s, 0x143, row_mask=0xC)  # row_bcast:31


def _lds_ld_tr(ptr, off):
    """ds_read_b64_tr_b16: in each 16-lane group, lane j addresses 4 b16 of row j;
    lane i receives element i % 4 of the rows addressed by lanes 4k + i // 4."""
    return _ld(ptr, off, 4, fx.BFloat16, fx.rocdl.cdna4.LDSReadTrans16_64b())


def _sel(cond, a, b):
    return cond.select(a, b)


def _sel_by_wave(w, values):
    """Pick ``values[w]`` for a runtime, wave-uniform ``w``."""
    out = fx.Int32(values[-1])
    for i in range(len(values) - 2, -1, -1):
        out = (w == i).select(fx.Int32(values[i]), out)
    return out


# Per-token factor rows in the ``fac`` LDS array (each BT fp32), all derived
# from the in-chunk log2 decay gl, the L2-norm reciprocals rq/rk and beta.
F_EA = 0  # gl + log2(beta * rk): row exponent of A
F_EB = 1  # gl - log2(rk): column exponent of A and P^T
F_EQ = 2  # gl + log2(scale * rq): row exponent of P (= column of P^T)
F_BETA = 3
F_BEGRK = 4  # beta * exp(gc) * rk
F_EGQ = 5  # scale * exp(gc) * rq
F_EDRK = 6  # exp(gc_last - gc) * rk
F_EG = 7  # exp(gc)
N_FAC = 8

# Strictly-lower off-diagonal 16x16 blocks (a > b) of a 64x64 chunk matrix:
# group wave wl takes OFF0[wl], and OFF1[wl] when wl < 2.
OFF0 = ((1, 0), (2, 1), (3, 1), (3, 2))
OFF1 = ((2, 0), (3, 0))
ZERO_TILE = N_A_TILES + 10
N_AP_TILES = ZERO_TILE + 1


@functools.lru_cache(maxsize=32)
def compile_gdn_fused_prefill(
    *,
    H: int,
    Hg: int,
    BV: int = 32,
    HAS_TRACK: bool = False,
    scale: float = KD**-0.5,
):
    """Compile the fused prefill; returns the @flyc.jit launcher.

    Launch grid is (V // BV, N * H); 512 threads per workgroup.
    """
    assert H % Hg == 0
    assert BV % 16 == 0 and VD % BV == 0
    NVB = BV // 16
    REP = H // Hg
    LOG2_SCALE = math.log2(scale)
    # v staging: thread (r_row, sub) loads V_EPT contiguous values of its token.
    V_EPT = BV // 8
    VROW = BV + 8

    @fx.struct
    class SharedStorage:
        # q tile, then two k tiles: chunk c's k is still read (transposed) by the
        # state update while chunk c + 1 is being staged.
        qk: fx.Array[fx.BFloat16, 3 * BT * ROW, 16]
        s: fx.Array[fx.BFloat16, BV * ROW, 16]
        ap: fx.Array[fx.BFloat16, N_AP_TILES * FRAG, 16]
        td: fx.Array[fx.BFloat16, GROUP * FRAG, 16]
        rhs: fx.Array[fx.BFloat16, GROUP * NVB * FRAG, 16]
        vn: fx.Array[fx.BFloat16, GROUP * NVB * FRAG, 16]
        vd: fx.Array[fx.BFloat16, GROUP * NVB * FRAG, 16]
        fac: fx.Array[fx.Float32, N_FAC * BT, 16]
        vt: fx.Array[fx.BFloat16, BT * VROW, 16]

    @flyc.kernel(name=f"gdn_fused_prefill_bv{BV}_h{H}_hg{Hg}_t{int(HAS_TRACK)}")
    def gdn_fused_kernel(
        q_t: fx.Tensor,
        k_t: fx.Tensor,
        v_t: fx.Tensor,
        g_t: fx.Tensor,
        beta_t: fx.Tensor,
        cu_t: fx.Tensor,
        idx_t: fx.Tensor,
        pool_t: fx.Tensor,
        o_t: fx.Tensor,
        trs_t: fx.Tensor,
        tri_t: fx.Tensor,
        q_st: fx.Int32,
        k_st: fx.Int32,
        v_st: fx.Int32,
        g_st: fx.Int32,
        b_st: fx.Int32,
        T_total: fx.Int32,
    ):
        tid = fx.Int32(fx.thread_idx.x)
        # Wave index as a scalar so per-wave selects and addresses stay in SGPRs.
        w = fx.Int32(rocdl.readfirstlane(T.i32, (tid // WAVE).ir_value()))
        wl = w % GROUP
        wh = w // GROUP
        lane = tid % WAVE
        c = lane % 16
        gq = lane // 16
        i_v = fx.Int32(fx.block_idx.x)
        i_nh = fx.Int32(fx.block_idx.y)
        i_n = i_nh // H
        i_h = i_nh % H
        i_hg = i_h // REP

        bos = fx.Int32(cu_t[i_n])
        eos = fx.Int32(cu_t[i_n + 1])
        T_local = eos - bos
        NT = (T_local + (BT - 1)) // BT
        slot = fx.Int32(idx_t[i_n])
        slot = (slot < 0).select(fx.Int32(0), slot)
        last_tok = (T_total - 1 < 0).select(fx.Int32(0), T_total - 1)

        def _view(tensor, base, n, records=None):
            view = fx.make_view(fx.get_iter(tensor) + base, fx.make_layout(n, 1))
            if records is None:
                return fx.get_iter(fx.rocdl.make_buffer_tensor(view))
            return fx.get_iter(
                fx.rocdl.make_buffer_tensor(view, num_records_bytes=records)
            )

        bos64 = fx.Int64(bos)
        q_it = _view(q_t, fx.Int64(i_hg) * KD, 1)
        k_it = _view(k_t, fx.Int64(i_hg) * KD, 1)
        v_it = _view(v_t, fx.Int64(i_h) * VD + fx.Int64(i_v) * BV, 1)
        g_it = _view(g_t, fx.Int64(i_h), 1)
        b_it = _view(beta_t, fx.Int64(i_h), 1)
        state_base = (fx.Int64(slot) * H + fx.Int64(i_h)) * (VD * KD) + fx.Int64(
            i_v
        ) * (BV * KD)
        pool_it = _view(pool_t, state_base, BV * KD)
        o_rec = fx.Int64(T_local) * (H * VD * 2)
        o_it = _view(
            o_t, (bos64 * H + fx.Int64(i_h)) * VD + fx.Int64(i_v) * BV, 1, o_rec
        )
        if const_expr(HAS_TRACK):
            trk = fx.Int32(tri_t[i_n])
            trs_base = (fx.Int64(i_n) * H + fx.Int64(i_h)) * (VD * KD) + fx.Int64(
                i_v
            ) * (BV * KD)
            trs_it = _view(trs_t, trs_base, BV * KD)

        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        p_qk = lds.qk.ptr
        p_s = lds.s.ptr
        p_ap = lds.ap.ptr
        p_td = lds.td.ptr
        p_rhs = lds.rhs.ptr
        p_vn = lds.vn.ptr
        p_vd = lds.vd.ptr
        p_f = lds.fac.ptr
        p_vt = lds.vt.ptr

        def _fac4(row, tok):
            return _lds_ld(p_f, row * BT + tok, 4, fx.Float32)

        def _fac1(row, tok):
            return _lds_ld(p_f, row * BT + tok, 1, fx.Float32)[0]

        # q/k staging: thread owns token row r and K columns 8*sub + 64*m.
        r_row = tid // 8
        sub = tid % 8

        def _tok(t_local):
            t = bos + t_local
            t = (t > last_tok).select(last_tok, t)
            return t

        def _prefetch(chunk):
            base = chunk * BT
            tq = _tok(base + r_row)
            qs = [
                _g_ld(q_it, tq * q_st + 8 * sub + 64 * m, 8, fx.BFloat16)
                for m in range(2)
            ]
            ks = [
                _g_ld(k_it, tq * k_st + 8 * sub + 64 * m, 8, fx.BFloat16)
                for m in range(2)
            ]
            vs = [_g_ld(v_it, tq * v_st + V_EPT * sub, V_EPT, fx.BFloat16)]
            tg = _tok(base + lane)
            gv = _g_ld(g_it, tg * g_st, 1, fx.Float32)[0]
            bv = _g_ld(b_it, tg * b_st, 1, fx.Float32)[0]
            return qs + ks + vs + [gv, bv]

        # State slice: group-0 wave wl owns K rows [32wl, 32wl + 32) of S^T, all
        # BV columns; group 1 carries an unused copy.
        def _s_off(fb, vb):
            return (16 * vb + c) * KD + 16 * fb + 4 * gq

        s_init = []
        for j in range_constexpr(2):
            for vb in range_constexpr(NVB):
                s_init.append(
                    _g_ld(pool_it, _s_off(2 * wl + j, vb), 4, fx.BFloat16).to(
                        fx.Float32
                    )
                )
        if wh == 0:
            for j in range_constexpr(2):
                for vb in range_constexpr(NVB):
                    _lds_st(
                        p_s,
                        (16 * vb + c) * ROW + 16 * (2 * wl + j) + 4 * gq,
                        s_init[j * NVB + vb].to(fx.BFloat16),
                        4,
                        fx.BFloat16,
                    )
        if w == 0:
            _lds_st(
                p_ap,
                ZERO_TILE * FRAG + lane * 4,
                fx.Vector.filled(4, 0.0, fx.BFloat16),
                4,
                fx.BFloat16,
            )

        N_S = 2 * NVB
        # Inputs are prefetched two chunks ahead to cover the load latency.
        pre0 = _prefetch(fx.Int32(0)) + _prefetch(fx.Int32(1))
        N_PRE = len(pre0) // 2
        ident = _vec(
            [((4 * gq + i) == c).select(fx.Float32(1.0), fx.Float32(0.0)) for i in range(4)]
        )

        def _mm_k4(a_row, b_row):
            """Raw 16x16 tile sum_k A[a_row + c][k] * B[b_row + c][k] over K."""
            acc = _zeros4()
            for s_ in range_constexpr(4):
                acc = _mfma32(
                    _lds_ld(p_qk, a_row + 32 * s_, 8, fx.BFloat16),
                    _lds_ld(p_qk, b_row + 32 * s_, 8, fx.BFloat16),
                    acc,
                )
            return acc

        def _a_tile(kbuf, ra, rb):
            acc = _mm_k4(
                kbuf + (16 * ra + c) * ROW + 8 * gq, kbuf + (16 * rb + c) * ROW + 8 * gq
            )
            ea4 = _fac4(F_EA, 16 * ra + 4 * gq)
            eb = _fac1(F_EB, 16 * rb + c)
            vals = _vec([acc[i] * _exp2(ea4[i] - eb) for i in range(4)])
            _lds_st(
                p_ap,
                (ra * (ra - 1) // 2 + rb) * FRAG + lane * 4,
                _bf16(vals),
                4,
                fx.BFloat16,
            )

        def _p_tile(kbuf, a, b, diag):
            # P^T[j][t], j-block b (rows, from k), t-block a (columns, from q).
            acc = _mm_k4(kbuf + (16 * b + c) * ROW + 8 * gq, (16 * a + c) * ROW + 8 * gq)
            eb4 = _fac4(F_EB, 16 * b + 4 * gq)
            eq = _fac1(F_EQ, 16 * a + c)
            vals = []
            for i in range_constexpr(4):
                val = acc[i] * _exp2(eq - eb4[i])
                if const_expr(diag):
                    val = ((4 * gq + i) <= c).select(val, fx.Float32(0.0))
                vals.append(val)
            _lds_st(
                p_ap,
                (N_A_TILES + a * (a + 1) // 2 + b) * FRAG + lane * 4,
                _bf16(_vec(vals)),
                4,
                fx.BFloat16,
            )

        for i_t, state in range(
            fx.Int64(0), fx.Int64(NT), fx.Int64(1), init=s_init + pre0
        ):
            ch = fx.Int32(i_t)
            kbuf = BT * ROW + (ch % 2) * (BT * ROW)
            S = [fx.Vector(state[j]) for j in range(N_S)]
            qr = [fx.Vector(state[N_S + m]) for m in range(2)]
            kr = [fx.Vector(state[N_S + 2 + m]) for m in range(2)]
            vrow = fx.Vector(state[N_S + 4])
            gv = fx.Float32(state[N_S + 5])
            bv = fx.Float32(state[N_S + 6])
            pre_next = [state[N_S + N_PRE + i] for i in range(N_PRE)]

            if const_expr(HAS_TRACK):
                if wh == 0:
                    if ch == trk:
                        for j in range_constexpr(2):
                            for vb in range_constexpr(NVB):
                                _g_st(
                                    trs_it,
                                    _s_off(2 * wl + j, vb),
                                    S[j * NVB + vb],
                                    4,
                                    fx.Float32,
                                )

            # ---- Phase 0: gates, norms, per-token factors, raw q/k to LDS ----
            L = T_local - ch * BT
            tok_ok = lane < L
            gv = tok_ok.select(gv, fx.Float32(0.0))
            bv = tok_ok.select(bv, fx.Float32(0.0))
            gl = _wave_scan(gv, lane) * LOG2E
            gl_last = fx.Float32(rocdl.readlane(T.f32, gl.ir_value(), BT - 1))
            _lds_st(p_vt, r_row * VROW + V_EPT * sub, vrow, V_EPT, fx.BFloat16)
            for m in range_constexpr(2):
                _lds_st(p_qk, r_row * ROW + 8 * sub + 64 * m, qr[m], 8, fx.BFloat16)
                _lds_st(
                    p_qk, kbuf + r_row * ROW + 8 * sub + 64 * m, kr[m], 8, fx.BFloat16
                )
            ssq = _sumsq_bf16x8(qr[1], _sumsq_bf16x8(qr[0], fx.Float32(0.0)))
            ssk = _sumsq_bf16x8(kr[1], _sumsq_bf16x8(kr[0], fx.Float32(0.0)))
            row_ok = r_row < L
            rq = row_ok.select(_rsq(_row_sum8(ssq) + L2NORM_EPS), fx.Float32(0.0))
            rk = row_ok.select(_rsq(_row_sum8(ssk) + L2NORM_EPS), fx.Float32(0.0))
            # This thread's token is r_row; its gates live in lane r_row % 64.
            src = r_row % WAVE
            gl_r = gpu.shuffle_idx(gl, src, WAVE)
            beta_r = gpu.shuffle_idx(bv, src, WAVE)
            eg_r = _exp2(gl_r)
            log_rk = _log2(rk)
            fac_vals = (
                (F_EA, gl_r + _log2(beta_r) + log_rk),
                (F_EB, gl_r - log_rk),
                (F_EQ, gl_r + _log2(rq) + LOG2_SCALE),
                (F_BETA, beta_r),
                (F_BEGRK, beta_r * eg_r * rk),
                (F_EGQ, eg_r * rq * scale),
                (F_EDRK, _exp2(gl_last - gl_r) * rk),
                (F_EG, eg_r),
            )
            if sub == 0:
                for fi, val in fac_vals:
                    _lds_st(p_f, fi * BT + r_row, _vec([val]), 1, fx.Float32)
            gpu.barrier()

            # ---- Phase 1: group 0 diag A + inverse and RHS; group 1 q S^T, A tiles ----
            nxt = pre_next + _prefetch(ch + 2)
            tw4 = 16 * wl + 4 * gq
            twc = 16 * wl + c
            sb = [
                [
                    _lds_ld(p_s, (16 * vb + c) * ROW + 8 * gq + 32 * s_, 8, fx.BFloat16)
                    for s_ in range(4)
                ]
                for vb in range(NVB)
            ]
            o_acc = [_zeros4() for _ in range(NVB)]
            if wh == 0:
                ea4w = _fac4(F_EA, tw4)
                eb4w = _fac4(F_EB, tw4)
                beta4w = _fac4(F_BETA, tw4)
                begrk4w = _fac4(F_BEGRK, tw4)
                ea_c = _fac1(F_EA, twc)
                eb_c = _fac1(F_EB, twc)
                k_row = kbuf + twc * ROW + 8 * gq
                ka = [_lds_ld(p_qk, k_row + 32 * s_, 8, fx.BFloat16) for s_ in range(4)]
                # Diagonal block of A, in both operand layouts (k k^T is symmetric).
                kk = _zeros4()
                for s_ in range_constexpr(4):
                    kk = _mfma32(ka[s_], ka[s_], kk)
                ab_l = []
                aa_l = []
                for i in range_constexpr(4):
                    r_i = 4 * gq + i
                    ab_l.append(
                        (r_i > c).select(kk[i] * _exp2(ea4w[i] - eb_c), fx.Float32(0.0))
                    )
                    aa_l.append(
                        (c > r_i).select(kk[i] * _exp2(ea_c - eb4w[i]), fx.Float32(0.0))
                    )
                # RHS = beta * v - beta * exp(gc) * rk * (k_raw S^T), token block wl.
                for vb in range_constexpr(NVB):
                    ks_acc = _zeros4()
                    for s_ in range_constexpr(4):
                        ks_acc = _mfma32(ka[s_], sb[vb][s_], ks_acc)
                    v4 = _lds_ld_tr(
                        p_vt, (16 * wl + 4 * gq + c // 4) * VROW + 16 * vb + 4 * (c % 4)
                    )
                    rhs = v4.to(fx.Float32) * beta4w - ks_acc * begrk4w
                    _lds_st(
                        p_rhs,
                        (wl * NVB + vb) * FRAG + lane * 4,
                        _bf16(rhs),
                        4,
                        fx.BFloat16,
                    )
                tdiag = _diag_inverse(_vec(aa_l), _vec(ab_l), ident)
                _lds_st(p_td, wl * FRAG + lane * 4, _bf16(tdiag), 4, fx.BFloat16)
            else:
                egq_c = _fac1(F_EGQ, twc)
                q_row = twc * ROW + 8 * gq
                qb = [_lds_ld(p_qk, q_row + 32 * s_, 8, fx.BFloat16) for s_ in range(4)]
                o_new = []
                for vb in range_constexpr(NVB):
                    qs_acc = _zeros4()
                    for s_ in range_constexpr(4):
                        qs_acc = _mfma32(sb[vb][s_], qb[s_], qs_acc)
                    o_new.append(qs_acc * egq_c)
                o_acc = o_new
                _a_tile(
                    kbuf,
                    _sel_by_wave(wl, [t[0] for t in OFF0]),
                    _sel_by_wave(wl, [t[1] for t in OFF0]),
                )
                if wl < 2:
                    _a_tile(
                        kbuf,
                        _sel_by_wave(wl, [t[0] for t in OFF1]),
                        _sel_by_wave(wl, [t[1] for t in OFF1]),
                    )

            gpu.barrier()

            # ---- Phase 3: group 0 Tinv row block + vnew; group 1 P^T tiles ----
            eg_last = fx.Float32(0.0)
            if wh == 0:
                zw = _vec(
                    [
                        _lds_ld(
                            p_td,
                            wl * FRAG + (c // 4) * 64 + (4 * gq + e) * 4 + c % 4,
                            1,
                            fx.BFloat16,
                        )[0].to(fx.Float32)
                        for e in range(4)
                    ]
                )
                z = [None] * 4
                z[3] = _vec([(wl == 3).select(zw[i], fx.Float32(0.0)) for i in range(4)])
                for j in range_constexpr(2, -1, -1):
                    y = _zeros4()
                    for kb in range_constexpr(j + 1, 4):
                        a_frag = _lds_ld(
                            p_ap, (kb * (kb - 1) // 2 + j) * FRAG + lane * 4, 4, fx.BFloat16
                        )
                        y = _mfma16(a_frag, _bf16(z[kb]), y)
                    td_frag = _lds_ld(p_td, j * FRAG + lane * 4, 4, fx.BFloat16)
                    zc = _mfma16(td_frag, _bf16(y), _zeros4())
                    z[j] = _vec(
                        [
                            (wl > j).select(
                                -zc[i], (wl == j).select(zw[i], fx.Float32(0.0))
                            )
                            for i in range(4)
                        ]
                    )
                zb = [_bf16(z[j]) for j in range(4)]
                edrk4w = _fac4(F_EDRK, tw4)
                for vb in range_constexpr(NVB):
                    vn = _zeros4()
                    for j in range_constexpr(4):
                        vn = _mfma16(
                            zb[j],
                            _lds_ld(
                                p_rhs, (j * NVB + vb) * FRAG + lane * 4, 4, fx.BFloat16
                            ),
                            vn,
                        )
                    _lds_st(
                        p_vn,
                        (wl * NVB + vb) * FRAG + lane * 4,
                        _bf16(vn),
                        4,
                        fx.BFloat16,
                    )
                    _lds_st(
                        p_vd,
                        (wl * NVB + vb) * FRAG + lane * 4,
                        _bf16(vn * edrk4w),
                        4,
                        fx.BFloat16,
                    )
                eg_last = _fac1(F_EG, BT - 1)
            else:
                _p_tile(kbuf, wl, wl, True)
                _p_tile(
                    kbuf,
                    _sel_by_wave(wl, [t[0] for t in OFF0]),
                    _sel_by_wave(wl, [t[1] for t in OFF0]),
                    False,
                )
                if wl < 2:
                    _p_tile(
                        kbuf,
                        _sel_by_wave(wl, [t[0] for t in OFF1]),
                        _sel_by_wave(wl, [t[1] for t in OFF1]),
                        False,
                    )

            gpu.barrier()

            # ---- Phase 4: group 0 state update; group 1 output ----
            new_s = S
            if wh == 0:
                upd = []
                for j in range_constexpr(2):
                    fb = 2 * wl + j
                    # k^T operand via transposed reads of the [token][K] tile.
                    kt_off = kbuf + (4 * gq + c // 4) * ROW + 16 * fb + 4 * (c % 4)
                    kt = [
                        _lds_ld_tr(p_qk, kt_off + (32 * s_ + 16 * h_) * ROW)
                        for s_ in range(2)
                        for h_ in range(2)
                    ]
                    for vb in range_constexpr(NVB):
                        acc = S[j * NVB + vb] * eg_last
                        for s_ in range_constexpr(2):
                            a8 = kt[2 * s_].shuffle(
                                kt[2 * s_ + 1], [0, 1, 2, 3, 4, 5, 6, 7]
                            )
                            b_lo = _lds_ld(
                                p_vd,
                                ((2 * s_) * NVB + vb) * FRAG + lane * 4,
                                4,
                                fx.BFloat16,
                            )
                            b_hi = _lds_ld(
                                p_vd,
                                ((2 * s_ + 1) * NVB + vb) * FRAG + lane * 4,
                                4,
                                fx.BFloat16,
                            )
                            acc = _mfma32(
                                a8, b_lo.shuffle(b_hi, [0, 1, 2, 3, 4, 5, 6, 7]), acc
                            )
                        upd.append(acc)
                        _lds_st(
                            p_s,
                            (16 * vb + c) * ROW + 16 * fb + 4 * gq,
                            _bf16(acc),
                            4,
                            fx.BFloat16,
                        )
                new_s = upd
            else:
                for vb in range_constexpr(NVB):
                    acc = o_acc[vb]
                    for b in range_constexpr(4):
                        p_idx = (b > wl).select(
                            fx.Int32(ZERO_TILE), N_A_TILES + wl * (wl + 1) // 2 + b
                        )
                        acc = _mfma16(
                            _lds_ld(
                                p_vn, (b * NVB + vb) * FRAG + lane * 4, 4, fx.BFloat16
                            ),
                            _lds_ld(p_ap, p_idx * FRAG + lane * 4, 4, fx.BFloat16),
                            acc,
                        )
                    _g_st(
                        o_it,
                        (ch * BT + twc) * (H * VD) + 16 * vb + 4 * gq,
                        _bf16(acc),
                        4,
                        fx.BFloat16,
                    )

            results = yield new_s + nxt

        if wh == 0:
            for j in range_constexpr(2):
                for vb in range_constexpr(NVB):
                    _g_st(
                        pool_it,
                        _s_off(2 * wl + j, vb),
                        _bf16(fx.Vector(results[j * NVB + vb])),
                        4,
                        fx.BFloat16,
                    )

    @flyc.jit
    def launch(
        q_t: fx.Tensor,
        k_t: fx.Tensor,
        v_t: fx.Tensor,
        g_t: fx.Tensor,
        beta_t: fx.Tensor,
        cu_t: fx.Tensor,
        idx_t: fx.Tensor,
        pool_t: fx.Tensor,
        o_t: fx.Tensor,
        trs_t: fx.Tensor,
        tri_t: fx.Tensor,
        q_st: fx.Int32,
        k_st: fx.Int32,
        v_st: fx.Int32,
        g_st: fx.Int32,
        b_st: fx.Int32,
        T_total: fx.Int32,
        grid_y: fx.Int32,
        stream: fx.Stream,
    ):
        gdn_fused_kernel(
            q_t,
            k_t,
            v_t,
            g_t,
            beta_t,
            cu_t,
            idx_t,
            pool_t,
            o_t,
            trs_t,
            tri_t,
            q_st,
            k_st,
            v_st,
            g_st,
            b_st,
            T_total,
        ).launch(grid=(VD // BV, grid_y, 1), block=(BLOCK, 1, 1), stream=stream)

    return launch


__all__ = ["compile_gdn_fused_prefill"]
