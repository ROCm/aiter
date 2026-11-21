# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Gated DeltaNet chunked prefill as two FlyDSL kernels (gfx950).

The single-kernel prefill (``gdn_fused_prefill.py``) walks a sequence's
64-token chunks serially in one workgroup per (sequence, value head, BV slice
of V), and redoes the state-independent work of every chunk (q/k norms and
gate factors, the triangular inverse of (I + A), the masked q k^T) on that
serial path. Here that work moves to a prep kernel with one workgroup per
(chunk, value head), all chunks in parallel. The serial kernel then keeps only
what depends on the state: k S^T, vnew = Tinv @ (beta v - ...), the state
update, q S^T and the output.

Both kernels run the fused kernel's own arithmetic on the same operands in
the same order; the hand-off quantities (the Tinv row blocks and P^T tiles as
the bf16 MFMA fragments the fused kernel keeps in registers / LDS, and the fp32
per-token factors) round-trip through global memory unchanged, so o, the pool
state and the track snapshot are bit-identical to the fused kernel's. A change
to the fused kernel's phase 0-2 arithmetic has to be mirrored here.

Hand-off layout, per prep slot s and value head h (s = bos // 64 + chunk + seq,
an injective map of (sequence, chunk) onto [0, (T - 1) // 64 + N)):
    pk  bf16 [s, h, wave 0..7, lane 0..63, 16]: waves 0-3 hold the Tinv row
        block of token block wl as 4 fragments, waves 4-7 the P^T tiles of
        output row block wl. Fragments above the diagonal are zero; those of
        waves 0, 1, 4 and 5 (fragments 2-3) are neither written nor read, the
        serial kernel substitutes zeros.
    fac f32  [s, h, 5, 64]: rows beta, beta*exp(gc)*rk, scale*exp(gc)*rq,
        exp(gc_last - gc)*rk, exp(gc).

The sequence count and the launch order are runtime arguments, so the compiled
variants are one prep kernel per head shape and the fused kernel's set of
serial ones.
"""

import functools
import math

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T

from ..kernels_common import LOG2E
from .gdn_fused_prefill import (
    BLOCK,
    BT,
    F_BEGRK,
    F_BETA,
    F_EA,
    F_EB,
    F_EDRK,
    F_EG,
    F_EGQ,
    F_EQ,
    FRAG,
    GROUP,
    KD,
    L2NORM_EPS,
    N_A_TILES,
    N_FAC,
    OFF0,
    OFF1,
    ROW,
    VD,
    WAVE,
    _bf16,
    _diag_inverse,
    _exp2,
    _g_ld,
    _g_st,
    _lds_ld,
    _lds_ld_tr,
    _lds_st,
    _log2,
    _mfma16,
    _mfma32,
    _row_sum8,
    _rsq,
    _sel_by_wave,
    _sumsq_bf16x8,
    _vec,
    _wave_scan,
    _zeros4,
)

# pk elements per (slot, head): 8 waves x 64 lanes x 4 fragments x 4 bf16.
PK_WAVE = WAVE * 16
PK_HEAD = 8 * PK_WAVE
# fac rows handed to the serial kernel, in this order (F_BETA .. F_EG).
N_FAC_OUT = F_EG - F_BETA + 1
FAC_HEAD = N_FAC_OUT * BT
# MI350X / MI355X: eight XCDs, each with its own L2; workgroups are dispatched to
# them round-robin by linear workgroup id. Both kernels launch a 1-D grid and
# map the workgroup id L to their work item either in an L2-friendly XCD order,
# where that order is a bijection onto the work items (a multiple of 8 k-head
# columns), or in the plain 2-D grid's order. Either way every item runs once,
# so only speed depends on this.
NUM_XCD = 8


def handoff_slots(T: int, N: int) -> int:
    """Prep slots s = bos // 64 + chunk + seq span [0, (T - 1) // 64 + N); one
    more is the dummy slot idle prep workgroups write."""
    return (T - 1) // BT + N + 1


def _buffer_view(tensor, base):
    view = fx.make_view(fx.get_iter(tensor) + base, fx.make_layout(1, 1))
    return fx.get_iter(fx.rocdl.make_buffer_tensor(view))


@functools.lru_cache(maxsize=32)
def compile_gdn_split_prep(*, H: int, Hg: int, scale: float = KD**-0.5):
    """Prep kernel launcher; (slots - 1) * H workgroups of 512 threads, where
    the launcher's ``grid_x`` is slots - 1 (the dummy slot's index).

    ``log_n`` = (N - 1).bit_length() bisection steps map a slot to its sequence.
    """
    assert H % Hg == 0
    REP = H // Hg
    LOG2_SCALE = math.log2(scale)

    @fx.struct
    class PrepStorage:
        qk: fx.Array[fx.BFloat16, 2 * BT * ROW, 16]
        ap: fx.Array[fx.BFloat16, N_A_TILES * FRAG, 16]
        td: fx.Array[fx.BFloat16, GROUP * FRAG, 16]
        # Only the A / P^T exponent rows (F_EA, F_EB, F_EQ) are read back here;
        # 40,704 bytes in all lets four workgroups share a CU.
        fac: fx.Array[fx.Float32, F_BETA * BT, 16]

    @flyc.kernel(name=f"gdn_split_prep_h{H}_hg{Hg}")
    def prep_kernel(
        q_t: fx.Tensor,
        k_t: fx.Tensor,
        g_t: fx.Tensor,
        beta_t: fx.Tensor,
        cu_t: fx.Tensor,
        idx_t: fx.Tensor,
        pk_t: fx.Tensor,
        fac_t: fx.Tensor,
        q_st: fx.Int32,
        k_st: fx.Int32,
        g_st: fx.Int32,
        b_st: fx.Int32,
        N_seq: fx.Int32,
        T_total: fx.Int32,
        log_n: fx.Int32,
        S_dummy: fx.Int32,
    ):
        tid = fx.Int32(fx.thread_idx.x)
        w = fx.Int32(rocdl.readfirstlane(T.i32, (tid // WAVE).ir_value()))
        wl = w % GROUP
        wh = w // GROUP
        lane = tid % WAVE
        c = lane % 16
        gq = lane // 16
        # Workgroup L runs on XCD L % NUM_XCD (round-robin dispatch). In the XCD
        # order the REP value heads of one k head and slot land on one XCD back
        # to back, so two of their three q/k tile reads hit its L2; it covers
        # the (slot, head) grid once when S_dummy * Hg is a multiple of
        # NUM_XCD. Otherwise L takes the plain (slot, head) grid's order.
        L = fx.Int32(fx.block_idx.x)
        jx = L // NUM_XCD
        qd = (jx // REP) * NUM_XCD + L % NUM_XCD
        xcd = (S_dummy * Hg) % NUM_XCD == 0
        i_s = xcd.select(qd // Hg, L % S_dummy)
        i_h = xcd.select((qd % Hg) * REP + jx % REP, L // S_dummy)
        i_hg = i_h // REP

        # Sequence of slot i_s: the last n with cu[n] // BT + n <= i_s.
        for _, bounds in range(0, log_n, 1, init=[fx.Int32(0), N_seq]):
            lo, hi = bounds
            mid = (lo + hi) // 2
            right = (fx.Int32(cu_t[mid]) // BT + mid) <= i_s
            results = yield [right.select(mid, lo), right.select(hi, mid)]
        i_n, _ = results
        bos = fx.Int32(cu_t[i_n])
        T_local = fx.Int32(cu_t[i_n + 1]) - bos
        NT = (T_local + (BT - 1)) // BT
        NT = (fx.Int32(idx_t[i_n]) < 0).select(fx.Int32(0), NT)
        ch = i_s - (bos // BT + i_n)
        # A slot before the batch's first sequence (cu[0] > 0) is idle too.
        ch = (ch < 0).select(NT, ch)
        last_tok = (T_total - 1 < 0).select(fx.Int32(0), T_total - 1)

        # Idle slots (no chunk of a live sequence) run on a clamped chunk and
        # write the dummy slot S_dummy, so the body needs no data-dependent branch.
        valid = ch < NT
        ch = valid.select(ch, fx.Int32(0))
        i_out = valid.select(i_s, S_dummy)
        q_it = _buffer_view(q_t, fx.Int64(i_hg) * KD)
        k_it = _buffer_view(k_t, fx.Int64(i_hg) * KD)
        g_it = _buffer_view(g_t, fx.Int64(i_h))
        b_it = _buffer_view(beta_t, fx.Int64(i_h))
        slot_head = fx.Int64(i_out) * H + fx.Int64(i_h)
        pk_it = _buffer_view(pk_t, slot_head * PK_HEAD)
        fac_it = _buffer_view(fac_t, slot_head * FAC_HEAD)

        lds = fx.SharedAllocator().allocate(PrepStorage).peek()
        p_qk = lds.qk.ptr
        p_ap = lds.ap.ptr
        p_td = lds.td.ptr
        p_f = lds.fac.ptr

        def _fac4(row, tok):
            return _lds_ld(p_f, row * BT + tok, 4, fx.Float32)

        def _fac1(row, tok):
            return _lds_ld(p_f, row * BT + tok, 1, fx.Float32)[0]

        r_row = tid // 8
        sub = tid % 8

        def _tok(t_local):
            t = bos + t_local
            return (t > last_tok).select(last_tok, t)

        base = ch * BT
        tq = _tok(base + r_row)
        qr = [
            _g_ld(q_it, tq * q_st + 8 * sub + 64 * m, 8, fx.BFloat16) for m in range(2)
        ]
        kr = [
            _g_ld(k_it, tq * k_st + 8 * sub + 64 * m, 8, fx.BFloat16) for m in range(2)
        ]
        tg = _tok(base + lane)
        gv = _g_ld(g_it, tg * g_st, 1, fx.Float32)[0]
        bv = _g_ld(b_it, tg * b_st, 1, fx.Float32)[0]
        kbuf = BT * ROW
        ident = _vec(
            [
                ((4 * gq + i) == c).select(fx.Float32(1.0), fx.Float32(0.0))
                for i in range(4)
            ]
        )

        def _mm_k4(a_row, b_row):
            acc = _zeros4()
            for s_ in range_constexpr(4):
                acc = _mfma32(
                    _lds_ld(p_qk, a_row + 32 * s_, 8, fx.BFloat16),
                    _lds_ld(p_qk, b_row + 32 * s_, 8, fx.BFloat16),
                    acc,
                )
            return acc

        def _a_tile(ra, rb):
            acc = _mm_k4(
                kbuf + (16 * ra + c) * ROW + 8 * gq,
                kbuf + (16 * rb + c) * ROW + 8 * gq,
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

        def _p_tile(a, b, diag):
            # P^T[j][t] tile (t-block a, j-block b) to output row block a, slot b.
            acc = _mm_k4(
                kbuf + (16 * b + c) * ROW + 8 * gq, (16 * a + c) * ROW + 8 * gq
            )
            eb4 = _fac4(F_EB, 16 * b + 4 * gq)
            eq = _fac1(F_EQ, 16 * a + c)
            vals = []
            for i in range_constexpr(4):
                val = acc[i] * _exp2(eq - eb4[i])
                if const_expr(diag):
                    val = ((4 * gq + i) <= c).select(val, fx.Float32(0.0))
                vals.append(val)
            _g_st(
                pk_it,
                (GROUP + a) * PK_WAVE + lane * 16 + 4 * b,
                _bf16(_vec(vals)),
                4,
                fx.BFloat16,
            )

        # ---- Phase 0: gates, norms, per-token factors (fused kernel's) ----
        L = T_local - ch * BT
        tok_ok = lane < L
        gv = tok_ok.select(gv, fx.Float32(0.0))
        bv = tok_ok.select(bv, fx.Float32(0.0))
        gl = _wave_scan(gv) * LOG2E
        gl_last = fx.Float32(rocdl.readlane(T.f32, gl.ir_value(), BT - 1))
        for m in range_constexpr(2):
            _lds_st(p_qk, r_row * ROW + 8 * sub + 64 * m, qr[m], 8, fx.BFloat16)
            _lds_st(p_qk, kbuf + r_row * ROW + 8 * sub + 64 * m, kr[m], 8, fx.BFloat16)
        ssq = _sumsq_bf16x8(qr[1], _sumsq_bf16x8(qr[0], fx.Float32(0.0)))
        ssk = _sumsq_bf16x8(kr[1], _sumsq_bf16x8(kr[0], fx.Float32(0.0)))
        row_ok = r_row < L
        rq = row_ok.select(_rsq(_row_sum8(ssq) + L2NORM_EPS), fx.Float32(0.0))
        rk = row_ok.select(_rsq(_row_sum8(ssk) + L2NORM_EPS), fx.Float32(0.0))
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
                if const_expr(fi < F_BETA):
                    _lds_st(p_f, fi * BT + r_row, _vec([val]), 1, fx.Float32)
                if const_expr(fi >= F_BETA):
                    _g_st(
                        fac_it,
                        (fi - F_BETA) * BT + r_row,
                        _vec([val]),
                        1,
                        fx.Float32,
                    )
        gpu.barrier()

        # ---- Phase 1: group 0 diagonal A blocks + inverse; group 1 A tiles ----
        tw4 = 16 * wl + 4 * gq
        twc = 16 * wl + c
        if wh == 0:
            ea4w = _fac4(F_EA, tw4)
            eb4w = _fac4(F_EB, tw4)
            ea_c = _fac1(F_EA, twc)
            eb_c = _fac1(F_EB, twc)
            k_row = kbuf + twc * ROW + 8 * gq
            ka = [_lds_ld(p_qk, k_row + 32 * s_, 8, fx.BFloat16) for s_ in range(4)]
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
            tdiag = _diag_inverse(_vec(aa_l), _vec(ab_l), ident)
            _lds_st(p_td, wl * FRAG + lane * 4, _bf16(tdiag), 4, fx.BFloat16)
        else:
            _a_tile(
                _sel_by_wave(wl, [t[0] for t in OFF0]),
                _sel_by_wave(wl, [t[1] for t in OFF0]),
            )
            if wl < 2:
                _a_tile(
                    _sel_by_wave(wl, [t[0] for t in OFF1]),
                    _sel_by_wave(wl, [t[1] for t in OFF1]),
                )

        gpu.barrier()

        # ---- Phase 2: group 0 Tinv row block -> pk; group 1 P^T tiles -> pk ----
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
                        p_ap,
                        (kb * (kb - 1) // 2 + j) * FRAG + lane * 4,
                        4,
                        fx.BFloat16,
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
            for j in range_constexpr(2):
                _g_st(
                    pk_it,
                    wl * PK_WAVE + lane * 16 + 4 * j,
                    _bf16(z[j]),
                    4,
                    fx.BFloat16,
                )
            # Fragments 2-3 of waves 0-1 are zero; the serial kernel never reads them.
            if wl >= 2:
                for j in range_constexpr(2, 4):
                    _g_st(
                        pk_it,
                        wl * PK_WAVE + lane * 16 + 4 * j,
                        _bf16(z[j]),
                        4,
                        fx.BFloat16,
                    )
        else:
            _p_tile(wl, wl, True)
            _p_tile(
                _sel_by_wave(wl, [t[0] for t in OFF0]),
                _sel_by_wave(wl, [t[1] for t in OFF0]),
                False,
            )
            if wl < 2:
                _p_tile(
                    _sel_by_wave(wl, [t[0] for t in OFF1]),
                    _sel_by_wave(wl, [t[1] for t in OFF1]),
                    False,
                )
            # Fragments above the diagonal of output row block wl are zero: (0, 1)
            # and (2, 3) are written; fragments 2-3 of waves 0-1 are not read.
            if wl < 1:
                _g_st(
                    pk_it,
                    (GROUP + wl) * PK_WAVE + lane * 16 + 4,
                    fx.Vector.filled(4, 0.0, fx.BFloat16),
                    4,
                    fx.BFloat16,
                )
            if wl == 2:
                _g_st(
                    pk_it,
                    (GROUP + wl) * PK_WAVE + lane * 16 + 12,
                    fx.Vector.filled(4, 0.0, fx.BFloat16),
                    4,
                    fx.BFloat16,
                )

    @flyc.jit
    def launch(
        q_t: fx.Tensor,
        k_t: fx.Tensor,
        g_t: fx.Tensor,
        beta_t: fx.Tensor,
        cu_t: fx.Tensor,
        idx_t: fx.Tensor,
        pk_t: fx.Tensor,
        fac_t: fx.Tensor,
        q_st: fx.Int32,
        k_st: fx.Int32,
        g_st: fx.Int32,
        b_st: fx.Int32,
        N_seq: fx.Int32,
        T_total: fx.Int32,
        log_n: fx.Int32,
        grid_x: fx.Int32,
        stream: fx.Stream,
    ):
        prep_kernel(
            q_t,
            k_t,
            g_t,
            beta_t,
            cu_t,
            idx_t,
            pk_t,
            fac_t,
            q_st,
            k_st,
            g_st,
            b_st,
            N_seq,
            T_total,
            log_n,
            grid_x,
        ).launch(grid=(grid_x * H, 1, 1), block=(BLOCK, 1, 1), stream=stream)

    return launch


@functools.lru_cache(maxsize=32)
def compile_gdn_split_serial(
    *,
    H: int,
    Hg: int,
    BV: int = 32,
    HAS_TRACK: bool = False,
    HAS_INIT_FLAG: bool = False,
):
    """Serial (state) kernel launcher; (V // BV) * N * H workgroups of 512
    threads, the fused kernel's work items.

    The fused kernel's chunk loop with the state-independent work replaced by
    loads of the prep kernel's hand-off.
    """
    assert H % Hg == 0
    assert BV % 16 == 0 and VD % BV == 0
    NVB = BV // 16
    NV = VD // BV
    REP = H // Hg
    V_EPT = BV // 8
    VROW = BV + 8

    @fx.struct
    class SharedStorage:
        # q tile, then two k tiles: chunk c's k is still read (transposed) by the
        # state update while chunk c + 1 is being staged.
        qk: fx.Array[fx.BFloat16, 3 * BT * ROW, 16]
        s: fx.Array[fx.BFloat16, BV * ROW, 16]
        rhs: fx.Array[fx.BFloat16, GROUP * NVB * FRAG, 16]
        vn: fx.Array[fx.BFloat16, GROUP * NVB * FRAG, 16]
        vd: fx.Array[fx.BFloat16, GROUP * NVB * FRAG, 16]
        fac: fx.Array[fx.Float32, N_FAC * BT, 16]
        vt: fx.Array[fx.BFloat16, BT * VROW, 16]

    @flyc.kernel(
        name=(
            f"gdn_split_serial_bv{BV}_h{H}_hg{Hg}"
            f"_t{int(HAS_TRACK)}_i{int(HAS_INIT_FLAG)}"
        )
    )
    def serial_kernel(
        q_t: fx.Tensor,
        k_t: fx.Tensor,
        v_t: fx.Tensor,
        cu_t: fx.Tensor,
        idx_t: fx.Tensor,
        pool_t: fx.Tensor,
        o_t: fx.Tensor,
        trs_t: fx.Tensor,
        tri_t: fx.Tensor,
        init_t: fx.Tensor,
        pk_t: fx.Tensor,
        fac_t: fx.Tensor,
        q_st: fx.Int32,
        k_st: fx.Int32,
        v_st: fx.Int32,
        T_total: fx.Int32,
        N_seq: fx.Int32,
    ):
        tid = fx.Int32(fx.thread_idx.x)
        w = fx.Int32(rocdl.readfirstlane(T.i32, (tid // WAVE).ir_value()))
        wl = w % GROUP
        wh = w // GROUP
        lane = tid % WAVE
        c = lane % 16
        gq = lane // 16
        # In the XCD order the V slices of the REP heads of one (sequence,
        # k head) land on one XCD back to back and share its L2; it covers the
        # grid once when N_seq * Hg is a multiple of NUM_XCD. Otherwise L takes
        # the plain (V slice, sequence * head) grid's order.
        L = fx.Int32(fx.block_idx.x)
        jx = L // NUM_XCD
        jr = jx // NV
        qd = (jr // REP) * NUM_XCD + L % NUM_XCD
        i_nh = L // NV
        xcd = (N_seq * Hg) % NUM_XCD == 0
        i_v = xcd.select(jx % NV, L % NV)
        i_n = xcd.select(qd // Hg, i_nh // H)
        i_h = xcd.select((qd % Hg) * REP + jr % REP, i_nh % H)
        i_hg = i_h // REP

        bos = fx.Int32(cu_t[i_n])
        eos = fx.Int32(cu_t[i_n + 1])
        T_local = eos - bos
        NT = (T_local + (BT - 1)) // BT
        slot_in = fx.Int32(idx_t[i_n])
        NT = (slot_in < 0).select(fx.Int32(0), NT)
        slot = (slot_in < 0).select(fx.Int32(0), slot_in)
        last_tok = (T_total - 1 < 0).select(fx.Int32(0), T_total - 1)
        # Prep slot of this sequence's chunk 0.
        slot0 = bos // BT + i_n

        def _view(tensor, base, records=None):
            view = fx.make_view(fx.get_iter(tensor) + base, fx.make_layout(1, 1))
            if records is None:
                return fx.get_iter(fx.rocdl.make_buffer_tensor(view))
            return fx.get_iter(
                fx.rocdl.make_buffer_tensor(view, num_records_bytes=records)
            )

        bos64 = fx.Int64(bos)
        q_it = _view(q_t, fx.Int64(i_hg) * KD)
        k_it = _view(k_t, fx.Int64(i_hg) * KD)
        v_it = _view(v_t, fx.Int64(i_h) * VD + fx.Int64(i_v) * BV)
        pk_it = _view(pk_t, fx.Int64(i_h) * PK_HEAD + fx.Int64(w) * PK_WAVE)
        fac_it = _view(fac_t, fx.Int64(i_h) * FAC_HEAD)
        state_base = (fx.Int64(slot) * H + fx.Int64(i_h)) * (VD * KD) + fx.Int64(
            i_v
        ) * (BV * KD)
        pool_it = _view(pool_t, state_base)
        o_rec = fx.Int64(T_local) * (H * VD * 2)
        o_it = _view(o_t, (bos64 * H + fx.Int64(i_h)) * VD + fx.Int64(i_v) * BV, o_rec)
        if const_expr(HAS_TRACK):
            trk = fx.Int32(tri_t[i_n])
            trs_base = (fx.Int64(i_n) * H + fx.Int64(i_h)) * (VD * KD) + fx.Int64(
                i_v
            ) * (BV * KD)
            trs_it = _view(trs_t, trs_base)

        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        p_qk = lds.qk.ptr
        p_s = lds.s.ptr
        p_rhs = lds.rhs.ptr
        p_vn = lds.vn.ptr
        p_vd = lds.vd.ptr
        p_f = lds.fac.ptr
        p_vt = lds.vt.ptr

        def _fac4(row, tok):
            return _lds_ld(p_f, row * BT + tok, 4, fx.Float32)

        def _fac1(row, tok):
            return _lds_ld(p_f, row * BT + tok, 1, fx.Float32)[0]

        r_row = tid // 8
        sub = tid % 8
        # fac staging: threads [0, 80) each move 4 floats of the 5 x 64 rows.
        f_row = tid // 16
        f_col = 4 * (tid % 16)
        f_ok = tid < N_FAC_OUT * 16
        f_off = (f_ok.select(f_row, fx.Int32(0))) * BT + f_col

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
            # Chunks past the sequence (prefetch overrun) read its last chunk.
            cs = (chunk >= NT).select(NT - 1, chunk)
            cs = (cs < 0).select(fx.Int32(0), cs)
            ps = slot0 + cs
            # Fragments 2-3 of waves 0-1 are zero; prep leaves them unwritten.
            pk1 = fx.Vector.filled(8, 0.0, fx.BFloat16)
            if wl >= 2:
                pk1 = _g_ld(pk_it, ps * (H * PK_HEAD) + lane * 16 + 8, 8, fx.BFloat16)
            pks = [_g_ld(pk_it, ps * (H * PK_HEAD) + lane * 16, 8, fx.BFloat16), pk1]
            fs = [_g_ld(fac_it, ps * (H * FAC_HEAD) + f_off, 4, fx.Float32)]
            return qs + ks + vs + pks + fs

        # State slice: group-0 wave wl owns K rows [32wl, 32wl + 32) of S^T, all
        # BV columns. Group 1 never reads it; it carries zeros through the loop.
        def _s_off(fb, vb):
            return (16 * vb + c) * KD + 16 * fb + 4 * gq

        s_init = [_zeros4() for _ in range(2 * NVB)]
        if wh == 0:
            s_init = [
                _g_ld(pool_it, _s_off(2 * wl + j, vb), 4, fx.BFloat16).to(fx.Float32)
                for j in range(2)
                for vb in range(NVB)
            ]
            if const_expr(HAS_INIT_FLAG):
                has_init = fx.Int32(init_t[i_n]) != 0
                s_init = [
                    _vec([has_init.select(s[i], fx.Float32(0.0)) for i in range(4)])
                    for s in s_init
                ]
            for j in range_constexpr(2):
                for vb in range_constexpr(NVB):
                    _lds_st(
                        p_s,
                        (16 * vb + c) * ROW + 16 * (2 * wl + j) + 4 * gq,
                        s_init[j * NVB + vb].to(fx.BFloat16),
                        4,
                        fx.BFloat16,
                    )

        N_S = 2 * NVB
        pre0 = _prefetch(fx.Int32(0)) + _prefetch(fx.Int32(1))
        N_PRE = len(pre0) // 2

        def _half(v8, h):
            v = fx.Vector(v8)
            return v.shuffle(v, [4 * h + e for e in range(4)])

        for i_t, state in range(
            fx.Int64(0), fx.Int64(NT), fx.Int64(1), init=s_init + pre0
        ):
            ch = fx.Int32(i_t)
            kbuf = BT * ROW + (ch % 2) * (BT * ROW)
            S = [fx.Vector(state[j]) for j in range(N_S)]
            qr = [fx.Vector(state[N_S + m]) for m in range(2)]
            kr = [fx.Vector(state[N_S + 2 + m]) for m in range(2)]
            vrow = fx.Vector(state[N_S + 4])
            pkr = [fx.Vector(state[N_S + 5 + m]) for m in range(2)]
            frow = fx.Vector(state[N_S + 7])
            pre_next = [state[N_S + N_PRE + i] for i in range(N_PRE)]
            # This wave's four hand-off fragments: Tinv row block (group 0) or
            # P^T tiles of its output row block (group 1).
            hf = [_half(pkr[j // 2], j % 2) for j in range(4)]

            if const_expr(HAS_TRACK):  # noqa: SIM102 - Preserve FlyDSL branches.
                if wh == 0:  # noqa: SIM102 - Preserve FlyDSL branches.
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

            # ---- Phase 0: raw q/k/v and the prep factors to LDS ----
            _lds_st(p_vt, r_row * VROW + V_EPT * sub, vrow, V_EPT, fx.BFloat16)
            for m in range_constexpr(2):
                _lds_st(p_qk, r_row * ROW + 8 * sub + 64 * m, qr[m], 8, fx.BFloat16)
                _lds_st(
                    p_qk, kbuf + r_row * ROW + 8 * sub + 64 * m, kr[m], 8, fx.BFloat16
                )
            if f_ok:
                _lds_st(p_f, F_BETA * BT + f_row * BT + f_col, frow, 4, fx.Float32)
            gpu.barrier()

            # ---- Phase 1: group 0 RHS; group 1 q S^T ----
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
                beta4w = _fac4(F_BETA, tw4)
                begrk4w = _fac4(F_BEGRK, tw4)
                k_row = kbuf + twc * ROW + 8 * gq
                ka = [_lds_ld(p_qk, k_row + 32 * s_, 8, fx.BFloat16) for s_ in range(4)]
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

            gpu.barrier()

            # ---- Phase 2: group 0 vnew = Tinv row block @ RHS ----
            eg_last = fx.Float32(0.0)
            if wh == 0:
                edrk4w = _fac4(F_EDRK, tw4)
                for vb in range_constexpr(NVB):
                    vn = _zeros4()
                    for j in range_constexpr(4):
                        vn = _mfma16(
                            hf[j],
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

            gpu.barrier()

            # ---- Phase 3: group 0 state update; group 1 output ----
            new_s = S
            if wh == 0:
                upd = []
                for j in range_constexpr(2):
                    fb = 2 * wl + j
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
                        acc = _mfma16(
                            _lds_ld(
                                p_vn, (b * NVB + vb) * FRAG + lane * 4, 4, fx.BFloat16
                            ),
                            hf[b],
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

        if slot_in >= 0:  # noqa: SIM102 - Preserve FlyDSL branches.
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
        cu_t: fx.Tensor,
        idx_t: fx.Tensor,
        pool_t: fx.Tensor,
        o_t: fx.Tensor,
        trs_t: fx.Tensor,
        tri_t: fx.Tensor,
        init_t: fx.Tensor,
        pk_t: fx.Tensor,
        fac_t: fx.Tensor,
        q_st: fx.Int32,
        k_st: fx.Int32,
        v_st: fx.Int32,
        T_total: fx.Int32,
        N_seq: fx.Int32,
        stream: fx.Stream,
    ):
        serial_kernel(
            q_t,
            k_t,
            v_t,
            cu_t,
            idx_t,
            pool_t,
            o_t,
            trs_t,
            tri_t,
            init_t,
            pk_t,
            fac_t,
            q_st,
            k_st,
            v_st,
            T_total,
            N_seq,
        ).launch(grid=(N_seq * (NV * H), 1, 1), block=(BLOCK, 1, 1), stream=stream)

    return launch


__all__ = [
    "FAC_HEAD",
    "PK_HEAD",
    "compile_gdn_split_prep",
    "compile_gdn_split_serial",
    "handoff_slots",
]
