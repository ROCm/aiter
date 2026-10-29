# SPDX-License-Identifier: MIT
"""The FFN seam of the DS V4.1 mono layer in Gluon: a faithful port of
``vllm/models/deepseek_v41/amd/mono/stages/seam.py`` (vLLM 8321d11), same tasks,
placements, mailbox regions, tags and arithmetic order:

    push    (160, 32 columns): this rank's attention partial (vLLM's unreduced
            wo_b output) to every rank's ATTN region (system-scope pairs)
    reduce  (160): the ranks' partials in rank order, fp32 -> bf16 -> PL (LDS)
    slice   (160, 32 columns x 4 streams): the owed post folded into the
            residual (R_new -> ``res_out``), the collapsed layer input (LIN) and
            the partial mix projections / sum of squares (PMIX)
    gate    (S): the 160 partials -> post / comb (Sinkhorn 20) and the next pre
    norm    (S): RMSNorm of LIN -> the MoE's bf16 input row, then its XRDY

Compile with ``enable_fp_fusion=False``: the reference rounds every product
(aiter's ``mhc_fused_post_pre_delayed_rmsnorm``); fmas are written out where it
uses them. The approximate math is the hardware's (``common.ops``)."""

import math

from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from ..common.build import NOINLINE

from ..common.ops import (
    bf16_pair,
    bf16_round,
    bf_hi,
    bf_lo,
    hw_exp2,
    hw_rcp,
    hw_rsq,
    lane_gather,
    sum_asc6,
    sum_desc6,
    xadd,
    xshfl,
)
from ..common.mx import clamp_fp8, fp8_pack4, vllm_mx_code
from ..common.sync import T512, fail, poll1, poll2, poll4, poll8, publish, put, store64

HC = gl.constexpr(4)
MIX = gl.constexpr(24)  # pre 4 | post 4 | comb 16
PART = gl.constexpr(25)  # a slice's partials a token: the mixes, then the sum of squares
COLS = gl.constexpr(32)  # hidden columns a slice task
KT = gl.constexpr(128)  # projection columns a slice task (HC x COLS)
GATE_BLOCK = gl.constexpr(8)
EPS = gl.constexpr(1e-20)
HC_EPS = gl.constexpr(1e-6)
SINKHORN = gl.constexpr(20)
POST_MULT = gl.constexpr(2.0)
LOG2E = gl.constexpr(1.0 / math.log(2.0))

# layouts: a thread an element; a flat 8-a-thread for MFMA operand gathers; the
# MFMA (16x16x4 f32, the 8 waves on the batch dimension)
FLAT8 = gl.constexpr(gl.BlockedLayout([8], [64], [8], [0]))
MMA4 = gl.constexpr(gl.amd.AMDMFMALayout(4, [16, 16, 4], False, [8, 1, 1]))
SMEM1 = gl.constexpr(gl.SwizzledSharedLayout(1, 1, 1, [0]))


@gluon.constexpr_function
def pow2(n):
    p = 1
    while p < int(n):
        p *= 2
    return p


def lds_words(s: int) -> dict:
    """The seam's LDS (f32 words): red, rl, fl, pl, each a power of two with a
    spare word past its live elements (masked scatters go there)."""

    def p2(n):
        p = 1
        while p < n:
            p *= 2
        return p

    return {
        "red": 4096,
        "rl": p2(s * 128 + 1),
        "fl": p2(24 * 128 + 1),
        "pl": p2(s * 32 + 1),
    }


# ---------------------------------------------------------------- push / reduce
@gluon.jit
def attn_pair(src, t, col, S: gl.constexpr, HIDDEN: gl.constexpr):
    return (src * S + t) * (HIDDEN // 2) + col // 2


@gluon.jit(noinline=NOINLINE)
def stage_push(part, task, tag, peers, rank, tid,
               S: gl.constexpr, TP: gl.constexpr, HIDDEN: gl.constexpr):  # fmt: skip
    """This rank's partial (bf16 [S, HIDDEN] at ``part``) at the task's 32
    columns, every token, to every rank's ATTN region, as the bits vLLM wrote."""
    pw = part.to(gl.pointer_type(gl.uint32))
    for pas in gl.static_range(0, S * 16, 512):
        idx = pas + tid
        active = idx < S * 16
        t = gl.minimum(idx // 16, S - 1)
        col = task * COLS + 2 * (idx % 16)
        w = gl.load(pw + (t * (HIDDEN // 2) + col // 2))
        pair = attn_pair(rank, t, col, S, HIDDEN)
        for p in gl.static_range(TP):
            put(peers[p], pair, w, tag, active, True)


@gluon.jit
def sum_ranks(own, pair, stride, tag, active, dl, health, rank, site,
              TP: gl.constexpr):  # fmt: skip
    """The TP ranks' bf16-pair partials at ``pair`` (rank r's at ``pair`` + r
    ``stride``) of the peer region ``own``, polled in one batch, summed in fp32
    in rank order 0 .. TP - 1 (``common.ops.sum_partials``)."""
    a = own.to(gl.uint64) + pair.to(gl.uint64) * 8
    st = stride * 8
    if TP == 2:
        w0, w1, f = poll2(a, a + st, tag, active, dl, health, True)
        lo = bf_lo(w0) + bf_lo(w1)
        hi = bf_hi(w0) + bf_hi(w1)
    elif TP == 4:
        w0, w1, w2, w3, f = poll4(a, a + st, a + 2 * st, a + 3 * st, tag, active, dl,
                                  health, True)  # fmt: skip
        lo = ((bf_lo(w0) + bf_lo(w1)) + bf_lo(w2)) + bf_lo(w3)
        hi = ((bf_hi(w0) + bf_hi(w1)) + bf_hi(w2)) + bf_hi(w3)
    else:
        gl.static_assert(TP == 8)
        w0, w1, w2, w3, w4, w5, w6, w7, f = poll8(
            a, a + st, a + 2 * st, a + 3 * st, a + 4 * st, a + 5 * st, a + 6 * st,
            a + 7 * st, tag, active, dl, health, True,
        )  # fmt: skip
        lo = bf_lo(w0) + bf_lo(w1)
        lo = lo + bf_lo(w2)
        lo = lo + bf_lo(w3)
        lo = lo + bf_lo(w4)
        lo = lo + bf_lo(w5)
        lo = lo + bf_lo(w6)
        lo = lo + bf_lo(w7)
        hi = bf_hi(w0) + bf_hi(w1)
        hi = hi + bf_hi(w2)
        hi = hi + bf_hi(w3)
        hi = hi + bf_hi(w4)
        hi = hi + bf_hi(w5)
        hi = hi + bf_hi(w6)
        hi = hi + bf_hi(w7)
    fail(health, f, rank, site, active)
    return lo, hi


@gluon.jit(noinline=NOINLINE)
def stage_reduce(task, tag, own, pl, tid, dl, health, rank,
                 S: gl.constexpr, TP: gl.constexpr, HIDDEN: gl.constexpr,
                 PL_SPARE: gl.constexpr):  # fmt: skip
    """The attention output at the task's 32 columns: the ranks' partials in
    rank order -> bf16 -> PL[t][32] (LDS), where the slice reads it."""
    for pas in gl.static_range(0, S * 16, 512):
        idx = pas + tid
        active = idx < S * 16
        t = gl.minimum(idx // 16, S - 1)
        cc = 2 * (idx % 16)
        col = task * COLS + cc
        lo, hi = sum_ranks(own, attn_pair(0, t, col, S, HIDDEN), S * (HIDDEN // 2),
                           tag, active, dl, health, rank, 1, TP)  # fmt: skip
        at = gl.where(active, t * COLS + cc, PL_SPARE)
        pl.scatter(bf16_round(lo), at, 0)
        pl.scatter(bf16_round(hi), gl.where(active, at + 1, PL_SPARE), 0)
    gl.barrier()


# ---------------------------------------------------------------- slice
@gluon.jit
def slice_rounds(res_in, comb_in, post_in, res_out, task, pl, rl, tid,
                 R0: gl.constexpr, NR: gl.constexpr, S: gl.constexpr,
                 HIDDEN: gl.constexpr, RL_SPARE: gl.constexpr, pend_g=None,
                 FOLD: gl.constexpr = True):  # fmt: skip
    """``stage_slice``'s rows of rounds R0 .. R0 + NR: every round's loads
    issued before any round's stores (a round's ``res_out`` store keeps a later
    round's loads behind it). Element e: token e / 128, stream (e % 128) / 32,
    column 32 task + e % 32; a round past S x 128 loads element S x 128 - 1 and
    stores nothing. The owed sublayer output: this task's reduction in LDS
    (``pl``), or (K1) ``pend_g``'s bf16 [S, HIDDEN]; ``FOLD`` off (a settled
    residual, no owed post): stream o as it arrives."""
    rin = res_in.to(gl.pointer_type(gl.bfloat16))
    cin = comb_in.to(gl.pointer_type(gl.float32))
    pin = post_in.to(gl.pointer_type(gl.float32))
    rout = res_out.to(gl.pointer_type(gl.bfloat16))
    c0 = task * COLS
    N: gl.constexpr = S * KT
    # loads
    r0s = ()
    for i in gl.static_range(NR):
        e = gl.minimum(tid + (R0 + i) * 512, N - 1)
        t = e // KT
        o = (e % KT) // COLS
        col = c0 + e % COLS
        rb = (t * HC) * HIDDEN + col
        r0s = r0s + ((
            gl.load(rin + rb).to(gl.float32),
            gl.load(rin + (rb + HIDDEN)).to(gl.float32),
            gl.load(rin + (rb + 2 * HIDDEN)).to(gl.float32),
            gl.load(rin + (rb + 3 * HIDDEN)).to(gl.float32),
            gl.load(cin + ((t * HC) * HC + o)),
            gl.load(cin + ((t * HC + 1) * HC + o)),
            gl.load(cin + ((t * HC + 2) * HC + o)),
            gl.load(cin + ((t * HC + 3) * HC + o)),
            gl.load(pin + (t * HC + o)),
        ),)  # fmt: skip
    # rows
    for i in gl.static_range(NR):
        v_e = tid + (R0 + i) * 512
        live = v_e < N
        e = gl.minimum(v_e, N - 1)
        t = e // KT
        o = (e % KT) // COLS
        cc = e % COLS
        col = c0 + cc
        r_0, r_1, r_2, r_3, cm0, cm1, cm2, cm3, post = r0s[i]
        if FOLD:
            if pend_g is None:
                pend = pl.gather(t * COLS + cc, 0)
            else:
                pend = gl.load(pend_g.to(gl.pointer_type(gl.bfloat16)) + (t * HIDDEN + col))
                pend = pend.to(gl.float32)
            # aiter's fold: ((post x + c0 r0) + c1 r1) + c2 r2, products
            # rounded, then fma(c3, r3, ..) and bf16 (what the collapse, the
            # sums of squares and the mix projection read)
            v = post * pend + (r_0 * cm0)
            v = v + cm1 * r_1
            v = v + cm2 * r_2
            v = bf16_round(gl.fma(cm3, r_3, v))
        else:
            v = gl.where(o == 0, r_0, gl.where(o == 1, r_1, gl.where(o == 2, r_2, r_3)))
        gl.store(rout + ((t * HC + o) * HIDDEN + col), v.to(gl.bfloat16), mask=live)
        rl.scatter(v, gl.where(live, t * KT + o * COLS + cc, RL_SPARE), 0)


@gluon.jit
def slice_mix(task, t0, n, tag, pmix, rl, fl, red, tid, S: gl.constexpr):
    """The mix projection partials of tokens t0 .. t0 + n: per wave w, A = fn
    rows (two 16-row tiles, rows past 24 zero) x B = R_new^T over K columns
    16 w .. 16 w + 16 (f32 MFMA 16x16x4, K in order); the waves' partials summed
    in wave order -> PMIX[(task S + t) 25 + j]."""
    p = gl.arange(0, 8 * 32 * 16, layout=FLAT8)
    # A[w][row][k] = fl[min(row, 23)][16 w + k]
    aw = p // (32 * 16)
    arow = (p // 16) % 32
    ak = p % 16
    av = fl.gather(gl.minimum(arow, MIX - 1) * KT + aw * 16 + ak, 0)
    av = gl.where(arow < MIX, av, 0.0)
    # B[w][k][tok] = rl[t0 + min(tok, n - 1)][16 w + k]
    q = gl.arange(0, 8 * 16 * 16, layout=FLAT8)
    bw = q // 256
    bk = (q // 16) % 16
    btok = t0 + gl.minimum(q % 16, n - 1)
    bv = rl.gather(btok * KT + bw * 16 + bk, 0)
    a = gl.convert_layout(gl.reshape(av, (8, 32, 16)), gl.DotOperandLayout(0, MMA4, 1))
    b = gl.convert_layout(gl.reshape(bv, (8, 16, 16)), gl.DotOperandLayout(1, MMA4, 1))
    acc = gl.amd.cdna4.mfma(a, b, gl.zeros((8, 32, 16), gl.float32, MMA4))
    # red[w][row][tok]
    red.store(gl.reshape(gl.convert_layout(acc, gl.BlockedLayout(
        [1, 1, 1], [1, 4, 16], [8, 1, 1], [2, 1, 0])), (4096,)))  # fmt: skip
    gl.barrier()
    tl = tid // MIX
    j = tid % MIX
    live = tid < MIX * n
    at = j * 16 + gl.minimum(tl, 15)
    tot = red.gather(at, 0)
    for w in gl.static_range(1, 8):
        tot = tot + red.gather(w * 512 + at, 0)
    put(pmix, (task * S + t0 + gl.minimum(tl, 15)) * PART + j, tot, tag, live)
    gl.barrier()


@gluon.jit(noinline=NOINLINE)
def stage_slice(res_in, comb_in, post_in, pre_in, hc_fn, res_out, task, tag, lin,
                pmix, pl, rl, fl, red, tid, lane, wave,
                S: gl.constexpr, HIDDEN: gl.constexpr, RL_SPARE: gl.constexpr, pend_g=None,
                FOLD: gl.constexpr = True):  # fmt: skip
    """R_new, the layer input and the partial projections of hidden columns
    32 task .. +32, all four streams, every token."""
    c0 = task * COLS
    ROUNDS: gl.constexpr = (S * KT + 511) // 512
    for r0 in gl.static_range(0, ROUNDS, 4):
        slice_rounds(res_in, comb_in, post_in, res_out, task, pl, rl, tid, r0,
                     min(4, ROUNDS - r0), S, HIDDEN, RL_SPARE, pend_g, FOLD)  # fmt: skip
    # fn as the original's two bf16 MFMAs take it: hi + lo, exact in fp32
    fn = hc_fn.to(gl.pointer_type(gl.float32))
    for i in gl.static_range(6):  # MIX x KT = 3072 = 6 x 512
        e = tid + 512 * i
        j = e // KT
        o = (e % KT) // COLS
        f = gl.load(fn + (j * (HC * HIDDEN) + o * HIDDEN + c0 + e % COLS))
        hi = bf16_round(f)
        fl.scatter(hi + bf16_round(f - hi), e, 0)
    gl.barrier()
    # the layer input: R_new's four streams by the incoming pre-mix, products
    # rounded, summed in order; two columns a thread, one bf16 pair a store
    pre = pre_in.to(gl.pointer_type(gl.float32))
    for pas in gl.static_range(0, S * 16, 512):
        idx = pas + tid
        active = idx < S * 16
        t = gl.minimum(idx // 16, S - 1)
        cc = 2 * (idx % 16)
        p0 = gl.load(pre + t * HC)
        p1 = gl.load(pre + (t * HC + 1))
        p2 = gl.load(pre + (t * HC + 2))
        p3 = gl.load(pre + (t * HC + 3))
        rb = t * KT + cc
        x0 = (((0.0 + p0 * rl.gather(rb, 0)) + p1 * rl.gather(rb + 32, 0))
              + p2 * rl.gather(rb + 64, 0)) + p3 * rl.gather(rb + 96, 0)  # fmt: skip
        x1 = (((0.0 + p0 * rl.gather(rb + 1, 0)) + p1 * rl.gather(rb + 33, 0))
              + p2 * rl.gather(rb + 65, 0)) + p3 * rl.gather(rb + 97, 0)  # fmt: skip
        put(lin, (t * HIDDEN + c0 + cc) // 2, bf16_pair(x0, x1), tag, active)
    # sums of squares: token t on wave t % 8, two columns a lane
    for i in gl.static_range((S + 7) // 8):
        t = wave + 8 * i
        tt = gl.minimum(t, S - 1)
        x0 = rl.gather(tt * KT + 2 * lane, 0)
        x1 = rl.gather(tt * KT + 2 * lane + 1, 0)
        sq = sum_desc6(x0 * x0 + x1 * x1, lane)
        put(pmix, (task * S + tt) * PART + MIX, sq, tag, (lane == 0) & (t < S))
    # the mixes, a token tile (16) at a time
    for t0 in gl.static_range(0, S, 16):
        slice_mix(task, t0, min(16, S - t0), tag, pmix, rl, fl, red, tid, S)


# ---------------------------------------------------------------- gate
@gluon.jit
def sigmoid(z):
    """Triton's ``tl.sigmoid``: 1 / (1 + exp2(-z log2 e))."""
    return 1.0 / (1.0 + hw_exp2(z * (-LOG2E)))


@gluon.jit
def _row(v, lane, q):
    return lane_gather(v, (lane - 8) // 4 * 4 + 8 + q)


@gluon.jit
def _col(v, lane, q):
    return lane_gather(v, 8 + 4 * q + (lane - 8) % 4)


@gluon.jit(noinline=NOINLINE)
def stage_gate(t, tag, pmix, red, hc_scale, hc_base, pre_out, post_out, comb_out,
               tid, lane, wave, dl, health, rank,
               S: gl.constexpr, HIDDEN: gl.constexpr):  # fmt: skip
    """Token t's 160 slices, in slice order, -> post / comb / next pre."""
    SLICES: gl.constexpr = HIDDEN // COLS
    GROUPS: gl.constexpr = SLICES // GATE_BLOCK  # 20
    g = tid // PART
    j = tid % PART
    live = tid < GROUPS * PART
    gg = gl.minimum(g, GROUPS - 1)
    base = pmix.to(gl.uint64)
    a0 = base + (((gg * GATE_BLOCK) * S + t) * PART + j).to(gl.uint64) * 8
    st = S * PART * 8
    v0, v1, v2, v3, v4, v5, v6, v7, f = poll8_slices(a0, st, tag, live, dl, health)
    fail(health, f, rank, 3, live)
    acc = v0.to(gl.float32, bitcast=True) + v1.to(gl.float32, bitcast=True)
    acc = acc + v2.to(gl.float32, bitcast=True)
    acc = acc + v3.to(gl.float32, bitcast=True)
    acc = acc + v4.to(gl.float32, bitcast=True)
    acc = acc + v5.to(gl.float32, bitcast=True)
    acc = acc + v6.to(gl.float32, bitcast=True)
    acc = acc + v7.to(gl.float32, bitcast=True)
    red.scatter(acc, gl.where(live, g * PART + j, 4095), 0)
    gl.barrier()
    # wave 0 (a value a lane: the lanes' j clamped to 23; lane-uniform work on
    # the other waves is discarded)
    jj = gl.minimum(lane, MIX - 1)
    mix = red.gather(jj, 0)
    sq = red.gather(gl.zeros_like(jj) + MIX, 0)
    for gi in gl.static_range(1, GROUPS):
        mix = mix + red.gather(gi * PART + jj, 0)
        sq = sq + red.gather(gl.zeros_like(jj) + (gi * PART + MIX), 0)
    scp = hc_scale.to(gl.pointer_type(gl.float32))
    sc0 = gl.load(scp)
    sc1 = gl.load(scp + 1)
    sc2 = gl.load(scp + 2)
    hb = gl.load(hc_base.to(gl.pointer_type(gl.float32)) + jj)
    v = mix * hw_rsq(gl.fma(sq, 1.0 / (HC * HIDDEN), EPS))
    w0 = wave == 0
    pre = sigmoid(gl.fma(v, sc0, hb))
    gl.store(pre_out.to(gl.pointer_type(gl.float32)) + (t * HC + lane), pre + HC_EPS,
             mask=w0 & (lane < HC))  # fmt: skip
    post = sigmoid(gl.fma(v, sc1, hb))
    gl.store(post_out.to(gl.pointer_type(gl.float32)) + (t * HC + lane - HC),
             post * POST_MULT, mask=w0 & (lane >= HC) & (lane < 2 * HC))  # fmt: skip
    cv = gl.fma(v, sc2, hb)
    m = gl.maximum(gl.maximum(_row(cv, lane, 0), _row(cv, lane, 1)),
                   gl.maximum(_row(cv, lane, 2), _row(cv, lane, 3)))  # fmt: skip
    cv = hw_exp2((cv - m) * LOG2E)
    rs = (_row(cv, lane, 0) + _row(cv, lane, 1)) + (_row(cv, lane, 2) + _row(cv, lane, 3))
    cv = gl.fma(cv, hw_rcp(rs), HC_EPS)
    cs = (_col(cv, lane, 0) + _col(cv, lane, 1)) + (_col(cv, lane, 2) + _col(cv, lane, 3))
    cv = cv * hw_rcp(cs + HC_EPS)
    for _ in gl.static_range(SINKHORN - 1):
        rs = (_row(cv, lane, 0) + _row(cv, lane, 1)) + (_row(cv, lane, 2)
                                                        + _row(cv, lane, 3))  # fmt: skip
        cv = cv * hw_rcp(rs + HC_EPS)
        cs = (_col(cv, lane, 0) + _col(cv, lane, 1)) + (_col(cv, lane, 2)
                                                        + _col(cv, lane, 3))  # fmt: skip
        cv = cv * hw_rcp(cs + HC_EPS)
    gl.store(comb_out.to(gl.pointer_type(gl.float32)) + (t * HC * HC + lane - 2 * HC), cv,
             mask=w0 & (lane >= 2 * HC) & (lane < MIX))  # fmt: skip
    gl.barrier()


@gluon.jit
def poll8_slices(a0, st, tag, live, dl, health):
    """Eight consecutive slices' pairs of one (token, j): ``st`` bytes apart."""
    return poll8(a0, a0 + st, a0 + 2 * st, a0 + 3 * st, a0 + 4 * st, a0 + 5 * st,
                 a0 + 6 * st, a0 + 7 * st, tag, live, dl, health)  # fmt: skip


# ---------------------------------------------------------------- norm
@gluon.jit(noinline=NOINLINE)
def stage_norm(t, tag, lin, norm_w, normed, xrdy, red, tid, lane, wave, dl, health,
               rank, HIDDEN: gl.constexpr, x8=None, x8s=None):  # fmt: skip
    """Token t's RMSNorm (rstd = rsq(fma(sum, 1/H, eps)), bf16((x rstd) w)) of
    LIN -> the MoE's bf16 input row (device scope), then XRDY[t]. 256 threads,
    3 chunks of 8 columns at 8 t + 2048 c (threads 256.. mirror thread 255).
    ``x8`` (K1's attention seam, upstream ``store_x8``): vLLM's MXFP8 of each 32
    columns (4 threads) -> X8 / X8S instead, then XRDY[2 t], [2 t + 1] (a flag
    a wqkv K half)."""
    tt = gl.minimum(tid, 255)
    mine = tid < 256
    lb = lin.to(gl.uint64)
    acc = gl.zeros_like(tid).to(gl.float32)
    xs = ()
    for ch in gl.static_range(3):
        colr = tt * 8 + 2048 * ch
        inrow = colr < HIDDEN
        col = gl.minimum(colr, HIDDEN - 8)
        a = lb + ((t * HIDDEN + col) // 2).to(gl.uint64) * 8
        w0, w1, w2, w3, f = poll4(a, a + 8, a + 16, a + 24, tag, mine, dl, health)
        fail(health, f, rank, 4, mine)
        v = (bf_lo(w0), bf_hi(w0), bf_lo(w1), bf_hi(w1), bf_lo(w2), bf_hi(w2),
             bf_lo(w3), bf_hi(w3))  # fmt: skip
        z = ()
        for k in gl.static_range(8):
            z = z + (gl.where(inrow, v[k], 0.0),)
        xs = xs + (z,)
    for ch in gl.static_range(3):
        for k in gl.static_range(8):
            acc = acc + xs[ch][k] * xs[ch][k]
    acc = sum_asc6(acc, lane)
    red.scatter(acc, gl.where((lane == 0) & mine, wave, 4095), 0)
    gl.barrier()
    tot = red.gather(lane % 4, 0)
    tot = xadd(xadd(tot, lane, 1), lane, 2)
    r = hw_rsq(gl.fma(tot, 1.0 / HIDDEN, EPS))
    wp = norm_w.to(gl.pointer_type(gl.bfloat16))
    nb = normed.to(gl.uint64)
    for ch in gl.static_range(3):
        colr = tt * 8 + 2048 * ch
        col = gl.minimum(colr, HIDDEN - 8)
        y = ()
        for k in gl.static_range(8):
            y = y + (bf16_round((xs[ch][k] * r) * gl.load(wp + (col + k)).to(gl.float32)),)
        lo = bf16_pair(y[0], y[1]).to(gl.uint64) | (bf16_pair(y[2], y[3]).to(gl.uint64) << 32)
        hi = bf16_pair(y[4], y[5]).to(gl.uint64) | (bf16_pair(y[6], y[7]).to(gl.uint64) << 32)
        live = mine & (colr < HIDDEN)
        if x8 is None:
            at = nb + ((t * HIDDEN + col) * 2).to(gl.uint64)
            store64(at, lo, live)
            store64(at + 8, hi, live)
        else:
            amax = gl.abs(y[0])
            for k in gl.static_range(1, 8):
                amax = gl.maximum(amax, gl.abs(y[k]))
            amax = gl.maximum(amax, xshfl(amax, lane, 1))
            amax = gl.maximum(amax, xshfl(amax, lane, 2))
            code = vllm_mx_code(amax)
            mul = ((254 - code) << 23).to(gl.float32, bitcast=True)
            w0 = fp8_pack4(clamp_fp8(y[0] * mul), clamp_fp8(y[1] * mul), clamp_fp8(y[2] * mul),
                           clamp_fp8(y[3] * mul))  # fmt: skip
            w1 = fp8_pack4(clamp_fp8(y[4] * mul), clamp_fp8(y[5] * mul), clamp_fp8(y[6] * mul),
                           clamp_fp8(y[7] * mul))  # fmt: skip
            xw = x8.to(gl.pointer_type(gl.uint32)) + t * (HIDDEN // 4) + col // 4
            gl.store(xw, w0, mask=live, cache_modifier=".wt")
            gl.store(xw + 1, w1, mask=live, cache_modifier=".wt")
            gl.store(x8s.to(gl.pointer_type(gl.int32)) + t * (HIDDEN // 32) + col // 32, code,
                     mask=live & (tt % 4 == 0), cache_modifier=".wt")  # fmt: skip
    if x8 is None:
        publish(xrdy, gl.zeros_like(tid) + t, gl.zeros_like(tid) + 1, tag, tid == 0)
    else:
        publish(xrdy, 2 * t + gl.minimum(tid, 1), gl.zeros_like(tid) + 1, tag, tid < 2)
    gl.barrier()
