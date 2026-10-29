# SPDX-License-Identifier: MIT
"""The DS V4.1 MoE's front stages in Gluon: a faithful port of
``vllm/models/deepseek_v41/amd/mono/stages/moe.py`` (vLLM 8321d11)
``stage_xq``, ``stage_router``, ``stage_route`` and ``load_route``:

    xq      (S): the row as MXFP8 for the routed experts (aiter's ceil code,
            amax floored 1e-10) -> X8R / X8RS and for the shared expert (vLLM's
            mxfp8_e4m3_quantize code) -> X8 / X8S, plain words, then X8RDY[t]
    router  (groups x 24 x 4: 16 rows x a quarter of K): bf16 GEMV, the waves'
            partials summed in wave order -> the part's f32 -> LOGIT
    route   (S, wave 0): the parts summed in part order; sqrtsoftplus + bias,
            a lane's 6 experts sorted, top-6 by ballot (ties: the lower lane),
            renormalized x 1.5 -> ROUTE (8 copies)
    table   (every CTA running ug / down): the union U of the picks in LDS"""

import math

from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from ..common.build import NOINLINE

from ..common.mx import clamp_fp8, code_ceil, fp8_pack4, rcp_pow2, vllm_mx_code
from ..common.ops import (
    ballot,
    bf_hi,
    bf_lo,
    hw_exp2,
    lane_gather,
    lanes_below,
    max_desc6,
    popcount32,
    popcount64,
)
from ..common.sync import (
    T512,
    fail,
    poll1,
    poll2,
    poll6x4,
    publish,
    put,
    store32,
    store64,
)

LOG2E = gl.constexpr(1.4426950408889634)
LN2 = gl.constexpr(0.6931471805599453)
ROUTE_COPIES = gl.constexpr(8)
MMA32 = gl.constexpr(gl.amd.AMDMFMALayout(4, [16, 16, 32], False, [8, 1, 1]))
# a router task's operands in registers, as the 16x16x32 MFMA lays them out:
# A [wave][row][k]: lane l row l % 16, 8 consecutive k at 8 (l // 16) a block;
# B [wave][k][token]: lane l token l % 16, the same k (the threads' order is what
# makes the conversion to the operand free: rows / tokens fastest)
A_LD = gl.constexpr(gl.BlockedLayout([1, 1, 8], [1, 16, 4], [8, 1, 1], [1, 2, 0]))
B_LD = gl.constexpr(gl.BlockedLayout([1, 8, 1], [1, 4, 16], [8, 1, 1], [2, 1, 0]))
RED3 = gl.constexpr(gl.BlockedLayout([1, 1, 1], [1, 4, 16], [8, 1, 1], [2, 1, 0]))


# ---------------------------------------------------------------- xq
@gluon.jit
def _mx_words(v, inv):
    """32 values -> 8 dwords of E4M3 (dword j: values 4j .. 4j + 3, RNE)."""
    out = ()
    for j in gl.static_range(8):
        out = out + (fp8_pack4(clamp_fp8(v[4 * j] * inv), clamp_fp8(v[4 * j + 1] * inv),
                               clamp_fp8(v[4 * j + 2] * inv),
                               clamp_fp8(v[4 * j + 3] * inv)),)  # fmt: skip
    return out


@gluon.jit
def _store_group(words_base, codes_base, t, g, ws, code, mine, HIDDEN: gl.constexpr):
    at = words_base.to(gl.uint64) + ((t * (HIDDEN // 4) + g * 8) * 4).to(gl.uint64)
    for j in gl.static_range(4):
        w = ws[2 * j].to(gl.uint64) | (ws[2 * j + 1].to(gl.uint64) << 32)
        store64(at + 8 * j, w, mine)
    cat = codes_base.to(gl.uint64) + ((t * (HIDDEN // 32) + g) * 4).to(gl.uint64)
    store32(cat, code, mine)


@gluon.jit(noinline=NOINLINE)
def stage_xq(t, x, x8r, x8rs, x8, x8s, x8rdy, tag, tid, HIDDEN: gl.constexpr):
    """Token t, a group of 32 a thread (thread g < 160; the rest mirror 159)."""
    G: gl.constexpr = HIDDEN // 32
    g = gl.minimum(tid, G - 1)
    mine = tid < G
    # the normed row: written once, whole 128 B lines, before its XRDY (which
    # the caller polled), never read before: plain loads (16 B each)
    xp = x.to(gl.pointer_type(gl.uint32)) + (t * HIDDEN + g * 32) // 2
    v = ()
    for q in gl.static_range(16):
        w = gl.load(xp + q)
        v = v + (bf_lo(w), bf_hi(w))
    amax = gl.abs(v[0])
    for i in gl.static_range(1, 32):
        amax = gl.maximum(amax, gl.abs(v[i]))
    # the routed experts' (aiter fused_mx_quant_moe_sort): amax floored 1e-10
    code_r = code_ceil(gl.maximum(amax, 1e-10) * (1.0 / 448.0))
    _store_group(x8r, x8rs, t, g, _mx_words(v, rcp_pow2(code_r)), code_r, mine, HIDDEN)
    # the shared expert's: vLLM's mxfp8_e4m3_quantize (2^(127 - code))
    code_v = vllm_mx_code(amax)
    inv_v = ((254 - code_v) << 23).to(gl.float32, bitcast=True)
    _store_group(x8, x8s, t, g, _mx_words(v, inv_v), code_v, mine, HIDDEN)
    publish(x8rdy, gl.zeros_like(tid) + t, gl.zeros_like(tid) + 1, tag, tid == 0)


# ---------------------------------------------------------------- router
@gluon.jit
def router_k(part, w, i, kin, HIDDEN: gl.constexpr):
    """K of a router lane's element: part ``part``'s steps w, w + 8, .. (32
    columns each) of wave w, step ``i``, column ``kin``."""
    PER_WAVE: gl.constexpr = HIDDEN // 32 // 8 // 4  # 5
    return (part * PER_WAVE * 8 + w + 8 * i) * 32 + kin


@gluon.jit(noinline=NOINLINE)
def stage_router(task, gate_w, x, logit, red, tag, tid, S: gl.constexpr,
                 EXPERTS: gl.constexpr, GROUPS: gl.constexpr, PER: gl.constexpr,
                 HIDDEN: gl.constexpr):  # fmt: skip
    """Router task (token group, row tile, K part) -> LOGIT parts, a token tile
    at a time: wave w's K steps w, w + 8, .. of the part (5 x 32 columns) in
    order on its accumulator, the waves summed in wave order. Every weight load,
    then every tile's x loads, issue before the first MFMA."""
    PER_GROUP: gl.constexpr = EXPERTS // 16 * 4
    STEPS: gl.constexpr = HIDDEN // 32 // 8 // 4  # 5
    tile = task % PER_GROUP // 4
    part = task % 4
    if GROUPS > 1:
        g0 = (task // PER_GROUP) * PER
    else:
        g0 = task * 0
    wa = gl.arange(0, 8, layout=gl.SliceLayout(1, gl.SliceLayout(2, A_LD)))
    ra = gl.arange(0, 16, layout=gl.SliceLayout(0, gl.SliceLayout(2, A_LD)))
    ka = gl.arange(0, 32, layout=gl.SliceLayout(0, gl.SliceLayout(1, A_LD)))
    rows = tile * 16 + ra[None, :, None]
    wb = gl.arange(0, 8, layout=gl.SliceLayout(1, gl.SliceLayout(2, B_LD)))
    kb = gl.arange(0, 32, layout=gl.SliceLayout(0, gl.SliceLayout(2, B_LD)))
    nb = gl.arange(0, 16, layout=gl.SliceLayout(0, gl.SliceLayout(1, B_LD)))
    wp = gate_w.to(gl.pointer_type(gl.bfloat16))
    xp = x.to(gl.pointer_type(gl.bfloat16))
    # weights: [wave][row][32 k] a step, nontemporal (read once a step)
    ws = ()
    for i in gl.static_range(STEPS):
        kk = router_k(part, wa[:, None, None], i, ka[None, None, :], HIDDEN)
        ws = ws + (gl.load(wp + rows * HIDDEN + kk, cache_modifier=".cg"),)
    # x: [wave][32 k][token] a step, of each token tile of the group
    xs = ()
    for j0 in gl.static_range(0, PER, 16):
        n = gl.minimum(min(16, PER - j0), S - (g0 + j0))
        tok = g0 + j0 + gl.maximum(gl.minimum(nb[None, None, :], n - 1), 0)
        for i in gl.static_range(STEPS):
            kx = router_k(part, wb[:, None, None], i, kb[None, :, None], HIDDEN)
            xs = xs + (gl.load(xp + tok * HIDDEN + kx),)
    for j0 in gl.static_range(0, PER, 16):
        n = gl.minimum(min(16, PER - j0), S - (g0 + j0))
        acc = gl.zeros((8, 16, 16), gl.float32, MMA32)
        for i in gl.static_range(STEPS):
            a = gl.convert_layout(ws[i], gl.DotOperandLayout(0, MMA32, 8))
            b = gl.convert_layout(xs[(j0 // 16) * STEPS + i], gl.DotOperandLayout(1, MMA32, 8))
            acc = gl.amd.cdna4.mfma(a, b, acc)
        red.slice(0, 2048).store(gl.reshape(gl.convert_layout(acc, RED3), (2048,)))
        gl.barrier()
        r = tid % 16
        tl = gl.minimum(tid // 16, 15)
        v = red.gather(r * 16 + tl, 0)
        for w in gl.static_range(1, 8):
            v = v + red.gather(w * 256 + r * 16 + tl, 0)
        live = tid < 16 * n
        t = g0 + j0 + tl
        put(logit, (t * EXPERTS + tile * 16 + r) * 4 + part, v, tag, live)
        gl.barrier()


# ---------------------------------------------------------------- router
# ---------------------------------------------------------------- route
@gluon.jit
def _score(x):
    """sqrt(log2(1 + exp2(x log2e)) ln2), x itself past 20 (topk_gating)."""
    lg = gl.inline_asm_elementwise("v_log_f32 $0, $1", "=v,v",
                                   [1.0 + hw_exp2(x * LOG2E)], gl.float32, True, 1)  # fmt: skip
    sp = gl.where(x > 20.0, x, lg * LN2)
    return gl.sqrt_rn(sp)


@gluon.jit
def _cx(vp, vq, op, oq, ip, iq):
    """A comparator: descending, ties kept in index order (swap if vp < vq)."""
    sw = vp < vq
    return (gl.where(sw, vq, vp), gl.where(sw, vp, vq), gl.where(sw, oq, op),
            gl.where(sw, op, oq), gl.where(sw, iq, ip), gl.where(sw, ip, iq))  # fmt: skip


@gluon.jit
def _ctz64(ball):
    lo = (ball & 0xFFFFFFFF).to(gl.uint32)
    hi = (ball >> 32).to(gl.uint32)
    flo = gl.inline_asm_elementwise("v_ffbl_b32 $0, $1", "=v,v", [lo], gl.int32, True, 1)
    fhi = gl.inline_asm_elementwise("v_ffbl_b32 $0, $1", "=v,v", [hi], gl.int32, True, 1)
    return gl.where(lo != 0, flo, gl.where(hi != 0, 32 + fhi, 0))


@gluon.jit(noinline=NOINLINE)
def stage_route(t, logit, route, bias, tag, tid, lane, wave, dl, health, rank,
                S: gl.constexpr, EXPERTS: gl.constexpr, TOPK: gl.constexpr,
                SCALE: gl.constexpr):  # fmt: skip
    """VLLM's topk_hash_softplus_sqrt for token t on wave 0 (f32 logits)."""
    gl.static_assert(EXPERTS == 384, "6 experts a lane (the target's routing)")
    w0 = wave == 0
    lb = logit.to(gl.uint64)
    e0 = lane
    base = lb + ((t * EXPERTS + e0) * 4).to(gl.uint64) * 8
    got = poll6x4(base, base + 64 * 32, base + 128 * 32, base + 192 * 32,
                  base + 256 * 32, base + 320 * 32, tag, w0, dl, health)  # fmt: skip
    fail(health, got[28], rank, 5, w0)
    bp = bias.to(gl.pointer_type(gl.float32))
    # the parts summed in part order (vLLM's gate: f32 out), each expert
    sv = ()
    so = ()
    si = ()
    for i in gl.static_range(6):
        tot = 0.0 + got[4 * i].to(gl.uint32).to(gl.float32, bitcast=True)
        for p in gl.static_range(1, 4):
            tot = tot + got[4 * i + p].to(gl.uint32).to(gl.float32, bitcast=True)
        e = lane + 64 * i
        sc = _score(tot)
        # aiter's NaN guards: a NaN score weighs 0 and never wins
        so = so + (gl.where(sc != sc, 0.0, sc),)
        sv = sv + (gl.maximum(sc + gl.load(bp + e), float("-inf")),)
        si = si + (e,)
    v0, v1, v2, v3, v4, v5 = sv
    o0, o1, o2, o3, o4, o5 = so
    i0, i1, i2, i3, i4, i5 = si
    # SORT_NETS[6]
    v0, v1, o0, o1, i0, i1 = _cx(v0, v1, o0, o1, i0, i1)
    v2, v3, o2, o3, i2, i3 = _cx(v2, v3, o2, o3, i2, i3)
    v4, v5, o4, o5, i4, i5 = _cx(v4, v5, o4, o5, i4, i5)
    v0, v2, o0, o2, i0, i2 = _cx(v0, v2, o0, o2, i0, i2)
    v1, v4, o1, o4, i1, i4 = _cx(v1, v4, o1, o4, i1, i4)
    v3, v5, o3, o5, i3, i5 = _cx(v3, v5, o3, o5, i3, i5)
    v0, v1, o0, o1, i0, i1 = _cx(v0, v1, o0, o1, i0, i1)
    v2, v3, o2, o3, i2, i3 = _cx(v2, v3, o2, o3, i2, i3)
    v4, v5, o4, o5, i4, i5 = _cx(v4, v5, o4, o5, i4, i5)
    v1, v2, o1, o2, i1, i2 = _cx(v1, v2, o1, o2, i1, i2)
    v3, v4, o3, o4, i3, i4 = _cx(v3, v4, o3, o4, i3, i4)
    v2, v3, o2, o3, i2, i3 = _cx(v2, v3, o2, o3, i2, i3)
    cursor = gl.zeros_like(lane)
    total = gl.zeros_like(lane).to(gl.float32)
    my_w = gl.zeros_like(lane).to(gl.float32)
    my_id = gl.zeros_like(lane)
    for k in gl.static_range(TOPK):
        cv = gl.where(cursor == 0, v0, gl.where(cursor == 1, v1, gl.where(
            cursor == 2, v2, gl.where(cursor == 3, v3, gl.where(
                cursor == 4, v4, gl.where(cursor == 5, v5, float("-inf")))))))  # fmt: skip
        ci = gl.where(cursor == 0, i0, gl.where(cursor == 1, i1, gl.where(
            cursor == 2, i2, gl.where(cursor == 3, i3, gl.where(
                cursor == 4, i4, gl.where(cursor == 5, i5, 0))))))  # fmt: skip
        co = gl.where(cursor == 0, o0, gl.where(cursor == 1, o1, gl.where(
            cursor == 2, o2, gl.where(cursor == 3, o3, gl.where(
                cursor == 4, o4, gl.where(cursor == 5, o5, 0.0))))))  # fmt: skip
        best = max_desc6(cv, lane)
        win = _ctz64(ballot(cv == best))
        win_id = lane_gather(ci, win)
        won = (cursor < 6) & (ci == win_id)
        w = lane_gather(gl.where(won, co, 0.0), win_id & 63)
        cursor = cursor + gl.where(won, 1, 0)
        total = total + w
        my_w = gl.where(lane == k, w, my_w)
        my_id = gl.where(lane == k, win_id, my_id)
    scale = SCALE / gl.maximum(total, 1e-20)
    mine = w0 & (lane < TOPK)
    for r in gl.static_range(ROUTE_COPIES):
        at = (r * S + t) * 2 * TOPK + lane
        put(route, at, my_id, tag, mine)
        put(route, at + TOPK, my_w * scale, tag, mine)
    gl.barrier()


# ---------------------------------------------------------------- the route table
@gluon.jit
def lane_prefix(v, lane):
    """The wave's inclusive prefix sum of v, lanes ascending."""
    incl = v
    for d in gl.static_range(6):
        moved = lane_gather(incl, gl.maximum(lane - (1 << d), 0))
        incl = incl + gl.where(lane >= (1 << d), moved, 0)
    return incl


@gluon.jit(noinline=NOINLINE)
def load_route(tab, red, route, bid, tag, tid, lane, wave, dl, health, rank,
               S: gl.constexpr, TOPK: gl.constexpr, EXPERTS: gl.constexpr,
               MAX_U: gl.constexpr, MULTI: gl.constexpr, SLOT_TILES: gl.constexpr,
               EXPERT_SLOT0: gl.constexpr, ORDER0: gl.constexpr,
               FIRST0: gl.constexpr, MASK0: gl.constexpr,
               TILE_FIRST0: gl.constexpr, TILE_SLOT0: gl.constexpr,
               TAB_SPARE: gl.constexpr):  # fmt: skip
    """Every token's (ids, weights) into LDS ``tab``, then the union U of the
    ids (ascending) and its derived tables (upstream ``load_route``)."""
    N: gl.constexpr = S * TOPK
    BASE: gl.constexpr = 2 * N
    copy_at = (bid % ROUTE_COPIES) * (2 * N)
    rb = route.to(gl.uint64)
    # a thread's route words in one poll batch (2 N <= 576: two passes)
    i0 = tid
    i1 = tid + 512
    a0 = rb + (copy_at + gl.minimum(i0, 2 * N - 1)).to(gl.uint64) * 8
    a1 = rb + (copy_at + gl.minimum(i1, 2 * N - 1)).to(gl.uint64) * 8
    if 2 * N > 512:
        w0, w1, f = poll2(a0, a1, tag, gl.zeros_like(tid) == 0, dl, health)
    else:
        w0, f = poll1(a0, tag, gl.zeros_like(tid) == 0, dl, health)
        w1 = w0
    fail(health, f, rank, 6, gl.zeros_like(tid) == 0)
    tab.scatter(w0.to(gl.int32, bitcast=True), gl.where(i0 < 2 * N, i0, TAB_SPARE), 0)
    if 2 * N > 512:
        tab.scatter(w1.to(gl.int32, bitcast=True), gl.where(i1 < 2 * N, i1, TAB_SPARE), 0)
    for pas in gl.static_range(0, 2 * EXPERTS, 512):
        idx = pas + tid
        tab.scatter(gl.zeros_like(tid), gl.where(idx < 2 * EXPERTS, MASK0 + idx, TAB_SPARE), 0)
    gl.barrier()
    # each pick ORs its token's bit into its expert's bitmap
    pk = gl.minimum(tid, N - 1)
    pt = pk // TOPK
    eid = tab.gather(pt * 2 * TOPK + pk % TOPK, 0)
    bit = gl.where(tid < N, 1 << (pt % 32), 0)
    tab.atomic_scatter_or(bit, MASK0 + eid * 2 + pt // 32, 0)
    gl.barrier()
    # a chunk of 64 experts a wave: its counts, prefixes and totals
    CHUNKS: gl.constexpr = EXPERTS // 64
    e = lane + 64 * gl.minimum(wave, CHUNKS - 1)
    count = popcount32(tab.gather(MASK0 + e * 2, 0)) + popcount32(tab.gather(MASK0 + e * 2 + 1, 0))
    ball = ballot(count != 0)
    incl = lane_prefix(count, lane)
    tiles = (count + 15) // 16
    tincl = lane_prefix(tiles, lane) if MULTI else tiles
    tot_u = popcount64(ball)
    tot_p = lane_gather(incl, gl.zeros_like(lane) + 63)
    tot_t = lane_gather(tincl, gl.zeros_like(lane) + 63)
    head = (wave < CHUNKS) & (lane == 0)
    red.scatter(tot_u.to(gl.float32, bitcast=True), gl.where(head, wave * 3, 4095), 0)
    red.scatter(tot_p.to(gl.float32, bitcast=True), gl.where(head, wave * 3 + 1, 4095), 0)
    red.scatter(tot_t.to(gl.float32, bitcast=True), gl.where(head, wave * 3 + 2, 4095), 0)
    gl.barrier()
    n_u = gl.zeros_like(lane)
    first = gl.zeros_like(lane)
    tile0 = gl.zeros_like(lane)
    for j in gl.static_range(CHUNKS - 1):
        below = j < wave
        n_u = n_u + gl.where(below, red.gather(gl.zeros_like(lane) + j * 3, 0).to(gl.int32, bitcast=True), 0)
        first = first + gl.where(below, red.gather(gl.zeros_like(lane) + j * 3 + 1, 0).to(gl.int32, bitcast=True), 0)
        tile0 = tile0 + gl.where(below, red.gather(gl.zeros_like(lane) + j * 3 + 2, 0).to(gl.int32, bitcast=True), 0)
    slot = n_u + lanes_below(ball)
    inw = wave < CHUNKS
    used = inw & (count != 0)
    tab.scatter(slot, gl.where(inw, EXPERT_SLOT0 + e, TAB_SPARE), 0)
    tab.scatter(e, gl.where(used, BASE + 1 + slot, TAB_SPARE), 0)
    tab.scatter(first + incl - count, gl.where(used, FIRST0 + slot, TAB_SPARE), 0)
    if MULTI:
        tf = tile0 + tincl - tiles
        tab.scatter(tf, gl.where(used, TILE_FIRST0 + slot, TAB_SPARE), 0)
        for j in gl.static_range(SLOT_TILES):
            tab.scatter(slot, gl.where(inw & (j < tiles), TILE_SLOT0 + tf + j, TAB_SPARE), 0)
    last = (wave == CHUNKS - 1) & (lane == 0)
    n_all = n_u + tot_u
    tab.scatter(n_all, gl.where(last, BASE, TAB_SPARE), 0)
    tab.scatter(gl.zeros_like(lane) + N, gl.where(last, FIRST0 + n_all, TAB_SPARE), 0)
    if MULTI:
        tab.scatter(tile0 + tot_t, gl.where(last, TILE_FIRST0 + n_all, TAB_SPARE), 0)
    gl.barrier()
    # each pick: its slot, and its place in the pick order (the tokens below
    # its own that picked the same expert)
    pslot = tab.gather(EXPERT_SLOT0 + eid, 0)
    live = tid < N
    tab.scatter(pslot, gl.where(live, BASE + 1 + MAX_U + tid, TAB_SPARE), 0)
    rank_ = gl.zeros_like(tid)
    for wd in gl.static_range(2):
        word = tab.gather(MASK0 + eid * 2 + wd, 0)
        lo = pt - 32 * wd
        keep = gl.where(lo >= 32, -1, gl.where(lo > 0, (1 << gl.maximum(lo, 0)) - 1, 0))
        rank_ = rank_ + popcount32(word & keep)
    pos = tab.gather(FIRST0 + pslot, 0) + rank_
    tab.scatter(tid, gl.where(live, ORDER0 + pos, TAB_SPARE), 0)
    gl.barrier()
