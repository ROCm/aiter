# SPDX-License-Identifier: MIT
"""The DS V4.1 MoE's down, combine and reduce stages in Gluon (``stages/moe.py``
``stage_down``, ``down_shared``, ``load_smid_tile``, ``down_combine``,
``stage_reduce``; vLLM 8321d11) on batched scaled MFMA (patched Triton):

    down    (320 tasks of 16 hidden rows; a CTA's one or two): the shared
            expert's down projection (vLLM's MXFP8 linear, its SMQ / SMC input)
            -> bf16 (LDS ``sdl``); then the routed experts of U, a wave a (task,
            slot) -- expert e's w2 rows (MXFP4, aiter's shuffle) times its picks'
            MXFP8 MID rows -- each pick's contribution -> LDS ``contrib``
    combine each token's routed contributions x routing weight, bf16, summed
            in bf16 in its top-k order, + the shared expert's -> bf16 pairs
            pushed to every rank's FFN region
    reduce  (160, 32 columns): the ranks' partials in rank order -> ``out``"""

from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from ..common.build import NOINLINE

from ..common.ops import bf16_pair, bf16_round
from ..common.sync import fail, load_dev32, poll1, poll4, put
from .moe_shared import AD, BD, MMA_S, SH3, _ar3, gather_nd, scatter_nd, wave_sum
from .moe_ug import scale_index
from .seam import sum_ranks

# MFMA operands loaded in the MFMA's own orders (free to convert): A [wave][row][k]
# lane row + 16 kq, B [wave][k][token] lane token + 16 kq, 16 K bytes a lane;
# scales a byte a lane, lane row (token) + 16 group
A_LD = gl.constexpr(gl.BlockedLayout([1, 1, 16], [1, 16, 4], [8, 1, 1], [1, 2, 0]))
S_LD = gl.constexpr(gl.BlockedLayout([1, 1, 1], [1, 16, 4], [8, 1, 1], [1, 2, 0]))
B_LD = gl.constexpr(gl.BlockedLayout([1, 16, 1], [1, 4, 16], [8, 1, 1], [2, 1, 0]))
SB_LD = gl.constexpr(gl.BlockedLayout([1, 1, 4], [1, 16, 4], [8, 1, 1], [2, 1, 0]))
C_LD = gl.constexpr(gl.BlockedLayout([1, 1, 1], [1, 4, 16], [8, 1, 1], [2, 1, 0]))
T512 = gl.constexpr(gl.BlockedLayout([1], [64], [8], [0]))
UNIT = gl.constexpr(127)


# ---------------------------------------------------------------- the shared down
@gluon.jit(noinline=NOINLINE)
def load_smid_tile(smq, smc, t0, n, sx32, sx8, sc8, tag, tid, dl, health, rank,
                   SH_INTER: gl.constexpr):  # fmt: skip
    """The shared expert's MXFP8 intermediate of tokens t0 .. t0 + n (SMQ / SMC:
    tagged words) into LDS, step-major: ``sx`` [8 R steps][16 tokens][128 bytes]
    (``sx32`` its u32 view, ``sx8`` its uint8 view; R = ``sh_rounds``: a wave
    steps w, w + 8, ..), ``sc8`` [8 R][16][4] codes; K past SH_INTER zero,
    codes past it unit."""
    R: gl.constexpr = sh_rounds(SH_INTER)
    WORDS: gl.constexpr = SH_INTER // 4
    GROUPS: gl.constexpr = SH_INTER // 32
    sx8.store(gl.zeros((8 * R, 16, 128), gl.uint8,
                       gl.BlockedLayout([1, 1, 16], [1, 8, 8], [8, 1, 1], [2, 1, 0])))  # fmt: skip
    sc8.store(gl.full((8 * R, 16, 4), UNIT, gl.uint8, SB_LD))
    gl.barrier()
    sb = smq.to(gl.uint64)
    # n x WORDS words, 4 a thread in one poll batch, then 1 more
    TOT: gl.constexpr = 16 * WORDS
    for r0 in gl.static_range(0, TOT, 2048):
        q0 = r0 + tid
        q1 = q0 + 512
        q2 = q0 + 1024
        q3 = q0 + 1536
        a0 = sb + ((t0 + gl.minimum(q0 // WORDS, n - 1)) * WORDS + q0 % WORDS).to(gl.uint64) * 8
        a1 = sb + ((t0 + gl.minimum(q1 // WORDS, n - 1)) * WORDS + q1 % WORDS).to(gl.uint64) * 8
        a2 = sb + ((t0 + gl.minimum(q2 // WORDS, n - 1)) * WORDS + q2 % WORDS).to(gl.uint64) * 8
        a3 = sb + ((t0 + gl.minimum(q3 // WORDS, n - 1)) * WORDS + q3 % WORDS).to(gl.uint64) * 8
        l0 = q0 < n * WORDS
        w0, w1, w2, w3, f = poll4(a0, a1, a2, a3, tag, l0, dl, health)
        fail(health, f, rank, 9, l0)
        for qi in gl.static_range(4):
            q = q0 + 512 * qi
            wv = (w0, w1, w2, w3)[qi]
            live = q < n * WORDS
            ww = q % WORDS
            at = ((ww // 32) * 16 + q // WORDS) * 32 + ww % 32
            # a masked lane writes 0 to the last word (the last step: K past
            # the intermediate, read as zero)
            sx32.scatter(gl.where(live, wv, 0), gl.where(live, at, 4096 * R - 1), 0)
    for c0 in gl.static_range(0, 16 * GROUPS, 512):
        cq = c0 + tid
        clive = cq < n * GROUPS
        ca = smc.to(gl.uint64) + ((t0 + gl.minimum(cq // GROUPS, n - 1)) * GROUPS
                                  + cq % GROUPS).to(gl.uint64) * 8  # fmt: skip
        cv, f = poll1(ca, tag, clive, dl, health)
        fail(health, f, rank, 10, clive)
        g = cq % GROUPS
        cat = ((g // 4) * 16 + cq // GROUPS) * 4 + g % 4
        sc8._reinterpret(gl.uint8, [512 * R], gl.SwizzledSharedLayout(1, 1, 1, [0])).scatter(
            gl.where(clive, cv, UNIT).to(gl.uint8), gl.where(clive, cat, 512 * R - 1), 0)
    gl.barrier()


@gluon.constexpr_function
def sh_rounds(sh_inter):
    """The shared down's K rounds: 8 steps of 128 a round (a step a wave)."""
    return -(-sh_inter // 1024)


@gluon.jit(noinline=NOINLINE)
def down_shared_task(task, sw2, sw2_s, sx8, sc8, SH_INTER: gl.constexpr, HIDDEN: gl.constexpr):
    """Rows 16 task .. of the shared down projection, [8][16][16] wave partials:
    wave w's K steps w, w + 8, .. (vLLM's row-major e4m3 [HIDDEN, SH_INTER])."""
    GROUPS: gl.constexpr = SH_INTER // 32
    R: gl.constexpr = sh_rounds(SH_INTER)
    SAL: gl.constexpr = gl.amd.cdna4.get_mfma_scale_layout(AD, [8, 16, 4])
    SBL: gl.constexpr = gl.amd.cdna4.get_mfma_scale_layout(BD, [8, 16, 4])
    wa, ra, ka = _ar3(8, 16, 128, A_LD)
    ws_, rs_, gs_ = _ar3(8, 16, 4, S_LD)
    row = task * 16 + ra
    acc = gl.zeros((8, 16, 16), gl.float32, MMA_S)
    for r in gl.static_range(R):
        k = (r * 8 + wa) * 128 + ka
        # masked, not clamped: a clamped index is not contiguous (byte loads)
        a = gl.load(sw2.to(gl.pointer_type(gl.uint8)) + row * SH_INTER + k, mask=k < SH_INTER,
                    other=0, cache_modifier=".cg")  # fmt: skip
        grp = (r * 8 + ws_) * 4 + gs_
        sa = gl.load(sw2_s.to(gl.pointer_type(gl.uint8)) + ((task * 16 + rs_) // 32) * GROUPS
                     + grp, mask=grp < GROUPS, other=UNIT)  # fmt: skip
        b = sx8.slice(r * 8, 8, 0).permute((0, 2, 1)).load(BD)
        sb = sc8.slice(r * 8, 8, 0).load(SBL)
        acc = gl.amd.cdna4.mfma_scaled(gl.convert_layout(a, AD), gl.convert_layout(sa, SAL),
                                       "e4m3", b, sb, "e4m3", acc)  # fmt: skip
    return acc


# ---------------------------------------------------------------- the routed down
@gluon.jit
def down_slot(task, u, nu, tab, w2, w2_s, mid, mids, contrib, j, NT: gl.constexpr,
              S: gl.constexpr, TOPK: gl.constexpr, FIRST0: gl.constexpr,
              ORDER0: gl.constexpr, INTER: gl.constexpr, INTER_REAL: gl.constexpr,
              HIDDEN: gl.constexpr, SCALE_COLS: gl.constexpr, J0: gl.constexpr = 0):  # fmt: skip
    """Down task ``task`` (the CTA's j-th) on wave w's slot u + w: the MFMA of the slot's picks
    first + J0 .. + min(count - J0, 16) over the INTER / 128 K steps (J0: a
    slot's pick tile past its first, 16 a tile), each pick's 16 rows ->
    ``contrib[j][pick][16]``."""
    STEPS: gl.constexpr = INTER // 128
    MID_BYTES: gl.constexpr = INTER
    PICKS: gl.constexpr = S * TOPK
    SAL: gl.constexpr = gl.amd.cdna4.get_mfma_scale_layout(AD, [8, 16, 4])
    SBL: gl.constexpr = gl.amd.cdna4.get_mfma_scale_layout(BD, [8, 16, 4])
    base_e = 2 * S * TOPK + 1
    # A: expert e_w's w2 rows 16 task .., its scale codes
    wa, ra, kb = _ar3(8, 16, 64, A_LD)
    ua = gl.minimum(u + wa, nu - 1)
    ea = gather_nd(tab, base_e + ua)
    ws_, rs_, gs_ = _ar3(8, 16, 4, S_LD)
    us = gl.minimum(u + ws_, nu - 1)
    es = gather_nd(tab, base_e + us)
    # B: the picks' MID rows, their codes (MIDS: a dword a code)
    wb, kk, nb = _ar3(8, 128, 16, B_LD)
    ub = gl.minimum(u + wb, nu - 1)
    fb = gather_nd(tab, FIRST0 + ub) + J0
    cb = gather_nd(tab, FIRST0 + ub + 1) - fb
    rowb = fb + gl.minimum(nb, gl.maximum(cb - 1, 0))
    wc, nc, gc = _ar3(8, 16, 4, S_LD)
    uc = gl.minimum(u + wc, nu - 1)
    fc = gather_nd(tab, FIRST0 + uc) + J0
    ccount = gather_nd(tab, FIRST0 + uc + 1) - fc
    rowc = fc + gl.minimum(nc, gl.maximum(ccount - 1, 0))
    w2p = w2.to(gl.pointer_type(gl.uint8))
    w2sp = w2_s.to(gl.pointer_type(gl.uint8))
    midp = mid.to(gl.pointer_type(gl.uint8))
    acc = gl.zeros((8, 16, 16), gl.float32, MMA_S)
    for st in gl.static_range(STEPS):
        off = (ea.to(gl.int64) * (HIDDEN * INTER // 2) + task * (16 * INTER // 2) + st * 1024
               + ((kb // 16) * 16 + ra) * 16 + kb % 16)  # fmt: skip
        a = gl.load(w2p + off, cache_modifier=".cg")
        sa = gl.load(w2sp + scale_index(es * HIDDEN + task * 16 + rs_, st * 4 + gs_, SCALE_COLS),
                     cache_modifier=".cg")  # fmt: skip
        kg = st * 128 + kk
        bv = gl.load(midp + rowb * MID_BYTES + kg)
        # past the rank's real intermediate MID is never written: zero
        bv = gl.where(kg < INTER_REAL, bv, 0)
        grp = st * 4 + gc
        ci = rowc * (INTER // 32) + (grp % 4) * STEPS + grp // 4
        sbv = load_dev32(mids.to(gl.uint64) + (ci * 4).to(gl.uint64)).to(gl.uint8)
        sbv = gl.where(grp * 32 < INTER_REAL, sbv, UNIT)
        acc = gl.amd.cdna4.mfma_scaled(gl.convert_layout(a, AD), gl.convert_layout(sa, SAL), "e2m1",
                                       gl.convert_layout(bv, BD), gl.convert_layout(sbv, SBL),
                                       "e4m3", acc)  # fmt: skip
    # each live pick's 16 rows -> contrib[j][pick][16]
    o = gl.convert_layout(acc, C_LD)
    wo, ro, no = _ar3(8, 16, 16, C_LD)
    uo = u + wo
    uoc = gl.minimum(uo, nu - 1)
    fo = gather_nd(tab, FIRST0 + uoc) + J0
    co = gather_nd(tab, FIRST0 + uoc + 1) - fo
    pick = gather_nd(tab, ORDER0 + fo + gl.minimum(no, gl.maximum(co - 1, 0)))
    live = (uo < nu) & (no < co)
    SPARE: gl.constexpr = 2 * PICKS * 16
    scatter_nd(contrib, o, gl.where(live, (j * PICKS + pick) * 16 + ro, SPARE))


@gluon.jit
def poll_slot_flags(u, nu, tab, ugf, tag, tid, dl, health, rank, UG_PARTS: gl.constexpr,
                    MULTI: gl.constexpr, SLOT_TILES: gl.constexpr,
                    TILE_FIRST0: gl.constexpr):  # fmt: skip
    """Wave w's slot u's ug flags: its MID rows written. One token tile: lane
    l < UG_PARTS polls part l; past it (upstream ``down_ug_flags``) lane l the
    flag of (the slot's pick tile l / UG_PARTS -- its last past its tiles --,
    part l % UG_PARTS)."""
    lane = tid % 64
    uc = gl.minimum(u, nu - 1)
    if MULTI:
        PER: gl.constexpr = SLOT_TILES * UG_PARTS
        gl.static_assert(PER <= 64)
        tile0 = tab.gather(TILE_FIRST0 + uc, 0)
        tiles = tab.gather(TILE_FIRST0 + uc + 1, 0) - tile0
        f = gl.minimum(lane, PER - 1)
        flag = (tile0 + gl.minimum(f // UG_PARTS, tiles - 1)) * UG_PARTS + f % UG_PARTS
        live = (lane < PER) & (u < nu)
    else:
        flag = uc * UG_PARTS + gl.minimum(lane, UG_PARTS - 1)
        live = (lane < UG_PARTS) & (u < nu)
    _, f = poll1(ugf.to(gl.uint64) + flag.to(gl.uint64) * 8, tag, live, dl, health)
    fail(health, f, rank, 11, live)


@gluon.jit
def round_picks(u, nu, tab, tid, FIRST0: gl.constexpr):
    """The most picks of a round's slots u .. u + 8 (uniform)."""
    us = u + tid
    uc = gl.minimum(us, nu - 1)
    n = tab.gather(FIRST0 + uc + 1, 0) - tab.gather(FIRST0 + uc, 0)
    return gl.max(gl.where((tid < 8) & (us < nu), n, 0), axis=0)


@gluon.jit(noinline=NOINLINE)
def stage_down(t0_task, t1_task, tab, w2, w2_s, mid, mids, ugf, contrib, tag, tid, dl, health,
               rank, NT: gl.constexpr, S: gl.constexpr, TOPK: gl.constexpr,
               FIRST0: gl.constexpr, ORDER0: gl.constexpr, UG_PARTS: gl.constexpr,
               INTER: gl.constexpr, INTER_REAL: gl.constexpr, HIDDEN: gl.constexpr,
               SCALE_COLS: gl.constexpr, MULTI: gl.constexpr, SLOT_TILES: gl.constexpr,
               TILE_FIRST0: gl.constexpr):  # fmt: skip
    """The routed down of this CTA's NT tasks: rounds of the slots of U, a wave a
    (task, slot); a round's slots' ug flags polled before their MID is read. A
    slot of more than 16 picks (S > 16): its next pick tiles' passes, when any
    of the round's slots has them."""
    SLOTS: gl.constexpr = 2 // NT
    zero = gl.zeros_like(tid)
    nu = gl.max(tab.gather(zero + 2 * S * TOPK, 0), axis=0)
    wave = tid // 64
    for rnd in range(0, (nu + 8 * SLOTS - 1) // (8 * SLOTS)):
        for i in gl.static_range(SLOTS):
            ut = wave + 8 * (SLOTS * rnd + i)
            poll_slot_flags(ut, nu, tab, ugf, tag, tid, dl, health, rank, UG_PARTS, MULTI,
                            SLOT_TILES, TILE_FIRST0)  # fmt: skip
            u = 8 * (SLOTS * rnd + i)  # wave w takes slot u + w
            # TODO(perf): FlyDSL issues the round's weights before this poll
            down_slot(t0_task, u, nu, tab, w2, w2_s, mid, mids, contrib, 0, NT, S, TOPK,
                      FIRST0, ORDER0, INTER, INTER_REAL, HIDDEN, SCALE_COLS)
            if NT == 2:
                down_slot(t1_task, u, nu, tab, w2, w2_s, mid, mids, contrib, 1, NT, S, TOPK,
                          FIRST0, ORDER0, INTER, INTER_REAL, HIDDEN, SCALE_COLS)
            if S > 16:
                most = round_picks(u, nu, tab, tid, FIRST0)
                for j0 in gl.static_range(16, S, 16):
                    if most > j0:
                        down_slot(t0_task, u, nu, tab, w2, w2_s, mid, mids, contrib, 0, NT, S,
                                  TOPK, FIRST0, ORDER0, INTER, INTER_REAL, HIDDEN, SCALE_COLS,
                                  j0)  # fmt: skip
                        if NT == 2:
                            down_slot(t1_task, u, nu, tab, w2, w2_s, mid, mids, contrib, 1, NT,
                                      S, TOPK, FIRST0, ORDER0, INTER, INTER_REAL, HIDDEN,
                                      SCALE_COLS, j0)  # fmt: skip
    gl.barrier()


# ---------------------------------------------------------------- combine / reduce
@gluon.jit(noinline=NOINLINE)
def down_combine(task, j, t0, n, tab, contrib, sdl, pair, peers, rank, tag, tid,
                 S: gl.constexpr, TOPK: gl.constexpr, TP: gl.constexpr,
                 HIDDEN: gl.constexpr, FFN_OFF: gl.constexpr):  # fmt: skip
    """Task ``task`` (the CTA's j-th), tokens t0 .. t0 + n: the routed
    contributions x weight, bf16, added in bf16 in top-k order (aiter's stage 2:
    bf16 atomics in no fixed order), + the shared expert's, bf16 -> pushed to
    every rank's FFN region."""
    PICKS: gl.constexpr = S * TOPK
    r = tid % 16
    tl = gl.minimum(tid // 16, 15)
    t = t0 + tl
    live = tid < 16 * n
    shared = sdl.gather((j * S + t) * 16 + r, 0)
    routed = gl.zeros_like(shared)
    for k in gl.static_range(TOPK):
        w = tab.gather(t * 2 * TOPK + TOPK + k, 0).to(gl.float32, bitcast=True)
        v = contrib.gather((j * PICKS + t * TOPK + k) * 16 + r, 0)
        routed = bf16_round(routed + bf16_round(v * w))
    pair.scatter(bf16_round(routed + shared), gl.where(live, tid, 511), 0)
    gl.barrier()
    rp = tid % 8
    tl2 = gl.minimum(tid // 8, 15)
    v0 = pair.gather(tl2 * 16 + 2 * rp, 0)
    v1 = pair.gather(tl2 * 16 + 2 * rp + 1, 0)
    col = task * 16 + 2 * rp
    at = (rank * S + t0 + tl2) * (HIDDEN // 2) + col // 2
    live2 = tid < 8 * n
    word = bf16_pair(v0, v1)
    for p in gl.static_range(TP):
        put(peers[p] + FFN_OFF, at, word, tag, live2, True)
    gl.barrier()


@gluon.jit(noinline=NOINLINE)
def stage_reduce(task, own, out, tag, tid, dl, health, rank, S: gl.constexpr,
                 TP: gl.constexpr, HIDDEN: gl.constexpr, FFN_OFF: gl.constexpr):  # fmt: skip
    """Columns 32 task ..: the ranks' partials summed in rank order -> bf16 -> out."""
    op = out.to(gl.pointer_type(gl.bfloat16))
    for pas in gl.static_range(0, S * 16, 512):
        idx = pas + tid
        live = idx < S * 16
        t = gl.minimum(idx // 16, S - 1)
        col = task * 32 + 2 * (idx % 16)
        lo, hi = sum_ranks(own + FFN_OFF, t * (HIDDEN // 2) + col // 2, S * (HIDDEN // 2), tag,
                           live, dl, health, rank, 12, TP)  # fmt: skip
        gl.store(op + t * HIDDEN + col, lo.to(gl.bfloat16), mask=live)
        gl.store(op + t * HIDDEN + col + 1, hi.to(gl.bfloat16), mask=live)
