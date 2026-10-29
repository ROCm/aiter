# SPDX-License-Identifier: MIT
"""The DS V4.1 FFN launch in Gluon: ``layer.build_mono_ffn`` / ``run_ffn``
(vLLM 8321d11) for a layer whose attention vLLM ran. One persistent launch,
256 CTAs x 512 threads, S rows:

    seam    push + reduce + slice (160 tasks from CTA 0), gate (S from CTA 160),
            norm (S from CTA 160 + S) -> the MoE input rows and their XRDY
    MoE     xq, router, route, shared, ug, down, reduce (``moe.py``)
    epoch   every CTA's mark checked by CTA 0, the MoE counter slots 128 epochs
            ahead cleared, the epoch moved on

Same launch ABI as FlyDSL's (``health``: the sticky health word of this rank,
the one addition), the same scratch / peer layout (upstream's
``layer.scratch_layout``, passed in as constexpr offsets), the same tags. The
addresses are pointer parameters, not integers: Triton knows a pointer
argument's 16-byte alignment and vectorizes the loads off it, while an address
cast from an integer loses it (byte loads, and past S = 4 the spills that come
with them). A raw address goes in through ``ptr``; inside, every pointer is
taken as bytes (``u8``) whatever the caller's dtype."""

from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from ..common.epoch import epoch_begin, epoch_end, load_epoch
from ..common.sync import SMEM_FLAT, T512, deadline, fail, poll1
from . import moe_down as md
from . import moe_front as mf
from . import moe_shared as ms
from . import moe_ug as mu
from . import seam

BLOCKS = gl.constexpr(256)


class ptr:
    """A raw device address as a Triton pointer argument (``*u8``)."""

    def __init__(self, address: int):
        import torch

        self.address = int(address)
        self.dtype = torch.uint8

    def data_ptr(self) -> int:
        return self.address


@gluon.jit
def u8(p):
    return p.to(gl.pointer_type(gl.uint8))


@gluon.constexpr_function
def cmin(a, b):
    return min(a, b)


@gluon.jit
def first_task(bid, BASE: gl.constexpr):
    """CTA ``bid``'s first task of a stage placed from CTA ``BASE``."""
    return (bid - BASE % 256 + 256) % 256


@gluon.jit
def peer_bases(peers, par, TP: gl.constexpr):
    """Every rank's peer buffer base + this epoch's parity half, loaded once."""
    pp = peers.to(gl.pointer_type(gl.int64))
    out = ()
    for p in gl.static_range(TP):
        out = out + (gl.load(pp + p) + par,)
    return out


@gluon.jit
def await_rows(xrdy, n, tag, tid, dl, health, rank):
    """XRDY of the first n tokens (thread t polls token t), then a barrier."""
    live = tid < n
    _, f = poll1(xrdy.to(gl.uint64) + gl.minimum(tid, n - 1).to(gl.uint64) * 8, tag, live, dl,
                 health)  # fmt: skip
    fail(health, f, rank, 13, live)
    gl.barrier()


@gluon.jit
def take_rank(counter, tab, tid, QSLOT: gl.constexpr, SPARE: gl.constexpr):
    """This CTA's arrival rank at a stage (a device-scope counter, one word a
    launch tag): which CTAs take the extra tasks."""
    zero = gl.zeros_like(tid)
    old = gl.atomic_add(counter.to(gl.pointer_type(gl.int32)) + zero, 1, mask=tid == 0,
                        sem="relaxed", scope="gpu")  # fmt: skip
    tab.scatter(old, gl.where(tid == 0, QSLOT, SPARE), 0)
    gl.barrier()
    return gl.max(tab.gather(zero + QSLOT, 0), axis=0)


@gluon.jit
def kffn(
    part, res_in, post_in, comb_in, pre_in, hc_fn, hc_scale, hc_base, ffn_w,
    res_out, post_out, comb_out, pre_out,
    gate_w, bias, w13, w13_s, w2, w2_s, sgu, sgu_s, sw2, sw2_s, out,
    scratch, sym, peers, rank, epoch, tl, health,
    S: gl.constexpr, TP: gl.constexpr, HIDDEN: gl.constexpr, HALF: gl.constexpr,
    FFN_OFF: gl.constexpr, LIN: gl.constexpr, PMIX: gl.constexpr, NORMED: gl.constexpr,
    XRDY: gl.constexpr, X8R: gl.constexpr, X8RS: gl.constexpr, X8: gl.constexpr,
    X8S: gl.constexpr, X8RDY: gl.constexpr, LOGIT: gl.constexpr, ROUTE: gl.constexpr,
    SMID: gl.constexpr, SMQ: gl.constexpr, SMC: gl.constexpr, MID: gl.constexpr,
    MIDS: gl.constexpr, UGF: gl.constexpr, EPOCH_COUNTERS: gl.constexpr,
    EXPERTS: gl.constexpr, TOPK: gl.constexpr, SCALE: gl.constexpr,
    ROUTER0: gl.constexpr, ROUTE0: gl.constexpr, SHARED0: gl.constexpr, UG0: gl.constexpr,
    ROUTER_TASKS: gl.constexpr, GROUPS: gl.constexpr, PER: gl.constexpr,
    MAX_U: gl.constexpr, UG_PARTS: gl.constexpr, FIRST0: gl.constexpr,
    ORDER0: gl.constexpr, EXPERT_SLOT0: gl.constexpr, MASK0: gl.constexpr,
    QUEUE0: gl.constexpr, TILE_FIRST0: gl.constexpr, TILE_SLOT0: gl.constexpr,
    MULTI: gl.constexpr, SLOT_TILES: gl.constexpr, TABW: gl.constexpr,
    SH_INTER: gl.constexpr, INTER: gl.constexpr, INTER_REAL: gl.constexpr,
    SCALE_COLS: gl.constexpr, SH_UNITS: gl.constexpr, MOE: gl.constexpr,
):  # fmt: skip
    """The FFN launch: ``ffn_half`` with this rank's partial pushed."""
    ffn_half(
        part, res_in, post_in, comb_in, pre_in, hc_fn, hc_scale, hc_base, ffn_w, res_out,
        post_out, comb_out, pre_out, gate_w, bias, w13, w13_s, w2, w2_s, sgu, sgu_s, sw2,
        sw2_s, out, scratch, sym, peers, rank, epoch, tl, health, S, TP, HIDDEN, HALF,
        FFN_OFF, LIN, PMIX, NORMED, XRDY, X8R, X8RS, X8, X8S, X8RDY, LOGIT, ROUTE, SMID,
        SMQ, SMC, MID, MIDS, UGF, EPOCH_COUNTERS, EXPERTS, TOPK, SCALE, ROUTER0, ROUTE0,
        SHARED0, UG0, ROUTER_TASKS, GROUPS, PER, MAX_U, UG_PARTS, FIRST0, ORDER0,
        EXPERT_SLOT0, MASK0, QUEUE0, TILE_FIRST0, TILE_SLOT0, MULTI, SLOT_TILES, TABW,
        SH_INTER, INTER, INTER_REAL, SCALE_COLS, SH_UNITS, MOE, True,
    )  # fmt: skip


@gluon.jit
def ffn_half(
    part, res_in, post_in, comb_in, pre_in, hc_fn, hc_scale, hc_base, ffn_w,
    res_out, post_out, comb_out, pre_out,
    gate_w, bias, w13, w13_s, w2, w2_s, sgu, sgu_s, sw2, sw2_s, out,
    scratch, sym, peers, rank, epoch, tl, health,
    S: gl.constexpr, TP: gl.constexpr, HIDDEN: gl.constexpr, HALF: gl.constexpr,
    FFN_OFF: gl.constexpr, LIN: gl.constexpr, PMIX: gl.constexpr, NORMED: gl.constexpr,
    XRDY: gl.constexpr, X8R: gl.constexpr, X8RS: gl.constexpr, X8: gl.constexpr,
    X8S: gl.constexpr, X8RDY: gl.constexpr, LOGIT: gl.constexpr, ROUTE: gl.constexpr,
    SMID: gl.constexpr, SMQ: gl.constexpr, SMC: gl.constexpr, MID: gl.constexpr,
    MIDS: gl.constexpr, UGF: gl.constexpr, EPOCH_COUNTERS: gl.constexpr,
    EXPERTS: gl.constexpr, TOPK: gl.constexpr, SCALE: gl.constexpr,
    ROUTER0: gl.constexpr, ROUTE0: gl.constexpr, SHARED0: gl.constexpr, UG0: gl.constexpr,
    ROUTER_TASKS: gl.constexpr, GROUPS: gl.constexpr, PER: gl.constexpr,
    MAX_U: gl.constexpr, UG_PARTS: gl.constexpr, FIRST0: gl.constexpr,
    ORDER0: gl.constexpr, EXPERT_SLOT0: gl.constexpr, MASK0: gl.constexpr,
    QUEUE0: gl.constexpr, TILE_FIRST0: gl.constexpr, TILE_SLOT0: gl.constexpr,
    MULTI: gl.constexpr, SLOT_TILES: gl.constexpr, TABW: gl.constexpr,
    SH_INTER: gl.constexpr, INTER: gl.constexpr, INTER_REAL: gl.constexpr,
    SCALE_COLS: gl.constexpr, SH_UNITS: gl.constexpr, MOE: gl.constexpr,
    PUSH: gl.constexpr,
):  # fmt: skip
    """The FFN launch's work on this CTA (upstream ``run_ffn``): the seam (with
    ``PUSH`` this rank's partial ``part`` pushed to every rank first; K2's
    wo_b pushed it already), the MoE, the epoch end. With ``PUSH`` it also
    marks the epoch begun (K2 marks it before its attention)."""
    part, res_in, post_in, comb_in, pre_in = u8(part), u8(res_in), u8(post_in), u8(comb_in), u8(pre_in)
    hc_fn, hc_scale, hc_base, ffn_w = u8(hc_fn), u8(hc_scale), u8(hc_base), u8(ffn_w)
    res_out, post_out, comb_out, pre_out = u8(res_out), u8(post_out), u8(comb_out), u8(pre_out)
    gate_w, bias, w13, w13_s, w2, w2_s = u8(gate_w), u8(bias), u8(w13), u8(w13_s), u8(w2), u8(w2_s)
    sgu, sgu_s, sw2, sw2_s, out = u8(sgu), u8(sgu_s), u8(sw2), u8(sw2_s), u8(out)
    scratch, sym, peers, epoch, health = u8(scratch), u8(sym), u8(peers), u8(epoch), u8(health)
    SLICES: gl.constexpr = HIDDEN // 32
    RLW: gl.constexpr = seam.pow2(S * 128 + 1)
    PLW: gl.constexpr = seam.pow2(S * 32 + 1)
    tid = gl.arange(0, 512, layout=T512)
    lane = tid % 64
    wave = tid // 64
    bid = gl.program_id(0)
    ep = load_epoch(epoch)
    par = (ep & 1).to(gl.int64) * HALF
    peer = peer_bases(peers, par, TP)
    own = sym + par
    if PUSH:
        epoch_begin(epoch, ep, bid, tid)
    tag = (ep << 1) + 2 + gl.zeros_like(tid)
    dl = deadline(tid)
    red = gl.allocate_shared_memory(gl.float32, [4096], SMEM_FLAT)
    rl = gl.allocate_shared_memory(gl.float32, [RLW], SMEM_FLAT)
    fl = gl.allocate_shared_memory(gl.float32, [4096], SMEM_FLAT)
    pl = gl.allocate_shared_memory(gl.float32, [PLW], SMEM_FLAT)
    sb = scratch
    lin = sb + LIN
    pmix = sb + PMIX
    normed = sb + NORMED
    xrdy = sb + XRDY

    # ---- the FFN seam
    for task in range(first_task(bid, 0), SLICES, 256):
        if PUSH:
            seam.stage_push(part, task, tag, peer, rank, tid, S, TP, HIDDEN)
        seam.stage_reduce(task, tag, own, pl, tid, dl, health, rank, S, TP, HIDDEN,
                          PLW - 1)  # fmt: skip
        seam.stage_slice(res_in, comb_in, post_in, pre_in, hc_fn, res_out, task, tag,
                         lin, pmix, pl, rl, fl, red, tid, lane, wave, S, HIDDEN,
                         RLW - 1)  # fmt: skip
    for t in range(first_task(bid, SLICES), S, 256):
        seam.stage_gate(t, tag, pmix, red, hc_scale, hc_base, pre_out, post_out,
                        comb_out, tid, lane, wave, dl, health, rank, S, HIDDEN)  # fmt: skip
    for t in range(first_task(bid, SLICES + S), S, 256):
        seam.stage_norm(t, tag, lin, ffn_w, normed, xrdy, red, tid, lane, wave, dl,
                        health, rank, HIDDEN)  # fmt: skip
    gl.barrier()

    if MOE:
        # ---- the MoE: LDS of its stages (after the seam's: no live overlap)
        xa = gl.allocate_shared_memory(gl.uint8, [4, 16, 8, 128], ms.SH4)
        xb = gl.allocate_shared_memory(gl.uint8, [16, 8, 128], ms.SH3)
        ca = gl.allocate_shared_memory(gl.uint64, [512], ms.SH1)
        cb = gl.allocate_shared_memory(gl.uint64, [128], ms.SH1)
        tab = gl.allocate_shared_memory(gl.int32, [TABW], SMEM_FLAT)
        pair = gl.allocate_shared_memory(gl.float32, [512], SMEM_FLAT)
        # xq
        for t in range(first_task(bid, 0), S, 256):
            await_rows(xrdy + t * 8, 1, tag, tid, dl, health, rank)
            mf.stage_xq(t, normed, sb + X8R, sb + X8RS, sb + X8, sb + X8S, sb + X8RDY, tag, tid,
                        HIDDEN)  # fmt: skip
        # router
        for task in range(first_task(bid, ROUTER0), ROUTER_TASKS, 256):
            await_rows(xrdy, S, tag, tid, dl, health, rank)
            mf.stage_router(task, gate_w, normed, sb + LOGIT, red, tag, tid, S, EXPERTS, GROUPS,
                            PER, HIDDEN)  # fmt: skip
        # route
        for t in range(first_task(bid, ROUTE0), S, 256):
            mf.stage_route(t, sb + LOGIT, sb + ROUTE, bias, tag, tid, lane, wave, dl, health,
                           rank, S, EXPERTS, TOPK, SCALE)  # fmt: skip
        # shared: a CTA's one unit (task, token tile) at most
        SH_TASKS: gl.constexpr = SH_INTER // 16
        unit = first_task(bid, SHARED0)
        if unit < SH_UNITS:
            st0 = unit // SH_TASKS * 16
            sn = gl.minimum(S - st0, 16)
            ms.load_x_tile(sb + X8, sb + X8S, sb + X8RDY, st0, sn, xa, xb, ca, cb, tab, tag, tid,
                           dl, health, rank, HIDDEN, False, ORDER0, TOPK)  # fmt: skip
            ms.stage_shared(unit % SH_TASKS, st0, sn, sgu, sgu_s, sb + SMID, sb + SMQ, sb + SMC,
                            xa, xb, ca, cb, red, pair, tag, tid, dl, health, rank, SH_INTER,
                            HIDDEN)  # fmt: skip
        # ug
        if MULTI:
            # past one token tile a task is (a slot's pick tile, part) and
            # gathers its picks' rows itself; the first round placed, the rest
            # from the queue (a CTA a shared / router task delayed takes fewer)
            mf.load_route(tab, red, sb + ROUTE, bid, tag, tid, lane, wave, dl, health, rank, S,
                          TOPK, EXPERTS, MAX_U, MULTI, SLOT_TILES, EXPERT_SLOT0, ORDER0, FIRST0,
                          MASK0, TILE_FIRST0, TILE_SLOT0, TABW - 1)  # fmt: skip
            total = mu.at(tab, TILE_FIRST0 + mu.at(tab, 2 * S * TOPK, tid), tid) * UG_PARTS
            ugq = epoch + (EPOCH_COUNTERS + (ep & 255)) * 4
            task = first_task(bid, UG0)
            while task < total:
                e, first, count = mu.tile_unit(tab, task // UG_PARTS, tid, S, TOPK, FIRST0,
                                               TILE_FIRST0, TILE_SLOT0)  # fmt: skip
                ms.load_x_tile(sb + X8R, sb + X8RS, sb + X8RDY, first, gl.minimum(count, 16), xa,
                               xb, ca, cb, tab, tag, tid, dl, health, rank, HIDDEN, True, ORDER0,
                               TOPK)  # fmt: skip
                mu.stage_ug(task, e, first, count, tab, w13, w13_s, sb + MID, sb + MIDS, sb + UGF,
                            xa, xb, ca, cb, red, tag, tid, lane, TOPK, UG_PARTS, ORDER0,
                            2 * INTER, INTER, HIDDEN, True)  # fmt: skip
                if total > 256:
                    task = take_rank(ugq, tab, tid, QUEUE0, TABW - 1) + 256
                else:
                    task = total
        else:
            # one token tile: every task placed (X8R and the route table into
            # LDS once a CTA)
            loaded = 0
            for task in range(first_task(bid, UG0), MAX_U * UG_PARTS, 256):
                if loaded == 0:
                    ms.load_x_tile(sb + X8R, sb + X8RS, sb + X8RDY, 0, S, xa, xb, ca, cb, tab, tag,
                                   tid, dl, health, rank, HIDDEN, False, ORDER0, TOPK)  # fmt: skip
                    mf.load_route(tab, red, sb + ROUTE, bid, tag, tid, lane, wave, dl, health,
                                  rank, S, TOPK, EXPERTS, MAX_U, MULTI, SLOT_TILES, EXPERT_SLOT0,
                                  ORDER0, FIRST0, MASK0, TILE_FIRST0, TILE_SLOT0,
                                  TABW - 1)  # fmt: skip
                loaded = 1
                u = task // UG_PARTS
                if u < mu.at(tab, 2 * S * TOPK, tid):
                    e, first, count = mu.slot_unit(tab, u, tid, S, TOPK, FIRST0)
                    mu.stage_ug(task, e, first, count, tab, w13, w13_s, sb + MID, sb + MIDS,
                                sb + UGF, xa, xb, ca, cb, red, tag, tid, lane, TOPK, UG_PARTS,
                                ORDER0, 2 * INTER, INTER, HIDDEN, False)  # fmt: skip
            if loaded == 0:
                mf.load_route(tab, red, sb + ROUTE, bid, tag, tid, lane, wave, dl, health, rank,
                              S, TOPK, EXPERTS, MAX_U, MULTI, SLOT_TILES, EXPERT_SLOT0, ORDER0,
                              FIRST0, MASK0, TILE_FIRST0, TILE_SLOT0, TABW - 1)  # fmt: skip
        # down: every CTA its task; the first 64 to arrive an extra one
        dq = epoch + (EPOCH_COUNTERS + 256 + (ep & 255)) * 4
        rk = take_rank(dq, tab, tid, QUEUE0, TABW - 1)
        PICKS: gl.constexpr = S * TOPK
        CW: gl.constexpr = seam.pow2(2 * PICKS * 16 + 1)
        SDW: gl.constexpr = seam.pow2(2 * S * 16 + 1)
        contrib = gl.allocate_shared_memory(gl.float32, [CW], SMEM_FLAT)
        sdl = gl.allocate_shared_memory(gl.float32, [SDW], SMEM_FLAT)
        SHR: gl.constexpr = md.sh_rounds(SH_INTER)
        sx32 = gl.allocate_shared_memory(gl.uint32, [4096 * SHR], ms.SH1)
        sx8 = sx32._reinterpret(gl.uint8, [8 * SHR, 16, 128], ms.SH3)
        sc8 = gl.allocate_shared_memory(gl.uint8, [8 * SHR, 16, 4], ms.SH3)
        r = tid % 16
        tl_ = gl.minimum(tid // 16, 15)
        two = rk < 64
        # the shared expert's down first, a token tile at a time: its input is
        # ready long before the routed results
        for t0 in gl.static_range(0, S, 16):
            sl = tid < 16 * cmin(16, S - t0)
            md.load_smid_tile(sb + SMQ, sb + SMC, t0, cmin(16, S - t0), sx32, sx8, sc8, tag, tid, dl, health,
                              rank, SH_INTER)  # fmt: skip
            v0 = ms.wave_sum(md.down_shared_task(bid, sw2, sw2_s, sx8, sc8, SH_INTER, HIDDEN),
                             red, r, tl_)  # fmt: skip
            sdl.scatter(bf16r(v0), gl.where(sl, (t0 + tl_) * 16 + r, SDW - 1), 0)
            if two:
                v1 = ms.wave_sum(md.down_shared_task(256 + rk, sw2, sw2_s, sx8, sc8, SH_INTER,
                                                     HIDDEN), red, r, tl_)  # fmt: skip
                sdl.scatter(bf16r(v1), gl.where(sl, (S + t0 + tl_) * 16 + r, SDW - 1), 0)
        if two:
            md.stage_down(bid, 256 + rk, tab, w2, w2_s, sb + MID, sb + MIDS, sb + UGF, contrib,
                          tag, tid, dl, health, rank, 2, S, TOPK, FIRST0, ORDER0, UG_PARTS, INTER,
                          INTER_REAL, HIDDEN, SCALE_COLS, MULTI, SLOT_TILES,
                          TILE_FIRST0)  # fmt: skip
            for t0 in gl.static_range(0, S, 16):
                md.down_combine(bid, 0, t0, cmin(16, S - t0), tab, contrib, sdl, pair, peer, rank,
                                tag, tid, S, TOPK, TP, HIDDEN, FFN_OFF)  # fmt: skip
                md.down_combine(256 + rk, 1, t0, cmin(16, S - t0), tab, contrib, sdl, pair, peer,
                                rank, tag, tid, S, TOPK, TP, HIDDEN, FFN_OFF)  # fmt: skip
        else:
            md.stage_down(bid, bid, tab, w2, w2_s, sb + MID, sb + MIDS, sb + UGF, contrib, tag, tid,
                          dl, health, rank, 1, S, TOPK, FIRST0, ORDER0, UG_PARTS, INTER, INTER_REAL,
                          HIDDEN, SCALE_COLS, MULTI, SLOT_TILES, TILE_FIRST0)  # fmt: skip
            for t0 in gl.static_range(0, S, 16):
                md.down_combine(bid, 0, t0, cmin(16, S - t0), tab, contrib, sdl, pair, peer, rank,
                                tag, tid, S, TOPK, TP, HIDDEN, FFN_OFF)  # fmt: skip
        # reduce
        for task in range(first_task(bid, 0), HIDDEN // 32, 256):
            md.stage_reduce(task, own, out, tag, tid, dl, health, rank, S, TP, HIDDEN, FFN_OFF)

    epoch_end(epoch, ep, bid, tid, health, rank, EPOCH_COUNTERS, 2)


@gluon.jit
def bf16r(v):
    return v.to(gl.bfloat16).to(gl.float32)


def build_constexprs(s: int, tp: int, moe: bool = True) -> dict:
    """The FFN launch's constexprs for width s, TP tp: upstream's own scratch
    layout (``layer.scratch_layout``, ``moe.ffn_offset``, the epoch buffer) and
    route shape (checked against ``plan.Route``). GPU image only (upstream
    imports aiter)."""
    from vllm.models.deepseek_v41.amd.mono import layer as L
    from vllm.models.deepseek_v41.amd.mono.stages import moe as M

    from .plan import Route, check_against_upstream

    check_against_upstream(s, tp)
    lay = L.scratch_layout(s, tp)
    r = Route(s, tp)
    g, per = r.router_groups
    off = {k: v[0] for k, v in lay.items()}
    return dict(
        S=s, TP=tp, HIDDEN=5120, HALF=L.peer_half_bytes(tp), FFN_OFF=M.ffn_offset(s, tp),
        LIN=off["lin"], PMIX=off["pmix"], NORMED=off["normed"], XRDY=off["xrdy_moe"],
        X8R=off["x8r"], X8RS=off["x8rs"], X8=off["x8"], X8S=off["x8s"], X8RDY=off["x8rdy"],
        LOGIT=off["logit"], ROUTE=off["route"], SMID=off["smid"], SMQ=off["smq"],
        SMC=off["smc"], MID=off["mid"], MIDS=off["mids"], UGF=off["ugf"],
        EPOCH_COUNTERS=L.EPOCH_COUNTERS, EXPERTS=r.experts, TOPK=r.topk, SCALE=M.SCALE,
        ROUTER0=r.router0, ROUTE0=r.route0, SHARED0=r.shared0, UG0=r.ug0,
        ROUTER_TASKS=r.router_tasks, GROUPS=g, PER=per, MAX_U=r.max_u, UG_PARTS=r.ug_parts,
        FIRST0=r.first0, ORDER0=r.order0, EXPERT_SLOT0=r.expert_slot0, MASK0=r.mask0,
        QUEUE0=r.queue0, TILE_FIRST0=r.tile_first0, TILE_SLOT0=r.tile_slot0, MULTI=r.multi,
        SLOT_TILES=r.slot_tiles, TABW=1 << r.tab_words.bit_length(), SH_INTER=r.sh_inter,
        INTER=r.inter, INTER_REAL=r.inter_real, SCALE_COLS=r.down_scale_cols,
        SH_UNITS=r.shared_units, MOE=moe,
    )  # fmt: skip
