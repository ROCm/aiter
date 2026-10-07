# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The mono decode layer's two launches (README): K1 = the attention seam
(ATOM's slice / gate, the norm into the front's MXFP8) + dsv41_mega_attn's
front; K2 = dsv41_mega_attn's back (wo_b pushed to every rank) + ATOM's FFN
seam (the TP sum folded in) + ATOM's MoE (vLLM numerics, ``atomv41.kernels``).

Tags: K1's hand-offs carry 2 e + 1, K2's 2 e + 2 (e: the launch pair's epoch,
moved on by K2's CTA 0 at its end), so K2's seam may reuse K1's seam regions.
Peer regions alternate by e's parity; every rank runs the same launches, so
the epochs agree across ranks.
"""

from dataclasses import dataclass

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import Int32, Int64, T

from aiter.ops.flydsl.kernels import buffer_ops as bo

from ..dsv41_mega_attn import back as mb_back
from ..dsv41_mega_attn import front as mb_front
from ..dsv41_mega_attn.device import BLOCKS, CM_DEV, THREADS, gstore, kernel_symbol, rsrc
from ..dsv41_mega_attn.plan import HIDDEN, Dims, back_scratch, front_scratch
from .atomfw.device.ops import bf16_round, bf_hi, bf_lo, butterfly, fp8_pack4, hw_rsq, traced
from .atomfw.device.ranks import peer_bases
from .atomfw.device.sync import preg, publish, sreg
from .atomfw.plan.build_key import key_tuple
from .atomfw.plan.execution import WAVES, first_task
from .atomv41.kernels import attn_post as ak2a
from .atomv41.kernels import attn_pre as ak1
from .atomv41.kernels import moe as ak2b
from .atomv41.kernels.debug import mailbox as atom_mailbox
from .atomv41.kernels.dims import Dims as ADims
from .atomv41.kernels.moe_shape import SORT_NETS, MoeBuild, route_shape
from .atomv41.sources import SOURCES

_STREAM = fx.Stream(None)
MAX_TOKENS = 48
X_WORDS = HIDDEN // 4
X_GROUPS = HIDDEN // 32
SLICES = ak1.SLICES  # 160
COUNTER_WORDS = 2 * 256  # the MoE's ug queue and down counts, a word a slot
COUNTER_BYTES = COUNTER_WORDS * 4


@dataclass(frozen=True)
class MonoBuild:
    tokens: int
    tp: int
    ratio: int  # the layer's compress ratio (its attention's key sources)
    timeline: bool = False


def _moe_key(key: MonoBuild) -> MoeBuild:
    return MoeBuild(tokens=key.tokens, tp=key.tp)


def _front_key(key: MonoBuild):
    return mb_front.FrontBuild(key.tokens, key.tp, key.ratio, key.timeline)


def _back_key(key: MonoBuild):
    return mb_back.BackBuild(key.tokens, key.tp, key.ratio, key.timeline)


def scratch_layout(s: int, tp: int) -> dict:
    """Every region of both launches -> (byte offset, bytes), disjoint: the
    MoE's counters first (at a fixed offset: their slots outlive a step width),
    then K1's front and seam, K2's back, the MoE's regions, the normed rows and
    their flags."""
    d = Dims(tp)
    out = {"ugq": (0, COUNTER_BYTES // 2), "dq": (COUNTER_BYTES // 2, COUNTER_BYTES // 2)}
    off = COUNTER_BYTES

    def place(regions):
        nonlocal off
        base = off
        for name, (o, n) in regions.items():
            out[name] = (base + o, n)
        off = max(off, base + max(o + n for o, n in regions.values()))
        off = -(-off // 256) * 256

    place(front_scratch(s, d))
    seam = ak1.scratch_layout(s)
    lo = min(seam[n][0] for n in ("lin", "pmix"))
    place({n: (seam[n][0] - lo, seam[n][1]) for n in ("lin", "pmix")})
    place(back_scratch(s, d, start=0))
    moe = {n: v for n, v in ak2b.scratch_layout(_moe_key(MonoBuild(s, tp, 1))).items() if n not in ("ugq", "dq")}
    lo = min(o for o, _ in moe.values())
    place({n: (o - lo, n_) for n, (o, n_) in moe.items()})
    place({"normed": (0, s * HIDDEN * 2), "xrdy_moe": (s * HIDDEN * 2 + 256, s * 8)})
    return out


def scratch_bytes(s: int = MAX_TOKENS, tp: int = 2) -> int:
    return max(o + n for o, n in scratch_layout(s, tp).values())


def peer_half_bytes(tp: int) -> int:
    """One parity's peer regions: the attention's partials, then the MoE's."""
    return ak2a.peer_bytes(MAX_TOKENS, tp) + ak2b.peer_bytes(MAX_TOKENS, tp)


def _epoch(epoch):
    return fx.Int32(bo.buffer_load(rsrc(epoch), 0, vec_width=1, dtype=T.i32))


def _mark_region(layout, scratch):
    return {name: scratch + fx.Int64(off) for name, (off, _) in layout.items()}


# ---------------------------------------------------------------- K1


@traced
def stage_norm_x8(ca, c, t):
    """The attention seam's RMSNorm of token t (ATOM ``stage_norm``'s: aiter's
    fused seam's rounding) -> bf16 -> vLLM's MXFP8 of each 32 (the attention's
    input quant) -> the front's X8 / X8S, plain at device scope, then its two
    XRDY flags (one a wqkv K half). 256 threads, 3 chunks of 8 at 8 t + 2048 c."""
    tid, lane, wave, red = ca["tid"], ca["lane"], ca["wave"], ca["red"]
    a = ca["args"]
    tt = fx.min(tid, 255)
    mine = tid < 256
    cols = [tt * 8 + 2048 * ch for ch in range(3)]
    xs = []
    for ch in range_constexpr(3):
        base = t * HIDDEN + fx.min(cols[ch], HIDDEN - 8)
        words = ca["poll"]([(ca["lin"], base // 2 + k, 1) for k in range(4)])
        v = []
        for k in range_constexpr(4):
            v += [bf_lo(words[k][0]), bf_hi(words[k][0])]
        xs.append([(cols[ch] < HIDDEN).select(x, fx.Float32(0.0)) for x in v])
    acc = fx.Float32(0.0)
    for ch in range_constexpr(3):
        for x in xs[ch]:
            acc = acc + x * x
    acc = butterfly(acc, (1, 2, 4, 8, 16, 32))
    if (lane == 0) & mine:
        fx.ptr_store(acc, red + wave)
    gpu.barrier()
    tot = butterfly(fx.ptr_load(red + lane % 4), (1, 2))
    r = hw_rsq(ak1.fma(tot, fx.Float32(1.0 / HIDDEN), fx.Float32(ak1.EPS)))
    for ch in range_constexpr(3):
        col = fx.min(cols[ch], HIDDEN - 8)
        ys = [bf16_round((xs[ch][j] * r) * ak1.ld_bf(a["attn_w"], col + j)) for j in range(8)]
        amax = abs(ys[0])
        for y in ys[1:]:
            amax = fx.max(amax, abs(y))
        code = ak2b.vllm_mx_code(butterfly(amax, (1, 2), fx.max))
        mul = ((254 - code) << 23).bitcast(fx.Float32)
        w0 = fp8_pack4(*[ak2b.clamp_fp8(ys[i] * mul) for i in range(4)])
        w1 = fp8_pack4(*[ak2b.clamp_fp8(ys[4 + i] * mul) for i in range(4)])
        if mine & (cols[ch] < HIDDEN):
            bo.buffer_store(
                fx.Vector.from_elements([w0, w1], fx.Int32), rsrc(c["x8"]),
                t * X_WORDS + col // 4, cache_modifier=CM_DEV,
            )  # fmt: skip
            if tt % 4 == 0:
                bo.buffer_store(code, rsrc(c["x8s"]), t * X_GROUPS + col // 32, cache_modifier=CM_DEV)
    rocdl.s_waitcnt(vmcnt=0)
    gpu.barrier()
    if tid < 2:
        c["mb"].put(c["xrdy"], 2 * t + tid, fx.Int32(1))
    gpu.barrier()


def build_mono_k1(key: MonoBuild):
    s, tp = key.tokens, key.tp
    assert 1 <= s <= MAX_TOKENS
    layout = scratch_layout(s, tp)
    fkey = _front_key(key)
    FrontLds = mb_front.front_lds(s, Dims(tp))
    NORM0 = SLICES  # the norms on the CTAs past the slices: the wqkv waits on them
    GATE0 = SLICES + s
    assert GATE0 + s <= BLOCKS

    @fx.struct
    class SeamLds:
        red: fx.Array[fx.Float32, WAVES * 2 * 64 * 4, 16]
        rl: fx.Array[fx.Float32, s * ak1.KT, 16]
        fl: fx.Array[fx.Float32, ak1.MIX * ak1.KT, 16]

    @fx.union
    class K1Lds:
        seam: SeamLds
        front: FrontLds

    name = kernel_symbol("dsv41_mono_k1", s=s, tp=tp, r=key.ratio, tl=key.timeline)
    keyed = key_tuple(key, SOURCES)

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def k1(
        res_in: Int64, pend: Int64, post_in: Int64, comb_in: Int64, pre_in: Int64,
        hc_fn: Int64, hc_scale: Int64, hc_base: Int64, attn_w: Int64,
        res_out: Int64, post_out: Int64, comb_out: Int64, pre_out: Int64,
        wqkv: Int64, wqkv_s: Int64, qn_w: Int64, kvn_w: Int64, wqb: Int64, wqb_s: Int64,
        cos_sin: Int64, pos: Int64, slot: Int64, swa: Int64, swa_stride: Int64,
        swa_block: Int32, q: Int64, swa_idx: Int64, swa_lens: Int64, comp_block: Int32,
        comp_bt: Int64, bt_stride: Int32, topk: Int64, t2r: Int64, kt: Int64, klen: Int64,
        scratch: Int64, epoch: Int64, tl: Int64,
    ):  # fmt: skip
        _ = keyed
        bid = fx.block_idx.x
        lds = fx.SharedAllocator().allocate(K1Lds)
        seam = lds.seam.peek()
        ep = _epoch(epoch)
        tag = (ep << 1) + 1
        fargs = {
            "x": 0, "x_stride": 0, "wqkv": wqkv, "wqkv_s": wqkv_s, "qn_w": qn_w,
            "kvn_w": kvn_w, "wqb": wqb, "wqb_s": wqb_s, "cos_sin": cos_sin, "pos": pos,
            "slot": slot, "swa": swa, "swa_stride": swa_stride, "swa_block": swa_block,
            "q": q, "swa_idx": swa_idx, "swa_lens": swa_lens, "comp_block": comp_block,
            "comp_bt": comp_bt, "bt_stride": bt_stride, "topk": topk, "t2r": t2r,
            "kt": kt, "klen": klen,
        }  # fmt: skip
        flayout = {n: layout[n] for n in front_scratch(s, Dims(tp))}
        c = mb_front.front_context(fkey, lds.front.peek(), fargs, scratch, tag, flayout)
        amb = atom_mailbox(tag - 1, scratch, -1, {})
        tid = fx.thread_idx.x
        ca = {
            "S": s, "tid": tid, "bid": bid, "lane": tid % 64, "wave": tid // 64,
            "rl": seam.rl.ptr, "fl": seam.fl.ptr, "red": seam.red.ptr,
            "put": amb.put, "put_bf": amb.put_bf, "put_words": amb.put_words, "poll": amb.poll,
            "args": {
                "res_in": res_in, "pend": pend, "post_in": post_in, "comb_in": comb_in,
                "pre_in": pre_in, "hc_fn": hc_fn, "hc_scale": hc_scale, "hc_base": hc_base,
                "attn_w": attn_w, "res_out": res_out, "post_out": post_out,
                "comb_out": comb_out, "pre_out": pre_out, "aux": 0,
            },
            "fold": True, "index": False, "aux": False, "ffn": False, "d": ADims(tp),
        }  # fmt: skip
        for region in ("lin", "pmix"):
            ca[region] = sreg(scratch, layout[region][0], region)
        if const_expr(key.timeline):
            c["tl"], c["tl_points"] = tl, mb_front.FRONT_POINTS

        def seam_slice():
            for task in range(first_task(bid, 0), SLICES, BLOCKS):
                ak1.stage_slice(ca, task)
            gpu.barrier()

        def seam_rest():
            for t in range(first_task(bid, NORM0), s, BLOCKS):
                stage_norm_x8(ca, c, t)
            for t in range(first_task(bid, GATE0), s, BLOCKS):
                ak1.stage_gate(ca, t)
            gpu.barrier()

        mb_front.run_front(c, fkey, bid, x8_given=True, before_wqkv=seam_slice, after_wqkv=seam_rest)

    @flyc.jit
    def launch(
        res_in: Int64, pend: Int64, post_in: Int64, comb_in: Int64, pre_in: Int64,
        hc_fn: Int64, hc_scale: Int64, hc_base: Int64, attn_w: Int64,
        res_out: Int64, post_out: Int64, comb_out: Int64, pre_out: Int64,
        wqkv: Int64, wqkv_s: Int64, qn_w: Int64, kvn_w: Int64, wqb: Int64, wqb_s: Int64,
        cos_sin: Int64, pos: Int64, slot: Int64, swa: Int64, swa_stride: Int64,
        swa_block: Int32, q: Int64, swa_idx: Int64, swa_lens: Int64, comp_block: Int32,
        comp_bt: Int64, bt_stride: Int32, topk: Int64, t2r: Int64, kt: Int64, klen: Int64,
        scratch: Int64, epoch: Int64, tl: Int64, stream: fx.Stream = _STREAM,
    ):  # fmt: skip
        _ = keyed
        k1(
            res_in, pend, post_in, comb_in, pre_in, hc_fn, hc_scale, hc_base, attn_w,
            res_out, post_out, comb_out, pre_out, wqkv, wqkv_s, qn_w, kvn_w, wqb, wqb_s,
            cos_sin, pos, slot, swa, swa_stride, swa_block, q, swa_idx, swa_lens, comp_block,
            comp_bt, bt_stride, topk, t2r, kt, klen, scratch, epoch, tl,
        ).launch(grid=(BLOCKS,), block=(THREADS,), stream=stream)  # fmt: skip

    return launch


# ---------------------------------------------------------------- K2


def build_mono_k2(key: MonoBuild):
    s, tp = key.tokens, key.tp
    assert 1 <= s <= MAX_TOKENS
    layout = scratch_layout(s, tp)
    bkey = _back_key(key)
    mkey = _moe_key(key)
    rs = route_shape(mkey)
    assert rs.experts // 64 in SORT_NETS
    BackM = mb_back.back_lds_members(s, Dims(tp))
    MoeLds = ak2b.moe_smem(s, rs, False)
    half = peer_half_bytes(tp)
    GATE0 = SLICES
    NORM0 = SLICES + s
    assert NORM0 + s <= BLOCKS

    @fx.struct
    class SeamLds:
        red: fx.Array[fx.Float32, WAVES * 2 * 64 * 4, 16]
        rl: fx.Array[fx.Float32, s * ak1.KT, 16]
        fl: fx.Array[fx.Float32, ak1.MIX * ak1.KT, 16]
        pl: fx.Array[fx.Float32, s * ak1.COLS, 16]

    @fx.union
    class K2Lds:
        split: BackM["split"]
        gemv: BackM["gemv"]
        seam: SeamLds
        moe: MoeLds

    name = kernel_symbol("dsv41_mono_k2", s=s, tp=tp, r=key.ratio, tl=key.timeline)
    keyed = key_tuple(key, SOURCES)

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def k2(
        q: Int64, swa: Int64, swa_stride: Int64, swa_block: Int32,
        comp: Int64, comp_stride: Int64, comp_block: Int32, kt: Int64, klen: Int64,
        pos: Int64, sink: Int64, qk_scale: Int32, cos_sin: Int64, woa: Int64, woa_s: Int64,
        wob: Int64, wob_s: Int64, zrec: Int64,
        res_in: Int64, post_in: Int64, comb_in: Int64, pre_in: Int64, hc_fn: Int64,
        hc_scale: Int64, hc_base: Int64, ffn_w: Int64,
        res_out: Int64, post_out: Int64, comb_out: Int64, pre_out: Int64,
        gate_w: Int64, bias: Int64, w13: Int64, w13_s: Int64, w2: Int64, w2_s: Int64,
        sgu: Int64, sgu_s: Int64, sw2: Int64, sw2_s: Int64, out: Int64,
        scratch: Int64, sym: Int64, peers: Int64, rank: Int32, epoch: Int64, tl: Int64,
    ):  # fmt: skip
        _ = keyed
        bid = fx.block_idx.x
        tid = fx.thread_idx.x
        lds = fx.SharedAllocator().allocate(K2Lds)
        seam = lds.seam.peek()
        ep = _epoch(epoch)
        tag = (ep << 1) + 2
        par = fx.Int64(ep & 1) * fx.Int64(half)
        bases = [b + par for b in peer_bases(peers, tp)]
        own = sym + par
        bargs = {
            "q": q, "swa": swa, "swa_stride": swa_stride, "swa_block": swa_block,
            "comp": comp, "comp_stride": comp_stride, "comp_block": comp_block,
            "kt": kt, "klen": klen, "pos": pos, "sink": sink, "qk_scale": qk_scale,
            "cos_sin": cos_sin, "woa": woa, "woa_s": woa_s, "wob": wob, "wob_s": wob_s,
            "out": 0, "out_stride": 0, "zrec": zrec,
        }  # fmt: skip
        blayout = {n: layout[n] for n in back_scratch(s, Dims(tp), start=0)}
        views = {"split": lds.split.peek(), "gemv": lds.gemv.peek()}
        c = mb_back.back_context(bkey, views, bargs, scratch, tag, blayout)
        amb = atom_mailbox(tag - 1, scratch, -1, {})

        def push(t, col, v0, v1):
            # this rank's bf16 partial pair to every rank's ATTN region
            for p in range_constexpr(tp):
                dst = preg(bases[p], 0, "attn")
                amb.put_bf(dst, 2 * ak2a.attn_region_pair(rank, t, s, col), [v0, v1])

        c["wob_out"] = push
        if const_expr(key.timeline):
            c["tl"], c["tl_points"] = tl, mb_back.BACK_POINTS
        mb_back.epoch_begin(c, epoch, ep)
        mb_back.run_back(c, bkey, bid)
        gpu.barrier()

        # ---- the FFN seam: the attention's TP sum folded into the residual
        ca = {
            "S": s, "tid": tid, "bid": bid, "lane": tid % 64, "wave": tid // 64,
            "rl": seam.rl.ptr, "fl": seam.fl.ptr, "red": seam.red.ptr, "pend_lds": seam.pl.ptr,
            "put": amb.put, "put_bf": amb.put_bf, "put_words": amb.put_words, "poll": amb.poll,
            "peer_addr": lambda p: bases[p], "rank": rank, "sym": own,
            "args": {
                "res_in": res_in, "post_in": post_in, "comb_in": comb_in, "pre_in": pre_in,
                "hc_fn": hc_fn, "hc_scale": hc_scale, "hc_base": hc_base, "attn_w": ffn_w,
                "res_out": res_out, "post_out": post_out, "comb_out": comb_out,
                "pre_out": pre_out, "normed": scratch + fx.Int64(layout["normed"][0]),
                "aux": 0,
            },
            "fold": True, "index": False, "aux": False, "ffn": True, "d": ADims(tp),
            "normed_cm": CM_DEV,
        }  # fmt: skip
        for region in ("lin", "pmix"):
            ca[region] = sreg(scratch, layout[region][0], region)
        mlayout = {n: layout[n] for n in ak2b.scratch_layout(mkey)}
        mlayout["xrdy"] = layout["xrdy_moe"]
        margs = ak2b.moe_args(
            scratch + fx.Int64(layout["normed"][0]), gate_w, bias, w13, w13_s, w2, w2_s,
            sgu, sgu_s, sw2, sw2_s, out,
        )  # fmt: skip
        cb = ak2b.moe_context(s, rs, lds.moe.peek(), amb, bases, mlayout, scratch, margs, rank, own)
        cb["x_cm"], cb["x_ready"] = CM_DEV, True
        cb["tag"] = ep & 255
        for task in range(first_task(bid, 0), SLICES, BLOCKS):
            ak2a.stage_reduce(ca, task)
            ak1.stage_slice(ca, task)
        for t in range(first_task(bid, GATE0), s, BLOCKS):
            ak1.stage_gate(ca, t)
        for t in range(first_task(bid, NORM0), s, BLOCKS):
            ak1.stage_norm(ca, t)
            publish(cb["put"], cb["xrdy"], t, 1, tid == 0)
        gpu.barrier()

        # ---- the MoE, its all-reduce into ``out``
        ak2b.run_moe(cb, mkey, bid, 0, 0)

        def reset():
            # the MoE counter slot 128 launch pairs ahead: its last use long
            # done, its next use far off
            slot = (ep + 128) & 255
            gstore(scratch + fx.Int64(layout["ugq"][0]) + fx.Int64(slot * 4), fx.Int32(0), words=1)
            gstore(scratch + fx.Int64(layout["dq"][0]) + fx.Int64(slot * 4), fx.Int32(0), words=1)

        mb_back.epoch_end(c, epoch, ep, reset)

    @flyc.jit
    def launch(
        q: Int64, swa: Int64, swa_stride: Int64, swa_block: Int32,
        comp: Int64, comp_stride: Int64, comp_block: Int32, kt: Int64, klen: Int64,
        pos: Int64, sink: Int64, qk_scale: Int32, cos_sin: Int64, woa: Int64, woa_s: Int64,
        wob: Int64, wob_s: Int64, zrec: Int64,
        res_in: Int64, post_in: Int64, comb_in: Int64, pre_in: Int64, hc_fn: Int64,
        hc_scale: Int64, hc_base: Int64, ffn_w: Int64,
        res_out: Int64, post_out: Int64, comb_out: Int64, pre_out: Int64,
        gate_w: Int64, bias: Int64, w13: Int64, w13_s: Int64, w2: Int64, w2_s: Int64,
        sgu: Int64, sgu_s: Int64, sw2: Int64, sw2_s: Int64, out: Int64,
        scratch: Int64, sym: Int64, peers: Int64, rank: Int32, epoch: Int64, tl: Int64,
        stream: fx.Stream = _STREAM,
    ):  # fmt: skip
        _ = keyed
        k2(
            q, swa, swa_stride, swa_block, comp, comp_stride, comp_block, kt, klen,
            pos, sink, qk_scale, cos_sin, woa, woa_s, wob, wob_s, zrec,
            res_in, post_in, comb_in, pre_in, hc_fn, hc_scale, hc_base, ffn_w,
            res_out, post_out, comb_out, pre_out,
            gate_w, bias, w13, w13_s, w2, w2_s, sgu, sgu_s, sw2, sw2_s, out,
            scratch, sym, peers, rank, epoch, tl,
        ).launch(grid=(BLOCKS,), block=(THREADS,), stream=stream)  # fmt: skip

    return launch
