# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""K2 ``layer_post``: K2a (``attn_post``) and K2b (``moe``) of one layer in one
launch -- the attention output to the MoE's reduced output.

K2b's stages follow K2a's on every CTA. A K2a launch ended at its norm and K2b
started over from x (``ffn_normed``); here the norm's CTAs publish each token's
row instead (device-scope stores, the load counter drained, then the token's
flag in XRDY), and xq and the router wait for it and read it at device scope:
the buffer is every layer's, so a plain load could hit this XCD's L2 copy of
the previous layer's.

Each half keeps its own mailbox tag and its scratch regions, K2b's past K2a's
(K2a's gate may still poll its partials while another CTA is past the norm).
Their LDS is one union: a CTA runs K2a's stages, then K2b's, with a barrier
between every two.
"""

from dataclasses import dataclass, field

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr.typing import Int32, Int64

from aiter.ops.flydsl.kernels.dsv41_mono.atomv41.kernels import attn_post as k2a
from aiter.ops.flydsl.kernels.dsv41_mono.atomv41.kernels import moe as k2b
from aiter.ops.flydsl.kernels.dsv41_mono.atomv41.kernels.attn_post import (
    attn_post_args,
    attn_post_context,
    attn_post_smem,
    lds_union,
    run_attn_post,
)
from aiter.ops.flydsl.kernels.dsv41_mono.atomv41.kernels.debug import mailbox
from aiter.ops.flydsl.kernels.dsv41_mono.atomv41.kernels.dims import Dims
from aiter.ops.flydsl.kernels.dsv41_mono.atomv41.kernels.moe import (
    moe_args,
    moe_context,
    moe_smem,
    run_moe,
)
from aiter.ops.flydsl.kernels.dsv41_mono.atomv41.kernels.moe_shape import (
    EXPERTS,
    SORT_NETS,
    TOPK,
    MoeBuild,
    route_shape,
)
from aiter.ops.flydsl.kernels.dsv41_mono.atomv41.sources import SOURCES
from aiter.ops.flydsl.kernels.dsv41_mono.atomfw.device.ops import CM_DEV, kernel_symbol, traced
from aiter.ops.flydsl.kernels.dsv41_mono.atomfw.device.ranks import peer_bases
from aiter.ops.flydsl.kernels.dsv41_mono.atomfw.device.stamps import stamp_begin, stamp_flush
from aiter.ops.flydsl.kernels.dsv41_mono.atomfw.device.sync import publish
from aiter.ops.flydsl.kernels.dsv41_mono.atomfw.plan.build_key import key_tuple
from aiter.ops.flydsl.kernels.dsv41_mono.atomfw.plan.execution import BLOCKS, THREADS
from aiter.ops.flydsl.kernels.dsv41_mono.atomfw.runtime.abi import KernelAbi

_CURRENT_STREAM = fx.Stream(None)

# ``timeline`` stamps: K2a's points, then K2b's
TL_STAGES = k2a.TL_STAGES + k2b.TL_STAGES
TL_POINTS = 1 + len(TL_STAGES)


@dataclass(frozen=True)
class LayerPostBuild:
    tokens: int
    index_bound_max: int = 0
    # the index plane is FP4 (``AttnPostBuild.index_fp4``)
    index_fp4: bool = False
    experts: int = EXPERTS
    topk: int = TOPK
    # the TP size: every width a rank holds (``Dims``)
    tp: int = 4
    timeline: bool = field(default=False, metadata={"sym": "tl"})
    # ``ATOM_MONO_DEBUG``: bounded mailbox waits recording at this scratch
    # offset (``debug``); -1 is the normal build
    diag_off: int = -1
    # FP4 groups an ug task (``moe_shape.RouteShape.ug_groups``)
    ug_groups: int = 2

    def halves(self):
        """(K2a's build key, K2b's). K2b's scratch starts at ``moe_base``, where
        this launch's layout puts it (its diag offset from there)."""
        key_a = k2a.AttnPostBuild(
            tokens=self.tokens,
            index_bound_max=self.index_bound_max,
            index_fp4=self.index_fp4,
            tp=self.tp,
            timeline=self.timeline,
            diag_off=self.diag_off,
        )
        moe_diag = self.diag_off - moe_base(key_a) if self.diag_off >= 0 else -1
        key_b = MoeBuild(
            tokens=self.tokens,
            experts=self.experts,
            topk=self.topk,
            tp=self.tp,
            timeline=self.timeline,
            diag_off=moe_diag,
            ug_groups=self.ug_groups,
        )
        return key_a, key_b


def moe_base(key_a) -> int:
    """K2b's scratch offset: past K2a's regions, in a launch of its own as in
    this one. Its regions are not all tagged pairs (the ug queue counts from
    the step's clear), so none may lie under K1's or K2a's."""
    return k2a.scratch_bytes(key_a)


# K2a's arguments, K2b's but its input (x: K2a's ``normed``), the shared tail
_K2A = k2a.ABI.names[: k2a.ABI.names.index("scratch")]
_K2B = [n for n in k2b.ABI.names[: k2b.ABI.names.index("scratch")] if n != "x"]
ABI = KernelAbi(
    (*_K2A, *_K2B, "scratch", "sym", "peers", "rank", "layer", "moe_layer", "tl")
)


def scratch_layout(key: LayerPostBuild) -> dict[str, tuple[int, int]]:
    """K2a's regions, then K2b's and XRDY (a token's normed row written)."""
    key_a, key_b = key.halves()
    out = dict(k2a.scratch_layout(key_a))
    base = moe_base(key_a)
    moe = k2b.scratch_layout(key_b)
    for name, (off, n) in moe.items():
        out[name] = (base + off, n)
    end = base + sum(n for _, n in moe.values())
    out["xrdy"] = (end, key.tokens * 8)
    return out


def scratch_bytes(key: LayerPostBuild) -> int:
    return max(o + n for o, n in scratch_layout(key).values())


@traced
def publish_x(ca, cb, t):
    """Token t's ``normed`` row, written at device scope by this CTA's norm, is
    readable by every CTA: the stores drained, then its flag."""
    publish(cb["put"], cb["xrdy"], t, 1, ca["tid"] == 0)


def build_layer_post(key: LayerPostBuild):
    s = key.tokens
    timeline = key.timeline
    assert 1 <= s <= k2b.MAX_TOKENS
    key_a, key_b = key.halves()
    rs = route_shape(key_b)
    assert rs.experts // 64 in SORT_NETS
    layout = scratch_layout(key)
    layout_a = {n: layout[n] for n in k2a.scratch_layout(key_a)}
    layout_b = {n: layout[n] for n in (*k2b.scratch_layout(key_b), "xrdy")}
    keyed = key_tuple(key, SOURCES)
    # K2a's stages, then K2b: one union (``attn_post_smem``)
    members = attn_post_smem(s, False, key.index_bound_max, Dims(key.tp))
    Halves = lds_union("K2Lds", {**members, "moe": moe_smem(s, rs, False)})

    @fx.struct
    class Stamps:
        tls: fx.Array[fx.Int64, TL_POINTS if timeline else 1, 16]

    name = kernel_symbol(
        "v41_layer_post", s=s, e=rs.experts, k=rs.topk, tl=int(timeline),
        ix=int(key.index_bound_max > 0), i4=int(key.index_fp4),
    )  # fmt: skip

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def layer_post(
        iact: Int32, iq: Int64, iw: Int64, plane: Int64, iqs: Int64,
        pscale: Int64, iplan: Int64, itab: Int64,
        itab_stride: Int32, itab_len: Int32, ibatch: Int64, ishift: Int32,
        ibound: Int64, ilog: Int64,
        lstride: Int32, isel: Int64, icand: Int64, ilift: Int32, iprod: Int32,
        ibmax: Int64, bstride: Int32, icout: Int64, irow: Int64, sbound: Int64,
        q: Int64, pool: Int64, sel: Int64, topk: Int32, kmeta: Int64, table: Int64,
        rows_per_page: Int32, page_rows: Int32, main_off: Int32, ring_off: Int32,
        ring_slots: Int32, sink: Int64,
        qk_scale: Int32, pos: Int64, cos: Int64, sin: Int64, woa: Int64, woa_s: Int64,
        wob: Int64, wob_s: Int64, res_in: Int64, post_in: Int64, comb_in: Int64,
        pre_in: Int64, hc_fn: Int64, hc_scale: Int64, hc_base: Int64, ffn_w: Int64,
        res_out: Int64, post_out: Int64, comb_out: Int64, pre_out: Int64,
        normed: Int64,
        gate_w: Int64, bias: Int64, w13: Int64, w13_s: Int64, w2: Int64,
        w2_s: Int64, sgu: Int64, sgu_s: Int64, sw2: Int64, sw2_s: Int64, out: Int64,
        scratch: Int64, sym: Int64, peers: Int64, rank: Int32, layer: Int32,
        moe_layer: Int32, tl: Int64,
    ):  # fmt: skip
        tid = fx.thread_idx.x
        bid = fx.block_idx.x
        alloc = fx.SharedAllocator()
        stamps = alloc.allocate(Stamps).peek()
        halves = alloc.allocate(Halves)
        lds_a = {name: getattr(halves, name).peek() for name in members}
        lds_b = halves.moe.peek()
        bases = peer_bases(peers, key.tp)
        args_a = attn_post_args(
            iact, iq, iw, plane, iqs, pscale, iplan, itab, itab_stride, itab_len,
            ibatch, ishift, ibound, ilog, lstride,
            isel, icand, ilift, iprod, ibmax, bstride, icout, irow, sbound,
            q, pool, sel, topk, kmeta, table, rows_per_page, page_rows, main_off,
            ring_off, ring_slots, sink, qk_scale, pos, cos, sin, woa, woa_s, wob,
            wob_s, res_in, post_in, comb_in, pre_in, hc_fn, hc_scale, hc_base, ffn_w,
            res_out, post_out, comb_out, pre_out, normed,
        )  # fmt: skip
        args_b = moe_args(
            normed, gate_w, bias, w13, w13_s, w2, w2_s, sgu, sgu_s, sw2, sw2_s, out
        )
        ca = attn_post_context(
            key_a, s, lds_a, mailbox(layer, scratch, key.diag_off, layout),
            bases, layout_a, scratch,
            args_a, rank, sym,
        )  # fmt: skip
        ca["normed_cm"] = CM_DEV
        cb = moe_context(
            s, rs, lds_b, mailbox(moe_layer, scratch, key.diag_off, layout),
            bases, layout_b, scratch,
            args_b, rank, sym,
        )  # fmt: skip
        cb["x_cm"], cb["x_ready"] = CM_DEV, True
        tls = stamps.tls.ptr
        stamp_begin(timeline, tls, tid, TL_POINTS)
        run_attn_post(ca, key_a, bid, tls, 0, lambda t: publish_x(ca, cb, t))
        run_moe(cb, key_b, bid, tls, len(k2a.TL_STAGES))
        stamp_flush(timeline, tls, tl, tid, bid, TL_POINTS)
        _ = keyed

    @flyc.jit
    def launch(
        iact: Int32, iq: Int64, iw: Int64, plane: Int64, iqs: Int64,
        pscale: Int64, iplan: Int64, itab: Int64,
        itab_stride: Int32, itab_len: Int32, ibatch: Int64, ishift: Int32,
        ibound: Int64, ilog: Int64,
        lstride: Int32, isel: Int64, icand: Int64, ilift: Int32, iprod: Int32,
        ibmax: Int64, bstride: Int32, icout: Int64, irow: Int64, sbound: Int64,
        q: Int64, pool: Int64, sel: Int64, topk: Int32, kmeta: Int64, table: Int64,
        rows_per_page: Int32, page_rows: Int32, main_off: Int32, ring_off: Int32,
        ring_slots: Int32, sink: Int64,
        qk_scale: Int32, pos: Int64, cos: Int64, sin: Int64, woa: Int64, woa_s: Int64,
        wob: Int64, wob_s: Int64, res_in: Int64, post_in: Int64, comb_in: Int64,
        pre_in: Int64, hc_fn: Int64, hc_scale: Int64, hc_base: Int64, ffn_w: Int64,
        res_out: Int64, post_out: Int64, comb_out: Int64, pre_out: Int64,
        normed: Int64,
        gate_w: Int64, bias: Int64, w13: Int64, w13_s: Int64, w2: Int64,
        w2_s: Int64, sgu: Int64, sgu_s: Int64, sw2: Int64, sw2_s: Int64, out: Int64,
        scratch: Int64, sym: Int64, peers: Int64, rank: Int32, layer: Int32,
        moe_layer: Int32, tl: Int64, stream: fx.Stream = _CURRENT_STREAM,
    ):  # fmt: skip
        _ = keyed
        layer_post(
            iact, iq, iw, plane, iqs, pscale, iplan, itab, itab_stride, itab_len,
            ibatch, ishift, ibound, ilog, lstride,
            isel, icand, ilift, iprod, ibmax, bstride, icout, irow, sbound,
            q, pool, sel, topk, kmeta, table, rows_per_page, page_rows, main_off,
            ring_off, ring_slots, sink, qk_scale, pos, cos, sin, woa, woa_s, wob,
            wob_s, res_in, post_in, comb_in, pre_in, hc_fn, hc_scale, hc_base, ffn_w,
            res_out, post_out, comb_out, pre_out, normed,
            gate_w, bias, w13, w13_s, w2, w2_s, sgu, sgu_s, sw2, sw2_s, out,
            scratch, sym, peers, rank, layer, moe_layer, tl,
        ).launch(grid=(BLOCKS,), block=(THREADS,), stream=stream)  # fmt: skip

    ABI.check(layer_post, launch)
    return launch
