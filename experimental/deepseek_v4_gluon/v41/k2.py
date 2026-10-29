# SPDX-License-Identifier: MIT
"""The DS V4.1 K2 launch in Gluon: ``layer.build_mono_k2`` (vLLM 8321d11), one
persistent launch a layer and rank, 256 CTAs x 512 threads, S rows: the
attention back (``back.py``: split, combine, wo_a, wo_b -- wo_b's bf16 pairs
pushed straight into every rank's ATTN region) then the FFN half
(``ffn.ffn_half`` without its push: the seam reduces what wo_b pushed). Same
launch ABI as FlyDSL's (+ ``health``), the same scratch / peer layout and tags.
An indexed build (``INDEX``) first writes the key table from vLLM's top-k
(``front.stage_kt``, the last S CTAs), each attention unit waiting on its
token's (KRDY)."""

from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from ..common.epoch import epoch_begin, load_epoch
from ..common.sync import SMEM_FLAT, T512, deadline, publish
from . import back as B
from . import front as F
from .ffn import ffn_half, first_task, peer_bases, u8


@gluon.jit
def k2(
    q, swa, swa_stride, swa_block, comp, comp_stride, comp_block, kt, klen, slot, swa_idx,
    swa_lens, t2r, topk, comp_bt, bt_stride, pos, sink, qk_scale, cos_sin, woa, woa_s, wob,
    wob_s, zrec,
    res_in, post_in, comb_in, pre_in, hc_fn, hc_scale, hc_base, ffn_w,
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
    HEADS: gl.constexpr, RATIO: gl.constexpr, INDEX: gl.constexpr, PART: gl.constexpr,
    PSTAT: gl.constexpr, SRDY: gl.constexpr, XO: gl.constexpr, XOS: gl.constexpr,
    CRDY: gl.constexpr, X8B: gl.constexpr, X8BS: gl.constexpr, ARDY: gl.constexpr,
    KRDY: gl.constexpr, SWA_W: gl.constexpr,
):  # fmt: skip
    q, swa, comp, kt, klen, zrec = u8(q), u8(swa), u8(comp), u8(kt), u8(klen), u8(zrec)
    slot, swa_idx, swa_lens, t2r, topk, comp_bt = (u8(slot), u8(swa_idx), u8(swa_lens), u8(t2r),
                                                   u8(topk), u8(comp_bt))  # fmt: skip
    pos, sink, cos_sin = u8(pos), u8(sink), u8(cos_sin)
    woa, woa_s, wob, wob_s = u8(woa), u8(woa_s), u8(wob), u8(wob_s)
    scratch, peers, epoch, health = u8(scratch), u8(peers), u8(epoch), u8(health)
    tid = gl.arange(0, 512, layout=T512)
    bid = gl.program_id(0)
    ep = load_epoch(epoch)
    peer = peer_bases(peers, (ep & 1).to(gl.int64) * HALF, TP)
    epoch_begin(epoch, ep, bid, tid)
    tag = (ep << 1) + 2 + gl.zeros_like(tid)
    dl = deadline(tid)
    sb = scratch
    # ---- the attention back (upstream ``back_plan``'s placement)
    NG: gl.constexpr = B.head_groups(HEADS, S)
    CK: gl.constexpr = B.chunk_keys(HEADS, S)
    NCHUNK: gl.constexpr = B.n_chunks(HEADS, S)
    HT: gl.constexpr = HEADS // 16
    NKT: gl.constexpr = 8 // NG
    N_SPLIT: gl.constexpr = S * NCHUNK
    N_COMB: gl.constexpr = S * (HEADS // 8)
    WOA_TASKS: gl.constexpr = (8 // TP) * 1024 // 32
    COMB0: gl.constexpr = N_SPLIT % 256
    WOA0: gl.constexpr = (N_SPLIT + N_COMB) % 256
    WOB0: gl.constexpr = (WOA0 + WOA_TASKS) % 256
    if INDEX:
        # the key table, from the top-k vLLM's indexer selected after K1, on
        # the last CTAs: each split / combine unit waits on its token's
        for t in range(first_task(bid, 256 - S), S, 256):
            F.stage_kt(t, slot, swa_idx, swa_lens, pos, t2r, topk, comp_bt, bt_stride,
                       comp_block, kt, klen, S, RATIO, SWA_W)  # fmt: skip
            publish(sb + KRDY, gl.zeros_like(tid) + t, gl.zeros_like(tid) + 1, tag, tid == 0)
    q_s = gl.allocate_shared_memory(gl.bfloat16, [16, HT, 16, 32], B.SH4)
    raw_n = gl.allocate_shared_memory(gl.uint8, [16, NG, CK, 32], B.SH4)
    raw_r = gl.allocate_shared_memory(gl.bfloat16, [2, NG, CK, 32], B.SH4)
    raw_s = gl.allocate_shared_memory(gl.uint8, [NG, CK, 32], B.SH3)
    p_s = gl.allocate_shared_memory(gl.bfloat16, [HT, 16, CK], B.SH3)
    st_s = gl.allocate_shared_memory(gl.float32, [B.p2(HEADS * (1 + NKT))], SMEM_FLAT)
    for unit in range(first_task(bid, 0), N_SPLIT, 256):
        B.stage_split(unit, q, kt, klen, swa, swa_stride, swa_block, comp, comp_stride,
                      comp_block, zrec, qk_scale, sb + PART, sb + PSTAT, sb + SRDY, sb + KRDY,
                      q_s, raw_n, raw_r, raw_s, p_s, st_s, tag, tid, dl, health, rank, S, HEADS,
                      RATIO, INDEX)  # fmt: skip
    for unit in range(first_task(bid, COMB0), N_COMB, 256):
        B.stage_combine(unit, klen, pos, sink, cos_sin, sb + PART, sb + PSTAT, sb + SRDY,
                        sb + KRDY, sb + XO, sb + XOS, sb + CRDY, tag, tid, dl, health, rank, S,
                        HEADS, INDEX)  # fmt: skip
    gl.barrier()
    x_s = gl.allocate_shared_memory(gl.uint8, [65536], B.SH1)
    c_s = gl.allocate_shared_memory(gl.uint8, [2048], B.SH1)
    red = gl.allocate_shared_memory(gl.float32, [4096], SMEM_FLAT)
    woa_task = first_task(bid, WOA0)
    if woa_task < WOA_TASKS:
        B.stage_woa(woa_task, woa, woa_s, sb + XO, sb + XOS, sb + X8B, sb + X8BS, sb + CRDY, x_s,
                    c_s, red, tag, tid, dl, health, rank, S, HEADS, TP)  # fmt: skip
        publish(sb + ARDY, gl.zeros_like(tid) + woa_task, gl.zeros_like(tid) + 1, tag, tid == 0)
    wob_task = first_task(bid, WOB0)
    wob2 = wob_task + 256
    B.stage_wob(wob_task, gl.minimum(wob2, HIDDEN // 16 - 1), wob2 < HIDDEN // 16, wob, wob_s,
                sb + X8B, sb + X8BS, sb + ARDY, peer, x_s, c_s, red, tag, tid, dl, health, rank, S,
                HEADS, TP, HIDDEN)  # fmt: skip
    gl.barrier()
    # ---- the FFN half: the seam reduces wo_b's pushes (no push of its own)
    ffn_half(
        scratch, res_in, post_in, comb_in, pre_in, hc_fn, hc_scale, hc_base, ffn_w, res_out,
        post_out, comb_out, pre_out, gate_w, bias, w13, w13_s, w2, w2_s, sgu, sgu_s, sw2,
        sw2_s, out, scratch, sym, peers, rank, epoch, tl, health, S, TP, HIDDEN, HALF,
        FFN_OFF, LIN, PMIX, NORMED, XRDY, X8R, X8RS, X8, X8S, X8RDY, LOGIT, ROUTE, SMID,
        SMQ, SMC, MID, MIDS, UGF, EPOCH_COUNTERS, EXPERTS, TOPK, SCALE, ROUTER0, ROUTE0,
        SHARED0, UG0, ROUTER_TASKS, GROUPS, PER, MAX_U, UG_PARTS, FIRST0, ORDER0,
        EXPERT_SLOT0, MASK0, QUEUE0, TILE_FIRST0, TILE_SLOT0, MULTI, SLOT_TILES, TABW,
        SH_INTER, INTER, INTER_REAL, SCALE_COLS, SH_UNITS, MOE, False,
    )  # fmt: skip


def build_constexprs(s: int, tp: int, ratio: int = 1, moe: bool = True,
                     index: bool = False) -> dict:  # fmt: skip
    """K2's constexprs for width s, TP tp: the FFN half's (``ffn.build_constexprs``)
    and the attention back's regions of upstream's own scratch layout. GPU image
    only (upstream imports aiter)."""
    from vllm.models.deepseek_v41.amd.mono import layer as L

    from .ffn import build_constexprs as ffn_constexprs

    c = ffn_constexprs(s, tp, moe)
    off = {k: v[0] for k, v in L.scratch_layout(s, tp).items()}
    c.update(HEADS=64 // tp, RATIO=ratio, INDEX=index, PART=off["part"], PSTAT=off["pstat"],
             SRDY=off["srdy"], XO=off["xo"], XOS=off["xos"], CRDY=off["crdy"], X8B=off["x8b"],
             X8BS=off["x8bs"], ARDY=off["ardy"], KRDY=off.get("krdy", off["ardy"]), SWA_W=128)
    return c
