# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2025-2026 FlyDSL Project Contributors

"""gfx1250 NVFP4 sparse-prefill attention with the tile's rows split over waves.

``pa_sparse_prefill_nvfp4_v2`` splits a page's 128 tokens over the CTA's four
waves, so every wave keeps every row's softmax state and O; the tile stops at
two rows. Here a CTA takes ``tile_rows`` (8) consecutive rows of one KV head and
wave w owns ``tile_rows // 4`` of them (rows 2w and 2w + 1 at 8: v2 tq2's state
a wave), walking every token of the pages its rows selected. A row of the
synthetic M3.1 picks then walks 3.0-3.5 union pages instead of v2's 8.8-9.4.

The union of the tile's page tables is built as in v1/v2. Each union page then
goes to one of two phases, by how many waves select it:

* phase A, ``share_min`` or more waves: a CTA ring of whole pages. Wave w DMAs
  plane w of the page (K, V, K scales, V scales: one tensor DMA a page and wave),
  and a CTA barrier a page publishes the planes and frees the oldest slot. Wave
  w then dequantizes the page's V quarter w once for the CTA (the 32x16x128
  selector WMMA, to FP16 in the PV A operand's order) into a shared V buffer,
  and a second barrier publishes it. Every wave passes every A page's barriers;
  it computes only the 32-token quarters its rows reach, its PV operands read
  from the shared V (V_AHEAD blocks in flight).
* phase B, the rest: each wave alone, over the 32-token quarters of its own
  pages, through a wave-private ring of v2's quarter slots (no barrier, V
  dequantized by the wave). A quarter's K/V pair of planes (the codes, then
  the scales) stages with one 2-D tensor DMA, its row stride the distance
  between the two pointers (``tdm_fold``), or two. With room, the B rings take
  A ring frames the A walk has freed.

Each wave walks a list of (page, quarter) items from LDS, A's in one loop and
B's in a second; the page step on a quarter is v2's (rows paired when both
selected the page, the query terms in VGPRs). The walk reads its metadata a
step ahead (the next item's word, the ring entry an A page stages, the page's
V planes before its barrier), so no LDS latency sits between a step's start
and its first DMA or WMMA. A row has one owner, so the epilogue scales and
stores without a cross-wave merge.

Numerics: QK is v2's (FP32 over the exact NVFP4 K and the query's two FP8
terms); a row's pages accumulate in A-then-B order. PV takes FP16 operands
where v2 takes BF16: NVFP4 V is exact in both, P rounds 8x finer. That margin
pays for a lazy softmax reference: a row's moves only when a head's maximum
passes it by ``2**REBASE_LOG2``, so the accumulator rescale (64 FP32 products a
row) runs on a few of a row's quarters instead of about half of them. P is
computed ``2**P_SHIFT`` above the reference: at most ``2**P_MAX_LOG2``, and
FP16-normal down to ``2**-(14 + P_SHIFT)`` of a head's running maximum. The
error vs FP32 is below v2's. BF16 output only: the wrapper routes the MXFP8 output (one
KV head) and small grids to v2.

``dbg`` (benchmark decomposition only): 1 the walk and its DMA without math,
2 prologue and epilogue only, 3 the math without DMA (barriers kept).
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, range_constexpr
from flydsl.expr.rocdl import tdm_ops
from flydsl.expr.typing import T

from .pa_decode.op_lds import _gptr
from .pa_sparse_prefill_nvfp4 import _bound_stride, _Lds
from .pa_sparse_prefill_nvfp4_v2 import (
    _E4M3_ONE,
    _PLANE_K,
    _PLANE_KS,
    _PLANE_V,
    _PLANE_VS,
    _ROW,
    _SROW,
    _TOKENS,
    HEAD_DIM,
    LOG2E,
    NWARP,
    PAGE,
    THREADS,
    TOPK,
    WAVE,
    WAVE_SLOT_BYTES,
    _attr,
    _bound_mask,
    _head_max,
    _probs,
    _qk,
    _query_terms,
    _rescale,
    _rows_operand,
    _selector_operand,
    _wmma_scale16_32x16,
)
from .utils import rcp_f32

TILE_ROWS = (4, 8)
QUARTERS = PAGE // _TOKENS
# A ring frame: one page's planes, each over all 128 tokens (an A ring slot), or
# four quarter slots of the B rings (v2's slot layout: a quarter's K, V, K
# scales and V scales back to back, so a K/V pair is one 2-D tensor DMA).
A_SLOT = NWARP * WAVE_SLOT_BYTES
A_PLANE_V = PAGE * _ROW
A_PLANE_KS = 2 * PAGE * _ROW
A_PLANE_VS = A_PLANE_KS + PAGE * _SROW
# The shared V of an A page: FP16 in the PV A operand's VGPR order, quarter q
# (32 tokens, dequantized by wave q), dims block b, lane half j at
# q * V_QUARTER + b * 1024 + j * 512, 16 B a lane.
V_QUARTER = _TOKENS * HEAD_DIM * 2
# Shared V blocks a wave keeps in flight: 5 issued before QK, then each 5 ahead of
# its PV (4: 0.958x, 6 and 8 spill in the loop; paired A/B on the M3.1 chunks).
V_AHEAD = 5
# Item word: bounds of the wave's rows (8 bits each), the token quarter, A's page
# head flag, the LDS frame and quarter slot it is read from, the A page's index
# in the A list and (A items) "this wave dequantizes its quarter of the V".
_Q_SHIFT, _HEAD_SHIFT, _SLOT_SHIFT, _FRAME_SHIFT, _IDX_SHIFT = 16, 18, 19, 21, 24
_DQ_SHIFT = 31
# FP16 P (see the module notes): at most 2**P_MAX_LOG2, computed P_SHIFT above
# the reference, which moves when a head's maximum passes it by 2**REBASE_LOG2
# (0: 0.959x, 4: 0.989x, 12: 1.000x of 8's speed; paired A/B on the M3.1 chunks).
P_MAX_LOG2 = 15
REBASE_LOG2 = 8
P_SHIFT = P_MAX_LOG2 - REBASE_LOG2

_CACHE = {}


def _b_alias(nb_a, nb_b):
    """The B rings fit the A frames the last A page leaves free."""
    return nb_b < nb_a


def _layout_v3(tq, nb_a, nb_b):
    """Byte offsets of the CTA's LDS regions."""
    entries = tq * TOPK
    # Frames 0 .. nb_a - 1: the A ring. The B rings: wave w's slot j is quarter
    # slot g % 4 of frame b0 + g // 4, g = w * nb_b + j, where b0 is the first A
    # frame free at the last A page (``b_slot``) or, without room there, nb_a.
    frames = nb_a if _b_alias(nb_a, nb_b) else nb_a + nb_b
    vbuf = frames * A_SLOT
    keys = vbuf + QUARTERS * V_QUARTER
    uidx = keys + 4 * entries
    upage = uidx + 4 * entries
    ubound = upage + 4 * entries
    counts = ubound + (entries * _bound_stride(tq) + 15) // 16 * 16
    alist = counts + 16
    items = alist + 8 * entries
    # A wave's items: at most one a page it does not select (an A page's
    # barrier) plus four quarters a page its rows select.
    per_wave = entries + QUARTERS * TOPK * (tq // NWARP)
    # + 32: the walk reads up to three items past the last.
    total = items + NWARP * per_wave * 8 + 32
    return {
        "vbuf": vbuf,
        "keys": keys,
        "uidx": uidx,
        "upage": upage,
        "ubound": ubound,
        "counts": counts,
        "alist": alist,
        "items": items,
        "per_wave": per_wave,
        "total": (total + 15) // 16 * 16,
    }


def _wmma_f16(a, b, acc):
    """16x16x32 FP16 WMMA, FP32 accumulator (raw ROCDL: Gfx1250Gemm.mma takes
    BF16 or FP8 operands only)."""
    return fx.Vector(
        fx.rocdl.wmma_f32_16x16x32_f16(
            fx.Vector.make_type(8, fx.Float32),
            fx.as_ir_value(a),
            fx.as_ir_value(b),
            fx.as_ir_value(acc),
            modC=_attr("#rocdl<wmma_c_modifier none>"),
            reuseA=False,
            reuseB=False,
        ).result
    )


def kernel_name(tq, share_min, nb_a, nb_b, waves_per_eu, dbg, tdm_fold=""):
    name = (
        f"pa_sparse_prefill_nvfp4_v3_d128_bs128_gqa16_top16_tq{tq}"
        f"_sm{share_min}_a{nb_a}b{nb_b}"
    )
    if tdm_fold:
        name += f"_tf{tdm_fold.replace('-', 'x')}"
    if dbg:
        name += f"_dbg{dbg}"
    if waves_per_eu:
        name += f"_w{waves_per_eu}"
    return name


def compile_pa_sparse_prefill_nvfp4_v3(
    *,
    tile_rows=8,
    softmax_scale,
    nvfp4_buffers=3,
    share_min=4,
    nb_b=2,
    waves_per_eu=3,
    quant_mx=False,
    dbg=0,
    tdm_fold="",
):
    """Compile (cached) the row-split reuse kernel; ``launch`` takes the same
    arguments as ``compile_pa_sparse_prefill_nvfp4``'s (the MXFP8 ones unused).

    ``tile_rows``: 8 (two rows a wave) or 4 (one). ``share_min``: waves that must
    select a page for the CTA ring (phase A); 1 runs every page there, 5 none.
    ``nvfp4_buffers`` / ``nb_b``: depth of the CTA ring / of each wave's quarter
    ring (``nb_b < nvfp4_buffers``: the B rings take A frames the A walk has
    freed). ``waves_per_eu``: the occupancy hint (0: none). ``tdm_fold``: per
    K/V pair of a B quarter's planes (the codes, then the scales) "k" or "v",
    the lower-addressed plane, which leads one 2-D tensor DMA and takes the
    pair's first slot plane, or "-" (two DMAs); the wrapper reads it off the
    pointers. "" (or "--"): four DMAs a quarter.
    """
    tq, nb_a, nb_b = int(tile_rows), int(nvfp4_buffers), int(nb_b)
    share_min, dbg = int(share_min), int(dbg)
    # A page's V planes are published at the previous page's first barrier
    # (its V reads then precede its own): page idx's DMA must be issued two
    # pages ahead, i.e. a CTA ring of 3.
    early_v = nb_a == 3
    if tq not in TILE_ROWS:
        raise ValueError(f"tile_rows must be one of {TILE_ROWS}")
    if nb_a not in (2, 3) or nb_b not in (2, 3):
        raise ValueError("nvfp4_buffers and nb_b must be 2 or 3")
    if not 1 <= share_min <= NWARP + 1:
        raise ValueError("share_min must be 1 to 5")
    if quant_mx:
        raise ValueError("v3 stores BF16 only; the MXFP8 output runs on v2")
    tdm_fold = "" if tdm_fold in ("", "--") else tdm_fold
    if tdm_fold and (len(tdm_fold) != 2 or set(tdm_fold) - set("kv-")):
        raise ValueError("tdm_fold must be two of 'k'/'v'/'-'")
    scale = float(softmax_scale)
    key = (tq, nb_a, nb_b, share_min, scale, int(waves_per_eu), dbg, tdm_fold)
    if key in _CACHE:
        return _CACHE[key]
    lay = _layout_v3(tq, nb_a, nb_b)
    rpw = tq // NWARP
    entries = tq * TOPK
    ub = _bound_stride(tq)
    per_wave = lay["per_wave"]
    assert entries <= THREADS and lay["total"] <= 327680
    # A folded pair's second plane lands one packed row after its first.
    assert _PLANE_V - _PLANE_K == _TOKENS * _ROW
    assert _PLANE_VS - _PLANE_KS == _TOKENS * _SROW
    # B slot plane offsets: a folded pair's lower-addressed plane takes the
    # pair's first plane.
    fold = tdm_fold or "--"
    b_dmas = sum(1 if lead != "-" else 2 for lead in fold)
    b_k, b_v = (_PLANE_V, _PLANE_K) if fold[0] == "v" else (_PLANE_K, _PLANE_V)
    b_ks, b_vs = (_PLANE_VS, _PLANE_KS) if fold[1] == "v" else (_PLANE_KS, _PLANE_VS)
    # A name, not lay["total"]: linters read a string in an annotation as a
    # forward reference.
    lds_words = lay["total"] // 4

    @fx.struct
    class SharedStorage:
        buf: fx.Array[fx.Int32, lds_words, 16]

    unit = scale * LOG2E

    @flyc.kernel(
        name=kernel_name(tq, share_min, nb_a, nb_b, waves_per_eu, dbg, tdm_fold),
        known_block_size=(THREADS, 1, 1),
    )
    def kernel(
        query: fx.Pointer,
        key: fx.Pointer,
        value: fx.Pointer,
        ks: fx.Pointer,
        vs: fx.Pointer,
        table: fx.Pointer,
        context: fx.Pointer,
        output: fx.Pointer,
        rows: fx.Int32,
        kv_heads: fx.Int32,
        table_stride: fx.Int32,
        sk_block: fx.Int32,
        sq_row: fx.Int32,
        sq_head: fx.Int32,
        pages: fx.Int32,
        quantized_output: fx.Pointer,
        output_scale: fx.Pointer,
        quant_rows: fx.Int32,
    ):
        tid = fx.Int32(gpu.thread_id("x"))
        warp = fx.Int32(fx.rocdl.readfirstlane(T.i32, tid // WAVE))
        lane = tid % WAVE
        lane16, half = lane % 16, lane // 16
        tile = fx.Int32(gpu.block_id("x"))
        kvh = fx.Int32(gpu.block_id("y"))
        storage = fx.SharedAllocator().allocate(SharedStorage).peek()
        mem = _Lds(fx.recast_iter(fx.Uint8, storage.buf.ptr))

        def lds_load(offset, dtype, n=1):
            return mem.load(offset, dtype, n)

        def lds_store(offset, value, dtype):
            mem.store(offset, value, dtype)

        def tdm(*args):
            mem.tdm_copy(*args)

        def uniform(value):
            return fx.Int32(fx.rocdl.readfirstlane(T.i32, value))

        table_i32 = fx.recast_iter(fx.Int32, table)
        context_i32 = fx.recast_iter(fx.Int32, context)

        # This wave's query rows load first: their latency overlaps the
        # page-table loads and the union build.
        query_bf16 = fx.recast_iter(fx.BFloat16, query)
        q_loaded = []
        for r in range_constexpr(rpw):
            row = tile * tq + warp * rpw + r
            seq = (row < rows).select(row * kv_heads + kvh, fx.Int32(0))
            qbase = fx.Int64(seq) * sq_row + lane16 * sq_head
            q_loaded.append(
                [
                    fx.Vector(
                        fx.ptr_load(
                            fx.add_offset(query_bf16, qbase + j * 32 + half * 16),
                            result_type=fx.Vector.make_type(16, fx.BFloat16),
                        )
                    )
                    for j in range_constexpr(4)
                ]
            )

        def union_barrier():
            """Publishes the union's LDS words with a fence over LDS only:
            gpu.barrier()'s workgroup fence also waits for every global load
            and store in flight (s_wait_loadcnt / storecnt 0), the query rows
            above included, which only the query terms read."""
            gpu.barrier(address_spaces=_attr("[#gpu.address_space<workgroup>]"))

        # ---- the union of the tile's page tables (as pa_sparse_prefill_nvfp4_v2) ----
        e_row = tid // TOPK
        e_slot = tid % TOPK
        g_row = tile * tq + e_row
        e_live = (tid < entries) & (g_row < rows)
        e_seq = e_live.select(g_row * kv_heads + kvh, fx.Int32(0))
        e_ctx = fx.Int32(context_i32[e_seq])
        e_page = fx.Int32(table_i32[e_seq * table_stride + e_slot])
        e_bound = fx.max(fx.Int32(0), fx.min(fx.Int32(PAGE), e_ctx - e_slot * PAGE))
        e_live = e_live & (e_page >= 0) & (e_page < pages)
        e_bound = e_live.select(e_bound, fx.Int32(0))
        e_valid = e_bound > 0
        e_key = e_valid.select(e_page, fx.Int32(-1))
        if tid < entries:
            lds_store(
                lay["keys"] + tid * 4,
                fx.Vector.from_elements([e_key], dtype=fx.Int32),
                fx.Int32,
            )
        zero_words = entries * ub // 4
        for i in range_constexpr((zero_words + THREADS - 1) // THREADS):
            w = i * THREADS + tid
            if w < zero_words:
                lds_store(
                    lay["ubound"] + w * 4,
                    fx.Vector.from_elements([fx.Int32(0)], dtype=fx.Int32),
                    fx.Int32,
                )
        union_barrier()
        # The first entry with this lane's key: scanned from the last entry to
        # the first, one select a key (the last match is the lowest index).
        # Testing "no match yet" in a forward scan puts a compare -> SGPR ->
        # scalar AND -> select chain in every one of the 128 steps.
        first = fx.Int32(entries)
        for i in range_constexpr(entries // 4):
            c = entries // 4 - 1 - i
            keys4 = lds_load(lay["keys"] + c * 16, fx.Int32, 4)
            for j in range_constexpr(4):
                e = 3 - j
                first = (keys4[e] == e_key).select(fx.Int32(c * 4 + e), first)
        is_rep = e_valid & (first == tid)
        rep_mask = fx.Int32(fx.rocdl.ballot(T.i32, is_rep))
        rank = fx.Int32(
            fx.rocdl.mbcnt_lo(
                fx.Int32.ir_type, rep_mask.ir_value(), fx.Int32(0).ir_value()
            )
        )
        if lane == 0:
            lds_store(
                lay["counts"] + warp * 4,
                fx.Vector.from_elements([fx.Int32(fx.ctpop(rep_mask))], dtype=fx.Int32),
                fx.Int32,
            )
        union_barrier()
        counts = lds_load(lay["counts"], fx.Int32, 4)
        wave_base = fx.Int32(0)
        for w in range_constexpr(1, NWARP):
            wave_base = wave_base + (warp >= w).select(counts[w - 1], fx.Int32(0))
        n_union = uniform(counts[0] + counts[1] + counts[2] + counts[3])
        if is_rep:
            u_rep = wave_base + rank
            lds_store(
                lay["uidx"] + tid * 4,
                fx.Vector.from_elements([u_rep], dtype=fx.Int32),
                fx.Int32,
            )
            lds_store(
                lay["upage"] + u_rep * 4,
                fx.Vector.from_elements([e_page], dtype=fx.Int32),
                fx.Int32,
            )
        union_barrier()
        if e_valid:
            u_entry = lds_load(lay["uidx"] + first * 4, fx.Int32, 1)[0]
            lds_store(
                lay["ubound"] + u_entry * ub + e_row,
                fx.Vector.from_elements([e_bound.to(fx.Int8)], dtype=fx.Int8),
                fx.Int8,
            )
        union_barrier()

        # ---- the A list (shared) and this wave's item list ----
        # Every wave derives the same A list from the union (and writes the same
        # words), so no barrier publishes it; the items are the wave's own.
        items_w = lay["items"] + warp * (per_wave * 8)

        def ballot(cond):
            return fx.Int32(fx.rocdl.ballot(T.i32, cond))

        def below(mask):
            """Set bits of ``mask`` under this lane."""
            return fx.Int32(
                fx.rocdl.mbcnt_lo(
                    fx.Int32.ir_type, mask.ir_value(), fx.Int32(0).ir_value()
                )
            )

        def put_items(count, at, page, info, slot_of):
            """Items ``at + t`` = (page, info | t << _Q_SHIFT | slot_of(t, at +
            ...)) for t < ``count`` (0..4 a lane), packed in lane order; returns
            ``at`` past them."""
            masks = [ballot(count > t) for t in range_constexpr(QUARTERS)]
            prefix = below(masks[0])
            total = fx.Int32(fx.ctpop(masks[0]))
            for t in range_constexpr(1, QUARTERS):
                prefix = prefix + below(masks[t])
                total = total + fx.Int32(fx.ctpop(masks[t]))
            for t in range_constexpr(QUARTERS):
                if count > t:
                    pos = at + prefix + t
                    word = info | (t << _Q_SHIFT) | slot_of(t, pos)
                    if const_expr(t == 0):
                        word = word | (1 << _HEAD_SHIFT)
                    lds_store(
                        items_w + pos * 8,
                        fx.Vector.from_elements([page, word], dtype=fx.Int32),
                        fx.Int32,
                    )
            return at + total

        n_a = fx.Int32(0)
        n_ai = fx.Int32(0)
        b_pending = []
        for k in range_constexpr((entries + WAVE - 1) // WAVE):
            u = k * WAVE + lane
            live = u < n_union
            page = lds_load(lay["upage"] + u * 4, fx.Int32, 1)[0]
            raw = lds_load(lay["ubound"] + u * ub, fx.Int32, ub // 4)
            bw = [live.select(raw[i], fx.Int32(0)) for i in range_constexpr(ub // 4)]
            count = fx.Int32(0)
            for w in range_constexpr(NWARP):
                at = w * rpw
                mask = (1 << (8 * rpw)) - 1
                sel = ((bw[at // 4] >> (8 * (at % 4))) & mask) != 0
                count = count + sel.select(fx.Int32(1), fx.Int32(0))
            top_all = fx.Int32(0)
            for t in range_constexpr(tq):
                top_all = fx.max(top_all, (bw[t // 4] >> (8 * (t % 4))) & 0xFF)
            word = bw[0]
            for i in range_constexpr(1, ub // 4):
                word = (warp * rpw >= 4 * i).select(bw[i], word)
            shift = (warp * rpw % 4) * 8
            mine = [(word >> (shift + 8 * r)) & 0xFF for r in range_constexpr(rpw)]
            top_mine = mine[0]
            for r in range_constexpr(1, rpw):
                top_mine = fx.max(top_mine, mine[r])
            quarters = (top_mine + (_TOKENS - 1)) // _TOKENS
            bounds_info = mine[0]
            for r in range_constexpr(1, rpw):
                bounds_info = bounds_info | (mine[r] << (8 * r))
            is_a = live & (count >= share_min)
            a_mask = ballot(is_a)
            a_idx = n_a + below(a_mask)
            if is_a:
                lds_store(
                    lay["alist"] + a_idx * 8,
                    fx.Vector.from_elements([page, top_all], dtype=fx.Int32),
                    fx.Int32,
                )
            # Every A page is at least one item (its barrier), quarters past the
            # wave's rows none.
            a_count = is_a.select(fx.max(quarters, fx.Int32(1)), fx.Int32(0))
            # A page a_idx's quarter t: frame a_idx % nb_a, slot t.
            a_frame = ((a_idx % nb_a) << _FRAME_SHIFT) | (a_idx << _IDX_SHIFT)
            # Wave w dequantizes quarter w if some row of the tile reaches it.
            dq = (top_all > warp * _TOKENS).select(fx.Int32(1), fx.Int32(0))
            a_frame = a_frame | (dq << _DQ_SHIFT)
            n_ai = put_items(
                a_count,
                n_ai,
                page,
                bounds_info | a_frame,
                lambda t, pos: fx.Int32(t << _SLOT_SHIFT),
            )
            n_a = n_a + fx.Int32(fx.ctpop(a_mask))
            is_b = live & (count < share_min)
            b_pending.append((is_b.select(quarters, fx.Int32(0)), page, bounds_info))

        # The B rings' first frame: with room, the frames of A pages n_a - nb_a
        # onward, free once every wave passed the last A page's barrier, where
        # the first B items are staged.
        if const_expr(_b_alias(nb_a, nb_b)):
            b_frame = n_a % nb_a
        else:
            b_frame = fx.Int32(nb_a)

        def b_slot(t, pos):
            """B item ``pos``: ring slot (pos - n_ai) % nb_b of this wave."""
            g = warp * nb_b + (pos - n_ai) % nb_b
            frame = b_frame + (g >> 2)
            if const_expr(_b_alias(nb_a, nb_b)):
                frame = frame % nb_a
            return (frame << _FRAME_SHIFT) | ((g & 3) << _SLOT_SHIFT)

        n_items = n_ai
        for b_count, page, bounds_info in b_pending:
            n_items = put_items(b_count, n_items, page, bounds_info, b_slot)
        n_a = uniform(n_a)
        n_ai = uniform(n_ai)
        n_items = uniform(n_items)
        if const_expr(dbg == 2):
            # Debug: prologue and epilogue only (no page walk).
            n_a = fx.Int32(0) * n_a
            n_ai = fx.Int32(0) * n_ai
            n_items = fx.Int32(0) * n_items

        def planes(info):
            """LDS offsets of an A item's K, V, K-scale and V-scale rows."""
            frame = ((info >> _FRAME_SHIFT) & 7) * A_SLOT
            slot = (info >> _SLOT_SHIFT) & (QUARTERS - 1)
            p_k = frame + slot * (_TOKENS * _ROW)
            p_ks = frame + A_PLANE_KS + slot * (_TOKENS * _SROW)
            return p_k, p_k + A_PLANE_V, p_ks, p_ks + PAGE * _SROW

        def b_base(info):
            """LDS offset of a B item's quarter slot."""
            frame = ((info >> _FRAME_SHIFT) & 7) * A_SLOT
            return frame + ((info >> _SLOT_SHIFT) & (QUARTERS - 1)) * WAVE_SLOT_BYTES

        def planes_b(info):
            """LDS offsets of a B item's K, V, K-scale and V-scale rows."""
            base = b_base(info)
            return base + b_k, base + b_v, base + b_ks, base + b_vs

        # ---- page DMA ----
        # A page of a scale plane in BF16 elements (the DMA's unit). The wrapper
        # keeps the stride a multiple of 16 B, so halving it first is exact and
        # spares the page DMAs a 64-bit signed division.
        scale_page = fx.Int64(sk_block >> 1)
        # A folded pair's lower-addressed plane leads its 2-D DMA: (leading
        # pointer, row stride in BF16 elements), once a CTA in SALU.
        pairs = []
        for lead, pair in zip(fold, ((key, value), (ks, vs))):
            near, far = pair[::-1] if lead == "v" else pair
            gap = fx.Int64(fx.ptrtoint(far)) - fx.Int64(fx.ptrtoint(near))
            pairs.append((near, far, gap >> 1))

        def stage_a(i, entry=None):
            """Wave w DMAs plane w of A page ``i`` (tokens past the page's
            largest bound zero-filled)."""
            if const_expr(entry is None):
                entry = lds_load(lay["alist"] + i * 8, fx.Int32, 2)
            page, top = uniform(entry[0]), uniform(entry[1])
            base = (i % nb_a) * A_SLOT
            for w, (ptr, plane, row) in enumerate(
                (
                    (key, 0, _ROW),
                    (value, A_PLANE_V, _ROW),
                    (ks, A_PLANE_KS, _SROW),
                    (vs, A_PLANE_VS, _SROW),
                )
            ):
                src = (
                    fx.Int64(page) * (PAGE * _ROW // 2)
                    if row == _ROW
                    else fx.Int64(page) * scale_page
                )
                if warp == w:
                    tdm(
                        _gptr(ptr, fx.BFloat16, 16),
                        src,
                        base + plane,
                        (PAGE * row // 2,),
                        (1,),
                        (top * (row // 2),),
                        8,
                        8,
                    )

        def stage_b(page, info):
            """This wave's DMAs of B item (page, info) into its quarter slot
            (v2's ``stage`` with the item's quarter for the wave's): per K/V
            pair of planes one 2-D DMA when folded, else two."""
            quarter = (info >> _Q_SHIFT) & (QUARTERS - 1)
            top = info & 0xFF
            for r in range_constexpr(1, rpw):
                top = fx.max(top, (info >> (8 * r)) & 0xFF)
            # The quarter's tokens to fetch (the rest zero-fill). No upper clamp:
            # an extent past the tile reads the same tile, and a min/max pair of
            # this uniform value becomes v_med3_i32 (no SALU form), which drags
            # every descriptor word through VGPRs and back with v_readfirstlane.
            valid = fx.max(fx.Int32(0), top - quarter * _TOKENS)
            base = b_base(info)
            payload = fx.Int64(page) * (PAGE * _ROW // 2) + fx.Int64(
                quarter * (_TOKENS * _ROW // 2)
            )
            scales = fx.Int64(page) * scale_page + fx.Int64(
                quarter * (_TOKENS * _SROW // 2)
            )
            for (near, far, stride), lead, plane, row, src in zip(
                pairs, fold, (_PLANE_K, _PLANE_KS), (_ROW, _SROW), (payload, scales)
            ):
                if const_expr(lead == "-"):
                    for ptr, at in ((near, plane), (far, plane + _TOKENS * row)):
                        tdm(
                            _gptr(ptr, fx.BFloat16, 16),
                            src,
                            base + at,
                            (_TOKENS * row // 2,),
                            (1,),
                            (valid * (row // 2),),
                            8,
                            8,
                        )
                else:
                    tdm(
                        _gptr(near, fx.BFloat16, 16),
                        src,
                        base + plane,
                        (2, _TOKENS * row // 2),
                        (stride, 1),
                        (None, valid * (row // 2)),
                        8,
                        8,
                    )

        def stage_b_head():
            for ahead in range_constexpr(nb_b - 1):
                if n_ai + ahead < n_items:
                    entry = lds_load(items_w + (n_ai + ahead) * 8, fx.Int32, 2)
                    stage_b(uniform(entry[0]), uniform(entry[1]))

        if const_expr(dbg != 3):
            for ahead in range_constexpr(nb_a - 1):
                if ahead < n_a:
                    stage_a(fx.Int32(ahead))
            if n_a == 0:
                stage_b_head()

        # ---- queries: this wave's rows split into FP8 terms, kept in VGPRs ----
        q_resident = [
            _query_terms(mem, q_loaded[r], half) for r in range_constexpr(rpw)
        ]
        selector = _selector_operand(lane16)
        initial = []
        for r in range_constexpr(rpw):
            initial.extend([fx.Float32(float("-inf")), fx.Float32(0.0)])
            initial.extend(
                [fx.Vector.filled(8, 0.0, fx.Float32) for _ in range_constexpr(8)]
            )

        def rebase(ref, l, acc, m):
            """Move the reference when a head's maximum ``m`` passed it by
            2**REBASE_LOG2 (v2's ``rebase_log2`` rule). ``ref`` carries the
            reference minus P_SHIFT, so P = 2**(score - ref) needs no shift;
            ``l`` is the softmax sum (P's scale cancels in the epilogue)."""
            grow = m > ref + fx.Float32(float(P_MAX_LOG2))
            moved = fx.Int32(fx.rocdl.ballot(T.i32, grow))
            if moved != fx.Int32(0):
                new_ref = grow.select(m - fx.Float32(float(P_SHIFT)), ref)
                safe = (new_ref > fx.Float32(float("-inf"))).select(
                    new_ref, fx.Float32(0.0)
                )
                corr = fx.Float32(fx.exp2(ref - safe, fastmath="afn"))
                acc = _rescale(acc, corr)
                l = l * corr
                ref = new_ref
            return ref, l, acc

        def v_tile(v_raw, b, lane_zero):
            """PV A operand of dims 16 b..: V^T dequantized by the selector WMMA."""
            sel_scale = fx.Int64(_E4M3_ONE << (8 * b)) | lane_zero
            a32 = fx.Vector.from_elements(
                [v_raw[0][0][e] for e in range_constexpr(8)]
                + [v_raw[1][0][e] for e in range_constexpr(8)],
                dtype=fx.Int32,
            )
            return _wmma_scale16_32x16(a32, selector, v_raw[2], sel_scale).to(
                fx.Float16
            )

        def tail_mask(raw, bound, quarter):
            """-inf past the row's in-page bound (its tail page: rare)."""
            lo = quarter * _TOKENS
            if (bound < lo + _TOKENS) & (bound > lo):
                raw = raw + _bound_mask(bound, quarter, half)
            return raw

        def v_private(p_v, p_vs):
            """V codes and scales of a quarter (the selector WMMA's operands)."""
            v_raw = [
                _rows_operand(lds_load, 0, p_v, p_vs, n, lane16, half)
                for n in range_constexpr(2)
            ]
            v_raw.append(lds_load(p_vs + lane * _SROW, fx.Int64, 1)[0])
            return v_raw

        def v_shared(vq, b):
            """PV A operand of dims 16 b.. from the shared V quarter at ``vq``."""
            at = vq + b * 1024 + lane * 16
            lo, hi = lds_load(at, fx.Int32, 4), lds_load(at + 512, fx.Int32, 4)
            return fx.Vector.from_elements(
                [lo[e] for e in range_constexpr(4)]
                + [hi[e] for e in range_constexpr(4)],
                dtype=fx.Int32,
            ).bitcast(fx.Float16)

        def v_quarter(idx):
            """This wave's quarter of A page ``idx``'s V planes."""
            frame = (idx % nb_a) * A_SLOT
            return v_private(
                frame + A_PLANE_V + warp * (_TOKENS * _ROW),
                frame + A_PLANE_VS + warp * (_TOKENS * _SROW),
            )

        def share_v(idx, k, v_raw=None):
            """Dequantize quarter ``warp`` of A page ``idx``'s V into the shared
            V (the operands ``v_tile`` gives each wave in the private walk)."""
            if const_expr(dbg == 1):
                # Debug: no attention math, the shared V included.
                return
            if const_expr(v_raw is None):
                v_raw = v_quarter(idx)
            lane_zero = fx.Int64((lane + k) >> 31)
            dst = lay["vbuf"] + warp * V_QUARTER + lane * 16
            for b in range_constexpr(8):
                words = v_tile(v_raw, b, lane_zero).bitcast(fx.Int32)
                for j in range_constexpr(2):
                    lds_store(
                        dst + b * 1024 + j * 512,
                        fx.Vector.from_elements(
                            [words[j * 4 + e] for e in range_constexpr(4)],
                            dtype=fx.Int32,
                        ),
                        fx.Int32,
                    )

        def run_rows(rows_state, ts, at):
            """One quarter of rows ``ts`` (all selected it): v2's [QK, maxima] |
            per row a rare rebase branch | [V operand, P, PV]. rows_state: per
            row [ref, l, acc x 8]; ``at``: the quarter's operands and bounds;
            its V source is ("private", p_v, p_vs, lane_zero) or ("lds", vq)."""
            k_tiles, v_src, bounds, quarter = at
            # Issued first: the LDS latency hides under QK.
            if const_expr(v_src[0] == "private"):
                v_raw = v_private(v_src[1], v_src[2])
            else:
                vts = [v_shared(v_src[1], b) for b in range_constexpr(V_AHEAD)]
            raws, maxes = [], []
            for t in ts:
                raw = tail_mask(_qk(q_resident[t], k_tiles), bounds[t], quarter)
                raws.append(raw)
                maxes.append(_head_max(raw, unit))
            cur, refs_p = [], []
            for i in range_constexpr(len(ts)):
                ref, l = rows_state[i * 10], rows_state[i * 10 + 1]
                acc = list(rows_state[i * 10 + 2 : i * 10 + 10])
                ref, l, acc = rebase(ref, l, acc, maxes[i])
                refs_p.append(ref)
                cur.append([ref, l, acc])
            ps, totals = [], []
            for i in range_constexpr(len(ts)):
                p, total = _probs(raws[i], refs_p[i], unit, fx.Float16)
                ps.append(p)
                totals.append(total)
            accs = [cur[i][2] for i in range_constexpr(len(ts))]
            for b in range_constexpr(8):
                if const_expr(v_src[0] == "private"):
                    vt = v_tile(v_raw, b, v_src[3])
                else:
                    # Block b + V_AHEAD loads while block b's PV runs.
                    if const_expr(b + V_AHEAD < 8):
                        vts.append(v_shared(v_src[1], b + V_AHEAD))
                    vt = vts[b]
                for i in range_constexpr(len(ts)):
                    accs[i][b] = _wmma_f16(vt, ps[i], accs[i][b])
            res = []
            for i in range_constexpr(len(ts)):
                res.extend([cur[i][0], cur[i][1] + totals[i]])
                res.extend(accs[i])
            return res

        def run_item(grp, info, k, shared):
            """The rows of this wave that reach item (info)'s quarter."""
            quarter = (info >> _Q_SHIFT) & (QUARTERS - 1)
            if const_expr(shared):
                p_k, p_v, p_ks, p_vs = planes(info)
            else:
                p_k, p_v, p_ks, p_vs = planes_b(info)
            bounds = [(info >> (8 * r)) & 0xFF for r in range_constexpr(rpw)]
            lo = quarter * _TOKENS
            k_tiles = [
                _rows_operand(lds_load, 0, p_k, p_ks, n, lane16, half)
                for n in range_constexpr(2)
            ]
            if const_expr(shared):
                v_src = ("lds", lay["vbuf"] + quarter * V_QUARTER)
            else:
                # VGPR block scales of the selector (see pa_sparse_prefill_nvfp4_v2).
                v_src = ("private", p_v, p_vs, fx.Int64((lane + k) >> 31))
            at = (k_tiles, v_src, bounds, quarter)
            if const_expr(dbg == 1):
                # Debug: the walk and its DMA only (no attention math).
                pass
            elif const_expr(rpw == 2):
                b0, b1 = bounds[0], bounds[1]
                if (b0 > lo) & (b1 > lo):
                    grp = run_rows(grp, (0, 1), at)
                if (b0 > lo) & (b1 <= lo):
                    grp = run_rows(grp[:10], (0,), at) + grp[10:]
                if (b0 <= lo) & (b1 > lo):
                    grp = grp[:10] + run_rows(grp[10:], (1,), at)
            else:
                if bounds[0] > lo:
                    grp = run_rows(grp, (0,), at)
            return grp

        # ---- the walk: this wave's A items (CTA ring), then its B items ----
        # The walk's LDS reads of its own metadata are issued a step ahead: the
        # next item's word (carried in an SGPR), an A page's ring entry and its
        # V planes before the page's barrier, a B item's staging (page, word)
        # an item early. Read when needed, each one's LDS latency stalled the
        # wave in every step (and the A page's lockstep path).
        ng = len(initial)

        def item_info(i):
            return lds_load(items_w + i * 8 + 4, fx.Int32, 1)[0]

        def plane_barrier():
            """CTA barrier that publishes the planes the waves' tensor DMAs
            wrote (each wave waited for its own first) without a fence:
            gpu.barrier() signals after a workgroup fence, s_wait_dscnt 0 (and
            loadcnt / storecnt 0), which would drain the LDS loads a step
            issues ahead of it. FlyDSL has no fence-free barrier, hence the raw
            ROCDL split barrier (x1.006 on the M3.1 chunks at the A page's
            barrier, paired A/B); the wavefront-scope fences emit no code and
            only keep LDS accesses on their side of it."""
            fx.memory_fence(syncscope="wavefront", ordering=fx.AtomicOrdering.Release)
            fx.rocdl.s_barrier_signal(-1)
            fx.rocdl.s_barrier_wait(-1)
            fx.memory_fence(syncscope="wavefront", ordering=fx.AtomicOrdering.Acquire)

        def publish_first():
            """Page 0's V planes published before the walk (page idx + 1's at
            page idx's first barrier)."""
            if n_a > 0:
                if (warp & 1) != 0:
                    if n_a > 1:
                        tdm_ops.tensor_wait(1)
                    else:
                        tdm_ops.tensor_wait(0)
                gpu.barrier()

        a_init = list(initial) + [uniform(item_info(fx.Int32(0)))]
        if const_expr(early_v and dbg != 3):
            publish_first()

        for k_i, state in range(fx.Int32(0), n_ai, fx.Int32(1), init=a_init):
            k = fx.Int32(k_i)
            st = list(state)
            info = st[ng]
            nxt = item_info(k + 1)
            if ((info >> _HEAD_SHIFT) & 1) != 0:
                idx = (info >> _IDX_SHIFT) & 0x7F
                a_entry = lds_load(lay["alist"] + (idx + (nb_a - 1)) * 8, fx.Int32, 2)
                v_early = None
                if const_expr(early_v):
                    v_early = v_quarter(idx)
                # Page idx: this wave's plane landed (only later A pages' DMAs
                # may be in flight; B's start after the last wait), the barrier
                # publishes every plane and frees page idx - 1's slot (and the
                # shared V). With early V the V and V-scale waves (1, 3) also
                # wait for page idx + 1's plane, which the barrier publishes.
                # No fence: the LDS loads in flight at it are this step's item
                # word, ring entry and page idx's V quarter (slot idx, not the
                # slot stage_a(idx + 2) overwrites); every read of page idx - 1
                # or the shared V that feeds the math completed at its WMMA,
                # and the second barrier drained the rest.
                if const_expr(dbg != 3):
                    if const_expr(early_v):
                        if (warp & 1) != 0:
                            tdm_ops.tensor_wait(0)
                        elif idx + 1 < n_a:
                            tdm_ops.tensor_wait(1)
                        else:
                            tdm_ops.tensor_wait(0)
                    elif const_expr(nb_a == 3):
                        if idx + 1 < n_a:
                            tdm_ops.tensor_wait(1)
                        else:
                            tdm_ops.tensor_wait(0)
                    else:
                        tdm_ops.tensor_wait(0)
                plane_barrier()
                if const_expr(dbg != 3):
                    if idx + (nb_a - 1) < n_a:
                        stage_a(idx + (nb_a - 1), a_entry)
                    if idx + 1 == n_a:
                        stage_b_head()
                if ((info >> _DQ_SHIFT) & 1) != 0:
                    share_v(idx, k, v_early)
                # Publishes page idx's V quarters.
                gpu.barrier()
            grp_a = run_item(st[:ng], info, k, True)
            final_a = yield grp_a + [uniform(nxt)]

        # Item n_ai's word came with the A walk; then (page, word) of the items
        # the first B step stages ahead.
        b_init = list(final_a)
        for j in range_constexpr(1, nb_b):
            e = lds_load(items_w + (n_ai + j) * 8, fx.Int32, 2)
            b_init.extend([uniform(e[0]), uniform(e[1])])
        for k_i, state in range(n_ai, n_items, fx.Int32(1), init=b_init):
            k = fx.Int32(k_i)
            st = list(state)
            # Carried: item k's word, then (page, word) of items k + 1 ..
            # k + nb_b - 1; item k + nb_b is read for the next step.
            info = st[ng]
            queue = st[ng + 1 :]
            nxt = lds_load(items_w + (k + nb_b) * 8, fx.Int32, 2)
            if const_expr(dbg != 3):
                # Item k is done when at most the later in-flight item's DMAs
                # (one a folded pair, two an unfolded one) remain.
                if const_expr(nb_b == 3):
                    if k + 1 < n_items:
                        tdm_ops.tensor_wait(b_dmas)
                    else:
                        tdm_ops.tensor_wait(0)
                else:
                    tdm_ops.tensor_wait(0)
                if k + (nb_b - 1) < n_items:
                    stage_b(queue[-2], queue[-1])
            grp_b = run_item(st[:ng], info, k, False)
            final = yield (
                grp_b + queue[1:2] + queue[2:] + [uniform(nxt[0]), uniform(nxt[1])]
            )

        # ---- epilogue: one owner a row, no merge ----
        tdm_ops.tensor_wait(0)
        out_bf16 = fx.recast_iter(fx.BFloat16, output)
        for r in range_constexpr(rpw):
            row = tile * tq + warp * rpw + r
            l_sum = final[r * 10 + 1]
            # A row without a visible key keeps O = 0: store zeros.
            inv = fx.Float32(rcp_f32((l_sum > 0.0).select(l_sum, fx.Float32(1.0))))
            if row < rows:
                seq = row * kv_heads + kvh
                for b in range_constexpr(8):
                    value_out = (
                        final[r * 10 + 2 + b]
                        * fx.Vector.from_elements([inv], dtype=fx.Float32).broadcast_to(
                            8
                        )
                    ).to(fx.BFloat16)
                    fx.ptr_store(
                        value_out,
                        fx.add_offset(
                            out_bf16,
                            fx.Int64(seq) * sq_row
                            + lane16 * sq_head
                            + b * 16
                            + half * 8,
                        ),
                    )

    @flyc.jit
    def launch(
        query: fx.Pointer,
        key: fx.Pointer,
        value: fx.Pointer,
        ks: fx.Pointer,
        vs: fx.Pointer,
        table: fx.Pointer,
        context: fx.Pointer,
        output: fx.Pointer,
        rows: fx.Int32,
        kv_heads: fx.Int32,
        table_stride: fx.Int32,
        sk_block: fx.Int32,
        sq_row: fx.Int32,
        sq_head: fx.Int32,
        pages: fx.Int32,
        quantized_output: fx.Pointer,
        output_scale: fx.Pointer,
        quant_rows: fx.Int32,
        stream: fx.Stream,
    ):
        kernel(
            query,
            key,
            value,
            ks,
            vs,
            table,
            context,
            output,
            rows,
            kv_heads,
            table_stride,
            sk_block,
            sq_row,
            sq_head,
            pages,
            quantized_output,
            output_scale,
            quant_rows,
            value_attrs=(
                {"rocdl.waves_per_eu": int(waves_per_eu)} if waves_per_eu else None
            ),
        ).launch(
            grid=((rows + tq - 1) // tq, kv_heads, 1),
            block=(THREADS, 1, 1),
            stream=stream,
        )

    compiled = {"launch": launch, "kernel": kernel, "lds": lay}
    _CACHE[key] = compiled
    return compiled
