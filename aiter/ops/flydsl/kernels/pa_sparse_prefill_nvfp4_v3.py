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
  and one CTA barrier a page publishes the planes and frees the oldest slot.
  Every wave passes every A page's barrier; it computes only the 32-token
  quarters its rows reach.
* phase B, the rest: each wave alone, over the 32-token quarters of its own
  pages, through a wave-private ring of v2's quarter slots (v2's four DMAs a
  quarter, no barrier).

Each wave walks one list of (page, quarter) items from LDS, A's first; the page
step on a quarter is v2's (rows paired when both selected the page, V
dequantized by the 32x16x128 selector WMMA, the query terms in VGPRs). A row
has one owner, so the epilogue scales and stores without a cross-wave merge.

Numerics are v2's per (row, page); a row's pages are accumulated in A-then-B
order, so the result matches v2 to rounding. BF16 output only: the wrapper
routes the MXFP8 output (one KV head) and small grids to v2.

``dbg`` (benchmark decomposition only): 1 the walk and its DMA without math,
2 prologue and epilogue only, 3 the math without DMA (barriers kept).
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, range_constexpr
from flydsl.expr.rocdl import tdm_ops
from flydsl.expr.typing import T

from .pa_decode.op_gemm import Gfx1250Gemm
from .pa_decode.op_lds import _gptr
from .pa_sparse_prefill_nvfp4 import _bound_stride, _Lds
from .pa_sparse_prefill_nvfp4_v2 import (
    _E4M3_ONE,
    _ROW,
    _SROW,
    _TOKENS,
    LOG2E,
    NWARP,
    PAGE,
    THREADS,
    TOPK,
    WAVE,
    WAVE_SLOT_BYTES,
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
# four quarter slots of the B rings at the same plane offsets.
A_SLOT = NWARP * WAVE_SLOT_BYTES
A_PLANE_V = PAGE * _ROW
A_PLANE_KS = 2 * PAGE * _ROW
A_PLANE_VS = A_PLANE_KS + PAGE * _SROW
# Item word: bounds of the wave's rows (8 bits each), the token quarter, A's page
# head flag, the LDS frame and quarter slot it is read from, and the A page's
# index in the A list.
_Q_SHIFT, _HEAD_SHIFT, _SLOT_SHIFT, _FRAME_SHIFT, _IDX_SHIFT = 16, 18, 19, 21, 24

_CACHE = {}


def _layout_v3(tq, nb_a, nb_b):
    """Byte offsets of the CTA's LDS regions."""
    entries = tq * TOPK
    # Frames 0 .. nb_a - 1: the A ring. Then the B rings: wave w's slot j is
    # quarter slot g % 4 of frame nb_a + g // 4, g = w * nb_b + j.
    keys = (nb_a + nb_b) * A_SLOT
    uidx = keys + 4 * entries
    upage = uidx + 4 * entries
    ubound = upage + 4 * entries
    counts = ubound + (entries * _bound_stride(tq) + 15) // 16 * 16
    alist = counts + 16
    items = alist + 8 * entries
    # A wave's items: at most one a page it does not select (an A page's
    # barrier) plus four quarters a page its rows select.
    per_wave = entries + QUARTERS * TOPK * (tq // NWARP)
    # + 16: the loop reads an item with the one it stages, past the last.
    total = items + NWARP * per_wave * 8 + 16
    return {
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


def kernel_name(tq, share_min, nb_a, nb_b, waves_per_eu, dbg):
    name = (
        f"pa_sparse_prefill_nvfp4_v3_d128_bs128_gqa16_top16_tq{tq}"
        f"_sm{share_min}_a{nb_a}b{nb_b}"
    )
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
):
    """Compile (cached) the row-split reuse kernel; ``launch`` takes the same
    arguments as ``compile_pa_sparse_prefill_nvfp4``'s (the MXFP8 ones unused).

    ``tile_rows``: 8 (two rows a wave) or 4 (one). ``share_min``: waves that must
    select a page for the CTA ring (phase A); 1 runs every page there, 5 none.
    ``nvfp4_buffers`` / ``nb_b``: depth of the CTA ring / of each wave's quarter
    ring. ``waves_per_eu``: the occupancy hint (0: none).
    """
    tq, nb_a, nb_b = int(tile_rows), int(nvfp4_buffers), int(nb_b)
    share_min, dbg = int(share_min), int(dbg)
    if tq not in TILE_ROWS:
        raise ValueError(f"tile_rows must be one of {TILE_ROWS}")
    if nb_a not in (2, 3) or nb_b not in (2, 3):
        raise ValueError("nvfp4_buffers and nb_b must be 2 or 3")
    if not 1 <= share_min <= NWARP + 1:
        raise ValueError("share_min must be 1 to 5")
    if quant_mx:
        raise ValueError("v3 stores BF16 only; the MXFP8 output runs on v2")
    scale = float(softmax_scale)
    key = (tq, nb_a, nb_b, share_min, scale, int(waves_per_eu), dbg)
    if key in _CACHE:
        return _CACHE[key]
    lay = _layout_v3(tq, nb_a, nb_b)
    rpw = tq // NWARP
    entries = tq * TOPK
    ub = _bound_stride(tq)
    per_wave = lay["per_wave"]
    assert entries <= THREADS and lay["total"] <= 327680
    # A name, not lay["total"]: linters read a string in an annotation as a
    # forward reference.
    lds_words = lay["total"] // 4

    @fx.struct
    class SharedStorage:
        buf: fx.Array[fx.Int32, lds_words, 16]

    unit = scale * LOG2E

    @flyc.kernel(
        name=kernel_name(tq, share_min, nb_a, nb_b, waves_per_eu, dbg),
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
        gpu.barrier()
        first = fx.Int32(entries)
        for c in range_constexpr(entries // 4):
            keys4 = lds_load(lay["keys"] + c * 16, fx.Int32, 4)
            for e in range_constexpr(4):
                hit = (keys4[e] == e_key) & (first == fx.Int32(entries))
                first = hit.select(fx.Int32(c * 4 + e), first)
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
        gpu.barrier()
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
        gpu.barrier()
        if e_valid:
            u_entry = lds_load(lay["uidx"] + first * 4, fx.Int32, 1)[0]
            lds_store(
                lay["ubound"] + u_entry * ub + e_row,
                fx.Vector.from_elements([e_bound.to(fx.Int8)], dtype=fx.Int8),
                fx.Int8,
            )
        gpu.barrier()

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

        def b_slot(t, pos):
            """B item ``pos``: ring slot (pos - n_ai) % nb_b of this wave."""
            g = warp * nb_b + (pos - n_ai) % nb_b
            return ((nb_a + (g >> 2)) << _FRAME_SHIFT) | ((g & 3) << _SLOT_SHIFT)

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
            """LDS offsets of an item's K, V, K-scale and V-scale rows."""
            frame = ((info >> _FRAME_SHIFT) & 7) * A_SLOT
            slot = (info >> _SLOT_SHIFT) & (QUARTERS - 1)
            p_k = frame + slot * (_TOKENS * _ROW)
            p_ks = frame + A_PLANE_KS + slot * (_TOKENS * _SROW)
            return p_k, p_k + A_PLANE_V, p_ks, p_ks + PAGE * _SROW

        # ---- page DMA ----
        def stage_a(i):
            """Wave w DMAs plane w of A page ``i`` (tokens past the page's
            largest bound zero-filled)."""
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
                    else fx.Int64(page) * fx.Int64(sk_block) // 2
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
            """This wave's four DMAs of B item (page, info) into its quarter
            slot (v2's ``stage`` with the item's quarter for the wave's)."""
            quarter = (info >> _Q_SHIFT) & (QUARTERS - 1)
            top = info & 0xFF
            for r in range_constexpr(1, rpw):
                top = fx.max(top, (info >> (8 * r)) & 0xFF)
            valid = fx.max(
                fx.Int32(0), fx.min(fx.Int32(_TOKENS), top - quarter * _TOKENS)
            )
            p_k, p_v, p_ks, p_vs = planes(info)
            payload = fx.Int64(page) * (PAGE * _ROW // 2) + fx.Int64(
                quarter * (_TOKENS * _ROW // 2)
            )
            scales = fx.Int64(page) * fx.Int64(sk_block) // 2 + fx.Int64(
                quarter * (_TOKENS * _SROW // 2)
            )
            for ptr, at in ((key, p_k), (value, p_v)):
                tdm(
                    _gptr(ptr, fx.BFloat16, 16),
                    payload,
                    at,
                    (_TOKENS * _ROW // 2,),
                    (1,),
                    (valid * (_ROW // 2),),
                    8,
                    8,
                )
            for ptr, at in ((ks, p_ks), (vs, p_vs)):
                tdm(
                    _gptr(ptr, fx.BFloat16, 16),
                    scales,
                    at,
                    (_TOKENS * _SROW // 2,),
                    (1,),
                    (valid * (_SROW // 2),),
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
            """Move the reference when a head's maximum ``m`` passed it (v2's
            rule at ``rebase_log2`` 0); ``l`` is the softmax sum."""
            grow = m > ref
            moved = fx.Int32(fx.rocdl.ballot(T.i32, grow))
            if moved != fx.Int32(0):
                new_ref = grow.select(m, ref)
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
                fx.BFloat16
            )

        def tail_mask(raw, bound, quarter):
            """-inf past the row's in-page bound (its tail page: rare)."""
            lo = quarter * _TOKENS
            if (bound < lo + _TOKENS) & (bound > lo):
                raw = raw + _bound_mask(bound, quarter, half)
            return raw

        def run_rows(rows_state, ts, at):
            """One quarter of rows ``ts`` (all selected it): v2's [QK, maxima] |
            per row a rare rebase branch | [V dequant, P, PV]. rows_state: per
            row [ref, l, acc x 8]; ``at``: the quarter's operands and bounds."""
            k_tiles, p_v, p_vs, bounds, quarter, lane_zero = at
            # Issued first: the LDS latency hides under QK.
            v_raw = [
                _rows_operand(lds_load, 0, p_v, p_vs, n, lane16, half)
                for n in range_constexpr(2)
            ]
            v_raw.append(lds_load(p_vs + lane * _SROW, fx.Int64, 1)[0])
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
                p, total = _probs(raws[i], refs_p[i], unit)
                ps.append(p)
                totals.append(total)
            accs = [cur[i][2] for i in range_constexpr(len(ts))]
            for b in range_constexpr(8):
                vt = v_tile(v_raw, b, lane_zero)
                for i in range_constexpr(len(ts)):
                    accs[i][b] = Gfx1250Gemm.mma(vt, ps[i], accs[i][b], 32, True)
            res = []
            for i in range_constexpr(len(ts)):
                res.extend([cur[i][0], cur[i][1] + totals[i]])
                res.extend(accs[i])
            return res

        # ---- the walk: this wave's A items (CTA ring), then its B items ----
        for k_i, state in range(fx.Int32(0), n_items, fx.Int32(1), init=initial):
            k = fx.Int32(k_i)
            # Item k and the item it stages (k + nb_b - 1) in one LDS read.
            entry = lds_load(items_w + k * 8, fx.Int32, 2 * nb_b)
            info = uniform(entry[1])
            quarter = (info >> _Q_SHIFT) & (QUARTERS - 1)
            idx = info >> _IDX_SHIFT
            in_a = k < n_ai
            if in_a:
                if ((info >> _HEAD_SHIFT) & 1) != 0:
                    # Page idx: this wave's plane landed (only later A pages'
                    # DMAs may be in flight; B's start after the last wait), the
                    # barrier publishes every plane and frees page idx - 1's slot.
                    if const_expr(dbg != 3):
                        if const_expr(nb_a == 3):
                            if idx + 1 < n_a:
                                tdm_ops.tensor_wait(1)
                            else:
                                tdm_ops.tensor_wait(0)
                        else:
                            tdm_ops.tensor_wait(0)
                    gpu.barrier()
                    if const_expr(dbg != 3):
                        if idx + (nb_a - 1) < n_a:
                            stage_a(idx + (nb_a - 1))
                        if idx + 1 == n_a:
                            stage_b_head()
            else:
                if const_expr(dbg != 3):
                    # Four DMAs an item: item k is done when at most the later
                    # in-flight items' remain.
                    if const_expr(nb_b == 3):
                        if k + 1 < n_items:
                            tdm_ops.tensor_wait(4)
                        else:
                            tdm_ops.tensor_wait(0)
                    else:
                        tdm_ops.tensor_wait(0)
                    if k + (nb_b - 1) < n_items:
                        stage_b(
                            uniform(entry[2 * nb_b - 2]), uniform(entry[2 * nb_b - 1])
                        )
            p_k, p_v, p_ks, p_vs = planes(info)
            bounds = [(info >> (8 * r)) & 0xFF for r in range_constexpr(rpw)]
            lo = quarter * _TOKENS
            k_tiles = [
                _rows_operand(lds_load, 0, p_k, p_ks, n, lane16, half)
                for n in range_constexpr(2)
            ]
            # VGPR block scales of the selector (see pa_sparse_prefill_nvfp4_v2).
            lane_zero = fx.Int64((lane + k) >> 31)
            at = (k_tiles, p_v, p_vs, bounds, quarter, lane_zero)
            grp = list(state)
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
            final = yield grp

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
