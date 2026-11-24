# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2025-2026 FlyDSL Project Contributors

"""gfx1250 NVFP4 sparse-prefill attention that reuses K/V across query tokens.

MiniMax-M3.1 sparse attention prefill: 16 query heads per KV head, head_dim 128,
NVFP4 K/V pages of 128 tokens, 16 selected pages a token. ``pa_decode``'s NVFP4
wave tile serves every prefill token as its own length-1 sequence, so every
token gathers its own 16 pages (18 KB each) and widens its own V: at the 14K-token
prefill that is 4.2 GB a layer of L2 traffic and 229K page dequantizations.

Here one CTA takes ``TQ`` consecutive rows (tokens) of one KV head. Its rows'
compacted page tables (the selector's emission: full pages in selection order,
the partial tail page last, ``context`` the rows' compacted lengths) are merged
into a union of distinct pages in LDS, together with each (union page, row) in-page
bound (128 for a full page, the tail's visible count, 0 when the row did not select
the page). The page loop then walks the union once:

* each wave DMAs its 32 tokens of the page (K, V payload and scale rows) into a
  wave-private ring, exactly as the decode wave tile does; tokens at or past the
  largest bound of the page arrive zero-filled;
* V is dequantized and transposed once per page (the scaled WMMA against the
  1.0 selector), K's raw e2m1 rows feed QK straight from LDS;
* every row that selected the page (a wave-uniform branch: rows that did not
  keep their state bitwise) runs the decode tile's per-page arithmetic: QK with
  its query split into FP8 terms, the in-page causal bound as the -inf start
  value of the QK accumulator, online softmax, BF16 PV.

After the loop the four waves merge each row through LDS and store the BF16
rows. A row's pages are accumulated in the union's order (first occurrence over
the tile's rows), not in its own table order, so the result matches the decode
tile to rounding (online softmax order), not bit for bit.

Inputs follow ATOM's flattened view (``sparse_attn_nvfp4._flydsl_attention``):
sequence ``s = row * kv_heads + kv_head`` has query ``[s, 16, 128]``, table row
``s`` of page ids of the ``[pages * kv_heads, 1, 128, ...]`` cache view, and
compacted length ``context[s]``.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, range_constexpr
from flydsl.expr.rocdl import tdm_ops
from flydsl.expr.typing import ReductionOp, T

from .pa_decode.op_gemm import Gfx1250Gemm
from .pa_decode.op_mxfp8 import E8M0_ONE, MxFp8Groups
from .pa_decode.op_lds import Gfx1250Memory, _gptr
from .pa_decode.pipeline_nvfp4 import (
    _E4M3_ONE,
    _MERGE_OUT,
    _PLANE_K,
    _PLANE_KS,
    _PLANE_V,
    _PLANE_VS,
    _ROW,
    _SROW,
    _TOKENS,
    WAVE_SLOT_BYTES,
    _e8m0_pow2,
    _f32_bits,
    _pow2,
    _wmma_scale16,
)
from .pa_decode.traits import LOG2E
from .utils import rcp_f32

WAVE = 32
NWARP = 4
THREADS = WAVE * NWARP
GROUP = 16  # query heads per KV head: the WMMA's 16 rows
HEAD_DIM = 128
PAGE = 128
TOPK = 16
# Query rows a CTA: the union build gives one (row, slot) entry to a thread.
TILE_ROWS = (2, 3, 4, 6, 8)
# Query split of a BF16 query (pipeline_nvfp4._query): two FP8 terms.
QUERY_TERMS = 2
Q_LANE_BYTES = 80


class _Lds(Gfx1250Memory):
    """``Gfx1250Memory``'s LDS accessors and tensor DMA over a raw LDS base
    (BF16-element DMA addressing, as the decode wave tile uses)."""

    class _T:
        kv_dtype = fx.BFloat16

    def __init__(self, base):  # noqa: D107 -- no PaDecodeContext here
        self.base = base
        self.t = self._T


def _bound_stride(tq):
    """Bytes of one union page's (row) bound row: whole dwords."""
    return (tq + 3) // 4 * 4


def _layout(tq, nb):
    """Byte offsets of the CTA's LDS regions."""
    ring = nb * NWARP * WAVE_SLOT_BYTES
    merge = NWARP * (_MERGE_OUT + 16 * 8)
    entries = tq * TOPK
    keys = max(ring, merge)
    uidx = keys + 4 * entries
    upage = uidx + 4 * entries
    ubound = upage + 4 * entries
    counts = ubound + (entries * _bound_stride(tq) + 15) // 16 * 16
    # Per (row, term, lane): the FP8 query operand (16 dwords) and its i64 block
    # scales (2 dwords), padded to 80 B so the operand reads stay 16 B aligned.
    qterms = counts + 16
    total = qterms + tq * QUERY_TERMS * WAVE * Q_LANE_BYTES
    return dict(
        ring=ring,
        merge=merge,
        keys=keys,
        uidx=uidx,
        upage=upage,
        ubound=ubound,
        counts=counts,
        qterms=qterms,
        total=(total + 15) // 16 * 16,
    )


_CACHE = {}


def kernel_name(tq, nb, q_regs, row_group, skip_rescale=False, fma_exp=False, quant_mx=False):
    name = f"pa_sparse_prefill_nvfp4_qbf16_d128_bs128_gqa16_top16_tq{tq}"
    if quant_mx:
        name += "_mxfp8only"
    if skip_rescale:
        name += "_sr"
    if fma_exp:
        name += "_fe"
    if nb != 3:
        name += f"_ring{nb}"
    if not q_regs:
        name += "_qlds"
    if row_group != 1:
        name += f"_rg{row_group}"
    return name


def compile_pa_sparse_prefill_nvfp4(
    *,
    tile_rows=4,
    softmax_scale,
    nvfp4_buffers=3,
    q_regs=True,
    row_group=1,
    skip_rescale=False,
    fma_exp=False,
    quant_mx=False,
):
    """Compile (cached) the reuse kernel for ``tile_rows`` query rows a CTA.

    ``q_regs``: every row's query terms stay in VGPRs through the page loop
    (else each row's are reread from LDS for every page it selected: fewer
    VGPRs, 4.6 KB of LDS reads a wave per (row, page)). ``row_group`` R: rows run
    in groups of R behind one wave-uniform branch (taken when any row of the
    group selected the page); inside a group every row runs, a row that did not
    select the page against an all -inf bias, which leaves its state bitwise
    unchanged, so the grouping changes speed only, never the result.
    ``skip_rescale``: skip the accumulator rescale when no row maximum of the
    wave grew (bitwise alike). ``fma_exp``: P = exp2(fma(s, c, -m)) without the
    rounded s * c (the maxima are unchanged; P differs by that rounding) and a
    pairwise row sum. ``quant_mx``: write o_proj's MXFP8 input instead of the
    BF16 rows (``op_mxfp8``: E4M3 values of the BF16 output and one E8M0 scale per
    32, bit for bit ``quantize_fp8`` of it), tokens past ``rows`` up to
    ``quant_rows`` (a padded token bucket; flattened row ``token * kv_heads +
    kv_head``) as FP8 zeros with unit scales.

    Returns ``{"launch", "kernel"}``; ``launch(query, key, value, ks, vs, table,
    context, output, rows, kv_heads, table_stride, scale_page_stride, sq_row,
    sq_head, pages, quantized_output, output_scale, quant_rows, stream)`` runs grid
    ``(ceil(max(rows, quant_rows) / tile_rows), kv_heads)``.
    """
    tq, nb, rg = int(tile_rows), int(nvfp4_buffers), int(row_group)
    q_regs = bool(q_regs)
    if tq not in TILE_ROWS:
        raise ValueError(f"tile_rows must be one of {TILE_ROWS}")
    if nb not in (2, 3, 4):
        raise ValueError("nvfp4_buffers must be 2, 3 or 4")
    if rg < 1 or tq % rg:
        raise ValueError("row_group must divide tile_rows")
    scale = float(softmax_scale)
    skip_rescale, fma_exp, quant_mx = bool(skip_rescale), bool(fma_exp), bool(quant_mx)
    key = (tq, nb, scale, q_regs, rg, skip_rescale, fma_exp, quant_mx)
    if key in _CACHE:
        return _CACHE[key]
    lay = _layout(tq, nb)
    entries = tq * TOPK
    ub = _bound_stride(tq)
    assert lay["total"] <= 327680

    @fx.struct
    class SharedStorage:
        buf: fx.Array[fx.Int32, lay["total"] // 4, 16]

    unit = scale * LOG2E

    def query_block(mem, values):
        """``Gfx1250Nvfp4WavePipeline._query_block`` of a live row."""
        block = values.to(fx.Float32)
        amax = fx.math.absf(block).reduce(ReductionOp.MAX)
        exponent = ((_f32_bits(amax) >> 23) & 0xFF) - 127
        residual = block
        out = []
        for i in range_constexpr(QUERY_TERMS):
            k = fx.max(fx.min(exponent - (7 + 4 * i), fx.Int32(120)), fx.Int32(-120))
            scaled = residual * fx.Vector.from_elements(
                [_pow2(-k)], dtype=fx.Float32
            ).broadcast_to(16)
            words = []
            backs = []
            for h in range_constexpr(2):
                pair = mem.fp8(
                    fx.Vector.from_elements(
                        [scaled[h * 8 + e] for e in range_constexpr(8)],
                        dtype=fx.Float32,
                    )
                )
                words.extend([pair[0], pair[1]])
                if const_expr(i + 1 < QUERY_TERMS):
                    backs.append(
                        fx.Vector(
                            fx.rocdl.cvt_scale_pk8_f32_fp8(
                                T.vec(8, T.f32),
                                fx.as_ir_value(
                                    fx.Vector.from_elements(
                                        [pair[0], pair[1]], dtype=fx.Int32
                                    )
                                ),
                                fx.as_ir_value(fx.Int32(0x7F7F7F7F)),
                                0,
                            )
                        )
                    )
            out.append((words, _e8m0_pow2(k)))
            if const_expr(i + 1 < QUERY_TERMS):
                quantized = fx.Vector.from_elements(
                    [backs[0][e] for e in range_constexpr(8)]
                    + [backs[1][e] for e in range_constexpr(8)],
                    dtype=fx.Float32,
                )
                residual = residual - quantized * fx.Vector.from_elements(
                    [_pow2(k)], dtype=fx.Float32
                ).broadcast_to(16)
        return out

    def query_scales(half, scale_words):
        """``Gfx1250Nvfp4WavePipeline._query_scales``."""
        result = []
        lower = half == 0
        for mine in scale_words:
            other = fx.Int32(mine.shuffle_xor(16, WAVE))
            even = lower.select(mine, other)
            odd = lower.select(other, mine)
            lo = (
                (even & 0xFF)
                | ((odd & 0xFF) << 8)
                | ((even & 0xFF00) << 8)
                | ((odd & 0xFF00) << 16)
            )
            hi = (
                ((even >> 16) & 0xFF)
                | (((odd >> 16) & 0xFF) << 8)
                | (((even >> 24) & 0xFF) << 16)
                | (((odd >> 24) & 0xFF) << 24)
            )
            result.append(
                fx.Vector.from_elements([lo, hi], dtype=fx.Int32).bitcast(fx.Int64)[0]
            )
        return result

    def query_terms(mem, loaded, half):
        """Per term (operand, i64 scales) of one row (``_query``)."""
        operands = [[] for _ in range_constexpr(QUERY_TERMS)]
        scale_words = [fx.Int32(0) for _ in range_constexpr(QUERY_TERMS)]
        for j in range_constexpr(4):
            for i, (words, code) in enumerate(query_block(mem, loaded[j])):
                operands[i].extend(words)
                scale_words[i] = scale_words[i] | (code << (8 * j))
        scales = query_scales(half, scale_words)
        return [
            (fx.Vector.from_elements(operands[i], dtype=fx.Int32), scales[i])
            for i in range_constexpr(QUERY_TERMS)
        ]

    def rows_operand(load, base, plane, n, lane16, half):
        """``Gfx1250Nvfp4WavePipeline._rows``: e2m1 rows (lane16 = token) of token
        tile ``n`` and the token's eight E4M3 block scales."""
        token = n * 16 + lane16
        row = base + plane + token * _ROW + half * 16
        words = []
        for j in range_constexpr(2):
            frag = load(row + j * 32, fx.Int32, 4)
            words.extend([frag[i] for i in range_constexpr(4)])
        scale_plane = _PLANE_KS if plane == _PLANE_K else _PLANE_VS
        scale = load(base + scale_plane + token * _SROW, fx.Int64, 1)[0]
        return fx.Vector.from_elements(words, dtype=fx.Int32), scale

    def selector_operand(lane16):
        one = fx.Int32(2) << ((lane16 % 8) * 4)
        even = (lane16 < 8).select(one, fx.Int32(0))
        odd = (lane16 >= 8).select(one, fx.Int32(0))
        return fx.Vector.from_elements([even, odd] * 4, dtype=fx.Int32)

    def bound_mask(bound, warp, half):
        """0 for this lane's in-page tokens below ``bound``, -inf at or past it."""
        out = []
        for i in range_constexpr(16):
            n, r = i // 8, i % 8
            token = warp * _TOKENS + n * 16 + half * 8 + r
            out.append((token < bound).select(fx.Float32(0.0), fx.Float32(float("-inf"))))
        return fx.Vector.from_elements(out, dtype=fx.Float32)

    def row_softmax(m, l, q, k_tiles, bias):
        """One row's online-softmax step over this wave's 32 tokens of the page
        (the decode wave tile's arithmetic): (new max, rescale, P operand, sum)."""
        raw = []
        for n in range_constexpr(2):
            k_rows, k_scale = k_tiles[n]
            scores = fx.Vector.from_elements(
                [bias[n * 8 + r] for r in range_constexpr(8)], dtype=fx.Float32
            )
            for operand, qscale in q:
                scores = _wmma_scale16(
                    k_rows, operand, scores, k_scale, qscale, "fp4_e2m1", "fp8_e4m3", "e8"
                )
            raw.extend([scores[r] for r in range_constexpr(8)])
        if const_expr(fma_exp):
            # max(s) * c == max(s * c) (rounding is monotone, c > 0); P takes the
            # unrounded product: exp2(fma(s, c, -m)).
            tile_max = fx.Vector.from_elements(raw, dtype=fx.Float32).reduce(ReductionOp.MAX)
            tile_max = fx.maxnumf(tile_max, tile_max.shuffle_xor(16, WAVE))
            tile_max = tile_max * fx.Float32(unit)
        else:
            scores = fx.Vector.from_elements(
                [x * fx.Float32(unit) for x in raw], dtype=fx.Float32
            )
            tile_max = scores.reduce(ReductionOp.MAX)
            tile_max = fx.maxnumf(tile_max, tile_max.shuffle_xor(16, WAVE))
        new_max = fx.maxnumf(m, tile_max)
        safe_max = (new_max > fx.Float32(float("-inf"))).select(new_max, fx.Float32(0.0))
        corr = fx.Float32(fx.exp2(m - safe_max, fastmath="afn"))
        if const_expr(fma_exp):
            probs = fx.Vector(
                fx.exp2(
                    fx.math.fma(
                        fx.Vector.from_elements(raw, dtype=fx.Float32),
                        fx.Vector.filled(16, unit, fx.Float32),
                        fx.Vector.from_elements([-safe_max], dtype=fx.Float32).broadcast_to(16),
                    ),
                    fastmath="afn",
                )
            )
        else:
            probs = fx.Vector(
                fx.exp2(
                    scores
                    - fx.Vector.from_elements([safe_max], dtype=fx.Float32).broadcast_to(16),
                    fastmath="afn",
                )
            )
        if const_expr(fma_exp):
            # Pairwise (packed) instead of a 16-add chain.
            part = probs
            width = 16
            while width > 2:
                width //= 2
                part = fx.Vector.from_elements(
                    [part[i] for i in range_constexpr(width)], dtype=fx.Float32
                ) + fx.Vector.from_elements(
                    [part[width + i] for i in range_constexpr(width)], dtype=fx.Float32
                )
            tile_sum = part[0] + part[1]
        else:
            tile_sum = probs.reduce(ReductionOp.ADD)
        tile_sum = tile_sum + tile_sum.shuffle_xor(16, WAVE)
        p_operand = probs.to(fx.BFloat16)
        l_new = l * corr + tile_sum
        return new_max, corr, p_operand, l_new

    def rescale(acc, corr):
        scaled = fx.Vector.from_elements(
            [acc[b][e] for b in range_constexpr(8) for e in range_constexpr(8)],
            dtype=fx.Float32,
        ) * fx.Vector.from_elements([corr], dtype=fx.Float32).broadcast_to(64)
        return [
            fx.Vector.from_elements(
                [scaled[b * 8 + e] for e in range_constexpr(8)], dtype=fx.Float32
            )
            for b in range_constexpr(8)
        ]

    def row_pv(acc, p_operand, v_ops):
        return [
            Gfx1250Gemm.mma(v_ops[b], p_operand, acc[b], 32, True)
            for b in range_constexpr(8)
        ]

    @flyc.kernel(name=kernel_name(tq, nb, q_regs, rg, skip_rescale, fma_exp, quant_mx), known_block_size=(THREADS, 1, 1))
    def pa_sparse_prefill_kernel(
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

        # Plain closures, not ``mem.`` calls: FlyDSL's rewriter carries a local
        # object whose method a runtime branch calls as that branch's state.
        def lds_load(offset, dtype, n=1):
            return mem.load(offset, dtype, n)

        def lds_store(offset, value, dtype):
            mem.store(offset, value, dtype)
        table_i32 = fx.recast_iter(fx.Int32, table)
        context_i32 = fx.recast_iter(fx.Int32, context)

        # ---- the union of the tile's page tables -------------------------
        # Entry e = (row e // 16, slot e % 16); one entry a thread.
        e_row = tid // TOPK
        e_slot = tid % TOPK
        g_row = tile * tq + e_row
        e_live = (tid < entries) & (g_row < rows)
        e_seq = e_live.select(g_row * kv_heads + kvh, fx.Int32(0))
        e_ctx = fx.Int32(context_i32[e_seq])
        e_page = fx.Int32(table_i32[e_seq * table_stride + e_slot])
        e_bound = fx.max(fx.Int32(0), fx.min(fx.Int32(PAGE), e_ctx - e_slot * PAGE))
        # A page id outside the cache (an unwritten padding row's table) selects
        # nothing: no DMA ever leaves the cache.
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
        # The (union page, row) bounds start at zero: rows without the page.
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
        # First occurrence of this entry's page over the tile's entries.
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
        n_union = fx.Int32(
            fx.rocdl.readfirstlane(
                T.i32, counts[0] + counts[1] + counts[2] + counts[3]
            )
        )
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

        # ---- page DMA ----------------------------------------------------
        def bounds_of(u):
            """The tile rows' in-page bounds of union page ``u`` (wave-uniform)."""
            words = lds_load(lay["ubound"] + u * ub, fx.Int32, ub // 4)
            out = []
            for t in range_constexpr(tq):
                word = fx.Int32(fx.rocdl.readfirstlane(T.i32, words[t // 4]))
                out.append((word >> (8 * (t % 4))) & 0xFF)
            return out

        def slot_base(slot):
            return (slot * NWARP + warp) * WAVE_SLOT_BYTES

        def stage(u, slot):
            """DMA this wave's 32 tokens of union page ``u`` into ``slot``; tokens
            at or past every row's bound arrive zero-filled."""
            page = fx.Int32(
                fx.rocdl.readfirstlane(T.i32, lds_load(lay["upage"] + u * 4, fx.Int32, 1)[0])
            )
            bounds = bounds_of(u)
            top = bounds[0]
            for t in range_constexpr(1, tq):
                top = fx.max(top, bounds[t])
            valid = fx.max(fx.Int32(0), fx.min(fx.Int32(_TOKENS), top - warp * _TOKENS))
            base = slot_base(slot)
            payload = fx.Int64(page) * (PAGE * _ROW // 2) + fx.Int64(
                warp * (_TOKENS * _ROW // 2)
            )
            scales = fx.Int64(page) * fx.Int64(sk_block) // 2 + fx.Int64(
                warp * (_TOKENS * _SROW // 2)
            )
            for ptr, plane in ((key, _PLANE_K), (value, _PLANE_V)):
                mem.tdm_copy(
                    _gptr(ptr, fx.BFloat16, 16),
                    payload,
                    base + plane,
                    (_TOKENS * _ROW // 2,),
                    (1,),
                    (valid * (_ROW // 2),),
                    8,
                    8,
                )
            for ptr, plane in ((ks, _PLANE_KS), (vs, _PLANE_VS)):
                mem.tdm_copy(
                    _gptr(ptr, fx.BFloat16, 16),
                    scales,
                    base + plane,
                    (_TOKENS * _SROW // 2,),
                    (1,),
                    (valid * (_SROW // 2),),
                    8,
                    8,
                )

        if n_union > 0:
            stage(fx.Int32(0), fx.Int32(0))
        for ahead in range_constexpr(1, nb - 1):
            if ahead < n_union:
                stage(fx.Int32(ahead), fx.Int32(ahead))

        # ---- queries: split once a CTA into LDS ------------------------------
        # Wave w splits rows w, w + 4, ... (every wave reads the same operands).
        query_bf16 = fx.recast_iter(fx.BFloat16, query)
        for t0 in range_constexpr(0, tq, NWARP):
            t_mine = t0 + warp
            row = tile * tq + t_mine
            seq = ((row < rows) & (t_mine < tq)).select(row * kv_heads + kvh, fx.Int32(0))
            qbase = fx.Int64(seq) * sq_row + lane16 * sq_head
            loaded = [
                fx.Vector(
                    fx.ptr_load(
                        fx.add_offset(query_bf16, qbase + j * 32 + half * 16),
                        result_type=fx.Vector.make_type(16, fx.BFloat16),
                    )
                )
                for j in range_constexpr(4)
            ]
            terms = query_terms(mem, loaded, half)
            if t_mine < tq:
                for i in range_constexpr(QUERY_TERMS):
                    operand, qscale = terms[i]
                    at = lay["qterms"] + ((t_mine * QUERY_TERMS + i) * WAVE + lane) * Q_LANE_BYTES
                    lds_store(at, operand, fx.Int32)
                    lds_store(
                        at + 64,
                        fx.Vector.from_elements([qscale], dtype=fx.Int64),
                        fx.Int64,
                    )
        gpu.barrier()

        def q_row(t):
            """Row ``t``'s query terms (operand, i64 scales) from LDS."""
            out = []
            for i in range_constexpr(QUERY_TERMS):
                at = lay["qterms"] + ((t * QUERY_TERMS + i) * WAVE + lane) * Q_LANE_BYTES
                parts = [lds_load(at + 16 * c, fx.Int32, 4) for c in range_constexpr(4)]
                words = fx.Vector.from_elements(
                    [parts[c][e] for c in range_constexpr(4) for e in range_constexpr(4)],
                    dtype=fx.Int32,
                )
                out.append((words, lds_load(at + 64, fx.Int64, 1)[0]))
            return out

        selector = selector_operand(lane16)
        zeros = fx.Vector.filled(8, 0.0, fx.Float32)
        zeros16 = fx.Vector.filled(16, 0.0, fx.Float32)
        initial = []
        for t in range_constexpr(tq):
            initial.extend([fx.Float32(float("-inf")), fx.Float32(0.0)])
            initial.extend([fx.Vector.filled(8, 0.0, fx.Float32) for _ in range_constexpr(8)])
        lo = warp * _TOKENS
        hi = lo + _TOKENS
        q_resident = [q_row(t) for t in range_constexpr(tq)] if q_regs else None
        neg16 = fx.Vector.filled(16, float("-inf"), fx.Float32)

        def row_bias(bound):
            """The QK start value of a row on this wave's 32 tokens: 0 when all
            are visible, -inf past the row's in-page bound (all -inf when the row
            did not select the page)."""
            bias = zeros16
            if bound < hi:
                bias = neg16
                if bound > lo:
                    bias = bound_mask(bound, warp, half)
            return bias

        # ---- the union walk -------------------------------------------------
        for u_i, state in range(fx.Int32(0), n_union, fx.Int32(1), init=initial):
            u = fx.Int32(u_i)
            slot = u % nb
            if const_expr(nb == 2):
                tdm_ops.tensor_wait(0)
            elif const_expr(nb == 3):
                if u + 1 < n_union:
                    tdm_ops.tensor_wait(4)
                else:
                    tdm_ops.tensor_wait(0)
            else:
                if u + 2 < n_union:
                    tdm_ops.tensor_wait(8)
                else:
                    if u + 1 < n_union:
                        tdm_ops.tensor_wait(4)
                    else:
                        tdm_ops.tensor_wait(0)
            if u + (nb - 1) < n_union:
                stage(u + (nb - 1), (u + (nb - 1)) % nb)
            base = slot_base(slot)
            bounds = bounds_of(u)
            k_tiles = [rows_operand(lds_load, base, _PLANE_K, n, lane16, half) for n in range_constexpr(2)]
            v_raw = [rows_operand(lds_load, base, _PLANE_V, n, lane16, half) for n in range_constexpr(2)]
            # VGPR block scales of the selector (pipeline_nvfp4: a uniform i64 is
            # read wrongly as an inline constant or SGPR pair); OR a lane-varying
            # zero the compiler cannot fold. Built per page from the loop index:
            # hoisted out of the loop, the i64 pairs (which the WMMA reads from
            # the low 256 VGPRs) were spilled to scratch past 256 VGPRs.
            lane_zero = fx.Int64((lane + u) >> 31)
            selector_scales = [
                fx.Int64(_E4M3_ONE << (8 * b)) | lane_zero for b in range_constexpr(8)
            ]
            v_ops = []
            for b in range_constexpr(8):
                pair = [
                    _wmma_scale16(
                        v_raw[n][0],
                        selector,
                        zeros,
                        v_raw[n][1],
                        selector_scales[b],
                        "fp4_e2m1",
                        "fp4_e2m1",
                    )
                    for n in range_constexpr(2)
                ]
                v_ops.append(
                    fx.Vector.from_elements(
                        [pair[0][e] for e in range_constexpr(8)]
                        + [pair[1][e] for e in range_constexpr(8)],
                        dtype=fx.Float32,
                    ).to(fx.BFloat16)
                )
            next_state = []
            for g in range_constexpr(tq // rg):
                group = list(state[g * rg * 10 : (g + 1) * rg * 10])
                top = bounds[g * rg]
                for t in range_constexpr(1, rg):
                    top = fx.max(top, bounds[g * rg + t])
                if top > lo:
                    updated = []
                    for i in range_constexpr(rg):
                        t = g * rg + i
                        m_old = group[i * 10]
                        m_t, corr, p_operand, l_t = row_softmax(
                            m_old,
                            group[i * 10 + 1],
                            q_resident[t] if q_regs else q_row(t),
                            k_tiles,
                            row_bias(bounds[t]),
                        )
                        acc_t = group[i * 10 + 2 : i * 10 + 10]
                        if const_expr(skip_rescale):
                            # The rescale is exactly 1 unless a row maximum of
                            # this wave grew (or 0 on a still-empty row, whose
                            # accumulator is 0): skip it then, bitwise alike.
                            grew = fx.Int32(fx.rocdl.ballot(T.i32, m_t != m_old))
                            if grew != fx.Int32(0):
                                acc_t = rescale(acc_t, corr)
                        else:
                            acc_t = rescale(acc_t, corr)
                        acc_t = row_pv(acc_t, p_operand, v_ops)
                        updated.extend([m_t, l_t])
                        updated.extend(acc_t)
                    group = updated
                next_state.extend(group)
            final = yield next_state

        # ---- merge the waves' partials row by row, store BF16 rows ---------
        tdm_ops.tensor_wait(0)
        out_bf16 = fx.recast_iter(fx.BFloat16, output)
        groups = MxFp8Groups(quantized_output, output_scale)

        def mx_store(values, offsets):
            # A lane pair (row, half 0 / 1) holds one 32-element group.
            groups.store(values, offsets, (16,), WAVE)

        def pad_row(row):
            """A padded bucket row: FP8 zeros and unit E8M0 scales."""
            per = GROUP * HEAD_DIM // THREADS  # 16 bytes a thread
            target = fx.recast_iter(
                fx.PointerType.get(fx.Int32.ir_type, fx.AddressSpace.Global, 4),
                fx.add_offset(
                    fx.recast_iter(fx.Int8, quantized_output),
                    fx.Int64(row) * (GROUP * HEAD_DIM) + tid * per,
                ),
            )
            fx.ptr_store(fx.Vector.filled(per // 4, 0, fx.Int32), target)
            scales = fx.recast_iter(fx.Int8, output_scale)
            if tid < GROUP * HEAD_DIM // 32:
                scales[fx.Int64(row) * (GROUP * HEAD_DIM // 32) + tid] = fx.Int32(E8M0_ONE).to(fx.Int8)
        stats = NWARP * _MERGE_OUT
        for t in range_constexpr(tq):
            gpu.barrier()
            for b in range_constexpr(8):
                lds_store(
                    warp * _MERGE_OUT + (b * 32 + lane) * 32,
                    final[t * 10 + 2 + b],
                    fx.Float32,
                )
            if half == 0:
                lds_store(
                    stats + (warp * 16 + lane16) * 8,
                    fx.Vector.from_elements(
                        [final[t * 10], final[t * 10 + 1]], dtype=fx.Float32
                    ),
                    fx.Float32,
                )
            gpu.barrier()
            wave_stats = [
                lds_load(stats + (w * 16 + lane16) * 8, fx.Float32, 2)
                for w in range_constexpr(NWARP)
            ]
            row_max = wave_stats[0][0]
            for w in range_constexpr(1, NWARP):
                row_max = fx.maxnumf(row_max, wave_stats[w][0])
            safe_max = (row_max > fx.Float32(float("-inf"))).select(
                row_max, fx.Float32(0.0)
            )
            weights = [
                fx.Float32(fx.exp2(wave_stats[w][0] - safe_max, fastmath="afn"))
                for w in range_constexpr(NWARP)
            ]
            denom = fx.Float32(0.0)
            for w in range_constexpr(NWARP):
                denom = denom + weights[w] * wave_stats[w][1]
            inv = fx.Float32(rcp_f32((denom > 0.0).select(denom, fx.Float32(1.0))))
            row = tile * tq + t
            if row < rows:
                seq = row * kv_heads + kvh
                values, offsets = [], []
                for vh in range_constexpr(2):
                    # MXFP8: wave w holds whole 32-dim groups (dims 32 w + 16 vh).
                    b = warp * 2 + vh if quant_mx else vh * 4 + warp
                    out = fx.Vector.filled(8, 0.0, fx.Float32)
                    for w in range_constexpr(NWARP):
                        out = out + lds_load(
                            w * _MERGE_OUT + (b * 32 + lane) * 32, fx.Float32, 8
                        ) * fx.Vector.from_elements(
                            [weights[w]], dtype=fx.Float32
                        ).broadcast_to(8)
                    value_out = (
                        out * fx.Vector.from_elements([inv], dtype=fx.Float32).broadcast_to(8)
                    ).to(fx.BFloat16)
                    d = b * 16 + half * 8
                    if const_expr(quant_mx):
                        values.append(value_out)
                        offsets.append(fx.Int64(seq) * (GROUP * HEAD_DIM) + lane16 * HEAD_DIM + d)
                    else:
                        offset = fx.Int64(seq) * sq_row + lane16 * sq_head + d
                        fx.ptr_store(value_out, fx.add_offset(out_bf16, offset))
                if const_expr(quant_mx):
                    mx_store(values, offsets)
            elif const_expr(quant_mx):
                if row < quant_rows:
                    pad_row(row * kv_heads + kvh)

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
        pa_sparse_prefill_kernel(
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
        ).launch(
            grid=((fx.max(rows, quant_rows) + tq - 1) // tq, kv_heads, 1),
            block=(THREADS, 1, 1),
            stream=stream,
        )

    compiled = {"launch": launch, "kernel": pa_sparse_prefill_kernel, "lds": lay}
    _CACHE[key] = compiled
    return compiled
