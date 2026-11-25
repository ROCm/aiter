# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2025-2026 FlyDSL Project Contributors

"""gfx1250 NVFP4 sparse-prefill attention that reuses K/V across query tokens.

MiniMax-M3.1 block-sparse attention: 16 query heads per KV head, head_dim 128,
NVFP4 K/V pages of 128 tokens, 16 selected pages a token. ``pa_decode``'s NVFP4
wave tile serves every prefill token as its own sequence and gathers its own 16
pages. Here a CTA takes two consecutive rows (tokens) of one KV head, merges
their compacted page tables into a union of distinct pages in LDS (with each
(page, row) in-page bound: 128 for a full page, the tail's visible count, 0 when
the row did not select it) and walks the union once:

* wave w DMAs tokens [32 w, 32 w + 32) of the page (K, V payload and scale rows)
  into a wave-private ring three pages deep, as the decode wave tile does;
* the rows run in one basic block when both selected the page ``[QK a, QK b,
  maxima] | rare rebase branches | [V dequant, P, PV a, PV b]`` and alone when
  one did. V is dequantized and transposed by the 32x16x128 e2m1 WMMA against a
  selector; the rows' query terms sit in VGPRs.

After the walk the four waves merge each row through LDS and store BF16 rows (or
o_proj's MXFP8 input). Numerics as the decode tile: QK in FP32 over the exact
NVFP4 K and the BF16 query split into two FP8 terms, BF16 P and V operands of PV,
FP32 accumulation, the running maximum as softmax reference. A row accumulates
its pages in the union's order, so the result equals the decode tile's to
rounding, not bit for bit.

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
from .pa_decode.op_lds import Gfx1250Memory, _gptr
from .pa_decode.op_mxfp8 import E8M0_ONE, MxFp8Groups
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
    _attr,
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
TQ = 2  # rows a CTA
RING = 3  # pages in flight a wave
WAVES_PER_EU = 3
QUERY_TERMS = 2  # the BF16 query split into FP8 terms (pipeline_nvfp4._query)
Q_LANE_BYTES = 80
ENTRIES = TQ * TOPK  # (row, slot) entries of the union build, one a thread
BOUND_STRIDE = 4  # bytes of a union page's row bounds: whole dwords

_CACHE = {}


class _Lds(Gfx1250Memory):
    """``Gfx1250Memory``'s LDS accessors and tensor DMA over a raw LDS base
    (BF16-element DMA addressing, as the decode wave tile uses)."""

    class _T:
        kv_dtype = fx.BFloat16

    def __init__(self, base):
        self.base = base
        self.t = self._T


def _layout():
    """Byte offsets of the CTA's LDS regions."""
    ring = RING * NWARP * WAVE_SLOT_BYTES
    merge = NWARP * (_MERGE_OUT + 16 * 8)
    keys = max(ring, merge)
    uidx = keys + 4 * ENTRIES
    upage = uidx + 4 * ENTRIES
    ubound = upage + 4 * ENTRIES
    counts = ubound + (ENTRIES * BOUND_STRIDE + 15) // 16 * 16
    # Per (row, term, lane): the FP8 query operand (16 dwords) and its i64 block
    # scales (2 dwords), padded to 80 B so the operand reads stay 16 B aligned.
    qterms = counts + 16
    total = qterms + TQ * QUERY_TERMS * WAVE * Q_LANE_BYTES
    return {
        "keys": keys,
        "uidx": uidx,
        "upage": upage,
        "ubound": ubound,
        "counts": counts,
        "qterms": qterms,
        "total": (total + 15) // 16 * 16,
    }


def _wmma_scale16_32x16(a, b, scale_a, scale_b):
    """32x16x128 e2m1 x e2m1 WMMA, E4M3 scales per 16 (raw ROCDL: no FlyDSL atom
    takes E4M3 block scales)."""
    return fx.Vector(
        fx.rocdl.wmma_scale16_f32_32x16x128_f4(
            fx.Vector.make_type(16, fx.Float32),
            fx.as_ir_value(a),
            fx.as_ir_value(b),
            fx.as_ir_value(fx.Vector.filled(16, 0.0, fx.Float32)),
            fx.as_ir_value(scale_a),
            fx.as_ir_value(scale_b),
            modC=_attr("#rocdl<wmma_c_modifier none>"),
            scaleAType=_attr("#rocdl<wmma_matrix_scale row0>"),
            fmtScaleA=_attr("#rocdl<wmma_matrix_scale_format e4m3>"),
            scaleBType=_attr("#rocdl<wmma_matrix_scale row0>"),
            fmtScaleB=_attr("#rocdl<wmma_matrix_scale_format e4m3>"),
            reuseA=False,
            reuseB=False,
        ).result
    )


def compile_pa_sparse_prefill_nvfp4(*, softmax_scale, quant_mx=False):
    """Compile (cached) the reuse kernel. ``quant_mx``: write o_proj's MXFP8 input
    (``op_mxfp8``'s group-32 quant of the BF16 rows) instead of BF16 rows."""
    scale = float(softmax_scale)
    key = (scale, bool(quant_mx))
    if key in _CACHE:
        return _CACHE[key]
    lay = _layout()
    assert ENTRIES <= THREADS and lay["total"] <= 327680
    # A name, not lay["total"]: linters read a string in an annotation as a
    # forward reference.
    lds_words = lay["total"] // 4

    @fx.struct
    class SharedStorage:
        buf: fx.Array[fx.Int32, lds_words, 16]

    unit = scale * LOG2E

    def query_block(mem, values):
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

    def rows_operand(load, base, plane, scale_plane, n, lane16, half):
        token = n * 16 + lane16
        row = base + plane + token * _ROW + half * 16
        words = []
        for j in range_constexpr(2):
            frag = load(row + j * 32, fx.Int32, 4)
            words.extend([frag[i] for i in range_constexpr(4)])
        sc = load(base + scale_plane + token * _SROW, fx.Int64, 1)[0]
        return fx.Vector.from_elements(words, dtype=fx.Int32), sc

    def selector_operand(lane16):
        one = fx.Int32(2) << ((lane16 % 8) * 4)
        even = (lane16 < 8).select(one, fx.Int32(0))
        odd = (lane16 >= 8).select(one, fx.Int32(0))
        return fx.Vector.from_elements([even, odd] * 4, dtype=fx.Int32)

    def bound_mask(bound, warp, half):
        out = []
        for i in range_constexpr(16):
            n, r = i // 8, i % 8
            token = warp * _TOKENS + n * 16 + half * 8 + r
            out.append(
                (token < bound).select(fx.Float32(0.0), fx.Float32(float("-inf")))
            )
        return fx.Vector.from_elements(out, dtype=fx.Float32)

    def qk(q, k_tiles):
        """Raw scores (16 a lane: tokens 8 half + r of both 16-token tiles)."""
        raw = []
        for n in range_constexpr(2):
            k_rows, k_scale = k_tiles[n]
            s = fx.Vector.filled(8, 0.0, fx.Float32)
            for operand, qscale in q:
                s = _wmma_scale16(
                    k_rows, operand, s, k_scale, qscale, "fp4_e2m1", "fp8_e4m3", "e8"
                )
            raw.extend([s[r] for r in range_constexpr(8)])
        return fx.Vector.from_elements(raw, dtype=fx.Float32)

    def head_max(raw):
        """This lane's head maximum over the wave's 32 tokens, in log2 units."""
        m = raw.reduce(ReductionOp.MAX)
        m = fx.maxnumf(m, m.shuffle_xor(16, WAVE))
        return m * fx.Float32(unit)

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

    def probs(raw, ref):
        """(BF16 P operand, its lane sum over both halves) relative to ``ref``."""
        safe = (ref > fx.Float32(float("-inf"))).select(ref, fx.Float32(0.0))
        p = fx.Vector(
            fx.exp2(
                fx.math.fma(
                    raw,
                    fx.Vector.filled(16, unit, fx.Float32),
                    fx.Vector.from_elements([-safe], dtype=fx.Float32).broadcast_to(16),
                ),
                fastmath="afn",
            )
        )
        part = p
        width = 16
        while width > 2:
            width //= 2
            part = fx.Vector.from_elements(
                [part[i] for i in range_constexpr(width)], dtype=fx.Float32
            ) + fx.Vector.from_elements(
                [part[width + i] for i in range_constexpr(width)], dtype=fx.Float32
            )
        total = part[0] + part[1]
        total = total + total.shuffle_xor(16, WAVE)
        return p.to(fx.BFloat16), total

    name = "pa_sparse_prefill_nvfp4_d128_bs128_gqa16_top16_tq2"
    if quant_mx:
        name += "_mxfp8only"

    @flyc.kernel(name=name, known_block_size=(THREADS, 1, 1))
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

        table_i32 = fx.recast_iter(fx.Int32, table)
        context_i32 = fx.recast_iter(fx.Int32, context)

        # The query rows' global loads issue first: their latency overlaps the
        # page-table loads and the union build. Wave t < TQ loads row t.
        query_bf16 = fx.recast_iter(fx.BFloat16, query)
        row = tile * TQ + warp
        seq = ((row < rows) & (warp < TQ)).select(row * kv_heads + kvh, fx.Int32(0))
        qbase = fx.Int64(seq) * sq_row + lane16 * sq_head
        q_loaded = [
            fx.Vector(
                fx.ptr_load(
                    fx.add_offset(query_bf16, qbase + j * 32 + half * 16),
                    result_type=fx.Vector.make_type(16, fx.BFloat16),
                )
            )
            for j in range_constexpr(4)
        ]

        # ---- the union of the tile's page tables ----
        e_row = tid // TOPK
        e_slot = tid % TOPK
        g_row = tile * TQ + e_row
        e_live = (tid < ENTRIES) & (g_row < rows)
        e_seq = e_live.select(g_row * kv_heads + kvh, fx.Int32(0))
        e_ctx = fx.Int32(context_i32[e_seq])
        e_page = fx.Int32(table_i32[e_seq * table_stride + e_slot])
        e_bound = fx.max(fx.Int32(0), fx.min(fx.Int32(PAGE), e_ctx - e_slot * PAGE))
        e_live = e_live & (e_page >= 0) & (e_page < pages)
        e_bound = e_live.select(e_bound, fx.Int32(0))
        e_valid = e_bound > 0
        e_key = e_valid.select(e_page, fx.Int32(-1))
        if tid < ENTRIES:
            lds_store(
                lay["keys"] + tid * 4,
                fx.Vector.from_elements([e_key], dtype=fx.Int32),
                fx.Int32,
            )
        zero_words = ENTRIES * BOUND_STRIDE // 4
        for i in range_constexpr((zero_words + THREADS - 1) // THREADS):
            w = i * THREADS + tid
            if w < zero_words:
                lds_store(
                    lay["ubound"] + w * 4,
                    fx.Vector.from_elements([fx.Int32(0)], dtype=fx.Int32),
                    fx.Int32,
                )
        gpu.barrier()
        first = fx.Int32(ENTRIES)
        for c in range_constexpr(ENTRIES // 4):
            keys4 = lds_load(lay["keys"] + c * 16, fx.Int32, 4)
            for e in range_constexpr(4):
                hit = (keys4[e] == e_key) & (first == fx.Int32(ENTRIES))
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
            fx.rocdl.readfirstlane(T.i32, counts[0] + counts[1] + counts[2] + counts[3])
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
                lay["ubound"] + u_entry * BOUND_STRIDE + e_row,
                fx.Vector.from_elements([e_bound.to(fx.Int8)], dtype=fx.Int8),
                fx.Int8,
            )
        gpu.barrier()

        # ---- page DMA ----
        # The union's page ids and bound words in VGPRs (lane i: union entry i):
        # the page loop reads them with v_readlane, no LDS round trip on its
        # critical path.
        page_reg = lds_load(lay["upage"] + lane * 4, fx.Int32, 1)[0]
        bound_reg = lds_load(lay["ubound"] + lane * BOUND_STRIDE, fx.Int32, 1)[0]

        def bounds_of(u):
            word = fx.Int32(fx.rocdl.readlane(T.i32, bound_reg, u % WAVE))
            return [(word >> (8 * t)) & 0xFF for t in range_constexpr(TQ)]

        def slot_base(slot):
            return (slot * NWARP + warp) * WAVE_SLOT_BYTES

        def stage(u, slot):
            page = fx.Int32(fx.rocdl.readlane(T.i32, page_reg, u % WAVE))
            bounds = bounds_of(u)
            top = fx.max(bounds[0], bounds[1])
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

        for ahead in range_constexpr(RING - 1):
            if ahead < n_union:
                stage(fx.Int32(ahead), fx.Int32(ahead))

        # ---- queries: split once a CTA into LDS, then into VGPRs ----
        terms = query_terms(mem, q_loaded, half)
        if warp < TQ:
            for i in range_constexpr(QUERY_TERMS):
                operand, qscale = terms[i]
                at = (
                    lay["qterms"]
                    + ((warp * QUERY_TERMS + i) * WAVE + lane) * Q_LANE_BYTES
                )
                lds_store(at, operand, fx.Int32)
                lds_store(
                    at + 64,
                    fx.Vector.from_elements([qscale], dtype=fx.Int64),
                    fx.Int64,
                )
        gpu.barrier()

        def q_row(t):
            out = []
            for i in range_constexpr(QUERY_TERMS):
                at = (
                    lay["qterms"] + ((t * QUERY_TERMS + i) * WAVE + lane) * Q_LANE_BYTES
                )
                parts = [lds_load(at + 16 * c, fx.Int32, 4) for c in range_constexpr(4)]
                words = fx.Vector.from_elements(
                    [
                        parts[c][e]
                        for c in range_constexpr(4)
                        for e in range_constexpr(4)
                    ],
                    dtype=fx.Int32,
                )
                out.append((words, lds_load(at + 64, fx.Int64, 1)[0]))
            return out

        selector = selector_operand(lane16)
        initial = []
        for t in range_constexpr(TQ):
            initial.extend([fx.Float32(float("-inf")), fx.Float32(0.0)])
            initial.extend(
                [fx.Vector.filled(8, 0.0, fx.Float32) for _ in range_constexpr(8)]
            )
        lo = warp * _TOKENS
        hi = lo + _TOKENS
        q_resident = [q_row(t) for t in range_constexpr(TQ)]

        def tail_mask(raw, bound):
            """-inf past the row's in-page bound (its tail page: rare)."""
            if (bound < hi) & (bound > lo):
                raw = raw + bound_mask(bound, warp, half)
            return raw

        def rebase(ref, l, acc, m):
            """Move the reference to the head maximum ``m`` where it grew."""
            grow = m > ref
            moved = fx.Int32(fx.rocdl.ballot(T.i32, grow))
            if moved != fx.Int32(0):
                new_ref = grow.select(m, ref)
                safe = (new_ref > fx.Float32(float("-inf"))).select(
                    new_ref, fx.Float32(0.0)
                )
                corr = fx.Float32(fx.exp2(ref - safe, fastmath="afn"))
                acc = rescale(acc, corr)
                l = l * corr
                ref = new_ref
            return ref, l, acc

        # ---- the union walk ----
        for u_i, state in range(fx.Int32(0), n_union, fx.Int32(1), init=initial):
            u = fx.Int32(u_i)
            slot = u % RING
            # Four DMAs a page: page u is done when at most the next page's remain.
            if u + 1 < n_union:
                tdm_ops.tensor_wait(4)
            else:
                tdm_ops.tensor_wait(0)
            if u + (RING - 1) < n_union:
                stage(u + (RING - 1), (u + (RING - 1)) % RING)
            base = slot_base(slot)
            bounds = bounds_of(u)
            k_tiles = [
                rows_operand(lds_load, base, _PLANE_K, _PLANE_KS, n, lane16, half)
                for n in range_constexpr(2)
            ]
            # VGPR block scales of the selector, built per page.
            lane_zero = fx.Int64((lane + u) >> 31)

            def v_tile(v_raw, b, lane_zero=lane_zero):
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

            def run_rows(rows_state, ts, base=base, k_tiles=k_tiles, bounds=bounds):
                """One step of rows ``ts`` (all selected the page for this wave):
                [QK, maxima] | per row a rare rebase branch | [V dequant, P, PV].
                rows_state: per row [ref, l, acc x 8]."""
                # Issued first: the LDS latency hides under QK.
                v_raw = [
                    rows_operand(lds_load, base, _PLANE_V, _PLANE_VS, n, lane16, half)
                    for n in range_constexpr(2)
                ]
                v_raw.append(lds_load(base + _PLANE_VS + lane * _SROW, fx.Int64, 1)[0])
                raws, maxes = [], []
                for t in ts:
                    raw = tail_mask(qk(q_resident[t], k_tiles), bounds[t])
                    raws.append(raw)
                    maxes.append(head_max(raw))
                cur = []
                for i, t in enumerate(ts):
                    ref, l = rows_state[i * 10], rows_state[i * 10 + 1]
                    acc = list(rows_state[i * 10 + 2 : i * 10 + 10])
                    ref, l, acc = rebase(ref, l, acc, maxes[i])
                    cur.append([ref, l, acc])
                ps, totals = [], []
                for i in range_constexpr(len(ts)):
                    p, total = probs(raws[i], cur[i][0])
                    ps.append(p)
                    totals.append(total)
                accs = [cur[i][2] for i in range_constexpr(len(ts))]
                for b in range_constexpr(8):
                    vt = v_tile(v_raw, b)
                    for i in range_constexpr(len(ts)):
                        accs[i][b] = Gfx1250Gemm.mma(vt, ps[i], accs[i][b], 32, True)
                res = []
                for i in range_constexpr(len(ts)):
                    res.extend([cur[i][0], cur[i][1] + totals[i]])
                    res.extend(accs[i])
                return res

            grp = list(state)
            b0, b1 = bounds[0], bounds[1]
            if (b0 > lo) & (b1 > lo):
                grp = run_rows(grp, (0, 1))
            if (b0 > lo) & (b1 <= lo):
                grp = run_rows(grp[:10], (0,)) + grp[10:]
            if (b0 <= lo) & (b1 > lo):
                grp = grp[:10] + run_rows(grp[10:], (1,))
            final = yield grp

        # ---- merge the waves' partials row by row, store ----
        tdm_ops.tensor_wait(0)
        out_bf16 = fx.recast_iter(fx.BFloat16, output)
        groups = MxFp8Groups(quantized_output, output_scale)

        def mx_store(values, offsets):
            groups.store(values, offsets, (16,), WAVE)

        def pad_row(row):
            per = GROUP * HEAD_DIM // THREADS
            target = fx.recast_iter(
                fx.PointerType.get(fx.Int32.ir_type, fx.AddressSpace.Global, 4),
                fx.add_offset(
                    fx.recast_iter(fx.Int8, quantized_output),
                    fx.Int64(row) * (GROUP * HEAD_DIM) + tid * per,
                ),
            )
            fx.ptr_store(fx.Vector.filled(per // 4, 0, fx.Int32), target)
            sc = fx.recast_iter(fx.Int8, output_scale)
            if tid < GROUP * HEAD_DIM // 32:
                sc[fx.Int64(row) * (GROUP * HEAD_DIM // 32) + tid] = fx.Int32(
                    E8M0_ONE
                ).to(fx.Int8)

        stats = NWARP * _MERGE_OUT
        for t in range_constexpr(TQ):
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
            # Each wave's (O, l) is relative to its own reference (-inf: zeros, weight 0).
            weights = [
                fx.Float32(fx.exp2(wave_stats[w][0] - safe_max, fastmath="afn"))
                for w in range_constexpr(NWARP)
            ]
            denom = fx.Float32(0.0)
            for w in range_constexpr(NWARP):
                denom = denom + weights[w] * wave_stats[w][1]
            inv = fx.Float32(rcp_f32((denom > 0.0).select(denom, fx.Float32(1.0))))
            row = tile * TQ + t
            if row < rows:
                seq = row * kv_heads + kvh
                values, offsets = [], []
                for vh in range_constexpr(2):
                    b = warp * 2 + vh if quant_mx else vh * 4 + warp
                    o = fx.Vector.filled(8, 0.0, fx.Float32)
                    for w in range_constexpr(NWARP):
                        o = o + lds_load(
                            w * _MERGE_OUT + (b * 32 + lane) * 32, fx.Float32, 8
                        ) * fx.Vector.from_elements(
                            [weights[w]], dtype=fx.Float32
                        ).broadcast_to(
                            8
                        )
                    value_out = (
                        o
                        * fx.Vector.from_elements([inv], dtype=fx.Float32).broadcast_to(
                            8
                        )
                    ).to(fx.BFloat16)
                    d = b * 16 + half * 8
                    if const_expr(quant_mx):
                        values.append(value_out)
                        offsets.append(
                            fx.Int64(seq) * (GROUP * HEAD_DIM) + lane16 * HEAD_DIM + d
                        )
                    else:
                        fx.ptr_store(
                            value_out,
                            fx.add_offset(
                                out_bf16, fx.Int64(seq) * sq_row + lane16 * sq_head + d
                            ),
                        )
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
            value_attrs={"rocdl.waves_per_eu": WAVES_PER_EU},
        ).launch(
            grid=((fx.max(rows, quant_rows) + TQ - 1) // TQ, kv_heads, 1),
            block=(THREADS, 1, 1),
            stream=stream,
        )

    compiled = {"launch": launch, "kernel": kernel}
    _CACHE[key] = compiled
    return compiled
