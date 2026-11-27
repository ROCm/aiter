# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2025-2026 FlyDSL Project Contributors
# ruff: noqa: B023
# (B023: the kernel body's closures over loop variables are traced and called inside the
# iteration that defines them, so each sees that iteration's values.)

"""gfx1250 NVFP4 sparse-prefill attention with K/V reuse, rows batched in pairs.

Same work split, union walk, DMA ring and epilogue as ``pa_sparse_prefill_nvfp4``
(a CTA takes ``tile_rows`` consecutive rows of one KV head and walks the union
of their pages; wave w owns tokens [32 w, 32 w + 32) of every page). What
changes is the page step of a wave:

* the tile's rows go in pairs; a pair that both selected the page runs as one
  basic block ``[QK a, QK b, maxima] | rare rebase branches | [V dequant, P, PV
  a, PV b]``, a page only one row of the pair selected runs that row alone;
* V is dequantized and transposed by the 32x16x128 e2m1 WMMA against the
  selector (one a 16-dim block for the wave's 32 tokens, half the instructions
  of two 16x16x128 ones; the output's VGPR order is the PV A operand's);
* the union's page ids and in-page bounds sit in VGPRs (``v_readlane`` a page),
  the query terms of a two-row tile in VGPRs (``q_regs``), 3 waves a SIMD.

Numerics: QK as the decode tile (FP32 over the exact NVFP4 K and the BF16 query
split into two FP8 terms), BF16 P and V operands of PV, FP32 accumulation. With
the default ``rebase_log2=0`` the softmax reference is the running maximum, as
the decode tile's (FP32-reference error equal to it, 99.6% of the outputs
bitwise equal); ``rebase_log2 > 0`` re-bases lazily (P <= 2**rebase_log2) and
measured slightly worse (rel. L2 +6%).

MI455X, one TP4 rank, us/layer, real MiniMax-M3.1 top-k (ATOM_M3_DUMP_PREFILL_TOPK,
code / English prose / AgentX prompts), paged decode today -> v1 (tq 4) -> v2:
14336 new on 99968 cached, layer 0: 309 -> 278 -> 251; layer 30: 307 -> 325 -> 271;
layer 59: 312 -> 276 -> 248; prose 16000 on 98304, layer 30: 346 -> 291 -> 271;
AgentX 32768 on 65280, layer 30: 700 -> 753 -> 634. Decomposition (layer 30,
v2): prologue + epilogue alone 40, the union walk's DMA alone 119, the math
without DMA 191, all 271: the DMA and the math do not overlap.

``dbg`` (benchmark decomposition only): 1 DMA walk without math, 2 prologue and
epilogue only, 3 math without DMA, 4/5 fewer DMA planes, 6 every DMA the first
union page.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, range_constexpr
from flydsl.expr.rocdl import tdm_ops
from flydsl.expr.typing import ReductionOp, T

from .kernels_common import create_llvm_ptr
from .pa_decode.op_gemm import Gfx1250Gemm
from .pa_decode.op_lds import _gptr
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
from .pa_sparse_prefill_nvfp4 import _bound_stride, _layout, _Lds
from .utils import rcp_f32

WAVE = 32
NWARP = 4
THREADS = WAVE * NWARP
GROUP = 16
HEAD_DIM = 128
PAGE = 128
TOPK = 16
TILE_ROWS = (2, 4, 6, 8)
QUERY_TERMS = 2
Q_LANE_BYTES = 80

# A page step's DMA extent (tokens, <= 128) sits in the union's bound words at
# this bit: there it reads as the codes' extent in BF16 elements (tokens x 32)
# at bit 16 of the tensor DMA descriptor word that holds it.
_EXT_SHIFT = 21
_EXT_MASK = 0xFF << _EXT_SHIFT

_CACHE = {}


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


def _query_block(mem, values):
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


def _query_scales(half, scale_words):
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


def _query_terms(mem, loaded, half):
    operands = [[] for _ in range_constexpr(QUERY_TERMS)]
    scale_words = [fx.Int32(0) for _ in range_constexpr(QUERY_TERMS)]
    for j in range_constexpr(4):
        for i, (words, code) in enumerate(_query_block(mem, loaded[j])):
            operands[i].extend(words)
            scale_words[i] = scale_words[i] | (code << (8 * j))
    scales = _query_scales(half, scale_words)
    return [
        (fx.Vector.from_elements(operands[i], dtype=fx.Int32), scales[i])
        for i in range_constexpr(QUERY_TERMS)
    ]


def _rows_operand(load, base, plane, scale_plane, n, lane16, half):
    token = n * 16 + lane16
    row = base + plane + token * _ROW + half * 16
    words = []
    for j in range_constexpr(2):
        frag = load(row + j * 32, fx.Int32, 4)
        words.extend([frag[i] for i in range_constexpr(4)])
    sc = load(base + scale_plane + token * _SROW, fx.Int64, 1)[0]
    return fx.Vector.from_elements(words, dtype=fx.Int32), sc


def _selector_operand(lane16):
    one = fx.Int32(2) << ((lane16 % 8) * 4)
    even = (lane16 < 8).select(one, fx.Int32(0))
    odd = (lane16 >= 8).select(one, fx.Int32(0))
    return fx.Vector.from_elements([even, odd] * 4, dtype=fx.Int32)


def _bound_mask(bound, warp, half):
    out = []
    for i in range_constexpr(16):
        n, r = i // 8, i % 8
        token = warp * _TOKENS + n * 16 + half * 8 + r
        out.append((token < bound).select(fx.Float32(0.0), fx.Float32(float("-inf"))))
    return fx.Vector.from_elements(out, dtype=fx.Float32)


def _qk(q, k_tiles):
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


def _head_max(raw, unit):
    """This lane's head maximum over the wave's 32 tokens, in log2 units."""
    m = raw.reduce(ReductionOp.MAX)
    m = fx.maxnumf(m, m.shuffle_xor(16, WAVE))
    return m * fx.Float32(unit)


def _rescale(acc, corr):
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


def _probs(raw, ref, unit, dtype=fx.BFloat16):
    """(P operand in ``dtype``, its lane sum over both halves) relative to ``ref``."""
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
    return p.to(dtype), total


def kernel_name(
    tq,
    nb,
    waves_per_eu,
    rebase_log2,
    v_hoist,
    fence,
    quant_mx,
    all_rows=False,
    v_wide=False,
    v_valu=0,
    q_regs=False,
    dbg=0,
    async_dma=False,
    cta_dma=False,
    tdm_fold="",
):
    name = f"pa_sparse_prefill_nvfp4_v2_d128_bs128_gqa16_top16_tq{tq}_rb{rebase_log2}"
    if tdm_fold:
        name += f"_tf{tdm_fold.replace('-', 'x')}"
    if cta_dma:
        name += "_cdma"
    if async_dma:
        name += "_adma"
    if dbg:
        name += f"_dbg{dbg}"
    if q_regs:
        name += "_qr"
    if v_valu:
        name += f"_vv{v_valu}"
    if v_wide:
        name += "_vw"
    if all_rows:
        name += "_all"
    if quant_mx:
        name += "_mxfp8only"
    if nb != 3:
        name += f"_ring{nb}"
    if waves_per_eu:
        name += f"_w{waves_per_eu}"
    if v_hoist:
        name += "_vh"
    if fence:
        name += f"_f{fence}"
    return name


def compile_pa_sparse_prefill_nvfp4_v2(
    *,
    tile_rows=2,
    softmax_scale,
    nvfp4_buffers=3,
    waves_per_eu=None,
    rebase_log2=0,
    v_hoist=None,
    fence=0,
    quant_mx=False,
    all_rows=False,
    v_wide=True,
    v_valu=0,
    q_regs=None,
    dbg=0,
    async_dma=False,
    cta_dma=False,
    tdm_fold="",
):
    """Compile (cached) the pair-batched reuse kernel; ``launch`` takes the same
    arguments as ``compile_pa_sparse_prefill_nvfp4``'s.

    ``rebase_log2``: a row's softmax reference moves only when a head's maximum
    passes it by 2**rebase_log2 (0: whenever it grows, the decode tile's rule).
    ``v_hoist``: dequantize V once a page before the row pairs (default for more
    than one pair) instead of inside each pair's block. ``fence`` k: a scheduling
    barrier after every k dim blocks of the V dequant + PV phase (bounds the
    dequantized V live at once; 0: none). ``waves_per_eu``: the occupancy hint (0: none).
    ``all_rows``: one basic block runs every row of the tile on a page any of
    them selected (rows that did not select it against an all -inf bias, state
    unchanged), instead of pairs that branch on their rows' selection.
    ``v_wide``: dequantize V with the 32x16x128 WMMA (one a dim block for the
    wave's 32 tokens) instead of two 16x16x128 ones (experimental layout).
    ``v_valu`` k: dequantize the first k of the 8 dim blocks with VALU instead
    (``ds_load_tr4_b64`` of the e2m1 V rows, unit-scale BF16 conversion, times
    the block scales from a transposed scale plane): moves V's dequantization
    from the matrix pipe to VALU (8: all of it). ``q_regs``: the rows' query
    terms stay in VGPRs (36 a row) instead of being reread from LDS every page.
    ``async_dma``: stage a page with per-lane ``global_load_async_to_lds`` (ten
    a page and wave; tokens past the rows' bounds read the page's token 0, finite
    data that every row masks) instead of four tensor DMAs.
    ``cta_dma``: wave w DMAs plane w (K, V, K scales, V scales) of the whole
    page, then a CTA barrier a page: one tensor DMA a page and wave instead of
    four (the waves walk the same union in lockstep anyway).
    ``tdm_fold``: stage a K/V pair of planes (the codes, then the scales) with
    one 2-D tensor DMA, its row stride the distance between the two pointers,
    instead of two. One letter a pair ("k"/"v": the lower-addressed plane, which
    leads the DMA and takes the pair's first slot plane; the wrapper reads it off
    the pointers; "-": two DMAs). True means "kk"; the tail stays zero-filled.
    "v" for the scales swaps the readers' two scale-plane offsets, which costs
    the page step more than the DMA it saves: the wrapper folds them K-first only.
    """
    tq, nb = int(tile_rows), int(nvfp4_buffers)
    # Defaults measured on real MiniMax-M3.1 picks (MI455X, 14336 new on 99968
    # cached): two rows a tile, the query terms in VGPRs, 3 waves a SIMD.
    if q_regs is None:
        q_regs = tq == 2
    if waves_per_eu is None:
        waves_per_eu = 3 if tq == 2 else 0
    if tq not in TILE_ROWS:
        raise ValueError(f"tile_rows must be one of {TILE_ROWS}")
    if nb not in (2, 3, 4):
        raise ValueError("nvfp4_buffers must be 2, 3 or 4")
    tdm_fold = "kk" if tdm_fold is True else (tdm_fold or "")
    if tdm_fold and (
        len(tdm_fold) != 2 or set(tdm_fold) - set("kv-") or tdm_fold == "--"
    ):
        raise ValueError("tdm_fold must be a bool or two of 'k'/'v'/'-'")
    if tdm_fold and (async_dma or cta_dma or dbg in (4, 5)):
        raise ValueError("tdm_fold stages every plane with tensor DMAs")
    # The per-wave tensor DMA ring always holds nb - 1 pages' DMAs (zero-extent
    # ones past the union's end): one unconditional tensor wait a page step,
    # and a page's descriptors built from operands prepared once a CTA.
    full_ring = not (async_dma or cta_dma or dbg == 3)
    pairs = tq // 2
    all_rows = bool(all_rows)
    v_hoist = (pairs > 1 and not all_rows) if v_hoist is None else bool(v_hoist)
    scale = float(softmax_scale)
    rb = int(rebase_log2)
    key = (
        tq,
        nb,
        scale,
        int(waves_per_eu),
        rb,
        v_hoist,
        int(fence),
        bool(quant_mx),
        all_rows,
        bool(v_wide),
        int(v_valu),
        bool(q_regs),
        int(dbg),
        bool(async_dma),
        bool(cta_dma),
        tdm_fold,
    )
    if key in _CACHE:
        return _CACHE[key]
    lay = dict(_layout(tq, nb))
    # Per wave: the page's V block scales transposed, [8 blocks][32 tokens] bytes.
    lay["vst"] = lay["total"]
    lay["total"] = lay["vst"] + NWARP * 256
    fence_after = [b for b in range(7) if fence and (b + 1) % int(fence) == 0]
    entries = tq * TOPK
    ub = _bound_stride(tq)
    assert entries <= THREADS and lay["total"] <= 327680
    # A folded pair's second plane lands one packed row after its first.
    assert _PLANE_V - _PLANE_K == _TOKENS * _ROW
    assert _PLANE_VS - _PLANE_KS == _TOKENS * _SROW
    # A name, not lay["total"]: linters read a string in an annotation as a
    # forward reference.
    lds_words = lay["total"] // 4

    @fx.struct
    class SharedStorage:
        buf: fx.Array[fx.Int32, lds_words, 16]

    unit = scale * LOG2E

    @flyc.kernel(
        name=kernel_name(
            tq,
            nb,
            waves_per_eu,
            rb,
            v_hoist,
            fence,
            quant_mx,
            all_rows,
            v_wide,
            v_valu,
            q_regs,
            dbg,
            async_dma,
            cta_dma,
            tdm_fold,
        ),
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

        table_i32 = fx.recast_iter(fx.Int32, table)
        context_i32 = fx.recast_iter(fx.Int32, context)

        # The query rows' global loads issue first: their latency overlaps the
        # page-table loads and the union build.
        query_bf16 = fx.recast_iter(fx.BFloat16, query)
        q_loaded = []
        for t0 in range_constexpr(0, tq, NWARP):
            t_mine = t0 + warp
            row = tile * tq + t_mine
            seq = ((row < rows) & (t_mine < tq)).select(
                row * kv_heads + kvh, fx.Int32(0)
            )
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

        # ---- the union of the tile's page tables (as pa_sparse_prefill_nvfp4) ----
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
        n_union = fx.Int32(
            fx.rocdl.readfirstlane(T.i32, counts[0] + counts[1] + counts[2] + counts[3])
        )
        if const_expr(dbg == 2):
            # Debug: prologue and epilogue only (no page walk).
            n_union = fx.Int32(0) * n_union
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

        # ---- page DMA (as pa_sparse_prefill_nvfp4) ----
        # The union's page ids and bound words in VGPRs (lane i: union entry i,
        # i + 32, ...): the page loop reads them with v_readlane, no LDS round
        # trip on its critical path.
        meta_regs = (entries + WAVE - 1) // WAVE
        page_regs = [
            lds_load(lay["upage"] + (k * WAVE + lane) * 4, fx.Int32, 1)[0]
            for k in range_constexpr(meta_regs)
        ]
        bound_regs = [
            [
                lds_load(lay["ubound"] + (k * WAVE + lane) * ub + 4 * w, fx.Int32, 1)[0]
                for w in range_constexpr(ub // 4)
            ]
            for k in range_constexpr(meta_regs)
        ]

        def meta_lane(regs, u):
            idx = u & (WAVE - 1)  # u >= 0: a mask, not a signed remainder
            v = fx.Int32(fx.rocdl.readlane(T.i32, regs[0], idx))
            for k in range_constexpr(1, meta_regs):
                v = (u >= k * WAVE).select(
                    fx.Int32(fx.rocdl.readlane(T.i32, regs[k], idx)), v
                )
            return v

        def bounds_of(u):
            out = []
            for t in range_constexpr(tq):
                word = meta_lane(
                    [bound_regs[k][t // 4] for k in range_constexpr(meta_regs)], u
                )
                out.append((word >> (8 * (t % 4))) & 0xFF)
            return out

        def page_of(u):
            return meta_lane(page_regs, u)

        if const_expr(cta_dma):
            # Page-wide planes in a slot; this wave's 32 tokens start at warp * 32.
            P_K = warp * (_TOKENS * _ROW)
            P_V = PAGE * _ROW + warp * (_TOKENS * _ROW)
            P_KS = 2 * PAGE * _ROW + warp * (_TOKENS * _SROW)
            P_VS = 2 * PAGE * _ROW + PAGE * _SROW + warp * (_TOKENS * _SROW)
        else:
            P_K, P_V, P_KS, P_VS = _PLANE_K, _PLANE_V, _PLANE_KS, _PLANE_VS

        # A page of a scale plane in BF16 elements (the TDM's unit). The wrapper
        # keeps the stride a multiple of 16 B, so halving it first is exact and
        # spares the page step a 64-bit floor division.
        scale_page = fx.Int64(sk_block >> 1)
        # The same in bytes, its low 4 bits known zero (see the ring's stage).
        scale_bytes = fx.Int64(sk_block & -16)

        if const_expr(tdm_fold):
            # A folded pair's lower-addressed plane leads its 2-D DMA and lands
            # in the pair's first slot plane: (leading pointer, row stride in
            # BF16 elements), once a CTA in SALU. The page step's DMAs then
            # build their descriptors in SALU as the 1-D ones do.
            fold = []
            for lead, pair in zip(tdm_fold, ((key, value), (ks, vs))):
                near, far = pair[::-1] if lead == "v" else pair
                gap = fx.Int64(fx.ptrtoint(far)) - fx.Int64(fx.ptrtoint(near))
                fold.append((near, gap >> 1))
            if const_expr(tdm_fold[0] == "v"):
                P_K, P_V = P_V, P_K
            if const_expr(tdm_fold[1] == "v"):
                P_KS, P_VS = P_VS, P_KS

        def slot_base(slot):
            if const_expr(cta_dma):
                return slot * (NWARP * WAVE_SLOT_BYTES)
            return (slot * NWARP + warp) * WAVE_SLOT_BYTES

        # This wave's quarter of a page plane, in BF16 elements from the page.
        quarter = {_ROW: _TOKENS * _ROW // 2, _SROW: _TOKENS * _SROW // 2}
        if const_expr(full_ring):
            # The ring's DMA operands, prepared once a CTA so that a page step
            # builds its descriptors with a few SALU and no dependent chain:
            # entries past the union's end read page 0 (an in-bounds address)
            # with no tokens, and every union entry carries this wave's DMA
            # extent (its tokens [32 w, top), top the tile's largest in-page
            # bound) at bit _EXT_SHIFT of its bound word. A page step reads it
            # with one v_readlane and an AND: its range is then known, so the
            # descriptor's extent clamp and high word fold away.
            for k in range_constexpr(meta_regs):
                live = k * WAVE + lane < n_union
                page_regs[k] = live.select(page_regs[k], fx.Int32(0))
                bound_regs[k] = [live.select(w, fx.Int32(0)) for w in bound_regs[k]]
            ext_regs = []
            for k in range_constexpr(meta_regs):
                top = fx.Int32(0)
                for t in range_constexpr(tq):
                    top = fx.max(top, (bound_regs[k][t // 4] >> (8 * (t % 4))) & 0xFF)
                ext = fx.max(fx.Int32(0), top - warp * _TOKENS) << _EXT_SHIFT
                if const_expr(meta_regs == 1):
                    # The prologue stages entries 0..nb-2 (from their bounds);
                    # a page step reads them again only past the union's end,
                    # where the lane index wraps: no tokens.
                    ext = (lane < nb - 1).select(fx.Int32(0), ext)
                if const_expr(tq % 4):
                    # The rows' bounds take the word's low tq % 4 bytes (all
                    # bounds_of reads); the extent goes in its top bits.
                    bound_regs[k][-1] = bound_regs[k][-1] | ext
                    ext_regs.append(bound_regs[k][-1])
                else:
                    ext_regs.append(ext)

        def stage_ring(u, base, step=False):
            """Stage union page u into the slot at LDS ``base`` (the per-wave
            tensor DMA ring); ``step``: a page step's (the prepared extent)."""
            page = page_of(u)
            if const_expr(dbg == 6):
                # Debug: every DMA fetches the tile's first union page (L2-hot).
                page = fx.Int32(
                    fx.rocdl.readfirstlane(
                        T.i32, lds_load(lay["upage"], fx.Int32, 1)[0]
                    )
                )
            if const_expr(step):
                word = meta_lane(ext_regs, u)
                if const_expr(meta_regs > 1):
                    # Past the union's end the lane index wraps onto live entries.
                    word = (u < n_union).select(word, fx.Int32(0))
                bits = word & _EXT_MASK
                # Extents in BF16 elements: tokens x 32 (codes), x 4 (scales).
                extent = {_ROW: bits >> 16, _SROW: bits >> 19}
            else:
                bounds = bounds_of(u)
                top = bounds[0]
                for t in range_constexpr(1, tq):
                    top = fx.max(top, bounds[t])
                # The quarter's tokens to fetch (the rest zero-fill). No upper
                # clamp: an extent past the tile reads the same tile (the copy
                # reads the tile only), and a min/max pair of this uniform value
                # becomes v_med3_i32 (no SALU form) that drags every descriptor
                # word through VGPRs and back with v_readfirstlane.
                valid = fx.max(fx.Int32(0), top - warp * _TOKENS)
                extent = {_ROW: valid * (_ROW // 2), _SROW: valid * (_SROW // 2)}
            # The copy doubles the halved byte offset back; the low bits known
            # zero fold the pair away (a page of scales is a multiple of 16 B).
            src = {
                _ROW: fx.Int64(page) * (PAGE * _ROW // 2),
                _SROW: (fx.Int64(page) * scale_bytes) >> 1,
            }

            def gptr(p, row):
                # The plane's pointer at this wave's quarter (loop-invariant).
                return fx.add_offset(_gptr(p, fx.BFloat16, 16), warp * quarter[row])

            if const_expr(tdm_fold):
                # A page's staging costs per tensor DMA it issues: one 2-D DMA
                # moves both planes of a folded pair (the same rows and bytes).
                for (ptr, far, plane, row), lead, (near, stride) in zip(
                    ((key, value, _PLANE_K, _ROW), (ks, vs, _PLANE_KS, _SROW)),
                    tdm_fold,
                    fold,
                ):
                    if const_expr(lead == "-"):
                        for p, at in ((ptr, plane), (far, plane + _TOKENS * row)):
                            mem.tdm_copy(
                                gptr(p, row),
                                src[row],
                                base + at,
                                (_TOKENS * row // 2,),
                                (1,),
                                (extent[row],),
                                8,
                                8,
                            )
                    else:
                        mem.tdm_copy(
                            gptr(near, row),
                            src[row],
                            base + plane,
                            (2, _TOKENS * row // 2),
                            (stride, 1),
                            (None, extent[row]),
                            8,
                            8,
                        )
                return
            for ptr, plane in (
                ((key, _PLANE_K),) if dbg == 5 else ((key, _PLANE_K), (value, _PLANE_V))
            ):
                mem.tdm_copy(
                    gptr(ptr, _ROW),
                    src[_ROW],
                    base + plane,
                    (_TOKENS * _ROW // 2,),
                    (1,),
                    (extent[_ROW],),
                    8,
                    8,
                )
            for ptr, plane in (() if dbg == 4 else ((ks, _PLANE_KS), (vs, _PLANE_VS))):
                mem.tdm_copy(
                    gptr(ptr, _SROW),
                    src[_SROW],
                    base + plane,
                    (_TOKENS * _SROW // 2,),
                    (1,),
                    (extent[_SROW],),
                    8,
                    8,
                )

        def stage_async(u, slot):
            page = page_of(u)
            bounds = bounds_of(u)
            top = bounds[0]
            for t in range_constexpr(1, tq):
                top = fx.max(top, bounds[t])
            valid = top - warp * _TOKENS
            base = slot_base(slot)
            for ptr, plane in ((key, _PLANE_K), (value, _PLANE_V)):
                gpage = fx.Int64(fx.ptrtoint(ptr)) + fx.Int64(page) * (PAGE * _ROW)
                for j in range_constexpr(4):
                    c = j * WAVE + lane  # 16-byte chunk of this wave's 32 x 64 B rows
                    own = fx.Int64(warp * (_TOKENS * _ROW) + c * 16)
                    src = ((c // 4) < valid).select(own, fx.Int64((c % 4) * 16))
                    mem.async_copy(gpage, src, base + plane + c * 16)
            for ptr, plane in ((ks, _PLANE_KS), (vs, _PLANE_VS)):
                gpage = fx.Int64(fx.ptrtoint(ptr)) + fx.Int64(page) * fx.Int64(sk_block)
                own = fx.Int64((warp * _TOKENS + lane) * _SROW)
                src = (lane < valid).select(own, fx.Int64(0))
                dst = create_llvm_ptr(
                    fx.Int64(fx.ptrtoint(mem.base))
                    + fx.Int64(base + plane + lane * _SROW),
                    address_space=3,
                )
                fx.rocdl.global_load_async_to_lds_b64(
                    create_llvm_ptr(gpage + src, address_space=1), dst, 0
                )

        def stage_cta(u, slot):
            page = page_of(u)
            bounds = bounds_of(u)
            top = bounds[0]
            for t in range_constexpr(1, tq):
                top = fx.max(top, bounds[t])
            base = slot * (NWARP * WAVE_SLOT_BYTES)
            for w, (ptr, plane, row) in enumerate(
                (
                    (key, 0, _ROW),
                    (value, PAGE * _ROW, _ROW),
                    (ks, 2 * PAGE * _ROW, _SROW),
                    (vs, 2 * PAGE * _ROW + PAGE * _SROW, _SROW),
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

        # async_dma and cta_dma stage with their own (dbg 3 stages nothing).
        stage = stage_cta if cta_dma else stage_async

        if const_expr(full_ring):
            # The ring always holds nb - 1 pages' DMAs, so the walk's waits need
            # no bound checks.
            for ahead in range_constexpr(nb - 1):
                stage_ring(fx.Int32(ahead), slot_base(fx.Int32(ahead)))
        elif const_expr(dbg != 3):
            if n_union > 0:
                stage(fx.Int32(0), fx.Int32(0))
            for ahead in range_constexpr(1, nb - 1):
                if ahead < n_union:
                    stage(fx.Int32(ahead), fx.Int32(ahead))

        # ---- queries: split once a CTA into LDS ----
        for t0 in range_constexpr(0, tq, NWARP):
            t_mine = t0 + warp
            loaded = q_loaded[t0 // NWARP]
            terms = _query_terms(mem, loaded, half)
            if t_mine < tq:
                for i in range_constexpr(QUERY_TERMS):
                    operand, qscale = terms[i]
                    at = (
                        lay["qterms"]
                        + ((t_mine * QUERY_TERMS + i) * WAVE + lane) * Q_LANE_BYTES
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

        selector = _selector_operand(lane16)
        zeros = fx.Vector.filled(8, 0.0, fx.Float32)
        zero_v = fx.Vector.filled(16, 0.0, fx.Float32).to(fx.BFloat16)
        initial = []
        for t in range_constexpr(tq):
            initial.extend([fx.Float32(float("-inf")), fx.Float32(0.0)])
            initial.extend(
                [fx.Vector.filled(8, 0.0, fx.Float32) for _ in range_constexpr(8)]
            )
        lo = warp * _TOKENS
        hi = lo + _TOKENS
        q_resident = [q_row(t) for t in range_constexpr(tq)] if q_regs else None
        if const_expr(full_ring):
            # The slot a page step stages (nb - 1 pages ahead), its LDS base
            # carried in an SGPR: slots nb - 1, 0, 1, ..., nb - 1, 0, ... (no
            # remainder and no slot arithmetic ahead of the step's DMAs).
            initial.append(slot_base(fx.Int32(nb - 1)))

        def tail_mask(raw, bound):
            """-inf past the row's in-page bound (its tail page: rare)."""
            if (bound < hi) & (bound > lo):
                raw = raw + _bound_mask(bound, warp, half)
            return raw

        def rebase(ref, l, acc, m):
            """Move the reference when a head's maximum ``m`` passed it by 2**rb."""
            if const_expr(rb == 0):
                grow = m > ref
            else:
                grow = m > ref + fx.Float32(float(rb))
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

        # ---- the union walk ----
        # Tensor DMAs a page and wave (``stage``): a folded pair issues one.
        page_dmas = 4 - (len(tdm_fold) - tdm_fold.count("-"))
        for u_i, state in range(fx.Int32(0), n_union, fx.Int32(1), init=initial):
            u = fx.Int32(u_i)
            slot = u % nb
            if const_expr(full_ring):
                ahead_base = state[-1]
                state = list(state)[:-1]
            if const_expr(dbg == 3):
                # Debug: the attention math on whatever the ring holds (no DMA).
                pass
            elif const_expr(cta_dma):
                # One DMA a page and wave: page u is done when at most the later
                # in-flight pages' remain; the barrier publishes every wave's plane.
                if const_expr(nb == 2):
                    tdm_ops.tensor_wait(0)
                elif const_expr(nb == 3):
                    if u + 1 < n_union:
                        tdm_ops.tensor_wait(1)
                    else:
                        tdm_ops.tensor_wait(0)
                else:
                    if u + 2 < n_union:
                        tdm_ops.tensor_wait(2)
                    else:
                        if u + 1 < n_union:
                            tdm_ops.tensor_wait(1)
                        else:
                            tdm_ops.tensor_wait(0)
                gpu.barrier()
            elif const_expr(async_dma):
                # Ten async loads a page: page u is done when at most the later
                # in-flight pages' remain.
                if const_expr(nb == 2):
                    fx.rocdl.s_wait_asynccnt(0)
                elif const_expr(nb == 3):
                    if u + 1 < n_union:
                        fx.rocdl.s_wait_asynccnt(10)
                    else:
                        fx.rocdl.s_wait_asynccnt(0)
                else:
                    if u + 2 < n_union:
                        fx.rocdl.s_wait_asynccnt(20)
                    else:
                        if u + 1 < n_union:
                            fx.rocdl.s_wait_asynccnt(10)
                        else:
                            fx.rocdl.s_wait_asynccnt(0)
            else:
                # Page u is done when only the nb - 2 later pages' DMAs remain
                # (the ring is always full: no bound checks, no branches).
                tdm_ops.tensor_wait((nb - 2) * page_dmas)
            if const_expr(full_ring):
                stage_ring(u + (nb - 1), ahead_base, True)
                # The page's DMAs issue together at the step head (spread over
                # the step they measured slower).
                fx.rocdl.sched_barrier(0)
            elif const_expr(dbg != 3):
                if u + (nb - 1) < n_union:
                    # The slot nb - 1 pages ahead, from this page's (no remainder).
                    ahead = slot + (nb - 1)
                    stage(u + (nb - 1), (ahead >= nb).select(ahead - nb, ahead))
            base = slot_base(slot)
            bounds = bounds_of(u)
            k_tiles = [
                _rows_operand(lds_load, base, P_K, P_KS, n, lane16, half)
                for n in range_constexpr(2)
            ]
            # VGPR block scales of the selector (see pa_sparse_prefill_nvfp4), built per page.
            lane_zero = fx.Int64((lane + u) >> 31)

            vst = lay["vst"] + warp * 256
            if const_expr(v_valu):
                # The page's V block scales transposed: lane = token writes its 8
                # bytes to [block][token] (one wave, no barrier; read after the
                # s_wait_dscnt of the loads).
                srow = lds_load(base + P_VS + lane * _SROW, fx.Int32, 2)
                for b in range_constexpr(8):
                    byte = (srow[b // 4] >> (8 * (b % 4))) & 0xFF
                    lds_store(
                        vst + b * 32 + lane,
                        fx.Vector.from_elements([byte.to(fx.Int8)], dtype=fx.Int8),
                        fx.Int8,
                    )
            # ds_load_tr4_b64 (probed on gfx1250): source lanes 8 g..8 g + 7 each
            # fetch 16 nibbles; output lane d gets, in dword w, nibble d % 16 of
            # the rows of group d // 16 + 2 w in lane order. Lane = token fetching
            # dims 16 b.. of its row gives lane (m, half) dims 16 b + m of tokens
            # 8 half + [0, 8) (dword 0) and 16 + 8 half + [0, 8) (dword 1): the
            # PV A operand's key order.
            tr_key = lane

            def v_tile_valu(b):
                ptr = mem.pointer(base + P_V + tr_key * _ROW + 8 * b, fx.Int32)
                nib = fx.Vector(fx.rocdl.ds_load_tr4_b64(T.vec(2, T.i32), ptr.llvm_ptr))
                vals = []
                for w in range_constexpr(2):
                    vals.append(
                        fx.Vector(
                            fx.rocdl.cvt_scale_pk8_bf16_fp4(
                                T.vec(8, T.bf16),
                                fx.as_ir_value(nib[w]),
                                fx.as_ir_value(fx.Int32(0x7F7F7F7F)),
                                0,
                            )
                        )
                    )
                scs = []
                for g in range_constexpr(2):
                    raw8 = lds_load(vst + b * 32 + g * 16 + half * 8, fx.Int32, 2)
                    scs.append(
                        fx.Vector(
                            fx.rocdl.cvt_scale_pk8_bf16_fp8(
                                T.vec(8, T.bf16),
                                fx.as_ir_value(raw8),
                                fx.as_ir_value(fx.Int32(0x7F7F7F7F)),
                                0,
                            )
                        )
                    )
                v16 = fx.Vector.from_elements(
                    [vals[0][e] for e in range_constexpr(8)]
                    + [vals[1][e] for e in range_constexpr(8)],
                    dtype=fx.BFloat16,
                )
                s16 = fx.Vector.from_elements(
                    [scs[0][e] for e in range_constexpr(8)]
                    + [scs[1][e] for e in range_constexpr(8)],
                    dtype=fx.BFloat16,
                )
                return v16 * s16

            def v_tile(v_raw, b):
                """PV A operand of dims 16 b..: V^T dequantized by the selector WMMA."""
                if const_expr(b < v_valu):
                    return v_tile_valu(b)
                sel_scale = fx.Int64(_E4M3_ONE << (8 * b)) | lane_zero
                if const_expr(v_wide):
                    a32 = fx.Vector.from_elements(
                        [v_raw[0][0][e] for e in range_constexpr(8)]
                        + [v_raw[1][0][e] for e in range_constexpr(8)],
                        dtype=fx.Int32,
                    )
                    return _wmma_scale16_32x16(a32, selector, v_raw[2], sel_scale).to(
                        fx.BFloat16
                    )
                pair = [
                    _wmma_scale16(
                        v_raw[n][0],
                        selector,
                        zeros,
                        v_raw[n][1],
                        sel_scale,
                        "fp4_e2m1",
                        "fp4_e2m1",
                    )
                    for n in range_constexpr(2)
                ]
                return fx.Vector.from_elements(
                    [pair[0][e] for e in range_constexpr(8)]
                    + [pair[1][e] for e in range_constexpr(8)],
                    dtype=fx.Float32,
                ).to(fx.BFloat16)

            v_hoisted = None
            if const_expr(v_hoist):
                top = bounds[0]
                for t in range_constexpr(1, tq):
                    top = fx.max(top, bounds[t])
                v_hoisted = [zero_v for _ in range_constexpr(8)]
                if top > lo:
                    v_raw_h = [
                        _rows_operand(lds_load, base, P_V, P_VS, n, lane16, half)
                        for n in range_constexpr(2)
                    ]
                    if const_expr(v_wide):
                        v_raw_h.append(
                            lds_load(base + P_VS + lane * _SROW, fx.Int64, 1)[0]
                        )
                    v_hoisted = [v_tile(v_raw_h, b) for b in range_constexpr(8)]

            def run_rows(rows_state, ts):
                """One step of rows ``ts`` (all selected the page for this wave):
                [QK, maxima] | per row a rare rebase branch | [V dequant, P, PV].
                rows_state: per row [ref, l, acc x 8]."""
                if const_expr(not v_hoist):
                    # Issued first: the LDS latency hides under QK.
                    v_raw = [
                        _rows_operand(lds_load, base, P_V, P_VS, n, lane16, half)
                        for n in range_constexpr(2)
                    ]
                    if const_expr(v_wide):
                        v_raw.append(
                            lds_load(base + P_VS + lane * _SROW, fx.Int64, 1)[0]
                        )
                raws, maxes = [], []
                for t in ts:
                    raw = tail_mask(
                        _qk(q_resident[t] if q_regs else q_row(t), k_tiles), bounds[t]
                    )
                    raws.append(raw)
                    maxes.append(_head_max(raw, unit))
                cur, refs_p = [], []
                for i, t in enumerate(ts):
                    ref, l = rows_state[i * 10], rows_state[i * 10 + 1]
                    acc = list(rows_state[i * 10 + 2 : i * 10 + 10])
                    if const_expr(all_rows):
                        # A row that did not select the page (on this wave) runs
                        # against a +inf reference: no rebase, P = 0, state kept.
                        sel = bounds[t] > lo
                        ref_in = sel.select(ref, fx.Float32(float("inf")))
                        ref_out, l, acc = rebase(ref_in, l, acc, maxes[i])
                        refs_p.append(ref_out)
                        ref = sel.select(ref_out, ref)
                    else:
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
                    vt = v_hoisted[b] if v_hoist else v_tile(v_raw, b)
                    for i in range_constexpr(len(ts)):
                        accs[i][b] = Gfx1250Gemm.mma(vt, ps[i], accs[i][b], 32, True)
                    if const_expr(b in fence_after):
                        fx.rocdl.sched_barrier(0)
                res = []
                for i in range_constexpr(len(ts)):
                    res.extend([cur[i][0], cur[i][1] + totals[i]])
                    res.extend(accs[i])
                return res

            next_state = []
            if const_expr(dbg in (1, 4, 5)):
                # Debug: the union walk and its DMA only (no attention math);
                # 4: payload only (no scale DMAs), 5: K payload only.
                next_state = list(state)
            elif const_expr(all_rows):
                top = bounds[0]
                for t in range_constexpr(1, tq):
                    top = fx.max(top, bounds[t])
                next_state = list(state)
                if top > lo:
                    next_state = run_rows(next_state, tuple(range(tq)))
            for g in range_constexpr(0 if (all_rows or dbg in (1, 4, 5)) else pairs):
                r0, r1 = 2 * g, 2 * g + 1
                grp = list(state[g * 20 : (g + 1) * 20])
                b0, b1 = bounds[r0], bounds[r1]
                if (b0 > lo) & (b1 > lo):
                    grp = run_rows(grp, (r0, r1))
                if (b0 > lo) & (b1 <= lo):
                    grp = run_rows(grp[:10], (r0,)) + grp[10:]
                if (b0 <= lo) & (b1 > lo):
                    grp = grp[:10] + run_rows(grp[10:], (r1,))
                next_state.extend(grp)
            if const_expr(full_ring):
                next_state.append(
                    (ahead_base >= slot_base(fx.Int32(nb - 1))).select(
                        slot_base(fx.Int32(0)), ahead_base + NWARP * WAVE_SLOT_BYTES
                    )
                )
            final = yield next_state

        # ---- merge the waves' partials row by row, store (as pa_sparse_prefill_nvfp4) ----
        if const_expr(async_dma):
            fx.rocdl.s_wait_asynccnt(0)
        else:
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
            # Each wave's (O, l) is relative to its own reference (-inf: zeros, weight 0).
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
                    pad_row(row)

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
            grid=((fx.max(rows, quant_rows) + tq - 1) // tq, kv_heads, 1),
            block=(THREADS, 1, 1),
            stream=stream,
        )

    compiled = {"launch": launch, "kernel": kernel, "lds": lay}
    _CACHE[key] = compiled
    return compiled
