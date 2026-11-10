# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2025-2026 FlyDSL Project Contributors

"""Ping-pong PA decode tile kernel for gfx950 BF16 linear KV.

One CTA owns a task's full query block (``query_length * 16`` rows of one KV
head) and streams 64-token K/V tiles into an LDS ring by async DMA. Its waves
split the block into row blocks of ``M_TILES`` 16-row tiles and each tile's
tokens into token slots of 16 or 32 tokens; wave ``w`` owns row block
``w % ROW_BLOCKS`` and token slot ``w // ROW_BLOCKS``, so the waves sharing a
SIMD work on the same rows. Each wave keeps its own online softmax, P stays in
registers (the QK accumulator layout is PV's B operand), and the token slots
merge through LDS at the end into the shared reducer's partial layout.
Query lengths that do not fill the last row block pad it; padded rows are
computed but never stored.
"""

import os

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl._mlir.dialects import rocdl as mlir_rocdl
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import const_expr, gpu, range_constexpr
from flydsl.expr.typing import T

from .buffer_ops import create_buffer_resource_from_addr
from .pa_decode.op_lds import PaDecodeLds
from .pa_decode.traits import KV_COMPUTE_BLOCK, LOG2E, WAVE
from .tensor_shim import buf_base_i64, buf_scalar_load, ptr_buf_tensor
from .utils import rcp_f32


def _pin(value):
    # Tied empty asm materializes the value here; LLVM otherwise sinks the
    # softmax past the stagger barrier into the MFMA phase.
    value = fx.Vector(value)
    words = value.bitcast(fx.Int32) if value.dtype.width < 32 else value
    tied = fx.Vector(
        llvm.inline_asm(
            words.ir_value().type, [words.ir_value()], "", "=v,0", has_side_effects=True
        )
    )
    return tied.bitcast(value.dtype) if value.dtype.width < 32 else tied


TILE = 64
MAX_QUERY_LENGTH = 8
# Scores may exceed the row's reference max by 2**RESCALE_LOG2 before P is
# rescaled; BF16 P and f32 l and O hold that range.
RESCALE_LOG2 = 8.0

_CACHE = {}


def _uni(v):
    return fx.Int32(fx.rocdl.readfirstlane(T.i32, fx.Int32(v)))


def _uni64(v):
    v = fx.Int64(v)
    lo = _uni(fx.Int32(v & fx.Int64(0xFFFFFFFF)))
    hi = _uni(fx.Int32(v >> 32))
    return (fx.Int64(hi) << 32) | (fx.Int64(lo) & fx.Int64(0xFFFFFFFF))


def _pin_s32(value):
    # Keeps scalar math where it is computed instead of sunk to its use.
    return fx.Int32(
        llvm.inline_asm(
            T.i32, [fx.Int32(value).ir_value()], "", "=s,0", has_side_effects=True
        )
    )


def _pin_f32(value):
    return fx.Float32(
        llvm.inline_asm(
            T.f32, [fx.Float32(value).ir_value()], "", "=v,0", has_side_effects=True
        )
    )


def _spread(costs, slots):
    # Consecutive items to slots, cumulative cost even across slots.
    total = sum(costs)
    plan = [[] for _ in range(slots)]
    acc = 0
    for i, c in enumerate(costs):
        plan[min(slots - 1, acc * slots // total)].append(i)
        acc += c
    return plan


def _soft_plan(slots):
    # Softmax of 16 scores in QK's MFMA shadows: exps in pairs, each pair's
    # sum one pair late, packing once eight exps are in, then the pair sums'
    # tree. Weights are issue slots.
    items = []
    for k in range(8):
        items += [("e", 2 * k), ("e", 2 * k + 1)]
        if k >= 1:
            items.append(("a", k - 1))
        if k == 4:
            items.append(("p", 0))
    items += [("a", 7), ("p", 1)]
    items += [("t", (f"s{2 * i}", f"s{2 * i + 1}", f"q{i}")) for i in range(4)]
    items += [("t", ("q0", "q1", "r0")), ("t", ("q2", "q3", "r1"))]
    items += [("t", ("r0", "r1", "sum")), ("l", None)]
    return items, _spread([4 if k == "p" else 1 for k, _ in items], slots)


def _tree_sum(xs):
    if len(xs) == 1:
        return xs[0]
    half = len(xs) // 2
    return _tree_sum(xs[:half]) + _tree_sum(xs[half:])


def _aux(value):
    return ir.IntegerAttr.get(ir.IntegerType.get_signless(32), value)


def pingpong_layout(query_length):
    """(M_TILES, ROW_BLOCKS, CHUNKS): 16-row tiles per wave, row blocks, and
    16-token chunks per wave; a 64-token tile has 64 // (16 * CHUNKS) slots."""
    override = os.environ.get("PP_LAYOUT")
    if override:
        return tuple(int(x) for x in override.split(","))
    if query_length == 1:
        return 1, 1, 1
    if query_length == 2:
        return 2, 1, 1
    if query_length <= 4:
        return 2, 2, 1
    if query_length <= 8:
        return 2, 4, 2
    # Prefill blocks: each wave covers all 64 tokens of a tile for 32 rows.
    return 2, 8, 4


def pingpong_supported(
    *,
    architecture,
    kv_dtype,
    query_dtype,
    linear_kv,
    head_dim,
    value_dim,
    query_group_size,
    query_length,
    block_size,
    per_token_kv,
    sliding_window,
):
    value_dim = head_dim if value_dim is None else value_dim
    return (
        architecture == "gfx950"
        and kv_dtype == "bf16"
        and query_dtype == "bf16"
        and linear_kv
        and not per_token_kv
        and sliding_window == 0
        and query_group_size == 16
        and 1 <= query_length <= MAX_QUERY_LENGTH
        and head_dim % 32 == 0
        and head_dim <= 256
        and value_dim % 16 == 0
        and block_size % TILE == 0
    )


def compile_pa_decode_pingpong(
    *,
    head_dim,
    value_dim,
    query_group_size,
    query_length,
    block_size,
    softmax_scale,
    stages=None,
    pair_ctas=0,
    varlen=False,
    window=0,
    sink_dtype=None,
    seq_query_length=0,
):
    """``pair_ctas`` > 0 gives each CTA two adjacent work slots.

    ``window`` > 0 (keys a query sees, its own included) builds a launch with
    one CTA per sequence and KV head and no plan: it reads each sequence's
    context from an int32 array in place of the work info and writes
    normalized output, folding in ``sink_dtype`` ("f32", "bf16", "f16" or
    None) per-head sink logits. Uniform window launches with
    ``seq_query_length`` (a multiple of ``query_length``) queries per sequence
    run each sequence as that many over ``query_length`` CTAs.

    ``varlen`` reads an int32 [num_seqs, 4] query map of (first query row,
    block-table row, valid queries, unused) per work-info sequence; queries at
    or past the valid count repeat the last one and the reducer drops them.
    Without it sequence s owns query rows [s * QLEN, (s + 1) * QLEN).
    """
    stages = stages or int(os.environ.get("PP_STAGES", "0"))
    prio = int(os.environ.get("PP_PRIO", "0"))
    pair_ctas = int(os.environ.get("PP_PAIR", pair_ctas))
    pair_tasks = pair_ctas > 0
    nocompute = int(os.environ.get("PP_NOCOMPUTE", "0"))
    # Windows' overlapping KV is reused across CTAs, so it stays cacheable.
    dma_aux = int(os.environ.get("PP_AUX", "0" if window else "2"))
    # Direct window output is written once: stream it past the reused KV.
    OAUX = int(os.environ.get("PP_OAUX", "2")) if window else 0
    M, R, CH = pingpong_layout(query_length)
    if window and varlen and query_length == 8 and not os.environ.get("PP_LAYOUT"):
        # Four waves on two 40 KiB stages fit two CTAs per CU, so one CTA's Q
        # load and output store overlap the other's tiles.
        M, R, CH = 2, 4, 4
    SLOT_TOKENS = 16 * CH
    S = TILE // SLOT_TOKENS
    NWAVES = R * S
    # Staggered prefill: two groups of one wave per SIMD, alternating tiles.
    stagger = int(os.environ.get("PP_STAGGER", "1")) if (NWAVES == 8 and S == 1) else 0
    # Four 40 KiB D192/V128 stages fill gfx950's 160 KiB LDS; a window's few
    # tiles gain nothing from a deeper ring, which the stagger still needs.
    stages = stages or (4 if stagger else 2 if window else 3)
    pin = int(os.environ.get("PP_PIN", "2"))
    ysched = int(os.environ.get("PP_YSCHED", "8"))
    grp_mod = int(os.environ.get("PP_GRP", "0"))
    nopk = int(os.environ.get("PP_NOPK", "1"))
    xissue = int(os.environ.get("PP_XISSUE", "1")) if stages >= 4 else 0
    unroll = int(os.environ.get("PP_UNROLL", "0"))
    fuse = (
        int(os.environ.get("PP_FUSE", "0" if window else "1"))
        if (NWAVES == 4 and S == 1)
        else 0
    )
    fsched = int(os.environ.get("PP_FSCHED", "2")) if fuse else 0
    # Both overlap a stage's DMA with reads of the stage before it.
    assert stages >= 3 or not (stagger or fuse), "a 2-stage ring races these schedules"
    lsum = int(os.environ.get("PP_LSUM", "0")) if (CH > 1 and (stagger or fuse)) else 0
    # 32x32x16 MFMAs on the staggered prefill: lane (n, h) owns row n of the
    # wave's 32 rows; the V transpose reads need DV in {128, 256}.
    mf32 = (
        int(os.environ.get("PP_MF32", "0"))
        if (stagger and M == 2 and CH == 4 and value_dim in (128, 256))
        else 0
    )
    mf32 = 0 if window else mf32
    lsum = 0 if mf32 else lsum
    yahead = int(os.environ.get("PP_YAHEAD", "4")) if mf32 else 0
    # K operands an opus-style LDS segment preloads; the rest load in QK.
    KPRE = int(os.environ.get("PP_KPRE", "6")) if mf32 == 2 else 0
    # LDS segment issuing V's DMA (K's goes in L2); 0 issues both in L2.
    VSEG = int(os.environ.get("PP_VSEG", "0")) if mf32 == 2 else 0
    # With VSEG 6, the LDS segment issuing a full tile's V copies.
    VSPLIT = (
        int(os.environ.get("PP_VSPLIT", "2")) if VSEG == 6 else (2 if VSEG == 8 else 0)
    )
    ANCHOR = int(os.environ.get("PP_ANCHOR", "0")) if mf32 == 2 else 0
    D = head_dim
    DV = value_dim
    GQA = query_group_size
    QLEN = query_length
    ROWS = QLEN * GQA
    assert R * M * 16 >= ROWS, "layout must cover every query row"
    key = (
        D,
        DV,
        GQA,
        QLEN,
        block_size,
        float(softmax_scale),
        stages,
        prio,
        pair_tasks,
        nocompute,
        dma_aux,
        (M, R, CH),
        bool(varlen),
        stagger,
        pin,
        ysched,
        grp_mod,
        nopk,
        xissue,
        unroll,
        lsum,
        fuse,
        fsched,
        mf32,
        yahead,
        KPRE,
        VSEG,
        VSPLIT,
        ANCHOR,
        window,
        sink_dtype,
        seq_query_length,
        OAUX,
    )
    if key in _CACHE:
        return _CACHE[key]

    K_ROW = D * 2
    V_ROW = DV * 2
    K_CPR = D // 8
    V_CPR = DV // 8
    K_STAGE = TILE * K_ROW
    V_STAGE = TILE * V_ROW
    STAGE = K_STAGE + V_STAGE
    # LDS DMAs may land out of issue order, so a partial vmcnt wait does not
    # cover the oldest tile. Waves form ISSUERS groups that take turns per
    # tile; a group's previous tile is its only one in flight when it waits.
    ISSUERS = 2 if stagger else stages - 1
    if mf32 == 2:
        # Every wave issues every tile and waits for it one tile later.
        ISSUERS = 1
    LEAD = stages - 1
    GROUP_WAVES = NWAVES // ISSUERS
    assert GROUP_WAVES * ISSUERS == NWAVES, "issuer groups must split the waves"
    K_COPIES = TILE * K_CPR // WAVE // GROUP_WAVES
    V_COPIES = TILE * V_CPR // WAVE // GROUP_WAVES
    assert K_COPIES * WAVE * GROUP_WAVES == TILE * K_CPR
    assert V_COPIES * WAVE * GROUP_WAVES == TILE * V_CPR
    KK = D // 32
    NB = DV // 16
    TILES_PER_PLAN = KV_COMPUTE_BLOCK // TILE
    # Merge scratch per (token slot > 0, row block): O [M][16 rows][DV] f32,
    # then m and l per row.
    MERGE_O = M * 16 * DV * 4
    MERGE_RB = MERGE_O + 2 * M * 16 * 4
    total_bytes = max(stages * STAGE, (S - 1) * R * MERGE_RB)
    scale_log2 = float(softmax_scale) * LOG2E

    @fx.struct
    class SharedStorage:
        buf: fx.Array[fx.Int32, total_bytes // 4, 16]

    class _LdsTraits:
        pass

    kernel_name = (
        f"pa_decode_pingpong_kvbf16_d{D}_v{DV}_bs{block_size}_gqa{GQA}_mtp{QLEN}"
        f"_m{M}r{R}c{CH}_s{stages}_p{prio}_c{int(pair_tasks)}_n{nocompute}_a{dma_aux}_x{stagger}{pin}y{ysched}g{grp_mod}k{nopk}i{xissue}u{unroll}l{lsum}f{fuse}{fsched}w{mf32}{yahead}{KPRE}{VSEG}{VSPLIT}{ANCHOR}"
    )
    # Online-softmax state: [m (SM), l (SM), O (SM * SNB)].
    SM = 1 if mf32 else M
    SNB = DV // 32 if mf32 else NB
    STATE = 2 * SM + SM * SNB
    KK32 = D // 16
    O_LEN = 16 if mf32 else 4
    y_reads = []
    if mf32:
        v_swizzle = lambda t: (t & 3) << 2
        need = [2] * (4 * SNB) + [1] * (2 * KK32)
        cum = [sum(need[: i + 1]) for i in range(len(need))]
        # Reads issued after MFMA i (the first entry precedes MFMA 0).
        issued = cum[yahead - 1]
        y_reads = []
        for i in range(len(need)):
            target = cum[min(i + yahead, len(need) - 1)]
            y_reads.append(target - issued)
            issued = target
        y_reads = [cum[yahead - 1]] + y_reads
    else:
        v_swizzle = lambda t: (t & 7) << 1
    if varlen:
        kernel_name += "_varlen"
    if window:
        kernel_name += f"_sw{window}" + (f"_sink{sink_dtype}" if sink_dtype else "")
    SPLIT = max(seq_query_length // query_length, 1) if window and not varlen else 1
    if SPLIT > 1:
        kernel_name += f"_qs{SPLIT}"
    if OAUX:
        kernel_name += f"_o{OAUX}"
    SINK_T = {"f32": fx.Float32, "bf16": fx.BFloat16, "f16": fx.Float16, None: None}[
        sink_dtype
    ]

    @flyc.jit
    def _task(
        pmax_ptr: fx.Pointer,
        psum_ptr: fx.Pointer,
        pout_ptr: fx.Pointer,
        query_ptr: fx.Pointer,
        key_cache_ptr: fx.Pointer,
        value_cache_ptr: fx.Pointer,
        block_tables_ptr: fx.Pointer,
        query_map_ptr: fx.Pointer,
        max_blocks_per_seq: fx.Int32,
        stride_q_row: fx.Int32,
        stride_q_head: fx.Int32,
        seq: fx.Int32,
        start: fx.Int32,
        end: fx.Int32,
        context: fx.Int32,
        slot: fx.Int32,
        work_capacity: fx.Int32,
        out_ptr: fx.Pointer,
        sinks_ptr: fx.Pointer,
        num_kv_heads: fx.Int32,
    ):
        lds = PaDecodeLds(_LdsTraits, SharedStorage)
        lds.allocate()
        kv_h = fx.Int32(gpu.block_id("y"))
        # Uniform, so the per-tile DMA address math stays on the SALU. The
        # window launch passes it: grid_dim is a dependent dispatch-packet load.
        n_kv = _uni(num_kv_heads) if const_expr(window) else _uni(gpu.grid_dim.y)
        tid = fx.Int32(gpu.thread_id("x"))
        warp = fx.Int32(fx.rocdl.readfirstlane(T.i32, tid // WAVE))
        lane = tid % WAVE
        g = lane // 16
        l16 = lane % 16
        ts = warp // R
        rb = warp % R
        if const_expr(stagger and grp_mod):
            issuer = warp % ISSUERS
            group_warp = warp // ISSUERS
        else:
            issuer = warp // GROUP_WAVES
            group_warp = warp % GROUP_WAVES
        grp = issuer
        if const_expr(mf32 == 2):
            grp = warp // (NWAVES // 2)
        neg_inf = fx.Float32(float("-inf"))
        zero = fx.Float32(0.0)
        # Query of each of the wave's M tiles; padded tiles reuse the last query.
        queries = [rb * M + m for m in range_constexpr(M)]
        if const_expr(varlen):
            qmap = fx.ptr_load(
                fx.add_offset(fx.recast_iter(fx.Int32, query_map_ptr), seq * 4),
                result_type=fx.Vector.make_type(4, fx.Int32),
            )
            first_row = _uni(fx.Int32(qmap[0]))
            table_seq = _uni(fx.Int32(qmap[1]))
            valid = _uni(fx.Int32(qmap[2]))
        else:
            first_row = seq * QLEN
            table_seq = seq // SPLIT if const_expr(SPLIT > 1) else seq
            valid = QLEN

        # Q: B operand of QK, lane (g, l16) holds head l16, dims 32kk + 8g.
        q_buf = ptr_buf_tensor(query_ptr, fx.BFloat16)
        q_tiled = fx.logical_divide(q_buf, fx.make_layout(1, 1))
        q_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.BFloat16)
        q_reg = fx.make_rmem_tensor(fx.make_layout(8, 1), fx.BFloat16)
        q_ops = []
        n32 = lane % 32
        h32 = lane // 32
        q32 = []
        if const_expr(mf32):
            # B operand of the 32x32 QK: lane (n, h) holds row n, dims 16kk + 8h.
            q_idx = rb * M + n32 // 16
            limit32 = context - valid + q_idx
            if const_expr(varlen):
                q_idx = (q_idx < valid).select(q_idx, valid - 1)
            elif const_expr(R * M > QLEN):
                q_idx = (q_idx < QLEN).select(q_idx, QLEN - 1)
            row_elem = (
                (first_row + q_idx) * stride_q_row
                + (kv_h * GQA + n32 % 16) * stride_q_head
                + h32 * 8
            )
            for kk in range_constexpr(KK32):
                fx.copy(q_atom, fx.slice(q_tiled, (None, row_elem + kk * 16)), q_reg)
                q32.append(fx.Vector(fx.memref_load_vec(q_reg)))
        for m in range_constexpr(0 if mf32 else M):
            q_idx = queries[m]
            if const_expr(varlen):
                q_idx = (q_idx < valid).select(q_idx, valid - 1)
            elif const_expr(R * M > QLEN):
                q_idx = (q_idx < QLEN).select(q_idx, QLEN - 1)
            q_row = first_row + q_idx
            head = kv_h * GQA + l16
            row_elem = q_row * stride_q_row + head * stride_q_head + g * 8
            ops_m = []
            for kk in range_constexpr(KK):
                fx.copy(q_atom, fx.slice(q_tiled, (None, row_elem + kk * 32)), q_reg)
                ops_m.append(fx.Vector(fx.memref_load_vec(q_reg)))
            q_ops.append(ops_m)

        k_base = buf_base_i64(key_cache_ptr)
        v_base = buf_base_i64(value_cache_ptr)
        bt = ptr_buf_tensor(block_tables_ptr, fx.Int32)
        k_page_elems = fx.Int64(n_kv) * fx.Int64(D * block_size)
        v_page_elems = fx.Int64(n_kv) * fx.Int64(DV * block_size)
        k_tok_bytes = n_kv * K_ROW
        v_tok_bytes = n_kv * V_ROW

        t_begin = start * TILES_PER_PLAN
        ctx_tiles = (context + (TILE - 1)) // TILE
        t_end_plan = end * TILES_PER_PLAN
        t_end = (t_end_plan < ctx_tiles).select(t_end_plan, ctx_tiles)
        # Tiles below t_split hold no position past any query's causal bound.
        unmasked = context - (valid - 1)
        unmasked = (unmasked > 0).select(unmasked, 0) // TILE
        t_split = (unmasked < t_end).select(unmasked, t_end)
        t_split = (t_split > t_begin).select(t_split, t_begin)
        if const_expr(window):
            # From the tile of the first query's first visible key, all masked.
            first_key = context - valid - (window - 1)
            t_begin = (first_key > 0).select(first_key, 0) // TILE
            t_end = ctx_tiles
            t_split = t_begin
        if const_expr(mf32 == 2):
            # The last tile's lookahead QK reads a stale stage: mask it whole.
            t_last = t_end * TILE - 1
            limit32 = (limit32 < t_last).select(limit32, t_last)

        # DMA copy i of a stage writes LDS bytes [1024 i, 1024 i + 1024); lane l
        # fetches the global chunk the swizzle stores at chunk 64 i + l.
        def dma_offsets(copies, cpr, tok_bytes, swizzle):
            offs = []
            for j in range_constexpr(copies):
                pos = (group_warp * copies + j) * WAVE + lane
                tok = pos // cpr
                offs.append(tok * tok_bytes + ((pos % cpr) ^ swizzle(tok)) * 16)
            return offs

        k_voff = dma_offsets(K_COPIES, K_CPR, k_tok_bytes, lambda t: (t >> 1) & 7)
        v_voff = dma_offsets(V_COPIES, V_CPR, v_tok_bytes, v_swizzle)

        def dma_addr(base, page_elems, row, tok0, phys, page_tok0):
            valid = context - tok0
            valid = (valid < TILE).select(valid, TILE)
            addr = _uni64(
                base
                + (
                    fx.Int64(phys) * page_elems
                    + fx.Int64((page_tok0 * n_kv + kv_h) * (row // 2))
                )
                * 2
            )
            # Rows at or past the context read as zero.
            nbytes = _uni((valid - 1) * n_kv * row + row)
            return addr, nbytes

        def dma_fire(addr, nbytes, voffs, copies, lds_off, soff=None, rs=None):
            if const_expr(rs is None):
                rs = create_buffer_resource_from_addr(
                    addr, num_records_bytes=fx.Int64(nbytes)
                )
            for j in range_constexpr(copies):
                dst = fx.to_llvm_ptr(
                    lds.pointer(
                        lds_off + (group_warp * copies + j) * (WAVE * 16), fx.Int32
                    )
                )
                fx.rocdl.raw_ptr_buffer_load_async_lds(
                    rs,
                    dst,
                    fx.Int32(16).ir_value(),
                    voffs[j].ir_value(),
                    fx.Int32(0 if soff is None else soff).ir_value(),
                    fx.Int32(0).ir_value(),
                    aux=_aux(dma_aux),
                )

        if const_expr(VSEG in (6, 8)):
            # Whole-cache resources: a full tile's page goes in soffset (the
            # host keeps each cache under 4 GiB); only a tile crossing the
            # context end needs its own bound for the zero fill.
            k_rs_all = create_buffer_resource_from_addr(
                _uni64(k_base + fx.Int64(kv_h * K_ROW)),
                num_records_bytes=fx.Int64(0xFFFFFFFF),
            )
            v_rs_all = create_buffer_resource_from_addr(
                _uni64(v_base + fx.Int64(kv_h * V_ROW)),
                num_records_bytes=fx.Int64(0xFFFFFFFF),
            )
            k_page_bytes = _uni(n_kv * (block_size * K_ROW))
            v_page_bytes = _uni(n_kv * (block_size * V_ROW))

        def dma(base, page_elems, row, voffs, copies, lds_off, tok0, phys, page_tok0):
            addr, nbytes = dma_addr(base, page_elems, row, tok0, phys, page_tok0)
            dma_fire(addr, nbytes, voffs, copies, lds_off)

        def tile_phys(tile):
            return fx.Int32(
                buf_scalar_load(
                    bt, table_seq * max_blocks_per_seq + (tile * TILE) // block_size
                )
            )

        def issue_half(tile, stage, phys, values):
            tok0 = tile * TILE
            if const_expr(values):
                dma(
                    v_base,
                    v_page_elems,
                    V_ROW,
                    v_voff,
                    V_COPIES,
                    stage * STAGE + K_STAGE,
                    tok0,
                    phys,
                    tok0 % block_size,
                )
            else:
                dma(
                    k_base,
                    k_page_elems,
                    K_ROW,
                    k_voff,
                    K_COPIES,
                    stage * STAGE,
                    tok0,
                    phys,
                    tok0 % block_size,
                )

        def issue(tile, stage):
            if const_expr(nocompute == 3):
                return
            tok0 = tile * TILE
            phys = fx.Int32(
                buf_scalar_load(bt, table_seq * max_blocks_per_seq + tok0 // block_size)
            )
            page_tok0 = tok0 % block_size
            if const_expr(nocompute == 2):
                phys = fx.Int32(1)
            stage_off = stage * STAGE
            dma(
                k_base,
                k_page_elems,
                K_ROW,
                k_voff,
                K_COPIES,
                stage_off,
                tok0,
                phys,
                page_tok0,
            )
            dma(
                v_base,
                v_page_elems,
                V_ROW,
                v_voff,
                V_COPIES,
                stage_off + K_STAGE,
                tok0,
                phys,
                page_tok0,
            )

        for s in range_constexpr(stages - 1):
            # Group 0 issues one tile later, in its softmax phase.
            lead_s = (
                LEAD - 1 if (stagger and xissue and s % 2 == 0 and mf32 != 2) else LEAD
            )
            if const_expr(s < lead_s):  # noqa: SIM102 (static guard of a dynamic if)
                if (issuer == s % ISSUERS) & (t_begin + s < t_end):
                    issue(t_begin + s, s)

        # Loop-invariant LDS read offsets within a stage.
        kq = l16 // 4
        qq = l16 % 4
        # The swizzles see neither the chunk nor bits above 8 chunks of a K row,
        # so those become instruction immediates over 2 K and NB V bases.
        swz = (l16 >> 1) & 7
        k_tok = ts * SLOT_TOKENS + l16
        k_lds = [
            k_tok * K_ROW + ((par * 4 + g) ^ swz) * 16 for par in range_constexpr(2)
        ]
        k_imm = [
            [c * 16 * K_ROW + (kk // 2) * 128 for kk in range_constexpr(KK)]
            for c in range_constexpr(CH)
        ]
        vswz = ((g * 4 + kq) & 7) << 1
        v_tok = ts * SLOT_TOKENS + g * 4 + kq
        v_lds = [
            K_STAGE + v_tok * V_ROW + ((nb * 2 + qq // 2) ^ vswz) * 16 + (qq % 2) * 8
            for nb in range_constexpr(NB)
        ]
        v_imm = [c * 16 * V_ROW for c in range_constexpr(CH)]
        # 32x32: K lane (n, h) reads token n, dims 16kk + 8h, over 4 bases
        # (kk % 4) with the rest immediate. Vᵀ group g reads 4 tokens of dims
        # 32 dblk + 16 (g % 2): lane (kq, qq) token 4h + kq, dims 4qq.., the
        # swizzle moving token kq's 4-chunk group to dblk ^ kq.
        swz32 = (n32 >> 1) & 7
        k32_lds = [
            n32 * K_ROW + ((2 * q4 + h32) ^ swz32) * 16 for q4 in range_constexpr(4)
        ]
        v32_lds = [
            K_STAGE
            + (4 * h32 + kq) * V_ROW
            + (4 * (kq ^ dblk) + 2 * (g % 2) + qq // 2) * 16
            + (qq % 2) * 8
            for dblk in range_constexpr(SNB if mf32 else 0)
        ]

        # Raw-score state [m (M), l (M), O (M * NB)]: m is a row-uniform
        # reference, rescaled only when a score exceeds it by RESCALE_LOG2 in
        # the log2 domain; l is per lane.
        init = (
            [neg_inf] * SM
            + [zero] * SM
            + [fx.full(O_LEN, 0.0, fx.Float32) for _ in range_constexpr(SM * SNB)]
        )
        limit = [context - valid + queries[m] for m in range_constexpr(M)]
        rescale_raw = RESCALE_LOG2 / scale_log2

        def scaled_ref(m_raw):
            return (m_raw > neg_inf).select(m_raw * fx.Float32(scale_log2), zero)

        def rescale(st, lmax):
            out_m = []
            out_l = []
            out_o = []
            for m in range_constexpr(M):
                rmax = lmax[m]
                for sh in (16, 32):
                    rmax = fx.maxnumf(rmax, rmax.shuffle_xor(sh, WAVE))
                m_new = fx.maxnumf(st[m], rmax)
                corr = fx.Float32(
                    fx.exp2(
                        st[m] * fx.Float32(scale_log2) - scaled_ref(m_new),
                        fastmath="fast",
                    )
                )
                out_m.append(m_new)
                out_l.append(st[M + m] * corr)
                corr_b = fx.full(4, corr, fx.Float32)
                out_o.extend(
                    st[2 * M + m * NB + nb] * corr_b for nb in range_constexpr(NB)
                )
            return out_m + out_l + out_o

        def stage_bases(stage):
            stage_off = stage * STAGE
            return [stage_off + b for b in k_lds], [stage_off + b for b in v_lds]

        def qk(stage, between=(), fence=0):
            """S = K Q^T for a stage: lane (g, l16) of chunk c holds token
            slot + 16c + 4g + i, row l16. One emit step of between follows
            each MFMA, with a compiler fence every fence MFMAs."""
            k_stage, _ = stage_bases(stage)
            s_acc = [[None] * CH for _ in range_constexpr(M)]
            queue = list(between)
            n_mfma = [0]

            def after_mfma():
                n_mfma[0] += 1
                if const_expr(len(queue) > 0):
                    queue.pop(0)()
                if const_expr(fence > 0 and n_mfma[0] % max(fence, 1) == 0):
                    fx.rocdl.sched_barrier(0)

            def k_chunk(c):
                return [
                    lds.load(k_stage[kk % 2] + k_imm[c][kk], fx.Int64, 2).bitcast(
                        fx.BFloat16
                    )
                    for kk in range_constexpr(KK)
                ]

            # Chunk c + 1's K reads issue under chunk c's MFMAs.
            k_ops = k_chunk(0)
            for c in range_constexpr(CH):
                k_next = k_chunk(c + 1) if c + 1 < CH else None
                for m in range_constexpr(M):
                    s_acc[m][c] = fx.full(4, 0.0, fx.Float32)
                for kk in range_constexpr(KK):
                    for m in range_constexpr(M):
                        s_acc[m][c] = fx.Vector(
                            fx.rocdl.mfma_f32_16x16x32_bf16(
                                T.f32x4, [k_ops[kk], q_ops[m][kk], s_acc[m][c], 0, 0, 0]
                            )
                        )
                        after_mfma()
                k_ops = k_next
            for step_fn in queue:
                step_fn()
            return [
                s_acc[m][c] for m in range_constexpr(M) for c in range_constexpr(CH)
            ]

        def soft(it, s_flat, state, masked, deferred=False):
            """Online softmax of one tile's scores: (state with O rescaled, P).
            deferred returns the P work as emit steps (one exp each, packing
            after each operand's last) and a list P fills in once they ran."""
            lane_tok = it * TILE + ts * SLOT_TOKENS + g * 4
            scores = []
            lmax = []
            for m in range_constexpr(M):
                vals = []
                for c in range_constexpr(CH):
                    for i in range_constexpr(4):
                        v = fx.Float32(s_flat[m * CH + c][i])
                        if const_expr(masked):
                            rel = limit[m] - lane_tok
                            keep = fx.Int32(c * 16 + i) <= rel
                            if const_expr(window):
                                keep = keep & (fx.Int32(c * 16 + i) > rel - window)
                            v = keep.select(v, neg_inf)
                        vals.append(v)
                mx = vals[0]
                for v in vals[1:]:
                    mx = fx.maxnumf(mx, v)
                scores.append(vals)
                lmax.append(mx)
            over = lmax[0] > state[0] + fx.Float32(rescale_raw)
            for m in range_constexpr(1, M):
                over = over | (lmax[m] > state[m] + fx.Float32(rescale_raw))
            st = list(state)
            if fx.Int64(fx.rocdl.ballot(T.i64, over.ir_value())) != fx.Int64(0):
                st = rescale(st, lmax)

            # One B operand per chunk pair (8 tokens a lane), or the chunk.
            width = 4 if CH == 1 else 8
            p_ops = []
            steps = []
            for m in range_constexpr(M):
                neg_ref = -scaled_ref(st[m])
                p = []

                def one(m=m, neg_ref=neg_ref, p=p):
                    v = scores[m][len(p)]
                    e = fx.Float32(
                        fx.exp2(
                            fx.fma(v, fx.Float32(scale_log2), neg_ref), fastmath="fast"
                        )
                    )
                    p.append(e)
                    if const_expr(not lsum):
                        st[M + m] = st[M + m] + e
                    if const_expr(len(p) % width == 0):
                        p_ops.append(
                            fx.Vector.from_elements(
                                p[len(p) - width :], dtype=fx.Float32
                            ).to(fx.BFloat16)
                        )

                steps.extend([one] * len(scores[m]))
            if const_expr(deferred):
                return st, p_ops, steps
            for step_fn in steps:
                step_fn()
            return st, p_ops

        def pv(stage, p_flat, st):
            """O += P V for a stage; the transpose read gives lane j dim j of
            tokens 4g + k, k = 0..3, in its 16-lane group."""
            _, v_stage = stage_bases(stage)
            pairs = max(CH // 2, 1)
            o_acc = [
                [st[2 * M + m * NB + nb] for nb in range_constexpr(NB)]
                for m in range_constexpr(M)
            ]

            def v_parts(nb):
                return [
                    fx.Vector(
                        fx.rocdl.ds_read_tr16_b64(
                            fx.Vector.make_type(4, fx.BFloat16),
                            fx.to_llvm_ptr(
                                lds.pointer(v_stage[nb] + v_imm[c], fx.BFloat16)
                            ),
                        ).result
                    )
                    for c in range_constexpr(CH)
                ]

            v_cur = v_parts(0)
            for nb in range_constexpr(NB):
                parts = v_cur
                v_cur = v_parts(nb + 1) if nb + 1 < NB else None
                for m in range_constexpr(M):
                    if const_expr(CH > 1):
                        for j in range_constexpr(CH // 2):
                            v_op = fx.Vector.from_elements(
                                [
                                    parts[2 * j + c][i]
                                    for c in range_constexpr(2)
                                    for i in range_constexpr(4)
                                ],
                                dtype=fx.BFloat16,
                            )
                            o_acc[m][nb] = fx.Vector(
                                fx.rocdl.mfma_f32_16x16x32_bf16(
                                    T.f32x4,
                                    [
                                        v_op,
                                        p_flat[m * pairs + j],
                                        o_acc[m][nb],
                                        0,
                                        0,
                                        0,
                                    ],
                                )
                            )
                    else:
                        o_acc[m][nb] = fx.Vector(
                            fx.rocdl.mfma_f32_16x16x16bf16_1k(
                                T.f32x4,
                                [
                                    parts[0].bitcast(fx.Int16),
                                    p_flat[m].bitcast(fx.Int16),
                                    o_acc[m][nb],
                                    0,
                                    0,
                                    0,
                                ],
                            )
                        )
            st = list(st)
            if const_expr(lsum):
                # Row sums of the bf16 P on the matrix pipe: every lane gets
                # its row's whole-tile sum.
                ones = fx.full(8, 1.0, fx.BFloat16)
                for m in range_constexpr(M):
                    acc = fx.full(4, 0.0, fx.Float32)
                    for j in range_constexpr(pairs):
                        acc = fx.Vector(
                            fx.rocdl.mfma_f32_16x16x32_bf16(
                                T.f32x4, [ones, p_flat[m * pairs + j], acc, 0, 0, 0]
                            )
                        )
                    st[M + m] = st[M + m] + fx.Float32(acc[0])
            return st[: 2 * M] + [
                o_acc[m][nb] for m in range_constexpr(M) for nb in range_constexpr(NB)
            ]

        def mfma32(a, b, c):
            return fx.Vector(
                fx.rocdl.mfma_f32_32x32x16_bf16(
                    fx.Vector.make_type(16, fx.Float32), [a, b, c, 0, 0, 0]
                )
            )

        def qk32(stage):
            """S^T = K Q^T per 32-token block bk: lane (n, h) holds row n,
            tokens 32bk + 8j + 4h + i at element 4j + i."""
            k_stage = [stage * STAGE + b for b in k32_lds]
            out = []
            for bk in range_constexpr(2):
                acc = fx.full(16, 0.0, fx.Float32)
                for kk in range_constexpr(KK32):
                    k_op = lds.load(
                        k_stage[kk % 4] + bk * 32 * K_ROW + (kk // 4) * 128, fx.Int64, 2
                    ).bitcast(fx.BFloat16)
                    acc = mfma32(k_op, q32[kk], acc)
                out.append(acc)
            return out

        def soft32(it, s_flat, state, masked):
            """(state with O rescaled, P): P operand b holds the lane's tokens
            16b + 4h + (0..3, 8..11), elements 8b..8b + 7 of its scores."""
            lane_tok = it * TILE + 4 * h32
            vals = []
            for bk in range_constexpr(2):
                for e in range_constexpr(16):
                    v = fx.Float32(s_flat[bk][e])
                    if const_expr(masked):
                        tok = bk * 32 + 8 * (e // 4) + e % 4
                        v = (fx.Int32(tok) <= limit32 - lane_tok).select(v, neg_inf)
                    vals.append(v)
            mx = vals[0]
            for v in vals[1:]:
                mx = fx.maxnumf(mx, v)
            st = list(state)
            over = mx > st[0] + fx.Float32(rescale_raw)
            if fx.Int64(fx.rocdl.ballot(T.i64, over.ir_value())) != fx.Int64(0):
                rmax = fx.maxnumf(mx, mx.shuffle_xor(32, WAVE))
                m_new = fx.maxnumf(st[0], rmax)
                corr = fx.Float32(
                    fx.exp2(
                        st[0] * fx.Float32(scale_log2) - scaled_ref(m_new),
                        fastmath="fast",
                    )
                )
                corr_b = fx.full(16, corr, fx.Float32)
                st = [m_new, st[1] * corr] + [o * corr_b for o in st[2:]]
            neg_ref = -scaled_ref(st[0])
            p = [
                fx.Float32(
                    fx.exp2(fx.fma(v, fx.Float32(scale_log2), neg_ref), fastmath="fast")
                )
                for v in vals
            ]
            # Pairwise, so the row sum is no serial add chain.
            st[1] = st[1] + _tree_sum(p)
            p_ops = [
                fx.Vector.from_elements(p[8 * b : 8 * b + 8], dtype=fx.Float32).to(
                    fx.BFloat16
                )
                for b in range_constexpr(4)
            ]
            return st, p_ops

        def pv32(stage, p_ops, st):
            """O^T += V^T P^T: lane (n, h) of dim block dblk holds row n, dims
            32 dblk + 8j + 4h + i at element 4j + i."""
            v_stage = [stage * STAGE + b for b in v32_lds]
            o = list(st[2:])

            def vt(dblk, b, half):
                return fx.Vector(
                    fx.rocdl.ds_read_tr16_b64(
                        fx.Vector.make_type(4, fx.BFloat16),
                        fx.to_llvm_ptr(
                            lds.pointer(
                                v_stage[dblk] + (16 * b + 8 * half) * V_ROW,
                                fx.BFloat16,
                            )
                        ),
                    ).result
                )

            for dblk in range_constexpr(SNB):
                for b in range_constexpr(4):
                    lo = vt(dblk, b, 0)
                    hi = vt(dblk, b, 1)
                    a = fx.Vector.from_elements(
                        [lo[i] for i in range_constexpr(4)]
                        + [hi[i] for i in range_constexpr(4)],
                        dtype=fx.BFloat16,
                    )
                    o[dblk] = mfma32(a, p_ops[b], o[dblk])
            return st[:2] + o

        def step(it, state, masked):
            rel = it - t_begin
            stage = rel % stages
            if issuer == rel % ISSUERS:
                fx.rocdl.s_waitcnt(vmcnt=0)
            gpu.barrier()
            nxt = it + (stages - 1)
            nxt_stage = (rel + (stages - 1)) % stages
            # Tile it + ISSUERS has tile it's issuers.
            if (issuer == rel % ISSUERS) & (nxt < t_end):
                issue(nxt, nxt_stage)
            if const_expr(nocompute in (1, 2)):
                return list(state)
            if const_expr(prio):
                fx.rocdl.s_setprio(1)
            s_flat = qk(stage)
            if const_expr(prio):
                fx.rocdl.s_setprio(0)
            st, p_flat = soft(it, s_flat, state, masked)
            if const_expr(prio):
                fx.rocdl.s_setprio(1)
            out = pv(stage, p_flat, st)
            if const_expr(prio):
                fx.rocdl.s_setprio(0)
            return out

        def smod(x):
            return x & (stages - 1) if stages & (stages - 1) == 0 else x % stages

        def stagger_step(it, carry, masked, g=None, par=None):
            # Group 1 runs one phase behind group 0: each phase X (softmax of
            # tile i) pairs with the other group's phase Y (PV of tile i, QK of
            # tile i + 1). Tile i + LEAD reuses tile i - 1's stage once group
            # 1's PV of it ends; a group waits for its tile two slots before
            # the first QK, when that tile is its only one in flight. g and
            # par, when given, fix the wave's group and i's parity at compile
            # time.
            rel = it - t_begin
            state = carry[:STATE]
            s_flat = carry[STATE:]
            nxt = it + LEAD

            def gated(group, rpar, bound, fn):
                if const_expr(
                    (g is None or g == group) and (par is None or par == rpar)
                ):
                    if const_expr(g is not None and par is not None):
                        if const_expr(bound is None):
                            fn()
                        else:
                            if bound:
                                fn()
                    else:
                        c = None
                        if const_expr(g is None):
                            c = grp == group
                        if const_expr(par is None):
                            pc = (rel & 1) == rpar
                            c = pc if c is None else c & pc
                        if const_expr(bound is not None):
                            c = c & bound
                        if c:
                            fn()

            gated(1, (1 - LEAD) % 2, nxt < t_end, lambda: issue(nxt, smod(rel + LEAD)))
            if const_expr(xissue):
                # Group 0's tile i + LEAD - 1 takes tile i - 2's stage, free
                # since group 1's PV of it in the previous slot.
                gated(
                    0,
                    (1 - LEAD) % 2,
                    nxt - 1 < t_end,
                    lambda: issue(nxt - 1, smod(rel + LEAD - 1)),
                )
            if const_expr(mf32):
                st, p_flat = soft32(it, s_flat, state, masked)
            else:
                st, p_flat = soft(it, s_flat, state, masked)
            if const_expr(pin):
                p_flat = [_pin(p) for p in p_flat]
                if const_expr(pin > 1 and SM == 1):
                    st[1] = fx.Float32(
                        llvm.inline_asm(
                            T.f32,
                            [fx.Float32(st[1]).ir_value()],
                            "",
                            "=v,0",
                            has_side_effects=True,
                        )
                    )
                elif const_expr(pin > 1):
                    sums = _pin(
                        fx.Vector.from_elements(st[SM : 2 * SM], dtype=fx.Float32)
                    )
                    st = st[:SM] + [sums[m] for m in range_constexpr(SM)] + st[2 * SM :]
            gated(0, 1, None, lambda: fx.rocdl.s_waitcnt(vmcnt=0))
            gpu.barrier()
            if const_expr(not xissue):
                gated(0, LEAD % 2, nxt < t_end, lambda: issue(nxt, smod(rel + LEAD)))
            if const_expr(prio):
                fx.rocdl.s_setprio(1)
            # The last tile's lookahead QK reads a stale stage; it is unused.
            if const_expr(mf32):
                state = pv32(smod(rel), p_flat, st)
                s_next = qk32(smod(rel + 1))
            else:
                state = pv(smod(rel), p_flat, st)
                s_next = qk(smod(rel + 1))
            if const_expr(ysched and mf32):
                # Each MFMA's LDS reads (two per PV, one per QK) issue yahead
                # MFMAs before it.
                fx.rocdl.sched_group_barrier(fx.rocdl.mask_dsrd, y_reads[0], 0)
                for n_ds_i in y_reads[1:]:
                    fx.rocdl.sched_group_barrier(fx.rocdl.mask_mfma, 1, 0)
                    if const_expr(n_ds_i > 0):
                        fx.rocdl.sched_group_barrier(fx.rocdl.mask_dsrd, n_ds_i, 0)
            elif const_expr(ysched):
                # One MFMA wave a SIMD: keep LDS reads ysched MFMAs ahead.
                n_ds = (NB + KK) * CH
                n_mfma = M * NB * max(CH // 2, 1) + M * KK * CH
                if const_expr(lsum):
                    n_mfma += M * max(CH // 2, 1)
                fx.rocdl.sched_group_barrier(fx.rocdl.mask_dsrd, ysched, 0)
                for _ in range_constexpr(n_ds - ysched):
                    fx.rocdl.sched_group_barrier(fx.rocdl.mask_mfma, 1, 0)
                    fx.rocdl.sched_group_barrier(fx.rocdl.mask_dsrd, 1, 0)
                fx.rocdl.sched_group_barrier(
                    fx.rocdl.mask_mfma, n_mfma - (n_ds - ysched), 0
                )
            if const_expr(prio):
                fx.rocdl.s_setprio(0)
            gated(1, 1, None, lambda: fx.rocdl.s_waitcnt(vmcnt=0))
            gpu.barrier()
            return list(state) + list(s_next)

        def k_reads(stage, bk, k0, k1):
            k_stage = [stage * STAGE + b for b in k32_lds]
            return [
                lds.load(
                    k_stage[kk % 4] + bk * 32 * K_ROW + (kk // 4) * 128, fx.Int64, 2
                ).bitcast(fx.BFloat16)
                for kk in range_constexpr(k0, k1)
            ]

        def qk_block(stage, bk, k_ops):
            # The LDS segment loaded the first KPRE operands; the rest issue
            # KPRE MFMAs ahead of their use.
            acc = fx.full(16, 0.0, fx.Float32)
            ops = list(k_ops)
            for kk in range_constexpr(KK32):
                acc = mfma32(ops[kk], q32[kk], acc)
                if const_expr(kk + KPRE < KK32):
                    ops.extend(k_reads(stage, bk, kk + KPRE, kk + KPRE + 1))
            return acc

        soft_items, soft_plan = _soft_plan(KK32)

        def soft_item(item, xs, l_in, env):
            kind, arg = item
            if const_expr(kind == "e"):
                env[f"p{arg}"] = fx.Float32(fx.exp2(xs[arg], fastmath="fast"))
            elif const_expr(kind == "a"):
                env[f"s{arg}"] = env[f"p{2 * arg}"] + env[f"p{2 * arg + 1}"]
            elif const_expr(kind == "p"):
                elems = [env[f"p{8 * arg + i}"] for i in range_constexpr(8)]
                env[f"P{arg}"] = _pin(
                    fx.Vector.from_elements(elems, dtype=fx.Float32).to(fx.BFloat16)
                )
            elif const_expr(kind == "t"):
                env[arg[2]] = env[arg[0]] + env[arg[1]]
            else:
                env["l"] = _pin_f32(l_in + env["sum"])

        def qk_soft(stage, bk, k_ops, xs, l_in, hooks=None):
            # QK block bk of the next tile, woven with this tile's softmax of
            # xs (exp2 arguments); returns scores, bf16 P operands, row sum.
            env = {}
            acc = fx.full(16, 0.0, fx.Float32)
            ops = list(k_ops)
            for kk in range_constexpr(KK32):
                acc = mfma32(ops[kk], q32[kk], acc)
                if const_expr(hooks is not None):
                    # Orders the MFMA before the slot's pinned scalar math.
                    acc = _pin(acc)
                if const_expr(kk + KPRE < KK32):
                    ops.extend(k_reads(stage, bk, kk + KPRE, kk + KPRE + 1))
                for j in range_constexpr(len(soft_plan[kk])):
                    soft_item(soft_items[soft_plan[kk][j]], xs, l_in, env)
                if const_expr(hooks is not None and kk in hooks):
                    hooks[kk]()
                fx.rocdl.sched_barrier(0)
            return acc, [env["P0"], env["P1"]], env["l"]

        def interleave_qk(n_valu, n_trans):
            for i in range_constexpr(KK32):
                fx.rocdl.sched_group_barrier(fx.rocdl.mask_mfma, 1, 0)
                if const_expr(i + KPRE < KK32):
                    fx.rocdl.sched_group_barrier(fx.rocdl.mask_dsrd, 1, 0)
                nv = n_valu * (i + 1) // KK32 - n_valu * i // KK32
                nt = n_trans * (i + 1) // KK32 - n_trans * i // KK32
                if const_expr(nv > 0):
                    fx.rocdl.sched_group_barrier(0x002, nv, 0)
                if const_expr(nt > 0):
                    fx.rocdl.sched_group_barrier(0x400, nt, 0)

        def v_reads(stage, d0, d1):
            v_stage = [stage * STAGE + b for b in v32_lds]

            def vt(dblk, b, half):
                return fx.Vector(
                    fx.rocdl.ds_read_tr16_b64(
                        fx.Vector.make_type(4, fx.BFloat16),
                        fx.to_llvm_ptr(
                            lds.pointer(
                                v_stage[dblk] + (16 * b + 8 * half) * V_ROW,
                                fx.BFloat16,
                            )
                        ),
                    ).result
                )

            ops = []
            for dblk in range_constexpr(d0, d1):
                for b in range_constexpr(4):
                    lo = vt(dblk, b, 0)
                    hi = vt(dblk, b, 1)
                    ops.append(
                        fx.Vector.from_elements(
                            [lo[i] for i in range_constexpr(4)]
                            + [hi[i] for i in range_constexpr(4)],
                            dtype=fx.BFloat16,
                        )
                    )
            return ops

        def masked_scores(s_blocks, it_n, masked):
            lane_tok = it_n * TILE + 4 * h32
            vals = []
            for bk in range_constexpr(2):
                for e in range_constexpr(16):
                    v = fx.Float32(s_blocks[bk][e])
                    if const_expr(masked):
                        tok = bk * 32 + 8 * (e // 4) + e % 4
                        v = (fx.Int32(tok) <= limit32 - lane_tok).select(v, neg_inf)
                    vals.append(v)
            return vals

        def lane_max(vals):
            mx = vals[0]
            for v in vals[1:]:
                mx = fx.maxnumf(mx, v)
            return mx

        def rescale32(st, mx):
            # Only when some lane's max passes its reference by RESCALE_LOG2.
            st = list(st)
            over = mx > st[0] + fx.Float32(rescale_raw)
            if fx.Int64(fx.rocdl.ballot(T.i64, over.ir_value())) != fx.Int64(0):
                rmax = fx.maxnumf(mx, mx.shuffle_xor(32, WAVE))
                m_new = fx.maxnumf(st[0], rmax)
                corr = fx.Float32(
                    fx.exp2(
                        st[0] * fx.Float32(scale_log2) - scaled_ref(m_new),
                        fastmath="fast",
                    )
                )
                corr_b = fx.full(16, corr, fx.Float32)
                st = [m_new, st[1] * corr] + [o * corr_b for o in st[2:]]
            return st

        def interleave(n_mfma, n_valu, n_trans):
            for i in range_constexpr(n_mfma):
                fx.rocdl.sched_group_barrier(fx.rocdl.mask_mfma, 1, 0)
                nv = n_valu * (i + 1) // n_mfma - n_valu * i // n_mfma
                nt = n_trans * (i + 1) // n_mfma - n_trans * i // n_mfma
                if const_expr(nv > 0):
                    fx.rocdl.sched_group_barrier(0x002, nv, 0)
                if const_expr(nt > 0):
                    fx.rocdl.sched_group_barrier(0x400, nt, 0)

        def seg_barrier():
            fx.rocdl.sched_barrier(0)
            gpu.barrier()
            fx.rocdl.sched_barrier(0)

        def opus_step(it, carry, masked):
            # Tile it: PV of it and QK of it + 1 in four MFMA segments, each
            # after an LDS segment that loads its operands; group 1 runs one
            # segment behind, so a SIMD's two waves alternate the two kinds.
            # Tile it's scores arrive as exp2 arguments, their exps and row
            # sums filling QK's MFMA shadows; tile it + 1's mask, max and
            # reference fill PV's. A rescale's correction of O and l waits
            # for the next L1. Every wave issues a share of tile x's DMA in
            # the L2 of tile x - 3, when tile x - 4's stage has drained, right
            # after waiting for its share of tile x - 1.
            rel = it - t_begin
            st = list(carry[:STATE])
            corr = fx.Float32(carry[STATE])
            x_cur = carry[STATE + 1 :]
            stage = smod(rel)
            nstage = smod(rel + 1)
            nxt = it + LEAD

            def p_pack(p, b0, b1):
                return [
                    _pin(
                        fx.Vector.from_elements(
                            p[8 * b : 8 * b + 8], dtype=fx.Float32
                        ).to(fx.BFloat16)
                    )
                    for b in range_constexpr(b0, b1)
                ]

            # L1
            if const_expr(VSEG not in (0, 7)):
                phys = tile_phys((nxt < t_end).select(nxt, it))

            def vdma():
                if const_expr(VSEG == 6):
                    tok6 = nxt * TILE
                    if (nxt < t_end) & (tok6 + TILE <= context):
                        fx.rocdl.s_waitcnt(vmcnt=K_COPIES)
                        dma_fire(
                            None,
                            None,
                            v_voff,
                            V_COPIES,
                            smod(rel + LEAD) * STAGE + K_STAGE,
                            phys * v_page_bytes + (tok6 % block_size) * v_tok_bytes,
                            v_rs_all,
                        )
                    elif nxt < t_end:
                        # L2 issued the tile whole.
                        fx.rocdl.s_waitcnt(vmcnt=K_COPIES + V_COPIES)
                    else:
                        fx.rocdl.s_waitcnt(vmcnt=0)
                elif nxt < t_end:
                    fx.rocdl.s_waitcnt(vmcnt=K_COPIES)
                    issue_half(nxt, smod(rel + LEAD), phys, True)

            if fx.Int64(
                fx.rocdl.ballot(T.i64, (corr != fx.Float32(1.0)).ir_value())
            ) != fx.Int64(0):
                corr_b = fx.full(16, corr, fx.Float32)
                st = [st[0], st[1] * corr] + [v * corr_b for v in st[2:]]
            k_ops = k_reads(nstage, 0, 0, KPRE)
            if const_expr(VSEG not in (0, 7)):
                phys = _pin_s32(phys)
            fx.rocdl.s_waitcnt(lgkmcnt=0)
            seg_barrier()
            # C1: tile nxt's DMA addresses in the MFMA shadows.
            hooks = None
            if const_expr(VSEG == 5):
                dmaenv = {}
                tok_n = nxt * TILE

                def prep(values):
                    def f():
                        # Pinned inputs hold the math in this slot.
                        ph = _pin_s32(phys)
                        tk = _pin_s32(tok_n)
                        if const_expr(values):
                            a, n = dma_addr(
                                v_base, v_page_elems, V_ROW, tk, ph, tk % block_size
                            )
                        else:
                            a, n = dma_addr(
                                k_base, k_page_elems, K_ROW, tk, ph, tk % block_size
                            )
                        dmaenv[values] = (
                            _pin_s32(fx.Int32(a & fx.Int64(0xFFFFFFFF))),
                            _pin_s32(fx.Int32(a >> 32)),
                            _pin_s32(n),
                        )

                    return f

                hooks = {2: prep(False), 6: prep(True)}
            s0, p_ops, lsum_ = qk_soft(nstage, 0, k_ops, x_cur[:16], st[1], hooks)
            s_next = [s0, None]
            seg_barrier()

            # L2: the DMA's scalar address math fits beside C1.
            def soff_dma():
                # Own copies retire in issue order: with V split off, only
                # tile x - 1's K must have landed; the V segment waits for
                # its V.
                if const_expr(VSPLIT == 2):
                    fx.rocdl.s_waitcnt(vmcnt=0)
                else:
                    fx.rocdl.s_waitcnt(vmcnt=V_COPIES)
                if nxt < t_end:
                    stage6 = smod(rel + LEAD) * STAGE
                    tok6 = nxt * TILE
                    if tok6 + TILE <= context:
                        ptok6 = tok6 % block_size
                        dma_fire(
                            None,
                            None,
                            k_voff,
                            K_COPIES,
                            stage6,
                            phys * k_page_bytes + ptok6 * k_tok_bytes,
                            k_rs_all,
                        )
                        if const_expr(VSPLIT == 2):
                            dma_fire(
                                None,
                                None,
                                v_voff,
                                V_COPIES,
                                stage6 + K_STAGE,
                                phys * v_page_bytes + ptok6 * v_tok_bytes,
                                v_rs_all,
                            )
                    else:
                        issue_half(nxt, smod(rel + LEAD), phys, False)
                        issue_half(nxt, smod(rel + LEAD), phys, True)

            if const_expr(VSEG == 6):
                soff_dma()
            elif const_expr(VSEG in (7, 8)):
                pass
            elif const_expr(VSEG == 5):
                fx.rocdl.s_waitcnt(vmcnt=0)
                if nxt < t_end:
                    stage_n = smod(rel + LEAD) * STAGE

                    def addr64(lo, hi):
                        return (fx.Int64(hi) << 32) | (
                            fx.Int64(lo) & fx.Int64(0xFFFFFFFF)
                        )

                    ka = dmaenv[False]
                    va = dmaenv[True]
                    dma_fire(addr64(ka[0], ka[1]), ka[2], k_voff, K_COPIES, stage_n)
                    dma_fire(
                        addr64(va[0], va[1]), va[2], v_voff, V_COPIES, stage_n + K_STAGE
                    )
            elif const_expr(VSEG):
                # Own copies retire in issue order: K of tile x - 1 before
                # its V, which the V segment waits for.
                if nxt < t_end:
                    fx.rocdl.s_waitcnt(vmcnt=V_COPIES)
                    issue_half(nxt, smod(rel + LEAD), phys, False)
                else:
                    fx.rocdl.s_waitcnt(vmcnt=0)
                if const_expr(VSEG == 2):
                    vdma()
            else:
                fx.rocdl.s_waitcnt(vmcnt=0)
                if nxt < t_end:
                    issue(nxt, smod(rel + LEAD))
            k_ops = k_reads(nstage, 1, 0, KPRE)
            fx.rocdl.s_waitcnt(lgkmcnt=0)
            seg_barrier()
            # C2
            s1, p_hi, lsum_ = qk_soft(nstage, 1, k_ops, x_cur[16:], lsum_)
            s_next[1] = s1
            p_ops = p_ops + p_hi
            seg_barrier()
            # L3
            half = SNB // 2
            if const_expr(VSEG == 3 or VSPLIT == 3):
                vdma()
            if const_expr(VSEG == 7):
                fx.rocdl.s_waitcnt(vmcnt=0)
                if nxt < t_end:
                    issue(nxt, smod(rel + LEAD))
            if const_expr(VSEG == 8):
                soff_dma()
            v_ops = v_reads(stage, 0, half)
            fx.rocdl.s_waitcnt(lgkmcnt=0)
            seg_barrier()
            # C3: tile it + 1's reference, moved when some lane's max passes
            # the old one by RESCALE_LOG2.
            o = st[2:]
            for j in range_constexpr(half):
                for b in range_constexpr(4):
                    o[j] = mfma32(v_ops[4 * j + b], p_ops[b], o[j])
            # Pinned after the barrier, the scores keep the softmax here.
            if const_expr(ANCHOR):
                s_next = [_pin(x) for x in s_next]
            vals = masked_scores(s_next, it + 1, masked)
            mx = lane_max(vals)
            over = mx > st[0] + fx.Float32(rescale_raw)
            any_over = fx.Int64(fx.rocdl.ballot(T.i64, over.ir_value())) != fx.Int64(0)
            rmax = fx.maxnumf(mx, mx.shuffle_xor(32, WAVE))
            m_new = any_over.select(fx.maxnumf(st[0], rmax), st[0])
            corr = fx.Float32(
                fx.exp2(
                    st[0] * fx.Float32(scale_log2) - scaled_ref(m_new), fastmath="fast"
                )
            )
            red = _pin(fx.Vector.from_elements([m_new, corr], dtype=fx.Float32))
            m_new = red[0]
            corr = red[1]
            interleave(4 * half, 16 + 12 + (64 if masked else 0), 1)
            seg_barrier()
            # L4
            if const_expr(VSEG == 4 or VSPLIT == 4):
                vdma()
            v_ops = v_reads(stage, half, SNB)
            fx.rocdl.s_waitcnt(lgkmcnt=0)
            seg_barrier()
            # C4
            for j in range_constexpr(half):
                for b in range_constexpr(4):
                    o[half + j] = mfma32(v_ops[4 * j + b], p_ops[b], o[half + j])
            if const_expr(ANCHOR):
                m_new = _pin_f32(m_new)
            neg_ref = -scaled_ref(m_new)
            x_next = [fx.fma(v, fx.Float32(scale_log2), neg_ref) for v in vals]
            interleave(4 * half, 33, 0)
            x_next = [
                _pin(
                    fx.Vector.from_elements(x_next[8 * c : 8 * c + 8], dtype=fx.Float32)
                )
                for c in range_constexpr(4)
            ]
            x_next = [x[i] for x in x_next for i in range_constexpr(8)]
            seg_barrier()
            return [m_new, lsum_] + o + [corr] + x_next

        def fused_step(it, carry, masked):
            # One wave a SIMD: softmax of tile i shares a block with tile
            # i + 1's QK, so its VALU fills the MFMA shadows. The barrier
            # retires tile i - 1's stage (PV i - 1, QK i done) and, after
            # its issuers' wait, publishes tile i + 1.
            rel = it - t_begin
            state = carry[: 2 * M + M * NB]
            s_flat = carry[2 * M + M * NB :]
            if issuer == (rel + 1) % ISSUERS:
                fx.rocdl.s_waitcnt(vmcnt=0)
            gpu.barrier()
            nxt = it + (stages - 1)
            if (issuer == rel % ISSUERS) & (nxt < t_end):
                issue(nxt, (rel + (stages - 1)) % stages)
            if const_expr(fsched >= 20):
                # Hand interleave: one score's scale and exp after each of
                # tile i + 1's QK MFMAs.
                st, p_flat, steps = soft(it, s_flat, state, masked, deferred=True)
                s_next = qk((rel + 1) % stages, steps, fsched - 19)
            else:
                st, p_flat = soft(it, s_flat, state, masked)
                # The last tile's lookahead QK reads a stale stage; it is unused.
                s_next = qk((rel + 1) % stages)
            state = pv(rel % stages, p_flat, st)
            qk_mfma = M * KK * CH
            pv_mfma = M * NB * max(CH // 2, 1) + (M * max(CH // 2, 1) if lsum else 0)
            if const_expr(fsched >= 20):
                pass
            elif const_expr(fsched >= 10):
                # Per score: scale (VALU), MFMA, exp (TRANS); LDS reads every
                # other group; then the rest of the MFMAs with packing VALU.
                n_exp = M * CH * 4
                fx.rocdl.sched_group_barrier(fx.rocdl.mask_dsrd, 4, 0)
                for i in range_constexpr(n_exp):
                    fx.rocdl.sched_group_barrier(0x002, fsched - 9, 0)
                    fx.rocdl.sched_group_barrier(fx.rocdl.mask_mfma, 1, 0)
                    fx.rocdl.sched_group_barrier(0x400, 1, 0)
                    if const_expr(i % 2 == 0):
                        fx.rocdl.sched_group_barrier(fx.rocdl.mask_dsrd, 1, 0)
                for i in range_constexpr(qk_mfma + pv_mfma - n_exp):
                    fx.rocdl.sched_group_barrier(fx.rocdl.mask_mfma, 1, 0)
                    fx.rocdl.sched_group_barrier(0x002, 1, 0)
                    fx.rocdl.sched_group_barrier(fx.rocdl.mask_dsrd, 1, 0)
            elif const_expr(fsched):
                # Softmax VALU under QK's MFMAs, LDS reads spread between.
                fx.rocdl.sched_group_barrier(fx.rocdl.mask_dsrd, 4, 0)
                for i in range_constexpr(qk_mfma):
                    fx.rocdl.sched_group_barrier(fx.rocdl.mask_mfma, 1, 0)
                    fx.rocdl.sched_group_barrier(0x2, fsched, 0)
                    if const_expr(i % 2 == 0):
                        fx.rocdl.sched_group_barrier(fx.rocdl.mask_dsrd, 1, 0)
                for i in range_constexpr(pv_mfma):
                    fx.rocdl.sched_group_barrier(fx.rocdl.mask_mfma, 1, 0)
                    fx.rocdl.sched_group_barrier(0x2, 1, 0)
                    fx.rocdl.sched_group_barrier(fx.rocdl.mask_dsrd, 1, 0)
            return list(state) + list(s_next)

        if const_expr(fuse):
            if issuer == 0:
                fx.rocdl.s_waitcnt(vmcnt=0)
            gpu.barrier()
            carry = list(init) + list(qk(0))
            for tile_i, c_in in range(t_begin, t_split, 1, init=carry):
                c_mid = yield fused_step(fx.Int32(tile_i), c_in, False)
            for tile_i, c_in in range(t_split, t_end, 1, init=c_mid):
                c_out = yield fused_step(fx.Int32(tile_i), c_in, True)
            results = c_out[: 2 * M + M * NB]
        elif const_expr(mf32 == 2):
            # Prologue: the first LEAD tiles land, every wave takes tile 0's
            # scores and reference, then group 1 falls one segment behind.
            fx.rocdl.s_waitcnt(vmcnt=0)
            gpu.barrier()
            vals0 = masked_scores(qk32(0), t_begin, True)
            st0 = rescale32(init, lane_max(vals0))
            neg_ref0 = -scaled_ref(st0[0])
            x0 = [fx.fma(v, fx.Float32(scale_log2), neg_ref0) for v in vals0]
            if grp == 1:
                gpu.barrier()
            carry = st0 + [fx.Float32(1.0)] + x0
            # Tile i's step masks tile i + 1's scores.
            u_end = t_split - 1
            u_end = (u_end > t_begin).select(u_end, t_begin)
            # In pairs, so the two score sets alternate registers.
            u_pairs = t_begin + ((u_end - t_begin) >> 1) * 2
            for tile_i, c_in in range(t_begin, u_pairs, 2, init=carry):
                c_one = opus_step(fx.Int32(tile_i), c_in, False)
                c_two = yield opus_step(fx.Int32(tile_i) + 1, c_one, False)
            c_mid = c_two
            if u_pairs < u_end:
                c_mid = opus_step(u_pairs, c_two, False)
            for tile_i, c_in in range(u_end, t_end, 1, init=c_mid):
                c_out = yield opus_step(fx.Int32(tile_i), c_in, True)
            # The last step's pending correction.
            corr_out = fx.Float32(c_out[STATE])
            corr_b = fx.full(16, corr_out, fx.Float32)
            results = [c_out[0], c_out[1] * corr_out] + [
                v * corr_b for v in c_out[2:STATE]
            ]
            if grp == 0:
                gpu.barrier()
        elif const_expr(stagger):
            # Prologue: tile 0 ready for group 0's QK, then group 1 falls one
            # phase behind.
            if grp == 0:
                fx.rocdl.s_waitcnt(vmcnt=0)
            gpu.barrier()
            if grp == 1:
                gpu.barrier()
            if const_expr(mf32):
                s0 = qk32(0)
            else:
                s0 = qk(0)
            if grp == 1:
                fx.rocdl.s_waitcnt(vmcnt=0)
            gpu.barrier()
            carry = list(init) + list(s0)
            if const_expr(unroll):
                # Per-group code, unmasked tiles in pairs: group and parity
                # tests fold at compile time.
                t_pairs = t_begin + ((t_split - t_begin) >> 1) * 2
                c_out = carry
                if const_expr(unroll == 2):
                    for tile_i, c_in in range(t_begin, t_pairs, 2, init=carry):
                        c_one = stagger_step(fx.Int32(tile_i), c_in, False, None, 0)
                        c_two = yield stagger_step(
                            fx.Int32(tile_i) + 1, c_one, False, None, 1
                        )
                    c_rem = c_two
                    if t_pairs < t_split:
                        c_rem = stagger_step(t_pairs, c_two, False, None, 0)
                    for tile_i, c_in in range(t_split, t_end, 1, init=c_rem):
                        c_out = yield stagger_step(fx.Int32(tile_i), c_in, True)
                else:
                    # Both groups' loops run in sequence, each empty for the
                    # other group's waves. Spelled out: an scf.if per group
                    # breaks dominance of values the helpers cache, and
                    # assignments in a range_constexpr body do not escape it.
                    mine0 = grp == 0
                    pairs0 = mine0.select(t_pairs, t_begin)
                    for tile_i, c_in in range(t_begin, pairs0, 2, init=carry):
                        c_one = stagger_step(fx.Int32(tile_i), c_in, False, 0, 0)
                        c_p0 = yield stagger_step(
                            fx.Int32(tile_i) + 1, c_one, False, 0, 1
                        )
                    c_r0 = c_p0
                    if mine0 & (t_pairs < t_split):
                        c_r0 = stagger_step(t_pairs, c_p0, False, 0, 0)
                    end0 = mine0.select(t_end, t_split)
                    for tile_i, c_in in range(t_split, end0, 1, init=c_r0):
                        c_m0 = yield stagger_step(fx.Int32(tile_i), c_in, True, 0)
                    mine1 = grp == 1
                    pairs1 = mine1.select(t_pairs, t_begin)
                    for tile_i, c_in in range(t_begin, pairs1, 2, init=c_m0):
                        c_one = stagger_step(fx.Int32(tile_i), c_in, False, 1, 0)
                        c_p1 = yield stagger_step(
                            fx.Int32(tile_i) + 1, c_one, False, 1, 1
                        )
                    c_r1 = c_p1
                    if mine1 & (t_pairs < t_split):
                        c_r1 = stagger_step(t_pairs, c_p1, False, 1, 0)
                    end1 = mine1.select(t_end, t_split)
                    for tile_i, c_in in range(t_split, end1, 1, init=c_r1):
                        c_m1 = yield stagger_step(fx.Int32(tile_i), c_in, True, 1)
                    c_out = c_m1
            else:
                for tile_i, c_in in range(t_begin, t_split, 1, init=carry):
                    c_mid = yield stagger_step(fx.Int32(tile_i), c_in, False)
                for tile_i, c_in in range(t_split, t_end, 1, init=c_mid):
                    c_out = yield stagger_step(fx.Int32(tile_i), c_in, True)
            results = c_out[:STATE]
            if grp == 0:
                gpu.barrier()
        else:
            for tile_i, state in range(t_begin, t_split, 1, init=init):
                interior = yield step(fx.Int32(tile_i), state, False)
            for tile_i, state in range(t_split, t_end, 1, init=interior):
                results = yield step(fx.Int32(tile_i), state, True)

        final = results
        fx.rocdl.s_waitcnt(vmcnt=0)
        gpu.barrier()

        M_FIN = 0 if mf32 else M
        m_fin = [final[m] * fx.Float32(scale_log2) for m in range_constexpr(M_FIN)]
        l_fin = []
        for m in range_constexpr(M_FIN):
            lr = final[M + m]
            if const_expr(lsum):
                # Each of the four lane groups holds the whole sum.
                lr = lr * fx.Float32(0.25)
            for sh in (16, 32):
                lr = lr + lr.shuffle_xor(sh, WAVE)
            l_fin.append(lr)
        o_fin = [
            [final[2 * M + m * NB + nb] for nb in range_constexpr(NB)]
            for m in range_constexpr(M_FIN)
        ]

        def scratch(slot_i):
            return ((slot_i - 1) * R + rb) * MERGE_RB

        def write_partial():
            base = scratch(ts)
            for m in range_constexpr(M):
                for nb in range_constexpr(NB):
                    lds.store(
                        base + ((m * 16 + l16) * DV + nb * 16 + g * 4) * 4,
                        fx.Float32,
                        o_fin[m][nb],
                    )
                # Every lane group holds the row's m and l; duplicate stores agree.
                lds.store(
                    base + MERGE_O + (m * 16 + l16) * 4,
                    fx.Float32,
                    fx.Vector.from_elements([m_fin[m]], dtype=fx.Float32),
                )
                lds.store(
                    base + MERGE_O + M * 64 + (m * 16 + l16) * 4,
                    fx.Float32,
                    fx.Vector.from_elements([l_fin[m]], dtype=fx.Float32),
                )

        def merge_store():
            partial = kv_h * work_capacity + slot
            # Per-partial resources: padded rows land past them and are dropped.
            pout_rs = create_buffer_resource_from_addr(
                _uni64(buf_base_i64(pout_ptr) + fx.Int64(partial) * (ROWS * DV * 2)),
                num_records_bytes=ROWS * DV * 2,
            )
            if const_expr(window):
                # The sequence's valid query rows of out; padded rows drop.
                row_bytes = n_kv * (GQA * DV * 2)
                pout_rs = create_buffer_resource_from_addr(
                    _uni64(
                        buf_base_i64(out_ptr)
                        + fx.Int64(first_row) * fx.Int64(row_bytes)
                    ),
                    num_records_bytes=fx.Int64(_uni(valid * row_bytes)),
                )
            pmax_rs = create_buffer_resource_from_addr(
                _uni64(buf_base_i64(pmax_ptr) + fx.Int64(partial) * (ROWS * 4)),
                num_records_bytes=ROWS * 4,
            )
            psum_rs = create_buffer_resource_from_addr(
                _uni64(buf_base_i64(psum_ptr) + fx.Int64(partial) * (ROWS * 4)),
                num_records_bytes=ROWS * 4,
            )
            soff = fx.Int32(0).ir_value()
            for m in range_constexpr(M):
                ms = [m_fin[m]]
                ls = [l_fin[m]]
                for j in range_constexpr(1, S):
                    base = scratch(j)
                    ms.append(
                        lds.load(base + MERGE_O + (m * 16 + l16) * 4, fx.Float32, 1)[0]
                    )
                    ls.append(
                        lds.load(
                            base + MERGE_O + M * 64 + (m * 16 + l16) * 4, fx.Float32, 1
                        )[0]
                    )
                mm = ms[0]
                for v in ms[1:]:
                    mm = fx.maxnumf(mm, v)
                if const_expr(SINK_T is not None):
                    # The sink is one more zero-value logit of the row.
                    sink = fx.Float32(
                        fx.ptr_load(
                            fx.add_offset(
                                fx.recast_iter(SINK_T, sinks_ptr), kv_h * GQA + l16
                            ),
                            result_type=fx.Vector.make_type(1, SINK_T),
                        )[0]
                    ) * fx.Float32(LOG2E)
                    mm = fx.maxnumf(mm, sink)
                safe = (mm > neg_inf).select(mm, zero)
                ws = [fx.Float32(fx.exp2(v - safe, fastmath="fast")) for v in ms]
                lt = ls[0] * ws[0]
                for j in range_constexpr(1, S):
                    lt = lt + ls[j] * ws[j]
                if const_expr(SINK_T is not None):
                    lt = lt + fx.Float32(fx.exp2(sink - safe, fastmath="fast"))
                inv = (lt > zero).select(fx.Float32(rcp_f32(lt)), zero)
                wb = [fx.full(4, w * inv, fx.Float32) for w in ws]
                row = queries[m] * GQA + l16
                if const_expr(window):
                    # Row of query q, head h in out [tokens, KV heads * GQA, DV].
                    row = (queries[m] * n_kv + kv_h) * GQA + l16
                for nb in range_constexpr(NB):
                    o = o_fin[m][nb] * wb[0]
                    for j in range_constexpr(1, S):
                        o1 = lds.load(
                            scratch(j) + ((m * 16 + l16) * DV + nb * 16 + g * 4) * 4,
                            fx.Float32,
                            4,
                        )
                        o = o + o1 * wb[j]
                    mlir_rocdl.raw_ptr_buffer_store(
                        o.to(fx.BFloat16).ir_value(),
                        pout_rs,
                        fx.Int32((row * DV + nb * 16 + g * 4) * 2).ir_value(),
                        soff,
                        aux=_aux(OAUX),
                    )
                if const_expr(not window):
                    mlir_rocdl.raw_ptr_buffer_store(
                        fx.Float32(mm * fx.Float32(1.0 / LOG2E)).ir_value(),
                        pmax_rs,
                        fx.Int32(row * 4).ir_value(),
                        soff,
                        aux=_aux(0),
                    )
                    mlir_rocdl.raw_ptr_buffer_store(
                        fx.Float32(lt).ir_value(),
                        psum_rs,
                        fx.Int32(row * 4).ir_value(),
                        soff,
                        aux=_aux(0),
                    )

        def store32():
            partial = kv_h * work_capacity + slot
            pout_rs = create_buffer_resource_from_addr(
                _uni64(buf_base_i64(pout_ptr) + fx.Int64(partial) * (ROWS * DV * 2)),
                num_records_bytes=ROWS * DV * 2,
            )
            pmax_rs = create_buffer_resource_from_addr(
                _uni64(buf_base_i64(pmax_ptr) + fx.Int64(partial) * (ROWS * 4)),
                num_records_bytes=ROWS * 4,
            )
            psum_rs = create_buffer_resource_from_addr(
                _uni64(buf_base_i64(psum_ptr) + fx.Int64(partial) * (ROWS * 4)),
                num_records_bytes=ROWS * 4,
            )
            soff = fx.Int32(0).ir_value()
            lt = final[1] + final[1].shuffle_xor(32, WAVE)
            inv = fx.full(
                4,
                fx.Float32((lt > zero).select(fx.Float32(rcp_f32(lt)), zero)),
                fx.Float32,
            )
            row = rb * (M * 16) + n32
            for dblk in range_constexpr(SNB):
                o = final[2 + dblk]
                for j in range_constexpr(4):
                    part = fx.Vector.from_elements(
                        [o[4 * j + i] for i in range_constexpr(4)], dtype=fx.Float32
                    )
                    mlir_rocdl.raw_ptr_buffer_store(
                        (part * inv).to(fx.BFloat16).ir_value(),
                        pout_rs,
                        fx.Int32(
                            (row * DV + dblk * 32 + 8 * j + 4 * h32) * 2
                        ).ir_value(),
                        soff,
                        aux=_aux(0),
                    )
            # Both lane halves hold the row's m and l; duplicate stores agree.
            mlir_rocdl.raw_ptr_buffer_store(
                fx.Float32(final[0] * fx.Float32(scale_log2 / LOG2E)).ir_value(),
                pmax_rs,
                fx.Int32(row * 4).ir_value(),
                soff,
                aux=_aux(0),
            )
            mlir_rocdl.raw_ptr_buffer_store(
                fx.Float32(lt).ir_value(),
                psum_rs,
                fx.Int32(row * 4).ir_value(),
                soff,
                aux=_aux(0),
            )

        if const_expr(mf32):
            store32()
        elif const_expr(S > 1):
            if ts > 0:
                write_partial()
            gpu.barrier()
            if ts == 0:
                merge_store()
        else:
            merge_store()
        # A fused pair's or next task's ring fill reuses the merge scratch.
        gpu.barrier()

    @flyc.kernel(name=kernel_name, known_block_size=(NWAVES * WAVE, 1, 1))
    def pa_decode_pingpong_kernel(
        pmax_ptr: fx.Pointer,
        psum_ptr: fx.Pointer,
        pout_ptr: fx.Pointer,
        query_ptr: fx.Pointer,
        key_cache_ptr: fx.Pointer,
        value_cache_ptr: fx.Pointer,
        block_tables_ptr: fx.Pointer,
        max_blocks_per_seq: fx.Int32,
        stride_q_row: fx.Int32,
        stride_q_head: fx.Int32,
        work_info_ptr: fx.Pointer,
        work_capacity: fx.Int32,
        query_map_ptr: fx.Pointer,
    ):
        work = fx.recast_iter(fx.Int32, work_info_ptr)

        def load_task(slot):
            return fx.ptr_load(
                fx.add_offset(work, slot * 4),
                result_type=fx.Vector.make_type(4, fx.Int32),
            )

        def run(slot, task, start, end):
            _task(
                pmax_ptr,
                psum_ptr,
                pout_ptr,
                query_ptr,
                key_cache_ptr,
                value_cache_ptr,
                block_tables_ptr,
                query_map_ptr,
                max_blocks_per_seq,
                stride_q_row,
                stride_q_head,
                fx.Int32(task[0]),
                start,
                end,
                fx.Int32(task[3]),
                slot,
                work_capacity,
                pout_ptr,
                pout_ptr,
                fx.Int32(0),
            )

        def empty_partial(slot):
            # Zero weight and zero output: the reducer scales output by weight.
            tid = fx.Int32(gpu.thread_id("x"))
            partial = fx.Int32(gpu.block_id("y")) * work_capacity + slot
            row = partial * ROWS + tid % ROWS
            fx.ptr_store(
                fx.Vector.from_elements([fx.Float32(float("-inf"))], dtype=fx.Float32),
                fx.add_offset(fx.recast_iter(fx.Float32, pmax_ptr), row),
            )
            fx.ptr_store(
                fx.Vector.from_elements([fx.Float32(0.0)], dtype=fx.Float32),
                fx.add_offset(fx.recast_iter(fx.Float32, psum_ptr), row),
            )
            pout = fx.recast_iter(fx.BFloat16, pout_ptr)
            chunks = ROWS * DV // 8
            for j in range_constexpr(-(-chunks // (NWAVES * WAVE))):
                idx = j * NWAVES * WAVE + tid
                # Overflow lanes rewrite the last chunk with the same zeros.
                idx = (idx < chunks).select(idx, chunks - 1)
                fx.ptr_store(
                    fx.full(8, 0.0, fx.BFloat16),
                    fx.add_offset(pout, partial * (ROWS * DV) + idx * 8),
                )

        if const_expr(pair_tasks):
            # Slots 2x and 2x + 1; a sequence's adjacent tasks run as one range.
            first = fx.Int32(gpu.block_id("x")) * 2
            second = first + 1
            has_second = second < work_capacity
            ta = load_task(first)
            tb = load_task(has_second.select(second, first))
            a_start = fx.Int32(ta[1])
            a_end = fx.Int32(ta[2])
            b_start = fx.Int32(tb[1])
            b_end = fx.Int32(tb[2])
            fuse = (
                has_second
                & (fx.Int32(ta[0]) == fx.Int32(tb[0]))
                & (a_end == b_start)
                & (a_start < a_end)
                & (b_start < b_end)
            )
            # One _task call site (it owns the LDS allocation): run 0 is the
            # first slot or the fused range, run 1 the unfused second slot.
            no_work = fx.Int32(0)
            second_live = has_second & ~fuse
            first_end = fuse.select(b_end, a_end)
            second_start = second_live.select(b_start, no_work)
            second_end = second_live.select(b_end, no_work)
            for run_i in range(0, 2, 1):
                is_first = fx.Int32(run_i) == 0
                start = is_first.select(a_start, second_start)
                end = is_first.select(first_end, second_end)
                if start < end:
                    _task(
                        pmax_ptr,
                        psum_ptr,
                        pout_ptr,
                        query_ptr,
                        key_cache_ptr,
                        value_cache_ptr,
                        block_tables_ptr,
                        query_map_ptr,
                        max_blocks_per_seq,
                        stride_q_row,
                        stride_q_head,
                        is_first.select(fx.Int32(ta[0]), fx.Int32(tb[0])),
                        start,
                        end,
                        is_first.select(fx.Int32(ta[3]), fx.Int32(tb[3])),
                        is_first.select(first, second),
                        work_capacity,
                        pout_ptr,
                        pout_ptr,
                        fx.Int32(0),
                    )
            if fuse:
                empty_partial(second)
        else:
            slot = fx.Int32(gpu.block_id("x"))
            task = load_task(slot)
            start = fx.Int32(task[1])
            end = fx.Int32(task[2])
            if start < end:
                run(slot, task, start, end)

    def _grid_x(work_capacity):
        # Outside the JIT rewriter, which would make the branch dynamic.
        if pair_tasks:
            return (work_capacity + 1) // 2
        return work_capacity

    @flyc.jit
    def pa_decode_pingpong_launch(
        pmax: fx.Pointer,
        psum: fx.Pointer,
        pout: fx.Pointer,
        query: fx.Pointer,
        key_cache: fx.Pointer,
        value_cache: fx.Pointer,
        block_tables: fx.Pointer,
        key_scale: fx.Pointer,
        value_scale: fx.Pointer,
        max_blocks_per_seq: fx.Int32,
        num_seqs: fx.Int32,
        num_kv_heads: fx.Int32,
        stride_ks_block: fx.Int32,
        stride_ks_head: fx.Int32,
        stride_q_row: fx.Int32,
        stride_q_head: fx.Int32,
        work_info: fx.Pointer,
        work_capacity: fx.Int32,
        query_map: fx.Pointer,
        stream: fx.Stream = fx.Stream(None),  # noqa: B008
    ):
        _ = kernel_name
        with CompilationContext.compile_hints({"fastmath": "contract"}):
            pa_decode_pingpong_kernel(
                pmax,
                psum,
                pout,
                query,
                key_cache,
                value_cache,
                block_tables,
                max_blocks_per_seq,
                stride_q_row,
                stride_q_head,
                work_info,
                work_capacity,
                query_map,
                **(
                    {
                        "value_attrs": {
                            "llvm.target_features": ir.Attribute.parse(
                                '#llvm.target_features<["-packed-fp32-ops"]>'
                            )
                        }
                    }
                    if nopk
                    else {}
                ),
            ).launch(
                grid=(_grid_x(work_capacity), num_kv_heads, 1),
                block=(NWAVES * WAVE, 1, 1),
                stream=stream,
            )

    @flyc.kernel(name=kernel_name, known_block_size=(NWAVES * WAVE, 1, 1))
    def pa_decode_pingpong_window_kernel(
        out_ptr: fx.Pointer,
        query_ptr: fx.Pointer,
        key_cache_ptr: fx.Pointer,
        value_cache_ptr: fx.Pointer,
        block_tables_ptr: fx.Pointer,
        max_blocks_per_seq: fx.Int32,
        stride_q_row: fx.Int32,
        stride_q_head: fx.Int32,
        contexts_ptr: fx.Pointer,
        sinks_ptr: fx.Pointer,
        query_map_ptr: fx.Pointer,
        num_kv_heads: fx.Int32,
    ):
        seq = fx.Int32(gpu.block_id("x"))
        context = _uni(
            fx.Int32(
                fx.ptr_load(
                    fx.add_offset(
                        fx.recast_iter(fx.Int32, contexts_ptr),
                        seq // SPLIT if const_expr(SPLIT > 1) else seq,
                    ),
                    result_type=fx.Vector.make_type(1, fx.Int32),
                )[0]
            )
        )
        if const_expr(SPLIT > 1):
            # Query block j of a sequence ends SPLIT - 1 - j blocks early.
            context = context - (SPLIT - 1 - seq % SPLIT) * QLEN
        # Unguarded, so the loads overlap: an empty pseudo-sequence runs no
        # tiles and its zero-record output resource drops the stores.
        _task(
            out_ptr,
            out_ptr,
            out_ptr,
            query_ptr,
            key_cache_ptr,
            value_cache_ptr,
            block_tables_ptr,
            query_map_ptr,
            max_blocks_per_seq,
            stride_q_row,
            stride_q_head,
            seq,
            fx.Int32(0),
            fx.Int32(0),
            context,
            fx.Int32(0),
            fx.Int32(0),
            out_ptr,
            sinks_ptr,
            num_kv_heads,
        )

    @flyc.jit
    def pa_decode_pingpong_window_launch(
        out: fx.Pointer,
        query: fx.Pointer,
        key_cache: fx.Pointer,
        value_cache: fx.Pointer,
        block_tables: fx.Pointer,
        max_blocks_per_seq: fx.Int32,
        stride_q_row: fx.Int32,
        stride_q_head: fx.Int32,
        contexts: fx.Pointer,
        sinks: fx.Pointer,
        query_map: fx.Pointer,
        num_seqs: fx.Int32,
        num_kv_heads: fx.Int32,
        stream: fx.Stream = fx.Stream(None),  # noqa: B008
    ):
        _ = kernel_name
        with CompilationContext.compile_hints({"fastmath": "contract"}):
            pa_decode_pingpong_window_kernel(
                out,
                query,
                key_cache,
                value_cache,
                block_tables,
                max_blocks_per_seq,
                stride_q_row,
                stride_q_head,
                contexts,
                sinks,
                query_map,
                num_kv_heads,
                **(
                    {
                        "value_attrs": {
                            "llvm.target_features": ir.Attribute.parse(
                                '#llvm.target_features<["-packed-fp32-ops"]>'
                            )
                        }
                    }
                    if nopk
                    else {}
                ),
            ).launch(
                grid=(num_seqs, num_kv_heads, 1),
                block=(NWAVES * WAVE, 1, 1),
                stream=stream,
            )

    if window:
        pa_decode_pingpong_launch = pa_decode_pingpong_window_launch
        pa_decode_pingpong_kernel = pa_decode_pingpong_window_kernel
    if NWAVES == 8:
        pa_decode_pingpong_launch.compile_hints["waves_per_eu"] = 2
    if const_expr(nopk):
        # Packed FP32 softmax VALU stalls behind the sibling wave's MFMAs.
        pa_decode_pingpong_launch.compile_hints["target_features"] = "-packed-fp32-ops"
    compiled = {
        "launch": pa_decode_pingpong_launch,
        "kernel": pa_decode_pingpong_kernel,
        "query_map": True,
    }
    _CACHE[key] = compiled
    return compiled
