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
from flydsl._mlir.dialects import rocdl as mlir_rocdl
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import const_expr, gpu, range_constexpr
from flydsl.expr.typing import T

from .buffer_ops import create_buffer_resource_from_addr
from .pa_decode.op_lds import PaDecodeLds
from .pa_decode.traits import KV_COMPUTE_BLOCK, LOG2E, WAVE
from .tensor_shim import buf_base_i64, buf_scalar_load, ptr_buf_tensor
from .utils import rcp_f32

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
    return 2, 4, 2


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
):
    stages = stages or int(os.environ.get("PP_STAGES", "3"))
    prio = int(os.environ.get("PP_PRIO", "0"))
    pair_ctas = int(os.environ.get("PP_PAIR", pair_ctas))
    pair_tasks = pair_ctas > 0
    nocompute = int(os.environ.get("PP_NOCOMPUTE", "0"))
    dma_aux = int(os.environ.get("PP_AUX", "2"))
    M, R, CH = pingpong_layout(query_length)
    SLOT_TOKENS = 16 * CH
    S = TILE // SLOT_TOKENS
    NWAVES = R * S
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
        pair_ctas,
        nocompute,
        dma_aux,
        (M, R, CH),
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
    ISSUERS = stages - 1
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
        f"_m{M}r{R}c{CH}_s{stages}_p{prio}_c{pair_ctas}_n{nocompute}_a{dma_aux}"
    )

    @flyc.jit
    def _task(
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
        seq: fx.Int32,
        start: fx.Int32,
        end: fx.Int32,
        context: fx.Int32,
        slot: fx.Int32,
        work_capacity: fx.Int32,
    ):
        lds = PaDecodeLds(_LdsTraits, SharedStorage)
        lds.allocate()
        kv_h = fx.Int32(gpu.block_id("y"))
        n_kv = fx.Int32(gpu.grid_dim.y)
        tid = fx.Int32(gpu.thread_id("x"))
        warp = fx.Int32(fx.rocdl.readfirstlane(T.i32, tid // WAVE))
        lane = tid % WAVE
        g = lane // 16
        l16 = lane % 16
        ts = warp // R
        rb = warp % R
        issuer = warp // GROUP_WAVES
        group_warp = warp % GROUP_WAVES
        neg_inf = fx.Float32(float("-inf"))
        zero = fx.Float32(0.0)
        # Query of each of the wave's M tiles; padded tiles reuse the last query.
        queries = [rb * M + m for m in range_constexpr(M)]

        # Q: B operand of QK, lane (g, l16) holds head l16, dims 32kk + 8g.
        q_buf = ptr_buf_tensor(query_ptr, fx.BFloat16)
        q_tiled = fx.logical_divide(q_buf, fx.make_layout(1, 1))
        q_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.BFloat16)
        q_reg = fx.make_rmem_tensor(fx.make_layout(8, 1), fx.BFloat16)
        q_ops = []
        for m in range_constexpr(M):
            q_idx = queries[m]
            if const_expr(R * M > QLEN):
                q_idx = (q_idx < QLEN).select(q_idx, QLEN - 1)
            q_row = seq * QLEN + q_idx
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
        unmasked = context - (QLEN - 1)
        unmasked = (unmasked > 0).select(unmasked, 0) // TILE
        t_split = (unmasked < t_end).select(unmasked, t_end)
        t_split = (t_split > t_begin).select(t_split, t_begin)

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
        v_voff = dma_offsets(V_COPIES, V_CPR, v_tok_bytes, lambda t: (t & 7) << 1)

        def dma(base, page_elems, row, voffs, copies, lds_off, tok0, phys, page_tok0):
            valid = context - tok0
            valid = (valid < TILE).select(valid, TILE)
            addr = (
                base
                + (
                    fx.Int64(phys) * page_elems
                    + fx.Int64((page_tok0 * n_kv + kv_h) * (row // 2))
                )
                * 2
            )
            # Rows at or past the context read as zero.
            nbytes = (valid - 1) * n_kv * row + row
            rs = create_buffer_resource_from_addr(
                _uni64(addr), num_records_bytes=fx.Int64(_uni(nbytes))
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
                    fx.Int32(0).ir_value(),
                    fx.Int32(0).ir_value(),
                    aux=_aux(dma_aux),
                )

        def issue(tile, stage):
            tok0 = tile * TILE
            phys = fx.Int32(
                buf_scalar_load(bt, seq * max_blocks_per_seq + tok0 // block_size)
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
            if (issuer == s) & (t_begin + s < t_end):
                issue(t_begin + s, s)

        # Loop-invariant LDS read offsets within a stage.
        kq = l16 // 4
        qq = l16 % 4
        k_rd = []
        v_rd = []
        for c in range_constexpr(CH):
            tok = ts * SLOT_TOKENS + c * 16 + l16
            swz = (tok >> 1) & 7
            k_rd.append(
                [tok * K_ROW + ((kk * 4 + g) ^ swz) * 16 for kk in range_constexpr(KK)]
            )
            vtok = ts * SLOT_TOKENS + c * 16 + g * 4 + kq
            vswz = ((g * 4 + kq) & 7) << 1
            v_rd.append(
                [
                    K_STAGE
                    + vtok * V_ROW
                    + ((nb * 2 + qq // 2) ^ vswz) * 16
                    + (qq % 2) * 8
                    for nb in range_constexpr(NB)
                ]
            )

        # Raw-score state [m (M), l (M), O (M * NB)]: m is a row-uniform
        # reference, rescaled only when a score exceeds it by RESCALE_LOG2 in
        # the log2 domain; l is per lane.
        init = (
            [neg_inf] * M
            + [zero] * M
            + [fx.full(4, 0.0, fx.Float32) for _ in range_constexpr(M * NB)]
        )
        limit = [context - QLEN + queries[m] for m in range_constexpr(M)]
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

            stage_off = stage * STAGE
            if const_expr(nocompute):
                return list(state)

            # QK: lane (g, l16) of chunk c holds token slot + 16c + 4g + i, row l16.
            s_acc = [[None] * CH for _ in range_constexpr(M)]
            if const_expr(prio):
                fx.rocdl.s_setprio(1)
            for c in range_constexpr(CH):
                for m in range_constexpr(M):
                    s_acc[m][c] = fx.full(4, 0.0, fx.Float32)
                for kk in range_constexpr(KK):
                    w = lds.load(stage_off + k_rd[c][kk], fx.Int64, 2)
                    k_op = w.bitcast(fx.BFloat16)
                    for m in range_constexpr(M):
                        s_acc[m][c] = fx.Vector(
                            fx.rocdl.mfma_f32_16x16x32_bf16(
                                T.f32x4, [k_op, q_ops[m][kk], s_acc[m][c], 0, 0, 0]
                            )
                        )
            if const_expr(prio):
                fx.rocdl.s_setprio(0)

            lane_tok = it * TILE + ts * SLOT_TOKENS + g * 4
            scores = []
            lmax = []
            for m in range_constexpr(M):
                vals = []
                for c in range_constexpr(CH):
                    for i in range_constexpr(4):
                        v = fx.Float32(s_acc[m][c][i])
                        if const_expr(masked):
                            v = (fx.Int32(c * 16 + i) <= limit[m] - lane_tok).select(
                                v, neg_inf
                            )
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

            p_ops = []
            l_new = []
            for m in range_constexpr(M):
                neg_ref = -scaled_ref(st[m])
                p = [
                    fx.Float32(
                        fx.exp2(
                            fx.fma(v, fx.Float32(scale_log2), neg_ref), fastmath="fast"
                        )
                    )
                    for v in scores[m]
                ]
                psum = st[M + m]
                for v in p:
                    psum = psum + v
                l_new.append(psum)
                p_ops.append(
                    fx.Vector.from_elements(p, dtype=fx.Float32).to(fx.BFloat16)
                )

            # PV: V^T (dims x tokens) by P (tokens x rows); the transpose read
            # gives lane j dim j of tokens 4g + k, k = 0..3, in its 16-lane group.
            o_acc = [
                [st[2 * M + m * NB + nb] for nb in range_constexpr(NB)]
                for m in range_constexpr(M)
            ]
            if const_expr(prio):
                fx.rocdl.s_setprio(1)
            for nb in range_constexpr(NB):
                parts = []
                for c in range_constexpr(CH):
                    r = fx.rocdl.ds_read_tr16_b64(
                        fx.Vector.make_type(4, fx.BFloat16),
                        fx.to_llvm_ptr(
                            lds.pointer(stage_off + v_rd[c][nb], fx.BFloat16)
                        ),
                    ).result
                    parts.append(fx.Vector(r))
                for m in range_constexpr(M):
                    if const_expr(CH == 2):
                        v_op = fx.Vector.from_elements(
                            [
                                parts[c][i]
                                for c in range_constexpr(2)
                                for i in range_constexpr(4)
                            ],
                            dtype=fx.BFloat16,
                        )
                        o_acc[m][nb] = fx.Vector(
                            fx.rocdl.mfma_f32_16x16x32_bf16(
                                T.f32x4, [v_op, p_ops[m], o_acc[m][nb], 0, 0, 0]
                            )
                        )
                    else:
                        o_acc[m][nb] = fx.Vector(
                            fx.rocdl.mfma_f32_16x16x16bf16_1k(
                                T.f32x4,
                                [
                                    parts[0].bitcast(fx.Int16),
                                    p_ops[m].bitcast(fx.Int16),
                                    o_acc[m][nb],
                                    0,
                                    0,
                                    0,
                                ],
                            )
                        )
            if const_expr(prio):
                fx.rocdl.s_setprio(0)
            return (
                st[:M]
                + l_new
                + [
                    o_acc[m][nb]
                    for m in range_constexpr(M)
                    for nb in range_constexpr(NB)
                ]
            )

        for tile_i, state in range(t_begin, t_split, 1, init=init):
            interior = yield step(fx.Int32(tile_i), state, False)
        for tile_i, state in range(t_split, t_end, 1, init=interior):
            results = yield step(fx.Int32(tile_i), state, True)

        final = results
        fx.rocdl.s_waitcnt(vmcnt=0)
        gpu.barrier()

        m_fin = [final[m] * fx.Float32(scale_log2) for m in range_constexpr(M)]
        l_fin = []
        for m in range_constexpr(M):
            lr = final[M + m]
            for sh in (16, 32):
                lr = lr + lr.shuffle_xor(sh, WAVE)
            l_fin.append(lr)
        o_fin = [
            [final[2 * M + m * NB + nb] for nb in range_constexpr(NB)]
            for m in range_constexpr(M)
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
                safe = (mm > neg_inf).select(mm, zero)
                ws = [fx.Float32(fx.exp2(v - safe, fastmath="fast")) for v in ms]
                lt = ls[0] * ws[0]
                for j in range_constexpr(1, S):
                    lt = lt + ls[j] * ws[j]
                inv = (lt > zero).select(fx.Float32(rcp_f32(lt)), zero)
                wb = [fx.full(4, w * inv, fx.Float32) for w in ws]
                row = queries[m] * GQA + l16
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
                        aux=_aux(0),
                    )
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

        if const_expr(S > 1):
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
                max_blocks_per_seq,
                stride_q_row,
                stride_q_head,
                fx.Int32(task[0]),
                start,
                end,
                fx.Int32(task[3]),
                slot,
                work_capacity,
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
                        max_blocks_per_seq,
                        stride_q_row,
                        stride_q_head,
                        is_first.select(fx.Int32(ta[0]), fx.Int32(tb[0])),
                        start,
                        end,
                        is_first.select(fx.Int32(ta[3]), fx.Int32(tb[3])),
                        is_first.select(first, second),
                        work_capacity,
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
        # Outside the JIT rewriter, which would make ``or`` a dynamic boolean.
        return pair_ctas or work_capacity

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
            ).launch(
                grid=(_grid_x(work_capacity), num_kv_heads, 1),
                block=(NWAVES * WAVE, 1, 1),
                stream=stream,
            )

    if NWAVES == 8:
        pa_decode_pingpong_launch.compile_hints["waves_per_eu"] = 2
    compiled = {
        "launch": pa_decode_pingpong_launch,
        "kernel": pa_decode_pingpong_kernel,
    }
    _CACHE[key] = compiled
    return compiled
