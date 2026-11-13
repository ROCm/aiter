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

from functools import cache

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl.expr import const_expr, gpu, range_constexpr
from flydsl.expr.typing import T

from .pa_decode.op_lds import PaDecodeLds
from .pa_decode.traits import KV_COMPUTE_BLOCK, LOG2E, WAVE
from .tensor_shim import buf_base_i64, buf_copy_store, buf_scalar_load, ptr_buf_tensor


def _pin(value):
    # Tied empty asm materializes the value here; LLVM otherwise sinks the
    # softmax past the stagger barrier into the MFMA phase. FlyDSL has no stable
    # optimization barrier.
    value = fx.Vector(value)
    words = value.bitcast(fx.Int32) if value.dtype.width < 32 else value
    tied = fx.Vector(
        llvm.inline_asm(
            words.ir_value().type, [words.ir_value()], "", "=v,0", has_side_effects=True
        )
    )
    return tied.bitcast(value.dtype) if value.dtype.width < 32 else tied


TILE = 64
# Scores may exceed the row's reference max by 2**RESCALE_LOG2 before P is
# rescaled; BF16 P and f32 l and O hold that range.
RESCALE_LOG2 = 8.0


def _rcp(x):
    # afn 1 / x lowers to v_rcp_f32.
    with fx.arith.fastmath("afn"):
        return fx.Float32(1.0) / fx.Float32(x)


def _uni(v):
    # readfirstlane makes v wave-uniform to LLVM, so the DMA address math on it
    # stays on the SALU. FlyDSL has no stable readfirstlane.
    return fx.Int32(fx.rocdl.readfirstlane(T.i32, fx.Int32(v)))


def _uni64(v):
    v = fx.Int64(v)
    lo = _uni(fx.Int32(v & fx.Int64(0xFFFFFFFF)))
    hi = _uni(fx.Int32(v >> 32))
    return (fx.Int64(hi) << 32) | (fx.Int64(lo) & fx.Int64(0xFFFFFFFF))


def _aux(value):
    # Cache-policy bits of the raw LDS DMA; see dma_fire.
    return ir.IntegerAttr.get(ir.IntegerType.get_signless(32), value)


def pingpong_layout(query_length):
    """(M_TILES, ROW_BLOCKS, CHUNKS): 16-row tiles per wave, row blocks, and
    16-token chunks per wave; a 64-token tile has 64 // (16 * CHUNKS) slots."""
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


@cache
def compile_pa_decode_pingpong(
    *,
    head_dim,
    value_dim,
    query_group_size,
    query_length,
    block_size,
    softmax_scale,
    varlen=False,
    window=0,
    sink_dtype=None,
    seq_query_length=0,
    direct_output=False,
):
    """The tile kernel's launch for one geometry.

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

    ``direct_output`` (varlen, plans with one task per pseudo-sequence) writes
    normalized rows to a contiguous [tokens, heads, DV] output passed in the
    launch's ``key_scale`` slot, so no reducer runs.
    """
    assert block_size % TILE == 0 and block_size & (block_size - 1) == 0
    # Windows' overlapping KV is reused across CTAs, so it stays cacheable.
    dma_aux = 0 if window else 2
    # Direct window output is written once: stream it past the reused KV.
    OAUX = 2 if window else 0
    M, R, CH = pingpong_layout(query_length)
    if window and varlen and query_length == 8:
        # Four waves on two 40 KiB stages fit two CTAs per CU, so one CTA's Q
        # load and output store overlap the other's tiles.
        M, R, CH = 2, 4, 4
    SLOT_TOKENS = 16 * CH
    S = TILE // SLOT_TOKENS
    NWAVES = R * S
    # Staggered prefill: two groups of one wave per SIMD, alternating tiles.
    stagger = NWAVES == 8 and S == 1
    # Four 40 KiB D192/V128 stages fill gfx950's 160 KiB LDS; a window's few
    # tiles gain nothing from a deeper ring, which the stagger still needs.
    stages = 4 if stagger else 2 if window else 3
    D = head_dim
    DV = value_dim
    GQA = query_group_size
    QLEN = query_length
    ROWS = QLEN * GQA
    assert R * M * 16 >= ROWS, "layout must cover every query row"
    DIRECT = bool(direct_output) and bool(varlen) and not window

    K_ROW = D * 2
    V_ROW = DV * 2
    K_CPR = D // 8
    V_CPR = DV // 8
    K_STAGE = TILE * K_ROW
    V_STAGE = TILE * V_ROW
    STAGE = K_STAGE + V_STAGE
    # Waves form ISSUERS groups that take turns per tile; a group's previous
    # tile is its only one in flight when it waits, so vmcnt(0) covers exactly
    # that tile.
    ISSUERS = 2 if stagger else stages - 1
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
        f"_m{M}r{R}c{CH}_s{stages}"
    )
    # Online-softmax state: [m (M), l (M), O (M * NB)].
    STATE = 2 * M + M * NB
    if varlen:
        kernel_name += "_varlen"
    if DIRECT:
        kernel_name += "_direct"
    if window:
        kernel_name += f"_sw{window}" + (f"_sink{sink_dtype}" if sink_dtype else "")
    SPLIT = max(seq_query_length // query_length, 1) if window and not varlen else 1
    if SPLIT > 1:
        kernel_name += f"_qs{SPLIT}"
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
        kv_h = fx.Int32(fx.block_idx.y)
        # Uniform, so the per-tile DMA address math stays on the SALU. The
        # window launch passes it: grid_dim is a dependent dispatch-packet load.
        n_kv = _uni(num_kv_heads) if const_expr(window) else _uni(gpu.grid_dim.y)
        tid = fx.Int32(fx.thread_idx.x)
        warp = _uni(tid // WAVE)
        lane = tid % WAVE
        g = lane // 16
        l16 = lane % 16
        ts = warp // R
        rb = warp % R
        issuer = warp // GROUP_WAVES
        group_warp = warp % GROUP_WAVES
        grp = issuer
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
        for m in range_constexpr(M):
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

        def dma_fire(addr, nbytes, voffs, copies, lds_off):
            # Raw LDS DMA: cdna4.BufferLoadAsyncLDS fixes the cache policy at 0
            # and tracks completion by asyncmark, while the paged KV stream is
            # non-temporal (aux 2) and its issuer groups wait on vmcnt.
            rs = fx.rocdl.get_buffer_rsrc(
                fx.get_iter(ptr_buf_tensor(addr, num_records_bytes=fx.Int64(nbytes)))
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

        page_shift = (block_size // TILE).bit_length() - 1
        k_page_bytes32 = _uni(n_kv * (block_size * K_ROW))
        v_page_bytes32 = _uni(n_kv * (block_size * V_ROW))

        def zext(x):
            return fx.Int64(x) & fx.Int64(0xFFFFFFFF)

        def issue(tile, stage):
            # Pages and in-page offsets fit 32 bits unsigned; only the page
            # base needs 64 (a 32 x 32 multiply). K and V share the row and
            # tail math.
            tok0 = tile * TILE
            phys = fx.Int32(
                buf_scalar_load(
                    bt, table_seq * max_blocks_per_seq + (tile >> page_shift)
                )
            )
            row_idx = (tok0 & (block_size - 1)) * n_kv + kv_h
            left = context - tok0
            rows_m1 = ((left < TILE).select(left, TILE) - 1) * n_kv
            stage_off = stage * STAGE

            def fire(base, page_bytes, row, voffs, copies, lds_off):
                addr = _uni64(
                    base + zext(phys) * zext(page_bytes) + zext(row_idx * row)
                )
                # Rows at or past the context read as zero.
                dma_fire(addr, _uni(rows_m1 * row + row), voffs, copies, lds_off)

            fire(k_base, k_page_bytes32, K_ROW, k_voff, K_COPIES, stage_off)
            fire(v_base, v_page_bytes32, V_ROW, v_voff, V_COPIES, stage_off + K_STAGE)

        for s in range_constexpr(stages - 1):
            # Group 0 issues one tile later, in its softmax phase.
            lead_s = LEAD - 1 if (stagger and s % 2 == 0) else LEAD
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

        # Raw-score state [m (M), l (M), O (M * NB)]: m is a row-uniform
        # reference, rescaled only when a score exceeds it by RESCALE_LOG2 in
        # the log2 domain; l is per lane.
        init = (
            [neg_inf] * M
            + [zero] * M
            + [fx.full(4, 0.0, fx.Float32) for _ in range_constexpr(M * NB)]
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
                    rmax = fx.maxnumf(
                        rmax, fx.Float32(fx.gpu.shuffle_xor(rmax, sh, WAVE))
                    )
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

        mma_k32 = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 32, fx.BFloat16))
        mma_k16 = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 16, fx.BFloat16))

        def mfma(atom, acc, a, b):
            """acc + a b on register fragments; the fragment copies fold away."""
            frags = []
            for value, dtype in ((acc, fx.Float32), (a, fx.BFloat16), (b, fx.BFloat16)):
                t = fx.make_rmem_tensor(len(value), dtype)
                t.store(fx.Vector(value))
                frags.append(t)
            fx.gemm(atom, frags[0], frags[1], frags[2], frags[0])
            return fx.Vector(frags[0].load())

        def qk(stage):
            """S = K Q^T for a stage: lane (g, l16) of chunk c holds token
            slot + 16c + 4g + i, row l16."""
            k_stage, _ = stage_bases(stage)
            s_acc = [[None] * CH for _ in range_constexpr(M)]

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
                        s_acc[m][c] = mfma(
                            mma_k32, s_acc[m][c], k_ops[kk], q_ops[m][kk]
                        )
                k_ops = k_next
            return [
                s_acc[m][c] for m in range_constexpr(M) for c in range_constexpr(CH)
            ]

        def soft(it, s_flat, state, masked):
            """Online softmax of one tile's scores: (state with O rescaled, P)."""
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
            # One SALU ballot; FlyDSL has no stable one, and a warp_reduce OR
            # would be six cross-lane shuffles every tile.
            if fx.Int64(fx.rocdl.ballot(T.i64, over.ir_value())) != fx.Int64(0):
                st = rescale(st, lmax)

            # One B operand per chunk pair (8 tokens a lane), or the chunk.
            width = 4 if CH == 1 else 8
            p_ops = []
            neg_refs = [-scaled_ref(st[m]) for m in range_constexpr(M)]
            for m in range_constexpr(M):
                neg_ref = neg_refs[m]
                p = []
                for v in scores[m]:
                    e = fx.Float32(
                        fx.exp2(
                            fx.fma(v, fx.Float32(scale_log2), neg_ref), fastmath="fast"
                        )
                    )
                    p.append(e)
                    st[M + m] = st[M + m] + e
                    if const_expr(len(p) % width == 0):
                        p_ops.append(
                            fx.Vector.from_elements(
                                p[len(p) - width :], dtype=fx.Float32
                            ).to(fx.BFloat16)
                        )
            return st, p_ops

        v_tr_atom = fx.make_copy_atom(fx.rocdl.cdna4.LDSReadTrans(16, 64), fx.BFloat16)
        v_tr_layout = fx.make_layout(4, 1)

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
                parts = []
                for c in range_constexpr(CH):
                    v_reg = fx.make_rmem_tensor(v_tr_layout, fx.BFloat16)
                    v_src = lds.pointer(v_stage[nb] + v_imm[c], fx.BFloat16)
                    fx.copy(v_tr_atom, fx.make_view(v_src, v_tr_layout), v_reg)
                    parts.append(fx.Vector(v_reg.load()))
                return parts

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
                            o_acc[m][nb] = mfma(
                                mma_k32, o_acc[m][nb], v_op, p_flat[m * pairs + j]
                            )
                    else:
                        o_acc[m][nb] = mfma(mma_k16, o_acc[m][nb], parts[0], p_flat[m])
            return st[: 2 * M] + [
                o_acc[m][nb] for m in range_constexpr(M) for nb in range_constexpr(NB)
            ]

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
            s_flat = qk(stage)
            st, p_flat = soft(it, s_flat, state, masked)
            return pv(stage, p_flat, st)

        def smod(x):
            return x & (stages - 1) if stages & (stages - 1) == 0 else x % stages

        def stagger_step(it, carry, masked):
            # Group 1 runs one phase behind group 0: each phase X (softmax of
            # tile i) pairs with the other group's phase Y (PV of tile i, QK of
            # tile i + 1). Tile i + LEAD reuses tile i - 1's stage once group
            # 1's PV of it ends; a group waits for its tile two slots before
            # the first QK, when that tile is its only one in flight.
            rel = it - t_begin
            state = carry[:STATE]
            s_flat = carry[STATE:]
            nxt = it + LEAD

            def gated(group, rpar, bound, fn):
                c = (grp == group) & ((rel & 1) == rpar)
                if const_expr(bound is not None):
                    c = c & bound
                if c:
                    fn()

            gated(1, (1 - LEAD) % 2, nxt < t_end, lambda: issue(nxt, smod(rel + LEAD)))
            # Group 0's tile i + LEAD - 1 takes tile i - 2's stage, free since
            # group 1's PV of it in the previous slot.
            gated(
                0,
                (1 - LEAD) % 2,
                nxt - 1 < t_end,
                lambda: issue(nxt - 1, smod(rel + LEAD - 1)),
            )
            st, p_flat = soft(it, s_flat, state, masked)
            p_flat = [_pin(p) for p in p_flat]
            sums = _pin(fx.Vector.from_elements(st[M : 2 * M], dtype=fx.Float32))
            st = st[:M] + [sums[m] for m in range_constexpr(M)] + st[2 * M :]
            gated(0, 1, None, lambda: fx.rocdl.s_waitcnt(vmcnt=0))
            gpu.barrier()
            # The last tile's lookahead QK reads a stale stage; it is unused.
            state = pv(smod(rel), p_flat, st)
            s_next = qk(smod(rel + 1))
            # One MFMA wave a SIMD: keep LDS reads 8 MFMAs ahead.
            lead_reads = 8
            n_ds = (NB + KK) * CH
            n_mfma = M * NB * max(CH // 2, 1) + M * KK * CH
            fx.rocdl.sched_dsrd(lead_reads)
            for _ in range_constexpr(n_ds - lead_reads):
                fx.rocdl.sched_mfma(1)
                fx.rocdl.sched_dsrd(1)
            fx.rocdl.sched_mfma(n_mfma - (n_ds - lead_reads))
            gated(1, 1, None, lambda: fx.rocdl.s_waitcnt(vmcnt=0))
            gpu.barrier()
            return list(state) + list(s_next)

        if const_expr(stagger):
            # Prologue: tile 0 ready for group 0's QK, then group 1 falls one
            # phase behind.
            if grp == 0:
                fx.rocdl.s_waitcnt(vmcnt=0)
            gpu.barrier()
            if grp == 1:
                gpu.barrier()
            s0 = qk(0)
            if grp == 1:
                fx.rocdl.s_waitcnt(vmcnt=0)
            gpu.barrier()
            carry = list(init) + list(s0)
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

        m_fin = [final[m] * fx.Float32(scale_log2) for m in range_constexpr(M)]
        l_fin = []
        for m in range_constexpr(M):
            lr = final[M + m]
            for sh in (16, 32):
                lr = lr + fx.Float32(fx.gpu.shuffle_xor(lr, sh, WAVE))
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
            pout_t = ptr_buf_tensor(
                _uni64(buf_base_i64(pout_ptr) + fx.Int64(partial) * (ROWS * DV * 2)),
                fx.BFloat16,
                unit_elems=4,
                num_records_bytes=ROWS * DV * 2,
            )
            if const_expr(window or DIRECT):
                # The sequence's valid query rows of out; padded rows drop.
                row_bytes = n_kv * (GQA * DV * 2)
                pout_t = ptr_buf_tensor(
                    _uni64(
                        buf_base_i64(out_ptr)
                        + fx.Int64(first_row) * fx.Int64(row_bytes)
                    ),
                    fx.BFloat16,
                    unit_elems=4,
                    num_records_bytes=fx.Int64(_uni(valid * row_bytes)),
                )
            pmax_t = ptr_buf_tensor(
                _uni64(buf_base_i64(pmax_ptr) + fx.Int64(partial) * (ROWS * 4)),
                fx.Float32,
                num_records_bytes=ROWS * 4,
            )
            psum_t = ptr_buf_tensor(
                _uni64(buf_base_i64(psum_ptr) + fx.Int64(partial) * (ROWS * 4)),
                fx.Float32,
                num_records_bytes=ROWS * 4,
            )
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
                inv = (lt > zero).select(_rcp(lt), zero)
                wb = [fx.full(4, w * inv, fx.Float32) for w in ws]
                row = queries[m] * GQA + l16
                if const_expr(window or DIRECT):
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
                    buf_copy_store(
                        pout_t,
                        row * (DV // 4) + nb * 4 + g,
                        o.to(fx.BFloat16),
                        fx.BFloat16,
                        unit_elems=4,
                        cache_modifier=OAUX,
                    )
                if const_expr(not (window or DIRECT)):
                    buf_copy_store(
                        pmax_t,
                        row,
                        fx.Float32(mm * fx.Float32(1.0 / LOG2E)),
                        fx.Float32,
                    )
                    buf_copy_store(psum_t, row, fx.Float32(lt), fx.Float32)

        if const_expr(S > 1):
            if ts > 0:
                write_partial()
            gpu.barrier()
            if ts == 0:
                merge_store()
        else:
            merge_store()
        # The next task's ring fill reuses the merge scratch.
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
        out_ptr: fx.Pointer,
    ):
        work = fx.recast_iter(fx.Int32, work_info_ptr)
        slot = fx.Int32(fx.block_idx.x)
        task = fx.ptr_load(
            fx.add_offset(work, slot * 4),
            result_type=fx.Vector.make_type(4, fx.Int32),
        )
        start = fx.Int32(task[1])
        end = fx.Int32(task[2])
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
                fx.Int32(task[0]),
                start,
                end,
                fx.Int32(task[3]),
                slot,
                work_capacity,
                out_ptr,
                pout_ptr,
                fx.Int32(0),
            )

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
            # Direct varlen output takes the unused key_scale slot.
            key_scale,
            # Packed FP32 softmax VALU stalls behind the sibling wave's MFMAs.
            # FlyDSL 0.5.0's target-features compile hint has no effect, so the
            # kernel attribute sets it.
            value_attrs={
                "llvm.target_features": ir.Attribute.parse(
                    '#llvm.target_features<["-packed-fp32-ops"]>'
                )
            },
        ).launch(
            grid=(work_capacity, num_kv_heads, 1),
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
        seq = fx.Int32(fx.block_idx.x)
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
            # Packed FP32 softmax VALU stalls behind the sibling wave's MFMAs.
            # FlyDSL 0.5.0's target-features compile hint has no effect, so the
            # kernel attribute sets it.
            value_attrs={
                "llvm.target_features": ir.Attribute.parse(
                    '#llvm.target_features<["-packed-fp32-ops"]>'
                )
            },
        ).launch(
            grid=(num_seqs, num_kv_heads, 1),
            block=(NWAVES * WAVE, 1, 1),
            stream=stream,
        )

    launch = pa_decode_pingpong_window_launch if window else pa_decode_pingpong_launch
    launch.compile_hints["fastmath"] = "contract"
    if NWAVES == 8:
        launch.compile_hints["waves_per_eu"] = 2
    return launch
