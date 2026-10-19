# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""BM16 inline-quant MXFP4 GEMM1 with the decode MoE sort folded into its head.

At decode M the a4w4 BM16 path launches a single-CTA sort that also zeroes the
atomic output (moe_buf) and then GEMM1. Here every GEMM1 CTA rebuilds the
padded per-expert block layout from topk_ids in LDS (histogram, block-count
scan), resolves the expert of its m-block and ranks that expert's routes into
its 16 rows. The n_block 0 CTA of each live m-block publishes the stage-2 sort
tables (sorted_token_ids, sorted_weights, sorted_expert_ids[m_block]), m-block 0
publishes num_valid_ids = [rows, M], and the moe_buf zero-init (atomic stage 2
only) is striped over the whole grid. M stays a runtime value; only the route
capacity bucket CAP (M * topk <= CAP) is a compile constant.

The GEMM itself is _gemm1_body; at small M the launch additionally splits K
over two CTAs and issues the hidden-state loads of the first K tiles before
the sort (token-row prologue), see the hooks documented on _gemm1_body.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith, const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T, as_ir_value

from . import dpp_utils
from .mxfp4_gemm1 import _bm_constants, _gemm1_body, gemm1_grid
from .mxfp4_gemm_common import (
    _global_i32_at,
    _global_i32_buffer_tiles,
    _global_i32_load,
    _global_scalar_tiles,
    _scalar_store,
    kStages,
)

BM = 16
BN = 256
BK = 256
# Experts 0..255 are histogrammed one per thread; one extra tail expert (the
# fused shared expert, id 256) is handled separately.
MAX_EXPERTS = 257
SORT_FOLD_CAPS = (256, 576)
# The first 14 kernel-argument dwords are preloaded into SGPRs.
KERNARG_PRELOAD_DWORDS = 14

# LDS word offsets of the sort scratch.
_W_CNT = 0  # 272: routes per expert
_W_WT = 272  # 16: per-wave scan totals
_W_MISC = 288  # 16: [0] expert of this m-block, [1] block index within it
_W_ROWS = 304  # 16: token row of each of the m-block's rows (pad = M)
_W_STOK = 320  # 16: sorted_token_ids slice
_W_WGT = 336  # 16: sorted_weights slice (f32 bits)
_SORT_WORDS = 352
_HIST_WORDS = 272


def sort_fold_cap(n_routes):
    """Smallest route-capacity bucket holding M * topk routes, or 0."""
    for cap in SORT_FOLD_CAPS:
        if n_routes <= cap:
            return cap
    return 0


def _ashr(x, n):
    return fx.Int32(arith.ShRSIOp(as_ir_value(x), as_ir_value(fx.Int32(n))).result)


def _lt_mask(a, b):
    """-1 where a < b, else 0 (|a - b| < 2**31)."""
    return _ashr(a - b, 31)


def _is_eq(a, b):
    """1 where a == b, else 0 (a, b >= 0)."""
    return fx.Int32(0) - _ashr((a ^ b) - fx.Int32(1), 31)


def _wave_inclusive_scan(x):
    """64-lane inclusive add-scan: row_shr 1/2/4/8, row_bcast15, row_bcast31."""
    for ctrl in (0x111, 0x112, 0x114, 0x118):
        x = x + fx.Int32(dpp_utils.update_dpp_i32(0, x, ctrl, 0xF, 0xF, True))
    x = x + fx.Int32(dpp_utils.update_dpp_i32(0, x, 0x142, 0xA, 0xF, True))
    x = x + fx.Int32(dpp_utils.update_dpp_i32(0, x, 0x143, 0xC, 0xF, True))
    return x


def _lds_atomic_add(tiles, idx, val):
    atom = fx.make_copy_atom(fx.UniversalAtomicAdd(fx.Int32), fx.Int32)
    r = fx.make_rmem_tensor(fx.make_layout(1, 1), fx.Int32)
    r.store(fx.Vector.from_elements([fx.Int32(val)], fx.Int32))
    fx.copy_atom_call(atom, r, fx.slice(tiles, (None, idx)))


def _lds_i32_tiles(base_ptr, word_off, n_words):
    flat = fx.make_view(
        fx.recast_iter(fx.Int32, fx.add_offset(base_ptr, word_off * 4)),
        fx.make_layout(n_words, 1),
    )
    return fx.logical_divide(flat, fx.make_layout(1, 1))


def compile_gemm1_sort_fold(
    *,
    D_HIDDEN,
    D_INTER,
    NE,
    TOPK,
    CAP,
    split_k=1,
    token_rows=False,
    zero_output=True,
):
    """Compile the sort-folded BM16 GEMM1 for one route-capacity bucket.

    split_k == 2 and token_rows are the small-M launch (M <= 7); CAP 576 and
    split_k == 2 also issue the first B tiles before the rank scan and load
    the hidden states one K tile ahead. zero_output zeroes moe_buf for an
    atomic stage 2; a reduce stage 2 writes every output row itself.
    """
    assert NE <= MAX_EXPERTS, "fold head handles experts 0..255 plus one tail"
    assert CAP in SORT_FOLD_CAPS, f"unsupported route capacity {CAP}"
    assert split_k in (1, 2)
    KH_TILE = BK // 2
    K = D_HIDDEN
    N_OUT = 2 * D_INTER
    assert K % BK == 0 and N_OUT % BN == 0
    NNB = N_OUT // BN
    K_TILES = K // BK
    k_stages = kStages
    assert K_TILES % split_k == 0 and K_TILES // split_k > k_stages
    early_b = token_rows or split_k > 1 or CAP > 256
    n_rep = (CAP + 255) // 256
    m_max = CAP // TOPK
    if token_rows:
        # LDS A row r holds token r, and rows >= M must read as zero, so the
        # 16-row tile also needs one pad row.
        assert m_max >= 1 and n_rep == 1
    # Waves rank all routes of the token-row bucket in private histograms.
    wave_sort = token_rows
    n_wave_reps = (15 * TOPK + 63) // 64
    if wave_sort:
        assert n_wave_reps <= 3
    k_w = K // 2  # bf16 [M, K] row as i32 words
    # Zero-init repetitions: the worst case over every M this bucket can see.
    n_rep_zero = 1
    for m in range(1, m_max + 1):
        g = gemm1_grid(m, BM, NE=NE, TOPK=TOPK, INTER=D_INTER, BN=BN)
        n_rep_zero = max(n_rep_zero, (m * k_w + g * 256 - 1) // (g * 256))
    n_pre = k_stages + 1 if token_rows else 0
    _, _, _, lds_bytes = _bm_constants(BM, BN, KH_TILE, K_TILES // split_k)

    name = f"gemm1_a4w4_sort_fold_h{K}_i{D_INTER}_ne{NE}_bm{BM}_c{CAP}"
    if split_k > 1:
        name += f"_sk{split_k}"
    if token_rows:
        name += "_tr"
    if not zero_output:
        name += "_nz"

    @fx.struct
    class SharedStorage:
        raw: fx.Array[fx.Uint8, lds_bytes, 16]
        srt: fx.Array[fx.Uint8, _SORT_WORDS * 4, 16]
        hst: fx.Array[fx.Uint8, (4 * _HIST_WORDS if wave_sort else 4) * 4, 16]

    @flyc.kernel(name=name, known_block_size=[256, 1, 1])
    def gemm1_sort_fold_kernel(
        # The preloaded window (first 14 dwords) holds the arguments the sort
        # head and the first B loads need. Under the token-row prologue it also
        # carries moe_buf: its zero-init stores are the last head stores.
        arg_bq: fx.Int64,
        arg_bscale: fx.Int64,
        i32_ntok: fx.Int32,
        i32_grid: fx.Int32,
        arg_hidden: fx.Int64,
        arg_topk: fx.Int64,
        arg_topkw: fx.Int64,
        arg_slot7: fx.Int64,
        arg_eids: fx.Int64,
        arg_cumsum: fx.Int64,
        arg_slot9: fx.Int64,
        arg_ascaleout: fx.Int64,
        arg_stok: fx.Int64,
        arg_swgt: fx.Int64,
        arg_sk_part: fx.Int64,
    ):
        arg_moebuf = arg_slot7 if token_rows else arg_slot9
        arg_aqout = arg_slot9 if token_rows else arg_slot7
        smem = fx.SharedAllocator().allocate(SharedStorage).peek()
        lds_raw_ptr = smem.raw.ptr
        sort_ptr = smem.srt.ptr
        tx_i32 = fx.Int32(gpu.thread_id("x"))
        bx_i32 = fx.Int32(gpu.block_id("x"))
        if const_expr(split_k > 1):
            sk_i32 = rocdl.readfirstlane(
                T.i32, as_ir_value(fx.Int32(gpu.block_id("y")))
            )
        else:
            sk_i32 = fx.Int32(0)
        lane = tx_i32 % fx.Int32(64)
        wave = rocdl.readfirstlane(T.i32, tx_i32 // fx.Int32(64))
        i32_routes = i32_ntok * fx.Int32(TOPK)

        cnt_tiles = _lds_i32_tiles(sort_ptr, _W_CNT, _HIST_WORDS)
        wt_tiles = _lds_i32_tiles(sort_ptr, _W_WT, 16)
        mi_tiles = _lds_i32_tiles(sort_ptr, _W_MISC, 16)
        rw_tiles = _lds_i32_tiles(sort_ptr, _W_ROWS, 16)
        st_tiles = _lds_i32_tiles(sort_ptr, _W_STOK, 16)
        wg_tiles = _lds_i32_tiles(sort_ptr, _W_WGT, 16)

        token_row_hidden = None
        if const_expr(token_rows):
            # topk ids first (sort critical path), then the token-row hidden
            # loads of the first K tiles, then everything else.
            if const_expr(wave_sort):
                wave_ids = []
                wave_ok = []
                for q in range_constexpr(n_wave_reps):
                    item = lane + fx.Int32(q * 64)
                    ok = _lt_mask(item, i32_routes)
                    wave_ids.append(_global_i32_at(arg_topk, item & ok))
                    wave_ok.append(ok)
                first_id = wave_ids[0]
                for q in range_constexpr(1, n_wave_reps):
                    first_id = (wave == fx.Int32(q)).select(wave_ids[q], first_id)
            else:
                first_id = _global_i32_at(
                    arg_topk, tx_i32 & _lt_mask(tx_i32, i32_routes)
                )
            h_tiles = _global_i32_buffer_tiles(
                arg_hidden, fx.Int64(i32_ntok * fx.Int32(K * 2)), 4
            )
            h_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.Int32)
            h_row = wave * fx.Int32(4) + lane // fx.Int32(16)
            h_voff = (
                h_row * fx.Int32(K * 2)
                + ((lane >> fx.Int32(2)) & fx.Int32(3)) * fx.Int32(64)
                + (lane & fx.Int32(3)) * fx.Int32(16)
            )
            token_row_hidden = []
            for kt in range_constexpr(n_pre):
                halves = []
                for h in range_constexpr(2):
                    soff = fx.Int32(kt * (BK * 2) + h * 256)
                    if const_expr(split_k > 1):
                        soff = soff + sk_i32 * fx.Int32((K_TILES // split_k) * BK * 2)
                    soff = rocdl.readfirstlane(T.i32, soff)
                    r = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Int32)
                    fx.copy(
                        h_atom,
                        fx.slice(h_tiles, (None, h_voff // fx.Int32(16))),
                        r,
                        soffset=soff // fx.Int32(4),
                    )
                    halves.append(r.load())
                token_row_hidden.append(halves)

        # topk ids/weights -> registers; routes past M * topk read as expert -1.
        tk_regs = []
        tkw_regs = []
        for rep in range_constexpr(n_rep):
            item = tx_i32 + fx.Int32(rep * 256)
            ok = _lt_mask(item, i32_routes)
            if const_expr(token_rows and rep == 0):
                tk_regs.append((first_id, ok))
            else:
                tk_regs.append((_global_i32_at(arg_topk, item & ok), ok))
            tkw_regs.append(_global_i32_at(arg_topkw, item & ok))

        # moe_buf zero-init, striped over the whole grid.
        mb_tiles = _global_scalar_tiles(arg_moebuf, fx.Int32, 1 << 30)
        n_dw = i32_ntok * fx.Int32(k_w)
        g_tid = (sk_i32 * i32_grid + bx_i32) * fx.Int32(256) + tx_i32
        g_stride = i32_grid * fx.Int32(256 * split_k)

        def zero_init():
            if const_expr(zero_output):
                for rz in range_constexpr(n_rep_zero):
                    zi = g_tid + g_stride * fx.Int32(rz)
                    if zi < n_dw:
                        _scalar_store(mb_tiles, zi, fx.Int32(0), fx.Int32)

        # Under the token-row prologue the zero-init stores go out after the
        # first B loads so they never sit ahead of them in the vmem queue.
        if const_expr(not token_rows):
            zero_init()

        tk_valid = [fx.Int32(0) - ok for (_, ok) in tk_regs]  # 1 iff a live route
        tk_ids = [(v & ok) | (fx.Int32(-1) ^ ok) for (v, ok) in tk_regs]
        if const_expr(wave_sort):
            # Every wave histograms all M * topk (<= 135) routes itself, so no
            # barrier is needed before the expert id and the first B loads.
            hist_view = fx.make_view(
                fx.recast_iter(fx.Int32, smem.hst.ptr),
                fx.make_layout(4 * _HIST_WORDS, 1),
            )
            hw_tiles = fx.logical_divide(hist_view, fx.make_layout(1, 1))
            hb = wave * fx.Int32(_HIST_WORDS)
            for q in range_constexpr(4):
                _scalar_store(
                    hw_tiles, hb + lane + fx.Int32(q * 64), fx.Int32(0), fx.Int32
                )
            if lane < fx.Int32(16):
                _scalar_store(
                    hw_tiles, hb + lane + fx.Int32(256), fx.Int32(0), fx.Int32
                )
            for q in range_constexpr(n_wave_reps):
                if wave_ok[q] != fx.Int32(0):
                    _lds_atomic_add(hw_tiles, hb + wave_ids[q], 1)
            # lane l owns experts 4l..4l+3; padded block counts and their scan
            cnt4 = [
                _global_i32_load(hw_tiles, hb + lane * fx.Int32(4) + fx.Int32(j))
                for j in range_constexpr(4)
            ]
            cnt_tail = _global_i32_load(hw_tiles, hb + fx.Int32(256))
            nb4 = [(c + fx.Int32(BM - 1)) >> fx.Int32(4) for c in cnt4]
            nb_tail = (cnt_tail + fx.Int32(BM - 1)) >> fx.Int32(4)
            p1 = nb4[0]
            p2 = p1 + nb4[1]
            p3 = p2 + nb4[2]
            lane_sum = p3 + nb4[3]
            lane_incl = _wave_inclusive_scan(lane_sum)
            t256 = fx.Int32(rocdl.readlane(T.i32, as_ir_value(lane_incl), 63))
            total_m_blocks = rocdl.readfirstlane(T.i32, as_ir_value(t256 + nb_tail))
            mbi_u = rocdl.readfirstlane(T.i32, as_ir_value(bx_i32 // fx.Int32(NNB)))
            rel = fx.Int32(mbi_u) - (lane_incl - lane_sum)
            one = fx.Int32(1)
            zero = fx.Int32(0)
            jj = (
                (rel >= p1).select(one, zero)
                + (rel >= p2).select(one, zero)
                + (rel >= p3).select(one, zero)
            )
            pj = (rel >= p3).select(
                p3, (rel >= p2).select(p2, (rel >= p1).select(p1, zero))
            )
            hit = (rel >= zero).select((rel < lane_sum).select(one, zero), zero)
            packed = (hit != zero).select(
                (lane * fx.Int32(4) + jj) | ((rel - pj) << fx.Int32(16)), zero
            )
            packed = fx.Int32(
                rocdl.readlane(T.i32, as_ir_value(_wave_inclusive_scan(packed)), 63)
            )
            packed = (fx.Int32(mbi_u) >= t256).select(
                fx.Int32(256) | ((fx.Int32(mbi_u) - t256) << fx.Int32(16)), packed
            )
            ws_expert = rocdl.readfirstlane(
                T.i32, as_ir_value(packed & fx.Int32(0xFFFF))
            )
            ws_block = rocdl.readfirstlane(T.i32, as_ir_value(packed >> fx.Int32(16)))
        else:
            # zero the histogram while the topk loads fly
            for rep in range_constexpr(2):
                ci = tx_i32 + fx.Int32(rep * 256)
                if ci < fx.Int32(_HIST_WORDS):
                    _scalar_store(cnt_tiles, ci, fx.Int32(0), fx.Int32)
            gpu.barrier()
            for rep in range_constexpr(n_rep):
                item = tx_i32 + fx.Int32(rep * 256)
                if item < i32_routes:
                    _lds_atomic_add(cnt_tiles, tk_ids[rep], 1)
            gpu.barrier()
            # Block-wide scan of padded block counts (thread t <-> expert t);
            # the tail expert 256 starts at the total of experts 0..255.
            cnt_t = _global_i32_load(cnt_tiles, tx_i32)
            nb_tail = (
                _global_i32_load(cnt_tiles, fx.Int32(256)) + fx.Int32(BM - 1)
            ) >> fx.Int32(4)
            nb_t = (cnt_t + fx.Int32(BM - 1)) >> fx.Int32(4)
            incl = _wave_inclusive_scan(nb_t)
            if lane == fx.Int32(63):
                _scalar_store(wt_tiles, wave, incl, fx.Int32)
            gpu.barrier()
            w0 = _global_i32_load(wt_tiles, fx.Int32(0))
            w1 = _global_i32_load(wt_tiles, fx.Int32(1))
            w2 = _global_i32_load(wt_tiles, fx.Int32(2))
            w3 = _global_i32_load(wt_tiles, fx.Int32(3))
            prefix = (
                (w0 & _lt_mask(fx.Int32(0), wave))
                + (w1 & _lt_mask(fx.Int32(1), wave))
                + (w2 & _lt_mask(fx.Int32(2), wave))
            )
            t256 = w0 + w1 + w2 + w3
            excl_t = incl + prefix - nb_t
            total_m_blocks = rocdl.readfirstlane(T.i32, as_ir_value(t256 + nb_tail))
            gpu.barrier()  # wt_tiles is reused by the rank scan

        bound = total_m_blocks * fx.Int32(NNB)
        if bx_i32 < bound:
            tile = bx_i32
            mbi = tile // fx.Int32(NNB)
            nbi = tile % fx.Int32(NNB)
            if const_expr(not wave_sort):
                # which expert owns this m-block?
                if mbi >= excl_t:
                    if mbi < excl_t + nb_t:
                        _scalar_store(mi_tiles, fx.Int32(0), tx_i32, fx.Int32)
                        _scalar_store(mi_tiles, fx.Int32(1), mbi - excl_t, fx.Int32)
                if tx_i32 == fx.Int32(0):
                    if mbi >= t256:
                        _scalar_store(mi_tiles, fx.Int32(0), fx.Int32(256), fx.Int32)
                        _scalar_store(mi_tiles, fx.Int32(1), mbi - t256, fx.Int32)
            if tx_i32 < fx.Int32(BM):
                # pad rows: token M, sorted id (topk << 24) | M, weight 0
                _scalar_store(rw_tiles, tx_i32, i32_ntok, fx.Int32)
                _scalar_store(
                    st_tiles, tx_i32, i32_ntok | fx.Int32(TOPK << 24), fx.Int32
                )
                _scalar_store(wg_tiles, tx_i32, fx.Int32(0), fx.Int32)
            if const_expr(wave_sort):
                # The pad-row init is ordered before the rank writes by the
                # rank scan's own barrier.
                fold_e = fx.Int32(ws_expert)
                row_lo = fx.Int32(ws_block) * fx.Int32(BM)
            else:
                gpu.barrier()
                fold_e = rocdl.readfirstlane(
                    T.i32, as_ir_value(_global_i32_load(mi_tiles, fx.Int32(0)))
                )
                row_lo = _global_i32_load(mi_tiles, fx.Int32(1)) * fx.Int32(BM)

            def rank_and_publish():
                if const_expr(token_rows):
                    zero_init()
                # Rank this expert's routes in route order; keep ranks
                # [row_lo, row_lo + 16).
                base = fx.Int32(0)
                for rep in range_constexpr(n_rep):
                    item = tx_i32 + fx.Int32(rep * 256)
                    x = _is_eq(tk_ids[rep], fold_e) & tk_valid[rep]
                    incl2 = _wave_inclusive_scan(x)
                    if const_expr(rep > 0):
                        gpu.barrier()
                    if lane == fx.Int32(63):
                        _scalar_store(wt_tiles, wave, incl2, fx.Int32)
                    gpu.barrier()
                    u0 = _global_i32_load(wt_tiles, fx.Int32(0))
                    u1 = _global_i32_load(wt_tiles, fx.Int32(1))
                    u2 = _global_i32_load(wt_tiles, fx.Int32(2))
                    u3 = _global_i32_load(wt_tiles, fx.Int32(3))
                    p = (
                        (u0 & _lt_mask(fx.Int32(0), wave))
                        + (u1 & _lt_mask(fx.Int32(1), wave))
                        + (u2 & _lt_mask(fx.Int32(2), wave))
                    )
                    d = base + p + incl2 - x - row_lo
                    if x == fx.Int32(1):
                        if fx.Uint32(d) < fx.Uint32(BM):
                            tok = item // fx.Int32(TOPK)
                            slot = item - tok * fx.Int32(TOPK)
                            _scalar_store(rw_tiles, d, tok, fx.Int32)
                            _scalar_store(
                                st_tiles, d, tok | (slot << fx.Int32(24)), fx.Int32
                            )
                            _scalar_store(wg_tiles, d, tkw_regs[rep], fx.Int32)
                    base = base + u0 + u1 + u2 + u3
                gpu.barrier()

                # publish this m-block's slice of the stage-2 sort tables
                if (nbi | sk_i32) == fx.Int32(0):
                    stok_g = _global_scalar_tiles(arg_stok, fx.Int32, 1 << 24)
                    swgt_g = _global_scalar_tiles(arg_swgt, fx.Int32, 1 << 24)
                    seid_g = _global_scalar_tiles(arg_eids, fx.Int32, 1 << 22)
                    if tx_i32 < fx.Int32(BM):
                        sp = mbi * fx.Int32(BM) + tx_i32
                        _scalar_store(
                            stok_g, sp, _global_i32_load(st_tiles, tx_i32), fx.Int32
                        )
                        _scalar_store(
                            swgt_g, sp, _global_i32_load(wg_tiles, tx_i32), fx.Int32
                        )
                    if tx_i32 == fx.Int32(0):
                        _scalar_store(seid_g, mbi, fold_e, fx.Int32)
                        if mbi == fx.Int32(0):
                            nv_g = _global_scalar_tiles(arg_cumsum, fx.Int32, 2)
                            _scalar_store(
                                nv_g,
                                fx.Int32(0),
                                total_m_blocks * fx.Int32(BM),
                                fx.Int32,
                            )
                            _scalar_store(nv_g, fx.Int32(1), i32_ntok, fx.Int32)

            if const_expr(not early_b):
                rank_and_publish()

            _gemm1_body(
                lds_raw_ptr,
                arg_hidden,
                arg_hidden,
                arg_bq,
                arg_bscale,
                arg_eids,
                arg_hidden,
                arg_aqout,
                arg_ascaleout,
                arg_hidden,
                arg_hidden,
                tile,
                lane,
                wave,
                True,
                i32_ntok,
                total_m_blocks,
                BM=BM,
                BN=BN,
                BK=BK,
                inline_quant=True,
                act="silu",
                swiglu_limit=float("inf"),
                K=K,
                N_OUT=N_OUT,
                NE=NE,
                native_scale_layout=True,
                k_stages=k_stages,
                fold_expert=fold_e,
                fold_rows=rw_tiles,
                fold_publish=rank_and_publish if early_b else None,
                fold_token_rows=token_row_hidden,
                split_k=split_k,
                sk_idx=sk_i32,
                arg_sk_part=arg_sk_part,
                b_scale_aux=2,
            )
        if const_expr(token_rows):
            if bx_i32 >= bound:
                zero_init()

    @flyc.jit
    def launch_gemm1_sort_fold(
        arg_bq: fx.Int64,
        arg_bscale: fx.Int64,
        i32_ntok: fx.Int32,
        i32_grid: fx.Int32,
        arg_hidden: fx.Int64,
        arg_topk: fx.Int64,
        arg_topkw: fx.Int64,
        arg_eids: fx.Int64,
        arg_cumsum: fx.Int64,
        arg_aqout: fx.Int64,
        arg_ascaleout: fx.Int64,
        arg_stok: fx.Int64,
        arg_swgt: fx.Int64,
        arg_moebuf: fx.Int64,
        arg_sk_part: fx.Int64,
        stream: fx.Stream,
    ):
        gemm1_sort_fold_kernel(
            arg_bq,
            arg_bscale,
            i32_ntok,
            i32_grid,
            arg_hidden,
            arg_topk,
            arg_topkw,
            arg_moebuf if token_rows else arg_aqout,
            arg_eids,
            arg_cumsum,
            arg_aqout if token_rows else arg_moebuf,
            arg_ascaleout,
            arg_stok,
            arg_swgt,
            arg_sk_part,
        ).launch(
            grid=(fx.Int64(i32_grid), split_k, 1),
            block=(256, 1, 1),
            stream=stream,
        )

    launch_gemm1_sort_fold.compile_hints = {
        "llvm_options": {"amdgpu-kernarg-preload-count": KERNARG_PRELOAD_DWORDS},
    }
    return launch_gemm1_sort_fold
