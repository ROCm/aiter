# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""gfx942 FlyDSL absorbed sparse MLA.

One CTA, two modes. BLOCK_Q==1 uses two waves that split D. BLOCK_Q==2/4 uses
one wave per query for QK and softmax over a shared KV tile, then each wave
computes a D/BLOCK_Q slice of the output for every query. Both modes:

* gather the next 16 KV rows with coalesced dwordx4 loads while the current
  tile's MFMA chain runs
* compute S^T (token on M, head on N): each lane owns one head, the softmax
  reduces across lanes 16 and 32 apart, and P^T is already the PV B operand
* compute out^T = V^T P^T; a lane owns four adjacent d, so one 8-byte V read
  per token feeds four MFMAs

``v_mfma_f32_16x16x16_bf16`` fragment map (device probe):

    C[i] = (A @ B^T)[lane_k + i, lane_row]
    lane_row = lane % 16, lane_k = (lane // 16) * 4
"""

from functools import lru_cache

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import fly as fly_dialect
from flydsl.expr import arith, const_expr, gpu, range_constexpr
from flydsl.expr.typing import T

from aiter.ops.flydsl.kernels.kernels_common import LOG2E
from aiter.ops.flydsl.kernels.tensor_shim import (
    AITER_FLYDSL_KERNARG_PRELOAD,
    AITER_FLYDSL_KERNARG_PRELOAD_COUNT,
)

WAVE = 64
HEAD_DIM = 512
NUM_HEADS = 16
VEC = 8
NEG_INF = float("-inf")
# Padded KV row pitch (elements), used only by Q-block.
LD_KV = 528
# MFMA batches between LDS-read issue and sched_barrier: QK pairs (2 MFMAs each)
# and, one-query only, PV groups (4 MFMAs each).
QK_BATCH_PAIRS_ONE_Q = 4
QK_BATCH_PAIRS_QBLOCK = 2
PV_BATCH_GROUPS_ONE_Q = 2
# Index vectors carried in loop state ahead of the tile being gathered.
# Q-block only: one-query is 4% slower with it (measured on the S^T kernel).
IDX_AHEAD_QBLOCK = 1
RESCALE_TAU = 8.0


def _mfma_ssa(mma, a, b, acc):
    out = fly_dialect.mma_atom_call_ssa(
        [fx.Vector.make_type(4, fx.Float32)],
        mma,
        a,
        b,
        acc,
    )
    if isinstance(out, (list, tuple)):
        out = out[0]
    return fx.Vector(out)


def _elem(acc4, i):
    return fx.Vector(acc4)[i]


def _zero4():
    return fx.Vector.filled(4, 0.0, fx.Float32)


def _exp2(x):
    return fx.Float32(fx.rocdl.exp2(fx.Float32.ir_type, x.ir_value()))


def _to_bf16(v):
    bits = v.bitcast(fx.Uint32) + fx.Uint32(0x00008000)
    return (bits >> fx.Uint32(16)).to(fx.Uint16).bitcast(fx.BFloat16)


def _scale_acc(acc4, alphas):
    return fx.Vector.from_elements(
        [_elem(acc4, i) * alphas[i] for i in range_constexpr(4)],
        fx.Float32,
    )


def _head_max(m, head_xor):
    # max over the four lanes (one per k-group) that hold one head
    for sh in head_xor:
        peer = m.shuffle_xor(sh, WAVE)
        m = (m > peer).select(m, peer)
    return m


def _rescale_all(acc, alpha):
    return [_scale_acc(a, [alpha] * 4) for a in acc]


# Trace-time layout helpers: loops walk static layouts, never index arithmetic.
def _py(x):
    """A static layout value (index, coordinate, extent) as Python ints."""
    return x.to_py_value()


def _ext(layout, mode):
    """Extent of one top-level mode of a static layout."""
    return _py(fx.size(fx.get_(fx.get_shape(layout), mode)))


def _rowmajor(*shape):
    return fx.make_ordered_layout(shape, tuple(reversed(range(len(shape)))))


def _colmajor(*shape):
    return fx.make_ordered_layout(shape, tuple(range(len(shape))))


def _unpack(state, counts):
    """Split flat loop state into consecutive parts of the given lengths."""
    it = iter(state)
    return [[next(it) for _ in range(n)] for n in counts]


def _halves(v8, halves):
    # 8 bf16 -> MFMA operands for tiles 2p (low half) and 2p+1 (high half); halves: (e, half) -> element
    v8 = fx.Vector(v8)
    return [
        fx.Vector.from_elements([v8[_py(fx.crd2idx((e, h), halves))] for e in range_constexpr(_ext(halves, 0))], fx.BFloat16)
        for h in range_constexpr(_ext(halves, 1))
    ]


@lru_cache(maxsize=8)
def compile_sparse_mla_qblock(block_q, has_sink=False):
    if block_q not in (1, 2, 4):
        raise ValueError("BLOCK_Q must be 1, 2, or 4")

    BLOCK_Q = int(block_q)
    HAS_SINK = bool(has_sink)
    ONE_Q = BLOCK_Q == 1
    # The bf16 one-query path uses two waves to split the 512-wide contraction
    # and output. Q-block keeps one wave per query over a shared KV tile.
    NWARP = 2 if ONE_Q else BLOCK_Q
    # arithmetic: the launch shape is needed before tracing, where no layout exists yet
    BLOCK_THREADS = NWARP * WAVE
    BLOCK_K = 16
    # D splits into 16-wide MFMA tiles: QK over NQK waves (one-query), PV over NPV waves.
    # Q-block: each wave owns a D/NWARP slice of the output for every query,
    # so one V^T operand feeds BLOCK_Q MFMAs; P^T and alpha go through LDS.
    NQK = NWARP if ONE_Q else 1
    NPV = NWARP
    QK_BATCH = QK_BATCH_PAIRS_ONE_Q if ONE_Q else QK_BATCH_PAIRS_QBLOCK
    IDX_AHEAD = 0 if ONE_Q else IDX_AHEAD_QBLOCK
    module_name = (
        f"sparse_mla_mfma_bq{BLOCK_Q}_sink{int(HAS_SINK)}_d{HEAD_DIM}_h{NUM_HEADS}"
    )

    # One-query fits 16 KB (4 CTAs/CU on gfx942): unpadded rows with the 16-B
    # column XORed by row & 7. Q-block keeps the padded row plus the exchange.
    SWZ = ONE_Q
    LD = HEAD_DIM if SWZ else LD_KV
    # Row extents only bound coordinates; strides carry the addresses.
    BIG = 1 << 24
    # bits per copy of 4 and 8 bf16
    COPY_BITS = {4: 64, 8: 128}

    def _lds_load(view, n):
        # A swizzled (composed-layout) view has no vector load, only copy atoms.
        r = fx.make_rmem_tensor(fx.make_layout(n, 1), fx.BFloat16)
        fx.copy(fx.make_copy_atom(fx.UniversalCopy(COPY_BITS[n]), fx.BFloat16), view, r)
        return r.load()

    def _lds_store(vec, view, n):
        r = fx.make_rmem_tensor(fx.make_layout(n, 1), fx.BFloat16)
        r.store(vec)
        fx.copy(fx.make_copy_atom(fx.UniversalCopy(COPY_BITS[n]), fx.BFloat16), r, view)

    def _swz(layout):
        # one-query: the 16-B chunk (offset bits 3..5) is XORed with row & 7 (bits 9..11)
        if const_expr(SWZ):
            return fx.make_composed_layout(fx.static(fx.SwizzleType.get(3, 3, 6)), layout)
        return layout

    @flyc.kernel(name=module_name, known_block_size=[BLOCK_THREADS, 1, 1])
    def sparse_mla_qblock_kernel(
        q_ptr: fx.Pointer,
        kv_ptr: fx.Pointer,
        idx_ptr: fx.Pointer,
        out_ptr: fx.Pointer,
        n_tok: fx.Int32,
        n_unique: fx.Int32,
        scale: fx.Float32,
        sink_ptr: fx.Pointer,
    ):
        # ---- static tilings: every trace-time loop walks one of these
        # D as 16-wide MFMA tiles, split over the QK and PV waves, and as 16-B chunks
        n_d_tiles = _ext(fx.zipped_divide(fx.make_layout(HEAD_DIM, 1), (16,)), 1)
        QK_D_TILES = _py(fx.shape_div(n_d_tiles, NQK))
        PV_D_TILES = _py(fx.shape_div(n_d_tiles, NPV))
        CHUNKS = _ext(fx.zipped_divide(fx.make_layout(HEAD_DIM, 1), (VEC,)), 1)
        # QK takes the tiles in pairs (one 16-B load feeds tiles 2p, 2p+1),
        # PV in groups of 4 (one V^T row read feeds tiles 4g .. 4g+3).
        qk_tiles = fx.zipped_divide(fx.make_layout(QK_D_TILES, 1), (2,))  # (half, pair)
        pv_tiles = fx.zipped_divide(fx.make_layout(PV_D_TILES, 1), (4,))  # (h, group)
        QK_PAIRS = _ext(qk_tiles, 1)
        PV_GROUPS = _ext(pv_tiles, 1)
        HALVES = _ext(qk_tiles, 0)
        H4 = _ext(pv_tiles, 0)
        qk_batches = fx.zipped_divide(fx.make_layout(QK_PAIRS, 1), (QK_BATCH,))  # (pair in batch, batch)
        # a pair's K: (pair % 2 inside the swizzled row, 64-element column block outside it)
        k_pairs = fx.zipped_divide(fx.make_layout(QK_PAIRS, 1), (2,))
        if const_expr(ONE_Q):
            pv_batches = fx.zipped_divide(fx.make_layout(PV_GROUPS, 1), (PV_BATCH_GROUPS_ONE_Q,))
        halves = fx.zipped_divide(fx.make_layout(VEC, 1), (H4,))  # 8 bf16 -> (e, half)
        # accumulators: (h, group, query) -> flat list position
        acc_l = _colmajor(H4, PV_GROUPS, 1 if ONE_Q else BLOCK_Q)
        N_ACC = _py(fx.size(acc_l))
        # this wave's gather rows: (row it, wave) -> KV-tile row
        gather_rows = fx.zipped_divide(fx.make_layout(BLOCK_K, 1), (_py(fx.ceil_div(BLOCK_K, NWARP)),))
        N_GATHER = _ext(gather_rows, 0)

        tid = fx.Int32(gpu.thread_id("x"))
        tc = fx.idx2crd(tid, _colmajor(WAVE, NWARP))
        lane = fx.get_(tc, 0).unpack()
        # SGPR warp id: readlane lane selects then need no VALU->SGPR hazard nops.
        warp_v = fx.get_(tc, 1).unpack()
        warp = fx.Int32(fx.rocdl.readfirstlane(T.i32, warp_v))
        # MFMA fragment coordinates: lane_row = lane % 16, k-group lane_g = lane // 16
        lanes = _colmajor(16, 4)
        lc = fx.idx2crd(lane, lanes)
        lane_row = fx.get_(lc, 0).unpack()
        lane_g = fx.get_(lc, 1).unpack()
        # XOR butterfly over the k-group mode: the lanes holding one head (lane_row)
        head_xor = [_py(fx.crd2idx((0, b), lanes)) for b in (1, 2)]
        tile = fx.Int32(gpu.block_id("x"))
        n_tok_i = fx.Int32(n_tok)
        # Q-block: token of (query j, tile)
        tokens = _colmajor(BLOCK_Q, BIG)
        if const_expr(ONE_Q):
            t_idx = tile
        else:
            t_idx = fx.get_scalar(fx.crd2idx((warp, tile), tokens))
        live = t_idx < n_tok_i
        # A dead Q-block wave (past the last token) reads token 0's Q; its P is zeroed below.
        q_row = live.select(t_idx, fx.Int32(0))
        qk_w = warp if const_expr(ONE_Q) else fx.Int32(0)

        # ---- memory layouts: every address below is a coordinate into one of these (elements)
        # q[token, head, qk wave, pair, k-group, 8]: MFMAs 2p, 2p+1 from one 16-B load
        q_g = fx.make_view(fx.rocdl.make_buffer_ptr(q_ptr), _rowmajor(BIG, NUM_HEADS, NQK, QK_PAIRS, 4, VEC))
        # out[token, head, pv wave, group, k-group, i, h]: acc (h, g)[i] is d = 64 g + 16 k-group + 4 i + h
        out_g = fx.make_view(fx.rocdl.make_buffer_ptr(out_ptr), _rowmajor(BIG, NUM_HEADS, NPV, PV_GROUPS, 4, 4, H4))
        # idx rows of n_unique, tiled by BLOCK_K: ((0, lane in tile), (row, k-tile)); the tile -> soffset
        n_u = fx.Int32(n_unique)
        idx_tiles = fx.zipped_divide(_rowmajor(BIG, n_u), (1, BLOCK_K))
        n_kt = fx.get_scalar(fx.size(fx.get_(fx.get_shape(idx_tiles), [1, 1])))
        idx_lane = fx.make_view(fx.rocdl.make_buffer_ptr(idx_ptr), _rowmajor(BLOCK_K, 1))[lane_row, None]
        idx_atom = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Int32)
        # tile kt + a: the next tiles a loop iteration loads ahead
        ahead = fx.make_layout((BIG, BIG), (1, 1))
        # kv[slot, 16-B chunk, 8]. The slot stays in voffset: moving it to soffset measured
        # 4% slower on the shared pattern (gfx950).
        kv_g = fx.make_view(fx.rocdl.make_buffer_ptr(kv_ptr), _rowmajor(BIG, CHUNKS, VEC))

        # LDS: the KV tile, and (Q-block) one P^T/alpha slot per thread: P^T (4 bf16) + alpha (f32)
        kv_tile_l = fx.make_layout((BLOCK_K, CHUNKS, VEC), (LD, VEC, 1))
        xch_l = _rowmajor(NWARP, WAVE, VEC)
        if const_expr(ONE_Q):
            @fx.struct
            class SharedStorage:
                kv: fx.Array[fx.BFloat16, _py(fx.cosize(kv_tile_l)), 16]
        else:
            @fx.struct
            class SharedStorage:
                kv: fx.Array[fx.BFloat16, _py(fx.cosize(kv_tile_l)), 16]
                xch: fx.Array[fx.BFloat16, _py(fx.cosize(xch_l)), 16]
        storage = fx.SharedAllocator().allocate(SharedStorage).peek()
        # LDS KV tile [row, 16-B chunk, 8]. The operand reads split D into 64-element column
        # blocks, which the swizzle never touches: the block is a pointer offset (an immediate),
        # and the swizzled layout covers the rest. K: [token, pair % 2, k-group, 8] at block
        # (qk wave, pair // 2); V^T rows: [(e, k-group), lane_row, 4] (token 4 k-group + e) at
        # block (pv wave, group).
        rows = fx.make_layout(BLOCK_K, LD)
        kv_tile = storage.kv.view(_swz(kv_tile_l))
        k_cols = _rowmajor(NQK, _ext(k_pairs, 1), 64)
        k_in = _rowmajor(2, 4, VEC)  # (pair % 2, k-group, 8) within a row
        k_rows = _swz(fx.make_layout((BLOCK_K, fx.get_shape(k_in)), (LD, fx.get_stride(k_in))))
        vt_cols = _rowmajor(NPV, PV_GROUPS, 64)
        vt_tok = fx.flat_divide(rows, (4,))  # token (e, k-group) -> row offset
        vt_rows = _swz(fx.make_layout((fx.get_shape(vt_tok), 16, 4), (fx.get_stride(vt_tok), 4, 1)))
        # one-query QK partials [(lane % 32, lane // 32), wave, 8], over the V^T rows already read
        part = storage.kv.view(fx.make_ordered_layout(((32, 2), NWARP, VEC), ((1, 3), 2, 0)))
        # wave w's peer, 1 - w (one-query: two waves)
        peer_of = fx.make_composed_layout(fx.make_layout(2, 1), 1, fx.make_layout(2, -1))
        # Q-block: (m, l) per lane after the loop [wave, lane, 4], and the P^T/alpha exchange [wave, lane, 8]
        ml_pub = storage.kv.view(_rowmajor(NWARP, WAVE, 4))
        if const_expr(not ONE_Q):
            xch = storage.xch.view(xch_l)

        qk_scale = fx.Float32(scale) * fx.Float32(LOG2E)
        # Keep -inf / 0.0 as raw arith constants: ambient fast_fp_math would
        # otherwise corrupt the empty-row ``== -inf`` guards (same pattern as
        # fused_compress_attn_hca).
        neg_inf = fx.Float32(arith.constant(NEG_INF, type=T.f32))
        zero_f = fx.Float32(arith.constant(0.0, type=T.f32))
        one_f = fx.Float32(arith.constant(1.0, type=T.f32))
        idx_row = q_row if const_expr(ONE_Q) else tile
        pv_w = warp
        mma = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 16, fx.BFloat16))

        # QK pairs MFMAs 2p, 2p+1 over one 16-B load per lane: lane (g, s)
        # owns dims 32p + 8g .. +7, low half for 2p and high half for 2p+1.
        q_frags = [
            _halves(q_g[q_row, lane_row, qk_w, pr, lane_g, None].load(), halves)
            for pr in range_constexpr(QK_PAIRS)
        ]

        def _load_idx(kt):
            # arithmetic: compare + select; past-the-end prefetches re-read tile 0 so the load
            # stays in the row (a ring layout would need a runtime divide by n_kt).
            kt = (kt < n_kt).select(kt, fx.Int32(0))
            off = fx.get_scalar(fx.crd2idx(((0, 0), (idx_row, kt)), idx_tiles))
            atom = fx.atom_set_value(idx_atom, "soffset", off)
            r = fx.make_rmem_tensor(fx.make_layout(1, 1), fx.Int32)
            fx.copy(atom, idx_lane, r)
            return fx.Int32(fx.Vector(r.load())[0])

        def _gather(idx_vec):
            frags = []
            for it in range_constexpr(N_GATHER):
                lane_sel = fx.get_scalar(fx.crd2idx((it, warp), gather_rows))
                slot = fx.Int32(fx.rocdl.readlane(T.i32, idx_vec, lane_sel))
                frags.append(kv_g[slot, lane, None].load())
            return frags

        def _store_kv(frags):
            for it in range_constexpr(N_GATHER):
                # warp_v (VGPR): the lane base absorbs the wave's rows, so row `it` is an immediate.
                _lds_store(fx.Vector(frags[it]), kv_tile[fx.crd2idx((it, warp_v), gather_rows), lane, None], VEC)

        def _k_a(pr):
            # A operands for MFMAs 2pr, 2pr+1: K[token = lane_row][d of q_frags[pr]].
            sub, blk = _py(fx.get_flat_coord(pr, k_pairs))
            k_op = (storage.kv.ptr + fx.crd2idx((qk_w, blk, 0), k_cols)).view(k_rows)
            return _halves(_lds_load(k_op[lane_row, (sub, lane_g, None)], VEC), halves)

        def _vt_rows(g):
            vt_op = (storage.kv.ptr + fx.crd2idx((pv_w, g, 0), vt_cols)).view(vt_rows)
            return [_lds_load(vt_op[(e, lane_g), lane_row, None], 4) for e in range_constexpr(4)]

        def _vt_group(g, raw=None):
            # A operands for tiles (h, g): row r is d = 64g + 4r + h.
            rows_g = [fx.Vector(r) for r in (raw if raw is not None else _vt_rows(g))]
            return [
                fx.Vector.from_elements([rows_g[e][h] for e in range_constexpr(4)], fx.BFloat16)
                for h in range_constexpr(H4)
            ]

        def _acc_at(acc, h, g, j):
            return acc[_py(fx.crd2idx((h, g, j), acc_l))]

        def _query_accs(acc, j):
            return [_acc_at(acc, h, g, j) for g in range_constexpr(PV_GROUPS) for h in range_constexpr(H4)]

        def _with_query_accs(acc, j, vals):
            acc = list(acc)
            pos = [_py(fx.crd2idx((h, g, j), acc_l)) for g in range_constexpr(PV_GROUPS) for h in range_constexpr(H4)]
            for p, v in zip(pos, vals):
                acc[p] = v
            return acc

        # Scores come out as S^T (token = lane_k + i, head = lane_row): every
        # lane owns one head, and P^T is already the PV B operand.
        m_i = neg_inf
        l_i = zero_f
        acc = [_zero4() for _ in range_constexpr(N_ACC)]
        prefetched = _gather(_load_idx(fx.Int32(0)))
        idx_q = [_load_idx(fx.get_scalar(fx.crd2idx((a, 1), ahead))) for a in range_constexpr(IDX_AHEAD)]
        loop_init = [m_i, l_i] + acc + prefetched + idx_q
        for kt, state in range(fx.Int32(0), n_kt, fx.Int32(1), init=loop_init):
            (m_i,), (l_i,), acc, prefetched, idx_q = _unpack(state, [1, 1, N_ACC, N_GATHER, IDX_AHEAD])
            m_i = fx.Float32(m_i)
            l_i = fx.Float32(l_i)
            acc = [fx.Vector(a) for a in acc]
            prefetched = [fx.Vector(v) for v in prefetched]
            idx_q = [fx.Int32(v) for v in idx_q]
            kt_i32 = fx.Int32(kt)
            kt_next = fx.get_scalar(fx.crd2idx((kt_i32, 1), ahead))

            fx.rocdl.s_waitcnt(vmcnt=0)
            _store_kv(prefetched)
            gpu.barrier()
            # Issue the next tile's gather before any LDS read so the bytes
            # stay outstanding across QK, softmax, and PV.
            next_prefetched = prefetched
            if const_expr(IDX_AHEAD == 0):
                idx_q = [_load_idx(kt_next)]
            idx_cur = idx_q[0]
            idx_q = idx_q[1:]
            if const_expr(IDX_AHEAD > 0):
                idx_q.append(_load_idx(fx.get_scalar(fx.crd2idx((kt_next, IDX_AHEAD), ahead))))
            if kt_next < n_kt:
                next_prefetched = _gather(idx_cur)
            fx.rocdl.sched_barrier(0)

            # One accumulator chain per half (tiles 2p and 2p+1), summed after the chain.
            s_accs = [_zero4() for _ in range_constexpr(HALVES)]
            fx.rocdl.s_setprio(1)
            for b in range_constexpr(_ext(qk_batches, 1)):
                pairs = [_py(fx.crd2idx((p, b), qk_batches)) for p in range_constexpr(_ext(qk_batches, 0))]
                buf = [_k_a(pr) for pr in pairs]
                fx.rocdl.sched_barrier(0)
                for pr, k in zip(pairs, buf):
                    for h in range_constexpr(HALVES):
                        s_accs[h] = _mfma_ssa(mma, k[h], q_frags[pr][h], s_accs[h])
            s_acc = s_accs[0]
            for a in s_accs[1:]:
                s_acc = fx.Vector.from_elements(
                    [_elem(s_acc, i) + _elem(a, i) for i in range_constexpr(4)], fx.Float32
                )

            fx.rocdl.s_setprio(0)
            if const_expr(ONE_Q):
                # Read this wave's whole V^T slice first: the partial below overwrites
                # rows 0-1 of the same D half (LDS ops retire in order per wave).
                v_raw = [_vt_rows(g) for g in range_constexpr(PV_GROUPS)]

                part[lane, warp, None].store(fx.Vector(s_acc).bitcast(fx.BFloat16))
                gpu.barrier()
                peer_w = fx.get_scalar(fx.crd2idx(warp, peer_of))
                peer = fx.Vector(part[lane, peer_w, None].load()).bitcast(fx.Float32)
                s_acc = fx.Vector.from_elements(
                    [_elem(s_acc, i) + peer[i] for i in range_constexpr(4)], fx.Float32
                )

            scores = [live.select(_elem(s_acc, i), neg_inf) for i in range_constexpr(4)]
            m_w = scores[0]
            for i in range_constexpr(1, 4):
                m_w = (m_w > scores[i]).select(m_w, scores[i])
            # Lazy rescale: keep the stale head max until some lane's local max grows by
            # more than RESCALE_TAU (log2 units), so p <= 2**RESCALE_TAU. Only then is the
            # cross-lane head max needed. -inf - -inf is NaN -> no move.
            moved = fx.Int64(fx.rocdl.ballot(fx.Int64.ir_type, (m_w - m_i) * qk_scale > fx.Float32(RESCALE_TAU)))
            m_n = m_i
            alpha = one_f
            if moved != fx.Int64(0):
                m_h = _head_max(m_w, head_xor)
                m_n = (m_h > m_i).select(m_h, m_i)
                alpha = (m_n == neg_inf).select(zero_f, _exp2((m_i - m_n) * qk_scale))
                if const_expr(ONE_Q):
                    acc = _rescale_all(acc, alpha)
            empty = m_n == neg_inf

            p_t = []
            for i in range_constexpr(4):
                p = empty.select(zero_f, _exp2((scores[i] - m_n) * qk_scale))
                p_t.append(live.select(p, zero_f))
            # l stays a per-lane partial (alpha is uniform across the head's
            # four lanes); the cross-lane sum happens once after the loop.
            l_i = l_i * alpha + ((p_t[0] + p_t[1]) + (p_t[2] + p_t[3]))
            m_i = m_n
            p_b = fx.Vector.from_elements([_to_bf16(p) for p in p_t], fx.BFloat16)

            if const_expr(ONE_Q):
                fx.rocdl.s_setprio(1)
                for b in range_constexpr(_ext(pv_batches, 1)):
                    groups = [_py(fx.crd2idx((gg, b), pv_batches)) for gg in range_constexpr(_ext(pv_batches, 0))]
                    va = [_vt_group(g, v_raw[g]) for g in groups]
                    fx.rocdl.sched_barrier(0)
                    for g, vg in zip(groups, va):
                        for h in range_constexpr(H4):
                            t = _py(fx.crd2idx((h, g, 0), acc_l))
                            acc[t] = _mfma_ssa(mma, vg[h], p_b, acc[t])
                fx.rocdl.s_setprio(0)
            else:
                pf = p_b.bitcast(fx.Float32)
                xch[warp_v, lane, None].store(
                    fx.Vector.from_elements([pf[0], pf[1], alpha, zero_f], fx.Float32).bitcast(fx.BFloat16)
                )
                fx.rocdl.sched_barrier(0)
                gpu.barrier()
                fx.rocdl.sched_barrier(0)
                xs = [
                    fx.Vector(xch[j, lane, None].load()).bitcast(fx.Float32)
                    for j in range_constexpr(BLOCK_Q)
                ]
                p_bs = [fx.Vector.from_elements([x[0], x[1]], fx.Float32).bitcast(fx.BFloat16) for x in xs]
                for j in range_constexpr(BLOCK_Q):
                    alpha_j = fx.Float32(xs[j][2])
                    # alpha is exactly 1 unless query j's max moved: skip when no lane's did.
                    moved_j = fx.Int64(fx.rocdl.ballot(fx.Int64.ir_type, alpha_j != one_f))
                    if moved_j != fx.Int64(0):
                        acc = _with_query_accs(acc, j, _rescale_all(_query_accs(acc, j), alpha_j))
                fx.rocdl.s_setprio(1)
                for g in range_constexpr(PV_GROUPS):
                    va = _vt_group(g)
                    fx.rocdl.sched_barrier(0)
                    for j in range_constexpr(BLOCK_Q):
                        for h in range_constexpr(H4):
                            t = _py(fx.crd2idx((h, g, j), acc_l))
                            acc[t] = _mfma_ssa(mma, va[h], p_bs[j], acc[t])
                fx.rocdl.s_setprio(0)
            gpu.barrier()
            prefetched = next_prefetched
            results = yield [m_i, l_i] + acc + prefetched + idx_q

        (m_i,), (l_i,), acc = _unpack(results, [1, 1, N_ACC])
        m_i = fx.Float32(m_i)
        l_i = fx.Float32(l_i)
        acc = [fx.Vector(a) for a in acc]
        for sh in head_xor:
            l_i = l_i + l_i.shuffle_xor(sh, WAVE)
        if const_expr(not ONE_Q):
            # Every wave stores some query's slice: publish (m, l) per query through LDS.
            gpu.barrier()
            ml_pub[warp, lane, None].store(fx.Vector.from_elements([m_i, l_i], fx.Float32).bitcast(fx.BFloat16))
            gpu.barrier()
            # Re-derive lane ids after the loop: hoisted copies spill at the 256 cap on gfx942.
            m1 = fx.Int32(-1).ir_value()
            lane = fx.Int32(fx.rocdl.mbcnt_hi(T.i32, m1, fx.rocdl.mbcnt_lo(T.i32, m1, fx.Int32(0).ir_value())))
            lc = fx.idx2crd(lane, lanes)
            lane_row = fx.get_(lc, 0).unpack()
            lane_g = fx.get_(lc, 1).unpack()

        tiny = fx.Float32(1.0e-30)
        head = lane_row

        def _inv(m_q, l_q):
            denom = l_q
            alpha_q = fx.Float32(1.0)
            if const_expr(HAS_SINK):
                sink = fx.make_view(fx.rocdl.make_buffer_ptr(sink_ptr), fx.make_layout(NUM_HEADS, 1))[head]
                m_scaled = m_q * fx.Float32(scale)
                m_final = (sink > m_scaled).select(sink, m_scaled)
                alpha_q = _exp2((m_scaled - m_final) * fx.Float32(LOG2E))
                sink_term = _exp2((sink - m_final) * fx.Float32(LOG2E))
                denom = denom * alpha_q + sink_term
            denom = (denom > tiny).select(denom, tiny)
            return alpha_q / denom

        def _store_out(t_q, j, inv):
            for g in range_constexpr(PV_GROUPS):
                for i in range_constexpr(4):
                    vals = [_to_bf16(_elem(_acc_at(acc, h, g, j), i) * inv) for h in range_constexpr(H4)]
                    out_g[t_q, head, pv_w, g, lane_g, i, None].store(fx.Vector.from_elements(vals, fx.BFloat16))

        if const_expr(ONE_Q):
            inv = _inv(m_i, l_i)
            if live:
                _store_out(t_idx, 0, inv)
        else:
            for j in range_constexpr(BLOCK_Q):
                t_j = fx.get_scalar(fx.crd2idx((j, tile), tokens))
                ml = fx.Vector(ml_pub[j, lane, None].load()).bitcast(fx.Float32)
                inv_j = _inv(fx.Float32(ml[0]), fx.Float32(ml[1]))
                if t_j < n_tok_i:
                    _store_out(t_j, j, inv_j)

    @flyc.jit
    def launch_sparse_mla_qblock(
        q_ptr: fx.Pointer,
        kv_ptr: fx.Pointer,
        idx_ptr: fx.Pointer,
        out_ptr: fx.Pointer,
        n_tok: fx.Int32,
        n_unique: fx.Int32,
        scale: fx.Float32,
        sink_ptr: fx.Pointer,
        n_tiles: fx.Int32,
        stream: fx.Stream,
    ):
        sparse_mla_qblock_kernel(
            q_ptr,
            kv_ptr,
            idx_ptr,
            out_ptr,
            n_tok,
            n_unique,
            scale,
            sink_ptr,
        ).launch(
            grid=(fx.Int64(n_tiles), 1, 1),
            block=(BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    launch_sparse_mla_qblock.compile_hints = {
        "waves_per_eu": 2,
        "fast_fp_math": True,
        "unsafe_fp_math": True,
        "llvm_options": {
            "amdgpu-kernarg-preload": AITER_FLYDSL_KERNARG_PRELOAD,
            "amdgpu-kernarg-preload-count": AITER_FLYDSL_KERNARG_PRELOAD_COUNT,
        },
    }
    return launch_sparse_mla_qblock
