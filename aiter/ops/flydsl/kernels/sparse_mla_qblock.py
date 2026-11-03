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
N_D_TILES = HEAD_DIM // 16
# 520 is the gfx942 gluon reference pitch.
LD_KV = HEAD_DIM + 8
# Independent QK accumulator chains, summed after the chain.
QK_ACCS = 2
MFMA_CHUNK_ONE_Q = 8
MFMA_CHUNK_QBLOCK = 4
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


def _head_max(m):
    # max over the four lanes (16 apart) that hold one head
    for sh in (16, 32):
        peer = m.shuffle_xor(sh, WAVE)
        m = (m > peer).select(m, peer)
    return m


def _rescale_all(acc, alpha):
    return [_scale_acc(a, [alpha] * 4) for a in acc]


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
    BLOCK_THREADS = NWARP * WAVE
    BLOCK_K = 16
    QK_D_TILES = N_D_TILES // NWARP if ONE_Q else N_D_TILES
    # Q-block: each wave owns a D/NWARP slice of the output for every query,
    # so one V^T operand feeds BLOCK_Q MFMAs; P^T and alpha go through LDS.
    PV_D_TILES = QK_D_TILES if ONE_Q else N_D_TILES // NWARP
    N_ACC = PV_D_TILES * (1 if ONE_Q else BLOCK_Q)
    N_GATHER = (BLOCK_K * (HEAD_DIM // VEC) + BLOCK_THREADS - 1) // BLOCK_THREADS
    MFMA_CHUNK = MFMA_CHUNK_ONE_Q if ONE_Q else MFMA_CHUNK_QBLOCK
    IDX_AHEAD = 0 if ONE_Q else IDX_AHEAD_QBLOCK
    module_name = (
        f"sparse_mla_mfma_bq{BLOCK_Q}_sink{int(HAS_SINK)}_d{HEAD_DIM}_h{NUM_HEADS}"
    )

    # P^T (4 bf16) + alpha (f32) per lane in one 16-B slot.
    XCH_ELEMS = 0 if ONE_Q else NWARP * WAVE * VEC
    # One-query fits 16 KB (4 CTAs/CU on gfx942): unpadded rows with the 16-B
    # column XORed by row & 7. Q-block keeps the padded row plus the exchange.
    SWZ = ONE_Q
    LD = HEAD_DIM if SWZ else LD_KV
    TILE_ELEMS = BLOCK_K * LD


    # Layout modes that split D: QK over NQK waves (one-query), PV over NPV waves.
    NQK = NWARP if ONE_Q else 1
    NPV = NWARP
    QK_PAIRS = QK_D_TILES // 2
    PV_GROUPS = PV_D_TILES // 4
    CHUNKS = HEAD_DIM // VEC
    # Row extents only bound coordinates; strides carry the addresses.
    BIG = 1 << 24

    if const_expr(ONE_Q):
        @fx.struct
        class SharedStorage:
            kv: fx.Array[fx.BFloat16, TILE_ELEMS, 16]
    else:
        @fx.struct
        class SharedStorage:
            kv: fx.Array[fx.BFloat16, TILE_ELEMS, 16]
            xch: fx.Array[fx.BFloat16, XCH_ELEMS, 16]

    def _lds_load(view, n):
        # A swizzled (composed-layout) view has no vector load, only copy atoms.
        r = fx.make_rmem_tensor(fx.make_layout(n, 1), fx.BFloat16)
        fx.copy(fx.make_copy_atom(fx.UniversalCopy(n * 16), fx.BFloat16), view, r)
        return r.load()

    def _lds_store(vec, view, n):
        r = fx.make_rmem_tensor(fx.make_layout(n, 1), fx.BFloat16)
        r.store(vec)
        fx.copy(fx.make_copy_atom(fx.UniversalCopy(n * 16), fx.BFloat16), r, view)

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
        tid = fx.Int32(gpu.thread_id("x"))
        tc = fx.idx2crd(tid, fx.make_layout((WAVE, NWARP), (1, WAVE)))
        lane = fx.get_(tc, 0).unpack()
        # SGPR warp id: readlane lane selects then need no VALU->SGPR hazard nops.
        warp_v = fx.get_(tc, 1).unpack()
        warp = fx.Int32(fx.rocdl.readfirstlane(T.i32, warp_v))
        # MFMA fragment coordinates: lane_row = lane % 16, k-group lane_g = lane // 16
        lc = fx.idx2crd(lane, fx.make_layout((16, 4), (1, 16)))
        lane_row = fx.get_(lc, 0).unpack()
        lane_g = fx.get_(lc, 1).unpack()
        tile = fx.Int32(gpu.block_id("x"))
        n_tok_i = fx.Int32(n_tok)
        # Q-block: token (query j of tile) = j + BLOCK_Q tile
        tokens = fx.make_layout((BLOCK_Q, BIG), (1, BLOCK_Q))
        if const_expr(ONE_Q):
            t_idx = tile
        else:
            t_idx = fx.get_scalar(fx.crd2idx((warp, tile), tokens))
        live = t_idx < n_tok_i
        q_row = live.select(t_idx, n_tok_i - fx.Int32(1))
        qk_w = warp if const_expr(ONE_Q) else fx.Int32(0)

        # ---- layouts: every address below is a coordinate into one of these (elements)
        # q[token, head, (qk wave, pair), k-group, 8]: MFMAs 2p, 2p+1 from one 16-B load
        q_g = fx.make_view(fx.rocdl.make_buffer_ptr(q_ptr), fx.make_layout(
            (BIG, NUM_HEADS, (NQK, QK_PAIRS), 4, VEC),
            (NUM_HEADS * HEAD_DIM, HEAD_DIM, (QK_PAIRS * 32, 32), VEC, 1)))
        # out[token, head, (pv wave, group), k-group, i, h]: acc[4g + h][i] is d = 64 g + 16 k-group + 4 i + h
        out_g = fx.make_view(fx.rocdl.make_buffer_ptr(out_ptr), fx.make_layout(
            (BIG, NUM_HEADS, (NPV, PV_GROUPS), 4, 4, 4),
            (NUM_HEADS * HEAD_DIM, HEAD_DIM, (PV_GROUPS * 64, 64), 16, 4, 1)))
        # idx[(row, k-tile)] -> soffset; lanes 16..63 alias 0..15 (stride 0) so no lane leaves the tile
        n_u = fx.Int32(n_unique)
        idx_tiles = fx.make_layout((BIG, BIG), (n_u, BLOCK_K))
        idx_lane = fx.slice(fx.make_view(fx.rocdl.make_buffer_ptr(idx_ptr), fx.make_layout(
            ((BLOCK_K, WAVE // BLOCK_K), 1), ((1, 0), 1))), (lane, None))
        idx_atom = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Int32)
        # kv[slot, 16-B chunk, 8]. The slot stays in voffset: moving it to soffset measured
        # 4% slower on the shared pattern (gfx950).
        kv_g = fx.make_view(fx.rocdl.make_buffer_ptr(kv_ptr), fx.make_layout(
            (BIG, CHUNKS, VEC), (HEAD_DIM, VEC, 1)))
        # this wave's gather rows -> KV-tile row
        gather_rows = fx.make_layout((NWARP, N_GATHER), (N_GATHER, 1))

        storage = fx.SharedAllocator().allocate(SharedStorage).peek()
        # LDS KV tile [row, 16-B chunk, 8]. The operand reads split D into 64-element column
        # blocks, which the swizzle never touches: the block is a pointer offset (an immediate),
        # and the swizzled layout covers the rest. K: [token, pair % 2, k-group, 8] at block
        # (qk wave, pair // 2); V^T rows: [k-group, e, lane_row, 4] (token 4 k-group + e) at
        # block (pv wave, group).
        kv_tile = storage.kv.view(_swz(fx.make_layout((BLOCK_K, CHUNKS, VEC), (LD, VEC, 1))))
        k_cols = fx.make_layout((NQK, QK_PAIRS // 2), (QK_PAIRS * 32, 64))
        k_rows = _swz(fx.make_layout((BLOCK_K, 2, 4, VEC), (LD, 32, VEC, 1)))
        vt_cols = fx.make_layout((NPV, PV_GROUPS), (PV_GROUPS * 64, 64))
        vt_rows = _swz(fx.make_layout((4, 4, 16, 4), (4 * LD, LD, 4, 1)))
        # one-query QK partials [(lane % 32, lane // 32), wave, 8], over the V^T rows already read
        part = storage.kv.view(fx.make_layout(((32, 2), NWARP, VEC), ((VEC, HEAD_DIM), HEAD_DIM // 2, 1)))
        # Q-block: (m, l) per lane after the loop [wave, lane, 4], and the P^T/alpha exchange [wave, lane, 8]
        ml_pub = storage.kv.view(fx.make_layout((NWARP, WAVE, 4), (WAVE * 4, 4, 1)))
        if const_expr(not ONE_Q):
            xch = storage.xch.view(fx.make_layout((NWARP, WAVE, VEC), (WAVE * VEC, VEC, 1)))

        qk_scale = fx.Float32(scale) * fx.Float32(LOG2E)
        # Keep -inf / 0.0 as raw arith constants: ambient fast_fp_math would
        # otherwise corrupt the empty-row ``== -inf`` guards (same pattern as
        # fused_compress_attn_hca).
        neg_inf = fx.Float32(arith.constant(NEG_INF, type=T.f32))
        zero_f = fx.Float32(arith.constant(0.0, type=T.f32))
        one_f = fx.Float32(arith.constant(1.0, type=T.f32))
        n_kt = (n_u + fx.Int32(BLOCK_K - 1)) // fx.Int32(BLOCK_K)
        idx_row = q_row if const_expr(ONE_Q) else tile
        pv_w = warp
        mma = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 16, fx.BFloat16))

        def _halves(v8):
            v8 = fx.Vector(v8)
            return [
                fx.Vector.from_elements([v8[h * 4 + e] for e in range_constexpr(4)], fx.BFloat16)
                for h in range_constexpr(2)
            ]

        # QK pairs MFMAs 2p, 2p+1 over one 16-B load per lane: lane (g, s)
        # owns dims 32p + 8g .. +7, low half for 2p and high half for 2p+1.
        q_frags = []
        for pr in range_constexpr(QK_PAIRS):
            q_frags += _halves(q_g[q_row, lane_row, (qk_w, pr), lane_g, None].load())

        def _load_idx(kt):
            # Past-the-end prefetches re-read tile 0 so the load stays in the row.
            kt = (kt < n_kt).select(kt, fx.Int32(0))
            atom = fx.atom_set_value(idx_atom, "soffset", fx.get_scalar(fx.crd2idx((idx_row, kt), idx_tiles)))
            r = fx.make_rmem_tensor(fx.make_layout(1, 1), fx.Int32)
            fx.copy(atom, idx_lane, r)
            return fx.Int32(fx.Vector(r.load())[0])

        def _gather(idx_vec):
            frags = []
            for it in range_constexpr(N_GATHER):
                lane_sel = fx.get_scalar(fx.crd2idx((warp, it), gather_rows))
                slot = fx.Int32(fx.rocdl.readlane(T.i32, idx_vec, lane_sel))
                frags.append(kv_g[slot, lane, None].load())
            return frags

        def _store_kv(frags):
            for it in range_constexpr(N_GATHER):
                # warp_v (VGPR): the lane base absorbs the wave's rows, so row `it` is an immediate.
                _lds_store(fx.Vector(frags[it]), kv_tile[fx.crd2idx((warp_v, it), gather_rows), lane, None], VEC)

        def _k_a(pr):
            # A operands for MFMAs 2pr, 2pr+1: K[token = lane_row][d of q_frags].
            k_op = (storage.kv.ptr + fx.crd2idx((qk_w, pr // 2), k_cols)).view(k_rows)
            return _halves(_lds_load(k_op[lane_row, pr % 2, lane_g, None], VEC))

        def _vt_rows(g):
            vt_op = (storage.kv.ptr + fx.crd2idx((pv_w, g), vt_cols)).view(vt_rows)
            return [_lds_load(vt_op[lane_g, e, lane_row, None], 4) for e in range_constexpr(4)]

        def _vt_group(g, raw=None):
            # A operands for tiles 4g .. 4g+3: tile 4g + h, row r is d = 64g + 4r + h.
            rows = [fx.Vector(r) for r in (raw if raw is not None else _vt_rows(g))]
            return [
                fx.Vector.from_elements([rows[e][h] for e in range_constexpr(4)], fx.BFloat16)
                for h in range_constexpr(4)
            ]

        # Scores come out as S^T (token = lane_k + i, head = lane_row): every
        # lane owns one head, and P^T is already the PV B operand.
        m_i = neg_inf
        l_i = zero_f
        acc = [_zero4() for _ in range_constexpr(N_ACC)]
        prefetched = _gather(_load_idx(fx.Int32(0)))
        idx_q = [_load_idx(fx.Int32(a + 1)) for a in range_constexpr(IDX_AHEAD)]
        loop_init = [m_i, l_i] + acc + prefetched + idx_q
        for kt, state in range(fx.Int32(0), n_kt, fx.Int32(1), init=loop_init):
            m_i = fx.Float32(state[0])
            l_i = fx.Float32(state[1])
            acc = [fx.Vector(state[2 + i]) for i in range_constexpr(N_ACC)]
            prefetched = [
                fx.Vector(state[2 + N_ACC + i]) for i in range_constexpr(len(prefetched))
            ]
            idx_q = [
                fx.Int32(state[2 + N_ACC + len(prefetched) + a]) for a in range_constexpr(IDX_AHEAD)
            ]
            kt_i32 = fx.Int32(kt)

            fx.rocdl.s_waitcnt(vmcnt=0)
            _store_kv(prefetched)
            gpu.barrier()
            # Issue the next tile's gather before any LDS read so the bytes
            # stay outstanding across QK, softmax, and PV.
            next_prefetched = prefetched
            if const_expr(IDX_AHEAD == 0):
                idx_q = [_load_idx(kt_i32 + fx.Int32(1))]
            idx_cur = idx_q[0]
            idx_q = idx_q[1:]
            if const_expr(IDX_AHEAD > 0):
                idx_q.append(_load_idx(kt_i32 + fx.Int32(IDX_AHEAD + 1)))
            if kt_i32 + fx.Int32(1) < n_kt:
                next_prefetched = _gather(idx_cur)
            fx.rocdl.sched_barrier(0)

            s_accs = [_zero4() for _ in range_constexpr(QK_ACCS)]
            fx.rocdl.s_setprio(1)
            n_qk = len(q_frags)
            qk_chunk = MFMA_CHUNK
            for chunk in range_constexpr(n_qk // qk_chunk):
                base = chunk * qk_chunk
                buf = []
                for i in range_constexpr(MFMA_CHUNK // 2):
                    buf += _k_a(chunk * (MFMA_CHUNK // 2) + i)
                fx.rocdl.sched_barrier(0)
                for i in range_constexpr(qk_chunk):
                    a = (base + i) % QK_ACCS
                    s_accs[a] = _mfma_ssa(
                        mma, buf[i], q_frags[base + i], s_accs[a]
                    )
            s_acc = s_accs[0]
            for a in range_constexpr(1, QK_ACCS):
                s_acc = fx.Vector.from_elements(
                    [_elem(s_acc, i) + _elem(s_accs[a], i) for i in range_constexpr(4)], fx.Float32
                )

            fx.rocdl.s_setprio(0)
            if const_expr(ONE_Q):
                # Read this wave's whole V^T slice first: the partial below overwrites
                # rows 0-1 of the same D half (LDS ops retire in order per wave).
                v_raw = [_vt_rows(g) for g in range_constexpr(PV_D_TILES // 4)]

                part[lane, warp, None].store(fx.Vector(s_acc).bitcast(fx.BFloat16))
                gpu.barrier()
                peer = fx.Vector(part[lane, warp ^ fx.Int32(1), None].load()).bitcast(fx.Float32)
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
                m_h = _head_max(m_w)
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
                for chunk in range_constexpr(PV_D_TILES // MFMA_CHUNK):
                    base = chunk * MFMA_CHUNK
                    va = []
                    for g in range_constexpr(MFMA_CHUNK // 4):
                        va += _vt_group(base // 4 + g, v_raw[base // 4 + g])
                    fx.rocdl.sched_barrier(0)
                    for i in range_constexpr(MFMA_CHUNK):
                        acc[base + i] = _mfma_ssa(mma, va[i], p_b, acc[base + i])
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
                        acc[j * PV_D_TILES:(j + 1) * PV_D_TILES] = _rescale_all(
                            acc[j * PV_D_TILES:(j + 1) * PV_D_TILES], alpha_j
                        )
                fx.rocdl.s_setprio(1)
                for g in range_constexpr(PV_D_TILES // 4):
                    va = _vt_group(g)
                    fx.rocdl.sched_barrier(0)
                    for j in range_constexpr(BLOCK_Q):
                        for h in range_constexpr(4):
                            t = j * PV_D_TILES + 4 * g + h
                            acc[t] = _mfma_ssa(mma, va[h], p_bs[j], acc[t])
                fx.rocdl.s_setprio(0)
            gpu.barrier()
            prefetched = next_prefetched
            results = yield [m_i, l_i] + acc + prefetched + idx_q

        m_i = fx.Float32(results[0])
        l_i = fx.Float32(results[1])
        acc = [fx.Vector(results[2 + i]) for i in range_constexpr(N_ACC)]
        for sh in (16, 32):
            l_i = l_i + l_i.shuffle_xor(sh, WAVE)
        if const_expr(not ONE_Q):
            # Every wave stores some query's slice: publish (m, l) per query through LDS.
            gpu.barrier()
            ml_pub[warp, lane, None].store(fx.Vector.from_elements([m_i, l_i], fx.Float32).bitcast(fx.BFloat16))
            gpu.barrier()
            # Re-derive lane ids after the loop: hoisted copies spill at the 256 cap on gfx942.
            m1 = fx.Int32(-1).ir_value()
            lane = fx.Int32(fx.rocdl.mbcnt_hi(T.i32, m1, fx.rocdl.mbcnt_lo(T.i32, m1, fx.Int32(0).ir_value())))
            lc = fx.idx2crd(lane, fx.make_layout((16, 4), (1, 16)))
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

        def _store_out(t_q, acc_q, inv):
            for g in range_constexpr(PV_GROUPS):
                for i in range_constexpr(4):
                    vals = [_to_bf16(_elem(acc_q[4 * g + h], i) * inv) for h in range_constexpr(4)]
                    out_g[t_q, head, (pv_w, g), lane_g, i, None].store(fx.Vector.from_elements(vals, fx.BFloat16))

        if const_expr(ONE_Q):
            inv = _inv(m_i, l_i)
            if live:
                _store_out(t_idx, acc, inv)
        else:
            for j in range_constexpr(BLOCK_Q):
                t_j = fx.get_scalar(fx.crd2idx((j, tile), tokens))
                ml = fx.Vector(ml_pub[j, lane, None].load()).bitcast(fx.Float32)
                inv_j = _inv(fx.Float32(ml[0]), fx.Float32(ml[1]))
                if t_j < n_tok_i:
                    _store_out(t_j, acc[j * PV_D_TILES:(j + 1) * PV_D_TILES], inv_j)

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
