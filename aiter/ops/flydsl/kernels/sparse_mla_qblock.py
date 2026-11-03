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

    if const_expr(ONE_Q):
        @fx.struct
        class SharedStorage:
            kv: fx.Array[fx.BFloat16, TILE_ELEMS, 16]
    else:
        @fx.struct
        class SharedStorage:
            kv: fx.Array[fx.BFloat16, TILE_ELEMS + XCH_ELEMS, 16]

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
        # SGPR warp id: readlane lane selects then need no VALU->SGPR hazard nops.
        warp = fx.Int32(fx.rocdl.readfirstlane(T.i32, tid // fx.Int32(WAVE)))
        lane = tid - warp * fx.Int32(WAVE)
        lane_row = lane % fx.Int32(16)
        lane_k = (lane // fx.Int32(16)) * fx.Int32(4)
        tile = fx.Int32(gpu.block_id("x"))
        n_tok_i = fx.Int32(n_tok)
        if const_expr(ONE_Q):
            t_idx = tile
        else:
            t_idx = tile * fx.Int32(BLOCK_Q) + warp
        live = t_idx < n_tok_i
        q_row = live.select(t_idx, n_tok_i - fx.Int32(1))

        storage = fx.SharedAllocator().allocate(SharedStorage)
        lds_kv = storage.kv.peek().ptr

        qk_scale = fx.Float32(scale) * fx.Float32(LOG2E)
        # Keep -inf / 0.0 as raw arith constants: ambient fast_fp_math would
        # otherwise corrupt the empty-row ``== -inf`` guards (same pattern as
        # fused_compress_attn_hca).
        neg_inf = fx.Float32(arith.constant(NEG_INF, type=T.f32))
        zero_f = fx.Float32(arith.constant(0.0, type=T.f32))
        one_f = fx.Float32(arith.constant(1.0, type=T.f32))
        n_u = fx.Int32(n_unique)
        n_kt = (n_u + fx.Int32(BLOCK_K - 1)) // fx.Int32(BLOCK_K)
        idx_row = q_row if const_expr(ONE_Q) else tile
        qk_d0 = warp * fx.Int32(QK_D_TILES * 16) if const_expr(ONE_Q) else fx.Int32(0)
        pv_d0 = qk_d0 if const_expr(ONE_Q) else warp * fx.Int32(PV_D_TILES * 16)
        mma = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 16, fx.BFloat16))
        v4bf16 = fx.Vector.make_type(4, fx.BFloat16)
        v8bf16 = fx.Vector.make_type(VEC, fx.BFloat16)

        def _halves(v8):
            v8 = fx.Vector(v8)
            return [
                fx.Vector.from_elements([v8[h * 4 + e] for e in range_constexpr(4)], fx.BFloat16)
                for h in range_constexpr(2)
            ]

        # QK pairs MFMAs 2p, 2p+1 over one 16-B load per lane: lane (g, s)
        # owns dims 32p + 8g .. +7, low half for 2p and high half for 2p+1.
        q_frags = []
        for pr in range_constexpr(QK_D_TILES // 2):
            q_off = (
                q_row * fx.Int32(NUM_HEADS * HEAD_DIM)
                + lane_row * fx.Int32(HEAD_DIM)
                + qk_d0
                + fx.Int32(pr * 32)
                + lane_k * fx.Int32(2)
            )
            q8 = fx.ptr_load(q_ptr + q_off, result_type=v8bf16)
            q_frags += _halves(q8)

        def _load_idx(k_base):
            # Address first, then load. A select around the load still issues
            # it for lanes 16..63 and walks off the end of the last query.
            # Past-the-end prefetches re-read tile 0 so the load stays in the row.
            k_base = (k_base < n_u).select(k_base, fx.Int32(0))
            ok = (lane < fx.Int32(BLOCK_K)) & (k_base + lane < n_u)
            off = ok.select(lane, fx.Int32(0))
            return fx.ptr_load(
                idx_ptr + idx_row * n_u + k_base + off,
                result_type=fx.Int32,
            )

        row0 = warp * fx.Int32(N_GATHER)

        def _row(r):
            return r * fx.Int32(LD)

        def _c16(c, r):
            # element offset of 16-B column c in row r
            return ((c ^ (r & fx.Int32(7))) if const_expr(SWZ) else c) * fx.Int32(VEC)

        if const_expr(SWZ):
            # Element offsets; the XOR'd column lives in bits 3..5. K: c16 = 4 pr + lane_k / 4,
            # so pr's parity flips bit 5. V: row lane_k + e flips c16 by e (lane_k % 4 == 0).
            k_base = _row(lane_row) + _c16(lane_k // fx.Int32(4), lane_row)
            v_base = (
                _row(lane_k)
                + _c16(lane_row // fx.Int32(2), lane_k)
                + (lane_row % fx.Int32(2)) * fx.Int32(4)
            )

        def _gather(idx_vec):
            frags = []
            for it in range_constexpr(N_GATHER):
                slot = fx.Int32(fx.rocdl.readlane(T.i32, idx_vec, row0 + fx.Int32(it)))
                frags.append(
                    fx.ptr_load(
                        kv_ptr + slot * fx.Int32(HEAD_DIM) + lane * fx.Int32(VEC),
                        result_type=v8bf16,
                    )
                )
            return frags

        def _store_kv(frags):
            for it in range_constexpr(N_GATHER):
                fx.ptr_store(
                    frags[it],
                    lds_kv + _row(row0 + fx.Int32(it)) + _c16(lane, row0 + fx.Int32(it)),
                )

        def _k_a(pr):
            # A operands for MFMAs 2pr, 2pr+1: K[token = lane_row][d of q_frags].
            if const_expr(SWZ):
                addr = lds_kv + ((k_base ^ fx.Int32((pr % 2) * 32)) + fx.Int32((pr // 2) * 64)) + qk_d0
            else:
                addr = lds_kv + _row(lane_row) + qk_d0 + fx.Int32(pr * 32) + lane_k * fx.Int32(2)
            k8 = fx.ptr_load(addr, result_type=v8bf16)
            return _halves(k8)

        def _vt_rows(g):
            return [
                fx.ptr_load(
                    (lds_kv + (((v_base + fx.Int32(e * LD)) ^ fx.Int32(e * 8)) + fx.Int32(g * 64)) + pv_d0) if const_expr(SWZ) else
                    (lds_kv + _row(lane_k + fx.Int32(e)) + pv_d0 + fx.Int32(g * 64) + lane_row * fx.Int32(4)),
                    result_type=v4bf16,
                )
                for e in range_constexpr(4)
            ]

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
        idx_q = [_load_idx(fx.Int32((a + 1) * BLOCK_K)) for a in range_constexpr(IDX_AHEAD)]
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
            k0 = kt_i32 * fx.Int32(BLOCK_K)

            fx.rocdl.s_waitcnt(vmcnt=0)
            _store_kv(prefetched)
            gpu.barrier()
            # Issue the next tile's gather before any LDS read so the bytes
            # stay outstanding across QK, softmax, and PV.
            next_prefetched = prefetched
            if const_expr(IDX_AHEAD == 0):
                idx_q = [_load_idx(k0 + fx.Int32(BLOCK_K))]
            idx_cur = idx_q[0]
            idx_q = idx_q[1:]
            if const_expr(IDX_AHEAD > 0):
                idx_q.append(_load_idx(k0 + fx.Int32((IDX_AHEAD + 1) * BLOCK_K)))
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

                def _part_addr(w):
                    return lds_kv + (lane // fx.Int32(32)) * fx.Int32(HEAD_DIM) + w * fx.Int32(HEAD_DIM // 2) + (lane % fx.Int32(32)) * fx.Int32(8)

                fx.ptr_store(fx.Vector(s_acc).bitcast(fx.BFloat16), _part_addr(warp))
                gpu.barrier()
                peer = fx.Vector(fx.ptr_load(_part_addr(warp ^ fx.Int32(1)), result_type=v8bf16)).bitcast(fx.Float32)
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
                xch = lds_kv + fx.Int32(TILE_ELEMS)
                pf = p_b.bitcast(fx.Float32)
                fx.ptr_store(
                    fx.Vector.from_elements([pf[0], pf[1], alpha, zero_f], fx.Float32).bitcast(fx.BFloat16),
                    xch + tid * fx.Int32(VEC),
                )
                fx.rocdl.sched_barrier(0)
                gpu.barrier()
                fx.rocdl.sched_barrier(0)
                xs = [
                    fx.Vector(fx.ptr_load(xch + fx.Int32(j * WAVE * VEC) + lane * fx.Int32(VEC), result_type=v8bf16)).bitcast(fx.Float32)
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
            fx.ptr_store(fx.Vector.from_elements([m_i, l_i], fx.Float32).bitcast(fx.BFloat16), lds_kv + tid * fx.Int32(4))
            gpu.barrier()
            # Re-derive lane ids after the loop: hoisted copies spill at the 256 cap on gfx942.
            m1 = fx.Int32(-1).ir_value()
            lane = fx.Int32(fx.rocdl.mbcnt_hi(T.i32, m1, fx.rocdl.mbcnt_lo(T.i32, m1, fx.Int32(0).ir_value())))
            lane_row = lane % fx.Int32(16)
            lane_k = (lane // fx.Int32(16)) * fx.Int32(4)

        tiny = fx.Float32(1.0e-30)
        head = lane_row

        def _inv(m_q, l_q):
            denom = l_q
            alpha_q = fx.Float32(1.0)
            if const_expr(HAS_SINK):
                sink = fx.ptr_load(sink_ptr + head, result_type=fx.Float32)
                m_scaled = m_q * fx.Float32(scale)
                m_final = (sink > m_scaled).select(sink, m_scaled)
                alpha_q = _exp2((m_scaled - m_final) * fx.Float32(LOG2E))
                sink_term = _exp2((sink - m_final) * fx.Float32(LOG2E))
                denom = denom * alpha_q + sink_term
            denom = (denom > tiny).select(denom, tiny)
            return alpha_q / denom

        def _store_out(t_q, acc_q, inv):
            out_row = out_ptr + t_q * fx.Int32(NUM_HEADS * HEAD_DIM) + head * fx.Int32(HEAD_DIM) + pv_d0
            # acc[4g + h][i] = out[head = lane_row][d = d0 + 64 g + 4 (lane_k + i) + h]
            for g in range_constexpr(PV_D_TILES // 4):
                for i in range_constexpr(4):
                    vals = [_to_bf16(_elem(acc_q[4 * g + h], i) * inv) for h in range_constexpr(4)]
                    fx.ptr_store(
                        fx.Vector.from_elements(vals, fx.BFloat16),
                        out_row + fx.Int32(g * 64) + (lane_k + fx.Int32(i)) * fx.Int32(4),
                    )

        if const_expr(ONE_Q):
            inv = _inv(m_i, l_i)
            if live:
                _store_out(t_idx, acc, inv)
        else:
            for j in range_constexpr(BLOCK_Q):
                t_j = tile * fx.Int32(BLOCK_Q) + fx.Int32(j)
                ml = fx.Vector(fx.ptr_load(lds_kv + fx.Int32(j * WAVE * 4) + lane * fx.Int32(4), result_type=v4bf16)).bitcast(fx.Float32)
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
