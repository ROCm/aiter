# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""gfx942 FlyDSL absorbed sparse MLA.

BQ=1: one CTA per query, 2 waves, BLOCK_K=16, warps split D.
``v_mfma_f32_16x16x16_bf16`` fragment map (device probe):

    C[i] = (A @ B^T)[lane_k + i, lane_row]
    lane_row = lane % 16, lane_k = (lane // 16) * 4

BQ=2/4: one wave per query, unique KV staged once.
"""

from threading import Lock

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, range_constexpr

from aiter.ops.flydsl.kernels.kernels_common import LOG2E
from aiter.ops.flydsl.kernels.tensor_shim import (
    AITER_FLYDSL_KERNARG_PRELOAD,
    AITER_FLYDSL_KERNARG_PRELOAD_COUNT,
    buf_copy_atom,
    buf_scalar_load,
    ptr_buf_tensor,
)

WAVE = 64
HEAD_DIM = 512
NUM_HEADS = 16
VEC = 8
NVEC = HEAD_DIM // VEC
NEG_INF = float("-inf")
N_D_TILES = HEAD_DIM // 16
# Eight bf16 elements of row padding spread column-wise MFMA reads across LDS
# banks. This matches the 520-element pitch in the gfx942 gluon reference.
LD_KV = HEAD_DIM + 8

_COMPILE_LOCK = Lock()
_CACHE = {}


def _mfma(a, b, acc):
    mma = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 16, fx.BFloat16))
    fa = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.BFloat16)
    fb = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.BFloat16)
    fc = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Float32)
    fx.memref_store_vec(a, fa)
    fx.memref_store_vec(b, fb)
    fx.memref_store_vec(acc, fc)
    fx.mma_atom_call(mma, fc, fa, fb, fc)
    return fx.memref_load_vec(fc)


def _elem(acc4, i):
    return fx.Vector(acc4)[i]


def _zero4():
    return fx.Vector.filled(4, 0.0, fx.Float32)


def _exp2(x):
    return fx.Float32(fx.rocdl.exp2(fx.Float32.ir_type, x.ir_value()))


def _to_bf16(v):
    bits = v.bitcast(fx.Uint32) + fx.Uint32(0x00008000)
    return (bits >> fx.Uint32(16)).to(fx.Uint16).bitcast(fx.BFloat16)


def _reduce16(val, add):
    val = fx.Float32(val)
    for sh in [8, 4, 2, 1]:
        peer = val.shuffle_xor(fx.Int32(sh), fx.Int32(64))
        if const_expr(add):
            val = val + peer
        else:
            val = (val > peer).select(val, peer)
    return val


def _scale_acc(acc4, alphas):
    return fx.Vector.from_elements(
        [_elem(acc4, i) * alphas[i] for i in range_constexpr(4)],
        fx.Float32,
    )


def compile_sparse_mla_qblock(block_q, has_sink=False):
    if block_q not in (1, 2, 4):
        raise ValueError("BLOCK_Q must be 1, 2, or 4")
    key = (int(block_q), int(has_sink))
    cached = _CACHE.get(key)
    if cached is not None:
        return cached

    BLOCK_Q = int(block_q)
    HAS_SINK = bool(has_sink)
    ONE_Q = BLOCK_Q == 1
    # The bf16 one-query path uses two waves to split the 512-wide contraction
    # and output. Q-block keeps one wave per query over a shared KV tile.
    NWARP = 2 if ONE_Q else BLOCK_Q
    BLOCK_THREADS = NWARP * WAVE
    BLOCK_K = 16
    K_SUBS = BLOCK_K // 16
    LD_P = BLOCK_K
    QK_D_TILES = N_D_TILES // NWARP if ONE_Q else N_D_TILES
    PV_D_TILES = N_D_TILES // NWARP if ONE_Q else N_D_TILES
    P_ELEMS = NUM_HEADS * LD_P if ONE_Q else NWARP * NUM_HEADS * LD_P
    KV_UNITS = BLOCK_K * NVEC
    N_GATHER = (KV_UNITS + BLOCK_THREADS - 1) // BLOCK_THREADS
    module_name = (
        f"sparse_mla_mfma_bq{BLOCK_Q}_sink{int(HAS_SINK)}_d{HEAD_DIM}_h{NUM_HEADS}"
    )

    @fx.struct
    class SharedStorage:
        kv: fx.Array[fx.BFloat16, BLOCK_K * LD_KV, 16]
        p: fx.Array[fx.BFloat16, P_ELEMS, 16]
        slots: fx.Array[fx.Int32, BLOCK_K if not ONE_Q else 1, 4]
        red: fx.Array[fx.Float32, NWARP * NUM_HEADS, 4]
        sred: fx.Array[
            fx.Float32,
            (NWARP * WAVE * 4 * K_SUBS) if ONE_Q else 1,
            4,
        ]

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
        warp = tid // fx.Int32(WAVE)
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

        kv_t = ptr_buf_tensor(kv_ptr, fx.BFloat16, unit_elems=VEC)
        idx_t = ptr_buf_tensor(idx_ptr, fx.Int32)
        atom_g = buf_copy_atom(16, fx.BFloat16)

        storage = fx.SharedAllocator().allocate(SharedStorage)
        lds_kv = storage.kv.peek().ptr
        lds_p = storage.p.peek().ptr
        red_lds = storage.red.peek().view(fx.make_layout(NWARP * NUM_HEADS, 1))
        if const_expr(ONE_Q):
            sred_lds = storage.sred.peek().view(
                fx.make_layout(NWARP * WAVE * 4 * K_SUBS, 1)
            )
        else:
            slots_lds = storage.slots.peek().view(fx.make_layout(BLOCK_K, 1))
        kv_lds_u = fx.logical_divide(
            storage.kv.peek().view(fx.make_layout((BLOCK_K * LD_KV,), (1,))),
            fx.make_layout(VEC, 1),
        )
        atom_s = fx.make_copy_atom(fx.UniversalCopy128b(), fx.BFloat16)
        zero_kv = fx.Vector.filled(VEC, 0.0, fx.BFloat16)

        qk_scale = fx.Float32(scale) * fx.Float32(LOG2E)
        neg_inf = fx.Float32(NEG_INF)
        zero_f = fx.Float32(0.0)
        n_u = fx.Int32(n_unique)
        n_kt = (n_u + fx.Int32(BLOCK_K - 1)) // fx.Int32(BLOCK_K)
        idx_row = q_row if const_expr(ONE_Q) else tile
        p_off = fx.Int32(0) if const_expr(ONE_Q) else warp * fx.Int32(NUM_HEADS * LD_P)
        qk_d0 = warp * fx.Int32(QK_D_TILES * 16) if const_expr(ONE_Q) else fx.Int32(0)
        pv_d0 = warp * fx.Int32(PV_D_TILES * 16) if const_expr(ONE_Q) else fx.Int32(0)

        q_frags = []
        for dt in range_constexpr(QK_D_TILES):
            q_off = (
                q_row * fx.Int32(NUM_HEADS * HEAD_DIM)
                + lane_row * fx.Int32(HEAD_DIM)
                + qk_d0
                + fx.Int32(dt * 16)
                + lane_k
            )
            q_frags.append(
                fx.ptr_load(
                    q_ptr + q_off, result_type=fx.Vector.make_type(4, fx.BFloat16)
                )
            )

        m_i = [neg_inf for _ in range_constexpr(4)]
        l_i = [zero_f for _ in range_constexpr(4)]
        acc = [_zero4() for _ in range_constexpr(PV_D_TILES)]

        def _prefetch_tile(k_base):
            frags = []
            for it in range_constexpr(N_GATHER):
                row = warp + fx.Int32(it * NWARP)
                k_pos = k_base + row
                slot = fx.Int32(buf_scalar_load(idx_t, idx_row * n_u + k_pos))
                frags.append(
                    fx.ptr_load(
                        kv_ptr + slot * fx.Int32(HEAD_DIM) + lane * fx.Int32(VEC),
                        result_type=fx.Vector.make_type(VEC, fx.BFloat16),
                    )
                )
            return frags

        if const_expr(ONE_Q):
            prefetched = _prefetch_tile(fx.Int32(0))
        else:
            prefetched = [zero_kv for _ in range_constexpr(N_GATHER)]
        loop_init = m_i + l_i + acc + prefetched
        for kt, state in range(fx.Int32(0), n_kt, fx.Int32(1), init=loop_init):
            state_off = 0
            m_i = [fx.Float32(state[state_off + i]) for i in range_constexpr(4)]
            state_off += 4
            l_i = [fx.Float32(state[state_off + i]) for i in range_constexpr(4)]
            state_off += 4
            acc = [fx.Vector(state[state_off + i]) for i in range_constexpr(PV_D_TILES)]
            state_off += PV_D_TILES
            prefetched = [
                fx.Vector(state[state_off + i]) for i in range_constexpr(N_GATHER)
            ]
            kt_i32 = fx.Int32(kt)
            k0 = kt_i32 * fx.Int32(BLOCK_K)

            if const_expr(ONE_Q):
                for it in range_constexpr(N_GATHER):
                    row = warp + fx.Int32(it * NWARP)
                    fx.ptr_store(
                        prefetched[it],
                        lds_kv + row * fx.Int32(LD_KV) + lane * fx.Int32(VEC),
                    )
                gpu.barrier()
                next_prefetched = prefetched
                if kt_i32 + fx.Int32(1) < n_kt:
                    next_prefetched = _prefetch_tile(k0 + fx.Int32(BLOCK_K))
            else:
                if tid < fx.Int32(BLOCK_K):
                    slots_lds[tid] = fx.Int32(idx_t[idx_row * n_u + k0 + tid])
                gpu.barrier()
                # All queries consume the union tile, so stage it once
                # cooperatively before any wave starts QK.
                gather_frags = [
                    fx.make_fragment_like(fx.slice(kv_t, (0, None)))
                    for _ in range_constexpr(N_GATHER)
                ]
                for it in range_constexpr(N_GATHER):
                    uidx = tid + fx.Int32(it * BLOCK_THREADS)
                    if uidx < fx.Int32(KV_UNITS):
                        row = uidx // fx.Int32(NVEC)
                        dvec = uidx - row * fx.Int32(NVEC)
                        fx.copy(
                            atom_g,
                            fx.slice(
                                kv_t,
                                (slots_lds[row] * fx.Int32(NVEC) + dvec, None),
                            ),
                            gather_frags[it],
                        )
                for it in range_constexpr(N_GATHER):
                    uidx = tid + fx.Int32(it * BLOCK_THREADS)
                    if uidx < fx.Int32(KV_UNITS):
                        row = uidx // fx.Int32(NVEC)
                        dvec = uidx - row * fx.Int32(NVEC)
                        lds_uidx = row * fx.Int32(LD_KV // VEC) + dvec
                        fx.copy(
                            atom_s,
                            gather_frags[it],
                            fx.slice(kv_lds_u, (None, lds_uidx)),
                        )
                gpu.barrier()

            s_acc = [_zero4() for _ in range_constexpr(K_SUBS)]
            for dt in range_constexpr(QK_D_TILES):
                for ks in range_constexpr(K_SUBS):
                    kv_row = fx.Int32(ks * 16) + lane_row
                    d_col = qk_d0 + fx.Int32(dt * 16) + lane_k
                    b = fx.ptr_load(
                        lds_kv + kv_row * fx.Int32(LD_KV) + d_col,
                        result_type=fx.Vector.make_type(4, fx.BFloat16),
                    )
                    s_acc[ks] = _mfma(q_frags[dt], b, s_acc[ks])

            if const_expr(ONE_Q):
                for ks in range_constexpr(K_SUBS):
                    for i in range_constexpr(4):
                        sred_lds[
                            (warp * fx.Int32(WAVE) + lane) * fx.Int32(4 * K_SUBS)
                            + fx.Int32(ks * 4 + i)
                        ] = _elem(s_acc[ks], i)
                gpu.barrier()
                summed_subs = []
                for ks in range_constexpr(K_SUBS):
                    summed = []
                    for i in range_constexpr(4):
                        score = zero_f
                        for w in range_constexpr(NWARP):
                            score = (
                                score
                                + sred_lds[
                                    (fx.Int32(w) * fx.Int32(WAVE) + lane)
                                    * fx.Int32(4 * K_SUBS)
                                    + fx.Int32(ks * 4 + i)
                                ]
                            )
                        summed.append(score)
                    summed_subs.append(fx.Vector.from_elements(summed, fx.Float32))
                s_acc = summed_subs

            scores = []
            for ks in range_constexpr(K_SUBS):
                score_ks = []
                for i in range_constexpr(4):
                    score_ks.append(live.select(_elem(s_acc[ks], i), neg_inf))
                scores.append(score_ks)

            for i in range_constexpr(4):
                m_w = _reduce16(scores[0][i], False)
                for ks in range_constexpr(1, K_SUBS):
                    peer_m = _reduce16(scores[ks][i], False)
                    m_w = (peer_m > m_w).select(peer_m, m_w)
                if lane_row == fx.Int32(0):
                    red_lds[warp * fx.Int32(NUM_HEADS) + lane_k + fx.Int32(i)] = m_w
            gpu.barrier()

            alphas = []
            m_new = []
            for i in range_constexpr(4):
                head = lane_k + fx.Int32(i)
                m_b = red_lds[warp * fx.Int32(NUM_HEADS) + head]
                m_old = m_i[i]
                m_n = (m_b > m_old).select(m_b, m_old)
                empty = m_n == neg_inf
                al = empty.select(zero_f, _exp2((m_old - m_n) * qk_scale))
                alphas.append(al)
                m_new.append(m_n)

            for dt in range_constexpr(PV_D_TILES):
                acc[dt] = _scale_acc(acc[dt], alphas)

            p_vals = []
            for ks in range_constexpr(K_SUBS):
                p_ks = []
                for i in range_constexpr(4):
                    empty = m_new[i] == neg_inf
                    p = empty.select(
                        zero_f, _exp2((scores[ks][i] - m_new[i]) * qk_scale)
                    )
                    p = live.select(p, zero_f)
                    p_ks.append(p)
                p_vals.append(p_ks)
            for i in range_constexpr(4):
                p_w = _reduce16(p_vals[0][i], True)
                for ks in range_constexpr(1, K_SUBS):
                    p_w = p_w + _reduce16(p_vals[ks][i], True)
                if lane_row == fx.Int32(0):
                    red_lds[warp * fx.Int32(NUM_HEADS) + lane_k + fx.Int32(i)] = p_w
            gpu.barrier()

            for i in range_constexpr(4):
                head = lane_k + fx.Int32(i)
                l_add = red_lds[warp * fx.Int32(NUM_HEADS) + head]
                l_i[i] = l_i[i] * alphas[i] + l_add
                m_i[i] = m_new[i]
                for ks in range_constexpr(K_SUBS):
                    kv_i = fx.Int32(ks * 16) + lane_row
                    fx.ptr_store(
                        _to_bf16(p_vals[ks][i]),
                        lds_p + p_off + (lane_k + fx.Int32(i)) * fx.Int32(LD_P) + kv_i,
                    )
            gpu.barrier()

            for dt in range_constexpr(PV_D_TILES):
                d_row = pv_d0 + fx.Int32(dt * 16) + lane_row
                for ks in range_constexpr(BLOCK_K // 16):
                    k_base = fx.Int32(ks * 16)
                    a = fx.ptr_load(
                        lds_p + p_off + lane_row * fx.Int32(LD_P) + k_base + lane_k,
                        result_type=fx.Vector.make_type(4, fx.BFloat16),
                    )
                    bs = []
                    for e in range_constexpr(4):
                        bs.append(
                            fx.ptr_load(
                                lds_kv
                                + (k_base + lane_k + fx.Int32(e)) * fx.Int32(LD_KV)
                                + d_row,
                                result_type=fx.BFloat16,
                            )
                        )
                    acc[dt] = _mfma(
                        a, fx.Vector.from_elements(bs, fx.BFloat16), acc[dt]
                    )
            gpu.barrier()
            if const_expr(ONE_Q):
                prefetched = next_prefetched
            results = yield m_i + l_i + acc + prefetched

        result_off = 0
        m_i = [fx.Float32(results[result_off + i]) for i in range_constexpr(4)]
        result_off += 4
        l_i = [fx.Float32(results[result_off + i]) for i in range_constexpr(4)]
        result_off += 4
        acc = [fx.Vector(results[result_off + i]) for i in range_constexpr(PV_D_TILES)]

        tiny = fx.Float32(1.0e-30)

        def _output_scale(i):
            denom = fx.Float32(l_i[i])
            alpha = fx.Float32(1.0)
            if const_expr(HAS_SINK):
                head = lane_k + fx.Int32(i)
                sink = fx.ptr_load(sink_ptr + head, result_type=fx.Float32)
                m_scaled = fx.Float32(m_i[i]) * fx.Float32(scale)
                m_final = (sink > m_scaled).select(sink, m_scaled)
                alpha = _exp2((m_scaled - m_final) * fx.Float32(LOG2E))
                sink_term = _exp2((sink - m_final) * fx.Float32(LOG2E))
                denom = denom * alpha + sink_term
            denom = (denom > tiny).select(denom, tiny)
            return alpha / denom

        if live:
            for i in range_constexpr(4):
                inv = _output_scale(i)
                head = lane_k + fx.Int32(i)
                for dt in range_constexpr(PV_D_TILES):
                    v = _elem(acc[dt], i) * inv
                    d = pv_d0 + fx.Int32(dt * 16) + lane_row
                    fx.ptr_store(
                        _to_bf16(v),
                        out_ptr
                        + t_idx * fx.Int32(NUM_HEADS * HEAD_DIM)
                        + head * fx.Int32(HEAD_DIM)
                        + d,
                    )

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
        "waves_per_eu": 1,
        "fast_fp_math": True,
        "unsafe_fp_math": True,
        "llvm_options": {
            "amdgpu-kernarg-preload": AITER_FLYDSL_KERNARG_PRELOAD,
            "amdgpu-kernarg-preload-count": AITER_FLYDSL_KERNARG_PRELOAD_COUNT,
        },
    }

    with _COMPILE_LOCK:
        cached = _CACHE.get(key)
        if cached is None:
            _CACHE[key] = launch_sparse_mla_qblock
            cached = launch_sparse_mla_qblock
    return cached
