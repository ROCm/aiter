# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""gfx942 FlyDSL absorbed sparse MLA.

One CTA, two modes. BLOCK_Q==1 uses two waves that split D. BLOCK_Q==2/4 uses
one wave per query over a shared KV tile. Both modes:

* gather the next 16 KV rows with coalesced dwordx4 loads while the current
  tile's MFMA chain runs
* reduce the 16-token softmax with DPP (no LDS broadcast)
* keep P in registers and transpose it with ds_bpermute, so each MFMA reads
  only K or V from LDS

``v_mfma_f32_16x16x16_bf16`` fragment map (device probe):

    C[i] = (A @ B^T)[lane_k + i, lane_row]
    lane_row = lane % 16, lane_k = (lane // 16) * 4
"""

from functools import lru_cache

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import fly as fly_dialect
from flydsl.expr import const_expr, gpu, range_constexpr
from flydsl.expr.typing import T

from aiter.ops.flydsl.kernels.dpp_utils import dpp_xor_f32
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
# Eight bf16 elements of row padding spread column-wise MFMA reads across LDS
# banks. This matches the 520-element pitch in the gfx942 gluon reference.
LD_KV = HEAD_DIM + 8


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


def _reduce16(val, add):
    """All-reduce 16 lanes that share a head group. DPP does not touch LDS."""
    val = fx.Float32(val)
    for sh in (8, 4, 2, 1):
        peer = fx.Float32(dpp_xor_f32(val, sh))
        if const_expr(add):
            val = val + peer
        else:
            val = (val > peer).select(val, peer)
    return val


def _bperm_f32(val, src_lane):
    bits = fx.Float32(val).bitcast(fx.Int32)
    recv = fx.rocdl.ds_bpermute(T.i32, src_lane * fx.Int32(4), bits)
    return fx.Int32(recv).bitcast(fx.Float32)


def _scale_acc(acc4, alphas):
    return fx.Vector.from_elements(
        [_elem(acc4, i) * alphas[i] for i in range_constexpr(4)],
        fx.Float32,
    )


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
    PV_D_TILES = QK_D_TILES
    N_GATHER = (BLOCK_K * (HEAD_DIM // VEC) + BLOCK_THREADS - 1) // BLOCK_THREADS
    # Eight PV MFMAs are 32 ds_read_u16s. On the four-query kernel that
    # overflows the 15-deep lgkmcnt counter and spills Q out of the 256-VGPR
    # occupancy-2 budget. Two MFMAs fit. The one-query kernel does not spill
    # at eight, and the wider batch is faster there.
    MFMA_CHUNK = 8 if ONE_Q else 2
    module_name = (
        f"sparse_mla_mfma_bq{BLOCK_Q}_sink{int(HAS_SINK)}_d{HEAD_DIM}_h{NUM_HEADS}"
    )

    @fx.struct
    class SharedStorage:
        kv: fx.Array[fx.BFloat16, BLOCK_K * LD_KV, 16]
        sred: fx.Array[fx.Float32, (NWARP * WAVE * 4) if ONE_Q else 1, 4]

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

        storage = fx.SharedAllocator().allocate(SharedStorage)
        lds_kv = storage.kv.peek().ptr
        if const_expr(ONE_Q):
            sred_lds = storage.sred.peek().view(
                fx.make_layout(NWARP * WAVE * 4, 1)
            )

        qk_scale = fx.Float32(scale) * fx.Float32(LOG2E)
        neg_inf = fx.Float32(NEG_INF)
        zero_f = fx.Float32(0.0)
        n_u = fx.Int32(n_unique)
        n_kt = (n_u + fx.Int32(BLOCK_K - 1)) // fx.Int32(BLOCK_K)
        idx_row = q_row if const_expr(ONE_Q) else tile
        qk_d0 = warp * fx.Int32(QK_D_TILES * 16) if const_expr(ONE_Q) else fx.Int32(0)
        pv_d0 = qk_d0
        mma = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 16, fx.BFloat16))
        v4bf16 = fx.Vector.make_type(4, fx.BFloat16)
        v8bf16 = fx.Vector.make_type(VEC, fx.BFloat16)

        q_frags = []
        for dt in range_constexpr(QK_D_TILES):
            q_off = (
                q_row * fx.Int32(NUM_HEADS * HEAD_DIM)
                + lane_row * fx.Int32(HEAD_DIM)
                + qk_d0
                + fx.Int32(dt * 16)
                + lane_k
            )
            q_frags.append(fx.ptr_load(q_ptr + q_off, result_type=v4bf16))

        def _load_idx(k_base):
            # Address first, then load. A select around the load still issues
            # it for lanes 16..63 and walks off the end of the last query.
            off = (lane < fx.Int32(16)).select(lane, fx.Int32(0))
            return fx.ptr_load(
                idx_ptr + idx_row * n_u + k_base + off,
                result_type=fx.Int32,
            )

        def _gather(idx_vec):
            frags = []
            for it in range_constexpr(N_GATHER):
                row = warp + fx.Int32(it * NWARP)
                slot = fx.Int32(fx.rocdl.readlane(T.i32, idx_vec, row))
                frags.append(
                    fx.ptr_load(
                        kv_ptr + slot * fx.Int32(HEAD_DIM) + lane * fx.Int32(VEC),
                        result_type=v8bf16,
                    )
                )
            return frags

        def _store_kv(frags):
            for it in range_constexpr(N_GATHER):
                row = warp + fx.Int32(it * NWARP)
                fx.ptr_store(
                    frags[it],
                    lds_kv + row * fx.Int32(LD_KV) + lane * fx.Int32(VEC),
                )

        def _qk_b(dt):
            return fx.ptr_load(
                lds_kv + lane_row * fx.Int32(LD_KV) + qk_d0 + fx.Int32(dt * 16) + lane_k,
                result_type=v4bf16,
            )

        def _pv_b(dt):
            d_row = pv_d0 + fx.Int32(dt * 16) + lane_row
            cols = []
            for e in range_constexpr(4):
                cols.append(
                    fx.ptr_load(
                        lds_kv + (lane_k + fx.Int32(e)) * fx.Int32(LD_KV) + d_row,
                        result_type=fx.BFloat16,
                    )
                )
            return fx.Vector.from_elements(cols, fx.BFloat16)

        m_i = [neg_inf for _ in range_constexpr(4)]
        l_i = [zero_f for _ in range_constexpr(4)]
        acc = [_zero4() for _ in range_constexpr(PV_D_TILES)]
        prefetched = _gather(_load_idx(fx.Int32(0)))
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

            fx.rocdl.s_waitcnt(vmcnt=0)
            _store_kv(prefetched)
            gpu.barrier()
            # Issue the next tile's gather before any LDS read so the bytes
            # stay outstanding across QK, softmax, and PV.
            next_prefetched = prefetched
            if kt_i32 + fx.Int32(1) < n_kt:
                next_prefetched = _gather(_load_idx(k0 + fx.Int32(BLOCK_K)))
            fx.rocdl.sched_barrier(0)

            s_acc = _zero4()
            for chunk in range_constexpr(QK_D_TILES // MFMA_CHUNK):
                base = chunk * MFMA_CHUNK
                buf = [_qk_b(base + i) for i in range_constexpr(MFMA_CHUNK)]
                fx.rocdl.sched_barrier(0)
                for i in range_constexpr(MFMA_CHUNK):
                    s_acc = _mfma_ssa(mma, q_frags[base + i], buf[i], s_acc)

            if const_expr(ONE_Q):
                for i in range_constexpr(4):
                    sred_lds[
                        (warp * fx.Int32(WAVE) + lane) * fx.Int32(4) + fx.Int32(i)
                    ] = _elem(s_acc, i)
                gpu.barrier()
                summed = []
                for i in range_constexpr(4):
                    score = zero_f
                    for w in range_constexpr(NWARP):
                        score = score + sred_lds[
                            (fx.Int32(w) * fx.Int32(WAVE) + lane) * fx.Int32(4)
                            + fx.Int32(i)
                        ]
                    summed.append(score)
                s_acc = fx.Vector.from_elements(summed, fx.Float32)

            scores = []
            for i in range_constexpr(4):
                scores.append(live.select(_elem(s_acc, i), neg_inf))

            # Every lane in the 16-wide head group already holds the reduced
            # max and sum, so the old LDS broadcast is unnecessary.
            m_new = []
            alphas = []
            for i in range_constexpr(4):
                m_w = _reduce16(scores[i], False)
                m_old = m_i[i]
                m_n = (m_w > m_old).select(m_w, m_old)
                empty = m_n == neg_inf
                al = empty.select(zero_f, _exp2((m_old - m_n) * qk_scale))
                alphas.append(al)
                m_new.append(m_n)

            for dt in range_constexpr(PV_D_TILES):
                acc[dt] = _scale_acc(acc[dt], alphas)

            p_heads = []
            for i in range_constexpr(4):
                empty = m_new[i] == neg_inf
                p = empty.select(zero_f, _exp2((scores[i] - m_new[i]) * qk_scale))
                p = live.select(p, zero_f)
                p_heads.append(p)
                l_add = _reduce16(p, True)
                l_i[i] = l_i[i] * alphas[i] + l_add
                m_i[i] = m_new[i]

            # QK left P as (head = lane_k+i, token = lane_row). PV wants
            # (head = lane_row, token = lane_k .. lane_k+3) packed as A.
            elem = lane_row % fx.Int32(4)
            src_base = (lane_row // fx.Int32(4)) * fx.Int32(16) + lane_k
            a_elems = []
            for e in range_constexpr(4):
                src_lane = src_base + fx.Int32(e)
                recvs = [_bperm_f32(p_heads[s], src_lane) for s in range_constexpr(4)]
                got = fx.Float32(recvs[0])
                for s in range_constexpr(1, 4):
                    got = (elem == fx.Int32(s)).select(fx.Float32(recvs[s]), got)
                a_elems.append(_to_bf16(got))
            p_a = fx.Vector.from_elements(a_elems, fx.BFloat16)

            for chunk in range_constexpr(PV_D_TILES // MFMA_CHUNK):
                base = chunk * MFMA_CHUNK
                vb = [_pv_b(base + i) for i in range_constexpr(MFMA_CHUNK)]
                fx.rocdl.sched_barrier(0)
                for i in range_constexpr(MFMA_CHUNK):
                    acc[base + i] = _mfma_ssa(mma, p_a, vb[i], acc[base + i])
            gpu.barrier()
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
        "waves_per_eu": 2,
        "fast_fp_math": True,
        "unsafe_fp_math": True,
        "llvm_options": {
            "amdgpu-kernarg-preload": AITER_FLYDSL_KERNARG_PRELOAD,
            "amdgpu-kernarg-preload-count": AITER_FLYDSL_KERNARG_PRELOAD_COUNT,
        },
    }
    return launch_sparse_mla_qblock
