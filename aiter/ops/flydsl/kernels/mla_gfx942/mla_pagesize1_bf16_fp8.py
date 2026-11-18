# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""gfx942/gfx950 persistent page-size-one MLA: BF16 Q, native E4M3 KV.

KV is E4M3FNUZ on gfx942 and OCP E4M3FN on gfx950, the formats each arch's
cvt_pk_f32_fp8 decodes.

Rows are packed as qpos * H + head. Four/eight waves share one gathered KV
slab; QK row tiles are wave-owned, PV output columns are wave-owned.
Only scalar specializations are captured by the cached builder.
"""

from functools import cache

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr import math as fmath
from flydsl.expr.typing import T

QK_DIM = 576
V_DIM = 512
INFO_COLS = 8

# A 576-wide Q/K row as (elem, lg, half, dc). Lane group lg reads 16 contiguous
# columns per dc, so one 16-byte load feeds two MFMA K steps (ds_read_b128).
# The permuted reduction order cancels in the dot product only if Q follows the
# same layout; the Q loads spell it out as offsets.
QK_COLS = ((8, 4, 2, 9), (1, 16, 8, 64))


def _buffer(ptr, dtype, layout):
    # The record bound is the layout's cosize: out-of-range loads return zero,
    # which the masked Q rows rely on. The 32-bit bound limits a view to 2 GiB.
    view = fx.make_view(fx.recast_iter(dtype, ptr), layout)
    return fx.Tensor(fx.rocdl.make_buffer_tensor(view, max_size=False))


def _rne(v):
    # Finite values only (probabilities and normalized outputs). Exact RNE.
    bits = v.bitcast(fx.Uint32)
    bits = bits + fx.Uint32(0x7FFF) + ((bits >> fx.Uint32(16)) & fx.Uint32(1))
    return (bits >> fx.Uint32(16)).to(fx.Uint16).bitcast(fx.BFloat16)


def _mfma(a, b, acc):
    mma = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 16, fx.BFloat16))
    fa = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.BFloat16)
    fb = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.BFloat16)
    fc = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Float32)
    fa.store(a)
    fb.store(b)
    fc.store(acc)
    fx.mma_atom_call(mma, fc, fa, fb, fc)
    return fc.load()


def _pack_exact(v):
    # Native unpack is FNUZ on gfx942 and OCP on gfx950; every finite value of
    # either E4M3 format is exactly BF16.
    packed = fx.rocdl.perm_b32(
        v[1].bitcast(fx.Int32), v[0].bitcast(fx.Int32), fx.Int32(0x07060302)
    )
    return fx.Vector.from_elements([fx.Int32(packed)], dtype=fx.Int32).bitcast(
        fx.BFloat16
    )


def _unhoisted(value):
    from flydsl._mlir.dialects import llvm

    return fx.Int32(
        llvm.inline_asm(
            T.i32,
            [fx.Int32(value).ir_value()],
            "v_mov_b32 $0, $1",
            "=v,v",
            has_side_effects=True,
        )
    )


def _uniform(value):
    # Scheduling metadata and wave ids are identical in all lanes of a wave.
    return fx.Int32(fx.rocdl.readfirstlane(T.i32, fx.Int32(value)))


def _frag(v, start):
    return fx.Vector.from_elements(
        [v[start + i] for i in range_constexpr(4)], dtype=fx.BFloat16
    )


# Eight waves for every row count. At one workgroup per CU this halves each
# thread's share of the KV gather and each wave's PV columns; measured
# 1.20-1.24x faster than four waves for 64 rows or fewer (H4/8/12, qlen 1-8).
WAVES = 8


@cache
def _build(H, QLEN, CAUSAL, WRITE_LSE):
    assert 1 <= H <= 16 and 1 <= QLEN <= 8 and H * QLEN <= 128
    NT = (H * QLEN + 15) // 16
    W = WAVES
    THREADS = W * 64
    CD = 32 // W
    OWN = (NT + W - 1) // W
    N = 32
    LD = 584
    LP = N + 8
    GATHER_BYTES = 48
    GATHER_LANES = QK_DIM // GATHER_BYTES
    # KV tokens of a tile in MFMA C-row order (i, lg, kn): the P tile and V^T.
    TOKENS = ((4, 4, N // 16), (1, 4, 16))
    # Wave-owned PV output columns: V^T rows (wave, cd, lr) and the matching
    # accumulator columns (wave, cd, lg, i).
    V_COLS = ((W, CD, 16), (V_DIM // W, 16, 1))
    OUT_COLS = ((W, CD, 4, 4), (V_DIM // W, 16, 4, 1))
    STAT_ALPHA, STAT_MAX, STAT_SUM = range(3)

    @fx.struct
    class Shared:
        kv: fx.Array[fx.BFloat16, N * LD, 16]
        p: fx.Array[fx.BFloat16, NT * 16 * LP, 16]
        stats: fx.Array[fx.Float32, NT * 16 * 3, 16]
        pages: fx.Array[fx.Uint64, N, 16]

    kernel_name = (
        f"mla_pagesize1_bf16_fp8_h{H}_q{QLEN}"
        f"{'_causal' if CAUSAL else ''}{'_lse' if WRITE_LSE else ''}"
    )

    @flyc.kernel(name=kernel_name, known_block_size=[THREADS, 1, 1])
    def stage1(
        partial: fx.Pointer,
        partial_lse: fx.Pointer,
        output: fx.Pointer,
        output_lse: fx.Pointer,
        q: fx.Pointer,
        kv: fx.Pointer,
        pages: fx.Pointer,
        indptr: fx.Pointer,
        info: fx.Pointer,
        kv_scale: fx.Pointer,
        scale: fx.Float32,
        total_q: fx.Int32,
        num_pages: fx.Int32,
        num_indices: fx.Int32,
        num_partial: fx.Int32,
        num_work: fx.Int32,
        num_cus: fx.Constexpr[int],
    ):
        tid = fx.thread_idx.x
        lane = tid % 64
        lr = lane % 16
        lg = lane // 16
        wave = _uniform(tid // 64)
        cu = fx.block_idx.x
        q_rows = total_q * H
        p_rows = num_partial * H
        gq = _buffer(
            q, fx.BFloat16, fx.make_layout((q_rows * (QK_DIM // 8), 8), (8, 1))
        )
        gp = _buffer(pages, fx.Int32, fx.make_layout(num_indices, 1))
        gi = _buffer(indptr, fx.Int32, fx.make_layout(num_cus + 1, 1))
        gw = _buffer(
            info, fx.Int32, fx.make_layout((num_work, INFO_COLS), (INFO_COLS, 1))
        )
        gr = _buffer(
            partial,
            fx.Float32,
            fx.make_layout((p_rows, OUT_COLS[0]), (V_DIM, OUT_COLS[1])),
        )
        gl = _buffer(partial_lse, fx.Float32, fx.make_layout(p_rows, 1))
        go = _buffer(
            output,
            fx.BFloat16,
            fx.make_layout((q_rows, OUT_COLS[0]), (V_DIM, OUT_COLS[1])),
        )
        gol = _buffer(output_lse, fx.Float32, fx.make_layout(q_rows, 1))
        g_scale = _buffer(kv_scale, fx.Float32, fx.make_layout(1, 1))
        descale = _uniform(g_scale[0].bitcast(fx.Int32)).bitcast(fx.Float32)
        score_scale = scale * descale * fx.Float32(1.4426950408889634)
        shared = fx.SharedAllocator().allocate(Shared).peek()
        # One padded KV slab, viewed per access pattern: K operand rows, the
        # gather's (8 BF16, part, chunk, gather lane) stores, and V^T columns.
        kv_k = fx.Tensor(
            shared.kv.view(fx.make_layout((N, QK_COLS[0]), (LD, QK_COLS[1])))
        )
        kv_gather = fx.Tensor(
            shared.kv.view(
                fx.make_layout(
                    (N, (8, GATHER_LANES, 3, 2)), (LD, (1, GATHER_BYTES, 16, 8))
                )
            )
        )
        kv_v = fx.Tensor(
            shared.kv.view(
                fx.make_layout(
                    (TOKENS[0], V_COLS[0]),
                    (tuple(s * LD for s in TOKENS[1]), V_COLS[1]),
                )
            )
        )
        p_lds = fx.Tensor(
            shared.p.view(fx.make_layout((NT * 16, TOKENS[0]), (LP, TOKENS[1])))
        )
        stats = fx.Tensor(shared.stats.view(fx.make_layout((NT * 16, 3), (3, 1))))
        s_pages = fx.Tensor(shared.pages.view(fx.make_layout(N, 1)))
        sk = shared.kv.ptr
        begin = _uniform(gi[cu])
        end = _uniform(gi[cu + 1])
        for wi in range(begin, end, 1):
            work = fx.Int64(wi)
            pr = _uniform(gw[(work, 1)])
            qs = _uniform(gw[(work, 2)])
            qe = _uniform(gw[(work, 3)])
            ks = _uniform(gw[(work, 4)])
            ke = _uniform(gw[(work, 5)])
            ko = _uniform(gw[(work, 6)])
            qlen = qe - qs
            M = qlen * H
            qregs = []
            for rt in range_constexpr(OWN):
                r = (wave + rt * W) * 16 + lr
                for dc in range_constexpr(9):
                    for half in range_constexpr(2):
                        # QK_COLS at (lg, half, dc), in 8-element units. Rows past
                        # M point one past the end, so the load returns zero.
                        # A QK_COLS view masking the row instead changes the KV
                        # loop's register allocation.
                        q_off = (
                            (fx.Int64(qs) * H + r) * QK_DIM
                            + dc * 64
                            + lg * 16
                            + half * 8
                        ) // 8
                        q_off = (r < M).select(
                            q_off, fx.Int64(total_q) * H * (QK_DIM // 8)
                        )
                        qregs.append(gq[(q_off, None)].load())
            init = [fx.Float32(float("-inf")) for _ in range_constexpr(OWN)]
            init += [fx.Float32(0.0) for _ in range_constexpr(OWN)]
            init += [fx.full(4, 0.0, fx.Float32) for _ in range_constexpr(NT * CD)]
            if NT < W:
                # Tile zero is primed once; all lanes issue clamped loads, including
                # the two waves whose gathered values are not stored to LDS.
                raw = [fx.full(4, 0, fx.Uint32) for _ in range_constexpr(3)]
                if ke > ks:
                    tok0 = (tid % (N * GATHER_LANES)) // GATHER_LANES
                    dc0 = tid % GATHER_LANES
                    idx0 = (ks + tok0 < ke).select(ks + tok0, ke - 1)
                    page0 = gp[idx0]
                    base0 = kv + fx.Int64(page0) * QK_DIM
                    for chunk in range_constexpr(3):
                        raw[chunk] = fx.ptr_load(
                            base0 + dc0 * GATHER_BYTES + chunk * 16,
                            result_type=fx.Vector.make_type(16, fx.Uint8),
                        ).bitcast(fx.Uint32)
                init += raw
            result = init
            for kb_raw, state in range(ks, ke, N, init=init):
                kb = fx.Int32(kb_raw)
                # NT<8: publish next-tile page bases through the original
                # table/B1. NT=8 keeps the current-tile baseline path.
                if tid < N:
                    lookahead = N if NT < W else 0
                    idx = (kb + lookahead + tid < ke).select(
                        kb + lookahead + tid, ke - 1
                    )
                    page = gp[idx]
                    fx.memref_store(
                        fx.ptrtoint(kv + fx.Int64(page) * QK_DIM).to(fx.Uint64),
                        s_pages,
                        tid,
                    )
                fx.barrier()
                gather_tid = _unhoisted(tid) if NT >= 7 else tid
                if wave < 6:
                    tok = gather_tid // GATHER_LANES
                    dc = gather_tid % GATHER_LANES
                    page_base = fx.inttoptr(kv.type, s_pages[tok]) if NT == W else kv
                    # Swap adjacent 16B BF16 writes in alternating four-lane
                    # groups. All eight 16B bank slots occur once per lane octet.
                    swap = (dc // 4) % 2
                    for chunk in range_constexpr(3):
                        d = dc * GATHER_BYTES + chunk * 16
                        # Keep 64-bit addressing for production pools >4 GiB.
                        packed = (
                            state[2 * OWN + NT * CD + chunk]
                            if NT < W
                            else fx.ptr_load(
                                page_base + d,
                                result_type=fx.Vector.make_type(16, fx.Uint8),
                            ).bitcast(fx.Uint32)
                        )
                        for piece in range_constexpr(2):
                            part = fx.Int32(piece) ^ swap
                            vals = []
                            for word in range_constexpr(2):
                                bits = (swap == 0).select(
                                    packed[piece * 2 + word],
                                    packed[(1 - piece) * 2 + word],
                                )
                                for half in range_constexpr(2):
                                    v = fx.Vector(
                                        fx.rocdl.cvt_pk_f32_fp8(
                                            T.f32x2, bits.to(fx.Int32), bool(half)
                                        )
                                    )
                                    pair = _pack_exact(v)
                                    vals += [pair[0], pair[1]]
                            kv_gather[(tok, (None, dc, chunk, part))].store(
                                fx.Vector.from_elements(vals, dtype=fx.BFloat16)
                            )
                fx.barrier()
                next_max = []
                next_sum = []
                for rt in range_constexpr(OWN):
                    tile = wave + rt * W
                    r = tile * 16 + lr
                    nm = state[rt]
                    sm = state[OWN + rt]
                    if tile < NT:
                        dots = [
                            fx.full(4, 0.0, fx.Float32)
                            for _ in range_constexpr(N // 16)
                        ]
                        if const_expr(NT <= 4):
                            # One K pair ahead (NT<=4 only: neutral at NT 5-6); fences bound live operands to two sets.
                            kvals = [
                                kv_k[(kn * 16 + lr, (None, lg, 0, 0))].load()
                                for kn in range_constexpr(N // 16)
                            ]
                            fx.rocdl.sched_barrier(0)
                            for step in range_constexpr(18):
                                if const_expr(step + 1 < 18):
                                    ns = step + 1
                                    next_k = [
                                        kv_k[
                                            (kn * 16 + lr, (None, lg, ns % 2, ns // 2))
                                        ].load()
                                        for kn in range_constexpr(N // 16)
                                    ]
                                fx.rocdl.sched_barrier(0)
                                qv = qregs[rt * 18 + step]
                                for sub in range_constexpr(2):
                                    for kn in range_constexpr(N // 16):
                                        dots[kn] = _mfma(
                                            _frag(kvals[kn], sub * 4),
                                            _frag(qv, sub * 4),
                                            dots[kn],
                                        )
                                fx.rocdl.sched_barrier(0)
                                if const_expr(step + 1 < 18):
                                    kvals = next_k
                        else:
                            for dc in range_constexpr(9):
                                for half in range_constexpr(2):
                                    # Prevent full-K CSE/hoisting across row tiles:
                                    # retaining the full slab alongside Q/O spills.
                                    kbase = _unhoisted(
                                        fx.get_scalar(
                                            fx.crd2idx(
                                                (lr, (0, lg, half, dc)), kv_k.layout
                                            )
                                        )
                                    )
                                    k_step = fx.Tensor(
                                        fx.make_view(
                                            sk + kbase,
                                            fx.make_layout((N // 16, 8), (16 * LD, 1)),
                                        )
                                    )
                                    kvals = [
                                        k_step[(kn, None)].load()
                                        for kn in range_constexpr(N // 16)
                                    ]
                                    qv = qregs[rt * 18 + dc * 2 + half]
                                    for sub in range_constexpr(2):
                                        for kn in range_constexpr(N // 16):
                                            dots[kn] = _mfma(
                                                _frag(kvals[kn], sub * 4),
                                                _frag(qv, sub * 4),
                                                dots[kn],
                                            )
                        visible_end = ke
                        if CAUSAL:
                            hide = qlen - 1 - r // H - ko
                            visible_end = ke - (hide > 0).select(hide, fx.Int32(0))
                        scores = []
                        valid = []
                        for kn in range_constexpr(N // 16):
                            for i in range_constexpr(4):
                                ok = (r < M) & (kb + kn * 16 + lg * 4 + i < visible_end)
                                valid.append(ok)
                                scores.append(
                                    ok.select(
                                        dots[kn][i] * score_scale,
                                        fx.Float32(float("-inf")),
                                    )
                                )
                        tm = scores[0]
                        for i in range_constexpr(1, N // 4):
                            tm = tm.maximumf(scores[i])
                        for delta in range_constexpr(2):
                            tm = tm.maximumf(fx.gpu.shuffle_xor(tm, 16 << delta, 64))
                        nm = state[rt].maximumf(tm)
                        safe = (nm > fx.Float32(float("-inf"))).select(
                            nm, fx.Float32(0.0)
                        )
                        alpha = fmath.exp2(state[rt] - safe)
                        probs = [
                            valid[i].select(
                                fmath.exp2(scores[i] - safe), fx.Float32(0.0)
                            )
                            for i in range_constexpr(N // 4)
                        ]
                        sm = probs[0]
                        for i in range_constexpr(1, N // 4):
                            sm = sm + probs[i]
                        for delta in range_constexpr(2):
                            sm = sm + fx.gpu.shuffle_xor(sm, 16 << delta, 64)
                        sm = state[OWN + rt] * alpha + sm
                        for kn in range_constexpr(N // 16):
                            for i in range_constexpr(4):
                                fx.memref_store(
                                    _rne(probs[kn * 4 + i]), p_lds, (r, (i, lg, kn))
                                )
                        if lg == 0:
                            fx.memref_store(alpha, stats, (r, STAT_ALPHA))
                            fx.memref_store(nm, stats, (r, STAT_MAX))
                            fx.memref_store(sm, stats, (r, STAT_SUM))
                    else:
                        nm = state[rt]
                        sm = state[OWN + rt]
                    next_max.append(nm)
                    next_sum.append(sm)
                next_raw = []
                if NT < W:
                    # After the complete QK/softmax phase: never between MFMAs.
                    fx.rocdl.sched_barrier(0)
                    pre_tid = _unhoisted(tid)
                    pre_tok = (pre_tid % (N * GATHER_LANES)) // GATHER_LANES
                    pre_dc = pre_tid % GATHER_LANES
                    pre_base = fx.inttoptr(kv.type, s_pages[pre_tok])
                    for chunk in range_constexpr(3):
                        next_raw.append(
                            fx.ptr_load(
                                pre_base + pre_dc * GATHER_BYTES + chunk * 16,
                                result_type=fx.Vector.make_type(16, fx.Uint8),
                            ).bitcast(fx.Uint32)
                        )
                    fx.rocdl.sched_barrier(0)
                fx.barrier()
                outs = []
                # PV: A=V^T, B=P^T. All rows, disjoint column slices per wave.
                for rt in range_constexpr(NT):
                    r = rt * 16 + lr
                    alpha = stats[(r, STAT_ALPHA)]
                    for cd in range_constexpr(CD):
                        old = state[2 * OWN + rt * CD + cd]
                        acc = old * alpha
                        for kn in range_constexpr(N // 16):
                            pv = p_lds[(r, (None, lg, kn))].load()
                            vv = fx.Vector.from_elements(
                                [
                                    kv_v[((i, lg, kn), (wave, cd, lr))]
                                    for i in range_constexpr(4)
                                ],
                                dtype=fx.BFloat16,
                            )
                            acc = _mfma(vv, pv, acc)
                        outs.append(acc)
                result = yield next_max + next_sum + outs + next_raw
            # Shared statistics are final after the last PV barrier. A work
            # with zero tokens never enters the loop and writes zero/-inf.
            fx.barrier()
            # Keep epilogue-only div/mod and addresses out of the KV loop's
            # live set. One opaque move per work item avoids spilling them.
            store_tid = _unhoisted(tid)
            store_lr = store_tid % 16
            store_lg = (store_tid % 64) // 16
            store_wave = store_tid // 64
            for rt in range_constexpr(NT):
                r = rt * 16 + store_lr
                if r < M:
                    mass = fx.Float32(0.0)
                    mx = fx.Float32(float("-inf"))
                    if ke > ks:
                        mass = stats[(r, STAT_SUM)]
                        mx = stats[(r, STAT_MAX)]
                    denom = (mass > 0).select(mass, fx.Float32(1.0))
                    factor = descale / denom
                    lse = (mass > 0).select(
                        mx * fx.Float32(0.6931471805599453) + fmath.log(denom),
                        fx.Float32(float("-inf")),
                    )
                    qpos = r // H
                    h = r % H
                    if pr >= 0:
                        out_row = fx.Int64(pr + qpos) * H + h
                        for cd in range_constexpr(CD):
                            acc = result[2 * OWN + rt * CD + cd]
                            cols = (store_wave, cd, store_lg, None)
                            gr[(out_row, cols)].store(acc * factor)
                        if (store_wave == 0) & (store_lg == 0):
                            fx.memref_store(lse, gl, out_row)
                    else:
                        out_row = fx.Int64(qs + qpos) * H + h
                        for cd in range_constexpr(CD):
                            acc = result[2 * OWN + rt * CD + cd]
                            cols = (store_wave, cd, store_lg, None)
                            value = fx.Vector.from_elements(
                                [_rne(acc[i] * factor) for i in range_constexpr(4)],
                                dtype=fx.BFloat16,
                            )
                            go[(out_row, cols)].store(value)
                        # Trace-time flag around a device test: they must not be merged.
                        if WRITE_LSE:  # noqa: SIM102
                            if (store_wave == 0) & (store_lg == 0):
                                fx.memref_store(lse, gol, out_row)

    return stage1


@flyc.jit
def launch_mla_pagesize1_bf16_fp8(
    partial: fx.Pointer,
    partial_lse: fx.Pointer,
    output: fx.Pointer,
    output_lse: fx.Pointer,
    q: fx.Pointer,
    kv: fx.Pointer,
    pages: fx.Pointer,
    indptr: fx.Pointer,
    info: fx.Pointer,
    kv_scale: fx.Pointer,
    scale: fx.Float32,
    total_q: fx.Int32,
    num_pages: fx.Int32,
    num_indices: fx.Int32,
    num_partial: fx.Int32,
    num_work: fx.Int32,
    num_cus: fx.Constexpr[int],
    num_heads: fx.Constexpr[int],
    max_seqlen_q: fx.Constexpr[int],
    causal: fx.Constexpr[bool],
    write_final_lse: fx.Constexpr[bool],
    stream: fx.Stream,
):
    kernel = _build(num_heads, max_seqlen_q, causal, write_final_lse)
    kernel(
        partial,
        partial_lse,
        output,
        output_lse,
        q,
        kv,
        pages,
        indptr,
        info,
        kv_scale,
        scale,
        total_q,
        num_pages,
        num_indices,
        num_partial,
        num_work,
        num_cus,
    ).launch(grid=(num_cus, 1, 1), block=(WAVES * 64, 1, 1), stream=stream)
