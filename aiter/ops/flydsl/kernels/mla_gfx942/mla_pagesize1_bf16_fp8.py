# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""gfx942 persistent page-size-one MLA: BF16 Q, E4M3FNUZ KV.

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


def _buffer(ptr, count, dtype, vec=1):
    view = fx.make_view(fx.recast_iter(dtype, ptr), fx.make_layout((1 << 31), 1))
    buf = fx.rocdl.make_buffer_tensor(
        view, num_records_bytes=fx.Int64(count) * (dtype.width // 8)
    )
    return fx.Tensor(
        fx.make_view(
            fx.get_iter(buf), fx.make_layout(((1 << 31) // vec, vec), (vec, 1))
        )
    )


def _ld(buf, index, dtype, vec=1):
    frag = fx.make_rmem_tensor(fx.make_layout(vec, 1), dtype)
    atom = fx.make_copy_atom(fx.rocdl.BufferCopy(dtype.width * vec), dtype)
    fx.copy_atom_call(atom, fx.slice(buf, (index, None)), frag)
    return frag.load() if vec != 1 else frag.load()[0]


def _st(buf, index, value, dtype):
    frag = fx.make_rmem_tensor(fx.make_layout(1, 1), dtype)
    frag.store(fx.Vector.from_elements([value], dtype=dtype))
    fx.copy_atom_call(
        fx.make_copy_atom(fx.rocdl.BufferCopy(dtype.width), dtype),
        frag,
        fx.slice(buf, (index, None)),
    )


def _stv(buf, index, value, dtype, vec=4):
    frag = fx.make_rmem_tensor(fx.make_layout(vec, 1), dtype)
    frag.store(value)
    fx.copy_atom_call(
        fx.make_copy_atom(fx.rocdl.BufferCopy(dtype.width * vec), dtype),
        frag,
        fx.slice(buf, (index, None)),
    )


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
    # gfx942 native unpack is FNUZ. Every finite FP8 value is exactly BF16.
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
    GATHER_LANES = 576 // GATHER_BYTES

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
        gq = _buffer(q, fx.Int64(total_q) * H * 576, fx.BFloat16, 8)
        gp = _buffer(pages, num_indices, fx.Int32)
        gi = _buffer(indptr, num_cus + 1, fx.Int32)
        gw = _buffer(info, fx.Int64(num_work) * 8, fx.Int32)
        gr = _buffer(partial, fx.Int64(num_partial) * H * 512, fx.Float32, 4)
        gl = _buffer(partial_lse, fx.Int64(num_partial) * H, fx.Float32)
        go = _buffer(output, fx.Int64(total_q) * H * 512, fx.BFloat16, 4)
        gol = _buffer(output_lse, fx.Int64(total_q) * H, fx.Float32)
        descale = _uniform(
            _ld(_buffer(kv_scale, 1, fx.Float32), 0, fx.Float32).bitcast(fx.Int32)
        ).bitcast(fx.Float32)
        score_scale = scale * descale * fx.Float32(1.4426950408889634)
        shared = fx.SharedAllocator().allocate(Shared).peek()
        sk = shared.kv.ptr
        sp = shared.p.ptr
        ss = shared.stats.ptr
        s_pages = shared.pages.ptr
        begin = _uniform(_ld(gi, cu, fx.Int32))
        end = _uniform(_ld(gi, cu + 1, fx.Int32))
        for wi in range(begin, end, 1):
            row = fx.Int64(wi) * 8
            pr = _uniform(_ld(gw, row + 1, fx.Int32))
            qs = _uniform(_ld(gw, row + 2, fx.Int32))
            qe = _uniform(_ld(gw, row + 3, fx.Int32))
            ks = _uniform(_ld(gw, row + 4, fx.Int32))
            ke = _uniform(_ld(gw, row + 5, fx.Int32))
            ko = _uniform(_ld(gw, row + 6, fx.Int32))
            qlen = qe - qs
            M = qlen * H
            # Permute reduction dimensions: four consecutive MFMA steps cover
            # each lane group's contiguous 16 values, enabling ds_read_b128.
            qregs = []
            for rt in range_constexpr(OWN):
                r = (wave + rt * W) * 16 + lr
                for dc in range_constexpr(9):
                    for half in range_constexpr(2):
                        off = (
                            (fx.Int64(qs) * H + r) * 576 + dc * 64 + lg * 16 + half * 8
                        ) // 8
                        off = (r < M).select(off, fx.Int64(total_q) * H * 72)
                        qregs.append(_ld(gq, off, fx.BFloat16, 8))
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
                    page0 = _ld(gp, idx0, fx.Int32)
                    base0 = kv + fx.Int64(page0) * 576
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
                    page = _ld(gp, idx, fx.Int32)
                    fx.ptr_store(
                        fx.ptrtoint(kv + fx.Int64(page) * 576).to(fx.Uint64),
                        s_pages + tid,
                    )
                fx.barrier()
                gather_tid = _unhoisted(tid) if NT >= 7 else tid
                if wave < 6:
                    tok = gather_tid // GATHER_LANES
                    dc = gather_tid % GATHER_LANES
                    page_base = (
                        fx.inttoptr(kv.type, fx.ptr_load(s_pages + tok))
                        if NT == W
                        else kv
                    )
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
                            fx.ptr_store(
                                fx.Vector.from_elements(vals, dtype=fx.BFloat16),
                                sk + tok * LD + d + part * 8,
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
                                fx.ptr_load(
                                    sk + kn * 16 * LD + lr * LD + lg * 16,
                                    result_type=fx.Vector.make_type(8, fx.BFloat16),
                                )
                                for kn in range_constexpr(N // 16)
                            ]
                            fx.rocdl.sched_barrier(0)
                            for step in range_constexpr(18):
                                if const_expr(step + 1 < 18):
                                    ns = step + 1
                                    kbase = (
                                        lr * LD
                                        + (ns // 2) * 64
                                        + lg * 16
                                        + (ns % 2) * 8
                                    )
                                    next_k = [
                                        fx.ptr_load(
                                            sk + kn * 16 * LD + kbase,
                                            result_type=fx.Vector.make_type(
                                                8, fx.BFloat16
                                            ),
                                        )
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
                                        lr * LD + dc * 64 + lg * 16 + half * 8
                                    )
                                    kvals = []
                                    for kn in range_constexpr(N // 16):
                                        kvals.append(
                                            fx.ptr_load(
                                                sk + kn * 16 * LD + kbase,
                                                result_type=fx.Vector.make_type(
                                                    8, fx.BFloat16
                                                ),
                                            )
                                        )
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
                                fx.ptr_store(
                                    _rne(probs[kn * 4 + i]),
                                    sp + r * LP + kn * 16 + lg * 4 + i,
                                )
                        if lg == 0:
                            fx.ptr_store(alpha, ss + r * 3)
                            fx.ptr_store(nm, ss + r * 3 + 1)
                            fx.ptr_store(sm, ss + r * 3 + 2)
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
                    pre_base = fx.inttoptr(kv.type, fx.ptr_load(s_pages + pre_tok))
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
                    alpha = fx.ptr_load(ss + r * 3)
                    for cd in range_constexpr(CD):
                        old = state[2 * OWN + rt * CD + cd]
                        acc = old * alpha
                        for kn in range_constexpr(N // 16):
                            pv = fx.ptr_load(
                                sp + r * LP + kn * 16 + lg * 4,
                                result_type=fx.Vector.make_type(4, fx.BFloat16),
                            )
                            vv = fx.Vector.from_elements(
                                [
                                    fx.ptr_load(
                                        sk
                                        + (kn * 16 + lg * 4 + i) * LD
                                        + wave * (512 // W)
                                        + cd * 16
                                        + lr
                                    )
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
                        mass = fx.ptr_load(ss + r * 3 + 2)
                        mx = fx.ptr_load(ss + r * 3 + 1)
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
                            d = store_wave * (512 // W) + cd * 16 + store_lg * 4
                            value = acc * factor
                            _stv(gr, (out_row * 512 + d) // 4, value, fx.Float32)
                        if (store_wave == 0) & (store_lg == 0):
                            _st(gl, out_row, lse, fx.Float32)
                    else:
                        out_row = fx.Int64(qs + qpos) * H + h
                        for cd in range_constexpr(CD):
                            acc = result[2 * OWN + rt * CD + cd]
                            d = store_wave * (512 // W) + cd * 16 + store_lg * 4
                            value = fx.Vector.from_elements(
                                [_rne(acc[i] * factor) for i in range_constexpr(4)],
                                dtype=fx.BFloat16,
                            )
                            _stv(go, (out_row * 512 + d) // 4, value, fx.BFloat16)
                        # Trace-time flag around a device test: they must not be merged.
                        if WRITE_LSE:  # noqa: SIM102
                            if (store_wave == 0) & (store_lg == 0):
                                _st(gol, out_row, lse, fx.Float32)

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
