# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""o_proj GEMV + its TP4 one-shot all-reduce + residual add + RMSNorm in one launch (gfx1250, decode rows).

The decode attention output of a TP4 model (MiniMax-M3.1: N 6144, K 2048) goes through two launches: the native
MXFP8 decode GEMM (``mxfp8_decode_gemm_gfx1250``) writes the bf16 TP-partial, then the k5 one-shot all-reduce
(``csrc/include/ar_k5_gfx1250.cuh``, Gluon order, bf16 out) pushes it to the peers, sums, adds the residual and
normalises. This launch does both with the same bits:

* workgroups ``0 .. N / cols - 1`` run the decode GEMM's tile (``ks`` 1) unchanged up to its output. Instead of a
  bf16 store, each forms its output columns' 16-byte packs (8 bf16 of a row), rewrites the 0xffff half-words to
  0x7fff (the k5 sender's sentinel rule) and stores every pack, system scope, into slot ``rank`` of buffer
  ``round`` of every rank's k5 inbox -- its own included (k5 reads its own rows from memory; here the own slot
  carries them, so a receiver needs no other wait). ``round`` is the k5 send counter of the pack's slab.
* workgroups ``N / cols ..``: k5's receiver, two rows a workgroup (waves 0-7 and 8-15), a wave a 96-pack slab:
  re-arms buffer ``(round + 2) % 3`` of the peers' slots (as k5 does), polls the four slots of buffer ``round``
  until no half-word is the sentinel, then k5's Gluon epilogue: the rank-order fp32 sum, its bf16 rounding, the
  fp32 residual add, the squares in (plane, element) order, the lane and wave pair trees, ``v_rcp_f32`` and
  ``v_rsq_f32``, eps added in fp64, ``(s * rstd) * (w + 1)``. Then it re-arms its own slot of buffer ``round``
  (only this launch writes and reads it) and advances the slab's send and receive counters, as one k5 call does.

The inbox, its counters and their peer mapping are k5's own (the custom all-reduce's IPC allocation): a launch
interleaves with k5 calls on the same counters. The arithmetic flags: no fused multiply-adds (separate multiplies
and adds), IEEE fp32 denormals (FlyDSL's default, as the Gluon-order k5 kernels are built).
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import llvm as _llvm
from flydsl.expr import const_expr, range_constexpr, rocdl
from flydsl.expr.typing import Constexpr, T
from flydsl.expr.typing import Vector as Vec

from aiter.ops.flydsl.kernels import tdm_ops_gfx1250 as tdm_ops
from aiter.ops.flydsl.kernels.buffer_ops import (
    buffer_load,
    buffer_store,
    create_buffer_resource_from_addr,
)
from aiter.ops.flydsl.kernels.gemm_common_gfx1250 import make_lds_copy_ops, pipeline_fence
from aiter.ops.flydsl.kernels.mxfp8_decode_gemm_gfx1250 import (
    A_KC,
    A_PAD,
    P_PAD,
    WAVE,
    WMMA_K,
    decode_gemm_config_error,
    decode_gemm_lds_bytes,
)
from aiter.ops.flydsl.kernels.tensor_shim import (
    AITER_FLYDSL_KERNARG_PRELOAD,
    AITER_FLYDSL_KERNARG_PRELOAD_COUNT,
    buf_copy_load,
    ptr_buf_tensor,
)

RANKS = 4
HIDDEN = 6144
ROW_PACKS = HIDDEN // 8  # 768
RWAVES = 8  # waves of a row (k5: kWaves)
SLAB = 96  # packs a wave (k5: kSlabPacks)
BUFS = 3
SCOPE_SYS = 3 << 3  # gfx12 CPol SCOPE field
VOLATILE = -(1 << 31)  # raw buffer intrinsics' aux bit 31 (as a signed i32)
SENT_LO = 0x7FFF7FFF
SENT_HI = -0x7FFF8000  # 0x80008000 as int32


def _bits(w):
    """One bit (the half-word's top bit) per half-word of ``w`` that is 0xffff (k5's sentinel_mask)."""
    x = w ^ fx.Int32(-1)
    t = ((x & fx.Int32(SENT_LO)) + fx.Int32(SENT_LO)) | x
    return (t & fx.Int32(SENT_HI)) ^ fx.Int32(SENT_HI)


def _bf_lo(w):
    return (w << fx.Int32(16)).bitcast(fx.Float32)


def _bf_hi(w):
    return (w & fx.Int32(-65536)).bitcast(fx.Float32)


def _f2bf2(lo, hi):
    """Two fp32 -> packed bf16 bits (round to nearest even, gfx1250's convert)."""
    v = Vec.from_elements([lo, hi], fx.Float32)
    return Vec(v.to(fx.BFloat16).bitcast(fx.Int32))[0]


def _rcp(v):
    """v_rcp_f32 by inline asm, as k5's rcp_asm: the compiler may not fold it to a correctly rounded 1/x."""
    return fx.Float32(
        _llvm.InlineAsmOp(
            res=T.f32,
            operands_=[fx.Float32(v).ir_value()],
            asm_string="v_rcp_f32 $0, $1",
            constraints="=v,v",
            has_side_effects=True,
            is_align_stack=False,
        ).result
    )


def _mem_clobber():
    """Keeps the poll loop's loads inside the loop (k5: asm volatile("" ::: "memory"))."""
    _llvm.InlineAsmOp(
        res=None, operands_=[], asm_string="", constraints="~{memory}", has_side_effects=True, is_align_stack=False
    )


def _rsq(v):
    return fx.Float32(_llvm.call_intrinsic(T.f32, "llvm.amdgcn.rsq.f32", [fx.Float32(v).ir_value()], [], []))


def _now32():
    return fx.Int32(fx.Int64(_llvm.call_intrinsic(T.i64, "llvm.readsteadycounter", [], [], [])) & fx.Int64(0x7FFFFFFF))


def _lds_barrier():
    rocdl.s_wait_dscnt(0)
    rocdl.s_barrier_signal(-1)
    rocdl.s_barrier_wait(-1)


@flyc.jit
def launch_oproj_ar(
    arg_a: fx.Pointer,
    arg_as: fx.Pointer,
    arg_w: fx.Pointer,
    arg_ws: fx.Pointer,
    arg_res: fx.Pointer,
    arg_nw: fx.Pointer,
    arg_out: fx.Pointer,
    arg_res_out: fx.Pointer,
    arg_ctr: fx.Pointer,
    i64_in0: fx.Int64,
    i64_in1: fx.Int64,
    i64_in2: fx.Int64,
    i64_in3: fx.Int64,
    i32_m: fx.Int32,
    i32_lda: fx.Int32,
    i32_slot: fx.Int32,
    i32_rank: fx.Int32,
    i32_maxit: fx.Int32,
    f64_eps: fx.Float64,
    stream: fx.Stream,
    N: Constexpr[int],
    K: Constexpr[int],
    tile_m: Constexpr[int],
    nb: Constexpr[int],
    kw: Constexpr[int],
    wn: Constexpr[int],
    wdepth: Constexpr[int],
    wcpol: Constexpr[int] = 0,
    atdm: Constexpr[int] = 0,
    max_rows: Constexpr[int] = 256,
    tl: Constexpr[int] = 0,
    arg_tl: fx.Pointer = None,
):
    """``tl`` (diagnostic): thread 0 of every workgroup stamps the 100 MHz steady counter into ``arg_tl``
    (int32 [grid, 4]): GEMV workgroups at entry, after the GEMV (packs formed), after the pushes are acknowledged;
    receivers at entry, when the poll starts, when the poll ends, after the epilogue's stores.

    ``arg_ctr``: the k5 context's counters (send [max_rows * 8], receive [max_rows * 8], err), ``i64_in*``: the
    four ranks' k5 inboxes, ``i32_slot`` packs a sender a buffer (k5: max_rows * 768). Rows ``i32_m`` <= tile_m.
    ``arg_out`` / ``arg_res_out``: bf16 [m, N] normed output and residual (k5's out / residual_out)."""
    ks = 1
    err = decode_gemm_config_error(N, K, tile_m, nb, ks, kw, wn, wdepth, atdm)
    if err is not None:
        raise ValueError(f"oproj_ar: {err}")
    if N != HIDDEN or kw * wn != 2 * RWAVES:
        raise ValueError("oproj_ar: N 6144 and 16 waves (two receiver rows a workgroup)")
    steps = K // WMMA_K
    steps_w = steps // kw
    cols = nb * 16
    ppr = cols // 8  # packs of a row a GEMV workgroup
    assert cols % 8 == 0 and SLAB % ppr == 0, "a workgroup's packs stay in one slab"
    mbt = tile_m // 16
    nbw = nb // wn
    block = kw * wn * WAVE
    wdepth = max(1, min(wdepth, steps_w))
    sa_opsel = mbt % 2 == 0
    sa_loads = mbt // 2 if sa_opsel else mbt
    KB4 = K // 128
    gemv_lds = decode_gemm_lds_bytes(tile_m, nb, kw, K, ks, atdm)
    PLANES = kw * tile_m * (nb * 16 * 4 + P_PAD)
    A_ROW = A_KC + A_PAD
    A_STAGE = tile_m * A_ROW
    P_ROW = cols * 4 + P_PAD
    P_PLANE = tile_m * P_ROW
    n_gemv = N // cols
    RED_OFF = (gemv_lds + 15) // 16 * 16  # receiver: 16 fp32 wave partials
    lds_bytes = RED_OFF + 2 * RWAVES * 4
    ctr_recv = max_rows * RWAVES
    ctr_err = 2 * max_rows * RWAVES
    kernel_name = (
        f"oproj_ar_k5_gfx1250{'_tl' if tl else ''}_n{N}_k{K}_t{tile_m}x{cols}_w{kw}x{wn}_d{wdepth}"
        f"{f'_c{wcpol}' if wcpol else ''}{'_at' if atdm else ''}"
    )

    @flyc.kernel(name=kernel_name, known_block_size=[block, 1, 1])
    def kernel_oproj_ar(
        arg_a: fx.Pointer,
        arg_as: fx.Pointer,
        arg_w: fx.Pointer,
        arg_ws: fx.Pointer,
        arg_res: fx.Pointer,
        arg_nw: fx.Pointer,
        arg_out: fx.Pointer,
        arg_res_out: fx.Pointer,
        arg_ctr: fx.Pointer,
        i64_in0: fx.Int64,
        i64_in1: fx.Int64,
        i64_in2: fx.Int64,
        i64_in3: fx.Int64,
        i32_m: fx.Int32,
        i32_lda: fx.Int32,
        i32_slot: fx.Int32,
        i32_rank: fx.Int32,
        i32_maxit: fx.Int32,
        f64_eps: fx.Float64,
        arg_tl: fx.Pointer,
    ):
        tid = fx.Int32(fx.thread_idx.x)
        bid = fx.Int32(fx.block_idx.x)
        tlp = fx.recast_iter(
            fx.PointerType.get(elem_ty=fx.Int32.ir_type, address_space=fx.AddressSpace.Global, alignment=4), arg_tl
        )

        def stamp(slot):
            if const_expr(tl):
                if tid == 0:
                    fx.ptr_store(_now32(), tlp + (bid * 4 + slot))

        stamp(0)
        wave = fx.Int32(rocdl.readfirstlane(T.i32, tid // WAVE))
        lane = tid % WAVE
        arena = fx.SharedAllocator(static=False)
        arena.allocate(lds_bytes)
        base_ptr = arena.base_ptr
        lds_idx = fx.Int64(fx.ptrtoint(base_ptr))
        lds_load_b128, lds_store_b128 = make_lds_copy_ops(128)
        ctr = fx.recast_iter(
            fx.PointerType.get(elem_ty=fx.Int32.ir_type, address_space=fx.AddressSpace.Global, alignment=4),
            arg_ctr,
        )
        inbox_bytes = fx.Int32(BUFS * RANKS) * i32_slot * fx.Int32(16)
        ins = [i64_in0, i64_in1, i64_in2, i64_in3]
        rsrc = [create_buffer_resource_from_addr(fx.Int64(p), num_records_bytes=inbox_bytes) for p in ins]

        if bid < n_gemv:
            # ---------------- the decode GEMM's tile (ks 1, rows <= tile_m) ----------------
            ntile = bid
            w_k = wave // wn
            w_n = wave % wn
            lane16 = lane % 16
            kgrp = lane // 16
            m0 = fx.Int32(0)
            ks0 = w_k * steps_w
            if const_expr(atdm):
                gA_base = fx.recast_iter(fx.Int8, arg_a)
                if wave < (K // A_KC):
                    c_off = fx.Int64(wave * A_KC)
                    desc = tdm_ops.make_tensor_descriptor_2d(
                        global_ptr=fx.Tensor(fx.make_view(fx.add_offset(gA_base, c_off),
                                                          fx.make_layout((tile_m, A_KC), (A_KC, 1)))),
                        lds_memref=fx.Tensor(fx.make_view(fx.add_offset(base_ptr, PLANES + wave * A_STAGE),
                                                          fx.make_layout((tile_m, A_KC), (A_ROW, 1)))),
                        global_offset=(0, 0),
                        tensor_shape=(tile_m, A_KC),
                        strides=(i32_lda, 1),
                        tile_shape=(tile_m, A_KC),
                        elem_bytes=1,
                        pad_interval=A_KC,
                        pad_amount=A_PAD,
                        num_warps=1,
                        workgroup_mask=0,
                        oob_outer_bound=i32_m,
                    )
                    tdm_ops.tensor_load_2d(desc)
            a_buf = ptr_buf_tensor(arg_a, fx.Int32, unit_elems=4, num_records_bytes=i32_m * i32_lda)
            sa_buf = ptr_buf_tensor(arg_as, fx.Int32, num_records_bytes=i32_m * (K // 32))
            b_buf = ptr_buf_tensor(arg_w, fx.Int32, unit_elems=4)
            sb_buf = ptr_buf_tensor(arg_ws, fx.Int32)
            lda16 = i32_lda // 16
            a_row = [(m0 + mi * 16 + lane16) * lda16 + kgrp for mi in range_constexpr(mbt)]
            nblk0 = ntile * nb + w_n * nbw
            b_base = [(nblk0 + j) * K + lane for j in range_constexpr(nbw)]
            sb_col = [(nblk0 + j) * 16 + lane16 for j in range_constexpr(nbw)]
            sb_base = [(c // 32) * (K // 4) + c % 32 for c in sb_col]
            sa_row = [
                (m0 + (2 * p) * 16 + lane) if const_expr(sa_opsel) else (m0 + p * 16 + lane16)
                for p in range_constexpr(sa_loads)
            ]
            sa_base = [r * KB4 for r in sa_row]

            def _cat16(v):
                v01 = v[0].shuffle(v[1], list(range(8)))
                v23 = v[2].shuffle(v[3], list(range(8)))
                return v01.shuffle(v23, list(range(16)))

            def load_step(i):
                ksg = ks0 + i
                bfr = [
                    [
                        buf_copy_load(b_buf, b_base[j] + ksg * 128 + jj * 32, fx.Int32, 4, cache_modifier=wcpol)
                        for jj in range_constexpr(4)
                    ]
                    for j in range_constexpr(nbw)
                ]
                sb = [buf_copy_load(sb_buf, sb_base[j] + ksg * 32, fx.Int32) for j in range_constexpr(nbw)]
                sa = [buf_copy_load(sa_buf, sa_base[p] + ksg, fx.Int32) for p in range_constexpr(sa_loads)]
                afr = None
                if const_expr(not atdm):
                    afr = [
                        [buf_copy_load(a_buf, a_row[mi] + ksg * 8 + jj * 2, fx.Int32, 4) for jj in range_constexpr(4)]
                        for mi in range_constexpr(mbt)
                    ]
                return bfr, sb, sa, afr

            def lds_a_frag(mi, i):
                s_ = w_k * steps_w + i
                base = PLANES + (s_ // 4) * A_STAGE + (mi * 16 + lane16) * A_ROW + (s_ % 4) * 128 + kgrp * 16
                return [Vec(lds_load_b128(lds_idx, base + 32 * jj)) for jj in range_constexpr(4)]

            atoms = [
                fx.make_mma_atom(
                    fx.rocdl.WMMAScale(
                        16, 16, WMMA_K, fx.Float8E4M3FN, fx.Float8E4M3FN, fx.Float32,
                        opsel_a=0, opsel_b=sel,
                    )
                )
                for sel in range_constexpr(2 if sa_opsel else 1)
            ]
            acc = [[fx.make_rmem_tensor(8, fx.Float32) for _ in range_constexpr(nbw)] for _ in range_constexpr(mbt)]
            for row in acc:
                for c in row:
                    c.store(fx.full(8, 0.0, fx.Float32))

            def _rmem(v):
                t = fx.make_rmem_tensor(16, fx.Int32)
                t.store(v)
                return t

            def compute_step(st, i):
                bfr, sb, sa, afr = st
                if const_expr(atdm):
                    afr = [lds_a_frag(mi, i) for mi in range_constexpr(mbt)]
                bt = [_rmem(_cat16(bfr[j])) for j in range_constexpr(nbw)]
                at = [_rmem(_cat16(afr[mi])) for mi in range_constexpr(mbt)]
                for mi in range_constexpr(mbt):
                    for j in range_constexpr(nbw):
                        fx.gemm(
                            atoms[mi % 2 if sa_opsel else 0],
                            acc[mi][j],
                            bt[j],
                            at[mi],
                            acc[mi][j],
                            scale_a=sb[j],
                            scale_b=sa[mi // 2 if sa_opsel else mi],
                        )

            inflight = [load_step(i) for i in range_constexpr(wdepth)]
            rocdl.sched_barrier(0)
            if const_expr(atdm):
                pipeline_fence(outstanding=0, use_cluster=False)
            for i in range_constexpr(steps_w):
                st = inflight.pop(0)
                if const_expr(i + wdepth < steps_w):
                    inflight.append(load_step(i + wdepth))
                    rocdl.sched_barrier(0)
                compute_step(st, i)

            for mi in range_constexpr(mbt):
                for j in range_constexpr(nbw):
                    row = mi * 16 + lane16
                    col = (w_n * nbw + j) * 16 + kgrp * 8
                    v = Vec(acc[mi][j].load())
                    off = w_k * P_PLANE + row * P_ROW + col * 4
                    lds_store_b128(lds_idx, off, v.shuffle(v, [0, 1, 2, 3]).bitcast(fx.Int32))
                    lds_store_b128(lds_idx, off + 16, v.shuffle(v, [4, 5, 6, 7]).bitcast(fx.Int32))
            _lds_barrier()

            def plane_sum(r, c4):
                tot = None
                for k in range_constexpr(kw):
                    part = Vec(lds_load_b128(lds_idx, k * P_PLANE + r * P_ROW + c4 * 16)).bitcast(fx.Float32)
                    tot = part if tot is None else tot + part
                return tot

            stamp(1)
            # ---------------- the packs, pushed into every rank's k5 inbox ----------------
            p_units = tile_m * ppr
            for t in range_constexpr((p_units + block - 1) // block):
                u = tid + t * block
                r = u // ppr
                c8 = u % ppr
                ok = r < i32_m
                if const_expr(p_units % block):
                    ok = ok & (u < p_units)
                if ok:
                    lo4 = plane_sum(r, 2 * c8)
                    hi4 = plane_sum(r, 2 * c8 + 1)
                    v8 = lo4.shuffle(hi4, list(range(8)))
                    pk = Vec(v8.to(fx.BFloat16).bitcast(fx.Int32))
                    words = [fx.Int32(pk[c]) for c in range_constexpr(4)]
                    words = [w_ ^ _bits(w_) for w_ in words]
                    data = Vec.from_elements(words, fx.Int32)
                    pcol = ntile * ppr + c8
                    p = r * ROW_PACKS + pcol
                    g = r * RWAVES + pcol // SLAB
                    rnd = fx.Int32(ctr[g])
                    off = ((rnd * RANKS + i32_rank) * i32_slot + p) * 16
                    for d in range_constexpr(RANKS):
                        buffer_store(data.ir_value(), rsrc[d], off, offset_is_bytes=True, cache_modifier=SCOPE_SYS)
            if const_expr(tl):
                rocdl.s_wait_storecnt(0)
                _lds_barrier()
            stamp(2)
        else:
            # ---------------- k5's receiver: two rows a workgroup, a wave a slab ----------------
            half = wave // RWAVES
            w = wave % RWAVES
            r = (bid - n_gemv) * 2 + half
            live = r < i32_m
            rr = live.select(r, fx.Int32(0))
            g = rr * RWAVES + w
            srnd = fx.Int32(rocdl.readfirstlane(T.i32, fx.Int32(ctr[g])))
            rrnd = fx.Int32(rocdl.readfirstlane(T.i32, fx.Int32(ctr[ctr_recv + g])))
            p0 = rr * ROW_PACKS + w * SLAB + lane
            res_rs = create_buffer_resource_from_addr(fx.Int64(fx.ptrtoint(arg_res)), num_records_bytes=i32_m * HIDDEN * 2)
            nw_rs = create_buffer_resource_from_addr(fx.Int64(fx.ptrtoint(arg_nw)), num_records_bytes=HIDDEN * 2)
            out_rs = create_buffer_resource_from_addr(fx.Int64(fx.ptrtoint(arg_out)), num_records_bytes=i32_m * HIDDEN * 2)
            ro_rs = create_buffer_resource_from_addr(fx.Int64(fx.ptrtoint(arg_res_out)), num_records_bytes=i32_m * HIDDEN * 2)
            # this rank's own inbox, one resource
            my_base = fx.Int64(i64_in0)
            for d in range_constexpr(1, RANKS):
                my_base = (i32_rank == d).select(fx.Int64(ins[d]), my_base)
            my_rs = create_buffer_resource_from_addr(my_base, num_records_bytes=inbox_bytes)
            rs = []
            wt = []
            for j in range_constexpr(3):
                rs.append(Vec(buffer_load(res_rs, (p0 + 32 * j) * 4, vec_width=4, dtype=T.i32)))
                wt.append(Vec(buffer_load(nw_rs, (w * SLAB + 32 * j + lane) * 4, vec_width=4, dtype=T.i32)))
            sent = Vec.from_elements([fx.Int32(-1)] * 4, fx.Int32)

            def my_store(off, data):
                buffer_store(data.ir_value(), my_rs, off, offset_is_bytes=True, cache_modifier=SCOPE_SYS)

            def my_load(off):
                # buffer_load takes a dword offset; not volatile: one wait a poll round, not one a load
                return Vec(buffer_load(my_rs, off // 4, vec_width=4, dtype=T.i32, cache_modifier=SCOPE_SYS))

            if live:
                # re-arm the peers' slots of buffer (round + 2) % 3, as k5 does
                rearm = (rrnd + 2) % BUFS
                for i in range_constexpr(1, RANKS):
                    s_ = (i32_rank + i) & (RANKS - 1)
                    for j in range_constexpr(3):
                        my_store(((rearm * RANKS + s_) * i32_slot + p0 + 32 * j) * 16, sent)

            def offs(s_, j):
                return ((rrnd * RANKS + s_) * i32_slot + p0 + 32 * j) * 16

            def load_all():
                _mem_clobber()
                return tuple(my_load(offs(s_, j)) for s_ in range_constexpr(RANKS) for j in range_constexpr(3))

            def pending(vs):
                m = fx.Int32(0)
                for v in vs:
                    for c in range_constexpr(4):
                        m = m | _bits(fx.Int32(v[c]))
                return fx.Int32(rocdl.ballot(T.i32, m != 0)) != 0

            stamp(1)
            (a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11) = load_all()
            pend = live & pending((a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11))
            it = fx.Int32(0)
            while pend & (it < i32_maxit):
                rocdl.s_sleep(1)
                (a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11) = load_all()
                pend = pending((a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11))
                it = it + 1
            if live & pend & (lane == 0):
                fx.atomic_add(ctr + ctr_err, fx.Int32(1), syncscope="agent")
            stamp(2)
            flat = (a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11)
            ins_v = [[flat[s_ * 3 + j] for j in range(3)] for s_ in range(RANKS)]

            # k5's Gluon epilogue
            s = [[None] * 8 for _ in range(3)]
            for j in range_constexpr(3):
                for k in range_constexpr(8):
                    acc_ = None
                    for q in range_constexpr(RANKS):
                        wq = fx.Int32(ins_v[q][j][k // 2])
                        x_ = _bf_hi(wq) if k % 2 else _bf_lo(wq)
                        acc_ = x_ if acc_ is None else acc_ + x_
                    s[j][k] = acc_
            sq = None
            for j in range_constexpr(3):
                o = []
                for c in range_constexpr(4):
                    ab = _f2bf2(s[j][2 * c], s[j][2 * c + 1])
                    rw = fx.Int32(rs[j][c])
                    s[j][2 * c] = _bf_lo(ab) + _bf_lo(rw)
                    s[j][2 * c + 1] = _bf_hi(ab) + _bf_hi(rw)
                for k in range_constexpr(8):
                    e = s[j][k] * s[j][k]
                    sq = e if sq is None else sq + e
                for c in range_constexpr(4):
                    o.append(_f2bf2(s[j][2 * c], s[j][2 * c + 1]))
                if live:
                    buffer_store(Vec.from_elements(o, fx.Int32).ir_value(), ro_rs, (p0 + 32 * j) * 16,
                                 offset_is_bytes=True)
            for d_ in range_constexpr(5):
                sq = sq + sq.shuffle_xor(fx.Int32(1 << d_), fx.Int32(WAVE))
            red = fx.recast_iter(
                fx.PointerType.get(elem_ty=fx.Float32.ir_type, address_space=fx.AddressSpace.Shared, alignment=4),
                base_ptr,
            ) + (RED_OFF // 4)
            if lane == 0:
                fx.ptr_store(sq, red + wave)
            _lds_barrier()
            pr = [fx.Float32(red[half * RWAVES + i]) for i in range_constexpr(RWAVES)]
            total = ((pr[0] + pr[1]) + (pr[2] + pr[3])) + ((pr[4] + pr[5]) + (pr[6] + pr[7]))
            mean = total * _rcp(fx.Float32(float(HIDDEN)))
            rstd = _rsq(fx.Float32(fx.Float64(mean) + f64_eps))
            for j in range_constexpr(3):
                o = []
                for c in range_constexpr(4):
                    wv = fx.Int32(wt[j][c])
                    y0 = (s[j][2 * c] * rstd) * (_bf_lo(wv) + fx.Float32(1.0))
                    y1 = (s[j][2 * c + 1] * rstd) * (_bf_hi(wv) + fx.Float32(1.0))
                    o.append(_f2bf2(y0, y1))
                if live:
                    buffer_store(Vec.from_elements(o, fx.Int32).ir_value(), out_rs, (p0 + 32 * j) * 16,
                                 offset_is_bytes=True)
            if live:
                # this launch's own slot of buffer round: consumed, re-armed for the next o_proj launch
                for j in range_constexpr(3):
                    my_store(offs(i32_rank, j), sent)
                if lane == 0:
                    fx.ptr_store((srnd + 1) % BUFS, ctr + g)
                    fx.ptr_store((rrnd + 1) % BUFS, ctr + (ctr_recv + g))
            if const_expr(tl):
                rocdl.s_wait_storecnt(0)
                _lds_barrier()
            stamp(3)

    mrecv = (i32_m + 1) // 2
    if arg_tl is None:
        arg_tl = arg_ctr
    kernel_oproj_ar(
        arg_a, arg_as, arg_w, arg_ws, arg_res, arg_nw, arg_out, arg_res_out, arg_ctr,
        i64_in0, i64_in1, i64_in2, i64_in3, i32_m, i32_lda, i32_slot, i32_rank, i32_maxit, f64_eps, arg_tl,
    ).launch(grid=(n_gemv + mrecv, 1, 1), block=(block, 1, 1), stream=stream)


launch_oproj_ar.compile_hints["llvm_options"] = {
    "amdgpu-kernarg-preload": AITER_FLYDSL_KERNARG_PRELOAD,
    "amdgpu-kernarg-preload-count": AITER_FLYDSL_KERNARG_PRELOAD_COUNT,
}
