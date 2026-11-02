# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
"""GEMM1 compute shared by fused MegaMoE v2 stage1 and its standalone interface."""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl.expr import const_expr, range_constexpr, rocdl
from flydsl.expr.typing import Vector as Vec

from ..tensor_shim import _run_compiled, ptr_buf_tensor
from .gemm_util import (
    _PACK,
    AS2RLoader,
    AScaleLoader,
    ATileLoader,
    BScaleLoader,
    BWeightLoader,
    MfmaScaleGU,
    PackedAScaleLoader,
    ScopedAS2RLoader,
    ScopedAScaleLoader,
    SiluQuantEpilogue,
    SwizzledQuantEpilogue,
    TileScheduler,
    UniformATileLoader,
    make_lds_dma_scopes,
    opaque_sgpr,
    wait_lds_barrier,
)


class _LdsF32View:
    def __init__(self, ptr):
        self.ptr = ptr


# fmt: off
@flyc.jit
def do_tile(m_tile, n_tile_base, expert, sched, a_gather, a_s2r, b_loader, b_scale, a_scale, mfma, epi, a_buf,
    a_scale_lds, a_lds_i32, K_ITERS, M_REPEAT, NUM_ACC_N, A_K_STEP_BYTES, pipe_weights,
    mfma_amajor, async_a_copy, trb_rsrc, tib_rsrc, indirect_input, pair_k):
# fmt: on
    N_ACC = M_REPEAT * NUM_ACC_N
    NUM_B_SCALE = NUM_ACC_N // _PACK
    NUM_A_SCALE = M_REPEAT // _PACK
    B_STATE_END = (
        N_ACC + NUM_ACC_N * _PACK
    )
    SB_STATE_END = B_STATE_END + NUM_B_SCALE
    last = fx.Int32(K_ITERS - 1)
    tile_row_base = trb_rsrc[m_tile]
    tile_input_base = tile_row_base
    if const_expr(indirect_input):
        tile_input_base = tib_rsrc[m_tile]
    b_row = sched.gate_base_row(expert) + n_tile_base
    a_gather.for_tile(tile_input_base)
    if const_expr(pipe_weights):
        if const_expr(async_a_copy):
            a_gather.prefetch_to_lds(
                fx.Int32(0),
                a_buf,
                fx.Int32(0),
            )
        else:
            a_gather.store(
                a_buf,
                a_gather.load_regs(fx.Int32(0)),
                fx.Int32(0),
            )
        a_scale.stage(a_scale_lds, tile_input_base)
        wait_lds_barrier(0 if async_a_copy else 63)
        b0 = b_loader.load_step(b_row, fx.Int32(0))
        init = [mfma.zero_value for _ in range(N_ACC)]
        init += [h for ni_list in b0 for h in ni_list]
        if const_expr(async_a_copy):
            init += b_scale.load_step(
                b_row,
                fx.Int32(0),
            )
            init += a_scale.load_step(
                a_scale_lds,
                fx.Int32(0),
            )
        def _kstep(sp_i, state):
            sp = fx.Int32(sp_i)
            acc = [Vec(a) for a in state[:N_ACC]]
            b_prev = [
                [Vec(state[N_ACC + ni * _PACK + ks]) for ks in range(_PACK)] for ni in range(NUM_ACC_N)
            ]
            cur_off = (sp & 1) * fx.Int32(a_lds_i32)
            nxt_off = ((sp + 1) & 1) * fx.Int32(a_lds_i32)
            spn = sp + 1
            if const_expr(async_a_copy):
                sb = [
                    fx.Int32(
                        state[B_STATE_END + group]
                    )
                    for group in range_constexpr(
                        NUM_B_SCALE
                    )
                ]
                sa = [
                    fx.Int32(
                        state[SB_STATE_END + group]
                    )
                    for group in range_constexpr(
                        NUM_A_SCALE
                    )
                ]
            else:
                sb = b_scale.load_step(b_row, sp)
                sa = a_scale.load_step(
                    a_scale_lds,
                    sp,
                )

            def a_load(mi, ks, _base=cur_off):
                return a_s2r.load_operand(a_buf, mi, ks, _base)

            if const_expr(async_a_copy):
                rocdl.sched_barrier(0)
                a_gather.prefetch_to_lds(
                    spn * fx.Int32(A_K_STEP_BYTES),
                    a_buf,
                    nxt_off,
                )
                rocdl.sched_barrier(0)
                sb_next = b_scale.load_step(
                    b_row,
                    spn,
                )
                sa_next = a_scale.load_step(
                    a_scale_lds,
                    spn,
                )
            else:
                a_regs = a_gather.load_regs(
                    spn * fx.Int32(A_K_STEP_BYTES)
                )

            def load_next(ni, _kn=spn):
                return b_loader.load_ni(b_row, ni, _kn)

            call_pipe = (
                mfma.call_pipe_am
                if mfma_amajor
                else mfma.call_pipe
            )
            acc, b_next = call_pipe(
                a_load,
                b_prev,
                acc,
                sa,
                sb,
                load_next,
            )
            if const_expr(async_a_copy):
                wait_lds_barrier(
                    NUM_ACC_N * _PACK
                    + NUM_B_SCALE
                )
            else:
                a_gather.store(a_buf, a_regs, nxt_off)
                wait_lds_barrier()
            yv = list(acc) + [h for ni_list in b_next for h in ni_list]
            if const_expr(async_a_copy):
                yv += sb_next
                yv += sa_next
            return yv

        if const_expr(pair_k):
            # Two K steps per iteration (ping-pong B registers): the rolled loop
            # copies the prefetched B tile into loop-carried registers, which
            # forces a vmcnt drain of the prefetch every step.
            pairs = (K_ITERS - 1) // 2
            for pr_i, state in range(0, pairs, 1, init=init):
                pr = fx.Int32(pr_i)
                mid = _kstep(pr * 2, state)
                yv = _kstep(pr * 2 + 1, mid)
                state = yield yv
            if const_expr((K_ITERS - 1) % 2):
                state = _kstep(fx.Int32(K_ITERS - 2), state)
        else:
            for sp_i, state in range(0, K_ITERS - 1, 1, init=init):
                yv = _kstep(sp_i, state)
                state = yield yv
        acc = [Vec(r) for r in state[:N_ACC]]
        b_prev = [
            [Vec(state[N_ACC + ni * _PACK + ks]) for ks in range(_PACK)]
            for ni in range(NUM_ACC_N)
        ]
        final_off = (last & 1) * fx.Int32(a_lds_i32)

        def final_a_load(mi, ks, _base=final_off):
            return a_s2r.load_operand(a_buf, mi, ks, _base)

        if const_expr(async_a_copy):
            sb = [
                fx.Int32(
                    state[B_STATE_END + group]
                )
                for group in range_constexpr(
                    NUM_B_SCALE
                )
            ]
            sa = [
                fx.Int32(
                    state[SB_STATE_END + group]
                )
                for group in range_constexpr(
                    NUM_A_SCALE
                )
            ]
        else:
            sb = b_scale.load_step(b_row, last)
            sa = a_scale.load_step(
                a_scale_lds,
                last,
            )
        acc = mfma.call_pipe_am_final(
            final_a_load,
            b_prev,
            acc,
            sa,
            sb,
        )
    else:
        if const_expr(async_a_copy):
            a_gather.prefetch_to_lds(
                fx.Int32(0),
                a_buf,
                fx.Int32(0),
            )
        else:
            a_gather.store(
                a_buf,
                a_gather.load_regs(fx.Int32(0)),
                fx.Int32(0),
            )
        a_scale.stage(a_scale_lds, tile_input_base)
        wait_lds_barrier(0 if async_a_copy else 63)
        init = [mfma.zero_value for _ in range(N_ACC)]
        for sp_i, state in range(0, K_ITERS, 1, init=init):
            sp = fx.Int32(sp_i)
            acc = [Vec(a) for a in state]
            cur_off = (sp & 1) * fx.Int32(a_lds_i32)
            nxt_off = ((sp + 1) & 1) * fx.Int32(a_lds_i32)
            spn = (sp + 1 < last).select(
                sp + 1,
                last,
            )

            def a_load(mi, ks, _base=cur_off):
                return a_s2r.load_operand(a_buf, mi, ks, _base)

            b = b_loader.load_step(b_row, sp)
            sa = a_scale.load_step(a_scale_lds, sp)
            sb = b_scale.load_step(b_row, sp)
            if const_expr(async_a_copy):
                rocdl.sched_barrier(0)
                a_gather.prefetch_to_lds(
                    spn * fx.Int32(A_K_STEP_BYTES),
                    a_buf,
                    nxt_off,
                )
                rocdl.sched_barrier(0)
            else:
                a_regs = a_gather.load_regs(
                    spn * fx.Int32(A_K_STEP_BYTES)
                )
            acc = mfma.call(
                a_load,
                b,
                acc,
                sa,
                sb,
            )
            if const_expr(async_a_copy):
                wait_lds_barrier(0)
            else:
                a_gather.store(a_buf, a_regs, nxt_off)
                wait_lds_barrier()
            state = yield list(acc)
        acc = [Vec(r) for r in state]
    # The epilogue aliases A_buf as cshuffle LDS after every wave finishes its final A ds_read.
    wait_lds_barrier()
    epi.store(acc, m_tile, tile_row_base, n_tile_base)


def _kstep_b_early(mfma, a_load, b_prev, acc, sa, sb, b_issue):
    """One A-major K step with A ds_read lookahead 1.

    ``b_issue()`` issues every B pack of the next K step up front, ksub-major
    (the order the next step consumes them; vmcnt retires in order), and
    returns them; None for the last step.
    """
    nn = mfma._num_acc_n
    groups = [(mi, ks) for mi in range(mfma._m_repeat) for ks in range(_PACK)]
    ngrp = len(groups)
    a_cur = a_load(*groups[0])
    b_next = b_issue() if b_issue is not None else None
    for g in range_constexpr(ngrp):
        mi, ksub = groups[g]
        a_nxt = a_load(*groups[g + 1]) if g + 1 < ngrp else None
        for ni in range_constexpr(nn):
            aidx = mfma.idx(mi, ni)
            acc[aidx] = mfma._mfma(
                a_cur, b_prev[ni][ksub], acc[aidx], sa[mi // _PACK],
                sb[ni // _PACK], ksub, mi % _PACK, ni % _PACK,
            )
        a_cur = a_nxt
    return acc, b_next


def _b_issuer(b_loader, b_row, num_acc_n, kstep):
    def _issue():
        out = [[None] * _PACK for _ in range(num_acc_n)]
        for ks in range_constexpr(_PACK):
            for ni in range_constexpr(num_acc_n):
                out[ni][ks] = b_loader._load_pack(b_row, ni, kstep, ks)
        return out

    return _issue


# fmt: off
@flyc.jit
def do_tile_prefill(m_tile, n_tile_base, expert, sched, a_gather, a_s2r, b_loader, b_scale, a_scale, mfma, epi,
    a_buf, a_scale_lds, a_lds_i32, K_ITERS, M_REPEAT, NUM_ACC_N, A_K_STEP_BYTES, trb_rsrc, tib_rsrc,
    indirect_input):
# fmt: on
    """128-row tile with LDS-DMA A (pipe_weights, A-major MFMA order).

    Per K step: the next step's A copy, B scale and A scale are issued
    first, then every next-step B pack, then the MFMAs.  The K loop is
    unrolled by two so the B double buffer alternates register sets (no
    loop-carried copies and no vmcnt(0) at the loop top), and the ping-pong
    A offsets go through ``opaque_sgpr`` so their addresses are not hoisted
    and spilled.
    """
    N_ACC = M_REPEAT * NUM_ACC_N
    NUM_B_SCALE = NUM_ACC_N // _PACK
    NUM_A_SCALE = M_REPEAT // _PACK
    NB = NUM_ACC_N * _PACK
    SB0 = N_ACC + NB
    SA0 = SB0 + NUM_B_SCALE
    last_i = K_ITERS - 1
    tile_row_base = trb_rsrc[m_tile]
    tile_input_base = tile_row_base
    if const_expr(indirect_input):
        tile_input_base = tib_rsrc[m_tile]
    b_row = sched.gate_base_row(expert) + n_tile_base
    a_gather.for_tile(tile_input_base)
    a_gather.prefetch_to_lds(fx.Int32(0), a_buf, fx.Int32(0))
    a_scale.stage(a_scale_lds, tile_input_base)
    wait_lds_barrier(0)

    def flat_b(bl):
        return [h for ni_list in bl for h in ni_list]

    def unpack(state):
        acc = [Vec(state[i]) for i in range(N_ACC)]
        bk = [[Vec(state[N_ACC + ni * _PACK + ks]) for ks in range(_PACK)]
              for ni in range(NUM_ACC_N)]
        sb = [fx.Int32(state[SB0 + g]) for g in range(NUM_B_SCALE)]
        sa = [fx.Int32(state[SA0 + g]) for g in range(NUM_A_SCALE)]
        return acc, bk, sb, sa

    def one_step(sp, cur_off, nxt_off, acc, b_cur, sb, sa, clamp=False):
        # K step sp reads ping-pong cur_off and prefetches step sp + 1 into
        # nxt_off.  With clamp the prefetch target may be K_ITERS; it is
        # clamped to the last step (a harmless re-copy into the idle half).
        spn = sp + fx.Int32(1)
        if const_expr(clamp):
            spn = (spn < fx.Int32(last_i)).select(spn, fx.Int32(last_i))

        def a_load(mi, ks, _base=cur_off):
            return a_s2r.load_operand(a_buf, mi, ks, _base)

        rocdl.sched_barrier(0)
        a_gather.prefetch_to_lds(spn * fx.Int32(A_K_STEP_BYTES), a_buf, nxt_off)
        rocdl.sched_barrier(0)
        sb_next = b_scale.load_step(b_row, spn)
        sa_next = a_scale.load_step(a_scale_lds, spn)
        acc, b_next = _kstep_b_early(
            mfma, a_load, b_cur, acc, sa, sb, _b_issuer(b_loader, b_row, NUM_ACC_N, spn)
        )
        wait_lds_barrier(NB + NUM_B_SCALE)
        return acc, b_next, sb_next, sa_next

    init = [mfma.zero_value for _ in range(N_ACC)]
    init += flat_b(b_loader.load_step(b_row, fx.Int32(0)))
    init += b_scale.load_step(b_row, fx.Int32(0))
    init += a_scale.load_step(a_scale_lds, fx.Int32(0))

    if const_expr(K_ITERS % 2 == 0):
        # Every K step runs inside the unrolled loop; the last step's
        # prefetch is clamped and drained before the epilogue.
        for pi, state in range(0, K_ITERS // 2, 1, init=init):
            acc, b_cur, sb, sa = unpack(state)
            sp0 = fx.Int32(pi) * fx.Int32(2)
            off0 = opaque_sgpr(fx.Int32(0))
            off1 = opaque_sgpr(fx.Int32(a_lds_i32))
            acc, b_mid, sb, sa = one_step(sp0, off0, off1, acc, b_cur, sb, sa)
            acc, b_nxt, sb, sa = one_step(sp0 + fx.Int32(1), off1, off0, acc,
                                          b_mid, sb, sa, clamp=True)
            yv = list(acc) + flat_b(b_nxt) + list(sb) + list(sa)
            state = yield yv
        acc = [Vec(state[i]) for i in range(N_ACC)]
        # Drain the clamped prefetch too: the epilogue aliases A_buf as
        # cshuffle LDS.
        wait_lds_barrier(0)
    else:
        for pi, state in range(0, last_i // 2, 1, init=init):
            acc, b_cur, sb, sa = unpack(state)
            sp0 = fx.Int32(pi) * fx.Int32(2)
            off0 = opaque_sgpr(fx.Int32(0))
            off1 = opaque_sgpr(fx.Int32(a_lds_i32))
            acc, b_mid, sb, sa = one_step(sp0, off0, off1, acc, b_cur, sb, sa)
            acc, b_nxt, sb, sa = one_step(sp0 + fx.Int32(1), off1, off0, acc,
                                          b_mid, sb, sa)
            yv = list(acc) + flat_b(b_nxt) + list(sb) + list(sa)
            state = yield yv
        acc, b_cur, sb, sa = unpack(state)
        final_off = opaque_sgpr(fx.Int32((last_i % 2) * a_lds_i32))

        def final_a_load(mi, ks, _base=final_off):
            return a_s2r.load_operand(a_buf, mi, ks, _base)

        acc, _unused = _kstep_b_early(mfma, final_a_load, b_cur, acc, sa, sb, None)
        # The epilogue aliases A_buf as cshuffle LDS.
        wait_lds_barrier()
    epi.store(acc, m_tile, tile_row_base, n_tile_base)


# fmt: off
@flyc.jit
def do_tile_pd2(m_tile, n_tile_base, expert, sched, a_gather, a_s2r, b_loader, b_scale, a_scale, mfma, epi,
    a_buf, a_scale_lds, a_lds_i32, K_ITERS, M_REPEAT, NUM_ACC_N, A_K_STEP_BYTES, trb_rsrc, tib_rsrc,
    indirect_input, N_DMA):
# fmt: on
    """Small decode tile with prefetch distance 2.

    One HBM round trip per K step bounds a decode tile at prefetch distance
    1.  Here the B packs, the B scales and the A LDS-DMA run two K steps
    ahead, through a 3-stage A LDS ring (stage k % 3); the loop is unrolled
    by three so every B register set is fixed, and the two non-issuing tail
    steps are peeled.
    """
    N_ACC = M_REPEAT * NUM_ACC_N
    NUM_B_SCALE = NUM_ACC_N // _PACK
    NUM_A_SCALE = M_REPEAT // _PACK
    NB = NUM_ACC_N * _PACK
    B0 = N_ACC
    B1 = B0 + NB
    S0 = B1 + NB
    S1 = S0 + NUM_B_SCALE
    SA = S1 + NUM_B_SCALE
    tile_row_base = trb_rsrc[m_tile]
    tile_input_base = tile_row_base
    if const_expr(indirect_input):
        tile_input_base = tib_rsrc[m_tile]
    b_row = sched.gate_base_row(expert) + n_tile_base
    b_pro = [b_loader.load_step(b_row, fx.Int32(0)),
             b_loader.load_step(b_row, fx.Int32(1))]
    sb_pro = [b_scale.load_step(b_row, fx.Int32(0)),
              b_scale.load_step(b_row, fx.Int32(1))]
    a_gather.for_tile(tile_input_base)
    a_gather.prefetch_to_lds(fx.Int32(0), a_buf, fx.Int32(0))
    a_gather.prefetch_to_lds(fx.Int32(A_K_STEP_BYTES), a_buf, fx.Int32(a_lds_i32))
    a_scale.stage(a_scale_lds, tile_input_base)
    wait_lds_barrier(0)

    def flat_b(bl):
        return [h for ni_list in bl for h in ni_list]

    def unb(state, base):
        return [[Vec(state[base + ni * _PACK + ks]) for ks in range(_PACK)]
                for ni in range(NUM_ACC_N)]

    def one_step(sp, acc, b_k, sb_k, sa_k, issue, last=False):
        # K step sp (runtime); ring stage sp % 3.
        cur_off = (sp % fx.Int32(3)) * fx.Int32(a_lds_i32)

        def a_load(mi, ks, _base=cur_off):
            return a_s2r.load_operand(a_buf, mi, ks, _base)

        sa_next = None
        if const_expr(not last):
            sa_next = a_scale.load_step(a_scale_lds, sp + fx.Int32(1))
        sb_n2 = None
        b_iss = None
        if const_expr(issue):
            sp2 = sp + fx.Int32(2)
            rocdl.sched_barrier(0)
            nxt_off = (sp2 % fx.Int32(3)) * fx.Int32(a_lds_i32)
            a_gather.prefetch_to_lds(sp2 * fx.Int32(A_K_STEP_BYTES), a_buf, nxt_off)
            rocdl.sched_barrier(0)
            sb_n2 = b_scale.load_step(b_row, sp2)
            b_iss = _b_issuer(b_loader, b_row, NUM_ACC_N, sp2)
        acc, b_n2 = _kstep_b_early(mfma, a_load, b_k, acc, sa_k, sb_k, b_iss)
        if const_expr(not last):
            if const_expr(issue):
                wait_lds_barrier(N_DMA + NUM_B_SCALE + NB)
            else:
                wait_lds_barrier(0)
        return acc, b_n2, sb_n2, sa_next

    init = [mfma.zero_value for _ in range(N_ACC)]
    init += flat_b(b_pro[0]) + flat_b(b_pro[1])
    init += list(sb_pro[0]) + list(sb_pro[1])
    init += a_scale.load_step(a_scale_lds, fx.Int32(0))

    n_issue = K_ITERS - 2
    n_trips = n_issue // 3
    for ti, state in range(0, n_trips, 1, init=init):
        acc = [Vec(state[i]) for i in range(N_ACC)]
        b0 = unb(state, B0)
        b1 = unb(state, B1)
        sb0 = [fx.Int32(state[S0 + g]) for g in range(NUM_B_SCALE)]
        sb1 = [fx.Int32(state[S1 + g]) for g in range(NUM_B_SCALE)]
        sa0 = [fx.Int32(state[SA + g]) for g in range(NUM_A_SCALE)]
        sp0 = fx.Int32(ti) * fx.Int32(3)
        acc, b2, sb2, sa1 = one_step(sp0, acc, b0, sb0, sa0, True)
        acc, b3, sb3, sa2 = one_step(sp0 + fx.Int32(1), acc, b1, sb1, sa1, True)
        acc, b4, sb4, sa3 = one_step(sp0 + fx.Int32(2), acc, b2, sb2, sa2, True)
        yv = list(acc) + flat_b(b3) + flat_b(b4) + list(sb3) + list(sb4) + list(sa3)
        state = yield yv
    acc = [Vec(state[i]) for i in range(N_ACC)]
    bA = unb(state, B0)
    bB = unb(state, B1)
    sbA = [fx.Int32(state[S0 + g]) for g in range(NUM_B_SCALE)]
    sbB = [fx.Int32(state[S1 + g]) for g in range(NUM_B_SCALE)]
    sa = [fx.Int32(state[SA + g]) for g in range(NUM_A_SCALE)]
    sp = n_trips * 3
    for _r in range_constexpr(n_issue % 3):
        acc, bC, sbC, sa = one_step(fx.Int32(sp), acc, bA, sbA, sa, True)
        bA, bB, sbA, sbB = bB, bC, sbB, sbC
        sp += 1
    acc, _u, _v, sa = one_step(fx.Int32(sp), acc, bA, sbA, sa, False)
    acc, _u, _v, _w = one_step(fx.Int32(sp + 1), acc, bB, sbB, sa, False, last=True)
    # The epilogue aliases A_buf as cshuffle LDS.
    wait_lds_barrier()
    epi.store(acc, m_tile, tile_row_base, n_tile_base)


def _tuned_prefill_tile(*, tuned, sort_block_m, tile_n, num_waves, pipe_weights, mfma_amajor,
                        async_a_copy, use_tile_resource, indexed_input, a_dtype, out_dtype, model_dim):
    return bool(
        tuned and sort_block_m == 128 and tile_n == 512 and num_waves == 8
        and pipe_weights and mfma_amajor and async_a_copy and use_tile_resource
        and not indexed_input and a_dtype == "fp8" and out_dtype == "fp8"
        and (model_dim // 32) % 16 == 0
    )


def _tuned_pd2_tile(*, tuned, sort_block_m, pipe_weights, async_a_copy, indexed_input, a_dtype,
                    total_threads):
    # The register-staged A variant (the smallest decode buckets) runs with
    # LDS-DMA A and prefetch distance 2 instead.
    return bool(
        tuned and sort_block_m == 32 and pipe_weights and not async_a_copy
        and not indexed_input and a_dtype == "fp8"
        and (sort_block_m * 16) % total_threads == 0
    )


def gemm1_a_lds_stages(*, tuned, sort_block_m, pipe_weights, async_a_copy, indexed_input,
                       a_dtype, total_threads):
    """Number of A-tile LDS stages build_fused_gemm1 needs for this variant."""
    pd2 = _tuned_pd2_tile(
        tuned=tuned, sort_block_m=sort_block_m, pipe_weights=pipe_weights,
        async_a_copy=async_a_copy, indexed_input=indexed_input, a_dtype=a_dtype,
        total_threads=total_threads,
    )
    return 3 if pd2 else 2


# fmt: off
def build_fused_gemm1(*, x_tensor, w_rsrc, sw_rsrc, sx_rsrc,
    out_rsrc, os_rsrc, trb_rsrc, tib_rsrc, expert_rsrc, out_tensor, a_buf, a_scale_lds, c_tile,
    model_dim, inter_dim, sort_block_m, tile_n, num_waves, n_per_wave, wave_id,
    m_repeat, num_acc_n, a_k_step_bytes, total_threads, k_iters, a_lds_i32, n_tiles,
    expert_offset, b_cache_modifier, swizzle_a, pipe_weights, mfma_amajor, async_a_copy,
    use_tile_resource, indirect_input, indexed_input=False, row_map_rsrc=None,
    source_rows=0, a_dtype="fp8", out_dtype="fp8", swiglu_limit=0.0, pair_k=False, evec,
    act="silu", situ_beta=1.0, situ_linear_beta=1.0, tuned=False):
    # fmt: on
    """Build the GEMM1 atoms and return its expert resolver and tile runner.

    ``tuned`` (compact Stage1) selects the tuned tile bodies where they apply:
    the 128-row prefill tile (do_tile_prefill), the register-staged decode
    tile as an LDS-DMA prefetch-distance-2 tile (do_tile_pd2), and a
    wave-uniform A loader for the other LDS-DMA variants.  Outputs are
    unchanged.
    """
    sched = TileScheduler(
        expert_rsrc=expert_rsrc,
        inter_dim=inter_dim,
        expert_offset=expert_offset,  # GLOBAL sorted_expert_id -> LOCAL w1 index
    )
    n_wave_base = wave_id * fx.Int32(n_per_wave)
    prefill_tile = _tuned_prefill_tile(
        tuned=tuned, sort_block_m=sort_block_m, tile_n=tile_n, num_waves=num_waves,
        pipe_weights=pipe_weights, mfma_amajor=mfma_amajor, async_a_copy=async_a_copy,
        use_tile_resource=use_tile_resource, indexed_input=indexed_input, a_dtype=a_dtype,
        out_dtype=out_dtype, model_dim=model_dim,
    )
    pd2_tile = _tuned_pd2_tile(
        tuned=tuned, sort_block_m=sort_block_m, pipe_weights=pipe_weights,
        async_a_copy=async_a_copy, indexed_input=indexed_input, a_dtype=a_dtype,
        total_threads=total_threads,
    )
    uniform_a = bool(tuned and not indexed_input and a_dtype == "fp8"
                     and (async_a_copy or pd2_tile))
    lds_scopes = make_lds_dma_scopes() if (prefill_tile or pd2_tile) else None

    # fmt: off
    a_pack = 2 if a_dtype == "fp4" else 1
    a_loader_cls = UniformATileLoader if uniform_a else ATileLoader
    a_gather = a_loader_cls(row_bytes=model_dim // a_pack, sort_block_m=sort_block_m,
        k_step_bytes=a_k_step_bytes, total_threads=total_threads, swizzle=swizzle_a,
        x_tensor=x_tensor, async_copy=async_a_copy or pd2_tile,
        indexed_input=indexed_input, row_map_rsrc=row_map_rsrc,
        source_rows=source_rows, a_dtype=a_dtype)
    # fmt: on
    if lds_scopes is not None:
        a_gather.lds_scopes = lds_scopes
        a_s2r = ScopedAS2RLoader(k_step_bytes=a_k_step_bytes, swizzle=swizzle_a,
                                 a_dtype=a_dtype, lds_scopes=lds_scopes)
    else:
        a_s2r = AS2RLoader(k_step_bytes=a_k_step_bytes, swizzle=swizzle_a, a_dtype=a_dtype)
    b_loader = BWeightLoader(
        w_rsrc=w_rsrc,
        num_acc_n=num_acc_n,
        model_dim=model_dim,
        cache_modifier=b_cache_modifier,
    )
    b_scale = BScaleLoader(scale_rsrc=sw_rsrc, num_acc_n=num_acc_n, model_dim=model_dim)
    if prefill_tile:
        a_scale = PackedAScaleLoader(
            scale_rsrc=sx_rsrc,
            m_repeat=m_repeat,
            model_dim=model_dim,
            sort_block_m=sort_block_m,
            total_threads=total_threads,
            indexed_input=indexed_input,
            row_map_rsrc=row_map_rsrc,
            lds_scopes=lds_scopes,
        )
    elif pd2_tile:
        a_scale = ScopedAScaleLoader(
            scale_rsrc=sx_rsrc,
            m_repeat=m_repeat,
            model_dim=model_dim,
            sort_block_m=sort_block_m,
            total_threads=total_threads,
            indexed_input=indexed_input,
            row_map_rsrc=row_map_rsrc,
            lds_scopes=lds_scopes,
        )
    else:
        a_scale = AScaleLoader(
            scale_rsrc=sx_rsrc,
            m_repeat=m_repeat,
            model_dim=model_dim,
            sort_block_m=sort_block_m,
            total_threads=total_threads,
            indexed_input=indexed_input,
            row_map_rsrc=row_map_rsrc,
        )
    mfma = MfmaScaleGU(m_repeat=m_repeat, num_acc_n=num_acc_n, a_dtype=a_dtype)
    epi_cls = SwizzledQuantEpilogue if prefill_tile else SiluQuantEpilogue
    # fmt: off
    epi = epi_cls(out_rsrc=out_rsrc, out_scale_rsrc=os_rsrc, sorted_rsrc=trb_rsrc, tokens=0,
        inter_dim=inter_dim, m_repeat=m_repeat, num_acc_n=num_acc_n, sort_block_m=sort_block_m, tile_n=tile_n,
        num_waves=num_waves, lds_out=c_tile, swiglu_limit=swiglu_limit, always_valid=True,
        out_tensor=out_tensor if use_tile_resource else None, evec=evec, out_dtype=out_dtype,
        act=act, situ_beta=situ_beta, situ_linear_beta=situ_linear_beta)
    # fmt: on

    def _decode(flat):
        m_tile = flat // fx.Int32(n_tiles)
        n_tile = flat - m_tile * fx.Int32(n_tiles)
        return m_tile, n_tile

    def expert_of_flat(flat):
        m_tile, _n = _decode(flat)
        return sched.expert_of(m_tile)

    def do_scheduled_tile(flat):
        m_tile, n_tile = _decode(flat)
        n_tile_base = n_wave_base + n_tile * fx.Int32(tile_n)
        expert = sched.expert_of(m_tile)
        # fmt: off
        if prefill_tile:
            do_tile_prefill(m_tile, n_tile_base, expert, sched, a_gather,
                a_s2r, b_loader, b_scale, a_scale, mfma, epi, a_buf,
                a_scale_lds, a_lds_i32, k_iters, m_repeat, num_acc_n,
                a_k_step_bytes, trb_rsrc, tib_rsrc, indirect_input)
        elif pd2_tile:
            do_tile_pd2(m_tile, n_tile_base, expert, sched, a_gather,
                a_s2r, b_loader, b_scale, a_scale, mfma, epi, a_buf,
                a_scale_lds, a_lds_i32, k_iters, m_repeat, num_acc_n,
                a_k_step_bytes, trb_rsrc, tib_rsrc, indirect_input,
                (sort_block_m * 16) // total_threads)
        else:
            do_tile(m_tile, n_tile_base, expert, sched, a_gather,
                a_s2r, b_loader, b_scale, a_scale, mfma, epi, a_buf,
                a_scale_lds, a_lds_i32, k_iters, m_repeat, num_acc_n,
                a_k_step_bytes, pipe_weights, mfma_amajor, async_a_copy,
                trb_rsrc, tib_rsrc, indirect_input, pair_k)
        # fmt: on

    return expert_of_flat, do_scheduled_tile


# fmt: off
@functools.cache
def compile_gemm1(
    *, model_dim: int, inter_dim: int, expert_offset: int = 0, sort_block_m: int = 32,
    tile_n: int = 256, tile_k: int = 256, num_waves: int = 4, pipe_weights: bool = True,
    mfma_amajor: bool = False, swizzle_a: bool = True, async_a_copy: bool = False,
    use_tile_resource: bool = True, waves_per_eu_hint: int = 2, b_cache_modifier: int = 0,
    a_dtype: str = "fp8", out_dtype: str = "fp8", swiglu_limit: float = 0.0,
    act: str = "silu", situ_beta: float = 1.0, situ_linear_beta: float = 1.0,
):
    # fmt: on
    """Compile standalone group GEMM1 from the fused Stage1 compute body."""
    num_waves = int(num_waves)
    assert num_waves > 1
    assert 1 <= waves_per_eu_hint <= 4
    assert tile_n % num_waves == 0
    assert (2 * inter_dim) % tile_n == 0
    assert tile_k == 256 and model_dim % tile_k == 0
    assert a_dtype in ("fp4", "fp8")
    assert out_dtype in ("fp4", "fp8")

    n_per_wave = tile_n // num_waves
    n_tiles = (2 * inter_dim) // tile_n
    m_repeat = sort_block_m // 16
    num_acc_n = n_per_wave // 16
    assert num_acc_n % 2 == 0 and m_repeat % 2 == 0

    a_k_step_bytes = tile_k // (2 if a_dtype == "fp4" else 1)
    out_row_bytes = inter_dim // (2 if out_dtype == "fp4" else 1)
    k_iters = model_dim // tile_k
    total_threads = num_waves * 64
    a_lds_size = sort_block_m * a_k_step_bytes
    a_lds_i32 = a_lds_size // 4
    cs_tile_n = tile_n // 2
    lds_pool_bytes = max(2 * a_lds_size, sort_block_m * cs_tile_n * 4)
    n_scale_bytes = sort_block_m * (model_dim // 32)

    @fx.struct
    class SharedStorage:
        pool: fx.Array[fx.Int8, lds_pool_bytes, 16]
        A_scale: fx.Array[fx.Int8, n_scale_bytes, 16]

    @flyc.kernel(known_block_size=[total_threads, 1, 1])
    def kernel(
        out: fx.Tensor, x: fx.Tensor, w: fx.Tensor, scale_x: fx.Tensor, scale_w: fx.Tensor,
        tile_row_base: fx.Tensor, expert_ids: fx.Tensor, out_scale: fx.Tensor, num_valid: fx.Int32,
        grid_x: fx.Int32,
    ):
        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        a_buf = lds.pool
        a_scale_lds = lds.A_scale
        c_tile = _LdsF32View(fx.recast_iter(fx.Float32, lds.pool.ptr))

        w_rsrc = ptr_buf_tensor(fx.get_iter(w), fx.Int32, unit_elems=4)
        sx_rsrc = ptr_buf_tensor(fx.get_iter(scale_x), fx.Int32, unit_elems=4)
        sw_rsrc = ptr_buf_tensor(fx.get_iter(scale_w), fx.Int32)
        trb_rsrc = ptr_buf_tensor(fx.get_iter(tile_row_base), fx.Int32)
        expert_rsrc = ptr_buf_tensor(fx.get_iter(expert_ids), fx.Int32)
        if const_expr(use_tile_resource):
            out_rsrc = None
        else:
            out_elem = fx.Int8 if const_expr(out_dtype == "fp4") else fx.Int16
            out_rsrc = ptr_buf_tensor(
                fx.get_iter(out),
                out_elem,
                num_records_bytes=num_valid * fx.Int32(out_row_bytes),
            )
        scale_cols = (inter_dim // 32 + 7) // 8 * 8
        os_rsrc = ptr_buf_tensor(
            fx.get_iter(out_scale),
            fx.Int8,
            num_records_bytes=num_valid * fx.Int32(scale_cols) + 8192,
        )
        wave_id = fx.thread_idx.x // 64

        _, run_tile = build_fused_gemm1(
            x_tensor=x, w_rsrc=w_rsrc, sw_rsrc=sw_rsrc,
            sx_rsrc=sx_rsrc, out_rsrc=out_rsrc, os_rsrc=os_rsrc, trb_rsrc=trb_rsrc,
            tib_rsrc=trb_rsrc,
            expert_rsrc=expert_rsrc, out_tensor=out, a_buf=a_buf,
            a_scale_lds=a_scale_lds, c_tile=c_tile, model_dim=model_dim, inter_dim=inter_dim,
            sort_block_m=sort_block_m, tile_n=tile_n, num_waves=num_waves, n_per_wave=n_per_wave,
            wave_id=wave_id, m_repeat=m_repeat, num_acc_n=num_acc_n, a_k_step_bytes=a_k_step_bytes,
            total_threads=total_threads, k_iters=k_iters, a_lds_i32=a_lds_i32, n_tiles=n_tiles,
            expert_offset=expert_offset, b_cache_modifier=b_cache_modifier, swizzle_a=swizzle_a,
            pipe_weights=pipe_weights, mfma_amajor=mfma_amajor, async_a_copy=async_a_copy,
            use_tile_resource=use_tile_resource, indirect_input=False,
            a_dtype=a_dtype, out_dtype=out_dtype,
            swiglu_limit=swiglu_limit, evec=8 if use_tile_resource else 2,
            act=act, situ_beta=situ_beta, situ_linear_beta=situ_linear_beta,

        )
        total_work = (num_valid // fx.Int32(sort_block_m)) * fx.Int32(n_tiles)
        for flat in range(fx.block_idx.x, total_work, grid_x):
            run_tile(flat)

    @flyc.jit
    def launch(
        out: fx.Tensor, x: fx.Tensor, w: fx.Tensor, scale_x: fx.Tensor, scale_w: fx.Tensor,
        tile_row_base: fx.Tensor, expert_ids: fx.Tensor, out_scale: fx.Tensor, num_valid: fx.Int32,
        grid_x: fx.Int32, stream: fx.Stream,
    ):
        kernel(
            out, x, w, scale_x, scale_w, tile_row_base, expert_ids, out_scale, num_valid, grid_x,
            value_attrs={
                "rocdl.waves_per_eu": waves_per_eu_hint,
                "rocdl.flat_work_group_size": f"{total_threads},{total_threads}",
            },
        ).launch(grid=(fx.Int64(grid_x), 1, 1), block=(total_threads, 1, 1), stream=stream)

    return launch


# fmt: off
def gemm1_kernel(
    out, x, w, scale_x, scale_w, tile_row_base, expert_ids, out_scale, num_valid, stream, *,
    model_dim: int, inter_dim: int, expert_offset: int = 0, sort_block_m: int = 32,
    tile_n: int = 256, tile_k: int = 256, num_waves: int = 4, grid_mult: int = 4,
    pipe_weights: bool = True, mfma_amajor: bool = False, swizzle_a: bool = True,
    async_a_copy: bool = False, use_tile_resource: bool = True, waves_per_eu_hint: int = 2,
    num_cu: int = 256, b_cache_modifier: int = 0, a_dtype: str = "fp8", out_dtype: str = "fp8",
    swiglu_limit: float = 0.0, act: str = "silu", situ_beta: float = 1.0, situ_linear_beta: float = 1.0,
):
    # fmt: on
    """Run standalone MegaMoEV2 group GEMM1 and return ``(out, out_scale)``."""
    num_valid = int(num_valid)
    if num_valid < 0 or num_valid % int(sort_block_m):
        raise ValueError("num_valid must be a non-negative multiple of sort_block_m")
    if num_valid == 0:
        return out, out_scale
    n_tiles = (2 * int(inter_dim)) // int(tile_n)
    total_work = (num_valid // int(sort_block_m)) * n_tiles
    grid_x = min(total_work, int(num_cu) * int(grid_mult))
    launch = compile_gemm1(
        model_dim=model_dim, inter_dim=inter_dim, expert_offset=expert_offset,
        sort_block_m=sort_block_m, tile_n=tile_n, tile_k=tile_k, num_waves=num_waves,
        pipe_weights=pipe_weights, mfma_amajor=mfma_amajor, swizzle_a=swizzle_a,
        async_a_copy=async_a_copy, use_tile_resource=use_tile_resource,
        waves_per_eu_hint=waves_per_eu_hint, b_cache_modifier=b_cache_modifier,
        a_dtype=a_dtype, out_dtype=out_dtype,
        swiglu_limit=swiglu_limit, act=act, situ_beta=situ_beta, situ_linear_beta=situ_linear_beta,
    )
    _run_compiled(
        launch, out, x, w.view(torch.uint8), scale_x, scale_w.view(torch.uint8),
        tile_row_base, expert_ids, out_scale,
        fx.Int32(num_valid), fx.Int32(grid_x), stream,
    )
    return out, out_scale
