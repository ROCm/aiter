# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
"""GEMM1 compute shared by fused MegaMoE v2 stage1 and its standalone interface."""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl.expr import const_expr, range_constexpr, rocdl
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec
from flydsl.expr.typing import as_ir_value as _raw

from ..tensor_shim import _run_compiled, ptr_buf_tensor
from .gemm_util import (
    _PACK,
    AS2RLoader,
    AScaleLoader,
    ATileLoader,
    BScaleLoader,
    BWeightLoader,
    MfmaScaleGU,
    SiluQuantEpilogue,
    TileScheduler,
    wait_lds_barrier,
)


class _LdsF32View:
    def __init__(self, ptr):
        self.ptr = ptr


# fmt: off
@flyc.jit
def do_tile(m_tile, n_tile_base, expert, sched, a_gather, a_s2r, b_loader, b_scale, a_scale, mfma, epi, a_buf,
    a_scale_lds, a_lds_i32, K_ITERS, M_REPEAT, NUM_ACC_N, A_K_STEP_BYTES, pipe_weights,
    mfma_amajor, async_a_copy, trb_rsrc, tib_rsrc, indirect_input, pair_k, narrow_final_a, tile_zero=False,
    row_map=None):
# fmt: on
    N_ACC = M_REPEAT * NUM_ACC_N
    NUM_B_SCALE = NUM_ACC_N // _PACK
    # A folded tile (M_REPEAT odd) stages a whole last scale group.
    NUM_A_SCALE = (M_REPEAT + 1) // _PACK
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
    if const_expr(row_map is not None):
        # The rows sit in the token-row region; row_map["wait"] returns once every row of the tile landed.
        assert not indirect_input
        a_srcs = a_gather.load_row_map(tile_row_base, row_map["srcmap"])
        s_srcs = a_scale.load_row_map(tile_row_base, row_map["srcmap"])
        row_map["wait"](a_srcs)
        a_gather.for_tile(tile_row_base, None, row_map, a_srcs)
    else:
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
        if const_expr(row_map is not None):
            a_scale.stage(a_scale_lds, tile_row_base, row_map, None, s_srcs)
        else:
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
                return a_s2r.load_operand_in_flight(a_buf, mi, ks, _base)

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
        if const_expr(tile_zero):
            # m_tile >= 0, so this adds 0, but the offset is opaque per tile: without it every tail A ds_read
            # address folds to a loop invariant, is hoisted to the preheader and spilled at the VGPR cap.
            final_off = final_off + (m_tile < fx.Int32(0)).select(m_tile, fx.Int32(0))

        def final_a_load(mi, ks, _base=final_off):
            return a_s2r.load_operand(a_buf, mi, ks, _base, narrow_copy=narrow_final_a)

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
        if const_expr(row_map is not None):
            a_scale.stage(a_scale_lds, tile_row_base, row_map, None, s_srcs)
        else:
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


# fmt: off
@flyc.jit
def do_tile_la(m_tile, n_tile_base, expert, sched, a_gather, a_s2r, b_loader, b_scale, a_scale, mfma, epi, a_buf,
    a_scale_lds, a_lds_i32, K_ITERS, M_REPEAT, NUM_ACC_N, A_K_STEP_BYTES, trb_rsrc, rows_rsrc, LA, row_map=None,
    meta=None, a_ahead=1):
# fmt: on
    """pipe_weights K-loop with B / B-scale issued LA K-steps ahead, unrolled xLA.

    The baseline x1 loop yields freshly loaded B registers that still interfere with the B registers feeding the
    MFMAs, so LLVM emits phi copies and drains the next-step loads before the back-edge: no real lookahead, one
    full HBM latency per K-step. Unrolling by LA rotates the LA-deep register queue back to identity each
    iteration, so the yielded sets coalesce with the loop arguments and loads stay in flight across the back-edge.
    Per-accumulator MFMA order (ksub 0 then 1, K-steps ascending) is unchanged -> bit-identical output.
    B-scale loads and epilogue stores are nt; the epilogue stores are clamped to the tile's valid rows.
    ``meta = (tile_row_base, valid_rows)`` replaces their loads (the caller's own plan); ``row_map`` as
    :func:`do_tile`, without the arrival wait.
    a_ahead > 1: the A rows of K-step k + a_ahead are loaded into registers at step k and written to LDS at step
    k + a_ahead - 1. With a_ahead=1 the end-of-step LDS write waits for a load issued at the start of the same step.
    """
    assert 1 <= a_ahead < K_ITERS
    N_ACC = M_REPEAT * NUM_ACC_N
    NUM_B = NUM_ACC_N * _PACK
    NUM_SB = NUM_ACC_N // _PACK
    sb_cm = 3
    valid_rows = None
    if const_expr(meta is not None):
        tile_row_base, valid_rows = meta
    else:
        tile_row_base = fx.Int32(rocdl.readfirstlane(T.i32, trb_rsrc[m_tile]))
    b_row = sched.gate_base_row(expert) + n_tile_base
    if const_expr(meta is None and rows_rsrc is not None):
        valid_rows = fx.Int32(rocdl.readfirstlane(T.i32, rows_rsrc[m_tile]))
    if const_expr(row_map is not None):
        a_srcs = a_gather.load_row_map(tile_row_base, row_map["srcmap"])
        s_srcs = a_scale.load_row_map(tile_row_base, row_map["srcmap"])
        a_gather.for_tile(tile_row_base, valid_rows, row_map, a_srcs)
        a_gather.store(a_buf, a_gather.load_regs(fx.Int32(0)), fx.Int32(0))
        a_scale.stage(a_scale_lds, tile_row_base, row_map, valid_rows, s_srcs)
    else:
        a_gather.for_tile(tile_row_base, valid_rows)
        a_gather.store(a_buf, a_gather.load_regs(fx.Int32(0)), fx.Int32(0))
        a_scale.stage(a_scale_lds, tile_row_base)
    wait_lds_barrier(63)
    init = [mfma.zero_value for _ in range(N_ACC)]
    for q in range_constexpr(LA):
        init += [h for ni_list in b_loader.load_step(b_row, fx.Int32(q)) for h in ni_list]
    for q in range_constexpr(LA):
        init += b_scale.load_step(b_row, fx.Int32(q), sb_cm)
    a_lds_bytes = [lb for lb, _c in a_gather._chunks]
    # a_ahead > 1: A register sets of K-steps k+1 .. k+a_ahead-1 (only in the unrolled K-loop).
    aq = [[v for _lb, v in a_gather.load_regs(fx.Int32(j * A_K_STEP_BYTES))] for j in range_constexpr(1, a_ahead)]

    def unpack(state):
        acc = [Vec(a) for a in state[:N_ACC]]
        bq = []
        for q in range_constexpr(LA):
            base = N_ACC + q * NUM_B
            bq.append([[Vec(state[base + ni * _PACK + ks]) for ks in range(_PACK)] for ni in range(NUM_ACC_N)])
        sbase = N_ACC + LA * NUM_B
        sbq = [[fx.Int32(state[sbase + q * NUM_SB + g]) for g in range(NUM_SB)] for q in range(LA)]
        return acc, bq, sbq

    def pack(acc, bq, sbq):
        out = list(acc)
        for b in bq:
            out += [h for ni_list in b for h in ni_list]
        for sb in sbq:
            out += list(sb)
        return out

    def step(sp, acc, bq, sbq, issue, a_next, a_issue=False):
        cur_off = (sp & fx.Int32(1)) * fx.Int32(a_lds_i32)
        nxt_off = ((sp + fx.Int32(1)) & fx.Int32(1)) * fx.Int32(a_lds_i32)
        kl = sp + fx.Int32(LA)
        if const_expr(a_issue):
            aq.append([v for _lb, v in a_gather.load_regs((sp + fx.Int32(a_ahead)) * fx.Int32(A_K_STEP_BYTES))])
        sb_new = b_scale.load_step(b_row, kl, sb_cm) if issue else None
        sa = a_scale.load_step(a_scale_lds, sp)
        a_regs = None
        if const_expr(a_next):
            if const_expr(a_ahead > 1):
                # Opaque to LLVM, which would otherwise fold the load + LDS write into an LDS DMA issued here.
                v4 = ir.VectorType.get([4], ir.IntegerType.get_signless(32))
                a_regs = [Vec(llvm.inline_asm(v4, [_raw(v)], "", "=v,0", has_side_effects=True)) for v in aq.pop(0)]
            else:
                a_regs = [v for _lb, v in a_gather.load_regs((sp + fx.Int32(1)) * fx.Int32(A_K_STEP_BYTES))]

        def a_load(mi, ks, _base=cur_off):
            return a_s2r.load_operand(a_buf, mi, ks, _base)

        def load_next(ni, _kn=kl):
            return b_loader.load_ni(b_row, ni, _kn)

        if const_expr(issue):
            acc, b_new = mfma.call_pipe(a_load, bq[0], acc, sa, sbq[0], load_next)
            bq = bq[1:] + [b_new]
            sbq = sbq[1:] + [sb_new]
        else:
            acc = mfma.call(a_load, bq[0], acc, sa, sbq[0])
            bq = bq[1:]
            sbq = sbq[1:]
        if const_expr(a_next):
            a_gather.store(a_buf, list(zip(a_lds_bytes, a_regs)), nxt_off)
            wait_lds_barrier()
        return acc, bq, sbq

    # a_ahead > 1 unrolls every K-step: a loop would rotate the register queues through phi copies, which wait for
    # the loads still in flight.
    n_loop = 0 if a_ahead > 1 else (K_ITERS - LA) // LA
    state = init
    if const_expr(n_loop > 0):
        for it_i, state in range(0, n_loop, 1, init=init):
            sp0 = fx.Int32(it_i) * fx.Int32(LA)
            acc, bq, sbq = unpack(state)
            for u in range_constexpr(LA):
                acc, bq, sbq = step(sp0 + fx.Int32(u), acc, bq, sbq, True, True)
            state = yield pack(acc, bq, sbq)
    acc, bq, sbq = unpack(state)
    for k in range_constexpr(n_loop * LA, K_ITERS):
        acc, bq, sbq = step(fx.Int32(k), acc, bq, sbq, k + LA <= K_ITERS - 1, k < K_ITERS - 1,
            a_ahead > 1 and k + a_ahead <= K_ITERS - 1)
    wait_lds_barrier()
    epi.store(acc, m_tile, tile_row_base, n_tile_base, valid_rows, 3)


# fmt: off
def build_fused_gemm1(*, x_tensor, w_rsrc, sw_rsrc, sx_rsrc,
    out_rsrc, os_rsrc, trb_rsrc, tib_rsrc, expert_rsrc, out_tensor, a_buf, a_scale_lds, c_tile,
    model_dim, inter_dim, sort_block_m, tile_n, num_waves, n_per_wave, wave_id,
    m_repeat, num_acc_n, a_k_step_bytes, total_threads, k_iters, a_lds_i32, n_tiles,
    expert_offset, b_cache_modifier, swizzle_a, pipe_weights, mfma_amajor, async_a_copy,
    use_tile_resource, indirect_input, indexed_input=False, row_map_rsrc=None,
    source_rows=0, a_dtype="fp8", out_dtype="fp8", swiglu_limit=0.0, pair_k=False, evec,
    act="silu", situ_beta=1.0, situ_linear_beta=1.0, narrow_final_a=False, rows_rsrc=None, lookahead=0,
    out_cache_modifier=0, tile_zero=False, fused_v8=False, epi_lean=0, row_map=None, short_rows=0, fold_rows=0,
    a_asm_reads=False, ascale_packed=False, plan_lookup=None, a_ahead=1):
    # fmt: on
    """Build the GEMM1 atoms and return its expert resolver and tile runner.

    ``lookahead > 0`` runs tiles with :func:`do_tile_la`, clamping each tile to its ``rows_rsrc`` row count;
    ``out_cache_modifier`` is the epilogue store cache policy and ``tile_zero`` is forwarded to :func:`do_tile`.
    The remaining knobs serve the fused kernels: ``fused_v8`` / ``epi_lean`` pick the fused evec 8 epilogue
    (SiluQuantEpilogue), ``row_map`` reads A through the srcmap (:func:`do_tile`), ``a_asm_reads`` /
    ``ascale_packed`` are the A loaders' modes, ``plan_lookup(m_tile) -> (expert, tile_row_base, rows)`` takes
    the lookahead tiles' metadata from the CTA's own plan and ``a_ahead`` is :func:`do_tile_la`'s.

    short_rows > 0 also returns a runner for tiles whose rows past the first short_rows are all padding: it
    computes and stores only those leading rows (m_repeat short_rows // 16) and leaves the rest unwritten.
    fold_rows > 0 also returns a runner for an m-tile followed by a tail tile of its expert with at most
    fold_rows real rows: one sort_block_m + fold_rows row tile (the rows are contiguous) storing both.
    """
    assert row_map is None or lookahead > 0 or (async_a_copy and a_k_step_bytes == 256)
    sched = TileScheduler(
        expert_rsrc=expert_rsrc,
        inter_dim=inter_dim,
        expert_offset=expert_offset,  # GLOBAL sorted_expert_id -> LOCAL w1 index
    )
    n_wave_base = wave_id * fx.Int32(n_per_wave)

    # fmt: off
    a_pack = 2 if a_dtype == "fp4" else 1
    a_gather = ATileLoader(row_bytes=model_dim // a_pack, sort_block_m=sort_block_m,
        k_step_bytes=a_k_step_bytes, total_threads=total_threads, swizzle=swizzle_a,
        x_tensor=x_tensor, async_copy=async_a_copy,
        indexed_input=indexed_input, row_map_rsrc=row_map_rsrc,
        source_rows=source_rows, a_dtype=a_dtype)
    # fmt: on
    a_s2r = AS2RLoader(k_step_bytes=a_k_step_bytes, swizzle=swizzle_a, a_dtype=a_dtype, asm_reads=a_asm_reads)
    b_loader = BWeightLoader(
        w_rsrc=w_rsrc,
        num_acc_n=num_acc_n,
        model_dim=model_dim,
        cache_modifier=b_cache_modifier,
    )
    b_scale = BScaleLoader(scale_rsrc=sw_rsrc, num_acc_n=num_acc_n, model_dim=model_dim)
    a_scale = AScaleLoader(
        scale_rsrc=sx_rsrc,
        m_repeat=m_repeat,
        model_dim=model_dim,
        sort_block_m=sort_block_m,
        total_threads=total_threads,
        indexed_input=indexed_input,
        row_map_rsrc=row_map_rsrc,
        packed=ascale_packed,
    )
    mfma = MfmaScaleGU(m_repeat=m_repeat, num_acc_n=num_acc_n, a_dtype=a_dtype)

    def make_epilogue(e_repeat, tile_rows=None):
        # fmt: off
        return SiluQuantEpilogue(out_rsrc=out_rsrc, out_scale_rsrc=os_rsrc, sorted_rsrc=trb_rsrc, tokens=0,
            inter_dim=inter_dim, m_repeat=e_repeat, num_acc_n=num_acc_n, sort_block_m=sort_block_m, tile_n=tile_n,
            num_waves=num_waves, lds_out=c_tile, swiglu_limit=swiglu_limit, always_valid=True,
            out_tensor=out_tensor if use_tile_resource else None, evec=evec, out_dtype=out_dtype,
            act=act, situ_beta=situ_beta, situ_linear_beta=situ_linear_beta, out_cache_modifier=out_cache_modifier,
            tile_rows=tile_rows, lean=epi_lean, fused_v8=fused_v8)
        # fmt: on

    epi = make_epilogue(m_repeat)

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
        meta = None
        if const_expr(plan_lookup is not None):
            # (local expert, tile row base, valid rows) from the CTA's own plan.
            expert, trb, rows = plan_lookup(m_tile)
            meta = (trb, rows)
        else:
            expert = sched.expert_of(m_tile)
        # fmt: off
        if const_expr(lookahead > 0):
            assert pipe_weights and not async_a_copy and not mfma_amajor and not indirect_input
            do_tile_la(m_tile, n_tile_base, expert, sched, a_gather, a_s2r, b_loader, b_scale, a_scale, mfma, epi,
                a_buf, a_scale_lds, a_lds_i32, k_iters, m_repeat, num_acc_n, a_k_step_bytes, trb_rsrc, rows_rsrc,
                lookahead, row_map, meta, a_ahead)
            return
        assert plan_lookup is None
        do_tile(m_tile, n_tile_base, expert, sched, a_gather,
            a_s2r, b_loader, b_scale, a_scale, mfma, epi, a_buf,
            a_scale_lds, a_lds_i32, k_iters, m_repeat, num_acc_n,
            a_k_step_bytes, pipe_weights, mfma_amajor, async_a_copy,
            trb_rsrc, tib_rsrc, indirect_input, pair_k, narrow_final_a, tile_zero, row_map)
        # fmt: on

    if const_expr(not short_rows):
        return expert_of_flat, do_scheduled_tile
    assert lookahead == 0 and not indirect_input and sort_block_m % short_rows == 0 and a_dtype == "fp8"
    s_repeat = short_rows // 16
    # fmt: off
    a_gather_s = ATileLoader(row_bytes=model_dim, sort_block_m=short_rows, k_step_bytes=a_k_step_bytes,
        total_threads=total_threads, swizzle=swizzle_a, x_tensor=x_tensor, async_copy=async_a_copy)
    a_scale_s = AScaleLoader(scale_rsrc=sx_rsrc, m_repeat=s_repeat, model_dim=model_dim, sort_block_m=short_rows,
        total_threads=total_threads, packed=ascale_packed)
    # fmt: on
    mfma_s = MfmaScaleGU(m_repeat=s_repeat, num_acc_n=num_acc_n)
    epi_s = make_epilogue(s_repeat, short_rows)

    def do_short_tile(flat):
        m_tile, n_tile = _decode(flat)
        n_tile_base = n_wave_base + n_tile * fx.Int32(tile_n)
        expert = sched.expert_of(m_tile)
        # fmt: off
        do_tile(m_tile, n_tile_base, expert, sched, a_gather_s, a_s2r, b_loader, b_scale, a_scale_s, mfma_s, epi_s,
            a_buf, a_scale_lds, a_lds_i32, k_iters, s_repeat, num_acc_n, a_k_step_bytes, pipe_weights, mfma_amajor,
            async_a_copy, trb_rsrc, tib_rsrc, False, pair_k, narrow_final_a, tile_zero, row_map)
        # fmt: on

    if const_expr(not fold_rows):
        return expert_of_flat, do_scheduled_tile, do_short_tile
    assert async_a_copy and pipe_weights and fold_rows in (16, short_rows)
    f_repeat = (sort_block_m + fold_rows) // 16
    # A and its scales are staged in whole 32-row groups (rows past the tail are its padding).
    f_rows = sort_block_m + short_rows
    # fmt: off
    a_gather_f = ATileLoader(row_bytes=model_dim, sort_block_m=f_rows, k_step_bytes=a_k_step_bytes,
        total_threads=total_threads, swizzle=swizzle_a, x_tensor=x_tensor, async_copy=async_a_copy)
    a_scale_f = AScaleLoader(scale_rsrc=sx_rsrc, m_repeat=f_rows // 16, model_dim=model_dim, sort_block_m=f_rows,
        total_threads=total_threads, packed=ascale_packed)
    # fmt: on
    mfma_f = MfmaScaleGU(m_repeat=f_repeat, num_acc_n=num_acc_n)
    epi_t = epi_s if fold_rows == short_rows else make_epilogue(fold_rows // 16, fold_rows)
    epi_f = _FoldEpilogue(epi, epi_t, m_repeat * num_acc_n, sort_block_m)

    def do_fold_tile(flat):
        m_tile, n_tile = _decode(flat)
        n_tile_base = n_wave_base + n_tile * fx.Int32(tile_n)
        expert = sched.expert_of(m_tile)
        # fmt: off
        do_tile(m_tile, n_tile_base, expert, sched, a_gather_f, a_s2r, b_loader, b_scale, a_scale_f, mfma_f, epi_f,
            a_buf, a_scale_lds, f_rows * a_k_step_bytes // 4, k_iters, f_repeat, num_acc_n, a_k_step_bytes,
            pipe_weights, mfma_amajor, async_a_copy, trb_rsrc, tib_rsrc, False, pair_k, narrow_final_a, tile_zero,
            row_map)
        # fmt: on

    return expert_of_flat, do_scheduled_tile, do_short_tile, do_fold_tile


class _FoldEpilogue:
    """Stores a folded tile: the head m-tile's accumulators, then the next m-tile's leading rows."""

    def __init__(self, head, tail, head_accs, sort_block_m):
        self._head, self._tail = head, tail
        self._head_accs, self._sort_block_m = head_accs, sort_block_m

    def store(self, acc, tile_i32, tile_row_base_i32, n_tile_base_i32):
        # Each store ends with an LDS barrier, so the second may reuse the cshuffle LDS.
        self._head.store(acc[: self._head_accs], tile_i32, tile_row_base_i32, n_tile_base_i32)
        self._tail.store(acc[self._head_accs :], tile_i32 + fx.Int32(1),
            tile_row_base_i32 + fx.Int32(self._sort_block_m), n_tile_base_i32)


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
