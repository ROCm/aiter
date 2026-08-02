# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
"""GEMM1 compute shared by fused MegaMoE v2 stage1 and its standalone interface."""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl.expr import const_expr, range_constexpr, rocdl
from flydsl.expr.typing import Vector as Vec

from ..tensor_shim import _run_compiled
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
    _buffer_load,
    _make_buffer,
    wait_lds_barrier,
)


class _LdsF32View:
    def __init__(self, ptr):
        self.ptr = ptr


@flyc.jit
# fmt: off
def do_tile(m_tile, n_tile_base, expert, sched, a_gather, a_s2r, b_loader, b_scale, a_scale, mfma, epi, a_buf,
    a_scale_lds, a_lds_i32, K_ITERS, M_REPEAT, NUM_ACC_N, A_K_STEP_BYTES, pipe_weights,
    mfma_amajor, async_a_copy, trb_rsrc):
# fmt: on
    N_ACC = M_REPEAT * NUM_ACC_N
    NUM_B_SCALE = NUM_ACC_N // _PACK
    NUM_A_SCALE = M_REPEAT // _PACK
    B_STATE_END = (
        N_ACC + NUM_ACC_N * _PACK
    )
    SB_STATE_END = B_STATE_END + NUM_B_SCALE
    last = fx.Int32(K_ITERS - 1)
    tile_row_base = _buffer_load(trb_rsrc, m_tile, fx.Int32)
    b_row = sched.gate_base_row(expert) + n_tile_base
    a_gather.for_tile(tile_row_base)
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
        a_scale.stage(a_scale_lds, tile_row_base)
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
        for sp_i, state in range(0, K_ITERS - 1, 1, init=init):
            sp = fx.Int32(sp_i)
            acc = [Vec(a) for a in state[:N_ACC]]
            b_prev = [
                [Vec(state[N_ACC + ni * _PACK + ks]) for ks in range(_PACK)] for ni in range(NUM_ACC_N)
            ]
            cur_off = (sp & fx.Int32(1)) * fx.Int32(a_lds_i32)
            nxt_off = ((sp + fx.Int32(1)) & fx.Int32(1)) * fx.Int32(a_lds_i32)
            spn = sp + fx.Int32(1)
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
            state = yield yv
        acc = [Vec(r) for r in state[:N_ACC]]
        b_prev = [
            [Vec(state[N_ACC + ni * _PACK + ks]) for ks in range(_PACK)]
            for ni in range(NUM_ACC_N)
        ]
        final_off = (last & fx.Int32(1)) * fx.Int32(a_lds_i32)

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
        a_scale.stage(a_scale_lds, tile_row_base)
        wait_lds_barrier(0 if async_a_copy else 63)
        init = [mfma.zero_value for _ in range(N_ACC)]
        for sp_i, state in range(0, K_ITERS, 1, init=init):
            sp = fx.Int32(sp_i)
            acc = [Vec(a) for a in state]
            cur_off = (sp & fx.Int32(1)) * fx.Int32(a_lds_i32)
            nxt_off = ((sp + fx.Int32(1)) & fx.Int32(1)) * fx.Int32(a_lds_i32)
            spn = (sp + fx.Int32(1) < last).select(
                sp + fx.Int32(1),
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
def build_fused_gemm1(*, x_tensor, w_rsrc, sw_rsrc, sx_rsrc,
    out_rsrc, os_rsrc, trb_rsrc, expert_rsrc, out_tensor, a_buf, a_scale_lds, c_tile,
    model_dim, inter_dim, sort_block_m, tile_n, num_waves, n_per_wave, wave_id,
    m_repeat, num_acc_n, a_k_step_bytes, total_threads, k_iters, a_lds_i32, n_tiles,
    expert_offset, b_cache_modifier, swizzle_a, pipe_weights, mfma_amajor, async_a_copy,
    use_tile_resource):
    # fmt: on
    """Build the GEMM1 atoms and return its expert resolver and tile runner."""
    sched = TileScheduler(
        expert_rsrc=expert_rsrc,
        inter_dim=inter_dim,
        expert_offset=expert_offset,  # GLOBAL sorted_expert_id -> LOCAL w1 index
    )
    n_wave_base = wave_id * fx.Int32(n_per_wave)

    # fmt: off
    a_gather = ATileLoader(row_bytes=model_dim, sort_block_m=sort_block_m,
        k_step_bytes=a_k_step_bytes, total_threads=total_threads, swizzle=swizzle_a,
        x_tensor=x_tensor, async_copy=async_a_copy)
    # fmt: on
    a_s2r = AS2RLoader(k_step_bytes=a_k_step_bytes, swizzle=swizzle_a)
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
    )
    mfma = MfmaScaleGU(m_repeat=m_repeat, num_acc_n=num_acc_n)
    # fmt: off
    epi = SiluQuantEpilogue(out_rsrc=out_rsrc, out_scale_rsrc=os_rsrc, sorted_rsrc=trb_rsrc, tokens=0,
        inter_dim=inter_dim, m_repeat=m_repeat, num_acc_n=num_acc_n, sort_block_m=sort_block_m, tile_n=tile_n,
        num_waves=num_waves, lds_out=c_tile, always_valid=True,
        out_tensor=out_tensor if use_tile_resource else None)
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
        do_tile(m_tile, n_tile_base, expert, sched, a_gather,
            a_s2r, b_loader, b_scale, a_scale, mfma, epi, a_buf,
            a_scale_lds, a_lds_i32, k_iters, m_repeat, num_acc_n,
            a_k_step_bytes, pipe_weights, mfma_amajor, async_a_copy,
            trb_rsrc)
        # fmt: on

    return expert_of_flat, do_scheduled_tile


# fmt: off
@functools.lru_cache(maxsize=None)
def compile_gemm1(
    *, model_dim: int, inter_dim: int, expert_offset: int = 0, sort_block_m: int = 32,
    tile_n: int = 256, tile_k: int = 256, num_waves: int = 4, pipe_weights: bool = True,
    mfma_amajor: bool = False, swizzle_a: bool = True, async_a_copy: bool = False,
    use_tile_resource: bool = True, waves_per_eu_hint: int = 2, b_cache_modifier: int = 0,
):
    # fmt: on
    """Compile standalone group GEMM1 from the fused Stage1 compute body."""
    num_waves = int(num_waves)
    assert num_waves > 1
    assert 1 <= waves_per_eu_hint <= 4
    assert tile_n % num_waves == 0
    assert (2 * inter_dim) % tile_n == 0
    assert tile_k == 256 and model_dim % tile_k == 0

    n_per_wave = tile_n // num_waves
    n_tiles = (2 * inter_dim) // tile_n
    m_repeat = sort_block_m // 16
    num_acc_n = n_per_wave // 16
    assert num_acc_n % 2 == 0 and m_repeat % 2 == 0

    a_k_step_bytes = tile_k
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

        w_rsrc = _make_buffer(w, fx.Int32, 4)
        sx_rsrc = _make_buffer(scale_x, fx.Int32, 4)
        sw_rsrc = _make_buffer(scale_w, fx.Int32)
        trb_rsrc = _make_buffer(tile_row_base, fx.Int32)
        expert_rsrc = _make_buffer(expert_ids, fx.Int32)
        if const_expr(use_tile_resource):
            out_rsrc = None
        else:
            out_rsrc = _make_buffer(
                out, fx.Int16, max_size=False, num_records_bytes=num_valid * fx.Int32(inter_dim)
            )
        scale_cols = (inter_dim // 32 + 7) // 8 * 8
        os_rsrc = _make_buffer(
            out_scale,
            fx.Int8,
            max_size=False,
            num_records_bytes=num_valid * fx.Int32(scale_cols) + fx.Int32(8192),
        )
        wave_id = fx.thread_idx.x // 64

        _, run_tile = build_fused_gemm1(
            x_tensor=x, w_rsrc=w_rsrc, sw_rsrc=sw_rsrc,
            sx_rsrc=sx_rsrc, out_rsrc=out_rsrc, os_rsrc=os_rsrc, trb_rsrc=trb_rsrc,
            expert_rsrc=expert_rsrc, out_tensor=out, a_buf=a_buf,
            a_scale_lds=a_scale_lds, c_tile=c_tile, model_dim=model_dim, inter_dim=inter_dim,
            sort_block_m=sort_block_m, tile_n=tile_n, num_waves=num_waves, n_per_wave=n_per_wave,
            wave_id=wave_id, m_repeat=m_repeat, num_acc_n=num_acc_n, a_k_step_bytes=a_k_step_bytes,
            total_threads=total_threads, k_iters=k_iters, a_lds_i32=a_lds_i32, n_tiles=n_tiles,
            expert_offset=expert_offset, b_cache_modifier=b_cache_modifier, swizzle_a=swizzle_a,
            pipe_weights=pipe_weights, mfma_amajor=mfma_amajor, async_a_copy=async_a_copy,
            use_tile_resource=use_tile_resource,
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
    num_cu: int = 256, b_cache_modifier: int = 0,
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
    )
    _run_compiled(
        launch, out, x, w.view(torch.uint8), scale_x, scale_w.view(torch.uint8), tile_row_base, expert_ids, out_scale,
        fx.Int32(num_valid), fx.Int32(grid_x), stream,
    )
    return out, out_scale
