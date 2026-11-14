# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""A grouped GEMM task of the MegaMoE TP kernel built on the dense gfx1250 MXFP4 GEMM's main
loop (kernels/gemm_a4w4_gfx1250.py: bounded TDM staging, two register banks of WMMA operands,
32x16x128 WMMA). Unlike gemm_tile (the grouped MoE kernel's loop) it speeds up with deeper
pipelines (tile_k 128 / 4 buffers: G1 -23%, G2 -21% in the dense benchmark at big M).

Same call interface as gemm_tile's gemm_task. Data layouts:
  A        [rows, K/2] fp4 row-major (G1: placed rows; G2: G1's output)
  A scales e8m0 row-major [rows, K/32] (placement / G1's output); a (tile_m, 4 B) TDM tile
           of them lands as the 32x4 LDS layout of one k128 (tile_k 128 only)
  B        moe_shuffle_weight (16-row groups of K/2 bytes, 16x16 shuffle), per expert
  B scales moe_shuffle_scale (32x4 layout), per expert
G1 (stage1): swiglu on the interleaved gate/up columns, MXFP4 out (+ row-major scales);
G2: bf16 rows."""

from __future__ import annotations

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, range_constexpr, rocdl
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

from aiter.utility.mx_types import MxDtypeInt as MxDtype

from .. import tdm_ops_gfx1250 as tdm_ops
from ..gemm_common_gfx1250 import (
    batched_silu_swiglu,
    make_lds_copy_ops,
    pipeline_fence,
    workgroup_barrier,
)
from ..kernels_common import ceildiv
from ..quant_utils import emit_amax_e8m0_native_scale, emit_cvt_scalef32_pk8_fp4_bf16
from .device import acq_agent, acq_wg, spin_agent

WAVE = 32
TILES_PER_GROUP = 16  # gemm_tile's DeepGEMM contiguous-M swizzle


def make_dense_task(
    *,
    K: int,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    m_warp: int,
    n_warp: int,
    num_buffers: int,
    n_experts: int,
    lds_arena,
    stage1: bool,
    a_row_bytes: int = 0,
    b_n: int = 0,
    b_k: int = 0,
    expert_given: bool = True,
    name: str = "",
):
    """-> (task, block). K: the A row's K (G1: H; G2: I as run); b_k < K: B rows hold b_k
    (G2: Ip), the K loop runs b_k; b_n: B rows per expert (G1: 2 Ip) when < the task's N.
    a_row_bytes: A row pitch (default K / 2)."""
    assert tile_k == 128 and 2 <= num_buffers <= 4, "row-major A scales land as 32x4 only at tile_k 128"

    assert tile_m % (16 * m_warp) == 0 and tile_n % (32 * n_warp) == 0
    KB = b_k or K  # the K the loop covers (B's)
    A_ROW = a_row_bytes or K // 2
    waves = m_warp * n_warp
    block = waves * WAVE
    wm_rep, wn_rep = tile_m // (16 * m_warp), tile_n // (32 * n_warp)
    assert not stage1 or wn_rep % 2 == 0, "G1: an MX block (64 N cols) spans two wn fragments"
    ks_count, pack_k = tile_k // 128, tile_k // 2
    b_group = 16
    a_row, b_row = pack_k + 16, pack_k * b_group
    sa_rows, sb_rows = max(1, tile_m // 32), tile_n // 32
    a_bytes, b_bytes = tile_m * a_row, (tile_n // b_group) * b_row
    sa_off, sb_off = a_bytes + b_bytes, a_bytes + b_bytes + sa_rows * tile_k
    pitch = ((sb_off + sb_rows * tile_k + 1023) // 1024) * 1024
    c_lds_row = tile_n + 8
    arena_bytes = max(num_buffers * pitch, 0 if stage1 else tile_m * c_lds_row * 2)
    # A scales row-major [rows, K / 32] (placement / G1's output): a (tile_m, 4 B) TDM tile of
    # them is exactly the 32x4 LDS layout of one k128 (32 rows x 4 blocks per super-row)
    SA_ROW = K // 32  # A scale bytes per row (the A row's K)
    SC_STRIDE_B = ceildiv(KB, 128) * 128
    KP_B = KB // 2  # B row bytes
    acc_vgprs, bank_vgprs = wm_rep * wn_rep * 16, 9 * wm_rep + 17 * wn_rep
    pipelined = ks_count % 2 == 0 and acc_vgprs + 2 * bank_vgprs <= 800

    def task(
        arg_c: fx.Pointer,
        arg_a: fx.Pointer,
        arg_b: fx.Pointer,
        arg_scale_a: fx.Pointer,
        arg_scale_b: fx.Pointer,
        arg_m_tile_map: fx.Pointer,
        arg_bias: fx.Pointer,
        arg_quant_scale: fx.Pointer,
        arg_ep_row_map: fx.Pointer,
        i32_m: fx.Int32,
        i32_n: fx.Int32,
        f32_swiglu_limit: fx.Float32,
        f32_situ_beta: fx.Float32,
        f32_situ_linear_beta: fx.Float32,
        task_id: fx.Int32,
        schedule_rows: fx.Int32,
        expert_hint: fx.Int32,
        wait_addr: fx.Int64,
        wait_target: fx.Int32,
        wait_err: fx.Int64,
    ):
        # mega passes its buffers as int32-typed pointers (engine: ptr_arg(t, fx.Int32)); the
        # offsets below are bytes, so address them through byte iterators
        arg_a = fx.recast_iter(fx.Uint8, arg_a)
        arg_b = fx.recast_iter(fx.Uint8, arg_b)
        arg_scale_a = fx.recast_iter(fx.Uint8, arg_scale_a)
        arg_scale_b = fx.recast_iter(fx.Uint8, arg_scale_b)
        out16 = fx.recast_iter(fx.Int16, arg_c)  # G1's fp4 output: 2-byte stores, even offsets
        qs8 = fx.recast_iter(fx.Uint8, arg_quant_scale)
        tid = fx.thread_idx.x
        wave = fx.Int32(rocdl.readfirstlane(T.i32, tid // WAVE))
        lane, lane16, kh = tid % WAVE, tid % 16, (tid % WAVE) // 16
        wm_base = (wave // n_warp) * (tile_m // m_warp)
        wn_base = (wave % n_warp) * (tile_n // n_warp)

        # task -> (m-tile, n-tile): gemm_tile's swizzle (groups of 16 m-tiles, n slowest)
        total_n_tiles = ceildiv(i32_n, tile_n)
        total_m_tiles = ceildiv(schedule_rows, tile_m)
        blocks_per_group = total_n_tiles * TILES_PER_GROUP
        group = task_id // blocks_per_group
        group_first = group * TILES_PER_GROUP
        in_group = task_id - group * blocks_per_group
        rem = total_m_tiles - group_first
        group_tiles = (rem < TILES_PER_GROUP).select(rem, TILES_PER_GROUP)
        m_tile = group_first + (in_group - (in_group // group_tiles) * group_tiles)
        n_tile = in_group // group_tiles
        bm = m_tile * tile_m
        bn = n_tile * tile_n

        i32_ptr = fx.PointerType.get(
            elem_ty=fx.Int32.ir_type, address_space=fx.AddressSpace.Global, alignment=4
        )
        tile_map = fx.recast_iter(i32_ptr, arg_m_tile_map)
        has_work = bm < tile_map[n_experts - 1]
        # (the bisect bounds exist on both branches: the AST rewrite yields them from the if)
        lo, hi = bm * 0, bm * 0 + n_experts
        if const_expr(expert_given):  # the caller's bisect
            expert = has_work.select(fx.Int32(expert_hint), fx.Int32(0))
        else:
            # the owning expert: the first whose end (tile_map) is past bm (gemm_tile's bisect;
            # the caller's hint is not computed when cfg.expert_hint is off)
            for _ in range_constexpr(max(1, (max(2, n_experts) - 1).bit_length() + 1)):
                mid = (lo + hi) >> 1
                mid_c = (mid < n_experts - 1).select(mid, n_experts - 1)
                go_right = tile_map[mid_c] <= bm
                lo = go_right.select(mid + 1, lo)
                hi = go_right.select(hi, mid)
            expert = has_work.select(lo, fx.Int32(0))
        # (never index past the last expert, whatever the hint)
        expert = (expert < n_experts).select(expert, n_experts - 1)
        mn_oob = has_work.select(tile_map[expert] - bm, fx.Int32(0))  # valid rows of the tile
        n_w = fx.Int32(b_n) if b_n else i32_n  # B rows per expert
        n_valid = n_w - bn  # valid B rows (output cols) of the tile

        # the caller's dependency (G2: every G1 tile of this m-tile), before any load
        if wait_target != fx.Int32(0):
            if tid == fx.Int32(0):
                spin_agent(wait_addr, wait_target, wait_err, 64)
                acq_agent()
            workgroup_barrier()
            acq_wg()

        base = lds_arena(arena_bytes)

        def view(ptr, shape, stride):
            return fx.Tensor(fx.make_view(ptr, fx.make_layout(shape, stride)))

        def stage(s):
            return fx.add_offset(base, s * pitch)

        def clear(ptr, size):
            for i in range_constexpr((size + block * 16 - 1) // (block * 16)):
                off = (i * block + tid) * 16
                if off < size:
                    fx.ptr_store(Vec.filled(16, 0, fx.Int8), ptr + off)

        a_base = arg_a + fx.Int64(bm) * fx.Int64(A_ROW)
        b_base = arg_b + (fx.Int64(expert) * fx.Int64(n_w) + fx.Int64(bn)) // b_group * fx.Int64(
            KP_B * b_group
        )
        sa_base = arg_scale_a + fx.Int64(bm) * fx.Int64(SA_ROW)
        sb_base = arg_scale_b + (fx.Int64(expert) * fx.Int64(n_w) + fx.Int64(bn)) // 32 * fx.Int64(
            SC_STRIDE_B
        )
        count = ceildiv(KB, tile_k)

        if has_work:
            # no full-arena clear per task (a persistent CTA runs many short tasks): rows / cols the
            # TDM skips (past the expert's rows, past b_n) keep stale LDS, which only reaches the
            # dropped rows / cols; the partial last K tile (both operands' tail must be 0) is cleared
            # in issue()

            def issue(s, kt):
                ptr = stage(s)
                ke = kt * tile_k
                if const_expr(KB % tile_k != 0):
                    if ke + tile_k > KB:
                        clear(ptr, pitch)
                        workgroup_barrier()
                def nonneg(v):
                    return max(0, v) if isinstance(v, int) else fx.max(0, v)

                remaining = nonneg(KB - ke)

                def copy(owner, src, offset, shape, stride, outer, inner, pad):
                    if wave == owner % waves:
                        desc = tdm_ops.make_tensor_descriptor_2d(
                            global_ptr=view(src, shape, (shape[1], 1)),
                            lds_memref=view(ptr + offset, shape, (shape[1] + pad, 1)),
                            global_offset=(0, 0),
                            tensor_shape=shape,
                            strides=(stride, 1),
                            tile_shape=shape,
                            elem_bytes=1,
                            pad_interval=shape[1] if pad else 0,
                            pad_amount=pad,
                            num_warps=1,
                            workgroup_mask=0,
                            oob_outer_bound=outer,
                            oob_inner_bound=inner,
                        )
                        tdm_ops.tensor_load_2d(desc)

                copy(0, a_base + fx.Int64(ke // 2), 0, (tile_m, pack_k), A_ROW,
                     fx.max(0, mn_oob), ceildiv(remaining, 32) * 16, 16)
                copy(1, b_base + fx.Int64(ke // 2 * b_group), a_bytes,
                     (tile_n // b_group, pack_k * b_group), KP_B * b_group,
                     fx.max(0, ceildiv(n_valid, b_group)), ceildiv(remaining, 32) * 16 * b_group, 0)
                copy(2, sa_base + fx.Int64(ke // 32), sa_off, (tile_m, 4), SA_ROW,
                     fx.max(0, mn_oob), nonneg(SA_ROW - ke // 32), 0)
                copy(3, sb_base + fx.Int64(ke), sb_off, (sb_rows, tile_k), SC_STRIDE_B,
                     fx.max(0, ceildiv(n_valid, 32)), nonneg(SC_STRIDE_B - ke), 0)

            ld32, _ = make_lds_copy_ops(32)
            ld128, _ = make_lds_copy_ops(128)
            acc = [fx.make_rmem_tensor(16, fx.Float32) for _ in range_constexpr(wm_rep * wn_rep)]
            for cf in acc:
                cf.store(Vec.filled(16, 0.0, fx.Float32))
            atoms = [
                fx.make_mma_atom(
                    rocdl.WMMAScale(32, 16, 128, fx.Float4E2M1FN, fx.Float4E2M1FN, fx.Float32,
                                    opsel_b=i)
                )
                for i in range_constexpr(2)
            ]

            def stage_addr(s):
                return fx.Int32(fx.ptrtoint(stage(s)))

            def new_bank():
                return (
                    [fx.make_rmem_tensor(8, fx.Int32) for _ in range_constexpr(wm_rep)],
                    [fx.make_rmem_tensor(16, fx.Int32) for _ in range_constexpr(wn_rep)],
                    [fx.make_rmem_tensor(1, fx.Int32) for _ in range_constexpr(wm_rep)],
                    [fx.make_rmem_tensor(1, fx.Int32) for _ in range_constexpr(wn_rep)],
                )

            banks = [new_bank() for _ in range_constexpr(2 if pipelined else 1)]
            a_lane = fx.Int32((wm_base + lane16) * a_row + kh * 16)
            b_lane = fx.Int32(a_bytes + (wn_base // 16) * b_row + kh * 16 * b_group + lane16 * 16)
            sa_lane = fx.Int32(sa_off + ((wm_base // 32) * (tile_k // 4) + lane) * 4)
            sb_lane = fx.Int32(sb_off + ((wn_base // 32) * (tile_k // 4) + lane) * 4)

            def load_bank(bank, addr, ks):
                acts, weights, ascales, bscales = bank
                a_ptr, b_ptr = addr + a_lane, addr + b_lane
                sa_ptr, sb_ptr = addr + sa_lane, addr + sb_lane
                for wm in range_constexpr(wm_rep):
                    off = wm * 16 * a_row + ks * 64
                    lo, hi = ld128(a_ptr, off), ld128(a_ptr, off + 32)
                    acts[wm].store(lo.shuffle(hi, list(range(8))))
                    ascales[wm].store(ld32(sa_ptr, ((wm // 2) * (tile_k // 4) + ks * 32) * 4))
                for wn in range_constexpr(wn_rep):
                    vs = []
                    for half in range_constexpr(2):
                        off = (wn * 2 + half) * b_row + ks * 64 * b_group
                        lo, hi = ld128(b_ptr, off), ld128(b_ptr, off + 32 * b_group)
                        vs.append(lo.shuffle(hi, list(range(8))))
                    weights[wn].store(vs[0].shuffle(vs[1], list(range(16))))
                    bscales[wn].store(ld32(sb_ptr, (wn * (tile_k // 4) + ks * 32) * 4))

            def mma_bank(bank):
                acts, weights, ascales, bscales = bank
                for wm in range_constexpr(wm_rep):
                    for wn in range_constexpr(wn_rep):
                        cf = acc[wm * wn_rep + wn]
                        sa_w, sb_w = bscales[wn].load()[0], ascales[wm].load()[0]
                        fx.gemm(atoms[wm % 2], cf, weights[wn], acts[wm], cf,
                                scale_a=sa_w, scale_b=sb_w)

            def plain_tile(slot, future=None):
                addr = stage_addr(slot)
                for ks in range_constexpr(ks_count):
                    load_bank(banks[0], addr, ks)
                    if const_expr(ks == 0 and future is not None):
                        rocdl.sched_barrier(0)
                        issue(future % num_buffers, future)
                        rocdl.sched_barrier(0)
                    rocdl.s_wait_dscnt(0)
                    mma_bank(banks[0])

            def run_tile(slot, next_slot, refill, outstanding):
                addr = stage_addr(slot)
                for ks in range_constexpr(ks_count):
                    if const_expr(ks + 1 < ks_count):
                        load_bank(banks[(ks + 1) % 2], addr, ks + 1)
                    elif const_expr(next_slot is not None):
                        rocdl.s_wait_dscnt(0)
                        pipeline_fence(outstanding)
                        if const_expr(refill is not None):
                            rocdl.sched_barrier(0)
                            issue(slot, refill)
                            rocdl.sched_barrier(0)
                        load_bank(banks[0], stage_addr(next_slot), 0)
                    rocdl.sched_barrier(0)
                    mma_bank(banks[ks % 2])
                    rocdl.sched_barrier(0)

            if const_expr(pipelined):
                for i in range_constexpr(num_buffers):
                    issue(i, min(i, count - 1))
                pipeline_fence(num_buffers - 1)
                load_bank(banks[0], stage_addr(0), 0)
                steady = max(0, count - num_buffers)
                for kt in range(steady):
                    run_tile(kt % num_buffers, (kt + 1) % num_buffers, kt + num_buffers,
                             num_buffers - 2)
                for j in range_constexpr(num_buffers):
                    if const_expr(steady + j < count):
                        nxt = (steady + j + 1) % num_buffers if j < num_buffers - 1 else None
                        run_tile((steady + j) % num_buffers, nxt, None, num_buffers - 2 - j)
            else:
                for i in range_constexpr(num_buffers - 1):
                    issue(i, min(i, count - 1))
                steady = max(0, count - (num_buffers - 1))
                for kt in range(steady):
                    pipeline_fence(num_buffers - 2)
                    plain_tile(kt % num_buffers, kt + num_buffers - 1)
                for j in range_constexpr(num_buffers - 1):
                    if const_expr(steady + j < count):
                        pipeline_fence(num_buffers - 2 - j)
                        plain_tile((steady + j) % num_buffers)
            pipeline_fence(0)

            if const_expr(stage1):
                # swiglu over the interleaved (gate, up) column pairs, MXFP4 per 32 outputs: a
                # lane holds 16 outputs of a block (two wn fragments x 8 pairs), the other 16
                # are lane ^ 16's (the amax crosses kh). Output col o = N col / 2.
                OUT_ROW = i32_n // 4  # fp4 bytes per output row (N / 2 outputs)
                SC_ROW_O = i32_n // 64  # its e8m0 bytes per row (row-major, G2's A scales)
                neg_limit = fx.Float32(0.0) - f32_swiglu_limit
                for wm in range_constexpr(wm_rep):
                    row_rel = wm_base + wm * 16 + lane16
                    row = bm + row_rel
                    row_ok = row_rel < mn_oob
                    for mb in range_constexpr(wn_rep // 2):
                        pairs = []
                        for sub in range_constexpr(2):
                            vals = Vec(acc[wm * wn_rep + mb * 2 + sub].load())
                            for p in range_constexpr(8):
                                pairs.append((vals[2 * p], vals[2 * p + 1]))
                        outs = batched_silu_swiglu(
                            pairs, swiglu=True, limit_f32=f32_swiglu_limit,
                            neg_limit_f32=neg_limit, range_constexpr=range_constexpr,
                        )
                        scale_f32, e8 = emit_amax_e8m0_native_scale(
                            outs, wave_size=WAVE, dtype=MxDtype.FP4_E2M1
                        )
                        # outs[sub * 8 + half * 4 + p]: N col wn_base + (mb*2+sub)*32 + half*16 + kh*8 + 2p
                        for sub in range_constexpr(2):
                            src = Vec.from_elements(outs[sub * 8 : sub * 8 + 8], fx.Float32)
                            packed = emit_cvt_scalef32_pk8_fp4_bf16(
                                src.to(fx.BFloat16).ir_value(), scale_f32, i32_ty=T.i32
                            )
                            ncol0 = bn + wn_base + (mb * 2 + sub) * 32 + kh * 8
                            o0 = ncol0 // 2  # outputs o0..o0+3 (half 0), o0+8.. (half 1)
                            if row_ok:
                                # bytes row * OUT_ROW + o0 / 2 (even) and + 4: in 2-byte elements
                                e0 = (fx.Int64(row) * fx.Int64(OUT_ROW) + fx.Int64(o0 // 2)) // 2
                                fx.ptr_store(fx.Int16(packed), out16 + e0)
                                fx.ptr_store(fx.Int16(packed >> fx.Int32(16)), out16 + e0 + 2)
                        # the block's e8m0 (row-major), by the kh 0 lane; blocks past the n_w / 2
                        # real outputs (stale B rows) get 2^0: G2 reads their scale group in its
                        # zero-filled K tail, where a stale 0xFF (NaN) would poison the row
                        kb = (bn + wn_base + mb * 64) // 64  # output block index
                        e8 = (kb * 64 < n_w).select(e8, fx.Int8(0x7F))
                        if (kh == fx.Int32(0)) & row_ok:
                            soff = fx.Int64(row) * fx.Int64(SC_ROW_O) + fx.Int64(kb)
                            fx.ptr_store(fx.Int8(e8), fx.recast_iter(fx.Int8, qs8 + soff))
            else:
                # bf16 rows: stage the tile in LDS, then consecutive lanes store consecutive
                # 16 B of a row (rows past the expert's valid rows dropped)
                workgroup_barrier()
                for wm in range_constexpr(wm_rep):
                    row_rel = wm_base + wm * 16 + lane16
                    for wn in range_constexpr(wn_rep):
                        values = acc[wm * wn_rep + wn].load().to(fx.BFloat16)
                        for half in range_constexpr(2):
                            col_rel = wn_base + wn * 32 + half * 16 + kh * 8
                            fx.ptr_store(
                                values.shuffle(values, list(range(half * 8, half * 8 + 8))),
                                fx.recast_iter(fx.BFloat16, base) + row_rel * c_lds_row + col_rel,
                            )
                workgroup_barrier()
                chunks = tile_n // 8
                outp = fx.recast_iter(fx.BFloat16, arg_c)
                for it in range_constexpr(tile_m * chunks // block):
                    idx = it * block + tid
                    row_rel, col_rel = idx // chunks, idx % chunks * 8
                    vals = fx.Vector(
                        fx.ptr_load(
                            fx.recast_iter(fx.BFloat16, base) + row_rel * c_lds_row + col_rel,
                            result_type=T.vec(8, fx.BFloat16.ir_type),
                        )
                    )
                    if row_rel < mn_oob:
                        fx.ptr_store(
                            vals,
                            outp + fx.Int64(bm + row_rel) * fx.Int64(i32_n) + fx.Int64(bn + col_rel),
                        )
                workgroup_barrier()

    return flyc.jit(task), block
