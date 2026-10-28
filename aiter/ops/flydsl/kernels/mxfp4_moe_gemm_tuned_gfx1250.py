# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Grouped contiguous-M MXFP4 MoE GEMM on the gemm_a4w4_256x256 schedule (gfx1250).

The mainloop is the standalone kernel's four-quadrant software pipeline
(``gemm_a4w4_256x256_gfx1250``). The MoE parts come from
``mxfp4_preshuffle_gfx1250_tdm``: the contiguous-M tile swizzle, the in-kernel
expert lookup through ``m_tile_map``, per-expert B / B-scale bases and row
bounds, the cluster multicast masks, the MoE A preshuffle
((M/16, K/128, 4, 16, 16 B)) and the wave-interleaved M16K4 A scales.

Only the plain epilogue is supported: BF16/F16 output, no activation, bias,
quantization or EP scatter.
"""

import math
import os
from collections.abc import Callable
from typing import NamedTuple

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, range_constexpr, rocdl
from flydsl.expr.rocdl import cluster
from flydsl.expr.typing import Constexpr, T, as_ir_value
from flydsl.expr.typing import Vector as Vec
from flydsl.runtime.device import get_rocm_arch as get_hip_arch

from aiter.jit.utils.chip_info import get_lds_capacity_bytes

from . import tdm_ops_gfx1250 as tdm_ops
from .gemm_common_gfx1250 import (
    make_lds_copy_ops,
    pipeline_fence,
    pipeline_fence_signal,
    pipeline_fence_wait,
    workgroup_barrier,
)
from .kernels_common import ceildiv, format_kernel_name

# Revolutions (num_buffers K-tiles each) between cluster barriers of an MxN
# cluster; 0 disables them. 1xN clusters never need one.
CLUSTER_SYNC_REVS = int(os.environ.get("AITER_FLYDSL_MOE_TUNED_SYNC_REVS", "8"))
if CLUSTER_SYNC_REVS < 0:
    raise ValueError("AITER_FLYDSL_MOE_TUNED_SYNC_REVS must be nonnegative")


def moe_tuned_supported(
    *,
    K,
    tile_m,
    tile_n,
    tile_k,
    m_warp,
    n_warp,
    num_buffers,
    cluster_m,
    cluster_n,
):
    """Whether the four-quadrant schedule covers this grouped GEMM shape."""
    k_tiles = K // tile_k if tile_k else 0
    return all(
        [
            (tile_m, tile_n, tile_k) == (256, 256, 256),
            (m_warp, n_warp, num_buffers) == (2, 2, 4),
            K % tile_k == 0,
            k_tiles >= num_buffers,
            k_tiles % num_buffers in (0, 2),
            cluster_m * cluster_n <= 16,
        ]
    )


class _Operand(NamedTuple):
    """LDS loaders and next-stage seed registers of one WMMA operand and its scales."""

    load: Callable  # (stage, fragment, k) -> (value, DS mark)
    load_scale: Callable  # (stage, scale row, k) -> (word, DS mark)
    seeds: list  # [bank][fragment][k]
    scale_seeds: list  # [bank][scale row][k]
    frags: int  # fragments per quadrant
    scale_rows: int  # scale rows the seed half loads
    width: int  # i32 lanes per fragment
    second_scale_rows: int  # scale rows the second half loads, after the seed rows


@flyc.jit
def launch_moe_gemm_a4w4_tuned(
    arg_c: fx.Pointer,
    arg_a: fx.Pointer,
    arg_b: fx.Pointer,
    arg_scale_a: fx.Pointer,
    arg_scale_b: fx.Pointer,
    arg_m_tile_map: fx.Pointer,
    i32_m: fx.Int32,
    stream: fx.Stream,
    N: fx.Int32,
    K: Constexpr[int],
    n_experts: Constexpr[int],
    tile_m: Constexpr[int],
    tile_n: Constexpr[int],
    tile_k: Constexpr[int],
    m_warp: Constexpr[int],
    n_warp: Constexpr[int],
    num_buffers: Constexpr[int],
    out_is_f16: Constexpr[int],
    cluster_m: Constexpr[int] = 1,
    cluster_n: Constexpr[int] = 1,
):
    """Launch the grouped MoE GEMM.

    ``i32_m`` is the contiguous-M capacity; ``m_tile_map`` holds each expert's
    cumulative valid-row end. A cluster spans ``cluster_n`` N-tiles along grid x
    and ``cluster_m`` M-tiles along grid y, as in the TDM grouped kernel, and
    ceil(N / tile_n) must be a multiple of ``cluster_n``.
    """
    assert moe_tuned_supported(
        K=K,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        m_warp=m_warp,
        n_warp=n_warp,
        num_buffers=num_buffers,
        cluster_m=cluster_m,
        cluster_n=cluster_n,
    ), "unsupported MoE tuned GEMM configuration"
    FENCE_WAIT_POS = 7  # WMMAs of slack between the READY signal and its matching wait
    WMMA_M = 32
    WMMA_N = 16
    WMMA_K = 128
    WAVE = 32
    PACK_TK = tile_k // 2  # A/B row bytes per K-tile (FP4 packed 2/byte)
    K_STEPS = tile_k // WMMA_K
    K_TILES = K // tile_k
    tail_mode = K_TILES % num_buffers
    SC_WORDS = tile_k // 4  # B-scale i32 words per super-row per K-tile
    SB_SUPERS = tile_n // 32
    warp_tile_m = tile_m // m_warp
    warp_tile_n = tile_n // n_warp
    wmma_m_rep = warp_tile_m // WMMA_N  # A fragments (16 rows each)
    wmma_n_rep = warp_tile_n // WMMA_M  # B fragments (32 rows each)
    half_m, half_n = wmma_m_rep // 2, wmma_n_rep // 2
    WMMA_PER_Q = half_m * half_n * K_STEPS  # WMMAs in one quadrant
    a_seed_scale_rows = (half_m + 1) // 2
    a_second_scale_rows = wmma_m_rep // 2 - a_seed_scale_rows
    n_acc = wmma_m_rep * wmma_n_rep
    num_waves = m_warp * n_warp
    block = num_waves * WAVE

    # MoE A preshuffle: one M16 group holds 16 rows x tile_k FP4 per K-tile,
    # K32 slabs of (16 rows x 16 B) side by side, and is dense in LDS.
    A_M16_TILE = 16 * PACK_TK
    A_GROUPS = tile_m // 16
    # A scales: one outer row per wave M tile, inner (k128, wm, lane16) dwords.
    AS_INNER = K_STEPS * wmma_m_rep * 16
    AS_SUPERS = m_warp
    AS_FULL_INNER = (K // WMMA_K) * wmma_m_rep * 16
    C_LDS_ROW = tile_n + 8
    B_LDS_ROW = PACK_TK * 16
    STAGE_A = A_GROUPS * A_M16_TILE
    STAGE_B = (tile_n // 16) * B_LDS_ROW
    STAGE_SA = AS_SUPERS * AS_INNER * 4
    STAGE_SB = SB_SUPERS * tile_k
    # Operand-major planar LDS: scales and A below, B on the next 64-KiB boundary above them.
    PLANAR_SA_BASE = 0
    PLANAR_A_BASE = PLANAR_SA_BASE + num_buffers * STAGE_SA
    PLANAR_SB_BASE = PLANAR_A_BASE + num_buffers * STAGE_A
    PLANAR_B_BASE = ((PLANAR_SB_BASE + num_buffers * STAGE_SB + 65535) // 65536) * 65536
    PLANAR_END = PLANAR_B_BASE + num_buffers * STAGE_B

    partial_elem = fx.Float16 if out_is_f16 else fx.BFloat16
    ARENA_B = max(PLANAR_END, tile_m * C_LDS_ROW * 2)
    lds_cap = get_lds_capacity_bytes(str(get_hip_arch()))
    if ARENA_B > lds_cap:
        raise RuntimeError(
            f"Shared Memory Overflow: requested {ARENA_B} bytes on "
            f"{get_hip_arch()}, limit is {lds_cap} bytes"
        )
    TILES_PER_GROUP = 16
    assert TILES_PER_GROUP % cluster_m == 0
    cluster_sync_revs = CLUSTER_SYNC_REVS if cluster_m > 1 else 0
    kernel_name = format_kernel_name(
        f"moe_a4w4_tuned_t{tile_m}x{tile_n}x{tile_k}"
        f"_mw{m_warp}_nw{n_warp}_nb{num_buffers}_K{K}_e{n_experts}"
        f"_cm{cluster_m}_cn{cluster_n}_sync{cluster_sync_revs}"
        + ("_f16" if out_is_f16 else "")
    )

    @flyc.kernel(name=kernel_name, known_block_size=[block, 1, 1])
    def kernel_moe_gemm_a4w4_tuned(
        arg_c: fx.Pointer,
        arg_a: fx.Pointer,
        arg_b: fx.Pointer,
        arg_scale_a: fx.Pointer,
        arg_scale_b: fx.Pointer,
        arg_m_tile_map: fx.Pointer,
        i32_m: fx.Int32,
        i32_n: fx.Int32,
    ):
        Kp16 = (K // 2) * 16
        tid = fx.thread_idx.x
        bid_x = fx.block_idx.x
        wave = fx.Int32(rocdl.readfirstlane(T.i32, tid // WAVE))
        lane = tid % WAVE
        lane16 = lane % 16
        kgrp = lane // 16
        wave_m = wave // n_warp
        wave_n = wave % n_warp

        # DeepGEMM contiguous-M swizzle, run at physical-cluster granularity.
        total_n_tiles = ceildiv(i32_n, tile_n)
        total_m_tiles = ceildiv(i32_m, tile_m)
        swz_id = bid_x // cluster_n if cluster_n > 1 else bid_x
        local_n = bid_x - swz_id * cluster_n if cluster_n > 1 else 0
        n_units = total_n_tiles // cluster_n if cluster_n > 1 else total_n_tiles
        local_m = fx.block_idx.y if cluster_m > 1 else 0
        m_units = (total_m_tiles + cluster_m - 1) // cluster_m
        group_m_units = TILES_PER_GROUP // cluster_m
        blocks_per_group = n_units * group_m_units
        group = swz_id // blocks_per_group
        group_first_tile = group * group_m_units
        in_group = swz_id - group * blocks_per_group
        rem_tiles = m_units - group_first_tile
        group_tiles = (rem_tiles < group_m_units).select(rem_tiles, group_m_units)
        m_unit = group_first_tile + (in_group - (in_group // group_tiles) * group_tiles)
        m_tile = m_unit * cluster_m + local_m
        blk_m = m_tile * tile_m
        n_unit = in_group // group_tiles
        blk_n = (
            (n_unit * cluster_n + local_n) * tile_n
            if cluster_n > 1
            else n_unit * tile_n
        )
        # A broadcasts across the cluster_n N-tiles of one M row.
        a_mask = ((1 << cluster_n) - 1) << (local_m * cluster_n) if cluster_n > 1 else 0
        blk_m64 = fx.Int64(blk_m)
        blk_n64 = fx.Int64(blk_n)
        n64 = fx.Int64(i32_n)

        i32_ptr = fx.PointerType.get(
            elem_ty=fx.Int32.ir_type, address_space=fx.AddressSpace.Global, alignment=4
        )
        tile_map = fx.recast_iter(i32_ptr, arg_m_tile_map)
        # Padding tiles of the worst-case grid exit here; cluster-uniform.
        has_work = blk_m < tile_map[n_experts - 1]
        expert = blk_m * 0 + n_experts
        if has_work:
            lo, hi = blk_m * 0, blk_m * 0 + n_experts
            for _ in range_constexpr(
                max(1, math.ceil(math.log2(max(2, n_experts))) + 1)
            ):
                mid = (lo + hi) >> 1
                mid_clamped = (mid < n_experts - 1).select(mid, n_experts - 1)
                go_right = tile_map[mid_clamped] <= blk_m
                lo = go_right.select(mid + 1, lo)
                hi = go_right.select(hi, mid)
            expert = lo
        eb64 = fx.Int64(expert)
        expert_c = (expert < n_experts).select(expert, n_experts - 1)
        mn_oob = tile_map[expert_c] - blk_m  # this expert's valid rows (A / C)

        b_mask = 0
        full_cluster = None
        if const_expr(cluster_m > 1):
            # B multicasts down the M peers that belong to the same expert.
            cluster_first_m = m_unit * cluster_m
            valid_m_tiles = (tile_map[n_experts - 1] + tile_m - 1) // tile_m
            full_cluster = cluster_first_m + cluster_m <= valid_m_tiles
            prev_expert = (expert > 0).select(expert - 1, 0)
            prev_expert = (prev_expert < n_experts).select(prev_expert, n_experts - 1)
            first_m = (expert > 0).select(
                (tile_map[prev_expert] + tile_m - 1) // tile_m, 0
            )
            end_m = (tile_map[expert_c] + tile_m - 1) // tile_m
            column_mask = fx.Int32(0)
            group_size = fx.Int32(0)
            for mi in range_constexpr(cluster_m):
                peer_m = cluster_first_m + mi
                same_expert = (peer_m >= first_m) & (peer_m < end_m)
                column_mask = column_mask | same_expert.select(1 << (mi * cluster_n), 0)
                group_size = group_size + same_expert.select(1, 0)
            use_b_mcast = full_cluster & (group_size > 1)
            b_mask = use_b_mcast.select(column_mask << local_n, 0)

        arena = fx.SharedAllocator(static=False)
        arena.allocate(ARENA_B)
        base_ptr = arena.base_ptr

        def _planar_base(offset, stride, stage):
            ptr = fx.add_offset(base_ptr, offset + stage * stride)
            return fx.Int64(fx.ptrtoint(ptr))

        def _view(ptr, shape, stride):
            return fx.Tensor(fx.make_view(ptr, fx.make_layout(shape, stride)))

        def _gv(base, off, shape, stride):
            return _view(fx.add_offset(base, off), shape, stride)

        def cluster_sync():
            cluster.cluster_barrier()

        if has_work:
            gA_base = fx.recast_iter(fx.Int8, arg_a)
            gB_base = fx.recast_iter(fx.Int8, arg_b)
            gSA_base = fx.recast_iter(fx.Int8, arg_scale_a)
            gSB_base = fx.recast_iter(fx.Int8, arg_scale_b)

            a_off0 = (blk_m64 // 16) * (16 * (K // 2))
            b_off0 = (eb64 * (n64 // 16) + blk_n64 // 16) * Kp16
            sa_off0 = (blk_m64 // warp_tile_m) * (AS_FULL_INNER * 4)
            sb_off0 = (eb64 * ((n64 + 31) // 32) + blk_n64 // 32) * K

            gA = _gv(gA_base, a_off0, (A_GROUPS, A_M16_TILE), (A_M16_TILE, 1))
            gB = _gv(gB_base, b_off0, (tile_n // 16, B_LDS_ROW), (B_LDS_ROW, 1))
            gSA = _gv(gSA_base, sa_off0, (AS_SUPERS, AS_INNER * 4), (AS_INNER * 4, 1))
            gSB = _gv(gSB_base, sb_off0, (SB_SUPERS, tile_k), (tile_k, 1))

            def _build_tdm_desc(owner):
                if const_expr(owner == 0):
                    tensor, offset, shape = gA, PLANAR_A_BASE, (A_GROUPS, A_M16_TILE)
                    stride, mask, bound, early = (
                        16 * (K // 2),
                        a_mask,
                        (mn_oob + 15) // 16,
                        False,
                    )
                elif const_expr(owner == 1):
                    tensor, offset, shape = gB, PLANAR_B_BASE, (tile_n // 16, B_LDS_ROW)
                    stride, mask, bound, early = Kp16, b_mask, None, False
                elif const_expr(owner == 2):
                    tensor, offset, shape = (
                        gSA,
                        PLANAR_SA_BASE,
                        (AS_SUPERS, AS_INNER * 4),
                    )
                    stride, mask, bound, early = AS_FULL_INNER * 4, a_mask, None, True
                else:
                    tensor, offset, shape = gSB, PLANAR_SB_BASE, (SB_SUPERS, tile_k)
                    stride, mask, bound, early = K, b_mask, None, True
                desc = tdm_ops.make_tensor_descriptor_2d(
                    global_ptr=tensor,
                    lds_memref=_view(
                        fx.add_offset(base_ptr, offset), shape, (shape[1], 1)
                    ),
                    global_offset=(0, 0),
                    tensor_shape=shape,
                    strides=(stride, 1),
                    tile_shape=shape,
                    elem_bytes=1,
                    pad_interval=0,
                    pad_amount=0,
                    num_warps=1,
                    workgroup_mask=mask,
                    early_timeout=early,
                    oob_outer_bound=bound,
                )
                return (
                    Vec(desc.dgroup0),
                    Vec(desc.dgroup1),
                    fx.Int32(shape[0] * shape[1]),
                    fx.Int32(shape[1]),
                )

            dgroup0 = Vec.filled(4, 0, fx.Int32)
            dgroup1 = Vec.filled(8, 0, fx.Int32)
            tdm_lds_step, tdm_global_step = fx.Int32(0), fx.Int32(0)
            if wave == 0:
                dgroup0, dgroup1, tdm_lds_step, tdm_global_step = _build_tdm_desc(0)
            elif wave == 1:
                dgroup0, dgroup1, tdm_lds_step, tdm_global_step = _build_tdm_desc(1)
            elif wave == 2:
                dgroup0, dgroup1, tdm_lds_step, tdm_global_step = _build_tdm_desc(2)
            else:
                dgroup0, dgroup1, tdm_lds_step, tdm_global_step = _build_tdm_desc(3)

            tdm_desc = tdm_ops.TDMDescriptor2D(
                as_ir_value(dgroup0), as_ir_value(dgroup1)
            )
            tdm_base_lds, tdm_base_lo, tdm_base_hi = dgroup0[1], dgroup0[2], dgroup0[3]

            def _prepare_tdm(slot, tile):
                desc = tdm_ops.update_tensor_descriptor_2d_lds_addr(
                    tdm_desc, tdm_base_lds + tdm_lds_step * slot
                )
                return tdm_ops.update_tensor_descriptor_2d_addr64(
                    desc, tdm_base_lo, tdm_base_hi, fx.Int32(tile) * tdm_global_step
                )

            wmb = wave_m * warp_tile_m
            wnb = wave_n * warp_tile_n

            # matrix_b_scale selects which half of the shared A-scale word this 16-row block uses.
            wmma_atoms = [
                fx.make_mma_atom(
                    fx.rocdl.WMMAScale(
                        WMMA_M,
                        WMMA_N,
                        WMMA_K,
                        fx.Float4E2M1FN,
                        fx.Float4E2M1FN,
                        fx.Float32,
                        opsel_b=sa_sel,
                    )
                )
                for sa_sel in range_constexpr(2)
            ]
            c_frags = [
                fx.make_rmem_tensor(16, fx.Float32) for _ in range_constexpr(n_acc)
            ]
            for cf in c_frags:
                cf.store(Vec.filled(16, 0.0, fx.Float32))

            def _rmem(n, v):
                t = fx.make_rmem_tensor(n, fx.Int32)
                t.store(v)
                return t

            def _mma_one(wm, wn, k, act, wt, sa_k, sb_k):
                idx = wm * wmma_n_rep + wn
                fx.gemm(
                    wmma_atoms[wm % 2],
                    c_frags[idx],
                    wt,
                    act,
                    c_frags[idx],
                    scale_a=sb_k[wn * K_STEPS + k],
                    scale_b=sa_k[(wm // 2) * K_STEPS + k],
                )

            def _mma_at(wm0, wn0, act, wt, sa_k, sb_k, pos, n_fast):
                """WMMA at linear position `pos` of the wm x wn x k block, minor index first."""
                n_minor = len(wt) if n_fast else len(act)
                minor, k, major = (
                    pos % n_minor,
                    (pos // n_minor) % K_STEPS,
                    pos // (n_minor * K_STEPS),
                )
                i, j = (major, minor) if n_fast else (minor, major)
                _mma_one(wm0 + i, wn0 + j, k, act[i][k], wt[j][k], sa_k, sb_k)

            if const_expr(cluster_sync_revs > 0):
                if full_cluster:
                    cluster_sync()
            # Keep fragment displacements as DS immediates inside the K loop.
            stage_a_addr, stage_b_addr, stage_sa_addr, stage_sb_addr = [], [], [], []
            a_byte = fx.Int64((wmb // 16) * A_M16_TILE + lane16 * 16 + kgrp * 256)
            b_byte = fx.Int64((wnb // 16) * B_LDS_ROW + kgrp * 256 + lane16 * 16)
            sa_byte = fx.Int64(wave_m * (AS_INNER * 4) + lane * 4)
            sb_byte = fx.Int64(((wnb // 32) * SC_WORDS + lane) * 4)
            for addr_stage in range_constexpr(num_buffers):
                stage_a_addr.append(
                    _planar_base(PLANAR_A_BASE, STAGE_A, addr_stage) + a_byte
                )
                stage_b_addr.append(
                    _planar_base(PLANAR_B_BASE, STAGE_B, addr_stage) + b_byte
                )
                stage_sa_addr.append(
                    _planar_base(PLANAR_SA_BASE, STAGE_SA, addr_stage) + sa_byte
                )
                stage_sb_addr.append(
                    _planar_base(PLANAR_SB_BASE, STAGE_SB, addr_stage) + sb_byte
                )

            lds_load_b32, _ = make_lds_copy_ops(32)
            lds_load_b128, _ = make_lds_copy_ops(128)

            class _DsOrder:
                """Counts issued DS loads so waits name a producer instead of a raw dscnt."""

                LIMIT = 63

                def __init__(self):
                    self.issued = 0

                def mark(self, ops):
                    self.issued += ops
                    return self.issued

                def wait(self, mark):
                    rocdl.s_wait_dscnt(min(self.issued - mark, self.LIMIT))

            ds = _DsOrder()

            DS_A_OPS, DS_B_OPS = 2, 4  # ds_load_b128 per A / B fragment

            def _stage_load_a(stage, wm, k):
                """16 rows x 128 FP4 of the MoE preshuffle: two K32 slab pairs 512 B apart."""
                row_off = wm * A_M16_TILE + k * (WMMA_K // 2) * 16
                v0 = lds_load_b128(stage_a_addr[stage], row_off)
                v1 = lds_load_b128(stage_a_addr[stage], row_off + 512)
                return v0.shuffle(v1, list(range(8))), ds.mark(DS_A_OPS)

            def _stage_load_b(stage, wn, k):
                """32 rows x 128 FP4: two stacked 16-row blocks, low 8 words then high 8."""
                col_off = wn * 2 * B_LDS_ROW + k * (WMMA_K // 2) * 16
                v = [
                    lds_load_b128(
                        stage_b_addr[stage], col_off + blk * B_LDS_ROW + half
                    )
                    for blk in range_constexpr(2)
                    for half in (0, 512)
                ]
                lo = v[0].shuffle(v[1], list(range(8)))
                hi = v[2].shuffle(v[3], list(range(8)))
                return lo.shuffle(hi, list(range(16))), ds.mark(DS_B_OPS)

            def _stage_load_sa(stage, sm, k):
                return lds_load_b32(
                    stage_sa_addr[stage], (k * wmma_m_rep + sm * 2) * 16 * 4
                )[0], ds.mark(1)

            def _stage_load_sb(stage, sn, k):
                return lds_load_b32(
                    stage_sb_addr[stage], (sn * SC_WORDS + k * 32) * 4
                )[0], ds.mark(1)

            def _load_scales(load_scale, stage, first, count):
                return [
                    load_scale(stage, first + s, k)[0]
                    for s in range_constexpr(count)
                    for k in range_constexpr(K_STEPS)
                ]

            # Separate seed banks prevent WMMA source/address register coalescing.
            def _pipe(count, width):
                return [
                    [
                        [
                            fx.make_rmem_tensor(width, fx.Int32)
                            for _ in range_constexpr(K_STEPS)
                        ]
                        for _ in range_constexpr(count)
                    ]
                    for _ in range_constexpr(2)
                ]

            op_a = _Operand(
                _stage_load_a,
                _stage_load_sa,
                _pipe(half_m, 8),
                _pipe(a_seed_scale_rows, 1),
                half_m,
                a_seed_scale_rows,
                8,
                a_second_scale_rows,
            )
            op_b = _Operand(
                _stage_load_b,
                _stage_load_sb,
                _pipe(half_n, 16),
                _pipe(half_n, 1),
                half_n,
                half_n,
                16,
                half_n,
            )
            seed_mark = [0, 0]  # DS mark after which each bank's seeds are complete

            def _seed_scales(op, stage, bank):
                for s in range_constexpr(op.scale_rows):
                    for k in range_constexpr(K_STEPS):
                        op.scale_seeds[bank][s][k].store(
                            Vec.from_elements([op.load_scale(stage, s, k)[0]], fx.Int32)
                        )

            def _seed_frag(op, stage, bank, i):
                for k in range_constexpr(K_STEPS):
                    op.seeds[bank][i][k].store(op.load(stage, i, k)[0])

            def _load_seeds(stage, bank):
                for op in (op_a, op_b):
                    _seed_scales(op, stage, bank)
                    for i in range_constexpr(op.frags):
                        _seed_frag(op, stage, bank, i)

            def _seeded(op, bank):
                return op.seeds[bank], [
                    reg.load()[0] for row in op.scale_seeds[bank] for reg in row
                ]

            def _sched_hint(groups):
                """Ask the machine scheduler for one WMMA per DS burst across the produced group."""
                rocdl.sched_mfma(1)
                rocdl.sched_dsrd(2)
                for _ in range_constexpr(groups):
                    rocdl.sched_mfma(1)
                    rocdl.sched_dsrd(4)
                rocdl.sched_barrier(0)

            def _quadrant(
                wm0, wn0, act, wt, sa_k, sb_k, n_fast, need, produce, mid=None
            ):
                """One quadrant; producer p issues after WMMA p, `mid` after WMMA FENCE_WAIT_POS."""
                rocdl.sched_barrier(0)
                ds.wait(need)
                rocdl.sched_barrier(0)
                n_lead = len(produce)
                assert (
                    n_lead <= WMMA_PER_Q
                ), f"{n_lead} producers do not fit {WMMA_PER_Q} WMMA slots"
                for pos in range_constexpr(WMMA_PER_Q):
                    _mma_at(wm0, wn0, act, wt, sa_k, sb_k, pos, n_fast)
                    if const_expr(mid is not None and pos == FENCE_WAIT_POS):
                        mid()
                    if const_expr(pos < n_lead):
                        produce[pos]()
                    if const_expr(pos == n_lead - 1):
                        _sched_hint(n_lead - 1)
                rocdl.sched_barrier(0)

            def _noop():
                pass

            def _compute_stage(stage, future_kt, fence_outstanding, has_next, parity):
                """One scheduled K-tile stage: four quadrants, each hosting one DS producer group."""
                next_stage = (stage + 1) % num_buffers
                bank, next_bank = stage % 2, next_stage % 2
                a_top, sa_top = _seeded(op_a, bank)
                b_left, sb_left = _seeded(op_b, bank)
                a_bottom = [[None] * K_STEPS for _ in range_constexpr(half_m)]
                b_right = [[None] * K_STEPS for _ in range_constexpr(half_n)]
                sa_bottom, sb_right = [], []

                def _produce_second_half(op, frags, scales):
                    """This stage's bottom A or right B: its scales, then one fragment per WMMA slot."""
                    out = [
                        lambda: scales.extend(
                            _load_scales(
                                op.load_scale,
                                stage,
                                op.scale_rows,
                                op.second_scale_rows,
                            )
                        )
                    ]
                    for i in range_constexpr(op.frags):
                        for k in range_constexpr(K_STEPS):

                            def _go(i=i, k=k):
                                frags[i][k] = _rmem(
                                    op.width, op.load(stage, op.frags + i, k)[0]
                                )

                            out.append(_go)
                    return out

                def _produce_seeds(op):
                    """The next stage's seeds: scales, then one fragment per slot (no-ops on the last stage)."""

                    def _scales():
                        if const_expr(has_next):
                            _seed_scales(op, next_stage, next_bank)

                    def _frag(i):
                        if const_expr(has_next):
                            _seed_frag(op, next_stage, next_bank, i)

                    return [_scales] + [
                        lambda i=i: _frag(i) for i in range_constexpr(op.frags)
                    ]

                TL, TR = (0, 0), (0, half_n)
                BL, BR = (half_m, 0), (half_m, half_n)
                if const_expr(parity == 0):
                    plan = [
                        (
                            TL,
                            a_top,
                            b_left,
                            lambda: _produce_second_half(op_b, b_right, sb_right),
                        ),
                        (
                            TR,
                            a_top,
                            b_right,
                            lambda: _produce_second_half(op_a, a_bottom, sa_bottom),
                        ),
                        (BL, a_bottom, b_left, lambda: _produce_seeds(op_a)),
                        (BR, a_bottom, b_right, lambda: _produce_seeds(op_b)),
                    ]
                else:
                    plan = [
                        (
                            TL,
                            a_top,
                            b_left,
                            lambda: _produce_second_half(op_a, a_bottom, sa_bottom),
                        ),
                        (
                            BL,
                            a_bottom,
                            b_left,
                            lambda: _produce_second_half(op_b, b_right, sb_right),
                        ),
                        (TR, a_top, b_right, lambda: _produce_seeds(op_b)),
                        (BR, a_bottom, b_right, lambda: _produce_seeds(op_a)),
                    ]

                def _run(idx, need, mid=None, lead=()):
                    (wm0, wn0), act, wt, producers = plan[idx]
                    produce = list(lead) + producers()
                    _quadrant(
                        wm0,
                        wn0,
                        act,
                        wt,
                        sa_top + sa_bottom,
                        sb_left + sb_right,
                        idx > 0,
                        need,
                        produce,
                        mid,
                    )

                _run(0, seed_mark[bank])
                _run(1, ds.issued)

                # All of this stage's reads have landed: signal READY now, wait for the
                # siblings FENCE_WAIT_POS WMMAs into the third quadrant, then refill the slot.
                need, mid = ds.issued, None
                refill = const_expr(future_kt is not None)
                prepared = _prepare_tdm(stage, future_kt) if const_expr(refill) else None
                rocdl.sched_barrier(0)
                ds.wait(need)
                rocdl.sched_barrier(0)
                if const_expr(has_next):
                    pipeline_fence_signal(
                        outstanding=fence_outstanding, use_cluster=False
                    )

                    def mid():
                        rocdl.sched_barrier(0)
                        pipeline_fence_wait(use_cluster=False)
                        if const_expr(refill):
                            tdm_ops.tensor_load_2d(prepared)
                        rocdl.sched_barrier(0)

                _run(2, need, mid, lead=[_noop] * (FENCE_WAIT_POS + 1))
                _run(3, need)
                seed_mark[next_bank] = ds.issued

            # Keep all four slots in flight and statically expand one revolution.
            for i in range_constexpr(num_buffers):
                tdm_ops.tensor_load_2d(_prepare_tdm(i, fx.Int32(i)))
            pipeline_fence(outstanding=num_buffers - 1, use_cluster=False)
            _load_seeds(0, 0)
            seed_mark[0] = ds.issued
            n_full = (K_TILES - num_buffers) // num_buffers

            def _run_all(parity):
                for rev in range(n_full):
                    for s in range_constexpr(num_buffers):
                        _compute_stage(
                            s, (rev + 1) * num_buffers + s, num_buffers - 2, True, parity
                        )
                    if const_expr(cluster_sync_revs > 0):
                        do_sync = (rev % cluster_sync_revs) == (cluster_sync_revs - 1)
                        if do_sync & full_cluster:
                            cluster_sync()

                if const_expr(tail_mode == 2):
                    for s in range_constexpr(2):
                        _compute_stage(
                            s,
                            n_full * num_buffers + s + num_buffers,
                            num_buffers - 2,
                            True,
                            parity,
                        )

                for j in range_constexpr(num_buffers):
                    _compute_stage(
                        (tail_mode + j) % num_buffers,
                        None,
                        num_buffers - 2 - j,
                        j < num_buffers - 1,
                        parity,
                    )

            wave_parity = fx.Int32(rocdl.readfirstlane(T.i32, wave % 2))
            if wave_parity == 0:
                _run_all(0)
            else:
                _run_all(1)

            rocdl.s_wait_dscnt(0)
            accs = [c_frags[idx].load() for idx in range_constexpr(n_acc)]

            # The epilogue restages C in this arena. Draining our own tensorcnt
            # suffices: peer multicast loads are pairwise matched with ours.
            pipeline_fence(outstanding=0, use_cluster=False)
            # acc (wm, wn) is 32 N-cols x 16 M-rows: lane holds M = wm*16+lane16 and, per 8-wide
            # half, N = wn*32 + half*16 + kgrp*8 + v.
            for wm in range_constexpr(wmma_m_rep):
                row_rel = wmb + wm * 16 + lane16
                for wn in range_constexpr(wmma_n_rep):
                    acc = accs[wm * wmma_n_rep + wn]
                    for half in range_constexpr(2):
                        col_rel = wnb + wn * 32 + half * 16 + kgrp * 8
                        h = acc.shuffle(acc, list(range(half * 8, half * 8 + 8))).to(
                            partial_elem
                        )
                        fx.ptr_store(
                            h.bitcast(fx.Int8),
                            base_ptr + (row_rel * C_LDS_ROW + col_rel) * 2,
                        )
            workgroup_barrier(use_cluster=False)
            ldc64 = fx.Int64(i32_n)
            c_off_rt = blk_m64 * ldc64 + blk_n64
            gtC = _gv(
                fx.recast_iter(partial_elem, arg_c),
                c_off_rt,
                (tile_m, C_LDS_ROW),
                (C_LDS_ROW, 1),
            )
            atomC = fx.rocdl.make_tdm_atom(
                gtC, [mn_oob, tile_n], strides=[ldc64, None], num_warps=num_waves
            )
            fx.copy(
                atomC,
                _view(
                    fx.recast_iter(partial_elem, base_ptr),
                    (tile_m, C_LDS_ROW),
                    (C_LDS_ROW, 1),
                ),
                gtC,
            )
            tdm_ops.tensor_wait(0)

    m_tiles = ceildiv(i32_m, tile_m)
    n_tiles = ceildiv(N, tile_n)
    grid = (((m_tiles + cluster_m - 1) // cluster_m) * n_tiles, cluster_m, 1)
    kargs = (arg_c, arg_a, arg_b, arg_scale_a, arg_scale_b, arg_m_tile_map, i32_m, N)
    if cluster_m > 1 or cluster_n > 1:
        kernel_moe_gemm_a4w4_tuned(
            *kargs,
            value_attrs={"rocdl.cluster_dims": f"{cluster_n},{cluster_m},1"},
        ).launch(
            grid=grid,
            block=(block, 1, 1),
            stream=stream,
            cluster=(cluster_n, cluster_m, 1),
        )
    else:
        kernel_moe_gemm_a4w4_tuned(*kargs).launch(
            grid=grid, block=(block, 1, 1), stream=stream
        )


launch_moe_gemm_a4w4_tuned.compile_hints["llvm_options"] = {
    "amdgpu-expert-scheduling-mode": True,
}
