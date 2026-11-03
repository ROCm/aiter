# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
# Vendored from FlyDSL kernels/gemm/gemm_a4w4_256x256_gfx1250.py.

"""A4W4 (MXFP4 E2M1 activation x MXFP4 E2M1 weight) 32x4-preshuffled MXscale GEMM for gfx1250."""

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
    make_gemm_grid,
    make_lds_copy_ops,
    pipeline_fence,
    pipeline_fence_signal,
    pipeline_fence_wait,
    workgroup_barrier,
)
from .gfx1250_cluster import compute_mcast_masks
from .kernels_common import format_kernel_name
from .splitk_fused_epilogue_gfx1250 import emit_fused_splitk_epilogue


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
def launch_gemm_a4w4_256x256(
    arg_c: fx.Pointer,
    arg_a: fx.Pointer,
    arg_b: fx.Pointer,
    arg_scale_a: fx.Pointer,
    arg_scale_b: fx.Pointer,
    i32_m: fx.Int32,
    stream: fx.Stream,
    N: fx.Int32,
    K: fx.Int32,
    i32_lda: fx.Int32,
    i32_ldc: fx.Int32,
    arg_out: fx.Pointer,
    tile_m: Constexpr[int],
    tile_n: Constexpr[int],
    tile_k: Constexpr[int],
    m_warp: Constexpr[int],
    n_warp: Constexpr[int],
    out_is_f16: Constexpr[int],
    num_buffers: Constexpr[int],
    cluster_m: Constexpr[int],
    cluster_n: Constexpr[int],
    split_k: Constexpr[int] = 1,
    a_preshuffle: Constexpr[bool] = False,
    fused_splitk: Constexpr[bool] = False,
    bounded_m: Constexpr[bool] = True,
):
    """Launch the A4W4 kernel.

    N must cover whole ``tile_n * cluster_n`` groups. M is unrestricted;
    preshuffled A must pad an odd M to a whole row pair. K per split must be a
    multiple of tile_k. Fused split-K requires 2/4/8 splits in one hardware cluster.
    """

    assert (
        tile_m in (128, 192, 256)
        and tile_n in (128, 256)
        and tile_k in (128, 256)
        and (m_warp, n_warp, num_buffers) == (2, 2, 4)
    ), "supported tile profiles are 128/192/256 x 128/256 x 128/256 with 2x2 waves and 4 buffers"
    assert (
        1 <= cluster_m < 16 and 1 <= cluster_n < 16 and 1 < cluster_m * cluster_n <= 16
    ), f"cluster_m*cluster_n must be 2..16, got {cluster_m}x{cluster_n}"
    assert split_k >= 1, f"split_k must be positive, got {split_k}"
    assert not fused_splitk or (
        split_k in (2, 4, 8) and tile_m % split_k == 0
    ), "fused split-K needs 2/4/8 splits dividing tile_m"
    # The fused split-K row rotation below masks with tile_m - 1.
    assert (
        not fused_splitk or tile_m & (tile_m - 1) == 0
    ), "fused split-K needs a 2^n tile_m"
    cluster_splitk = fused_splitk and split_k > 1
    cluster_k = split_k if cluster_splitk else 1
    assert (
        cluster_m * cluster_n * cluster_k <= 16
    ), "all K splits must fit in a 16-workgroup cluster"
    cluster_sync_revs = 8
    # The four-quadrant software pipeline; other profiles run the sequential K-tile schedule.
    scheduled = tile_m in (192, 256) and tile_n == 256 and tile_k == 256
    FENCE_WAIT_POS = 7  # WMMAs of slack between the READY signal and its matching wait
    WMMA_M = 32
    WMMA_N = 16
    WMMA_K = 128
    WAVE = fx.num_warp_threads()
    PACK_TK = tile_k // 2  # A/B row bytes per K-tile (FP4 packed 2/byte)
    K_STEPS = tile_k // WMMA_K
    SC_WORDS = tile_k // 4  # scale i32 words per super-row per K-tile
    SA_SUPERS = tile_m // 32
    SB_SUPERS = tile_n // 32
    warp_tile_m = tile_m // m_warp
    warp_tile_n = tile_n // n_warp
    wmma_m_rep = warp_tile_m // WMMA_N  # A fragments (16 rows each)
    wmma_n_rep = warp_tile_n // WMMA_M  # B fragments (32 rows each)
    half_m, half_n = wmma_m_rep // 2, wmma_n_rep // 2
    WMMA_PER_Q = half_m * half_n * K_STEPS  # WMMAs in one quadrant
    # One A scale word covers two 16-row fragments. The seed half takes every scale row its
    # fragments touch; with an odd half_m (a 96-row warp tile) the middle row is shared and the
    # second half loads only the rows past it.
    a_seed_scale_rows = (half_m + 1) // 2
    a_second_scale_rows = wmma_m_rep // 2 - a_seed_scale_rows
    n_acc = wmma_m_rep * wmma_n_rep
    num_waves = m_warp * n_warp
    block = num_waves * WAVE
    LDS_PAD_A = 16
    A_SHUFFLE_BYTES = 128  # the shuffle's major tile, in bytes -- not a K-element count
    A_PAIR = 2 if a_preshuffle else 1
    assert not a_preshuffle or tile_m % 2 == 0, "a_preshuffle needs an even tile_m"
    assert (
        not a_preshuffle or PACK_TK == A_SHUFFLE_BYTES
    ), f"a_preshuffle needs a {A_SHUFFLE_BYTES}-byte A row per K-tile, got {PACK_TK}"
    A_LDS_ROWS = tile_m // A_PAIR
    A_TDM_ROW = A_PAIR * PACK_TK
    A_LDS_ROW = A_TDM_ROW + LDS_PAD_A
    C_LDS_ROW = tile_n + 8
    B_LDS_ROW = PACK_TK * 16
    STAGE_A = A_LDS_ROWS * A_LDS_ROW
    STAGE_B = (tile_n // 16) * B_LDS_ROW
    STAGE_SA = SA_SUPERS * tile_k
    STAGE_SB = SB_SUPERS * tile_k
    # Operand-major planar LDS: scales and A below, B on the next 64-KiB boundary above them.
    PLANAR_SA_BASE = 0
    PLANAR_A_BASE = PLANAR_SA_BASE + num_buffers * STAGE_SA
    PLANAR_SB_BASE = PLANAR_A_BASE + num_buffers * STAGE_A
    PLANAR_B_BASE = ((PLANAR_SB_BASE + num_buffers * STAGE_SB + 65535) // 65536) * 65536
    PLANAR_END = PLANAR_B_BASE + num_buffers * STAGE_B

    partial_elem = (
        fx.Float32 if split_k > 1 else (fx.Float16 if out_is_f16 else fx.BFloat16)
    )
    partial_bytes = 4 if split_k > 1 else 2
    ARENA_B = max(PLANAR_END, tile_m * C_LDS_ROW * partial_bytes)
    lds_cap = get_lds_capacity_bytes(str(get_hip_arch()))
    if ARENA_B > lds_cap:
        raise RuntimeError(
            f"Shared Memory Overflow: requested {ARENA_B} bytes on "
            f"{get_hip_arch()}, limit is {lds_cap} bytes"
        )
    kernel_name = format_kernel_name(
        f"gemm_a4w4_mx32_compute_t{tile_m}x{tile_n}x{tile_k}"
        f"_mw{m_warp}_nw{n_warp}_nb{num_buffers}_sk{split_k}"
        f"_cm{cluster_m}_cn{cluster_n}"
        + ("_apre" if a_preshuffle else "")
        + ("_fsk" if cluster_splitk else "")
    )

    @flyc.kernel(name=kernel_name, known_block_size=[block, 1, 1])
    def kernel_gemm_a4w4_256x256(
        arg_c: fx.Pointer,
        arg_a: fx.Pointer,
        arg_b: fx.Pointer,
        arg_scale_a: fx.Pointer,
        arg_scale_b: fx.Pointer,
        i32_m: fx.Int32,
        i32_n: fx.Int32,
        i32_k: fx.Int32,
        i32_lda: fx.Int32,
        i32_ldc: fx.Int32,
        arg_out: fx.Pointer,
        tail_mode: Constexpr[int],
    ):

        K_TILES = i32_k // (tile_k * split_k)
        k64 = fx.Int64(i32_k)
        i32_lda_b = i32_lda // 2  # A row stride in bytes (FP4 packed)
        lda64 = fx.Int64(i32_lda_b)
        ldc64 = fx.Int64(i32_ldc)
        Kp16 = (k64 // 2) * 16

        tid = fx.thread_idx.x
        bid_x, bid_y, bid_z = fx.block_idx
        if const_expr(split_k > 1):
            m_chunks = fx.grid_dim.z // split_k
            split_idx = bid_z // m_chunks
            m_chunk = bid_z - split_idx * m_chunks
            kt_base = fx.Int64(split_idx) * fx.Int64(K_TILES)
        else:
            split_idx = fx.Int32(0)
            m_chunk = bid_z
            kt_base = fx.Int64(0)
        wave = fx.Int32(rocdl.readfirstlane(T.i32, tid // WAVE))
        lane = tid % WAVE
        lane16 = lane % 16
        kgrp = lane // 16
        wave_m = wave // n_warp
        wave_n = wave % n_warp
        local_x, local_y = cluster.compute_cluster_position()
        a_mask, b_mask = compute_mcast_masks(local_x, local_y, cluster_m, cluster_n)
        if const_expr(cluster_splitk):
            cluster_plane = fx.Int32(rocdl.cluster_workgroup_id_z()) * (
                cluster_m * cluster_n
            )
            a_mask = fx.Int32(a_mask) << cluster_plane
            b_mask = fx.Int32(b_mask) << cluster_plane
        blk_m = (m_chunk * fx.grid_dim.x + bid_x) * tile_m
        blk_n = bid_y * tile_n
        blk_m64 = fx.Int64(blk_m)
        blk_n64 = fx.Int64(blk_n)
        mn_oob = i32_m - blk_m  # valid M rows (A / C)
        a_oob = (mn_oob + 1) >> 1 if const_expr(a_preshuffle) else mn_oob
        sa_oob = (i32_m + 31) // 32 - blk_m // 32  # valid M-supers (scale-A)

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

        oc = fx.Float16 if out_is_f16 else fx.BFloat16
        if const_expr(cluster_splitk):
            flat_tile = bid_y * fx.Int32(fx.grid_dim.x) * (
                fx.Int32(fx.grid_dim.z) // split_k
            ) + (m_chunk * fx.Int32(fx.grid_dim.x) + bid_x)
        gA_base = fx.recast_iter(fx.Int8, arg_a)
        gB_base = fx.recast_iter(fx.Int8, arg_b)

        k_elem0 = kt_base * tile_k
        a_off0 = blk_m64 * lda64 + kt_base * (PACK_TK * A_PAIR)
        b_off0 = (blk_n64 // 16) * Kp16 + k_elem0 * 8
        sa_off0 = (blk_m64 // 32) * k64 + k_elem0
        sb_off0 = (blk_n64 // 32) * k64 + k_elem0

        gA = _gv(gA_base, a_off0, (A_LDS_ROWS, A_TDM_ROW), (A_TDM_ROW, 1))
        gB = _gv(gB_base, b_off0, (tile_n // 16, B_LDS_ROW), (B_LDS_ROW, 1))
        gSA = _gv(arg_scale_a, sa_off0, (SA_SUPERS, tile_k), (tile_k, 1))
        gSB = _gv(arg_scale_b, sb_off0, (SB_SUPERS, tile_k), (tile_k, 1))

        def _build_tdm_desc(owner):
            if const_expr(owner == 0):
                tensor, offset, shape, lds_stride = (
                    gA,
                    PLANAR_A_BASE,
                    (A_LDS_ROWS, A_TDM_ROW),
                    A_LDS_ROW,
                )
                stride, mask, bound, pad, early = (
                    i32_lda_b * A_PAIR,
                    a_mask,
                    a_oob,
                    LDS_PAD_A,
                    False,
                )
            elif const_expr(owner == 1):
                tensor, offset, shape, lds_stride = (
                    gB,
                    PLANAR_B_BASE,
                    (tile_n // 16, B_LDS_ROW),
                    B_LDS_ROW,
                )
                stride, mask, bound, pad, early = i32_k * 8, b_mask, None, 0, False
            elif const_expr(owner == 2):
                tensor, offset, shape, lds_stride = (
                    gSA,
                    PLANAR_SA_BASE,
                    (SA_SUPERS, tile_k),
                    tile_k,
                )
                stride, mask, bound, pad, early = i32_k, a_mask, sa_oob, 0, True
            else:
                tensor, offset, shape, lds_stride = (
                    gSB,
                    PLANAR_SB_BASE,
                    (SB_SUPERS, tile_k),
                    tile_k,
                )
                stride, mask, bound, pad, early = i32_k, b_mask, None, 0, True
            desc = tdm_ops.make_tensor_descriptor_2d(
                global_ptr=tensor,
                lds_memref=_view(
                    fx.add_offset(base_ptr, offset), shape, (lds_stride, 1)
                ),
                global_offset=(0, 0),
                tensor_shape=shape,
                strides=(stride, 1),
                tile_shape=shape,
                elem_bytes=1,
                pad_interval=shape[1] if pad else 0,
                pad_amount=pad,
                num_warps=1,
                workgroup_mask=mask,
                early_timeout=early,
                oob_outer_bound=bound,
            )
            return (
                Vec(desc.dgroup0),
                Vec(desc.dgroup1),
                fx.Int32(shape[0] * lds_stride),
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

        tdm_desc = tdm_ops.TDMDescriptor2D(as_ir_value(dgroup0), as_ir_value(dgroup1))
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
        c_frags = [fx.make_rmem_tensor(16, fx.Float32) for _ in range_constexpr(n_acc)]
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

        cluster.cluster_barrier()
        # Keep fragment displacements as DS immediates inside the K loop.
        stage_a_addr, stage_b_addr, stage_sa_addr, stage_sb_addr = [], [], [], []
        sa_row, sb_col = wmb + lane, wnb + lane
        if const_expr(a_preshuffle):
            a_pair = wave_m * (warp_tile_m // 2) + (lane16 >> 1)
            a_byte = fx.Int64(
                a_pair * A_LDS_ROW + (lane16 & 1) * A_SHUFFLE_BYTES + kgrp * 16
            )
            a_frag_row_step = (16 // A_PAIR) * A_LDS_ROW
        else:
            a_byte = fx.Int64((wmb + lane16) * A_LDS_ROW + kgrp * 16)
            a_frag_row_step = 16 * A_LDS_ROW
        b_byte = fx.Int64((wnb // 16) * B_LDS_ROW + kgrp * 256 + lane16 * 16)
        sa_byte = fx.Int64(((sa_row // 32) * SC_WORDS + sa_row % 32) * 4)
        sb_byte = fx.Int64(((sb_col // 32) * SC_WORDS + sb_col % 32) * 4)
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
            """16 rows x 128 FP4: lane holds row wm*16+lane16, K-bytes kgrp*16 + 32*j."""
            row_off = wm * a_frag_row_step + k * (WMMA_K // 2)
            v0 = lds_load_b128(stage_a_addr[stage], row_off)
            v1 = lds_load_b128(stage_a_addr[stage], row_off + 32)
            return v0.shuffle(v1, list(range(8))), ds.mark(DS_A_OPS)

        def _stage_load_b(stage, wn, k):
            """32 rows x 128 FP4: two stacked 16-row blocks, low 8 words then high 8."""
            col_off = wn * 2 * B_LDS_ROW + k * (WMMA_K // 2) * 16
            v = [
                lds_load_b128(stage_b_addr[stage], col_off + blk * B_LDS_ROW + half)
                for blk in range_constexpr(2)
                for half in (0, 512)
            ]
            lo = v[0].shuffle(v[1], list(range(8)))
            hi = v[2].shuffle(v[3], list(range(8)))
            return lo.shuffle(hi, list(range(16))), ds.mark(DS_B_OPS)

        def _stage_load_sa(stage, sm, k):
            return lds_load_b32(stage_sa_addr[stage], (sm * SC_WORDS + k * 32) * 4)[
                0
            ], ds.mark(1)

        def _stage_load_sb(stage, sn, k):
            return lds_load_b32(stage_sb_addr[stage], (sn * SC_WORDS + k * 32) * 4)[
                0
            ], ds.mark(1)

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

        def _quadrant(wm0, wn0, act, wt, sa_k, sb_k, n_fast, need, produce, mid=None):
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

        def _compute_generic_tile(stage):
            """Sequential K-tile schedule for the smaller M/N and K profiles."""
            act = [
                [
                    _rmem(8, _stage_load_a(stage, wm, k)[0])
                    for k in range_constexpr(K_STEPS)
                ]
                for wm in range_constexpr(wmma_m_rep)
            ]
            wt = [
                [
                    _rmem(16, _stage_load_b(stage, wn, k)[0])
                    for k in range_constexpr(K_STEPS)
                ]
                for wn in range_constexpr(wmma_n_rep)
            ]
            sa_k = _load_scales(_stage_load_sa, stage, 0, wmma_m_rep // 2)
            sb_k = _load_scales(_stage_load_sb, stage, 0, wmma_n_rep)
            rocdl.s_wait_dscnt(0)
            for wm in range_constexpr(wmma_m_rep):
                for wn in range_constexpr(wmma_n_rep):
                    for k in range_constexpr(K_STEPS):
                        _mma_one(wm, wn, k, act[wm][k], wt[wn][k], sa_k, sb_k)

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
                            op.load_scale, stage, op.scale_rows, op.second_scale_rows
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
                pipeline_fence_signal(outstanding=fence_outstanding, use_cluster=False)

                def mid():
                    rocdl.sched_barrier(0)
                    pipeline_fence_wait(use_cluster=False)
                    if const_expr(refill):
                        tdm_ops.tensor_load_2d(prepared)
                    rocdl.sched_barrier(0)

            _run(2, need, mid, lead=[_noop] * (FENCE_WAIT_POS + 1))
            _run(3, need)
            seed_mark[next_bank] = ds.issued

        if const_expr(scheduled):
            # Keep all four slots in flight and statically expand one revolution.
            # tail_mode: K tiles left after the full revolutions (0 or 2), or -1 for any count.
            last_kt = K_TILES - 1
            for i in range_constexpr(num_buffers):
                seed_kt = fx.Int32(i)
                if const_expr(tail_mode == -1):
                    seed_kt = (seed_kt < last_kt).select(seed_kt, last_kt)
                tdm_ops.tensor_load_2d(_prepare_tdm(i, seed_kt))
            pipeline_fence(outstanding=num_buffers - 1, use_cluster=False)
            _load_seeds(0, 0)
            seed_mark[0] = ds.issued

            if const_expr(tail_mode == -1):
                n_full = (K_TILES + num_buffers - 1) // num_buffers - 1
                drain_s = K_TILES - n_full * num_buffers
            else:
                n_full = (K_TILES - num_buffers) // num_buffers

            def _run_all(parity):
                for rev in range(n_full):
                    do_sync = (rev % cluster_sync_revs) == (cluster_sync_revs - 1)
                    for s in range_constexpr(num_buffers):
                        future_kt = (rev + 1) * num_buffers + s
                        if const_expr(tail_mode == -1):
                            future_kt = (future_kt < last_kt).select(future_kt, last_kt)
                        _compute_stage(s, future_kt, num_buffers - 2, True, parity)
                    if do_sync:
                        cluster.cluster_barrier()

                if const_expr(tail_mode == 2):
                    # K is runtime; this fixed schedule serves all lengths with
                    # two tiles beyond a full four-slot revolution.
                    for s in range_constexpr(2):
                        _compute_stage(
                            s,
                            n_full * num_buffers + s + num_buffers,
                            num_buffers - 2,
                            True,
                            parity,
                        )

                for j in range_constexpr(num_buffers):
                    stage = (2 + j) % num_buffers if const_expr(tail_mode == 2) else j
                    valid = j < drain_s if const_expr(tail_mode == -1) else True
                    if valid:
                        _compute_stage(
                            stage,
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
        else:
            # Each group fills distinct LDS slots. Peers meet only before the
            # next group can reuse those slots for another multicast.
            for group in range((K_TILES + num_buffers - 1) // num_buffers):
                for slot in range_constexpr(num_buffers):
                    kt = group * num_buffers + slot
                    if kt < K_TILES:
                        tdm_ops.tensor_load_2d(_prepare_tdm(slot, kt))
                pipeline_fence(outstanding=0, use_cluster=False)
                for slot in range_constexpr(num_buffers):
                    if group * num_buffers + slot < K_TILES:
                        _compute_generic_tile(slot)
                cluster.cluster_barrier()

        rocdl.s_wait_dscnt(0)
        accs = [c_frags[idx].load() for idx in range_constexpr(n_acc)]

        pipeline_fence(outstanding=0, use_cluster=True)
        # acc (wm, wn) is 32 N-cols x 16 M-rows: lane holds M = wm*16+lane16 and, per 8-wide
        # half, N = wn*32 + half*16 + kgrp*8 + v.
        for wm in range_constexpr(wmma_m_rep):
            row_rel = wmb + wm * 16 + lane16
            if const_expr(cluster_splitk):
                # Place local rows after all peer rows for the shared reducer.
                row_rel = (row_rel + tile_m - (split_idx + 1) * (tile_m // split_k)) & (
                    tile_m - 1
                )
            for wn in range_constexpr(wmma_n_rep):
                acc = accs[wm * wmma_n_rep + wn]
                for half in range_constexpr(2):
                    col_rel = wnb + wn * 32 + half * 16 + kgrp * 8
                    h = acc.shuffle(acc, list(range(half * 8, half * 8 + 8))).to(
                        partial_elem
                    )
                    fx.ptr_store(
                        h.bitcast(fx.Int8),
                        base_ptr + (row_rel * C_LDS_ROW + col_rel) * partial_bytes,
                    )
        workgroup_barrier(use_cluster=False)
        c_off_rt = blk_m64 * ldc64 + blk_n64
        if const_expr(cluster_splitk):
            emit_fused_splitk_epilogue(
                elem=oc,
                tid=tid,
                block=block,
                tile_m=tile_m,
                tile_n=tile_n,
                lds_base_ptr=base_ptr,
                partials=arg_c,
                out=arg_out,
                c_off=c_off_rt,
                ldc64=ldc64,
                c_lds_row=C_LDS_ROW,
                split_k=split_k,
                split_idx=split_idx,
                mn_oob=mn_oob,
                flat_tile=flat_tile,
                bounded_m=bounded_m,
            )
        else:
            if const_expr(split_k > 1):
                c_off_rt += fx.Int64(split_idx) * fx.Int64(i32_m) * ldc64
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

    grid_arg = make_gemm_grid(
        i32_m,
        N,
        tile_m,
        tile_n,
        cluster_m,
        split_k=split_k,
        fused_splitk=cluster_splitk,
    )

    # Runtime N/K shape checks belong to the caller. These schedule variants
    # depend on the K-tile remainder, never on K's value in the JIT cache key.
    def _launch(tail_mode):
        kernel_gemm_a4w4_256x256(
            arg_c,
            arg_a,
            arg_b,
            arg_scale_a,
            arg_scale_b,
            i32_m,
            N,
            K,
            i32_lda,
            i32_ldc,
            arg_out,
            tail_mode,
            value_attrs={"rocdl.cluster_dims": f"{cluster_m},{cluster_n},{cluster_k}"},
        ).launch(
            grid=grid_arg,
            block=(block, 1, 1),
            stream=stream,
            cluster=(cluster_m, cluster_n, cluster_k),
        )

    if const_expr(scheduled):
        per_split_tiles = K // (tile_k * split_k)
        if (per_split_tiles >= num_buffers) & (per_split_tiles % num_buffers == 2):
            _launch(2)
        elif (per_split_tiles >= num_buffers) & (per_split_tiles % num_buffers == 0):
            _launch(0)
        else:
            _launch(-1)
    else:
        _launch(-1)


launch_gemm_a4w4_256x256.compile_hints["llvm_options"] = {
    "amdgpu-expert-scheduling-mode": True,
}
