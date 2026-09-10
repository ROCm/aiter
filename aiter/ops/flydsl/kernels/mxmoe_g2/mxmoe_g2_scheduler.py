# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2025-2026 FlyDSL Project Contributors
"""GEMM2 tile schedulers: persist_flat / one-shot (sp0 | spart) / persist-M."""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, rocdl
from flydsl.expr.typing import T

from ..mxfp4_gemm_common import _udiv, global_typed_ptr


def _flat_persistent_tile(pid, bound):
    """Round-robin a linear (M*N) work-unit id across 8 XCDs.

    persist_flat only; second-stage M-group swizzle omitted (g2 xcd_swizzle=0).
    """
    nxcd = fx.Int32(8)
    xc = fx.Int32(fx.Uint32(pid) % fx.Uint32(nxcd))
    xr = fx.Int32(fx.Uint32(bound) % fx.Uint32(nxcd))
    return xc * _udiv(bound, nxcd) + fx.min(xc, xr) + _udiv(pid, nxcd)


def spart_output_tile_index(
    block_1d_id, M0, N0, group_num, m01, nmajor=False, *, g2_spart=0
):
    """ck_tile GemmSpatiallyLocalTilePartitioner::GetOutputTileIndex.

    1D block id -> spatially-local (m_block_idx, n_block_idx).
    block_1d_id/M0 runtime; N0/group_num/m01/g2_spart compile-time.
    g2_spart<=0: identity N-fast linear mapping.
    """
    n0 = fx.Int32(N0)
    if const_expr(g2_spart <= 0):
        m_block_idx = _udiv(block_1d_id, n0)
        n_block_idx = block_1d_id - m_block_idx * n0
        return m_block_idx, n_block_idx

    gn = fx.Int32(group_num)
    m01c = fx.Int32(m01)

    # group_size = ceil(M0*N0 / GroupNum); big_group_num = GroupNum - (group_size*GroupNum - M0*N0)
    mn = M0 * n0
    group_size = _udiv(mn + gn - fx.Int32(1), gn)
    big_group_num = gn - (group_size * gn - mn)

    group_id_y = _udiv(block_1d_id, gn)
    group_id_x = block_1d_id - group_id_y * gn

    # remap = group_id_x <= big_group_num ? gx*gs + gy : gx*gs + big - gx + gy
    remap_a = group_id_x * group_size + group_id_y
    remap_b = group_id_x * group_size + big_group_num - group_id_x + group_id_y
    remap = (group_id_x <= big_group_num).select(remap_a, remap_b)

    if nmajor:
        if m01 != 1:
            raise AssertionError("nmajor requires m01==1")
        idx_N0 = _udiv(remap, M0)
        return remap - idx_N0 * M0, idx_N0

    idx_M0 = _udiv(remap, n0)
    idx_N0 = remap - idx_M0 * n0

    # M0_tmp = M0 / M01 ; M0_mod_M01 = M0 - M0_tmp*M01 ; M01_adapt = (idx_M0 < M0 - M0_mod) ? M01 : M0_mod
    M0_tmp = _udiv(M0, m01c)
    M0_mod = M0 - M0_tmp * m01c
    M01_adapt = (idx_M0 < (M0 - M0_mod)).select(m01c, M0_mod)

    idx_M00 = _udiv(idx_M0, m01c)
    idx_M01 = idx_M0 - idx_M00 * m01c
    idx_local = idx_N0 + idx_M01 * n0

    N_out = _udiv(idx_local, M01_adapt)
    loc_mod = idx_local - N_out * M01_adapt

    m_block_idx = loc_mod + idx_M00 * m01c
    n_block_idx = N_out
    return m_block_idx, n_block_idx


def g2_launch_grid_x(
    persist_flat, i32_max_m_blocks, i32_grid_blocks, num_n_blocks, cu_num
):
    """Host-side grid.x: persist_flat uses a CU-sized grid when work is large."""
    if const_expr(persist_flat):
        total_work = i32_max_m_blocks * num_n_blocks
        return (total_work > fx.Int32(4 * cu_num)).select(
            fx.Int32(cu_num), total_work
        )
    return i32_grid_blocks * num_n_blocks


@flyc.jit
def schedule_g2_tiles(
    bx_i32,
    arg_cumsum,
    num_n_blocks,
    *,
    persist_flat,
    persist,
    g2_spart,
    BM,
    cu_num,
    g2_group_num,
    g2_m01,
    issue_all_a_loads,
    run_unit,
    issue_and_run,
):
    """Pick one compile-time tile schedule and issue A + run_unit for this block."""
    if const_expr(persist_flat):
        # Flat persist-M+N: grid-stride all (m,n) tiles; 8-XCD remap only (g2 xcd_swizzle=0).
        cumsum0 = global_typed_ptr(arg_cumsum, T.i32)[0]
        bound = _udiv(cumsum0, BM) * fx.Int32(num_n_blocks)
        grid_nb = fx.Int32(gpu.grid_dim.x)
        if bx_i32 < bound:
            unit_bx = _flat_persistent_tile(bx_i32, bound)
            issue_and_run(unit_bx, _udiv(unit_bx, num_n_blocks))
        for iv in range(bx_i32 + grid_nb, bound, gpu.grid_dim.x):
            gpu.barrier()
            unit_bx = _flat_persistent_tile(fx.Int32(iv), bound)
            issue_and_run(unit_bx, _udiv(unit_bx, num_n_blocks))
    elif const_expr(not persist):
        # One-shot: one block, one (m,n) tile. sp0 is identity; else remap.
        # issue_and_run wraps issue_all_a_loads + run_unit (A after cumsum).
        cumsum0 = global_typed_ptr(arg_cumsum, T.i32)[0]
        total_m_blocks = _udiv(cumsum0, BM)
        bound = total_m_blocks * fx.Int32(num_n_blocks)

        if fx.Int32(bx_i32) < bound:
            m_block_idx, n_block_idx = spart_output_tile_index(
                bx_i32,
                total_m_blocks,
                num_n_blocks,
                g2_group_num,
                g2_m01,
                g2_spart=g2_spart,
            )
            issue_and_run(
                m_block_idx * fx.Int32(num_n_blocks) + n_block_idx,
                m_block_idx,
                mn_idx=(m_block_idx, n_block_idx),
            )
    else:
        # Persistent-m: fixed cu_num*num_n_blocks grid; each block grid-strides m-tiles by cu_num (aiter `_persist`).
        m_tile0 = _udiv(bx_i32, num_n_blocks)
        n_block = bx_i32 - m_tile0 * fx.Int32(num_n_blocks)
        c_stride = fx.Int32(cu_num)

        cumsum0 = global_typed_ptr(arg_cumsum, T.i32)[0]
        total_m_blocks = _udiv(cumsum0, BM)
        # ceil((total_m_blocks - m_tile0) / cu_num), clamped to 0 when m_tile0 >= total_m_blocks.
        diff = total_m_blocks - m_tile0
        rem = (diff > fx.Int32(0)).select(diff, fx.Int32(0))
        n_iters = _udiv(rem + c_stride - fx.Int32(1), c_stride)
        for _it in range(
            fx.Int32(0),
            n_iters,
            fx.Int32(1),
        ):
            m_block = m_tile0 + fx.Int32(_it) * c_stride
            unit_bx = m_block * fx.Int32(num_n_blocks) + n_block
            gpu.barrier()  # persist: separate prev-iter epilog C-slab LDS reads from this iter's A-load into the shared LDS union
            issue_all_a_loads(m_block * fx.Int32(BM))
            rocdl.sched_barrier(0)
            if fx.Int32(m_block) < total_m_blocks:
                run_unit(unit_bx)
