# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Fused split-K reduction shared by the K-split workgroups in one cluster."""

import flydsl.expr as fx
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr.rocdl import cluster
from flydsl.expr.typing import T

from .communication_ops_utils import traced
from .gemm_common_gfx1250 import workgroup_barrier
from .tensor_shim import buf_copy_load, ptr_buf_tensor

CPOL_DEVICE = 16
CPOL_STORE_DEVICE = CPOL_DEVICE | 3
VEC = 8
UNROLL = 32
MAX_PARTIAL_VECTORS = 128  # 512 dwords per load batch, including split-K 8.


@traced
def emit_fused_splitk_epilogue(
    *,
    elem,
    tid,
    block,
    tile_m,
    tile_n,
    lds_base_ptr,
    partials,
    out,
    c_off,
    ldc64,
    c_lds_row,
    split_k,
    split_idx,
    mn_oob,
    flat_tile,
):
    """Reduce disjoint row ranges using all of a tile's split-K workgroups.

    The launch clusters every K split of a tile together in grid.z, so the
    cluster barrier alone orders the exchange: no flags, no spinning, and no
    resident-grid precondition.  Each workgroup owns ``tile_m // split_k``
    rows, keeps its own stripe in LDS, and publishes only the peer rows --
    the caller has already rotated its LDS rows so those are contiguous.
    """
    reduce_m = tile_m // split_k
    reduce_row = split_idx * reduce_m
    lanes_per_row = tile_n // VEC
    rows_per_iter = block // lanes_per_row
    partial_vectors = MAX_PARTIAL_VECTORS // (block // 128)
    unroll = min(UNROLL, partial_vectors // split_k, reduce_m // rows_per_iter)
    row0 = tid // lanes_per_row
    col = (tid % lanes_per_row) * VEC
    plane = fx.Int64(tile_m * tile_n)
    partial_base = (
        fx.recast_iter(
            fx.PointerType.get(elem.ir_type, partials.address_space), partials
        )
        + fx.Int64(flat_tile) * split_k * plane
    )
    lds_out = fx.recast_iter(elem, lds_base_ptr)

    # The rotated peer rows occupy the prefix of each padded scratch plane.
    # Even on an M tail the whole prefix fits; the output store's M bound
    # discards the invalid logical rows.
    peer_m = tile_m - reduce_m
    partial_out = partial_base + fx.Int64(split_idx) * plane
    layout = fx.make_layout((peer_m, c_lds_row), (c_lds_row, 1))
    dst = fx.Tensor(fx.make_view(partial_out, layout))
    store = fx.rocdl.make_tdm_atom(
        dst,
        [peer_m, tile_n],
        strides=[fx.Int64(tile_n), None],
        num_warps=block // 32,
        cache_modifier=CPOL_STORE_DEVICE,
    )
    fx.copy(store, fx.Tensor(fx.make_view(lds_out, layout)), dst)
    fx.rocdl.tdm_ops.tensor_wait(0)
    cluster.cluster_barrier()

    output_base = (
        fx.recast_iter(fx.PointerType.get(elem.ir_type, out.address_space), out)
        + c_off
        + fx.Int64(reduce_row) * ldc64
    )
    # Select the cyclic peer stripe in each uniform buffer base. Keeping that
    # offset out of the per-vector indices avoids repeated vector arithmetic.
    buffers = [
        ptr_buf_tensor(
            partial_base
            + fx.Int64(s) * plane
            + fx.Int64((split_idx + split_k - s - 1) & (split_k - 1))
            * (reduce_m * tile_n),
            elem,
            unit_elems=VEC,
            num_records_bytes=fx.Int64(reduce_m * tile_n * 2),
        )
        for s in range_constexpr(split_k)
    ]
    for batch in range(reduce_m // (rows_per_iter * unroll)):
        partial_indices = [
            tid + (batch * unroll + u) * block for u in range_constexpr(unroll)
        ]
        local_parts = [
            fx.Vector(
                fx.ptr_load(
                    lds_out
                    + (peer_m + row0 + (batch * unroll + u) * rows_per_iter) * c_lds_row
                    + col,
                    result_type=T.vec(VEC, elem.ir_type),
                )
            )
            for u in range_constexpr(unroll)
        ]
        parts = [
            [fx.make_rmem_tensor(VEC, elem) for s in range_constexpr(split_k)]
            for u in range_constexpr(unroll)
        ]
        # Uniform branches skip the unpublished local stripe completely.
        # Stage all peer loads before converting or summing the fragments.
        for s in range_constexpr(split_k):
            for u in range_constexpr(unroll):
                parts[u][s].store(local_parts[u])
            if split_idx != fx.Int32(s):
                for u in range_constexpr(unroll):
                    parts[u][s].store(
                        buf_copy_load(
                            buffers[s],
                            partial_indices[u],
                            elem,
                            VEC,
                            cache_modifier=CPOL_DEVICE,
                        )
                    )
        for u in range_constexpr(unroll):
            acc = parts[u][0].load().extf(T.vec(VEC, T.f32))
            for s in range_constexpr(1, split_k):
                acc = acc + parts[u][s].load().extf(T.vec(VEC, T.f32))
            row = row0 + (batch * unroll + u) * rows_per_iter
            fx.ptr_store(acc.to(elem), lds_out + row * c_lds_row + col)

    workgroup_barrier(use_cluster=False)
    remaining = mn_oob - reduce_row
    valid_rows = (remaining > 0).select(remaining, fx.Int32(0))
    layout = fx.make_layout((reduce_m, c_lds_row), (c_lds_row, 1))
    tile_out = fx.Tensor(fx.make_view(output_base, layout))
    store_atom = fx.rocdl.make_tdm_atom(
        tile_out,
        [valid_rows, tile_n],
        strides=[ldc64, None],
        num_warps=block // 32,
    )
    fx.copy(store_atom, fx.Tensor(fx.make_view(lds_out, layout)), tile_out)
    fx.rocdl.tdm_ops.tensor_wait(0)
