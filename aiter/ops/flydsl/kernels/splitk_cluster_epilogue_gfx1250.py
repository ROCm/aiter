# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Experimental reduction shared by the K-split workgroups in one cluster."""

import flydsl.expr as fx
from flydsl.expr import range_constexpr
from flydsl.expr.rocdl import cluster
from flydsl.expr.typing import T

from .communication_ops_utils import traced
from .gemm_common_gfx1250 import workgroup_barrier
from .splitk_epilogue_gfx1250 import CPOL_DEVICE, MAX_PARTIAL_VECTORS, UNROLL, VEC
from .tensor_shim import buf_copy_load, ptr_buf_tensor


@traced
def emit_cluster_splitk_reduce_epilogue(
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

    The launch must cluster every K split together in grid.z. Each wave has
    waited for its device-scope partial store before entering here. The
    cluster barrier publishes all partials without counters or polling.
    """
    cluster.cluster_barrier()
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
    output_base = (
        fx.recast_iter(fx.PointerType.get(elem.ir_type, out.address_space), out)
        + c_off
        + fx.Int64(reduce_row) * ldc64
    )
    buffers = [
        ptr_buf_tensor(
            partial_base + fx.Int64(s) * plane,
            elem,
            unit_elems=VEC,
            num_records_bytes=fx.Int64(mn_oob * tile_n * 2),
        )
        for s in range_constexpr(split_k)
    ]
    lds_out = fx.recast_iter(elem, lds_base_ptr)
    for batch in range(reduce_m // (rows_per_iter * unroll)):
        partial_indices = [
            tid + (batch * unroll + u) * block + reduce_row * lanes_per_row
            for u in range_constexpr(unroll)
        ]
        parts = [
            [
                buf_copy_load(buffers[s], idx, elem, VEC, cache_modifier=CPOL_DEVICE)
                for s in range_constexpr(split_k)
            ]
            for idx in partial_indices
        ]
        for u in range_constexpr(unroll):
            acc = parts[u][0].extf(T.vec(VEC, T.f32))
            for s in range_constexpr(1, split_k):
                acc = acc + parts[u][s].extf(T.vec(VEC, T.f32))
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
