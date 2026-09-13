# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Reduction shared by the K-split workgroups in one cluster."""

import flydsl.expr as fx
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr.rocdl import cluster
from flydsl.expr.typing import T

from .communication_ops_utils import traced
from .gemm_common_gfx1250 import workgroup_barrier
from .splitk_epilogue_gfx1250 import (
    CPOL_DEVICE,
    CPOL_STORE_DEVICE,
    MAX_PARTIAL_VECTORS,
    UNROLL,
    VEC,
)
from .tensor_shim import buf_copy_load, buf_copy_store, ptr_buf_tensor


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
    bounded_m=False,
    reuse_lds=False,
):
    """Reduce disjoint row ranges using all of a tile's split-K workgroups.

    The launch must cluster every K split together in grid.z. The caller has
    either published the full partial or retained it in LDS for ``reuse_lds``.
    In that mode the caller cyclically rotates the LDS rows so the local row
    stripe is last. Peer rows are published by one contiguous TDM store, and
    the local rounded partial stays in LDS. The reduced stripe is written
    directly to global memory. Both modes preserve the split-order FP32 sum.
    """
    reduce_m = tile_m // split_k
    reduce_row = split_idx * reduce_m
    lanes_per_row = tile_n // VEC
    rows_per_iter = block // lanes_per_row
    partial_vectors = MAX_PARTIAL_VECTORS // (block // 128)
    unroll = min(UNROLL, partial_vectors // split_k, reduce_m // rows_per_iter)
    row0 = tid // lanes_per_row
    col = (tid % lanes_per_row) * VEC
    peer_m = tile_m - reduce_m
    plane = fx.Int64((peer_m if reuse_lds else tile_m) * tile_n)
    partial_base = (
        fx.recast_iter(
            fx.PointerType.get(elem.ir_type, partials.address_space), partials
        )
        + fx.Int64(flat_tile) * split_k * plane
    )
    lds_out = fx.recast_iter(elem, lds_base_ptr)
    if const_expr(reuse_lds):
        # Each scratch plane stores just the rotated peer rows, including on an
        # M tail. The final output descriptor discards invalid logical rows.
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
    remaining = mn_oob - reduce_row
    valid_rows = (remaining > 0).select(remaining, fx.Int32(0))
    if const_expr(reuse_lds and bounded_m):
        output_buffer = ptr_buf_tensor(
            output_base,
            elem,
            unit_elems=VEC,
            num_records_bytes=fx.Int64(valid_rows) * ldc64 * 2,
        )
    # Select the cyclic peer stripe in each uniform buffer base. Keeping that
    # offset out of the per-vector indices avoids repeated vector arithmetic.
    buffers = [
        ptr_buf_tensor(
            partial_base
            + fx.Int64(s) * plane
            + (
                fx.Int64((split_idx + split_k - s - 1) & (split_k - 1))
                * (reduce_m * tile_n)
                if const_expr(reuse_lds)
                else 0
            ),
            elem,
            unit_elems=VEC,
            num_records_bytes=(
                fx.Int64(reduce_m * tile_n * 2)
                if const_expr(reuse_lds)
                else fx.Int64(mn_oob * tile_n * 2)
            ),
        )
        for s in range_constexpr(split_k)
    ]
    if const_expr(reuse_lds and split_k == 2):
        peer_buffer = ptr_buf_tensor(
            partial_base + fx.Int64(1 - split_idx) * plane,
            elem,
            unit_elems=VEC,
            num_records_bytes=fx.Int64(reduce_m * tile_n * 2),
        )
    for batch in range(reduce_m // (rows_per_iter * unroll)):
        partial_indices = [
            tid
            + (batch * unroll + u) * block
            + (0 if const_expr(reuse_lds) else reduce_row * lanes_per_row)
            for u in range_constexpr(unroll)
        ]
        if const_expr(reuse_lds and split_k == 2):
            # With two splits, both peer stripes begin at offset zero. The
            # two-term FP32 sum is commutative, so no split-index branch is needed.
            peer_parts = [
                buf_copy_load(peer_buffer, idx, elem, VEC, cache_modifier=CPOL_DEVICE)
                for idx in partial_indices
            ]
            local_parts = [
                fx.Vector(
                    fx.ptr_load(
                        lds_out
                        + (peer_m + row0 + (batch * unroll + u) * rows_per_iter)
                        * c_lds_row
                        + col,
                        result_type=T.vec(VEC, elem.ir_type),
                    )
                )
                for u in range_constexpr(unroll)
            ]
            values = [[local_parts[u], peer_parts[u]] for u in range_constexpr(unroll)]
        elif const_expr(reuse_lds):
            local_parts = [
                fx.Vector(
                    fx.ptr_load(
                        lds_out
                        + (
                            tile_m
                            - reduce_m
                            + row0
                            + (batch * unroll + u) * rows_per_iter
                        )
                        * c_lds_row
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
                    # The descriptor selects the peer stripe once per wave.
                    # The local source points to unpublished rows and is skipped.
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
            values = [
                [parts[u][s].load() for s in range_constexpr(split_k)]
                for u in range_constexpr(unroll)
            ]
        else:
            values = [
                [
                    buf_copy_load(
                        buffers[s], idx, elem, VEC, cache_modifier=CPOL_DEVICE
                    )
                    for s in range_constexpr(split_k)
                ]
                for idx in partial_indices
            ]
        for u in range_constexpr(unroll):
            acc = values[u][0].extf(T.vec(VEC, T.f32))
            for s in range_constexpr(1, split_k):
                acc = acc + values[u][s].extf(T.vec(VEC, T.f32))
            row = row0 + (batch * unroll + u) * rows_per_iter
            if const_expr(reuse_lds):
                if const_expr(bounded_m):
                    buf_copy_store(
                        output_buffer,
                        row * fx.Int32(ldc64 // VEC) + col // VEC,
                        acc.to(elem),
                        elem,
                        VEC,
                    )
                else:
                    fx.ptr_store(acc.to(elem), output_base + row * ldc64 + col)
            else:
                fx.ptr_store(acc.to(elem), lds_out + row * c_lds_row + col)

    if const_expr(not reuse_lds):
        workgroup_barrier(use_cluster=False)
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
