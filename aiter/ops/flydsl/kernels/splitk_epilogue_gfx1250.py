# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""In-kernel split-K reduction for the gfx1250 a8w8 GEMM."""

import flydsl.expr as fx
from flydsl._mlir.dialects import llvm as llvm_dialect
from flydsl.expr import range_constexpr
from flydsl.expr.typing import T, as_ir_value

from .communication_ops_utils import traced
from .gemm_common_gfx1250 import workgroup_barrier
from .tensor_shim import buf_copy_load, ptr_buf_tensor

# gfx1250 cpol[4:3] selects device scope for both TDM stores and buffer loads.
# Only the partials need to cross shader-engine caches; avoid a whole-cache fence.
CPOL_DEVICE = 16
# Keep the published partials in the device cache for the reducer.
CPOL_STORE_DEVICE = CPOL_DEVICE | 3
FLAG_STRIDE_I32 = 32
VEC = 8
UNROLL = 32
MAX_PARTIAL_VECTORS = 128  # 512 dwords per load batch, including split-K 8.


@traced
def emit_splitk_reduce_epilogue(
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
    mn_oob,
    flat_tile,
    arg_flag,
):
    """Let the last arriving split reduce this tile without another launch.

    The caller stores its partial with device scope, waits for the TDM store,
    and synchronizes the workgroup before entering here. A single integer
    counter elects the reducer; all other workgroups can exit immediately.
    Partials keep the output dtype, matching the separate reduction kernel,
    and are accumulated in FP32 in split order before the final conversion.
    """
    flag = fx.recast_iter(fx.PointerType.get(T.i32, arg_flag.address_space), arg_flag)
    flag_ptr = fx.to_llvm_ptr(flag + flat_tile * FLAG_STRIDE_I32)
    # Keep the election result in row padding, outside the reduced C tile.
    shared_flag = fx.recast_iter(fx.Int32, lds_base_ptr + tile_n * 2)
    if tid == fx.Int32(0):
        arrival = fx.Int32(
            llvm_dialect.atomicrmw(
                llvm_dialect.AtomicBinOp.add,
                flag_ptr,
                as_ir_value(fx.Int32(1)),
                llvm_dialect.AtomicOrdering.monotonic,
                syncscope="agent",
                alignment=4,
            )
        )
        fx.ptr_store(arrival, shared_flag)
    workgroup_barrier(use_cluster=False)

    arrival = fx.Int32(fx.ptr_load(shared_flag))
    if arrival == fx.Int32(split_k - 1):
        lanes_per_row = tile_n // VEC
        rows_per_iter = block // lanes_per_row
        unroll = min(UNROLL, MAX_PARTIAL_VECTORS // split_k, tile_m // rows_per_iter)
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
        )
        # Bound each plane at M so vector loads of the final M tile zero-fill.
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
        for batch in range(tile_m // (rows_per_iter * unroll)):
            partial_indices = [
                tid + (batch * unroll + u) * block for u in range_constexpr(unroll)
            ]
            parts = [
                [
                    buf_copy_load(
                        buffers[s], idx, elem, VEC, cache_modifier=CPOL_DEVICE
                    )
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
        layout = fx.make_layout((tile_m, c_lds_row), (c_lds_row, 1))
        tile_out = fx.Tensor(fx.make_view(output_base, layout))
        store_atom = fx.rocdl.make_tdm_atom(
            tile_out,
            [mn_oob, tile_n],
            strides=[ldc64, None],
            num_warps=block // 32,
        )
        fx.copy(store_atom, fx.Tensor(fx.make_view(lds_out, layout)), tile_out)
        fx.rocdl.tdm_ops.tensor_wait(0)
        if tid == fx.Int32(0):
            # Undo this launch's arrivals without overwriting another increment.
            llvm_dialect.atomicrmw(
                llvm_dialect.AtomicBinOp.add,
                flag_ptr,
                as_ir_value(fx.Int32(-split_k)),
                llvm_dialect.AtomicOrdering.monotonic,
                syncscope="agent",
                alignment=4,
            )
