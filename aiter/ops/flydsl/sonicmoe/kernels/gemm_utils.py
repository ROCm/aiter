# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
# Modifications Copyright (C) 2026 Advanced Micro Devices, Inc.

"""Low-level gfx950 GEMM helpers shared by SonicMoE grouped kernels."""

import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl.expr import const_expr, range_constexpr, rocdl

GFX950_DMA_BYTES = 16
GFX950_WAVE_SIZE = 64


def __barrier(vmcnt=0):
    llvm.InlineAsmOp(
        None,
        [],
        f"s_waitcnt vmcnt({vmcnt})\n\ts_barrier",
        "",
        has_side_effects=True,
    )


def make_lds_layout(rows, block_k):
    swizzle = fx.static(fx.SwizzleType.get(3, 3, 3))
    return fx.make_composed_layout(
        swizzle,
        fx.make_ordered_layout((rows, block_k), (1, 0)),
    )


def make_transposed_lds_layout(rows, block_k):
    base_layout = fx.make_ordered_layout((rows, block_k), (0, 1))
    if const_expr(rows == 64):
        return fx.make_composed_layout(
            fx.static(fx.SwizzleType.get(2, 4, 2)), base_layout
        )
    if const_expr(rows == 128):
        return fx.make_composed_layout(
            fx.static(fx.SwizzleType.get(2, 4, 3)), base_layout
        )
    if const_expr(rows == 256):
        return fx.make_composed_layout(
            fx.static(fx.SwizzleType.get(2, 4, 4)), base_layout
        )
    return base_layout


def get_wave_lds_offset(tid, async_load_bytes):
    return rocdl.readfirstlane(
        fx.Int64.ir_type,
        fx.Int64(tid // GFX950_WAVE_SIZE * GFX950_WAVE_SIZE * async_load_bytes),
    )


def make_wave_lds_ptr(ptr, wave_offset):
    return fx.recast_iter(fx.Int8, ptr) + fx.Int32(wave_offset)


def swizzled_col_idx(row, col, layout, block_k):
    elem_offset = fx.get_scalar(fx.crd2idx((row, col), layout))
    return elem_offset % block_k


def transposed_contiguous_idx(idx, k_idx, layout, rows):
    elem_offset = fx.get_scalar(fx.crd2idx((idx, k_idx), layout))
    return elem_offset % rows


def buffer_load_lds_inline(rsrc, lds_ptr, global_offset, dma_bytes):
    instruction = {
        16: "buffer_load_dwordx4",
        8: "buffer_load_dwordx2",
        4: "buffer_load_dword",
    }[dma_bytes]
    llvm.InlineAsmOp(
        None,
        [
            llvm.IntToPtrOp(
                ir.Type.parse("!llvm.ptr<3>"),
                fx.as_ir_value(fx.ptrtoint(lds_ptr)),
            ).result,
            fx.as_ir_value(global_offset),
            fx.as_ir_value(rsrc),
        ],
        f"s_mov_b32 m0, $0\n\t{instruction} $1, $2, 0 offen sc0 lds",
        "s,v,s",
        has_side_effects=True,
    )


def async_load_to_lds(
    lds_base,
    rsrc,
    lds_layout,
    outer_tile_size,
    outer_bound,
    global_outer_offset,
    leading_stride,
    load_iters,
    is_k_major,
    k_tile,
    context,
):
    (
        wave_offset,
        tid,
        block_threads,
        async_load_vec_size,
        ldg_x_threads,
        ks_begin,
        block_k,
        in_data_bytes,
        async_load_bytes,
    ) = context
    lds_ptr = make_wave_lds_ptr(lds_base, wave_offset)
    for index in range_constexpr(load_iters):
        global_tid = block_threads * index + tid
        if const_expr(is_k_major):
            outer_x_threads = outer_tile_size // async_load_vec_size
            outer_lds_idx = global_tid % outer_x_threads * async_load_vec_size
            k_local_idx = global_tid // outer_x_threads
            outer_local_idx = transposed_contiguous_idx(
                outer_lds_idx,
                k_local_idx,
                lds_layout,
                outer_tile_size,
            )
            global_k_idx = ks_begin + k_tile * block_k + k_local_idx
        else:
            outer_local_idx = global_tid // ldg_x_threads
            k_local_idx = global_tid % ldg_x_threads * async_load_vec_size
            global_k_idx = (
                ks_begin
                + k_tile * block_k
                + swizzled_col_idx(
                    outer_local_idx,
                    k_local_idx,
                    lds_layout,
                    block_k,
                )
            )
        global_outer_idx = global_outer_offset + outer_local_idx
        safe_outer_idx = (global_outer_idx < outer_bound).select(global_outer_idx, 0)
        if const_expr(is_k_major):
            global_offset = (
                global_k_idx * leading_stride + safe_outer_idx
            ) * in_data_bytes
        else:
            global_offset = (
                safe_outer_idx * leading_stride + global_k_idx
            ) * in_data_bytes
        buffer_load_lds_inline(rsrc, lds_ptr, global_offset, async_load_bytes)
        if index < load_iters - 1:
            lds_ptr = lds_ptr + block_threads * async_load_bytes
