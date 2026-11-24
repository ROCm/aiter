# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""One-pass MoE route-gather (scatter-copy) kernel (FlyDSL).

Background
----------
Before the stage1 GEMM, each token's quantized payload (and per-token scale)
must be copied from the flat per-token layout into the grouped per-expert
layout the masked grouped kernel consumes::

    for e in range(E):
        toks = tokens routed to expert e          # n = counts[e] of them
        grouped[e, :n] = a_payload[toks]          # copy each token's row

This kernel does that copy in one pass. Each destination grouped row is mapped
to its source token row via a precomputed ``dst_src`` map (``-1`` => leave the
destination untouched, i.e. an unused padding slot). One thread-block per
destination row copies the whole row; threads stride over the row's elements.

The copy is byte-exact. We pick the widest vectorized buffer op whose access
size divides ``row_bytes`` (which keeps every row base naturally aligned), using
the same fallback ladder as the MoE GEMM X-load path:

    row_bytes % 16 == 0 -> dwordx4 (16B / vec4 i32)   -- fastest
    row_bytes %  8 == 0 -> dwordx2 ( 8B / vec2 i32)
    row_bytes %  4 == 0 -> dword   ( 4B / scalar i32)
    otherwise           -> byte    ( 1B / scalar i8)   -- e.g. a 90B scale row

Source and destination rows share the same byte width.

Grid  : (num_dst_rows, 1, 1)   -- num_dst_rows = E * max_m
Block : (BLOCK_THREADS, 1, 1)
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr.typing import Int32

from aiter.ops.flydsl.kernels.tensor_shim import (
    AITER_FLYDSL_KERNARG_PRELOAD,
    AITER_FLYDSL_KERNARG_PRELOAD_COUNT,
    buf_copy_atom,
    ptr_buf_tensor,
)

BLOCK_THREADS = 256


def build_moe_scatter_copy_token_module(row_bytes: int):
    """Return a JIT launcher that gathers source rows into destination rows.

    Parameters
    ----------
    row_bytes : int   byte width of one row (same for src and dst).

    Launcher signature: ``(src, dst, dst_src, num_dst, stream=...)`` where
    ``src``/``dst`` are uint8 tensors viewed as (rows, row_bytes) and
    ``dst_src`` is an int32 (num_dst,) map from dst row -> src row (-1 to skip).
    """
    assert row_bytes > 0
    # Fallback ladder (mirrors moe_gemm_2stage X-load): pick the widest buffer op
    # whose access size divides row_bytes. Because each width divides row_bytes,
    # every row base (row * row_bytes) is naturally aligned for that width, and the
    # row is covered by a whole number of units with no in-row remainder.
    if row_bytes % 16 == 0:
        vec_width, unit_bytes, use_dword, width_tag = 4, 16, True, "dwx4"
    elif row_bytes % 8 == 0:
        vec_width, unit_bytes, use_dword, width_tag = 2, 8, True, "dwx2"
    elif row_bytes % 4 == 0:
        vec_width, unit_bytes, use_dword, width_tag = 1, 4, True, "dw"
    else:
        vec_width, unit_bytes, use_dword, width_tag = 1, 1, False, "by"

    # Number of vectorized copy units (buffer ops) per row.
    n_units = row_bytes // unit_bytes

    module_name = f"moe_scatter_copy_token_b{row_bytes}_{width_tag}"

    @flyc.kernel(name=module_name)
    def scatter_copy_kernel(
        src: fx.Pointer,  # (num_src, row_bytes) uint8
        dst: fx.Pointer,  # (num_dst, row_bytes) uint8
        dst_src: fx.Pointer,  # (num_dst,) int32  -- src row per dst row, -1=skip
        num_dst: Int32,
    ):
        cdt = fx.Int32 if use_dword else fx.Int8

        # Uint32 for the counts/indices (`<` lowers to ult); the source row is
        # signed because -1 is the "skip this destination row" sentinel.
        tid_i32 = fx.Uint32(fx.thread_idx.x)
        bid_i32 = fx.Uint32(fx.block_idx.x)

        if bid_i32 < fx.Uint32(num_dst):
            srow = ptr_buf_tensor(dst_src)[bid_i32]
            if fx.Int32(srow) >= 0:
                # A row is a contiguous run of n_units copy units.
                if const_expr(vec_width > 1):
                    src_t = ptr_buf_tensor(src, cdt, unit_elems=vec_width)
                    dst_t = ptr_buf_tensor(dst, cdt, unit_elems=vec_width)
                    atom = buf_copy_atom(unit_bytes, cdt)
                    # Hoisted: a per-iteration fragment costs more than the copy.
                    frag = fx.make_fragment_like(fx.slice(src_t, (0, None)))
                else:
                    src_t = ptr_buf_tensor(src, cdt)
                    dst_t = ptr_buf_tensor(dst, cdt)
                src_unit_base = fx.Uint32(srow) * n_units
                dst_unit_base = bid_i32 * n_units

                for it in range_constexpr(
                    (n_units + BLOCK_THREADS - 1) // BLOCK_THREADS
                ):
                    # One buffer op per unit; threads stride over the row.
                    uidx = tid_i32 + it * BLOCK_THREADS
                    if uidx < fx.Uint32(n_units):
                        s_unit = src_unit_base + uidx
                        d_unit = dst_unit_base + uidx
                        if const_expr(vec_width > 1):
                            fx.copy(atom, fx.slice(src_t, (s_unit, None)), frag)
                            fx.copy(atom, frag, fx.slice(dst_t, (d_unit, None)))
                        else:
                            dst_t[d_unit] = src_t[s_unit]

    @flyc.jit
    def launch_scatter_copy(
        src: fx.Pointer,
        dst: fx.Pointer,
        dst_src: fx.Pointer,
        num_dst: fx.Int32,
        stream: fx.Stream,
    ):
        launcher = scatter_copy_kernel(src, dst, dst_src, num_dst)
        launcher.launch(
            grid=(fx.Int64(num_dst), 1, 1),
            block=(BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    launch_scatter_copy.compile_hints = {
        "llvm_options": {
            "amdgpu-kernarg-preload": AITER_FLYDSL_KERNARG_PRELOAD,
            "amdgpu-kernarg-preload-count": AITER_FLYDSL_KERNARG_PRELOAD_COUNT,
        },
    }

    return launch_scatter_copy


@functools.cache
def build_moe_scatter_copy_compact_module(row_bytes: int):
    """Build a compact-wire row scatter.

    Compact MegaMoE dispatch stores one quantized payload row for each
    ``(source rank, source token)``. ``ep_rowmap`` maps every final
    expert-contiguous row back to the packed route metadata for that payload.
    This kernel materializes those rows before GEMM1, so GEMM1 can keep its
    ordinary contiguous-A TDM path instead of issuing indexed TDM gathers.
    """
    assert row_bytes > 0
    if row_bytes % 16 == 0:
        vec_width, unit_bytes, use_dword, width_tag = 4, 16, True, "dwx4"
    elif row_bytes % 8 == 0:
        vec_width, unit_bytes, use_dword, width_tag = 2, 8, True, "dwx2"
    elif row_bytes % 4 == 0:
        vec_width, unit_bytes, use_dword, width_tag = 1, 4, True, "dw"
    else:
        vec_width, unit_bytes, use_dword, width_tag = 1, 1, False, "by"
    n_units = row_bytes // unit_bytes
    module_name = f"moe_scatter_copy_compact_b{row_bytes}_{width_tag}"

    @flyc.kernel(name=module_name)
    def scatter_copy_compact_kernel(
        src: fx.Pointer,  # compact-wire rows, uint8
        dst: fx.Pointer,  # contiguous expert rows, uint8
        ep_rowmap: fx.Pointer,  # (num_dst, 2) i32: packed_meta, route weight
        src_row_stride_units: Int32,
        topk: Int32,
        num_dst: Int32,
    ):
        cdt = fx.Int32 if use_dword else fx.Int8
        tid_i32 = fx.Uint32(fx.thread_idx.x)
        bid_i32 = fx.Uint32(fx.block_idx.x)

        if bid_i32 < fx.Uint32(num_dst):
            packed_meta = ptr_buf_tensor(ep_rowmap)[bid_i32 * 2]
            # packed_meta is source_rank * max_tokens * topk + token * topk
            # + route. Negative is the padding-row sentinel.
            if fx.Int32(packed_meta) >= 0:
                src_row = fx.Int32(packed_meta) // topk
                if const_expr(vec_width > 1):
                    src_t = ptr_buf_tensor(src, cdt, unit_elems=vec_width)
                    dst_t = ptr_buf_tensor(dst, cdt, unit_elems=vec_width)
                    atom = buf_copy_atom(unit_bytes, cdt)
                    frag = fx.make_fragment_like(fx.slice(src_t, (0, None)))
                else:
                    src_t = ptr_buf_tensor(src, cdt)
                    dst_t = ptr_buf_tensor(dst, cdt)
                src_unit_base = fx.Uint32(src_row) * fx.Uint32(src_row_stride_units)
                dst_unit_base = bid_i32 * n_units
                for it in range_constexpr(
                    (n_units + BLOCK_THREADS - 1) // BLOCK_THREADS
                ):
                    uidx = tid_i32 + it * BLOCK_THREADS
                    if uidx < fx.Uint32(n_units):
                        s_unit = src_unit_base + uidx
                        d_unit = dst_unit_base + uidx
                        if const_expr(vec_width > 1):
                            fx.copy(atom, fx.slice(src_t, (s_unit, None)), frag)
                            fx.copy(atom, frag, fx.slice(dst_t, (d_unit, None)))
                        else:
                            dst_t[d_unit] = src_t[s_unit]

    @flyc.jit
    def launch_scatter_copy_compact(
        src: fx.Pointer,
        dst: fx.Pointer,
        ep_rowmap: fx.Pointer,
        src_row_stride_units: fx.Int32,
        topk: fx.Int32,
        num_dst: fx.Int32,
        stream: fx.Stream,
    ):
        launcher = scatter_copy_compact_kernel(
            src, dst, ep_rowmap, src_row_stride_units, topk, num_dst
        )
        launcher.launch(
            grid=(fx.Int64(num_dst), 1, 1),
            block=(BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    launch_scatter_copy_compact.compile_hints = {
        "llvm_options": {
            "amdgpu-kernarg-preload": AITER_FLYDSL_KERNARG_PRELOAD,
            "amdgpu-kernarg-preload-count": AITER_FLYDSL_KERNARG_PRELOAD_COUNT,
        },
    }
    return launch_scatter_copy_compact


@functools.cache
def build_moe_scatter_copy_compact_pair_module(payload_bytes: int, scale_bytes: int):
    """Build one compact-wire scatter for a payload and its scale row.

    The compact wire keeps both planes at the same row stride. Combining them
    avoids launching a second grid of one block per expert row solely to copy
    E8M0 scales.
    """
    assert (
        payload_bytes % 16 == 0 and scale_bytes % 16 == 0
    ), "paired compact scatter needs 16-byte rows"
    payload_units = payload_bytes // 16
    scale_units = scale_bytes // 16

    nwarps = BLOCK_THREADS // 32

    @flyc.kernel(name=f"moe_scatter_copy_compact_pair_p{payload_bytes}_s{scale_bytes}")
    def scatter_copy_compact_pair_kernel(
        src_payload: fx.Pointer,
        src_scale: fx.Pointer,
        dst_payload: fx.Pointer,
        dst_scale: fx.Pointer,
        ep_rowmap: fx.Pointer,
        masked_m: fx.Pointer,
        psum: fx.Pointer,
        src_row_stride_units: Int32,
        topk: Int32,
        n_split: Int32,
    ):
        # One column of blocks per local expert. masked_m/psum name the live
        # rows; the aligned gaps between experts are not launched.
        expert = fx.Uint32(fx.block_idx.x)
        part = fx.Uint32(fx.block_idx.y)
        tid = fx.Uint32(fx.thread_idx.x)
        lane = tid & fx.Uint32(31)
        warp = tid >> 5
        count = fx.Uint32(ptr_buf_tensor(masked_m)[expert])
        end = fx.Uint32(ptr_buf_tensor(psum)[expert])
        start = end - count
        row_base = part * fx.Uint32(nwarps) + warp
        stride = fx.Uint32(n_split) * fx.Uint32(nwarps)
        src_t = ptr_buf_tensor(src_payload, fx.Int32, unit_elems=4)
        src_s = ptr_buf_tensor(src_scale, fx.Int32, unit_elems=4)
        dst_t = ptr_buf_tensor(dst_payload, fx.Int32, unit_elems=4)
        dst_s = ptr_buf_tensor(dst_scale, fx.Int32, unit_elems=4)
        atom = buf_copy_atom(16, fx.Int32)
        frag = fx.make_fragment_like(fx.slice(src_t, (0, None)))
        for row_i in range(row_base, count, stride):
            row = start + row_i
            packed_meta = ptr_buf_tensor(ep_rowmap)[row * 2]
            if fx.Int32(packed_meta) >= 0:
                src_row = fx.Int32(packed_meta) // topk
                src_base = fx.Uint32(src_row) * fx.Uint32(src_row_stride_units)
                dst_payload_base = row * fx.Uint32(payload_units)
                dst_scale_base = row * fx.Uint32(scale_units)
                for it in range_constexpr((payload_units + 31) // 32):
                    unit = lane + fx.Uint32(it * 32)
                    if unit < fx.Uint32(payload_units):
                        fx.copy(atom, fx.slice(src_t, (src_base + unit, None)), frag)
                        fx.copy(
                            atom,
                            frag,
                            fx.slice(dst_t, (dst_payload_base + unit, None)),
                        )
                for it in range_constexpr((scale_units + 31) // 32):
                    unit = lane + fx.Uint32(it * 32)
                    if unit < fx.Uint32(scale_units):
                        fx.copy(atom, fx.slice(src_s, (src_base + unit, None)), frag)
                        fx.copy(
                            atom,
                            frag,
                            fx.slice(dst_s, (dst_scale_base + unit, None)),
                        )

    @flyc.jit
    def launch_scatter_copy_compact_pair(
        src_payload: fx.Pointer,
        src_scale: fx.Pointer,
        dst_payload: fx.Pointer,
        dst_scale: fx.Pointer,
        ep_rowmap: fx.Pointer,
        masked_m: fx.Pointer,
        psum: fx.Pointer,
        src_row_stride_units: fx.Int32,
        topk: fx.Int32,
        n_experts: fx.Int32,
        n_split: fx.Int32,
        stream: fx.Stream,
    ):
        launcher = scatter_copy_compact_pair_kernel(
            src_payload,
            src_scale,
            dst_payload,
            dst_scale,
            ep_rowmap,
            masked_m,
            psum,
            src_row_stride_units,
            topk,
            n_split,
        )
        launcher.launch(
            grid=(fx.Int64(n_experts), fx.Int64(n_split), 1),
            block=(BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    launch_scatter_copy_compact_pair.compile_hints = {
        "llvm_options": {
            "amdgpu-kernarg-preload": AITER_FLYDSL_KERNARG_PRELOAD,
            "amdgpu-kernarg-preload-count": AITER_FLYDSL_KERNARG_PRELOAD_COUNT,
        },
    }
    return launch_scatter_copy_compact_pair
