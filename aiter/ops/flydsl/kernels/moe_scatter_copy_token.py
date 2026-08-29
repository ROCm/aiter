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
            if fx.Int32(srow) >= fx.Int32(0):
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
