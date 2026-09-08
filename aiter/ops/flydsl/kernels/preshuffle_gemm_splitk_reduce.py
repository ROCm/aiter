# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""fp32 split-K reduction for ``preshuffle_gemm_splitk``.

Sums the ``split_k`` fp32 partial tiles produced by the split-K preshuffle GEMM and
downcasts once, on the final store. Modelled on
``gemm_a8w8_splitk_reduce_gfx1250`` but the partials stay fp32 the whole way — that
kernel truncates to bf16 before summing, which is exactly what this path must not do.

Workspace layout is ``(split_k, m_pad, N)`` fp32 with ``m_pad`` a whole number of GEMM
M-tiles. The grid is one row per block-y, so the padding rows are simply never visited
and no index arithmetic has to know about them.
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, range_constexpr
from flydsl.expr.typing import T

from aiter.ops.flydsl.kernels import buffer_ops
from aiter.ops.flydsl.kernels.kernels_common import format_kernel_name

BLOCK = 256
VEC = 4  # 16B per thread per slice


@functools.lru_cache(maxsize=64)
def compile_preshuffle_gemm_splitk_reduce(
    *, N: int, split_k: int, out_dtype: str = "bf16"
):
    """Compile the fp32 split-K reduce.

    Signature: fn(out, workspace, M, m_pad, stream).
    """
    if out_dtype not in ("bf16", "fp16"):
        raise ValueError(f"out_dtype must be bf16/fp16, got {out_dtype!r}")
    out_elem = fx.BFloat16 if out_dtype == "bf16" else fx.Float16
    out_bytes = 2
    span = BLOCK * VEC
    n_tiles = (N + span - 1) // span
    name = format_kernel_name(f"preshuffle_gemm_splitk_reduce_{out_dtype}_sk{split_k}")

    @flyc.kernel(name=name, known_block_size=[BLOCK, 1, 1])
    def reduce_kernel(
        arg_out: fx.Tensor,
        arg_ws: fx.Tensor,
        i32_m_pad: fx.Int32,
    ):
        vec_out = T.vec(VEC, out_elem.ir_type)  # needs the kernel's MLIR context
        tid = gpu.thread_id("x")
        tile = fx.Int32(gpu.block_id("x"))
        row = fx.Int32(gpu.block_id("y"))
        col = tile * fx.Int32(span) + fx.Int32(tid) * fx.Int32(VEC)

        # Each resource covers exactly one row, so the N tail falls off the descriptor
        # instead of needing a predicate.
        ws_row_bytes = fx.Int32(N * 4)
        acc = None
        for s in range_constexpr(split_k):
            rsrc = buffer_ops.create_buffer_resource(
                arg_ws,
                max_size=False,
                num_records_bytes=ws_row_bytes,
                base_byte_offset=(fx.Int64(s) * fx.Int64(i32_m_pad) + fx.Int64(row))
                * fx.Int64(N * 4),
            )
            part = fx.Vector(
                buffer_ops.buffer_load(rsrc, col, vec_width=VEC, dtype=T.f32)
            )
            acc = part if const_expr(s == 0) else acc + part

        out_rsrc = buffer_ops.create_buffer_resource(
            arg_out,
            max_size=False,
            num_records_bytes=fx.Int32(N * out_bytes),
            base_byte_offset=fx.Int64(row) * fx.Int64(N * out_bytes),
        )
        buffer_ops.buffer_store(acc.truncf(vec_out), out_rsrc, col)

    @flyc.jit
    def launch(
        arg_out: fx.Tensor,
        arg_ws: fx.Tensor,
        i32_m: fx.Int32,
        i32_m_pad: fx.Int32,
        stream: fx.Stream,
    ):
        reduce_kernel(arg_out, arg_ws, i32_m_pad).launch(
            grid=(n_tiles, i32_m, 1),
            block=(BLOCK, 1, 1),
            stream=stream,
        )

    return launch
