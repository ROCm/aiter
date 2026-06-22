# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""DeepGEMM-contiguous M-tile prefix sum (FlyDSL), single-block serial scan.

Given per-expert row counts ``masked_m`` (E,), computes the tile-aligned
exclusive prefix sum used by the contiguous grouped-GEMM scheduler:

    aligned[e]   = ceil(masked_m[e] / tile_m) * tile_m
    starts[e]    = sum(aligned[0:e])            # exclusive prefix sum
    psum[e]      = starts[e] + masked_m[e]      # actual end (NOT tile-aligned)
    contiguous_m = max(tile_m, sum(aligned))

Replaces ``torch.cumsum``, which on ROCm lowers to a rocprim scan (a
``trampoline_kernel`` plus an internal D2D temp copy) on every call. The number
of experts E is tiny, so a single-thread serial scan in one block is cheaper and
copy-free.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith, buffer_ops
from flydsl.expr.typing import T, Int32
from flydsl.expr.arith import ArithValue, CmpIPredicate
from flydsl.compiler.kernel_function import CompilationContext

from flydsl._mlir import ir
from flydsl._mlir.dialects import scf


def build_moe_contiguous_psum_module():
    """Return a JIT launcher computing the tile-aligned prefix sum in one pass.

    Launcher: ``(masked_m, starts, psum, contiguous_m, experts, tile_m, stream=)``
      masked_m     : (E,)  int32  in   per-expert row counts
      starts       : (E,)  int32  out  exclusive prefix sum of aligned counts
      psum         : (E,)  int32  out  starts[e] + masked_m[e]
      contiguous_m : (1,)  int32  out  max(tile_m, sum(aligned))
    """

    @flyc.kernel(name="moe_contiguous_psum")
    def psum_kernel(
        masked_m: fx.Tensor,  # (E,) int32 in
        starts: fx.Tensor,  # (E,) int32 out
        psum: fx.Tensor,  # (E,) int32 out
        contiguous_m: fx.Tensor,  # (1,) int32 out
        experts: Int32,
        tile_m: Int32,
    ):
        i32 = T.i32
        tid = ArithValue(fx.thread_idx.x)
        is_leader = arith.cmpi(CmpIPredicate.eq, tid, arith.constant(0, type=i32))
        _if = scf.IfOp(is_leader)
        with ir.InsertionPoint(_if.then_block):
            m_rsrc = buffer_ops.create_buffer_resource(masked_m, max_size=True)
            s_rsrc = buffer_ops.create_buffer_resource(starts, max_size=True)
            p_rsrc = buffer_ops.create_buffer_resource(psum, max_size=True)
            c_rsrc = buffer_ops.create_buffer_resource(contiguous_m, max_size=True)

            tile_v = ArithValue(tile_m)
            tile_minus_1 = tile_v - arith.constant(1, type=i32)

            c0_idx = arith.index(0)
            c1_idx = arith.index(1)
            e_upper = arith.index_cast(T.index, experts)
            loop = scf.ForOp(c0_idx, e_upper, c1_idx, [arith.constant(0, type=i32)])
            with ir.InsertionPoint(loop.body):
                e = loop.induction_variable
                cur = loop.inner_iter_args[0]
                e_i32 = arith.index_cast(i32, e)
                m = buffer_ops.buffer_load(m_rsrc, e_i32, vec_width=1, dtype=i32)
                # aligned = ((m + tile_m - 1) // tile_m) * tile_m  (unsigned floor)
                q = arith.divui(ArithValue(m) + tile_minus_1, tile_v)
                aligned = ArithValue(q) * tile_v
                # starts[e] = cur ; psum[e] = cur + m
                buffer_ops.buffer_store(cur, s_rsrc, e_i32)
                buffer_ops.buffer_store(ArithValue(cur) + ArithValue(m), p_rsrc, e_i32)
                next_cur = ArithValue(cur) + ArithValue(aligned)
                scf.YieldOp([next_cur])
            final_cur = loop.results[0]
            # contiguous_m = max(tile_m, final_cur)
            gt = arith.cmpi(CmpIPredicate.sgt, final_cur, tile_v)
            cm = arith.select(gt, final_cur, tile_v)
            buffer_ops.buffer_store(cm, c_rsrc, arith.constant(0, type=i32))
            scf.YieldOp([])

    @flyc.jit
    def launch_psum(
        masked_m: fx.Tensor,
        starts: fx.Tensor,
        psum: fx.Tensor,
        contiguous_m: fx.Tensor,
        experts: fx.Int32,
        tile_m: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            pass
        psum_kernel(masked_m, starts, psum, contiguous_m, experts, tile_m).launch(
            grid=(arith.index(1), 1, 1),
            block=(64, 1, 1),
            stream=stream,
        )

    return launch_psum
