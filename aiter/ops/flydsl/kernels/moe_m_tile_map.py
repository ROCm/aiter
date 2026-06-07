# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""One-pass MoE m-tile-map kernel (FlyDSL).

Device-side replacement for the host-side packing loop in
``moe_grouped_gemm_mxscale_gfx1250._make_m_tile_map`` (which did a
``valid_tiles.cpu().tolist()`` device->host sync + a Python comprehension).

Given ``m_tile_prefix`` (the (E+1,) int32 cumulative tile counts from
``_make_m_tile_prefix``), pack the per-expert valid M-tiles into the flat
grouped-persistent schedule::

    for j in [0, prefix[e+1] - prefix[e]):
        m_tile_map[prefix[e] + j] = e * max_m_tiles + j

The prefix already encodes both the per-expert tile count (successive
difference) and the write offset (the prefix value itself), so:

  * no host sync is needed -- the count lives in ``prefix[E]`` and the
    persistent GEMM reads ``m_tile_map[0 : prefix[E]]`` only;
  * the output buffer is sized to the max ``E * max_m_tiles`` and any rows past
    ``prefix[E]`` are never read;
  * each lane's write range ``[prefix[e], prefix[e+1])`` is disjoint, so the
    one-warp fan-out below is race-free.

One warp iterates all experts: lane ``l`` handles experts ``l, l+64, ...``.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith
from flydsl.expr.typing import T, Int32
from flydsl.expr.arith import ArithValue
from flydsl.compiler.kernel_function import CompilationContext

from flydsl._mlir import ir
from flydsl._mlir.dialects import scf
from flydsl.expr import buffer_ops

BLOCK_THREADS = 64  # one wavefront iterates all experts


def build_moe_m_tile_map_module():
    """Return a JIT launcher that fills ``m_tile_map`` from ``m_tile_prefix``.

    Launcher: ``(m_tile_prefix, m_tile_map, experts, max_m_tiles, stream=...)``
      m_tile_prefix : (E+1,)           int32  cumulative per-expert tile counts
      m_tile_map    : (E*max_m_tiles,) int32  out: packed global tile ids
      experts       : int32                    number of (local) experts E
      max_m_tiles   : int32                    ceil(max_m / tile_m)
    """

    @flyc.kernel(name="moe_m_tile_map")
    def m_tile_map_kernel(
        m_tile_prefix: fx.Tensor,  # (E+1,) int32
        m_tile_map: fx.Tensor,     # (E*max_m_tiles,) int32 out
        experts: Int32,
        max_m_tiles: Int32,
    ):
        i32 = T.i32
        idx_t = ir.IndexType.get()
        prefix_rsrc = buffer_ops.create_buffer_resource(m_tile_prefix, max_size=True)
        map_rsrc = buffer_ops.create_buffer_resource(m_tile_map, max_size=True)

        # Outer loop: e = lane; e < experts; e += BLOCK_THREADS (one warp covers all).
        lane_idx = arith.index_cast(idx_t, ArithValue(fx.thread_idx.x))
        experts_idx = arith.index_cast(idx_t, ArithValue(experts))
        step_idx = arith.constant(BLOCK_THREADS, index=True)

        e_for = scf.ForOp(lane_idx, experts_idx, step_idx)
        with ir.InsertionPoint(e_for.body):
            e_i32 = arith.index_cast(i32, e_for.induction_variable)
            e1_i32 = ArithValue(e_i32) + arith.constant(1, type=i32)

            # base = prefix[e] (write offset);  cnt = prefix[e+1] - prefix[e].
            base = buffer_ops.buffer_load(prefix_rsrc, e_i32, vec_width=1, dtype=i32)
            nxt = buffer_ops.buffer_load(prefix_rsrc, e1_i32, vec_width=1, dtype=i32)
            cnt_i32 = ArithValue(nxt) - ArithValue(base)
            tile_base = ArithValue(e_i32) * ArithValue(max_m_tiles)  # e*max_m_tiles

            # Inner loop: j in [0, cnt) -> m_tile_map[base + j] = e*max_m_tiles + j.
            j_for = scf.ForOp(
                arith.constant(0, index=True),
                arith.index_cast(idx_t, cnt_i32),
                arith.constant(1, index=True),
            )
            with ir.InsertionPoint(j_for.body):
                j_i32 = arith.index_cast(i32, j_for.induction_variable)
                val = ArithValue(tile_base) + ArithValue(j_i32)
                pos = ArithValue(base) + ArithValue(j_i32)
                buffer_ops.buffer_store(val, map_rsrc, pos)
                scf.YieldOp([])
            scf.YieldOp([])

    @flyc.jit
    def launch_m_tile_map(
        m_tile_prefix: fx.Tensor,
        m_tile_map: fx.Tensor,
        experts: fx.Int32,
        max_m_tiles: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            pass

        gx = arith.constant(1, index=True)
        m_tile_map_kernel(
            m_tile_prefix, m_tile_map, experts, max_m_tiles
        ).launch(
            grid=(gx, 1, 1),
            block=(BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    return launch_m_tile_map
