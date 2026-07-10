# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Helpers for grouped persistent MoE M-tile scheduling.

The grouped gfx1250 GEMM consumes a compact stream of M tiles:

``m_tile_prefix[e]``
    Cumulative tile count before expert ``e``.

``m_tile_map[prefix[e] + local_tile]``
    Packed tile id ``e * max_m_tiles + local_tile``.

These tiny kernels build those tensors on device so the persistent GEMM path
does not need host-side packing.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import scf
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import arith, buffer_ops
from flydsl.expr.arith import ArithValue, CmpIPredicate
from flydsl.expr.typing import T, Int32

from aiter.ops.flydsl.kernels.tensor_shim import ptr_rsrc, MOE_KERNARG_PRELOAD_COUNT

BLOCK_THREADS = 256


def _valid_tiles(masked_rsrc, expert, max_m, tile_m):
    i32 = T.i32
    c0 = arith.constant(0, type=i32)
    valid_m = buffer_ops.buffer_load(masked_rsrc, expert, vec_width=1, dtype=i32)
    valid_m = arith.maxsi(valid_m, c0)
    valid_m = arith.minsi(valid_m, max_m)
    return (valid_m + tile_m - arith.constant(1, type=i32)) // tile_m


def _emit_prefix_sum(masked_rsrc, expert, max_m, tile_m):
    i32 = T.i32
    c0 = arith.constant(0, index=True)
    c1 = arith.constant(1, index=True)
    expert_idx = arith.index_cast(T.index, expert)
    init_acc = arith.constant(0, type=i32)

    loop = scf.ForOp(c0, expert_idx, c1, [init_acc])
    loop_ip = ir.InsertionPoint(loop.body)
    loop_ip.__enter__()

    cur = arith.index_cast(i32, loop.induction_variable)
    acc = ArithValue(loop.inner_iter_args[0])
    tiles = _valid_tiles(masked_rsrc, cur, max_m, tile_m)
    scf.YieldOp([acc + tiles])

    loop_ip.__exit__(None, None, None)
    return ArithValue(loop.results[0])


def build_moe_m_tile_prefix_map_module():
    """Return a Python launcher computing prefix and compact tile map together.

    Launcher:
    ``(masked_m, m_tile_prefix, m_tile_map, experts, max_m, tile_m,
    max_m_tiles, stream=...)``.
    """
    import torch

    map_launch = build_moe_m_tile_map_module()

    def launch_m_tile_prefix_map(
        masked_m,
        m_tile_prefix,
        m_tile_map,
        experts,
        max_m,
        tile_m,
        max_m_tiles,
        stream=None,
    ):
        valid_m = masked_m[: int(experts)].to(dtype=torch.int32)
        valid_m = valid_m.clamp(min=0, max=int(max_m))
        valid_tiles = torch.div(
            valid_m + (int(tile_m) - 1),
            int(tile_m),
            rounding_mode="floor",
        )
        m_tile_prefix[0].zero_()
        torch.cumsum(valid_tiles, dim=0, out=m_tile_prefix[1:])
        from aiter.ops.flydsl.kernels.tensor_shim import ptr_arg

        map_launch(
            ptr_arg(m_tile_prefix),
            ptr_arg(m_tile_map),
            int(experts),
            int(max_m_tiles),
            stream=stream,
        )

    return launch_m_tile_prefix_map


def build_moe_m_tile_map_module():
    """Return a JIT launcher computing compact tile map from an existing prefix."""

    @flyc.kernel(name="moe_m_tile_map", known_block_size=[BLOCK_THREADS, 1, 1])
    def m_tile_map_kernel(
        m_tile_prefix: fx.Pointer,
        m_tile_map: fx.Pointer,
        experts: Int32,
        max_m_tiles: Int32,
    ):
        i32 = T.i32
        expert = ArithValue(fx.block_idx.x)
        tid = ArithValue(fx.thread_idx.x)
        prefix_rsrc = ptr_rsrc(m_tile_prefix)
        map_rsrc = ptr_rsrc(m_tile_map)

        expert_valid = arith.cmpi(CmpIPredicate.ult, expert, ArithValue(experts))
        if_expert = scf.IfOp(expert_valid)
        with ir.InsertionPoint(if_expert.then_block):
            prefix = buffer_ops.buffer_load(prefix_rsrc, expert, vec_width=1, dtype=i32)
            next_prefix = buffer_ops.buffer_load(
                prefix_rsrc,
                expert + arith.constant(1, type=i32),
                vec_width=1,
                dtype=i32,
            )
            tiles = next_prefix - prefix
            e_base = expert * ArithValue(max_m_tiles)
            c_threads = arith.constant(BLOCK_THREADS, type=i32)
            max_tiles_idx = arith.index_cast(T.index, max_m_tiles)
            c0 = arith.constant(0, index=True)
            c1 = arith.constant(1, index=True)
            trips = (max_tiles_idx + arith.index(BLOCK_THREADS - 1)) // arith.index(
                BLOCK_THREADS
            )

            loop = scf.ForOp(c0, trips, c1)
            loop_ip = ir.InsertionPoint(loop.body)
            loop_ip.__enter__()
            it = arith.index_cast(i32, loop.induction_variable)
            local_tile = it * c_threads + tid
            tile_ok = arith.cmpi(CmpIPredicate.ult, local_tile, tiles)
            if_tile = scf.IfOp(tile_ok)
            with ir.InsertionPoint(if_tile.then_block):
                buffer_ops.buffer_store(
                    e_base + local_tile, map_rsrc, prefix + local_tile
                )
                scf.YieldOp([])
            scf.YieldOp([])
            loop_ip.__exit__(None, None, None)
            scf.YieldOp([])

    @flyc.jit
    def launch_m_tile_map(
        m_tile_prefix: fx.Pointer,
        m_tile_map: fx.Pointer,
        experts: fx.Int32,
        max_m_tiles: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            pass

        gx = arith.index_cast(T.index, experts)
        m_tile_map_kernel(
            m_tile_prefix,
            m_tile_map,
            experts,
            max_m_tiles,
        ).launch(
            grid=(gx, 1, 1),
            block=(BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    launch_m_tile_map.compile_hints = {
        "llvm_options": {
            "amdgpu-kernarg-preload": True,
            "amdgpu-kernarg-preload-count": MOE_KERNARG_PRELOAD_COUNT,
        },
    }

    return launch_m_tile_map
