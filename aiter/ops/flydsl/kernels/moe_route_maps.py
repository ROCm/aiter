# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""One-pass MoE route -> grouped-row map kernel (FlyDSL), atomic-scatter argsort.

Computes ``src_rows`` (a.k.a. c_map): for each route ``i = t*topk + k`` of the
flattened ``topk_ids``, the grouped-GEMM row that token ``t``'s k-th expert
occupies = ``expert*max_m + slot``. The within-expert ``slot`` is claimed via a
global ``atomicAdd`` on a per-expert counter -- no host-side argsort / nonzero /
one-hot. This is the SGLang ``compute_arg_sorts`` idea, adapted to our padded
per-expert blocks: pre-initialize ``atomic_buffer[e] = e*max_m`` so the atomic
returns the grouped row directly.

One thread per route (grid covers ceil(numel/BLOCK)). The within-expert order is
atomic-race order (nondeterministic), which is fine: scatter-copy and
gather-reduce both key off the *same* src_rows, and the grouped GEMM is
order-agnostic within ``[0, counts[e])``.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith
from flydsl.expr.typing import T, Int32
from flydsl.expr.arith import ArithValue, CmpIPredicate
from flydsl.compiler.kernel_function import CompilationContext

from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm, scf
from flydsl.expr import buffer_ops

BLOCK_THREADS = 256


def build_moe_route_maps_module():
    """Return a JIT launcher computing src_rows (route -> grouped row).

    Launcher: ``(topk_ids, atomic_buffer, c_map, numel, grid_blocks, stream=...)``
      topk_ids      : (numel,) int32   flattened expert ids
      atomic_buffer : (E,)     int32   pre-init to e*max_m (mutated)
      c_map         : (numel,) int32   out: grouped row per route
    """
    @flyc.kernel(name="moe_route_maps")
    def route_maps_kernel(
        topk_ids: fx.Tensor,       # (numel,) int32
        atomic_buffer: fx.Tensor,  # (E,) int32, pre-init e*max_m
        c_map: fx.Tensor,          # (numel,) int32 out
        numel: Int32,
    ):
        i32 = T.i32
        route = ArithValue(fx.block_idx.x) * arith.constant(
            BLOCK_THREADS, type=i32
        ) + ArithValue(fx.thread_idx.x)
        in_range = arith.cmpi(CmpIPredicate.ult, route, ArithValue(numel))
        _if = scf.IfOp(in_range)
        with ir.InsertionPoint(_if.then_block):
            topk_rsrc = buffer_ops.create_buffer_resource(topk_ids, max_size=True)
            c_rsrc = buffer_ops.create_buffer_resource(c_map, max_size=True)

            e = buffer_ops.buffer_load(topk_rsrc, route, vec_width=1, dtype=i32)

            # &atomic_buffer[e] : base address + e*4 bytes -> llvm global ptr
            base_idx = buffer_ops.extract_base_index(atomic_buffer, address_space=1)
            e_idx = arith.index_cast(T.index, e)
            addr = fx.Index(base_idx) + fx.Index(e_idx) * fx.Index(4)
            ptr = buffer_ops.create_llvm_ptr(addr, address_space=1)
            ptr = ptr._value if hasattr(ptr, "_value") else ptr

            # atomicAdd(&atomic_buffer[e], 1) -> grouped row (buffer pre-init e*max_m)
            start = llvm.AtomicRMWOp(
                llvm.AtomicBinOp.add,
                ptr,
                arith.constant(1, type=i32),
                llvm.AtomicOrdering.monotonic,
                syncscope="agent",
                alignment=4,
            ).result

            buffer_ops.buffer_store(start, c_rsrc, route)
            scf.YieldOp([])

    @flyc.jit
    def launch_route_maps(
        topk_ids: fx.Tensor,
        atomic_buffer: fx.Tensor,
        c_map: fx.Tensor,
        numel: fx.Int32,
        grid_blocks: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            pass

        gx = arith.index_cast(T.index, grid_blocks)
        route_maps_kernel(topk_ids, atomic_buffer, c_map, numel).launch(
            grid=(gx, 1, 1),
            block=(BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    return launch_route_maps
