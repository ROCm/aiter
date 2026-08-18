# SPDX-License-Identifier: MIT
"""Single-CTA byte copy + release signal, standing in for one RDMA put."""

from __future__ import annotations

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import arith, gpu
from flydsl.expr.arith import ArithValue
from flydsl.expr.typing import Int32, Int64, T

from aiter.ops.flydsl.kernels import buffer_ops
from .. import comm_ops

BLOCK_THREADS = 256


def build_copy_put_signal_module():
    """Return launcher ``(src, dst, signal, nbytes, generation, stream)``."""

    @flyc.kernel(name="megamoe_tile_copy_put_signal", known_block_size=[BLOCK_THREADS, 1, 1])
    def copy_put_signal_kernel(
        src: fx.Tensor,
        dst: fx.Tensor,
        signal: fx.Tensor,
        nbytes: Int32,
        generation: Int64,
    ):
        tid = fx.Int32(gpu.thread_id("x"))
        src_rsrc = buffer_ops.create_buffer_resource(src, max_size=True)
        dst_rsrc = buffer_ops.create_buffer_resource(dst, max_size=True)
        sig_rsrc = buffer_ops.create_buffer_resource(signal, max_size=True)

        nbytes_i32 = ArithValue(nbytes)
        vec_units = nbytes_i32 // fx.Int32(16)
        for unit in range(tid, vec_units, fx.Int32(BLOCK_THREADS)):
            # buffer offsets are in dtype elements: four i32 values = 16 bytes.
            elem = unit * fx.Int32(4)
            value = buffer_ops.buffer_load(
                src_rsrc, elem, vec_width=4, dtype=T.i32
            )
            buffer_ops.buffer_store(value, dst_rsrc, elem)

        tail_base = vec_units * fx.Int32(16)
        tail = tail_base + tid
        if tail < nbytes_i32:
            value = buffer_ops.buffer_load(src_rsrc, tail, vec_width=1, dtype=T.i8)
            buffer_ops.buffer_store(value, dst_rsrc, tail)

        gpu.barrier()
        if tid == fx.Int32(0):
            comm_ops.fence_system_release()
            buffer_ops.buffer_store(ArithValue(generation), sig_rsrc, fx.Int32(0))

    @flyc.jit
    def launch_copy_put_signal(
        src: fx.Tensor,
        dst: fx.Tensor,
        signal: fx.Tensor,
        nbytes: fx.Int32,
        generation: fx.Int64,
        stream: fx.Stream,
    ):
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            pass
        launch = copy_put_signal_kernel(src, dst, signal, nbytes, generation)
        launch.launch(
            grid=(1, 1, 1),
            block=(BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    return launch_copy_put_signal
