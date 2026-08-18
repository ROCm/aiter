# SPDX-License-Identifier: MIT
"""Final two-node EP-partial combine.

The kernel waits for one local and one returned remote node partial, performs
the elementwise add in FP32, stores BF16 or FP32, and publishes an absolute
generation.  It intentionally does not perform tensor-parallel reduction;
the published result is the input to a later TP all-reduce when TP is enabled.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, rocdl
from flydsl.expr.typing import T

from .. import comm_ops
from ..gemm_common import global_typed_ptr

from .hier_sync import publish_generation_system, wait_i64_at_least_system


def _normalize_dtype(value: str, name: str) -> str:
    value = str(value).lower()
    aliases = {"bfloat16": "bf16", "f32": "fp32", "float32": "fp32"}
    value = aliases.get(value, value)
    if value not in ("bf16", "fp32"):
        raise ValueError(f"{name} must be 'bf16' or 'fp32'")
    return value


def compile_final_combine(
    *,
    D_HIDDEN: int,
    local_dtype: str = "bf16",
    remote_dtype: str | None = None,
    output_dtype: str = "bf16",
    threads: int = 256,
):
    """Compile the final two-node EP combine kernel.

    Inputs and output use contiguous ``[source_capacity, D_HIDDEN]`` logical
    layouts. ``local_ready[source]`` and ``remote_ready[source]`` are absolute
    generations. One CTA owns one source row, so publication needs no extra
    hidden-tile done counter.
    """

    if D_HIDDEN <= 0:
        raise ValueError("D_HIDDEN must be positive")
    local_dtype = _normalize_dtype(local_dtype, "local_dtype")
    remote_dtype = _normalize_dtype(
        local_dtype if remote_dtype is None else remote_dtype,
        "remote_dtype",
    )
    output_dtype = _normalize_dtype(output_dtype, "output_dtype")
    if threads not in (64, 128, 256):
        raise ValueError("threads must be one of 64,128,256")

    name = (
        f"megamoe_tile_final_combine_v1_h{D_HIDDEN}_"
        f"l{local_dtype}_r{remote_dtype}_o{output_dtype}_t{threads}"
    )

    @flyc.kernel(name=name, known_block_size=[threads, 1, 1])
    def kernel(
        arg_local_partial: fx.Int64,
        arg_remote_partial: fx.Int64,
        arg_local_ready: fx.Int64,
        arg_remote_ready: fx.Int64,
        arg_output: fx.Int64,
        arg_final_output_ready: fx.Int64,
        generation: fx.Int64,
    ):
        source = fx.Int32(gpu.block_id("x"))
        tx = fx.Int32(gpu.thread_id("x"))

        if tx == fx.Int32(0):
            wait_i64_at_least_system(
                arg_local_ready + fx.Int64(source) * fx.Int64(8),
                generation,
            )
            wait_i64_at_least_system(
                arg_remote_ready + fx.Int64(source) * fx.Int64(8),
                generation,
            )
        gpu.barrier()
        comm_ops.fence_system_acquire()

        if const_expr(local_dtype == "bf16"):
            local = global_typed_ptr(arg_local_partial, T.bf16, align=2)
        else:
            local = global_typed_ptr(arg_local_partial, T.f32, align=4)
        if const_expr(remote_dtype == "bf16"):
            remote = global_typed_ptr(arg_remote_partial, T.bf16, align=2)
        else:
            remote = global_typed_ptr(arg_remote_partial, T.f32, align=4)
        if const_expr(output_dtype == "bf16"):
            output = global_typed_ptr(arg_output, T.bf16, align=2)
        else:
            output = global_typed_ptr(arg_output, T.f32, align=4)

        row_base = fx.Int64(source) * fx.Int64(D_HIDDEN)
        for col in range(tx, D_HIDDEN, threads):
            index = row_base + fx.Int64(col)
            value = fx.Float32(fx.ptr_load(local + index)) + fx.Float32(
                fx.ptr_load(remote + index)
            )
            if const_expr(output_dtype == "bf16"):
                fx.ptr_store(fx.BFloat16(value), output + index)
            else:
                fx.ptr_store(value, output + index)

        rocdl.s_waitcnt(0)
        gpu.barrier()
        if tx == fx.Int32(0):
            publish_generation_system(
                arg_final_output_ready + fx.Int64(source) * fx.Int64(8),
                generation,
            )

    @flyc.jit
    def launch_final_combine_v1(
        arg_local_partial: fx.Int64,
        arg_remote_partial: fx.Int64,
        arg_local_ready: fx.Int64,
        arg_remote_ready: fx.Int64,
        arg_output: fx.Int64,
        arg_final_output_ready: fx.Int64,
        generation: fx.Int64,
        i32_active_sources: fx.Int32,
        stream: fx.Stream,
    ):
        kernel(
            arg_local_partial,
            arg_remote_partial,
            arg_local_ready,
            arg_remote_ready,
            arg_output,
            arg_final_output_ready,
            generation,
        ).launch(
            grid=(i32_active_sources, 1, 1),
            block=(threads, 1, 1),
            stream=stream,
        )

    launch_final_combine_v1.kernel_name = name
    launch_final_combine_v1.local_dtype = local_dtype
    launch_final_combine_v1.remote_dtype = remote_dtype
    launch_final_combine_v1.output_dtype = output_dtype
    launch_final_combine_v1.output_contract = (
        "two-node-ep-partial-sum;no-tp-reduction"
    )
    launch_final_combine_v1.requires_tp_all_reduce = True
    return launch_final_combine_v1


__all__ = ["compile_final_combine"]
