# SPDX-License-Identifier: MIT
"""Node-local reduction of rank-local EP partials.

This is deliberately a separate kernel from GMM2.  GMM2 produces one weighted
partial per source token and local EP rank; this kernel waits until the node's
rank slots are complete, sums those slots in FP32, and publishes one absolute
generation per source token.  Tensor-parallel reduction is a later pipeline
stage and is not part of this contract.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T

from .. import comm_ops
from ..gemm_common import global_typed_ptr

from .hier_sync import publish_generation_system, wait_i32_count_system


def compile_node_partial_reduce(
    *,
    D_HIDDEN: int,
    NUM_RANKS: int = 8,
    output_dtype: str = "bf16",
    threads: int = 256,
):
    """Compile a node-local EP-partial reducer.

    ``arg_rank_partials`` is a contiguous BF16 slab with logical shape
    ``[NUM_RANKS, source_capacity, D_HIDDEN]``.  Slots which do not contribute
    to a source token must be zero-filled. ``rank_route_expected[source]`` is
    the number of remote/local slot-completion publications expected for that
    token; it gates reads but is not a rank mask.

    The accumulation type is always FP32. ``output_dtype`` selects only the
    final node-partial storage type. No TP reduction is performed here.
    """

    output_dtype = str(output_dtype).lower()
    if D_HIDDEN <= 0:
        raise ValueError("D_HIDDEN must be positive")
    if not 1 <= NUM_RANKS <= 8:
        raise ValueError("NUM_RANKS must be in [1, 8]")
    if output_dtype not in ("bf16", "fp32"):
        raise ValueError("output_dtype must be 'bf16' or 'fp32'")
    if threads not in (64, 128, 256):
        raise ValueError("threads must be one of 64,128,256")

    out_tag = "bf16" if output_dtype == "bf16" else "fp32"
    name = (
        f"megamoe_tile_node_partial_reduce_v1_h{D_HIDDEN}_"
        f"r{NUM_RANKS}_{out_tag}_t{threads}"
    )

    @flyc.kernel(name=name, known_block_size=[threads, 1, 1])
    def kernel(
        arg_rank_partials: fx.Int64,
        arg_rank_route_ready: fx.Int64,
        arg_rank_route_expected: fx.Int64,
        arg_node_partial: fx.Int64,
        arg_node_partial_ready: fx.Int64,
        generation: fx.Int64,
        i32_source_capacity: fx.Int32,
    ):
        source = fx.Int32(gpu.block_id("x"))
        tx = fx.Int32(gpu.thread_id("x"))

        if tx == fx.Int32(0):
            wait_i32_count_system(
                arg_rank_route_ready,
                arg_rank_route_expected,
                source,
            )
        gpu.barrier()
        # The polling acquire belongs to the leader. Every worker invalidates
        # its own view before reading its slice of the partial payload.
        comm_ops.fence_system_acquire()

        partials = global_typed_ptr(arg_rank_partials, T.bf16, align=2)
        if const_expr(output_dtype == "bf16"):
            output = global_typed_ptr(arg_node_partial, T.bf16, align=2)
        else:
            output = global_typed_ptr(arg_node_partial, T.f32, align=4)

        source_stride = fx.Int64(i32_source_capacity) * fx.Int64(D_HIDDEN)
        source_base = fx.Int64(source) * fx.Int64(D_HIDDEN)
        for col in range(tx, D_HIDDEN, threads):
            out_index = source_base + fx.Int64(col)
            total = fx.Float32(0.0)
            for rank in range_constexpr(NUM_RANKS):
                in_index = fx.Int64(rank) * source_stride + out_index
                value = fx.ptr_load(partials + in_index)
                total = total + fx.Float32(value)
            if const_expr(output_dtype == "bf16"):
                fx.ptr_store(fx.BFloat16(total), output + out_index)
            else:
                fx.ptr_store(total, output + out_index)

        rocdl.s_waitcnt(0)
        gpu.barrier()
        if tx == fx.Int32(0):
            publish_generation_system(
                arg_node_partial_ready + fx.Int64(source) * fx.Int64(8),
                generation,
            )

    @flyc.jit
    def launch_node_partial_v1(
        arg_rank_partials: fx.Int64,
        arg_rank_route_ready: fx.Int64,
        arg_rank_route_expected: fx.Int64,
        arg_node_partial: fx.Int64,
        arg_node_partial_ready: fx.Int64,
        generation: fx.Int64,
        i32_source_capacity: fx.Int32,
        i32_active_sources: fx.Int32,
        stream: fx.Stream,
    ):
        kernel(
            arg_rank_partials,
            arg_rank_route_ready,
            arg_rank_route_expected,
            arg_node_partial,
            arg_node_partial_ready,
            generation,
            i32_source_capacity,
        ).launch(
            grid=(i32_active_sources, 1, 1),
            block=(threads, 1, 1),
            stream=stream,
        )

    launch_node_partial_v1.kernel_name = name
    launch_node_partial_v1.num_ranks = NUM_RANKS
    launch_node_partial_v1.input_dtype = "bf16"
    launch_node_partial_v1.output_dtype = output_dtype
    launch_node_partial_v1.input_layout = (
        f"[{NUM_RANKS}, source_capacity, {D_HIDDEN}]"
    )
    launch_node_partial_v1.output_contract = (
        "fp32-accumulated-node-local-ep-partial;no-tp-reduction"
    )
    launch_node_partial_v1.requires_zero_filled_missing_rank_slots = True
    return launch_node_partial_v1


def compile_node_partial_reduce_lsa(
    *,
    D_HIDDEN: int,
    NUM_RANKS: int = 8,
    output_dtype: str = "bf16",
    threads: int = 256,
):
    """Compile the zero-copy CCO-LSA variant of the node reducer.

    Every peer registers an identically sized CCO window and places its BF16
    ``[source_capacity, D_HIDDEN]`` rank partial at ``partial_offset``. The
    kernel receives the local ``RegisteredWindow.handle`` and resolves peer
    VAs with the public ``ccoGetLsaPeerPtr`` bridge; it never reconstructs a
    MORI-private peer stride. Missing peer rows retain the same zero-fill
    contract as :func:`compile_node_partial_reduce`.
    """

    # Keep the CCO/MORI dependency lazy: importing the ordinary copy-slab
    # reducer must remain possible when the optional CCO package is absent.
    from ..cco.ops import lsa_ptr

    output_dtype = str(output_dtype).lower()
    if D_HIDDEN <= 0:
        raise ValueError("D_HIDDEN must be positive")
    if not 1 <= NUM_RANKS <= 8:
        raise ValueError("NUM_RANKS must be in [1, 8]")
    if output_dtype not in ("bf16", "fp32"):
        raise ValueError("output_dtype must be 'bf16' or 'fp32'")
    if threads not in (64, 128, 256):
        raise ValueError("threads must be one of 64,128,256")

    name = (
        f"megamoe_tile_node_partial_reduce_lsa_v1_h{D_HIDDEN}_"
        f"r{NUM_RANKS}_{output_dtype}_t{threads}"
    )

    @flyc.kernel(name=name, known_block_size=[threads, 1, 1])
    def kernel(
        arg_window_handle: fx.Int64,
        u64_partial_offset: fx.Int64,
        arg_rank_route_ready: fx.Int64,
        arg_rank_route_expected: fx.Int64,
        arg_node_partial: fx.Int64,
        arg_node_partial_ready: fx.Int64,
        generation: fx.Int64,
    ):
        source = fx.Int32(gpu.block_id("x"))
        tx = fx.Int32(gpu.thread_id("x"))

        if tx == fx.Int32(0):
            wait_i32_count_system(
                arg_rank_route_ready,
                arg_rank_route_expected,
                source,
            )
        gpu.barrier()
        comm_ops.fence_system_acquire()

        peer_partials = [
            global_typed_ptr(
                lsa_ptr(
                    arg_window_handle,
                    fx.Int32(rank),
                    u64_partial_offset,
                ),
                T.bf16,
                align=2,
            )
            for rank in range_constexpr(NUM_RANKS)
        ]
        if const_expr(output_dtype == "bf16"):
            output = global_typed_ptr(arg_node_partial, T.bf16, align=2)
        else:
            output = global_typed_ptr(arg_node_partial, T.f32, align=4)

        row_base = fx.Int64(source) * fx.Int64(D_HIDDEN)
        for col in range(tx, D_HIDDEN, threads):
            index = row_base + fx.Int64(col)
            total = fx.Float32(0.0)
            for rank in range_constexpr(NUM_RANKS):
                total = total + fx.Float32(
                    fx.ptr_load(peer_partials[rank] + index)
                )
            if const_expr(output_dtype == "bf16"):
                fx.ptr_store(fx.BFloat16(total), output + index)
            else:
                fx.ptr_store(total, output + index)

        rocdl.s_waitcnt(0)
        gpu.barrier()
        if tx == fx.Int32(0):
            publish_generation_system(
                arg_node_partial_ready + fx.Int64(source) * fx.Int64(8),
                generation,
            )

    @flyc.jit
    def launch_node_partial_lsa_v1(
        arg_window_handle: fx.Int64,
        u64_partial_offset: fx.Int64,
        arg_rank_route_ready: fx.Int64,
        arg_rank_route_expected: fx.Int64,
        arg_node_partial: fx.Int64,
        arg_node_partial_ready: fx.Int64,
        generation: fx.Int64,
        i32_active_sources: fx.Int32,
        stream: fx.Stream,
    ):
        kernel(
            arg_window_handle,
            u64_partial_offset,
            arg_rank_route_ready,
            arg_rank_route_expected,
            arg_node_partial,
            arg_node_partial_ready,
            generation,
        ).launch(
            grid=(i32_active_sources, 1, 1),
            block=(threads, 1, 1),
            stream=stream,
        )

    launch_node_partial_lsa_v1.kernel_name = name
    launch_node_partial_lsa_v1.num_ranks = NUM_RANKS
    launch_node_partial_lsa_v1.input_dtype = "bf16"
    launch_node_partial_lsa_v1.output_dtype = output_dtype
    launch_node_partial_lsa_v1.input_layout = (
        f"peer_window+offset -> [{NUM_RANKS}][source_capacity,{D_HIDDEN}]"
    )
    launch_node_partial_lsa_v1.output_contract = (
        "zero-copy-lsa-fp32-node-local-ep-partial;no-tp-reduction"
    )
    launch_node_partial_lsa_v1.requires_registered_window_handle = True
    launch_node_partial_lsa_v1.requires_zero_filled_missing_rank_slots = True
    return launch_node_partial_lsa_v1


__all__ = ["compile_node_partial_reduce", "compile_node_partial_reduce_lsa"]
