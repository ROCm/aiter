# SPDX-License-Identifier: MIT
"""Device-side node epoch gate for CCO-LSA rank partials."""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import gpu

from .. import comm_ops

from .hier_sync import publish_generation_system, wait_i64_at_least_system


def compile_rank_partial_epoch_gate_lsa(
    *, NUM_RANKS: int = 8, threads: int = 64
):
    """Compile a publish-before-wait gate for one node's rank partials.

    The launcher must run on the same stream after the local rank-partial
    payload writer. Thread zero release-publishes the local absolute
    generation first; one lane per LSA peer then waits for that peer's matching
    generation with system-acquire semantics. Returning from this kernel gates
    a later same-stream node reducer without a host synchronize or Gloo
    barrier.

    ``u64_ready_offset`` is relative to every identically registered CCO
    window. ``arg_local_ready`` is the already-resolved local pointer for the
    same offset.
    """

    if not 1 <= int(NUM_RANKS) <= 64:
        raise ValueError("NUM_RANKS must be in [1, 64]")
    if int(threads) not in (64, 128, 256):
        raise ValueError("threads must be one of 64,128,256")
    if int(threads) < int(NUM_RANKS):
        raise ValueError("threads must cover every LSA peer rank")

    # Keep the optional CCO dependency lazy for ordinary compute-only imports.
    from ..cco.ops import lsa_ptr

    name = f"megamoe_tile_rank_partial_epoch_lsa_r{NUM_RANKS}_t{threads}"

    @flyc.kernel(name=name, known_block_size=[threads, 1, 1])
    def kernel(
        arg_window_handle: fx.Int64,
        arg_local_ready: fx.Int64,
        u64_ready_offset: fx.Int64,
        generation: fx.Int64,
    ):
        tx = fx.Int32(gpu.thread_id("x"))

        # Publish before polling so every rank can make progress even when all
        # eight gate CTAs become resident at the same time.
        if tx == fx.Int32(0):
            publish_generation_system(arg_local_ready, generation)
        gpu.barrier()

        if tx < fx.Int32(NUM_RANKS):
            peer_ready = lsa_ptr(
                arg_window_handle,
                tx,
                u64_ready_offset,
            )
            wait_i64_at_least_system(peer_ready, generation)
        gpu.barrier()

        # The wait primitive already acquires on each polling lane. Keep an
        # acquire on every gate worker as an explicit kernel-boundary contract;
        # the downstream reducer also acquires before its peer payload loads.
        comm_ops.fence_system_acquire()

    @flyc.jit
    def launch_rank_partial_epoch_lsa(
        arg_window_handle: fx.Int64,
        arg_local_ready: fx.Int64,
        u64_ready_offset: fx.Int64,
        generation: fx.Int64,
        stream: fx.Stream,
    ):
        kernel(
            arg_window_handle,
            arg_local_ready,
            u64_ready_offset,
            generation,
        ).launch(
            grid=(1, 1, 1),
            block=(threads, 1, 1),
            stream=stream,
        )

    launch_rank_partial_epoch_lsa.kernel_name = name
    launch_rank_partial_epoch_lsa.num_ranks = int(NUM_RANKS)
    launch_rank_partial_epoch_lsa.threads = int(threads)
    launch_rank_partial_epoch_lsa.ready_kind = "absolute-generation"
    launch_rank_partial_epoch_lsa.publish_before_wait = True
    launch_rank_partial_epoch_lsa.requires_registered_window_handle = True
    launch_rank_partial_epoch_lsa.memory_order = "system-release-acquire"
    return launch_rank_partial_epoch_lsa


__all__ = ["compile_rank_partial_epoch_gate_lsa"]
