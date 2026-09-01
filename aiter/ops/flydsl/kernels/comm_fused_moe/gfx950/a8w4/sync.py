# SPDX-License-Identifier: Apache-2.0
"""Synchronization primitives shared by communication-fused MoE kernels."""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import gpu, ptrtoint

from .... import communication_ops_utils as comm_ops

FLAT_VA_RANK_STRIDE = 1 << 32


def peer_base(flat_base, peer):
    return flat_base + fx.Int64(peer) * fx.Int64(FLAT_VA_RANK_STRIDE)


@functools.cache
def compile_epoch_barrier(tp_size: int):
    """Publish one symmetric workspace epoch and acquire all TP peers."""

    @flyc.kernel(
        name=f"comm_fused_moe_epoch_tp{tp_size}",
        known_block_size=[64, 1, 1],
    )
    def kernel(
        workspace: fx.Pointer,
        flat_base: fx.Int64,
        ready_offset: fx.Int64,
    ):
        tid = fx.Int32(gpu.thread_idx.x)
        local_base = fx.Int64(ptrtoint(workspace))
        expected = fx.Int64(
            comm_ops.load_i64_global(local_base + ready_offset)
        ) + fx.Int64(1)
        if tid == fx.Int32(0):
            comm_ops.store_i64_global_system(local_base + ready_offset, expected)
        gpu.barrier()
        if tid < fx.Int32(tp_size):
            peer_addr = peer_base(flat_base, tid)
            comm_ops.spin_until_ge_i64(peer_addr + ready_offset, expected)
            comm_ops.fence_system_acquire()

    def launch(workspace, flat_base, ready_offset, stream):
        kernel(workspace, flat_base, ready_offset).launch(
            grid=(1, 1, 1), block=(64, 1, 1), stream=stream
        )

    launch.__name__ = f"launch_comm_fused_moe_epoch_tp{tp_size}"
    return flyc.jit(launch)
