# SPDX-License-Identifier: MIT
"""Small device-side synchronization atoms shared by hierarchical kernels."""

import flydsl.compiler as flyc
import flydsl.expr as fx

from .. import comm_ops


@flyc.jit
def wait_i64_at_least_system_load(addr: fx.Int64, expected: fx.Int64):
    """Poll an arena generation written by a NIC/peer, then acquire payload.

    A system-scope atomic load is used deliberately. A plain LLVM load is not
    a safe polling primitive for memory that changes only through RDMA, while
    a fetch-add of zero needlessly turns every poll into a contended RMW.
    """

    seen = fx.Int64(comm_ops.load_i64_global_system(addr))
    while seen < expected:
        seen = fx.Int64(comm_ops.load_i64_global_system(addr))
    comm_ops.fence_system_acquire()


wait_i64_at_least_system = wait_i64_at_least_system_load


@flyc.jit
def wait_i32_count_system_load(
    ready_addr: fx.Int64, expected_addr: fx.Int64, index: fx.Int32
):
    byte = fx.Int64(index) * fx.Int64(4)
    expected = fx.Int32(comm_ops.load_i32_global_system(expected_addr + byte))
    ready = fx.Int32(comm_ops.load_i32_global_system(ready_addr + byte))
    while ready < expected:
        ready = fx.Int32(comm_ops.load_i32_global_system(ready_addr + byte))
    comm_ops.fence_system_acquire()


wait_i32_count_system = wait_i32_count_system_load


@flyc.jit
def publish_generation_system(addr: fx.Int64, generation: fx.Int64):
    comm_ops.fence_system_release()
    comm_ops.store_i64_global_system(addr, generation)


@flyc.jit
def increment_i32_system(addr: fx.Int64, index: fx.Int32):
    return fx.Int32(
        comm_ops.atomic_add_system(
            addr + fx.Int64(index) * fx.Int64(4), fx.Int32(1)
        )
    )
