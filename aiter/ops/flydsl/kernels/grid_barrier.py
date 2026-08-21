# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Device-wide barrier for persistent FlyDSL grids.

A grid barrier only terminates if every block is resident at the same time: a
block still waiting to be scheduled can never reach the barrier the resident
ones are spinning on. That makes it unusable for a grid sized by tile count --
which is what the MoE GEMMs launch -- and usable only for a *persistent* grid,
launched at ``max_coresident_blocks`` and looping over tiles internally.

The counter is monotonic and keyed by a compile-time generation index rather
than reset between barriers: a block that has passed barrier ``g`` may reach
``g+1`` and bump the counter while a slower block is still spinning on ``g``,
so an exact-equality wait would miss its wakeup. The host zeroes the counter
once before the launch; it ends at ``n_barriers * expected_blocks``.
"""

import torch
from flydsl._mlir import ir
from flydsl._mlir.dialects import scf
from flydsl.expr import arith, gpu
from flydsl.expr.arith import CmpIPredicate
from flydsl.expr.arith import _to_raw as _raw
from flydsl.expr.typing import T

from aiter.ops.flydsl.kernels import communication_ops_utils as comm_ops

# gfx1250 holds 320 KB of LDS per CU and schedules at most this many waves of
# work per CU; both cap how many blocks can be co-resident.
_MAX_BLOCKS_PER_CU = 8


def emit_grid_barrier(bar_addr_i64, expected_blocks, generation: int, tid):
    """Emit one device-wide barrier.

    ``bar_addr_i64`` is a zeroed i32 counter, ``expected_blocks`` the runtime
    block count the grid was launched with, ``generation`` the 0-based index of
    this barrier within the kernel, and ``tid`` the block-local thread id.

    Raw ``scf.IfOp`` rather than a Python ``if``: FlyDSL only AST-rewrites
    dynamic conditionals inside a ``@flyc.kernel`` body, so a predicate written
    normally in this module-level emitter would be evaluated host-side.
    """
    i32 = T.i32
    # s_barrier syncs waves but emits no implicit waitcnt, so drain the memory
    # counters first or this block's stores may not be visible to the peers it
    # is about to release.
    comm_ops.waitcnt_all()
    comm_ops.fence_agent_release()
    gpu.barrier()

    is_leader = arith.cmpi(CmpIPredicate.eq, _raw(tid), arith.constant(0, type=i32))
    _if = scf.IfOp(is_leader)
    with ir.InsertionPoint(_if.then_block):
        comm_ops.atomic_add_agent(bar_addr_i64, arith.constant(1, type=i32))
        target = arith.muli(
            _raw(expected_blocks), arith.constant(generation + 1, type=i32)
        )
        # spin_until_gt(target-1) == wait for >=target; see the module docstring
        # on why this cannot be an equality wait.
        comm_ops.spin_until_gt_i32(
            bar_addr_i64, arith.subi(target, arith.constant(1, type=i32))
        )
        scf.YieldOp([])

    gpu.barrier()
    comm_ops.fence_agent_acquire()


def max_coresident_blocks(
    lds_bytes_per_block: int,
    threads_per_block: int,
    device: torch.device | int | None = None,
) -> int:
    """Blocks that fit on the device at once, the ceiling for a grid barrier.

    Deliberately conservative: overestimating deadlocks the barrier, while
    underestimating only costs a persistent kernel some parallelism.
    """
    props = torch.cuda.get_device_properties(device)
    cus = int(props.multi_processor_count)
    lds_per_cu = int(props.shared_memory_per_block)

    by_lds = lds_per_cu // max(int(lds_bytes_per_block), 1)
    by_threads = 2048 // max(int(threads_per_block), 1)
    per_cu = max(1, min(by_lds, by_threads, _MAX_BLOCKS_PER_CU))
    return cus * per_cu


def grid_barrier_counter(device=None) -> torch.Tensor:
    """A zeroed counter sized for one kernel's worth of grid barriers."""
    return torch.zeros(1, dtype=torch.int32, device=device)
