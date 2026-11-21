# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""MI300X (gfx942) facts the kernels rely on, and helpers tied to the wave / XCD structure.

Grids sized from ``NUM_CU`` (persistent kernels, grid barriers) need the whole GPU: with
fewer free CUs they deadlock instead of slowing down.
"""

import flydsl.expr as fx
from flydsl._mlir.dialects import llvm
from flydsl.expr import rocdl
from flydsl.expr.typing import T

from .mem import _raw

NUM_CU = 304
LDS_BYTES_MAX = 65536  # per WG; a kernel over it fails at launch without raising

# cache policy bits of buffer / global loads and stores (the aux operand)
CM_SC0, CM_NT, CM_SC1 = 1, 2, 16
CM_SYS = CM_SC0 | CM_SC1  # system scope: loads bypass, stores write through the XCD L2s


def _u(v):
    """Wave-uniform copy (readfirstlane) of an i32."""
    # readfirstlane is unstable (not in fx.rocdl.__all__, FlyDSL 0.5.0) and has no stable
    # counterpart: fx.gpu offers lane_id and the shuffles, whose results are VGPRs.
    return fx.Int32(rocdl.readfirstlane(T.i32, _raw(fx.Int32(v))))


def _readlane(v, lane):
    """The i32 ``v`` of the wave-uniform, run-time ``lane``, in an SGPR."""
    # readlane is unstable (not in fx.rocdl.__all__). Its stable counterpart,
    # fx.gpu.shuffle_idx, gives v_readlane only for a compile-time lane; for a run-time
    # one it is a ds_bpermute into a VGPR (gfx942 compile-only probe).
    return fx.Int32(rocdl.readlane(T.i32, _raw(fx.Int32(v)), _raw(fx.Int32(lane))))


def _sleep(n):
    """s_sleep n (about 64 n cycles) between two polls of a spin loop."""
    # s_sleep is unstable (not in fx.rocdl.__all__) and has no stable counterpart.
    rocdl.s_sleep(n)


def _sched_fence():
    """No instruction is scheduled across this point (sched_barrier 0)."""
    # sched_barrier is unstable (not in fx.rocdl.__all__). The stable fx.rocdl.sched_*
    # helpers are sched_group_barriers: they order groups of instructions, not a fence.
    rocdl.sched_barrier(0)


def _opaque_tx(tx):
    """(tx, lane, wave) re-derived from an asm copy of tx right before a tile body:
    LLVM cannot hoist the body's per-lane address math (out of the tile loop / to
    the kernel start), whose long live ranges would spill."""
    # Raw asm: FlyDSL 0.5.0 has no stable opaque value. kernels_common.pin_i32 is a
    # pure asm, which loop-invariant code motion still hoists; this one has side
    # effects, so it stays in the tile body. With a plain tx the main kernel spills 84
    # VGPRs (320 B scratch per lane) and 67 SGPRs, against 0 and 9 (gfx942
    # compile-only, every config).
    t = fx.Int32(
        llvm.inline_asm(
            T.i32, [_raw(tx)], "v_mov_b32 $0, $1", "=v,v", has_side_effects=True
        )
    )
    return (
        t,
        t % fx.Int32(64),
        _u(t // fx.Int32(64)),
    )
