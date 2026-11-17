# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Hooks: code that a persistent multi-role FlyDSL GEMM runs inside its kernel.

A host GEMM computes ``roles`` column blocks of one fused weight (Q, K and V for the
pre-attention monokernel), persistent over one workgroup per CU, and calls three
``@flyc.jit`` hooks at fixed points. The hooks are traced into the GEMM's kernel, so their
code runs between the GEMM's tiles without another launch. The INT8 4-wave GEMM
(``compile_int8_gemm_4w(roles=..., hooks=...)``) is the first host; any GEMM that keeps
this contract can host the same hooks.

Arguments
    ``role``: the role, a Python int. Hooks are traced once per role, so ``const_expr``
    branches on it cost nothing.
    ``aux``: Int64 address of the caller's aux buffer. It starts with the host's tile
    counters (below); the rest belongs to the hooks.
    ``args``: six Int64 values the caller passes to the launcher, opaque to the host.
    ``c_m``: Int32 rows of C.
    ``row_base``, ``col_base`` (tile hook only): Int32 first row and column of C of the
    calling wave's tile.

What the host provides
    - C is complete for every counted tile: ``[roles, M, role_cols]`` BF16, plus whatever the
      host's epilogue writes for the role (for the INT8 GEMM: per-head sums of squares and
      maxima, and optional per-column maxima).
    - Tile counters: one Int32 per role at ``aux + 4 * role``. Thread 0 of the workgroup adds
      1 per tile with an agent-scope release, after every wave's stores of the tile and the
      tile hook have completed. The host never resets the counters: a caller that reuses
      ``aux`` across calls waits for ``call * tiles_per_role``.
    - Order: each workgroup takes its tiles of role 0, then role 1, and so on. All of its
      tiles of a role are counted before its first tile of the next role starts.
    - Every workgroup is resident for the whole kernel (one per CU). A hook may therefore
      spin on another workgroup's counters of the same or an earlier role, but never on a
      later role's.

When each hook runs
    ``tile(role, aux, args, c_m, row_base, col_base)``: after each tile, in every wave of
    the workgroup, once the workgroup's global loads and stores have completed (vmcnt 0)
    and its waves have met at a barrier; before the tile is counted. A hook that writes
    global memory waits for its stores (vmcnt 0) and meets the other waves at a barrier
    before returning, so the count's release covers them.
    ``first_block(role, aux, args, c_m)``: roles 1 and later, once per workgroup, right after
    its first tile of the role is counted. Work deferred from the previous role (its
    quantize and copies) runs here, under the next role's tiles.
    ``role(role, aux, args, c_m)``: every workgroup, after its last tile of the role is
    counted and its loads have drained, including a workgroup with no tile of the role.

Rules for hooks
    - Every thread of the workgroup enters each hook and leaves it; hooks branch by thread
      or workgroup inside.
    - Hooks own no LDS: the host's LDS holds its operand buffers.
    - Hooks add to the host's registers; the host's accumulators are dead at every hook.
    - A runtime ``if`` may not call methods of a Python list; build lists with
      comprehensions outside the branch.
    - Spin-waits use acquire after release-ordered writes by the other side
      (``communication_ops_utils.spin_until_ge_*``).

``key`` names the hooks' code in the kernel name. Two different hooks with one key share a
compiled binary, so a key covers everything the hooks were built from.
"""

from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class GemmHooks:
    """The three hooks of a host GEMM, under this module's contract."""

    role: Callable
    first_block: Callable | None = None
    tile: Callable | None = None
    key: str = ""
