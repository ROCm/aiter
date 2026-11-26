# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Hooks: code that a persistent multi-role FlyDSL GEMM runs inside its kernel.

A host GEMM computes ``roles`` column blocks of one fused weight, persistent over its
workgroups, and calls up to three hooks at fixed points. The hooks are traced into the
host's kernel, so their code runs between the host's output blocks without another
launch. The contract is written for the pre-attention monokernel's fused Q/K/V
projection, and it encodes that use: the host's epilogue applies its Q/K prep to every role
but the last and per-column maxima (``col_amax_offset``) to the last role only, C is
planar, ``[roles, M, role_cols]``, while ``col_base`` counts columns of the fused weight,
and every host passes the hooks six opaque Int64 values. The host side of the protocol is
the functions below, which every host calls.

Terms: a *block* is a workgroup's 256x256 output block, the unit the counters count; a
*tile* is one wave's 128x128 part of a block.

Arguments
    ``role``: the role, a Python int. Hooks are traced once per role, so ``const_expr``
    branches on it cost nothing.
    ``aux``: Int64 address of the caller's aux buffer. It starts with the host's block
    counters (below); the rest belongs to the hooks.
    ``args``: six Int64 values the caller passes to the launcher, opaque to the host.
    ``c_m``: Int32 rows of C.
    ``row_base``, ``col_base`` (tile hook only): Int32 first row of C and first column of
    the fused weight of the calling wave's tile.

What the host provides
    - C is complete for every counted block, plus whatever the host's epilogue writes for
      the role (per-head maxima and row-factor partials, and optional per-column maxima).
    - Block counters: one Int32 per role at ``aux + 4 * role``. Thread 0 of the workgroup
      adds 1 per block with an agent-scope release, after every wave's stores of the block
      and the tile hook have completed. The host never resets the counters: a caller that
      reuses ``aux`` across calls waits for ``call * blocks_per_role``.
    - Order: each workgroup takes its blocks of role 0, then role 1, and so on. All of its
      blocks of a role are counted before its first block of the next role starts.

What the caller provides
    - Residency: every workgroup is resident for the whole kernel, one per CU. The host
      checks that its grid fits the device (``validate_roles``), but it cannot see a CU
      mask (``HSA_CU_MASK``) or another kernel running beside it, such as an RCCL
      all-to-all; the caller rules both out. A hook may then spin on another workgroup's
      counters of the same or an earlier role, never of a later role. Without residency
      such a spin hangs; a bounded spin turns the hang into an error.

When each hook runs
    ``tile(role, aux, args, c_m, row_base, col_base)``: after each block, in every wave,
    with that wave's tile, once the block's loads and stores have completed (vmcnt 0) and
    the waves have met at a barrier; before the block is counted. A hook that writes global
    memory waits for its stores (vmcnt 0) and meets the other waves at a barrier before
    returning, so the count's release covers them.
    ``first_block(role, aux, args, c_m)``: roles 1 and later, right after the workgroup's
    first block of the role is counted. It runs only on workgroups with a block of the role
    (at M = 1184 and role_cols = 5120, 156 of 256 workgroups have none). Work deferred from
    the previous role (its quantize and copies) runs here, under the role's blocks.
    ``role_end(role, aux, args, c_m)``: every workgroup, after its last block of the role is
    counted and its loads have drained, including a workgroup with no block of the role.

Rules for hooks
    - Hooks are FlyDSL functions traced into the kernel (``@flyc.jit``); the host does not
      check this.
    - Every thread of the workgroup enters each hook and leaves it; hooks branch by thread
      or workgroup inside.
    - Hooks own no LDS: the host's LDS holds its operand buffers.
    - Hooks add to the host's registers; the host's accumulators are dead at every hook.
    - A runtime ``if`` may not call methods of a Python list; build lists with
      comprehensions outside the branch.
    - Spin-waits use acquire after release-ordered writes by the other side
      (``communication_ops_utils.spin_until_ge_*``).

``key`` names the hooks in the kernel name. Kernel names are for attribution in profiles
and disassembly; FlyDSL keys compiled binaries on the traced code.
"""

import re
from collections.abc import Callable
from dataclasses import dataclass

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr

from aiter.jit.utils.chip_info import get_cdna4_num_xcds, get_cu_num
from aiter.ops.flydsl.kernels.communication_ops_utils import _to_ptr_global
from aiter.ops.flydsl.kernels.gemm_a8w8_8wave import _xcd_swizzle_any


@dataclass(frozen=True)
class GemmHooks:
    """The hooks of a host GEMM, in the order they run, under this module's contract."""

    tile: Callable | None = None
    first_block: Callable | None = None
    role_end: Callable | None = None
    key: str = ""


def validate_roles(
    *,
    roles,
    role_cols,
    block_n,
    hooks,
    prep,
    num_cu,
    num_xcds,
    xcd_swizzle,
    col_amax_offset,
):
    """Raise for a roles or hooks configuration that a host would run differently from
    what was asked, or whose grid cannot be resident on this device."""
    if roles < 1:
        raise ValueError(f"roles must be at least 1, got {roles}")
    if roles == 1:
        if hooks is not None:
            raise ValueError("hooks need roles > 1: a one-role host calls none of them")
        if col_amax_offset is not None:
            raise ValueError("col_amax_offset needs roles > 1")
        return
    if hooks is None:
        raise ValueError("roles > 1 needs hooks")
    if prep is None:
        raise ValueError(
            "roles > 1 needs a prep, which applies to every role but the last"
        )
    if role_cols <= 0 or role_cols % block_n:
        raise ValueError(
            f"role_cols must be a positive multiple of {block_n}, got {role_cols}"
        )
    if xcd_swizzle <= 0:
        raise ValueError("roles > 1 needs the swizzled block order (xcd_swizzle > 0)")
    if num_cu <= 0 or num_cu % num_xcds:
        raise ValueError(
            f"roles > 1 needs a persistent grid of a multiple of {num_xcds} XCDs, "
            f"got num_cu={num_cu}"
        )
    device_cu, device_xcds = get_cu_num(), get_cdna4_num_xcds()
    if num_cu > device_cu or num_xcds != device_xcds:
        raise ValueError(
            f"unsupported on this device: the hooks need all {num_cu} workgroups resident "
            f"over {num_xcds} XCDs, and the device has {device_cu} CUs over {device_xcds}"
        )


def name_suffix(*, roles, role_cols, hooks, col_amax_offset):
    """The kernel-name part for the roles and hooks, with the key made symbol-safe."""
    if roles == 1:
        return ""
    key = re.sub(r"[^0-9A-Za-z_]", "_", hooks.key)
    return (
        f"_roles{roles}x{role_cols}_hook{key}"
        + ("_th" if hooks.tile is not None else "")
        + ("_fb" if hooks.first_block is not None else "")
        + (f"_cmax{col_amax_offset}" if col_amax_offset is not None else "")
    )


def role_block_coords(
    block, role, *, m_blocks, role_cols, block_n, xcd_swizzle, num_xcds
):
    """Output block (m, n) of ``block``, counted within ``role``, in the swizzled order;
    ``n`` counts blocks of the fused weight. A persistent grid of a multiple of
    ``num_xcds`` workgroups keeps each block on its workgroup's XCD."""
    role_blocks = role_cols // block_n
    m, n = _xcd_swizzle_any(m_blocks, role_blocks, xcd_swizzle, block, num_xcds)
    return m, n + role * role_blocks


def make_host_steps(hooks):
    """The host's two steps around the hooks:

    ``after_block(role, aux, args, c_m, row_base, col_base, block, block0)``, right after
    the block's stores are issued: waits for every wave's stores of the block (vmcnt 0)
    and meets the waves at a barrier, then runs the tile hook, counts the block and, for
    the workgroup's first block of a role after the first, runs the first-block hook.
    ``end_role(role, aux, args, c_m)``, after the workgroup's last block of the role once
    its loads have drained: the role-end hook.
    """

    @flyc.jit
    def after_block(role, aux, args, c_m, row_base, col_base, block, block0):
        fx.rocdl.s_waitcnt(vmcnt=0)
        fx.barrier()
        if const_expr(hooks.tile is not None):
            hooks.tile(role, aux, args, c_m, row_base, col_base)
        if fx.thread_idx.x == 0:
            fx.atomic_add(
                _to_ptr_global(aux + role * 4, fx.Int32),
                fx.Int32(1),
                syncscope=fx.rocdl.SyncScope.Agent,
                ordering=fx.AtomicOrdering.Release,
            )
        # A trace-time branch around a runtime one; they do not merge.
        if const_expr(hooks.first_block is not None and role > 0):  # noqa: SIM102
            if fx.Int32(block) == block0:
                hooks.first_block(role, aux, args, c_m)

    def end_role(role, aux, args, c_m):
        if hooks.role_end is not None:
            hooks.role_end(role, aux, args, c_m)

    return after_block, end_role
