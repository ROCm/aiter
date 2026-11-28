# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Device helpers for the MegaMoE EP kernel: buffer / LDS access, memory-scope
atomics and waits, and bf16 packing, imported by the other EP modules from here."""

from __future__ import annotations

from flydsl._mlir.dialects import llvm

from ..mega_moe_tp.common import (  # noqa: F401  (re-exported device primitives)
    ACQ_REL,
    AUX_SYS,
    V4I,
    _ctpop,
    alive,
    before,
    bf16x8_to_f32,
    bld,
    bst,
    g_add_agent,
    g_ld_sys,
    g_or_agent,
    g_st_sys,
    i32,
    lds_atomic_add,
    lds_atomic_or,
    lds_ld_acq,
    lds_ld_i32,
    lds_st,
    lds_st_rel,
    now,
    pack_bf16x8,
    poll_sys_ge,
    rsrc,
    uni,
    uni64,
    wait_lgkm0,
    wait_vm,
    wave_rank,
)


def device_trap():
    """Abort the kernel (``s_trap 2``, what ``llvm.trap`` lowers to on AMDGPU)."""
    llvm.InlineAsmOp(None, [], "s_trap 2", "", has_side_effects=True)
