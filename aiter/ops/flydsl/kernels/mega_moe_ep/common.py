# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Device helpers for the MegaMoE EP kernel.

These are the TP kernel's arena / quant / sync primitives (:mod:`mega_moe_tp.common`)
re-exported unchanged: the EP kernel reuses the same hipIpc symmetric-arena buffer
access, E8M0 / FP4 / FP8 packing and epoch-flag spin mechanics. EP-only device
helpers are added at the bottom. Importing from here (rather than reaching into the TP
package) keeps the EP device code's dependency surface in one place.
"""

from __future__ import annotations

from flydsl._mlir.dialects import llvm

from ..mega_moe_tp.common import (  # noqa: F401  (re-exported device primitives)
    ACQ,
    ACQ_REL,
    AUX_SC1,
    AUX_SYS,
    DEADLINE,
    MAX_TP,
    MONO,
    REL,
    V4I,
    _ctpop,
    alive,
    amax,
    before,
    bf16x8_to_f32,
    bld,
    bst,
    cat8,
    dma4,
    dma16,
    e8_scale,
    fp4_pack,
    fp8x4_pack,
    fp8x4_unpack,
    g_add_agent,
    g_ld_i32,
    g_ld_sys,
    g_or_agent,
    g_st_sys,
    gptr,
    i32,
    lds_atomic_add,
    lds_atomic_or,
    lds_cas,
    lds_ld,
    lds_ld_acq,
    lds_ld_i32,
    lds_p,
    lds_ptr,
    lds_st,
    lds_st_rel,
    mfma,
    mxfp8x8,
    now,
    pack_bf16x8,
    poll_sys_ge,
    rsrc,
    scales4,
    spin0,
    spin_lds_ge,
    sum_live,
    uni,
    uni64,
    wait_lgkm0,
    wait_vm,
    wave_rank,
    wave_red,
    widen,
)

# One expert-parallel peer per rank (<= 8 ranks on one node, same as the TP group).
MAX_NPES = MAX_TP


def device_trap():
    """Abort the kernel (``s_trap 2``, what ``llvm.trap`` lowers to on AMDGPU)."""
    llvm.InlineAsmOp(None, [], "s_trap 2", "", has_side_effects=True)
