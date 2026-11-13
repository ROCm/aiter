# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Optional ``sched_vgpr`` / ``sched_bank`` hints for gfx1250 MXFP8 compute GEMM.

Mirrors the FMHA HD192 / MXFP4 pin style, but uses flydsl 0.4.2
``rocdl.experimental.sched_*`` (``fx.set_register`` is 0.5+ only).

Env (all default off so stock path stays bit-stable unless opted in)::

    AITER_FLYDSL_MXFP8_SCHED_VGPR=1   pin acc / A / B windows
    AITER_FLYDSL_MXFP8_ASM_PIPE=1     widen WAR + ASM-style DS bursts / hints
    AITER_FLYDSL_MXFP8_PIN_SOFT=1     advisory pins instead of strict
"""

from __future__ import annotations

import os

from flydsl.expr import const_expr

try:
    from flydsl.expr.rocdl.experimental import sched_bank, sched_vgpr
except ImportError:  # pragma: no cover - older wheels without the LLVM patch
    sched_bank = None
    sched_vgpr = None

_SCHED_VGPR = int(os.environ.get("AITER_FLYDSL_MXFP8_SCHED_VGPR", "0"))
_ASM_PIPE = int(os.environ.get("AITER_FLYDSL_MXFP8_ASM_PIPE", "0"))
_PIN_SOFT = int(os.environ.get("AITER_FLYDSL_MXFP8_PIN_SOFT", "0"))

if _SCHED_VGPR not in (0, 1):
    raise ValueError("AITER_FLYDSL_MXFP8_SCHED_VGPR must be 0 or 1")
if _ASM_PIPE not in (0, 1):
    raise ValueError("AITER_FLYDSL_MXFP8_ASM_PIPE must be 0 or 1")
if _PIN_SOFT not in (0, 1):
    raise ValueError("AITER_FLYDSL_MXFP8_PIN_SOFT must be 0 or 1")

ENABLE_SCHED_VGPR = bool(_SCHED_VGPR and sched_vgpr is not None)
ENABLE_ASM_PIPE = bool(_ASM_PIPE)
PIN_SOFT = bool(_PIN_SOFT)

# Acc fragments (v8f32) sit in the upper half so A/B ping-pong can own 0..511,
# matching the MXFP4 explicit-partition convention (acc @512+).
ACC_VGPR_BASE = 512
# A/B 16-i32 fragments (ds_load_b128 x4 joined): two banks of ping-pong slots.
# Keep a single slot window (no phase split): dual-phase strict pins spilled.
A_VGPR_BASE = 0
B_VGPR_BASE = 256
FRAG_VGPRS = 16  # 16 x i32


def pin_vgpr(raw_val, regnum: int, soft: bool | None = None):
    """Identity at IR level; guides RA to ``regnum`` (start of wide value)."""
    if const_expr(not ENABLE_SCHED_VGPR) or sched_vgpr is None:
        return raw_val
    soft = PIN_SOFT if soft is None else soft
    try:
        return sched_vgpr(raw_val, regnum, soft=soft)
    except TypeError:  # older experimental signature
        return sched_vgpr(raw_val, regnum)


def pin_bank(raw_val, bank: int, soft: bool | None = None):
    if const_expr(not ENABLE_SCHED_VGPR) or sched_bank is None:
        return raw_val
    soft = PIN_SOFT if soft is None else soft
    try:
        return sched_bank(raw_val, bank, soft=soft)
    except TypeError:
        return sched_bank(raw_val, bank)


def pin_acc_vec(raw_val, frag_idx: int):
    """Pin one v8f32 accumulator fragment."""
    return pin_vgpr(raw_val, ACC_VGPR_BASE + frag_idx * 8)


def pin_a_frag(raw_val, slot: int, phase: int = 0):
    """Pin one 16-i32 A fragment (``slot`` in 0..half_m-1 within a stage).

    ``phase`` is accepted for call-site compatibility but ignored: dual-phase
    strict pins caused VGPR pressure / spills on the short-K WAR path.
    """
    del phase
    return pin_vgpr(raw_val, A_VGPR_BASE + slot * FRAG_VGPRS)


def pin_b_frag(raw_val, slot: int, phase: int = 0):
    del phase
    return pin_vgpr(raw_val, B_VGPR_BASE + slot * FRAG_VGPRS)
