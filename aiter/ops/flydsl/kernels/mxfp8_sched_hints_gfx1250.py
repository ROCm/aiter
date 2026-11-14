# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Optional ``sched_vgpr`` / ``sched_bank`` hints for gfx1250 MXFP8 compute GEMM.

Mirrors the FMHA HD192 / MXFP4 pin style, but uses flydsl 0.4.2
``rocdl.experimental.sched_*`` (``fx.set_register`` is 0.5+ only).

``sched_vgpr`` only pins **register-held results** (ds_load destinations,
WMMA accumulators, scale words) — not LDS address SSA / index values.

Opt-in env (default off keeps the stock path bit-stable)::

    AITER_FLYDSL_MXFP8_SCHED_VGPR=1   pin Acc / A / B / scale result windows
    AITER_FLYDSL_MXFP8_ASM_PIPE=1     ASM-style DS bursts / sched hints
    AITER_FLYDSL_MXFP8_PIN_SOFT=1     advisory pins instead of strict
    AITER_FLYDSL_MXFP8_SA_PANEL_CLASS0=1|2|4|8  short-K SA panel (default 4)
    AITER_FLYDSL_MXFP8_PIN_LAYOUT=mxfp4|asm
    AITER_FLYDSL_MXFP8_PIN_B128=0|1   pin each ds_load_b128 dest (default 0)
    AITER_FLYDSL_MXFP8_SHORT_BEAT=0|1 light beats on short-K (default 0)
    AITER_FLYDSL_MXFP8_BANK_HINT=0|1|2
    AITER_FLYDSL_MXFP8_SAME_MSB=0|1   ASM absolute pins + bank (measured regress)

Winning mid-K behavior (dense producers, load-first on persistent, scale pin
on class1+ps) is hardcoded in the GEMM kernel when ``ASM_PIPE`` /
``SCHED_VGPR`` are on — no extra knobs.
"""

from __future__ import annotations

import os

import flydsl.expr as fx
from flydsl.expr import const_expr

try:
    from flydsl.expr.rocdl.experimental import sched_bank, sched_vgpr
except ImportError:  # pragma: no cover - older wheels without the LLVM patch
    sched_bank = None
    sched_vgpr = None

_SCHED_VGPR = int(os.environ.get("AITER_FLYDSL_MXFP8_SCHED_VGPR", "0"))
_ASM_PIPE = int(os.environ.get("AITER_FLYDSL_MXFP8_ASM_PIPE", "0"))
_PIN_SOFT = int(os.environ.get("AITER_FLYDSL_MXFP8_PIN_SOFT", "0"))
# Default 4: short-K A-scale panel (c_relocate needs stages==1 and regresses
# on the gemm-request huge-N short-K shape when forced).
_SA_PANEL_CLASS0 = int(os.environ.get("AITER_FLYDSL_MXFP8_SA_PANEL_CLASS0", "4"))
_PIN_LAYOUT = os.environ.get("AITER_FLYDSL_MXFP8_PIN_LAYOUT", "mxfp4").strip().lower()
_PIN_B128 = int(os.environ.get("AITER_FLYDSL_MXFP8_PIN_B128", "0"))
_SHORT_BEAT = int(os.environ.get("AITER_FLYDSL_MXFP8_SHORT_BEAT", "0"))
_BANK_HINT = int(os.environ.get("AITER_FLYDSL_MXFP8_BANK_HINT", "0"))
_SAME_MSB = int(os.environ.get("AITER_FLYDSL_MXFP8_SAME_MSB", "0"))

if _SCHED_VGPR not in (0, 1):
    raise ValueError("AITER_FLYDSL_MXFP8_SCHED_VGPR must be 0 or 1")
if _ASM_PIPE not in (0, 1):
    raise ValueError("AITER_FLYDSL_MXFP8_ASM_PIPE must be 0 or 1")
if _PIN_SOFT not in (0, 1):
    raise ValueError("AITER_FLYDSL_MXFP8_PIN_SOFT must be 0 or 1")
if _SA_PANEL_CLASS0 not in (1, 2, 4, 8):
    raise ValueError("AITER_FLYDSL_MXFP8_SA_PANEL_CLASS0 must be 1, 2, 4, or 8")
if _PIN_LAYOUT not in ("mxfp4", "asm"):
    raise ValueError("AITER_FLYDSL_MXFP8_PIN_LAYOUT must be mxfp4 or asm")
if _PIN_B128 not in (0, 1):
    raise ValueError("AITER_FLYDSL_MXFP8_PIN_B128 must be 0 or 1")
if _SHORT_BEAT not in (0, 1):
    raise ValueError("AITER_FLYDSL_MXFP8_SHORT_BEAT must be 0 or 1")
if _BANK_HINT not in (0, 1, 2):
    raise ValueError("AITER_FLYDSL_MXFP8_BANK_HINT must be 0, 1, or 2")
if _SAME_MSB not in (0, 1):
    raise ValueError("AITER_FLYDSL_MXFP8_SAME_MSB must be 0 or 1")

ENABLE_SCHED_VGPR = bool(_SCHED_VGPR and sched_vgpr is not None)
ENABLE_ASM_PIPE = bool(_ASM_PIPE)
SA_PANEL_CLASS0 = _SA_PANEL_CLASS0
# Mid-K keeps the legacy fat A-scale panel (not env-tunable).
SA_PANEL_CLASS1 = 8
PIN_SOFT = bool(_PIN_SOFT)
ENABLE_SAME_MSB = bool(_SAME_MSB and ENABLE_SCHED_VGPR)
PIN_LAYOUT = "asm" if ENABLE_SAME_MSB else _PIN_LAYOUT
ENABLE_PIN_B128 = bool((_PIN_B128 or ENABLE_SAME_MSB) and ENABLE_SCHED_VGPR)
ENABLE_SHORT_BEAT = bool(_SHORT_BEAT)
BANK_HINT = 0 if ENABLE_SAME_MSB else (_BANK_HINT if ENABLE_SCHED_VGPR else 0)
ENABLE_BANK_HINT = bool(BANK_HINT)

# ── mxfp4 partition (legacy / measured best on gemm-request) ───────────
ACC_VGPR_BASE = 512
A_VGPR_BASE = 0
B_VGPR_BASE = 256
FRAG_VGPRS = 16  # 16 x i32

# ── ASM CO-inspired result windows (PIN_LAYOUT=asm) ───────────────────
ASM_ACC_LOCAL0 = 4
ASM_ACC_PER_BANK = 16  # 16 × v8f32 = 128 VGPRs per bank
ASM_FRAG_BASE = 128
ASM_SCALE_BASE = 196


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


def _pin_abs(raw_val, regnum: int):
    """``sched_vgpr(regnum)`` plus matching ``sched_bank(regnum//256)``."""
    v = pin_vgpr(raw_val, regnum)
    if const_expr(ENABLE_SAME_MSB):
        v = pin_bank(v, regnum // 256)
    return v


def pin_acc_vec(raw_val, frag_idx: int):
    """Pin one v8f32 accumulator fragment (WMMA result / srcC)."""
    if const_expr(PIN_LAYOUT == "asm"):
        bank = frag_idx // ASM_ACC_PER_BANK
        local = ASM_ACC_LOCAL0 + (frag_idx % ASM_ACC_PER_BANK) * 8
        v = _pin_abs(raw_val, bank * 256 + local)
    else:
        v = pin_vgpr(raw_val, ACC_VGPR_BASE + frag_idx * 8)
    if const_expr(BANK_HINT > 0):
        v = pin_bank(v, 2)
    return v


def pin_a_frag(raw_val, slot: int, phase: int = 0):
    """Pin one 16-i32 A fragment — joined load result (not the LDS address)."""
    del phase
    if const_expr(PIN_LAYOUT == "asm"):
        v = _pin_abs(raw_val, ASM_FRAG_BASE + slot * FRAG_VGPRS)
    else:
        v = pin_vgpr(raw_val, A_VGPR_BASE + slot * FRAG_VGPRS)
    if const_expr(BANK_HINT == 1):
        v = pin_bank(v, 0)
    elif const_expr(BANK_HINT == 2):
        v = pin_bank(v, 2)
    return v


def pin_b_frag(raw_val, slot: int, phase: int = 0):
    """Pin one 16-i32 B fragment (joined ds_load result)."""
    del phase
    if const_expr(PIN_LAYOUT == "asm"):
        v = _pin_abs(raw_val, 256 + ASM_FRAG_BASE + slot * FRAG_VGPRS)
    else:
        v = pin_vgpr(raw_val, B_VGPR_BASE + slot * FRAG_VGPRS)
    if const_expr(BANK_HINT == 1):
        v = pin_bank(v, 0 if const_expr(PIN_LAYOUT != "asm") else 1)
    elif const_expr(BANK_HINT == 2):
        v = pin_bank(v, 2)
    return v


def _frag_window_base(kind: str) -> int:
    if const_expr(PIN_LAYOUT == "asm"):
        return ASM_FRAG_BASE if kind == "a" else (256 + ASM_FRAG_BASE)
    return A_VGPR_BASE if kind == "a" else B_VGPR_BASE


def pin_frag_b128(raw_val, kind: str, slot: int, part: int):
    """Pin one ``ds_load_b128`` destination (4×i32)."""
    if const_expr(not ENABLE_PIN_B128):
        return raw_val
    reg = _frag_window_base(kind) + slot * FRAG_VGPRS + part * 4
    return _pin_abs(raw_val, reg)


def pin_scale_word(raw_val, slot: int):
    """Pin one i32 scale *load result* (ASM v196..v203 data window)."""
    if const_expr(not ENABLE_SCHED_VGPR):
        return raw_val
    return pin_vgpr(fx.Int32(raw_val), ASM_SCALE_BASE + slot)
