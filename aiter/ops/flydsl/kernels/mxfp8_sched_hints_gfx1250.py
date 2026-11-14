# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Optional ``sched_vgpr`` / ``sched_bank`` hints for gfx1250 MXFP8 compute GEMM.

Mirrors the FMHA HD192 / MXFP4 pin style, but uses flydsl 0.4.2
``rocdl.experimental.sched_*`` (``fx.set_register`` is 0.5+ only).

``sched_vgpr`` only pins **register-held results** (ds_load destinations,
WMMA accumulators, scale words) — not LDS address SSA / index values.

Env (all default off so stock path stays bit-stable unless opted in)::

    AITER_FLYDSL_MXFP8_SCHED_VGPR=1   pin acc / A / B / scale result windows
    AITER_FLYDSL_MXFP8_ASM_PIPE=1     widen WAR + ASM-style DS bursts / hints
    AITER_FLYDSL_MXFP8_PIN_SOFT=1     advisory pins instead of strict
    AITER_FLYDSL_MXFP8_SA_PANEL_CLASS0=1|2|4|8  short-K SA panel stages (default 4)
        1 enables persistent ``c_relocate`` but measured regress on huge-N
        short-K; 2/4/8 widen the A-scale panel (c_relocate off).
    AITER_FLYDSL_MXFP8_SA_PANEL_CLASS1=1|2|4|8  mid-K SA panel stages (default 8)
        1 enables persistent ``c_relocate`` on nb2 mid-K (e.g. K=6144).
    AITER_FLYDSL_MXFP8_PIN_LAYOUT=mxfp4|asm
        mxfp4 — Acc@512, A@0, B@256 (legacy MXFP4 partition; measured best)
        asm   — Acc striped low+MSB like ASM; A@128 bank0; B@128 bank1
    AITER_FLYDSL_MXFP8_PIN_B128=0|1
        also pin each ds_load_b128 *dest* (4×i32) inside the frag window
        (default 0 — measured regress vs join-level frag pin).
    AITER_FLYDSL_MXFP8_PIN_SCALE=0|1
        pin scale ds_load_b32 *results* into v196+ (default 1). Kernel gates
        to mid-K class1 + persistent only (helps 25600×6144; skips short-K).
    AITER_FLYDSL_MXFP8_SHORT_BEAT=0|1
        on short-K (class 0) still emit per-producer ``sched_mfma/dsrd`` beats
        to cut ``va_vdst(0)`` between WMMA and ds_load (default 0).
    AITER_FLYDSL_MXFP8_LOAD_FIRST=0|1|2
        in ``_quad``, issue producer ds_loads before the WMMA at the same slot.
        0 off; 1 force on; 2 auto (default) — on for mid/long-K + persistent
        (helps huge-N K=6144; avoids short-K / non-persistent regress).
    AITER_FLYDSL_MXFP8_BANK_HINT=0|1|2
        0 off; 1 A/B→bank0 Acc→bank2; 2 A/B+Acc all bank2.
    AITER_FLYDSL_MXFP8_SAME_MSB=0|1
        Force ASM-style absolute pins + ``sched_bank(reg//256)`` on Acc and on
        each ``ds_load_b128`` dest so WMMA/load share a bank (default 0).
        Implies PIN_LAYOUT=asm and PIN_B128=1. Measured regress.
    AITER_FLYDSL_MXFP8_NARROW_MSB=0|1
        Narrower than SAME_MSB: ASM-style join pins + ``sched_bank`` on Acc/A/B
        results only — does **not** force PIN_B128 (default 0).
    AITER_FLYDSL_MXFP8_MIDK_Q0=0|2|4
        Mid-K (class 1) Q0 producer start slot (default 0 = dense). 4 = legacy
        WAR gap; 2 = middle ground.
    AITER_FLYDSL_MXFP8_MIDK_SOFT_ASM=0|1
        Soft ASM-layout join + PIN_B128 (default 0). Measured slight win on
        K=6144 vs hard SAME_MSB regress; does not force sched_bank.
    AITER_FLYDSL_MXFP8_MIDK_WD=0|1
        Soft-pin A/B (join + b128) into Acc banks 2/3 so WMMA→ds_load can
        stay WD-adjacent without low-field SAME_MSB (default 0).
    AITER_FLYDSL_MXFP8_MIDK_WAR=0|1
        Enable a4w4 WAR soft-pipe for mx32 mid-K (class 1); default 0 (legacy
        ``_quad`` path). Non-mx short-K WAR is unchanged.
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
# Default 4: short-K A-scale panel (c_relocate requires stages==1 and regresses
# on the gemm-request huge-N short-K shape when forced).
_SA_PANEL_CLASS0 = int(os.environ.get("AITER_FLYDSL_MXFP8_SA_PANEL_CLASS0", "4"))
# Default 8: mid-K fat A-scale panel (legacy). 1 unlocks c_relocate on nb2.
_SA_PANEL_CLASS1 = int(os.environ.get("AITER_FLYDSL_MXFP8_SA_PANEL_CLASS1", "8"))
_PIN_LAYOUT = os.environ.get("AITER_FLYDSL_MXFP8_PIN_LAYOUT", "mxfp4").strip().lower()
_PIN_B128 = int(os.environ.get("AITER_FLYDSL_MXFP8_PIN_B128", "0"))
_PIN_SCALE = int(os.environ.get("AITER_FLYDSL_MXFP8_PIN_SCALE", "1"))
_SHORT_BEAT = int(os.environ.get("AITER_FLYDSL_MXFP8_SHORT_BEAT", "0"))
# Default 2=auto: mid/long-K persistent measured faster on gemm-request
# 25600x6144; force-on regresses short-K / small non-persistent shapes.
_LOAD_FIRST = int(os.environ.get("AITER_FLYDSL_MXFP8_LOAD_FIRST", "2"))
_BANK_HINT = int(os.environ.get("AITER_FLYDSL_MXFP8_BANK_HINT", "0"))
_SAME_MSB = int(os.environ.get("AITER_FLYDSL_MXFP8_SAME_MSB", "0"))
_NARROW_MSB = int(os.environ.get("AITER_FLYDSL_MXFP8_NARROW_MSB", "0"))
_MIDK_Q0 = int(os.environ.get("AITER_FLYDSL_MXFP8_MIDK_Q0", "0"))
_MIDK_SOFT_ASM = int(os.environ.get("AITER_FLYDSL_MXFP8_MIDK_SOFT_ASM", "0"))
_MIDK_WD = int(os.environ.get("AITER_FLYDSL_MXFP8_MIDK_WD", "0"))
_MIDK_WAR = int(os.environ.get("AITER_FLYDSL_MXFP8_MIDK_WAR", "0"))

if _SCHED_VGPR not in (0, 1):
    raise ValueError("AITER_FLYDSL_MXFP8_SCHED_VGPR must be 0 or 1")
if _ASM_PIPE not in (0, 1):
    raise ValueError("AITER_FLYDSL_MXFP8_ASM_PIPE must be 0 or 1")
if _PIN_SOFT not in (0, 1):
    raise ValueError("AITER_FLYDSL_MXFP8_PIN_SOFT must be 0 or 1")
if _SA_PANEL_CLASS0 not in (1, 2, 4, 8):
    raise ValueError("AITER_FLYDSL_MXFP8_SA_PANEL_CLASS0 must be 1, 2, 4, or 8")
if _SA_PANEL_CLASS1 not in (1, 2, 4, 8):
    raise ValueError("AITER_FLYDSL_MXFP8_SA_PANEL_CLASS1 must be 1, 2, 4, or 8")
if _PIN_LAYOUT not in ("mxfp4", "asm"):
    raise ValueError("AITER_FLYDSL_MXFP8_PIN_LAYOUT must be mxfp4 or asm")
if _PIN_B128 not in (0, 1):
    raise ValueError("AITER_FLYDSL_MXFP8_PIN_B128 must be 0 or 1")
if _PIN_SCALE not in (0, 1):
    raise ValueError("AITER_FLYDSL_MXFP8_PIN_SCALE must be 0 or 1")
if _SHORT_BEAT not in (0, 1):
    raise ValueError("AITER_FLYDSL_MXFP8_SHORT_BEAT must be 0 or 1")
if _LOAD_FIRST not in (0, 1, 2):
    raise ValueError("AITER_FLYDSL_MXFP8_LOAD_FIRST must be 0, 1, or 2")
if _BANK_HINT not in (0, 1, 2):
    raise ValueError("AITER_FLYDSL_MXFP8_BANK_HINT must be 0, 1, or 2")
if _SAME_MSB not in (0, 1):
    raise ValueError("AITER_FLYDSL_MXFP8_SAME_MSB must be 0 or 1")
if _NARROW_MSB not in (0, 1):
    raise ValueError("AITER_FLYDSL_MXFP8_NARROW_MSB must be 0 or 1")
if _MIDK_Q0 not in (0, 2, 4):
    raise ValueError("AITER_FLYDSL_MXFP8_MIDK_Q0 must be 0, 2, or 4")
if _MIDK_SOFT_ASM not in (0, 1):
    raise ValueError("AITER_FLYDSL_MXFP8_MIDK_SOFT_ASM must be 0 or 1")
if _MIDK_WD not in (0, 1):
    raise ValueError("AITER_FLYDSL_MXFP8_MIDK_WD must be 0 or 1")
if _MIDK_WAR not in (0, 1):
    raise ValueError("AITER_FLYDSL_MXFP8_MIDK_WAR must be 0 or 1")

ENABLE_SCHED_VGPR = bool(_SCHED_VGPR and sched_vgpr is not None)
ENABLE_ASM_PIPE = bool(_ASM_PIPE)
SA_PANEL_CLASS0 = _SA_PANEL_CLASS0
SA_PANEL_CLASS1 = _SA_PANEL_CLASS1
# SAME_MSB: Acc striped by bank at low indices; each ds_load_b128 dest gets
# absolute sched_vgpr + matching sched_bank so RA does not encode as low+MSB.
ENABLE_SAME_MSB = bool(_SAME_MSB and ENABLE_SCHED_VGPR)
# NARROW_MSB: asm join pins + bank, without PIN_B128 (avoids SAME_MSB regress).
ENABLE_NARROW_MSB = bool(_NARROW_MSB and ENABLE_SCHED_VGPR and not ENABLE_SAME_MSB)
# Soft ASM+b128: advisory pins only (no sched_bank). Slight 6144 win vs base.
ENABLE_MIDK_SOFT_ASM = bool(
    _MIDK_SOFT_ASM
    and ENABLE_SCHED_VGPR
    and not ENABLE_SAME_MSB
    and not ENABLE_NARROW_MSB
)
# Overlay A/B into Acc banks 2/3 (soft + b128); keeps PIN_LAYOUT=mxfp4 Acc@512.
ENABLE_MIDK_WD = bool(
    _MIDK_WD
    and ENABLE_SCHED_VGPR
    and not ENABLE_SAME_MSB
    and not ENABLE_NARROW_MSB
    and not ENABLE_MIDK_SOFT_ASM
)
MIDK_Q0_START = _MIDK_Q0
ENABLE_MIDK_WAR = bool(_MIDK_WAR)
PIN_SOFT = bool(_PIN_SOFT or ENABLE_MIDK_SOFT_ASM or ENABLE_MIDK_WD)
PIN_LAYOUT = (
    "asm"
    if (ENABLE_SAME_MSB or ENABLE_NARROW_MSB or ENABLE_MIDK_SOFT_ASM)
    else _PIN_LAYOUT
)
ENABLE_PIN_B128 = bool(
    (_PIN_B128 or ENABLE_SAME_MSB or ENABLE_MIDK_SOFT_ASM or ENABLE_MIDK_WD)
    and ENABLE_SCHED_VGPR
)
ENABLE_PIN_SCALE = bool(_PIN_SCALE and ENABLE_SCHED_VGPR)
ENABLE_SHORT_BEAT = bool(_SHORT_BEAT)
ENABLE_LOAD_FIRST = _LOAD_FIRST == 1  # force on
ENABLE_LOAD_FIRST_AUTO = _LOAD_FIRST == 2  # mid/long-K + persistent
BANK_HINT = (
    0
    if (ENABLE_SAME_MSB or ENABLE_NARROW_MSB or ENABLE_MIDK_SOFT_ASM)
    else (_BANK_HINT if ENABLE_SCHED_VGPR else 0)
)
ENABLE_BANK_HINT = bool(BANK_HINT)

# ── mxfp4 partition (legacy / measured best on gemm-request) ───────────
# Acc fragments (v8f32) sit in the upper half so A/B ping-pong can own 0..511.
ACC_VGPR_BASE = 512
# MIDK_WD: park A/B inside Acc banks (2/3) so WMMA and ds_load share MSB.
A_VGPR_BASE = 512 + 128 if ENABLE_MIDK_WD else 0
B_VGPR_BASE = 768 + 128 if ENABLE_MIDK_WD else 256
FRAG_VGPRS = 16  # 16 x i32

# ── ASM CO-inspired result windows (PIN_LAYOUT=asm) ───────────────────
# Acc: v[4:131] × 4 MSB banks  → flat bank*256 + (4 + local*8)
# A/B frags: v[128:191] overlay; B lives in bank1 same low indices.
# Scale *data* (ds_load_b32 dest): v196..v203 — not LDS address VGPRs.
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
    """``sched_vgpr(regnum)`` plus matching ``sched_bank(regnum//256)``.

    Absolute numbers alone can still be rewritten as low-field + ``s_set_vgpr_msb``;
    tying the bank hint to the same absolute slot makes RA prefer a stable MSB
    view so WMMA and a following ``ds_load`` can share it.
    """
    v = pin_vgpr(raw_val, regnum)
    if const_expr(ENABLE_SAME_MSB or ENABLE_NARROW_MSB):
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
    # BANK_HINT=1/2: Acc→bank2 (legacy experiments; ignored when SAME_MSB).
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
    """Pin one ``ds_load_b128`` destination (4×i32) — the real load dest.

    Join-level / ``sched_bank`` alone does not move these; ISA still shows
    ``v[0:3]`` + ``s_set_vgpr_msb``. Absolute ``sched_vgpr`` (+ bank under
    SAME_MSB) is what can keep WMMA and ds_load in one MSB window.
    """
    if const_expr(not ENABLE_PIN_B128):
        return raw_val
    reg = _frag_window_base(kind) + slot * FRAG_VGPRS + part * 4
    return _pin_abs(raw_val, reg)


def pin_scale_word(raw_val, slot: int):
    """Pin one i32 scale *load result* (ASM v196..v203 data window)."""
    if const_expr(not ENABLE_PIN_SCALE):
        return raw_val
    return pin_vgpr(fx.Int32(raw_val), ASM_SCALE_BASE + slot)
