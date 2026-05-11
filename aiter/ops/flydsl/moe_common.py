# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Common types shared across MoE FlyDSL kernel modules."""

import os
from enum import Enum
from typing import Optional


class GateMode(str, Enum):
    """Gate/Up computation strategy for stage1 GEMM.

    SEPARATED:      Two separate B-tile streams (gate + up), default mode.
    MOCK_GATE_ONLY: Single B-tile stream over full [0, 2*inter_dim), simulates
                    gate-only by doubling grid X on top of SEPARATED layout.
                    Requires split-K (k_batch>1).  NOT true gate-only.
    GATE_ONLY:      Reserved for future true gate-only implementation.
    INTERLEAVE:     Weight rows interleave gate/up (gate[0], up[0], gate[1], ...).
                    pack_N=2 routes even/odd N subtiles.  NOT tied to split-K.
    """

    SEPARATED = "separated"
    MOCK_GATE_ONLY = "mock_gate_only"
    GATE_ONLY = "gate_only"
    INTERLEAVE = "interleave"


# ---------------------------------------------------------------------------
# GUI (gate_up_interleave) dispatch policy
# ---------------------------------------------------------------------------
# AITER_MOE_GUI controls when the FlyDSL stage1 path picks the
# gate_up_interleave layout instead of separated.  The mode names mirror the
# weight-row layout they request:
#
#   auto  (default) - historical behavior: only a8w4 (a=fp8 + w=fp4) uses GUI
#   gugu            - "gate-up gate-up" interleaved rows; force GUI for any
#                     fp4-weight path (a4w4 and a8w4 both fold gate+up)
#   gguu            - "gate-gate up-up" separated rows; disable GUI entirely
#                     (downgrades caller-explicit INTERLEAVE to SEPARATED too)
#
# The env var is read at kernel-registry import time (`_register_all_configs`
# in moe_kernels.py) and again per-call by `default_gate_mode`. Switching
# value at runtime affects newly compiled kernels and dispatch decisions, but
# the *registered* candidate set is fixed at import. To change the registered
# set you must restart the python process.

_VALID_GUI_MODES = ("auto", "gugu", "gguu")
_ENV_VAR = "AITER_MOE_GUI"


def get_gui_mode() -> str:
    """Return current AITER_MOE_GUI policy ('auto', 'gugu', or 'gguu')."""
    val = os.environ.get(_ENV_VAR, "auto").lower()
    if val not in _VALID_GUI_MODES:
        raise ValueError(
            f"{_ENV_VAR} must be one of {_VALID_GUI_MODES}, got {val!r}"
        )
    return val


def is_gui_for(
    a_dtype: str, b_dtype: str, mode: Optional[str] = None
) -> bool:
    """Decide whether stage1 should fold gate+up (GUI / interleave layout)
    for the given dtype pair under the current AITER_MOE_GUI policy.

    GUI requires `b_dtype == "fp4"` regardless of mode (the kernel only
    supports fp4 weights).  Within that:
      - gguu: never GUI
      - gugu: any fp4-weight path -> GUI (a4w4 included)
      - auto: only a=fp8 -> GUI (legacy behavior)
    """
    mode = mode or get_gui_mode()
    if mode == "gguu":
        return False
    if b_dtype != "fp4":
        return False
    if mode == "gugu":
        return True
    return a_dtype == "fp8"


def default_gate_mode(a_dtype: str, b_dtype: str) -> str:
    """Default `gate_mode` string for the given dtype pair under the active
    AITER_MOE_GUI policy.  Returned as the string literal (not GateMode) to
    match `flydsl_moe_stage1`'s wire format."""
    return GateMode.INTERLEAVE.value if is_gui_for(a_dtype, b_dtype) else GateMode.SEPARATED.value
