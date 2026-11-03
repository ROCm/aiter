# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Deprecated alias of :mod:`aiter.fused_moe_bf16` (#7150).

The module name baked a backend into a module that carries several backends
(asm, ck, torch). Import :mod:`aiter.fused_moe_bf16` instead. Importing this
alias is side-effect-free; the first attribute access warns once.
"""

from __future__ import annotations

import warnings

from . import fused_moe_bf16 as _fused_moe_bf16

_warned = False


def __getattr__(name: str):
    global _warned
    if not _warned:
        _warned = True
        warnings.warn(
            "aiter.fused_moe_bf16_asm is deprecated: use aiter.fused_moe_bf16. "
            "The alias is removed after the deprecation window tracked in #7150 "
            "(https://github.com/ROCm/aiter/issues/7150).",
            DeprecationWarning,
            stacklevel=2,
        )
    return getattr(_fused_moe_bf16, name)
