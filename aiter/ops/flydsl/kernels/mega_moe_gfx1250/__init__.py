# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""gfx1250 MegaMoE with a single lazy public entry point."""

import importlib

__all__ = ["MegaMoEGfx1250"]


def __getattr__(name):
    if name != "MegaMoEGfx1250":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(importlib.import_module(f"{__name__}.mega_moe"), name)


def __dir__():
    return sorted(list(globals()) + __all__)
