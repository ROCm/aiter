# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""gfx1250 MegaMoE with a single lazy public entry point."""

import importlib
import os


def _pin_mori_jit_arch():
    """Make mori's JIT target the GPU's real ISA (e.g. gfx1250-strict on MI450 A0).

    mori.jit.config.detect_gpu_arch() runs `rocm_agent_enumerator` without
    `-name`, which reports the base ISA (gfx1250) on A0, so the EP dispatch
    kernel was built for gfx1250 and hipModuleLoad rejected it there ("device
    kernel image is invalid"). A single arch in MORI_GPU_ARCHS overrides that
    detection; a user-set value is left alone.
    """
    if os.environ.get("MORI_GPU_ARCHS"):
        return
    try:
        from aiter.jit.utils.chip_info import get_gfx_runtime  # rocminfo, no HIP init

        arch = get_gfx_runtime()
    except Exception:  # noqa: BLE001 -- no GPU: leave mori's own detection
        return
    # Only the A0 case; every other GPU keeps mori's own detection unchanged.
    if arch == "gfx1250-strict":
        os.environ["MORI_GPU_ARCHS"] = arch


_pin_mori_jit_arch()

__all__ = [
    "CombineKind",
    "DispatchKind",
    "ExpertComputeConfig",
    "MegaMoEGeometry",
    "MegaMoEGfx1250",
    "MegaMoERecipe",
    "read_dispatch_wire_env",
]


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(importlib.import_module(f"{__name__}.mega_moe"), name)


def __dir__():
    return sorted(list(globals()) + __all__)
