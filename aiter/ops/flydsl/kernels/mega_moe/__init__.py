# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""MegaMoEV2 fused MoE operator with lazy public imports."""

import importlib

_LAZY = {
    "MegaMoEConfig": "mega_moe_config",
    "MegaMoEV2": "mega_moe_v2",
    "Stage1Config": "mega_moe_config",
    "Stage2Config": "mega_moe_config",
    "compile_gemm1": "gemm1",
    "gemm1_kernel": "gemm1",
    "select_mega_moe_config": "mega_moe_config",
}

__all__ = list(_LAZY)


def __getattr__(name):
    sub = _LAZY.get(name)
    if sub is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(importlib.import_module(f"{__name__}.{sub}"), name)


def __dir__():
    return sorted(list(globals()) + __all__)
