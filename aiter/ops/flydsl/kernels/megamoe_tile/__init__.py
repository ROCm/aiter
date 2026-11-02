# SPDX-License-Identifier: MIT
"""MegaMoE across two nodes (EP16, A4W4) with lazy public imports."""

import importlib

_LAZY = {
    "MegaMoEInternode": "mega_moe_internode",
}

__all__ = list(_LAZY)


def __getattr__(name):
    submodule = _LAZY.get(name)
    if submodule is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(importlib.import_module(f"{__name__}.{submodule}"), name)


def __dir__():
    return sorted(list(globals()) + __all__)
