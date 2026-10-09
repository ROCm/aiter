# SPDX-License-Identifier: MIT
"""K3 EP16 MegaMoE Tile operator with lazy public imports."""

import importlib


_LAZY = {
    "MegaMoETileA4W4": "mega_moe_tile_a4w4",
    "Stage1ArenaLayout": "stage1_abi",
    "Stage1ArenaRegion": "stage1_abi",
    "Stage1DispatchWire": "stage1_abi",
    "Stage2NodePartialWire": "stage2_abi",
    "TwoKernelArenaLayout": "stage1_abi",
    "Stage2ArenaLayout": "stage2_abi",
    "compile_megamoe_tile_ep16_stage1": "stage1",
    "validate_public_stage1_contract": "stage1_abi",
}

__all__ = list(_LAZY)


def __getattr__(name):
    submodule = _LAZY.get(name)
    if submodule is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(importlib.import_module(f"{__name__}.{submodule}"), name)


def __dir__():
    return sorted(list(globals()) + __all__)
