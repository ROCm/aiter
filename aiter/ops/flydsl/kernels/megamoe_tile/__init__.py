# SPDX-License-Identifier: MIT
"""K3 EP16 MegaMoE Tile operator with lazy public imports."""

import importlib


_LAZY = {
    "HierarchicalMegaMoEV2": "mega_moe_tile_a4w4",
    "K3DispatchWireLayout": "wire",
    "K3PartialWireLayout": "wire",
    "MegaMoETileA4W4": "mega_moe_tile_a4w4",
    "PreparedA4W4Weights": "compute_v2",
    "SUPPORTED_ACTIVATIONS": "activation",
    "Stage1ArenaLayout": "stage1_abi",
    "Stage1ArenaRegion": "stage1_abi",
    "Stage1DispatchWire": "stage1_abi",
    "Stage2NodePartialWire": "stage2_abi",
    "TwoKernelArenaLayout": "stage1_abi",
    "Stage2ArenaLayout": "stage2_abi",
    "a4w4_dense_reference": "compute_v2",
    "apply_gate_up": "activation",
    "compile_megamoe_tile_ep16_stage1": "stage1",
    "compile_megamoe_tile_ep16_stage2_a4w4": "stage2",
    "normalize_activation": "activation",
    "pack_dispatch_records": "wire",
    "prepare_local_a4w4_weights": "compute_v2",
    "run_local_ep_a4w4": "compute_v2",
    "run_local_ep_a4w4_silu": "compute_v2",
    "unpack_dispatch_records": "wire",
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
