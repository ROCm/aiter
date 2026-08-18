# SPDX-License-Identifier: MIT
"""K3 EP16 MegaMoE Tile operator with lazy public imports."""

import importlib


_LAZY = {
    "ArenaRegion": "runtime",
    "CopyTransport": "transport",
    "EpochPhase": "runtime",
    "HierCcoArenaLayout": "runtime",
    "HierEpoch": "runtime",
    "HierEpochPointers": "runtime",
    "HierMegaMoETileConfig": "config",
    "HierarchicalMegaMoEV2": "mega_moe_tile_a4w4",
    "K3DispatchWireLayout": "wire",
    "K3PartialWireLayout": "wire",
    "KernelResources": "profiling",
    "KimiK3A4W4Shape": "config",
    "LayeredHierPipeline": "runtime",
    "LogicalTopology": "topology",
    "MegaMoETileA4W4": "mega_moe_tile_a4w4",
    "MoriShmemTransport": "transport",
    "PersistentH1Workspace": "workspace",
    "PreparedA4W4Weights": "compute_v2",
    "RoutePlan": "topology",
    "SUPPORTED_ACTIVATIONS": "activation",
    "Stage1ArenaLayout": "stage1_abi",
    "Stage1ArenaRegion": "stage1_abi",
    "Stage1DispatchWire": "stage1_abi",
    "Stage2NodePartialWire": "stage2_abi",
    "TwoKernelArenaLayout": "stage1_abi",
    "Stage2ArenaLayout": "stage2_abi",
    "TransportKind": "config",
    "a4w4_dense_reference": "compute_v2",
    "apply_gate_up": "activation",
    "build_route_plan": "topology",
    "compile_megamoe_tile_ep16_stage1": "stage1",
    "compile_megamoe_tile_ep16_stage2_a4w4": "stage2",
    "dense_moe_reference": "reference",
    "hidden_fraction": "profiling",
    "hierarchical_moe_reference": "reference",
    "normalize_activation": "activation",
    "pack_dispatch_records": "wire",
    "prepare_local_a4w4_weights": "compute_v2",
    "run_local_ep_a4w4": "compute_v2",
    "run_local_ep_a4w4_silu": "compute_v2",
    "silu_gate": "reference",
    "situ_v2": "reference",
    "swiglu_gate": "reference",
    "timeline_overlap_ratio": "profiling",
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
