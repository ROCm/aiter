# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Lightweight FHMoE contract predicates shared by runtime and AOT.

This module must remain free of imports from the top-level ``aiter`` namespace
so it is safe when ``AITER_AOT_IMPORT=1`` exposes only the lightweight JIT core.
"""

# Reduce-mode stage 2 writes [M, topk, model_dim] BF16 route output through a
# buffer resource with a 32-bit byte extent/offset. Keep the largest supported
# M at or below UINT32_MAX bytes until that kernel uses 64-bit/rebased addressing.
_HY4_ROUTE_OUTPUT_BYTES_PER_TOKEN = 9 * 6144 * 2
HY4_FHMOE_MAX_TOKENS = ((1 << 32) - 1) // _HY4_ROUTE_OUTPUT_BYTES_PER_TOKEN


def _is_hy4_mxfp8_fhmoe_contract(
    *,
    model_dim: int,
    inter_dim: int,
    experts: int,
    topk: int,
    routed_mxfp8: bool,
    hidden_pad: int,
    intermediate_pad: int,
    gate_interleaved: bool,
    doweight_stage1: bool,
    shared_expert_id: int,
) -> bool:
    """Return whether stage 1 has HY4-compatible geometry and layout.

    This is a compatibility predicate, not model identity. Runtime callers
    must also inspect ``clamp_shared`` before enabling unclamped-shared
    semantics; compatible clamped calls use the same tuned table.
    """
    return (
        (model_dim, inter_dim, experts, topk) == (6144, 256, 257, 9)
        and routed_mxfp8
        and hidden_pad == 0
        and intermediate_pad == 0
        and gate_interleaved
        and not doweight_stage1
        and shared_expert_id == 256
    )
