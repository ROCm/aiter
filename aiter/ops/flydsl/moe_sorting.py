# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL MoE sorting kernel — drop-in replacement for CK/Opus moe_sorting_fwd.

Provides `flydsl_moe_sorting_fwd()` with the same signature as
`aiter.moe_sorting_fwd()` so it can be used as a direct dispatch target
in `_moe_sorting_impl()`.

Workspace is pre-allocated here (not inside the kernel) so that CUDA graph
capture sees deterministic allocations.
"""

import torch

_workspace_cache = {}


def flydsl_moe_sorting_herd(
    ids,
    values,
    sorted_ids,
    sorted_weights,
    sorted_experts,
    valid_ids,
    moe_buf,
    experts,
    block_size,
    routed_scaling_factor,
):
    from .kernels.moe_sorting_herd import build_moe_sorting_herd_module
    from .kernels.tensor_shim import _run_compiled

    tokens, width = ids.shape
    if not (1 <= tokens <= 128 and 1 < width <= experts <= 1024):
        raise ValueError("HERD sorting requires M<=128 and 1<=K<E<=1024")
    if not (
        ids.dtype == torch.int32
        and values.dtype == torch.float32
        and ids.shape == values.shape
        and ids.is_contiguous()
        and values.is_contiguous()
        and ids.device == values.device
    ):
        raise ValueError("HERD requires contiguous int32 IDs/fp32 weights on one GPU")
    kept_ids = torch.empty((tokens, width - 1), dtype=torch.int32, device=ids.device)
    kept_weights = torch.empty_like(kept_ids, dtype=torch.float32)
    buf = (
        moe_buf.view(torch.int32)
        if moe_buf.numel()
        else torch.empty((0, 0), dtype=torch.int32, device=ids.device)
    )
    threads = max(128, 1 << (experts - 1).bit_length())
    blocks = 1 + min(256, (buf.numel() + threads * 4 - 1) // (threads * 4))
    launch = build_moe_sorting_herd_module(experts, width - 1, tokens, block_size)
    _run_compiled(
        launch,
        ids,
        values,
        kept_ids,
        kept_weights,
        sorted_ids,
        sorted_weights,
        sorted_experts,
        valid_ids,
        buf,
        tokens,
        buf.numel(),
        float(routed_scaling_factor),
        blocks,
        torch.cuda.current_stream(ids.device),
        specialization_key=(buf.shape,),
    )
    return kept_ids, kept_weights


def flydsl_moe_sorting_fwd(
    topk_ids,
    topk_weights,
    sorted_ids,
    sorted_weights,
    sorted_expert_ids,
    num_valid_ids,
    moe_buf,
    num_experts,
    unit_size,
    expert_mask=None,
    num_local_tokens=None,
):
    from .kernels.moe_sorting_kernel import (
        moe_sorting_flydsl,
        moe_sorting_get_workspace_size,
    )

    M = topk_ids.shape[0]
    topk = topk_ids.shape[1]
    device = topk_ids.device

    # Pre-allocate workspace (cached per device for CUDA graph compatibility).
    # A larger workspace can satisfy smaller requests, so we keep the largest seen.
    ws_size = moe_sorting_get_workspace_size(M, num_experts, topk, unit_size)
    workspace = None
    if ws_size > 0:
        workspace = _workspace_cache.get(device)
        if workspace is None or workspace.numel() < ws_size:
            workspace = torch.empty(ws_size, dtype=torch.int32, device=device)
            _workspace_cache[device] = workspace

    moe_sorting_flydsl(
        topk_ids,
        topk_weights,
        sorted_ids,
        sorted_weights,
        sorted_expert_ids,
        num_valid_ids,
        moe_buf,
        num_experts,
        unit_size,
        expert_mask,
        num_local_tokens,
        workspace,
    )
