# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Public entry points of aiter.moev2, with the legacy signatures.

`fused_moe` here always runs the v2 code: it ignores AITER_MOEV2 and raises when no
registered path claims the call. The legacy `aiter.fused_moe.fused_moe` is the entry
point that honours the switch.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from aiter import ActivationType, QuantType

from . import paths

if TYPE_CHECKING:
    from aiter.ops.flydsl.kernels.mega_moe_gfx1250.types import Stage2ScatterContext


def fused_moe(
    hidden_states,
    w1,
    w2,
    topk_weight,
    topk_ids,
    expert_mask: torch.Tensor | None = None,
    activation=ActivationType.Silu,
    quant_type=QuantType.No,
    doweight_stage1=False,
    w1_scale: torch.Tensor | None = None,
    w2_scale: torch.Tensor | None = None,
    a1_scale: torch.Tensor | None = None,
    a2_scale: torch.Tensor | None = None,
    block_size_M=None,
    num_local_tokens: torch.Tensor | None = None,
    moe_sorting_dispatch_policy=0,
    dtype=None,
    hidden_pad=0,
    intermediate_pad=0,
    bias1=None,
    bias2=None,
    splitk=0,
    swiglu_limit=None,
    beta=None,
    linear_beta=None,
    gate_mode: str | None = "separated",
    shared_w1: torch.Tensor | None = None,
    shared_w2: torch.Tensor | None = None,
    shared_w1_scale: torch.Tensor | None = None,
    shared_w2_scale: torch.Tensor | None = None,
    shared_expert_id: int = -1,
    stage2_scatter: Stage2ScatterContext | None = None,
    output: torch.Tensor | None = None,
    quant_type_a: QuantType | None = None,
    quant_dtype_a: torch.dtype | None = None,
    quant_dtype_a2: torch.dtype | None = None,
    clamp_shared: bool = True,
    herd_topk: int = 0,
    herd_routed_scaling_factor: float = 1.0,
    deterministic: bool = False,
):
    """Same contract as `aiter.fused_moe.fused_moe`, served by aiter.moev2."""
    call = dict(locals())
    for path in paths.registered().values():
        if path.claims(call):
            return path.run(call)
    raise NotImplementedError(
        "no aiter.moev2 path handles this fused_moe call yet; "
        "use aiter.fused_moe.fused_moe (the legacy code)"
    )
