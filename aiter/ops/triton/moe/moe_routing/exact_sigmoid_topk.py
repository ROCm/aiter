# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import torch

from aiter.ops.triton._triton_kernels.moe.moe_routing.exact_sigmoid_topk import (
    _exact_sigmoid_biased_topk,
)


def exact_sigmoid_biased_topk(
    gating_output: torch.Tensor,
    correction_bias: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    need_renorm: bool,
    routed_scaling_factor: float,
) -> None:
    """Run the gfx1250 exact-contract sigmoid + bias top-k specialization."""
    _exact_sigmoid_biased_topk[(gating_output.shape[0],)](
        gating_output,
        correction_bias,
        topk_weights,
        topk_ids,
        gating_output.stride(0),
        topk_weights.stride(0),
        topk_ids.stride(0),
        gating_output.shape[0],
        routed_scaling_factor,
        N_EXPERTS=896,
        TOPK=16,
        BLOCK_N=1024,
        NEED_RENORM=need_renorm,
        num_warps=4,
    )
