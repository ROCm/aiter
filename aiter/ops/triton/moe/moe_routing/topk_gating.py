# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Caller-owned MoE gating launchers without bitmatrix routing metadata."""

import torch
import triton

from aiter.ops.triton._triton_kernels.moe.moe_routing.topk_gating import (
    _exact_sigmoid_biased_topk,
    _sqrtsoftplus_384_topk,
)
from aiter.ops.triton.utils._triton.arch_info import get_arch

SQRTSOFTPLUS_384_EXPERTS = 384
# The lane layout of the HIP kernel that topk_gating runs for 384 experts:
# (waves for each row, lanes for each wave, experts for each lane).
SQRTSOFTPLUS_384_LAYOUTS = {
    "gfx950": (1, 64, 6),  # topk_gating_kernel_opt
    "gfx1250": (4, 32, 3),  # topk_gating_kernel_opt_multiwave<.., 384, 4, ..>
    "gfx1250-strict": (4, 32, 3),
}


def exact_sigmoid_biased_topk(
    gating_output: torch.Tensor,
    correction_bias: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    need_renorm: bool,
    routed_scaling_factor: float,
    num_valid_rows: torch.Tensor | None = None,
) -> None:
    """Write the gfx1250 HIP-compatible result into caller-owned buffers.
    Rows at or past `num_valid_rows` get ids -1 and weights 0."""
    num_experts = gating_output.shape[1]
    topk = topk_ids.shape[1]
    _exact_sigmoid_biased_topk[(gating_output.shape[0],)](
        gating_output,
        correction_bias,
        topk_weights,
        topk_ids,
        num_valid_rows,
        gating_output.stride(0),
        topk_weights.stride(0),
        topk_ids.stride(0),
        gating_output.shape[0],
        routed_scaling_factor,
        N_EXPERTS=num_experts,
        TOPK=topk,
        TOPK_PAD=triton.next_power_of_2(topk),
        BLOCK_N=triton.next_power_of_2(num_experts),
        NEED_RENORM=need_renorm,
        HAS_NUM_VALID_ROWS=num_valid_rows is not None,
        num_warps=4,
    )


def sqrtsoftplus_384_topk(
    gating_output: torch.Tensor,
    correction_bias: torch.Tensor | None,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    routed_scaling_factor: float,
    num_valid_rows: torch.Tensor | None = None,
) -> None:
    """Write the renormalized sqrtsoftplus top-k of 384 experts into
    caller-owned buffers, bit for bit as the HIP topk_gating kernel.

    The bias only selects. The weights are the scores of the selected experts,
    renormalized and multiplied by routed_scaling_factor. Rows at or past
    num_valid_rows get ids -1 and weights 0.

    Limits: an arch in SQRTSOFTPLUS_384_LAYOUTS, BF16 gating_output of
    [rows, 384] with unit column stride, an FP32 bias of [384] or None, FP32
    topk_weights and INT32 topk_ids of [rows, topk] with topk from 1 to 8.
    On gfx1250 the HIP kernel uses this layout only up to 1024 rows.
    """
    if num_valid_rows is not None and (
        num_valid_rows.dtype != torch.int32 or num_valid_rows.numel() < 1
    ):
        raise ValueError(
            "num_valid_rows must be an int32 tensor of at least one element"
        )
    waves, lanes, ept = SQRTSOFTPLUS_384_LAYOUTS[get_arch()]
    topk = topk_ids.shape[1]
    _sqrtsoftplus_384_topk[(gating_output.shape[0],)](
        gating_output,
        correction_bias,
        topk_weights,
        topk_ids,
        num_valid_rows,
        gating_output.stride(0),
        topk_weights.stride(0),
        topk_ids.stride(0),
        routed_scaling_factor,
        TOPK=topk,
        TOPK_PAD=triton.next_power_of_2(topk),
        HAS_BIAS=correction_bias is not None,
        HAS_NUM_VALID_ROWS=num_valid_rows is not None,
        WAVES=waves,
        LANES=lanes,
        EPT=ept,
        num_warps=1,
    )
