# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Caller-owned MoE gating launchers without bitmatrix routing metadata."""

from typing import NamedTuple

import torch
import triton

from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.ops.triton._triton_kernels.moe.moe_routing.topk_gating import (
    _exact_sigmoid_biased_topk,
    _sqrtsoftplus_384_topk,
)

SQRTSOFTPLUS_384_EXPERTS = 384


class SqrtSoftplus384Arch(NamedTuple):
    """How one arch runs the 384-expert sqrtsoftplus top-k."""

    # The lane layout of the HIP kernel that topk_gating runs for 384 experts.
    waves: int  # waves for each row
    lanes: int  # lanes for each wave
    ept: int  # experts for each lane
    # The rows that topk_gating sends to the Triton kernel.
    max_rows: int  # most rows; the HIP kernel takes larger calls
    graph_only: bool  # only under CUDA graph capture; eager calls run HIP


# Keyed by get_gfx_runtime(). The HIP kernel and its layout:
# gfx950: topk_gating_kernel_opt, for all row counts; it is the faster one
# above 2048 rows.
# gfx1250: topk_gating_kernel_opt_multiwave<.., 384, 4, ..>, up to 1024 rows.
# Eager calls stay on HIP: there the Triton launch costs more host time than
# the kernel saves.
SQRTSOFTPLUS_384_ARCHS = {
    "gfx950": SqrtSoftplus384Arch(1, 64, 6, 2048, True),
    "gfx1250": SqrtSoftplus384Arch(4, 32, 3, 1024, True),
    "gfx1250-strict": SqrtSoftplus384Arch(4, 32, 3, 1024, True),
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

    Limits: an arch in SQRTSOFTPLUS_384_ARCHS, BF16 gating_output of
    [rows, 384] with unit column stride, an FP32 bias of [384] or None, FP32
    topk_weights and INT32 topk_ids of [rows, topk] with topk from 1 to 8.
    The result follows the HIP kernel bit for bit up to the max_rows of the
    arch; on gfx1250 the HIP kernel uses another layout above that.
    """
    if num_valid_rows is not None and (
        num_valid_rows.dtype != torch.int32 or num_valid_rows.numel() < 1
    ):
        raise ValueError(
            "num_valid_rows must be an int32 tensor of at least one element"
        )
    arch = SQRTSOFTPLUS_384_ARCHS.get(get_gfx_runtime())
    if arch is None:
        raise ValueError(
            f"sqrtsoftplus_384_topk supports {sorted(SQRTSOFTPLUS_384_ARCHS)}, "
            f"not {get_gfx_runtime()}"
        )
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
        WAVES=arch.waves,
        LANES=arch.lanes,
        EPT=arch.ept,
        num_warps=1,
    )
