# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Caller-owned MoE gating launchers without bitmatrix routing metadata."""

from typing import NamedTuple

import torch
import triton

from aiter.ops.triton._triton_kernels.moe.moe_routing.topk_gating import (
    _exact_sigmoid_biased_topk,
    _sqrtsoftplus_384_topk,
)

SQRTSOFTPLUS_384_EXPERTS = 384


class SqrtSoftplus384Arch(NamedTuple):
    """How one arch runs the 384-expert sqrtsoftplus top-k."""

    # The most rows that topk_gating sends to the Triton kernel; the HIP kernel
    # takes larger calls.
    max_rows: int


# Keyed by get_gfx_runtime(). The kernel has one lane layout, that of
# topk_gating_kernel_opt: one wave of 64 lanes, six experts in each lane.
# gfx950 runs that HIP kernel for all row counts; under graph replay it is the
# faster one above 2048 rows. An arch whose HIP kernel has another layout needs its own kernel
# path and test before it can have an entry.
# No other arch is in the table. On gfx1250 the HIP single-wave kernel for 384
# experts (topk_gating_kernel_opt_vwave) is at least as fast as the Triton
# kernel; gfx1250-strict and gfx942 have not been measured.
# topk_gating sends a renormalized 384-expert sqrtsoftplus call without shared
# columns to the Triton kernel on an arch in the table, up to max_rows rows,
# in eager mode and in CUDA graphs alike. The choice is static, from the arch
# and the call's shape. Every other call runs HIP.
SQRTSOFTPLUS_384_ARCHS = {
    "gfx950": SqrtSoftplus384Arch(max_rows=2048),
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


def _sqrtsoftplus_384_topk_launch(
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

    Private: topk_gating is the entry point. It checks the limits below and
    sends any other call to HIP. Limits: a device
    arch in SQRTSOFTPLUS_384_ARCHS, BF16 gating_output of [rows, 384] with
    unit column stride and 1 to max_rows rows, an FP32 bias of [384] or None,
    FP32 topk_weights and INT32 topk_ids of [rows, topk] with unit column
    stride and topk from 1 to 8.
    """
    if num_valid_rows is not None and (
        num_valid_rows.dtype != torch.int32 or num_valid_rows.numel() < 1
    ):
        raise ValueError(
            "num_valid_rows must be an int32 tensor of at least one element"
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
        float(routed_scaling_factor),
        TOPK=topk,
        # next_power_of_2(topk), without its per-call overhead
        TOPK_PAD=1 << (topk - 1).bit_length(),
        HAS_BIAS=correction_bias is not None,
        HAS_NUM_VALID_ROWS=num_valid_rows is not None,
        num_warps=1,
    )
