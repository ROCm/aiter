# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

import torch
from torch import Tensor

from ...jit.core import compile_ops
from ..enum import ActivationType

_OPUS_MOE_STAGE1_A8W4_SCALE_GROUP = 32


def _opus_runtime_swiglu_limit(
    swiglu_limit: float | None,
    activation: int,
) -> float:
    """Normalize the clamp bound passed to Opus stage1 kernels.

    ``None`` and ``0.0`` mean no configured clamp. SwiGLU retains its 7.0
    default; Silu and Situv2 encode no clamp as positive infinity.
    """
    if activation == ActivationType.Swiglu.value:
        return float(swiglu_limit) if swiglu_limit else 7.0
    return float(swiglu_limit) if swiglu_limit else float("inf")


def _contiguous(tensor: Tensor) -> Tensor:
    return tensor if tensor.is_contiguous() else tensor.contiguous()


def _gen_opus_moe_stage1_a8w4_fake_tensors(
    hidden_states: Tensor,
    w1: Tensor,
    hidden_scale: Tensor,
    w1_scale: Tensor,
    bias: Tensor | None,
    sorted_token_ids: Tensor,
    sorted_expert_ids: Tensor,
    num_valid_ids: Tensor,
    out: Tensor,
    out_scale: Tensor,
) -> Tensor:
    return out


@compile_ops(
    "module_moe_opus",
    fc_name="opus_moe_stage1_a8w4_fwd",
    gen_fake=_gen_opus_moe_stage1_a8w4_fake_tensors,
    develop=True,
)
def _opus_moe_stage1_a8w4_fwd_raw(
    hidden_states: Tensor,
    w1: Tensor,
    hidden_scale: Tensor,
    w1_scale: Tensor,
    bias: Tensor | None,
    sorted_token_ids: Tensor,
    sorted_expert_ids: Tensor,
    num_valid_ids: Tensor,
    out: Tensor,
    out_scale: Tensor,
    topk: int,
    block_m: int,
    kernelName: str,
    inter_dim_pad: int,
    activation: int,
    swiglu_limit: float,
    situ_beta: float,
    situ_linear_beta: float,
) -> Tensor: ...


def _make_out_scale(
    *,
    sorted_token_ids: Tensor,
    sorted_expert_ids: Tensor,
    block_m: int,
    inter_dim: int,
) -> Tensor:
    sorted_size = max(
        int(sorted_token_ids.numel()),
        int(sorted_expert_ids.numel()) * int(block_m),
    )
    padded_rows = (sorted_size + 255) // 256 * 256
    scale_cols = inter_dim // _OPUS_MOE_STAGE1_A8W4_SCALE_GROUP
    padded_cols = (scale_cols + 7) // 8 * 8
    return torch.zeros(
        (padded_rows, padded_cols),
        dtype=torch.float8_e8m0fnu,
        device=sorted_token_ids.device,
    )


def opus_moe_stage1_a8w4_fwd(
    hidden_states: Tensor,
    w1: Tensor,
    hidden_scale: Tensor,
    w1_scale: Tensor,
    sorted_token_ids: Tensor,
    sorted_expert_ids: Tensor,
    num_valid_ids: Tensor,
    *,
    topk: int,
    inter_dim_pad: int,
    block_m: int,
    kernelName: str,
    activation: int = ActivationType.Silu.value,
    bias: Tensor | None = None,
    out: Tensor | None = None,
    out_scale: Tensor | None = None,
    output_sorted: bool = False,
    swiglu_limit: float | None = None,
    situ_beta: float = 4.0,
    situ_linear_beta: float = 25.0,
) -> tuple[Tensor, Tensor]:
    block_m = int(block_m)
    kernelName = str(kernelName)
    activation = int(getattr(activation, "value", activation))
    swiglu_limit = _opus_runtime_swiglu_limit(swiglu_limit, activation)
    inter_dim = int(w1.shape[1]) // 2
    if out is None:
        out_shape = (
            (
                max(
                    int(sorted_token_ids.numel()),
                    int(sorted_expert_ids.numel()) * block_m,
                ),
                inter_dim,
            )
            if output_sorted
            else (hidden_states.shape[0], int(topk), inter_dim)
        )
        out = torch.empty(
            out_shape,
            dtype=torch.float8_e4m3fn,
            device=hidden_states.device,
        )
    if out_scale is None:
        out_scale = _make_out_scale(
            sorted_token_ids=sorted_token_ids,
            sorted_expert_ids=sorted_expert_ids,
            block_m=block_m,
            inter_dim=out.shape[-1],
        )

    _opus_moe_stage1_a8w4_fwd_raw(
        _contiguous(hidden_states),
        _contiguous(w1),
        _contiguous(hidden_scale),
        _contiguous(w1_scale),
        _contiguous(bias) if bias is not None else None,
        _contiguous(sorted_token_ids),
        _contiguous(sorted_expert_ids),
        _contiguous(num_valid_ids),
        _contiguous(out),
        _contiguous(out_scale),
        int(topk),
        int(block_m),
        kernelName,
        int(inter_dim_pad),
        activation,
        float(swiglu_limit),
        float(situ_beta),
        float(situ_linear_beta),
    )
    return out, out_scale


__all__ = [
    "opus_moe_stage1_a8w4_fwd",
]
