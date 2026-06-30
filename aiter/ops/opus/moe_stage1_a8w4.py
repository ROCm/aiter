# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor

from ...jit.core import compile_ops
from ..opus_moe_stage1_a8w4_meta import (
    OPUS_A8W4_STAGE1_KID_P0_BM32_BN384_A_REUSE_MFMA,
    OPUS_A8W4_STAGE1_KID_P0_BM16_BN64_SBM32_G1_A_REUSE_MFMA,
    OPUS_A8W4_STAGE1_KID_P0_BM16_BN64_SBM32_G1_KW2_A_REUSE_MFMA,
    OPUS_A8W4_STAGE1_KID_P0_BM16_BN384_G1_KW4_A_REUSE_MFMA,
    OPUS_A8W4_STAGE1_KID_P0_BM64_BN384_GATE_UP_GROUP_SPLIT,
    OPUS_A8W4_STAGE1_KID_P0_BM128_BN256_GATE_UP_GROUP_SPLIT,
    OPUS_A8W4_STAGE1_KID_P0_BM64_BN256_GATE_UP_GROUP_SPLIT,
)

_OPUS_MOE_STAGE1_A8W4_P0_BM32_BN384_A_REUSE_MFMA_KERNEL_ID = (
    OPUS_A8W4_STAGE1_KID_P0_BM32_BN384_A_REUSE_MFMA
)
_OPUS_MOE_STAGE1_A8W4_P0_BM16_BN64_SBM32_G1_A_REUSE_MFMA_KERNEL_ID = (
    OPUS_A8W4_STAGE1_KID_P0_BM16_BN64_SBM32_G1_A_REUSE_MFMA
)
_OPUS_MOE_STAGE1_A8W4_P0_BM16_BN64_SBM32_G1_KW2_A_REUSE_MFMA_KERNEL_ID = (
    OPUS_A8W4_STAGE1_KID_P0_BM16_BN64_SBM32_G1_KW2_A_REUSE_MFMA
)
_OPUS_MOE_STAGE1_A8W4_P0_BM16_BN384_G1_KW4_A_REUSE_MFMA_KERNEL_ID = (
    OPUS_A8W4_STAGE1_KID_P0_BM16_BN384_G1_KW4_A_REUSE_MFMA
)
_OPUS_MOE_STAGE1_A8W4_P0_BM64_BN384_GATE_UP_GROUP_SPLIT_KERNEL_ID = (
    OPUS_A8W4_STAGE1_KID_P0_BM64_BN384_GATE_UP_GROUP_SPLIT
)
_OPUS_MOE_STAGE1_A8W4_P0_BM128_BN256_GATE_UP_GROUP_SPLIT_KERNEL_ID = (
    OPUS_A8W4_STAGE1_KID_P0_BM128_BN256_GATE_UP_GROUP_SPLIT
)
_OPUS_MOE_STAGE1_A8W4_P0_BM64_BN256_GATE_UP_GROUP_SPLIT_KERNEL_ID = (
    OPUS_A8W4_STAGE1_KID_P0_BM64_BN256_GATE_UP_GROUP_SPLIT
)
_OPUS_MOE_STAGE1_A8W4_P0_KERNEL_ID = (
    _OPUS_MOE_STAGE1_A8W4_P0_BM128_BN256_GATE_UP_GROUP_SPLIT_KERNEL_ID
)
_OPUS_MOE_STAGE1_A8W4_BLOCK_M = 128
_OPUS_MOE_STAGE1_A8W4_INTER_DIM = 512
_OPUS_MOE_STAGE1_A8W4_SCALE_GROUP = 32


def _contiguous(tensor: Tensor) -> Tensor:
    return tensor if tensor.is_contiguous() else tensor.contiguous()


def _gen_opus_moe_stage1_a8w4_fake_tensors(
    hidden_states: Tensor,
    w1: Tensor,
    hidden_scale: Tensor,
    w1_scale: Tensor,
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
    sorted_token_ids: Tensor,
    sorted_expert_ids: Tensor,
    num_valid_ids: Tensor,
    out: Tensor,
    out_scale: Tensor,
    block_m: int,
    kernel_id: int,
    inter_dim_pad: int,
    model_dim_pad: int,
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
    return torch.empty(
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
    model_dim_pad: int = 0,
    out: Optional[Tensor] = None,
    out_scale: Optional[Tensor] = None,
    block_m: int = _OPUS_MOE_STAGE1_A8W4_BLOCK_M,
    kernel_id: int = _OPUS_MOE_STAGE1_A8W4_P0_KERNEL_ID,
) -> tuple[Tensor, Tensor]:
    if out is None:
        out = torch.empty(
            (hidden_states.shape[0], int(topk), _OPUS_MOE_STAGE1_A8W4_INTER_DIM),
            dtype=torch.float8_e4m3fn,
            device=hidden_states.device,
        )
    if out_scale is None:
        out_scale = _make_out_scale(
            sorted_token_ids=sorted_token_ids,
            sorted_expert_ids=sorted_expert_ids,
            block_m=int(block_m),
            inter_dim=out.shape[-1],
        )

    _opus_moe_stage1_a8w4_fwd_raw(
        _contiguous(hidden_states),
        _contiguous(w1),
        _contiguous(hidden_scale),
        _contiguous(w1_scale),
        _contiguous(sorted_token_ids),
        _contiguous(sorted_expert_ids),
        _contiguous(num_valid_ids),
        _contiguous(out),
        _contiguous(out_scale),
        int(block_m),
        int(kernel_id),
        int(inter_dim_pad),
        int(model_dim_pad),
    )
    return out, out_scale


__all__ = [
    "_OPUS_MOE_STAGE1_A8W4_P0_KERNEL_ID",
    "_OPUS_MOE_STAGE1_A8W4_P0_BM32_BN384_A_REUSE_MFMA_KERNEL_ID",
    "_OPUS_MOE_STAGE1_A8W4_P0_BM16_BN64_SBM32_G1_A_REUSE_MFMA_KERNEL_ID",
    "_OPUS_MOE_STAGE1_A8W4_P0_BM16_BN64_SBM32_G1_KW2_A_REUSE_MFMA_KERNEL_ID",
    "_OPUS_MOE_STAGE1_A8W4_P0_BM16_BN384_G1_KW4_A_REUSE_MFMA_KERNEL_ID",
    "_OPUS_MOE_STAGE1_A8W4_P0_BM64_BN384_GATE_UP_GROUP_SPLIT_KERNEL_ID",
    "_OPUS_MOE_STAGE1_A8W4_P0_BM128_BN256_GATE_UP_GROUP_SPLIT_KERNEL_ID",
    "_OPUS_MOE_STAGE1_A8W4_P0_BM64_BN256_GATE_UP_GROUP_SPLIT_KERNEL_ID",
    "opus_moe_stage1_a8w4_fwd",
]
