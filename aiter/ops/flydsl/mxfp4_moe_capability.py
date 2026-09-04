# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Dispatch capability checks shared by the A4W4 SiTUv2 low-M FMoE fast paths.

Both the BM16 inline-sort two-stage path and the direct raw-topk M1 executor
have to reject the same set of unsupported invocations and resolve legacy
metadata instead. The checks live here so neither copy can drift.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from aiter.utility import dtypes


def enum_name(value: Any) -> str:
    """Normalize an enum, its value, or a plain string to a bare lowercase name.

    ActivationType is backed by module_aiter_core, which is not importable
    while AOT job discovery runs; comparing names avoids the import.
    """
    return str(value).split(".")[-1].strip().lower()


def scale_cols(size: int, group_size: int = 32) -> int:
    """Scale columns per row, padded the way ``e8m0_shuffle`` pads them."""
    return ((size // group_size + 7) // 8) * 8


def metadata_kernel_name(metadata: Any, stage: int) -> str:
    partial = getattr(metadata, f"stage{stage}", None)
    keywords = getattr(partial, "keywords", None) or {}
    for key in (f"kernelName{stage}", "kernelName"):
        value = keywords.get(key)
        if isinstance(value, str) and value:
            return value
    return ""


@dataclass(frozen=True)
class MoeCall:
    """The part of a ``fused_moe`` invocation that dispatch decisions read.

    Host-only: shapes, dtypes and options. No tensor value is ever read, so
    building one never synchronizes with the device.
    """

    hidden_states: torch.Tensor
    w1: torch.Tensor
    w2: torch.Tensor
    topk_weight: torch.Tensor
    topk_ids: torch.Tensor
    w1_scale: torch.Tensor | None
    w2_scale: torch.Tensor | None
    dtype: torch.dtype
    q_dtype_a: torch.dtype
    q_dtype_w: torch.dtype
    quant_type: Any
    activation: Any
    gate_mode: Any
    isG1U1: bool
    doweight_stage1: bool
    # Optional beyond this point: absent means "not requested".
    expert_mask: torch.Tensor | None = None
    num_local_tokens: torch.Tensor | None = None
    bias1: torch.Tensor | None = None
    bias2: torch.Tensor | None = None
    a1_scale: torch.Tensor | None = None
    a2_scale: torch.Tensor | None = None
    stage2_scatter: Any = None
    hidden_pad: int = 0
    intermediate_pad: int = 0
    block_size_M: int | None = None
    beta: float | None = None
    linear_beta: float | None = None

    @property
    def tokens(self) -> int:
        return int(self.hidden_states.shape[0])

    @property
    def hidden(self) -> int:
        return int(self.hidden_states.shape[1])

    @property
    def experts(self) -> int:
        return int(self.w1.shape[0])

    @property
    def inter(self) -> int:
        return int(self.w2.shape[-1]) * 2

    @property
    def topk(self) -> int:
        return int(self.topk_ids.shape[1])

    @property
    def tensors(self) -> tuple[torch.Tensor, ...]:
        return (
            self.hidden_states,
            self.w1,
            self.w2,
            self.topk_weight,
            self.topk_ids,
            self.w1_scale,
            self.w2_scale,
        )


#: Stage1 activations the low-M fast paths emit. SiTUv2 additionally consumes
#: the two beta kernel arguments; silu ignores them.
LOWM_ACTIVATIONS = ("situv2", "silu")


def check_a4w4_lowm(call: MoeCall) -> tuple[bool, str]:
    """Whether the A4W4 low-M fast paths may execute this call."""
    if call.expert_mask is not None or call.num_local_tokens is not None:
        return False, "expert-parallel masking or local-token metadata"
    if call.hidden_states.dtype != dtypes.bf16 or call.dtype != dtypes.bf16:
        return False, "activations and output must be bf16"
    if call.q_dtype_a != dtypes.fp4x2 or call.q_dtype_w != dtypes.fp4x2:
        return False, "requires MXFP4 activations and weights"
    if enum_name(call.activation) not in LOWM_ACTIVATIONS:
        return False, f"unsupported activation {call.activation}"
    if enum_name(call.quant_type) != "per_1x32":
        return False, f"unsupported quant type {call.quant_type}"
    if enum_name(call.gate_mode) != "separated":
        return False, f"unsupported gate mode {call.gate_mode}"
    if not call.isG1U1 or call.doweight_stage1:
        return False, "requires g1u1 weights with doweight_stage1=False"
    if not (
        getattr(call.w1, "is_shuffled", False)
        and getattr(call.w2, "is_shuffled", False)
    ):
        return False, "weights are not both preshuffled"

    s1, s2 = call.w1_scale, call.w2_scale
    if s1 is None or s2 is None:
        return False, "missing MXFP4 weight scales"
    if s1.dtype != dtypes.fp8_e8m0 or s2.dtype != dtypes.fp8_e8m0:
        return False, "MXFP4 weight scales must be e8m0"
    if s1.numel() < call.experts * 2 * call.inter * scale_cols(call.hidden) or (
        s2.numel() < call.experts * call.hidden * scale_cols(call.inter)
    ):
        # Padded, not exact: a non-256-aligned inter_dim (Kimi-K3 I=384 gives
        # 12 valid of 16 columns) makes the shuffled stride wider than the
        # payload, and the kernels address the padded stride.
        return False, "MXFP4 weight scales are smaller than the padded stride"

    if not all(tensor.is_contiguous() for tensor in call.tensors):
        return False, "non-contiguous tensors"
    if len({tensor.device for tensor in call.tensors}) != 1:
        return False, "cross-device tensors"

    for name in ("bias1", "bias2", "stage2_scatter", "a1_scale", "a2_scale"):
        if getattr(call, name) is not None:
            return False, f"unsupported option {name}"
    if call.hidden_pad or call.intermediate_pad:
        return False, "hidden/intermediate padding"
    return True, ""
