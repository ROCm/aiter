# SPDX-License-Identifier: MIT
# Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

from torch import Tensor
from typing import List, Optional
from ..jit.core import compile_ops, CK_DIR, AITER_CSRC_DIR, AITER_ROOT_DIR, AITER_CORE_DIR
import torch.nn.functional as F


@compile_ops("module_moe_asm")
def topk_softmax(
    topk_weights: Tensor,
    topk_indices: Tensor,
    token_expert_indices: Tensor,
    gating_output: Tensor,
    need_renorm: bool,
): ...


@compile_ops("module_moe_asm")
def moe_sum(input: Tensor, output: Tensor): ...


@compile_ops("module_moe_asm")
def moe_align_block_size(
    topk_ids: Tensor,
    num_experts: int,
    block_size: int,
    sorted_token_ids: Tensor,
    experts_ids: Tensor,
    token_nums: Tensor,
    num_tokens_post_pad: Tensor,
): ...


@compile_ops("module_moe_asm")
def fmoe(
    out: Tensor,
    input: Tensor,
    gate: Tensor,
    down: Tensor,
    sorted_token_ids: Tensor,
    sorted_weight_buf: Tensor,
    sorted_expert_ids: Tensor,
    num_valid_ids: Tensor,
    topk: int,
): ...


@compile_ops("module_moe_asm", fc_name='ActivationType')
class _ActivationType():
    ...


ActivationType = _ActivationType(0)


@compile_ops("module_moe_asm")
def fmoe_int8_g1u0(
    out: Tensor,
    input: Tensor,
    gate: Tensor,
    down: Tensor,
    sorted_token_ids: Tensor,
    sorted_weight_buf: Tensor,
    sorted_expert_ids: Tensor,
    num_valid_ids: Tensor,
    topk: int,
    input_scale: Tensor,
    fc1_scale: Tensor,
    fc2_scale: Tensor,
    fc2_smooth_scale: Tensor,
    activation: Optional[ActivationType] = ActivationType.Silu,
): ...


@compile_ops("module_moe_asm")
def fmoe_g1u1(
    out: Tensor,
    input: Tensor,
    gate: Tensor,
    down: Tensor,
    sorted_token_ids: Tensor,
    sorted_weight_buf: Tensor,
    sorted_expert_ids: Tensor,
    num_valid_ids: Tensor,
    topk: int,
    input_scale: Tensor,
    fc1_scale: Tensor,
    fc2_scale: Tensor,
    fc2_smooth_scale: Optional[Tensor] = None,
    activation: Optional[ActivationType] = ActivationType.Silu,
): ...


@compile_ops("module_moe_asm")
def fmoe_int8_g1u0_a16(
    out: Tensor,
    input: Tensor,  # bf16
    gate: Tensor,
    down: Tensor,
    sorted_token_ids: Tensor,
    sorted_weight_buf: Tensor,
    sorted_expert_ids: Tensor,
    num_valid_ids: Tensor,
    topk: int,
    fc1_scale: Tensor,
    fc2_scale: Tensor,
    fc1_smooth_scale: Tensor,
    fc2_smooth_scale: Tensor,
): ...


@compile_ops("module_moe_asm")
def fmoe_fp8_g1u1_a16(
    out: Tensor,
    input: Tensor,  # bf16
    gate: Tensor,
    down: Tensor,
    sorted_token_ids: Tensor,
    sorted_weight_buf: Tensor,
    sorted_expert_ids: Tensor,
    num_valid_ids: Tensor,
    topk: int,
    fc1_scale: Tensor,
    fc2_scale: Tensor,
    fc1_smooth_scale: Tensor,
    fc2_smooth_scale: Tensor,
): ...


@compile_ops("module_moe_asm")
def fmoe_fp8_blockscale_g1u1(
    out: Tensor,
    input: Tensor,
    gate: Tensor,
    down: Tensor,
    sorted_token_ids: Tensor,
    sorted_weight_buf: Tensor,
    sorted_expert_ids: Tensor,
    num_valid_ids: Tensor,
    topk: int,
    fc1_scale: Tensor,
    fc2_scale: Tensor,
    input_scale: Optional[Tensor] = None,
    fc_scale_blkn: Optional[Tensor] = 128,
    fc_scale_blkk: Optional[Tensor] = 128,
    fc2_smooth_scale: Optional[Tensor] = None,
): ...


@compile_ops("module_moe")
def ck_moe(
    hidden_states: Tensor,
    w1: Tensor,
    w2: Tensor,
    topk_weight: Tensor,
    topk_ids: Tensor,
    fc1_scale: Optional[Tensor] = None,
    fc2_scale: Optional[Tensor] = None,
    fc1_smooth_scale: Optional[Tensor] = None,
    fc2_smooth_scale: Optional[Tensor] = None,
    block_m: Optional[int] = 32,
    expert_mask: Optional[Tensor] = None
): ...


@compile_ops("module_moe_ck2stages")
def ck_moe_stage1(
    hidden_states: Tensor,
    w1: Tensor,
    w2: Tensor,
    sorted_token_ids: Tensor,
    sorted_expert_ids: Tensor,
    num_valid_ids: Tensor,
    out: Tensor,
    topk: int,
    w1_scale: Optional[Tensor] = None,
    a1_scale: Optional[Tensor] = None,
    block_m: Optional[int] = 32
): ...


@compile_ops("module_moe_ck2stages")
def ck_moe_stage2(
    inter_states: Tensor,
    w1: Tensor,
    w2: Tensor,
    sorted_token_ids: Tensor,
    sorted_expert_ids: Tensor,
    sorted_weights: Tensor,
    num_valid_ids: Tensor,
    out: Tensor,
    topk: int,
    w2_scale: Optional[Tensor] = None,
    a2_scale: Optional[Tensor] = None,
    block_m: Optional[int] = 32
): ...
