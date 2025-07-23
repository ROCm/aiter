# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import torch
from torch import Tensor
from typing import Optional
from ..jit.core import compile_ops, AITER_CSRC_DIR
from .enum import ActivationType, Enum, QuantType
from ..utility import dtypes
import functools

torch.int4 = getattr(torch, "int4", torch.uint32)


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
    sorted_weights: Tensor,
    sorted_expert_ids: Tensor,
    num_valid_ids: Tensor,
    topk: int,
    activation: Optional[Enum] = ActivationType.Silu,
): ...


@compile_ops("module_moe_asm")
def fmoe_int8_g1u0(
    out: Tensor,
    input: Tensor,
    gate: Tensor,
    down: Tensor,
    sorted_token_ids: Tensor,
    sorted_weights: Tensor,
    sorted_expert_ids: Tensor,
    num_valid_ids: Tensor,
    topk: int,
    input_scale: Tensor,
    fc1_scale: Tensor,
    fc2_scale: Tensor,
    fc2_smooth_scale: Tensor,
    activation: Optional[Enum] = ActivationType.Silu,
): ...


@compile_ops("module_moe_asm")
def fmoe_g1u1(
    out: Tensor,
    input: Tensor,
    gate: Tensor,
    down: Tensor,
    sorted_token_ids: Tensor,
    sorted_weights: Tensor,
    sorted_expert_ids: Tensor,
    num_valid_ids: Tensor,
    topk: int,
    input_scale: Tensor,
    fc1_scale: Tensor,
    fc2_scale: Tensor,
    fc2_smooth_scale: Optional[Tensor] = None,
    activation: Optional[Enum] = ActivationType.Silu,
): ...


@compile_ops("module_moe_asm")
def fmoe_g1u1_tkw1(
    out: Tensor,
    input: Tensor,
    gate: Tensor,
    down: Tensor,
    sorted_token_ids: Tensor,
    sorted_weights: Tensor,
    sorted_expert_ids: Tensor,
    num_valid_ids: Tensor,
    topk: int,
    input_scale: Tensor,
    fc1_scale: Tensor,
    fc2_scale: Tensor,
    fc2_smooth_scale: Optional[Tensor] = None,
    activation: Optional[Enum] = ActivationType.Silu,
): ...


@compile_ops("module_moe_asm")
def fmoe_int8_g1u0_a16(
    out: Tensor,
    input: Tensor,  # bf16
    gate: Tensor,
    down: Tensor,
    sorted_token_ids: Tensor,
    sorted_weights: Tensor,
    sorted_expert_ids: Tensor,
    num_valid_ids: Tensor,
    topk: int,
    fc1_scale: Tensor,
    fc2_scale: Tensor,
    fc1_smooth_scale: Tensor,
    fc2_smooth_scale: Tensor,
): ...


@compile_ops("module_moe_asm")
def fmoe_g1u1_a16(
    out: Tensor,
    input: Tensor,  # bf16
    gate: Tensor,
    down: Tensor,
    sorted_token_ids: Tensor,
    sorted_weights: Tensor,
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
    sorted_weights: Tensor,
    sorted_expert_ids: Tensor,
    num_valid_ids: Tensor,
    topk: int,
    input_scale: Tensor,
    fc1_scale: Tensor,
    fc2_scale: Tensor,
    fc_scale_blkn: int = 128,
    fc_scale_blkk: int = 128,
    fc2_smooth_scale: Optional[Tensor] = None,
    activation: ActivationType = ActivationType.Silu,
): ...


@compile_ops("module_moe_asm")
def moe_stage1_g1u1(
    input: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    sorted_expert_ids: torch.Tensor,
    num_valid_ids: torch.Tensor,
    out: torch.Tensor,
    inter_dim: int,
    kernelName: str,
    block_m: int,
    ksplit: int = 0,
    activation: ActivationType = ActivationType.Silu,
    quant_type: QuantType = QuantType.No,
    a1_scale: Optional[torch.Tensor] = None,
    w1_scale: Optional[torch.Tensor] = None,
    sorted_weights: Optional[torch.Tensor] = None,
) -> None: ...


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
    kernelName: str = "",
    w1_scale: Optional[Tensor] = None,
    a1_scale: Optional[Tensor] = None,
    block_m: Optional[int] = 32,
    sorted_weights: Optional[Tensor] = None,
): ...


@compile_ops("module_moe_ck2stages")
def ck_moe_stage2(
    inter_states: Tensor,
    w1: Tensor,
    w2: Tensor,
    sorted_token_ids: Tensor,
    sorted_expert_ids: Tensor,
    num_valid_ids: Tensor,
    out: Tensor,
    topk: int,
    kernelName: str = "",
    w2_scale: Optional[Tensor] = None,
    a2_scale: Optional[Tensor] = None,
    block_m: Optional[int] = 32,
    sorted_weights: Optional[Tensor] = None,
): ...


dtype2str_dict = {
    dtypes.fp16: "f16",
    dtypes.bf16: "b16",
    dtypes.fp8: "f8",
    dtypes.i8: "i8",
    torch.uint8: "fp4x2",
    torch.uint32: "i4",
}


@functools.lru_cache(maxsize=1024)
def get_moe_stage_module(
    input_dtype,
    weight_dtype,
    output_dtype,
    activation,
    quant_type,
    mul_routed_weight_stage,
):
    Adtype = dtype2str_dict[input_dtype]
    Bdtype = dtype2str_dict[weight_dtype]
    Cdtype = dtype2str_dict[output_dtype]

    act = str(activation).split(".")[-1].lower()
    quant_type = str(quant_type).split(".")[-1].lower()

    md_name = ("_").join(
        [
            "module_moe_ck2stages",
            Adtype,
            Bdtype,
            Cdtype,
            act,
            quant_type,
            f"mulWeightStage{mul_routed_weight_stage}",
        ]
    )

    blob_gen_cmd = [
        f"{AITER_CSRC_DIR}/ck_gemm_moe_2stages_codegen/gen_instances.py -a {Adtype} -b {Bdtype} -c {Cdtype} -q {quant_type} -act {act} -m {mul_routed_weight_stage} -w {{}}"
    ]

    return md_name, blob_gen_cmd


def ck_moe_stage1_fwd(
    hidden_states: Tensor,
    w1: Tensor,
    w2: Tensor,
    sorted_token_ids: Tensor,
    sorted_expert_ids: Tensor,
    num_valid_ids: Tensor,
    out: Tensor,
    topk: int,
    kernelName: str = "",
    w1_scale: Optional[Tensor] = None,
    a1_scale: Optional[Tensor] = None,
    block_m: Optional[int] = 32,
    sorted_weights: Optional[Tensor] = None,
    quant_type: QuantType = QuantType.No,
    activation: ActivationType = ActivationType.Silu,
):
    mul_routed_weight_stage = 2 if sorted_weights is None else 1
    md_name, blob_gen_cmd = get_moe_stage_module(
        hidden_states.dtype,
        w1.dtype,
        out.dtype,
        activation,
        quant_type,
        mul_routed_weight_stage,
    )

    ck_moe_stage1(
        hidden_states,
        w1,
        w2,
        sorted_token_ids,
        sorted_expert_ids,
        num_valid_ids,
        out,
        topk,
        kernelName,
        w1_scale,
        a1_scale,
        block_m,
        sorted_weights,
        custom_build_args={"md_name": md_name, "blob_gen_cmd": blob_gen_cmd},
    )
    return out


def ck_moe_stage2_fwd(
    inter_states: Tensor,
    w1: Tensor,
    w2: Tensor,
    sorted_token_ids: Tensor,
    sorted_expert_ids: Tensor,
    num_valid_ids: Tensor,
    out: Tensor,
    topk: int,
    kernelName: str = "",
    w2_scale: Optional[Tensor] = None,
    a2_scale: Optional[Tensor] = None,
    block_m: Optional[int] = 32,
    sorted_weights: Optional[Tensor] = None,
    quant_type: QuantType = QuantType.No,
    activation: ActivationType = ActivationType.Silu,
):
    mul_routed_weight_stage = 1 if sorted_weights is None else 2

    md_name, blob_gen_cmd = get_moe_stage_module(
        inter_states.dtype,
        w1.dtype,
        out.dtype,
        activation,
        quant_type,
        mul_routed_weight_stage,
    )

    ck_moe_stage2(
        inter_states,
        w1,
        w2,
        sorted_token_ids,
        sorted_expert_ids,
        num_valid_ids,
        out,
        topk,
        kernelName,
        w2_scale,
        a2_scale,
        block_m,
        sorted_weights,
        custom_build_args={"md_name": md_name, "blob_gen_cmd": blob_gen_cmd},
    )
    return out
