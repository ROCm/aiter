# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

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
) -> None: ...


@compile_ops("module_moe_asm")
def topk_softmax_asm(
    topk_weights: Tensor,
    topk_indices: Tensor,
    token_expert_indices: Tensor,
    gating_output: Tensor,
    need_renorm: bool,
) -> None: ...


@compile_ops("module_moe_topk")
def topk_sigmoid(
    topk_weights: Tensor, topk_indices: Tensor, gating_output: Tensor
) -> None: ...


@compile_ops("module_moe_asm")
def moe_sum(input: Tensor, output: Tensor) -> None: ...


@compile_ops("module_moe_asm")
def moe_align_block_size(
    topk_ids: Tensor,
    num_experts: int,
    block_size: int,
    sorted_token_ids: Tensor,
    experts_ids: Tensor,
    token_nums: Tensor,
    num_tokens_post_pad: Tensor,
) -> None: ...


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
) -> None: ...


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
    activation: Optional[Enum] = ActivationType.Silu.value,
) -> None: ...


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
    kernelName: str,
    fc2_smooth_scale: Optional[Tensor] = None,
    activation: Optional[Enum] = ActivationType.Silu.value,
) -> None: ...


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
    kernelName: str,
    fc2_smooth_scale: Optional[Tensor] = None,
    activation: Optional[Enum] = ActivationType.Silu.value,
) -> None: ...


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
) -> None: ...


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
    activation: Optional[Enum] = ActivationType.Silu.value,
) -> None: ...


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
    kernelName: str,
    fc_scale_blkn: int = 128,
    fc_scale_blkk: int = 128,
    fc2_smooth_scale: Optional[Tensor] = None,
    activation: Optional[Enum] = ActivationType.Silu.value,
) -> None: ...

@compile_ops("module_moe_asm")
def fmoe_fp8_blockscale_with_xquant_g1u1(
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
    kernelName: str,
    fc_scale_blkn: int = 128,
    fc_scale_blkk: int = 128,
    fc2_smooth_scale: Optional[Tensor] = None,
    activation: Optional[Enum] = ActivationType.Silu.value,
) -> None: ...


@compile_ops("module_moe_asm")
def moe_stage1_g1u1(
    input: Tensor,
    w1: Tensor,
    w2: Tensor,
    sorted_token_ids: Tensor,
    sorted_expert_ids: Tensor,
    num_valid_ids: Tensor,
    out: Tensor,
    inter_dim: int,
    kernelName: str,
    block_m: int,
    ksplit: int = 0,
    activation: Optional[Enum] = ActivationType.Silu.value,
    quant_type: Optional[Enum] = QuantType.No.value,
    a1_scale: Optional[Tensor] = None,
    w1_scale: Optional[Tensor] = None,
    sorted_weights: Optional[Tensor] = None,
) -> None: ...


def cmdGenFunc_ck_moe_stage(
    hidden_states: Tensor,
    w1: Tensor,
    w2: Tensor,
    sorted_token_ids: Tensor,
    sorted_expert_ids: Tensor,
    num_valid_ids: Tensor,
    out: Tensor,
    topk: int,
    kernelName: Optional[str] = None,
    w1_scale: Optional[Tensor] = None,
    a1_scale: Optional[Tensor] = None,
    block_m: Optional[int] = 32,
    sorted_weights: Optional[Tensor] = None,
    quant_type: int = 0,
    activation: int = 0,
    splitk: int = 1,
    use_non_temporal_load: bool = False,
    dst_type: Optional[str] = None,
    is_shuffled: bool = True,
):

    mul_routed_weight_stage = 2 if sorted_weights is None else 1
    is_splitk = splitk > 1
    outtype = str2dtype_dict[dst_type] if is_splitk else out.dtype
    md_name, blob_gen_cmd = get_moe_stage_module(
        hidden_states.dtype,
        w1.dtype,
        outtype,
        activation,
        quant_type,
        mul_routed_weight_stage,
        is_shuffled,
        is_splitk,
    )
    return {
        "md_name": md_name,
        "blob_gen_cmd": blob_gen_cmd,
    }


def cmdGenFunc_ck_moe_stage2(
    hidden_states: Tensor,
    w1: Tensor,
    w2: Tensor,
    sorted_token_ids: Tensor,
    sorted_expert_ids: Tensor,
    num_valid_ids: Tensor,
    out: Tensor,
    topk: int,
    kernelName: Optional[str] = None,
    w1_scale: Optional[Tensor] = None,
    a1_scale: Optional[Tensor] = None,
    block_m: Optional[int] = 32,
    sorted_weights: Optional[Tensor] = None,
    quant_type: int = 0,
    activation: int = 0,
    splitk: int = 1,
    use_non_temporal_load: bool = False,
    dst_type: Optional[str] = None,
    is_shuffled: bool = True,
):

    mul_routed_weight_stage = 1 if sorted_weights is None else 2
    md_name, blob_gen_cmd = get_moe_stage_module(
        hidden_states.dtype,
        w1.dtype,
        out.dtype,
        activation,
        quant_type,
        mul_routed_weight_stage,
        is_shuffled,
    )
    return {
        "md_name": md_name,
        "blob_gen_cmd": blob_gen_cmd,
    }


@compile_ops("module_moe_ck2stages", gen_func=cmdGenFunc_ck_moe_stage)
def ck_moe_stage1(
    hidden_states: Tensor,
    w1: Tensor,
    w2: Tensor,
    sorted_token_ids: Tensor,
    sorted_expert_ids: Tensor,
    num_valid_ids: Tensor,
    out: Tensor,
    topk: int,
    kernelName: Optional[str] = None,
    w1_scale: Optional[Tensor] = None,
    a1_scale: Optional[Tensor] = None,
    block_m: Optional[int] = 32,
    sorted_weights: Optional[Tensor] = None,
    quant_type: int = 0,
    activation: int = 0,
    splitk: Optional[int] = 1,
    use_non_temporal_load: bool = False,
    dst_type: Optional[str] = None,
    is_shuffled: bool = True,
) -> None: ...


@compile_ops("module_moe_ck2stages", gen_func=cmdGenFunc_ck_moe_stage2)
def ck_moe_stage2(
    inter_states: Tensor,
    w1: Tensor,
    w2: Tensor,
    sorted_token_ids: Tensor,
    sorted_expert_ids: Tensor,
    num_valid_ids: Tensor,
    out: Tensor,
    topk: int,
    kernelName: Optional[str] = None,
    w2_scale: Optional[Tensor] = None,
    a2_scale: Optional[Tensor] = None,
    block_m: Optional[int] = 32,
    sorted_weights: Optional[Tensor] = None,
    quant_type: int = 0,
    activation: int = 0,
    splitk: int = 1,
    use_non_temporal_load: bool = False,
    dst_type: Optional[str] = None,
    is_shuffled: bool = True,
) -> None: ...


@compile_ops("module_moe_cktile2stages", fc_name="cktile_moe_gemm1")
def moe_cktile2stages_gemm1_ck(
    XQ: Tensor,
    WQ: Tensor,
    Y: Tensor,
    sorted_ids: Tensor,
    sorted_expert_ids: Tensor,
    max_token_ids: Tensor,
    topk: int,
    n_padded_zeros: Optional[int] = 0,
    k_padded_zeros: Optional[int] = 0,
    topk_weight: Optional[Tensor] = None,
    x_scale: Optional[Tensor] = None,
    w_scale: Optional[Tensor] = None,
    exp_bias: Optional[Tensor] = None,
    activation: Optional[int] = 0,
    block_m: Optional[int] = 32,
    split_k: Optional[int] = 1,
) -> Tensor: ...


def moe_cktile2stages_gemm1(
    XQ: Tensor,
    WQ: Tensor,
    Y: Tensor,
    sorted_ids: Tensor,
    sorted_expert_ids: Tensor,
    max_token_ids: Tensor,
    topk: int,
    n_padded_zeros: Optional[int] = 0,
    k_padded_zeros: Optional[int] = 0,
    topk_weight: Optional[Tensor] = None,
    x_scale: Optional[Tensor] = None,
    w_scale: Optional[Tensor] = None,
    exp_bias: Optional[Tensor] = None,
    activation: Optional[int] = 0,
    block_m: Optional[int] = 32,
    split_k: Optional[int] = 1,
):
    return moe_cktile2stages_gemm1_ck(
        XQ,
        WQ,
        Y,
        sorted_ids,
        sorted_expert_ids,
        max_token_ids,
        topk,
        n_padded_zeros,
        k_padded_zeros,
        topk_weight,
        x_scale,
        w_scale,
        exp_bias,
        activation,
        block_m,
        split_k,
    )


@compile_ops("module_moe_cktile2stages", fc_name="cktile_moe_gemm2")
def moe_cktile2stages_gemm2_ck(
    XQ: Tensor,
    WQ: Tensor,
    Y: Tensor,
    sorted_ids: Tensor,
    sorted_expert_ids: Tensor,
    max_token_ids: Tensor,
    topk: int,
    n_padded_zeros: Optional[int] = 0,
    k_padded_zeros: Optional[int] = 0,
    topk_weight: Optional[Tensor] = None,
    x_scale: Optional[Tensor] = None,
    w_scale: Optional[Tensor] = None,
    exp_bias: Optional[Tensor] = None,
    activation: Optional[int] = 0,
    block_m: Optional[int] = 32,
    split_k: Optional[int] = 1,
) -> Tensor: ...


def moe_cktile2stages_gemm2(
    XQ: Tensor,
    WQ: Tensor,
    Y: Tensor,
    sorted_ids: Tensor,
    sorted_expert_ids: Tensor,
    max_token_ids: Tensor,
    topk: int,
    n_padded_zeros: Optional[int] = 0,
    k_padded_zeros: Optional[int] = 0,
    topk_weight: Optional[Tensor] = None,
    x_scale: Optional[Tensor] = None,
    w_scale: Optional[Tensor] = None,
    exp_bias: Optional[Tensor] = None,
    activation: Optional[int] = 0,
    block_m: Optional[int] = 32,
    split_k: Optional[int] = 1,
):
    return moe_cktile2stages_gemm2_ck(
        XQ,
        WQ,
        Y,
        sorted_ids,
        sorted_expert_ids,
        max_token_ids,
        topk,
        n_padded_zeros,
        k_padded_zeros,
        topk_weight,
        x_scale,
        w_scale,
        exp_bias,
        activation,
        block_m,
        split_k,
    )


dtype2str_dict = {
    dtypes.fp16: "f16",
    dtypes.bf16: "b16",
    dtypes.fp8: "f8",
    dtypes.i8: "i8",
    dtypes.fp4x2: "fp4x2",
    torch.uint32: "i4",
    torch.int4: "i4",
}

str2dtype_dict = {
    "f16": dtypes.fp16,
    "b16": dtypes.bf16,
}


@functools.lru_cache(maxsize=1024)
def get_moe_stage_module(
    input_dtype,
    weight_dtype,
    output_dtype,
    activation,
    quant_type,
    mul_routed_weight_stage,
    preshuffle_mode=False,
    is_splitk=False,
):
    if isinstance(activation, int):
        activation = ActivationType(activation)
    if isinstance(quant_type, int):
        quant_type = QuantType(quant_type)

    Adtype = dtype2str_dict[input_dtype]
    Bdtype = dtype2str_dict[weight_dtype]
    Cdtype = dtype2str_dict[output_dtype]

    preshuffle_str = ""
    if preshuffle_mode and weight_dtype == dtypes.fp4x2:
        preshuffle_str = "--preshuffle"

    splitk_str = "--issplitk" if is_splitk else ""
    quant_type = (
        QuantType.per_1x128 if quant_type == QuantType.per_128x128 else quant_type
    )
    act = str(activation).split(".")[-1].lower()
    quant_type = str(quant_type).split(".")[-1].lower()

    md_name = ("_").join(
        [
            "module_moe_ck2stages",
            Adtype,
            Bdtype,
            "preshuffle_on" if preshuffle_mode else "preshuffle_off",
            Cdtype,
            act,
            quant_type,
            f"mulWeightStage{mul_routed_weight_stage}",
            "splitk" if is_splitk else "",
        ]
    )
    blob_gen_cmd = [
        f"{AITER_CSRC_DIR}/ck_gemm_moe_2stages_codegen/gen_instances.py -a {Adtype} -b {Bdtype} -c {Cdtype} -q {quant_type} -act {act} -m {mul_routed_weight_stage} {preshuffle_str} {splitk_str} -w {{}}"
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
    splitk: Optional[int] = 1,
    use_non_temporal_load: Optional[bool] = False,
    dst_type: Optional[torch.dtype] = None,
):
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
        quant_type.value,
        activation.value,
        int(splitk) if splitk is not None else splitk,
        use_non_temporal_load,
        None if dst_type is None else dtype2str_dict[dst_type],
        is_shuffled=getattr(w1, "is_shuffled", False),
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
    use_non_temporal_load: Optional[bool] = False,
):
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
        quant_type.value,
        activation.value,
        use_non_temporal_load=use_non_temporal_load,
        is_shuffled=getattr(w2, "is_shuffled", False),
    )
    return out
