# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import functools
import os
from dataclasses import dataclass
from typing import Callable, Optional

import torch

import aiter

# from aiter import get_torch_quant as get_quant
from aiter import ActivationType, QuantType, dtypes
from aiter import get_hip_quant as get_quant
from aiter import logger
from aiter.jit.core import AITER_CONFIGS, PY, bd_dir, get_asm_dir, mp_lock
from aiter.jit.utils.chip_info import get_cu_num, get_gfx
from aiter.jit.utils.torch_guard import torch_compile_guard
from aiter.ops.triton.quant.fused_mxfp4_quant import fused_dynamic_mxfp4_quant_moe_sort
from aiter.ops.triton.moe.moe_align_block_size import moe_align_block_size_triton
from aiter.utility import fp4_utils

BLOCK_SIZE_M = 32


def moe_sorting_triton(
    topk_ids,
    topk_weights,
    num_experts,
    model_dim,
    moebuf_dtype,
    block_size=BLOCK_SIZE_M,
):
    device = topk_ids.device
    M, topk = topk_ids.shape
    total = topk_ids.numel()
    max_num_tokens_padded = int(total + num_experts * block_size - topk)
    max_num_m_blocks = int((max_num_tokens_padded + block_size - 1) // block_size)

    sorted_ids = torch.empty(max_num_tokens_padded, dtype=dtypes.i32, device=device)
    sorted_expert_ids = torch.empty(max_num_m_blocks, dtype=dtypes.i32, device=device)
    num_tokens_post_pad = torch.empty(1, dtype=dtypes.i32, device=device)

    moe_align_block_size_triton(
        topk_ids,
        num_experts,
        block_size,
        sorted_ids,
        sorted_expert_ids,
        num_tokens_post_pad,
    )

    # Gather sorted_weights from topk_weights using sorted_ids
    flat_weights = topk_weights.reshape(-1).to(dtypes.fp32)
    sorted_weights = torch.zeros(
        max_num_tokens_padded, dtype=dtypes.fp32, device=device
    )
    valid_mask = sorted_ids < total
    clamped_ids = sorted_ids.clamp(min=0, max=total - 1).long()
    sorted_weights[valid_mask] = flat_weights[clamped_ids[valid_mask]]

    # Build num_valid_ids[2] for compatibility with CK format
    num_valid_ids = torch.empty(2, dtype=dtypes.i32, device=device)
    num_valid_ids[0] = num_tokens_post_pad[0]
    num_valid_ids[1] = num_tokens_post_pad[0]

    moe_buf = torch.empty((M, model_dim), dtype=moebuf_dtype, device=device)

    return sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_buf


def moe_sorting(
    topk_ids,
    topk_weights,
    num_experts,
    model_dim,
    moebuf_dtype,
    block_size=BLOCK_SIZE_M,
    expert_mask=None,
    num_local_tokens=None,
    dispatch_policy=0,
):
    try:
        device = topk_ids.device
        M, topk = topk_ids.shape
        max_num_tokens_padded = int(topk_ids.numel() + num_experts * block_size - topk)

        max_num_m_blocks = int((max_num_tokens_padded + block_size - 1) // block_size)
        sorted_ids = torch.empty(max_num_tokens_padded, dtype=dtypes.i32, device=device)
        sorted_weights = torch.empty(
            max_num_tokens_padded, dtype=dtypes.fp32, device=device
        )
        sorted_expert_ids = torch.empty(
            max_num_m_blocks, dtype=dtypes.i32, device=device
        )
        num_valid_ids = torch.empty(2, dtype=dtypes.i32, device=device)
        moe_buf = torch.empty((M, model_dim), dtype=moebuf_dtype, device=device)

        aiter.moe_sorting_fwd(
            topk_ids,
            topk_weights,
            sorted_ids,
            sorted_weights,
            sorted_expert_ids,
            num_valid_ids,
            moe_buf,
            num_experts,
            int(block_size),
            expert_mask,
            num_local_tokens,
            dispatch_policy,
        )
    except Exception as e:
        if expert_mask is not None:
            logger.error(
                f"CK moe_sorting failed with expert_mask set, Triton fallback does not support EP masking: {e}"
            )
            raise
        logger.warning(f"CK moe_sorting_fwd unavailable ({e}), falling back to Triton")
        return moe_sorting_triton(
            topk_ids, topk_weights, num_experts, model_dim, moebuf_dtype, block_size
        )
    return sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_buf


# Lru cache will using hash to create key, which makes error when w1,w2 shape is symint.
# We can use torch.compile(dynamic=False) to avoid
@functools.lru_cache(maxsize=2048)
def get_inter_dim(w1_shape, w2_shape):
    E, _, model_dim = w1_shape
    E, model_dim, inter_dim = w2_shape

    int4_war = model_dim // w1_shape[-1]
    inter_dim *= int4_war
    return E, model_dim, inter_dim


def fused_moe(
    hidden_states,
    w1,  # [expert(local_expert:EP), inter_dim*2, dim] N,K
    w2,  # [expert(local_expert:EP), dim, inter_dim]
    topk_weight,
    topk_ids,
    expert_mask: Optional[torch.tensor] = None,  # EP
    activation=ActivationType.Silu,
    quant_type=QuantType.No,
    doweight_stage1=False,
    # following for quant
    w1_scale: Optional[torch.tensor] = None,  # [expert(local_expert:EP), inter_dim, 1]
    w2_scale: Optional[torch.tensor] = None,  # [expert(local_expert:EP), model_dim, 1]
    a1_scale: Optional[torch.tensor] = None,  # [expert(local_expert:EP), 1, model_dim]
    a2_scale: Optional[torch.tensor] = None,  # [expert(local_expert:EP), 1, inter_dim]
    # following for tuning
    block_size_M=None,
    num_local_tokens: Optional[torch.tensor] = None,
    moe_sorting_dispatch_policy=0,
    dtype=None,
    # following for cktile support
    hidden_pad=0,
    intermediate_pad=0,
    bias1=None,
    bias2=None,
    splitk=0,
):
    if not block_size_M:
        block_size_M = -1
    return fused_moe_(
        hidden_states=hidden_states,
        w1=w1,
        w2=w2,
        topk_weight=topk_weight,
        topk_ids=topk_ids,
        expert_mask=expert_mask,
        activation=activation.value,
        quant_type=quant_type.value,
        doweight_stage1=doweight_stage1,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        a1_scale=a1_scale,
        a2_scale=a2_scale,
        block_size_M=block_size_M,
        num_local_tokens=num_local_tokens,
        moe_sorting_dispatch_policy=moe_sorting_dispatch_policy,
        dtype=dtype,
        hidden_pad=hidden_pad,
        intermediate_pad=intermediate_pad,
        bias1=bias1,
        bias2=bias2,
    )


def fused_moe_fake(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,  # [expert(local_expert:EP), inter_dim*2, dim] N,K
    w2: torch.Tensor,  # [expert(local_expert:EP), dim, inter_dim]
    topk_weight: torch.Tensor,
    topk_ids: torch.Tensor,
    expert_mask: Optional[torch.Tensor] = None,  # EP
    activation: int = ActivationType.Silu.value,
    quant_type: int = QuantType.No.value,
    doweight_stage1: bool = False,
    # following for quant
    w1_scale: Optional[torch.Tensor] = None,  # [expert(local_expert:EP), inter_dim, 1]
    w2_scale: Optional[torch.Tensor] = None,  # [expert(local_expert:EP), model_dim, 1]
    a1_scale: Optional[torch.Tensor] = None,  # [expert(local_expert:EP), 1, model_dim]
    a2_scale: Optional[torch.Tensor] = None,  # [expert(local_expert:EP), 1, inter_dim]
    # following for tuning
    block_size_M: int = -1,
    num_local_tokens: Optional[torch.Tensor] = None,
    moe_sorting_dispatch_policy: bool = 0,
    dtype: Optional[torch.dtype] = None,
    hidden_pad: int = 0,
    intermediate_pad: int = 0,
    bias1: Optional[torch.Tensor] = None,
    bias2: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    device = topk_ids.device
    M, topk = topk_ids.shape
    dtype = hidden_states.dtype if dtype is None else dtype
    model_dim = w2.shape[1]
    moe_buf = torch.empty((M, model_dim), dtype=dtype, device=device)
    return moe_buf


@torch_compile_guard(gen_fake=fused_moe_fake)
def fused_moe_(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,  # [expert(local_expert:EP), inter_dim*2, dim] N,K
    w2: torch.Tensor,  # [expert(local_expert:EP), dim, inter_dim]
    topk_weight: torch.Tensor,
    topk_ids: torch.Tensor,
    expert_mask: Optional[torch.Tensor] = None,  # EP
    activation: int = ActivationType.Silu.value,
    quant_type: int = QuantType.No.value,
    doweight_stage1: bool = False,
    # following for quant
    w1_scale: Optional[torch.Tensor] = None,  # [expert(local_expert:EP), inter_dim, 1]
    w2_scale: Optional[torch.Tensor] = None,  # [expert(local_expert:EP), model_dim, 1]
    a1_scale: Optional[torch.Tensor] = None,  # [expert(local_expert:EP), 1, model_dim]
    a2_scale: Optional[torch.Tensor] = None,  # [expert(local_expert:EP), 1, inter_dim]
    # following for tuning
    block_size_M: int = -1,
    num_local_tokens: Optional[torch.Tensor] = None,
    moe_sorting_dispatch_policy: bool = 0,
    dtype: Optional[torch.dtype] = None,
    hidden_pad: int = 0,
    intermediate_pad: int = 0,
    bias1: Optional[torch.Tensor] = None,
    bias2: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    # We do such convert since custom_op schema restriction on block_size_M, and Enum type
    activation = ActivationType(activation)
    quant_type = QuantType(quant_type)
    if block_size_M == -1:
        block_size_M = None
    """user API"""
    M, topk = topk_ids.shape
    E, model_dim, inter_dim = get_inter_dim(w1.shape, w2.shape)

    assert w1.shape[1] in [
        inter_dim,
        inter_dim * 2,
    ], f"Invalid MoE weight: {w1.shape=} {w2.shape=}"
    isG1U1 = inter_dim != w1.shape[1]
    isShuffled = getattr(w1, "is_shuffled", False)

    global_E = E
    if expert_mask is not None:
        global_E = expert_mask.numel()
    dtype = hidden_states.dtype if dtype is None else dtype
    assert dtype in [
        dtypes.fp16,
        dtypes.bf16,
    ], f"Fused_moe unsupported out dtype: {dtype}"
    quant_type = quant_remap.get(quant_type, quant_type)
    q_dtype_w = w1.dtype
    q_dtype_a = w1.dtype if w1.dtype != torch.uint32 else dtypes.fp8
    bf16_fp8_bound = 512
    if quant_type == QuantType.per_1x32:
        if activation == ActivationType.Swiglu:
            if get_gfx() != "gfx950" or M < bf16_fp8_bound:
                q_dtype_a = dtypes.bf16
            elif M >= bf16_fp8_bound:
                q_dtype_a = dtypes.fp8
        else:
            q_dtype_a = dtypes.fp4x2

    metadata = get_2stage_cfgs(
        get_padded_M(M),  # consider token_num > 1024 as prefill
        model_dim,
        inter_dim,
        E,
        topk,
        dtype,
        q_dtype_a,
        q_dtype_w,
        quant_type,
        isG1U1,
        activation,
        doweight_stage1,
        hidden_pad,
        intermediate_pad,
        isShuffled,
    )

    block_size_M = metadata.block_m if block_size_M is None else block_size_M
    # Ensure block_size_M is int (metadata.block_m from CSV may be float)
    if block_size_M is not None:
        block_size_M = int(block_size_M)

    sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_buf = moe_sorting(
        topk_ids,
        topk_weight,
        global_E,
        model_dim,
        dtype,
        block_size_M,
        expert_mask,
        num_local_tokens,
        moe_sorting_dispatch_policy,
    )

    if metadata.run_1stage:
        return metadata.stage1(
            hidden_states,
            w1,
            w2,
            topk,
            sorted_ids,
            sorted_weights,
            sorted_expert_ids,
            num_valid_ids,
            moe_buf,
            isG1U1,
            block_size_M,
            # activation=activation,
            # quant_type=quant_type,
            q_dtype_a=q_dtype_a,
            q_dtype_w=q_dtype_w,
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            a1_scale=a1_scale,
            a2_scale=a2_scale,
            num_local_tokens=num_local_tokens,
            M=M,
            device=topk_ids.device,
            doweight_stage1=doweight_stage1,
        )
    else:
        return fused_moe_2stages(
            hidden_states,
            w1,
            w2,
            topk,
            sorted_ids,
            sorted_weights,
            sorted_expert_ids,
            num_valid_ids,
            moe_buf,
            isG1U1,
            block_size_M,
            activation=activation,
            quant_type=quant_type,
            doweight_stage1=doweight_stage1,
            q_dtype_a=q_dtype_a,
            q_dtype_w=q_dtype_w,
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            a1_scale=a1_scale,
            a2_scale=a2_scale,
            num_local_tokens=num_local_tokens,
            # following for cktile support
            hidden_pad=hidden_pad,
            intermediate_pad=intermediate_pad,
            bias1=bias1,
            bias2=bias2,
        )


def fused_moe_1stage(
    hidden_states,
    w1,  # [expert(local_expert:EP), inter_dim*2, dim] N,K
    w2,  # [expert(local_expert:EP), dim, inter_dim]
    topk,
    sorted_ids,
    sorted_weights,
    sorted_expert_ids,
    num_valid_ids,
    moe_buf,
    isG1U1,
    block_size_M=32,
    activation=ActivationType.Silu,
    quant_type=QuantType.No,
    kernelName: str = "",
    # following for quant
    q_dtype_a=None,
    q_dtype_w=None,
    w1_scale=None,  # [expert(local_expert:EP), inter_dim, 1]
    w2_scale=None,  # [expert(local_expert:EP), model_dim, 1]
    a1_scale=None,  # [expert(local_expert:EP), 1, model_dim]
    a2_scale=None,  # [expert(local_expert:EP), 1, inter_dim]
    num_local_tokens: Optional[torch.tensor] = None,
    M: int = None,
    device=None,
    doweight_stage1: bool = None,
):
    if quant_type == QuantType.No and activation == ActivationType.Silu and not isG1U1:
        # pure bf16
        aiter.fmoe(
            moe_buf,
            hidden_states,
            w1,
            w2,
            sorted_ids,
            sorted_weights,
            sorted_expert_ids,
            num_valid_ids,
            topk,
        )
    elif quant_type == QuantType.per_Token and doweight_stage1 and isG1U1:
        a8_type = w1.dtype
        _, model_dim, _ = w2.shape

        a8 = torch.empty((M, model_dim), dtype=a8_type, device=device)
        a8_scale = torch.empty(M, dtype=dtypes.fp32, device=device)
        aiter.dynamic_per_token_scaled_quant(a8, hidden_states, a8_scale)

        aiter.fmoe_g1u1_tkw1(
            moe_buf,
            a8,
            w1,
            w2,
            sorted_ids,
            sorted_weights,
            sorted_expert_ids,
            num_valid_ids,
            topk,
            a8_scale,
            w1_scale,
            w2_scale,
            kernelName,
            a2_scale,
            activation,
        )
    elif quant_type == QuantType.per_1x32 and activation == ActivationType.Swiglu:
        # Swiglu+per_1x32: skip normal quantization, pass activations directly
        if q_dtype_a == dtypes.fp8:
            # fp8 prefill path: convert to fp8 with dummy e8m0 scales
            a1 = hidden_states.to(dtypes.fp8)
            M_sort = sorted_ids.shape[0]
            N = a1.shape[-1]
            a1_scale = torch.ones(
                [M_sort, N // 32], dtype=dtypes.fp8_e8m0, device=a1.device
            )
        else:
            # bf16 decode path: pass bf16 activations directly
            a1 = hidden_states
            a1_scale = None
        E, model_dim, inter_dim = get_inter_dim(w1.shape, w2.shape)
        w1_scale = w1_scale.view(E, -1)
        w2_scale = w2_scale.view(E, -1)
        aiter.fmoe_g1u1(
            moe_buf,
            a1,
            w1,
            w2,
            sorted_ids,
            sorted_weights,
            sorted_expert_ids,
            num_valid_ids,
            topk,
            a1_scale,
            w1_scale,
            w2_scale,
            kernelName,
            fc2_smooth_scale=None,
            activation=activation,
        )
    else:
        quant_func = get_quant(quant_type)
        if hidden_states.dtype != q_dtype_a:
            if quant_type == QuantType.per_1x128:
                quant_func = functools.partial(quant_func, transpose_scale=True)
            a1, a1_scale = quant_func(
                hidden_states,
                scale=a1_scale,
                quant_dtype=q_dtype_a,
                num_rows=num_local_tokens,
            )
        else:
            assert (
                a1_scale is not None or quant_type == QuantType.No
            ), "a1_scale must be provided for quantized input for fused_moe"
            a1 = hidden_states
            if quant_type == QuantType.per_1x128:
                scale_t = torch.empty_like(a1_scale)
                aiter.partial_transpose(scale_t, a1_scale, num_rows=num_local_tokens)
                a1_scale = scale_t

        token_num = hidden_states.shape[0]
        E, model_dim, inter_dim = get_inter_dim(w1.shape, w2.shape)
        if quant_type == QuantType.per_1x32:
            a1_scale = fp4_utils.moe_mxfp4_sort(
                a1_scale,
                sorted_ids,
                num_valid_ids,
                token_num,
                block_size_M,
            )
            w1_scale = w1_scale.view(E, -1)
            w2_scale = w2_scale.view(E, -1)

        if quant_type == QuantType.per_1x128:
            fmoe_func = functools.partial(
                aiter.fmoe_fp8_blockscale_g1u1,
                fc_scale_blkn=128,
                fc_scale_blkk=128,
            )
        elif isG1U1:
            fmoe_func = aiter.fmoe_g1u1
        else:
            aiter.fmoe_int8_g1u0(
                moe_buf,
                a1,
                w1,
                w2,
                sorted_ids,
                sorted_weights,
                sorted_expert_ids,
                num_valid_ids,
                topk,
                a1_scale,
                w1_scale,
                w2_scale,
                fc2_smooth_scale=None,
                activation=activation,
            )
            return moe_buf

        fmoe_func(
            moe_buf,
            a1,
            w1,
            w2,
            sorted_ids,
            sorted_weights,
            sorted_expert_ids,
            num_valid_ids,
            topk,
            a1_scale,
            w1_scale,
            w2_scale,
            kernelName,
            fc2_smooth_scale=None,
            activation=activation,
        )
    return moe_buf


@functools.lru_cache(maxsize=2048)
def get_block_size_M(token, topk, expert, inter_dim):
    cu_num = get_cu_num()
    tileN = 128
    tgN = (inter_dim + tileN - 1) // tileN
    support_list = [32, 64, 128]

    tmp = []
    for el in support_list:
        max_num_tokens = token * topk + expert * el - topk
        tg_num = tgN * (max_num_tokens + el - 1) // el
        rnd = (tg_num + cu_num - 1) // cu_num
        empty = cu_num - tg_num % cu_num
        tmp.append((rnd, empty, el))
    return sorted(tmp, key=lambda x: x[:2])[0][-1]


@functools.lru_cache(maxsize=2048)
def use_nt(token, topk, e):
    use_nt = int(os.environ.get("AITER_USE_NT", "-1"))
    if use_nt != -1:
        return bool(use_nt)
    return (token * topk // e) < 64


@functools.lru_cache(maxsize=2048)
def get_ksplit(token, topk, expert, inter_dim, model_dim):
    aiter_ksplit = int(os.environ.get("AITER_KSPLIT", "0"))
    if aiter_ksplit != 0:
        return aiter_ksplit
    # only for moe_blk gemm1 a8w8 decode scenario
    if token * topk > expert:
        return 0
    cu_num = get_cu_num()
    tileN = 128

    tgM = token * topk  # decode tile num
    tgN = (inter_dim + tileN - 1) // tileN

    tg_num = tgN * tgM
    # if all cu already active
    if tg_num >= cu_num:
        return 0
    tilek = 256
    split_max = (cu_num + tg_num - 1) // tg_num
    # at least split = 2
    for i in reversed(range(2, split_max + 1)):
        if (model_dim % i == 0) and ((model_dim // i) % tilek == 0):
            return i
    return 0


cfg_2stages = None
# fmt: off
fused_moe_1stage_dict = {
    "gfx942":
    {
        # activation,                    quant_type,        dtype,    q_dtype_a,    q_dtype_w,   isG1U1,    doweight_stage1,      API
        (ActivationType.Silu,          QuantType.No,  dtypes.bf16,   dtypes.bf16,   dtypes.bf16,   False,   False) : aiter.fmoe,
        (ActivationType.Silu,          QuantType.No,  dtypes.fp16,   dtypes.fp16,   dtypes.fp16,   False,   False) : aiter.fmoe,
        (ActivationType.Gelu,   QuantType.per_Token,  dtypes.bf16,    dtypes.fp8,   dtypes.i4x2,    True,   False) : aiter.fmoe_g1u1,
        (ActivationType.Silu,    QuantType.per_1x32,  dtypes.bf16,  dtypes.fp4x2,  dtypes.fp4x2,    True,   False) : aiter.fmoe_g1u1,
        (ActivationType.Silu,   QuantType.per_Token,  dtypes.bf16,     dtypes.i8,     dtypes.i8,    True,   False) : aiter.fmoe_g1u1,
        (ActivationType.Gelu,   QuantType.per_Token,  dtypes.bf16,     dtypes.i8,     dtypes.i8,    True,   False) : aiter.fmoe_g1u1,
        (ActivationType.Silu,   QuantType.per_Token,  dtypes.bf16,    dtypes.fp8,    dtypes.fp8,    True,   False) : aiter.fmoe_g1u1,
        (ActivationType.Gelu,   QuantType.per_Token,  dtypes.bf16,    dtypes.fp8,    dtypes.fp8,    True,   False) : aiter.fmoe_g1u1,
        (ActivationType.Silu,   QuantType.per_1x128,  dtypes.bf16,    dtypes.fp8,    dtypes.fp8,    True,   False) : aiter.fmoe_g1u1,
        (ActivationType.Silu,   QuantType.per_Token,  dtypes.bf16,     dtypes.i8,     dtypes.i8,   False,   False) : aiter.fmoe_int8_g1u0,
        (ActivationType.Gelu,   QuantType.per_Token,  dtypes.bf16,     dtypes.i8,     dtypes.i8,   False,   False) : aiter.fmoe_int8_g1u0,
    },
    "gfx950":
    {
        (ActivationType.Silu,    QuantType.per_1x32,   dtypes.bf16,   dtypes.fp4x2,  dtypes.fp4x2,    True,   False) : aiter.fmoe_g1u1,
        (ActivationType.Silu,   QuantType.per_1x128,   dtypes.bf16,     dtypes.fp8,    dtypes.fp8,    True,   False) : aiter.fmoe_fp8_blockscale_g1u1,
        (ActivationType.Silu,   QuantType.per_Token,   dtypes.bf16,    dtypes.bf16,   dtypes.bf16,   False,   False) : aiter.fmoe,
        (ActivationType.Silu,   QuantType.per_Token,   dtypes.bf16,     dtypes.fp8,    dtypes.fp8,    True,   True)  : aiter.fmoe_g1u1_tkw1,
        (ActivationType.Swiglu, QuantType.per_1x32,   dtypes.bf16,   dtypes.bf16,  dtypes.fp4x2,    True,   False) : aiter.fmoe_g1u1,
        (ActivationType.Swiglu, QuantType.per_1x32,   dtypes.bf16,    dtypes.fp8,  dtypes.fp4x2,    True,   False) : aiter.fmoe_g1u1,
    }
}
# fmt: on

quant_remap = {QuantType.per_128x128: QuantType.per_1x128}


def nextPow2(n):
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


def get_padded_M(M):
    padded_m = M
    if M >= 1 and M <= 16:
        # decoding policy may be changed in the future.
        padded_m = nextPow2(padded_m)
    elif M < 1024:
        padded_m = nextPow2(padded_m)
    elif M < 2048:
        padded_m = 1024
    elif M < 16384:
        padded_m = 2048
    else:
        padded_m = 16384
    return padded_m


@dataclass
class MOEMetadata:
    stage1: Callable
    stage2: Callable
    block_m: int
    ksplit: int
    run_1stage: bool = False
    has_bias: bool = False
    use_non_temporal_load: bool = True


@functools.lru_cache(maxsize=2048)
def get_2stage_cfgs(
    token,
    model_dim,
    inter_dim,
    expert,
    topk,
    dtype,
    q_dtype_a,
    q_dtype_w,
    q_type,
    use_g1u1,
    activation,
    doweight_stage1,
    hidden_pad,
    intermediate_pad,
    is_shuffled=True,
):
    def get_cfg_2stages(tune_file):
        import pandas as pd

        cfg_2stages = pd.read_csv(tune_file)
        cfg_2stages = cfg_2stages.set_index(
            [
                "cu_num",
                "token",
                "model_dim",
                "inter_dim",
                "expert",
                "topk",
                "act_type",
                "dtype",
                "q_dtype_a",
                "q_dtype_w",
                "q_type",
                "use_g1u1",
                "doweight_stage1",
            ]
        ).to_dict("index")
        return cfg_2stages

    global cfg_2stages
    config_path = os.path.dirname(AITER_CONFIGS.AITER_CONFIG_FMOE_FILE)
    tune_file = AITER_CONFIGS.AITER_CONFIG_FMOE_FILE
    untune_file = os.path.join(config_path, "untuned_fmoe.csv")
    profile_file = os.path.join(config_path, "profile_fmoe.csv")
    if cfg_2stages is None:
        cfg_2stages = get_cfg_2stages(tune_file)
    cu_num = get_cu_num()
    keys = (
        cu_num,
        token,
        model_dim,
        inter_dim,
        expert,
        topk,
        str(activation),
        str(dtype),
        str(q_dtype_a),
        str(q_dtype_w),
        str(q_type),
        use_g1u1,
        doweight_stage1,
    )

    def MainFunc():
        with open(untune_file, "a") as f:
            if os.path.getsize(untune_file) == 0:
                f.write(
                    "token,model_dim,inter_dim,expert,topk,act_type,dtype,q_dtype_a,q_dtype_w,q_type,use_g1u1,doweight_stage1"
                )
            q_dtype_ws = q_dtype_w if q_dtype_w != torch.uint32 else "torch.int4"
            f.write(
                f"\n{token},{model_dim},{inter_dim},{expert},{topk},{activation},{dtype},{q_dtype_a},{q_dtype_ws},{q_type},{int(use_g1u1)},{int(doweight_stage1)}"
            )
        logger.info("\033[34m Start tuning fmoe")
        os.system(
            f"{PY} {get_asm_dir()}/fmoe_2stages/tune.py -i {untune_file} -o {tune_file} -o2 {profile_file} --last"
        )

    def FinalFunc():
        logger.info(
            f"[Hint] tuned configs are saved in {tune_file}, you can set AITER_CONFIG_FMOE to this file to use tuned configs"
        )
        logger.info("\033[0m")

    def use_cfg():
        problem_type = (activation, dtype, q_dtype_a, q_dtype_w, q_type)
        bypass_type = (
            ActivationType.Silu,
            dtypes.bf16,
            dtypes.fp8,
            dtypes.fp8,
            QuantType.per_1x128,
        )
        if problem_type == bypass_type and (token * topk) <= 128:  # bypass tuned
            aiter.logger.info("bypass tuned results for fp8 blockscale")
            return False
        return True

    # cfg = cfg_2stages.get(keys, None)
    cfg = cfg_2stages.get(keys, None) if cfg_2stages and use_cfg() else None
    if cfg is None and os.environ.get("AITER_ONLINE_TUNE", "0") == "1":
        lock_path = os.path.join(bd_dir, f"lock_fmoe_tune_{keys}")
        mp_lock(lock_path, MainFunc=MainFunc, FinalFunc=FinalFunc)
        cfg_2stages = get_cfg_2stages(tune_file)
        # cfg = cfg_2stages.get(keys, None)
        cfg = cfg_2stages.get(keys, None) if cfg_2stages else None
        if cfg is None:
            logger.warning(f"Fmoe tuning not support for {keys}")
    use_non_temporal_load = False
    if cfg is None or int(os.environ.get("AITER_BYPASS_TUNE_CONFIG", "0")):
        ksplit = 0
        kernelName1 = ""
        kernelName2 = ""
        run_1stage = False
        if (
            activation,
            q_type,
            dtype,
            q_dtype_a,
            q_dtype_w,
            use_g1u1,
            doweight_stage1,
        ) in fused_moe_1stage_dict[get_gfx()]:
            if q_type == QuantType.per_1x128:
                # for fp8 blockscale, ck has better performance so disable assembly kernel
                run_1stage = token > 32 and (inter_dim % 256 == 0)
            elif q_type == QuantType.per_Token and q_dtype_w == dtypes.i8:
                run_1stage = token > 32
            elif q_type == QuantType.per_Token and q_dtype_w == dtypes.fp8:
                run_1stage = token > 16
            elif q_type == QuantType.per_1x32:
                run_1stage = True
            elif q_type != QuantType.per_1x32:
                run_1stage = token < 256

        block_m = (
            BLOCK_SIZE_M
            if run_1stage
            else (
                (64 if token > 32 else 16)
                if q_type == QuantType.per_1x128
                else get_block_size_M(token, topk, expert, inter_dim)
            )
        )
        ksplit = (
            ksplit
            if (run_1stage)
            else (
                get_ksplit(token, topk, expert, inter_dim, model_dim)
                if q_type in [QuantType.per_1x128, QuantType.per_1x32]
                else ksplit
            )
        )
        use_non_temporal_load = use_nt(token, topk, expert)
        aiter.logger.info(
            f"run_1stage = {run_1stage}, ksplit = {ksplit} q_type = {q_type} block_m = {block_m} use_nt = {use_non_temporal_load}, estimated_m_per_expert = {token * topk // expert}"
        )
    else:
        block_m = cfg["block_m"]
        ksplit = cfg["ksplit"]
        kernelName1 = cfg["kernelName1"]
        kernelName2 = cfg["kernelName2"]
        run_1stage = cfg.get("run_1stage", False)

    tag = f"({kernelName1=}, {kernelName2=})"
    logger.info(
        f"[fused_moe] using {'1stage' if run_1stage else '2stage'} {'default' if cfg is None else tag} for {keys} "
    )

    def get_block_m() -> int:
        if q_dtype_a == dtypes.fp8:
            return 32
        else:
            return 16 if token < 2048 else 32 if token < 16384 else 64

    if run_1stage:
        return MOEMetadata(
            functools.partial(
                fused_moe_1stage,
                kernelName=kernelName1,
                activation=activation,
                quant_type=q_type,
            ),
            None,
            block_m,
            ksplit,
            run_1stage,
        )
    if (
        dtype in [dtypes.bf16, dtypes.fp16]
        and q_type == QuantType.per_1x32
        and activation == ActivationType.Swiglu
    ):
        return MOEMetadata(
            functools.partial(
                cktile_moe_stage1,
                n_pad_zeros=intermediate_pad // 64 * 64 * (2 if use_g1u1 else 1),
                k_pad_zeros=hidden_pad // 128 * 128,
                activation=activation,
            ),
            functools.partial(
                cktile_moe_stage2,
                n_pad_zeros=hidden_pad // 64 * 64,
                k_pad_zeros=intermediate_pad // 128 * 128,
                activation=activation,
            ),
            get_block_m(),
            ksplit,
            False,
            True,
        )
    elif (
        dtype in [dtypes.bf16, dtypes.fp16]
        and q_type == QuantType.per_1x32
        and q_dtype_w in [dtypes.fp4x2]
        and ksplit > 1
        and is_shuffled
    ):
        return MOEMetadata(
            functools.partial(
                cktile_moe_stage1,
                n_pad_zeros=intermediate_pad // 64 * 64 * (2 if use_g1u1 else 1),
                k_pad_zeros=hidden_pad // 128 * 128,
                activation=activation,
                split_k=ksplit,
            ),
            functools.partial(
                cktile_moe_stage2,
                n_pad_zeros=hidden_pad // 64 * 64,
                k_pad_zeros=intermediate_pad // 128 * 128,
                activation=activation,
            ),
            16 if token < 2048 else 32 if token < 16384 else 64,
            ksplit,
            run_1stage,
        )

    if (kernelName1 and "ck2stages" in kernelName1) or (
        not kernelName1
        and (
            (q_type == QuantType.per_1x128 and doweight_stage1)
            or q_dtype_w
            in [
                dtypes.bf16,
                dtypes.fp16,
                torch.uint32,
                dtypes.fp4x2,
                dtypes.fp8,
            ]
        )
    ):
        return MOEMetadata(
            functools.partial(
                ck_moe_stage1,
                kernelName=kernelName1,
                activation=activation,
                quant_type=q_type,
                dtype=dtype,
                splitk=ksplit,
                use_non_temporal_load=use_non_temporal_load,
            ),
            functools.partial(
                aiter.ck_moe_stage2_fwd,
                kernelName=kernelName2,
                activation=activation,
                quant_type=q_type,
                use_non_temporal_load=use_non_temporal_load,
            ),
            block_m,
            int(ksplit),
            run_1stage,
        )

    # TODO: remove when stage2 support more size
    tmpList = [16, 32, 64, 128]
    if block_m not in tmpList:
        tag = ""
        block_m = ([el for el in tmpList if block_m < el] + [128])[0]

    return MOEMetadata(
        functools.partial(
            asm_stage1,
            kernelName=kernelName1,
            activation=activation,
            quant_type=q_type,
        ),
        functools.partial(
            aiter.ck_moe_stage2_fwd,
            kernelName=kernelName2,
            activation=activation,
            quant_type=q_type,
        ),
        block_m,
        ksplit,
        run_1stage,
    )


def fused_moe_2stages(
    hidden_states,
    w1,  # [expert(local_expert:EP), inter_dim*2, dim] N,K
    w2,  # [expert(local_expert:EP), dim, inter_dim]
    topk,
    sorted_ids,
    sorted_weights,
    sorted_expert_ids,
    num_valid_ids,
    moe_out,
    isG1U1,
    block_size_M,
    activation=ActivationType.Silu,
    quant_type=QuantType.No,
    doweight_stage1=False,
    # following for quant
    q_dtype_a=None,
    q_dtype_w=None,
    w1_scale=None,  # [expert(local_expert:EP), inter_dim, 1]
    w2_scale=None,  # [expert(local_expert:EP), model_dim, 1]
    a1_scale=None,  # [expert(local_expert:EP), 1, model_dim]
    a2_scale=None,  # [expert(local_expert:EP), 1, inter_dim]
    num_local_tokens: Optional[torch.tensor] = None,
    # following for cktile support
    hidden_pad=0,
    intermediate_pad=0,
    bias1=None,
    bias2=None,
):
    quant_func = get_quant(quant_type)
    token_num_quant_moe_sort_switch = 1024
    token_num, _ = hidden_states.shape
    E, model_dim, inter_dim = get_inter_dim(w1.shape, w2.shape)
    dtype = moe_out.dtype
    device = hidden_states.device
    is_shuffled = getattr(w1, "is_shuffled", False)
    metadata = get_2stage_cfgs(
        get_padded_M(token_num),  # consider token_num > 1024 as prefill
        model_dim,
        inter_dim,
        E,
        topk,
        dtype,
        q_dtype_a,
        q_dtype_w,
        quant_type,
        isG1U1,
        activation,
        doweight_stage1,
        hidden_pad,
        intermediate_pad,
        is_shuffled,
    )
    if (
        quant_type == QuantType.per_1x32
        and dtype in [dtypes.bf16, dtypes.fp16]
        and w1.dtype == dtypes.fp4x2
        and (
            q_dtype_a in [dtypes.bf16, dtypes.fp16]
            and activation == ActivationType.Swiglu
            or (q_dtype_a in [dtypes.fp4x2] and metadata.ksplit > 1 and is_shuffled)
        )
    ):
        a1 = hidden_states.to(dtype)
        a1_scale = None
    elif (
        quant_type == aiter.QuantType.per_1x32
        and dtype in [dtypes.bf16, dtypes.fp16]
        and q_dtype_a == dtypes.fp8
        and w1.dtype == dtypes.fp4x2
        and activation == aiter.ActivationType.Swiglu
    ):
        a1 = hidden_states.to(dtypes.fp8)
        M = sorted_ids.shape[0]
        N = a1.shape[-1]
        a1_scale = torch.ones([M, N // 32], dtype=dtypes.fp8_e8m0, device=a1.device)

    elif quant_type == QuantType.per_1x32:
        if token_num <= token_num_quant_moe_sort_switch:
            a1, a1_scale = fused_dynamic_mxfp4_quant_moe_sort(
                hidden_states,
                sorted_ids=sorted_ids,
                num_valid_ids=num_valid_ids,
                token_num=token_num,
                topk=1,
                block_size=block_size_M,
            )
        else:
            a1, a1_scale = quant_func(
                hidden_states,
                scale=a1_scale,
                quant_dtype=q_dtype_a,
                num_rows=num_local_tokens,
            )
            a1_scale = fp4_utils.moe_mxfp4_sort(
                a1_scale,
                sorted_ids=sorted_ids,
                num_valid_ids=num_valid_ids,
                token_num=token_num,
                block_size=block_size_M,
            )
    elif hidden_states.dtype != q_dtype_a:
        if quant_type == QuantType.per_1x128 and metadata.stage1.func is asm_stage1:
            quant_func = functools.partial(quant_func, transpose_scale=True)
        a1, a1_scale = quant_func(
            hidden_states,
            scale=a1_scale,
            quant_dtype=q_dtype_a,
            num_rows=num_local_tokens,
        )
    else:
        assert (
            a1_scale is not None or quant_type == QuantType.No
        ), "a1_scale must be provided for quantized input for fused_moe"
        a1 = hidden_states
    if quant_type == QuantType.per_1x128 and metadata.stage1.func is asm_stage1:
        ratio = a1_scale.element_size() // a1.element_size()
        a2 = torch.empty(
            (token_num + (token_num * ratio + 127) // 128, topk, inter_dim),
            dtype=q_dtype_a,
            device=device,
        )
    else:
        a2 = torch.empty(
            (token_num, topk, inter_dim),
            dtype=dtype,
            device=device,
        )
    extra_stage1_args = {}
    extra_stage2_args = {}
    if (
        not metadata.run_1stage
        and metadata.has_bias
        and dtype in [dtypes.bf16, dtypes.fp16]
        and quant_type == QuantType.per_1x32
        and activation == ActivationType.Swiglu
    ):
        extra_stage1_args["bias1"] = bias1
        extra_stage2_args["bias2"] = bias2
    a2 = metadata.stage1(
        a1,
        w1,
        w2,
        sorted_ids,
        sorted_expert_ids,
        num_valid_ids,
        a2,
        topk,
        block_m=block_size_M,
        a1_scale=a1_scale,
        w1_scale=(
            w1_scale.view(dtypes.fp8_e8m0) if w1.dtype == dtypes.fp4x2 else w1_scale
        ),
        sorted_weights=sorted_weights if doweight_stage1 else None,
        **extra_stage1_args,
    )
    if (
        quant_type == QuantType.per_1x32
        and dtype in [dtypes.bf16, dtypes.fp16]
        and w1.dtype == dtypes.fp4x2
        and (
            q_dtype_a in [dtypes.bf16, dtypes.fp16]
            and activation == ActivationType.Swiglu
            or (metadata.ksplit > 1 and is_shuffled)
        )
    ):
        a2_scale = None
    elif (
        quant_type == aiter.QuantType.per_1x32
        and dtype in [dtypes.bf16]
        and q_dtype_a == dtypes.fp8
        and w1.dtype == dtypes.fp4x2
        and activation == aiter.ActivationType.Swiglu
    ):
        a2 = a2.to(dtypes.fp8)
        a2_scale = a1_scale
    elif quant_type == QuantType.per_1x32:
        a2 = a2.view(-1, inter_dim)
        if token_num <= token_num_quant_moe_sort_switch:
            a2, a2_scale = fused_dynamic_mxfp4_quant_moe_sort(
                a2,
                sorted_ids=sorted_ids,
                num_valid_ids=num_valid_ids,
                token_num=token_num,
                topk=topk,
                block_size=block_size_M,
            )
        else:
            a2, a2_scale = quant_func(
                a2,
                scale=a2_scale,
                quant_dtype=q_dtype_a,
                num_rows=num_local_tokens,
                num_rows_factor=topk,
            )
            a2_scale = fp4_utils.moe_mxfp4_sort(
                a2_scale[: token_num * topk, :].view(token_num, topk, -1),
                sorted_ids=sorted_ids,
                num_valid_ids=num_valid_ids,
                token_num=token_num,
                block_size=block_size_M,
            )
        a2 = a2.view(token_num, topk, -1)
    elif quant_type == QuantType.per_1x128 and metadata.stage1.func is asm_stage1:
        a2_v = a2[:token_num, :, :]
        a2_scale = (
            a2[token_num:, ...]
            .view(-1)[: token_num * topk * inter_dim * ratio // 128]
            .view(dtypes.fp32)
            .view(token_num, -1)
        )
        a2 = a2_v
    else:
        a2, a2_scale = quant_func(
            a2,
            scale=a2_scale,
            quant_dtype=q_dtype_a,
            num_rows=num_local_tokens,
            num_rows_factor=topk,
        )
        a2 = a2.view(token_num, topk, inter_dim)

    metadata.stage2(
        a2,
        w1,
        w2,
        sorted_ids,
        sorted_expert_ids,
        num_valid_ids,
        moe_out,
        topk,
        w2_scale=(
            w2_scale.view(dtypes.fp8_e8m0) if w2.dtype == dtypes.fp4x2 else w2_scale
        ),
        a2_scale=a2_scale,
        block_m=block_size_M,
        sorted_weights=sorted_weights if not doweight_stage1 else None,
        **extra_stage2_args,
    )

    return moe_out


def torch_moe_act(act_input, torch_act, inter_dim):
    if act_input.shape[-1] == inter_dim:
        return torch_act(act_input)
    else:
        gate, up = act_input.split([inter_dim, inter_dim], dim=-1)
        return torch_act(gate) * up


def asm_stage1(
    input,
    w1,
    w2,
    sorted_ids,
    sorted_expert_ids,
    num_valid_ids,
    out,  # [token_num, topk, inter_dim]
    topk,
    block_m: int,
    kernelName: str = "",
    ksplit: int = 0,
    activation=ActivationType.Silu,
    quant_type=QuantType.No,
    a1_scale=None,
    w1_scale=None,
    sorted_weights=None,
):
    dtype = dtypes.bf16  # out.dtype, asm only support bf16
    if quant_type != QuantType.per_1x128:
        out = out.view(dtype)
    device = out.device
    token_num, _, _ = out.shape
    E, model_dim, inter_dim = get_inter_dim(w1.shape, w2.shape)

    if quant_type == QuantType.per_Tensor:
        a1_scale = a1_scale.view(1, 1).repeat(token_num, 1)
        w1_scale = w1_scale.view(E, 1).repeat(1, w1.shape[1])
        quant_type = QuantType.per_Token

    tmp_out = out
    if ksplit > 0:
        tmp_out = torch.zeros(
            (token_num, topk, w1.shape[1]),
            dtype=dtypes.fp32,
            device=device,
        ).view(dtype)

    aiter.moe_stage1_g1u1(
        input,
        w1,
        w2,
        sorted_ids,
        sorted_expert_ids,
        num_valid_ids,
        tmp_out,
        inter_dim,
        kernelName,
        block_m,
        ksplit=ksplit,
        activation=activation,
        quant_type=quant_type,
        a1_scale=a1_scale,
        w1_scale=w1_scale,
        sorted_weights=sorted_weights,
    )
    if ksplit > 0:
        if activation == ActivationType.Silu:
            aiter.silu_and_mul(out, tmp_out.view(dtypes.fp32))
        else:
            aiter.gelu_and_mul(out, tmp_out.view(dtypes.fp32))
    return out


def torch_moe(
    hidden_states,
    w1,
    w2,
    topk_weight,
    topk_ids,
    # following for int8 quant
    fc1_scale=None,  # [expert(local_expert:EP), inter_dim, 1]
    fc2_scale=None,  # [expert(local_expert:EP), model_dim, 1]
    fc1_smooth_scale=None,  # [expert(local_expert:EP), 1, model_dim]
    fc2_smooth_scale=None,  # [expert(local_expert:EP), 1, inter_dim]
    expert_mask=None,
    activation=ActivationType.Silu,
):
    computeType = dtypes.fp32
    dtype = hidden_states.dtype
    torch_act = aiter.get_torch_act(activation)
    hidden_states = hidden_states.to(computeType)
    w1 = w1.to(computeType)
    w2 = w2.to(computeType)
    B, D = hidden_states.shape
    topk = topk_weight.shape[1]
    if expert_mask is not None:
        local_expert_hash = expert_mask.cumsum(0, dtype=dtypes.i32) - 1
        local_expert_hash[expert_mask == 0] = -1
        topk_ids = local_expert_hash[topk_ids]

    hidden_states = hidden_states.view(B, -1, D).repeat(1, topk, 1)
    out = torch.zeros(
        (B, topk, D),
        dtype=computeType,
        device=hidden_states.device,
    )

    inter_dim = w2.shape[2]

    if fc1_scale is not None:
        # gose to quant D_w8a8/w8a8
        expert = w1.shape[0]
        w2D = w2.shape[-1]
        w1 = (w1.view(-1, D) * fc1_scale.view(-1, 1)).view(expert, -1, D)
        w2 = (w2.view(-1, w2D) * fc2_scale.view(-1, 1)).view(expert, -1, w2D)

    if fc1_smooth_scale is not None:
        expert = fc1_smooth_scale.shape[0]
        fc1_smooth_scale = fc1_smooth_scale.view(expert, -1)
        fc2_smooth_scale = fc2_smooth_scale.view(expert, -1)

    for E_id in range(w1.shape[0]):
        mask = topk_ids == E_id
        if mask.sum():
            sub_tokens = hidden_states[mask]
            if fc1_smooth_scale is not None:
                sub_tokens = sub_tokens * (fc1_smooth_scale[E_id])

            act_input = sub_tokens @ (w1[E_id].transpose(0, 1))
            act_out = torch_moe_act(act_input, torch_act, inter_dim)
            if fc2_smooth_scale is not None:
                act_out = act_out * (fc2_smooth_scale[E_id])
            out[mask] = act_out @ (w2[E_id].transpose(0, 1))

    return (out * topk_weight.view(B, -1, 1)).sum(dim=1).to(dtype)


# temp workaround for swiglu
def swiglu(x_glu, x_linear, alpha: float = 1.702, limit: float = 7.0):
    # Clamp the input values
    x_glu = x_glu.clamp(min=None, max=limit)
    x_linear = x_linear.clamp(min=-limit, max=limit)
    out_glu = x_glu * torch.sigmoid(alpha * x_glu)
    # Note we add an extra bias of 1 to the linear layer
    return out_glu * (x_linear + 1)


def torch_moe_stage1(
    hidden_states,
    w1,  # E, inter_dim*2, model_dim
    w2,  # E, model_dim, inter_dim
    topk_weight,
    topk_ids,
    dtype=dtypes.fp16,
    activation=ActivationType.Silu,
    quant_type=QuantType.No,
    # following for quant
    a1_scale=None,  # [token, 1]
    w1_scale=None,  # [expert, inter_dim, 1]
    w1_bias=None,  # [expert, inter_dim, 1]
    doweight=False,
):
    quant_type = quant_remap.get(quant_type, quant_type)
    ctype = dtypes.fp32  # compute type
    B, D = hidden_states.shape
    topk = topk_weight.shape[1]
    N = w1.shape[1]
    E, model_dim, inter_dim = get_inter_dim(w1.shape, w2.shape)
    if quant_type == QuantType.per_1x32:
        from aiter.utility import fp4_utils

        w1 = fp4_utils.mxfp4_to_f32(w1)
        w1_scale = fp4_utils.e8m0_to_f32(w1_scale)
        if a1_scale is not None:  # skip a16w4
            hidden_states = fp4_utils.mxfp4_to_f32(hidden_states)
            a1_scale = fp4_utils.e8m0_to_f32(a1_scale)
        else:  # a16w4
            hidden_states = hidden_states.to(ctype)

    else:
        hidden_states = hidden_states.to(ctype)
        w1 = w1.to(ctype)

    if quant_type in [QuantType.per_Token, QuantType.per_Tensor]:
        w1 = w1 * w1_scale.view(w1_scale.shape[0], -1, 1)
        hidden_states = hidden_states * a1_scale
    # per_128x128
    elif quant_type in [QuantType.per_128x128, QuantType.per_1x128]:
        w1_shape = w1.shape
        w1 = w1.view(
            w1.shape[0], w1.shape[1] // 128, 128, w1.shape[2] // 128, 128
        ) * w1_scale.view(
            w1_scale.shape[0], w1.shape[1] // 128, 1, w1.shape[2] // 128, 1
        )
        w1 = w1.view(w1_shape)

        a1_scale = a1_scale.view(hidden_states.shape[0], -1, 1)
        a1_scale = a1_scale.repeat(
            1, 1, hidden_states.shape[-1] // a1_scale.shape[1]
        ).view(hidden_states.shape[0], -1)
        hidden_states = hidden_states * a1_scale
    elif quant_type == QuantType.No:
        pass
    elif quant_type == QuantType.per_1x32:
        w1_shape = w1.shape
        w1 = w1.view(E, N, model_dim // 32, 32) * w1_scale.view(
            E, N, model_dim // 32, 1
        )
        w1 = w1.view(w1_shape)

        a1_shape = hidden_states.shape
        hidden_states = hidden_states.view(a1_shape[0], a1_shape[1] // 32, 32)
        if a1_scale is not None:
            a1_scale = a1_scale[: a1_shape[0]]
            hidden_states = hidden_states * a1_scale.view(
                a1_shape[0], a1_shape[1] // 32, 1
            )
        hidden_states = hidden_states.view(a1_shape)
    else:
        assert False, f"Unsupported quant_type: {quant_type}"

    hidden_states = hidden_states.view(B, -1, model_dim).repeat(1, topk, 1)

    out = torch.zeros(
        (B, topk, N),
        dtype=ctype,
        device=hidden_states.device,
    )
    for E_id in range(w1.shape[0]):
        mask = topk_ids == E_id
        if mask.sum():
            sub_tokens = hidden_states[mask]
            act_input = sub_tokens @ (w1[E_id].transpose(0, 1))
            if doweight:
                act_input = act_input * topk_weight[mask].view(-1, 1)
            out[mask] = act_input
            if w1_bias is not None:
                out[mask] = out[mask] + w1_bias[E_id].view(1, -1)
    use_g1u1 = w1.shape[1] == (2 * inter_dim)
    use_swiglu = activation == aiter.ActivationType.Swiglu
    torch_act = aiter.get_torch_act(activation)
    if use_g1u1:
        gate, up = out.split([inter_dim, inter_dim], dim=-1)
        if use_swiglu:
            out = swiglu(gate, up)
        else:
            out = torch_act(gate) * up
    else:
        out = torch_act(out)
    return out.to(dtype)


def torch_moe_stage2(
    hidden_states,
    w1,  # E, inter_dim*2, model_dim
    w2,  # E, model_dim, inter_dim
    topk_weights,
    topk_ids,
    dtype=dtypes.fp16,
    quant_type=QuantType.No,
    w2_scale=None,  # [1]
    a2_scale=None,  # [expert]]'
    w2_bias=None,
    doweight=True,
):
    ctype = dtypes.fp32  # compute type
    E, model_dim, inter_dim = get_inter_dim(w1.shape, w2.shape)
    if quant_type == QuantType.per_1x32:
        from aiter.utility import fp4_utils

        w2 = fp4_utils.mxfp4_to_f32(w2)
        w2_scale = fp4_utils.e8m0_to_f32(w2_scale)
        if a2_scale is not None:
            hidden_states = fp4_utils.mxfp4_to_f32(hidden_states)
            a2_scale = fp4_utils.e8m0_to_f32(a2_scale)
        else:  # a16w4
            hidden_states = hidden_states.to(ctype)
    else:
        hidden_states = hidden_states.to(ctype)
        w2 = w2.to(ctype)

    token_num, topk = topk_ids.shape
    hidden_states = hidden_states.view(token_num, topk, inter_dim)

    if quant_type in [QuantType.per_Token, QuantType.per_Tensor]:
        hidden_states = hidden_states * a2_scale.view(a2_scale.shape[0], -1, 1)
        w2 = w2 * w2_scale.view(w2_scale.shape[0], -1, 1)
    elif quant_type in [QuantType.per_128x128, QuantType.per_1x128]:
        a2_scale = a2_scale.view(hidden_states.shape[0], topk, -1, 1)
        a2_scale = a2_scale.repeat(1, 1, 1, 128).view(hidden_states.shape[0], topk, -1)
        hidden_states = hidden_states * a2_scale

        w2_shape = w2.shape
        w2 = w2.view(
            w2.shape[0], w2.shape[1] // 128, 128, w2.shape[2] // 128, 128
        ) * w2_scale.view(
            w2_scale.shape[0], w2.shape[1] // 128, 1, w2.shape[2] // 128, 1
        )
        w2 = w2.view(w2_shape)
    elif quant_type == QuantType.per_1x32:
        a2_shape = hidden_states.shape
        if a2_scale is not None:
            a2_scale = a2_scale[: a2_shape[0] * topk]
            a2_scale = a2_scale.view(token_num, topk, inter_dim // 32, 1)
            hidden_states = (
                hidden_states.view(token_num, topk, inter_dim // 32, 32) * a2_scale
            )
        hidden_states = hidden_states.view(a2_shape)

        w2_shape = w2.shape
        w2 = w2.view(E, model_dim, inter_dim // 32, 32) * w2_scale.view(
            E, model_dim, inter_dim // 32, 1
        )
        w2 = w2.view(w2_shape)

    out = torch.zeros(
        (token_num, topk, model_dim),
        dtype=ctype,
        device=hidden_states.device,
    )
    for E_id in range(w1.shape[0]):
        mask = topk_ids == E_id
        if mask.sum():
            sub_tokens = hidden_states[mask]
            act_input = sub_tokens @ (w2[E_id].transpose(0, 1))
            out[mask] = act_input
            if w2_bias is not None:
                out[mask] = out[mask] + w2_bias[E_id].view(1, -1)
    if doweight:
        out = out * topk_weights.view(token_num, -1, 1)
    return out.sum(1).to(dtype)


def ck_moe_stage1(
    hidden_states,
    w1,  # [E, inter_dim*2, model_dim]
    w2,  # [E, model_dim, inter_dim]
    sorted_token_ids,  # [max_num_tokens_padded]
    sorted_expert_ids,  # [max_num_m_blocks]
    num_valid_ids,  # [1]
    out,
    topk,
    block_m,
    a1_scale,
    w1_scale,
    kernelName="",
    sorted_weights=None,
    quant_type=aiter.QuantType.No,
    activation=ActivationType.Gelu,
    splitk=1,
    use_non_temporal_load=False,
    dtype=None,
):
    token_num = hidden_states.shape[0]
    is_splitk = quant_type is aiter.QuantType.per_1x128 and splitk > 1
    tmp_out = (
        torch.zeros(
            (token_num, topk, w1.shape[1]), dtype=dtypes.fp32, device=out.device
        )
        if is_splitk
        else out
    )
    aiter.ck_moe_stage1_fwd(
        hidden_states,
        w1,
        w2,
        sorted_token_ids,
        sorted_expert_ids,
        num_valid_ids,
        tmp_out,
        topk,
        kernelName,
        w1_scale,
        a1_scale,
        block_m,
        sorted_weights,
        quant_type,
        activation,
        splitk if is_splitk else 0,
        use_non_temporal_load,
        out.dtype,
    )
    if is_splitk:
        if activation == ActivationType.Silu:
            aiter.silu_and_mul(out, tmp_out.view(dtypes.fp32))
        else:
            aiter.gelu_and_mul(out, tmp_out.view(dtypes.fp32))
    return out


def cktile_moe_stage1(
    hidden_states,
    w1,
    w2,
    sorted_token_ids,
    sorted_expert_ids,
    num_valid_ids,
    out,
    topk,
    block_m,
    a1_scale,
    w1_scale,
    sorted_weights=None,
    n_pad_zeros=0,
    k_pad_zeros=0,
    bias1=None,
    activation=ActivationType.Silu,
    split_k=1,
    dtype=torch.bfloat16,
):
    token_num = hidden_states.shape[0]
    _, n1, k1 = w1.shape
    _, k2, n2 = w2.shape
    D = n2 if k2 == k1 else n2 * 2  # bit4 format
    # max_num_tokens_padded = sorted_expert_ids.shape[0]*block_size

    if w1.dtype is torch.uint32:
        D = D * 8

    out = torch.empty((token_num, topk, D), dtype=dtype, device=hidden_states.device)
    tmp_out = (
        torch.zeros(
            (token_num, topk, w1.shape[1]), dtype=hidden_states.dtype, device=out.device
        )
        if split_k > 1
        else out
    )

    # print("Run cktile_moe_stage1: M=%d, N(N*2)=%d, K=%d, topk=%d, expert=%d"%(token_num, w1.shape[1], hidden_states.shape[1], topk, w1.shape[0]))
    aiter.moe_cktile2stages_gemm1(
        hidden_states,
        w1,
        tmp_out,
        sorted_token_ids,
        sorted_expert_ids,
        num_valid_ids,
        topk,
        n_pad_zeros,
        k_pad_zeros,
        sorted_weights,
        a1_scale,
        w1_scale,
        bias1,
        activation,
        block_m,
        split_k,
    )

    if split_k > 1:
        if activation == ActivationType.Silu:
            aiter.silu_and_mul(out, tmp_out)  # TODO: support fp32 splitk
        else:
            aiter.gelu_and_mul(out, tmp_out)
    return out


def cktile_moe_stage2(
    a2,
    w1,
    w2,
    sorted_token_ids,
    sorted_expert_ids,
    num_valid_ids,
    out,
    topk,
    w2_scale,
    a2_scale,
    block_m,
    activation=ActivationType.Swiglu,
    sorted_weights=None,
    zeros_out=False,
    n_pad_zeros=0,
    k_pad_zeros=0,
    bias2=None,
):
    token_num = a2.shape[0]
    D = w2.shape[1]
    # max_num_tokens_padded = sorted_expert_ids.shape[0]*block_size

    # out = torch.empty(
    #     (token_num, D),
    #     dtype=a2.dtype,
    #     device=a2.device,
    # )
    # if zeros_out:
    #     out.fill_(0)
    # print("Run cktile_moe_stage2: M=%d, N=%d, K=%d, topk=%d, expert=%d"%(a2.shape[0]*a2.shape[1], w2.shape[1], a2.shape[2], topk, w2.shape[0]))
    aiter.moe_cktile2stages_gemm2(
        a2,
        w2,
        out,
        sorted_token_ids,
        sorted_expert_ids,
        num_valid_ids,
        topk,
        n_pad_zeros,
        k_pad_zeros,
        sorted_weights,
        a2_scale,
        w2_scale,
        bias2,
        activation,
        block_m,
    )
    return out


def fused_topk(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    topk_ids: Optional[torch.Tensor] = None,
    topk_weights: Optional[torch.Tensor] = None,
):
    assert hidden_states.shape[0] == gating_output.shape[0], "Number of tokens mismatch"

    M, _ = hidden_states.shape
    expert = gating_output.shape[1]

    token_expert_indicies = torch.empty(
        M, topk, dtype=dtypes.i32, device=hidden_states.device
    )

    if (expert, topk) in [
        (128, 4),
        (128, 6),
        (128, 8),
        (256, 6),
        (256, 8),
    ] and gating_output.dtype in [dtypes.bf16, dtypes.fp32]:
        if topk_weights is None:
            topk_weights = torch.empty(
                (M + 3) // 4 * 4, topk, dtype=dtypes.fp32, device=hidden_states.device
            )
        if topk_ids is None:
            topk_ids = torch.empty(
                (M + 3) // 4 * 4, topk, dtype=dtypes.i32, device=hidden_states.device
            )
        aiter.topk_softmax_asm(
            topk_weights,
            topk_ids,
            token_expert_indicies,
            gating_output,
            renormalize,
        )
        topk_weights = topk_weights[:M, :]
        topk_ids = topk_ids[:M, :]
    else:
        if topk_weights is None:
            topk_weights = torch.empty(
                M, topk, dtype=dtypes.fp32, device=hidden_states.device
            )
        if topk_ids is None:
            topk_ids = torch.empty(
                M, topk, dtype=dtypes.i32, device=hidden_states.device
            )
        aiter.topk_softmax(
            topk_weights,
            topk_ids,
            token_expert_indicies,
            gating_output,
            renormalize,
        )

    del token_expert_indicies  # Not used. Will be used in the future.

    # if renormalize:
    #     topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

    return topk_weights, topk_ids
