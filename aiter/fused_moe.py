# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import functools
import os
from dataclasses import dataclass
from typing import Callable, Optional

import aiter
import torch

# from aiter import get_torch_quant as get_quant
from aiter import ActivationType, QuantType, dtypes
from aiter import get_hip_quant as get_quant
from aiter import logger
from aiter.jit.core import AITER_CONFIGS, AITER_CSRC_DIR, PY, bd_dir, mp_lock
from aiter.jit.utils.chip_info import get_cu_num, get_gfx
from aiter.jit.utils.torch_guard import torch_compile_guard
from aiter.ops.flydsl.utils import is_flydsl_available
from aiter import fused_dynamic_mxfp4_quant_moe_sort, mxfp4_moe_sort_fwd

BLOCK_SIZE_M = 32

_USE_OPUS_MOE_SORTING = os.environ.get("AITER_USE_OPUS_MOE_SORTING", "0") == "1"


def _moe_sorting_impl(
    topk_ids,
    topk_weights,
    num_experts,
    model_dim,
    moebuf_dtype,
    block_size,
    expert_mask,
    num_local_tokens,
    dispatch_policy,
    use_opus,
):
    device = topk_ids.device
    M, topk = topk_ids.shape
    max_num_tokens_padded = int(topk_ids.numel() + num_experts * block_size - topk)

    max_num_m_blocks = int((max_num_tokens_padded + block_size - 1) // block_size)
    sorted_ids = torch.empty(max_num_tokens_padded, dtype=dtypes.i32, device=device)
    sorted_weights = torch.empty(
        max_num_tokens_padded, dtype=dtypes.fp32, device=device
    )
    sorted_expert_ids = torch.empty(max_num_m_blocks, dtype=dtypes.i32, device=device)
    num_valid_ids = torch.empty(2, dtype=dtypes.i32, device=device)
    moe_buf = torch.empty((M, model_dim), dtype=moebuf_dtype, device=device)

    fwd_fn = aiter.moe_sorting_opus_fwd if use_opus else aiter.moe_sorting_fwd
    fwd_fn(
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
        return _moe_sorting_impl(
            topk_ids,
            topk_weights,
            num_experts,
            model_dim,
            moebuf_dtype,
            block_size,
            expert_mask,
            num_local_tokens,
            dispatch_policy,
            use_opus=_USE_OPUS_MOE_SORTING,
        )
    except Exception as e:
        logger.error(f"Error in moe_sorting: {e}")
        max_num_tokens_padded = int(
            topk_ids.numel() + num_experts * block_size - topk_ids.shape[1]
        )
        topk = topk_ids.shape[1]
        logger.error(
            f"Moe_sorting info: {max_num_tokens_padded=} {block_size=} {num_experts=} {topk=} {topk_ids.shape=}"
        )
        raise e


# Lru cache will using hash to create key, which makes error when w1,w2 shape is symint.
# We can use torch.compile(dynamic=False) to avoid
@functools.lru_cache(maxsize=2048)
def get_inter_dim(w1_shape, w2_shape, q_dtype_w=None):
    E, _, model_dim = w1_shape
    E, model_dim, inter_dim = w2_shape

    if q_dtype_w is None:
        int4_war = model_dim // w1_shape[-1]
    elif q_dtype_w == dtypes.fp4x2:
        int4_war = 2
    elif q_dtype_w == torch.uint32:
        int4_war = model_dim // w1_shape[-1]
    else:
        int4_war = 1
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
    q_type2: Optional[QuantType] = None,
    q_dtype_a2: Optional[torch.dtype] = None,
    q_dtype_w2: Optional[torch.dtype] = None,
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
        q_type2=q_type2.value if isinstance(q_type2, QuantType) else q_type2,
        q_dtype_a2=q_dtype_a2,
        q_dtype_w2=q_dtype_w2,
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
    q_type2: Optional[int] = None,
    q_dtype_a2: Optional[torch.dtype] = None,
    q_dtype_w2: Optional[torch.dtype] = None,
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
    q_type2: Optional[int] = None,
    q_dtype_a2: Optional[torch.dtype] = None,
    q_dtype_w2: Optional[torch.dtype] = None,
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
    q_type2 = quant_type if q_type2 is None else QuantType(q_type2)
    if block_size_M == -1:
        block_size_M = None
    """user API"""
    M, topk = topk_ids.shape
    q_dtype_w2 = w2.dtype if q_dtype_w2 is None else q_dtype_w2
    E, model_dim, inter_dim = get_inter_dim(w1.shape, w2.shape, q_dtype_w=q_dtype_w2)

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
    q_type2 = quant_remap.get(q_type2, q_type2)
    q_dtype_w = w1.dtype
    q_dtype_a = w1.dtype if w1.dtype != torch.uint32 else dtypes.fp8
    # If input is already FP8-quantized (e.g. from FP8 dispatch) with block scale,
    # use FP8 as activation dtype to skip redundant re-quantization
    if (
        quant_type == QuantType.per_1x128
        and hidden_states.dtype == dtypes.fp8
        and a1_scale is not None
    ):
        q_dtype_a = dtypes.fp8
    bf16_fp8_bound = 512
    if quant_type == QuantType.per_1x32:
        if activation == ActivationType.Swiglu:
            if get_gfx() != "gfx950" or M < bf16_fp8_bound:
                q_dtype_a = dtypes.bf16
            elif M >= bf16_fp8_bound:
                q_dtype_a = dtypes.fp8
        else:
            q_dtype_a = dtypes.fp4x2
    q_dtype_a2 = q_dtype_a if q_dtype_a2 is None else q_dtype_a2

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
        q_dtype_a2=q_dtype_a2,
        q_dtype_w2=q_dtype_w2,
        q_type2=q_type2,
    )

    if metadata.run_1stage:
        block_size_M1 = metadata.block_m if block_size_M is None else block_size_M
        block_size_M2 = block_size_M1
    else:
        if block_size_M is None:
            block_size_M1 = metadata.block_m
            block_size_M2 = metadata.block_m2
        else:
            block_size_M1 = block_size_M
            block_size_M2 = block_size_M

    block_size_M1 = int(block_size_M1)
    block_size_M2 = int(block_size_M2)

    if metadata.run_1stage or block_size_M1 == block_size_M2:
        sorted_ids1, sorted_weights1, sorted_expert_ids1, num_valid_ids1, moe_buf = (
            moe_sorting(
                topk_ids,
                topk_weight,
                global_E,
                model_dim,
                dtype,
                block_size_M1,
                expert_mask,
                num_local_tokens,
                moe_sorting_dispatch_policy,
            )
        )
        sorted_ids2 = sorted_ids1
        sorted_weights2 = sorted_weights1
        sorted_expert_ids2 = sorted_expert_ids1
        num_valid_ids2 = num_valid_ids1
    else:
        sorted_ids1, sorted_weights1, sorted_expert_ids1, num_valid_ids1, _ = moe_sorting(
            topk_ids,
            topk_weight,
            global_E,
            model_dim,
            dtype,
            block_size_M1,
            expert_mask,
            num_local_tokens,
            moe_sorting_dispatch_policy,
        )
        sorted_ids2, sorted_weights2, sorted_expert_ids2, num_valid_ids2, moe_buf = (
            moe_sorting(
                topk_ids,
                topk_weight,
                global_E,
                model_dim,
                dtype,
                block_size_M2,
                expert_mask,
                num_local_tokens,
                moe_sorting_dispatch_policy,
            )
        )
        # Different block_m can legitimately produce different padded valid-id
        # counts, so do not enforce equality across the two sorting passes.
        if int(num_valid_ids1[0].item()) != int(num_valid_ids2[0].item()):
            pass
            # logger.warning(
            #     f"[fused_moe] dual-sorting valid-id counts differ with "
            #     f"block_m(stage1)={block_size_M1}, block_m2(stage2)={block_size_M2}: "
            #     f"{int(num_valid_ids1[0].item())} vs {int(num_valid_ids2[0].item())}"
            # )

    if metadata.run_1stage:
        return metadata.stage1(
            hidden_states,
            w1,
            w2,
            topk,
            sorted_ids1,
            sorted_weights1,
            sorted_expert_ids1,
            num_valid_ids1,
            moe_buf,
            isG1U1,
            block_size_M1,
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
            sorted_ids1,
            sorted_weights1,
            sorted_expert_ids1,
            num_valid_ids1,
            moe_buf,
            isG1U1,
            block_size_M1,
            activation=activation,
            quant_type=quant_type,
            doweight_stage1=doweight_stage1,
            q_dtype_a=q_dtype_a,
            q_dtype_w=q_dtype_w,
            q_type2=q_type2,
            q_dtype_a2=q_dtype_a2,
            q_dtype_w2=q_dtype_w2,
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            a1_scale=a1_scale,
            a2_scale=a2_scale,
            sorted_ids2=sorted_ids2,
            sorted_weights2=sorted_weights2,
            sorted_expert_ids2=sorted_expert_ids2,
            num_valid_ids2=num_valid_ids2,
            block_size_M2=block_size_M2,
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
        E, model_dim, inter_dim = get_inter_dim(w1.shape, w2.shape, q_dtype_w=w2.dtype)
        if quant_type == QuantType.per_1x32:
            a1_scale = mxfp4_moe_sort_fwd(
                a1_scale,
                sorted_ids=sorted_ids,
                num_valid_ids=num_valid_ids,
                token_num=token_num,
                cols=model_dim,
            )
            w1_scale = w1_scale.view(E, -1)
            w2_scale = w2_scale.view(E, -1)

        if quant_type == QuantType.per_1x128:
            fmoe_func = functools.partial(
                aiter.fmoe_fp8_blockscale_g1u1,
                fc_scale_blkn=128,
                fc_scale_blkk=128,
                block_size_M=block_size_M,
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
        (ActivationType.Gelu,   QuantType.per_1x128,   dtypes.bf16,     dtypes.fp8,    dtypes.fp8,    True,   False) : aiter.fmoe_fp8_blockscale_g1u1,
        (ActivationType.Silu,   QuantType.per_Token,   dtypes.bf16,    dtypes.bf16,   dtypes.bf16,   False,   False) : aiter.fmoe,
        (ActivationType.Silu,   QuantType.per_Token,   dtypes.bf16,     dtypes.fp8,    dtypes.fp8,    True,   True)  : aiter.fmoe_g1u1_tkw1,
        (ActivationType.Silu,   QuantType.per_Token,   dtypes.bf16,     dtypes.fp8,    dtypes.fp8,    True,   False) : aiter.fmoe_g1u1,
        (ActivationType.Gelu,   QuantType.per_Token,   dtypes.bf16,     dtypes.fp8,    dtypes.fp8,    True,   False) : aiter.fmoe_g1u1,
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
    # Keep compatibility with historical tuning keys for this shape.
    if padded_m == 20480:
        return 20480
    if M < 32768:
        padded_m = nextPow2(padded_m)
    else:
        padded_m = 32768
    return padded_m


@dataclass
class MOEMetadata:
    stage1: Callable
    stage2: Callable
    block_m: int
    ksplit: int
    block_m2: Optional[int] = None
    run_1stage: bool = False
    has_bias: bool = False
    use_non_temporal_load: bool = True
    fuse_fp4_quant: bool = False

    def __post_init__(self):
        if self.block_m2 is None:
            self.block_m2 = self.block_m


def _flydsl_stage1_wrapper(
    hidden_states,
    w1,
    w2,
    sorted_token_ids,
    sorted_expert_ids,
    num_valid_ids,
    out,
    topk,
    kernelName="",
    activation=ActivationType.Silu,
    w1_scale=None,
    a1_scale=None,
    sorted_weights=None,
    fuse_fp4_quant=False,
    fuse_sort_scale=False,
    **_kwargs,
):
    parsed = aiter.ops.flydsl.moe_kernels.get_flydsl_kernel_params(kernelName)
    if parsed is None:
        raise ValueError(f"Invalid FlyDSL kernel name: {kernelName}")
    _, _, inter_dim = get_inter_dim(w1.shape, w2.shape, q_dtype_w=w2.dtype)
    use_g1u1 = w1.shape[1] == (2 * inter_dim)
    if activation == ActivationType.Swiglu:
        act = "swiglu"
    elif activation == ActivationType.Gelu:
        act = "gelu"
    else:
        act = "silu"
    _fq = fuse_fp4_quant or parsed.get("fuse_fp4_quant", False)
    _fss = fuse_sort_scale or (_fq and not fuse_sort_scale)
    return aiter.ops.flydsl.flydsl_moe_stage1(
        a=hidden_states,
        w1=w1,
        sorted_token_ids=sorted_token_ids,
        sorted_expert_ids=sorted_expert_ids,
        num_valid_ids=num_valid_ids,
        out=out,
        topk=topk,
        tile_m=parsed["tile_m"],
        tile_n=parsed["tile_n"],
        tile_k=parsed["tile_k"],
        a_dtype=parsed["a_dtype"],
        b_dtype=parsed["b_dtype"],
        out_dtype=parsed["out_dtype"],
        act=act,
        use_g1u1=use_g1u1,
        w1_scale=w1_scale,
        a1_scale=a1_scale,
        sorted_weights=sorted_weights,
        fuse_fp4_quant=_fq,
        fuse_sort_scale=_fss,
        use_async_copy=parsed.get("use_async_copy", False),
        k_batch=parsed.get("k_batch", 1),
        waves_per_eu=parsed.get("waves_per_eu", 3),
        b_nt=parsed.get("b_nt", 2),
        gate_only=parsed.get("gate_only", False),
    )


def _flydsl_stage2_wrapper(
    inter_states,
    w1,
    w2,
    sorted_token_ids,
    sorted_expert_ids,
    num_valid_ids,
    out,
    topk,
    kernelName="",
    w2_scale=None,
    a2_scale=None,
    sorted_weights=None,
    **_kwargs,
):

    parsed = aiter.ops.flydsl.moe_kernels.get_flydsl_kernel_params(kernelName)
    if parsed is None:
        raise ValueError(f"Invalid FlyDSL kernel name: {kernelName}")
    return aiter.ops.flydsl.flydsl_moe_stage2(
        inter_states=inter_states,
        w2=w2,
        sorted_token_ids=sorted_token_ids,
        sorted_expert_ids=sorted_expert_ids,
        num_valid_ids=num_valid_ids,
        out=out,
        topk=topk,
        tile_m=parsed["tile_m"],
        tile_n=parsed["tile_n"],
        tile_k=parsed["tile_k"],
        a_dtype=parsed["a_dtype"],
        b_dtype=parsed["b_dtype"],
        out_dtype=parsed["out_dtype"],
        mode=parsed.get("mode", "atomic"),
        w2_scale=w2_scale,
        a2_scale=a2_scale,
        sorted_weights=sorted_weights,
        sort_block_m=parsed.get("sort_block_m", 0),
        use_async_copy=parsed.get("use_async_copy", False),
        waves_per_eu=parsed.get("waves_per_eu", 3),
        b_nt=parsed.get("b_nt", 2),
        mfma_variant=parsed.get("mfma_variant", None),
        # Keep stage2 persist behavior aligned with kernel naming.
        # For migrated old kernels (non `_persist` names), force legacy non-persistent path.
        persist=parsed.get("persist", False),
    )


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
    q_dtype_a2=None,
    q_dtype_w2=None,
    q_type2=None,
):
    # Hybrid quant: stage2 triple defaults to stage1's when not provided.
    if q_dtype_a2 is None:
        q_dtype_a2 = q_dtype_a
    if q_dtype_w2 is None:
        q_dtype_w2 = q_dtype_w
    if q_type2 is None:
        q_type2 = q_type

    _INDEX_COLS = [
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
        "q_dtype_a2",
        "q_dtype_w2",
        "q_type2",
    ]

    def _backfill_stage2_cols(df):
        """Old csv has no _2 columns -> stage2 triple equals stage1 triple."""
        for src, dst in (
            ("q_dtype_a", "q_dtype_a2"),
            ("q_dtype_w", "q_dtype_w2"),
            ("q_type", "q_type2"),
        ):
            if dst not in df.columns:
                df[dst] = df[src]
            else:
                df[dst] = df[dst].fillna(df[src])
        return df

    def get_cfg_2stages(tune_file):
        import pandas as pd

        df = pd.read_csv(tune_file)
        if "_tag" in df.columns:
            df = df[df["_tag"].fillna("") == ""]
        df = _backfill_stage2_cols(df)
        df = df.set_index(_INDEX_COLS).to_dict("index")
        return df

    _flydsl_fallback_cache = {}

    def get_flydsl_fallback_cfgs(tune_file):
        """Return fallback configs (rows tagged ``flydsl_fallback``)."""
        if tune_file in _flydsl_fallback_cache:
            return _flydsl_fallback_cache[tune_file]
        import pandas as pd

        if not os.path.exists(tune_file):
            _flydsl_fallback_cache[tune_file] = {}
            return {}
        df = pd.read_csv(tune_file)
        if "_tag" not in df.columns:
            _flydsl_fallback_cache[tune_file] = {}
            return {}
        fb_df = df[df["_tag"] == "flydsl_fallback"]
        if fb_df.empty:
            _flydsl_fallback_cache[tune_file] = {}
            return {}
        fb_df = _backfill_stage2_cols(fb_df.copy())
        result = fb_df.set_index(_INDEX_COLS).to_dict("index")
        _flydsl_fallback_cache[tune_file] = result
        return result

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
        str(q_dtype_a2),
        str(q_dtype_w2),
        str(q_type2),
    )

    def MainFunc():
        # Detect whether the existing file already has the hybrid _2 columns.
        # Old files only carry 12 base columns; appending 15-col rows would
        # break pandas. We honor the existing header so legacy files keep
        # working when hybrid mode is not requested.
        is_hybrid = (
            (str(q_dtype_a2), str(q_dtype_w2), str(q_type2))
            != (str(q_dtype_a), str(q_dtype_w), str(q_type))
        )
        has_hybrid_header = False
        if os.path.getsize(untune_file) > 0:
            with open(untune_file, "r") as fr:
                first_line = fr.readline().strip()
            has_hybrid_header = "q_type2" in first_line.split(",")
        with open(untune_file, "a") as f:
            if os.path.getsize(untune_file) == 0:
                f.write(
                    "token,model_dim,inter_dim,expert,topk,act_type,dtype,q_dtype_a,q_dtype_w,q_type,use_g1u1,doweight_stage1,q_dtype_a2,q_dtype_w2,q_type2"
                )
                has_hybrid_header = True
            q_dtype_ws = q_dtype_w if q_dtype_w != torch.uint32 else "torch.int4"
            q_dtype_w2s = q_dtype_w2 if q_dtype_w2 != torch.uint32 else "torch.int4"
            base_row = f"\n{token},{model_dim},{inter_dim},{expert},{topk},{activation},{dtype},{q_dtype_a},{q_dtype_ws},{q_type},{int(use_g1u1)},{int(doweight_stage1)}"
            if has_hybrid_header:
                f.write(
                    f"{base_row},{q_dtype_a2},{q_dtype_w2s},{q_type2}"
                )
            else:
                if is_hybrid:
                    raise RuntimeError(
                        f"hybrid quant requested ({q_dtype_a}->{q_dtype_a2}, "
                        f"{q_dtype_w}->{q_dtype_w2}, {q_type}->{q_type2}) but "
                        f"{untune_file} uses legacy header without _2 columns. "
                        "Migrate the file or remove it to enable hybrid mode."
                    )
                f.write(base_row)
        logger.info("\033[34m Start tuning fmoe")
        os.system(
            f"{PY} {AITER_CSRC_DIR}/ck_gemm_moe_2stages_codegen/gemm_moe_tune.py -i {untune_file} -o {tune_file} -o2 {profile_file} --last"
        )

    def FinalFunc():
        logger.info(
            f"[Hint] tuned configs are saved in {tune_file}, you can set AITER_CONFIG_FMOE to this file to use tuned configs"
        )
        logger.info("\033[0m")

    cfg = cfg_2stages.get(keys, None) if cfg_2stages else None
    if cfg is None and os.environ.get("AITER_ONLINE_TUNE", "0") == "1":
        lock_path = os.path.join(bd_dir, f"lock_fmoe_tune_{keys}")
        mp_lock(lock_path, MainFunc=MainFunc, FinalFunc=FinalFunc)
        cfg_2stages = get_cfg_2stages(tune_file)
        cfg = cfg_2stages.get(keys, None) if cfg_2stages else None
        if cfg is None:
            logger.warning(f"Fmoe tuning not support for {keys}")
    if cfg is not None and not is_flydsl_available():
        kn1 = str(cfg.get("kernelName1", ""))
        kn2 = str(cfg.get("kernelName2", ""))
        if kn1.startswith("flydsl_") or kn2.startswith("flydsl_"):
            fallback_cfgs = get_flydsl_fallback_cfgs(tune_file)
            fallback = fallback_cfgs.get(keys)
            if fallback is not None:
                cfg = fallback
                logger.info(
                    f"[fused_moe] flydsl unavailable, using fallback config for {keys}"
                )
            else:
                cfg = None
                logger.warning(
                    f"[fused_moe] flydsl unavailable and no fallback for {keys}, "
                    "using default heuristics"
                )

    def _is_missing_number(value):
        # CSV values may come in as None/NaN; both should fall back to defaults.
        return value is None or (isinstance(value, float) and value != value)

    use_non_temporal_load = False
    block_m2 = None
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
                run_1stage = token > 32 and (inter_dim % 128 == 0)
            elif q_type == QuantType.per_Token and q_dtype_w == dtypes.i8:
                run_1stage = token > 32
            elif q_type == QuantType.per_Token and q_dtype_w == dtypes.fp8:
                run_1stage = token > 16 or inter_dim % 128 != 0
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
        block_m2 = int(block_m)
        use_non_temporal_load = use_nt(token, topk, expert)
        aiter.logger.info(
            f"run_1stage = {run_1stage}, ksplit = {ksplit} q_type = {q_type} block_m = {block_m} use_nt = {use_non_temporal_load}, estimated_m_per_expert = {token * topk // expert}"
        )
    else:
        cfg_block_m = cfg.get("block_m", BLOCK_SIZE_M)
        if _is_missing_number(cfg_block_m):
            cfg_block_m = BLOCK_SIZE_M
        block_m2 = cfg.get("block_m2", cfg_block_m)
        if _is_missing_number(block_m2):
            block_m2 = cfg_block_m
        block_m = int(cfg_block_m)
        block_m2 = int(block_m2)
        if int(os.environ.get("AITER_KSPLIT", "0")) != -1:
            ksplit = cfg.get("ksplit1", cfg.get("ksplit", 0))
        else:
            ksplit = 0
        kernelName1 = cfg["kernelName1"]
        kernelName2 = cfg["kernelName2"]
        run_1stage = cfg.get("run_1stage", False)
        if not is_shuffled and not run_1stage:
            logger.warning(
                f"[fused_moe] tuned config found for {keys} but is_shuffled=False. "
                "Tuned kernels are optimized for preshuffled weights (preshuffle_on). "
                "Running with preshuffle_off may produce incorrect results."
            )

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
        # never hard code block_m for 1-stage since it can be tuned by kernel itself, and we have different heuristics for different quant types
        # # TODO: enable this approach for other quant types and archs
        # if q_type == QuantType.per_1x128 and get_gfx() == "gfx950":
        #     tkn_per_epr = token * topk // expert
        #     block_m = 64 if tkn_per_epr > 32 else block_m
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
            block_m2=block_m2,
            run_1stage=run_1stage,
        )
    is_flydsl1 = bool(kernelName1) and kernelName1.startswith("flydsl_")
    is_flydsl2 = bool(kernelName2) and kernelName2.startswith("flydsl_")
    if (is_flydsl1 or is_flydsl2) and is_flydsl_available():
        _s1_fq = is_flydsl1 and "_fq" in kernelName1
        if is_flydsl1:
            stage1_func = functools.partial(
                _flydsl_stage1_wrapper,
                kernelName=kernelName1,
                activation=activation,
            )
        else:
            stage1_func = functools.partial(
                ck_moe_stage1,
                kernelName=kernelName1,
                activation=activation,
                quant_type=q_type,
                dtype=dtype,
                splitk=ksplit,
                use_non_temporal_load=use_non_temporal_load,
            )

        if is_flydsl2:
            stage2_func = functools.partial(
                _flydsl_stage2_wrapper,
                kernelName=kernelName2,
            )
        else:
            stage2_func = functools.partial(
                aiter.ck_moe_stage2_fwd,
                kernelName=kernelName2,
                activation=activation,
                quant_type=q_type2,
                use_non_temporal_load=use_non_temporal_load,
            )

        return MOEMetadata(
            stage1_func,
            stage2_func,
            block_m,
            int(ksplit),
            block_m2=block_m2,
            run_1stage=run_1stage,
            fuse_fp4_quant=_s1_fq and q_type2 == QuantType.per_1x32,
        )
    if (
        dtype in [dtypes.bf16, dtypes.fp16]
        and q_type == QuantType.per_1x32
        and activation == ActivationType.Swiglu
    ):
        _bm_cktile = get_block_m()
        return MOEMetadata(
            functools.partial(
                cktile_moe_stage1,
                n_pad_zeros=intermediate_pad // 64 * 64 * (2 if use_g1u1 else 1),
                k_pad_zeros=hidden_pad // 128 * 128,
                activation=activation,
                split_k=max(ksplit, 1),
            ),
            functools.partial(
                cktile_moe_stage2,
                n_pad_zeros=hidden_pad // 64 * 64,
                k_pad_zeros=intermediate_pad // 128 * 128,
                activation=activation,
            ),
            _bm_cktile,
            ksplit,
            block_m2=_bm_cktile,
            run_1stage=False,
            has_bias=True,
        )
    elif (
        dtype in [dtypes.bf16, dtypes.fp16]
        and q_type == QuantType.per_1x32
        and q_dtype_w in [dtypes.fp4x2]
        and ksplit > 1
        and is_shuffled
    ):
        _bm_cktile = 16 if token < 2048 else 32 if token < 16384 else 64
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
            _bm_cktile,
            ksplit,
            block_m2=_bm_cktile,
            run_1stage=run_1stage,
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
        if kernelName2 and kernelName2.startswith("flydsl_") and is_flydsl_available():
            stage2_func = functools.partial(
                _flydsl_stage2_wrapper,
                kernelName=kernelName2,
            )
        else:
            stage2_func = functools.partial(
                aiter.ck_moe_stage2_fwd,
                kernelName=kernelName2,
                activation=activation,
                quant_type=q_type2,
                use_non_temporal_load=use_non_temporal_load,
            )
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
            stage2_func,
            block_m,
            int(ksplit),
            block_m2=block_m2,
            run_1stage=run_1stage,
        )

    # TODO: remove when stage2 support more size
    tmpList = [16, 32, 64, 128]
    if block_m not in tmpList:
        tag = ""
        block_m = ([el for el in tmpList if block_m < el] + [128])[0]
        block_m2 = block_m

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
            quant_type=q_type2,
        ),
        block_m,
        ksplit,
        block_m2=block_m2,
        run_1stage=run_1stage,
    )


def _quantize_stage2_per_1x32(
    a2,
    a2_scale,
    quant_func2,
    q_dtype_a2,
    sorted_ids2,
    num_valid_ids2,
    token_num,
    topk,
    inter_dim,
    num_local_tokens,
    block_size_M2,
    is_flydsl_stage2,
):
    a2 = a2.view(-1, inter_dim)
    if is_flydsl_stage2 and q_dtype_a2 == dtypes.fp4x2:
        # FlyDSL stage2 loads A2 by original token/topk row, but its scale
        # buffer follows the sorted MoE row order.
        a2, a2_scale_unsorted = quant_func2(
            a2,
            scale=a2_scale,
            quant_dtype=q_dtype_a2,
            num_rows=num_local_tokens,
            num_rows_factor=topk,
        )
        a2_scale = mxfp4_moe_sort_fwd(
            a2_scale_unsorted,
            sorted_ids=sorted_ids2,
            num_valid_ids=num_valid_ids2,
            token_num=token_num,
            cols=inter_dim,
        )
    else:
        a2, a2_scale = fused_dynamic_mxfp4_quant_moe_sort(
            a2,
            sorted_ids=sorted_ids2,
            num_valid_ids=num_valid_ids2,
            token_num=token_num,
            topk=topk,
            block_size=block_size_M2,
            num_rows=num_local_tokens,
        )
    return a2.view(token_num, topk, -1), a2_scale


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
    q_type2=None,
    q_dtype_a2=None,
    q_dtype_w2=None,
    w1_scale=None,  # [expert(local_expert:EP), inter_dim, 1]
    w2_scale=None,  # [expert(local_expert:EP), model_dim, 1]
    a1_scale=None,  # [expert(local_expert:EP), 1, model_dim]
    a2_scale=None,  # [expert(local_expert:EP), 1, inter_dim]
    sorted_ids2=None,
    sorted_weights2=None,
    sorted_expert_ids2=None,
    num_valid_ids2=None,
    block_size_M2=None,
    num_local_tokens: Optional[torch.tensor] = None,
    # following for cktile support
    hidden_pad=0,
    intermediate_pad=0,
    bias1=None,
    bias2=None,
):
    quant_func = get_quant(quant_type)
    q_type2 = quant_type if q_type2 is None else QuantType(q_type2)
    q_type2 = quant_remap.get(q_type2, q_type2)
    quant_func2 = get_quant(q_type2)
    q_dtype_a2 = q_dtype_a if q_dtype_a2 is None else q_dtype_a2
    q_dtype_w2 = w2.dtype if q_dtype_w2 is None else q_dtype_w2
    token_num, _ = hidden_states.shape
    if sorted_ids2 is None:
        sorted_ids2 = sorted_ids
    if sorted_weights2 is None:
        sorted_weights2 = sorted_weights
    if sorted_expert_ids2 is None:
        sorted_expert_ids2 = sorted_expert_ids
    if num_valid_ids2 is None:
        num_valid_ids2 = num_valid_ids
    if block_size_M2 is None:
        block_size_M2 = block_size_M
    E, model_dim, inter_dim = get_inter_dim(w1.shape, w2.shape, q_dtype_w=q_dtype_w2)
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
        q_dtype_a2=q_dtype_a2,
        q_dtype_w2=q_dtype_w2,
        q_type2=q_type2,
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
        if hidden_states.dtype == dtypes.fp4x2 and a1_scale is not None:
            # Input is already quantized to fp4x2 (e.g., from FP4 dispatch),
            # skip re-quantization, only sort the scale
            a1 = hidden_states
            a1_scale = mxfp4_moe_sort_fwd(
                a1_scale,
                sorted_ids=sorted_ids,
                num_valid_ids=num_valid_ids,
                token_num=token_num,
                cols=model_dim,
            )
        else:
            a1, a1_scale = fused_dynamic_mxfp4_quant_moe_sort(
                hidden_states,
                sorted_ids=sorted_ids,
                num_valid_ids=num_valid_ids,
                token_num=token_num,
                topk=topk,
                block_size=block_size_M,
                num_rows=num_local_tokens,
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
        None if metadata.fuse_fp4_quant else a2,
        topk,
        block_m=block_size_M,
        a1_scale=a1_scale,
        w1_scale=(
            w1_scale.view(dtypes.fp8_e8m0) if w1.dtype == dtypes.fp4x2 else w1_scale
        ),
        sorted_weights=sorted_weights if doweight_stage1 else None,
        **extra_stage1_args,
    )
    if metadata.fuse_fp4_quant and isinstance(a2, tuple):
        a2_raw, a2_scale = a2[0], a2[1]
        _fp4_bytes = token_num * topk * (inter_dim // 2)
        a2 = (
            a2_raw.view(-1)
            .view(torch.uint8)[:_fp4_bytes]
            .view(dtypes.fp4x2)
            .reshape(token_num, topk, -1)
        )
    elif (
        q_type2 == QuantType.per_1x32
        and dtype in [dtypes.bf16, dtypes.fp16]
        and w1.dtype == dtypes.fp4x2
        and (
            q_dtype_a2 in [dtypes.bf16, dtypes.fp16]
            and activation == ActivationType.Swiglu
            or (metadata.ksplit > 1 and is_shuffled)
        )
    ):
        a2_scale = None
    elif (
        q_type2 == aiter.QuantType.per_1x32
        and dtype in [dtypes.bf16]
        and q_dtype_a2 == dtypes.fp8
        and w1.dtype == dtypes.fp4x2
        and activation == aiter.ActivationType.Swiglu
    ):
        a2 = a2.to(dtypes.fp8)
        a2_scale = a1_scale
    elif q_type2 == QuantType.per_1x32:
        a2, a2_scale = _quantize_stage2_per_1x32(
            a2,
            a2_scale,
            quant_func2,
            q_dtype_a2,
            sorted_ids2,
            num_valid_ids2,
            token_num,
            topk,
            inter_dim,
            num_local_tokens,
            block_size_M2,
            getattr(metadata.stage2, "func", None) is _flydsl_stage2_wrapper,
        )
    elif q_type2 == QuantType.per_1x128 and quant_type == QuantType.per_1x128 and metadata.stage1.func is asm_stage1:
        a2_v = a2[:token_num, :, :]
        a2_scale = (
            a2[token_num:, ...]
            .view(-1)[: token_num * topk * inter_dim * ratio // 128]
            .view(dtypes.fp32)
            .view(token_num, -1)
        )
        a2 = a2_v
    else:
        a2, a2_scale = quant_func2(
            a2,
            scale=a2_scale,
            quant_dtype=q_dtype_a2,
            num_rows=num_local_tokens,
            num_rows_factor=topk,
        )
        a2 = a2.view(token_num, topk, inter_dim)

    metadata.stage2(
        a2,
        w1,
        w2,
        sorted_ids2,
        sorted_expert_ids2,
        num_valid_ids2,
        moe_out,
        topk,
        w2_scale=(
            w2_scale.view(dtypes.fp8_e8m0) if w2.dtype == dtypes.fp4x2 else w2_scale
        ),
        a2_scale=a2_scale,
        block_m=block_size_M2,
        sorted_weights=sorted_weights2 if not doweight_stage1 else None,
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
    E, model_dim, inter_dim = get_inter_dim(w1.shape, w2.shape, q_dtype_w=w2.dtype)

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
    E, model_dim, inter_dim = get_inter_dim(w1.shape, w2.shape, q_dtype_w=w2.dtype)
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
    E, model_dim, inter_dim = get_inter_dim(w1.shape, w2.shape, q_dtype_w=w2.dtype)
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
    if is_splitk:
        # CK kernel zeros this buffer via hipMemsetAsync when KBatch > 1
        sorted_size = min(token_num * topk * block_m, sorted_token_ids.shape[0])
        tmp_out = torch.empty(
            (sorted_size, w1.shape[1]), dtype=dtypes.fp32, device=out.device
        )
    else:
        tmp_out = out
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
        valid_out = tmp_out[: token_num * topk, :]
        if activation == ActivationType.Silu:
            aiter.silu_and_mul(out, valid_out.view(dtypes.fp32))
        else:
            aiter.gelu_and_mul(out, valid_out.view(dtypes.fp32))
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
    kernel_name="",
):
    token_num = hidden_states.shape[0]
    _, n1, k1 = w1.shape
    _, k2, n2 = w2.shape
    D = n2 if k2 == k1 else n2 * 2  # bit4 format
    # max_num_tokens_padded = sorted_expert_ids.shape[0]*block_size

    if w1.dtype is torch.uint32:
        D = D * 8

    out = torch.empty((token_num, topk, D), dtype=dtype, device=hidden_states.device)
    # WARNING: when split_k > 1, this allocation has the same undersized buffer
    # pattern fixed in ck_moe_stage1 (see ROCm/aiter#2508). If the CK tile
    # kernel calls hipMemsetAsync with sorted_size rows, this will overflow.
    # When fp32 splitk is enabled, apply the same fix: use sorted_size =
    # min(token_num * topk * block_m, sorted_token_ids.shape[0]) and slice
    # valid_out = tmp_out[:token_num * topk, :] before silu_and_mul/gelu_and_mul.
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
        kernel_name,
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
    kernel_name="",
):
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
        kernel_name=kernel_name,
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

    if (
        get_gfx() in ["gfx942", "gfx950"]
        and (expert, topk)
        in [
            (128, 4),
            (128, 6),
            (128, 8),
            (256, 6),
            (256, 8),
            (384, 8),
        ]
        and gating_output.dtype in [dtypes.bf16, dtypes.fp32]
        and gating_output.is_contiguous()
    ):
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
