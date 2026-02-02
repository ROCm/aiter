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
from aiter.ops.triton.fused_mxfp4_quant import fused_dynamic_mxfp4_quant_moe_sort
from aiter.utility import fp4_utils
from aiter.utility.fp4_utils import moe_mxfp4_sort

BLOCK_SIZE_M = 32


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
    device = topk_ids.device
    M, topk = topk_ids.shape
    max_num_tokens_padded = topk_ids.numel() + num_experts * block_size - topk

    max_num_m_blocks = int((max_num_tokens_padded + block_size - 1) // block_size)
    sorted_ids = torch.empty((max_num_tokens_padded,), dtype=dtypes.i32, device=device)
    sorted_weights = torch.empty(
        (max_num_tokens_padded,), dtype=dtypes.fp32, device=device
    )
    sorted_expert_ids = torch.empty(
        (max_num_m_blocks,), dtype=dtypes.i32, device=device
    )
    num_valid_ids = torch.empty((2), dtype=dtypes.i32, device=device)
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
        block_size,
        expert_mask,
        num_local_tokens,
        dispatch_policy,
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
    use_flydsl=False,
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
        use_flydsl=use_flydsl,
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
    use_flydsl: bool = False,
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
    use_flydsl: bool = False,
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
    )

    block_size_M = metadata.block_m if block_size_M is None else block_size_M

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
            sorted_ids=sorted_ids,
            sorted_weights=sorted_weights,
            sorted_expert_ids=sorted_expert_ids,
            num_valid_ids=num_valid_ids,
            moe_out=moe_buf,
            isG1U1=isG1U1,
            block_size_M=block_size_M,
            activation=activation,
            quant_type=quant_type,
            doweight_stage1=doweight_stage1,
            q_dtype_a=q_dtype_a,
            q_dtype_w=q_dtype_w,
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            a1_scale=a1_scale,
            a2_scale=a2_scale,
            expert_mask=expert_mask,
            num_local_tokens=num_local_tokens,
            topk_ids=topk_ids,
            topk_weights=topk_weight,
            # following for cktile support
            hidden_pad=hidden_pad,
            intermediate_pad=intermediate_pad,
            bias1=bias1,
            bias2=bias2,
            use_flydsl=use_flydsl,
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
    tgN = (inter_dim * 2 + tileN - 1) // tileN

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
    }
}
# fmt: on

quant_remap = {QuantType.per_128x128: QuantType.per_1x128}


def nextPow2(n):
    if n <= 0:
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
    if cfg is None or int(os.environ.get("AITER_BYPASS_TUNE_CONFIG", "0")):
        ksplit = 0
        kernelName1 = ""
        kernelName2 = ""
        run_1stage = False
        # Force disable 1stage for now.
        run_1stage = False

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
                if q_type == QuantType.per_1x128
                else ksplit
            )
        )
        aiter.logger.info(
            f"run_1stage = {run_1stage}, ksplit = {ksplit} q_type = {q_type}"
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
            ),
            functools.partial(
                cktile_moe_stage2,
                n_pad_zeros=hidden_pad // 64 * 64,
                k_pad_zeros=intermediate_pad // 128 * 128,
            ),
            get_block_m(),
            ksplit,
            False,
            True,
        )
    if (
        "ck2stages" in kernelName1
        or (q_type == QuantType.per_1x128 and doweight_stage1)
        or q_dtype_w
        in [
            dtypes.bf16,
            dtypes.fp16,
            torch.uint32,
            dtypes.fp4x2,
            dtypes.fp8,
            dtypes.i8,
        ]
    ):
        use_flydsl_stage2 = (
            os.environ.get("AITER_USE_FLYDSL_MOE", "1")
            in ("1", "true", "True", "YES", "yes")
            and q_type == QuantType.per_Token
            and q_dtype_a in (dtypes.fp8, dtypes.i8)
            and q_dtype_w in (dtypes.fp8, dtypes.i8)
            and use_g1u1
            and activation == ActivationType.Silu
        )
        use_flydsl_stage1 = use_flydsl_stage2
        flydsl_block_m = int(block_m) if block_m is not None else 64
        stage1_func = (
            functools.partial(flydsl_moe_stage1)
            if use_flydsl_stage1
            else functools.partial(
                ck_moe_stage1,
                kernelName=kernelName1,
                activation=activation,
                quant_type=q_type,
                splitk=ksplit,
            )
        )
        stage2_func = (
            functools.partial(
                flydsl_moe_stage2,
            )
            if use_flydsl_stage2
            else functools.partial(
                aiter.ck_moe_stage2_fwd,
                kernelName=kernelName2,
                activation=activation,
                quant_type=q_type,
            )
        )
        if os.environ.get("AITER_FLYDSL_DEBUG", "0") == "1":
            logger.info("use_flydsl_stage2=%s", use_flydsl_stage2)
            logger.info("use_flydsl_stage1=%s", use_flydsl_stage1)
            logger.info("flydsl_block_m=%s", flydsl_block_m)
            logger.info("stage1_func=%s", stage1_func)
            logger.info("stage2_func=%s", stage2_func)
            logger.info("block_m=%s", block_m)
            logger.info("ksplit=%s", ksplit)
            logger.info("run_1stage=%s", run_1stage)
        return MOEMetadata(
            stage1_func,
            stage2_func,
            flydsl_block_m if use_flydsl_stage1 else block_m,
            int(ksplit),
            run_1stage,
        )

    # TODO: remove when stage2 support more size
    tmpList = [32, 64, 128]
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
    expert_mask: Optional[torch.Tensor] = None,
    num_local_tokens: Optional[torch.tensor] = None,
    topk_ids: Optional[torch.Tensor] = None,
    topk_weights: Optional[torch.Tensor] = None,
    # following for cktile support
    hidden_pad=0,
    intermediate_pad=0,
    bias1=None,
    bias2=None,
    use_flydsl: bool = False,
):
    quant_func = get_quant(quant_type)
    token_num_quant_moe_sort_switch = 1024
    token_num, _ = hidden_states.shape
    E, model_dim, inter_dim = get_inter_dim(w1.shape, w2.shape)
    dtype = moe_out.dtype
    device = hidden_states.device
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
    )

    # SmoothQuant integration for FlyDSL int8 path:
    # When using flydsl 2-stage, callers may pass per-expert smooth scales via a1_scale/a2_scale
    # (shapes [E, model_dim] and [E, inter_dim] respectively). We must NOT pass these as "static"
    # quant scales into per-token quant; instead:
    # - stage1: build per-(token,slot) int8 X + per-(token,slot) scale_x
    # - stage2: optionally apply smooth scale to a2 (per expert) before dynamic quantization.
    is_flydsl_stage1 = getattr(metadata.stage1, "func", None) is flydsl_moe_stage1
    smooth_a1 = None
    smooth_a2 = None
    if quant_type == QuantType.per_Token and is_flydsl_stage1:
        if (
            a1_scale is not None
            and a1_scale.numel() == E * model_dim
            and a1_scale.shape[-1] == model_dim
        ):
            smooth_a1 = a1_scale.view(E, model_dim).to(dtypes.fp32)
            a1_scale = None
        if (
            a2_scale is not None
            and a2_scale.numel() == E * inter_dim
            and a2_scale.shape[-1] == inter_dim
        ):
            smooth_a2 = a2_scale.view(E, inter_dim).to(dtypes.fp32)
            a2_scale = None
    if (
        quant_type == QuantType.per_1x32
        and dtype in [dtypes.bf16, dtypes.fp16]
        and q_dtype_a in [dtypes.bf16, dtypes.fp16]
        and w1.dtype == dtypes.fp4x2
        and activation == ActivationType.Swiglu
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
            a1_scale = moe_mxfp4_sort(
                a1_scale,
                sorted_ids=sorted_ids,
                num_valid_ids=num_valid_ids,
                token_num=token_num,
                block_size=block_size_M,
            )
    elif hidden_states.dtype != q_dtype_a:
        if quant_type == QuantType.per_1x128 and metadata.stage1.func is asm_stage1:
            quant_func = functools.partial(quant_func, transpose_scale=True)
        if smooth_a1 is not None and q_dtype_a == dtypes.i8:
            # Align with aiter asm path: use device kernel `moe_smoothquant_fwd` to build
            # per-(token,slot) int8 activations + per-(token,slot) scale_x.
            if topk_ids is None:
                raise RuntimeError("FlyDSL int8smooth requires topk_ids")
            # `moe_smoothquant_fwd` expects:
            # - out: [token*topk, model_dim] int8
            # - input: [token, model_dim] bf16/fp16
            # - x_scale: [E, model_dim] (or [E,1,model_dim]) fp32
            # - topk_ids: [token, topk] i32
            # - y_scale: [token*topk] fp32
            a1 = torch.empty(
                (token_num * topk, model_dim), dtype=dtypes.i8, device=device
            )
            a1_scale = torch.empty((token_num * topk), dtype=dtypes.fp32, device=device)
            topk_ids_sq = topk_ids
            if expert_mask is not None and smooth_a1.shape[0] != expert_mask.numel():
                topk_ids_sq = _get_local_expert_hash(expert_mask)[topk_ids_sq]

            aiter.moe_smoothquant_fwd(
                a1,
                hidden_states,
                smooth_a1.contiguous(),
                topk_ids_sq.to(dtypes.i32),
                a1_scale,
            )
        else:
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
    if is_flydsl_stage1:
        # FlyDSL stage1 needs the original token count (it may receive X already expanded to token*topk).
        extra_stage1_args["tokens"] = token_num
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
        dtype=dtype,
        **extra_stage1_args,
    )

    if (
        quant_type == QuantType.per_1x32
        and dtype in [dtypes.bf16, dtypes.fp16]
        and q_dtype_a in [dtypes.bf16, dtypes.fp16]
        and w1.dtype == dtypes.fp4x2
        and activation == ActivationType.Swiglu
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
            a2_scale = moe_mxfp4_sort(
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
        if smooth_a2 is not None and q_dtype_a == dtypes.i8:
            # Fuse stage2 SmoothQuant (scale select + mul + per-row dynamic quant) into ONE kernel.
            # This avoids extra vectorized gather/copy/mul kernels that show up in profiling.
            #
            # Uses `smooth_per_token_scaled_quant`, which applies:
            #   y = input * smooth_scale[smooth_scale_map[row]]
            #   row_scale = max(abs(y)) / 127
            #   out = round(y / row_scale)
            #
            # NOTE: `smooth_scale_map` is indexed by flattened row id (blockIdx.x), so the
            # input layout must match `topk_ids` flatten order. For A2 we keep token-major:
            #   row = token*topk + slot  (input is [token, topk, inter_dim] contiguous)
            if topk_ids is None:
                raise RuntimeError(
                    "SmoothQuant stage2 requires topk_ids for smooth_scale_map"
                )
            a2_in = a2.view(token_num, topk, inter_dim).contiguous()
            a2_out = torch.empty(
                (token_num, topk, inter_dim), dtype=q_dtype_a, device=device
            )
            a2_scale = torch.empty(
                (token_num, topk, 1), dtype=dtypes.fp32, device=device
            )
            topk_ids_sq2 = topk_ids
            if expert_mask is not None and smooth_a2.shape[0] != expert_mask.numel():
                topk_ids_sq2 = _get_local_expert_hash(expert_mask)[topk_ids_sq2]

            aiter.smooth_per_token_scaled_quant(
                a2_out,
                a2_in,
                a2_scale,
                smooth_a2.contiguous(),
                topk_ids_sq2.contiguous(),
                num_rows=num_local_tokens,
                num_rows_factor=topk,
            )
            a2 = a2_out
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
    use_swiglu = (a1_scale is None) and (quant_type == QuantType.per_1x32)
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
    dtype=None,
):
    token_num = hidden_states.shape[0]
    tmp_out = (
        torch.zeros(
            (token_num, topk, w1.shape[1]), dtype=dtypes.fp32, device=out.device
        )
        if splitk > 1
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
        splitk,
        out.dtype,
    )
    if splitk > 1:
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

    # print("Run cktile_moe_stage1: M=%d, N(N*2)=%d, K=%d, topk=%d, expert=%d"%(token_num, w1.shape[1], hidden_states.shape[1], topk, w1.shape[0]))
    aiter.moe_cktile2stages_gemm1(
        hidden_states,
        w1,
        out,
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
        block_m,
    )
    return out


# ---- FlyDSL compile/import caches (avoid per-call overhead) ----
_FLYDSL_COMPILE_MOE_GEMM1 = None
_FLYDSL_COMPILE_MOE_GEMM2 = None
_FLYDSL_MOE_GEMM1_CACHE = {}
_FLYDSL_MOE_GEMM2_CACHE = {}
_FLYDSL_SCRATCH_CACHE = {}
_LOCAL_EXPERT_HASH_CACHE = {}


def _get_local_expert_hash(expert_mask: torch.Tensor) -> torch.Tensor:
    """Cache global->local expert id map for EP."""
    key = (
        int(expert_mask.device.index),
        int(expert_mask.numel()),
        int(expert_mask.data_ptr()),
    )
    cached = _LOCAL_EXPERT_HASH_CACHE.get(key)
    if cached is not None and cached.device == expert_mask.device:
        return cached
    local_expert_hash = expert_mask.cumsum(0, dtype=dtypes.i32)
    local_expert_hash[local_expert_hash > 0] -= 1
    _LOCAL_EXPERT_HASH_CACHE[key] = local_expert_hash
    return local_expert_hash


def _get_flydsl_compile_fns():
    global _FLYDSL_COMPILE_MOE_GEMM1, _FLYDSL_COMPILE_MOE_GEMM2
    if _FLYDSL_COMPILE_MOE_GEMM1 is not None and _FLYDSL_COMPILE_MOE_GEMM2 is not None:
        return _FLYDSL_COMPILE_MOE_GEMM1, _FLYDSL_COMPILE_MOE_GEMM2

    import sys

    DSL2_ROOT = os.environ.get("DSL2_ROOT", "/data/felix/dsl2")
    FLIR_ROOT = os.environ.get("FLIR_ROOT", "/workspace/FLIR")
    if DSL2_ROOT not in sys.path:
        sys.path.insert(0, DSL2_ROOT)
    if FLIR_ROOT not in sys.path:
        sys.path.insert(0, FLIR_ROOT)
    from kernels.moe_gemm_2stage import compile_moe_gemm1, compile_moe_gemm2  # type: ignore

    _FLYDSL_COMPILE_MOE_GEMM1 = compile_moe_gemm1
    _FLYDSL_COMPILE_MOE_GEMM2 = compile_moe_gemm2
    return _FLYDSL_COMPILE_MOE_GEMM1, _FLYDSL_COMPILE_MOE_GEMM2


def flydsl_moe_stage1(
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
    dtype=torch.bfloat16,
    **_kwargs,
):
    if a1_scale is None or w1_scale is None:
        raise RuntimeError("FlyDSL stage1 requires a1_scale and w1_scale")

    # `tokens` is the original token count. For smoothquant int8 we may receive X already expanded
    # to [tokens*topk, model_dim].
    tokens = int(_kwargs.get("tokens", hidden_states.shape[0]))
    token_num = tokens
    E, _, model_dim = w1.shape
    inter_dim = w2.shape[2]

    # IMPORTANT: stage1/stage2 tile_m must match the sorting `block_m` used to build `sorted_*`.
    # Allow env override only when it matches `block_m`.
    tile_m = block_m if block_m is not None else 64
    tile_m_env = os.environ.get("AITER_FLYDSL_MOE_TILE_M1", "")
    if tile_m_env:
        try:
            tile_m_env_i = int(tile_m_env)
            if block_m is None or tile_m_env_i == int(block_m):
                tile_m = tile_m_env_i
        except ValueError:
            pass
    tile_n = int(os.environ.get("AITER_FLYDSL_MOE_TILE_N1", "128"))
    tile_k = int(os.environ.get("AITER_FLYDSL_MOE_TILE_K1", "128"))
    stage1_cshuffle = os.environ.get("AITER_FLYDSL_STAGE1_CSHUFFLE", "0") in (
        "1",
        "true",
        "True",
        "YES",
        "yes",
    )

    sorted_ids = sorted_token_ids.contiguous()
    sorted_eids = sorted_expert_ids.contiguous()
    sorted_size = int(sorted_ids.numel())
    size_expert_ids_total = int(sorted_eids.numel())
    blocks = int(size_expert_ids_total)

    if sorted_weights is None:
        sorted_w = torch.empty((0,), dtype=dtypes.fp32, device=sorted_ids.device)
        doweight_stage1 = False
    else:
        sorted_w = sorted_weights
        doweight_stage1 = True

    debug_flydsl = os.environ.get("AITER_FLYDSL_DEBUG", "0") == "1"
    if debug_flydsl:
        logger.info(
            "[flydsl] stage1 inputs: tokens=%d topk=%d model_dim=%d inter_dim=%d block_m=%d",
            token_num,
            topk,
            model_dim,
            inter_dim,
            tile_m,
        )

    # Infer in_dtype: fp8 / int8 / int8smooth (expanded X + per-route scale).
    x_rows = int(hidden_states.shape[0])
    is_token_slot = x_rows == (token_num * int(topk))
    x_q = hidden_states.contiguous().view(x_rows, model_dim)
    scale_x_1d = a1_scale.view(-1).contiguous()
    w1_flat = w1.contiguous().view(E * (2 * inter_dim), model_dim)
    w1_scale_1d = w1_scale.view(-1).contiguous()

    compile_moe_gemm1, _compile_moe_gemm2 = _get_flydsl_compile_fns()

    out_dtype = "bf16" if dtype == torch.bfloat16 else "f16"

    # Optional: run stage1 with fp16 output (and optional CShuffle) then cast to bf16.
    # This can be faster than bf16 output on some configs.
    stage1_fp16_out = os.environ.get("AITER_FLYDSL_STAGE1_FP16_OUT", "0") in (
        "1",
        "true",
        "True",
        "YES",
        "yes",
    )
    keep_fp16_for_stage2 = (
        True  # hack here, bf16 out is slow. temp use fp16, todo fix bf16 atomic
    )
    out_kernel = out
    out_dtype_kernel = out_dtype
    if stage1_fp16_out and dtype == torch.bfloat16:
        key_s = ("stage1_fp16_out", int(out.device.index), tuple(out.shape))
        out_kernel = _FLYDSL_SCRATCH_CACHE.get(key_s)
        if out_kernel is None or out_kernel.device != out.device:
            out_kernel = torch.empty(out.shape, device=out.device, dtype=dtypes.fp16)
            _FLYDSL_SCRATCH_CACHE[key_s] = out_kernel
        out_kernel.zero_()
        out_dtype_kernel = "f16"
    if w1.dtype == dtypes.fp8:
        in_dtype = "fp8"
    elif w1.dtype == dtypes.i8:
        in_dtype = "int8smooth" if is_token_slot else "int8"
    else:
        raise ValueError(f"FlyDSL stage1 unsupported w1 dtype: {w1.dtype}")

    key1 = (
        model_dim,
        inter_dim,
        E,
        topk,
        tile_m,
        tile_n,
        tile_k,
        bool(doweight_stage1),
        in_dtype,
        out_dtype_kernel,
        bool(stage1_cshuffle),  # use_cshuffle_epilog
    )
    exe1 = _FLYDSL_MOE_GEMM1_CACHE.get(key1)
    if exe1 is None:
        exe1 = compile_moe_gemm1(
            model_dim=model_dim,
            inter_dim=inter_dim,
            experts=E,
            topk=topk,
            tile_m=tile_m,
            tile_n=tile_n,
            tile_k=tile_k,
            doweight_stage1=bool(doweight_stage1),
            in_dtype=in_dtype,
            out_dtype=out_dtype_kernel,
            use_cshuffle_epilog=bool(stage1_cshuffle),
        )
        _FLYDSL_MOE_GEMM1_CACHE[key1] = exe1
    exe1(
        out_kernel,
        x_q,
        w1_flat,
        scale_x_1d,
        w1_scale_1d,
        sorted_ids,
        sorted_eids,
        sorted_w.view(-1).contiguous(),
        num_valid_ids,
        token_num,
        inter_dim,
        model_dim,
        int(blocks),
    )
    if out_kernel is not out:
        if keep_fp16_for_stage2:
            # Downstream stage2 smoothquant/quant will consume fp16 directly.
            return out_kernel
        out.copy_(out_kernel.to(dtypes.bf16))
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
        block_m,
    )
    return out


def flydsl_moe_stage2(
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
    sorted_weights=None,
    **_kwargs,
):
    if w2_scale is None or a2_scale is None:
        raise RuntimeError("FlyDSL stage2 requires a2_scale and w2_scale")

    token_num, _, inter_dim = a2.shape
    model_dim = w2.shape[1]
    E = w2.shape[0]

    tile_m = block_m if block_m is not None else 64
    tile_m_env = os.environ.get("AITER_FLYDSL_MOE_TILE_M2", "")
    if tile_m_env:
        try:
            tile_m_env_i = int(tile_m_env)
            if block_m is None or tile_m_env_i == int(block_m):
                tile_m = tile_m_env_i
        except ValueError:
            pass
    tile_n = int(os.environ.get("AITER_FLYDSL_MOE_TILE_N2", "256"))
    tile_k = int(os.environ.get("AITER_FLYDSL_MOE_TILE_K2", "128"))
    stage2_cshuffle = os.environ.get("AITER_FLYDSL_STAGE2_CSHUFFLE", "1") in (
        "1",
        "true",
        "True",
        "YES",
        "yes",
    )

    sorted_ids = sorted_token_ids.contiguous()
    sorted_eids = sorted_expert_ids.contiguous()
    sorted_size = int(sorted_ids.numel())
    size_expert_ids_total = int(sorted_eids.numel())
    blocks = int(size_expert_ids_total)

    if sorted_weights is None:
        sorted_w = torch.zeros(
            sorted_ids.shape, dtype=dtypes.fp32, device=sorted_ids.device
        )
    else:
        sorted_w = sorted_weights

    # Optional: avoid global atomics in stage2 by writing per-(token,slot) output and reducing.
    # This is often faster for large topk / EP, and also reduces atomic contention.
    # Support both legacy no_atomics (torch.sum) and new compile_moe_gemm2_ex (FlyDSL reduce kernel).
    no_atomics = os.environ.get("AITER_FLYDSL_STAGE2_NO_ATOMICS", "0") in (
        "1",
        "true",
        "True",
        "YES",
        "yes",
    )
    # New env var: use compile_moe_gemm2_ex with mode selection (AUTO/ATOMIC/REDUCE).
    use_gemm2_ex = os.environ.get("AITER_FLYDSL_USE_GEMM2_EX", "0") in (
        "1",
        "true",
        "True",
        "YES",
        "yes",
    )
    gemm2_mode = os.environ.get("AITER_FLYDSL_GEMM2_MODE", "AUTO").upper().strip()

    a2_qt_flat = a2.contiguous().view(-1)
    a2_scale_1d = a2_scale.view(-1).contiguous()
    w2_flat = w2.contiguous().view(E * model_dim, inter_dim)
    w2_scale_1d = w2_scale.view(-1).contiguous()

    # NOTE: FlyDSL stage2 uses global atomics; keep output dtype consistent with `out`.
    _compile_moe_gemm1, compile_moe_gemm2 = _get_flydsl_compile_fns()

    if out.dtype == dtypes.bf16:
        out_dtype = "bf16"
    elif out.dtype == dtypes.fp16:
        out_dtype = "fp16"
    elif out.dtype == dtypes.fp32:
        out_dtype = "f32"
    else:
        raise ValueError(
            f"FlyDSL stage2 only supports out dtype in (fp16, bf16, fp32), got {out.dtype}"
        )

    # Optional fast path: run fp16 atomics then cast to bf16.
    # This can be significantly faster than bf16 global atomics on some configs.
    use_fp16_out = (
        os.environ.get("AITER_FLYDSL_STAGE2_FP16_OUT", "0")
        in ("1", "true", "True", "YES", "yes")
        and out.dtype == dtypes.bf16
        and not no_atomics
        and not use_gemm2_ex  # compile_moe_gemm2_ex handles dtype internally
    )
    out_kernel = out
    out_dtype_kernel = out_dtype
    if use_fp16_out:
        # Reuse fp16 scratch to avoid per-iter allocation overhead.
        key_s = ("stage2_fp16_out", int(out.device.index), tuple(out.shape))
        out_kernel = _FLYDSL_SCRATCH_CACHE.get(key_s)
        if out_kernel is None or out_kernel.device != out.device:
            out_kernel = torch.empty(out.shape, device=out.device, dtype=dtypes.fp16)
            _FLYDSL_SCRATCH_CACHE[key_s] = out_kernel
        out_kernel.zero_()
        out_dtype_kernel = "fp16"
    if w2.dtype == dtypes.fp8:
        in_dtype = "fp8"
    elif w2.dtype == dtypes.i8:
        in_dtype = "int8"
    else:
        raise ValueError(f"FlyDSL stage2 unsupported w2 dtype: {w2.dtype}")

    # Use compile_moe_gemm2_ex when requested (supports both atomic and reduce modes).
    if use_gemm2_ex:
        # Import at runtime to avoid circular dependency issues.
        try:
            from kernels.moe_gemm_2stage import compile_moe_gemm2_ex, MoeGemm2Mode
        except ImportError:
            logger.warning(
                "AITER_FLYDSL_USE_GEMM2_EX=1 but compile_moe_gemm2_ex not available. "
                "Falling back to compile_moe_gemm2."
            )
            use_gemm2_ex = False

    if use_gemm2_ex:
        # Map env var to MoeGemm2Mode enum.
        if gemm2_mode == "ATOMIC":
            mode_val = MoeGemm2Mode.ATOMIC
        elif gemm2_mode == "REDUCE":
            mode_val = MoeGemm2Mode.REDUCE
        else:
            mode_val = MoeGemm2Mode.AUTO

        key2_ex = (
            "gemm2_ex",
            model_dim,
            inter_dim,
            E,
            topk,
            tile_m,
            tile_n,
            tile_k,
            bool(sorted_weights is not None),
            in_dtype,
            out_dtype_kernel,
            bool(stage2_cshuffle),
            mode_val,
        )
        exe2 = _FLYDSL_MOE_GEMM2_CACHE.get(key2_ex)
        if exe2 is None:
            exe2 = compile_moe_gemm2_ex(
                model_dim=model_dim,
                inter_dim=inter_dim,
                experts=E,
                topk=topk,
                tile_m=tile_m,
                tile_n=tile_n,
                tile_k=tile_k,
                doweight_stage2=bool(sorted_weights is not None),
                in_dtype=in_dtype,
                out_dtype=out_dtype_kernel,
                use_cshuffle_epilog=bool(stage2_cshuffle),
                mode=mode_val,
                tokens_hint=token_num,
            )
            _FLYDSL_MOE_GEMM2_CACHE[key2_ex] = exe2

        # Launch kernel (wrapper handles atomic vs reduce internally).
        exe2(
            out_kernel if not use_fp16_out else out_kernel,
            a2_qt_flat,
            w2_flat.view(-1),
            a2_scale_1d,
            w2_scale_1d,
            sorted_ids,
            sorted_eids,
            sorted_w.view(-1).contiguous(),
            num_valid_ids,
            token_num,
            model_dim,
            inter_dim,
            int(blocks),
        )
        if use_fp16_out:
            out.copy_(out_kernel.to(dtypes.bf16))
        return out

    # Legacy path: use compile_moe_gemm2 with accumulate flag.
    key2 = (
        model_dim,
        inter_dim,
        E,
        topk,
        tile_m,
        tile_n,
        tile_k,
        bool(sorted_weights is not None),
        in_dtype,
        out_dtype_kernel,
        bool(not no_atomics),
        bool(stage2_cshuffle),
    )
    exe2 = _FLYDSL_MOE_GEMM2_CACHE.get(key2)
    if exe2 is None:
        exe2 = compile_moe_gemm2(
            model_dim=model_dim,
            inter_dim=inter_dim,
            experts=E,
            topk=topk,
            tile_m=tile_m,
            tile_n=tile_n,
            tile_k=tile_k,
            doweight_stage2=bool(sorted_weights is not None),
            in_dtype=in_dtype,
            out_dtype=out_dtype_kernel,
            accumulate=not no_atomics,
            use_cshuffle_epilog=bool(stage2_cshuffle),
        )
        _FLYDSL_MOE_GEMM2_CACHE[key2] = exe2
    if not no_atomics:
        exe2(
            out_kernel,
            a2_qt_flat,
            w2_flat.view(-1),
            a2_scale_1d,
            w2_scale_1d,
            sorted_ids,
            sorted_eids,
            sorted_w.view(-1).contiguous(),
            num_valid_ids,
            token_num,
            model_dim,
            inter_dim,
            int(blocks),
        )
        if use_fp16_out:
            out.copy_(out_kernel.to(dtypes.bf16))
        return out

    # Scatter per-(token,slot) output then reduce over topk on the host side (legacy torch.sum path).
    # IMPORTANT: scatter mode does not use atomics and may not write every element
    # (e.g. masked/fake expert rows). Zero-init to keep unwritten entries deterministic.
    key_s = (
        "stage2_scatter_slots",
        int(out.device.index),
        tuple((token_num, topk, model_dim)),
        str(out.dtype),
    )
    out_slots = _FLYDSL_SCRATCH_CACHE.get(key_s)
    if (
        out_slots is None
        or out_slots.device != out.device
        or out_slots.dtype != out.dtype
    ):
        out_slots = torch.empty(
            (token_num, topk, model_dim), device=out.device, dtype=out.dtype
        )
        _FLYDSL_SCRATCH_CACHE[key_s] = out_slots
    out_slots.zero_()
    exe2(
        out_slots.view(-1),
        a2_qt_flat,
        w2_flat.view(-1),
        a2_scale_1d,
        w2_scale_1d,
        sorted_ids,
        sorted_eids,
        sorted_w.view(-1).contiguous(),
        num_valid_ids,
        token_num,
        model_dim,
        inter_dim,
        int(blocks),
    )
    out.copy_(out_slots.sum(dim=1, dtype=torch.float32).to(out.dtype))
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
        get_gfx() == "gfx942"
        and (expert, topk) in [(128, 6), (128, 8), (256, 6), (256, 8)]
        and gating_output.dtype == dtypes.fp32
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
