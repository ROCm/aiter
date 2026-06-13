# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import torch
import os
from typing import Optional, Callable
from dataclasses import dataclass
import functools
import aiter
from aiter import logger
from aiter import ActivationType, QuantType, dtypes
from aiter.utility import fp4_utils

# from aiter import get_torch_quant as get_quant
from aiter import get_hip_quant as get_quant
from aiter.utility.fp4_utils import moe_mxfp4_sort
from aiter.jit.core import (
    AITER_ROOT_DIR,
    AITER_CSRC_DIR,
    PY,
    bd_dir,
    mp_lock,
)
from aiter.jit.utils.chip_info import get_cu_num
from aiter.fused_moe import moe_sorting

BLOCK_SIZE_M = 32


@functools.lru_cache(maxsize=1024)
def get_dp_shared_expert_token_range(token_num, dp_size, rank):
    per_dp_token = (token_num + dp_size - 1) // dp_size
    start = min(per_dp_token * rank, token_num)
    end = min(per_dp_token * (rank + 1), token_num)
    return start, end


@functools.lru_cache(maxsize=1024)
def get_dp_shared_expert_stage1_moe_sorting_result(
    token_num, share_expert_num, device, block_size_M, share_expert_score=1.0
):
    topk_ids_list = [[i] * token_num for i in range(share_expert_num)]
    topk_ids = torch.tensor(topk_ids_list, dtype=dtypes.i32, device=device).view(-1, 1)
    topk_weights = torch.empty(
        (token_num * share_expert_num), dtype=dtypes.fp32, device=device
    )
    topk_weights.fill_(share_expert_score)
    sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_buf = moe_sorting(
        topk_ids,
        topk_weights,
        share_expert_num,
        model_dim=0,
        moebuf_dtype=dtypes.fp32,
        block_size=block_size_M,
        expert_mask=None,
        num_local_tokens=None,
        dispatch_policy=0,
    )
    del moe_buf
    del sorted_weights
    return topk_ids, sorted_ids, sorted_expert_ids, num_valid_ids


@functools.lru_cache(maxsize=1024)
def get_dp_shared_expert_stage2_moe_sorting_result(
    token_num, share_expert_num, device, block_size_M, share_expert_score=1.0
):
    topk_ids_list = [list(range(share_expert_num)) for i in range(token_num)]
    topk_ids = torch.tensor(topk_ids_list, dtype=dtypes.i32, device=device)
    topk_weights = torch.empty(
        (token_num, share_expert_num), dtype=dtypes.fp32, device=device
    )
    topk_weights.fill_(share_expert_score)
    sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_buf = moe_sorting(
        topk_ids,
        topk_weights,
        share_expert_num,
        model_dim=0,
        moebuf_dtype=dtypes.fp32,
        block_size=block_size_M,
        expert_mask=None,
        num_local_tokens=None,
        dispatch_policy=0,
    )
    del moe_buf
    return topk_ids, sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids


@functools.lru_cache(maxsize=1024)
def get_inter_dim(w1_shape, w2_shape):
    E, _, model_dim = w1_shape
    E, model_dim, inter_dim = w2_shape

    int4_war = model_dim // w1_shape[-1]
    inter_dim *= int4_war
    return E, model_dim, inter_dim


def nextPow2(n):
    if n <= 0:
        return 1
    return 1 << (n - 1).bit_length()


def get_padded_M(M):
    padded_m = M
    if M >= 1 and M <= 16:
        padded_m = 16
    elif M < 1024:
        padded_m = nextPow2(padded_m)
    else:
        padded_m = 1024
    return padded_m


def fused_moe_dp_share_expert(
    hidden_states,
    w1,  # [expert(local_expert:EP), inter_dim*2, dim] N,K
    w2,  # [expert(local_expert:EP), dim, inter_dim]
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
    dp_size=1,
    dp_rank=0,
    moe_buf: Optional[
        torch.tensor
    ] = None,  # you can use no-shared expert result here, it will atomic add to it
):
    """user API"""
    orig_M, model_dim = hidden_states.shape
    start, end = get_dp_shared_expert_token_range(orig_M, dp_size, dp_rank)
    hidden_states_dp = hidden_states[start:end]
    M = end - start
    E, model_dim, inter_dim = get_inter_dim(w1.shape, w2.shape)
    topk = E

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
    q_dtype_a = dtypes.fp4x2 if quant_type == QuantType.per_1x32 else q_dtype_a

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
    )

    block_size_M = metadata.block_m if block_size_M is None else block_size_M

    if moe_buf is None:
        moe_buf = torch.zeros_like(hidden_states, dtype=dtype)
    moe_buf_dp = moe_buf[start:end]
    if M == 0:
        return moe_buf
    fused_moe_2stages(
        hidden_states_dp,
        w1,
        w2,
        topk,
        moe_buf_dp,
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
    )
    return moe_buf


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
    # following for quant
    q_dtype_a=None,
    q_dtype_w=None,
    w1_scale=None,  # [expert(local_expert:EP), inter_dim, 1]
    w2_scale=None,  # [expert(local_expert:EP), model_dim, 1]
    a1_scale=None,  # [expert(local_expert:EP), 1, model_dim]
    a2_scale=None,  # [expert(local_expert:EP), 1, inter_dim]
    num_local_tokens: Optional[torch.tensor] = None,
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
            fmoe_func = aiter.fmoe_int8_g1u0

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
            fc2_smooth_scale=None,
            activation=activation,
        )
    return moe_buf


@functools.lru_cache(maxsize=1024)
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


cfg_2stages = None
# fmt: off
fused_moe_1stage_dict = {
    "gfx942":
    {
        # activation,                    quant_type,        dtype,    q_dtype_a,    q_dtype_w,   isG1U1,      API
        (ActivationType.Silu,          QuantType.No,  dtypes.bf16,   dtypes.bf16,   dtypes.bf16,   False) : aiter.fmoe,
        (ActivationType.Silu,          QuantType.No,  dtypes.fp16,   dtypes.fp16,   dtypes.fp16,   False) : aiter.fmoe,
        (ActivationType.Gelu,   QuantType.per_Token,  dtypes.bf16,    dtypes.fp8,   dtypes.i4x2,    True) : aiter.fmoe_g1u1,
        (ActivationType.Silu,    QuantType.per_1x32,  dtypes.bf16,  dtypes.fp4x2,  dtypes.fp4x2,    True) : aiter.fmoe_g1u1,
        (ActivationType.Silu,   QuantType.per_Token,  dtypes.bf16,     dtypes.i8,     dtypes.i8,    True) : aiter.fmoe_g1u1,
        (ActivationType.Gelu,   QuantType.per_Token,  dtypes.bf16,     dtypes.i8,     dtypes.i8,    True) : aiter.fmoe_g1u1,
        (ActivationType.Silu,   QuantType.per_Token,  dtypes.bf16,    dtypes.fp8,    dtypes.fp8,    True) : aiter.fmoe_g1u1,
        (ActivationType.Gelu,   QuantType.per_Token,  dtypes.bf16,    dtypes.fp8,    dtypes.fp8,    True) : aiter.fmoe_g1u1,
        (ActivationType.Silu,   QuantType.per_1x128,  dtypes.bf16,    dtypes.fp8,    dtypes.fp8,    True) : aiter.fmoe_g1u1,
        (ActivationType.Silu,   QuantType.per_Token,  dtypes.bf16,     dtypes.i8,     dtypes.i8,   False) : aiter.fmoe_int8_g1u0,
        (ActivationType.Gelu,   QuantType.per_Token,  dtypes.bf16,     dtypes.i8,     dtypes.i8,   False) : aiter.fmoe_int8_g1u0,
    },
    "gfx950":
    {
        (ActivationType.Silu,    QuantType.per_1x32,   dtypes.bf16,   dtypes.fp4x2,  dtypes.fp4x2,    True) : aiter.fmoe_g1u1,
        (ActivationType.Silu,   QuantType.per_1x128,   dtypes.bf16,     dtypes.fp8,    dtypes.fp8,    True) : aiter.fmoe_fp8_blockscale_g1u1,
    }
}
# fmt: on

quant_remap = {QuantType.per_128x128: QuantType.per_1x128}


@dataclass
class MOEMetadata:
    stage1: Callable
    stage2: Callable
    block_m: int
    ksplit: int
    run_1stage: bool = False


@functools.lru_cache(maxsize=1024)
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
    config_path = f"{AITER_ROOT_DIR}/aiter/configs/"
    tune_file = os.path.join(config_path, "tuned_fmoe.csv")
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
            q_dtype_ws = q_dtype_w if q_dtype_w != torch.uint32 else "torch.int4"
            f.write(
                f"\n{token},{model_dim},{inter_dim},{expert},{topk},{activation},{dtype},{q_dtype_a},{q_dtype_ws},{q_type},{int(use_g1u1)},{int(doweight_stage1)}"
            )
        logger.info("\033[34m Start tuning fmoe")
        os.system(
            f"{PY} {AITER_CSRC_DIR}/ck_gemm_moe_2stages_codegen/gemm_moe_tune.py -i {untune_file} -o {tune_file} -o2 {profile_file} --last"
        )

    def FinalFunc():
        logger.info("\033[0m")

    cfg = cfg_2stages.get(keys, None)
    if cfg is None and os.environ.get("AITER_ONLINE_TUNE", "0") == "1":
        lock_path = os.path.join(bd_dir, f"lock_fmoe_tune_{keys}")
        mp_lock(lock_path, MainFunc=MainFunc, FinalFunc=FinalFunc)
        cfg_2stages = get_cfg_2stages(tune_file)
        cfg = cfg_2stages.get(keys, None)
        if cfg is None:
            logger.warning(f"Fmoe tuning not support for {keys}")

    if cfg is None:
        ksplit = 0
        kernelName1 = ""
        kernelName2 = ""
        run_1stage = False
        # if (
        #     not doweight_stage1
        #     and (
        #         activation,
        #         q_type,
        #         dtype,
        #         q_dtype_a,
        #         q_dtype_w,
        #         use_g1u1,
        #     )
        #     in fused_moe_1stage_dict[get_gfx()]
        # ):
        #     if q_type == QuantType.per_1x128:
        #         run_1stage = True and (inter_dim % 256 == 0)
        #     elif q_type == QuantType.per_Token and q_dtype_w in [dtypes.i8, dtypes.fp8]:
        #         run_1stage = token > 32
        #     else:
        #         run_1stage = token < 256
        block_m = (
            BLOCK_SIZE_M
            if run_1stage
            else (
                64
                if q_type == QuantType.per_1x128
                else get_block_size_M(token, topk, expert, inter_dim)
            )
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

    if "ck2stages" in kernelName1 or q_dtype_w in [
        dtypes.bf16,
        dtypes.fp16,
        torch.uint32,
        torch.uint8,
    ]:
        return MOEMetadata(
            functools.partial(
                aiter.ck_moe_stage1_fwd,
                kernelName=kernelName1,
                activation=activation,
                quant_type=q_type,
                dst_type=dtype,
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
):

    quant_func = get_quant(quant_type)

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
    )

    if a1_scale is not None and hidden_states.dtype != q_dtype_a:
        sm1_scale = a1_scale
        sm2_scale = a2_scale
        a1_scale = None
        a2_scale = None
        topk_ids, sorted_ids, sorted_expert_ids, num_valid_ids = (
            get_dp_shared_expert_stage1_moe_sorting_result(
                token_num, E, device, block_size_M
            )
        )
    else:
        sm1_scale = None
        sm2_scale = None
        topk_ids, sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids = (
            get_dp_shared_expert_stage2_moe_sorting_result(
                token_num, E, device, block_size_M
            )
        )

    if sm1_scale is not None:
        a1_scale = torch.empty((token_num, topk, 1), device=device, dtype=dtypes.fp32)
        a1 = torch.empty(
            (token_num, topk, model_dim),
            dtype=q_dtype_a,
            device=device,
        )
        hidden_states = hidden_states.view(1, token_num, model_dim).expand(topk, -1, -1)
        # aiter.moe_smoothquant_fwd(
        #     a1, hidden_states, sm1_scale, topk_ids, a1_scale
        # )
        aiter.smooth_per_token_scaled_quant(
            a1, hidden_states, a1_scale, sm1_scale, topk_ids
        )
        a1 = a1.view(-1, model_dim)
    elif quant_type == QuantType.per_1x32:
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

    a2 = metadata.stage1(
        a1,
        w1,
        w2,
        sorted_ids,
        sorted_expert_ids,
        num_valid_ids,
        a2 if sm1_scale is None else a2.view(-1, 1, inter_dim),
        topk if sm1_scale is None else 1,
        block_m=block_size_M,
        a1_scale=a1_scale,
        w1_scale=w1_scale,
        sorted_weights=None,
    )

    topk_ids, sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids = (
        get_dp_shared_expert_stage2_moe_sorting_result(
            token_num, E, device, block_size_M
        )
    )

    if sm2_scale is not None:
        a2_scale = torch.empty((token_num, topk, 1), device=device, dtype=dtypes.fp32)
        a2_tmp = torch.empty(
            (token_num, topk, inter_dim),
            dtype=q_dtype_a,
            device=device,
        )
        a2 = a2.view(topk, token_num, inter_dim).permute(1, 0, 2)
        aiter.smooth_per_token_scaled_quant(a2_tmp, a2, a2_scale, sm2_scale, topk_ids)
        a2 = a2_tmp
    elif quant_type == QuantType.per_1x32:
        a2 = a2.view(-1, inter_dim)
        a2, a2_scale = quant_func(
            a2,
            scale=a2_scale,
            quant_dtype=q_dtype_a,
            num_rows=num_local_tokens,
            num_rows_factor=topk,
        )
        a2 = a2.view(token_num, topk, -1)
        a2_scale = moe_mxfp4_sort(
            a2_scale[: token_num * topk, :].view(token_num, topk, -1),
            sorted_ids=sorted_ids,
            num_valid_ids=num_valid_ids,
            token_num=token_num,
            block_size=block_size_M,
        )

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
        w2_scale=w2_scale,
        a2_scale=a2_scale,
        block_m=block_size_M,
        sorted_weights=sorted_weights if not doweight_stage1 else None,
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
            aiter.silu_and_mul(out, tmp_out.view(dtypes.fp32).to(dtype))
        else:
            aiter.gelu_and_mul(out, tmp_out.view(dtypes.fp32).to(dtype))
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
    from aiter import pertoken_quant

    computeType = dtypes.fp32
    dtype = hidden_states.dtype
    torch_act = aiter.get_torch_act(activation)
    hidden_states = hidden_states.to(computeType)
    quant_dtype = w1.dtype
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
            sub_tokens_q, scale = pertoken_quant(sub_tokens, quant_dtype=quant_dtype)
            sub_tokens = sub_tokens_q.to(computeType) * scale.view(-1, 1)
            if fc1_smooth_scale is not None:
                sub_tokens = sub_tokens * (fc1_smooth_scale[E_id])

            act_input = sub_tokens @ (w1[E_id].transpose(0, 1))
            act_out = (
                torch_moe_act(act_input, torch_act, inter_dim).to(dtype).to(computeType)
            )
            if fc2_smooth_scale is not None:
                act_out = act_out * (fc2_smooth_scale[E_id])
            act_out_q, scale = pertoken_quant(act_out, quant_dtype=quant_dtype)
            act_out = act_out_q.to(computeType) * scale.view(-1, 1)
            out[mask] = act_out @ (w2[E_id].transpose(0, 1))

    return (out * topk_weight.view(B, -1, 1)).to(dtype).sum(dim=1)
