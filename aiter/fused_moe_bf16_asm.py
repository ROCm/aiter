# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import os
from dataclasses import dataclass
from typing import Optional

import pandas as pd
import torch
import torch.nn.functional as F

import aiter
from aiter import (
    ActivationType,
    QuantType,
    dtypes,
    get_hip_quant,
    logger,
    pertoken_quant,
)
from aiter.fused_moe import fused_moe

BLOCK_SIZE_M = 32


def moe_sorting_ck(
    topk_ids,
    topk_weights,
    num_experts,
    model_dim,
    moebuf_dtype,
    block_size=BLOCK_SIZE_M,
    expert_mask=None,
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
    )
    return sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_buf


def asm_moe_stage2(
    inter_states,
    w1,
    w2,
    sorted_ids,
    sorted_expert_ids,
    num_valid_ids,
    out,
    topk,
    kernelName,
    w2_scale,
    a2_scale,
    block_m,
    sorted_weights,
    quant_type,
    activation,
    splitk,
):
    return aiter.moe_stage2_g1u1(
        inter_states,
        w1,
        w2,
        sorted_ids,
        sorted_expert_ids,
        num_valid_ids,
        out,
        topk,
        kernelName,
        block_m,
        w2_scale,
        a2_scale,
        None,
        None,
        sorted_weights,
        quant_type,
        activation,
        splitk,
    )


@dataclass
class AsmInt8Config:
    run_1stage: bool = False
    run_2stage: bool = False
    block_m: int = 32
    ksplit: int = 0
    kernelName1: str = ""
    kernelName2: str = ""
    # If True, stage1 outputs int8 (fused quant in kernel). If False, stage1 outputs bf16 and we quantize before stage2.
    stage1_fused_quant: bool = False


_TUNED_CONFIGS = None


def _load_tuned_configs():
    global _TUNED_CONFIGS
    if _TUNED_CONFIGS is not None:
        return _TUNED_CONFIGS

    try:
        tune_file = os.path.join(os.path.dirname(__file__), "configs/tuned_fmoe.csv")
        if not os.path.exists(tune_file):
            _TUNED_CONFIGS = {}
            return _TUNED_CONFIGS

        cfg_2stages = pd.read_csv(tune_file)
        # Ensure columns are correct types if needed
        # We need to match the key structure from fused_moe.py
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
        _TUNED_CONFIGS = cfg_2stages
    except Exception as e:
        logger.warning(f"Failed to load tuned_fmoe.csv: {e}")
        _TUNED_CONFIGS = {}

    return _TUNED_CONFIGS


def get_asm_int8_config(
    token,
    model_dim,
    inter_dim,
    expert,
    topk,
    quant_type,
    dtype,
    is_smoothquant,
) -> AsmInt8Config:
    """
    Get the configuration for ASM Int8 kernel execution.
    It attempts to load tuned configurations from a CSV file or falls back to heuristics.
    """
    config = AsmInt8Config()

    # Try to load from CSV
    tuned_configs = _load_tuned_configs()

    # Construct key
    # We map types to strings as they appear in CSV
    # dtype passed here is int8 if smoothquant is True, but CSV uses model dtype (bf16)
    # We hardcode torch.bfloat16 as this file is bf16 specific
    dtype_str = "torch.bfloat16"
    q_dtype_a_str = "torch.int8"
    q_dtype_w_str = "torch.int8"

    # act_type: ActivationType.Silu -> "ActivationType.Silu"
    act_type_str = "ActivationType.Silu"

    q_type_str = str(quant_type)

    # cu_num
    try:
        cu_num = torch.cuda.get_device_properties(0).multi_processor_count
    except Exception:
        cu_num = 80  # Default fallback

    # doweight_stage1 is 0 based on _asm_moe_2stages_int8
    doweight_stage1 = 0
    use_g1u1 = 1

    key = (
        cu_num,
        token,
        model_dim,
        inter_dim,
        expert,
        topk,
        act_type_str,
        dtype_str,
        q_dtype_a_str,
        q_dtype_w_str,
        q_type_str,
        use_g1u1,
        doweight_stage1,
    )

    if key in tuned_configs:
        entry = tuned_configs[key]
        config.run_2stage = True  # If it is in the CSV, it is likely a 2-stage config
        if "run_1stage" in entry and entry["run_1stage"] == 1:
            config.run_1stage = True
            config.run_2stage = False

        config.block_m = int(entry["block_m"])
        config.ksplit = int(entry["ksplit"])
        config.kernelName1 = entry.get("kernelName1", "")
        config.kernelName2 = entry.get("kernelName2", "")
        # stage1_fused_quant: 1/True = stage1 outputs int8; 0/False or missing = stage1 outputs bf16
        if "stage1_fused_quant" in entry:
            val = entry["stage1_fused_quant"]
            if isinstance(val, bool):
                config.stage1_fused_quant = val
            else:
                config.stage1_fused_quant = bool(int(val))
        return config

    if is_smoothquant and (
        dtype == dtypes.i8 or dtype == torch.int8 or str(dtype) == "torch.int8"
    ):
        # Check if inter_dim is supported by the 2-stage Int8 kernel.
        # block_m must match available tile_m in the multix CSV
        # (smoothquant 2-stage always uses multix path).
        # Available multix kernels: 64x384, 64x320, 64x192, 48x128, 32x128
        if inter_dim % 384 == 0:
            config.run_2stage = True
            config.block_m = 64
            config.ksplit = 0
            config.kernelName1 = ""
            config.kernelName2 = ""
            return config
        elif inter_dim % 320 == 0:
            config.run_2stage = True
            config.block_m = 64
            config.ksplit = 0
            config.kernelName1 = ""
            config.kernelName2 = ""
            return config
        elif inter_dim % 192 == 0:
            config.run_2stage = True
            config.block_m = 64
            config.ksplit = 0
            config.kernelName1 = ""
            config.kernelName2 = ""
            return config
        elif inter_dim % 128 == 0:
            config.run_2stage = True
            config.block_m = 32
            config.ksplit = 0
            config.kernelName1 = ""
            config.kernelName2 = ""
            return config

    config.block_m = 32
    return config


def _asm_moe_2stages_a8(
    M,
    topk,
    inter_dim,
    a8,
    w1,
    w2,
    sorted_ids,
    sorted_expert_ids,
    num_valid_ids,
    moe_buf,
    config: AsmInt8Config,
    a8_scale,
    w1_scale,
    fc2_smooth_scale,
    w2_scale,
    sorted_weights,
    activation,
    topk_ids=None,
    local_expert_hash=None,
):
    """
    2-stage ASM MoE: stage1 (a8 @ w1 -> inter_states) then stage2 (inter_states @ w2 -> moe_buf).
    If config.stage1_fused_quant: stage1 outputs int8 + per-128 scale; stage2 uses them as-is.
    Else: stage1 outputs bf16; we quantize to int8 + (M, topk) scale via smooth_per_token_scaled_quant, then stage2.
    """
    device = a8.device
    doweight_stage1 = False

    if config.stage1_fused_quant:
        # Stage1 outputs int8 into flat buffer; a2_scale is per-128-elements.
        total_tokens = M * topk
        data_size = total_tokens * inter_dim
        scale_size = data_size // 32  # 4 bytes scale per 128 bytes data
        flat_buffer = torch.empty(
            data_size + scale_size, dtype=dtypes.i8, device=device
        )
        inter_states = flat_buffer[:data_size].view(M, topk, inter_dim)
        scale_view = flat_buffer[data_size:].view(torch.float32)

        aiter.moe_stage1_g1u1(
            a8,
            w1,
            w2,
            sorted_ids,
            sorted_expert_ids,
            num_valid_ids,
            inter_states,
            inter_dim,
            config.kernelName1,
            config.block_m,
            config.ksplit,
            activation,
            QuantType.per_Token,
            a8_scale,
            w1_scale,
            None,
            None,
            fc2_smooth_scale,
            w2_scale,
            sorted_weights if doweight_stage1 else None,
        )

        num_scales = inter_dim // 128
        a2_scale = scale_view.view(M, topk, num_scales)
        a2 = inter_states
    else:
        # Stage1 outputs bf16; quantize to a2 (int8) + a2_scale (M, topk) for stage2.
        inter_states = torch.empty(
            (M, topk, inter_dim), dtype=dtypes.bf16, device=device
        )
        a2 = torch.empty((M, topk, inter_dim), dtype=dtypes.i8, device=device)
        a2_scale = torch.empty((M, topk), dtype=dtypes.fp32, device=device)

        aiter.moe_stage1_g1u1(
            a8,
            w1,
            w2,
            sorted_ids,
            sorted_expert_ids,
            num_valid_ids,
            inter_states,
            inter_dim,
            config.kernelName1,
            config.block_m,
            config.ksplit,
            activation,
            QuantType.per_Token,
            a8_scale,
            w1_scale,
            None,
            None,
            fc2_smooth_scale,
            w2_scale,
            sorted_weights if doweight_stage1 else None,
        )

        aiter.smooth_per_token_scaled_quant(
            a2,
            inter_states.view(M, topk, inter_dim),
            a2_scale,
            fc2_smooth_scale,
            topk_ids,
            smooth_scale_map_hash=local_expert_hash,
            enable_ps=True,
        )

    asm_moe_stage2(
        a2,
        w1,
        w2,
        sorted_ids,
        sorted_expert_ids,
        num_valid_ids,
        moe_buf,
        topk,
        config.kernelName2,
        w2_scale,
        a2_scale,
        config.block_m,
        sorted_weights if not doweight_stage1 else None,
        QuantType.per_Token.value,
        activation.value,
        config.ksplit,
    )

    return moe_buf


def _run_asm_moe_int8(
    hidden_states,
    w1,
    w2,
    sorted_ids,
    sorted_weights,
    sorted_expert_ids,
    num_valid_ids,
    topk_ids,
    topk,
    moe_buf,
    config,
    fc1_smooth_scale,
    fc2_smooth_scale,
    w1_scale,
    w2_scale,
    activation,
    per_tensor_quant_scale,
    expert_mask,
    lastdim_mul,
    local_expert_hash=None,
):
    M, _ = topk_ids.shape
    device = topk_ids.device
    _, model_dim, inter_dim = w2.shape

    # Clone topk_ids only when 2-stage and we may overwrite it (fc1_smooth + moe_smoothquant_fwd path).
    use_ref_input_quant = False
    topk_ids_for_scale = (
        topk_ids.clone()
        if (config.run_2stage and fc1_smooth_scale is not None)
        else None
    )
    if expert_mask is not None and local_expert_hash is None:
        local_expert_hash = expert_mask.cumsum(0, dtype=dtypes.i32).clone()
        local_expert_hash[local_expert_hash > 0] -= 1
        local_expert_hash[expert_mask == 0] = -1

    # a8w8 fmoe, opt: smooth quant
    a8_type = (
        w1.dtype if w1.dtype != dtypes.i32 and w1.dtype != torch.uint32 else dtypes.fp8
    )
    is_int8 = w1.dtype == dtypes.i8 or w1.dtype == torch.int8
    if fc1_smooth_scale is not None:
        use_ref_input_quant = config.run_2stage and is_int8
        if use_ref_input_quant:
            # Reference path: smooth_per_token_scaled_quant for input; keep topk_ids (global) for stage2.
            a8 = torch.empty((topk * M, model_dim), dtype=a8_type, device=device)
            a8_scale = torch.empty((topk * M), dtype=dtypes.fp32, device=device)
            aiter.smooth_per_token_scaled_quant(
                a8.view(topk, M, model_dim).transpose(0, 1),
                hidden_states.view(M, 1, model_dim).expand(-1, topk, -1),
                a8_scale,
                fc1_smooth_scale,
                topk_ids,
                smooth_scale_map_hash=local_expert_hash,
                enable_ps=True,
            )
            a8 = a8.view(-1, model_dim).view(topk, M, model_dim)
        else:
            if is_int8:
                a8 = torch.empty((topk, M, model_dim), dtype=a8_type, device=device)
                a8_scale = torch.empty((topk, M, 1), dtype=dtypes.fp32, device=device)
            else:
                a8 = torch.empty((topk * M, model_dim), dtype=a8_type, device=device)
                a8_scale = torch.empty((topk * M), dtype=dtypes.fp32, device=device)
            if expert_mask is not None:
                topk_ids = local_expert_hash[topk_ids]
            aiter.moe_smoothquant_fwd(
                a8, hidden_states, fc1_smooth_scale, topk_ids, a8_scale
            )
    else:
        if w1.dtype == dtypes.fp8 or w1.dtype in (dtypes.i32, torch.uint32):
            a8 = torch.empty((M, model_dim), dtype=a8_type, device=device)
            a8_scale = torch.empty(M, dtype=dtypes.fp32, device=device)
            if per_tensor_quant_scale is None:
                aiter.dynamic_per_token_scaled_quant(a8, hidden_states, a8_scale)
            else:
                aiter.static_per_tensor_quant(a8, hidden_states, per_tensor_quant_scale)
                a8_scale.fill_(per_tensor_quant_scale)
        elif w1.dtype == dtypes.i8:
            a8 = torch.empty((M, model_dim), dtype=w1.dtype, device=device)
            a8_scale = torch.empty(M, dtype=dtypes.fp32, device=device)
            fc1_smooth_scale = torch.ones(model_dim, dtype=dtypes.fp32, device=device)
            aiter.smoothquant_fwd(a8, hidden_states, fc1_smooth_scale, a8_scale)
        else:
            logger.warning("FMOE fall into pure torch quant...")
            a8, a8_scale = aiter.pertoken_quant(hidden_states, quant_dtype=w1.dtype)
    # two stage: both paths handled inside _asm_moe_2stages_a8 via config.stage1_fused_quant
    if config.run_2stage:
        # Pass global topk_ids when we kept it (ref path); else pass cloned original (topk_ids_for_scale or topk_ids if no clone).
        ids_for_scale = (
            topk_ids if use_ref_input_quant else (topk_ids_for_scale or topk_ids)
        )
        return _asm_moe_2stages_a8(
            M,
            topk,
            inter_dim,
            a8,
            w1,
            w2,
            sorted_ids,
            sorted_expert_ids,
            num_valid_ids,
            moe_buf,
            config,
            a8_scale,
            w1_scale,
            fc2_smooth_scale,
            w2_scale,
            sorted_weights,
            activation,
            topk_ids=ids_for_scale,
            local_expert_hash=local_expert_hash,
        )

    # one stage
    if w2.shape[2] * lastdim_mul == w1.shape[1]:
        aiter.fmoe_int8_g1u0(
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
            fc2_smooth_scale,
            activation,
        )
    elif w2.shape[2] * 2 * lastdim_mul == w1.shape[1]:
        kernel_name = ""
        if config.run_1stage and config.kernelName1:
            kernel_name = config.kernelName1
        aiter.fmoe_g1u1(
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
            kernel_name,
            fc2_smooth_scale,
            activation,
        )

    else:
        raise ValueError(f"Invalid MoE weight: {w1.shape=} {w2.shape=} {lastdim_mul}")

    #   fc2_smooth_scale)
    return moe_buf


def _run_asm_moe_bf16(
    moe_buf,
    hidden_states,
    w1,
    w2,
    sorted_ids,
    sorted_weights,
    sorted_expert_ids,
    num_valid_ids,
    topk,
):
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


def _run_asm_moe_a16(
    moe_buf,
    hidden_states,
    w1,
    w2,
    sorted_ids,
    sorted_weights,
    sorted_expert_ids,
    num_valid_ids,
    topk,
    w1_scale,
    w2_scale,
    fc1_smooth_scale,
    fc2_smooth_scale,
    activation,
    inter_dim,
):
    # a16w8 smooth quant fmoe
    if w1.dtype in [dtypes.fp8, dtypes.i8] and inter_dim * 2 == w1.shape[1]:
        aiter.fmoe_g1u1_a16(
            moe_buf,
            hidden_states,
            w1,
            w2,
            sorted_ids,
            sorted_weights,
            sorted_expert_ids,
            num_valid_ids,
            topk,
            w1_scale,
            w2_scale,
            fc1_smooth_scale,
            fc2_smooth_scale,
            activation,
        )
    elif w1.dtype == dtypes.i8 and inter_dim == w1.shape[1]:
        aiter.fmoe_int8_g1u0_a16(
            moe_buf,
            hidden_states,
            w1,
            w2,
            sorted_ids,
            sorted_weights,
            sorted_expert_ids,
            num_valid_ids,
            topk,
            w1_scale,
            w2_scale,
            fc1_smooth_scale,
            fc2_smooth_scale,
        )
    else:
        raise ValueError(f"Invalid args: {w1.dtype} {w1.shape=} {w2.shape=}")


def _run_asm_moe_block_scale(
    moe_buf,
    hidden_states,
    w1,
    w2,
    sorted_ids,
    sorted_weights,
    sorted_expert_ids,
    num_valid_ids,
    topk,
    w1_scale,
    w2_scale,
    block_shape,
    dtype,
    M,
    model_dim,
):
    assert (
        dtype == torch.bfloat16
    ), "asm_moe for block_scale only support bfloat16 hidden_states"
    assert block_shape == (
        128,
        128,
    ), "asm_moe for block_scale only support (128, 128)"
    assert (
        w1.dtype == torch.float8_e4m3fnuz
    ), "asm_moe for block_scale only support float8_e4m3fnuz weight"
    assert w2.shape[2] * 2 == w1.shape[1], "aiter moe for block_scale only support g1u1"
    scale_blk_n, scale_blk_k = block_shape
    hidden_states = hidden_states.view(M * model_dim // scale_blk_k, scale_blk_k)

    a1_q, a1_scale = pertoken_quant(
        hidden_states.view(-1, model_dim // scale_blk_k, scale_blk_k),
        quant_dtype=torch.float8_e4m3fnuz,
    )
    a1_q = a1_q.view(-1, model_dim)
    a1_scale = a1_scale.squeeze(-1).t().contiguous()

    aiter.fmoe_fp8_blockscale_g1u1(
        moe_buf,
        a1_q,
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
        "",
        scale_blk_n,
        scale_blk_k,
        None,
    )


def asm_moe(
    hidden_states,
    w1,  # [expert(local_expert:EP), inter_dim*2, dim] N,K
    w2,  # [expert(local_expert:EP), dim, inter_dim]
    topk_weight,
    topk_ids,
    # following for int8 quant
    fc1_scale=None,  # [expert(local_expert:EP), inter_dim, 1]
    fc2_scale=None,  # [expert(local_expert:EP), model_dim, 1]
    fc1_smooth_scale=None,  # [expert(local_expert:EP), 1, model_dim]
    fc2_smooth_scale=None,  # [expert(local_expert:EP), 1, inter_dim]
    a16=False,
    per_tensor_quant_scale=None,
    block_shape=None,
    expert_mask=None,
    activation=ActivationType.Silu,
    local_expert_hash=None,
):
    # Map legacy parameters to fused_moe naming convention
    w1_scale = fc1_scale
    w2_scale = fc2_scale

    E, model_dim, inter_dim = w2.shape
    global_E = E
    if expert_mask is not None:
        global_E = expert_mask.numel()
    M, topk = topk_ids.shape
    dtype = hidden_states.dtype
    lastdim_mul = 8 if w1.dtype in {dtypes.i32, torch.uint32} else 1

    # Check for 2-stage execution
    is_smoothquant = fc1_smooth_scale is not None

    config = get_asm_int8_config(
        M,
        model_dim,
        inter_dim,
        global_E,
        topk,
        QuantType.per_Token,
        w1.dtype,
        is_smoothquant,
    )

    block_m_sorting = BLOCK_SIZE_M
    if not a16 and config.block_m > 0:
        block_m_sorting = config.block_m

    # Unified sorting logic
    sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_buf = (
        moe_sorting_ck(
            topk_ids,
            topk_weight,
            global_E,
            model_dim,
            dtype,
            block_m_sorting,
            expert_mask,
        )
    )

    if w1_scale is None:
        # pure bf16
        _run_asm_moe_bf16(
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
    elif a16:
        # a16w8 smooth quant fmoe
        _run_asm_moe_a16(
            moe_buf,
            hidden_states,
            w1,
            w2,
            sorted_ids,
            sorted_weights,
            sorted_expert_ids,
            num_valid_ids,
            topk,
            w1_scale,
            w2_scale,
            fc1_smooth_scale,
            fc2_smooth_scale,
            activation,
            inter_dim,
        )
    elif block_shape is not None:
        # a:fp8
        _run_asm_moe_block_scale(
            moe_buf,
            hidden_states,
            w1,
            w2,
            sorted_ids,
            sorted_weights,
            sorted_expert_ids,
            num_valid_ids,
            topk,
            w1_scale,
            w2_scale,
            block_shape,
            dtype,
            M,
            model_dim,
        )
    else:
        # a:int8 w8/4
        return _run_asm_moe_int8(
            hidden_states,
            w1,
            w2,
            sorted_ids,
            sorted_weights,
            sorted_expert_ids,
            num_valid_ids,
            topk_ids,
            topk,
            moe_buf,
            config,
            fc1_smooth_scale,
            fc2_smooth_scale,
            w1_scale,
            w2_scale,
            activation,
            per_tensor_quant_scale,
            expert_mask,
            lastdim_mul,
            local_expert_hash=local_expert_hash,
        )

    return moe_buf


def asm_moe_tkw1(
    hidden_states,
    w1,  # [expert(local_expert:EP), inter_dim*2, dim] N,K
    w2,  # [expert(local_expert:EP), dim, inter_dim]
    topk_weight,
    topk_ids,
    # following for int8 quant
    fc1_scale=None,  # [expert(local_expert:EP), inter_dim, 1]
    fc2_scale=None,  # [expert(local_expert:EP), model_dim, 1]
    fc1_smooth_scale=None,  # [expert(local_expert:EP), 1, model_dim]
    fc2_smooth_scale=None,  # [expert(local_expert:EP), 1, inter_dim]
    a16=False,
    per_tensor_quant_scale=None,
    expert_mask=None,
    activation=ActivationType.Silu,
):
    return fused_moe(
        hidden_states,
        w1,
        w2,
        topk_weight,
        topk_ids,
        expert_mask=expert_mask,
        activation=activation,
        quant_type=QuantType.per_Token,
        doweight_stage1=True,
        w1_scale=fc1_scale,
        w2_scale=fc2_scale,
        a1_scale=fc1_smooth_scale,
        a2_scale=fc2_smooth_scale,
    )


def get_block_size(token, topk, expert):
    token_per_expert = token * topk / expert
    support_list = [32, 64, 128]
    for el in support_list:
        if token_per_expert <= el * 4:
            return el
    return support_list[-1]


# Only support fp8 per tensor quant
def ck_moe_2stages(
    a1,
    w1,  # [expert(local_expert:EP), inter_dim(*2), dim] N,K
    w2,  # [expert(local_expert:EP), dim, inter_dim]
    topk_weight,
    topk_ids,
    # following for int8 quant
    quant_type=QuantType.No,
    fc1_scale=None,  # [expert(local_expert:EP), inter_dim, 1]
    fc2_scale=None,  # [expert(local_expert:EP), model_dim, 1]
    a1_scale=None,  # [1]
    a2_scale=None,  # [1]
    block_size=None,
    expert_mask=None,
    activation=ActivationType.Silu,
    doweight_stage1=False,
):

    quant_func = get_hip_quant(quant_type)
    q_dtype_a = w1.dtype if w1.dtype != torch.uint32 else torch.float8_e4m3fnuz

    E, model_dim, inter_dim = w2.shape
    if w1.dtype is torch.uint32:
        inter_dim = inter_dim * 8

    global_E = E
    if expert_mask is not None:
        global_E = expert_mask.numel()
    M, topk = topk_ids.shape
    dtype = a1.dtype
    device = topk_ids.device
    if block_size is None:
        block_size = get_block_size(M, topk, E)
    sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_buf = (
        moe_sorting_ck(
            topk_ids, topk_weight, global_E, model_dim, dtype, block_size, expert_mask
        )
    )
    # print("block_size:", block_size, sorted_expert_ids)
    a1, a1_scale = quant_func(a1, scale=a1_scale, quant_dtype=q_dtype_a)

    a2 = torch.empty(
        (M, topk, inter_dim),
        dtype=dtype,
        device=device,
    )

    if activation == ActivationType.Silu:
        act_op = 1  # silu_and_mul
    else:
        act_op = 0  # gelu_and_mul

    aiter.ck_moe_stage1(
        a1,
        w1,
        w2,
        sorted_ids,
        sorted_expert_ids,
        num_valid_ids,
        a2,
        topk,
        fc1_scale,
        a1_scale,
        block_size,
        sorted_weights if doweight_stage1 else None,
        act_op,
    )

    if quant_type == QuantType.per_Token:
        a2 = a2.view(M, -1)
    a2, a2_scale = quant_func(a2, scale=a2_scale, quant_dtype=q_dtype_a)
    a2 = a2.view(M, topk, -1)

    aiter.ck_moe_stage2(
        a2,
        w1,
        w2,
        sorted_ids,
        sorted_expert_ids,
        num_valid_ids,
        moe_buf,
        topk,
        fc2_scale,
        a2_scale,
        block_size,
        sorted_weights if not doweight_stage1 else None,
    )
    return moe_buf


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
    if w2.shape[2] * 2 == w1.shape[1]:
        # g1u1(w1 include gate and up)
        moeType = "g1u1"
    else:
        # g1u0(w1 only include gate)
        moeType = "g1u0"

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
            if moeType == "g1u1":
                gate, up = act_input.split([inter_dim, inter_dim], dim=-1)
                if activation == ActivationType.Gelu:
                    act_out = F.gelu(gate) * up
                else:
                    act_out = F.silu(gate) * up
            else:
                if activation == ActivationType.Gelu:
                    act_out = F.gelu(act_input)
                else:
                    act_out = F.silu(act_input)
            if fc2_smooth_scale is not None:
                act_out = act_out * (fc2_smooth_scale[E_id])
            out[mask] = act_out @ (w2[E_id].transpose(0, 1))

    return (out * topk_weight.view(B, -1, 1)).sum(dim=1).to(dtype)


def torch_moe_tkw1(
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
    if w2.shape[2] * 2 == w1.shape[1]:
        # g1u1(w1 include gate and up)
        moeType = "g1u1"
    else:
        # g1u0(w1 only include gate)
        moeType = "g1u0"

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
            if moeType == "g1u1":
                gate, up = act_input.split([inter_dim, inter_dim], dim=-1)
                gate = gate * (topk_weight.view(B, -1, 1)[mask])
                up = up * (topk_weight.view(B, -1, 1)[mask])
                if activation == ActivationType.Gelu:
                    act_out = F.gelu(gate) * up
                else:
                    act_out = F.silu(gate) * up
            else:
                if activation == ActivationType.Gelu:
                    act_out = F.gelu(act_input)
                else:
                    act_out = F.silu(act_input)
            if fc2_smooth_scale is not None:
                act_out = act_out * (fc2_smooth_scale[E_id])
            act_out, act_out_scale = pertoken_quant(
                act_out, quant_dtype=dtypes.fp8, dtypeMax=None
            )
            out[mask] = (
                act_out.to(computeType)
                @ (w2[E_id].transpose(0, 1))
                * act_out_scale.view(-1, 1)
            )

    return out.sum(dim=1).to(dtype)


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

    if topk_weights is None:
        topk_weights = torch.empty(
            M, topk, dtype=dtypes.fp32, device=hidden_states.device
        )
    if topk_ids is None:
        topk_ids = torch.empty(M, topk, dtype=dtypes.i32, device=hidden_states.device)
    token_expert_indicies = torch.empty(
        M, topk, dtype=dtypes.i32, device=hidden_states.device
    )

    aiter.topk_softmax(
        topk_weights,
        topk_ids,
        token_expert_indicies,
        gating_output.float(),  # TODO(woosuk): Optimize this.
        renormalize,
    )
    del token_expert_indicies  # Not used. Will be used in the future.

    return topk_weights, topk_ids
