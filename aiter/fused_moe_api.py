# SPDX-License-Identifier: MIT
# Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

import torch
import torch.nn.functional as F
import numpy as np
import sys
import os
from typing import Any, Callable, Dict, Optional, Tuple
import aiter
from aiter import logger
from aiter.fused_moe_bf16_asm import asm_moe


def aiter_moe(hidden_states,  # not quant
              w1,  # [expert(local_expert:EP), inter_dim*2, dim] N,K
              w2,  # [expert(local_expert:EP), dim, inter_dim]
              topk_weight, topk_ids,
              # following for int8 quant
              fc1_scale=None,  # [expert(local_expert:EP), inter_dim, 1]
              fc2_scale=None,  # [expert(local_expert:EP), model_dim, 1]
              fc1_smooth_scale=None,  # [expert(local_expert:EP), 1, model_dim]
              fc2_smooth_scale=None,  # [expert(local_expert:EP), 1, inter_dim]
              a16=False,
              acitvation=None,
              per_tensor_quant_scale=None,
              block_shape=None,
              expert_mask=None,
              ):
    useInt4Weight = True if w1.dtype in [torch.int32, torch.uint32] else False
    lastdim_mul = 8 if useInt4Weight else 1
    g1u1 = True if w1.shape[1] == w2.shape[2] * 2 * lastdim_mul else False
    dtype = hidden_states.dtype
    if acitvation is None:
        acitvation = 'silu' if g1u1 else 'gelu'
    assert acitvation in ['silu', 'gelu'], "aiter moe only support silu and gelu activation,\
        by default, 'silu' is used for g1u1 and 'gelu' is used for g1u0"

    if a16 == True:
        assert dtype == torch.bfloat16, "aiter a16 asm_moe only support bfloat16 hidden_states"
        assert w2.shape[2] % 512 == 0 or w2.shape[2] % 320 == 0, "aiter a16 asm_moe only support w2.shape[2] % 512 == 0 or w2.shape[2] % 320 == 0"
        assert (g1u1 and w1.dtype == torch.float8_e4m3fnuz) or (not g1u1 and w1.dtype ==
                                                                torch.int8), "aiter a16 asm_moe only support g1u1 with fp8 or g1u0 with int8"
        assert fc1_smooth_scale is not None and fc2_smooth_scale is not None, "aiter a16 asm_moe need smoothquant(per channel)"
        assert fc1_scale is not None and fc2_scale is not None, "aiter a16 asm_moe need w_scale(per channel)"
        assert per_tensor_quant_scale is None, "aiter a16 asm_moe not support per_tensor_quant_scale"
        return asm_moe(hidden_states, w1, w2, topk_weight, topk_ids, fc1_scale, fc2_scale, 
                       fc1_smooth_scale, fc2_smooth_scale, True, None, expert_mask=expert_mask)
    
    elif useInt4Weight:
        assert dtype == torch.bfloat16, "aiter a8wint4 asm_moe only support bfloat16 hidden_states"
        assert g1u1, "aiter a8wint4 asm_moe only support g1u1"
        assert fc1_smooth_scale is None and fc2_smooth_scale is None, "aiter a8wint4 asm_moe not support smoothquant"
        return asm_moe(hidden_states, w1, w2, topk_weight, topk_ids, fc1_scale, fc2_scale, 
                       fc1_smooth_scale, fc2_smooth_scale, False, per_tensor_quant_scale, expert_mask=expert_mask, activation=acitvation)
    
    elif block_shape is not None:
        assert dtype == torch.bfloat16, "aiter moe for block_scale only support bfloat16 hidden_states"
        assert block_shape == (
            128, 128), "aiter moe for block_scale only support (128, 128)"
        assert fc1_smooth_scale is None and fc2_smooth_scale is None, "aiter moe for block_scale not support smoothquant"
        assert per_tensor_quant_scale is None, "aiter moe for block_scale not support per_tensor_quant_scale"
        assert g1u1, "aiter moe for block_scale only support g1u1"
        assert acitvation == 'silu', "aiter moe for block_scale only support silu acitvation"
        return asm_moe(hidden_states, w1, w2, topk_weight, topk_ids, fc1_scale, fc2_scale, 
                       fc1_smooth_scale, fc2_smooth_scale, False, None, block_shape=block_shape, expert_mask=expert_mask)
    
    elif fc1_smooth_scale is not None and fc2_smooth_scale is not None and w1.dtype in [torch.float8_e4m3fnuz, torch.int8]:
        assert dtype == torch.bfloat16, "aiter asm_moe for smoothquant only support bfloat16 hidden_states"
        if g1u1:
            assert acitvation == 'silu', "aiter asm_moe for g1u1 smoothquant only support silu acitvation"
        else:
            assert acitvation == 'gelu', "aiter asm_moe for g1u0 smoothquant only support gelu acitvation"
        assert g1u1 or (not g1u1 and w1.dtype ==
                        torch.int8), "aiter asm_moe for smoothquant not support g1u0 fp8 smoothquant"
        return asm_moe(hidden_states, w1, w2, topk_weight, topk_ids, fc1_scale, fc2_scale, 
                       fc1_smooth_scale, fc2_smooth_scale, False, per_tensor_quant_scale, expert_mask=expert_mask)
    
    elif fc1_smooth_scale is None and fc2_smooth_scale is None and w1.dtype in [torch.float8_e4m3fnuz, torch.int8]:
        assert dtype == torch.bfloat16, "aiter asm_moe for fp8/int8 quant only support bfloat16 hidden_states"
        assert g1u1, "aiter asm_moe for fp8/int8 quant only support g1u1"
        assert acitvation == 'silu', "aiter asm_moe for fp8/int8 quant only support silu acitvation"
        return asm_moe(hidden_states, w1, w2, topk_weight, topk_ids, fc1_scale, fc2_scale, 
                       fc1_smooth_scale, fc2_smooth_scale, False, per_tensor_quant_scale, expert_mask=expert_mask)
    
    elif fc1_scale is None and fc2_scale is None:
        assert fc1_smooth_scale is None and fc2_smooth_scale is None, "aiter moe for no quant not support smoothquant"
        assert per_tensor_quant_scale is None, "aiter moe for no quant not support per_tensor_quant_scale"
        if not g1u1 and acitvation == 'gelu':
            return asm_moe(hidden_states, w1, w2, topk_weight, topk_ids, fc1_scale, fc2_scale, 
                           fc1_smooth_scale, fc2_smooth_scale, False, per_tensor_quant_scale, expert_mask=expert_mask)
        else:
            block_m = 32
            return aiter.ck_moe(hidden_states, w1, w2, topk_weight, topk_ids, fc1_scale, fc2_scale, 
                                fc1_smooth_scale, fc2_smooth_scale, block_m, expert_mask, acitvation)
