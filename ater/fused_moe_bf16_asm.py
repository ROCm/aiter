import torch
import torch.nn.functional as F
import numpy as np
import sys
import os
from typing import Any, Callable, Dict, Optional, Tuple
import ater
BLOCK_SIZE_M = 32


def moe_sorting_ck(topk_ids, topk_weights, num_experts, model_dim, moebuf_dtype):
    block_size = BLOCK_SIZE_M
    device = topk_ids.device
    M, topk = topk_ids.shape
    topk = topk_ids.shape[1]
    max_num_tokens_padded = topk_ids.numel() + num_experts * block_size - topk
    max_num_m_blocks = int((max_num_tokens_padded+block_size-1)//block_size)
    sorted_ids = torch.empty((max_num_tokens_padded, ),
                             dtype=torch.int32,
                             device=device)
    sorted_weights = torch.empty((max_num_tokens_padded, ),
                                 dtype=torch.float,
                                 device=device)
    sorted_expert_ids = torch.empty((max_num_m_blocks, ),
                                    dtype=torch.int32,
                                    device=device)
    num_tokens_post_pad = torch.empty((1),
                                      dtype=torch.int32,
                                      device=device)
    moe_buf = torch.empty((M, model_dim),
                          dtype=moebuf_dtype,
                          device=device)
    ater.moe_sorting_fwd(topk_ids, topk_weights, sorted_ids, sorted_weights,  sorted_expert_ids,
                         num_tokens_post_pad, moe_buf, num_experts, BLOCK_SIZE_M)
    return sorted_ids, sorted_weights, sorted_expert_ids, num_tokens_post_pad, moe_buf


def asm_moe(hidden_states, w1, w2, topk_weight, topk_ids,
            # following for int8 quant
            fc1_scale=None,  # [expert, inter_dim, 1]
            fc2_scale=None,  # [expert, model_dim, 1]
            fc1_smooth_scale=None,  # [expert, 1, model_dim]
            fc2_smooth_scale=None,  # [expert, 1, inter_dim]
            a16=False
            ):
    E, _, model_dim = w1.shape
    M, topk = topk_ids.shape
    dtype = hidden_states.dtype
    device = topk_ids.device
    sorted_ids, sorted_weights, sorted_expert_ids, num_tokens_post_padded, moe_buf = moe_sorting_ck(topk_ids, topk_weight, E,
                                                                                                    model_dim, dtype)
    if fc1_scale is None:
        ater.fmoe(moe_buf, hidden_states, w1, w2, sorted_ids,
                  sorted_weights, sorted_expert_ids, num_tokens_post_padded, topk)
    elif a16:
        ater.fmoe_int8_g1u0_a16(moe_buf, hidden_states, w1, w2, sorted_ids,
                                sorted_weights, sorted_expert_ids, num_tokens_post_padded,
                                topk,
                                fc1_scale,
                                fc2_scale,
                                fc1_smooth_scale,
                                fc2_smooth_scale)
    else:
        a8 = torch.empty((topk * M, model_dim),
                         dtype=torch.int8, device=device)
        a8_scale = torch.empty((topk * M), dtype=torch.float, device=device)

        ater.moe_smoothquant_fwd(
            a8, hidden_states, fc1_smooth_scale, topk_ids, a8_scale)
        ater.fmoe_int8_g1u0(moe_buf, a8, w1, w2, sorted_ids,
                            sorted_weights, sorted_expert_ids, num_tokens_post_padded,
                            topk,
                            a8_scale,
                            fc1_scale,
                            fc2_scale,
                            fc2_smooth_scale)
    return moe_buf


def torch_moe(hidden_states, w1, w2, topk_weight, topk_ids,
              # following for int8 quant
              fc1_scale=None,  # [expert, inter_dim, 1]
              fc2_scale=None,  # [expert, model_dim, 1]
              fc1_smooth_scale=None,  # [expert, 1, model_dim]
              fc2_smooth_scale=None,  # [expert, 1, inter_dim]
              ):
    B, D = hidden_states.shape
    topk = topk_weight.shape[1]
    dtype = hidden_states.dtype
    hidden_states = hidden_states.view(
        B, -1, D).repeat(1, topk, 1)
    out = torch.zeros(
        (B, topk, D),
        dtype=dtype,
        device=hidden_states.device,
    )
    # gose to quant D_w8a8/w8a8
    if fc1_scale is not None:
        expert = w1.shape[0]
        w2D = w2.shape[-1]
        w1 = (w1.view(-1, D).to(fc1_scale) *
              fc1_scale.view(-1, 1)).to(dtype).view(expert, -1, D)
        w2 = (w2.view(-1, w2D).to(fc2_scale) *
              fc2_scale.view(-1, 1)).to(dtype).view(expert, -1, w2D)
    if fc1_smooth_scale is not None:
        expert = fc1_smooth_scale.shape[0]
        fc1_smooth_scale = fc1_smooth_scale.view(expert, -1).to(dtype)
        fc2_smooth_scale = fc2_smooth_scale.view(expert, -1).to(dtype)

    for E_id in range(w1.shape[0]):
        mask = topk_ids == E_id
        if mask.sum():
            sub_tokens = hidden_states[mask]
            if fc1_smooth_scale is not None:
                sub_tokens = sub_tokens * (
                    fc1_smooth_scale[E_id])
            act_input = sub_tokens @ (w1[E_id].transpose(0, 1))
            act_out = F.gelu(act_input)
            if fc2_smooth_scale is not None:
                act_out = act_out * (
                    fc2_smooth_scale[E_id])
            out[mask] = act_out @ (w2[E_id].transpose(0, 1))

    return (
        out * topk_weight.view(B, -1, 1).to(out.dtype)
    ).sum(dim=1)
