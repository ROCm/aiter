# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import torch
from aiter import QuantType, dtypes, ActivationType
from aiter.fused_moe import fused_topk

try:
    from aiter.ops.triton.moe.moe_op_e2e import (
        e2e_moe as triton_e2e_moe,
        moe_set_use_persistent_kernel as triton_e2e_moe_set_persistent,
    )
    from aiter.ops.triton.utils.types import torch_to_triton_dtype
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False


def torch_moe_align_block_size_ref(topk_ids, block_size, num_experts):
    def _moe_align_block_size(topk_ids, num_experts, top_k, block_size, sorted_token_ids, expert_ids, num_tokens_post_pad):
        M, top_k = topk_ids.shape
        expert_to_tokens = [[] for _ in range(num_experts)]
        for token_id in range(M):
            for j in range(top_k):
                e_id = topk_ids[token_id, j].item()
                expert_to_tokens[e_id].append(token_id * top_k + j)
        reordered_token_ids = []
        reordered_expert_ids = []
        for e_id in range(num_experts):
            tokens_for_expert = expert_to_tokens[e_id]
            num_tokens = len(tokens_for_expert)
            n_blocks = (num_tokens + block_size - 1) // block_size
            padded_size = n_blocks * block_size
            reordered_token_ids.extend(tokens_for_expert)
            reordered_expert_ids.extend([e_id] * n_blocks)
            if padded_size > num_tokens:
                pad_count = padded_size - num_tokens
                reordered_token_ids.extend([topk_ids.numel()] * pad_count)
        token_length = len(reordered_token_ids)
        expert_length = len(reordered_expert_ids)
        sorted_token_ids[:token_length] = torch.tensor(reordered_token_ids, dtype=sorted_token_ids.dtype, device=sorted_token_ids.device)
        expert_ids[:expert_length] = torch.tensor(reordered_expert_ids, dtype=expert_ids.dtype, device=expert_ids.device)
        if token_length < sorted_token_ids.numel():
            sorted_token_ids[token_length:] = topk_ids.numel()
        if expert_length < expert_ids.numel():
            expert_ids[expert_length:] = topk_ids.numel()
        num_tokens_post_pad.fill_(token_length)
    
    top_k = topk_ids.shape[1]
    sorted_ids = torch.empty((topk_ids.numel() + num_experts * (block_size - 1),), dtype=torch.int32, device=topk_ids.device)
    expert_ids = torch.empty((topk_ids.numel() + num_experts,), dtype=torch.int32, device=topk_ids.device)
    sorted_ids.fill_(topk_ids.numel())
    num_tokens_post_pad = torch.empty((1), dtype=torch.int32, device=topk_ids.device)
    _moe_align_block_size(topk_ids, num_experts, top_k, block_size, sorted_ids, expert_ids, num_tokens_post_pad)
    return sorted_ids, expert_ids, num_tokens_post_pad


def generate_data_triton_1stage(token, model_dim, inter_dim, expert, topk, dtype, blockM, device="cuda"):
    torch.manual_seed(0)
    input = torch.randn((token, model_dim), dtype=dtype, device=device) / 10
    w1 = torch.randn((expert, inter_dim * 2, model_dim), dtype=dtype, device=device) / 10
    w2 = torch.randn((expert, model_dim, inter_dim), dtype=dtype, device=device)
    score = torch.randn((token, expert), dtype=dtype, device=device)
    topk_weights, topk_ids = fused_topk(input, score, topk, True)
    sorted_token_ids, expert_ids, num_tokens_post_padded = torch_moe_align_block_size_ref(topk_ids, blockM, expert)
    return (input, input, w1, w2, sorted_token_ids, topk_weights, expert_ids, num_tokens_post_padded, None, None, None, w1, w2, topk_weights, topk_ids)


def run_triton_1stage(hidden_states, a1_qt, w1_qt, w2_qt, sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, w1_scale, a1_scale, w2_scale, topk_ids, topk, config, dtype, persistent):
    if not TRITON_AVAILABLE:
        raise RuntimeError("Triton kernels not available")
    triton_e2e_moe_set_persistent(persistent)
    token_num = a1_qt.shape[0]
    model_dim = hidden_states.shape[1]
    out = torch.zeros((token_num, topk, model_dim), dtype=dtype, device=a1_qt.device)
    intermediate = None
    if persistent:
        inter_dim = w2_qt.shape[2]
        intermediate = torch.zeros((token_num * topk, inter_dim), dtype=torch.float32, device=a1_qt.device)
    out = triton_e2e_moe(a1_qt, w1_qt, w2_qt, intermediate, out, a1_scale, w1_scale, w2_scale, sorted_weights, sorted_ids, topk_ids, sorted_expert_ids, num_valid_ids, 0, topk, False, False, config)
    if out.dim() == 3:
        out = out.sum(dim=1)
    return out


def torch_triton_e2e_moe_ref(a, w1, w2, topk_ids, topk_weights, dtype):
    M, top_k = topk_ids.shape
    a_expanded = a.unsqueeze(1).repeat(1, top_k, 1)
    w1_indexed = w1[topk_ids]
    intermediate = torch.einsum("mek,menk->men", a_expanded.to(dtype), w1_indexed.to(dtype))
    from aiter.ops.triton.utils.moe_common import torch_silu_and_mul_ref
    silu_out = torch_silu_and_mul_ref(intermediate.view(-1, intermediate.shape[-1]))
    silu_out = silu_out.view(M, top_k, -1)
    w2_indexed = w2[topk_ids]
    c = torch.einsum("mek,menk->men", silu_out.to(dtype), w2_indexed.to(dtype))
    return c.sum(dim=1)


def gen_triton_1stage_task(info):
    if not TRITON_AVAILABLE:
        return []
    cu_num, token, model_dim, inter_dim, expert, topk, act_type, dtype, q_dtype_a, q_dtype_w, q_type, use_g1u1, doweight_stage1 = info
    if q_type != QuantType.No or not use_g1u1 or act_type != ActivationType.Silu or doweight_stage1 == 1:
        return []
    tasks_triton = []
    block_m_values = [16, 32, 64]
    for persistent in [False, True]:
        for blockM in block_m_values:
            if persistent:
                for N1 in [128]:
                    for N2 in [64]:
                        for K in [64]:
                            config = {"BLOCK_SIZE_M": blockM, "BLOCK_SIZE_N1": N1, "BLOCK_SIZE_N2": N2, "BLOCK_SIZE_K1": K, "BLOCK_SIZE_K2": K}
                            kernel_name = f"triton_e2e_moe_persistent_M{blockM}_N1-{N1}_N2-{N2}_K{K}"
                            tasks_triton.append(((info, "triton_1stage", kernel_name, blockM), generate_data_triton_1stage, (token, model_dim, inter_dim, expert, topk, dtype, blockM), run_triton_1stage, ([0, 1, 2, 3, 4, 5, 6, 7, 9, 8, 10, 14], topk, config, dtype, persistent), {}, torch_triton_e2e_moe_ref, ([0, 2, 3, 14, 13], dtype), {}, (None), 0.01, 0.2, True))
            else:
                for N in [64, 128, 256]:
                    for K1 in [32, 64]:
                        for K2 in [32, 64]:
                            for GROUP_M in [1, 2]:
                                config = {"BLOCK_SIZE_M": blockM, "BLOCK_SIZE_N": N, "BLOCK_SIZE_K1": K1, "BLOCK_SIZE_K2": K2, "GROUP_SIZE_M": GROUP_M}
                                kernel_name = f"triton_e2e_moe_non_persistent_M{blockM}_N{N}_K1-{K1}_K2-{K2}_GM{GROUP_M}"
                                tasks_triton.append(((info, "triton_1stage", kernel_name, blockM), generate_data_triton_1stage, (token, model_dim, inter_dim, expert, topk, dtype, blockM), run_triton_1stage, ([0, 1, 2, 3, 4, 5, 6, 7, 9, 8, 10, 14], topk, config, dtype, persistent), {}, torch_triton_e2e_moe_ref, ([0, 2, 3, 14, 13], dtype), {}, (None), 0.01, 0.2, True))
    return tasks_triton
