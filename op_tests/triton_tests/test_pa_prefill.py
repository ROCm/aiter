# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import math
import random
import pytest
import torch
import triton
from aiter.ops.triton.pa_prefill import context_attention_fwd


STR_DTYPE_TO_TORCH_DTYPE = {
    "half": torch.half,
    "bfloat16": torch.bfloat16,
    "float": torch.float,
    "fp8": torch.uint8,
    "fp8_e4m3": torch.uint8,
    "fp8_e5m2": torch.uint8,
}


NUM_HEADS = [64]
NUM_QUERIES_PER_KV = [1, 8, 64]
HEAD_SIZES = [128, 96, 24]
DTYPES = [torch.float16]
CUDA_DEVICES = [f"cuda:{i}" for i in range(1 if torch.cuda.device_count() == 1 else 2)]
SLIDING_WINDOW = [0, 16, 64, 128, 256, 512, 2048]
KV_CACHE_DTYPES = ["auto", "fp8", "fp8_e5m2"]


def context_attention_fwd_torch(
    query,  # [num_tokens, H, D]
    k,  # [num_tokens, Hkv, D]
    v,  # [num_tokens, Hkv, D]
    output,  # [num_tokens, H, D]
    k_cache,  # [B, Hkv, D/8, Blk_sz, 8]
    v_cache,  # [B, Hkv, D, Blk_sz]
    b_start_loc,  # [B+1]
    b_seq_len,  # [B]
    k_scale,
    v_scale,
    alibi_slopes=None,
    sliding_window=None,
    sm_scale=None,
):
    # Setup
    num_blocks = b_seq_len.shape[0]
    head_dim = query.shape[-1]
    num_heads = query.shape[1]
    num_kv_heads = k.shape[1]
    num_queries_per_kv = num_heads // num_kv_heads
    device = query.device

    # Softmax scale fallback
    if sm_scale is None:
        sm_scale = 1.0 / (head_dim**0.5)

    is_kv_cache_fp8 = k_cache.dtype == torch.uint8

    # Cast all inputs to float32
    query = query.to(torch.float32)
    k = k.to(torch.float32)
    v = v.to(torch.float32)
    k_cache = k_cache.to(torch.float32)
    v_cache = v_cache.to(torch.float32)

    for b in range(num_blocks):
        q_start = b_start_loc[b]
        q_end = b_start_loc[b + 1]
        q_len = q_end - q_start
        ctx_len = b_seq_len[b] - q_len

        q = query[q_start:q_end]  # [q_len, H, D]
        k_local = k[q_start:q_end]  # [q_len, Hkv, D]
        v_local = v[q_start:q_end]  # [q_len, Hkv, D]

        for h in range(num_heads):
            kv_h = h // num_queries_per_kv

            qh = q[:, h]  # [q_len, D]

            kc = k_cache[:, kv_h]  # [B, D//8, Blk_sz, 8]
            kc = kc.permute(0, 2, 1, 3).reshape(-1, head_dim)  # [B * Blk_sz, D]
            kc = kc[:ctx_len]  # [ctx_len, D]

            vc = v_cache[:, kv_h]  # [B, D, Blk_sz]
            vc = vc.permute(0, 2, 1).reshape(-1, head_dim)  # [B * Blk_sz, D]
            vc = vc[:ctx_len]  # [ctx_len, D]

            if is_kv_cache_fp8:
                kc = kc * k_scale
                vc = vc * v_scale

            # Compute query against context
            qk_ctx = torch.matmul(qh, kc.T)
            qk_ctx *= sm_scale

            if sliding_window and sliding_window > 0:
                q_pos = torch.arange(ctx_len, ctx_len + q_len, device=device)
                k_pos = torch.arange(ctx_len, device=device)
                rel_dist = q_pos[:, None] - k_pos[None, :]
                qk_ctx = qk_ctx.masked_fill(rel_dist >= sliding_window, -1e4)

            if alibi_slopes is not None:
                alibi_slope = alibi_slopes[h]
                q_pos = torch.arange(ctx_len, ctx_len + q_len, device=device)[:, None]
                k_pos = torch.arange(ctx_len, device=device)[None, :]
                rel_pos = k_pos - q_pos
                alibi_bias = rel_pos.to(torch.float32) * alibi_slope
                mask = (rel_pos <= 0) & (q_pos < b_seq_len[b])
                alibi_bias = torch.where(mask, alibi_bias, float("-inf"))
                qk_ctx += alibi_bias

            p_ctx = torch.softmax(qk_ctx, dim=-1)
            acc = torch.matmul(p_ctx, vc)

            # Compute query against itself (with causal mask)
            kh = k_local[:, kv_h]  # [q_len, D]
            vh = v_local[:, kv_h]  # [q_len, D]

            qk_self = torch.matmul(qh, kh.T)
            qk_self *= sm_scale

            causal_mask = torch.triu(
                torch.ones(q_len, q_len, dtype=torch.bool, device=device), 1
            )
            qk_self = qk_self.masked_fill(causal_mask, float("-inf"))

            if sliding_window and sliding_window > 0:
                q_pos = torch.arange(q_len, device=device)
                k_pos = torch.arange(q_len, device=device)
                rel_dist = q_pos[:, None] - k_pos[None, :]
                qk_self = qk_self.masked_fill(rel_dist >= sliding_window, -10000)

            if alibi_slopes is not None:
                alibi_slope = alibi_slopes[h]
                q_pos = torch.arange(q_len, device=device)[:, None]
                k_pos = torch.arange(q_len, device=device)[None, :]
                rel_pos = k_pos - q_pos
                alibi_bias = rel_pos.to(torch.float32) * alibi_slope
                mask = (rel_pos <= 0) & (q_pos < q_len)
                alibi_bias = torch.where(mask, alibi_bias, float("-inf"))
                qk_self += alibi_bias

            p_self = torch.softmax(qk_self, dim=-1)
            acc_self = torch.matmul(p_self, vh)

            # Output
            acc_total = acc + acc_self
            output[q_start:q_end, h] = acc_total.to(output.dtype)
    return


def _get_alibi_slopes(total_num_heads: int) -> torch.Tensor:
    closest_power_of_2 = 2 ** math.floor(math.log2(total_num_heads))
    base = torch.tensor(
        2 ** (-(2 ** -(math.log2(closest_power_of_2) - 3))),
        dtype=torch.float32,
    )
    powers = torch.arange(1, 1 + closest_power_of_2, dtype=torch.int32)
    slopes = torch.pow(base, powers)

    if closest_power_of_2 != total_num_heads:
        extra_base = torch.tensor(
            2 ** (-(2 ** -(math.log2(2 * closest_power_of_2) - 3))),
            dtype=torch.float32,
        )
        num_remaining_heads = min(
            closest_power_of_2, total_num_heads - closest_power_of_2
        )
        extra_powers = torch.arange(
            start=1, end=1 + 2 * num_remaining_heads, step=2, dtype=torch.int32
        )
        slopes = torch.cat([slopes, torch.pow(extra_base, extra_powers)], dim=0)
    return slopes


def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)


def input_helper(
    BS,
    MAX_SEQ_LEN,
    MAX_CTX_LEN,
    cache_size,
    block_size,
    max_block_per_request,
    num_heads: int,
    head_size: int,
    num_queries_per_kv: int,
    dtype: torch.dtype,
    kv_cache_dtype: str,
    device: str,
    use_alibi_slope: bool,
):
    seed_everything(0)
    torch.set_default_device(device)

    # Need this, otherwise when we capture the graph the process
    # for GPU 1 would run on both GPU0 and GPU1 and things would hang
    #
    # see also similar issue: https://github.com/Dao-AILab/flash-attention/issues/523
    torch.cuda.set_device(device)

    if use_alibi_slope:
        alibi_slopes = _get_alibi_slopes(num_heads).to(device)

    query_lens = [random.randint(16, MAX_SEQ_LEN) for _ in range(BS)]
    ctx_lens = [random.randint(16, MAX_CTX_LEN) for _ in range(BS)]
    seq_lens = [a + b for a, b in zip(query_lens, ctx_lens)]
    num_kv_heads = num_heads // num_queries_per_kv

    num_tokens = sum(query_lens)
    query = torch.empty(num_tokens, num_heads, head_size, dtype=dtype)
    query.uniform_(-1e-3, 1e-3)
    output = torch.empty(num_tokens, num_heads, head_size, dtype=dtype)

    kv = torch.empty(sum(seq_lens), 2, num_kv_heads, head_size, dtype=dtype)
    kv.uniform_(-1e-3, 1e-3)
    key, value = kv.unbind(dim=1)

    if kv_cache_dtype == "auto":
        cache_dtype = dtype
    else:
        cache_dtype = STR_DTYPE_TO_TORCH_DTYPE[kv_cache_dtype]
    k_cache = torch.zeros(
        cache_size, block_size, num_kv_heads, head_size, dtype=cache_dtype
    )
    v_cache = torch.zeros(
        cache_size, block_size, num_kv_heads, head_size, dtype=cache_dtype
    )
    k = torch.zeros(sum(query_lens), num_kv_heads, head_size, dtype=dtype)
    v = torch.zeros(sum(query_lens), num_kv_heads, head_size, dtype=dtype)
    values = torch.arange(0, cache_size, dtype=torch.long)
    values = values[torch.randperm(cache_size)]
    block_table = values[: BS * max_block_per_request].view(BS, max_block_per_request)
    b_seq_len = torch.tensor(seq_lens, dtype=torch.long)
    b_ctx_len = torch.tensor(ctx_lens, dtype=torch.long)
    b_start_loc = torch.cumsum(torch.tensor([0] + query_lens, dtype=torch.long), dim=0)
    max_input_len = MAX_SEQ_LEN
    # copy kv to cache
    b_seq_start_loc = torch.cumsum(
        torch.tensor([0] + seq_lens[:-1], dtype=torch.long), dim=0
    )
    for i in range(BS):
        for j in range(query_lens[i]):
            k[b_start_loc[i] + j].copy_(key[b_seq_start_loc[i] + b_ctx_len[i] + j])
            v[b_start_loc[i] + j].copy_(value[b_seq_start_loc[i] + b_ctx_len[i] + j])
        cur_ctx = 0
        block_id = 0
        while cur_ctx < b_ctx_len[i]:
            start_loc = b_seq_start_loc[i] + cur_ctx
            if cur_ctx + block_size > b_ctx_len[i]:
                end_loc = b_seq_start_loc[i] + b_ctx_len[i]
            else:
                end_loc = start_loc + block_size
            start_slot = block_table[i, block_id] * block_size
            end_slot = start_slot + end_loc - start_loc
            k_cache.view(-1, num_kv_heads, head_size)[start_slot:end_slot].copy_(
                key[start_loc:end_loc]
            )
            v_cache.view(-1, num_kv_heads, head_size)[start_slot:end_slot].copy_(
                value[start_loc:end_loc]
            )
            cur_ctx += block_size
            block_id += 1
    # transpose K_cache[num_blocks, block_size, num_kv_heads, head_size]
    # to K_cache[num_blocks, num_kv_heads, head_size/8, block_size, 8]
    k_cache = (
        k_cache.view(-1, block_size, num_kv_heads, head_size // 8, 8)
        .permute(0, 2, 3, 1, 4)
        .contiguous()
    )
    # transpose V_cache[num_blocks, block_size, num_kv_heads, head_size]
    # to V_cache[num_blocks, num_kv_heads, head_size, block_size]
    v_cache = (
        v_cache.view(-1, block_size, num_kv_heads, head_size)
        .permute(0, 2, 3, 1)
        .contiguous()
    )
    k_scale = v_scale = torch.tensor(1.0, dtype=torch.float32, device=device)

    if use_alibi_slope:
        return (
            query,
            k,
            v,
            output,
            k_cache,
            v_cache,
            block_table,
            b_start_loc,
            b_seq_len,
            max_input_len,
            k_scale,
            v_scale,
            alibi_slopes,
        )
    else:
        return (
            query,
            k,
            v,
            output,
            k_cache,
            v_cache,
            block_table,
            b_start_loc,
            b_seq_len,
            max_input_len,
            k_scale,
            v_scale,
            None,
        )


@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("num_queries_per_kv", NUM_QUERIES_PER_KV)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("kv_cache_dtype", KV_CACHE_DTYPES)
@pytest.mark.parametrize("device", CUDA_DEVICES)
@pytest.mark.parametrize("sliding_window", SLIDING_WINDOW)
@torch.inference_mode()
def test_contexted_kv_attention(
    num_heads: int,
    num_queries_per_kv: int,
    head_size: int,
    sliding_window: int,
    dtype: torch.dtype,
    kv_cache_dtype: str,
    device: str,
) -> None:

    torch.cuda.empty_cache()  # Helps avoid hangs in large tests
    (
        query,
        k,
        v,
        output,
        k_cache,
        v_cache,
        block_table,
        b_start_loc,
        b_seq_len,
        max_input_len,
        k_scale,
        v_scale,
        _,
    ) = input_helper(
        BS=10,
        MAX_SEQ_LEN=1024,
        MAX_CTX_LEN=1024,
        cache_size=640,
        block_size=32,
        max_block_per_request=64,
        num_heads=num_heads,
        head_size=head_size,
        num_queries_per_kv=num_queries_per_kv,
        dtype=dtype,
        kv_cache_dtype=kv_cache_dtype,
        device=device,
        use_alibi_slope=False,
    )
    output_torch = torch.empty_like(output)
    output_triton = output

    # Run Triton
    context_attention_fwd(
        query,
        k,
        v,
        output_triton,
        kv_cache_dtype,
        k_cache,
        v_cache,
        block_table,
        b_start_loc,
        b_seq_len,
        max_input_len,
        k_scale,
        v_scale,
        sliding_window=sliding_window,
    )
    # Run Torch
    context_attention_fwd_torch(
        query,
        k,
        v,
        output_torch,
        k_cache,
        v_cache,
        b_start_loc,
        b_seq_len,
        k_scale,
        v_scale,
        sliding_window=sliding_window,
    )

    triton.testing.assert_close(output_triton, output_torch, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("num_queries_per_kv", NUM_QUERIES_PER_KV)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("kv_cache_dtype", KV_CACHE_DTYPES)
@pytest.mark.parametrize("device", CUDA_DEVICES)
@torch.inference_mode()
def test_contexted_kv_attention_alibi(
    num_heads: int,
    num_queries_per_kv: int,
    head_size: int,
    dtype: torch.dtype,
    kv_cache_dtype: str,
    device: str,
) -> None:
    torch.cuda.empty_cache()  # Helps avoid hangs in large tests
    (
        query,
        k,
        v,
        output,
        k_cache,
        v_cache,
        block_table,
        b_start_loc,
        b_seq_len,
        max_input_len,
        k_scale,
        v_scale,
        alibi_slopes,
    ) = input_helper(
        BS=10,
        MAX_SEQ_LEN=1024,
        MAX_CTX_LEN=1024,
        cache_size=640,
        block_size=32,
        max_block_per_request=64,
        num_heads=num_heads,
        head_size=head_size,
        num_queries_per_kv=num_queries_per_kv,
        dtype=dtype,
        kv_cache_dtype=kv_cache_dtype,
        device=device,
        use_alibi_slope=True,
    )
    output_torch = torch.empty_like(output)
    output_triton = output

    # Run Triton
    context_attention_fwd(
        query,
        k,
        v,
        output_triton,
        kv_cache_dtype,
        k_cache,
        v_cache,
        block_table,
        b_start_loc,
        b_seq_len,
        max_input_len,
        k_scale,
        v_scale,
        alibi_slopes=alibi_slopes,
    )
    # Run Torch
    context_attention_fwd_torch(
        query,
        k,
        v,
        output_torch,
        k_cache,
        v_cache,
        b_start_loc,
        b_seq_len,
        k_scale,
        v_scale,
        alibi_slopes=alibi_slopes,
    )

    triton.testing.assert_close(output_triton, output_torch, atol=1e-2, rtol=1e-2)
