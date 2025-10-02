# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
import math
import random

import pytest
import torch

import triton
import triton.language as tl
from aiter.test_common import checkAllclose, perftest, benchmark
from aiter.ops.triton.pa_mqa_logits import deepgemm_fp8_paged_mqa_logits_stage1

@triton.jit
def kernel_fp8_paged_mqa_logits_stage1(
    batch_size,
    next_n,
    heads_num,
    Q_buffer,
    stride_q_batch,
    stride_q_next_n,
    stride_q_heads,
    KV_buffer,
    stride_k_seq,
    scale_buffer,
    scale_buffer_stride,
    context_len_ptr,
    kv_indices,
    weights,
    stride_w_batch,
    Out_buffer,
    stride_out_heads,
    stride_out_batch,
    max_model_len,
    max_blk_len,
    ChunkQ: tl.constexpr,
    ChunkK: tl.constexpr,
    HiddenDim: tl.constexpr,
    SplitKV: tl.constexpr = 1,
):
    pid = tl.program_id(0)
    num_block_q_head = tl.cdiv(heads_num, ChunkQ)

    pid_q_head, remain_pid = pid % num_block_q_head, pid // num_block_q_head
    pid_next_n, remain_pid = remain_pid % next_n, remain_pid // next_n
    pid_batch, pid_split_kv = remain_pid % batch_size, remain_pid // batch_size

    context_length = tl.load(context_len_ptr + pid_batch)

    context_chunk_num = tl.cdiv(context_length, ChunkK)
    split_context_chunk_num = tl.cdiv(context_chunk_num, SplitKV)

    split_context_start = (pid_split_kv * split_context_chunk_num) * ChunkK
    split_context_length = min(context_length - split_context_start, split_context_chunk_num * ChunkK)

    q = tl.load(
        Q_buffer
        + pid_batch * stride_q_batch
        + pid_next_n * stride_q_next_n
        + ((pid_q_head * ChunkQ + tl.arange(0, ChunkQ)) * stride_q_heads)[:, None]
        + tl.arange(0, HiddenDim)[None, :],
    )
    scale_weight = tl.load(weights + (pid_batch * next_n + pid_next_n) * stride_w_batch + pid_q_head * ChunkQ + tl.arange(0, ChunkQ))

    for context_idx in range(split_context_start, split_context_start + split_context_length, ChunkK):
        mask_kv = context_idx + tl.arange(0, ChunkK) < context_length
        context_kv_idx = tl.load(kv_indices + pid_batch * max_blk_len + context_idx + tl.arange(0, ChunkK), mask=mask_kv, other=0)

        k = tl.load(
            KV_buffer + context_kv_idx[:, None] * stride_k_seq + tl.arange(0, HiddenDim)[None, :],
            mask=mask_kv[:, None],
            other=0.0,
        )
        k_scale_f = tl.load(scale_buffer + context_kv_idx[:, None] * scale_buffer_stride)

        o = tl.dot(q, k.T)
        o = o * k_scale_f.T
        o = tl.maximum(o, 0.0)
        o = o * scale_weight[None, :].T

        mask = context_idx + tl.arange(0, ChunkK) <= context_length - pid_next_n
        o = tl.where(mask[None, :], o, float("-inf"))

        tl.store(
            Out_buffer
            + (pid_batch * next_n + pid_next_n) * stride_out_batch
            + (pid_q_head * ChunkQ + tl.arange(0, ChunkQ)[:, None, None]) * stride_out_heads
            + (context_idx + tl.arange(0, ChunkK)[None, None, :]),
            o[:, None, :],
        )

def ref_fp8_paged_mqa_logits(q: torch.Tensor, kv_cache: torch.Tensor,
                             weights: torch.Tensor, context_lens: torch.Tensor, block_tables: torch.Tensor,
                             max_model_len: int):
    def cdiv(a: int, b: int) -> int:
        """Ceiling division."""
        return -(a // -b)
    batch_size, next_n, heads, dim = q.size()
    num_block, block_size, _, dim = kv_cache.size()
    logits = torch.full([heads, batch_size * next_n, max_model_len], float('-inf'), device=q.device, dtype=torch.float32)
    context_lens = context_lens.tolist()
    for i in range(batch_size):
        context_len = context_lens[i]
        q_offsets = torch.arange(context_len - next_n, context_len, device='cuda')
        weight_slice = weights[i * next_n:(i + 1) * next_n, :].transpose(0, 1).contiguous()
        for block_rk in range(cdiv(context_len, block_size)):
            block_idx = block_tables[i][block_rk]
            qx, kx = q[i], kv_cache[block_idx]
            k_offsets = torch.arange(block_rk * block_size, (block_rk + 1) * block_size, device='cuda')
            mask = (k_offsets[None, :] < context_len) & (k_offsets[None, :] <= q_offsets[:, None])
            s = torch.where(mask[None, :, :], (qx.transpose(0, 1) @ kx.transpose(0, 1).transpose(1, 2)).to(logits.dtype), float('-inf'))
            s = torch.relu(s) * weight_slice[..., None]
            s = s.sum(dim=0)
            logits[:, i * next_n:(i + 1) * next_n, block_rk * block_size: (block_rk + 1) * block_size] = torch.where(k_offsets[None, None, :] <= q_offsets[None, :, None], s, float('-inf'))
    return logits


def triton_fp8_paged_mqa_logits_stage1(
    q_fp8: torch.Tensor,  # dtype = float8
    kv_cache_fp8: torch.Tensor,  # dtype = float8 [num_blocks, 1, 1, D+4]
    weights: torch.Tensor,  # dtype = float32
    out_qk: torch.Tensor,  # dtype = float32
    context_lens: torch.Tensor,
    kv_indices: torch.Tensor,
    max_model_len: int,
):
    batch_size, next_n, heads, hidden_dim = q_fp8.size()
    _, max_blk_len = kv_indices.size()
    kv_cache_fp8, kv_cache_scale = kv_cache_fp8[..., :hidden_dim], kv_cache_fp8[..., hidden_dim:]
    # Since the triton don't have the reinterpret_cast, we slice the scale out and view it as float
    kv_cache_scale = kv_cache_scale.view(torch.float32)
    kv_cache_fp8 = kv_cache_fp8.view(torch.float8_e4m3fnuz)

    config = {
        "ChunkQ": 32,
        "ChunkK": 64,
        "HiddenDim": hidden_dim,
        "SplitKV": 5,
    }
    assert heads % config["ChunkQ"] == 0

    grid = (batch_size * next_n * (heads // config["ChunkQ"] * config["SplitKV"]),)
    deepgemm_fp8_paged_mqa_logits_stage1[grid](
        batch_size,
        next_n,
        heads,
        q_fp8,
        q_fp8.stride(0),
        q_fp8.stride(1),
        q_fp8.stride(2),
        kv_cache_fp8,
        kv_cache_fp8.stride(0),
        kv_cache_scale,
        kv_cache_scale.stride(0),
        context_lens,
        kv_indices,
        weights,
        weights.stride(0),
        out_qk,
        out_qk.stride(0),
        out_qk.stride(1),
        max_model_len,
        max_blk_len,
        **config,
    )

def kv_cache_cast_to_fp8(x: torch.Tensor) -> torch.Tensor:
    num_blocks, block_size, num_heads, head_dim = x.shape
    assert num_heads == 1
    x_amax = x.abs().float().amax(dim=3, keepdim=True).clamp(1e-4)
    sf = x_amax / 240.0
    x_scaled = (x * (1.0 / sf)).to(torch.float8_e4m3fnuz)
    x_fp8 = torch.empty((num_blocks, block_size * (head_dim + 4)), device=x.device, dtype=torch.uint8)
    x_fp8[ :, : block_size * head_dim] = x_scaled.view(num_blocks, block_size * head_dim).view(dtype=torch.uint8)
    x_fp8[ :, block_size * head_dim :] = sf.view(num_blocks, block_size).view(dtype=torch.uint8)
    return x_fp8.view(num_blocks, block_size, num_heads, head_dim + 4)

def cdiv(x: int, y: int) -> int:
    return (x + y - 1) // y

def test_deepgemm_fp8_paged_mqa_logits():
    torch.manual_seed(0)
    random.seed(0)

    max_model_len = 4096
    num_blocks = 111 * 1000 * 3
    blocksize = 1
    for batch_size, next_n in [(4, 1), (2, 2)]:
        for heads, index_dim in [(32, 128)]:
            for avg_kv in (2048,):
                var_ratio = 0.4
                context_lens = torch.randint(int((1 - var_ratio) * avg_kv), int(((1 + var_ratio)) * avg_kv) + 1, (batch_size,)).cuda().to(torch.int32)
                prefix_sum_context_lens = torch.zeros((batch_size + 1,), device="cuda", dtype=torch.int32)
                prefix_sum_context_lens[1:] = torch.cumsum(context_lens, dim=0)

                print(context_lens)
                print(prefix_sum_context_lens)

                q = torch.randn(
                    (batch_size, next_n, heads, index_dim),
                    device="cuda",
                    dtype=torch.bfloat16,
                )
                kv_cache = torch.randn(
                    (num_blocks, blocksize, 1, index_dim),
                    device='cuda',
                    dtype=torch.bfloat16
                )
                weights = torch.randn(
                    (batch_size * next_n, heads),
                    device="cuda",
                    dtype=torch.float32,
                )
                qk_datatype = torch.float8_e4m3fnuz
                max_block_len = (context_lens.max().item() + blocksize - 1) // blocksize * blocksize
                block_tables = torch.zeros((batch_size, max_block_len), device='cuda', dtype=torch.int32)

                counter = 0
                block_idx_pool = list(range(num_blocks))
                random.shuffle(block_idx_pool)
                for i in range(batch_size):
                    ctx_len = context_lens[i].item()
                    for j in range(cdiv(ctx_len, blocksize)):
                        block_tables[i][j] = block_idx_pool[counter]
                        counter += 1

                q_fp8 = q.to(qk_datatype)
                kv_cache_fp8 = kv_cache_cast_to_fp8(kv_cache)
                # kv_cache_fp8 = kv_cache.to(qk_datatype)

                kv_indices = torch.zeros(prefix_sum_context_lens[-1], device="cuda", dtype=torch.int32)
                for i in range(batch_size):
                    ctx_len = int(context_lens[i].item())
                    kv_indices[prefix_sum_context_lens[i] : prefix_sum_context_lens[i + 1]] = torch.randperm(max_model_len, device="cuda")[:ctx_len]

                ref_qk = ref_fp8_paged_mqa_logits(
                    q,
                    kv_cache,
                    weights,
                    context_lens, 
                    block_tables,
                    max_model_len
                )

                out_qk = torch.full(
                    (heads, batch_size * next_n, max_model_len),
                    float("-inf"),
                    device="cuda",
                    dtype=torch.float32,
                )
                deepgemm_fp8_paged_mqa_logits_stage1(q_fp8, kv_cache_fp8, weights, out_qk, context_lens, block_tables, max_model_len)
                out_qk = torch.sum(out_qk, dim=0)

                positions = torch.arange(max_model_len, device="cuda").unsqueeze(0).expand(heads, batch_size * next_n, -1)
                row_indices = torch.arange(batch_size * next_n, device="cuda") // next_n
                next_n_offset = torch.arange(batch_size * next_n, device="cuda") % next_n
                mask = positions <= (context_lens[row_indices] - next_n + next_n_offset).unsqueeze(1)

                out_qk = out_qk.masked_fill(~mask, 0)
                ref_qk = ref_qk.masked_fill(~mask, 0)

                # aiter_match = checkAllclose(out_qk, ref_qk, atol=1e-2, rtol=1e-2)
                def calc_diff(x: torch.Tensor, y: torch.Tensor):
                    x, y = x.double(), y.double()
                    denominator = (x * x + y * y).sum()
                    sim = 2 * (x * y).sum() / denominator
                    return 1 - sim

                # follow the de
                diff = calc_diff(out_qk, ref_qk)
                assert diff < 1e-3
                print("test pass!")
                # print(diff)


if __name__ == "__main__":
    test_deepgemm_fp8_paged_mqa_logits()
