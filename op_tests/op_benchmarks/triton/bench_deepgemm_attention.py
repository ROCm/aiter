# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
import math
import random

import pytest
import torch

import triton
import triton.language as tl
from aiter.test_common import checkAllclose, perftest, benchmark


def ref_fp8_paged_mqa_logits_stage1(
    q: torch.Tensor,  # dtype = float8
    kv_cache: torch.Tensor,  # dtype = float8
    weights: torch.Tensor,  # dtype = float32
    prefix_sum_context_lens: torch.Tensor,
    kv_indices: torch.Tensor,
    max_model_len: int,
):
    batch_size, next_n, heads, dim = q.size()
    seq_kv, _, dim = kv_cache.size()  # 3d

    qk_Head_BatchxMTP_MaxLength = torch.full(
        [heads, batch_size * next_n, max_model_len],
        float("-inf"),
        device=q.device,
        dtype=torch.float32,
    )
    prefix_sum_context_lens = prefix_sum_context_lens.tolist()
    for i in range(batch_size):
        context_len = prefix_sum_context_lens[i + 1] - prefix_sum_context_lens[i]

        q_offsets = torch.arange(context_len - next_n, context_len, device="cuda")

        qx, kx = q[i], kv_cache[kv_indices[prefix_sum_context_lens[i] : prefix_sum_context_lens[i + 1]]]
        k_offsets = torch.arange(0, context_len, device="cuda")
        mask = k_offsets[None, :] <= q_offsets[:, None]

        s = torch.where(
            mask[None, :, :],
            (qx.transpose(0, 1) @ kx.transpose(0, 1).transpose(1, 2)).to(qk_Head_BatchxMTP_MaxLength.dtype),
            float("-inf"),
        )
        weight_slice = weights[i * next_n : (i + 1) * next_n, :].transpose(0, 1).contiguous()
        s = torch.relu(s) * weight_slice[..., None]
        qk_Head_BatchxMTP_MaxLength[:, i * next_n : (i + 1) * next_n, :context_len] = s

        # following part is finished by framework later
        #   s = s.sum(dim=0)
        #   logits[i * next_n : (i + 1) * next_n, :context_len] = torch.where(k_offsets[None, :] <= q_offsets[:, None], s, float("-inf"))

    return qk_Head_BatchxMTP_MaxLength


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
    prefix_sum_context_lens,
    kv_indices,
    weights,
    stride_w_batch,
    Out_buffer,
    stride_out_heads,
    stride_out_batch,
    max_model_len,
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

    context_start = tl.load(prefix_sum_context_lens + pid_batch)
    context_end = tl.load(prefix_sum_context_lens + pid_batch + 1)

    context_length = context_end - context_start
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
        context_kv_idx = tl.load(kv_indices + context_start + context_idx + tl.arange(0, ChunkK), mask=mask_kv, other=0)

        k = tl.load(
            KV_buffer + context_kv_idx[:, None] * stride_k_seq + tl.arange(0, HiddenDim)[None, :],
            mask=mask_kv[:, None],
            other=0.0,
        )

        o = tl.dot(q, k.T)
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


def triton_fp8_paged_mqa_logits_stage1(
    q_fp8: torch.Tensor,  # dtype = float8
    kv_cache_fp8: torch.Tensor,  # dtype = float8
    weights: torch.Tensor,  # dtype = float32
    out_qk: torch.Tensor,  # dtype = float32
    prefix_sum_context_lens: torch.Tensor,
    kv_indices: torch.Tensor,
    max_model_len: int,
):
    batch_size, next_n, heads, hidden_dim = q_fp8.size()

    config = {
        "ChunkQ": 32,
        "ChunkK": 64,
        "HiddenDim": hidden_dim,
        "SplitKV": 5,
    }
    assert heads % config["ChunkQ"] == 0

    grid = (batch_size * next_n * (heads // config["ChunkQ"] * config["SplitKV"]),)
    kernel_fp8_paged_mqa_logits_stage1[grid](
        batch_size,
        next_n,
        heads,
        q_fp8,
        q_fp8.stride(0),
        q_fp8.stride(1),
        q_fp8.stride(2),
        kv_cache_fp8,
        kv_cache_fp8.stride(0),
        prefix_sum_context_lens,
        kv_indices,
        weights,
        weights.stride(0),
        out_qk,
        out_qk.stride(0),
        out_qk.stride(1),
        max_model_len,
        **config,
    )


def test_deepgemm_fp8_paged_mqa_logits():
    torch.manual_seed(0)
    random.seed(0)

    max_model_len = 4096
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
                    dtype=torch.float32,
                )
                kv_cache = torch.randn(
                    (max_model_len, 1, index_dim),
                    device="cuda",
                    dtype=torch.float32,
                )
                weights = torch.randn(
                    (batch_size * next_n, heads),
                    device="cuda",
                    dtype=torch.float32,
                )
                qk_datatype = torch.float8_e4m3fnuz

                q_fp8 = q.to(qk_datatype)
                kv_cache_fp8 = kv_cache.to(qk_datatype)

                kv_indices = torch.zeros(prefix_sum_context_lens[-1], device="cuda", dtype=torch.int32)
                for i in range(batch_size):
                    ctx_len = int(context_lens[i].item())
                    kv_indices[prefix_sum_context_lens[i] : prefix_sum_context_lens[i + 1]] = torch.randperm(max_model_len, device="cuda")[:ctx_len]

                ref_qk = ref_fp8_paged_mqa_logits_stage1(
                    # convert qk type back to float32 to make sure reference use the same data in calculation
                    q_fp8.to(torch.float32),
                    kv_cache_fp8.to(torch.float32),
                    weights,
                    prefix_sum_context_lens,
                    kv_indices,
                    max_model_len,
                )

                out_qk = torch.full(
                    (heads, batch_size * next_n, max_model_len),
                    float("-inf"),
                    device="cuda",
                    dtype=torch.float32,
                )
                triton_fp8_paged_mqa_logits_stage1(q_fp8, kv_cache_fp8, weights, out_qk, prefix_sum_context_lens, kv_indices, max_model_len)

                positions = torch.arange(max_model_len, device="cuda").unsqueeze(0).expand(heads, batch_size * next_n, -1)
                row_indices = torch.arange(batch_size * next_n, device="cuda") // next_n
                next_n_offset = torch.arange(batch_size * next_n, device="cuda") % next_n
                mask = positions <= (context_lens[row_indices] - next_n + next_n_offset).unsqueeze(1)

                out_qk = out_qk.masked_fill(~mask, 0)
                ref_qk = ref_qk.masked_fill(~mask, 0)

                aiter_match = checkAllclose(out_qk, ref_qk, atol=1e-2, rtol=1e-2)
                print(aiter_match)


if __name__ == "__main__":
    test_deepgemm_fp8_paged_mqa_logits()
