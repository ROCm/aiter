# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
import math
import random

import pytest
import torch

import triton
import triton.language as tl

from aiter.ops.triton.pa_mqa_logits import deepgemm_fp8_paged_mqa_logits, deepgemm_fp8_paged_mqa_logits_stage1
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
    logits = torch.full([batch_size * next_n, max_model_len], float("-inf"), device=q.device, dtype=torch.float32)

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

        s = s.sum(dim=0)
        logits[i * next_n : (i + 1) * next_n, :context_len] = torch.where(k_offsets[None, :] <= q_offsets[:, None], s, float("-inf"))

    return qk_Head_BatchxMTP_MaxLength, logits


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

                ref_qk, ref_logits = ref_fp8_paged_mqa_logits_stage1(
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
                out_logits = torch.full(
                    (batch_size * next_n, max_model_len),
                    float("-inf"),
                    device="cuda",
                    dtype=torch.float32,
                )
                deepgemm_fp8_paged_mqa_logits_stage1(q_fp8, kv_cache_fp8, weights, out_qk, prefix_sum_context_lens, kv_indices, max_model_len)
                deepgemm_fp8_paged_mqa_logits(q_fp8, kv_cache_fp8, weights, out_logits, prefix_sum_context_lens, kv_indices, max_model_len)

                positions_qk = torch.arange(max_model_len, device="cuda").unsqueeze(0).expand(heads, batch_size * next_n, -1)
                positions_logits = torch.arange(max_model_len, device="cuda").unsqueeze(0).expand(batch_size * next_n, -1)
                row_indices = torch.arange(batch_size * next_n, device="cuda") // next_n
                next_n_offset = torch.arange(batch_size * next_n, device="cuda") % next_n
                mask_qk = positions_qk <= (context_lens[row_indices] - next_n + next_n_offset).unsqueeze(1)
                mask_logits = positions_logits <= (context_lens[row_indices] - next_n + next_n_offset).unsqueeze(1)

                out_qk = out_qk.masked_fill(~mask_qk, 0)
                ref_qk = ref_qk.masked_fill(~mask_qk, 0)

                out_logits = out_logits.masked_fill(~mask_logits, 0)
                ref_logits = ref_logits.masked_fill(~mask_logits, 0)

                checkAllclose(out_qk, ref_qk, atol=1e-2, rtol=1e-2)
                checkAllclose(out_logits, ref_logits, atol=1e-2, rtol=1e-2)


if __name__ == "__main__":
    test_deepgemm_fp8_paged_mqa_logits()
