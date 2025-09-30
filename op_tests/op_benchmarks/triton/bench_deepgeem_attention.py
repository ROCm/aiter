# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import math
import random

import pytest
import torch


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
        weight_slice = weights[i * next_n : (i + 1) * next_n, :].transpose(0, 1).contiguous()
        qx, kx = q[i], kv_cache[kv_indices[prefix_sum_context_lens[i] : prefix_sum_context_lens[i + 1]]]
        k_offsets = torch.arange(0, context_len, device="cuda")
        mask = k_offsets[None, :] <= q_offsets[:, None]
        s = torch.where(
            mask[None, :, :],
            (qx.transpose(0, 1) @ kx.transpose(0, 1).transpose(1, 2)).to(qk_Head_BatchxMTP_MaxLength.dtype),
            float("-inf"),
        )
        s = torch.relu(s) * weight_slice[..., None]
        qk_Head_BatchxMTP_MaxLength[:, i * next_n : (i + 1) * next_n, :context_len] = s

        # following part is finished by framework later
        #   s = s.sum(dim=0)
        #   logits[i * next_n : (i + 1) * next_n, :context_len] = torch.where(k_offsets[None, :] <= q_offsets[:, None], s, float("-inf"))

    return qk_Head_BatchxMTP_MaxLength


def test_deepgemm_fp8_paged_mqa_logits():
    torch.manual_seed(0)
    random.seed(0)

    max_model_len = 4096
    for batch_size, next_n in [(4, 1), (2, 2)]:
        for heads, index_dim in [(32, 128)]:
            for avg_kv in (2048,):
                context_lens = torch.randint(int(0.8 * avg_kv), int(1.2 * avg_kv), (batch_size,)).cuda().to(torch.int32)
                prefix_sum_context_lens = torch.zeros((batch_size + 1,), device="cuda", dtype=torch.int32)
                prefix_sum_context_lens[1:] = torch.cumsum(context_lens, dim=0)

                q = torch.randn(
                    (batch_size, next_n, heads, index_dim),
                    device="cuda",
                    dtype=torch.bfloat16,
                )
                kv_cache = torch.randn(
                    (max_model_len, 1, index_dim),
                    device="cuda",
                    dtype=torch.bfloat16,
                )
                weights = torch.randn(
                    (batch_size * next_n, heads),
                    device="cuda",
                    dtype=torch.float32,
                )

                kv_indices = torch.zeros(prefix_sum_context_lens[-1], device="cuda", dtype=torch.int32)
                for i in range(batch_size):
                    ctx_len = int(context_lens[i].item())
                    kv_indices[prefix_sum_context_lens[i] : prefix_sum_context_lens[i + 1]] = torch.randperm(ctx_len, device="cuda")

                ref_qk = ref_fp8_paged_mqa_logits_stage1(
                    q,
                    kv_cache,
                    weights,
                    prefix_sum_context_lens,
                    kv_indices,
                    max_model_len,
                )

                # q_fp8 = q.to(torch.float8_e4m3fn)
                # kv_cache_fp8 = kv_cache_cast_to_fp8(kv_cache)

                out_qk = torch.empty(
                    (heads, batch_size * next_n, max_model_len),
                    device="cuda",
                    dtype=torch.float32,
                )
                #

                print(ref_qk.shape)

                positions = torch.arange(max_model_len, device="cuda").unsqueeze(0).expand(batch_size * next_n, -1)
                row_indices = torch.arange(batch_size * next_n, device="cuda") // next_n
                next_n_offset = torch.arange(batch_size * next_n, device="cuda") % next_n
                mask = positions <= (context_lens[row_indices] - next_n + next_n_offset).unsqueeze(1)

                out_qk = out_qk.masked_fill(~mask, 0)
                ref_qk = ref_qk.masked_fill(~mask, 0)

                diff = torch.abs(out_qk - ref_qk)
                avg_diff = torch.mean(diff)
                max_diff = torch.max(diff)

                print("avg_diff:", avg_diff.item())
                print("max_diff:", max_diff.item())


if __name__ == "__main__":
    test_deepgemm_fp8_paged_mqa_logits()
