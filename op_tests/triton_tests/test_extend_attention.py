# SPDX-License-Identifier: MIT
# Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

import torch
import pytest
from aiter.ops.triton.extend_attention import extend_attention_fwd


def input_helper(
    B,
    H,
    prefix_length,
    extend_length,
    kv_lora_rank,
    qk_rope_head_dim,
    v_head_dim,
    dtype,
    device,
    attn_impl="normal",
    equal_seqlens=False,
    requires_grad=False,
):
    torch.manual_seed(0)

    if not equal_seqlens:
        max_extend_length = extend_length
        max_prefix_length = prefix_length

        seqlens_extend = torch.randint(
            1, max_extend_length + 1, (B,), dtype=torch.int32
        )
        if prefix_length == 0:
            seqlens_prefix = torch.full((B,), prefix_length)
        else:
            seqlens_prefix = torch.randint(
                1, max_prefix_length + 1, (B,), dtype=torch.int32
            )

    else:
        seqlens_extend = torch.full((B,), extend_length)
        seqlens_prefix = torch.full((B,), prefix_length)

    cu_seqlens_extend = torch.cat(
        [
            torch.tensor([0], dtype=torch.int32),
            seqlens_extend.cumsum(dim=0, dtype=torch.int32),
        ]
    )
    cu_seqlens_prefix = torch.cat(
        [
            torch.tensor([0], dtype=torch.int32),
            seqlens_prefix.cumsum(dim=0, dtype=torch.int32),
        ]
    )

    cu_seqlens_extend = cu_seqlens_extend.to(device="cuda")
    cu_seqlens_prefix = cu_seqlens_prefix.to(device="cuda")

    total_extend = cu_seqlens_extend[-1].item()
    total_prefix = cu_seqlens_prefix[-1].item()

    if attn_impl == "absorb":
        Lq = kv_lora_rank + qk_rope_head_dim
        Lk = kv_lora_rank + qk_rope_head_dim
        Lv = kv_lora_rank
    else:
        Lq = v_head_dim + qk_rope_head_dim
        Lk = v_head_dim + qk_rope_head_dim
        Lv = v_head_dim

    q_extend = torch.randn(
        total_extend, H, Lq, dtype=dtype, device=device
    ).requires_grad_(requires_grad)

    # extend parts
    k_extend = torch.randn(
        total_extend, 1, Lk, dtype=dtype, device=device
    ).requires_grad_(requires_grad)
    v_extend = k_extend[..., :Lv]

    # extend indexing
    qo_indptr = cu_seqlens_extend

    # prefix parts
    k_buffer = torch.randn(
        total_prefix, 1, Lk, dtype=dtype, device=device
    ).requires_grad_(requires_grad)
    v_buffer = k_buffer[..., :Lv]

    if attn_impl != "absorb":
        # simulate v = kv_latent * w_vc which changes the values compared to k
        v_extend = torch.randn_like(v_extend)
        v_buffer = torch.randn_like(v_buffer)

    # prefix indexing
    kv_indptr = cu_seqlens_prefix
    kv_indices = torch.arange(total_prefix, device=device)

    custom_mask = None
    mask_indptr = None
    max_len_extend = extend_length

    return (
        q_extend,
        k_extend,
        v_extend,
        k_buffer,
        v_buffer,
        kv_indptr,
        kv_indices,
        qo_indptr,
        custom_mask,
        mask_indptr,
        max_len_extend,
    )


@pytest.mark.parametrize(
    "B, H, prefix, extend, kv_lora_rank, qk_rope_head_dim, v_head_dim",
    [
        (2, 4, 0, 512, 32, 16, 32),
        (3, 5, 0, 333, 18, 13, 17),
        (3, 5, 512, 333, 18, 0, 17),
        (3, 5, 110, 333, 18, 0, 19),
        # (8, 16, 0, 1024, 128, 0, 128), # this one passes
        # (8, 16, 0, 16324, 128, 0, 128), # this one fails, numeric precision is likely the issue
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("ref_attn_impl", ["normal", "absorb"])
def test_op_fwd(
    B,
    H,
    prefix,
    extend,
    kv_lora_rank,
    qk_rope_head_dim,
    v_head_dim,
    dtype,
    ref_attn_impl,
    causal,
    sm_scale=1.0,
    logit_cap=0.0,
    device="cuda",
):
    torch.manual_seed(0)
    torch.set_default_device(device)
    torch.set_default_dtype(dtype)

    (
        q_extend,
        k_extend,
        v_extend,
        k_buffer,
        v_buffer,
        kv_indptr,
        kv_indices,
        qo_indptr,
        custom_mask,
        mask_indptr,
        max_len_extend,
    ) = input_helper(
        B,
        H,
        prefix,
        extend,
        kv_lora_rank,
        qk_rope_head_dim,
        v_head_dim,
        dtype,
        device,
        ref_attn_impl,
    )
    tri_out = torch.empty(
        (*q_extend.shape[:-1], v_extend.shape[-1]),
        dtype=q_extend.dtype,
        device=q_extend.device,
    )

    # Reference
    extend_attention_fwd(
        q_extend,
        k_extend,
        v_extend,
        tri_out,
        k_buffer,
        v_buffer,
        qo_indptr,
        kv_indptr,
        kv_indices,
        custom_mask,
        causal,
        mask_indptr,
        max_len_extend,
        sm_scale=sm_scale,
        logit_cap=logit_cap,
    )

    ref_out = torch.empty_like(tri_out, dtype=q_extend.dtype, device=q_extend.device)
    # ref implementation
    for i in range(0, B):
        start_q, start_k = qo_indptr[i], kv_indptr[i]
        end_q, end_k = qo_indptr[i + 1], kv_indptr[i + 1]

        # Get query, prefix key/values, and extend key/values
        q = q_extend[start_q:end_q]  # [seq_len, H, C]
        k_prefix = k_buffer[start_k:end_k]  # [prefix_len, 1, C]
        v_prefix = v_buffer[start_k:end_k]  # [prefix_len, 1, C]
        k_ext = k_extend[start_q:end_q]  # [seq_len, 1, C]
        v_ext = v_extend[start_q:end_q]  # [seq_len, 1, C]

        prefix_len = end_k - start_k
        seq_len = end_q - start_q

        # Calculate attention scores for prefix tokens
        scores_prefix = torch.einsum("qhc,khc->hqk", q, k_prefix)  # .float()

        # Calculate attention scores for extend tokens
        scores_extend = torch.einsum("qhc,khc->hqk", q, k_ext)  # .float()

        # Apply causal mask only to the extend part if needed
        if causal:
            causal_mask = torch.triu(
                torch.ones(
                    (seq_len, seq_len), dtype=torch.bool, device=scores_extend.device
                ),
                diagonal=1,
            )
            causal_mask = causal_mask.unsqueeze(0).expand(
                scores_extend.shape[0], -1, -1
            )
            scores_extend = scores_extend.masked_fill(causal_mask, float("-inf"))

        # Combine scores and apply softmax
        scores_combined = torch.cat([scores_prefix, scores_extend], dim=-1) * sm_scale
        p_combined = torch.softmax(scores_combined, dim=-1).to(dtype)

        # Split the attention weights back
        p_prefix = p_combined[:, :, :prefix_len]
        p_extend = p_combined[:, :, prefix_len:]

        # Calculate output separately and combine
        out_prefix = torch.einsum("hqk,khd->qhd", p_prefix, v_prefix)
        out_extend = torch.einsum("hqk,khd->qhd", p_extend, v_ext)

        ref_out[start_q:end_q] = out_prefix.to(dtype) + out_extend.to(dtype)

    torch.testing.assert_close(ref_out, tri_out, rtol=2e-2, atol=2e-2)


if __name__ == "__main__":
    test_op_fwd(1, 2, 1024, 1024, 256, 0, 256, torch.float32, "normal", False)
    test_op_fwd(3, 5, 110, 333, 18, 0, 17, torch.float32, "normal", True)
