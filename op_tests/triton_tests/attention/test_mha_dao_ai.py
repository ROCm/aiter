# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import torch
import pytest
from aiter.ops.triton.attention.mha import (
    flash_attn_func,
    flash_attn_varlen_func,
    mha_set_impl,
)
from aiter.test_mha_common import (
    attention_ref,
    generate_random_padding_mask,
    generate_qkv,
)


@pytest.fixture
def dao_ai_impl():
    """Set mha impl to dao_ai for the test, restore to default on cleanup."""
    mha_set_impl("dao_ai")
    yield
    mha_set_impl("default")


@pytest.mark.parametrize(
    "BATCH, SEQLEN_Q, SEQLEN_K, NUM_Q_HEADS, NUM_K_HEADS, HEAD_SZ, CAUSAL, VARLEN, BWD",
    [
        (1, 128, 128, 8, 8, 64, True, False, False),  # fwd causal
        (2, 256, 256, 16, 4, 128, False, False, False),  # fwd GQA non-causal
        (1, 128, 128, 8, 8, 64, True, True, False),  # fwd_varlen causal
        (1, 128, 128, 8, 8, 64, True, False, True),  # bwd causal
        (1, 128, 128, 8, 8, 64, True, True, True),  # bwd_varlen causal
    ],
)
def test_mha_dao_ai(
    dao_ai_impl,
    BATCH: int,
    SEQLEN_Q: int,
    SEQLEN_K: int,
    NUM_Q_HEADS: int,
    NUM_K_HEADS: int,
    HEAD_SZ: int,
    CAUSAL: bool,
    VARLEN: bool,
    BWD: bool,
    dtype=torch.float16,
):
    """Test dao_ai impl dispatch for fwd/bwd x varlen against PyTorch reference."""
    torch.cuda.empty_cache()
    torch.manual_seed(20)

    q = torch.randn(
        BATCH,
        SEQLEN_Q,
        NUM_Q_HEADS,
        HEAD_SZ,
        device="cuda",
        dtype=dtype,
        requires_grad=BWD,
    )
    k = torch.randn(
        BATCH,
        SEQLEN_K,
        NUM_K_HEADS,
        HEAD_SZ,
        device="cuda",
        dtype=dtype,
        requires_grad=BWD,
    )
    v = torch.randn(
        BATCH,
        SEQLEN_K,
        NUM_K_HEADS,
        HEAD_SZ,
        device="cuda",
        dtype=dtype,
        requires_grad=BWD,
    )

    if VARLEN:
        query_padding_mask = generate_random_padding_mask(
            SEQLEN_Q, BATCH, "cuda", mode="full"
        )
        key_padding_mask = generate_random_padding_mask(
            SEQLEN_K, BATCH, "cuda", mode="full"
        )
        (
            q_unpad,
            k_unpad,
            v_unpad,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            q,
            k,
            v,
            output_pad_fn,
            dq_pad_fn,
            dk_pad_fn,
        ) = generate_qkv(q, k, v, query_padding_mask, key_padding_mask, kvpacked=False)
        q_unpad.requires_grad_(BWD)
        k_unpad.requires_grad_(BWD)
        v_unpad.requires_grad_(BWD)

        with torch.set_grad_enabled(BWD):
            triton_out = flash_attn_varlen_func(
                q_unpad,
                k_unpad,
                v_unpad,
                cu_seqlens_q,
                cu_seqlens_k,
                max_seqlen_q,
                max_seqlen_k,
                causal=CAUSAL,
            )
    else:
        query_padding_mask = None
        key_padding_mask = None
        with torch.set_grad_enabled(BWD):
            triton_out = flash_attn_func(q, k, v, causal=CAUSAL)

    # Forward check against PyTorch reference
    q_ref = q.detach().clone().requires_grad_(BWD)
    k_ref = k.detach().clone().requires_grad_(BWD)
    v_ref = v.detach().clone().requires_grad_(BWD)
    torch_out, _, _ = attention_ref(
        q_ref,
        k_ref,
        v_ref,
        causal=CAUSAL,
        query_padding_mask=query_padding_mask,
        key_padding_mask=key_padding_mask,
    )
    if VARLEN:
        triton_out_padded = output_pad_fn(triton_out)
        torch.testing.assert_close(triton_out_padded, torch_out, atol=1e-2, rtol=1e-2)
    else:
        torch.testing.assert_close(triton_out, torch_out, atol=1e-2, rtol=1e-2)

    # Backward check against PyTorch reference
    if BWD:
        do = torch.randn_like(q)
        if VARLEN:
            triton_out = output_pad_fn(triton_out)
            triton_dq, triton_dk, triton_dv = torch.autograd.grad(
                triton_out, (q_unpad, k_unpad, v_unpad), do
            )
            triton_dq = dq_pad_fn(triton_dq)
            triton_dk = dk_pad_fn(triton_dk)
            triton_dv = dk_pad_fn(triton_dv)
        else:
            triton_dq, triton_dk, triton_dv = torch.autograd.grad(
                triton_out, (q, k, v), do
            )

        torch_dq, torch_dk, torch_dv = torch.autograd.grad(
            torch_out, (q_ref, k_ref, v_ref), do
        )

        torch.testing.assert_close(
            triton_dq,
            torch_dq,
            atol=1e-2,
            rtol=1e-2,
            msg=lambda msg: f"dao_ai bwd dq mismatch\n\n{msg}\n",
        )
        torch.testing.assert_close(
            triton_dk,
            torch_dk,
            atol=1e-2,
            rtol=1e-2,
            msg=lambda msg: f"dao_ai bwd dk mismatch\n\n{msg}\n",
        )
        torch.testing.assert_close(
            triton_dv,
            torch_dv,
            atol=1e-2,
            rtol=1e-2,
            msg=lambda msg: f"dao_ai bwd dv mismatch\n\n{msg}\n",
        )
