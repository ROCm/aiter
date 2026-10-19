# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch

from aiter import logger
from aiter.ops.triton._triton_kernels.flash_attn_triton_amd.utils import FP8_ARCHS
from aiter.ops.triton.attention.mha import (
    mha_set_use_fused_bwd_kernel,
)
from aiter.ops.triton.attention.mha_v3 import (
    flash_attn_fp8_func,
    flash_attn_varlen_fp8_func,
)
from aiter.ops.triton.utils._triton.arch_info import get_arch
from aiter.test_mha_common import (
    attention_ref,
    attention_ref_with_tol,
    generate_qkv,
    generate_random_padding_mask,
)

arch = get_arch()

pytestmark = pytest.mark.skipif(
    arch not in FP8_ARCHS, reason=f"FP8 not supported on {arch}"
)


def assert_cosine_similarity(actual, expected, threshold=0.96, norm_floor=1e-3):
    """Assert that two tensors have high cosine similarity."""
    a = actual.float().flatten()
    b = expected.float().flatten()
    # NOTE: cosine similarity is unstable for near-zero tensors
    if b.norm().item() > norm_floor:
        cos_sim = torch.nn.functional.cosine_similarity(
            a.unsqueeze(0), b.unsqueeze(0)
        ).item()
        assert cos_sim >= threshold, f"Cosine similarity {cos_sim:.6f} < {threshold}"


def fp8_assert_close(tensor_a, tensor_b, atol=1.0, cos_sim_threshold=0.96):
    """FP8 quality check: max absolute error + cosine similarity."""
    a = tensor_a.float().flatten()
    b = tensor_b.float().flatten()

    max_abs = (a - b).abs().max().item()
    assert max_abs <= atol, f"Max absolute error {max_abs:.4f} > {atol}"

    assert_cosine_similarity(tensor_a, tensor_b, cos_sim_threshold)


@pytest.mark.parametrize("BATCH", [1, 4])
@pytest.mark.parametrize(
    "SEQLEN_Q, SEQLEN_K",
    [(1, 1), (64, 128), (2048, 2048)],
)
@pytest.mark.parametrize("NUM_Q_HEADS, NUM_K_HEADS", [(1, 1), (48, 8)])
@pytest.mark.parametrize("CAUSAL", [(True), (False)])
def test_mha(
    BATCH: int,
    SEQLEN_Q: int,
    SEQLEN_K: int,
    NUM_Q_HEADS: int,
    NUM_K_HEADS: int,
    CAUSAL: bool,
    dtype=torch.bfloat16,
):
    HEAD_SZ: int = 128

    if CAUSAL and (SEQLEN_Q * SEQLEN_K > 128 * 128):
        pytest.skip(
            "FP8+CAUSAL for big sequence lenghts results in random precision errors"
        )

    torch.cuda.empty_cache()
    torch.manual_seed(20)
    q = torch.randn((BATCH, SEQLEN_Q, NUM_Q_HEADS, HEAD_SZ), device="cuda", dtype=dtype)
    k = torch.randn((BATCH, SEQLEN_K, NUM_K_HEADS, HEAD_SZ), device="cuda", dtype=dtype)
    v = torch.randn((BATCH, SEQLEN_K, NUM_K_HEADS, HEAD_SZ), device="cuda", dtype=dtype)

    triton_out = flash_attn_fp8_func(
        q,
        k,
        v,
        causal=CAUSAL,
    )

    logger.debug("triton_out.shape=%s, triton_out=%s", triton_out.shape, triton_out)

    torch_out = attention_ref(q, k, v, causal=CAUSAL)
    torch_out, attention_scores, _ = torch_out

    logger.debug("torch_out.shape=%s, torch_out=%s", torch_out.shape, torch_out)
    logger.debug(
        "attention_scores.shape=%s, attention_scores=%s",
        attention_scores.shape,
        attention_scores,
    )

    fp8_assert_close(triton_out, torch_out.to(triton_out.dtype))


@pytest.mark.parametrize("BATCH", [1, 4])
@pytest.mark.parametrize(
    "SEQLEN_Q, SEQLEN_K",
    [(1, 1), (64, 128), (2048, 2048)],
)
@pytest.mark.parametrize("NUM_Q_HEADS, NUM_K_HEADS", [(1, 1), (48, 8)])
@pytest.mark.parametrize("CAUSAL", [(True), (False)])
def test_mha_varlen(
    BATCH: int,
    SEQLEN_Q: int,
    SEQLEN_K: int,
    NUM_Q_HEADS: int,
    NUM_K_HEADS: int,
    CAUSAL: bool,
    dtype=torch.bfloat16,
):
    HEAD_SZ: int = 128

    torch.set_printoptions(threshold=10000)
    torch.cuda.empty_cache()
    torch.manual_seed(20)

    q = torch.randn((BATCH, SEQLEN_Q, NUM_Q_HEADS, HEAD_SZ), device="cuda", dtype=dtype)
    k = torch.randn((BATCH, SEQLEN_K, NUM_K_HEADS, HEAD_SZ), device="cuda", dtype=dtype)
    v = torch.randn((BATCH, SEQLEN_K, NUM_K_HEADS, HEAD_SZ), device="cuda", dtype=dtype)
    query_padding_mask = generate_random_padding_mask(
        SEQLEN_Q, BATCH, "cuda", mode="random"
    )
    key_padding_mask = generate_random_padding_mask(
        SEQLEN_K, BATCH, "cuda", mode="random"
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
        _,
        _,
    ) = generate_qkv(q, k, v, query_padding_mask, key_padding_mask, kvpacked=False)

    logger.debug(
        "query_padding_mask.shape=%s query_padding_mask=%s",
        query_padding_mask.shape,
        query_padding_mask,
    )
    logger.debug(
        "key_padding_mask.shape=%s key_padding_mask=%s",
        key_padding_mask.shape,
        key_padding_mask,
    )
    logger.debug("q.shape=%s q=%s", q.shape, q)
    logger.debug("k.shape=%s k=%s", k.shape, k)
    logger.debug("v.shape=%s v=%s", v.shape, v)
    logger.debug("q_unpad.shape=%s q_unpad=%s", q_unpad.shape, q_unpad)
    logger.debug("k_unpad.shape=%s k_unpad=%s", k_unpad.shape, k_unpad)
    logger.debug("v_unpad.shape=%s v_unpad=%s", v_unpad.shape, v_unpad)
    logger.debug("max_seqlens_q=%d", max_seqlen_q)
    logger.debug("max_seqlens_k=%d", max_seqlen_k)
    logger.debug("cu_seqlens_q=%s", cu_seqlens_q)
    logger.debug("cu_seqlens_k=%s", cu_seqlens_k)

    triton_out = flash_attn_varlen_fp8_func(
        q_unpad,
        k_unpad,
        v_unpad,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        causal=CAUSAL,
    )

    triton_out = output_pad_fn(triton_out)

    logger.debug("triton_out.shape=%s, triton_out=%s", triton_out.shape, triton_out)

    torch_out = attention_ref(
        q,
        k,
        v,
        query_padding_mask=query_padding_mask,
        key_padding_mask=key_padding_mask,
        causal=CAUSAL,
    )
    torch_out, attention_scores, _ = torch_out

    logger.debug("torch_out.shape=%s, torch_out=%s", torch_out.shape, torch_out)
    logger.debug(
        "attention_scores.shape=%s, attention_scores=%s",
        attention_scores.shape,
        attention_scores,
    )

    fp8_assert_close(triton_out, torch_out.to(triton_out.dtype))


# Production shapes based on real models:
#   HQ=32, HK=8:  Llama 3 8B (GQA 4:1)
#   HQ=64, HK=8:  Llama 3 70B (GQA 8:1)
#   HQ=32, HK=32: Llama 2 7B (MHA)
@pytest.mark.parametrize("BATCH", [1, 4])
@pytest.mark.parametrize("SEQLEN_Q", [512, 2048])
@pytest.mark.parametrize("SEQLEN_K", [512, 2048])
@pytest.mark.parametrize("NUM_Q_HEADS", [32, 64])
@pytest.mark.parametrize("CAUSAL", [True, False])
@pytest.mark.parametrize("FUSED", [False, True])
def test_mha_backward(
    BATCH: int,
    SEQLEN_Q: int,
    SEQLEN_K: int,
    NUM_Q_HEADS: int,
    CAUSAL: bool,
    FUSED: bool,
    dtype=torch.bfloat16,
):
    HEAD_SZ: int = 128
    NUM_K_HEADS: int = 8

    if FUSED and CAUSAL:
        pytest.skip("FUSED+CAUSAL results in NaNs")
    if CAUSAL:
        pytest.skip("FP8+CAUSAL results in random precision errors")

    torch.cuda.empty_cache()
    torch.manual_seed(20)
    mha_set_use_fused_bwd_kernel(FUSED)

    q = torch.randn(BATCH, SEQLEN_Q, NUM_Q_HEADS, HEAD_SZ, device="cuda", dtype=dtype)
    k = torch.randn(BATCH, SEQLEN_K, NUM_K_HEADS, HEAD_SZ, device="cuda", dtype=dtype)
    v = torch.randn(BATCH, SEQLEN_K, NUM_K_HEADS, HEAD_SZ, device="cuda", dtype=dtype)
    q.requires_grad = True
    k.requires_grad = True
    v.requires_grad = True
    do = torch.randn_like(q)

    # Triton forward + backward
    with torch.enable_grad():
        triton_out = flash_attn_fp8_func(q, k, v, causal=CAUSAL)

    triton_dq, triton_dk, triton_dv = torch.autograd.grad(
        triton_out, (q, k, v), do.clone()
    )

    # Reference forward + backward with adaptive tolerances
    torch_out, torch_grads, fwd_tol, bwd_tols = attention_ref_with_tol(
        q,
        k,
        v,
        do,
        is_fp8=True,
        causal=CAUSAL,
    )
    torch_dq, torch_dk, torch_dv = torch_grads

    # Check quality
    triton_vals = [triton_out, triton_dq, triton_dk, triton_dv]
    ref_vals = [torch_out, torch_dq, torch_dk, torch_dv]
    tols = [fwd_tol] + bwd_tols
    for tri, ref, (atol, rtol) in zip(triton_vals, ref_vals, tols):
        torch.testing.assert_close(tri, ref.to(tri.dtype), atol=atol, rtol=rtol)
        assert_cosine_similarity(tri, ref)


@pytest.mark.parametrize("BATCH", [1, 4])
@pytest.mark.parametrize("SEQLEN_Q", [512, 2048])
@pytest.mark.parametrize("SEQLEN_K", [512, 2048])
@pytest.mark.parametrize("NUM_Q_HEADS", [32, 64])
@pytest.mark.parametrize("CAUSAL", [True, False])
@pytest.mark.parametrize("FUSED", [False, True])
def test_mha_backward_varlen(
    BATCH: int,
    SEQLEN_Q: int,
    SEQLEN_K: int,
    NUM_Q_HEADS: int,
    CAUSAL: bool,
    FUSED: bool,
    dtype=torch.bfloat16,
):
    HEAD_SZ: int = 128
    NUM_K_HEADS: int = 8

    if FUSED and CAUSAL:
        pytest.skip("FUSED+CAUSAL results in NaNs")

    torch.cuda.empty_cache()
    torch.manual_seed(20)
    mha_set_use_fused_bwd_kernel(FUSED)

    q = torch.randn(BATCH, SEQLEN_Q, NUM_Q_HEADS, HEAD_SZ, device="cuda", dtype=dtype)
    k = torch.randn(BATCH, SEQLEN_K, NUM_K_HEADS, HEAD_SZ, device="cuda", dtype=dtype)
    v = torch.randn(BATCH, SEQLEN_K, NUM_K_HEADS, HEAD_SZ, device="cuda", dtype=dtype)
    q.requires_grad = True
    k.requires_grad = True
    v.requires_grad = True

    query_padding_mask = generate_random_padding_mask(
        SEQLEN_Q, BATCH, "cuda", mode="random"
    )
    key_padding_mask = generate_random_padding_mask(
        SEQLEN_K, BATCH, "cuda", mode="random"
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

    q_unpad.requires_grad = True
    k_unpad.requires_grad = True
    v_unpad.requires_grad = True
    do = torch.randn_like(q)

    # Triton varlen forward + backward
    with torch.enable_grad():
        triton_out = flash_attn_varlen_fp8_func(
            q_unpad,
            k_unpad,
            v_unpad,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            causal=CAUSAL,
        )

    triton_out = output_pad_fn(triton_out)
    triton_dq, triton_dk, triton_dv = torch.autograd.grad(
        triton_out, (q_unpad, k_unpad, v_unpad), do.clone()
    )
    triton_dq = dq_pad_fn(triton_dq)
    triton_dk = dk_pad_fn(triton_dk)
    triton_dv = dk_pad_fn(triton_dv)

    # Reference forward + backward with adaptive tolerances
    torch_out, torch_grads, fwd_tol, bwd_tols = attention_ref_with_tol(
        q,
        k,
        v,
        do,
        is_fp8=True,
        query_padding_mask=query_padding_mask,
        key_padding_mask=key_padding_mask,
        causal=CAUSAL,
    )
    torch_dq, torch_dk, torch_dv = torch_grads

    # Check quality
    triton_vals = [triton_out, triton_dq, triton_dk, triton_dv]
    ref_vals = [torch_out, torch_dq, torch_dk, torch_dv]
    tols = [fwd_tol] + bwd_tols
    for tri, ref, (atol, rtol) in zip(triton_vals, ref_vals, tols):
        torch.testing.assert_close(tri, ref.to(tri.dtype), atol=atol, rtol=rtol)
        assert_cosine_similarity(tri, ref)
