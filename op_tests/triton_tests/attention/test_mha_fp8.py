# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import torch
import pytest
import logging
from aiter.ops.triton.attention.mha import (
    mha_set_use_fused_bwd_kernel,
)
from aiter.ops.triton.attention.mha_v3 import (
    flash_attn_fp8_func,
    flash_attn_varlen_fp8_func,
)
from aiter.test_mha_common import (
    attention_ref,
    attention_ref_with_tol,
    generate_random_padding_mask,
    generate_qkv,
)

from aiter.ops.triton.utils._triton.arch_info import get_arch
from aiter.ops.triton._triton_kernels.flash_attn_triton_amd.utils import FP8_ARCHS

arch = get_arch()

pytestmark = pytest.mark.skipif(
    arch not in FP8_ARCHS, reason=f"FP8 not supported on {arch}"
)


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
DEBUG_MODE = False


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

    if DEBUG_MODE:
        print(f"triton_out.shape={triton_out.shape}, triton_out={triton_out}")

    torch_out = attention_ref(q, k, v, causal=CAUSAL)
    torch_out, attention_scores, _ = torch_out

    if DEBUG_MODE:
        print(f"torch_out.shape={torch_out.shape}, torch_out={torch_out}")
        print(
            f"attention_scores.shape={attention_scores.shape}, attention_scores={attention_scores}"
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

    # TO DO: remove once Triton/LLVM async-copy compiler issue is fixed
    if arch == "gfx950" and CAUSAL:
        pytest.skip(
            "Known gfx950 FP8 varlen MHA compiler crash with async copy enabled"
        )

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

    if DEBUG_MODE:
        print(
            f"query_padding_mask.shape={query_padding_mask.shape} query_padding_mask={query_padding_mask}"
        )
        print(
            f"key_padding_mask.shape={key_padding_mask.shape} key_padding_mask={key_padding_mask}"
        )

        print(f"q.shape={q.shape} q={q}")
        print(f"k.shape={k.shape} k={k}")
        print(f"v.shape={v.shape} v={v}")
        print(f"q_unpad.shape={q_unpad.shape} q_unpad={q_unpad}")
        print(f"k_unpad.shape={k_unpad.shape} k_unpad={k_unpad}")
        print(f"v_unpad.shape={v_unpad.shape} v_unpad={v_unpad}")
        print(f"max_seqlens_q={max_seqlen_q }")
        print(f"max_seqlens_k={max_seqlen_k }")
        print(f"cu_seqlens_q={cu_seqlens_q }")
        print(f"cu_seqlens_k={cu_seqlens_k }")

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

    if DEBUG_MODE:
        print(f"triton_out.shape={triton_out.shape}, triton_out={triton_out}")

    torch_out = attention_ref(
        q,
        k,
        v,
        query_padding_mask=query_padding_mask,
        key_padding_mask=key_padding_mask,
        causal=CAUSAL,
    )
    torch_out, attention_scores, _ = torch_out

    if DEBUG_MODE:
        print(f"torch_out.shape={torch_out.shape}, torch_out={torch_out}")
        print(
            f"attention_scores.shape={attention_scores.shape}, attention_scores={attention_scores}"
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

    # TO DO: Remove  once the Triton/LLVM async-copy compiler issue is fixed
    if arch == "gfx950" and (
        (CAUSAL and not FUSED) or (not CAUSAL and NUM_Q_HEADS == 32)
    ):
        pytest.skip(
            "Known gfx950 FP8 backward varlen MHA compiler crash with async copy enabled"
        )

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
