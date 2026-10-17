# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch

from aiter import logger
from aiter.ops.triton._triton_kernels.flash_attn_triton_amd.utils import FP8_ARCHS
from aiter.ops.triton.attention.mha import (
    flash_attn_varlen_func,
    mha_set_use_fused_bwd_kernel,
)
from aiter.ops.triton.attention.mha_v3 import (
    flash_attn_fp8_func,
    flash_attn_varlen_fp8_func,
)
from aiter.ops.triton.utils._triton.arch_info import get_arch
from aiter.ops.triton.utils.types import get_fp8_e4m3_dtype
from aiter.test_mha_common import (
    attention_ref,
    attention_ref_with_tol,
    generate_qkv,
    generate_random_padding_mask,
    quantize_fp8_per_bh,
)
from op_tests.triton_tests.attention.mha_test_utils import skip_if_gluon_unsupported

arch = get_arch()

# The mha_v3 FP8 API (high-precision inputs, quantized inside) runs on
# FP8_ARCHS; gfx1250 FP8 goes through the Gluon backend tests below.
requires_mha_v3_fp8 = pytest.mark.skipif(
    arch not in FP8_ARCHS, reason=f"mha_v3 FP8 not supported on {arch}"
)
requires_gfx1250 = pytest.mark.skipif(
    arch != "gfx1250", reason="covers the gfx1250 Gluon MHA kernel"
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


@requires_mha_v3_fp8
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


@requires_mha_v3_fp8
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


def _assert_lse_close(lse, lse_ref):
    # Rows with no visible key (causal with SEQLEN_Q > SEQLEN_K) are -inf.
    finite = torch.isfinite(lse_ref)
    assert torch.equal(torch.isfinite(lse), finite)
    torch.testing.assert_close(lse[finite], lse_ref[finite], atol=1e-2, rtol=1e-2)


def _dequantize_fp8_per_bh(x_fp8, descale, cu_seqlens):
    """Inverse of quantize_fp8_per_bh for thd input: the values the kernel sees."""
    seqlens = (cu_seqlens[1:] - cu_seqlens[:-1]).long()
    batch_idx = torch.repeat_interleave(
        torch.arange(seqlens.numel(), device=x_fp8.device), seqlens
    )
    return x_fp8.float() * descale[batch_idx].unsqueeze(-1)


# FP8 with sinks, sliding windows (including a zero-left window) and LSE, which
# the other FP8 Gluon tests don't cover. The reference runs on the dequantized
# inputs, so the tolerances only cover the kernel itself (mostly P rounded to
# FP8).
@requires_gfx1250
@pytest.mark.parametrize("BATCH", [1, 4])
@pytest.mark.parametrize("SEQLEN_Q, SEQLEN_K", [(128, 128), (100, 300), (300, 100)])
@pytest.mark.parametrize("NUM_Q_HEADS, NUM_K_HEADS", [(8, 8), (16, 4)])
@pytest.mark.parametrize("HEAD_SZ", [64, 128])
@pytest.mark.parametrize(
    "CAUSAL, WINDOW_SIZE_LEFT, SINK",
    [
        (False, -1, False),
        (True, -1, False),
        (True, -1, True),
        (True, 32, True),
        (False, 0, False),
        (True, 0, True),
    ],
)
def test_mha_varlen_fp8_sink_window_gluon(
    BATCH: int,
    SEQLEN_Q: int,
    SEQLEN_K: int,
    NUM_Q_HEADS: int,
    NUM_K_HEADS: int,
    HEAD_SZ: int,
    CAUSAL: bool,
    WINDOW_SIZE_LEFT: int,
    SINK: bool,
):
    skip_if_gluon_unsupported("gluon", head_dim=HEAD_SZ, v_head_dim=HEAD_SZ)

    torch.manual_seed(20)
    torch.cuda.empty_cache()
    q = torch.randn((BATCH, SEQLEN_Q, NUM_Q_HEADS, HEAD_SZ), device="cuda")
    k = torch.randn((BATCH, SEQLEN_K, NUM_K_HEADS, HEAD_SZ), device="cuda")
    v = torch.randn((BATCH, SEQLEN_K, NUM_K_HEADS, HEAD_SZ), device="cuda")
    sink = torch.randn((NUM_Q_HEADS,), device="cuda") if SINK else None
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
        _,
        _,
        _,
        output_pad_fn,
        dq_pad_fn,
        dk_pad_fn,
    ) = generate_qkv(q, k, v, query_padding_mask, key_padding_mask, kvpacked=False)

    fp8_dtype = get_fp8_e4m3_dtype()
    q_fp8, q_descale = quantize_fp8_per_bh(q_unpad.detach(), fp8_dtype, cu_seqlens_q)
    k_fp8, k_descale = quantize_fp8_per_bh(k_unpad.detach(), fp8_dtype, cu_seqlens_k)
    v_fp8, v_descale = quantize_fp8_per_bh(v_unpad.detach(), fp8_dtype, cu_seqlens_k)

    gluon_out, gluon_lse = flash_attn_varlen_func(
        q_fp8,
        k_fp8,
        v_fp8,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        causal=CAUSAL,
        window_size=(WINDOW_SIZE_LEFT, -1),
        return_lse=True,
        sink=sink,
        backend="gluon",
        q_descale=q_descale,
        k_descale=k_descale,
        v_descale=v_descale,
    )
    torch_out, _, torch_lse = attention_ref(
        dq_pad_fn(_dequantize_fp8_per_bh(q_fp8, q_descale, cu_seqlens_q)),
        dk_pad_fn(_dequantize_fp8_per_bh(k_fp8, k_descale, cu_seqlens_k)),
        dk_pad_fn(_dequantize_fp8_per_bh(v_fp8, v_descale, cu_seqlens_k)),
        query_padding_mask=query_padding_mask,
        key_padding_mask=key_padding_mask,
        causal=CAUSAL,
        window_size=(WINDOW_SIZE_LEFT, -1),
        sink=sink,
    )
    seqlens_q = (cu_seqlens_q[1:] - cu_seqlens_q[:-1]).tolist()
    torch_lse = torch.cat(
        [torch_lse[b, :, :n].transpose(0, 1) for b, n in enumerate(seqlens_q)]
    )

    assert gluon_out.dtype == torch.bfloat16
    fp8_assert_close(
        output_pad_fn(gluon_out), torch_out, atol=0.25, cos_sim_threshold=0.999
    )
    _assert_lse_close(gluon_lse, torch_lse)


# Production shapes based on real models:
#   HQ=32, HK=8:  Llama 3 8B (GQA 4:1)
#   HQ=64, HK=8:  Llama 3 70B (GQA 8:1)
#   HQ=32, HK=32: Llama 2 7B (MHA)
@requires_mha_v3_fp8
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


@requires_mha_v3_fp8
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
