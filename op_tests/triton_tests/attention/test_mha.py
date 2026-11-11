# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.


import math

import pytest
import torch

from aiter import logger
from aiter.ops.triton._gluon_kernels.gfx950.attention.mha import (
    _get_config as _get_gluon_config,
)
from aiter.ops.triton._triton_kernels.flash_attn_triton_amd import flash_attn_2
from aiter.ops.triton.attention.mha import (
    _flash_attn_forward,
    flash_attn_func,
    flash_attn_varlen_func,
    mha_set_impl,
    mha_set_use_fused_bwd_kernel,
    mha_set_use_int64_strides,
)
from aiter.ops.triton.utils._triton.arch_info import get_arch, is_fp8_avail
from aiter.ops.triton.utils.types import get_fp8_e4m3_dtype
from aiter.test_mha_common import (
    attention_ref,
    attention_ref_with_tol,
    generate_qkv,
    generate_random_padding_mask,
    quantize_fp8_per_bh,
)
from op_tests.triton_tests.attention.mha_test_utils import (
    pad_rearrange_dropout_mask,
    skip_if_gluon_unsupported,
    skip_if_triton_padded_head_miscompiled,
)


@pytest.fixture(params=[False, True])
def FUSED(request):
    mha_set_use_fused_bwd_kernel(request.param)
    yield request.param
    mha_set_use_fused_bwd_kernel(False)


def _test_mha_impl(
    BATCH: int,
    SEQLEN_Q: int,
    SEQLEN_K: int,
    NUM_Q_HEADS: int,
    NUM_K_HEADS: int,
    HEAD_SZ: int,
    DROPOUT: float,
    RETURN_LSE: bool,
    RETURN_SOFTMAX: bool,
    CAUSAL: bool,
    backend: str = "triton",
    dtype=torch.bfloat16,
):
    skip_if_gluon_unsupported(backend, dropout_p=DROPOUT)

    torch.manual_seed(20)
    torch.cuda.empty_cache()
    q = torch.randn((BATCH, SEQLEN_Q, NUM_Q_HEADS, HEAD_SZ), device="cuda", dtype=dtype)
    k = torch.randn((BATCH, SEQLEN_K, NUM_K_HEADS, HEAD_SZ), device="cuda", dtype=dtype)
    v = torch.randn((BATCH, SEQLEN_K, NUM_K_HEADS, HEAD_SZ), device="cuda", dtype=dtype)

    dropout_mask = None
    triton_out = flash_attn_func(
        q,
        k,
        v,
        dropout_p=DROPOUT,
        causal=CAUSAL,
        return_lse=RETURN_LSE,
        return_attn_probs=RETURN_SOFTMAX,
        backend=backend,
    )

    lse = None
    sd_mask = None
    if RETURN_LSE or RETURN_SOFTMAX:
        lse = triton_out[1] if RETURN_LSE else None
        if RETURN_SOFTMAX:
            sd_mask = triton_out[2] if RETURN_LSE else triton_out[1]
        triton_out = triton_out[0]
        if lse is not None:
            logger.debug("lse.shape=%s, lse=%s", lse.shape, lse)

    if DROPOUT > 0.0 and RETURN_SOFTMAX:
        dropout_mask = sd_mask >= 0
        logger.debug("sd_mask.shape=%s, sd_mask=%s", sd_mask.shape, sd_mask)
        logger.debug(
            "dropout_mask.shape=%s, dropout_mask=%s",
            dropout_mask.shape,
            dropout_mask,
        )

    logger.debug("triton_out.shape=%s, triton_out=%s", triton_out.shape, triton_out)

    torch_out = attention_ref(
        q, k, v, dropout_p=DROPOUT, dropout_mask=dropout_mask, causal=CAUSAL
    )
    torch_out, attention_scores, lse_ref = torch_out
    logger.debug("torch_out.shape=%s, torch_out=%s", torch_out.shape, torch_out)
    logger.debug(
        "attention_scores.shape=%s, attention_scores=%s",
        attention_scores.shape,
        attention_scores,
    )

    torch.testing.assert_close(triton_out, torch_out, atol=1e-2, rtol=1e-2)

    if RETURN_LSE:
        # Fully-masked causal rows are -inf in the reference (and Triton) and 0
        # in Gluon; compare only rows that attended at least one key.
        finite = torch.isfinite(lse_ref)
        torch.testing.assert_close(
            lse[finite].float(), lse_ref[finite].float(), atol=1e-2, rtol=1e-2
        )
    if RETURN_SOFTMAX:
        # Triton only writes S_dmask when dropout is on; Gluon fills it at
        # dropout_p == 0 as well. The returned score is not the same as the reference because they
        # are not adjusted as new maxes per block are found. So we just check the
        # shape and dtype.
        if sd_mask is None:
            assert backend == "triton" and DROPOUT == 0.0
        else:
            assert sd_mask.dtype == torch.float32
            assert sd_mask.shape == (BATCH, NUM_Q_HEADS, SEQLEN_Q, SEQLEN_K)


@pytest.mark.parametrize("BATCH", [1, 30, 50])
@pytest.mark.parametrize(
    "SEQLEN_Q, SEQLEN_K",
    [(1, 1), (128, 128), (32, 16), (64, 128), (2048, 2048)],
)
@pytest.mark.parametrize("NUM_Q_HEADS, NUM_K_HEADS", [(1, 1), (8, 1), (48, 8)])
@pytest.mark.parametrize("HEAD_SZ", [33, 64, 128])
@pytest.mark.parametrize("CAUSAL", [(True), (False)])
@pytest.mark.parametrize("backend", ["triton", "gluon"])
def test_mha(
    BATCH: int,
    SEQLEN_Q: int,
    SEQLEN_K: int,
    NUM_Q_HEADS: int,
    NUM_K_HEADS: int,
    HEAD_SZ: int,
    CAUSAL: bool,
    backend: str,
    dtype=torch.bfloat16,
):
    skip_if_triton_padded_head_miscompiled(
        backend, HEAD_SZ, CAUSAL, SEQLEN_K, NUM_K_HEADS
    )
    _test_mha_impl(
        BATCH,
        SEQLEN_Q,
        SEQLEN_K,
        NUM_Q_HEADS,
        NUM_K_HEADS,
        HEAD_SZ,
        DROPOUT=0.0,
        RETURN_LSE=False,
        RETURN_SOFTMAX=False,
        CAUSAL=CAUSAL,
        backend=backend,
        dtype=dtype,
    )


def _fp8_assert_close(actual, expected, atol=1.0, cos_sim_threshold=0.94):
    """fp8 comparison: bounded max-abs error plus cosine similarity.

    fp8 attention has larger elementwise error than bf16, so we check direction
    (cosine similarity) in addition to a bounded absolute error.
    """
    a = actual.float().flatten()
    b = expected.float().flatten()
    max_abs = (a - b).abs().max().item()
    assert max_abs <= atol, f"Max absolute error {max_abs:.4f} > {atol}"
    if b.norm().item() > 1e-3:
        cos_sim = torch.nn.functional.cosine_similarity(
            a.unsqueeze(0), b.unsqueeze(0)
        ).item()
        assert (
            cos_sim >= cos_sim_threshold
        ), f"Cosine similarity {cos_sim:.6f} < {cos_sim_threshold}"


@pytest.mark.parametrize("BATCH", [1, 4])
@pytest.mark.parametrize("SEQLEN_Q, SEQLEN_K", [(128, 128), (512, 2048)])
@pytest.mark.parametrize("NUM_Q_HEADS, NUM_K_HEADS", [(8, 8), (16, 4)])
@pytest.mark.parametrize("HEAD_SZ", [64, 128])
@pytest.mark.parametrize("CAUSAL", [True, False])
def test_mha_fp8_gluon(
    BATCH: int,
    SEQLEN_Q: int,
    SEQLEN_K: int,
    NUM_Q_HEADS: int,
    NUM_K_HEADS: int,
    HEAD_SZ: int,
    CAUSAL: bool,
    dtype=torch.bfloat16,
):
    skip_if_gluon_unsupported("gluon")

    torch.manual_seed(20)
    torch.cuda.empty_cache()
    q = torch.randn((BATCH, SEQLEN_Q, NUM_Q_HEADS, HEAD_SZ), device="cuda", dtype=dtype)
    k = torch.randn((BATCH, SEQLEN_K, NUM_K_HEADS, HEAD_SZ), device="cuda", dtype=dtype)
    v = torch.randn((BATCH, SEQLEN_K, NUM_K_HEADS, HEAD_SZ), device="cuda", dtype=dtype)

    fp8_dtype = get_fp8_e4m3_dtype()
    q_fp8, q_descale = quantize_fp8_per_bh(q, fp8_dtype)
    k_fp8, k_descale = quantize_fp8_per_bh(k, fp8_dtype)
    v_fp8, v_descale = quantize_fp8_per_bh(v, fp8_dtype)

    gluon_out = flash_attn_func(
        q_fp8,
        k_fp8,
        v_fp8,
        causal=CAUSAL,
        backend="gluon",
        q_descale=q_descale,
        k_descale=k_descale,
        v_descale=v_descale,
    )

    torch_out, _, _ = attention_ref(q, k, v, causal=CAUSAL)

    _fp8_assert_close(gluon_out, torch_out.to(gluon_out.dtype))


@pytest.mark.parametrize("BATCH", [1, 4])
@pytest.mark.parametrize("SEQLEN_Q, SEQLEN_K", [(128, 128), (512, 2048)])
@pytest.mark.parametrize("NUM_Q_HEADS, NUM_K_HEADS", [(8, 8), (16, 4)])
@pytest.mark.parametrize("HEAD_SZ", [64, 128])
@pytest.mark.parametrize("CAUSAL", [True, False])
def test_mha_varlen_fp8_gluon(
    BATCH: int,
    SEQLEN_Q: int,
    SEQLEN_K: int,
    NUM_Q_HEADS: int,
    NUM_K_HEADS: int,
    HEAD_SZ: int,
    CAUSAL: bool,
    dtype=torch.bfloat16,
):
    skip_if_gluon_unsupported("gluon")

    torch.manual_seed(20)
    torch.cuda.empty_cache()
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

    # generate_qkv returns grad-tracking tensors; detach so the quantization below
    # doesn't build an autograd graph for this forward-only kernel.
    fp8_dtype = get_fp8_e4m3_dtype()
    q_fp8, q_descale = quantize_fp8_per_bh(q_unpad.detach(), fp8_dtype, cu_seqlens_q)
    k_fp8, k_descale = quantize_fp8_per_bh(k_unpad.detach(), fp8_dtype, cu_seqlens_k)
    v_fp8, v_descale = quantize_fp8_per_bh(v_unpad.detach(), fp8_dtype, cu_seqlens_k)

    gluon_out = flash_attn_varlen_func(
        q_fp8,
        k_fp8,
        v_fp8,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        causal=CAUSAL,
        backend="gluon",
        q_descale=q_descale,
        k_descale=k_descale,
        v_descale=v_descale,
    )
    gluon_out = output_pad_fn(gluon_out)

    torch_out, _, _ = attention_ref(
        q,
        k,
        v,
        query_padding_mask=query_padding_mask,
        key_padding_mask=key_padding_mask,
        causal=CAUSAL,
    )

    _fp8_assert_close(gluon_out, torch_out.to(gluon_out.dtype))


@pytest.mark.parametrize(
    "FP8, VARLEN, CAUSAL, HEAD_SZ, OUT_DTYPE",
    [
        (True, True, True, 256, torch.bfloat16),
        (True, False, False, 128, torch.bfloat16),
        (True, True, False, 128, torch.float16),
        (True, False, True, 128, None),
        (False, False, True, 128, None),
        (False, True, False, 128, torch.float16),
    ],
)
def test_mha_forward_out_dtype(
    FP8: bool,
    VARLEN: bool,
    CAUSAL: bool,
    HEAD_SZ: int,
    OUT_DTYPE: torch.dtype,
    dtype=torch.bfloat16,
):
    """out_dtype changes only the final store: the output is the fp32 output cast."""
    if FP8 and not is_fp8_avail():
        pytest.skip(f"FP8 not supported on {get_arch()}")
    BATCH, SEQLEN, NUM_Q_HEADS, NUM_K_HEADS = 2, 256, 8, 2
    torch.manual_seed(20)
    q = torch.randn((BATCH, SEQLEN, NUM_Q_HEADS, HEAD_SZ), device="cuda", dtype=dtype)
    k = torch.randn((BATCH, SEQLEN, NUM_K_HEADS, HEAD_SZ), device="cuda", dtype=dtype)
    v = torch.randn((BATCH, SEQLEN, NUM_K_HEADS, HEAD_SZ), device="cuda", dtype=dtype)
    cu_seqlens = None
    if VARLEN:
        q, k, v = (x.flatten(0, 1) for x in (q, k, v))
        cu_seqlens = torch.arange(
            0, (BATCH + 1) * SEQLEN, SEQLEN, device="cuda", dtype=torch.int32
        )
    descales = {}
    if FP8:
        fp8_dtype = get_fp8_e4m3_dtype()
        q, descales["descale_q"] = quantize_fp8_per_bh(q, fp8_dtype, cu_seqlens)
        k, descales["descale_k"] = quantize_fp8_per_bh(k, fp8_dtype, cu_seqlens)
        v, descales["descale_v"] = quantize_fp8_per_bh(v, fp8_dtype, cu_seqlens)

    def forward(**kwargs):
        return _flash_attn_forward(
            q,
            k,
            v,
            dropout_p=0.0,
            softmax_scale=HEAD_SZ**-0.5,
            causal=CAUSAL,
            window_size_left=-1,
            window_size_right=-1,
            bias=None,
            alibi_slopes=None,
            return_lse=True,
            return_softmax=False,
            max_seqlen_q=SEQLEN,
            max_seqlen_k=SEQLEN,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            **descales,
            **kwargs,
        )[:2]

    ref, ref_lse = forward(out_dtype=torch.float32)
    out, lse = forward() if OUT_DTYPE is None else forward(out_dtype=OUT_DTYPE)
    expected_dtype = OUT_DTYPE or (torch.float32 if FP8 else dtype)
    assert (
        out.dtype == expected_dtype
    ), f"output dtype is {out.dtype}, expected {expected_dtype}"
    assert (
        lse.dtype == torch.float32
    ), f"softmax_lse dtype is {lse.dtype}, expected torch.float32"
    assert torch.equal(lse, ref_lse), "softmax_lse changed with out_dtype"
    expected = ref.to(expected_dtype)
    # An fp16 store can fuse the 1/l multiply into the cast (v_fma_mixlo_f16) and round
    # once. Then about 1 in 16k values differ by 1 ulp from the double-rounded cast.
    finfo = torch.finfo(expected_dtype)
    torch.testing.assert_close(
        out.float(),
        expected.float(),
        atol=finfo.smallest_normal * finfo.eps,
        rtol=finfo.eps,
        msg=lambda m: f"output is not within 1 ulp of the cast float32 output\n{m}",
    )
    n_diff = int((out != expected).sum())
    assert n_diff <= out.numel() // 1000, (
        f"{n_diff} of {out.numel()} values differ from the cast float32 output, "
        "more than 1 in 1000"
    )


@pytest.mark.parametrize(
    "IMPL, OUT_DTYPE, ERROR",
    [("default", torch.int32, ValueError), ("dao_ai", torch.float16, AssertionError)],
)
def test_mha_forward_out_dtype_rejected(IMPL: str, OUT_DTYPE: torch.dtype, ERROR: type):
    """Unsupported out_dtype values fail before the kernel launch."""
    q = torch.randn((1, 128, 4, 64), device="cuda", dtype=torch.bfloat16)
    mha_set_impl(IMPL)
    try:
        with pytest.raises(ERROR, match="out_dtype"):
            _flash_attn_forward(
                q,
                q,
                q,
                dropout_p=0.0,
                softmax_scale=64**-0.5,
                causal=False,
                window_size_left=-1,
                window_size_right=-1,
                bias=None,
                alibi_slopes=None,
                return_lse=False,
                return_softmax=False,
                max_seqlen_q=128,
                max_seqlen_k=128,
                out_dtype=OUT_DTYPE,
            )
    finally:
        mha_set_impl("default")


# Short K (fewer than 8 key tiles) runs the one-tile loop; long K, the pipeline.
@pytest.mark.parametrize("SEQLEN_Q, SEQLEN_K", [(128, 128), (300, 1100)])
@pytest.mark.parametrize("HEAD_SZ", [53, 64, 96, 128])
@pytest.mark.parametrize("CAUSAL", [True, False])
def test_mha_fp32_gluon(SEQLEN_Q: int, SEQLEN_K: int, HEAD_SZ: int, CAUSAL: bool):
    """fp32 inputs on the fp32 MFMA path, padded heads included."""
    skip_if_gluon_unsupported("gluon")

    torch.manual_seed(21)
    q = torch.randn((2, SEQLEN_Q, 4, HEAD_SZ), device="cuda", dtype=torch.float32)
    k = torch.randn((2, SEQLEN_K, 2, HEAD_SZ), device="cuda", dtype=torch.float32)
    v = torch.randn_like(k)

    gluon_out, gluon_lse = flash_attn_func(
        q, k, v, causal=CAUSAL, return_lse=True, backend="gluon"
    )
    torch_out, _, torch_lse = attention_ref(q, k, v, causal=CAUSAL)

    torch.testing.assert_close(gluon_out, torch_out, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(gluon_lse, torch_lse, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("HEAD_SZ", [53, 128])
@pytest.mark.parametrize("CAUSAL", [True, False])
def test_mha_varlen_fp32_gluon(HEAD_SZ: int, CAUSAL: bool):
    """Packed fp32 sequences, one-tile and pipelined lengths in one launch."""
    skip_if_gluon_unsupported("gluon")

    torch.manual_seed(22)
    seqlens_q, seqlens_k = [300, 77, 129], [1100, 513, 129]
    cu_seqlens_q = torch.tensor([0] + seqlens_q, device="cuda").cumsum(0).int()
    cu_seqlens_k = torch.tensor([0] + seqlens_k, device="cuda").cumsum(0).int()
    q = torch.randn((sum(seqlens_q), 4, HEAD_SZ), device="cuda", dtype=torch.float32)
    k = torch.randn((sum(seqlens_k), 2, HEAD_SZ), device="cuda", dtype=torch.float32)
    v = torch.randn_like(k)

    gluon_out, gluon_lse = flash_attn_varlen_func(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max(seqlens_q),
        max(seqlens_k),
        causal=CAUSAL,
        return_lse=True,
        backend="gluon",
    )

    for b in range(len(seqlens_q)):
        rows = slice(cu_seqlens_q[b], cu_seqlens_q[b + 1])
        keys = slice(cu_seqlens_k[b], cu_seqlens_k[b + 1])
        torch_out, _, torch_lse = attention_ref(
            q[None, rows], k[None, keys], v[None, keys], causal=CAUSAL
        )
        torch.testing.assert_close(gluon_out[rows], torch_out[0], atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(
            gluon_lse[rows], torch_lse[0].T, atol=1e-4, rtol=1e-4
        )


@pytest.mark.parametrize("DTYPE", ["bf16", "fp16", "fp8"])
@pytest.mark.parametrize("HEAD_SZ", [64, 128])
@pytest.mark.parametrize("CAUSAL", [True, False])
@pytest.mark.parametrize("JUMP", ["below", "at", "above"])
def test_mha_gluon_lazy_rescale_threshold(
    DTYPE: str, HEAD_SZ: int, CAUSAL: bool, JUMP: str
):
    """Row-max jumps just below, exactly at and just above the lazy-rescale
    threshold, on keys long enough for the pipelined K loop."""
    skip_if_gluon_unsupported("gluon")
    SEQLEN_Q, SEQLEN_K, PLATEAU = 256, 2048, 512

    is_fp8 = DTYPE == "fp8"
    threshold = _get_gluon_config(
        is_fp8=is_fp8, causal=CAUSAL, v_head_dim=HEAD_SZ, seqlen_k=SEQLEN_K
    )["LAZY_RESCALE_THRESHOLD"]
    assert threshold > 0, "this shape must select a lazy-rescale config"
    jump = threshold + {"below": -1.0, "at": 0.0, "above": 1.0}[JUMP]

    # softmax_scale = ln 2 and Q = e_0 make each score, in log2 units, K[..., 0]:
    # four plateaus of keys, each `jump` above the one before.  Every level is
    # exact in bf16, fp16 and e4m3, so "at" lands on the threshold exactly.
    torch.manual_seed(23)
    q = torch.zeros((1, SEQLEN_Q, 2, HEAD_SZ), device="cuda", dtype=torch.float32)
    q[..., 0] = 1
    k = torch.randn((1, SEQLEN_K, 2, HEAD_SZ), device="cuda", dtype=torch.float32)
    levels = (torch.arange(SEQLEN_K, device="cuda") // PLATEAU).to(torch.float32)
    k[..., 0] = (levels * jump)[:, None]
    v = torch.randn_like(k)
    kwargs = {}
    if is_fp8:
        fp8_dtype = get_fp8_e4m3_dtype()
        q, k, v = q.to(fp8_dtype), k.to(fp8_dtype), v.to(fp8_dtype)
        descale = torch.ones((1, 2), device="cuda", dtype=torch.float32)
        kwargs = {"q_descale": descale, "k_descale": descale, "v_descale": descale}
    else:
        dtype = torch.bfloat16 if DTYPE == "bf16" else torch.float16
        q, k, v = q.to(dtype), k.to(dtype), v.to(dtype)

    gluon_out, gluon_lse = flash_attn_func(
        q,
        k,
        v,
        softmax_scale=math.log(2.0),
        causal=CAUSAL,
        return_lse=True,
        backend="gluon",
        **kwargs,
    )
    # attention_ref scales by 1/sqrt(HEAD_SZ); fold in ln 2 instead.
    torch_out, _, torch_lse = attention_ref(
        q.float() * (math.log(2.0) * HEAD_SZ**0.5),
        k.float(),
        v.float(),
        causal=CAUSAL,
    )

    tol = {"bf16": 1e-2, "fp16": 2e-3, "fp8": 3e-2}[DTYPE]
    torch.testing.assert_close(gluon_out.float(), torch_out, atol=tol, rtol=tol)
    torch.testing.assert_close(gluon_lse, torch_lse, atol=1e-3, rtol=1e-4)


@pytest.mark.parametrize("CAUSAL", [False, True])
def test_mha_fp8_gluon_long_tail(CAUSAL: bool):
    """A long fp8 tail far below the row max keeps its weight."""
    skip_if_gluon_unsupported("gluon")
    SEQLEN_K, HEAD_SZ, SCALE = 8192, 128, 0.125

    # Logits [0, -10, -10, ...] and V = [0, 1, 1, ...]: the output is the tail's
    # total weight, 8191 * e^-10 / (1 + 8191 * e^-10) = 0.271.
    q = torch.zeros((1, 128, 1, HEAD_SZ), device="cuda", dtype=torch.float32)
    q[..., 0] = 1
    k = torch.zeros((1, SEQLEN_K, 1, HEAD_SZ), device="cuda", dtype=torch.float32)
    k[:, 1:, :, 0] = -10 / SCALE
    v = torch.ones((1, SEQLEN_K, 1, HEAD_SZ), device="cuda", dtype=torch.float32)
    v[:, 0] = 0
    fp8_dtype = get_fp8_e4m3_dtype()
    q, k, v = q.to(fp8_dtype), k.to(fp8_dtype), v.to(fp8_dtype)
    descale = torch.ones((1, 1), device="cuda", dtype=torch.float32)

    gluon_out = flash_attn_func(
        q,
        k,
        v,
        softmax_scale=SCALE,
        causal=CAUSAL,
        q_descale=descale,
        k_descale=descale,
        v_descale=descale,
        backend="gluon",
    )
    # attention_ref scales by 1/sqrt(HEAD_SZ); fold in SCALE instead.
    torch_out, _, _ = attention_ref(
        q.float() * (SCALE * HEAD_SZ**0.5), k.float(), v.float(), causal=CAUSAL
    )

    # Equal tail P all round the same way in e4m3, giving 0.262; P lifted short
    # of e4m3's top gave 0.419.
    torch.testing.assert_close(gluon_out.float(), torch_out, atol=0.015, rtol=0)


def test_mha_fp16_gluon_small_scale():
    """fp16 Q times a tiny softmax scale must not underflow before Q@K^T."""
    skip_if_gluon_unsupported("gluon")
    SEQLEN_K, HEAD_SZ, SCALE = 1024, 64, 1e-6

    # Q = 1/64, K = +-65504, V = +-1 with the same sign per key: every score is
    # +-0.0655, which moves the output by about 0.065.  Q scaled in fp16 first
    # (1/64 * 1e-6 * log2(e)) would round to zero and lose that.
    torch.manual_seed(26)
    sign = torch.randint(0, 2, (1, SEQLEN_K, 1, 1), device="cuda") * 2 - 1
    sign = sign.to(torch.float16)
    q = torch.full((1, 128, 1, HEAD_SZ), 1 / 64, device="cuda", dtype=torch.float16)
    k = (sign * 65504).expand(-1, -1, -1, HEAD_SZ).contiguous()
    v = sign.expand(-1, -1, -1, HEAD_SZ).contiguous()

    gluon_out = flash_attn_func(q, k, v, softmax_scale=SCALE, backend="gluon")
    torch_out, _, _ = attention_ref(
        q.float() * (SCALE * HEAD_SZ**0.5), k.float(), v.float()
    )

    torch.testing.assert_close(gluon_out.float(), torch_out, atol=3e-3, rtol=3e-3)


@pytest.mark.parametrize("NUM_Q_HEADS, NUM_K_HEADS", [(1, 1), (8, 1)])
@pytest.mark.parametrize("DROPOUT, RETURN_LSE, RETURN_SOFTMAX, ", [(0.2, True, True)])
@pytest.mark.parametrize("CAUSAL", [(True), (False)])
def test_mha_with_dropout(
    NUM_Q_HEADS: int,
    NUM_K_HEADS: int,
    DROPOUT: float,
    RETURN_LSE: bool,
    RETURN_SOFTMAX: bool,
    CAUSAL: bool,
    dtype=torch.bfloat16,
):
    batch = 2
    seqlen_q = 510
    seqlen_k = 1020
    head_size = 128
    _test_mha_impl(
        batch,
        seqlen_q,
        seqlen_k,
        NUM_Q_HEADS,
        NUM_K_HEADS,
        head_size,
        DROPOUT=DROPOUT,
        RETURN_LSE=RETURN_LSE,
        RETURN_SOFTMAX=RETURN_SOFTMAX,
        CAUSAL=CAUSAL,
        dtype=dtype,
    )


@pytest.mark.parametrize("backend", ["triton", "gluon"])
@pytest.mark.parametrize("CAUSAL", [True, False])
@pytest.mark.parametrize(
    "RETURN_LSE, RETURN_SOFTMAX",
    [(True, False), (False, True), (True, True)],
)
def test_mha_return_lse_softmax(
    backend: str,
    CAUSAL: bool,
    RETURN_LSE: bool,
    RETURN_SOFTMAX: bool,
    dtype=torch.bfloat16,
):
    _test_mha_impl(
        BATCH=2,
        SEQLEN_Q=128,
        SEQLEN_K=64,
        NUM_Q_HEADS=8,
        NUM_K_HEADS=2,
        HEAD_SZ=64,
        DROPOUT=0.0,
        RETURN_LSE=RETURN_LSE,
        RETURN_SOFTMAX=RETURN_SOFTMAX,
        CAUSAL=CAUSAL,
        backend=backend,
        dtype=dtype,
    )


# LLaMA 3 405B config
@pytest.mark.parametrize("backend", ["triton", "gluon"])
def test_mha_int64_strides(
    backend: str,
    dtype=torch.bfloat16,
    test_backward=True,
):
    BATCH = 1
    SEQLEN_Q, SEQLEN_K = 1, 1
    NUM_Q_HEADS, NUM_K_HEADS = 128, 8
    HEAD_SZ = 128
    CAUSAL = True
    DROPOUT = 0.0
    """
    In the absence of strides being int64, parts of the offset computation is done in 32 bit and overflows resulting in segfaults.
    """
    # The Gluon backend is forward-only.
    return_lse = True
    test_backward = test_backward and backend != "gluon"
    skip_if_gluon_unsupported(backend, dropout_p=DROPOUT)

    torch.cuda.empty_cache()
    torch.manual_seed(20)
    # use int64 strides for the backend under test.
    # NOTE: if you set this to false this test case will segfault
    mha_set_use_int64_strides(True)

    # generate inputs with large strides
    def _generate_input(
        batch: int, seqlen: int, nheads: int, dim_size: int, large_stride: bool = False
    ) -> torch.Tensor:
        seqlens = torch.full((batch,), seqlen, device="cuda")
        cu_seqlens = torch.cat(
            [
                torch.tensor([0], dtype=torch.int32, device="cuda"),
                seqlens.cumsum(dim=0, dtype=torch.int32),
            ]
        )
        total_seqlen = cu_seqlens[-1].item()

        if large_stride:
            x_dummy = torch.randn(
                (total_seqlen, nheads, 1024 * 1024 * 64), dtype=dtype, device="cuda"
            ).requires_grad_(True)
            x = x_dummy[:seqlen, :nheads, :dim_size]
        else:
            x = torch.randn(
                (total_seqlen, nheads, dim_size), dtype=dtype, device="cuda"
            ).requires_grad_(True)
        return x, cu_seqlens, seqlen

    # inputs
    q, cu_seqlens_q, max_seqlens_q = _generate_input(
        BATCH, SEQLEN_Q, NUM_Q_HEADS, HEAD_SZ, large_stride=True
    )
    k, cu_seqlens_k, max_seqlens_k = _generate_input(
        BATCH, SEQLEN_K, NUM_K_HEADS, HEAD_SZ
    )
    v, _, _ = _generate_input(BATCH, SEQLEN_K, NUM_K_HEADS, HEAD_SZ)
    do = torch.randn_like(q)

    logger.debug("q: %s %s", q.shape, q.stride())
    logger.debug("k: %s %s", k.shape, k.stride())
    logger.debug("v: %s %s", v.shape, v.stride())
    logger.debug("cu_seqlens_q: %s %s", cu_seqlens_q.shape, cu_seqlens_q.stride())
    logger.debug("cu_seqlens_k: %s %s", cu_seqlens_k.shape, cu_seqlens_k.stride())

    out = flash_attn_varlen_func(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlens_q,
        max_seqlens_k,
        dropout_p=DROPOUT,
        causal=CAUSAL,
        return_lse=return_lse,
        backend=backend,
    )
    triton_out, lse = out
    assert lse.dtype == torch.float32
    assert lse.shape == (q.shape[0], NUM_Q_HEADS)
    if test_backward:
        triton_dq, triton_dk, triton_dv = torch.autograd.grad(
            triton_out, (q, k, v), do.clone()
        )

    # NOTE: use fwd output to wait not exit program before kernel finishes
    logger.info("triton_out: %s", triton_out)
    if test_backward:
        logger.info("triton_dq: %s %s", triton_dq.shape, triton_dq.stride())
        logger.info("triton_dk: %s %s", triton_dk.shape, triton_dk.stride())
        logger.info("triton_dv: %s %s", triton_dv.shape, triton_dv.stride())


def _test_mha_varlen_impl(
    BATCH: int,
    SEQLEN_Q: int,
    SEQLEN_K: int,
    NUM_Q_HEADS: int,
    NUM_K_HEADS: int,
    HEAD_SZ: int,
    DROPOUT: float,
    RETURN_LSE: bool,
    RETURN_SOFTMAX: bool,
    CAUSAL: bool,
    backend: str = "triton",
    dtype=torch.bfloat16,
):
    skip_if_gluon_unsupported(backend, dropout_p=DROPOUT)

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

    triton_out = flash_attn_varlen_func(
        q_unpad,
        k_unpad,
        v_unpad,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        dropout_p=DROPOUT,
        causal=CAUSAL,
        return_lse=RETURN_LSE,
        return_attn_probs=RETURN_SOFTMAX,
        backend=backend,
    )

    lse = None
    sd_mask = None
    if RETURN_LSE or RETURN_SOFTMAX:
        lse = triton_out[1] if RETURN_LSE else None
        if RETURN_SOFTMAX:
            sd_mask = triton_out[2] if RETURN_LSE else triton_out[1]
        triton_out = triton_out[0]
        if lse is not None:
            logger.debug("lse.shape=%s, lse=%s", lse.shape, lse)

    dropout_mask = None
    if DROPOUT > 0.0 and RETURN_SOFTMAX:
        dropout_mask = sd_mask >= 0
        dropout_mask = pad_rearrange_dropout_mask(
            dropout_mask,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            SEQLEN_Q,
            SEQLEN_K,
            NUM_Q_HEADS,
        )
        dropout_mask = dropout_mask > 0
        logger.debug(
            "dropout_mask.shape=%s, dropout_mask=%s",
            dropout_mask.shape,
            dropout_mask,
        )
    triton_out = output_pad_fn(triton_out)
    logger.debug("triton_out.shape=%s, triton_out=%s", triton_out.shape, triton_out)

    torch_out = attention_ref(
        q,
        k,
        v,
        query_padding_mask=query_padding_mask,
        key_padding_mask=key_padding_mask,
        dropout_p=DROPOUT,
        dropout_mask=dropout_mask,
        causal=CAUSAL,
    )
    torch_out, attention_scores, lse_ref = torch_out

    logger.debug("torch_out.shape=%s, torch_out=%s", torch_out.shape, torch_out)
    logger.debug(
        "attention_scores.shape=%s, attention_scores=%s",
        attention_scores.shape,
        attention_scores,
    )

    torch.testing.assert_close(
        triton_out, torch_out.to(triton_out.dtype), atol=1e-1, rtol=1e-1
    )

    if RETURN_LSE:
        lse_ref = torch.cat(
            [
                lse_ref[b, :, query_padding_mask[b]].transpose(0, 1)
                for b in range(BATCH)
            ],
            dim=0,
        )
        finite = torch.isfinite(lse_ref)
        torch.testing.assert_close(
            lse[finite].float(), lse_ref[finite].float(), atol=1e-2, rtol=1e-2
        )
    if RETURN_SOFTMAX:
        if sd_mask is None:
            assert backend == "triton" and DROPOUT == 0.0
        else:
            assert sd_mask.dtype == torch.float32
            assert sd_mask.shape == (BATCH, NUM_Q_HEADS, max_seqlen_q, max_seqlen_k)


@pytest.mark.parametrize("BATCH", [1, 4, 30, 50])
@pytest.mark.parametrize(
    "SEQLEN_Q, SEQLEN_K",
    [(1, 1), (128, 128), (32, 16), (64, 128), (2048, 2048)],
)
@pytest.mark.parametrize(
    "NUM_Q_HEADS, NUM_K_HEADS", [(1, 1), (8, 1), (16, 16), (48, 8)]
)
@pytest.mark.parametrize("HEAD_SZ", [8, 32, 33, 128])
@pytest.mark.parametrize("CAUSAL", [(True), (False)])
@pytest.mark.parametrize("backend", ["triton", "gluon"])
def test_mha_varlen(
    BATCH: int,
    SEQLEN_Q: int,
    SEQLEN_K: int,
    NUM_Q_HEADS: int,
    NUM_K_HEADS: int,
    HEAD_SZ: int,
    CAUSAL: bool,
    backend: str,
    dtype=torch.bfloat16,
):
    skip_if_triton_padded_head_miscompiled(
        backend, HEAD_SZ, CAUSAL, SEQLEN_K, NUM_K_HEADS
    )
    _test_mha_varlen_impl(
        BATCH,
        SEQLEN_Q,
        SEQLEN_K,
        NUM_Q_HEADS,
        NUM_K_HEADS,
        HEAD_SZ,
        DROPOUT=0.0,
        RETURN_LSE=False,
        RETURN_SOFTMAX=False,
        CAUSAL=CAUSAL,
        backend=backend,
        dtype=dtype,
    )


@pytest.mark.parametrize("NUM_Q_HEADS, NUM_K_HEADS", [(1, 1), (8, 1)])
@pytest.mark.parametrize("DROPOUT, RETURN_LSE, RETURN_SOFTMAX, ", [(0.2, True, True)])
@pytest.mark.parametrize("CAUSAL", [(True), (False)])
def test_mha_varlen_with_dropout(
    NUM_Q_HEADS: int,
    NUM_K_HEADS: int,
    DROPOUT: float,
    RETURN_LSE: bool,
    RETURN_SOFTMAX: bool,
    CAUSAL: bool,
    dtype=torch.bfloat16,
):
    batch = 2
    seqlen_q = 510
    seqlen_k = 1020
    head_size = 128
    _test_mha_varlen_impl(
        batch,
        seqlen_q,
        seqlen_k,
        NUM_Q_HEADS,
        NUM_K_HEADS,
        head_size,
        DROPOUT=DROPOUT,
        RETURN_LSE=RETURN_LSE,
        RETURN_SOFTMAX=RETURN_SOFTMAX,
        CAUSAL=CAUSAL,
        dtype=dtype,
    )


@pytest.mark.parametrize("backend", ["triton", "gluon"])
@pytest.mark.parametrize("CAUSAL", [True, False])
@pytest.mark.parametrize(
    "RETURN_LSE, RETURN_SOFTMAX",
    [(True, False), (False, True), (True, True)],
)
def test_mha_varlen_return_lse_softmax(
    backend: str,
    CAUSAL: bool,
    RETURN_LSE: bool,
    RETURN_SOFTMAX: bool,
    dtype=torch.bfloat16,
):
    _test_mha_varlen_impl(
        BATCH=2,
        SEQLEN_Q=128,
        SEQLEN_K=64,
        NUM_Q_HEADS=8,
        NUM_K_HEADS=2,
        HEAD_SZ=64,
        DROPOUT=0.0,
        RETURN_LSE=RETURN_LSE,
        RETURN_SOFTMAX=RETURN_SOFTMAX,
        CAUSAL=CAUSAL,
        backend=backend,
        dtype=dtype,
    )


# Production shapes based on real models:
#   HQ=32, HK=8:  Llama 3 8B (GQA 4:1)
#   HQ=64, HK=8:  Llama 3 70B (GQA 8:1)
#   HQ=32, HK=32: Llama 2 7B (MHA)
@pytest.mark.parametrize("BATCH", [1, 4])
@pytest.mark.parametrize("SEQLEN_Q", [512, 1024, 2048])
@pytest.mark.parametrize("SEQLEN_K", [512, 1024, 2048])
@pytest.mark.parametrize("NUM_Q_HEADS", [32, 64])
@pytest.mark.parametrize("NUM_K_HEADS", [8])
@pytest.mark.parametrize("HEAD_SZ", [128])
@pytest.mark.parametrize("CAUSAL", [True, False])
@pytest.mark.parametrize("DROPOUT", [0.0, 0.2])
def test_mha_backward(
    BATCH: int,
    SEQLEN_Q: int,
    SEQLEN_K: int,
    NUM_Q_HEADS: int,
    NUM_K_HEADS: int,
    HEAD_SZ: int,
    CAUSAL: bool,
    DROPOUT: float,
    FUSED: bool,
    dtype=torch.bfloat16,
):
    HAS_DROPOUT = DROPOUT > 0.0

    if FUSED and CAUSAL:
        pytest.skip("FUSED+CAUSAL results in NaNs")
    if CAUSAL and HAS_DROPOUT:
        pytest.skip("CAUSAL+DROPOUT backward results in NaNs")

    torch.cuda.empty_cache()
    torch.manual_seed(20)

    q = torch.randn(BATCH, SEQLEN_Q, NUM_Q_HEADS, HEAD_SZ, device="cuda", dtype=dtype)
    k = torch.randn(BATCH, SEQLEN_K, NUM_K_HEADS, HEAD_SZ, device="cuda", dtype=dtype)
    v = torch.randn(BATCH, SEQLEN_K, NUM_K_HEADS, HEAD_SZ, device="cuda", dtype=dtype)
    q.requires_grad = True
    k.requires_grad = True
    v.requires_grad = True
    do = torch.randn_like(q)

    # Triton forward + backward
    with torch.enable_grad():
        triton_out = flash_attn_func(
            q,
            k,
            v,
            dropout_p=DROPOUT,
            causal=CAUSAL,
            return_lse=HAS_DROPOUT,
            return_attn_probs=HAS_DROPOUT,
        )
        if HAS_DROPOUT:
            dropout_mask = triton_out[2] >= 0
            triton_out = triton_out[0]
        else:
            dropout_mask = None
    triton_dq, triton_dk, triton_dv = torch.autograd.grad(
        triton_out, (q, k, v), do.clone()
    )

    # Reference forward + backward with adaptive tolerances
    torch_out, torch_grads, fwd_tol, bwd_tols = attention_ref_with_tol(
        q,
        k,
        v,
        do,
        is_fp8=False,
        dropout_p=DROPOUT,
        dropout_mask=dropout_mask,
        causal=CAUSAL,
    )
    torch_dq, torch_dk, torch_dv = torch_grads

    # Check quality
    triton_vals = [triton_out, triton_dq, triton_dk, triton_dv]
    ref_vals = [torch_out, torch_dq, torch_dk, torch_dv]
    tols = [fwd_tol] + bwd_tols
    for tri, ref, (atol, rtol) in zip(triton_vals, ref_vals, tols):
        torch.testing.assert_close(tri, ref.to(tri.dtype), atol=atol, rtol=rtol)


@pytest.mark.parametrize("head_size", [33, 53, 64])
@pytest.mark.parametrize("varlen", [False, True])
@pytest.mark.parametrize("seqlen_q,seqlen_k", [(97, 113), (113, 97)])
@pytest.mark.parametrize("strided", [False, True])
def test_mha_dao_ai_backward_head_size(
    head_size, varlen, seqlen_q, seqlen_k, strided, monkeypatch
):
    from aiter.ops.triton._triton_kernels.flash_attn_triton_amd import bwd, interface_v2
    from aiter.ops.triton.attention import mha

    monkeypatch.setattr(interface_v2, "BWD_MODE", "fused")
    kernel = bwd.bwd_kernel_fused_causal
    original_run = kernel.run
    launches = 0

    def trace_run(*args, **kwargs):
        nonlocal launches
        result = original_run(*args, **kwargs)
        launches += 1
        return result

    monkeypatch.setattr(kernel, "run", trace_run)

    monkeypatch.setattr(mha, "_MHA_IMPL", "dao_ai")
    torch.manual_seed(20)
    batch = 2
    q = torch.randn(
        batch,
        seqlen_q,
        4,
        head_size,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    k = torch.randn(
        batch,
        seqlen_k,
        2,
        head_size,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    v = torch.randn_like(k, requires_grad=True)
    do = torch.randn_like(q)
    inputs = (q, k, v)
    if varlen:
        inputs = tuple(x.flatten(0, 1) for x in inputs)
    if strided:
        # Change token/head strides after packing so flatten cannot undo them.
        inputs = tuple(
            x.transpose(-3, -2).contiguous().transpose(-3, -2).detach().requires_grad_()
            for x in inputs
        )
        assert all(not x.is_contiguous() for x in inputs)
    if varlen:
        cu_q = torch.arange(
            0, (batch + 1) * seqlen_q, seqlen_q, device="cuda", dtype=torch.int32
        )
        cu_k = torch.arange(
            0, (batch + 1) * seqlen_k, seqlen_k, device="cuda", dtype=torch.int32
        )
        out = flash_attn_varlen_func(
            *inputs, cu_q, cu_k, seqlen_q, seqlen_k, causal=True, backend="triton"
        )
    else:
        out = flash_attn_func(*inputs, causal=True, backend="triton")
    grads = torch.autograd.grad(out, inputs, do.reshape_as(out))
    assert launches > 0, "Expected the Dao-AI fused causal backward kernel"
    ref_out, ref_grads, fwd_tol, bwd_tols = attention_ref_with_tol(
        q, k, v, do, causal=True
    )
    torch.testing.assert_close(
        out, ref_out.reshape_as(out).to(out.dtype), atol=fwd_tol[0], rtol=fwd_tol[1]
    )
    for inp, grad, ref, (atol, rtol) in zip(inputs, grads, ref_grads, bwd_tols):
        assert grad.shape == inp.shape
        torch.testing.assert_close(
            grad, ref.reshape_as(grad).to(grad.dtype), atol=atol, rtol=rtol
        )


@pytest.mark.parametrize("BATCH", [2, 4])
@pytest.mark.parametrize("SEQLEN_Q", [256, 512])
@pytest.mark.parametrize("SEQLEN_K", [256, 512])
@pytest.mark.parametrize("NUM_Q_HEADS", [32])
@pytest.mark.parametrize("NUM_K_HEADS", [8])
@pytest.mark.parametrize("HEAD_SZ", [128])
@pytest.mark.parametrize("CAUSAL", [False])
def test_mha_backward_sbhd_do(
    BATCH: int,
    SEQLEN_Q: int,
    SEQLEN_K: int,
    NUM_Q_HEADS: int,
    NUM_K_HEADS: int,
    HEAD_SZ: int,
    CAUSAL: bool,
    FUSED: bool,
    dtype=torch.bfloat16,
):
    """Verify backward correctness when dO has SBHD memory layout (strides differ from O).

    Creates dO as a (seqlen, batch, nheads, headdim) tensor transposed to
    (batch, seqlen, nheads, headdim), so its strides are different from the
    contiguous BSHD output tensor. This exercises the independent stride
    handling for dO in _bwd_preprocess.
    """
    if FUSED and CAUSAL:
        pytest.skip("FUSED+CAUSAL results in NaNs")

    torch.cuda.empty_cache()
    torch.manual_seed(42)

    q = torch.randn(BATCH, SEQLEN_Q, NUM_Q_HEADS, HEAD_SZ, device="cuda", dtype=dtype)
    k = torch.randn(BATCH, SEQLEN_K, NUM_K_HEADS, HEAD_SZ, device="cuda", dtype=dtype)
    v = torch.randn(BATCH, SEQLEN_K, NUM_K_HEADS, HEAD_SZ, device="cuda", dtype=dtype)
    q.requires_grad = True
    k.requires_grad = True
    v.requires_grad = True

    # dO in SBHD memory layout: (seqlen, batch, nheads, headdim) viewed as BSHD
    do_sbhd = torch.randn(
        SEQLEN_Q, BATCH, NUM_Q_HEADS, HEAD_SZ, device="cuda", dtype=dtype
    )
    do = do_sbhd.transpose(0, 1)  # shape is BSHD but strides are SBHD
    assert not do.is_contiguous(), "dO should be non-contiguous (SBHD strides)"

    # Reference: use contiguous dO for the reference computation
    do_contig = do.contiguous()

    # Triton forward + backward with SBHD-strided dO
    with torch.enable_grad():
        triton_out = flash_attn_func(q, k, v, causal=CAUSAL)
    triton_dq, triton_dk, triton_dv = torch.autograd.grad(triton_out, (q, k, v), do)

    # Reference forward + backward (with contiguous dO)
    torch_out, torch_grads, fwd_tol, bwd_tols = attention_ref_with_tol(
        q,
        k,
        v,
        do_contig,
        causal=CAUSAL,
    )
    torch_dq, torch_dk, torch_dv = torch_grads

    triton_vals = [triton_out, triton_dq, triton_dk, triton_dv]
    ref_vals = [torch_out, torch_dq, torch_dk, torch_dv]
    tols = [fwd_tol] + bwd_tols
    for tri, ref, (atol, rtol) in zip(triton_vals, ref_vals, tols):
        torch.testing.assert_close(tri, ref.to(tri.dtype), atol=atol, rtol=rtol)


@pytest.mark.parametrize("SEQLEN_Q", [512, 2048])
@pytest.mark.parametrize("SEQLEN_K", [512, 2048])
@pytest.mark.parametrize("NUM_Q_HEADS", [32, 64])
@pytest.mark.parametrize("CAUSAL", [True, False])
@pytest.mark.parametrize("DROPOUT", [0.0, 0.2])
def test_mha_backward_varlen(
    SEQLEN_Q: int,
    SEQLEN_K: int,
    NUM_Q_HEADS: int,
    CAUSAL: bool,
    DROPOUT: float,
    FUSED: bool,
    dtype=torch.bfloat16,
):
    BATCH = 3
    HEAD_SZ = 128
    NUM_K_HEADS = 8
    HAS_DROPOUT = DROPOUT > 0.0

    if FUSED and CAUSAL:
        pytest.skip("FUSED+CAUSAL results in NaNs")
    if CAUSAL and HAS_DROPOUT:
        pytest.skip("CAUSAL+DROPOUT backward results in NaNs")

    torch.cuda.empty_cache()
    torch.manual_seed(20)

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
        triton_out = flash_attn_varlen_func(
            q_unpad,
            k_unpad,
            v_unpad,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            dropout_p=DROPOUT,
            causal=CAUSAL,
            return_lse=HAS_DROPOUT,
            return_attn_probs=HAS_DROPOUT,
        )
        if HAS_DROPOUT:
            dropout_mask = (
                pad_rearrange_dropout_mask(
                    triton_out[2] >= 0,
                    cu_seqlens_q,
                    cu_seqlens_k,
                    max_seqlen_q,
                    max_seqlen_k,
                    SEQLEN_Q,
                    SEQLEN_K,
                    NUM_Q_HEADS,
                )
                > 0
            )
            triton_out = triton_out[0]
        else:
            dropout_mask = None
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
        is_fp8=False,
        query_padding_mask=query_padding_mask,
        key_padding_mask=key_padding_mask,
        dropout_p=DROPOUT,
        dropout_mask=dropout_mask,
        causal=CAUSAL,
    )
    torch_dq, torch_dk, torch_dv = torch_grads

    # Check quality
    triton_vals = [triton_out, triton_dq, triton_dk, triton_dv]
    ref_vals = [torch_out, torch_dq, torch_dk, torch_dv]
    tols = [fwd_tol] + bwd_tols
    for tri, ref, (atol, rtol) in zip(triton_vals, ref_vals, tols):
        torch.testing.assert_close(tri, ref.to(tri.dtype), atol=atol, rtol=rtol)


@pytest.mark.parametrize("bwd_mode", [None, "fused", "split"])
def test_mha_backward_varlen_very_negative_logits(bwd_mode, monkeypatch):
    # Rows whose logits are all very negative (logsumexp < -88) used to overflow exp(0 - lse) to inf
    # on the zero-loaded keys of the last partial key block of the unmasked dq loop, and inf * 0 put
    # NaN in dq. Seen on real Qwen2.5-1.5B activations. The sequence lengths are not block multiples
    # so that partial block exists.
    torch.cuda.empty_cache()
    torch.manual_seed(0)
    mha_set_use_fused_bwd_kernel(False)
    NUM_HEADS, HEAD_SZ = 4, 128
    seqlens = [300, 250]
    total = sum(seqlens)
    u = torch.nn.functional.normalize(
        torch.randn(HEAD_SZ, device="cuda", dtype=torch.float32), dim=0
    )
    a = 34.0  # q . k ~ -a^2, so logits ~ -a^2 / sqrt(HEAD_SZ) ~ -102
    q = (
        -a * u
        + 0.5
        * torch.randn(total, NUM_HEADS, HEAD_SZ, device="cuda", dtype=torch.float32)
    ).to(torch.bfloat16)
    k = (
        a * u
        + 0.5
        * torch.randn(total, NUM_HEADS, HEAD_SZ, device="cuda", dtype=torch.float32)
    ).to(torch.bfloat16)
    v = torch.randn(total, NUM_HEADS, HEAD_SZ, device="cuda", dtype=torch.bfloat16)
    do = torch.randn(total, NUM_HEADS, HEAD_SZ, device="cuda", dtype=torch.bfloat16)
    q.requires_grad = True
    k.requires_grad = True
    v.requires_grad = True
    cu_seqlens = torch.tensor(
        [0, *torch.tensor(seqlens, device="cuda").cumsum(0).tolist()],
        device="cuda",
        dtype=torch.int32,
    )

    if bwd_mode is None:
        triton_out = flash_attn_varlen_func(
            q, k, v, cu_seqlens, cu_seqlens, max(seqlens), max(seqlens), causal=True
        )
        triton_dq, triton_dk, triton_dv = torch.autograd.grad(triton_out, (q, k, v), do)
    else:
        # The flash_attn_2 backward has "fused" and "split" modes with different dq loops; monkeypatch restores the mode.
        monkeypatch.setattr(flash_attn_2, "BWD_MODE", bwd_mode)
        scale = HEAD_SZ**-0.5
        out, lse, _, _ = flash_attn_2.varlen_fwd(
            q,
            k,
            v,
            None,
            cu_seqlens,
            cu_seqlens,
            seqused_k=None,
            leftpad_k=None,
            block_table_=None,
            alibi_slopes=None,
            max_seqlen_q=max(seqlens),
            max_seqlen_k=max(seqlens),
            dropout_p=0.0,
            softmax_scale=scale,
            zero_tensors=False,
            causal=True,
            window_size_left=-1,
            window_size_right=-1,
            softcap=0.0,
            return_softmax=False,
        )
        triton_dq, triton_dk, triton_dv, _ = flash_attn_2.varlen_bwd(
            do,
            q,
            k,
            v,
            out,
            lse,
            None,
            None,
            None,
            cu_seqlens,
            cu_seqlens,
            None,
            max_seqlen_q=max(seqlens),
            max_seqlen_k=max(seqlens),
            dropout_p=0.0,
            softmax_scale=scale,
            zero_tensors=False,
            causal=True,
            window_size_left=-1,
            window_size_right=-1,
            softcap=0.0,
            deterministic=False,
        )

    # Reference: fp32 attention per sequence.
    q_ref, k_ref, v_ref = (t.detach().float().requires_grad_(True) for t in (q, k, v))
    ref_out = torch.cat(
        [
            attention_ref(
                q_ref[s:e][None], k_ref[s:e][None], v_ref[s:e][None], causal=True
            )[0][0]
            for s, e in zip(cu_seqlens[:-1].tolist(), cu_seqlens[1:].tolist())
        ]
    )
    torch_dq, torch_dk, torch_dv = torch.autograd.grad(
        ref_out, (q_ref, k_ref, v_ref), do.float()
    )

    for tri, ref in zip(
        (triton_dq, triton_dk, triton_dv), (torch_dq, torch_dk, torch_dv)
    ):
        assert torch.isfinite(tri).all()
        # |q| and |k| are ~34 here, so bf16 rounding alone gives ~2% error (the fused backward matches it).
        torch.testing.assert_close(tri, ref.to(tri.dtype), atol=5e-2, rtol=5e-2)


@pytest.mark.parametrize(
    "causal, head_dim_v, seqlen_q, seqlen_k, batch_heads, expected",
    [
        (True, 128, None, None, None, "mid_head"),  # no shape info
        (True, 128, 128, 512, 512, "default"),  # causal rule, both at boundary
        (True, 128, 128, 512, 511, "mid_head"),  # one program short
        (True, 128, 128, 513, 512, "mid_head"),  # KV one token too long
        (True, 96, 512, 512, 128, "default"),  # 128 * cdiv(512, 128) = 512
        (False, 128, 128, 512, 4096, "mid_head"),  # causal rule only
        (False, 97, 128, 256, 1024, "default"),  # non-causal wide-head rule
        (False, 96, 128, 256, 1024, "mid_head"),  # head below min_head_dim_v
        (False, 97, 128, 257, 1024, "mid_head"),
        (False, 97, 128, 256, 1023, "mid_head"),
        (False, 65, 128, 128, 1024, "default"),  # short-KV rule
        (False, 65, 128, 128, 1023, "mid_head"),
        (True, 64, 128, 128, 4096, "small_head"),  # outside mid_head range
        (True, 160, 128, 128, 4096, "default"),
    ],
)
def test_mha_fwd_mid_head_selection(
    causal, head_dim_v, seqlen_q, seqlen_k, batch_heads, expected
):
    from aiter.ops.triton._triton_kernels.attention.mha import _get_config
    from aiter.ops.triton.utils._triton.arch_info import get_arch
    from aiter.ops.triton.utils.config_utils import (
        load_config_json,
        resolve_config_dir,
    )

    fwd_cfg = load_config_json(
        f"{resolve_config_dir('attention', 'MHA', backend='triton')}/DEFAULT.json"
    )["fwd"]
    if get_arch() != "gfx950":
        # Arches without mid_head keep the pre-existing selection.
        assert "mid_head_skip" not in fwd_cfg or "mid_head" in fwd_cfg
        pytest.skip("mid_head_skip rules are only defined for gfx950")
    cfg = _get_config(
        False,
        torch.bfloat16,
        head_dim_v=head_dim_v,
        causal=causal,
        max_seqlen_q=seqlen_q,
        max_seqlen_k=seqlen_k,
        batch_heads=batch_heads,
    )
    assert cfg == fwd_cfg[expected]
