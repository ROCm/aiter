# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import torch
import pytest
import logging
from aiter.ops.triton.attention.mha import (
    flash_attn_func,
    flash_attn_varlen_func,
    mha_set_use_fused_bwd_kernel,
    mha_set_use_int64_strides,
)
from aiter.ops.triton.attention.mha_v3 import (
    flash_attn_fp8_func,
    flash_attn_varlen_fp8_func,
)
from aiter.test_mha_common import (
    attention_ref,
    generate_random_padding_mask,
    generate_qkv,
)
from op_tests.triton_tests.attention.mha_test_utils import pad_rearrange_dropout_mask

from aiter.ops.triton.utils._triton.arch_info import get_arch
from aiter.ops.triton._triton_kernels.flash_attn_triton_amd.utils import FP8_ARCHS

arch = get_arch()
_supports_fp8 = arch in FP8_ARCHS

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
DEBUG_MODE = False


def _attention_ref_with_tol(q, k, v, do, is_fp8=False, **kwargs):
    """Run attention reference and compute adaptive tolerances.

    Follows the upstream flash attention tolerance pattern
    (see tests/test_flash_attn.py in Dao-AILab/flash-attention). Runs two
    PyTorch references (upcast and non-upcast) and uses the gap between them as
    a baseline for tolerance.

    Returns (out, (dq, dk, dv), fwd_tol, [dq_tol, dk_tol, dv_tol])
    where each tol is (atol, rtol).
    """
    has_dropout = kwargs.get("dropout_p", 0.0) > 0.0

    def _run_ref(upcast, reorder_ops=False):
        q_ = q.detach().clone().requires_grad_(True)
        k_ = k.detach().clone().requires_grad_(True)
        v_ = v.detach().clone().requires_grad_(True)
        with torch.enable_grad():
            out, _, _ = attention_ref(
                q_, k_, v_, upcast=upcast, reorder_ops=reorder_ops, **kwargs
            )
        dq, dk, dv = torch.autograd.grad(out, (q_, k_, v_), do)
        return out, dq, dk, dv

    def _tol(ref_val, pt_val, is_forward=False):
        baseline = (pt_val - ref_val).abs().max().item()
        if is_fp8:
            mult = 4
            atol_floor = 5e-1 if is_forward else 1.0
            rtol_floor = 1e-1
        elif has_dropout:
            # Dropout scaling (1/(1-p)) amplifies precision errors in the
            # fused kernel differently than in the reference. The baseline
            # between two references uses the same mask so it underestimates
            # the kernel-vs-reference gap.
            mult = 2
            atol_floor = 1e-1 if is_forward else 2.0
            rtol_floor = 1e-1
        else:
            mult = 2
            atol_floor = 1e-2 if is_forward else 1.5e-2
            rtol_floor = 1e-5
        atol = max(mult * baseline, atol_floor)
        return atol, rtol_floor

    out, dq, dk, dv = _run_ref(upcast=True)
    out_pt, dq_pt, dk_pt, dv_pt = _run_ref(upcast=False, reorder_ops=True)

    fwd_tol = _tol(out, out_pt, is_forward=True)
    bwd_tols = [_tol(dq, dq_pt), _tol(dk, dk_pt), _tol(dv, dv_pt)]

    return out, (dq, dk, dv), fwd_tol, bwd_tols


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
    FP8: bool,
    dtype=torch.float16,
):
    torch.cuda.empty_cache()
    q = torch.randn((BATCH, SEQLEN_Q, NUM_Q_HEADS, HEAD_SZ), device="cuda", dtype=dtype)
    k = torch.randn((BATCH, SEQLEN_K, NUM_K_HEADS, HEAD_SZ), device="cuda", dtype=dtype)
    v = torch.randn((BATCH, SEQLEN_K, NUM_K_HEADS, HEAD_SZ), device="cuda", dtype=dtype)

    dropout_mask = None
    if FP8:
        if not _supports_fp8:
            pytest.skip(f"FP8 not supported on {arch}")
        if DROPOUT > 0.0 or RETURN_LSE or RETURN_SOFTMAX:
            pytest.skip(
                "FP8 mode does not support dropout_p, return_lse, or return_attn_probs"
            )
        if CAUSAL and (SEQLEN_Q * SEQLEN_K > 128 * 128):
            pytest.skip(
                "FP8+CAUSAL for big sequence lenghts results in random precision errors"
            )

        triton_out = flash_attn_fp8_func(
            q,
            k,
            v,
            causal=CAUSAL,
        )
    else:
        triton_out = flash_attn_func(
            q,
            k,
            v,
            dropout_p=DROPOUT,
            causal=CAUSAL,
            return_lse=RETURN_LSE,
            return_attn_probs=RETURN_SOFTMAX,
        )

    if RETURN_LSE:
        assert len(triton_out) > 1
        lse = triton_out[1]
        if DEBUG_MODE:
            print(f"lse.shape={lse.shape}, lse={lse}")

    if DROPOUT > 0.0 and RETURN_SOFTMAX:
        if RETURN_LSE:
            assert len(triton_out) == 3
            sd_mask = triton_out[2]
        else:
            assert len(triton_out) == 2
            sd_mask = triton_out[1]
        dropout_mask = sd_mask >= 0
        if DEBUG_MODE:
            print(f"sd_mask.shape={sd_mask.shape}, sd_mask={sd_mask}")
            print(
                f"dropout_mask.shape={dropout_mask.shape}, dropout_mask={dropout_mask}"
            )

    if RETURN_SOFTMAX or RETURN_LSE:
        triton_out = triton_out[0]
    if DEBUG_MODE:
        print(f"triton_out.shape={triton_out.shape}, triton_out={triton_out}")

    torch_out = attention_ref(
        q, k, v, dropout_p=DROPOUT, dropout_mask=dropout_mask, causal=CAUSAL
    )
    torch_out, attention_scores, _ = torch_out
    if DEBUG_MODE:
        print(f"torch_out.shape={torch_out.shape}, torch_out={torch_out}")
        print(
            f"attention_scores.shape={attention_scores.shape}, attention_scores={attention_scores}"
        )

    if FP8:
        fp8_assert_close(triton_out, torch_out.to(triton_out.dtype))
    else:
        torch.testing.assert_close(triton_out, torch_out, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("BATCH", [1, 30, 50])
@pytest.mark.parametrize(
    "SEQLEN_Q, SEQLEN_K",
    [(1, 1), (128, 128), (32, 16), (64, 128), (2048, 2048)],
)
@pytest.mark.parametrize("NUM_Q_HEADS, NUM_K_HEADS", [(1, 1), (8, 8), (48, 8)])
@pytest.mark.parametrize("HEAD_SZ", [64, 128])
@pytest.mark.parametrize("CAUSAL", [(True), (False)])
@pytest.mark.parametrize("FP8", [(True), (False)])
def test_mha(
    BATCH: int,
    SEQLEN_Q: int,
    SEQLEN_K: int,
    NUM_Q_HEADS: int,
    NUM_K_HEADS: int,
    HEAD_SZ: int,
    CAUSAL: bool,
    FP8: bool,
    dtype=torch.bfloat16,
):
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
        FP8=FP8,
        dtype=dtype,
    )


@pytest.mark.parametrize("NUM_Q_HEADS, NUM_K_HEADS", [(1, 1), (8, 1)])
@pytest.mark.parametrize("DROPOUT, RETURN_LSE, RETURN_SOFTMAX, ", [(0.2, True, True)])
@pytest.mark.parametrize("CAUSAL", [(True), (False)])
@pytest.mark.parametrize("FP8", [(True), (False)])
def test_mha_with_dropout(
    NUM_Q_HEADS: int,
    NUM_K_HEADS: int,
    DROPOUT: float,
    RETURN_LSE: bool,
    RETURN_SOFTMAX: bool,
    CAUSAL: bool,
    FP8: bool,
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
        FP8=FP8,
        dtype=dtype,
    )


# LLaMA 3 405B config
def test_mha_int64_strides(
    dtype=torch.float16,
    device="cuda",
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
    torch.cuda.empty_cache()
    torch.manual_seed(20)
    # use int64 strides.
    mha_set_use_int64_strides(
        True
    )  # NOTE: if you set this to false this test case will segfault

    # generate inputs with large strides
    def _generate_input(
        batch: int, seqlen: int, nheads: int, dim_size: int, large_stride: bool = False
    ) -> torch.Tensor:
        seqlens = torch.full((batch,), seqlen)
        cu_seqlens = torch.cat(
            [
                torch.tensor([0], dtype=torch.int32),
                seqlens.cumsum(dim=0, dtype=torch.int32),
            ]
        ).to(device="cuda")
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

    if DEBUG_MODE:
        print()
        print("q:", q.shape, q.stride())
        print("k:", k.shape, k.stride())
        print("v:", v.shape, v.stride())
        print("cu_seqlens_q:", cu_seqlens_q.shape, cu_seqlens_q.stride())
        print("cu_seqlens_k:", cu_seqlens_k.shape, cu_seqlens_k.stride())

    triton_out, _ = flash_attn_varlen_func(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlens_q,
        max_seqlens_k,
        dropout_p=DROPOUT,
        causal=CAUSAL,
        return_lse=True,
    )
    if test_backward:
        triton_dq, triton_dk, triton_dv = torch.autograd.grad(
            triton_out, (q, k, v), do.clone()
        )

    # NOTE: use fwd output to wait not exit program before kernel finishes
    print("triton_out:", triton_out)
    if test_backward:
        print("triton_dq:", triton_dq.shape, triton_dq.stride())
        print("triton_dk:", triton_dk.shape, triton_dk.stride())
        print("triton_dv:", triton_dv.shape, triton_dv.stride())


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
    FP8: bool,
    dtype=torch.float16,
):
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
        dq_pad_fn,
        dk_pad_fn,
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
    if FP8:
        if not _supports_fp8:
            pytest.skip(f"FP8 not supported on {arch}")
        if DROPOUT > 0.0 or RETURN_LSE or RETURN_SOFTMAX:
            pytest.skip(
                "FP8 varlen mode does not support dropout_p, return_lse, or return_attn_probs"
            )

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
    else:
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
        )

    if RETURN_LSE:
        assert len(triton_out) > 1
        lse = triton_out[1]
        if DEBUG_MODE:
            print(f"lse.shape={lse.shape}, lse={lse}")

    dropout_mask = None
    if DROPOUT > 0.0 and RETURN_SOFTMAX:
        if RETURN_LSE:
            assert len(triton_out) == 3
            sd_mask = triton_out[2]
        else:
            assert len(triton_out) == 2
            sd_mask = triton_out[1]
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
        if DEBUG_MODE:
            # print(f"sd_mask.shape={sd_mask.shape}, sd_mask={sd_mask}")
            print(
                f"dropout_mask.shape={dropout_mask.shape}, dropout_mask={dropout_mask}"
            )
    if RETURN_SOFTMAX or RETURN_LSE:
        triton_out = output_pad_fn(triton_out[0])
    else:
        triton_out = output_pad_fn(triton_out)
    if DEBUG_MODE:
        print(f"triton_out.shape={triton_out.shape}, triton_out={triton_out}")

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
    torch_out, attention_scores, _ = torch_out

    if DEBUG_MODE:
        print(f"torch_out.shape={torch_out.shape}, torch_out={torch_out}")
        print(
            f"attention_scores.shape={attention_scores.shape}, attention_scores={attention_scores}"
        )

    if FP8:
        fp8_assert_close(triton_out, torch_out.to(triton_out.dtype))
    else:
        torch.testing.assert_close(
            triton_out, torch_out.to(triton_out.dtype), atol=1e-1, rtol=1e-1
        )


@pytest.mark.parametrize("BATCH", [1, 4, 30, 50])
@pytest.mark.parametrize(
    "SEQLEN_Q, SEQLEN_K",
    [(1, 1), (128, 128), (32, 16), (64, 128), (2048, 2048)],
)
@pytest.mark.parametrize(
    "NUM_Q_HEADS, NUM_K_HEADS", [(1, 1), (16, 16), (2, 1), (48, 8)]
)
@pytest.mark.parametrize("HEAD_SZ", [8, 32, 128])
@pytest.mark.parametrize("CAUSAL", [(True), (False)])
@pytest.mark.parametrize("FP8", [(False), (True)])
def test_mha_varlen(
    BATCH: int,
    SEQLEN_Q: int,
    SEQLEN_K: int,
    NUM_Q_HEADS: int,
    NUM_K_HEADS: int,
    HEAD_SZ: int,
    CAUSAL: bool,
    FP8: bool,
    dtype=torch.float16,
):
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
        FP8=FP8,
        dtype=dtype,
    )


@pytest.mark.parametrize("NUM_Q_HEADS, NUM_K_HEADS", [(1, 1), (8, 1)])
@pytest.mark.parametrize("DROPOUT, RETURN_LSE, RETURN_SOFTMAX, ", [(0.2, True, True)])
@pytest.mark.parametrize("CAUSAL", [(True), (False)])
@pytest.mark.parametrize("FP8", [(False), (True)])
def test_mha_varlen_with_dropout(
    NUM_Q_HEADS: int,
    NUM_K_HEADS: int,
    DROPOUT: float,
    RETURN_LSE: bool,
    RETURN_SOFTMAX: bool,
    CAUSAL: bool,
    FP8: bool,
    dtype=torch.float16,
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
        FP8=FP8,
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
@pytest.mark.parametrize("FUSED", [False, True])
@pytest.mark.parametrize("FP8", [True, False])
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
    FP8: bool,
    dtype=torch.float16,
):
    HAS_DROPOUT = DROPOUT > 0.0

    if FP8 and not _supports_fp8:
        pytest.skip(f"FP8 not supported on {arch}")
    if FUSED and CAUSAL:
        pytest.skip("FUSED+CAUSAL results in NaNs")
    if FP8 and HAS_DROPOUT:
        pytest.skip("FP8 does not support dropout")
    if CAUSAL and HAS_DROPOUT:
        pytest.skip("CAUSAL+DROPOUT backward results in NaNs")
    if FP8 and CAUSAL:
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
        if FP8:
            triton_out = flash_attn_fp8_func(q, k, v, causal=CAUSAL)
            dropout_mask = None
        else:
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
    torch_out, torch_grads, fwd_tol, bwd_tols = _attention_ref_with_tol(
        q,
        k,
        v,
        do,
        is_fp8=FP8,
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
        if FP8:
            assert_cosine_similarity(tri, ref)


@pytest.mark.parametrize("BATCH", [2, 4])
@pytest.mark.parametrize("SEQLEN_Q", [256, 512])
@pytest.mark.parametrize("SEQLEN_K", [256, 512])
@pytest.mark.parametrize("NUM_Q_HEADS", [32])
@pytest.mark.parametrize("NUM_K_HEADS", [8])
@pytest.mark.parametrize("HEAD_SZ", [128])
@pytest.mark.parametrize("CAUSAL", [False])
@pytest.mark.parametrize("FUSED", [False, True])
def test_mha_backward_sbhd_do(
    BATCH: int,
    SEQLEN_Q: int,
    SEQLEN_K: int,
    NUM_Q_HEADS: int,
    NUM_K_HEADS: int,
    HEAD_SZ: int,
    CAUSAL: bool,
    FUSED: bool,
    dtype=torch.float16,
):
    """Verify backward correctness when dO has SBHD memory layout (strides differ from O).

    Creates dO as a (seqlen, batch, nheads, headdim) tensor transposed to
    (batch, seqlen, nheads, headdim), so its strides are different from the
    contiguous BSHD output tensor. This exercises the independent stride
    handling for dO in _bwd_preprocess.
    """
    torch.cuda.empty_cache()
    torch.manual_seed(42)
    if FUSED and CAUSAL:
        pytest.skip("FUSED+CAUSAL results in NaNs")

    mha_set_use_fused_bwd_kernel(FUSED)

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
    torch_out, torch_grads, fwd_tol, bwd_tols = _attention_ref_with_tol(
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
@pytest.mark.parametrize("FUSED", [False, True])
@pytest.mark.parametrize("FP8", [True, False])
def test_mha_backward_varlen(
    SEQLEN_Q: int,
    SEQLEN_K: int,
    NUM_Q_HEADS: int,
    CAUSAL: bool,
    DROPOUT: float,
    FUSED: bool,
    FP8: bool,
    dtype=torch.float16,
):
    BATCH = 3
    HEAD_SZ = 128
    NUM_K_HEADS = 8
    HAS_DROPOUT = DROPOUT > 0.0

    if FP8 and not _supports_fp8:
        pytest.skip(f"FP8 not supported on {arch}")
    if FUSED and CAUSAL:
        pytest.skip("FUSED+CAUSAL results in NaNs")
    if FP8 and HAS_DROPOUT:
        pytest.skip("FP8 does not support dropout")
    if CAUSAL and HAS_DROPOUT:
        pytest.skip("CAUSAL+DROPOUT backward results in NaNs")

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
        if FP8:
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
            dropout_mask = None
        else:
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
    torch_out, torch_grads, fwd_tol, bwd_tols = _attention_ref_with_tol(
        q,
        k,
        v,
        do,
        is_fp8=FP8,
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
        if FP8:
            assert_cosine_similarity(tri, ref)
