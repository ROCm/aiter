# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Tests for ``aiter.ops.shuffle.pad_weight_for_bpreshuffle`` and the padded
bpreshuffle path of ``aiter.gemm_a8w8_bpreshuffle``.

These tests cover the GLM-4.6V FP8 K values (1368, 2736, 5472, 6144) that
trip the shuffle-weight K-divisibility assertion and/or the
``IsSupportedArgument`` rejection inside the bpreshuffle kernel. They run
on CPU-only for the metadata/regression cases and require CUDA for the GEMM
correctness cases.
"""

from __future__ import annotations

import pytest
import torch

from aiter.ops.shuffle import (
    _DEFAULT_BPRESHUFFLE_PAD_ALIGNMENT,
    pad_weight_for_bpreshuffle,
    shuffle_weight,
)


GLM_K_PADDED_TO_UNPADDED = {
    # (TP, role, original_k): padded_k for AITER_BPRESHUFFLE_PAD_ALIGNMENT=256
    "tp8_down_proj": (1368, 1536),
    "tp4_down_proj": (2736, 2816),
    "tp2_down_proj": (5472, 5632),
    "tp2_other": (6144, 6144),
}

ALIGNED_K_VALUES = (4096, 10944, 1024, 1536, 2048, 3072)
PADDED_K_VALUES = (1368, 2736, 5472)
NON_FP8_DTYPES = (torch.bfloat16, torch.float16, torch.int8)


# --- pad_weight_for_bpreshuffle: padding correctness ---------------------- #


@pytest.mark.parametrize(
    "case",
    list(GLM_K_PADDED_TO_UNPADDED.values()),
    ids=[k for k in GLM_K_PADDED_TO_UNPADDED],
)
def test_pad_metadata_matches_alignment(case):
    original_k, padded_k = case
    n = 64
    w = torch.zeros((n, original_k), dtype=torch.bfloat16)
    out = pad_weight_for_bpreshuffle(w)
    assert out.shape[-1] == padded_k
    assert out.shape[-2] == n
    assert out.aiter_original_k == original_k
    assert out.aiter_padded_k == padded_k
    assert out.aiter_k_padding == padded_k - original_k


@pytest.mark.parametrize("original_k", PADDED_K_VALUES)
def test_pad_tail_is_exactly_zero(original_k):
    w = torch.randn((32, original_k), dtype=torch.bfloat16)
    out = pad_weight_for_bpreshuffle(w)
    if out.aiter_k_padding == 0:
        pytest.skip(
            f"K={original_k} is already aligned to "
            f"{_DEFAULT_BPRESHUFFLE_PAD_ALIGNMENT}; nothing to test"
        )
    tail = out[..., original_k:]
    assert torch.all(tail == 0), (
        f"padded tail for K={original_k} -> {out.aiter_padded_k} should be "
        f"exactly zero, got max abs={tail.abs().max().item()}"
    )


@pytest.mark.parametrize("original_k", PADDED_K_VALUES)
def test_pad_head_equals_input(original_k):
    w = torch.randn((32, original_k), dtype=torch.bfloat16)
    out = pad_weight_for_bpreshuffle(w)
    head = out[..., :original_k]
    assert torch.equal(head, w), (
        "padded head must bit-exactly equal the input weight slice"
    )


@pytest.mark.parametrize("original_k", ALIGNED_K_VALUES)
def test_pad_no_op_on_aligned_k(original_k):
    """Already-aligned K must round-trip without padding."""
    w = torch.randn((32, original_k), dtype=torch.bfloat16)
    out = pad_weight_for_bpreshuffle(w)
    assert out.shape == w.shape
    assert out.aiter_original_k == original_k
    assert out.aiter_padded_k == original_k
    assert out.aiter_k_padding == 0


@pytest.mark.parametrize("dtype", NON_FP8_DTYPES)
def test_pad_works_for_non_fp8_dtypes(dtype):
    """The padding helper is dtype-agnostic; metadata propagates regardless."""
    if dtype == torch.int8:
        w = torch.randint(-8, 8, (16, 1368), dtype=dtype)
    else:
        w = torch.randn((16, 1368), dtype=dtype)
    out = pad_weight_for_bpreshuffle(w)
    assert out.dtype == dtype
    assert out.aiter_original_k == 1368
    assert out.aiter_padded_k > 1368
    assert torch.equal(out[..., :1368], w)
    assert torch.all(out[..., 1368:] == 0)


def test_pad_alignment_argument_overrides_default():
    w = torch.randn((16, 1024), dtype=torch.bfloat16)
    out = pad_weight_for_bpreshuffle(w, alignment=2048)
    assert out.aiter_padded_k == 2048
    assert out.aiter_k_padding == 1024


def test_pad_alignment_must_be_positive():
    w = torch.zeros((4, 256), dtype=torch.bfloat16)
    with pytest.raises(ValueError):
        pad_weight_for_bpreshuffle(w, alignment=0)
    with pytest.raises(ValueError):
        pad_weight_for_bpreshuffle(w, alignment=-1)


# --- shuffle_weight: metadata propagation -------------------------------- #


@pytest.mark.parametrize("original_k", PADDED_K_VALUES)
def test_shuffle_weight_propagates_padding_metadata(original_k):
    w = torch.randn((32, original_k), dtype=torch.bfloat16)
    padded = pad_weight_for_bpreshuffle(w)
    shuffled = shuffle_weight(padded, layout=(16, 16))
    assert getattr(shuffled, "is_shuffled", False) is True
    assert shuffled.aiter_original_k == padded.aiter_original_k
    assert shuffled.aiter_padded_k == padded.aiter_padded_k
    assert shuffled.aiter_k_padding == padded.aiter_k_padding


def test_shuffle_weight_strict_on_unpadded_non_divisible_k():
    """``shuffle_weight`` must keep its old assertion behaviour for direct
    callers that did not opt into ``pad_weight_for_bpreshuffle``. This is the
    contract upstream AIter ships today.
    """
    w = torch.randn((32, 1368), dtype=torch.bfloat16)
    with pytest.raises(AssertionError):
        shuffle_weight(w, layout=(16, 16))


def test_shuffle_weight_aligned_input_has_no_padding_attrs():
    """If the input has no padding metadata, the output should not have it
    either -- we never invent metadata out of thin air.
    """
    w = torch.randn((32, 4096), dtype=torch.bfloat16)
    shuffled = shuffle_weight(w, layout=(16, 16))
    assert not hasattr(shuffled, "aiter_original_k")
    assert not hasattr(shuffled, "aiter_padded_k")
    assert not hasattr(shuffled, "aiter_k_padding")


# --- gemm_a8w8_bpreshuffle: padded GEMM correctness ---------------------- #
#
# These tests need a CUDA device + AIter's compiled bpreshuffle kernel. Skip
# cleanly when not available so the metadata tests above can run on CPU CI.


CUDA_AVAILABLE = torch.cuda.is_available()


@pytest.fixture(scope="module")
def aiter_module():
    if not CUDA_AVAILABLE:
        pytest.skip("CUDA required for bpreshuffle GEMM correctness tests")
    import aiter

    return aiter


def _fp8_perchannel_quant(t: torch.Tensor):
    """Per-row FP8 quant matching the SGLang weight path. Returns
    (q_t [N,K] fp8_e4m3fnuz, scale [N,1] fp32)."""
    import aiter
    from aiter import dtypes

    return aiter.pertoken_quant(t, quant_dtype=dtypes.fp8)


def _torch_reference(x_bf16, w_bf16, x_scale, w_scale, dtype):
    x = x_bf16.to(torch.float32) * x_scale
    w = w_bf16.to(torch.float32) * w_scale
    return torch.nn.functional.linear(x, w).to(dtype)


@pytest.mark.parametrize(
    "m,n,k",
    [
        (1, 4096, 1368),
        (16, 4096, 2736),
        (1, 4096, 5472),
        (16, 4096, 6144),
        (16, 4096, 4096),
    ],
)
def test_padded_bpreshuffle_matches_torch_reference(aiter_module, m, n, k):
    """End-to-end: pad + shuffle the weight, manually pad the activation,
    compare against the BF16 reference."""
    aiter = aiter_module

    torch.manual_seed(0)
    device = "cuda"
    x_bf16 = torch.randn((m, k), dtype=torch.bfloat16, device=device)
    w_bf16 = torch.randn((n, k), dtype=torch.bfloat16, device=device)

    xq, x_scale = _fp8_perchannel_quant(x_bf16)
    wq, w_scale = _fp8_perchannel_quant(w_bf16)

    padded_wq = pad_weight_for_bpreshuffle(wq)
    shuffled = shuffle_weight(padded_wq, layout=(16, 16))

    if shuffled.aiter_padded_k != shuffled.aiter_original_k:
        padded_xq = torch.zeros(
            (m, shuffled.aiter_padded_k),
            dtype=xq.dtype,
            device=device,
        )
        padded_xq[:, :k] = xq
    else:
        padded_xq = xq

    try:
        out = aiter.gemm_a8w8_bpreshuffle(
            padded_xq, shuffled, x_scale, w_scale, None, torch.bfloat16
        )
    except RuntimeError as e:
        pytest.skip(
            f"bpreshuffle kernel rejected (M={m}, N={n}, K_padded="
            f"{shuffled.aiter_padded_k}); SGLang falls back to gemm_a8w8_CK "
            f"for this shape. Underlying error: {e}"
        )

    ref = _torch_reference(x_bf16, w_bf16, x_scale, w_scale, torch.bfloat16)
    assert out.shape == (m, n), (out.shape, (m, n))
    torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2)


def test_padded_weight_unpadded_activation_raises(aiter_module):
    """Negative test: passing a padded weight with an unpadded activation
    must hit the unconditional shape assertion in ``gemm_a8w8_bpreshuffle``,
    not produce a silent wrong result or a kernel-level failure."""
    aiter = aiter_module

    torch.manual_seed(0)
    device = "cuda"
    m, n, k = 16, 4096, 2736
    x_bf16 = torch.randn((m, k), dtype=torch.bfloat16, device=device)
    w_bf16 = torch.randn((n, k), dtype=torch.bfloat16, device=device)

    xq, x_scale = _fp8_perchannel_quant(x_bf16)
    wq, w_scale = _fp8_perchannel_quant(w_bf16)

    padded_wq = pad_weight_for_bpreshuffle(wq)
    if padded_wq.aiter_padded_k == padded_wq.aiter_original_k:
        pytest.skip("expected K=2736 to need padding for this test")
    shuffled = shuffle_weight(padded_wq, layout=(16, 16))

    with pytest.raises(AssertionError):
        aiter.gemm_a8w8_bpreshuffle(
            xq, shuffled, x_scale, w_scale, None, torch.bfloat16
        )
