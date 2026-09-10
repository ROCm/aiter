# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import math
import os

import pytest
import torch

from aiter.ops.mha import flash_attn_varlen_func, fmha_v3_varlen_fwd


def _make_packed(sq: int, sk: int, h: int, seed: int = 0):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    q = torch.randn(sq, h, 192, dtype=torch.bfloat16, generator=generator).cuda()
    k = torch.randn(sk, h, 192, dtype=torch.bfloat16, generator=generator).cuda()
    v = torch.randn(sk, h, 128, dtype=torch.bfloat16, generator=generator).cuda()
    cu_q = torch.tensor([0, sq], dtype=torch.int32, device="cuda")
    cu_k = torch.tensor([0, sk], dtype=torch.int32, device="cuda")
    return q, k, v, cu_q, cu_k


def _run_v3(q, k, v, cu_q, cu_k, scale, split_mode, *, return_lse=False):
    previous = os.environ.get("AITER_FMHA_HD192_SPLIT_KV")
    os.environ["AITER_FMHA_HD192_SPLIT_KV"] = split_mode
    try:
        out, lse, _, _ = fmha_v3_varlen_fwd(
            q,
            k,
            v,
            cu_q,
            cu_k,
            q.shape[0],
            k.shape[0],
            0,
            0.0,
            scale,
            0.0,
            False,
            False,
            -1,
            -1,
            return_lse,
            False,
            1,
        )
    finally:
        if previous is None:
            os.environ.pop("AITER_FMHA_HD192_SPLIT_KV", None)
        else:
            os.environ["AITER_FMHA_HD192_SPLIT_KV"] = previous
    return (out, lse) if return_lse else out


def _production_asm(q, k, v, cu_q, cu_k, scale, *, return_lse=False):
    return _run_v3(
        q, k, v, cu_q, cu_k, scale, "0", return_lse=return_lse
    )


def _split_asm(q, k, v, cu_q, cu_k, scale, *, return_lse=False):
    return _run_v3(
        q, k, v, cu_q, cu_k, scale, "force", return_lse=return_lse
    )


def _cosine_difference(reference: torch.Tensor, actual: torch.Tensor) -> float:
    ref = reference.double()
    got = actual.double()
    return 1.0 - 2.0 * (ref * got).sum().item() / max(
        (ref.square() + got.square()).sum().item(), 1e-12
    )


def _assert_close(reference: torch.Tensor, actual: torch.Tensor) -> None:
    assert _cosine_difference(reference, actual) < 1e-4
    torch.testing.assert_close(actual, reference, rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize(
    "sq,sk",
    [
        (1, 96),
        (31, 129),
        (32, 160),
        (33, 191),
        (127, 192),
        (128, 193),
        (129, 224),
        (257, 511),
    ],
)
def test_unified_split3_boundaries(sq, sk):
    q, k, v, cu_q, cu_k = _make_packed(sq, sk, 4, seed=sq + sk)
    scale = 1.0 / math.sqrt(192)
    reference = _production_asm(q, k, v, cu_q, cu_k, scale)
    actual = _split_asm(q, k, v, cu_q, cu_k, scale)
    assert torch.isfinite(actual).all()
    _assert_close(reference, actual)


def test_unified_split3_lse_and_determinism():
    sq, sk, h = 257, 511, 12
    q, k, v, cu_q, cu_k = _make_packed(sq, sk, h, seed=19)
    scale = 0.125
    reference, reference_lse = _production_asm(
        q, k, v, cu_q, cu_k, scale, return_lse=True
    )
    actual, actual_lse = _split_asm(
        q, k, v, cu_q, cu_k, scale, return_lse=True
    )
    _assert_close(reference, actual)
    torch.testing.assert_close(actual_lse, reference_lse, rtol=2e-4, atol=2e-4)
    for _ in range(100):
        repeat = _split_asm(q, k, v, cu_q, cu_k, scale)
        assert torch.equal(actual, repeat)


def test_ticket_shape_public_dispatch():
    sq, sk, h = 4096, 42700, 12
    q, k, v, cu_q, cu_k = _make_packed(sq, sk, h, seed=23)
    scale = 1.0 / math.sqrt(192)
    reference = _production_asm(q, k, v, cu_q, cu_k, scale)
    out = torch.empty(sq, h, 128, dtype=torch.bfloat16, device="cuda")
    previous = os.environ.get("AITER_FMHA_HD192_SPLIT_KV")
    os.environ["AITER_FMHA_HD192_SPLIT_KV"] = "1"
    try:
        actual, lse = flash_attn_varlen_func(
            q,
            k,
            v,
            cu_q,
            cu_k,
            sq,
            sk,
            softmax_scale=scale,
            causal=False,
            return_lse=True,
            out=out,
        )
    finally:
        if previous is None:
            os.environ.pop("AITER_FMHA_HD192_SPLIT_KV", None)
        else:
            os.environ["AITER_FMHA_HD192_SPLIT_KV"] = previous
    assert actual.data_ptr() == out.data_ptr()
    assert torch.isfinite(lse).all()
    _assert_close(reference, actual)
