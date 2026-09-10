# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import math

import pytest
import torch

import aiter
from aiter.ops.mha import flash_attn_varlen_func, fmha_v3_varlen_fwd

_DEVICE_NAME = torch.cuda.get_device_name()
pytestmark = pytest.mark.skipif(
    aiter.get_gfx() != "gfx942"
    or not any(device in _DEVICE_NAME for device in ("MI300X", "MI325X")),
    reason="split-KV ASM is validated only on gfx942 MI300X/MI325X",
)


def _make_packed(sq: int, sk: int, h: int, seed: int = 0):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    q = torch.randn(sq, h, 192, dtype=torch.bfloat16, generator=generator).cuda()
    k = torch.randn(sk, h, 192, dtype=torch.bfloat16, generator=generator).cuda()
    v = torch.randn(sk, h, 128, dtype=torch.bfloat16, generator=generator).cuda()
    cu_q = torch.tensor([0, sq], dtype=torch.int32, device="cuda")
    cu_k = torch.tensor([0, sk], dtype=torch.int32, device="cuda")
    return q, k, v, cu_q, cu_k


def _run_v3(q, k, v, cu_q, cu_k, scale, num_splits, *, return_lse=False):
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
        num_splits=num_splits,
    )
    return (out, lse) if return_lse else out


def _production_asm(q, k, v, cu_q, cu_k, scale, *, return_lse=False):
    return _run_v3(
        q, k, v, cu_q, cu_k, scale, 1, return_lse=return_lse
    )


def _split_asm(q, k, v, cu_q, cu_k, scale, num_splits=3, *, return_lse=False):
    return _run_v3(
        q, k, v, cu_q, cu_k, scale, num_splits, return_lse=return_lse
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


def test_splitkv_one_routes_to_production():
    sq, sk, h = 129, 511, 4
    q, k, v, cu_q, cu_k = _make_packed(sq, sk, h, seed=17)
    scale = 1.0 / math.sqrt(192)
    reference, reference_lse = _production_asm(
        q, k, v, cu_q, cu_k, scale, return_lse=True
    )
    actual, actual_lse = _run_v3(
        q, k, v, cu_q, cu_k, scale, 1, return_lse=True
    )
    assert torch.equal(actual, reference)
    assert torch.equal(actual_lse, reference_lse)


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
def test_splitkv_boundaries(sq, sk):
    q, k, v, cu_q, cu_k = _make_packed(sq, sk, 4, seed=sq + sk)
    scale = 1.0 / math.sqrt(192)
    reference = _production_asm(q, k, v, cu_q, cu_k, scale)
    actual = _split_asm(q, k, v, cu_q, cu_k, scale)
    assert torch.isfinite(actual).all()
    _assert_close(reference, actual)


@pytest.mark.parametrize("num_splits", range(2, 9))
def test_splitkv_counts(num_splits):
    sq, sk, h = 129, 2048, 4
    q, k, v, cu_q, cu_k = _make_packed(sq, sk, h, seed=100 + num_splits)
    scale = 1.0 / math.sqrt(192)
    reference, reference_lse = _production_asm(
        q, k, v, cu_q, cu_k, scale, return_lse=True
    )
    actual, actual_lse = _split_asm(
        q, k, v, cu_q, cu_k, scale, num_splits, return_lse=True
    )
    _assert_close(reference, actual)
    torch.testing.assert_close(actual_lse, reference_lse, rtol=2e-4, atol=2e-4)


def test_splitkv_rejects_empty_final_partition():
    sq, sk, h = 129, 511, 4
    q, k, v, cu_q, cu_k = _make_packed(sq, sk, h, seed=21)
    with pytest.raises(RuntimeError, match="empty final KV partition"):
        _split_asm(q, k, v, cu_q, cu_k, 1.0 / math.sqrt(192), num_splits=5)


def test_public_explicit_splitkv_dispatch():
    sq, sk, h = 129, 2048, 4
    q, k, v, cu_q, cu_k = _make_packed(sq, sk, h, seed=22)
    scale = 1.0 / math.sqrt(192)
    reference = _production_asm(q, k, v, cu_q, cu_k, scale)
    actual = flash_attn_varlen_func(
        q,
        k,
        v,
        cu_q,
        cu_k,
        sq,
        sk,
        softmax_scale=scale,
        causal=False,
        num_splits=5,
    )
    _assert_close(reference, actual)


def test_public_splitkv_rejects_incompatible_input():
    sq, sk, h = 129, 2048, 4
    q, k, v, cu_q, cu_k = _make_packed(sq, sk, h, seed=24)
    with pytest.raises(ValueError, match="requires the gfx942"):
        flash_attn_varlen_func(
            q,
            k,
            v,
            cu_q,
            cu_k,
            sq,
            sk,
            dropout_p=0.1,
            num_splits=5,
        )


def test_splitkv_lse_and_determinism():
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


def test_automatic_splitkv_dispatch():
    sq, sk, h = 4096, 42700, 12
    q, k, v, cu_q, cu_k = _make_packed(sq, sk, h, seed=23)
    scale = 1.0 / math.sqrt(192)
    reference = _production_asm(q, k, v, cu_q, cu_k, scale)
    out = torch.empty(sq, h, 128, dtype=torch.bfloat16, device="cuda")
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
        num_splits=0,
    )
    assert actual.data_ptr() == out.data_ptr()
    assert torch.isfinite(lse).all()
    _assert_close(reference, actual)
