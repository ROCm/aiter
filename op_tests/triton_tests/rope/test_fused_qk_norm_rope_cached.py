# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch

from aiter.ops.triton.rope.fused_qk_norm_rope_cached import fused_qk_norm_rope_cached


def reference(q, k, qw, kw, cache, eps):
    """The unfused expression: per-head RMSNorm, then partial NeoX RoPE."""

    def rope(x, w):
        d = x.shape[-1]
        x = torch.nn.functional.rms_norm(x, (d,), w, eps)
        rot = cache.shape[-1]
        half = rot // 2
        cos_h, sin_h = cache.split(half, dim=-1)
        cos = torch.cat((cos_h, cos_h), dim=-1).unsqueeze(1)
        sin = torch.cat((sin_h, sin_h), dim=-1).unsqueeze(1)
        x_rot, x_pass = x[..., :rot], x[..., rot:]
        x1, x2 = torch.chunk(x_rot, 2, dim=-1)
        rotated = x_rot * cos + torch.cat((-x2, x1), dim=-1) * sin
        return torch.cat((rotated, x_pass), dim=-1)

    return rope(q, qw), rope(k, kw)


def make(T, H, D, R, dtype, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    q = torch.randn(T, H, D, generator=g, device="cuda", dtype=dtype)
    k = torch.randn(T, H, D, generator=g, device="cuda", dtype=dtype)
    qw = torch.randn(D, generator=g, device="cuda", dtype=dtype)
    kw = torch.randn(D, generator=g, device="cuda", dtype=dtype)
    ang = torch.randn(T, R // 2, generator=g, device="cuda", dtype=torch.float32)
    cache = torch.cat((ang.cos(), ang.sin()), dim=-1).to(dtype)
    return q, k, qw, kw, cache


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize(
    "T,H,D,R",
    [
        (128, 56, 128, 96),  # MiniMax-H3: 0.75 partial rotary, 56 heads
        (1, 56, 128, 96),  # single token
        (77, 7, 128, 96),  # one Ulysses-8 head slice, ragged token count
        (64, 8, 64, 64),  # full rotary
        (64, 8, 128, 64),  # half rotary
        (33, 3, 32, 16),  # small and non-power-of-two head count
    ],
)
def test_matches_norm_then_rope(dtype, T, H, D, R):
    q, k, qw, kw, cache = make(T, H, D, R, dtype)
    wq, wk = reference(q.clone(), k.clone(), qw, kw, cache, 1e-5)
    gq, gk = fused_qk_norm_rope_cached(q, k, qw, kw, cache, eps=1e-5)
    tol = 3e-2 if dtype is torch.bfloat16 else 2e-5
    torch.testing.assert_close(gq.float(), wq.float(), atol=tol, rtol=tol)
    torch.testing.assert_close(gk.float(), wk.float(), atol=tol, rtol=tol)


def test_is_in_place():
    q, k, qw, kw, cache = make(32, 8, 128, 96, torch.bfloat16)
    gq, gk = fused_qk_norm_rope_cached(q, k, qw, kw, cache)
    assert gq.data_ptr() == q.data_ptr() and gk.data_ptr() == k.data_ptr()


def test_q_and_k_use_their_own_weights():
    """A kernel that applied q_weight to both would pass every symmetric test."""
    q, k, qw, kw, cache = make(16, 4, 128, 96, torch.float32)
    kw = kw * 3.0
    wq, wk = reference(q.clone(), k.clone(), qw, kw, cache, 1e-5)
    gq, gk = fused_qk_norm_rope_cached(q, k, qw, kw, cache)
    torch.testing.assert_close(gq, wq, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(gk, wk, atol=2e-5, rtol=2e-5)
    assert not torch.allclose(gq, gk)


def test_untouched_tail_is_passed_through():
    """Dims beyond the rotated width get the norm but no rotation."""
    q, k, qw, kw, cache = make(16, 4, 128, 96, torch.float32)
    q0 = q.clone()
    fused_qk_norm_rope_cached(q, k, qw, kw, cache)
    normed = torch.nn.functional.rms_norm(q0, (128,), qw, 1e-5)
    torch.testing.assert_close(q[..., 96:], normed[..., 96:], atol=2e-5, rtol=2e-5)


def test_rejects_bad_cache_width():
    q, k, qw, kw, cache = make(8, 2, 64, 32, torch.float32)
    with pytest.raises(AssertionError):
        fused_qk_norm_rope_cached(q, k, qw, kw, cache[:, :31])


def test_operates_on_strided_views_of_a_packed_qkv():
    """q and k are slices of one qkv projection; the op must work in place on
    those views, since materialising them is the copy it exists to avoid."""
    T, H, D, R = 64, 8, 128, 96
    inner = H * D
    qkv = torch.randn(T, 3 * inner, device="cuda", dtype=torch.float32)
    ref_qkv = qkv.clone()
    q, k, _ = qkv.split(inner, dim=-1)
    q, k = q.view(T, H, D), k.view(T, H, D)
    assert not q.is_contiguous()
    _, _, qw, kw, cache = make(T, H, D, R, torch.float32)

    rq, rk, _ = ref_qkv.split(inner, dim=-1)
    wq, wk = reference(rq.view(T, H, D), rk.view(T, H, D), qw, kw, cache, 1e-5)
    fused_qk_norm_rope_cached(q, k, qw, kw, cache, eps=1e-5)

    torch.testing.assert_close(q, wq, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(k, wk, atol=2e-5, rtol=2e-5)
    # v must be untouched.
    torch.testing.assert_close(qkv[:, 2 * inner :], ref_qkv[:, 2 * inner :])
