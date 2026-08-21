# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Tests for gfx950 hd=72 FlyDSL varlen FMHA."""

from __future__ import annotations

import math
import os

import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("flydsl")
from aiter.ops.flydsl import is_flydsl_available
from aiter.ops.flydsl.fmha_kernels import flydsl_flash_attn_varlen_func

if not is_flydsl_available():
    pytest.skip("flydsl is not available", allow_module_level=True)


def _is_gfx950() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        arch = torch.cuda.get_device_properties(0).gcnArchName
    except Exception:  # noqa: BLE001
        return False
    return arch.lower().split(":")[0].startswith("gfx950")


pytestmark = pytest.mark.skipif(
    not _is_gfx950(),
    reason="gfx950 hd=72 FlyDSL FMHA is gfx950 only",
)

HEAD_DIM = 72
SOFTMAX_SCALE = 1.0 / math.sqrt(HEAD_DIM)


def _pack_varlen(seqlens, num_heads, head_dim, dtype, seed=0):
    device = "cuda"
    g = torch.Generator(device=device).manual_seed(seed)
    cu = torch.zeros(len(seqlens) + 1, dtype=torch.int32, device=device)
    cu[1:] = torch.cumsum(torch.tensor(seqlens, dtype=torch.int32, device=device), 0)
    total = int(cu[-1].item())
    q = torch.randn(total, num_heads, head_dim, generator=g, dtype=dtype, device=device)
    k = torch.randn(total, num_heads, head_dim, generator=g, dtype=dtype, device=device)
    v = torch.randn(total, num_heads, head_dim, generator=g, dtype=dtype, device=device)
    return q, k, v, cu, max(seqlens)


def _ref_varlen(q, k, v, cu, scale):
    outs = []
    batch = cu.numel() - 1
    for i in range(batch):
        s = int(cu[i].item())
        e = int(cu[i + 1].item())
        if e == s:
            continue
        qi = q[s:e].transpose(0, 1).float()
        ki = k[s:e].transpose(0, 1).float()
        vi = v[s:e].transpose(0, 1).float()
        oi = F.scaled_dot_product_attention(qi, ki, vi, scale=scale, is_causal=False)
        outs.append(oi.transpose(0, 1).to(q.dtype))
    return torch.cat(outs, 0)


def _assert_close(out, ref, min_cos=0.99, mean_cos=0.999):
    assert out.shape == ref.shape
    assert out.dtype == ref.dtype
    cos = F.cosine_similarity(
        out.float().reshape(-1, HEAD_DIM),
        ref.float().reshape(-1, HEAD_DIM),
        dim=1,
    )
    assert cos.min().item() > min_cos, f"min_cos={cos.min().item():.6f}"
    assert cos.mean().item() > mean_cos, f"mean_cos={cos.mean().item():.6f}"


@pytest.mark.parametrize("seqlen", [128, 256])
def test_flydsl_fmha_gfx950_single_seq(seqlen):
    q, k, v, cu, max_s = _pack_varlen([seqlen], num_heads=16, head_dim=HEAD_DIM, dtype=torch.bfloat16)
    out = flydsl_flash_attn_varlen_func(
        q, k, v, cu, cu, max_s, max_s, softmax_scale=SOFTMAX_SCALE, causal=False
    )
    assert out is not None
    ref = _ref_varlen(q, k, v, cu, SOFTMAX_SCALE)
    _assert_close(out, ref)


def test_flydsl_fmha_gfx950_packed_remainders():
    seqlens = [17, 64, 80, 130]
    q, k, v, cu, max_s = _pack_varlen(seqlens, num_heads=16, head_dim=HEAD_DIM, dtype=torch.bfloat16)
    out = flydsl_flash_attn_varlen_func(
        q, k, v, cu, cu, max_s, max_s, softmax_scale=SOFTMAX_SCALE, causal=False
    )
    assert out is not None
    ref = _ref_varlen(q, k, v, cu, SOFTMAX_SCALE)
    _assert_close(out, ref)


def test_flydsl_fmha_gfx950_skips_causal():
    q, k, v, cu, max_s = _pack_varlen([128], num_heads=4, head_dim=HEAD_DIM, dtype=torch.bfloat16)
    got = flydsl_flash_attn_varlen_func(
        q, k, v, cu, cu, max_s, max_s, softmax_scale=SOFTMAX_SCALE, causal=True
    )
    assert got is None


def test_flydsl_fmha_gfx950_skips_other_head_dim():
    q, k, v, cu, max_s = _pack_varlen([128], num_heads=4, head_dim=64, dtype=torch.bfloat16)
    got = flydsl_flash_attn_varlen_func(
        q, k, v, cu, cu, max_s, max_s, softmax_scale=1.0 / math.sqrt(64), causal=False
    )
    assert got is None


@pytest.mark.skipif(
    os.environ.get("AITER_FMHA_GFX950_BENCH", "0") != "1",
    reason="set AITER_FMHA_GFX950_BENCH=1 to time vs CK",
)
@pytest.mark.parametrize("num_seqs", [2, 4, 8, 10])
def test_flydsl_fmha_gfx950_bench_vs_ck(num_seqs):
    from aiter.ops.mha import flash_attn_varlen_func as mha_varlen

    seqlens = [5776] * num_seqs
    q, k, v, cu, max_s = _pack_varlen(
        seqlens, num_heads=16, head_dim=HEAD_DIM, dtype=torch.bfloat16, seed=1
    )
    os.environ["AITER_FLYDSL_FMHA_HD72"] = "1"
    out_fd = flydsl_flash_attn_varlen_func(
        q, k, v, cu, cu, max_s, max_s, softmax_scale=SOFTMAX_SCALE, causal=False
    )
    assert out_fd is not None

    def _sync_time(fn, iters=20, warmup=5):
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) * 1000.0 / iters

    fly_us = _sync_time(
        lambda: flydsl_flash_attn_varlen_func(
            q, k, v, cu, cu, max_s, max_s, softmax_scale=SOFTMAX_SCALE, causal=False
        )
    )
    old = os.environ.get("AITER_FLYDSL_FMHA_HD72", "1")
    os.environ["AITER_FLYDSL_FMHA_HD72"] = "0"
    try:
        ck_us = _sync_time(
            lambda: mha_varlen(
                q,
                k,
                v,
                cu,
                cu,
                max_s,
                max_s,
                softmax_scale=SOFTMAX_SCALE,
                causal=False,
            )
        )
    finally:
        os.environ["AITER_FLYDSL_FMHA_HD72"] = old
    print(f"num_seqs={num_seqs} flydsl={fly_us:.1f}us ck={ck_us:.1f}us")
    assert fly_us > 0 and ck_us > 0
