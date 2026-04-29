# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Tests for ``flydsl_flash_attn_func`` (gfx1201 / RDNA4)."""

from __future__ import annotations

from typing import Tuple

import pytest
import torch
import torch.nn.functional as F

flydsl = pytest.importorskip("flydsl")
from aiter.ops.flydsl import is_flydsl_available, flydsl_flash_attn_func  # noqa: E402

if not is_flydsl_available():
    pytest.skip("flydsl is not available", allow_module_level=True)


def _is_gfx1201() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        arch = torch.cuda.get_device_properties(0).gcnArchName
    except Exception:
        return False
    return arch.lower().split(":")[0].startswith("gfx1201")


pytestmark = pytest.mark.skipif(
    not _is_gfx1201(),
    reason="flydsl_flash_attn_func is gfx1201/RDNA4 only",
)


def _ref_sdpa_bshd(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """SDPA reference with BSHD inputs/outputs."""
    out_bhsd = F.scaled_dot_product_attention(
        q.transpose(1, 2).contiguous(),
        k.transpose(1, 2).contiguous(),
        v.transpose(1, 2).contiguous(),
        is_causal=False,
    )
    return out_bhsd.transpose(1, 2).contiguous()


def _make_qkv(
    batch: int,
    seq_len: int,
    num_heads: int,
    head_dim: int,
    dtype: torch.dtype,
    seed: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    g = torch.Generator(device="cuda").manual_seed(seed)
    shape = (batch, seq_len, num_heads, head_dim)
    q = torch.randn(shape, generator=g, dtype=dtype, device="cuda")
    k = torch.randn(shape, generator=g, dtype=dtype, device="cuda")
    v = torch.randn(shape, generator=g, dtype=dtype, device="cuda")
    return q, k, v


@pytest.mark.parametrize(
    "batch,seq_len,num_heads,head_dim",
    [
        # Aligned production-like Wan2.1 1.3B shape, padded to multiple of 128.
        (1, 32768, 12, 128),
        # Smaller aligned shape (sanity).
        (2, 1024, 8, 128),
        # Unaligned shape — exercises the auto-padding path. 32760 → 32768.
        (1, 32760, 12, 128),
    ],
)
def test_flydsl_fmha_correctness_bf16(batch, seq_len, num_heads, head_dim):
    q, k, v = _make_qkv(batch, seq_len, num_heads, head_dim, torch.bfloat16)
    out = flydsl_flash_attn_func(q, k, v, causal=False)
    ref = _ref_sdpa_bshd(q, k, v)

    assert out.shape == ref.shape == (batch, seq_len, num_heads, head_dim)
    assert out.dtype == ref.dtype == torch.bfloat16

    cos = F.cosine_similarity(
        out.float().reshape(-1, head_dim),
        ref.float().reshape(-1, head_dim),
        dim=1,
    )
    # bf16 attention is noisy; cosine is the right correctness signal.
    assert cos.min().item() > 0.99, f"min_cos={cos.min().item():.6f}"
    assert cos.mean().item() > 0.999, f"mean_cos={cos.mean().item():.6f}"


def test_flydsl_fmha_rejects_cross_attention():
    q = torch.randn(1, 1024, 12, 128, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(1, 512, 12, 128, dtype=torch.bfloat16, device="cuda")
    v = torch.randn(1, 512, 12, 128, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(ValueError, match="self-attention"):
        flydsl_flash_attn_func(q, k, v)


def test_flydsl_fmha_rejects_unsupported_head_dim():
    q = torch.randn(1, 256, 8, 48, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(ValueError, match="head_dim"):
        flydsl_flash_attn_func(q, q.clone(), q.clone())


def test_flydsl_fmha_rejects_dtype_mismatch():
    q = torch.randn(1, 1024, 8, 128, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(1, 1024, 8, 128, dtype=torch.float16, device="cuda")
    v = torch.randn(1, 1024, 8, 128, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(ValueError, match="dtype"):
        flydsl_flash_attn_func(q, k, v)
