# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Tests for ``flydsl_sage_attn_func`` (CDNA / gfx942 / gfx950)."""

from __future__ import annotations

from typing import Tuple

import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("flydsl")
from aiter.ops.flydsl import is_flydsl_available, flydsl_sage_attn_func  # noqa: E402

if not is_flydsl_available():
    pytest.skip("flydsl is not available", allow_module_level=True)

try:
    from aiter.ops.triton.attention.fav3_sage import fav3_sage_wrapper_func
    HAS_TRITON_SAGE = True
except Exception:
    HAS_TRITON_SAGE = False


def _is_cdna() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        arch = torch.cuda.get_device_properties(0).gcnArchName
    except Exception:
        return False
    arch_base = arch.lower().split(":")[0]
    return arch_base.startswith("gfx942") or arch_base.startswith("gfx950")


pytestmark = pytest.mark.skipif(
    not _is_cdna(),
    reason="flydsl_sage_attn_func is gfx942/gfx950 (CDNA) only",
)


def _ref_sdpa_bshd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
    softmax_scale: float | None = None,
) -> torch.Tensor:
    """SDPA reference with BSHD inputs/outputs (runs in fp32 for accuracy)."""
    q_f32 = q.float().transpose(1, 2).contiguous()  # BHSD
    k_f32 = k.float().transpose(1, 2).contiguous()
    v_f32 = v.float().transpose(1, 2).contiguous()
    scale = softmax_scale or (q.shape[-1] ** -0.5)
    out_bhsd = F.scaled_dot_product_attention(
        q_f32, k_f32, v_f32, is_causal=causal, scale=scale
    )
    return out_bhsd.transpose(1, 2).contiguous()  # BSHD


def _make_qkv(
    batch: int,
    seq_len: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    dtype: torch.dtype,
    seed: int = 42,
    device: str = "cuda",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    g = torch.Generator(device=device).manual_seed(seed)
    q = torch.randn((batch, seq_len, num_q_heads, head_dim), generator=g, dtype=dtype, device=device)
    k = torch.randn((batch, seq_len, num_kv_heads, head_dim), generator=g, dtype=dtype, device=device)
    v = torch.randn((batch, seq_len, num_kv_heads, head_dim), generator=g, dtype=dtype, device=device)
    return q, k, v


# Int8 quantization introduces ~1-2% noise; use cos_sim > 0.97 as the bound.
_COS_MIN = 0.97
_COS_MEAN_MIN = 0.99


@pytest.mark.parametrize(
    "batch,seq_len,num_q_heads,num_kv_heads,head_dim",
    [
        (1, 4096, 8, 8, 128),    # standard MHA, seq aligned to 256
        (2, 2048, 16, 16, 128),  # larger heads
        (1, 4096, 16, 4, 128),   # GQA: 4 KV groups for 16 Q heads
        (1, 4096, 8, 1, 128),    # MQA: 1 KV head
    ],
)
def test_flydsl_sage_correctness_bf16(batch, seq_len, num_q_heads, num_kv_heads, head_dim):
    q, k, v = _make_qkv(batch, seq_len, num_q_heads, num_kv_heads, head_dim, torch.bfloat16)
    out = flydsl_sage_attn_func(q, k, v, causal=False)
    ref = _ref_sdpa_bshd(q, k, v, causal=False)

    assert out.shape == (batch, seq_len, num_q_heads, head_dim), f"shape mismatch: {out.shape}"
    assert out.dtype == torch.bfloat16, f"expected bf16 output, got {out.dtype}"

    cos = F.cosine_similarity(
        out.float().reshape(-1, head_dim),
        ref.float().reshape(-1, head_dim),
        dim=1,
    )
    assert cos.min().item() > _COS_MIN, (
        f"min_cos={cos.min().item():.4f} < {_COS_MIN} "
        f"(batch={batch}, seq={seq_len}, Hq={num_q_heads}, Hkv={num_kv_heads})"
    )
    assert cos.mean().item() > _COS_MEAN_MIN, (
        f"mean_cos={cos.mean().item():.4f} < {_COS_MEAN_MIN}"
    )


def test_flydsl_sage_correctness_causal():
    """Causal masking: lower-triangular attention."""
    batch, seq_len, num_q_heads, num_kv_heads, head_dim = 1, 4096, 8, 8, 128
    q, k, v = _make_qkv(batch, seq_len, num_q_heads, num_kv_heads, head_dim, torch.bfloat16)
    out = flydsl_sage_attn_func(q, k, v, causal=True)
    ref = _ref_sdpa_bshd(q, k, v, causal=True)

    cos = F.cosine_similarity(
        out.float().reshape(-1, head_dim),
        ref.float().reshape(-1, head_dim),
        dim=1,
    )
    assert cos.min().item() > _COS_MIN, f"causal min_cos={cos.min().item():.4f}"
    assert cos.mean().item() > _COS_MEAN_MIN, f"causal mean_cos={cos.mean().item():.4f}"


def test_flydsl_sage_bhsd_layout():
    """BHSD layout produces the same result as BSHD (transposed)."""
    batch, seq_len, num_q_heads, head_dim = 1, 4096, 8, 128
    q_bshd, k_bshd, v_bshd = _make_qkv(batch, seq_len, num_q_heads, num_q_heads, head_dim, torch.bfloat16)

    out_bshd = flydsl_sage_attn_func(q_bshd, k_bshd, v_bshd, layout="bshd")

    # Run with BHSD inputs
    q_bhsd = q_bshd.permute(0, 2, 1, 3).contiguous()
    k_bhsd = k_bshd.permute(0, 2, 1, 3).contiguous()
    v_bhsd = v_bshd.permute(0, 2, 1, 3).contiguous()
    out_bhsd = flydsl_sage_attn_func(q_bhsd, k_bhsd, v_bhsd, layout="bhsd")

    # out_bhsd should be BHSD; convert back to BSHD for comparison
    out_bhsd_as_bshd = out_bhsd.permute(0, 2, 1, 3).contiguous()

    cos = F.cosine_similarity(
        out_bshd.float().reshape(-1, head_dim),
        out_bhsd_as_bshd.float().reshape(-1, head_dim),
        dim=1,
    )
    assert cos.min().item() > 0.999, f"layout mismatch min_cos={cos.min().item():.6f}"


def test_flydsl_sage_rejects_non_cdna():
    """Reject non-CDNA GPUs."""
    import unittest.mock as mock

    mock_props = mock.MagicMock()
    mock_props.gcnArchName = "gfx1201"
    with mock.patch("torch.cuda.get_device_properties", return_value=mock_props):
        q = torch.randn(1, 256, 8, 128, dtype=torch.bfloat16, device="cuda")
        with pytest.raises(ValueError, match="gfx942"):
            flydsl_sage_attn_func(q, q.clone(), q.clone())


def test_flydsl_sage_rejects_gqa_invalid():
    """num_q_heads not divisible by num_kv_heads must raise."""
    q = torch.randn(1, 256, 7, 128, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(1, 256, 3, 128, dtype=torch.bfloat16, device="cuda")
    v = torch.randn(1, 256, 3, 128, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(ValueError, match="divisible"):
        flydsl_sage_attn_func(q, k, v)


def test_flydsl_sage_rejects_bad_head_dim():
    """head_dim < 64 must raise."""
    q = torch.randn(1, 256, 8, 32, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(ValueError, match="head_dim"):
        flydsl_sage_attn_func(q, q.clone(), q.clone())


def test_flydsl_sage_rejects_dtype_mismatch():
    """Mismatched q/k/v dtypes must raise."""
    q = torch.randn(1, 256, 8, 128, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(1, 256, 8, 128, dtype=torch.float16, device="cuda")
    v = torch.randn(1, 256, 8, 128, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(ValueError, match="dtype"):
        flydsl_sage_attn_func(q, k, v)


def test_flydsl_sage_padded_seq_len():
    """seq_len not aligned to block_m (256) is padded transparently."""
    batch, seq_len, num_q_heads, head_dim = 1, 3000, 8, 128  # 3000 % 256 != 0
    q, k, v = _make_qkv(batch, seq_len, num_q_heads, num_q_heads, head_dim, torch.bfloat16)
    out = flydsl_sage_attn_func(q, k, v, causal=False)
    ref = _ref_sdpa_bshd(q, k, v, causal=False)

    assert out.shape == (batch, seq_len, num_q_heads, head_dim), f"shape: {out.shape}"
    cos = F.cosine_similarity(
        out.float().reshape(-1, head_dim),
        ref.float().reshape(-1, head_dim),
        dim=1,
    )
    assert cos.min().item() > _COS_MIN, f"padded seq min_cos={cos.min().item():.4f}"


@pytest.mark.skipif(not HAS_TRITON_SAGE, reason="fav3_sage_wrapper_func not available")
@pytest.mark.parametrize(
    "batch,seq_q,seq_k,num_q_heads,num_kv_heads,head_dim,causal",
    [
        (1,  4096,  4096,  8,  8, 128, False),
        (1,  4096,  4096,  8,  8, 128, True),
        (1, 16384, 16384, 24, 24, 128, False),
        (2,  4096,  4096, 16,  4, 128, False),
        (1,  3000,  3000,  8,  8, 128, False),
    ],
)
def test_flydsl_sage_vs_triton(batch, seq_q, seq_k, num_q_heads, num_kv_heads, head_dim, causal):
    """FlyDSL output matches Triton fav3_sage on the same quantized inputs."""
    q, k, v = _make_qkv(batch, seq_q, num_q_heads, num_kv_heads, head_dim, torch.bfloat16)
    if seq_k != seq_q:
        _, k, v = _make_qkv(batch, seq_k, num_q_heads, num_kv_heads, head_dim, torch.bfloat16, seed=99)

    flydsl_out = flydsl_sage_attn_func(q, k, v, causal=causal)
    triton_out = fav3_sage_wrapper_func(q, k, v, causal=causal)
    torch.cuda.synchronize()

    cos = F.cosine_similarity(
        flydsl_out.float().reshape(-1, head_dim),
        triton_out.float().reshape(-1, head_dim),
        dim=1,
    )
    assert cos.min().item() > _COS_MIN, (
        f"FlyDSL vs Triton min_cos={cos.min().item():.4f} < {_COS_MIN} "
        f"(B={batch} Sq={seq_q} Sk={seq_k} Hq={num_q_heads} Hkv={num_kv_heads} causal={causal})"
    )
    assert cos.mean().item() > _COS_MEAN_MIN, (
        f"FlyDSL vs Triton mean_cos={cos.mean().item():.4f} < {_COS_MEAN_MIN}"
    )
