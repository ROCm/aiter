# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Tests for ``flydsl_sage_attn_mxfp4_func`` and ``fav3_sage_mxfp4_flydsl_wrapper``
(CDNA / gfx950 only).

Two reference paths:
  1. SDPA fp32                       — looser bound (cos_min ≥ 0.95). Sanity check.
  2. Triton MXFP4 with shared inputs — tight bound (cos_min ≥ 0.999). Most
                                       discriminating: both kernels see the
                                       SAME quantized inputs, so any difference
                                       isolates kernel implementation.
"""

from __future__ import annotations

from typing import Tuple

import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("flydsl")
from aiter.ops.flydsl import (  # noqa: E402
    is_flydsl_available,
    flydsl_sage_attn_mxfp4_func,
    fav3_sage_mxfp4_flydsl_wrapper,
)
from aiter.ops.triton.attention.fav3_sage_attention_mxfp4_wrapper import (  # noqa: E402
    fav3_sage_mxfp4_wrapper as _triton_mxfp4_wrapper,
    fav3_sage_mxfp4_func as _triton_mxfp4_func,
)
from aiter.ops.triton.quant.sage_attention_quant_wrappers import (  # noqa: E402
    sage_quant_mxfp4 as _triton_sage_quant_mxfp4,
)
from aiter.utility.dtypes import fp8 as _fp8_dtype  # noqa: E402

if not is_flydsl_available():
    pytest.skip("flydsl is not available", allow_module_level=True)


def _is_gfx950() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        arch = torch.cuda.get_device_properties(0).gcnArchName
    except Exception:
        return False
    return arch.lower().split(":")[0].startswith("gfx950")


pytestmark = pytest.mark.skipif(
    not _is_gfx950(),
    reason="MXFP4 sage attention requires gfx950",
)


# FP4 noisier than INT8 — looser bounds vs SDPA fp32.
_COS_MIN_SDPA = 0.95
_COS_MEAN_MIN_SDPA = 0.97

# vs Triton with shared inputs: tight (kernel-implementation-only diff).
_COS_MIN_TRITON = 0.999


def _ref_sdpa_bshd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
    softmax_scale: float | None = None,
) -> torch.Tensor:
    q32 = q.float().transpose(1, 2).contiguous()
    k32 = k.float().transpose(1, 2).contiguous()
    v32 = v.float().transpose(1, 2).contiguous()
    if k32.shape[1] != q32.shape[1]:
        rep = q32.shape[1] // k32.shape[1]
        k32 = k32.repeat_interleave(rep, dim=1)
        v32 = v32.repeat_interleave(rep, dim=1)
    scale = softmax_scale or (q.shape[-1] ** -0.5)
    out_bhsd = F.scaled_dot_product_attention(
        q32, k32, v32, is_causal=causal, scale=scale
    )
    return out_bhsd.transpose(1, 2).contiguous()


def _make_qkv(
    batch: int, seq_len: int, num_q_heads: int, num_kv_heads: int,
    head_dim: int, dtype: torch.dtype, seed: int = 42,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    g = torch.Generator(device="cuda").manual_seed(seed)
    q = torch.randn((batch, seq_len, num_q_heads, head_dim),
                    generator=g, dtype=dtype, device="cuda")
    k = torch.randn((batch, seq_len, num_kv_heads, head_dim),
                    generator=g, dtype=dtype, device="cuda")
    v = torch.randn((batch, seq_len, num_kv_heads, head_dim),
                    generator=g, dtype=dtype, device="cuda")
    return q, k, v


def _cos_stats(out: torch.Tensor, ref: torch.Tensor, head_dim: int):
    cos = F.cosine_similarity(
        out.float().reshape(-1, head_dim),
        ref.float().reshape(-1, head_dim),
        dim=1,
    )
    return cos.min().item(), cos.mean().item()


# ============================================================================
# Correctness vs SDPA fp32 (full pipeline including quant)
# ============================================================================

@pytest.mark.parametrize(
    "batch,seq_len,num_q_heads,num_kv_heads",
    [
        (1, 4096, 8, 8),
        (2, 2048, 16, 16),
        (1, 4096, 16, 4),    # GQA 4:1
        (1, 4096, 8, 1),     # MQA
    ],
)
def test_flydsl_mxfp4_vs_sdpa(batch, seq_len, num_q_heads, num_kv_heads):
    head_dim = 128
    q, k, v = _make_qkv(batch, seq_len, num_q_heads, num_kv_heads,
                        head_dim, torch.bfloat16)
    out = fav3_sage_mxfp4_flydsl_wrapper(
        q, k, v, causal=False, layout="bshd", q_smooth=False,
    )
    ref = _ref_sdpa_bshd(q, k, v, causal=False)

    assert out.shape == (batch, seq_len, num_q_heads, head_dim), (
        f"shape mismatch: {out.shape}"
    )
    assert out.dtype == torch.bfloat16
    cos_min, cos_mean = _cos_stats(out, ref, head_dim)
    assert cos_min > _COS_MIN_SDPA, (
        f"min_cos={cos_min:.4f} < {_COS_MIN_SDPA} "
        f"(B={batch} S={seq_len} Hq={num_q_heads} Hkv={num_kv_heads})"
    )
    assert cos_mean > _COS_MEAN_MIN_SDPA, (
        f"mean_cos={cos_mean:.4f} < {_COS_MEAN_MIN_SDPA}"
    )


def test_flydsl_mxfp4_causal_vs_sdpa():
    q, k, v = _make_qkv(1, 4096, 8, 8, 128, torch.bfloat16)
    out = fav3_sage_mxfp4_flydsl_wrapper(q, k, v, causal=True, layout="bshd")
    ref = _ref_sdpa_bshd(q, k, v, causal=True)
    cos_min, cos_mean = _cos_stats(out, ref, 128)
    assert cos_min > _COS_MIN_SDPA, f"causal min_cos={cos_min:.4f}"
    assert cos_mean > _COS_MEAN_MIN_SDPA, f"causal mean_cos={cos_mean:.4f}"


# ============================================================================
# Correctness vs Triton MXFP4 with SHARED quantized inputs (most discriminating)
# ============================================================================

def _shared_quant_run(q, k, v, causal, q_smoothing, layout="bshd"):
    """Quantize once via Triton; run BOTH kernels with the same inputs."""
    fp8_type = _fp8_dtype
    fp8_max = torch.finfo(fp8_type).max
    (q_q, q_d, k_q, k_d, v_q, v_d, delta_s) = _triton_sage_quant_mxfp4(
        q, k, v, fp8_type, fp8_max,
        BLKQ=256, BLKK=64, layout=layout, q_smoothing=q_smoothing,
    )
    triton_out = _triton_mxfp4_func(
        q=q_q, k=k_q, v=v_q,
        q_descale=q_d, k_descale=k_d, v_descale=v_d,
        bias=delta_s, causal=causal, layout=layout,
    )
    flydsl_out = flydsl_sage_attn_mxfp4_func(
        q=q_q, k=k_q, v=v_q,
        q_descale=q_d, k_descale=k_d, v_descale=v_d,
        bias=delta_s, causal=causal, layout=layout,
    )
    return flydsl_out, triton_out


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("q_smoothing", [False, True])
def test_flydsl_mxfp4_vs_triton_shared_inputs(causal, q_smoothing):
    q, k, v = _make_qkv(1, 4096, 8, 8, 128, torch.bfloat16)
    flydsl_out, triton_out = _shared_quant_run(q, k, v, causal, q_smoothing)
    assert flydsl_out.shape == triton_out.shape
    cos_min, cos_mean = _cos_stats(flydsl_out, triton_out, 128)
    assert cos_min > _COS_MIN_TRITON, (
        f"FlyDSL vs Triton (shared q-inputs) min_cos={cos_min:.6f} < {_COS_MIN_TRITON} "
        f"(causal={causal} q_smooth={q_smoothing})"
    )


# ============================================================================
# Layout invariance
# ============================================================================

def test_flydsl_mxfp4_bhsd_vs_bshd():
    q, k, v = _make_qkv(1, 4096, 8, 8, 128, torch.bfloat16)
    out_bshd = fav3_sage_mxfp4_flydsl_wrapper(q, k, v, causal=False, layout="bshd")

    qb = q.permute(0, 2, 1, 3).contiguous()
    kb = k.permute(0, 2, 1, 3).contiguous()
    vb = v.permute(0, 2, 1, 3).contiguous()
    out_bhsd = fav3_sage_mxfp4_flydsl_wrapper(qb, kb, vb, causal=False, layout="bhsd")
    out_bhsd_as_bshd = out_bhsd.permute(0, 2, 1, 3).contiguous()

    cos_min, _ = _cos_stats(out_bshd, out_bhsd_as_bshd, 128)
    assert cos_min > 0.999, f"BHSD vs BSHD min_cos={cos_min:.6f}"


# ============================================================================
# Padded sequence (not multiple of BLOCK_M)
# ============================================================================

def test_flydsl_mxfp4_padded_seq():
    q, k, v = _make_qkv(1, 3000, 8, 8, 128, torch.bfloat16)  # 3000 % 256 != 0
    out = fav3_sage_mxfp4_flydsl_wrapper(q, k, v, causal=False, layout="bshd")
    ref = _ref_sdpa_bshd(q, k, v, causal=False)
    cos_min, _ = _cos_stats(out, ref, 128)
    assert cos_min > _COS_MIN_SDPA, f"padded min_cos={cos_min:.4f}"


# ============================================================================
# Negative tests
# ============================================================================

def test_flydsl_mxfp4_rejects_block_lut():
    q, k, v = _make_qkv(1, 256, 4, 4, 128, torch.bfloat16)
    with pytest.raises(NotImplementedError, match="block-sparse"):
        fav3_sage_mxfp4_flydsl_wrapper(
            q, k, v, causal=False, block_lut=("dummy", "dummy", "dummy"),
        )


def test_flydsl_mxfp4_rejects_no_hadamard():
    q, k, v = _make_qkv(1, 256, 4, 4, 128, torch.bfloat16)
    with pytest.raises(NotImplementedError, match="hadamard"):
        fav3_sage_mxfp4_flydsl_wrapper(
            q, k, v, causal=False, hadamard_rotation=False,
        )


def test_flydsl_mxfp4_rejects_bad_head_dim():
    # head_dim=64 → packed bytes = 32. Skip via Triton quant first to get uint8.
    q = torch.randn(1, 256, 8, 64, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(ValueError, match="head_dim"):
        fav3_sage_mxfp4_flydsl_wrapper(q, q.clone(), q.clone(), causal=False)
