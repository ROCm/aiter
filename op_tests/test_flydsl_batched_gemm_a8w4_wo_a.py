#!/usr/bin/env python3

# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Step 0 correctness gate for routing deepseek-v4 wo_a (grouped output LoRA)
through the flydsl strided-batched a8w4 kernel
(``aiter/ops/flydsl/kernels/mxfp4_preshuffle_batched_gemm_gfx1250_tdm.py``).

The wo_a batched GEMM in ATOM is currently BF16 x BF16:

    o  : [M(tokens), B(groups), K]  bf16
    wo_a: [B, N(o_lora_rank), K]    bf16
    y  : [M, B, N] = einsum("mbk,bnk->mbn")

This test pins down the a8w4 operand layout (mbn) by:
  1. quantizing o -> MXFP8 (e4m3 + e8m0) via ``quant_act_mxfp8_mbn``,
  2. quantizing/preshuffling wo_a -> MXFP4 + n32k4 scale via
     ``preshuffle_a8w4_weight_mbn``,
  3. running ``flydsl_batched_gemm_a8w4_v2`` (layout='mbn'),
and comparing against the BF16 einsum reference.

Because a8w4 is lossy vs BF16, the gate is cosine-sim + mean-rel-error, not
bit-exact. Run on a gfx1250 box:

    pytest -q op_tests/test_flydsl_batched_gemm_a8w4_wo_a.py
    # or a single shape with verbose stats:
    python op_tests/test_flydsl_batched_gemm_a8w4_wo_a.py
"""

from __future__ import annotations

import pytest
import torch

from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl.batched_gemm_mxfp4 import (
    flydsl_batched_gemm_a8w4_v2,
    preshuffle_a8w4_weight_mbn,
    quant_act_mxfp8_mbn,
)

torch.set_default_device("cuda")

pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]

_IS_GFX1250 = get_gfx() == "gfx1250"
_skip_gfx1250 = pytest.mark.skipif(
    not _IS_GFX1250, reason="flydsl batched a8w4 kernel is gfx1250-only"
)


def _ref_bf16(o_mbn: torch.Tensor, w_bnk: torch.Tensor) -> torch.Tensor:
    """BF16 reference: y[m,b,n] = sum_k o[m,b,k] * w[b,n,k]."""
    return torch.einsum("mbk,bnk->mbn", o_mbn.float(), w_bnk.float())


def _stats(ref: torch.Tensor, got: torch.Tensor):
    ref = ref.float().flatten()
    got = got.float().flatten()
    cos = torch.nn.functional.cosine_similarity(ref, got, dim=0).item()
    denom = ref.abs().mean().clamp_min(1e-6)
    mre = (ref - got).abs().mean().item() / denom.item()
    max_ae = (ref - got).abs().max().item()
    return cos, mre, max_ae


def _run_a8w4_mbn(o_mbn: torch.Tensor, w_bnk: torch.Tensor, dtype=torch.bfloat16):
    """Full quant + preshuffle + launch, mirroring the intended _attn_post path."""
    B, N, K = w_bnk.shape
    a_fp8, a_scales = quant_act_mxfp8_mbn(o_mbn)          # [M,B,K], [M//32,B,(K//32)*32]
    w_codes, w_scales = preshuffle_a8w4_weight_mbn(w_bnk)  # [B,N//16,(K//2)*16], [B,N//32,(K//32)*32]
    y = flydsl_batched_gemm_a8w4_v2(
        a_fp8, w_codes, a_scales, w_scales, N=N, dtype=dtype, layout="mbn"
    )  # [B, M, N] view
    return y.transpose(0, 1).contiguous()  # -> [M, B, N] to match reference


# (M, B, N, K): decode-ish, mid, prefill-ish. N=o_lora_rank=1024, K per-group=4096
# match the dsv4-pro wo_a shape. B = n_local_groups (use small B for test speed).
_SHAPES = [
    (32, 2, 1024, 4096),
    (128, 2, 1024, 4096),
    (1024, 2, 1024, 4096),
    (4096, 4, 1024, 4096),
    (17, 2, 1024, 4096),  # M not a multiple of 32 -> exercises super padding/OOB
]


def _ref_quant_n32k4_mbn(o_mbn: torch.Tensor):
    """Reference (pre-fusion) path: dynamic_mxfp8_quant + torch n32k4 reshuffle.

    Kept here as the bit-exact oracle for the fused
    ``dynamic_mxfp8_quant_n32k4_mbn`` kernel (方案 B).
    """
    from aiter.ops.triton.quant import dynamic_mxfp8_quant
    from aiter.ops.shuffle import shuffle_scale_n32k4

    M, B, K = o_mbn.shape
    a_fp8, a_scale = dynamic_mxfp8_quant(o_mbn.reshape(M * B, K))
    a_fp8 = a_fp8.view(M, B, K).contiguous()
    a_scale = a_scale.view(M, B, K // 32)
    M32 = ((M + 31) // 32) * 32
    if M32 != M:
        pad = torch.zeros(
            (M32 - M, B, K // 32), dtype=a_scale.dtype, device=a_scale.device
        )
        a_scale = torch.cat([a_scale, pad], dim=0)
    a_sh = shuffle_scale_n32k4(a_scale.transpose(0, 1).contiguous(), experts_cnt=B)
    a_sh = a_sh.transpose(0, 1).contiguous()  # [M32//32, B, (K//32)*32]
    return a_fp8, a_sh


@_skip_gfx1250
@pytest.mark.parametrize("M,B,N,K", _SHAPES)
def test_fused_quant_n32k4_matches_torch(M, B, N, K):
    """方案 B: fused quant+preshuffle scale must be bit-exact vs the torch path.

    Only the padded (m >= M) super rows may differ; both are pre-zeroed so the
    comparison covers the full buffer.
    """
    from aiter.ops.triton.quant import dynamic_mxfp8_quant_n32k4_mbn

    torch.manual_seed(0)
    o = torch.randn(M, B, K, dtype=torch.bfloat16) * 0.1

    ref_fp8, ref_scale = _ref_quant_n32k4_mbn(o)
    got_fp8, got_scale = dynamic_mxfp8_quant_n32k4_mbn(o)

    assert got_fp8.shape == ref_fp8.shape
    assert got_scale.shape == ref_scale.shape, (
        f"{tuple(got_scale.shape)} vs {tuple(ref_scale.shape)}"
    )
    # fp8 payload: bit-exact (same quant math).
    assert torch.equal(got_fp8.view(torch.uint8), ref_fp8.view(torch.uint8))
    # e8m0 scale in n32k4 layout: bit-exact.
    assert torch.equal(got_scale, ref_scale)


@_skip_gfx1250
@pytest.mark.parametrize("M,B,N,K", _SHAPES)
def test_wo_a_a8w4_matches_bf16(M, B, N, K):
    torch.manual_seed(0)
    o = torch.randn(M, B, K, dtype=torch.bfloat16) * 0.1
    w = torch.randn(B, N, K, dtype=torch.bfloat16) * 0.1

    ref = _ref_bf16(o, w)
    got = _run_a8w4_mbn(o, w)

    assert got.shape == ref.shape, f"{tuple(got.shape)} vs {tuple(ref.shape)}"
    cos, mre, max_ae = _stats(ref, got)
    # a8w4 (MXFP8 x MXFP4) vs BF16: expect high cosine sim, few-percent MRE.
    assert cos > 0.99, f"cos={cos:.5f} mre={mre:.4f} max_ae={max_ae:.4f}"
    assert mre < 0.06, f"cos={cos:.5f} mre={mre:.4f} max_ae={max_ae:.4f}"


if __name__ == "__main__":
    if not _IS_GFX1250:
        raise SystemExit(f"requires gfx1250, got {get_gfx()}")
    for (M, B, N, K) in _SHAPES:
        torch.manual_seed(0)
        o = torch.randn(M, B, K, dtype=torch.bfloat16) * 0.1
        w = torch.randn(B, N, K, dtype=torch.bfloat16) * 0.1
        ref = _ref_bf16(o, w)
        got = _run_a8w4_mbn(o, w)
        cos, mre, max_ae = _stats(ref, got)
        print(
            f"M={M:>5} B={B} N={N} K={K} | cos={cos:.5f} "
            f"mre={mre:.4f} max_ae={max_ae:.4f}"
        )
