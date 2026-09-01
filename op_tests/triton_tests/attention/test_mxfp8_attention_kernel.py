# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Correctness tests for the MXFP8 Flash Attention v2 kernel.

Tests the forward path (attn_fwd_mxfp8) via mxfp8_attention_forward with
use_mxfp8=False (non-quantized path), comparing against a PyTorch reference.
Only runs on FP8-capable architectures (gfx942, gfx950).
"""

import math

import pytest
import torch

from aiter.ops.triton.attention.mxfp8_attention import mxfp8_attention_forward
from aiter.ops.triton.utils._triton.arch_info import get_arch

pytestmark = pytest.mark.skipif(
    get_arch() != "gfx950",
    reason=f"MXFP8 Flash Attention requires gfx950 (CDNA4), got {get_arch()}",
)


def _ref_attention(q, k, v, causal=False, sm_scale=None):
    """Pure-PyTorch scaled-dot-product attention reference."""
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(q.shape[-1])
    # q/k/v: [B, H, S, D] (bhsd layout)
    scores = torch.einsum("bhmd,bhnd->bhmn", q.float(), k.float()) * sm_scale
    if causal:
        S_q, S_k = q.shape[2], k.shape[2]
        mask = torch.tril(
            torch.ones(S_q, S_k, device=q.device, dtype=torch.bool),
            diagonal=S_k - S_q,
        )
        scores = scores.masked_fill(~mask, float("-inf"))
    p = torch.softmax(scores, dim=-1)
    return torch.einsum("bhmn,bhnd->bhmd", p, v.float()).to(q.dtype)


def _dummy_scales(B, H, S, D, quant_block_size, dtype, device):
    """Create dummy e8m0 scale tensors (all 127 = scale 1.0) for use_mxfp8=False."""
    scale_blocks = (D + quant_block_size - 1) // quant_block_size
    q_scale = torch.full((B, H, S, scale_blocks), 127, dtype=torch.uint8, device=device)
    return q_scale


@pytest.mark.parametrize(
    "B, H, S, D",
    [
        (1, 4, 64, 64),
        (2, 8, 128, 128),
        (1, 4, 64, 128),
    ],
)
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_mxfp8_attn_fwd_nofp8(B, H, S, D, causal, dtype):
    """mxfp8_attention_forward with use_mxfp8=False matches PyTorch reference."""
    torch.manual_seed(0)
    sm_scale = 1.0 / math.sqrt(D)
    quant_block_size = 32

    q = torch.randn(B, S, H, D, dtype=dtype, device="cuda") * 0.1
    k = torch.randn(B, S, H, D, dtype=dtype, device="cuda") * 0.1
    v = torch.randn(B, S, H, D, dtype=dtype, device="cuda") * 0.1

    q_scale = _dummy_scales(B, H, S, D, quant_block_size, dtype, "cuda")
    k_scale = _dummy_scales(B, H, S, D, quant_block_size, dtype, "cuda")
    v_scale = _dummy_scales(B, H, S, D, quant_block_size, dtype, "cuda")

    out, _lse, _ = mxfp8_attention_forward(
        q,
        k,
        v,
        q_scale=q_scale,
        k_scale=k_scale,
        v_scale=v_scale,
        sm_scale=sm_scale,
        causal=causal,
        use_mxfp8=False,
        block_m=64,
        block_n=64,
        quant_block_size=quant_block_size,
        layout="bshd",
    )

    # Convert to bhsd for reference
    q_ref = q.permute(0, 2, 1, 3)
    k_ref = k.permute(0, 2, 1, 3)
    v_ref = v.permute(0, 2, 1, 3)
    ref = _ref_attention(q_ref, k_ref, v_ref, causal=causal, sm_scale=sm_scale)
    ref = ref.permute(0, 2, 1, 3)  # back to bshd

    torch.testing.assert_close(
        out.float(),
        ref.float(),
        atol=1e-2,
        rtol=1e-2,
        msg=f"Mismatch at (B={B},H={H},S={S},D={D},causal={causal})",
    )
