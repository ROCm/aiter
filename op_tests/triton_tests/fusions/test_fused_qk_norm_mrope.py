# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch

from aiter.ops.triton.fusions.fused_qk_norm_mrope import fused_qk_norm_mrope
from aiter.ops.triton.utils._triton import arch_info

pytestmark = pytest.mark.skipif(
    arch_info.get_arch() != "gfx1250", reason="gfx1250 Gluon kernel"
)

HEAD_DIM = 256
ROPE_HALF = 32
MAX_POS = 4096
SECTION = [11, 11, 10]


def make_inputs(num_tokens, q_heads, k_heads):
    torch.manual_seed(0)
    # Q and K are column slices of one [T, gate | q | k | v] projection output, as in Qwen3.5.
    qkv = torch.randn(
        (num_tokens, 2 * (q_heads + k_heads) * HEAD_DIM),
        dtype=torch.bfloat16,
        device="cuda",
    )
    q = qkv[:, q_heads * HEAD_DIM : 2 * q_heads * HEAD_DIM]
    k = qkv[:, 2 * q_heads * HEAD_DIM : (2 * q_heads + k_heads) * HEAD_DIM]
    pos_t = torch.randint(0, MAX_POS - 16, (num_tokens,), device="cuda")
    return {
        "q": q,
        "k": k,
        "q_weight": torch.randn(HEAD_DIM, dtype=torch.bfloat16, device="cuda") * 0.2,
        "k_weight": torch.randn(HEAD_DIM, dtype=torch.bfloat16, device="cuda") * 0.2,
        "eps": 1e-6,
        # H and W differ from T, as for an image token.
        "positions": torch.stack([pos_t, pos_t + 3, pos_t + 11]),
        "cos": torch.rand((MAX_POS, ROPE_HALF), dtype=torch.float32, device="cuda"),
        "sin": torch.rand((MAX_POS, ROPE_HALF), dtype=torch.float32, device="cuda"),
        "mrope_section": SECTION,
    }


def torch_ref(x, weight, heads, inputs):
    """Gemma RMSNorm in fp32, then Neox RoPE on values [0, 64) with interleaved MRoPE positions."""
    x = x.float().unflatten(-1, (heads, HEAD_DIM))
    normed = (
        x
        * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + inputs["eps"])
        * (weight.float() + 1)
    )
    i = torch.arange(ROPE_HALF, device="cuda")
    use_h = (i % 3 == 1) & (i < 3 * SECTION[1])
    use_w = (i % 3 == 2) & (i < 3 * SECTION[2])
    t, h, w = inputs["positions"]
    pos = torch.where(use_h, h[:, None], torch.where(use_w, w[:, None], t[:, None]))
    # Give cos and sin a head axis.
    cos, sin = inputs["cos"][pos, i][:, None], inputs["sin"][pos, i][:, None]
    x1, x2 = normed[..., :ROPE_HALF], normed[..., ROPE_HALF : 2 * ROPE_HALF]
    out = normed.clone()
    out[..., :ROPE_HALF] = x1 * cos - x2 * sin
    out[..., ROPE_HALF : 2 * ROPE_HALF] = x2 * cos + x1 * sin
    return out.flatten(-2)


@pytest.mark.parametrize("num_tokens", [1, 37, 300])
@pytest.mark.parametrize("q_heads, k_heads", [(4, 1), (16, 2)])
def test_fused_qk_norm_mrope(num_tokens, q_heads, k_heads):
    inputs = make_inputs(num_tokens, q_heads, k_heads)
    q_out, k_out = fused_qk_norm_mrope(**inputs)
    ref_q = torch_ref(inputs["q"], inputs["q_weight"], q_heads, inputs)
    ref_k = torch_ref(inputs["k"], inputs["k_weight"], k_heads, inputs)
    # One bf16 rounding of each output value.
    torch.testing.assert_close(q_out.float(), ref_q, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(k_out.float(), ref_k, atol=1e-2, rtol=1e-2)
