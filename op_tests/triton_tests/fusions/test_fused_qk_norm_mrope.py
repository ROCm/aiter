# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch

from aiter.ops.triton.fusions.fused_qk_norm_mrope import (
    fused_qk_norm_mrope,
    fused_qk_norm_mrope_cache,
)
from aiter.ops.triton.utils._triton import arch_info

pytestmark = pytest.mark.skipif(
    arch_info.get_arch() != "gfx1250", reason="gfx1250 Gluon kernel"
)

HEAD_DIM = 256
ROPE_HALF = 32
MAX_POS = 4096
SECTION = [11, 11, 10]
BLOCK = 16
X = 8  # bf16 values in 16 bytes


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


def make_cache_inputs(num_tokens, q_heads, k_heads):
    inputs = make_inputs(num_tokens, q_heads, k_heads)
    num_blocks = num_tokens // BLOCK + 4
    # A V with strided rows, and a different slot for each token. Some tokens are pad tokens.
    inputs["v"] = torch.randn(
        (num_tokens, 2 * k_heads * HEAD_DIM), dtype=torch.bfloat16, device="cuda"
    )[:, : k_heads * HEAD_DIM]
    # The last tokens have no slot (graph padding).
    slots = torch.randperm(num_blocks * BLOCK, device="cuda")[
        : num_tokens - num_tokens // 10
    ]
    slots[2::7] = -1
    inputs["slot_mapping"] = slots
    inputs["k_cache"] = torch.zeros(
        (num_blocks, k_heads, HEAD_DIM // X, BLOCK, X),
        dtype=torch.bfloat16,
        device="cuda",
    )
    inputs["v_cache"] = torch.zeros(
        (num_blocks, k_heads, BLOCK // X, HEAD_DIM, X),
        dtype=torch.bfloat16,
        device="cuda",
    )
    return inputs


@pytest.mark.parametrize("num_tokens", [1, 37, 300])
@pytest.mark.parametrize("q_heads, k_heads", [(4, 1), (16, 2)])
def test_fused_qk_norm_mrope_cache(num_tokens, q_heads, k_heads):
    inputs = make_cache_inputs(num_tokens, q_heads, k_heads)
    q_out, k_out = fused_qk_norm_mrope_cache(**inputs)
    ref_q = torch_ref(inputs["q"], inputs["q_weight"], q_heads, inputs)
    ref_k = torch_ref(inputs["k"], inputs["k_weight"], k_heads, inputs)
    torch.testing.assert_close(q_out.float(), ref_q, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(k_out.float(), ref_k, atol=1e-2, rtol=1e-2)

    # The tokens with a slot fill their cache rows. All other rows stay zero.
    slots = inputs["slot_mapping"]
    written = torch.zeros(num_tokens, dtype=torch.bool, device="cuda")
    written[: len(slots)] = slots >= 0
    blk, off = (
        slots[written[: len(slots)]] // BLOCK,
        slots[written[: len(slots)]] % BLOCK,
    )
    ref_k_cache = torch.zeros_like(inputs["k_cache"], dtype=torch.float32)
    ref_k_cache[blk, :, :, off, :] = ref_k[written].unflatten(
        -1, (k_heads, HEAD_DIM // X, X)
    )
    ref_v_cache = torch.zeros_like(inputs["v_cache"])
    ref_v_cache[blk, :, off // X, :, off % X] = inputs["v"][written].unflatten(
        -1, (k_heads, HEAD_DIM)
    )
    torch.testing.assert_close(
        inputs["k_cache"].float(), ref_k_cache, atol=1e-2, rtol=1e-2
    )
    torch.testing.assert_close(inputs["v_cache"], ref_v_cache, atol=0, rtol=0)


@pytest.mark.parametrize("num_tokens", [1, 37, 300])
@pytest.mark.parametrize("kv_scale", [1.0, 0.5])
def test_fused_qk_norm_mrope_cache_fp8(num_tokens, kv_scale):
    """An fp8 cache: K and V stored as value / kv_scale clamped to the fp8 range, Q
    returned in fp8 (clamped, unscaled); each value rounded to bf16 first."""
    import aiter

    fp8 = aiter.dtypes.fp8
    fp8_max = torch.finfo(fp8).max
    q_heads, k_heads, block, x = 16, 2, 64, 16  # 16 fp8 values in 16 bytes
    inputs = make_inputs(num_tokens, q_heads, k_heads)
    num_blocks = num_tokens // block + 4
    v = torch.randn(
        (num_tokens, 2 * k_heads * HEAD_DIM), dtype=torch.bfloat16, device="cuda"
    )[:, : k_heads * HEAD_DIM]
    v[:, :8] = 3000.0  # past the fp8 range: stored clamped
    inputs["v"] = v
    slots = torch.randperm(num_blocks * block, device="cuda")[:num_tokens]
    slots[2::7] = -1
    inputs["slot_mapping"] = slots
    inputs["k_cache"] = torch.zeros(
        (num_blocks, k_heads, HEAD_DIM // x, block, x), dtype=torch.uint8, device="cuda"
    ).view(fp8)
    inputs["v_cache"] = torch.zeros(
        (num_blocks, k_heads, block // x, HEAD_DIM, x), dtype=torch.uint8, device="cuda"
    ).view(fp8)
    q_out, k_out = fused_qk_norm_mrope_cache(**inputs, kv_scale=kv_scale)
    ref_q = torch_ref(inputs["q"], inputs["q_weight"], q_heads, inputs).bfloat16()
    ref_k = torch_ref(inputs["k"], inputs["k_weight"], k_heads, inputs).bfloat16()
    assert q_out.dtype == fp8 and k_out.dtype == torch.bfloat16
    to_fp8 = lambda t: t.float().clamp(-fp8_max, fp8_max).to(fp8).float()
    torch.testing.assert_close(q_out.float(), to_fp8(ref_q), atol=0.5, rtol=0.125)
    torch.testing.assert_close(k_out.float(), ref_k.float(), atol=1e-2, rtol=1e-2)

    written = slots >= 0
    blk, off = slots[written] // block, slots[written] % block
    ref_k_cache = torch.zeros(
        inputs["k_cache"].shape, dtype=torch.float32, device="cuda"
    )
    ref_k_cache[blk, :, :, off, :] = to_fp8(ref_k.float() / kv_scale)[written].unflatten(
        -1, (k_heads, HEAD_DIM // x, x)
    )
    ref_v_cache = torch.zeros(
        inputs["v_cache"].shape, dtype=torch.float32, device="cuda"
    )
    ref_v_cache[blk, :, off // x, :, off % x] = to_fp8(v.float() / kv_scale)[
        written
    ].unflatten(-1, (k_heads, HEAD_DIM))
    torch.testing.assert_close(
        inputs["k_cache"].float(), ref_k_cache, atol=0.5, rtol=0.125
    )
    # V takes no arithmetic before the cast but the scale: exact.
    assert not torch.isnan(inputs["v_cache"].float()).any()
    torch.testing.assert_close(inputs["v_cache"].float(), ref_v_cache, atol=0, rtol=0)
