# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch

from aiter.ops.triton.fusions.fused_qk_norm_rope_group_quant import (
    fused_qk_norm_rope_group_quant,
)
from aiter.ops.triton.utils._triton import arch_info
from aiter.ops.triton.utils.types import e4m3_dtype

pytestmark = pytest.mark.skipif(
    arch_info.get_arch() != "gfx1250", reason="gfx1250 Gluon kernel"
)

NUM_HEADS = 128
HEAD_DIM = 512
ROPE_DIM = 64
NOPE_DIM = HEAD_DIM - ROPE_DIM
GROUP_SIZE = 64
MAX_POS = 4096


def make_inputs(num_tokens):
    torch.manual_seed(0)
    swa_dest_rows = torch.randperm(num_tokens, device="cuda").int()
    # Some tokens have no SWA row.
    swa_dest_rows[1::5] = -1
    batch_ids = torch.zeros(num_tokens, dtype=torch.int32, device="cuda")
    # Some tokens are pad tokens.
    batch_ids[3::5] = -1
    return {
        "q": torch.randn(num_tokens, NUM_HEADS, HEAD_DIM, device="cuda").bfloat16(),
        "kv": torch.randn(num_tokens, HEAD_DIM, device="cuda").bfloat16(),
        "k_weight": torch.rand(HEAD_DIM, device="cuda").bfloat16() + 0.5,
        "positions": torch.randint(0, MAX_POS, (num_tokens,), device="cuda"),
        "cos": torch.rand(MAX_POS, ROPE_DIM // 2, device="cuda").bfloat16(),
        "sin": torch.rand(MAX_POS, ROPE_DIM // 2, device="cuda").bfloat16(),
        "eps": 1e-6,
        "swa_nope": torch.zeros(num_tokens, HEAD_DIM, dtype=e4m3_dtype, device="cuda"),
        "swa_rope": torch.zeros(
            num_tokens, ROPE_DIM, dtype=torch.bfloat16, device="cuda"
        ),
        "swa_dest_rows": swa_dest_rows,
        "batch_ids": batch_ids,
    }


def torch_ref(x, weight, inputs):
    """RMSNorm in fp32, then RoPE on the last ROPE_DIM values."""
    x = x.float()
    inv_rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + inputs["eps"])
    normed = x * inv_rms * weight
    cos = inputs["cos"][inputs["positions"]].float()
    sin = inputs["sin"][inputs["positions"]].float()
    if x.dim() == 3:
        # Q is [T, H, HEAD_DIM]. Give cos and sin a head axis.
        cos, sin = cos[:, None], sin[:, None]
    even, odd = normed[..., NOPE_DIM::2], normed[..., NOPE_DIM + 1 :: 2]
    rope_even = even * cos - odd * sin
    rope_odd = odd * cos + even * sin
    rope = torch.stack([rope_even, rope_odd], -1)
    return normed[..., :NOPE_DIM], rope.flatten(-2)


def dequant(nope):
    """fp8 values times the e8m0 scale of their group."""
    num_groups = NOPE_DIM // GROUP_SIZE
    # The row keeps two copies of each scale byte. Read the first copy.
    scale_bytes = nope.view(torch.uint8)[..., NOPE_DIM : NOPE_DIM + 2 * num_groups : 2]
    # An e8m0 byte b is the scale 2 ** (b - 127).
    scales = torch.exp2(scale_bytes.float() - 127)
    values = nope[..., :NOPE_DIM].float().unflatten(-1, (num_groups, GROUP_SIZE))
    return (values * scales[..., None]).flatten(-2)


def ceil_scale_bytes(normed):
    """The e8m0 scale byte of each group: amax / 448, rounded up to a power of two.
    The minimum amax is 1e-8. The kernel uses _mxfp8_downcast_op with SCALING_MODE=1."""
    amax = normed.unflatten(-1, (NOPE_DIM // GROUP_SIZE, GROUP_SIZE)).abs().amax(-1)
    bits = (amax.clamp_min(1e-8) * (1 / 448)).view(torch.int32)
    # The biased exponent, plus 1 if the value is not a power of two.
    return ((bits >> 23) + ((bits & 0x7FFFFF) != 0)).to(torch.uint8)


@pytest.mark.parametrize("num_tokens", [37, 300, 600])
def test_fused_qk_norm_rope_group_quant(num_tokens):
    inputs = make_inputs(num_tokens)
    # The scale rounding modes differ only when amax / 448 is a power of two or
    # amax is 0. Random data does not have these cases. Add them to two Q rows.
    # Token 0, head 0: the RMS is 8, so in group 0 amax / 448 = 14 / 8 / 448 = 2^-8.
    row = torch.full((HEAD_DIM,), 8.0, device="cuda")
    row[0], row[1], row[2:5] = 14.0, 4.0, 6.0  # 14**2 + 4**2 + 3 * 6**2 = 5 * 8**2
    inputs["q"][0, 0] = row
    # Token 1, head 0: group 1 is all zeros.
    inputs["q"][1, 0, GROUP_SIZE : 2 * GROUP_SIZE] = 0
    q_nope, q_rope, k_nope, k_rope = fused_qk_norm_rope_group_quant(**inputs)

    # Q has no RMSNorm weight. Use 1.0.
    ref_q_nope, ref_q_rope = torch_ref(inputs["q"], 1.0, inputs)
    ref_k_nope, ref_k_rope = torch_ref(inputs["kv"], inputs["k_weight"], inputs)
    # fp8 e4m3 has 3 mantissa bits. The rounding error is at most 1/16 of the value.
    torch.testing.assert_close(dequant(q_nope), ref_q_nope, atol=1e-2, rtol=1 / 16)
    torch.testing.assert_close(dequant(k_nope), ref_k_nope, atol=1e-2, rtol=1 / 16)
    torch.testing.assert_close(q_rope.float(), ref_q_rope, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(k_rope.float(), ref_k_rope, atol=1e-2, rtol=1e-2)

    # Both copies of each scale byte must be exact.
    num_groups = NOPE_DIM // GROUP_SIZE
    for nope, ref_nope in ((q_nope, ref_q_nope), (k_nope, ref_k_nope)):
        scale_bytes = nope.view(torch.uint8)[..., NOPE_DIM : NOPE_DIM + 2 * num_groups]
        ref_bytes = ceil_scale_bytes(ref_nope).repeat_interleave(2, dim=-1)
        torch.testing.assert_close(scale_bytes, ref_bytes)

    # Tokens with a negative SWA row or batch id get no SWA write.
    written = (inputs["swa_dest_rows"] >= 0) & (inputs["batch_ids"] >= 0)
    dest_rows = inputs["swa_dest_rows"][written]
    ref_swa_nope = torch.zeros_like(ref_k_nope)
    ref_swa_nope[dest_rows] = dequant(k_nope)[written]
    ref_swa_rope = torch.zeros_like(k_rope)
    ref_swa_rope[dest_rows] = k_rope[written]
    torch.testing.assert_close(dequant(inputs["swa_nope"]), ref_swa_nope)
    torch.testing.assert_close(inputs["swa_rope"], ref_swa_rope)
