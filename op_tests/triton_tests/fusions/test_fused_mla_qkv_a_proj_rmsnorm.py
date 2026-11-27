# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Tests for the fused MLA qkv_a projection + q/kv RMSNorm (FP8 block-scaled weight).

The reference is pure torch in fp64: the same dynamic per-1x128 FP8 e4m3 activation quant,
the dequantized product, a BF16 round of the projection (the kernel commits to BF16 there,
as an unfused GEMM output would) and the two RMSNorms in fp32.

Three input draws: a benign draw, a cancellation draw (``hidden = [x, -x]`` against a weight
whose two K halves differ by one FP8 code in a quarter of the entries, so the split-K
partials are large and nearly cancel) and a wide draw (per-row magnitudes over 2^-12..2^12
and norm weights over 2^-4..2^4). The cancellation draw fails when the split-K partials are
stored in BF16.
"""

import pytest
import torch

from aiter.ops.shuffle import shuffle_weight
from aiter.ops.triton.fusions.fused_mla_qkv_a_proj_rmsnorm import (
    fused_mla_qkv_a_proj_rmsnorm,
    fused_mla_qkv_a_proj_rmsnorm_supported,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")

# GLM-5.2: q_lora_rank 2048, kv_lora_rank 512, qk_rope_head_dim 64, hidden 6144.
_Q, _KV, _ROPE, _K = 2048, 512, 64, 6144
_N = _Q + _KV + _ROPE
_EPS = 1e-5
_FP8 = torch.float8_e4m3fn
_FP8_MAX = 448.0
_GROUP = 128
_BATCHES = (1, 2, 4, 8, 16, 32)
_DRAWS = ("benign", "cancel", "wide")


def _quant_1x128(x):
    """Dynamic per-(row, 128) FP8 e4m3 quant in fp32; returns the dequantized fp64 value."""
    m, k = x.shape
    groups = x.float().view(m, k // _GROUP, _GROUP)
    scale = groups.abs().amax(-1, keepdim=True).clamp_min(1e-10) / _FP8_MAX
    codes = (groups / scale).clamp(-_FP8_MAX, _FP8_MAX).to(_FP8)
    return (codes.double() * scale.double()).view(m, k)


def ref_qkv_a_proj_rmsnorm(hidden, weight, weight_scale, q_norm_weight, kv_norm_weight):
    """fp64 reference on the row-major (unshuffled) weight."""
    n = weight.shape[0]
    w_scale = weight_scale.double().repeat_interleave(_GROUP, 0)[:n]
    w_scale = w_scale.repeat_interleave(_GROUP, 1)
    projected = _quant_1x128(hidden) @ (weight.double() * w_scale).t()
    projected = projected.to(torch.bfloat16).float()

    def rmsnorm(x, gamma):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + _EPS) * gamma.float()

    q, kv, rope = projected.split([_Q, _KV, _ROPE], dim=-1)
    return rmsnorm(q, q_norm_weight), rmsnorm(kv, kv_norm_weight), rope


def generate_inputs(m, draw, seed=0):
    gen = torch.Generator(device="cuda").manual_seed(seed * 100_003 + m)

    def randn(*shape):
        return torch.randn(*shape, generator=gen, device="cuda", dtype=torch.float32)

    def rand(*shape):
        return torch.rand(*shape, generator=gen, device="cuda", dtype=torch.float32)

    n_blocks, k_blocks = -(-_N // _GROUP), _K // _GROUP
    q_norm_weight = (1.0 + 0.25 * randn(_Q)).to(torch.bfloat16)
    kv_norm_weight = (1.0 + 0.25 * randn(_KV)).to(torch.bfloat16)
    if draw == "benign":
        hidden = randn(m, _K).to(torch.bfloat16)
        weight = (0.25 * randn(_N, _K)).to(_FP8)
        weight_scale = (0.5 + rand(n_blocks, k_blocks)) * 2.0**-6
    else:
        half = _K // 2
        x_half = randn(m, half)
        if draw == "wide":
            exponents = torch.randint(-12, 13, (m, 1), generator=gen, device="cuda")
            x_half = x_half * torch.exp2(exponents.float())

            def wide(width):
                magnitude = torch.exp2(rand(width) * 8 - 4)
                sign = torch.where(rand(width) < 0.5, -1.0, 1.0)
                return (magnitude * sign).to(torch.bfloat16)

            q_norm_weight, kv_norm_weight = wide(_Q), wide(_KV)
        x_half = x_half.to(torch.bfloat16)
        hidden = torch.cat([x_half, -x_half], 1).contiguous()
        w_half = (0.25 * randn(_N, half)).to(_FP8)
        bits = w_half.view(torch.uint8)
        nudge = (rand(_N, half) < 0.25) & ((bits & 0x7F) < 0x70)
        w_other = torch.where(nudge, bits ^ 1, bits).view(_FP8)
        weight = torch.cat([w_half, w_other], 1).contiguous()
        scale_half = (0.5 + rand(n_blocks, k_blocks // 2)) * 2.0**-6
        weight_scale = torch.cat([scale_half, scale_half], 1).contiguous()
    return hidden, weight, weight_scale.contiguous(), q_norm_weight, kv_norm_weight


def _require_supported(*args):
    ok, reason = fused_mla_qkv_a_proj_rmsnorm_supported(*args, rope_dim=_ROPE)
    if not ok:
        pytest.skip(f"unsupported here: {reason}")


@pytest.mark.parametrize("draw", _DRAWS)
@pytest.mark.parametrize("m", _BATCHES, ids=lambda m: f"M{m}")
def test_matches_reference(m, draw):
    hidden, weight, weight_scale, q_norm_weight, kv_norm_weight = generate_inputs(
        m, draw
    )
    shuffled = shuffle_weight(weight, layout=(16, 16))
    _require_supported(hidden, shuffled, weight_scale, q_norm_weight, kv_norm_weight)

    q, kv, rope = fused_mla_qkv_a_proj_rmsnorm(
        hidden,
        shuffled,
        weight_scale,
        q_norm_weight,
        kv_norm_weight,
        rope_dim=_ROPE,
        eps=_EPS,
    )
    ref_q, ref_kv, ref_rope = ref_qkv_a_proj_rmsnorm(
        hidden, weight, weight_scale, q_norm_weight, kv_norm_weight
    )
    for name, out, ref in (
        ("q", q, ref_q),
        ("kv", kv, ref_kv),
        ("k_pe", rope, ref_rope),
    ):
        assert out.shape == ref.shape and out.dtype is torch.bfloat16, name
        assert torch.isfinite(out).all(), f"{name} has non-finite values"
        torch.testing.assert_close(out.float(), ref, atol=2**-5, rtol=1e-2, msg=name)


@pytest.mark.parametrize("m", (1, 8, 32), ids=lambda m: f"M{m}")
def test_matches_aiter_chain(m):
    """Within the same tolerance of the AITER chain it replaces (quant + GEMM + fused norm)."""
    from aiter import dtypes
    from aiter.ops.enum import QuantType
    from aiter.ops.fused_qk_rmsnorm_group_quant import fused_qk_rmsnorm
    from aiter.ops.gemm_op_a8w8 import gemm_a8w8_blockscale_bpreshuffle
    from aiter.ops.quant import get_hip_quant

    hidden, weight, weight_scale, q_norm_weight, kv_norm_weight = generate_inputs(
        m, "cancel"
    )
    shuffled = shuffle_weight(weight, layout=(16, 16))
    _require_supported(hidden, shuffled, weight_scale, q_norm_weight, kv_norm_weight)

    q, kv, rope = fused_mla_qkv_a_proj_rmsnorm(
        hidden,
        shuffled,
        weight_scale,
        q_norm_weight,
        kv_norm_weight,
        rope_dim=_ROPE,
        eps=_EPS,
    )

    x_fp8, x_scale = get_hip_quant(QuantType.per_1x128)(hidden, quant_dtype=dtypes.fp8)
    x_scale = (
        x_scale.t().contiguous().t()
    )  # column-major, as the bpreshuffle GEMM reads it
    projected = gemm_a8w8_blockscale_bpreshuffle(
        x_fp8, shuffled, x_scale, weight_scale, dtypes.bf16
    )
    chain_q = torch.empty((m, _Q), dtype=torch.bfloat16, device="cuda")
    chain_kv = torch.empty((m, _KV), dtype=torch.bfloat16, device="cuda")
    fused_qk_rmsnorm(
        q_out_quantized=chain_q,
        q=projected[:, :_Q],
        q_weight=q_norm_weight,
        q_epsilon=_EPS,
        k_out=chain_kv,
        k=projected[:, _Q : _Q + _KV],
        k_weight=kv_norm_weight,
        k_epsilon=_EPS,
        quant_type=QuantType.No,
    )
    torch.testing.assert_close(q, chain_q, atol=2**-5, rtol=1e-2)
    torch.testing.assert_close(kv, chain_kv, atol=2**-5, rtol=1e-2)
    torch.testing.assert_close(rope, projected[:, _Q + _KV :], atol=2**-5, rtol=1e-2)


def test_determinism():
    hidden, weight, weight_scale, q_norm_weight, kv_norm_weight = generate_inputs(
        8, "wide"
    )
    shuffled = shuffle_weight(weight, layout=(16, 16))
    _require_supported(hidden, shuffled, weight_scale, q_norm_weight, kv_norm_weight)
    first = fused_mla_qkv_a_proj_rmsnorm(
        hidden, shuffled, weight_scale, q_norm_weight, kv_norm_weight, rope_dim=_ROPE
    )
    second = fused_mla_qkv_a_proj_rmsnorm(
        hidden, shuffled, weight_scale, q_norm_weight, kv_norm_weight, rope_dim=_ROPE
    )
    for a, b in zip(first, second):
        assert torch.equal(a, b)


def test_unsupported_calls_report_reason():
    hidden, weight, weight_scale, q_norm_weight, kv_norm_weight = generate_inputs(
        3, "benign"
    )
    shuffled = shuffle_weight(weight, layout=(16, 16))
    ok, reason = fused_mla_qkv_a_proj_rmsnorm_supported(
        hidden, shuffled, weight_scale, q_norm_weight, kv_norm_weight, rope_dim=_ROPE
    )
    assert not ok and "M=3" in reason

    bad_scale = weight_scale.t().contiguous()
    ok, reason = fused_mla_qkv_a_proj_rmsnorm_supported(
        hidden[:1], shuffled, bad_scale, q_norm_weight, kv_norm_weight, rope_dim=_ROPE
    )
    assert not ok and "weight_scale" in reason

    ok, reason = fused_mla_qkv_a_proj_rmsnorm_supported(
        hidden[:1].to(device="cpu"),
        shuffled,
        weight_scale,
        q_norm_weight,
        kv_norm_weight,
    )
    assert not ok and "CUDA" in reason

    with pytest.raises(ValueError):
        fused_mla_qkv_a_proj_rmsnorm(
            hidden,
            shuffled,
            weight_scale,
            q_norm_weight,
            kv_norm_weight,
            rope_dim=_ROPE,
        )
