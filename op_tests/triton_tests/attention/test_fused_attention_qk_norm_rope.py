# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch

from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.ops.triton.attention.fused_attention_qk_norm_rope import (
    fused_attention_qk_norm_rope,
    fused_attention_qk_norm_rope_supported,
)

QUERY_HEADS = 4
HEAD_DIM = 256
ROTARY_DIM = 64
PACKED = (QUERY_HEADS + 1) * 512
EPS = 1.0e-6

requires_gfx950 = pytest.mark.skipif(
    not torch.cuda.is_available() or get_gfx_runtime() != "gfx950",
    reason="requires gfx950",
)

# One row count per schedule band: decode (<=12288), m128_m256 (<=24576),
# prefill (>24576). The crossovers are covered at their edges.
ROWS = [1, 64, 1024, 12288, 14336, 16384, 24576, 28672, 32768]


def _make_inputs(rows: int):
    device = torch.device("cuda")
    gen = torch.Generator(device=device).manual_seed(rows)
    packed = (
        torch.randn(rows, PACKED, generator=gen, device=device, dtype=torch.float32)
        * 0.1
    ).to(torch.bfloat16)
    q_norm = (
        torch.randn(HEAD_DIM, generator=gen, device=device, dtype=torch.float32) * 0.1
    )
    k_norm = (
        torch.randn(HEAD_DIM, generator=gen, device=device, dtype=torch.float32) * 0.1
    )
    positions = torch.arange(rows, device=device, dtype=torch.int64)
    cos_sin_cache = (
        torch.randn(
            rows + 8, ROTARY_DIM, generator=gen, device=device, dtype=torch.float32
        )
        * 0.5
    )
    return packed, q_norm, k_norm, positions, cos_sin_cache


def _reference(packed, q_norm, k_norm, positions, cos_sin_cache):
    rows = packed.shape[0]
    device = packed.device
    half = ROTARY_DIM // 2
    trig = cos_sin_cache[positions]
    cos = trig[:, :half]
    sin = trig[:, half:ROTARY_DIM]

    def normalize(values, weight):
        inv_rms = torch.rsqrt(values.float().square().mean(-1, keepdim=True) + EPS)
        return (values.float() * inv_rms * (1.0 + weight.float())).to(torch.bfloat16)

    def rope(values):
        out = values.float()
        lo = out[:, :half]
        hi = out[:, half:ROTARY_DIM]
        rotated = out.clone()
        rotated[:, :half] = lo * cos - hi * sin
        rotated[:, half:ROTARY_DIM] = hi * cos + lo * sin
        return rotated.to(torch.bfloat16)

    query = torch.empty(
        rows, QUERY_HEADS, HEAD_DIM, device=device, dtype=torch.bfloat16
    )
    gate = torch.empty(
        rows, QUERY_HEADS * HEAD_DIM, device=device, dtype=torch.bfloat16
    )
    for head in range(QUERY_HEADS):
        slab = packed[:, head * 512 : head * 512 + 256]
        query[:, head] = rope(normalize(slab, q_norm))
        gate[:, head * 256 : (head + 1) * 256] = packed[
            :, head * 512 + 256 : head * 512 + 512
        ]
    key = rope(
        normalize(packed[:, QUERY_HEADS * 512 : QUERY_HEADS * 512 + 256], k_norm)
    )
    key = key.view(rows, 1, HEAD_DIM)
    value = packed[:, QUERY_HEADS * 512 + 256 :].view(rows, 1, HEAD_DIM)
    return query, gate, key, value


@pytest.mark.parametrize("rows", ROWS)
@requires_gfx950
def test_matches_reference(rows):
    inputs = _make_inputs(rows)
    ok, reason = fused_attention_qk_norm_rope_supported(*inputs)
    if not ok:
        # The mid/large bands need the Gluon dialect (Triton 3.8); without it
        # the call is unsupported and the native path runs instead.
        pytest.skip(reason)
    query, gate, key, value = fused_attention_qk_norm_rope(*inputs)
    ref_query, ref_gate, ref_key, ref_value = _reference(*inputs)
    # Q/K round to BF16 before and after RoPE; allow a couple of BF16 ULPs.
    torch.testing.assert_close(query, ref_query, atol=0.0625, rtol=2.0e-2)
    torch.testing.assert_close(key, ref_key, atol=0.0625, rtol=2.0e-2)
    # Gate and value are passthrough views, so they must be exact.
    torch.testing.assert_close(gate, ref_gate.view(gate.shape), atol=0.0, rtol=0.0)
    torch.testing.assert_close(value, ref_value, atol=0.0, rtol=0.0)


@requires_gfx950
def test_supported_rejects_bad_calls():
    packed, q_norm, k_norm, positions, cos_sin_cache = _make_inputs(64)
    # Wrong packed width.
    ok, _ = fused_attention_qk_norm_rope_supported(
        packed[:, :-512], q_norm, k_norm, positions, cos_sin_cache
    )
    assert not ok
    # Wrong dtype.
    ok, _ = fused_attention_qk_norm_rope_supported(
        packed.float(), q_norm, k_norm, positions, cos_sin_cache
    )
    assert not ok
    # positions too short.
    ok, _ = fused_attention_qk_norm_rope_supported(
        packed, q_norm, k_norm, positions[:10], cos_sin_cache
    )
    assert not ok
