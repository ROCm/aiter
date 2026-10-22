# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""BF16 cache ragged-attention oracle, dispatch, stream and input checks."""

import pytest
import torch
from test_fused_norm_rope_attn_rope_cast import csr, dequantize, quantize_fp8, rotate
from test_fused_norm_rope_attn_rope_cast_paged import dense

from aiter import fused_norm_rope_attn_rope_cast_bf16

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or not torch.cuda.get_device_properties(0).gcnArchName.startswith("gfx950"),
    reason="requires gfx950",
)


@pytest.mark.parametrize("heads", [8, 16, 32, 64, 128])
@pytest.mark.parametrize("raw,q_norm", [(False, False), (True, False), (True, True)])
@pytest.mark.parametrize("singleton", [False, True])
def test_bf16_dense(q_norm, heads, raw, singleton):
    torch.manual_seed(16100 + heads)
    kv = torch.randn(1201, 512, device="cuda", dtype=torch.bfloat16) * 0.25
    rows, clean = [], []
    for n in [0, 1, 31, 32, 33, 65, 127, 1153]:
        ids = torch.randperm(len(kv), device="cuda")[:n].int()
        j = torch.arange(n, device="cuda")
        ids[(j >= 32) & (j < 64)] = -1
        if n > 2:
            ids[-2] = 2**31 - 1
            ids[-1] = len(kv) - 1
        rows.append(ids)
        clean.append(ids[(ids >= 0) & (ids < len(kv))])
    ids, ptr = csr(rows, torch.int32, 2**31 - 1)
    q = torch.randn(8, heads, 512, device="cuda", dtype=torch.bfloat16) * 0.25
    angle = torch.randn(64, 32, device="cuda")
    cos, sin = angle.cos().bfloat16(), angle.sin().bfloat16()
    pos = torch.arange(8, device="cuda", dtype=torch.int32) * 7
    sink = torch.linspace(-3, 3, heads, device="cuda")
    sink[::11] = -torch.inf
    norm_eps = 0.03125
    normalized = (
        (
            q.float()
            * torch.rsqrt(q.float().square().mean(-1, keepdim=True) + norm_eps)
        ).bfloat16()
        if q_norm
        else q
    )
    qr = rotate(normalized, cos, sin, pos).bfloat16() if raw else q
    expected, maxima, lses = dense(qr, kv, clean, cos, sin, pos, sink)
    debug = torch.empty_like(q, dtype=torch.float32)
    out = fused_norm_rope_attn_rope_cast_bf16(
        q,
        kv[:, None] if singleton else kv,
        ids,
        ptr,
        cos,
        sin,
        pos,
        attn_sink=sink,
        q_is_rotated=not raw,
        q_norm=q_norm,
        norm_eps=norm_eps,
        debug_out=debug,
    )
    torch.testing.assert_close(debug, expected, rtol=0.02, atol=0.008)
    assert float((debug - expected).norm() / expected.norm()) < 0.005
    torch.testing.assert_close(out[2], maxima, rtol=1e-4, atol=3e-5)
    torch.testing.assert_close(out[3], lses, rtol=1e-4, atol=3e-5)
    assert float((dequantize(*out[:2]) - expected).norm() / expected.norm()) < 0.035


@pytest.mark.parametrize("heads", [64, 128])
@pytest.mark.parametrize("tokens", [0, 1024, 1025, 2048, 2049])
def test_bf16_stream_replay(heads, tokens):
    kv = (
        torch.arange(1, 34, device="cuda", dtype=torch.bfloat16)[:, None]
        .expand(-1, 512)
        .contiguous()
    )
    quant_values, quant_codes = quantize_fp8(kv)
    quant_reference = quant_values.float() * torch.exp2(
        quant_codes.int().float() - 127
    ).repeat_interleave(32, -1)
    q = torch.zeros(tokens, heads, 512, device="cuda", dtype=torch.bfloat16)
    ids = torch.arange(tokens, device="cuda", dtype=torch.int32) % len(kv)
    ptr = torch.arange(tokens + 1, device="cuda", dtype=torch.int32)
    cos = torch.ones(1, 32, device="cuda", dtype=torch.bfloat16)
    sin = torch.zeros_like(cos)
    pos = torch.zeros(tokens, device="cuda", dtype=torch.int32)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        out = fused_norm_rope_attn_rope_cast_bf16(
            q, kv, ids, ptr, cos, sin, pos, q_is_rotated=False
        )
        if not tokens:
            assert all(x.numel() == 0 for x in out)
            return
        buffers = dict(zip(("out", "out_scale", "max_logits", "lse"), out))
        expected = [x.clone() for x in out]
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            fused_norm_rope_attn_rope_cast_bf16(
                q, kv, ids, ptr, cos, sin, pos, q_is_rotated=False, **buffers
            )
        for _ in range(3):
            for x in out:
                x.view(torch.uint8).fill_(255)
            graph.replay()
            for x, y in zip(out, expected):
                torch.testing.assert_close(
                    x.view(torch.uint8), y.view(torch.uint8), rtol=0, atol=0
                )
        if tokens:
            reference = quant_reference[ids.long()][:, None].expand_as(q)
            torch.testing.assert_close(dequantize(*out[:2]), reference, rtol=0, atol=0)
        assert torch.count_nonzero(out[2]) == torch.count_nonzero(out[3]) == 0
    torch.cuda.current_stream().wait_stream(stream)


@pytest.mark.parametrize(
    "bad", ["dtype", "shape", "strides", "indices", "paged", "empty_cache"]
)
def test_bf16_contract(bad):
    q = torch.zeros(2, 64, 512, device="cuda", dtype=torch.bfloat16)
    kv = torch.ones(3, 512, device="cuda", dtype=torch.bfloat16)
    ids = torch.tensor([0, 2], device="cuda", dtype=torch.int32)
    ptr = torch.arange(3, device="cuda", dtype=torch.int32)
    cos = torch.ones(1, 32, device="cuda", dtype=torch.bfloat16)
    sin = torch.zeros_like(cos)
    pos = torch.zeros(2, device="cuda", dtype=torch.int32)
    kw = {}
    if bad == "dtype":
        kv = kv.float()
    elif bad == "shape":
        kv = kv.view(3, 2, 256)
    elif bad == "strides":
        kv = torch.ones(3, 1024, device="cuda", dtype=torch.bfloat16)[:, ::2]
    elif bad == "indices":
        ids = ids.long()
    elif bad == "paged":
        kw = {
            "cache_page_size": 32,
            "cache_scales": torch.zeros(16, device="cuda", dtype=torch.uint8),
        }
    elif bad == "empty_cache":
        out = fused_norm_rope_attn_rope_cast_bf16(q, kv[:0], ids, ptr, cos, sin, pos)
        assert torch.count_nonzero(out[0].view(torch.uint8)) == 0
        assert torch.isneginf(out[2]).all() and torch.isposinf(out[3]).all()
        return
    with pytest.raises((ValueError, RuntimeError)):
        fused_norm_rope_attn_rope_cast_bf16(q, kv, ids, ptr, cos, sin, pos, **kw)
