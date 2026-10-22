# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Public dispatch, checked/unchecked tiles, prepared Q and replay for fused attention."""

import pytest
import test_fused_norm_rope_attn_rope_cast_fp8_qk as oracle
import torch
from test_fused_norm_rope_attn_rope_cast import csr, dequantize, quantize_fp8, rotate
from test_fused_norm_rope_attn_rope_cast_fp8_qk import dense, fixture

from aiter import fused_norm_rope_attn_rope_cast

pytestmark = oracle.pytestmark


@pytest.mark.parametrize("page", [0, 128])
@pytest.mark.parametrize("raw,q_norm", [(False, False), (True, False), (True, True)])
@pytest.mark.parametrize("pos64", [False, True])
@pytest.mark.parametrize("mode", ["mixed", "valid", "truncated"])
def test_fused_norm_rope_attn_rope_cast_contract(q_norm, page, raw, pos64, mode):
    args, kw, rows, kv, sink = fixture(128, page, raw)
    q, pool, _ids, _ptr, cos, sin, pos = args
    if pos64:
        pos = pos.long()
    if mode == "valid":
        rows = [
            (torch.arange(1153, device="cuda") * 17 + i) % len(kv)
            for i in range(len(rows))
        ]
    # Padded CSR selects the new dispatch while retaining empty, invalid and tail queries.
    rows = [
        torch.cat((r, torch.full((1153 - len(r),), -1, device="cuda", dtype=r.dtype)))
        for r in rows
    ]
    ids, ptr = csr(rows, torch.int32, -1)
    n = len(kv)
    if mode == "truncated":
        n = 83
        kv = kv[:n]
        if page:
            kw["cache_scales"] = kw["cache_scales"][: n * 16]
        else:
            pool = pool[: 4 + n * 528]
    if not page:
        ids = torch.where(
            (ids >= 0) & (ids < 1280),
            ids.long() * 1056 + 9,
            torch.full_like(ids.long(), -1),
        )
    norm_eps = 0.03125
    normalized = (
        (
            q.float()
            * torch.rsqrt(q.float().square().mean(-1, keepdim=True) + norm_eps)
        ).bfloat16()
        if q_norm
        else q
    )
    debug = torch.empty_like(q, dtype=torch.float32)
    common = dict(**kw, fp8_qk=True, attn_sink=sink, debug_out=debug)
    out = fused_norm_rope_attn_rope_cast(
        q,
        pool,
        ids,
        ptr,
        cos,
        sin,
        pos,
        q_is_rotated=not raw,
        q_norm=q_norm,
        norm_eps=norm_eps,
        **common,
    )
    expected, mx, lse = dense(normalized, kv, rows, cos, sin, pos, sink, raw)
    torch.testing.assert_close(out[2], mx, rtol=1e-4, atol=3e-5)
    torch.testing.assert_close(out[3], lse, rtol=1e-4, atol=3e-5)
    torch.testing.assert_close(debug, expected, rtol=0.02, atol=0.008)
    assert float((debug - expected).norm() / expected.norm().clamp_min(1e-10)) < 0.005
    assert (
        float(
            (dequantize(*out[:2]) - expected).norm() / expected.norm().clamp_min(1e-10)
        )
        < 0.035
    )
    rotated = rotate(normalized, cos, sin, pos).bfloat16() if raw else q
    q8, qs = quantize_fp8(rotated)
    prepared = fused_norm_rope_attn_rope_cast(
        q8.reshape_as(q),
        pool,
        ids,
        ptr,
        cos,
        sin,
        pos,
        q_scale=qs.reshape(q.shape[0], 128, 16),
        **common,
    )
    for x, y in zip(out, prepared):
        torch.testing.assert_close(
            x.view(torch.uint8), y.view(torch.uint8), rtol=0, atol=0
        )
    buffers = {"out": out[0], "out_scale": out[1], "max_logits": out[2], "lse": out[3]}
    refs = [x.clone() for x in out]
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fused_norm_rope_attn_rope_cast(
            q,
            pool,
            ids,
            ptr,
            cos,
            sin,
            pos,
            q_is_rotated=not raw,
            q_norm=q_norm,
            norm_eps=norm_eps,
            **common,
            **buffers,
        )
    for x in out:
        x.view(torch.uint8).fill_(255)
    graph.replay()
    for x, y in zip(out, refs):
        torch.testing.assert_close(
            x.view(torch.uint8), y.view(torch.uint8), rtol=0, atol=0
        )
