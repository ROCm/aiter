# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Independent quantized-Q oracle, cache layout, masking and replay checks."""

import pytest
import torch
from test_fused_norm_rope_attn_rope_cast import (
    csr,
    dequantize,
    pack_rows,
    quantize_fp8,
    rotate,
    simple_inputs,
)

from aiter import fused_norm_rope_attn_rope_cast

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or not torch.cuda.get_device_properties(0).gcnArchName.startswith("gfx950"),
    reason="requires gfx950",
)


def fixture(heads, page, raw_q, seed=711):
    torch.manual_seed(seed + heads)
    n = 1280
    # Group scales deliberately span six powers of two; uniform scales hide lane errors.
    gain = torch.exp2(
        torch.arange(16, device="cuda").float() % 7 - 3
    ).repeat_interleave(32)
    kv = (torch.randn(n, 512, device="cuda") * 0.2 * gain).bfloat16()
    values, codes = quantize_fp8(kv)
    decoded = (
        values.float() * torch.exp2(codes.float() - 127).repeat_interleave(32, -1)
    ).bfloat16()
    counts = [0, 1, 31, 32, 33, 65, 129, 1153]
    rows = []
    for count in counts:
        ix = torch.randperm(n, device="cuda")[:count].int()
        if count > 1:
            ix[::37] = -1
            ix[32:64] = n + 7
        rows.append(ix)
    ids, ptr = csr(rows, torch.int32, n + 7)
    kw = {}
    if page:
        # Independent data/scale bases with padding, including a non-specialized B=7.
        pages = (n + page - 1) // page
        ds, ss = page * 512 + 16, page * 16 + 16
        data = torch.zeros(pages, ds, device="cuda", dtype=torch.uint8)
        scales = torch.zeros(pages, ss, device="cuda", dtype=torch.uint8)
        ix = torch.arange(n, device="cuda")
        data[:, : page * 512].view(pages, page, 512)[ix // page, ix % page] = (
            values.view(torch.uint8)
        )
        scales[:, : page * 16].view(pages, page, 16)[ix // page, ix % page] = codes
        pool = data.flatten()
        kw = {
            "cache_scales": scales.flatten(),
            "cache_page_size": page,
            "cache_num_rows": n,
            "cache_data_page_stride": ds,
            "cache_scale_page_stride": ss,
        }
    else:
        pool = torch.cat(
            (
                torch.zeros(4, device="cuda", dtype=torch.uint8),
                pack_rows(values, codes).flatten(),
            )
        )
        ids = torch.where(
            (ids >= 0) & (ids < n),
            ids.long() * 1056 + 9,
            torch.full_like(ids.long(), -1),
        )
    q = (
        torch.randn(len(counts), heads, 512, device="cuda") * 0.2 * gain.flip(0)
    ).bfloat16()
    angle = torch.randn(16, 32, device="cuda")
    cos, sin = angle.cos().bfloat16(), angle.sin().bfloat16()
    pos = torch.arange(len(counts), device="cuda", dtype=torch.int32)
    sink = torch.linspace(-3, 3, heads, device="cuda")
    sink[::11] = -torch.inf
    return (q, pool, ids, ptr, cos, sin, pos), kw, rows, decoded, sink


def dense(q, kv, rows, cos, sin, pos, sink, raw_q):
    rq = rotate(q, cos, sin, pos).bfloat16() if raw_q else q
    q8, qs = quantize_fp8(rq)
    qref = (
        q8.float() * torch.exp2(qs.float() - 127).repeat_interleave(32, -1)
    ).reshape_as(q)
    results = []
    maxima = []
    lses = []
    for t, ix in enumerate(rows):
        valid = ix[(ix >= 0) & (ix < kv.shape[0])].long()
        if not len(valid):
            results.append(torch.zeros_like(q[t], dtype=torch.float32))
            maxima.append(torch.full((q.shape[1],), -torch.inf, device=q.device))
            lses.append(torch.full((q.shape[1],), torch.inf, device=q.device))
            continue
        selected = kv[valid].float()
        logits = qref[t] @ selected.T * 512**-0.5
        maxima.append(logits.max(-1).values)
        lses.append(logits.logsumexp(-1))
        p = torch.cat((logits, sink[:, None]), -1).softmax(-1)[:, : len(valid)]
        results.append(p @ selected)
    return (
        rotate(torch.stack(results).bfloat16(), cos, sin, pos, True),
        torch.stack(maxima),
        torch.stack(lses),
    )


@pytest.mark.parametrize("heads", [64, 128])
@pytest.mark.parametrize("page", [0, 7, 32, 128])
@pytest.mark.parametrize("raw_q,q_norm", [(False, False), (True, False), (True, True)])
def test_fp8_qk_dense(q_norm, heads, page, raw_q):
    args, kw, rows, kv, sink = fixture(heads, page, raw_q)
    q, _, _, _, cos, sin, pos = args
    norm_eps = 0.03125
    normalized = (
        (
            q.float()
            * torch.rsqrt(q.float().square().mean(-1, keepdim=True) + norm_eps)
        ).bfloat16()
        if q_norm
        else q
    )
    expected, mx, lse = dense(normalized, kv, rows, cos, sin, pos, sink, raw_q)
    debug = torch.empty_like(q, dtype=torch.float32)
    out = fused_norm_rope_attn_rope_cast(
        *args,
        **kw,
        attn_sink=sink,
        q_is_rotated=not raw_q,
        q_norm=q_norm,
        norm_eps=norm_eps,
        fp8_qk=True,
        debug_out=debug,
    )
    torch.testing.assert_close(out[2], mx, rtol=1e-4, atol=3e-5)
    torch.testing.assert_close(out[3], lse, rtol=1e-4, atol=3e-5)
    torch.testing.assert_close(debug, expected, rtol=0.02, atol=0.008)
    assert float((debug - expected).norm() / expected.norm()) < 0.005
    assert float((dequantize(*out[:2]) - expected).norm() / expected.norm()) < 0.035
    assert torch.count_nonzero(debug[0]) == 0


@pytest.mark.parametrize("kind", ["heads", "bf16", "extend"])
def test_fp8_qk_reject(kind):
    args = simple_inputs(heads=32 if kind == "heads" else 64)
    kw = {}
    if kind == "bf16":
        args[1] = torch.ones(1, 512, device="cuda", dtype=torch.bfloat16)
        args[2] = torch.zeros(1, device="cuda", dtype=torch.int32)
    if kind == "extend":
        kw = {
            "extend_kv": torch.ones(1, 512, device="cuda", dtype=torch.bfloat16),
            "extend": torch.zeros(1, device="cuda", dtype=torch.int32),
            "extend_indptr": args[3],
        }
    with pytest.raises((ValueError, RuntimeError), match="FP8 QK requires"):
        fused_norm_rope_attn_rope_cast(*args, **kw, fp8_qk=True)


def test_fp8_qk_replay():
    args, kw, _, _, sink = fixture(128, 128, True, seed=821)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        out = fused_norm_rope_attn_rope_cast(
            *args, **kw, fp8_qk=True, q_is_rotated=False, attn_sink=sink
        )
        refs = [x.clone() for x in out]
        buffers = {
            "out": out[0],
            "out_scale": out[1],
            "max_logits": out[2],
            "lse": out[3],
        }
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            fused_norm_rope_attn_rope_cast(
                *args, **kw, **buffers, fp8_qk=True, q_is_rotated=False, attn_sink=sink
            )
        for _ in range(5):
            for x in out:
                x.view(torch.uint8).fill_(255)
            graph.replay()
            for x, y in zip(out, refs):
                torch.testing.assert_close(
                    x.view(torch.uint8), y.view(torch.uint8), rtol=0, atol=0
                )
    torch.cuda.current_stream().wait_stream(stream)


def test_fp8_qk_zero_tokens():
    out = fused_norm_rope_attn_rope_cast(*simple_inputs(0), fp8_qk=True)
    assert all(x.numel() == 0 for x in out)


@pytest.mark.parametrize("heads", [64, 128])
@pytest.mark.parametrize("page", [0, 7, 32, 128])
def test_fp8_prepared_q(heads, page):
    args, kw, _, _, sink = fixture(heads, page, True)
    q, _pool, _ids, _ptr, cos, sin, pos = args
    q8, qs = quantize_fp8(rotate(q, cos, sin, pos).bfloat16())
    q8 = q8.reshape_as(q)
    qs = qs.reshape(q.shape[0], heads, 16)
    out = fused_norm_rope_attn_rope_cast(
        *args, **kw, fp8_qk=True, q_is_rotated=False, attn_sink=sink
    )
    prepared = fused_norm_rope_attn_rope_cast(
        q8, *args[1:], **kw, fp8_qk=True, q_scale=qs, attn_sink=sink
    )
    for x, y in zip(out, prepared):
        torch.testing.assert_close(
            x.view(torch.uint8), y.view(torch.uint8), rtol=0, atol=0
        )


@pytest.mark.parametrize("bad", ["flag", "rotate", "shape", "dtype", "q_dtype"])
def test_prepared_q_contract(bad):
    args = simple_inputs()
    q8, qs = quantize_fp8(args[0])
    args[0] = q8.reshape_as(args[0])
    qs = qs.reshape(1, 64, 16)
    kw = {"fp8_qk": True, "q_scale": qs}
    if bad == "flag":
        kw["fp8_qk"] = False
    if bad == "rotate":
        kw["q_is_rotated"] = False
    if bad == "shape":
        kw["q_scale"] = qs.reshape(-1)
    if bad == "dtype":
        kw["q_scale"] = qs.int()
    if bad == "q_dtype":
        args[0] = args[0].bfloat16()
    with pytest.raises((ValueError, RuntimeError)):
        fused_norm_rope_attn_rope_cast(*args, **kw)
