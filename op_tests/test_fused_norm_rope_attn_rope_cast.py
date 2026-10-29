# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Fused norm/RoPE/attention/cast: dense references, cache modes and replay.

Direct execution runs contract/replay regressions before the correctness
and performance sweep. For example: -b 1 128 -H 64 --topk 512 --modes prepared.
"""

import argparse
import itertools
import math
from unittest import TestCase
from unittest.mock import patch

import pandas as pd
import torch

import aiter
from aiter import (
    dtypes,
    fused_norm_rope_attn_rope_cast_bf16,
)
from aiter import (
    fused_norm_rope_attn_rope_cast as fused_v41,
)
from aiter import (
    fused_norm_rope_attn_rope_cast_legacy as fused_norm_rope_attn_rope_cast,
)
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.fused_norm_rope_attn_rope_cast import (
    _bf16_kv_rows,
    _fused_norm_rope_attn_rope_cast_legacy_with_debug,
    _fused_norm_rope_attn_rope_cast_with_debug,
)
from aiter.ops.fused_norm_rope_attn_rope_cast import (
    _fused_norm_rope_attn_rope_cast_fwd as fused_v41_fwd,
)
from aiter.ops.fused_norm_rope_attn_rope_cast import (
    _fused_norm_rope_attn_rope_cast_legacy_fwd as fused_norm_rope_attn_rope_cast_fwd,
)
from aiter.test_common import benchmark, checkAllclose, run_perftest

SUPPORTED_GFX = ["gfx950"]  # The fused FP8 MFMA kernels target gfx950 only.


def _raises(expected, match=None):
    case = TestCase()
    if match is None:
        return case.assertRaises(expected)
    return case.assertRaisesRegex(expected, match)


def _call_v41(*args, **kwargs):
    if "debug_out" in kwargs:
        return _fused_norm_rope_attn_rope_cast_with_debug(*args, **kwargs)
    return fused_v41(*args, **kwargs)


def _call_legacy(*args, **kwargs):
    if "debug_out" in kwargs:
        return _fused_norm_rope_attn_rope_cast_legacy_with_debug(*args, **kwargs)
    return fused_norm_rope_attn_rope_cast(*args, **kwargs)


def _call_bf16(q, kv, indices, indptr, cos, sin, positions, **kwargs):
    if "debug_out" in kwargs:
        return _fused_norm_rope_attn_rope_cast_legacy_with_debug(
            q, _bf16_kv_rows(kv), indices, indptr, cos, sin, positions, **kwargs
        )
    return fused_norm_rope_attn_rope_cast_bf16(
        q, kv, indices, indptr, cos, sin, positions, **kwargs
    )


def quantize_fp8(x):
    groups = x.float().reshape(-1, 16, 32)
    exponent = (groups.abs().amax(-1).clamp_min(1e-30) / 448).log2().ceil()
    scale = torch.exp2(exponent)
    values = (groups / scale[..., None]).to(torch.float8_e4m3fn).reshape(-1, 512)
    return values, (exponent + 127).to(torch.uint8)


def pack_rows(values, scales):
    return torch.cat((values.view(torch.uint8), scales), dim=-1).contiguous()


def decode(pool, ids):
    # CPU reference preserves FP32 subnormals for the full E8M0 codec range.
    offsets = (ids.cpu() >> 1).long()
    records = pool.cpu().flatten()[offsets[:, None] + torch.arange(528)[None, :]]
    values = records[:, :512].contiguous().view(torch.float8_e4m3fn).float()
    scales = torch.exp2(records[:, 512:].int().float() - 127).repeat_interleave(32, -1)
    return (values * scales).bfloat16().to(pool.device)


def csr(rows, dtype, garbage):
    counts = [len(row) for row in rows]
    ptr = torch.tensor([0, *counts], device="cuda", dtype=torch.int32).cumsum(
        0, dtype=torch.int32
    )
    ids = torch.full((sum(counts) + 71,), garbage, device="cuda", dtype=dtype)
    if sum(counts):
        ids[: sum(counts)] = torch.cat(rows).to(dtype)
    return ids, ptr


def rotate(x, cos, sin, positions, inverse=False):
    result = x.float().clone()
    pairs = x.float()[..., -64:].reshape(x.shape[0], x.shape[1], 32, 2)
    c = cos[positions.long()].float()[:, None, :]
    s = sin[positions.long()].float()[:, None, :] * (-1 if inverse else 1)
    result[..., -64:] = torch.stack(
        (pairs[..., 0] * c - pairs[..., 1] * s, pairs[..., 0] * s + pairs[..., 1] * c),
        -1,
    ).flatten(-2)
    return result


def dense_legacy_reference(
    q, pool, ids, ptr, cos, sin, pos, sink, extra=None, eids=None, eptr=None
):
    outputs = []
    maxima = []
    lses = []
    for t in range(q.shape[0]):
        rows = decode(pool, ids[int(ptr[t]) : int(ptr[t + 1])])
        if extra is not None:
            rows = torch.cat(
                (rows, extra[eids[int(eptr[t]) : int(eptr[t + 1])].long()])
            )
        if not rows.shape[0]:
            outputs.append(torch.zeros_like(q[t], dtype=torch.float32))
            maxima.append(torch.full((q.shape[1],), -torch.inf, device=q.device))
            lses.append(torch.full((q.shape[1],), torch.inf, device=q.device))
            continue
        logits = q[t].float() @ rows.float().T * 512**-0.5
        maxima.append(logits.max(-1).values)
        lses.append(logits.logsumexp(-1))
        all_logits = (
            torch.cat((logits, sink[:, None]), -1) if sink is not None else logits
        )
        probability = all_logits.softmax(-1)[:, : rows.shape[0]]
        outputs.append(probability @ rows.float())
    return (
        rotate(torch.stack(outputs).bfloat16(), cos, sin, pos, True),
        torch.stack(maxima),
        torch.stack(lses),
    )


def dequantize(out, scales):
    t, groups, _ = out.shape
    return (out.float() * scales.float().repeat_interleave(32, -1)).reshape(
        t, groups * 8, 512
    )


def simple_inputs(tokens=1, heads=64):
    q = torch.zeros(tokens, heads, 512, device="cuda", dtype=torch.bfloat16)
    pool = pack_rows(
        *quantize_fp8(torch.ones(1, 512, device="cuda", dtype=torch.bfloat16))
    ).flatten()
    ids = torch.ones(tokens, device="cuda", dtype=torch.int64)
    ptr = torch.arange(tokens + 1, device="cuda", dtype=torch.int32)
    cos = torch.ones(1, 32, device="cuda", dtype=torch.bfloat16)
    return [
        q,
        pool,
        ids,
        ptr,
        cos,
        torch.zeros_like(cos),
        torch.zeros(tokens, device="cuda", dtype=torch.int32),
    ]


def dense_bf16_reference(q, decoded, rows, cos, sin, pos, sink, extra=None, erows=None):
    ys, ms, ls = [], [], []
    for t, ids in enumerate(rows):
        kv = decoded[ids.long()]
        if extra is not None:
            kv = torch.cat((kv, extra[erows[t].long()]))
        if not len(kv):
            ys.append(torch.zeros_like(q[t], dtype=torch.float32))
            ms.append(torch.full((q.shape[1],), -torch.inf, device="cuda"))
            ls.append(torch.full((q.shape[1],), torch.inf, device="cuda"))
            continue
        logits = q[t].float() @ kv.float().T * 512**-0.5
        ms.append(logits.max(-1).values)
        ls.append(logits.logsumexp(-1))
        p = torch.cat((logits, sink[:, None]), -1).softmax(-1)[:, : len(kv)]
        ys.append(p @ kv.float())
    return (
        rotate(torch.stack(ys).bfloat16(), cos, sin, pos, True),
        torch.stack(ms),
        torch.stack(ls),
    )


def make_fp8_qk_inputs(heads, raw_q, seed=711):
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


def dense_fp8_qk_reference(q, kv, rows, cos, sin, pos, sink, raw_q):
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


# Legacy cache and dispatch.


def check_ragged_dense(q_norm, heads, raw_q, with_extend):
    torch.manual_seed(9100 + heads)
    values = torch.randn(1200, 512, device="cuda", dtype=torch.bfloat16) * 0.5
    pool = torch.cat(
        (
            torch.zeros(4, device="cuda", dtype=torch.uint8),
            pack_rows(*quantize_fp8(values)).flatten(),
        )
    )
    all_ids = torch.arange(1200, device="cuda", dtype=torch.int64) * 1056 + 9
    counts = [0, 1, 31, 32, 33, 65, 127, 1153]
    rows = []
    clean = []
    for n in counts:
        ids = all_ids[torch.randperm(1200, device="cuda")[:n]].clone()
        ordinal = torch.arange(n, device="cuda")
        ids[((ordinal >= 32) & (ordinal < 64)) | (ordinal % 71 == 0)] = -1
        if n > 2:
            ids[-2] = 2**45
        rows.append(ids)
        clean.append(ids[(ids >= 0) & (ids < 2**45)])
    ids, ptr = csr(rows, torch.int64, 2**45)
    cids, cptr = csr(clean, torch.int64, 2**45)
    q = torch.randn(8, heads, 512, device="cuda", dtype=torch.bfloat16) * 0.5
    angle = torch.randn(64, 32, device="cuda")
    cos = angle.cos().bfloat16()
    sin = angle.sin().bfloat16()
    pos = torch.arange(8, device="cuda", dtype=torch.int64) * 7
    sink = torch.linspace(-3, 3, heads, device="cuda")
    sink[::11] = -torch.inf
    extra = eids = eptr = ceids = ceptr = None
    options = {}
    if with_extend:
        extra = torch.randn(193, 512, device="cuda", dtype=torch.bfloat16) * 0.5
        erows = []
        clean_erows = []
        for n in [0, 1, 31, 32, 33, 65, 127, 129]:
            selected = torch.randperm(193, device="cuda")[:n].int()
            invalid = torch.arange(n, device="cuda") % 19 == 0
            clean_erows.append(selected[~invalid])
            selected[invalid] = 200
            erows.append(selected)
        eids, eptr = csr(erows, torch.int32, 2**30)
        ceids, ceptr = csr(clean_erows, torch.int32, 2**30)
        options = {"extend_kv": extra, "extend": eids, "extend_indptr": eptr}
    norm_eps = 0.03125
    normalized = (
        (
            q.float()
            * torch.rsqrt(q.float().square().mean(-1, keepdim=True) + norm_eps)
        ).bfloat16()
        if q_norm
        else q
    )
    qr = rotate(normalized, cos, sin, pos).bfloat16() if raw_q else q
    expected, maxima, lses = dense_legacy_reference(
        qr, pool, cids, cptr, cos, sin, pos, sink, extra, ceids, ceptr
    )
    debug = torch.empty_like(q, dtype=torch.float32)
    out = _call_legacy(
        q,
        pool,
        ids,
        ptr,
        cos,
        sin,
        pos,
        attn_sink=sink,
        q_is_rotated=not raw_q,
        q_norm=q_norm,
        norm_eps=norm_eps,
        debug_out=debug,
        **options,
    )
    torch.testing.assert_close(debug, expected, rtol=0.02, atol=0.008)
    assert float((debug - expected).norm() / expected.norm()) < 0.005
    torch.testing.assert_close(out[2], maxima, rtol=1e-4, atol=3e-5)
    torch.testing.assert_close(out[3], lses, rtol=1e-4, atol=3e-5)
    quant = dequantize(*out[:2])
    assert float((quant - expected).norm() / expected.norm()) < 0.035
    assert torch.count_nonzero(debug[0]) == 0


def check_invalid_metadata(bad):
    args = simple_inputs()
    kw = {}
    if bad == "q_dtype":
        args[0] = args[0].float()
    elif bad == "q_stride":
        args[0] = torch.empty(1, 64, 1024, device="cuda", dtype=torch.bfloat16)[
            ..., ::2
        ]
    elif bad == "q_alignment":
        args[0] = torch.empty(args[0].numel() + 1, device="cuda", dtype=torch.bfloat16)[
            1:
        ].view_as(args[0])
    elif bad == "pool_alignment":
        args[1] = torch.cat((args[1], args[1]))[1:529]
    elif bad == "positions_dtype":
        args[-1] = args[-1].float()
    elif bad == "csr_dtype":
        args[3] = args[3].long()
    elif bad == "missing_extend":
        kw["extend_kv"] = args[0][0]
    elif bad == "output_shape":
        kw["out"] = torch.empty(1, 64, 512, device="cuda", dtype=torch.float8_e4m3fn)
    elif bad == "scale":
        kw["softmax_scale"] = float("nan")
    with _raises((ValueError, RuntimeError)):
        _call_legacy(*args, **kw)


def check_zero_tokens():
    out = _call_legacy(*simple_inputs(0))
    assert all(x.numel() == 0 for x in out)


def check_nondefault_stream_and_buffers():
    args = simple_inputs(5)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        out = _call_legacy(*args)
        again = _call_legacy(
            *args, out=out[0], out_scale=out[1], max_logits=out[2], lse=out[3]
        )
        assert all(x is y for x, y in zip(out, again))
    torch.cuda.current_stream().wait_stream(stream)
    torch.testing.assert_close(
        dequantize(*out[:2]), torch.ones_like(args[0]).float(), rtol=0, atol=0
    )


def check_prefetch_codec_empty_tiles_and_replay(heads):
    codes = torch.arange(256, device="cuda", dtype=torch.uint8).repeat(2)
    codes[codes == 127] = 126
    codes[codes == 255] = 254
    scales = torch.tensor([0, 1, 80, 127, 140, 200], device="cuda", dtype=torch.uint8)
    records = torch.cat(
        (codes.expand(6, -1), scales[:, None].expand(-1, 16)), 1
    ).contiguous()
    pool = torch.cat(
        (torch.zeros(4, device="cuda", dtype=torch.uint8), records.flatten())
    )
    row_ids = torch.arange(6, device="cuda", dtype=torch.int64) * 1056 + 9
    ids = torch.full((6, 65), -1, device="cuda", dtype=torch.int64)
    ptr = torch.arange(7, device="cuda", dtype=torch.int32) * 65
    q = torch.zeros(6, heads, 512, device="cuda", dtype=torch.bfloat16)
    cos = torch.ones(1, 32, device="cuda", dtype=torch.bfloat16)
    sin = torch.zeros_like(cos)
    pos = torch.zeros(6, device="cuda", dtype=torch.int32)
    debug = torch.empty_like(q, dtype=torch.float32)
    args = (q, pool, ids.flatten(), ptr, cos, sin, pos)
    out = _call_legacy(*args, debug_out=debug, q_is_rotated=False)
    buffers = {
        "out": out[0],
        "out_scale": out[1],
        "max_logits": out[2],
        "lse": out[3],
        "debug_out": debug,
        "q_is_rotated": False,
    }
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _call_legacy(*args, **buffers)
    expected = decode(pool, row_ids).float()[:, None, :].expand_as(debug)
    for slot in (64, 32, -1, 0, 64):
        ids.fill_(-1)
        if slot >= 0:
            ids[:, slot].copy_(row_ids)
        debug.fill_(float("nan"))
        graph.replay()
        if slot < 0:
            assert torch.count_nonzero(debug) == 0
            assert torch.isposinf(out[3]).all()
        else:
            torch.testing.assert_close(debug, expected, rtol=0, atol=0)
            torch.testing.assert_close(out[3], torch.zeros_like(out[3]), rtol=0, atol=0)


def check_prefetch_long_invalid_decode(heads, raw_q):
    torch.manual_seed(31500 + heads)
    counts = [0, 1, 31, 32, 33, 127, 513, 1153]
    values = torch.randn(1200, 512, device="cuda", dtype=torch.bfloat16) * 0.5
    pool = torch.cat(
        (
            torch.zeros(4, device="cuda", dtype=torch.uint8),
            pack_rows(*quantize_fp8(values)).flatten(),
        )
    )
    all_ids = torch.arange(1200, device="cuda", dtype=torch.int64) * 1056 + 9
    rows = []
    clean = []
    for n in counts:
        ids = all_ids[torch.randperm(1200, device="cuda")[:n]].clone()
        ordinal = torch.arange(n, device="cuda")
        ids[((ordinal >= 32) & (ordinal < 64)) | (ordinal % 73 == 0)] = -1
        rows.append(ids)
        clean.append(ids[ids >= 0])
    ids, ptr = csr(rows, torch.int64, 2**45)
    cids, cptr = csr(clean, torch.int64, 2**45)
    q = torch.randn(8, heads, 512, device="cuda", dtype=torch.bfloat16) * 0.5
    angles = torch.randn(64, 32, device="cuda")
    cos = angles.cos().bfloat16()
    sin = angles.sin().bfloat16()
    pos = torch.arange(8, device="cuda", dtype=torch.int64) * 7
    sink = torch.linspace(-3, 3, heads, device="cuda")
    sink[::11] = -torch.inf
    qr = rotate(q, cos, sin, pos).bfloat16() if raw_q else q
    expected, maxima, lses = dense_legacy_reference(
        qr, pool, cids, cptr, cos, sin, pos, sink
    )
    debug = torch.empty_like(q, dtype=torch.float32)
    out = _call_legacy(
        q,
        pool,
        ids,
        ptr,
        cos,
        sin,
        pos,
        attn_sink=sink,
        q_is_rotated=not raw_q,
        debug_out=debug,
    )
    torch.testing.assert_close(debug, expected, rtol=0.02, atol=0.008)
    assert float((debug - expected).norm() / expected.norm()) < 0.005
    torch.testing.assert_close(out[2], maxima, rtol=1e-4, atol=3e-5)
    torch.testing.assert_close(out[3], lses, rtol=1e-4, atol=3e-5)


def check_prefetch_dispatch_boundary(heads, tokens):
    q = torch.zeros(tokens, heads, 512, device="cuda", dtype=torch.bfloat16)
    row = (
        torch.arange(512, device="cuda")
        .remainder(17)
        .sub(8)
        .mul(0.125)
        .bfloat16()[None, :]
    )
    pool = pack_rows(*quantize_fp8(row)).flatten()
    ids = torch.ones(tokens * 65, device="cuda", dtype=torch.int64)
    ptr = torch.arange(tokens + 1, device="cuda", dtype=torch.int32) * 65
    cos = torch.ones(1, 32, device="cuda", dtype=torch.bfloat16)
    sin = torch.zeros_like(cos)
    pos = torch.zeros(tokens, device="cuda", dtype=torch.int32)
    out = _call_legacy(q, pool, ids, ptr, cos, sin, pos, q_is_rotated=False)
    value = dequantize(*out[:2])
    torch.testing.assert_close(
        value, row.float()[None, :, :].expand_as(value), rtol=0, atol=0
    )
    torch.testing.assert_close(
        out[3], torch.full_like(out[3], 65.0).log(), rtol=1e-6, atol=1e-6
    )


def check_h128_load_schedule_dispatch_boundary(tokens, raw_q):
    # Uniform QK and one repeated physical row have an analytic attention
    # result. K33 exercises both panels across the H128 grid-size switch.
    heads = 128
    q = torch.zeros(tokens, heads, 512, device="cuda", dtype=torch.bfloat16)
    row = (
        torch.arange(512, device="cuda")
        .remainder(17)
        .sub(8)
        .mul(0.125)
        .bfloat16()[None, :]
    )
    pool = pack_rows(*quantize_fp8(row)).flatten()
    ids = torch.ones(tokens * 33, device="cuda", dtype=torch.int64)
    ptr = torch.arange(tokens + 1, device="cuda", dtype=torch.int32) * 33
    cos = torch.ones(1, 32, device="cuda", dtype=torch.bfloat16)
    sin = torch.zeros_like(cos)
    pos = torch.zeros(tokens, device="cuda", dtype=torch.int64)
    out = _call_legacy(q, pool, ids, ptr, cos, sin, pos, q_is_rotated=not raw_q)
    actual = dequantize(*out[:2])
    torch.testing.assert_close(
        actual, row.float()[None, :, :].expand_as(actual), rtol=0, atol=0
    )
    torch.testing.assert_close(out[2], torch.zeros_like(out[2]), rtol=0, atol=0)
    torch.testing.assert_close(
        out[3], torch.full_like(out[3], 33.0).log(), rtol=1e-6, atol=1e-6
    )


# BF16 cache.


def check_bf16_dense(q_norm, heads, raw, singleton):
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
    expected, maxima, lses = dense_bf16_reference(qr, kv, clean, cos, sin, pos, sink)
    debug = torch.empty_like(q, dtype=torch.float32)
    out = _call_bf16(
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


def check_bf16_stream_replay(heads, tokens):
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
        out = _call_bf16(q, kv, ids, ptr, cos, sin, pos, q_is_rotated=False)
        if not tokens:
            assert all(x.numel() == 0 for x in out)
            return
        buffers = dict(zip(("out", "out_scale", "max_logits", "lse"), out))
        expected = [x.clone() for x in out]
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            _call_bf16(q, kv, ids, ptr, cos, sin, pos, q_is_rotated=False, **buffers)
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


def check_bf16_contract(bad):
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
    elif bad == "empty_cache":
        out = _call_bf16(q, kv[:0], ids, ptr, cos, sin, pos)
        assert torch.count_nonzero(out[0].view(torch.uint8)) == 0
        assert torch.isneginf(out[2]).all() and torch.isposinf(out[3]).all()
        return
    with _raises((ValueError, RuntimeError)):
        _call_bf16(q, kv, ids, ptr, cos, sin, pos, **kw)


# FP8 QK and prepared Q.


def check_fp8_qk_dense(q_norm, heads, raw_q):
    args, kw, rows, kv, sink = make_fp8_qk_inputs(heads, raw_q)
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
    expected, mx, lse = dense_fp8_qk_reference(
        normalized, kv, rows, cos, sin, pos, sink, raw_q
    )
    debug = torch.empty_like(q, dtype=torch.float32)
    out = _call_legacy(
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


def check_fp8_qk_reject(kind):
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
    with _raises((ValueError, RuntimeError), match="FP8 QK requires"):
        _call_legacy(*args, **kw, fp8_qk=True)


def check_fp8_qk_replay():
    args, kw, _, _, sink = make_fp8_qk_inputs(128, True, seed=821)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        out = _call_legacy(*args, **kw, fp8_qk=True, q_is_rotated=False, attn_sink=sink)
        refs = [x.clone() for x in out]
        buffers = {
            "out": out[0],
            "out_scale": out[1],
            "max_logits": out[2],
            "lse": out[3],
        }
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            _call_legacy(
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


def check_fp8_qk_zero_tokens():
    out = _call_legacy(*simple_inputs(0), fp8_qk=True)
    assert all(x.numel() == 0 for x in out)


def check_fp8_prepared_q(heads):
    args, kw, _, _, sink = make_fp8_qk_inputs(heads, True)
    q, _pool, _ids, _ptr, cos, sin, pos = args
    q8, qs = quantize_fp8(rotate(q, cos, sin, pos).bfloat16())
    q8 = q8.reshape_as(q)
    qs = qs.reshape(q.shape[0], heads, 16)
    out = _call_legacy(*args, **kw, fp8_qk=True, q_is_rotated=False, attn_sink=sink)
    prepared = _call_legacy(
        q8, *args[1:], **kw, fp8_qk=True, q_scale=qs, attn_sink=sink
    )
    for x, y in zip(out, prepared):
        torch.testing.assert_close(
            x.view(torch.uint8), y.view(torch.uint8), rtol=0, atol=0
        )


def check_prepared_q_contract(bad):
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
    with _raises((ValueError, RuntimeError)):
        _call_legacy(*args, **kw)


# Prefill integration.


def check_fused_norm_rope_attn_rope_cast_contract(q_norm, raw, pos64, mode):
    args, kw, rows, kv, sink = make_fp8_qk_inputs(128, raw)
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
        pool = pool[: 4 + n * 528]
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
    out = _call_legacy(
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
    expected, mx, lse = dense_fp8_qk_reference(
        normalized, kv, rows, cos, sin, pos, sink, raw
    )
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
    prepared = _call_legacy(
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
        _call_legacy(
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


# Optional Q RMSNorm.


def check_norm_contract(direct, bad):
    args = simple_inputs()
    kw = {"q_norm": True, "q_is_rotated": False}
    if bad == "rotated":
        kw["q_is_rotated"] = True
    elif bad == "prepared":
        args[0] = args[0].to(torch.float8_e4m3fn)
        kw.update(
            fp8_qk=True,
            q_is_rotated=True,
            q_scale=torch.full((1, 64, 16), 127, device="cuda", dtype=torch.uint8),
        )
    else:
        kw["norm_eps"] = {
            "zero": 0,
            "negative": -1,
            "nan": float("nan"),
            "infinity": float("inf"),
            "underflow": 1e-40,
        }[bad]
    with _raises((ValueError, RuntimeError), match="q_norm|norm_eps"):
        if direct:
            q = args[0]
            outputs = [
                torch.empty((1, 8, 4096), device=q.device, dtype=torch.float8_e4m3fn),
                torch.empty((1, 8, 128), device=q.device, dtype=torch.float8_e8m0fnu),
                torch.empty((1, 64), device=q.device),
                torch.empty((1, 64), device=q.device),
            ]
            fused_norm_rope_attn_rope_cast_fwd(
                *args, *outputs, None, None, None, None, None, 512**-0.5, **kw
            )
        else:
            _call_legacy(*args, **kw)


def check_norm_zero_q(tokens):
    args = simple_inputs(tokens)
    out = _call_legacy(*args, q_norm=True, q_is_rotated=False)
    assert all(torch.isfinite(x.float()).all() for x in out)
    assert torch.count_nonzero(out[2]) == torch.count_nonzero(out[3]) == 0


def check_norm_recomputed_on_graph_replay():

    args, kw, rows, kv, sink = make_fp8_qk_inputs(128, True)
    q, pool, _, _, cos, sin, pos = args
    rows = [
        (torch.arange(1153, device=q.device) * 17 + i) % len(kv)
        for i in range(len(rows))
    ]
    ids, ptr = csr(rows, torch.int32, -1)
    ids = ids.long() * 1056 + 9
    args = (q, pool, ids, ptr, cos, sin, pos)
    kw.update(
        q_norm=True, norm_eps=0.125, q_is_rotated=False, fp8_qk=True, attn_sink=sink
    )
    out = _call_legacy(*args, **kw)
    buffers = dict(zip(("out", "out_scale", "max_logits", "lse"), out))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _call_legacy(*args, **kw, **buffers)
    # Make the norm scale change after capture, including zero-valued heads.
    q.mul_(0.25)
    q[:, ::11] = 0
    graph.replay()
    normalized = (
        q.float()
        * torch.rsqrt(q.float().square().mean(-1, keepdim=True) + kw["norm_eps"])
    ).bfloat16()
    _, mx, lse = dense_fp8_qk_reference(normalized, kv, rows, cos, sin, pos, sink, True)
    torch.testing.assert_close(out[2], mx, rtol=1e-4, atol=3e-5)
    torch.testing.assert_close(out[3], lse, rtol=1e-4, atol=3e-5)


# Two-source FP8 views and independent CSR ranges.


def two_source_inputs(
    heads=64,
    layout="interleaved",
    mode="raw",
    wide=False,
    with_sink=True,
    valid_indices=False,
):
    torch.manual_seed(9821 + heads)
    tokens = 8 if valid_indices and wide else 6
    decoded = []
    views = []
    for source, count in enumerate((37, 79)):
        original = torch.randn(count, 512, device="cuda", dtype=torch.bfloat16) * 0.25
        values, scales = quantize_fp8(original)
        if layout == "interleaved":
            cache = pack_rows(values, scales)
            data, sf = cache[:, :512].view(torch.float8_e4m3fn), cache[:, 512:]
        elif layout == "separate":
            data, sf = values, scales
        else:
            data_storage = torch.full(
                (count, 548 + source * 16), 255, device="cuda", dtype=torch.uint8
            )
            scale_storage = torch.full(
                (count, 36 + source * 8), 255, device="cuda", dtype=torch.uint8
            )
            data = data_storage[:, 4:516].view(torch.float8_e4m3fn)
            sf = scale_storage[:, 4:20]
            data.copy_(values)
            sf.copy_(scales)
        decoded.append(
            (
                values.float()
                * torch.exp2(scales.float() - 127).repeat_interleave(32, -1)
            ).bfloat16()
        )
        views.append((data, sf.view(torch.float8_e8m0fnu)))
    counts = (
        ([0, 1, 31, 33, 65, 513], [0, 2, 33, 65, 97, 640])
        if wide
        else ([0, 1, 7, 31, 32, 65], [0, 0, 3, 9, 33, 47])
    )
    if valid_indices:
        # Exercise empty sources and exact/partial tile boundaries with valid CSR rows.
        counts = (
            ([0, 1, 0, 31, 64, 65, 513, 128], [0, 2, 2, 33, 64, 97, 640, 0])
            if wide
            else ([0, 1, 31, 64, 128, 129], [1, 0, 33, 64, 128, 65])
        )
    source_rows, indices, pointers = [], [], []
    for source, sizes in enumerate(counts):
        rows = []
        for n in sizes:
            ids = torch.randint(
                len(decoded[source]), (n,), device="cuda", dtype=torch.int32
            )
            if not valid_indices:
                ids[::11] = -1
                ids[1::17] = len(decoded[source]) + 101
            rows.append(ids)
        ids, ptr = csr(rows, torch.int32, 2**31 - 1)
        if wide and source == 0:
            padded = torch.full(
                (tokens * 1200,), 2**31 - 1, device="cuda", dtype=torch.int32
            )
            padded[: ids.numel()] = ids
            ids = padded
        source_rows.append(rows)
        indices.append(ids)
        pointers.append(ptr)
    q = torch.randn(tokens, heads, 512, device="cuda", dtype=torch.bfloat16) * 0.125
    angles = torch.randn(tokens, 32, device="cuda")
    cos, sin = angles.cos().bfloat16(), angles.sin().bfloat16()
    pos = torch.arange(tokens, device="cuda", dtype=torch.int64)
    sink = torch.linspace(-3, 3, heads, device="cuda")
    sink[::11] = -torch.inf
    if not with_sink:
        sink.fill_(-torch.inf)
    reference_q = q
    if mode == "norm":
        reference_q = (
            q.float() * torch.rsqrt(q.float().square().mean(-1, keepdim=True) + 0.125)
        ).bfloat16()
    q_scale = None
    if mode == "rotated":
        q = rotate(q, cos, sin, pos).bfloat16()
        reference_q = q
    elif mode == "prepared":
        q8, qs = quantize_fp8(rotate(q, cos, sin, pos).bfloat16())
        q, q_scale = q8.reshape_as(q), qs.reshape(tokens, heads, 16).view(
            torch.float8_e8m0fnu
        )
    rows = []
    for t in range(tokens):
        prefix = source_rows[0][t]
        extend = source_rows[1][t]
        rows.append(
            torch.cat(
                (
                    prefix[(prefix >= 0) & (prefix < 37)],
                    extend[(extend >= 0) & (extend < 79)] + 37,
                )
            )
        )
    kwargs = {
        "q": q,
        "q_scale": q_scale,
        "unified_kv": views[0][0],
        "unified_kv_scale": views[0][1],
        "kv_indices_prefix": indices[0],
        "kv_indptr_prefix": pointers[0],
        "kv": views[1][0],
        "kv_scale": views[1][1],
        "kv_indices_extend": indices[1],
        "kv_indptr_extend": pointers[1],
        "cos": cos,
        "sin": sin,
        "positions": pos,
        "attn_sink": sink if with_sink else None,
        "q_is_rotated": mode in ("prepared", "rotated"),
        "q_norm": mode == "norm",
        "norm_eps": 0.125,
    }
    expected = dense_fp8_qk_reference(
        reference_q, torch.cat(decoded), rows, cos, sin, pos, sink, mode != "rotated"
    )
    return kwargs, expected


def check_two_source_dense(heads, layout, mode, wide, with_sink, valid_indices):
    kwargs, (expected, mx, lse) = two_source_inputs(
        heads, layout, mode, wide, with_sink, valid_indices
    )
    debug = torch.empty_like(kwargs["q"], dtype=torch.float32)
    out = _call_v41(
        **kwargs,
        debug_out=debug,
        max_logits=torch.empty_like(mx),
        lse=torch.empty_like(lse),
    )
    torch.testing.assert_close(out[2], mx, rtol=1e-4, atol=3e-5)
    torch.testing.assert_close(out[3], lse, rtol=1e-4, atol=3e-5)
    torch.testing.assert_close(debug, expected, rtol=0.02, atol=0.008)
    assert float((debug - expected).norm() / expected.norm()) < 0.005
    assert float((dequantize(*out[:2]) - expected).norm() / expected.norm()) < 0.035
    if not valid_indices or wide:
        assert torch.count_nonzero(debug[0]) == 0


def check_two_source_empty_and_bounds(heads, empty, large_grid):
    tokens = 0 if empty == "tokens" else 512 if large_grid else 3
    q = torch.zeros(tokens, heads, 512, device="cuda", dtype=torch.bfloat16)
    counts = (
        0 if empty in ("prefix", "both") else 2,
        0 if empty in ("extend", "both") else 7,
    )
    sources = []
    for rows, value in zip(counts, (2.0, 1.0)):
        sources.append(
            quantize_fp8(
                torch.full((rows, 512), value, device="cuda", dtype=torch.bfloat16)
            )
        )
    # A valid last extend row lies outside the prefix's row count. Negative and
    # oversized ids must be masked independently for both sources.
    prefix = torch.tensor([1, -1, 100] * tokens, device="cuda", dtype=torch.int32)
    extend = torch.tensor([6, -1, 100] * tokens, device="cuda", dtype=torch.int32)
    ptr = torch.arange(tokens + 1, device="cuda", dtype=torch.int32) * 3
    if heads == 128:
        prefix = torch.cat(
            (
                prefix,
                torch.full(
                    (max(tokens, 1) * 1200,),
                    2**31 - 1,
                    device="cuda",
                    dtype=torch.int32,
                ),
            )
        )
    kwargs = {
        "q": q,
        "q_scale": None,
        "unified_kv": sources[0][0],
        "unified_kv_scale": sources[0][1],
        "kv_indices_prefix": prefix,
        "kv_indptr_prefix": ptr,
        "cos": torch.ones(1, 32, device="cuda", dtype=torch.bfloat16),
        "sin": torch.zeros(1, 32, device="cuda", dtype=torch.bfloat16),
        "positions": torch.zeros(tokens, device="cuda", dtype=torch.int32),
    }
    if empty != "omitted":
        kwargs.update(
            kv=sources[1][0],
            kv_scale=sources[1][1],
            kv_indices_extend=extend,
            kv_indptr_extend=ptr,
        )
    debug = torch.empty_like(q, dtype=torch.float32)
    out = _call_v41(
        **kwargs, debug_out=debug, lse=torch.empty(tokens, heads, device="cuda")
    )
    live = int(counts[0] > 0) + int(counts[1] > 0 and empty != "omitted")
    expected = (
        2 * int(counts[0] > 0) + int(counts[1] > 0 and empty != "omitted")
    ) / max(live, 1)
    torch.testing.assert_close(debug, torch.full_like(debug, expected), rtol=0, atol=0)
    if live:
        torch.testing.assert_close(
            out[3],
            torch.full_like(out[3], float(torch.log(torch.tensor(float(live))))),
            rtol=1e-5,
            atol=1e-6,
        )
    else:
        assert torch.isposinf(out[3]).all()
    assert out[2] is None


def check_two_source_rescale(heads, mixed_scales):
    kwargs, _ = two_source_inputs(heads, "padded", "prepared", wide=True)
    scales = kwargs["q_scale"].view(torch.uint8)
    if mixed_scales:
        scales.random_(118, 131)
    else:
        scales.fill_(127)
    query = kwargs["q"].float() * torch.exp2(scales.float() - 127).repeat_interleave(
        32, -1
    )
    debug = torch.empty_like(query)
    stats = torch.empty(query.shape[:2], device="cuda")
    out = _call_v41(
        **kwargs, debug_out=debug, max_logits=torch.empty_like(stats), lse=stats
    )
    for t in range(len(query)):
        selected = []
        for data, scale, ids, ptr in [
            (
                kwargs["unified_kv"],
                kwargs["unified_kv_scale"],
                kwargs["kv_indices_prefix"],
                kwargs["kv_indptr_prefix"],
            ),
            (
                kwargs["kv"],
                kwargs["kv_scale"],
                kwargs["kv_indices_extend"],
                kwargs["kv_indptr_extend"],
            ),
        ]:
            rows = ids[int(ptr[t]) : int(ptr[t + 1])].long()
            rows = rows[(rows >= 0) & (rows < len(data))]
            decoded = data.float() * torch.exp2(
                scale.view(torch.uint8).float() - 127
            ).repeat_interleave(32, -1)
            selected.append(decoded[rows])
        values = torch.cat(selected)
        if not len(values):
            assert torch.count_nonzero(debug[t]) == 0
            continue
        logits = query[t] @ values.T * 512**-0.5
        probs = torch.cat((logits, kwargs["attn_sink"][:, None]), -1).softmax(-1)[
            :, : len(values)
        ]
        expected = rotate(
            (probs @ values).bfloat16()[None],
            kwargs["cos"],
            kwargs["sin"],
            kwargs["positions"][t : t + 1],
            inverse=True,
        )[0]
        torch.testing.assert_close(
            out[2][t], logits.max(-1).values, rtol=1e-4, atol=3e-5
        )
        torch.testing.assert_close(
            out[3][t], logits.logsumexp(-1), rtol=1e-4, atol=3e-5
        )
        assert float((debug[t] - expected).norm() / expected.norm()) < 0.005


def check_two_source_graph_replay(heads, mode, valid_indices):
    input_mode = "prepared" if mode.startswith("prepared") else mode
    kwargs, _ = two_source_inputs(
        heads, "padded", input_mode, wide=True, valid_indices=valid_indices
    )
    out = _call_v41(**kwargs)
    assert out[2:] == (None, None)
    buffers = {"out": out[0], "out_scale": out[1]}
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            _call_v41(**kwargs, **buffers)
        # Change the second source after capture, including its scales and CSR.
        kwargs["kv"].view(torch.uint8).zero_()
        kwargs["kv_scale"].view(torch.uint8).fill_(127)
        kwargs["kv_indices_extend"].fill_(-1)
        if mode == "prepared":
            kwargs["q_scale"].view(torch.uint8).fill_(127)
        elif mode == "prepared_mixed":
            kwargs["q_scale"].view(torch.uint8).random_(118, 131)
        else:
            kwargs["q"].mul_(0.5)
        expected = _call_v41(**kwargs)
        for _ in range(3):
            out[0].view(torch.uint8).fill_(255)
            out[1].view(torch.uint8).fill_(255)
            graph.replay()
            for actual, reference in zip(out[:2], expected[:2]):
                torch.testing.assert_close(
                    actual.view(torch.uint8),
                    reference.view(torch.uint8),
                    rtol=0,
                    atol=0,
                )
    torch.cuda.current_stream().wait_stream(stream)


def check_two_source_contract(bad):
    kwargs, _ = two_source_inputs(mode="prepared")
    if bad == "prefix_dtype":
        kwargs["unified_kv"] = kwargs["unified_kv"].bfloat16()
    elif bad == "scale_dtype":
        kwargs["kv_scale"] = kwargs["kv_scale"].float()
    elif bad == "prefix_rows":
        kwargs["unified_kv_scale"] = kwargs["unified_kv_scale"][:-1]
    elif bad == "extend_rows":
        kwargs["kv_scale"] = kwargs["kv_scale"][:-1]
    elif bad == "prefix_indices":
        kwargs["kv_indices_prefix"] = kwargs["kv_indices_prefix"].long()
    elif bad == "extend_indices":
        kwargs["kv_indices_extend"] = kwargs["kv_indices_extend"].long()
    elif bad == "prefix_ptr":
        kwargs["kv_indptr_prefix"] = kwargs["kv_indptr_prefix"].long()
    elif bad == "missing_scale":
        kwargs["kv_scale"] = None
    elif bad == "missing_ptr":
        kwargs["kv_indptr_extend"] = None
    elif bad == "channel_stride":
        kwargs["unified_kv"] = torch.empty(
            37, 1024, device="cuda", dtype=torch.float8_e4m3fn
        )[:, ::2]
    elif bad == "row_stride":
        kwargs["unified_kv"] = torch.empty(
            37, 529, device="cuda", dtype=torch.float8_e4m3fn
        )[:, :512]
    elif bad == "unaligned":
        kwargs["unified_kv"] = torch.empty(
            37, 516, device="cuda", dtype=torch.float8_e4m3fn
        )[:, 1:513]
    elif bad == "q_scale_dtype":
        kwargs["q_scale"] = kwargs["q_scale"].float()
    elif bad == "prepared_rotation":
        kwargs["q_is_rotated"] = False
    elif bad == "prepared_norm":
        kwargs["q_norm"] = True
    elif bad == "out_alias":
        kwargs["out"] = kwargs["q"].view(6, 8, 4096)
    elif bad == "stat_alias":
        stats = torch.empty(6, 64, device="cuda")
        kwargs.update(lse=stats, max_logits=stats)
    elif bad == "norm_eps":
        kwargs["norm_eps"] = 0
    elif bad == "backend":
        kwargs["backend"] = "invalid"
    with _raises((ValueError, RuntimeError)):
        _call_v41(**kwargs)


def check_two_source_h64_causal_prefill(
    tokens, topk, mode, heads=64, splits=1, sliding_window=False
):
    # Long causal CSR rows exercise the 64-row pipeline, including its source
    # switch and final partial tile. Check all heads against a dense oracle.
    torch.manual_seed(23917 + topk)
    prefix_rows = 65536
    values, scales = quantize_fp8(
        torch.randn(prefix_rows + tokens, 512, device="cuda", dtype=torch.bfloat16)
        * 0.25
    )
    packed = pack_rows(values, scales)
    prefix, extend = packed[:prefix_rows], packed[prefix_rows:]
    pids = torch.randint(prefix_rows, (tokens, topk), device="cuda", dtype=torch.int32)
    pp = torch.arange(tokens + 1, device="cuda", dtype=torch.int32) * topk
    lengths = torch.arange(1, tokens + 1, device="cuda", dtype=torch.int32).clamp_max(
        topk
    )
    ep = torch.cat(
        (torch.zeros(1, device="cuda", dtype=torch.int32), lengths.cumsum(0).int())
    )
    columns = torch.arange(topk, device="cuda", dtype=torch.int32).expand(tokens, -1)
    eids = columns[columns < lengths[:, None]]
    if sliding_window:
        prefix_ids, extend_ids = [], []
        for j in range(tokens):
            old_swa = max(127 - j, 0)
            prefix_ids.append(
                torch.cat(
                    (
                        pids[j],
                        torch.arange(
                            prefix_rows - old_swa,
                            prefix_rows,
                            device="cuda",
                            dtype=torch.int32,
                        ),
                    )
                )
            )
            extend_ids.append(
                torch.arange(max(0, j - 127), j + 1, device="cuda", dtype=torch.int32)
            )
        pids, pp = csr(prefix_ids, torch.int32, 2**31 - 1)
        eids, ep = csr(extend_ids, torch.int32, 2**31 - 1)
    q = torch.randn(tokens, heads, 512, device="cuda", dtype=torch.bfloat16) * 0.125
    angles = torch.randn(tokens, 32, device="cuda")
    cos, sin = angles.cos().bfloat16(), angles.sin().bfloat16()
    pos = torch.arange(tokens, device="cuda", dtype=torch.int32)
    sink = torch.linspace(-3, 3, heads, device="cuda")
    sink[::11] = -torch.inf
    reference_q = q
    if mode == "norm":
        reference_q = (
            q.float() * torch.rsqrt(q.float().square().mean(-1, keepdim=True) + 0.125)
        ).bfloat16()
    query, qscale = q, None
    if mode == "prepared":
        query, qscale = quantize_fp8(rotate(q, cos, sin, pos).bfloat16())
        query = query.reshape_as(q)
        qscale = qscale.reshape(tokens, heads, 16).view(torch.float8_e8m0fnu)
    debug = torch.empty_like(q, dtype=torch.float32)
    workspace = None
    if splits > 1:
        count = tokens * splits * heads * 516
        storage = torch.full((count + 8,), 0x5A5A5A5A, device="cuda", dtype=torch.int32)
        workspace = (
            storage[4:-4].view(torch.float32).reshape(tokens, splits, heads, 516)
        )
        workspace.fill_(torch.nan)
    out = _call_v41(
        q=query,
        q_scale=qscale,
        unified_kv=prefix[:, :512].view(torch.float8_e4m3fn),
        unified_kv_scale=prefix[:, 512:].view(torch.float8_e8m0fnu),
        kv_indices_prefix=pids.flatten(),
        kv_indptr_prefix=pp,
        kv=extend[:, :512].view(torch.float8_e4m3fn),
        kv_scale=extend[:, 512:].view(torch.float8_e8m0fnu),
        kv_indices_extend=eids,
        kv_indptr_extend=ep,
        cos=cos,
        sin=sin,
        positions=pos,
        attn_sink=sink,
        q_is_rotated=mode == "prepared",
        q_norm=mode == "norm",
        norm_eps=0.125,
        is_decode=splits > 1,
        num_splits=splits,
        workspace=workspace,
        debug_out=debug,
        max_logits=torch.empty(tokens, heads, device="cuda"),
        lse=torch.empty(tokens, heads, device="cuda"),
    )
    probes = sorted({0, tokens // 2 - 1, tokens // 2, tokens - 1})
    if sliding_window:
        probes = sorted(set(probes) | {126, 127, 128})
    pids = pids.flatten()
    rows = [
        torch.cat(
            (
                pids[int(pp[t]) : int(pp[t + 1])],
                eids[int(ep[t]) : int(ep[t + 1])] + prefix_rows,
            )
        )
        for t in probes
    ]
    decoded = (
        values.float() * torch.exp2(scales.float() - 127).repeat_interleave(32, -1)
    ).bfloat16()
    expected, mx, lse = dense_fp8_qk_reference(
        reference_q[probes], decoded, rows, cos, sin, pos[probes], sink, True
    )
    torch.testing.assert_close(out[2][probes], mx, rtol=1e-4, atol=3e-5)
    torch.testing.assert_close(out[3][probes], lse, rtol=1e-4, atol=3e-5)
    torch.testing.assert_close(debug[probes], expected, rtol=0.02, atol=0.008)
    assert float((debug[probes] - expected).norm() / expected.norm()) < 0.005
    assert (
        float((dequantize(*out[:2])[probes] - expected).norm() / expected.norm())
        < 0.035
    )

    if workspace is not None:
        assert bool((storage[:4] == 0x5A5A5A5A).all())
        assert bool((storage[-4:] == 0x5A5A5A5A).all())


def check_two_source_h64_q_scale_groups(rotated, valid_indices, extend):
    # Different powers of two expose scale-byte and Q-fragment permutations
    # that nearly uniform random magnitudes can hide.
    kwargs, _ = two_source_inputs(wide=True, valid_indices=valid_indices)
    q = kwargs["q"]
    groups = torch.arange(16, device=q.device)
    heads = torch.arange(64, device=q.device)
    factors = torch.exp2(((groups[None, :] * 3 + heads[:, None]) % 13 - 6).float())
    q.mul_(factors.repeat_interleave(32, -1).bfloat16())
    prepared_q = rotate(q, kwargs["cos"], kwargs["sin"], kwargs["positions"]).bfloat16()
    if rotated:
        kwargs.update(q=prepared_q, q_is_rotated=True)
    if not extend:
        kwargs.update(
            kv=None, kv_scale=None, kv_indices_extend=None, kv_indptr_extend=None
        )
    kwargs.update(
        max_logits=torch.empty(q.shape[:2], device=q.device),
        lse=torch.empty(q.shape[:2], device=q.device),
    )
    actual = _call_v41(**kwargs)
    q8, scales = quantize_fp8(prepared_q)
    kwargs.update(
        q=q8.reshape_as(q),
        q_scale=scales.reshape(*q.shape[:2], 16).view(torch.float8_e8m0fnu),
        q_is_rotated=True,
        max_logits=torch.empty_like(actual[2]),
        lse=torch.empty_like(actual[3]),
    )
    expected = _call_v41(**kwargs)
    for x, y in zip(actual, expected):
        torch.testing.assert_close(
            x.view(torch.uint8), y.view(torch.uint8), rtol=0, atol=0
        )


def check_two_source_rope_storage_offset(heads, mode):
    kwargs, (_, mx, lse) = two_source_inputs(heads, "padded", mode, True, True, True)
    reference = _call_v41(
        **kwargs, max_logits=torch.empty_like(mx), lse=torch.empty_like(lse)
    )
    # Contiguous BF16 views can start two or six bytes past an aligned allocation.
    for name, offset in (("cos", 1), ("sin", 3)):
        original = kwargs[name]
        storage = torch.empty(
            original.numel() + offset, device=original.device, dtype=original.dtype
        )
        view = storage[offset:].view_as(original)
        view.copy_(original)
        assert view.is_contiguous() and view.data_ptr() % 8 != 0
        kwargs[name] = view
    actual = _call_v41(
        **kwargs, max_logits=torch.empty_like(mx), lse=torch.empty_like(lse)
    )
    for out, expected in zip(actual, reference):
        torch.testing.assert_close(
            out.view(torch.uint8), expected.view(torch.uint8), rtol=0, atol=0
        )


def check_two_source_scale_output_storage_offset(heads, mode, offset):
    kwargs, (_, mx, lse) = two_source_inputs(heads, "padded", mode, True, True, True)
    reference = _call_v41(
        **kwargs, max_logits=torch.empty_like(mx), lse=torch.empty_like(lse)
    )
    storage = torch.empty(
        reference[1].numel() + offset, device="cuda", dtype=torch.uint8
    )
    scales = storage[offset:].view(reference[1].dtype).view_as(reference[1])
    assert scales.is_contiguous() and scales.data_ptr() % 4 != 0
    actual = _call_v41(
        **kwargs,
        out_scale=scales,
        max_logits=torch.empty_like(mx),
        lse=torch.empty_like(lse),
    )
    assert actual[1].data_ptr() == scales.data_ptr()
    for out, expected in zip(actual, reference):
        torch.testing.assert_close(
            out.view(torch.uint8), expected.view(torch.uint8), rtol=0, atol=0
        )


def check_two_source_index_storage_offset(heads, mode, offset, valid_indices):
    kwargs, (_, mx, lse) = two_source_inputs(
        heads, "padded", mode, True, True, valid_indices
    )
    reference = _call_v41(
        **kwargs, max_logits=torch.empty_like(mx), lse=torch.empty_like(lse)
    )
    for name in ("kv_indices_prefix", "kv_indices_extend"):
        original = kwargs[name]
        storage = torch.empty(
            original.numel() + offset, device=original.device, dtype=original.dtype
        )
        view = storage[offset:].view_as(original)
        view.copy_(original)
        assert view.is_contiguous() and view.data_ptr() % 16 != 0
        kwargs[name] = view
    actual = _call_v41(
        **kwargs, max_logits=torch.empty_like(mx), lse=torch.empty_like(lse)
    )
    for out, expected in zip(actual, reference):
        torch.testing.assert_close(
            out.view(torch.uint8), expected.view(torch.uint8), rtol=0, atol=0
        )


def check_fp8_decode_dense(heads, layout, mode, splits, valid_indices, with_sink):
    kwargs, (expected, mx, lse) = two_source_inputs(
        heads, layout, mode, True, with_sink, valid_indices
    )
    debug = torch.empty_like(kwargs["q"], dtype=torch.float32)
    out = _call_v41(
        **kwargs,
        is_decode=True,
        num_splits=splits,
        debug_out=debug,
        max_logits=torch.empty_like(mx),
        lse=torch.empty_like(lse),
    )
    torch.testing.assert_close(out[2], mx, rtol=1e-4, atol=3e-5)
    torch.testing.assert_close(out[3], lse, rtol=1e-4, atol=3e-5)
    torch.testing.assert_close(debug, expected, rtol=0.02, atol=0.008)
    assert float((debug - expected).norm() / expected.norm()) < 0.005
    assert float((dequantize(*out[:2]) - expected).norm() / expected.norm()) < 0.035


def check_fp8_decode_ring_view(heads, splits, sources):
    q = torch.zeros(1, heads, 512, device="cuda", dtype=torch.float8_e4m3fn)
    qs = torch.full((1, heads, 16), 127, device="cuda", dtype=torch.uint8)
    pv, ps = quantize_fp8(torch.full((97, 512), 2.0, device="cuda"))
    pool = pack_rows(pv, ps)
    ring_values = (1 + torch.arange(71, device="cuda") % 8).float()
    ev, es = quantize_fp8(ring_values[:, None].expand(71, 512))
    ring = pack_rows(ev, es)[7:]
    pi = torch.tensor([2, 90], device="cuda", dtype=torch.int32)
    ei = torch.arange(59, 99, device="cuda", dtype=torch.int32) % 64
    pn = pi.numel() if sources in ("both", "prefix") else 0
    en = ei.numel() if sources in ("both", "extend") else 0
    pp = torch.tensor([0, pn], device="cuda", dtype=torch.int32)
    ep = torch.tensor([0, en], device="cuda", dtype=torch.int32)
    debug = torch.empty(1, heads, 512, device="cuda")
    mx, lse = torch.empty(1, heads, device="cuda"), torch.empty(1, heads, device="cuda")
    kwargs = {
        "q": q,
        "q_scale": qs,
        "unified_kv": pool[:, :512].view(torch.float8_e4m3fn),
        "unified_kv_scale": pool[:, 512:],
        "kv_indices_prefix": pi,
        "kv_indptr_prefix": pp,
        "kv": ring[:, :512].view(torch.float8_e4m3fn),
        "kv_scale": ring[:, 512:],
        "kv_indices_extend": ei,
        "kv_indptr_extend": ep,
        "cos": torch.ones(1, 32, device="cuda", dtype=torch.bfloat16),
        "sin": torch.zeros(1, 32, device="cuda", dtype=torch.bfloat16),
        "positions": torch.zeros(1, device="cuda", dtype=torch.int32),
        "is_decode": True,
        "num_splits": splits,
        "debug_out": debug,
        "max_logits": mx,
        "lse": lse,
    }
    _call_v41(**kwargs)
    expected = (2 * pn + ring_values[7:][ei[:en].long()].sum()) / max(1, pn + en)
    torch.testing.assert_close(
        debug, torch.full_like(debug, expected.bfloat16().float()), rtol=0, atol=0
    )
    torch.testing.assert_close(
        mx, torch.full_like(mx, 0 if pn + en else -torch.inf), rtol=0, atol=0
    )
    expected_lse = (
        torch.log(torch.tensor(float(pn + en), device="cuda")) if pn + en else torch.inf
    )
    torch.testing.assert_close(
        lse, torch.full_like(lse, expected_lse), rtol=1e-4, atol=3e-5
    )


def check_fp8_decode_graph_replay(heads, mode):
    kwargs, (_, mx, lse) = two_source_inputs(heads, "padded", mode, True, True, True)
    t = kwargs["q"].shape[0]
    kwargs.update(
        is_decode=True,
        num_splits=4,
        workspace=torch.empty(t, 4, heads, 516, device="cuda"),
        out=torch.empty(t, heads // 8, 4096, device="cuda", dtype=torch.float8_e4m3fn),
        out_scale=torch.empty(
            t, heads // 8, 128, device="cuda", dtype=torch.float8_e8m0fnu
        ),
        max_logits=torch.empty_like(mx),
        lse=torch.empty_like(lse),
    )
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        _call_v41(**kwargs)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            _call_v41(**kwargs)
        for empty in [False, True, False]:
            kwargs["kv_indices_prefix"].fill_(-1 if empty else 4)
            kwargs["kv_indices_extend"].fill_(-1 if empty else 8)
            kwargs["kv_indptr_prefix"][1:-1].zero_()
            kwargs["kv_indptr_extend"][1:-1].zero_()
            expected = tuple(x.clone() for x in _call_v41(**kwargs))
            kwargs["workspace"].fill_(torch.nan)
            graph.replay()
            for actual, ref in zip(
                (
                    kwargs["out"],
                    kwargs["out_scale"],
                    kwargs["max_logits"],
                    kwargs["lse"],
                ),
                expected,
            ):
                torch.testing.assert_close(
                    actual.view(torch.uint8), ref.view(torch.uint8), rtol=0, atol=0
                )
    torch.cuda.current_stream().wait_stream(stream)


def check_fp8_decode_contract(bad):
    kwargs, _ = two_source_inputs(mode="prepared")
    t, h, _ = kwargs["q"].shape
    kwargs.update(is_decode=True, num_splits=2)
    if bad == "phase":
        kwargs["is_decode"] = False
    elif bad == "prefill_workspace":
        kwargs.update(
            is_decode=False,
            num_splits=0,
            workspace=torch.empty(t, 2, h, 516, device="cuda"),
        )
    elif bad == "count":
        kwargs["num_splits"] = 33
    elif bad == "fraction":
        kwargs["num_splits"] = 1.5
    elif bad == "workspace_shape":
        kwargs["workspace"] = torch.empty(t, 2, h, 512, device="cuda")
    elif bad == "workspace_dtype":
        kwargs["workspace"] = torch.empty(
            t, 2, h, 516, device="cuda", dtype=torch.bfloat16
        )
    elif bad == "workspace_overlap":
        kwargs["workspace"] = torch.empty(t, 2, h, 516, device="cuda")
        kwargs["debug_out"] = (
            kwargs["workspace"].flatten()[: t * h * 512].view(t, h, 512)
        )
    else:
        kwargs["num_splits"] = 1
        kwargs["workspace"] = torch.empty(t, 1, h, 516, device="cuda")
    with _raises((ValueError, RuntimeError)):
        _call_v41(**kwargs)


def check_fp8_prefill_auto_split(heads, mode):
    kwargs, (expected, mx, lse) = two_source_inputs(heads, "padded", mode, True)
    kwargs.update(
        debug_out=torch.empty_like(expected),
        max_logits=torch.empty_like(mx),
        lse=torch.empty_like(lse),
    )
    with patch(
        "aiter.ops.fused_norm_rope_attn_rope_cast._fused_norm_rope_attn_rope_cast_fwd",
        wraps=fused_v41_fwd,
    ) as native:
        out = _call_v41(**kwargs)
        splits, workspace = native.call_args.args[23:25]
        assert 1 < splits <= 32
        assert workspace.shape == (expected.shape[0], splits, heads, 516)
        torch.testing.assert_close(kwargs["debug_out"], expected, rtol=0.02, atol=0.008)
        torch.testing.assert_close(out[2], mx, rtol=1e-4, atol=3e-5)
        torch.testing.assert_close(out[3], lse, rtol=1e-4, atol=3e-5)
        out = _call_v41(**kwargs, num_splits=1)
        assert native.call_args.args[23] == 1
        assert native.call_args.args[24] is None
        torch.testing.assert_close(kwargs["debug_out"], expected, rtol=0.02, atol=0.008)
        torch.testing.assert_close(out[2], mx, rtol=1e-4, atol=3e-5)
        torch.testing.assert_close(out[3], lse, rtol=1e-4, atol=3e-5)


def check_fp8_decode_auto_and_poisoned_workspace(heads, mode, valid_indices, splits):
    kwargs, (expected, mx, lse) = two_source_inputs(
        heads, "padded", mode, True, False, valid_indices
    )
    debug = torch.empty_like(expected)
    # Automatic dispatch must handle spare index-buffer capacity and ragged rows.
    out = _call_v41(
        **kwargs,
        is_decode=True,
        debug_out=debug,
        max_logits=torch.empty_like(mx),
        lse=torch.empty_like(lse),
    )
    torch.testing.assert_close(debug, expected, rtol=0.02, atol=0.008)
    torch.testing.assert_close(out[2], mx, rtol=1e-4, atol=3e-5)
    torch.testing.assert_close(out[3], lse, rtol=1e-4, atol=3e-5)
    # More splits than live tiles must overwrite every partial, including empties.
    t = kwargs["q"].shape[0]
    workspace = torch.full((t, splits, heads, 516), torch.nan, device="cuda")
    for ids in (kwargs["kv_indices_prefix"], kwargs["kv_indices_extend"]):
        ids.fill_(-1)
    for sink in (
        None,
        torch.full((heads,), -torch.inf, device="cuda"),
        torch.full((heads,), 100.0, device="cuda"),
    ):
        kwargs["attn_sink"] = sink
        workspace.fill_(torch.nan)
        out = _call_v41(
            **kwargs,
            is_decode=True,
            num_splits=splits,
            workspace=workspace,
            debug_out=debug,
            max_logits=torch.empty_like(mx),
            lse=torch.empty_like(lse),
        )
        assert torch.count_nonzero(debug) == 0
        assert torch.count_nonzero(dequantize(*out[:2])) == 0
        assert torch.isneginf(out[2]).all()
        assert torch.isposinf(out[3]).all()


def check_fp8_decode_large_grid_replay(heads, mode, valid_indices, layout, splits):
    kwargs, (expected, mx, lse) = two_source_inputs(
        heads, layout, mode, True, True, valid_indices
    )
    # The single path must exercise the 32-head tile at both H64 and H128.
    original_tokens = kwargs["q"].shape[0]
    target_tokens = 128 // (heads // 64) if splits == 1 else original_tokens * 22
    repeats = (target_tokens + original_tokens - 1) // original_tokens
    for key in ("q", "q_scale"):
        if kwargs[key] is not None:
            x = kwargs[key]
            kwargs[key] = (
                x.view(torch.uint8).repeat(repeats, 1, 1)[:target_tokens].view(x.dtype)
            )
    kwargs["positions"] = kwargs["positions"].repeat(repeats)[:target_tokens]
    for source in ("prefix", "extend"):
        pk, ik = "kv_indptr_" + source, "kv_indices_" + source
        ptr, ids = kwargs[pk], kwargs[ik]
        count = int(ptr[-1])
        kwargs[ik] = ids[:count].repeat(repeats)
        kwargs[pk] = torch.cat(
            [ptr[:-1] + i * count for i in range(repeats)] + [ptr[-1:] * repeats]
        )
        kwargs[pk] = kwargs[pk][: target_tokens + 1].contiguous()
        kwargs[ik] = kwargs[ik][: int(kwargs[pk][-1])].contiguous()
    expected = expected.repeat(repeats, 1, 1)[:target_tokens]
    mx = mx.repeat(repeats, 1)[:target_tokens]
    lse = lse.repeat(repeats, 1)[:target_tokens]
    t = kwargs["q"].shape[0]
    kwargs.update(
        is_decode=True,
        num_splits=splits,
        workspace=(
            torch.full((t, splits, heads, 516), torch.nan, device="cuda")
            if splits > 1
            else None
        ),
        debug_out=torch.empty_like(expected),
        max_logits=torch.empty_like(mx),
        lse=torch.empty_like(lse),
    )
    out = _call_v41(**kwargs)
    kwargs["out"], kwargs["out_scale"] = out[:2]
    torch.testing.assert_close(kwargs["debug_out"], expected, rtol=0.02, atol=0.008)
    torch.testing.assert_close(out[2], mx, rtol=1e-4, atol=3e-5)
    torch.testing.assert_close(out[3], lse, rtol=1e-4, atol=3e-5)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _call_v41(**kwargs)
    for empty in [False, True, False]:
        if empty:
            kwargs["kv_indices_prefix"].fill_(-1)
            kwargs["kv_indices_extend"].fill_(-1)
        elif torch.all(kwargs["kv_indices_prefix"] == -1):
            kwargs["kv_indices_prefix"].fill_(4)
            kwargs["kv_indices_extend"].fill_(8)
        if empty:
            kwargs["kv_indptr_prefix"][1:-1].zero_()
            kwargs["kv_indptr_extend"][1:-1].zero_()
        refs = tuple(x.clone() for x in _call_v41(**kwargs))
        for _ in range(4):
            if kwargs["workspace"] is not None:
                kwargs["workspace"].fill_(torch.nan)
            graph.replay()
            for actual, ref in zip(out, refs):
                torch.testing.assert_close(
                    actual.view(torch.uint8), ref.view(torch.uint8), rtol=0, atol=0
                )


def check_shared_kv_wave_boundaries(tokens, mode, valid_indices, is_decode, heads):
    _check_shared_kv_wave_boundary(tokens, mode, valid_indices, is_decode, heads)


def _check_shared_kv_wave_boundary(
    tokens,
    mode,
    valid_indices,
    is_decode,
    heads,
    capacity=1200,
    layout="padded",
    exercise_common=False,
):
    kwargs, (expected, mx, lse) = two_source_inputs(
        heads, layout, mode, wide=True, valid_indices=valid_indices
    )
    selected = (torch.arange(tokens, device="cuda") + 5) % kwargs["q"].shape[0]
    for key in ("q", "q_scale"):
        if kwargs[key] is not None:
            value = kwargs[key]
            kwargs[key] = (
                value.view(torch.uint8).index_select(0, selected).view(value.dtype)
            )
    kwargs["positions"] = kwargs["positions"].index_select(0, selected)
    if exercise_common:
        kwargs["positions"] = kwargs["positions"].int()
    for source in ("prefix", "extend"):
        ptr = kwargs[f"kv_indptr_{source}"].tolist()
        ids = kwargs[f"kv_indices_{source}"]
        rows = [ids[ptr[i] : ptr[i + 1]] for i in selected.tolist()]
        new_ids, new_ptr = csr(rows, torch.int32, 2**31 - 1)
        kwargs[f"kv_indices_{source}"] = new_ids
        kwargs[f"kv_indptr_{source}"] = new_ptr
    ids = kwargs["kv_indices_prefix"]
    total = tokens * capacity - kwargs["kv_indices_extend"].numel()
    assert total >= ids.numel()
    padded = torch.full((total,), 2**31 - 1, device="cuda", dtype=torch.int32)
    padded[: ids.numel()] = ids
    kwargs["kv_indices_prefix"] = padded
    expected, mx, lse = [x.index_select(0, selected) for x in (expected, mx, lse)]
    kwargs.update(
        is_decode=is_decode,
        debug_out=torch.empty_like(expected),
        max_logits=torch.empty_like(mx),
        lse=torch.empty_like(lse),
    )
    out = _call_v41(**kwargs)
    torch.testing.assert_close(out[2], mx, rtol=1e-4, atol=3e-5)
    torch.testing.assert_close(out[3], lse, rtol=1e-4, atol=3e-5)
    torch.testing.assert_close(kwargs["debug_out"], expected, rtol=0.02, atol=0.008)
    assert float((kwargs["debug_out"] - expected).norm() / expected.norm()) < 0.005
    assert float((dequantize(*out[:2]) - expected).norm() / expected.norm()) < 0.035
    # Replay must overwrite every head, including after invalid and empty rows.
    kwargs["out"], kwargs["out_scale"] = out[:2]
    outputs = (*out, kwargs["debug_out"])
    refs = tuple(x.clone() for x in outputs)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _call_v41(**kwargs)
    for _ in range(2):
        for x in outputs:
            x.view(torch.uint8).fill_(255)
        graph.replay()
        for x, ref in zip(outputs, refs):
            torch.testing.assert_close(
                x.view(torch.uint8), ref.view(torch.uint8), rtol=0, atol=0
            )

    if exercise_common:
        # With 32-bit positions, omitted diagnostic outputs must agree with the
        # general route. Toggle each host guard while retaining identical inputs.
        for variant in ("common", "positions64", "sinkless", "max", "debug", "no_lse"):
            call = dict(kwargs)
            call["max_logits"] = None
            call["debug_out"] = None
            if variant == "positions64":
                call["positions"] = call["positions"].long()
            elif variant == "sinkless":
                call["attn_sink"] = None
            elif variant == "max":
                call["max_logits"] = torch.empty_like(mx)
            elif variant == "debug":
                call["debug_out"] = torch.empty_like(expected)
            elif variant == "no_lse":
                call["lse"] = None
            general = dict(
                call,
                debug_out=torch.empty_like(expected),
                max_logits=torch.empty_like(mx),
                lse=torch.empty_like(lse),
            )
            reference = [x.clone() for x in _call_v41(**general)]
            actual = _call_v41(**call)
            for value, ref in zip(actual, reference):
                if value is not None:
                    torch.testing.assert_close(
                        value.view(torch.uint8), ref.view(torch.uint8), rtol=0, atol=0
                    )
            if call["debug_out"] is not None:
                torch.testing.assert_close(
                    call["debug_out"], general["debug_out"], rtol=0, atol=0
                )
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                _call_v41(**call)
            for value in actual:
                if value is not None:
                    value.view(torch.uint8).fill_(255)
            graph.replay()
            for value, ref in zip(actual, reference):
                if value is not None:
                    torch.testing.assert_close(
                        value.view(torch.uint8), ref.view(torch.uint8), rtol=0, atol=0
                    )


def check_h64_common_prefill_contract(tokens, mode, layout, valid_indices):
    _check_shared_kv_wave_boundary(
        tokens, mode, valid_indices, False, 64, layout=layout, exercise_common=True
    )


def check_mid_batch_attention_routes(tokens, heads, mode, layout):
    # Include empty/invalid rows and spare CSR capacity in both Q/output paths.
    _check_shared_kv_wave_boundary(
        tokens, mode, False, True, heads, 640, layout, exercise_common=True
    )


def check_mid_batch_attention_capacity(tokens, heads, capacity, mode):
    _check_shared_kv_wave_boundary(
        tokens, mode, True, True, heads, capacity, "interleaved", exercise_common=True
    )


def check_h64_large_prefill_capacity(tokens, capacity, mode, layout, valid_indices):
    # Spare CSR allocation must not turn padding into attended keys at the route boundary.
    _check_shared_kv_wave_boundary(
        tokens, mode, valid_indices, False, 64, capacity, layout, exercise_common=True
    )


def check_h128_large_shared_kv_capacity(tokens, capacity, mode, layout, valid_indices):
    # Check ragged rows and unused CSR capacity on both sides of the shared-KV threshold.
    _check_shared_kv_wave_boundary(
        tokens, mode, valid_indices, False, 128, capacity, layout, exercise_common=True
    )


def check_native_split_integer_width_and_error_recovery(bad_splits):
    kwargs, _ = two_source_inputs(128, "padded", "raw")
    tokens, heads, _ = kwargs["q"].shape
    outputs = _call_v41(
        **kwargs,
        max_logits=torch.empty((tokens, heads), device="cuda"),
        lse=torch.empty((tokens, heads), device="cuda"),
    )
    references = [x.clone() for x in outputs]
    native = dict(kwargs)
    native.update(
        out=outputs[0],
        out_scale=outputs[1],
        max_logits=outputs[2],
        lse=outputs[3],
        debug_out=None,
        q_norm=False,
        norm_eps=1e-6,
        softmax_scale=512**-0.5,
    )
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        # Both values would become a valid split count if the FFI narrowed to int32.
        with _raises(RuntimeError, match="num_splits must be in"):
            fused_v41_fwd(**native, num_splits=bad_splits)
        for output in outputs:
            output.view(torch.uint8).fill_(255)
        # Default split/workspace values and a fresh call must work after the error.
        fused_v41_fwd(**native)
    torch.cuda.current_stream().wait_stream(stream)
    for output, reference in zip(outputs, references):
        torch.testing.assert_close(
            output.view(torch.uint8), reference.view(torch.uint8), rtol=0, atol=0
        )


def check_h64_prefill_address_width(wide_view, rows, prepared, tokens):
    # Exercise unsigned offsets beyond 2 GiB and the fallback beyond 4 GiB.
    # All four independently strided views must participate in the dispatch guard.
    heads = 64
    row_stride = 2**30 + 4
    kwargs = {}
    for key in ("unified_kv", "unified_kv_scale", "kv", "kv_scale"):
        scale = key.endswith("scale")
        width = 16 if scale else 512
        if key == wide_view:
            storage = torch.empty(
                (rows - 1) * row_stride + width, device="cuda", dtype=torch.uint8
            )
            storage[:528] = 0
            view = storage.as_strided((rows, width), (row_stride, 1))
        else:
            view = torch.empty(rows, width, device="cuda", dtype=torch.uint8)
        view.fill_(127 if scale else 0x38)
        kwargs[key] = view.view(torch.float8_e8m0fnu if scale else torch.float8_e4m3fn)
    for source, count in (("prefix", 1024), ("extend", 128)):
        kwargs[f"kv_indices_{source}"] = torch.full(
            (tokens * count,), rows - 1, device="cuda", dtype=torch.int32
        )
        kwargs[f"kv_indptr_{source}"] = (
            torch.arange(tokens + 1, device="cuda", dtype=torch.int32) * count
        )
    q = torch.zeros(tokens, heads, 512, device="cuda", dtype=torch.bfloat16)
    kwargs.update(
        q=q.to(torch.float8_e4m3fn) if prepared else q,
        q_scale=(
            torch.full((tokens, heads, 16), 127, device="cuda", dtype=torch.uint8).view(
                torch.float8_e8m0fnu
            )
            if prepared
            else None
        ),
        q_is_rotated=True,
        cos=torch.ones(1, 32, device="cuda", dtype=torch.bfloat16),
        sin=torch.zeros(1, 32, device="cuda", dtype=torch.bfloat16),
        positions=torch.zeros(tokens, device="cuda", dtype=torch.int32),
        debug_out=torch.empty_like(q, dtype=torch.float32),
    )
    out = _call_v41(**kwargs)
    torch.testing.assert_close(
        kwargs["debug_out"], torch.ones_like(q, dtype=torch.float32), rtol=0, atol=0
    )
    torch.testing.assert_close(
        dequantize(*out[:2]), torch.ones_like(q, dtype=torch.float32), rtol=0, atol=0
    )


def check_q_and_debug_16byte_storage_offsets(api, heads, q_norm):
    if api == "two_source":
        kwargs, _ = two_source_inputs(
            heads, "padded", "norm" if q_norm else "raw", wide=True
        )
        q = kwargs["q"]
        kwargs.update(
            max_logits=torch.empty(q.shape[:2], device=q.device),
            lse=torch.empty(q.shape[:2], device=q.device),
        )
    else:
        args, kwargs, _, _, sink = make_fp8_qk_inputs(heads, True)
        q = args[0]
        kwargs.update(
            attn_sink=sink,
            fp8_qk=api == "legacy_fp8",
            q_is_rotated=False,
            q_norm=q_norm,
            norm_eps=0.125,
        )

    def run(query, debug, buffers=None):
        call = dict(kwargs, debug_out=debug)
        call.update(buffers or {})
        if api == "two_source":
            call["q"] = query
            return _call_v41(**call)
        return _call_legacy(query, *args[1:], **call)

    aligned_debug = torch.empty_like(q, dtype=torch.float32)
    expected = tuple(x.clone() for x in run(q, aligned_debug))
    expected_debug = aligned_debug.clone()
    for shifted in ("q", "debug", "both"):
        query = q
        debug = torch.empty_like(aligned_debug)
        guards = []
        for name, original, offset in (("q", q, 8), ("debug", aligned_debug, 4)):
            if shifted not in (name, "both"):
                continue
            storage = torch.full(
                (original.numel() + 2 * offset,),
                3.0,
                device=original.device,
                dtype=original.dtype,
            )
            view = storage[offset:-offset].view_as(original)
            assert view.is_contiguous() and view.data_ptr() % 32 == 16
            guards.append((storage, offset))
            if name == "q":
                query = view.copy_(q)
            else:
                debug = view
        actual = run(query, debug)
        buffers = dict(zip(("out", "out_scale", "max_logits", "lse"), actual))
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run(query, debug, buffers)
        for replay in (False, True):
            if replay:
                for x in (*actual, debug):
                    x.view(torch.uint8).fill_(255)
                graph.replay()
            for value, ref in zip((*actual, debug), (*expected, expected_debug)):
                torch.testing.assert_close(
                    value.view(torch.uint8), ref.view(torch.uint8), rtol=0, atol=0
                )
            for storage, offset in guards:
                assert torch.all(storage[:offset] == 3)
                assert torch.all(storage[-offset:] == 3)
            torch.testing.assert_close(query, q, rtol=0, atol=0)


def check_softmax_scale_base2_range_and_error_recovery(api, direct):
    args = simple_inputs(tokens=1, heads=64)
    q, pool, _ids, ptr, cos, sin, pos = args
    debug = torch.empty_like(q, dtype=torch.float32)
    if api == "two_source":
        kwargs = {
            "q": q,
            "q_scale": None,
            "unified_kv": pool[:512].view(torch.float8_e4m3fn).view(1, 512),
            "unified_kv_scale": pool[512:].view(torch.float8_e8m0fnu).view(1, 16),
            "kv_indices_prefix": torch.zeros(1, device=q.device, dtype=torch.int32),
            "kv_indptr_prefix": ptr,
            "kv": None,
            "kv_scale": None,
            "kv_indices_extend": None,
            "kv_indptr_extend": None,
            "attn_sink": None,
            "cos": cos,
            "sin": sin,
            "positions": pos,
            "q_is_rotated": True,
            "q_norm": False,
            "norm_eps": 1e-6,
            "debug_out": debug,
            "max_logits": torch.empty(q.shape[:2], device=q.device),
            "lse": torch.empty(q.shape[:2], device=q.device),
        }
        out = _call_v41(**kwargs)
        kwargs.update(out=out[0], out_scale=out[1])

        def run(scale):
            if direct:
                fused_v41_fwd(**kwargs, softmax_scale=scale)
            else:
                _call_v41(**kwargs, softmax_scale=scale)

    else:
        out = _call_legacy(*args, debug_out=debug)
        buffers = dict(zip(("out", "out_scale", "max_logits", "lse"), out))

        def run(scale):
            if direct:
                fused_norm_rope_attn_rope_cast_fwd(
                    *args, *out, None, None, None, None, debug, scale, True
                )
            else:
                _call_legacy(*args, **buffers, debug_out=debug, softmax_scale=scale)

    # 2e38 remains finite in base-2 units; zero Q must retain zero logits.
    run(2e38)
    torch.testing.assert_close(debug, torch.ones_like(debug), rtol=0, atol=0)
    torch.testing.assert_close(out[2], torch.zeros_like(out[2]), rtol=0, atol=0)
    torch.testing.assert_close(out[3], torch.zeros_like(out[3]), rtol=0, atol=0)
    for scale in (3e38, torch.finfo(torch.float32).max):
        with _raises(RuntimeError, match="softmax_scale.*base-2"):
            run(scale)
        run(2e38)
        torch.testing.assert_close(debug, torch.ones_like(debug), rtol=0, atol=0)


def check_public_api_has_no_debug_output(api):
    import inspect

    args = simple_inputs()
    debug = torch.empty_like(args[0], dtype=torch.float32)
    if api == "two_source":
        kwargs, _ = two_source_inputs()
        fn, positional = fused_v41, ()
    elif api == "bf16":
        args[1] = torch.ones(1, 512, device="cuda", dtype=torch.bfloat16)
        args[2] = torch.zeros(1, device="cuda", dtype=torch.int32)
        fn, positional, kwargs = fused_norm_rope_attn_rope_cast_bf16, args, {}
    else:
        fn, positional, kwargs = fused_norm_rope_attn_rope_cast, args, {}
    assert "debug_out" not in inspect.signature(fn).parameters
    with _raises(TypeError, match="debug_out"):
        fn(*positional, **kwargs, debug_out=debug)
    assert not hasattr(aiter, "fused_norm_rope_attn_rope_cast_fwd")
    assert not hasattr(aiter, "fused_norm_rope_attn_rope_cast_legacy_fwd")
    assert not hasattr(aiter, "_fused_norm_rope_attn_rope_cast_with_debug")


def make_benchmark_inputs(
    batch, heads, topk, window, pool_rows, dtype, mode, stage, layout
):
    torch.manual_seed(8801 + batch + heads + topk + window)
    extend_rows = batch * window if stage == "decode" else batch
    caches = []
    for rows in (pool_rows, extend_rows):
        values, scales = quantize_fp8(
            torch.randn(rows, 512, device="cuda", dtype=dtype) * 0.25
        )
        if layout == "interleaved":
            # ATOM's group32 cache is [S,528] bytes. Both views have row stride
            # 528; making either contiguous would change the measured call.
            packed = pack_rows(values, scales)
            values = packed[:, :512].view(torch.float8_e4m3fn)
            scales = packed[:, 512:]
        caches.append((values, scales.view(torch.float8_e8m0fnu)))
    main_rows = pool_rows - window if stage == "prefill" else pool_rows
    if topk > main_rows:
        raise ValueError("topk must fit the prefix pool excluding the prefill SWA ring")
    prefix_ids = torch.stack(
        [torch.randperm(main_rows, device="cuda")[:topk] for _ in range(batch)]
    ).int()
    token = torch.arange(batch, device="cuda", dtype=torch.int32)
    columns = torch.arange(window, device="cuda", dtype=torch.int32)
    prefix_lengths = torch.full_like(token, topk)
    if stage == "decode":
        # Independent requests, each with its own window in ring-buffer order.
        extend_ids = (
            token[:, None] * window + (columns + 37) % max(window, 1)
        ).flatten()
        lengths = torch.full_like(token, window)
    else:
        # Continued prefill consumes the old SWA tail before switching to extend.
        old_lengths = (window - 1 - token).clamp_min(0)
        old_begin = pool_rows + token - window + 1
        old_ids = pool_rows - window + (old_begin[:, None] + columns) % max(window, 1)
        prefix_ids = torch.cat((prefix_ids, old_ids), dim=1)[
            torch.cat(
                (
                    torch.ones_like(prefix_ids, dtype=torch.bool),
                    columns < old_lengths[:, None],
                ),
                dim=1,
            )
        ]
        prefix_lengths += old_lengths
        lengths = (token + 1).clamp_max(window)
        extend_ids = (token[:, None] - lengths[:, None] + 1 + columns)[
            columns < lengths[:, None]
        ]
    extend_ptr = torch.cat(
        (torch.zeros(1, device="cuda", dtype=torch.int32), lengths.cumsum(0).int())
    )
    q = torch.randn(batch, heads, 512, device="cuda", dtype=dtype) * 0.125
    angles = torch.randn(batch, 32, device="cuda")
    cos, sin = angles.cos().bfloat16(), angles.sin().bfloat16()
    q_scale = None
    if mode in ("rotated", "prepared"):
        q = rotate(q, cos, sin, token).bfloat16()
    if mode == "prepared":
        values, scales = quantize_fp8(q)
        q = values.reshape(batch, heads, 512)
        q_scale = scales.reshape(batch, heads, 16).view(torch.float8_e8m0fnu)
    sink = torch.linspace(-2, 2, heads, device="cuda")
    sink[::11] = -torch.inf
    return {
        "q": q,
        "q_scale": q_scale,
        "unified_kv": caches[0][0],
        "unified_kv_scale": caches[0][1],
        "kv_indices_prefix": prefix_ids.flatten(),
        "kv_indptr_prefix": torch.cat(
            (
                torch.zeros(1, device="cuda", dtype=torch.int32),
                prefix_lengths.cumsum(0).int(),
            )
        ),
        "kv": caches[1][0],
        "kv_scale": caches[1][1],
        "kv_indices_extend": extend_ids,
        "kv_indptr_extend": extend_ptr,
        "cos": cos,
        "sin": sin,
        "positions": token,
        "attn_sink": sink,
        "q_is_rotated": mode in ("rotated", "prepared"),
        "q_norm": mode == "norm",
        "is_decode": stage == "decode",
    }


def run_torch(args):
    """Independent FP32 attention oracle; BF16 rounding follows the op contract.

    Reconstruct each source from its CSR and process every query in bounded
    chunks, so large-T sweeps do not materialize [T,H,topk] attention scores.
    This reference is never timed.
    """
    q = args["q"]
    if args["q_scale"] is None:
        if args["q_norm"]:
            q = (
                q.float()
                * torch.rsqrt(q.float().square().mean(-1, keepdim=True) + 1e-6)
            ).bfloat16()
        if not args["q_is_rotated"]:
            q = rotate(q, args["cos"], args["sin"], args["positions"]).bfloat16()
        values, scales = quantize_fp8(q)
        qref = (
            values.float() * torch.exp2(scales.float() - 127).repeat_interleave(32, -1)
        ).reshape_as(q)
    else:
        qref = q.float() * args["q_scale"].float().repeat_interleave(32, -1)
    sources = []
    for name, value_key, scale_key in (
        ("prefix", "unified_kv", "unified_kv_scale"),
        ("extend", "kv", "kv_scale"),
    ):
        decoded = args[value_key].float() * args[scale_key].float().repeat_interleave(
            32, -1
        )
        ptr = args[f"kv_indptr_{name}"].cpu().tolist()
        sources.append((decoded, args[f"kv_indices_{name}"], ptr))
    expected = torch.empty_like(q, dtype=dtypes.fp32)
    for start in range(0, q.shape[0], 32):
        end = min(start + 32, q.shape[0])
        chunks, masks = [], []
        for decoded, indices, ptr in sources:
            lengths = torch.tensor(
                [ptr[i + 1] - ptr[i] for i in range(start, end)], device=q.device
            )
            width = max(ptr[i + 1] - ptr[i] for i in range(start, end))
            if width == 0:
                continue
            columns = torch.arange(width, device=q.device)
            mask = columns < lengths[:, None]
            offsets = torch.tensor(ptr[start:end], device=q.device)[:, None] + columns
            ids = indices[offsets.masked_fill(~mask, 0)].long()
            chunks.append(decoded[ids])
            masks.append(mask)
        kv = torch.cat(chunks, dim=1)
        mask = torch.cat(masks, dim=1)
        logits = (qref[start:end] @ kv.transpose(1, 2)) * 512**-0.5
        logits.masked_fill_(~mask[:, None, :], -torch.inf)
        sink = args["attn_sink"][None, :, None].expand(end - start, -1, -1)
        probabilities = torch.cat((logits, sink), dim=-1).softmax(-1)[..., :-1]
        attention = (probabilities @ kv).bfloat16()
        expected[start:end] = rotate(
            attention,
            args["cos"],
            args["sin"],
            args["positions"][start:end],
            inverse=True,
        )
    return expected


@benchmark()
def test_fused_norm_rope_attn_rope_cast(
    batch, heads, topk, window, pool_rows, dtype, mode, stage, layout, splits
):
    args = make_benchmark_inputs(
        batch, heads, topk, window, pool_rows, dtype, mode, stage, layout
    )
    args["num_splits"] = splits
    # Model output is [T,H/8,4096] FP8 plus row-layout group32 scales. Allocate
    # both before timing. Optional LSE/max/debug outputs are not model outputs.
    args["out"] = torch.empty(
        batch, heads // 8, 4096, device="cuda", dtype=torch.float8_e4m3fn
    )
    args["out_scale"] = torch.empty(
        batch, heads // 8, 128, device="cuda", dtype=torch.float8_e8m0fnu
    )
    if splits > 1:
        args["workspace"] = torch.empty(
            batch, splits, heads, 516, device="cuda", dtype=dtypes.fp32
        )
    # For splits=0 the public wrapper selects/allocates split scratch itself;
    # graph timing excludes the host allocation, just as it excludes dispatch.
    ref = run_torch(args)
    candidates = {"fused": lambda: fused_v41(**args)}

    # Effective QK+PV FLOPs (4*H*D per selected KV row). TB/s is logical I/O:
    # selected KV values/scales once per query, CSR, Q, RoPE tables and outputs.
    # These are algorithmic metrics, not measured HBM traffic; implementation
    # scratch and cache reuse are excluded. All fused stages remain in the time.
    selected_rows = (
        args["kv_indices_prefix"].numel() + args["kv_indices_extend"].numel()
    )
    assert selected_rows == batch * (topk + window)
    flops = 4 * heads * 512 * selected_rows
    nbytes = sum(
        args[indices].numel()
        * (512 * args[values].element_size() + 16 * args[scales].element_size())
        for indices, values, scales in (
            ("kv_indices_prefix", "unified_kv", "unified_kv_scale"),
            ("kv_indices_extend", "kv", "kv_scale"),
        )
    )
    for key in (
        "q",
        "q_scale",
        "kv_indices_prefix",
        "kv_indptr_prefix",
        "kv_indices_extend",
        "kv_indptr_extend",
        "cos",
        "sin",
        "positions",
        "attn_sink",
        "out",
        "out_scale",
    ):
        tensor = args[key]
        if tensor is not None:
            nbytes += tensor.numel() * tensor.element_size()
    ret = {"gfx": get_gfx()}
    for name, fn in candidates.items():
        out, us = run_perftest(fn, testGraph=True, num_rotate_args=1)
        actual = dequantize(*out[:2]).to(dtypes.fp32)
        err = checkAllclose(
            ref.to(dtypes.fp32),
            actual,
            rtol=0.02,
            atol=0.008,
            tol_err_ratio=0,
            msg=f"{name}: dequantized output",
        )
        assert err == 0, f"{name}: output exceeds reference tolerance"
        assert float((actual - ref).norm() / ref.norm().clamp_min(1e-30)) < 0.035
        # Attention rounding can cross an E8M0 boundary relative to the dense
        # oracle. Check scales exactly against torch quantization of the kernel's
        # pre-quant output, obtained in a separate, untimed diagnostic call.
        debug = torch.empty_like(ref)
        checked = _call_v41(**args, debug_out=debug)
        debug_err = checkAllclose(
            ref,
            debug,
            rtol=0.02,
            atol=0.008,
            tol_err_ratio=0,
            msg=f"{name}: pre-quant output",
        )
        assert debug_err == 0
        assert float((debug - ref).norm() / ref.norm().clamp_min(1e-30)) < 0.005
        groups = debug.reshape(-1, 16, 32)
        # frexp preserves the FP32 boundary: log2 can round a scale just above
        # 2**n to n and incorrectly select the smaller E8M0 scale.
        mantissa, exponent = torch.frexp(groups.abs().amax(-1).clamp_min(1e-8) / 448)
        exponent = exponent - (mantissa == 0.5).int()
        scales = (exponent + 127).to(torch.uint8)
        values = (groups * torch.exp2(-exponent.float())[..., None]).to(
            torch.float8_e4m3fn
        )
        scale_err = checkAllclose(
            scales.reshape_as(checked[1]).to(dtypes.fp32),
            checked[1].view(torch.uint8).to(dtypes.fp32),
            rtol=0,
            atol=0,
            tol_err_ratio=0,
            msg=f"{name}: scale bytes",
        )
        quant_err = checkAllclose(
            values.reshape_as(checked[0]).to(dtypes.fp32),
            checked[0].to(dtypes.fp32),
            rtol=0,
            atol=0,
            tol_err_ratio=0,
            msg=f"{name}: FP8 quantization",
        )
        assert scale_err == quant_err == 0
        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = flops / us / 1e6
        ret[f"{name} TB/s"] = nbytes / us / 1e6
        ret[f"{name} err"] = err
    return ret


def check_norm_subnormal_products(api, heads, tokens, splits, capacity):
    q_value, epsilon = 2.0**-64, 2.0**-126
    scalar = torch.tensor(q_value, dtype=torch.float32, device="cpu")
    normalized = float((scalar * torch.rsqrt(scalar.square() + epsilon)).bfloat16())
    assert normalized == 0.447265625
    q = torch.full((tokens, heads, 512), q_value, dtype=torch.bfloat16, device="cuda")
    kv = torch.arange(3, device="cuda").bfloat16()[:, None].expand(-1, 512).contiguous()
    packed = pack_rows(*quantize_fp8(kv))
    cos = torch.ones(1, 32, device="cuda", dtype=torch.bfloat16)
    sin = torch.zeros_like(cos)
    pos = torch.zeros(tokens, device="cuda", dtype=torch.int32)
    if api == "two_source":
        indices = torch.full(
            (tokens * (capacity - 1),), -1, device="cuda", dtype=torch.int32
        )
        indices[: tokens * 2] = torch.tensor(
            [0, 1], device="cuda", dtype=torch.int32
        ).repeat(tokens)
        kwargs = {
            "q_scale": None,
            "unified_kv": packed[:2, :512].view(torch.float8_e4m3fn),
            "unified_kv_scale": packed[:2, 512:],
            "kv_indices_prefix": indices,
            "kv_indptr_prefix": torch.arange(
                tokens + 1, device="cuda", dtype=torch.int32
            )
            * 2,
            "kv": packed[2:, :512].view(torch.float8_e4m3fn),
            "kv_scale": packed[2:, 512:],
            "kv_indices_extend": torch.zeros(tokens, device="cuda", dtype=torch.int32),
            "kv_indptr_extend": torch.arange(
                tokens + 1, device="cuda", dtype=torch.int32
            ),
            "is_decode": splits > 1,
            "num_splits": splits,
            "cos": cos,
            "sin": sin,
            "positions": pos,
            "softmax_scale": 1 / 512,
            "q_is_rotated": False,
        }

        def run(query, norm, debug):
            return _call_v41(
                q=query, **kwargs, q_norm=norm, norm_eps=epsilon, debug_out=debug
            )

    else:
        ids = torch.arange(3, device="cuda", dtype=torch.int32).repeat(tokens)
        pool = kv if api == "bf16" else packed.flatten()
        if api == "legacy":
            ids = ids.long() * 1056 + 1
        ptr = torch.arange(tokens + 1, device="cuda", dtype=torch.int32) * 3

        def run(query, norm, debug):
            return _call_legacy(
                query,
                pool,
                ids,
                ptr,
                cos,
                sin,
                pos,
                q_norm=norm,
                norm_eps=epsilon,
                q_is_rotated=False,
                softmax_scale=1 / 512,
                debug_out=debug,
            )

    actual = torch.empty_like(q, dtype=torch.float32)
    expected = torch.empty_like(actual)
    result = run(q, True, actual)
    reference = run(torch.full_like(q, normalized), False, expected)
    torch.testing.assert_close(actual, expected, rtol=0.001, atol=0.001)
    torch.testing.assert_close(
        dequantize(*result[:2]), dequantize(*reference[:2]), rtol=0, atol=0
    )


def check_csr_capacity_limit(api, source):
    limit = (2**31 - 1) // 2 - 256
    if api == "two_source":
        kwargs, _ = two_source_inputs(mode="prepared")
        key = "kv_indices_" + source
        saved = kwargs[key]

        def run(indices):
            kwargs[key] = indices
            return _call_v41(**kwargs)

    else:
        args = simple_inputs()
        if api == "bf16":
            args[1] = torch.ones(1, 512, device="cuda", dtype=torch.bfloat16)
            args[2] = args[2].int().zero_()
        kwargs = {}
        if source == "extend":
            kwargs.update(
                extend_kv=torch.ones(1, 512, device="cuda", dtype=torch.bfloat16),
                extend=torch.zeros(1, device="cuda", dtype=torch.int32),
                extend_indptr=torch.tensor([0, 1], device="cuda", dtype=torch.int32),
            )
        saved = args[2] if source == "prefix" else kwargs["extend"]

        def run(indices):
            if source == "prefix":
                args[2] = indices
            else:
                kwargs["extend"] = indices
            return _call_legacy(*args, **kwargs)

    for capacity in (limit + 1, 2**31 - 1):
        with _raises(RuntimeError, match="CSR capacity exceeds"):
            run(saved[:1].expand(capacity))
    with _raises(RuntimeError, match="wrong shape or strides"):
        run(saved[:1].expand(limit))
    run(saved)


def check_decode_optional_statistics(heads, mode, splits, valid_indices):
    kwargs, (expected, mx, lse) = two_source_inputs(
        heads, "padded", mode, True, True, valid_indices
    )
    tokens = kwargs["q"].shape[0]
    workspace = torch.empty(tokens, splits, heads, 516, device="cuda")
    kwargs.update(is_decode=True, num_splits=splits, workspace=workspace)
    for option in ("output", "max_logits", "lse", "debug_out"):
        call = dict(kwargs)
        if option != "output":
            call[option] = torch.empty_like(expected if option == "debug_out" else mx)
        actual = _call_v41(**call)
        error = (dequantize(*actual[:2]) - expected).norm() / expected.norm()
        assert float(error) < 0.035
        if option == "max_logits":
            torch.testing.assert_close(actual[2], mx, rtol=1e-4, atol=3e-5)
        elif option == "lse":
            torch.testing.assert_close(actual[3], lse, rtol=1e-4, atol=3e-5)
        elif option == "debug_out":
            torch.testing.assert_close(call[option], expected, rtol=0.02, atol=0.008)
        call.update(out=actual[0], out_scale=actual[1])
        outputs = [x for x in actual if x is not None]
        if option == "debug_out":
            outputs.append(call[option])
        references = [x.clone() for x in outputs]
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            _call_v41(**call)
        for _ in range(2):
            for x in [workspace, *outputs]:
                x.view(torch.uint8).fill_(255)
            graph.replay()
            for x, ref in zip(outputs, references):
                torch.testing.assert_close(
                    x.view(torch.uint8), ref.view(torch.uint8), rtol=0, atol=0
                )


def check_decode_small_grid_output(heads, mode, splits, valid_indices):
    kwargs, (expected, _, _) = two_source_inputs(
        heads, "padded", mode, True, True, valid_indices
    )
    for key in ("q", "q_scale", "positions"):
        if kwargs[key] is not None:
            kwargs[key] = kwargs[key][:1]
    for source in ("prefix", "extend"):
        kwargs[f"kv_indptr_{source}"] = kwargs[f"kv_indptr_{source}"][:2]
    expected = expected[:1]
    workspace = (
        torch.full((1, splits, heads, 516), torch.nan, device="cuda")
        if splits > 1
        else None
    )
    kwargs.update(is_decode=True, num_splits=splits, workspace=workspace)
    actual = _call_v41(**kwargs)
    error = (dequantize(*actual[:2]) - expected).norm() / expected.norm().clamp_min(
        1e-30
    )
    assert float(error) < 0.035
    kwargs.update(out=actual[0], out_scale=actual[1])
    refs = [x.clone() for x in actual[:2]]
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _call_v41(**kwargs)
    for _ in range(3):
        for x in actual[:2]:
            x.view(torch.uint8).fill_(255)
        if workspace is not None:
            workspace.fill_(torch.nan)
        graph.replay()
        for x, ref in zip(actual[:2], refs):
            torch.testing.assert_close(
                x.view(torch.uint8), ref.view(torch.uint8), rtol=0, atol=0
            )


def check_small_head_causal_prefill(heads, mode, splits):
    check_two_source_h64_causal_prefill(160, 512, mode, heads, splits, True)


def check_h32_split_head_tile_boundary(tokens, splits, mode):
    check_two_source_h64_causal_prefill(tokens, 512, mode, 32, splits)


def check_h16_long_list_auto_split(tokens, topk, mode):
    # Exercise the long-list policy with dense output/max/LSE checks around
    # the grid and allocated-row boundaries, including normalized raw Q.
    check_two_source_h64_causal_prefill(tokens, topk, mode, 16, 0)


def check_h32_auto_split_boundary(tokens, mode):
    check_two_source_h64_causal_prefill(tokens, 2048, mode, 32, 0)


def check_small_head_split_storage(heads, mode, splits):
    kwargs, (expected, mx, lse) = two_source_inputs(
        heads, "padded", mode, True, True, True
    )
    guards = []

    def buffer(shape, dtype, offset):
        count = math.prod(shape) * torch.empty((), dtype=dtype).element_size()
        storage = torch.full(
            (offset + count + 16,), 0x5A, device="cuda", dtype=torch.uint8
        )
        guards.append((storage, offset, count))
        return storage[offset : offset + count].view(dtype).reshape(shape)

    q = kwargs["q"]
    view = buffer(q.shape, q.dtype, 32 if mode == "prepared" else 16)
    view.copy_(q)
    kwargs["q"] = view
    t = q.shape[0]
    kwargs.update(
        is_decode=True,
        num_splits=splits,
        out=buffer((t, heads // 8, 4096), torch.float8_e4m3fn, 16),
        out_scale=buffer((t, heads // 8, 128), torch.float8_e8m0fnu, 1),
        debug_out=buffer((t, heads, 512), torch.float32, 16),
        max_logits=buffer((t, heads), torch.float32, 4),
        lse=buffer((t, heads), torch.float32, 4),
    )
    if splits > 1:
        kwargs["workspace"] = buffer((t, splits, heads, 516), torch.float32, 16)
        kwargs["workspace"].fill_(torch.nan)
    for auxiliary in (True, False):
        if not auxiliary:
            for name in ("debug_out", "max_logits", "lse"):
                kwargs.pop(name)
        if splits > 1:
            kwargs["workspace"].fill_(torch.nan)
        out = _call_v41(**kwargs)
        if auxiliary:
            torch.testing.assert_close(
                kwargs["debug_out"], expected, rtol=0.02, atol=0.008
            )
            torch.testing.assert_close(out[2], mx, rtol=1e-4, atol=3e-5)
            torch.testing.assert_close(out[3], lse, rtol=1e-4, atol=3e-5)
        assert float((dequantize(*out[:2]) - expected).norm() / expected.norm()) < 0.035
        for storage, offset, count in guards:
            assert bool((storage[:offset] == 0x5A).all())
            assert bool((storage[offset + count :] == 0x5A).all())


def run_contract_checks():
    """Run every correctness case during the same direct invocation used by CI."""
    checks = [
        (
            check_h16_long_list_auto_split,
            [
                ("mode", ["raw", "prepared", "norm"]),
                (
                    "tokens,topk",
                    [
                        (63, 2048),
                        (64, 2048),
                        (65, 2048),
                        (127, 2048),
                        (128, 2048),
                        (129, 2048),
                        (255, 2048),
                        (256, 2048),
                        (257, 2048),
                        (128, 1919),
                        (128, 1920),
                        (256, 1919),
                        (256, 1920),
                        (256, 4096),
                    ],
                ),
            ],
        ),
        (
            check_h32_auto_split_boundary,
            [
                ("mode", ["raw", "prepared", "norm"]),
                ("tokens", [17, 24, 31, 32, 33, 48, 64, 65, 80]),
            ],
        ),
        (
            check_h32_split_head_tile_boundary,
            [
                ("mode", ["raw", "prepared", "norm"]),
                ("tokens,splits", [(64, 2), (65, 2), (32, 4), (33, 4)]),
            ],
        ),
        (
            check_decode_small_grid_output,
            [
                ("heads", [64, 128]),
                ("mode", ["raw", "prepared", "norm"]),
                ("splits", [0, 4, 8, 16, 32]),
                ("valid_indices", [False, True]),
            ],
        ),
        (
            check_decode_optional_statistics,
            [
                ("heads", [64, 128]),
                ("mode", ["raw", "norm", "prepared"]),
                ("splits", [2, 3, 4, 8, 16, 32]),
                ("valid_indices", [False, True]),
            ],
        ),
        (
            check_norm_subnormal_products,
            [
                (
                    "api,heads,tokens,splits,capacity",
                    [
                        ("two_source", 16, 1, 1, 1024),
                        ("two_source", 16, 128, 1, 1024),
                        ("two_source", 16, 1, 4, 1024),
                        ("two_source", 16, 128, 4, 1024),
                        ("two_source", 64, 1, 1, 3),
                        ("two_source", 64, 128, 1, 1024),
                        ("two_source", 64, 4096, 1, 1024),
                        ("two_source", 64, 1, 4, 1024),
                        ("two_source", 128, 1, 1, 3),
                        ("two_source", 128, 128, 1, 1024),
                        ("two_source", 128, 4096, 1, 1024),
                        ("two_source", 128, 1, 4, 1024),
                        ("legacy", 64, 1, 1, 3),
                        ("legacy", 128, 1, 1, 3),
                        ("bf16", 64, 1, 1, 3),
                        ("bf16", 128, 1, 1, 3),
                    ],
                )
            ],
        ),
        (
            check_csr_capacity_limit,
            [
                ("api", ["legacy", "bf16", "two_source"]),
                ("source", ["prefix", "extend"]),
            ],
        ),
        (
            check_ragged_dense,
            [
                ("with_extend", [False, True]),
                ("raw_q,q_norm", [(False, False), (True, False), (True, True)]),
                ("heads", [8, 16, 32, 64, 128]),
            ],
        ),
        (
            check_invalid_metadata,
            [
                (
                    "bad",
                    [
                        "q_dtype",
                        "q_stride",
                        "q_alignment",
                        "pool_alignment",
                        "positions_dtype",
                        "csr_dtype",
                        "missing_extend",
                        "output_shape",
                        "scale",
                    ],
                )
            ],
        ),
        (check_zero_tokens, []),
        (check_nondefault_stream_and_buffers, []),
        (check_prefetch_codec_empty_tiles_and_replay, [("heads", [64, 128])]),
        (
            check_prefetch_long_invalid_decode,
            [("raw_q", [False, True]), ("heads", [64, 128])],
        ),
        (
            check_prefetch_dispatch_boundary,
            [("tokens", [1023, 1024, 1025]), ("heads", [64, 128])],
        ),
        (
            check_h128_load_schedule_dispatch_boundary,
            [("raw_q", [False, True]), ("tokens", [2047, 2048, 2049])],
        ),
        (
            check_bf16_dense,
            [
                ("singleton", [False, True]),
                ("raw,q_norm", [(False, False), (True, False), (True, True)]),
                ("heads", [8, 16, 32, 64, 128]),
            ],
        ),
        (
            check_bf16_stream_replay,
            [("tokens", [0, 1024, 1025, 2048, 2049]), ("heads", [64, 128])],
        ),
        (
            check_bf16_contract,
            [("bad", ["dtype", "shape", "strides", "indices", "empty_cache"])],
        ),
        (
            check_fp8_qk_dense,
            [
                ("raw_q,q_norm", [(False, False), (True, False), (True, True)]),
                ("heads", [64, 128]),
            ],
        ),
        (check_fp8_qk_reject, [("kind", ["heads", "bf16", "extend"])]),
        (check_fp8_qk_replay, []),
        (check_fp8_qk_zero_tokens, []),
        (check_fp8_prepared_q, [("heads", [64, 128])]),
        (
            check_prepared_q_contract,
            [("bad", ["flag", "rotate", "shape", "dtype", "q_dtype"])],
        ),
        (
            check_fused_norm_rope_attn_rope_cast_contract,
            [
                ("mode", ["mixed", "valid", "truncated"]),
                ("pos64", [False, True]),
                ("raw,q_norm", [(False, False), (True, False), (True, True)]),
            ],
        ),
        (
            check_norm_contract,
            [
                (
                    "bad",
                    [
                        "rotated",
                        "prepared",
                        "zero",
                        "negative",
                        "nan",
                        "infinity",
                        "underflow",
                    ],
                ),
                ("direct", [False, True]),
            ],
        ),
        (check_norm_zero_q, [("tokens", [0, 3])]),
        (check_norm_recomputed_on_graph_replay, []),
        (
            check_small_head_causal_prefill,
            [
                ("splits", [1, 3]),
                ("mode", ["raw", "prepared", "norm"]),
                ("heads", [16, 32]),
            ],
        ),
        (
            check_small_head_split_storage,
            [
                ("splits", [1, 2, 3, 4, 5, 7, 32]),
                ("mode", ["raw", "rotated", "prepared", "norm"]),
                ("heads", [16, 32]),
            ],
        ),
        (
            check_two_source_dense,
            [
                ("valid_indices", [False, True]),
                ("with_sink", [False, True]),
                ("wide", [False, True]),
                ("mode", ["raw", "rotated", "prepared", "norm"]),
                ("layout", ["interleaved", "separate", "padded"]),
                ("heads", [16, 32, 64, 128]),
            ],
        ),
        (
            check_two_source_empty_and_bounds,
            [
                ("large_grid", [False, True]),
                ("empty", ["prefix", "extend", "both", "omitted", "tokens"]),
                ("heads", [16, 32, 64, 128]),
            ],
        ),
        (
            check_two_source_rescale,
            [("mixed_scales", [False, True]), ("heads", [16, 64, 128])],
        ),
        (
            check_two_source_graph_replay,
            [
                ("valid_indices", [False, True]),
                ("mode", ["raw", "prepared", "prepared_mixed", "norm"]),
                ("heads", [16, 32, 64, 128]),
            ],
        ),
        (
            check_two_source_contract,
            [
                (
                    "bad",
                    [
                        "prefix_dtype",
                        "scale_dtype",
                        "prefix_rows",
                        "extend_rows",
                        "prefix_indices",
                        "extend_indices",
                        "prefix_ptr",
                        "missing_scale",
                        "missing_ptr",
                        "channel_stride",
                        "row_stride",
                        "unaligned",
                        "q_scale_dtype",
                        "prepared_rotation",
                        "prepared_norm",
                        "out_alias",
                        "stat_alias",
                        "norm_eps",
                        "backend",
                    ],
                )
            ],
        ),
        (
            check_two_source_h64_causal_prefill,
            [
                ("mode", ["raw", "prepared", "norm"]),
                (
                    "tokens,topk",
                    [
                        (512, 512),
                        (768, 512),
                        (769, 512),
                        (1024, 448),
                        (1024, 512),
                        (1024, 768),
                        (1024, 1024),
                    ],
                ),
            ],
        ),
        (
            check_two_source_h64_q_scale_groups,
            [
                ("extend", [False, True]),
                ("valid_indices", [False, True]),
                ("rotated", [False, True]),
            ],
        ),
        (
            check_two_source_rope_storage_offset,
            [
                ("mode", ["raw", "rotated", "prepared", "norm"]),
                ("heads", [16, 32, 64, 128]),
            ],
        ),
        (
            check_two_source_scale_output_storage_offset,
            [
                ("offset", [1, 3]),
                ("mode", ["raw", "prepared"]),
                ("heads", [16, 32, 64, 128]),
            ],
        ),
        (
            check_two_source_index_storage_offset,
            [
                ("valid_indices", [False, True]),
                ("offset", [1, 3]),
                ("mode", ["raw", "prepared"]),
                ("heads", [16, 32, 64, 128]),
            ],
        ),
        (
            check_fp8_decode_dense,
            [
                ("with_sink", [False, True]),
                ("valid_indices", [False, True]),
                ("splits", [2, 4, 7]),
                ("mode", ["raw", "rotated", "prepared", "norm"]),
                ("layout", ["interleaved", "separate", "padded"]),
                ("heads", [16, 32, 64, 128]),
            ],
        ),
        (
            check_fp8_decode_ring_view,
            [
                ("sources", ["both", "prefix", "extend", "empty"]),
                ("splits", [1, 3, 4, 32]),
                ("heads", [64, 128]),
            ],
        ),
        (
            check_fp8_decode_graph_replay,
            [("mode", ["raw", "prepared", "norm"]), ("heads", [16, 32, 64, 128])],
        ),
        (
            check_fp8_prefill_auto_split,
            [("heads", [16, 32, 64, 128]), ("mode", ["raw", "prepared", "norm"])],
        ),
        (
            check_fp8_decode_contract,
            [
                (
                    "bad",
                    [
                        "phase",
                        "prefill_workspace",
                        "count",
                        "fraction",
                        "workspace_shape",
                        "workspace_dtype",
                        "workspace_overlap",
                        "single_workspace",
                    ],
                )
            ],
        ),
        (
            check_fp8_decode_auto_and_poisoned_workspace,
            [
                ("splits", [4, 32]),
                ("valid_indices", [False, True]),
                ("mode", ["raw", "prepared", "norm"]),
                ("heads", [16, 32, 64, 128]),
            ],
        ),
        (
            check_fp8_decode_large_grid_replay,
            [
                ("splits", [1, 2, 4]),
                ("layout", ["interleaved", "separate", "padded"]),
                ("valid_indices", [False, True]),
                ("mode", ["raw", "rotated", "prepared", "norm"]),
                ("heads", [64, 128]),
            ],
        ),
        (
            check_shared_kv_wave_boundaries,
            [
                ("heads", [64, 128]),
                ("is_decode", [False, True]),
                ("valid_indices", [False, True]),
                ("mode", ["raw", "prepared"]),
                (
                    "tokens",
                    [
                        320,
                        321,
                        384,
                        385,
                        512,
                        513,
                        640,
                        641,
                        768,
                        769,
                        896,
                        897,
                        1024,
                        1025,
                        2048,
                        2049,
                        4096,
                        1536,
                        2047,
                        3072,
                        4095,
                        4097,
                        8192,
                    ],
                ),
            ],
        ),
        (
            check_h64_common_prefill_contract,
            [
                ("valid_indices", [False, True]),
                ("layout", ["interleaved", "separate", "padded"]),
                ("mode", ["raw", "prepared", "rotated", "norm"]),
                ("tokens", [511, 512, 1024, 1025]),
            ],
        ),
        (
            check_mid_batch_attention_routes,
            [
                ("layout", ["interleaved", "separate", "padded"]),
                ("mode", ["raw", "prepared", "rotated", "norm"]),
                ("heads", [64, 128]),
                (
                    "tokens",
                    [
                        64,
                        65,
                        127,
                        128,
                        129,
                        255,
                        256,
                        257,
                        383,
                        384,
                        385,
                        768,
                        769,
                        1024,
                    ],
                ),
            ],
        ),
        (
            check_mid_batch_attention_capacity,
            [
                ("mode", ["raw", "prepared"]),
                ("capacity", [511, 512, 513, 1023, 1024, 1025]),
                ("tokens,heads", [(128, 64), (256, 64), (128, 128), (256, 128)]),
            ],
        ),
        (
            check_h64_large_prefill_capacity,
            [
                ("valid_indices", [False, True]),
                ("layout", ["interleaved", "separate", "padded"]),
                ("mode", ["raw", "rotated", "prepared", "norm"]),
                ("capacity", [1023, 1024, 1025]),
                ("tokens", [2048, 4096, 8192, 16384]),
            ],
        ),
        (
            check_h128_large_shared_kv_capacity,
            [
                ("valid_indices", [False, True]),
                ("layout", ["interleaved", "separate", "padded"]),
                ("mode", ["raw", "rotated", "prepared", "norm"]),
                ("capacity", [1023, 1024, 1025]),
                ("tokens", [1025, 2048, 4096, 16384]),
            ],
        ),
        (
            check_native_split_integer_width_and_error_recovery,
            [("bad_splits", [4294967297, -4294967295])],
        ),
        (
            check_h64_prefill_address_width,
            [
                ("tokens", [512, 2048]),
                ("prepared", [False, True]),
                ("rows", [3, 5]),
                ("wide_view", ["unified_kv", "unified_kv_scale", "kv", "kv_scale"]),
            ],
        ),
        (
            check_q_and_debug_16byte_storage_offsets,
            [
                ("q_norm", [False, True]),
                ("heads", [64, 128]),
                ("api", ["legacy", "legacy_fp8", "two_source"]),
            ],
        ),
        (
            check_softmax_scale_base2_range_and_error_recovery,
            [("direct", [False, True]), ("api", ["legacy", "two_source"])],
        ),
        (
            check_public_api_has_no_debug_output,
            [("api", ["legacy", "bf16", "two_source"])],
        ),
    ]
    rows = []
    for fn, axes in checks:
        passed = 0
        for values in itertools.product(*(values for _, values in axes)):
            kwargs = {}
            for (names, _), value in zip(axes, values):
                names = names.split(",")
                values_for_axis = (value,) if len(names) == 1 else value
                kwargs.update(zip(names, values_for_axis))
            try:
                fn(**kwargs)
            except Exception as error:
                raise AssertionError(f"{fn.__name__} failed with {kwargs}") from error
            passed += 1
        rows.append({"check": fn.__name__, "passed": passed})
        aiter.logger.info("%s: %d cases passed", fn.__name__, passed)
    aiter.logger.info(
        "fused_norm_rope_attn_rope_cast correctness (%d cases, markdown):\n%s",
        sum(row["passed"] for row in rows),
        pd.DataFrame(rows).to_markdown(index=False),
    )
    return rows


def main():
    if not torch.cuda.is_available() or get_gfx() not in SUPPORTED_GFX:
        aiter.logger.warning("fused_norm_rope_attn_rope_cast requires gfx950; skipping")
        return
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="Fused v4.1 correctness/performance sweep; torch is reference only.",
    )
    parser.add_argument(
        "-d",
        "--dtype",
        type=dtypes.str2Dtype,
        choices=[dtypes.bf16],
        nargs="+",
        default=[dtypes.bf16],
    )
    parser.add_argument(
        "-b",
        "--batch",
        type=int,
        nargs="+",
        default=[1, 128, 1024],
        help="Query count T (independent requests in decode)",
    )
    parser.add_argument(
        "-H",
        "--heads",
        type=int,
        choices=[16, 32, 64, 128],
        nargs="+",
        default=[16, 32, 64, 128],
    )
    parser.add_argument(
        "--topk",
        type=int,
        nargs="+",
        default=[512, 1024, 2048],
        help="Selected prefix rows per query, excluding the old prefill SWA tail",
    )
    parser.add_argument(
        "--window",
        type=int,
        nargs="+",
        default=[128],
        help="Causal prefill window / per-request decode history",
    )
    parser.add_argument("--pool-rows", type=int, nargs="+", default=[65536])
    parser.add_argument(
        "--modes",
        choices=["unprepared", "prepared", "rotated", "norm"],
        nargs="+",
        default=["unprepared", "prepared"],
    )
    parser.add_argument(
        "--stages",
        choices=["decode", "prefill"],
        nargs="+",
        default=["decode", "prefill"],
    )
    parser.add_argument(
        "-l",
        "--layout",
        choices=["interleaved", "separate"],
        nargs="+",
        default=["interleaved"],
    )
    parser.add_argument(
        "--splits",
        type=int,
        choices=range(33),
        nargs="+",
        default=[0],
        help="0: public API auto selection; 1..32: explicit split count (prefill: 0/1)",
    )
    options = parser.parse_args()
    if (
        min(options.batch + options.topk + options.pool_rows) <= 0
        or min(options.window) < 0
    ):
        parser.error(
            "batch, topk and pool-rows must be positive; window must be nonnegative"
        )
    if max(options.topk) > min(options.pool_rows):
        parser.error("topk must not exceed pool-rows")
    if "prefill" in options.stages and max(options.topk) + max(options.window) > min(
        options.pool_rows
    ):
        parser.error("prefill topk + window must not exceed pool-rows")
    torch.backends.cuda.matmul.allow_tf32 = False
    run_contract_checks()
    rows = []
    for (
        batch,
        heads,
        topk,
        window,
        pool_rows,
        dtype,
        mode,
        stage,
        layout,
        splits,
    ) in itertools.product(
        options.batch,
        options.heads,
        options.topk,
        options.window,
        options.pool_rows,
        options.dtype,
        options.modes,
        options.stages,
        options.layout,
        options.splits,
    ):
        if stage == "prefill" and splits > 1:
            aiter.logger.warning(
                "prefill supports splits=0/1; skipping splits=%s", splits
            )
            continue
        rows.append(
            test_fused_norm_rope_attn_rope_cast(
                batch,
                heads,
                topk,
                window,
                pool_rows,
                dtype,
                mode,
                stage,
                layout,
                splits,
            )
        )
    df = pd.DataFrame(rows)
    aiter.logger.info(
        "TFLOPS: effective QK+PV; TB/s: logical I/O, excluding scratch/cache effects."
    )
    aiter.logger.info(
        "fused_norm_rope_attn_rope_cast summary (markdown):\n%s",
        df.to_markdown(index=False),
    )


if __name__ == "__main__":
    main()
