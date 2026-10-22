# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Fused norm/RoPE/attention/cast: dense references, cache modes and replay.

Run with pytest, or directly with --benchmark for a synthetic timing fixture.
"""

import argparse

import pytest
import torch

from aiter import fused_norm_rope_attn_rope_cast, fused_norm_rope_attn_rope_cast_bf16
from aiter.ops.fused_norm_rope_attn_rope_cast import fused_norm_rope_attn_rope_cast_fwd
from aiter.test_common import perftest

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or not torch.cuda.get_device_properties(0).gcnArchName.startswith("gfx950"),
    reason="requires gfx950",
)


# Shared fixtures and independent dense references.


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


def make_paged_cache(n=1201, block=32, separate=False, padding=0):
    pages = (n + block - 1) // block
    dp = block * (512 if separate else 528) + padding
    sp = block * 16 + padding + 4 if separate else dp
    data = torch.full((pages * dp,), 255, device="cuda", dtype=torch.uint8)
    scales = (
        torch.full((pages * sp,), 255, device="cuda", dtype=torch.uint8)
        if separate
        else data[block * 512 :]
    )
    slots = torch.arange(n, device="cuda")
    values = (torch.randn(n, 512, device="cuda") * 2).to(torch.float8_e4m3fn)
    codes = torch.randint(122, 128, (n, 16), device="cuda", dtype=torch.uint8)
    d = slots // block * dp + slots % block * 512
    s = slots // block * sp + slots % block * 16
    data[d[:, None] + torch.arange(512, device="cuda")] = values.view(torch.uint8)
    scales[s[:, None] + torch.arange(16, device="cuda")] = codes
    kw = {
        "cache_scales": scales,
        "cache_page_size": block,
        "cache_num_rows": n,
        "cache_data_page_stride": dp,
        "cache_scale_page_stride": sp,
    }
    return data, kw


def decode_pages(data, kw):
    # Decode bytes on CPU, independently of the kernel and fixture values.
    data, scale = data.cpu(), kw["cache_scales"].cpu()
    b = kw["cache_page_size"]
    rows = []
    for slot in range(kw["cache_num_rows"]):
        page, row = divmod(slot, b)
        d = page * kw["cache_data_page_stride"] + row * 512
        s = page * kw["cache_scale_page_stride"] + row * 16
        v = data[d : d + 512].contiguous().view(torch.float8_e4m3fn).float()
        c = scale[s : s + 16].int().float()
        rows.append((v * torch.exp2(c - 127).repeat_interleave(32)).bfloat16())
    return torch.stack(rows).cuda()


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


def make_fp8_qk_inputs(heads, page, raw_q, seed=711):
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


@pytest.mark.parametrize("heads", [8, 16, 32, 64, 128])
@pytest.mark.parametrize("raw_q,q_norm", [(False, False), (True, False), (True, True)])
@pytest.mark.parametrize("with_extend", [False, True])
def test_ragged_dense(q_norm, heads, raw_q, with_extend):
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
    out = fused_norm_rope_attn_rope_cast(
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


@pytest.mark.parametrize(
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
def test_invalid_metadata(bad):
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
    with pytest.raises((ValueError, RuntimeError)):
        fused_norm_rope_attn_rope_cast(*args, **kw)


def test_zero_tokens():
    out = fused_norm_rope_attn_rope_cast(*simple_inputs(0))
    assert all(x.numel() == 0 for x in out)


def test_nondefault_stream_and_buffers():
    args = simple_inputs(5)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        out = fused_norm_rope_attn_rope_cast(*args)
        again = fused_norm_rope_attn_rope_cast(
            *args, out=out[0], out_scale=out[1], max_logits=out[2], lse=out[3]
        )
        assert all(x is y for x, y in zip(out, again))
    torch.cuda.current_stream().wait_stream(stream)
    torch.testing.assert_close(
        dequantize(*out[:2]), torch.ones_like(args[0]).float(), rtol=0, atol=0
    )


@pytest.mark.parametrize("heads", [64, 128])
def test_prefetch_codec_empty_tiles_and_replay(heads):
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
    out = fused_norm_rope_attn_rope_cast(*args, debug_out=debug, q_is_rotated=False)
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
        fused_norm_rope_attn_rope_cast(*args, **buffers)
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


@pytest.mark.parametrize("heads", [64, 128])
@pytest.mark.parametrize("raw_q", [False, True])
def test_prefetch_long_invalid_decode(heads, raw_q):
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
    out = fused_norm_rope_attn_rope_cast(
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


@pytest.mark.parametrize("heads", [64, 128])
@pytest.mark.parametrize("tokens", [1023, 1024, 1025])
def test_prefetch_dispatch_boundary(heads, tokens):
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
    out = fused_norm_rope_attn_rope_cast(
        q, pool, ids, ptr, cos, sin, pos, q_is_rotated=False
    )
    value = dequantize(*out[:2])
    torch.testing.assert_close(
        value, row.float()[None, :, :].expand_as(value), rtol=0, atol=0
    )
    torch.testing.assert_close(
        out[3], torch.full_like(out[3], 65.0).log(), rtol=1e-6, atol=1e-6
    )


@pytest.mark.parametrize("tokens", [2047, 2048, 2049])
@pytest.mark.parametrize("raw_q", [False, True])
def test_h128_load_schedule_dispatch_boundary(tokens, raw_q):
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
    out = fused_norm_rope_attn_rope_cast(
        q, pool, ids, ptr, cos, sin, pos, q_is_rotated=not raw_q
    )
    actual = dequantize(*out[:2])
    torch.testing.assert_close(
        actual, row.float()[None, :, :].expand_as(actual), rtol=0, atol=0
    )
    torch.testing.assert_close(out[2], torch.zeros_like(out[2]), rtol=0, atol=0)
    torch.testing.assert_close(
        out[3], torch.full_like(out[3], 33.0).log(), rtol=1e-6, atol=1e-6
    )


# Paged cache.


@pytest.mark.parametrize("heads", [8, 16, 32, 64, 128])
@pytest.mark.parametrize("raw,q_norm", [(False, False), (True, False), (True, True)])
@pytest.mark.parametrize("extend", [False, True])
@pytest.mark.parametrize(
    "layout", [(32, False, 0), (128, False, 512), (32, True, 68), (33, True, 4)]
)
def test_paged_dense(q_norm, heads, raw, extend, layout):
    torch.manual_seed(16200 + heads)
    data, kw = make_paged_cache(block=layout[0], separate=layout[1], padding=layout[2])
    decoded = decode_pages(data, kw)
    rows, clean = [], []
    for n in [0, 1, 31, 32, 33, 65, 127, 1153]:
        ids = torch.randperm(len(decoded), device="cuda")[:n].int()
        j = torch.arange(n, device="cuda")
        ids[(j >= 32) & (j < 64)] = -1
        if n > 2:
            ids[-2] = 2**31 - 1
            ids[-1] = len(decoded) - 1
        rows.append(ids)
        clean.append(ids[(ids >= 0) & (ids < len(decoded))])
    ids, ptr = csr(rows, torch.int32, 2**31 - 1)
    q = torch.randn(8, heads, 512, device="cuda", dtype=torch.bfloat16) * 0.5
    angles = torch.randn(64, 32, device="cuda")
    cos, sin = angles.cos().bfloat16(), angles.sin().bfloat16()
    pos = torch.arange(8, device="cuda", dtype=torch.int32) * 7
    sink = torch.linspace(-3, 3, heads, device="cuda")
    sink[::11] = -torch.inf
    extra = erows = None
    if extend:
        extra = torch.randn(129, 512, device="cuda", dtype=torch.bfloat16) * 0.5
        erows = [
            torch.arange(n, device="cuda", dtype=torch.int32)
            for n in [0, 1, 31, 32, 33, 65, 127, 129]
        ]
        eids, eptr = csr(erows, torch.int32, 2**30)
        kw.update(extend_kv=extra, extend=eids, extend_indptr=eptr)
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
    expected, maxima, lses = dense_bf16_reference(
        qr, decoded, clean, cos, sin, pos, sink, extra, erows
    )
    debug = torch.empty_like(q, dtype=torch.float32)
    out = fused_norm_rope_attn_rope_cast(
        q,
        data,
        ids,
        ptr,
        cos,
        sin,
        pos,
        q_is_rotated=not raw,
        q_norm=q_norm,
        norm_eps=norm_eps,
        attn_sink=sink,
        debug_out=debug,
        **kw,
    )
    torch.testing.assert_close(debug, expected, rtol=0.02, atol=0.008)
    assert float((debug - expected).norm() / expected.norm()) < 0.005
    torch.testing.assert_close(out[2], maxima, rtol=1e-4, atol=3e-5)
    torch.testing.assert_close(out[3], lses, rtol=1e-4, atol=3e-5)
    assert float((dequantize(*out[:2]) - expected).norm() / expected.norm()) < 0.035


@pytest.mark.parametrize("heads", [8, 64, 128])
@pytest.mark.parametrize("tokens", [1024, 1025, 2047, 2048, 2049])
@pytest.mark.parametrize("raw", [False, True])
def test_paged_dispatch_stream_replay(heads, tokens, raw):
    torch.manual_seed(heads + tokens)
    args = simple_inputs(tokens, heads)
    data, kw = make_paged_cache(65, 32, True, 4)
    decoded = decode_pages(data, kw)
    args[1] = data
    args[2] = (torch.arange(tokens, device="cuda") % 65).int()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        out = fused_norm_rope_attn_rope_cast(*args, **kw, q_is_rotated=not raw)
        buffers = dict(zip(("out", "out_scale", "max_logits", "lse"), out))
        expected = [x.clone() for x in out]
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            fused_norm_rope_attn_rope_cast(*args, **kw, **buffers, q_is_rotated=not raw)
        for _ in range(3):
            for x in out:
                x.view(torch.uint8).fill_(255)
            graph.replay()
            for i, x in enumerate(out):
                torch.testing.assert_close(
                    x.view(torch.uint8), expected[i].view(torch.uint8), rtol=0, atol=0
                )
        quant = dequantize(*out[:2])
        ref = decoded[args[2].long()][:, None, :].expand_as(args[0]).float()
        torch.testing.assert_close(quant, ref, rtol=0, atol=0)
        assert torch.count_nonzero(out[2]) == torch.count_nonzero(out[3]) == 0
    torch.cuda.current_stream().wait_stream(stream)


@pytest.mark.parametrize("heads", [8, 64, 128])
@pytest.mark.parametrize(
    "bad", ["data_short", "scale_short", "row_limit", "huge_stride", "empty"]
)
def test_paged_bounds(heads, bad):
    args = simple_inputs(4, heads)
    data, kw = make_paged_cache(64, 32, True, 0)
    args[2] = torch.tensor([0, 31, 32, 63], device="cuda", dtype=torch.int32)
    if bad == "data_short":
        data = data[: 32 * 512 - 1]
    elif bad == "scale_short":
        kw["cache_scales"] = kw["cache_scales"][: 32 * 16 - 1]
    elif bad == "row_limit":
        kw["cache_num_rows"] = 1
    elif bad == "huge_stride":
        kw["cache_data_page_stride"] = 2**63 - 4
        args[2][-1] = 2**31 - 1
        kw["cache_num_rows"] = 2**31 - 1
    else:
        data = data[:0]
        kw["cache_scales"] = kw["cache_scales"][:0]
    args[1] = data
    debug = torch.empty_like(args[0], dtype=torch.float32)
    out = fused_norm_rope_attn_rope_cast(*args, **kw, debug_out=debug)
    live = []
    for s in args[2].tolist():
        p, r = divmod(s, kw["cache_page_size"])
        live.append(
            0 <= s < kw["cache_num_rows"]
            and p * kw["cache_data_page_stride"] + (r + 1) * 512 <= data.numel()
            and p * kw["cache_scale_page_stride"] + (r + 1) * 16
            <= kw["cache_scales"].numel()
        )
    for i, valid in enumerate(live):
        if valid:
            assert torch.isfinite(debug[i]).all()
            assert torch.count_nonzero(debug[i]) > 0
            assert torch.count_nonzero(out[2][i]) == torch.count_nonzero(out[3][i]) == 0
        else:
            assert torch.count_nonzero(debug[i]) == 0
            assert torch.isneginf(out[2][i]).all() and torch.isposinf(out[3][i]).all()


@pytest.mark.parametrize(
    "bad",
    [
        "missing_scales",
        "legacy_metadata",
        "slot_dtype",
        "scale_dtype",
        "scale_alignment",
        "scale_stride",
        "short_page",
        "negative_rows",
    ],
)
def test_paged_metadata(bad):
    args = simple_inputs()
    data, kw = make_paged_cache(32)
    args[1], args[2] = data, torch.zeros(1, device="cuda", dtype=torch.int32)
    if bad == "missing_scales":
        kw["cache_scales"] = None
    elif bad == "legacy_metadata":
        kw["cache_page_size"] = 0
    elif bad == "slot_dtype":
        args[2] = args[2].long()
    elif bad == "scale_dtype":
        kw["cache_scales"] = kw["cache_scales"].float()
    elif bad == "scale_alignment":
        kw["cache_scales"] = kw["cache_scales"][1:]
    elif bad == "scale_stride":
        kw["cache_scales"] = kw["cache_scales"][::2]
    elif bad == "short_page":
        kw["cache_data_page_stride"] = 511
    else:
        kw["cache_num_rows"] = -1
    with pytest.raises((RuntimeError, ValueError)):
        fused_norm_rope_attn_rope_cast(*args, **kw)


def test_paged_large_row_offset():
    # The selected row starts beyond signed int32 bytes, within one large page.
    # Only the selected bytes are initialized; other rows must never be read.
    slot = 2**22 + 1
    block = 2**23
    data = torch.empty((slot + 1) * 512, device="cuda", dtype=torch.uint8)
    scales = torch.empty((slot + 1) * 16, device="cuda", dtype=torch.uint8)
    data[-512:] = 0x38  # E4M3FN 1
    scales[-16:] = 127  # E8M0 1
    args = simple_inputs()
    args[1] = data
    args[2] = torch.tensor([slot], device="cuda", dtype=torch.int32)
    debug = torch.empty_like(args[0], dtype=torch.float32)
    out = fused_norm_rope_attn_rope_cast(
        *args,
        cache_scales=scales,
        cache_page_size=block,
        cache_num_rows=slot + 1,
        cache_data_page_stride=block * 512,
        cache_scale_page_stride=block * 16,
        debug_out=debug,
    )
    torch.testing.assert_close(debug, torch.ones_like(debug), rtol=0, atol=0)
    assert torch.count_nonzero(out[2]) == torch.count_nonzero(out[3]) == 0


# BF16 cache.


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
    expected, maxima, lses = dense_bf16_reference(qr, kv, clean, cos, sin, pos, sink)
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


# FP8 QK and prepared Q.


@pytest.mark.parametrize("heads", [64, 128])
@pytest.mark.parametrize("page", [0, 7, 32, 128])
@pytest.mark.parametrize("raw_q,q_norm", [(False, False), (True, False), (True, True)])
def test_fp8_qk_dense(q_norm, heads, page, raw_q):
    args, kw, rows, kv, sink = make_fp8_qk_inputs(heads, page, raw_q)
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
    args, kw, _, _, sink = make_fp8_qk_inputs(128, 128, True, seed=821)
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
    args, kw, _, _, sink = make_fp8_qk_inputs(heads, page, True)
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


# Prefill integration.


@pytest.mark.parametrize("page", [0, 128])
@pytest.mark.parametrize("raw,q_norm", [(False, False), (True, False), (True, True)])
@pytest.mark.parametrize("pos64", [False, True])
@pytest.mark.parametrize("mode", ["mixed", "valid", "truncated"])
def test_fused_norm_rope_attn_rope_cast_contract(q_norm, page, raw, pos64, mode):
    args, kw, rows, kv, sink = make_fp8_qk_inputs(128, page, raw)
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


# Optional Q RMSNorm.


@pytest.mark.parametrize("direct", [False, True])
@pytest.mark.parametrize(
    "bad", ["rotated", "prepared", "zero", "negative", "nan", "infinity", "underflow"]
)
def test_norm_contract(direct, bad):
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
    with pytest.raises((ValueError, RuntimeError), match="q_norm|norm_eps"):
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
            fused_norm_rope_attn_rope_cast(*args, **kw)


@pytest.mark.parametrize("tokens", [0, 3])
def test_norm_zero_q(tokens):
    args = simple_inputs(tokens)
    out = fused_norm_rope_attn_rope_cast(*args, q_norm=True, q_is_rotated=False)
    assert all(torch.isfinite(x.float()).all() for x in out)
    assert torch.count_nonzero(out[2]) == torch.count_nonzero(out[3]) == 0


@pytest.mark.parametrize("page", [0, 128])
def test_norm_recomputed_on_graph_replay(page):

    args, kw, rows, kv, sink = make_fp8_qk_inputs(128, page, True)
    q, pool, _, _, cos, sin, pos = args
    rows = [
        (torch.arange(1153, device=q.device) * 17 + i) % len(kv)
        for i in range(len(rows))
    ]
    ids, ptr = csr(rows, torch.int32, -1)
    if not page:
        ids = ids.long() * 1056 + 9
    args = (q, pool, ids, ptr, cos, sin, pos)
    kw.update(
        q_norm=True, norm_eps=0.125, q_is_rotated=False, fp8_qk=True, attn_sink=sink
    )
    out = fused_norm_rope_attn_rope_cast(*args, **kw)
    buffers = dict(zip(("out", "out_scale", "max_logits", "lse"), out))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fused_norm_rope_attn_rope_cast(*args, **kw, **buffers)
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


@perftest()
def run_benchmark(args, buffers):
    return fused_norm_rope_attn_rope_cast(*args, **buffers)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--tokens", type=int, default=560)
    parser.add_argument("--heads", type=int, choices=[8, 16, 32, 64, 128], default=64)
    parser.add_argument("--kv", type=int, default=640)
    options, pytest_args = parser.parse_known_args()
    if options.benchmark:
        args = simple_inputs(options.tokens, options.heads)
        args[2] = args[2].repeat_interleave(options.kv)
        args[3] *= options.kv
        out = fused_norm_rope_attn_rope_cast(*args)
        buffers = {
            "out": out[0],
            "out_scale": out[1],
            "max_logits": out[2],
            "lse": out[3],
        }
        _, us = run_benchmark(args, buffers)
        print(f"T={options.tokens} H={options.heads} K={options.kv}: {us:.3f} us")
        print(
            "Synthetic repeated-row fixture; producer and layout preparation excluded."
        )
    else:
        raise SystemExit(pytest.main([__file__, "-q", *pytest_args]))
