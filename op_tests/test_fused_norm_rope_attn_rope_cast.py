# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Independent dense oracle and replay/dispatch regressions for DSV4.1 fusion.

Run directly, or with pytest. No ATOM checkout or cache extension is required.
"""

import argparse
from types import SimpleNamespace

import pytest
import torch

from aiter import fused_norm_rope_attn_rope_cast
from aiter.test_common import perftest

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or not torch.cuda.get_device_properties(0).gcnArchName.startswith("gfx950"),
    reason="requires gfx950",
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


def oracle(q, pool, ids, ptr, cos, sin, pos, sink, extra=None, eids=None, eptr=None):
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


fused = SimpleNamespace(prefill=fused_norm_rope_attn_rope_cast, dequantize=dequantize)


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
    expected, maxima, lses = oracle(
        qr, pool, cids, cptr, cos, sin, pos, sink, extra, ceids, ceptr
    )
    debug = torch.empty_like(q, dtype=torch.float32)
    out = fused.prefill(
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
        fused.prefill(*args, **kw)


def test_zero_tokens():
    out = fused.prefill(*simple_inputs(0))
    assert all(x.numel() == 0 for x in out)


def test_nondefault_stream_and_buffers():
    args = simple_inputs(5)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        out = fused.prefill(*args)
        again = fused.prefill(
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
    out = fused.prefill(*args, debug_out=debug, q_is_rotated=False)
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
        fused.prefill(*args, **buffers)
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
    expected, maxima, lses = oracle(qr, pool, cids, cptr, cos, sin, pos, sink)
    debug = torch.empty_like(q, dtype=torch.float32)
    out = fused.prefill(
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
    out = fused.prefill(q, pool, ids, ptr, cos, sin, pos, q_is_rotated=False)
    value = fused.dequantize(*out[:2])
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
    out = fused.prefill(q, pool, ids, ptr, cos, sin, pos, q_is_rotated=not raw_q)
    actual = fused.dequantize(*out[:2])
    torch.testing.assert_close(
        actual, row.float()[None, :, :].expand_as(actual), rtol=0, atol=0
    )
    torch.testing.assert_close(out[2], torch.zeros_like(out[2]), rtol=0, atol=0)
    torch.testing.assert_close(
        out[3], torch.full_like(out[3], 33.0).log(), rtol=1e-6, atol=1e-6
    )


@perftest()
def run_benchmark(args, buffers):
    return fused.prefill(*args, **buffers)


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
        out = fused.prefill(*args)
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
