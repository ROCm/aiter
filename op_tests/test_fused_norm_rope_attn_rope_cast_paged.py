# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Independent paged-cache decoding, ragged attention, bounds and replay checks."""

import pytest
import torch
from test_fused_norm_rope_attn_rope_cast import csr, dequantize, rotate, simple_inputs

from aiter import fused_norm_rope_attn_rope_cast

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or not torch.cuda.get_device_properties(0).gcnArchName.startswith("gfx950"),
    reason="requires gfx950",
)


def make_cache(n=1201, block=32, separate=False, padding=0):
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


def dense(q, decoded, rows, cos, sin, pos, sink, extra=None, erows=None):
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


@pytest.mark.parametrize("heads", [8, 16, 32, 64, 128])
@pytest.mark.parametrize("raw,q_norm", [(False, False), (True, False), (True, True)])
@pytest.mark.parametrize("extend", [False, True])
@pytest.mark.parametrize(
    "layout", [(32, False, 0), (128, False, 512), (32, True, 68), (33, True, 4)]
)
def test_paged_dense(q_norm, heads, raw, extend, layout):
    torch.manual_seed(16200 + heads)
    data, kw = make_cache(block=layout[0], separate=layout[1], padding=layout[2])
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
    expected, maxima, lses = dense(
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
        **kw
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
    data, kw = make_cache(65, 32, True, 4)
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
    data, kw = make_cache(64, 32, True, 0)
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
    data, kw = make_cache(32)
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
        debug_out=debug
    )
    torch.testing.assert_close(debug, torch.ones_like(debug), rtol=0, atol=0)
    assert torch.count_nonzero(out[2]) == torch.count_nonzero(out[3]) == 0
