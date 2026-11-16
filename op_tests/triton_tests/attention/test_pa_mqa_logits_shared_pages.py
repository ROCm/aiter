# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch

from aiter.ops.triton._triton_kernels.attention.pa_mqa_logits_shared_pages import (
    _pa_mqa_logits_shared_pages_kernel,
)
from aiter.ops.triton.attention.pa_mqa_logits_shared_pages import (
    has_launch_config,
    pa_mqa_logits_shared_pages,
)
from aiter.ops.triton.utils._triton import arch_info

pytestmark = pytest.mark.skipif(
    not has_launch_config(), reason="no launch config for this arch"
)

FP8_NEG_ZERO = 0x80


def _fp8(shape, dtype):
    x = torch.randn(shape, dtype=torch.float32, device="cuda").to(dtype)
    if dtype != torch.float8_e4m3fn:
        return x
    # -0.0 in e4m3fn has the NaN encoding of e4m3fnuz.
    neg_zero = torch.rand(shape, dtype=torch.float32, device="cuda") < 0.01
    return x.view(torch.uint8).masked_fill(neg_zero, FP8_NEG_ZERO).view(dtype)


def make_inputs(
    batch,
    next_n,
    heads,
    page_size,
    max_len,
    layout,
    head_dim=256,
    dtype=torch.float8_e4m3fn,
    seed=0,
):
    """Random inputs in the packed page layout.

    ``layout`` sets the page mapping and the lengths:

    - ``"shared"``: all requests use the same pages, with causal lengths, as
      the rows of one prefill chunk do.
    - ``"mixed"``: two owners, so that one row block maps to two page sets.
    - ``"distinct"``: each request has its own pages and a random length.
    """
    torch.manual_seed(seed)
    pages_per_req = -(-max_len // page_size)
    if layout == "shared":
        owners = torch.zeros(batch, dtype=torch.int64, device="cuda")
    elif layout == "mixed":
        owners = torch.arange(batch, device="cuda") * 2 // max(batch, 1)
    else:
        owners = torch.arange(batch, device="cuda")
    num_owners = {"shared": 1, "mixed": 2}.get(layout, batch)
    num_pages = num_owners * pages_per_req
    owner_pages = torch.randperm(num_pages, device="cuda").view(
        num_owners, pages_per_req
    )
    block_tables = owner_pages[owners].to(torch.int32)

    if layout == "distinct":
        lengths = torch.randint(
            next_n, max_len + 1, (batch,), dtype=torch.int32, device="cuda"
        )
    else:
        lengths = torch.linspace(
            next_n, max_len, batch, dtype=torch.float32, device="cuda"
        ).to(torch.int32)

    keys = _fp8((num_pages, page_size * head_dim), dtype).view(torch.uint8)
    scales = torch.rand((num_pages, page_size), dtype=torch.float32, device="cuda")
    kv_cache = torch.cat([keys, scales.view(torch.uint8)], dim=1).view(
        num_pages, page_size, 1, head_dim + 4
    )
    q = _fp8((batch, next_n, heads, head_dim), dtype)
    weights = torch.randn((batch * next_n, heads), dtype=torch.float32, device="cuda")
    return q, kv_cache, weights, lengths, block_tables


def reference(q, kv_cache, weights, lengths, block_tables, max_len):
    batch, next_n, heads, head_dim = q.shape
    num_pages, page_size = kv_cache.shape[:2]
    packed = kv_cache.view(num_pages, -1)
    keys = packed[:, : page_size * head_dim].view(q.dtype).double()
    keys = keys.view(num_pages, page_size, head_dim)
    scales = packed[:, page_size * head_dim :].view(torch.float32).double()

    k = torch.arange(max_len, device=q.device)
    page = block_tables.long()[:, k // page_size]
    key = keys[page, k % page_size]
    scale = scales[page, k % page_size]

    dots = torch.einsum("bnhd,bkd->bnhk", q.double(), key).relu()
    w = weights.double().reshape(batch, next_n, heads, 1)
    logits = (dots * w).sum(dim=2) * scale[:, None, :]
    visible = lengths.reshape(batch, -1)[:, -1:] - next_n + 1
    visible = visible + torch.arange(next_n, device=q.device)
    logits = logits.masked_fill(k >= visible[:, :, None], float("-inf"))
    return logits.reshape(batch * next_n, max_len).float()


def run(q, kv_cache, weights, lengths, block_tables, max_len, width=None):
    out = torch.full(
        (q.shape[0] * q.shape[1], width or max_len),
        float("-inf"),
        dtype=torch.float32,
        device="cuda",
    )
    pa_mqa_logits_shared_pages(
        q, kv_cache, weights, out, lengths, block_tables, max_len
    )
    return out[:, :max_len]


def check(inputs, max_len, **kwargs):
    out = run(*inputs, max_len, **kwargs)
    torch.testing.assert_close(out, reference(*inputs, max_len), rtol=1e-3, atol=1e-2)


@pytest.mark.parametrize("heads", [4, 8, 16])
@pytest.mark.parametrize("page_size", [16, 32, 48, 256])
@pytest.mark.parametrize(
    "batch, next_n, max_len, layout",
    [
        (64, 1, 1500, "shared"),
        (37, 1, 300, "mixed"),
        (19, 1, 777, "distinct"),
        (9, 3, 1100, "distinct"),
    ],
)
@torch.inference_mode()
def test_pa_mqa_logits_shared_pages(heads, page_size, batch, next_n, max_len, layout):
    inputs = make_inputs(batch, next_n, heads, page_size, max_len, layout)
    check(inputs, max_len)


@pytest.mark.parametrize("head_dim", [128, 256])
@pytest.mark.parametrize(
    "dtype",
    [
        torch.float8_e4m3fn,
        pytest.param(
            torch.float8_e4m3fnuz,
            marks=pytest.mark.skipif(
                arch_info.get_arch() != "gfx942", reason="e4m3fnuz is gfx942 only"
            ),
        ),
    ],
)
@torch.inference_mode()
def test_pa_mqa_logits_shared_pages_dtype(head_dim, dtype):
    inputs = make_inputs(40, 1, 8, 64, 900, "mixed", head_dim=head_dim, dtype=dtype)
    check(inputs, 900)


@torch.inference_mode()
def test_pa_mqa_logits_shared_pages_layouts():
    batch, groups, heads, max_len = 40, 2, 8, 1000
    q, kv_cache, weights, lengths, block_tables = make_inputs(
        batch, groups, heads, 32, max_len, "shared"
    )
    # One query group of a [tokens, groups, heads, dim] tensor, a (B, next_n)
    # lengths table, and an output row wider than max_len.
    q = q[:, 1:]
    weights = weights.view(batch, groups, heads)[:, 1]
    lengths = lengths[:, None].contiguous()
    check((q, kv_cache, weights, lengths, block_tables), max_len, width=1100)

    next_n = 3
    q, kv_cache, weights, lengths, block_tables = make_inputs(
        batch, next_n, heads, 32, max_len, "distinct"
    )
    offsets = torch.arange(next_n, dtype=torch.int32, device="cuda")
    table = (lengths[:, None] - next_n + 1 + offsets).contiguous()
    check((q, kv_cache, weights, table, block_tables), max_len)


@torch.inference_mode()
def test_pa_mqa_logits_shared_pages_writes_only_visible_columns():
    q, kv_cache, weights, lengths, block_tables = make_inputs(
        32, 1, 8, 32, 500, "distinct"
    )
    inputs = (q, kv_cache, weights, lengths, block_tables)
    out = torch.full((32, 600), 7.0, dtype=torch.float32, device="cuda")
    pa_mqa_logits_shared_pages(q, kv_cache, weights, out, lengths, block_tables, 500)
    ref = reference(*inputs, 500)
    hidden = torch.isneginf(ref)
    assert (out[:, :500][hidden] == 7.0).all()
    assert (out[:, 500:] == 7.0).all()
    torch.testing.assert_close(
        out[:, :500][~hidden], ref[~hidden], rtol=1e-3, atol=1e-2
    )


@torch.inference_mode()
def test_pa_mqa_logits_shared_pages_request_boundary():
    # Two requests meet inside one row block, as at a prefill chunk boundary.
    q, kv_cache, weights, lengths, block_tables = make_inputs(
        64, 1, 8, 32, 3000, "distinct"
    )
    lengths = torch.full_like(lengths, 3000)
    block_tables[:37] = block_tables[0]
    check((q, kv_cache, weights, lengths, block_tables), 3000)


@torch.inference_mode()
def test_pa_mqa_logits_shared_pages_e4m3fn_path(monkeypatch):
    if arch_info.get_arch() != "gfx942":
        pytest.skip("the e4m3fn-as-e4m3fnuz path is gfx942 only")
    inputs = make_inputs(21, 1, 16, 32, 900, "mixed")
    check(inputs, 900)

    # Run the plain e4m3fn path of other archs. Triton emulates it on gfx942.
    monkeypatch.setattr(arch_info, "get_arch", lambda: "gfx950")
    check(inputs, 900)


@torch.inference_mode()
def test_pa_mqa_logits_shared_pages_one_binary_per_shape():
    def binaries():
        caches = _pa_mqa_logits_shared_pages_kernel.device_caches.values()
        return sum(len(c[0]) for c in caches)

    run(*make_inputs(8, 1, 8, 32, 50, "shared"), 50)
    before = binaries()
    for max_len in (100, 300, 700, 1000, 3000):
        run(*make_inputs(8, 1, 8, 32, max_len, "shared"), max_len)

    assert binaries() == before


@torch.inference_mode()
def test_pa_mqa_logits_shared_pages_graph_replay():
    # The launch must read the lengths on the device, so a replay follows them.
    q, kv_cache, weights, lengths, block_tables = make_inputs(
        64, 1, 8, 32, 2000, "mixed"
    )
    out = torch.full((64, 2000), float("-inf"), dtype=torch.float32, device="cuda")

    def launch():
        pa_mqa_logits_shared_pages(
            q, kv_cache, weights, out, lengths, block_tables, 2000
        )

    launch()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch()
    lengths.copy_(lengths // 2)
    out.fill_(float("-inf"))
    graph.replay()

    ref = reference(q, kv_cache, weights, lengths, block_tables, 2000)
    torch.testing.assert_close(out, ref, rtol=1e-3, atol=1e-2)


@torch.inference_mode()
def test_pa_mqa_logits_shared_pages_empty():
    q, kv_cache, weights, lengths, block_tables = make_inputs(
        0, 1, 16, 32, 64, "distinct"
    )
    out = torch.empty((0, 64), dtype=torch.float32, device="cuda")

    result = pa_mqa_logits_shared_pages(
        q, kv_cache, weights, out, lengths, block_tables, 64
    )

    assert result is out
