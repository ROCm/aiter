# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Native BF16 compute/cache regressions for gfx1250 planned PA decode."""

import importlib.util

import pytest
import torch

from op_tests.test_flydsl_pa_decode import _make_inputs, _reference


@pytest.fixture
def device(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("ROCm is not available")
    if torch.cuda.get_device_properties(0).gcnArchName.split(":")[0] != "gfx1250":
        pytest.skip("requires gfx1250")
    if importlib.util.find_spec("flydsl") is None:
        pytest.skip("FlyDSL is not installed")
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    return torch.device("cuda")


def _inputs(
    device,
    length,
    *,
    query_length=1,
    group=1,
    dim=128,
    trans_v=True,
    strides=False,
    page_ids=None,
    pool_pages=None,
    page=128,
    divisor=8,
    cache_dtype=torch.float8_e4m3fn,
):
    pages = (length + page - 1) // page
    if page_ids is None:
        page_ids = torch.randperm(pages, device=device, dtype=torch.int32)
    table = page_ids.reshape(1, pages)
    pool_pages = pages if pool_pages is None else pool_pages
    # Representable values isolate lane/operand mapping from FP8 cache rounding.
    torch.manual_seed(17)
    key = torch.randint(-4, 5, (pages, 1, page, dim), device=device).float() / divisor
    value = torch.randint(-4, 5, key.shape, device=device).float() / divisor
    qshape = (query_length, group, dim)
    query = (torch.randint(-4, 5, qshape, device=device).float() / divisor).bfloat16()
    if strides:
        query_view = torch.empty_strided(
            qshape,
            (group * (dim + 1) + 3, dim + 1, 1),
            dtype=query.dtype,
            device=device,
        )
        query_view.copy_(query)
        query = query_view
    chunk = 16 // torch.empty((), dtype=cache_dtype).element_size()
    cache_key = torch.empty(
        (pool_pages, 1, dim // chunk, page, chunk), dtype=cache_dtype, device=device
    )
    vshape = (
        (pool_pages, 1, page // chunk, dim, chunk)
        if trans_v
        else (pool_pages, 1, dim, page)
    )
    cache_value = torch.empty(vshape, dtype=cache_dtype, device=device)
    packed_key = (
        key.reshape(pages, 1, page, dim // chunk, chunk)
        .permute(0, 1, 3, 2, 4)
        .contiguous()
        .to(cache_key.dtype)
    )
    packed_value = (
        (
            value.reshape(pages, 1, page // chunk, chunk, dim).permute(0, 1, 2, 4, 3)
            if trans_v
            else value.permute(0, 1, 3, 2)
        )
        .contiguous()
        .to(cache_value.dtype)
    )
    # FP8 index_copy is not implemented by all bring-up PyTorch builds.
    for logical, physical in enumerate(page_ids.cpu().tolist()):
        cache_key[physical].copy_(packed_key[logical])
        cache_value[physical].copy_(packed_value[logical])
    if pool_pages == pages:
        reference_key, reference_value = torch.empty_like(key), torch.empty_like(value)
        reference_key[page_ids.long()] = key
        reference_value[page_ids.long()] = value
        key, value = reference_key, reference_value
    lengths = torch.tensor([length], dtype=torch.int32, device=device)
    return query, key, value, cache_key, cache_value, table, lengths


def _pack(key, value, trans_v):
    pages, heads, page, dim = key.shape
    kc = (
        key.reshape(pages, heads, page, dim // 8, 8).permute(0, 1, 3, 2, 4).contiguous()
    )
    vc = (
        value.reshape(pages, heads, page // 8, 8, dim).permute(0, 1, 2, 4, 3)
        if trans_v
        else value.permute(0, 1, 3, 2)
    ).contiguous()
    return kc, vc


@pytest.mark.parametrize(
    "query_dtype", [torch.bfloat16, torch.float16], ids=["bf16-q", "fp16-q"]
)
@pytest.mark.parametrize("trans_v", [False, True], ids=["plain-v", "transposed-v"])
@pytest.mark.parametrize(
    "ql,heads,group,dim,page,parts,window,pattern",
    [
        (1, 1, 8, 128, 16, 1, 0, "random"),
        (3, 2, 4, 64, 64, 3, 0, "random"),
        (4, 1, 16, 128, 128, 7, 257, "random"),
        (2, 2, 8, 256, 16, 7, 0, "random"),
        (1, 1, 4, 384, 64, 1, 0, "random"),
        (2, 1, 8, 512, 64, 3, 0, "random"),
        (1, 1, 4, 768, 128, 1, 0, "random"),
        (1, 1, 4, 1024, 128, 1, 0, "random"),
        (1, 1, 8, 128, 16, 1, 0, "small_query"),
        (1, 1, 16, 128, 128, 1, 0, "probability_tail"),
    ],
)
def test_bf16_cache(
    ql, heads, group, dim, page, parts, window, pattern, query_dtype, trans_v, device
):
    from aiter.ops.flydsl.pa_decode import pa_decode, plan_pa_decode

    q, k, v, table, lengths = _make_inputs(
        ql, heads, group, dim, page, torch.bfloat16, pattern, device
    )
    q = q.to(query_dtype)
    kc, vc = _pack(k, v, trans_v)
    sinks = torch.linspace(-2, 2, q.shape[1], dtype=torch.float32, device=device)
    sinks[0], sinks[-1] = float("-inf"), float("inf")
    plan = plan_pa_decode(
        lengths, heads, max_partitions=parts, query_length=ql, sliding_window=window
    )
    out = torch.full_like(q, float("nan"))
    # Default compute_type is inferred from the cache dtype; no scale buffers
    # are allocated or read for native BF16.
    pa_decode(
        out,
        q,
        kc,
        vc,
        lengths,
        table,
        dim**-0.5,
        ql,
        sinks=sinks,
        sliding_window=window,
        work_plan=plan,
    )
    expected = _reference(q, k.float(), v.float(), table, lengths, ql, window, sinks)
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out.float(), expected, atol=0.002, rtol=0.01)


@pytest.mark.parametrize("trans_v", [False, True])
def test_bf16_graph_refresh(trans_v, device):
    from aiter.ops.flydsl.pa_decode import pa_decode, plan_pa_decode

    q, k, v, kc, vc, table, lengths = _inputs(
        device,
        1027,
        query_length=4,
        group=8,
        trans_v=trans_v,
        strides=True,
        cache_dtype=torch.bfloat16,
    )
    plan = plan_pa_decode(
        lengths, 1, max_partitions=8, sliding_window=257, query_length=4
    )
    rows = 32
    sums = torch.empty((1, plan.capacity, rows), device=device)
    maxima = torch.empty_like(sums)
    partial = torch.empty((*sums.shape, 128), dtype=q.dtype, device=device)
    backing = torch.empty(4 * (8 * 129 + 3) + 1, dtype=q.dtype, device=device)
    out = backing[1:].as_strided(q.shape, (8 * 129 + 3, 129, 1))

    def launch():
        pa_decode(
            out,
            q,
            kc,
            vc,
            lengths,
            table,
            128**-0.5,
            4,
            compute_type=torch.bfloat16,
            work_plan=plan,
            sliding_window=257,
            exp_sums=sums,
            max_logits=maxima,
            temporary_output=partial,
        )

    launch()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        launch()
    torch.cuda.current_stream().wait_stream(stream)
    for length in (1027, 1024, 1023, 257, 256, 255, 1, 0, 1027):
        lengths.fill_(length)
        plan_pa_decode(lengths, 1, plan=plan, sliding_window=257, query_length=4)
        sums.fill_(float("nan"))
        maxima.fill_(float("nan"))
        partial.fill_(float("nan"))
        graph.replay()
        expected = _reference(q, k, v, table, lengths, 4, 257, None)
        torch.testing.assert_close(out.float(), expected, atol=0.002, rtol=0.01)


@pytest.mark.parametrize("boundary", [2**31, 2**32], ids=["2GiB", "4GiB"])
@pytest.mark.parametrize("trans_v", [False, True])
def test_bf16_wide_addresses(boundary, trans_v, device):
    from aiter.ops.flydsl.pa_decode import pa_decode, plan_pa_decode

    page_bytes = 128 * 128 * 2
    high_page = boundary // page_bytes
    pool_pages = high_page + 2
    if torch.cuda.mem_get_info(device)[0] < 2 * pool_pages * page_bytes + 2**30:
        pytest.skip("insufficient memory for wide-address test")
    ids = torch.tensor([high_page + 1, high_page], dtype=torch.int32, device=device)
    q, k, v, kc, vc, table, lengths = _inputs(
        device,
        255,
        cache_dtype=torch.bfloat16,
        trans_v=trans_v,
        page_ids=ids,
        pool_pages=pool_pages,
    )
    plan = plan_pa_decode(lengths, 1, max_partitions=1)
    out = torch.empty_like(q)
    pa_decode(out, q, kc, vc, lengths, table, 128**-0.5, 1, work_plan=plan)
    expected = _reference(
        q,
        k,
        v,
        torch.arange(2, device=device, dtype=torch.int32).reshape(1, 2),
        lengths,
        1,
        0,
        None,
    )
    torch.testing.assert_close(out.float(), expected, atol=0.002, rtol=0.01)


def test_bf16_reject_scales(device):
    from aiter.ops.flydsl.pa_decode import pa_decode, plan_pa_decode

    q, _k, _v, kc, vc, table, lengths = _inputs(device, 128, cache_dtype=torch.bfloat16)
    plan = plan_pa_decode(lengths, 1, max_partitions=1)
    with pytest.raises(ValueError, match="does not use key_scale"):
        pa_decode(
            torch.empty_like(q),
            q,
            kc,
            vc,
            lengths,
            table,
            128**-0.5,
            1,
            key_scale=torch.ones(1, device=device),
            work_plan=plan,
        )


@pytest.mark.parametrize("trans_v", [False, True])
def test_bf16_large_mtp(trans_v, device):
    from aiter.ops.flydsl.pa_decode import pa_decode, plan_pa_decode

    # BF16 D1024/MTP4/G16 needs one K/V buffer and reuses one Q tile.
    q, k, v, kc, vc, table, lengths = _inputs(
        device,
        257,
        query_length=4,
        group=16,
        dim=1024,
        trans_v=trans_v,
        cache_dtype=torch.bfloat16,
    )
    plan = plan_pa_decode(lengths, 1, max_partitions=1, query_length=4)
    out = torch.empty_like(q)
    pa_decode(out, q, kc, vc, lengths, table, 1024**-0.5, 4, work_plan=plan)
    expected = _reference(q, k, v, table, lengths, 4, 0, None)
    torch.testing.assert_close(out.float(), expected, atol=0.002, rtol=0.01)
