# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""gfx1250 PA decode device regressions beyond the shared correctness matrix."""

import importlib.util

import pytest
import torch

from op_tests.test_flydsl_pa_decode import _reference


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


@pytest.mark.parametrize("trans_v", [False, True], ids=["plain-v", "transposed-v"])
@pytest.mark.parametrize(
    "page,dim", [(16, 64), (16, 128), (64, 128), (128, 256), (16, 768), (128, 384)]
)
def test_dma_partial_pages(page, dim, trans_v, device):
    from aiter.ops.flydsl.pa_decode import pa_decode, plan_pa_decode

    # A tile crosses shuffled page boundaries and ends after one token of the
    # final page. Cover TDM and the non-power-of-two vector DMA fallback.
    q, k, v, kc, vc, table, lengths = _inputs(
        device,
        2 * page + 1,
        query_length=4,
        group=8,
        dim=dim,
        trans_v=trans_v,
        page=page,
        divisor=16,
    )
    plan = plan_pa_decode(lengths, 1, max_partitions=1, query_length=4)
    output = torch.full_like(q, float("nan"))
    scale = torch.ones((kc.shape[0], 1, page, 1), dtype=torch.float32, device=device)
    pa_decode(
        output,
        q,
        kc,
        vc,
        lengths,
        table,
        dim**-0.5,
        4,
        compute_type=kc.dtype,
        key_scale=scale,
        value_scale=scale,
        work_plan=plan,
    )
    expected = _reference(q, k, v, table, lengths, 4, 0, None)
    torch.testing.assert_close(output.float(), expected, atol=0.005, rtol=0.005)


@pytest.mark.parametrize("parts", [1, 4, 8, 32, 33, 64, 65, 256])
def test_partition_boundaries(parts, device):
    from aiter.ops.flydsl.pa_decode import pa_decode, plan_pa_decode

    if parts > torch.cuda.get_device_properties(device).multi_processor_count:
        pytest.skip("partition cap exceeds device CUs")
    q, k, v, kc, vc, table, lengths = _inputs(device, parts * 256, group=1)
    plan = plan_pa_decode(lengths, 1, max_partitions=parts, workgroup_budget=parts)
    assert plan.reduce_info[0, 1].item() == parts
    output = torch.full_like(q, float("nan"))
    scale = torch.ones((kc.shape[0], 1, 128, 1), dtype=torch.float32, device=device)
    pa_decode(
        output,
        q,
        kc,
        vc,
        lengths,
        table,
        128**-0.5,
        1,
        compute_type=kc.dtype,
        key_scale=scale,
        value_scale=scale,
        work_plan=plan,
    )
    expected = _reference(q, k, v, table, lengths, 1, 0, None)
    torch.testing.assert_close(output.float(), expected, atol=0.005, rtol=0.005)


@pytest.mark.parametrize("window", [0, 257])
@pytest.mark.parametrize("per_token", [False, True])
def test_graph_refresh(window, per_token, device):
    from aiter.ops.flydsl.pa_decode import pa_decode, plan_pa_decode

    q, k, v, kc, vc, table, lengths = _inputs(
        device, 1027, query_length=4, group=8, strides=True
    )
    plan = plan_pa_decode(
        lengths, 1, max_partitions=65, sliding_window=window, query_length=4
    )
    # Odd output strides and an offset ensure stores honor scalar alignment.
    backing = torch.empty(4 * (8 * 129 + 3) + 1, dtype=q.dtype, device=device)
    output = backing[1:].as_strided(q.shape, (8 * 129 + 3, 129, 1))
    rows = 4 * 8
    stats = torch.empty((1, plan.capacity, rows), dtype=torch.float32, device=device)
    maxima = torch.empty_like(stats)
    partial = torch.empty((*stats.shape, 128), dtype=q.dtype, device=device)
    scale = torch.ones(
        (kc.shape[0], 1, 128, 1) if per_token else (1,),
        dtype=torch.float32,
        device=device,
    )
    sinks = torch.linspace(-2, 2, 8, device=device)
    sinks[0], sinks[-1] = float("-inf"), float("inf")

    def launch():
        pa_decode(
            output,
            q,
            kc,
            vc,
            lengths,
            table,
            128**-0.5,
            4,
            compute_type=kc.dtype,
            key_scale=scale,
            value_scale=scale,
            exp_sums=stats,
            max_logits=maxima,
            temporary_output=partial,
            sinks=sinks,
            sliding_window=window,
            work_plan=plan,
            max_context_length=1027,
        )

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        launch()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        launch()
    pointers = plan.work_info.data_ptr(), plan.reduce_info.data_ptr()
    for length in (1027, 1024, 1023, 257, 256, 255, 1, 0, 1027):
        lengths.fill_(length)
        refreshed = plan_pa_decode(
            lengths, 1, plan=plan, sliding_window=window, query_length=4
        )
        assert refreshed is plan
        assert pointers == (plan.work_info.data_ptr(), plan.reduce_info.data_ptr())
        # Poison inactive scratch to expose accidental reads of cleared tasks.
        stats.fill_(float("nan"))
        maxima.fill_(float("nan"))
        partial.fill_(float("nan"))
        graph.replay()
        expected = _reference(q, k, v, table, lengths, 4, window, sinks)
        torch.testing.assert_close(output.float(), expected, atol=0.005, rtol=0.005)


@pytest.mark.parametrize("boundary", [2**31, 2**32], ids=["2GiB", "4GiB"])
@pytest.mark.parametrize("trans_v", [False, True], ids=["plain-v", "transposed-v"])
@pytest.mark.parametrize("per_token", [False, True])
def test_wide_cache_addresses(boundary, trans_v, per_token, device):
    from aiter.ops.flydsl.pa_decode import pa_decode, plan_pa_decode

    # Allocate lazily and touch only the high pages; cached offsets must retain
    # their high bits for both the K and V physical layouts.
    bytes_per_page = 128 * 128
    high_page = boundary // bytes_per_page
    pool_pages = high_page + 2
    if torch.cuda.mem_get_info(device)[0] < 2 * pool_pages * bytes_per_page + 2**30:
        pytest.skip("insufficient free VRAM for wide-address regression")
    ids = torch.tensor([high_page + 1, high_page], dtype=torch.int32, device=device)
    q, k, v, kc, vc, table, lengths = _inputs(
        device, 255, trans_v=trans_v, page_ids=ids, pool_pages=pool_pages
    )
    # Reference uses the logical page order, independently of the large pool.
    logical_table = torch.arange(2, device=device, dtype=torch.int32).reshape(1, 2)
    plan = plan_pa_decode(lengths, 1, max_partitions=1)
    scale = torch.ones(
        (kc.shape[0], 1, 128, 1) if per_token else (1,),
        dtype=torch.float32,
        device=device,
    )
    output = torch.full_like(q, float("nan"))
    pa_decode(
        output,
        q,
        kc,
        vc,
        lengths,
        table,
        128**-0.5,
        1,
        compute_type=kc.dtype,
        key_scale=scale,
        value_scale=scale,
        work_plan=plan,
    )
    expected = _reference(q, k, v, logical_table, lengths, 1, 0, None)
    torch.testing.assert_close(output.float(), expected, atol=0.005, rtol=0.005)


@pytest.mark.parametrize("per_token", [False, True])
def test_large_head_single_buffer(per_token, device):
    from aiter.ops.flydsl.pa_decode import pa_decode, plan_pa_decode

    # D1024/MTP4/G16 cannot double-buffer K/V and Q within 320 KiB LDS.
    # Exercise multiple 64-token iterations and the one-buffer retirement path.
    q, k, v, kc, vc, table, lengths = _inputs(
        device, 257, query_length=4, group=16, dim=1024
    )
    plan = plan_pa_decode(lengths, 1, max_partitions=1)
    output = torch.full_like(q, float("nan"))
    scale = torch.ones(
        (kc.shape[0], 1, 128, 1) if per_token else (1,),
        dtype=torch.float32,
        device=device,
    )
    pa_decode(
        output,
        q,
        kc,
        vc,
        lengths,
        table,
        1024**-0.5,
        4,
        compute_type=kc.dtype,
        key_scale=scale,
        value_scale=scale,
        work_plan=plan,
    )
    expected = _reference(q, k, v, table, lengths, 4, 0, None)
    torch.testing.assert_close(output.float(), expected, atol=0.005, rtol=0.005)
