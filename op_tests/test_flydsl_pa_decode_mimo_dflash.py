# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 FlyDSL Project Contributors

"""MiMo Qlen8 full-attention checks for the reused downstream wave kernels."""

import math

import pytest
import torch

from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.ops.flydsl.pa_decode import pa_decode, plan_pa_decode

PAGE = 64
QLEN = 8
HEADS = 16
VALUE_DIM = 128


@pytest.fixture(autouse=True)
def require_gfx950():
    if not torch.cuda.is_available() or get_gfx_runtime() != "gfx950":
        pytest.skip("MiMo FP8 Qlen8 wave decode requires gfx950")


def _pack_caches(k: torch.Tensor, v: torch.Tensor):
    pages, page, kv_heads, key_dim = k.shape
    k_cache = (
        k.reshape(pages, page, kv_heads, key_dim // 16, 16)
        .permute(0, 2, 3, 1, 4)
        .contiguous()
    )
    v_cache = (
        v.reshape(pages, page // 16, 16, kv_heads, VALUE_DIM)
        .permute(0, 3, 1, 4, 2)
        .contiguous()
    )
    return k_cache, v_cache


def _workspace(batch, parts):
    shape = (batch, 1, parts, QLEN * HEADS)
    max_logits = torch.empty(shape, dtype=torch.float32, device="cuda")
    exp_sums = torch.empty_like(max_logits)
    temporary_output = torch.empty(
        (*shape, VALUE_DIM), dtype=torch.bfloat16, device="cuda"
    )
    return max_logits, exp_sums, temporary_output


@pytest.mark.parametrize(
    ("head_dim", "contexts", "parts"),
    [
        (128, (257,), 1),
        (192, (511,), 1),
        (192, (513,), 4),
        (128, (1027, 769), 4),
        (192, (1027, 769), 4),
        (192, (2049, 1025), 8),
        (192, (1027,) * 9, 8),  # B*NP=72 selects the throughput wave kernel.
        (192, (8193,) * 8, 32),
        pytest.param(192, (65536,), 64, id="long-context-np64"),
    ],
)
def test_mimo_fp8_qlen8_full_matches_independent_fp32(head_dim, contexts, parts):
    """Reference uses physical page lookup and bottom-right causal masking."""
    torch.manual_seed(20260925)
    batch = len(contexts)
    max_pages = (max(contexts) + PAGE - 1) // PAGE
    pages = batch * max_pages
    k = (torch.randn((pages, PAGE, 1, head_dim), device="cuda") * 0.5).to(
        torch.float8_e4m3fn
    )
    v = (torch.randn((pages, PAGE, 1, VALUE_DIM), device="cuda") * 0.5).to(
        torch.float8_e4m3fn
    )
    key_cache, value_cache = _pack_caches(k, v)
    query = (torch.randn((batch * QLEN, HEADS, head_dim), device="cuda") * 0.5).to(
        torch.bfloat16
    )
    output = torch.empty(
        (batch * QLEN, HEADS, VALUE_DIM), dtype=torch.bfloat16, device="cuda"
    )
    table = torch.randperm(pages, device="cuda").to(torch.int32).view(batch, max_pages)
    lengths = torch.tensor(contexts, dtype=torch.int32, device="cuda")
    key_scale = torch.tensor([0.75], dtype=torch.float32, device="cuda")
    value_scale = torch.tensor([1.25], dtype=torch.float32, device="cuda")
    pmax, psum, pout = _workspace(batch, parts)
    pa_decode(
        output,
        query,
        key_cache,
        value_cache,
        lengths,
        table,
        head_dim**-0.5,
        QLEN,
        parts,
        compute_type=key_cache.dtype,
        key_scale=key_scale,
        value_scale=value_scale,
        max_logits=pmax,
        exp_sums=psum,
        temporary_output=pout,
    )

    reference = torch.empty_like(output, dtype=torch.float32)
    for b, context in enumerate(contexts):
        positions = torch.arange(context, device="cuda")
        physical_pages = table[b, positions // PAGE].long()
        offsets = positions % PAGE
        keys = k[physical_pages, offsets, 0].float() * key_scale
        values = v[physical_pages, offsets, 0].float() * value_scale
        for i in range(QLEN):
            visible = context - QLEN + i + 1
            logits = (query[b * QLEN + i].float() @ keys[:visible].T) / math.sqrt(
                head_dim
            )
            reference[b * QLEN + i] = torch.softmax(logits, dim=1) @ values[:visible]
    error = output.float() - reference
    relative_l2 = torch.linalg.vector_norm(error) / torch.linalg.vector_norm(reference)
    if max(contexts) <= 512:
        assert relative_l2 < 0.005
        torch.testing.assert_close(output.float(), reference, rtol=0.02, atol=0.001)
    else:
        assert relative_l2 < 0.035
        torch.testing.assert_close(output.float(), reference, rtol=0.02, atol=0.004)


def test_mimo_fp8_qlen8_full_graph_replays_current_metadata():
    head_dim, context, parts = 192, 513, 4
    pages_per_table = (context + 1 + PAGE - 1) // PAGE
    pages = 2 * pages_per_table
    query = torch.zeros((QLEN, HEADS, head_dim), dtype=torch.bfloat16, device="cuda")
    k = torch.zeros((pages, PAGE, 1, head_dim), device="cuda").to(torch.float8_e4m3fn)
    v = torch.zeros((pages, PAGE, 1, VALUE_DIM), device="cuda").to(torch.bfloat16)
    v[0, 0, 0, 0] = 1
    second_marker = context + 1 - QLEN + 1
    v[pages_per_table + second_marker // PAGE, second_marker % PAGE, 0, 0] = 1
    key_cache, value_cache = _pack_caches(k, v.to(torch.float8_e4m3fn))
    output = torch.empty((QLEN, HEADS, VALUE_DIM), dtype=torch.bfloat16, device="cuda")
    table = torch.arange(pages_per_table, device="cuda", dtype=torch.int32)[
        None, :
    ].contiguous()
    lengths = torch.tensor([context], dtype=torch.int32, device="cuda")
    scale = torch.ones(1, dtype=torch.float32, device="cuda")
    pmax, psum, pout = _workspace(1, parts)

    def launch():
        pa_decode(
            output,
            query,
            key_cache,
            value_cache,
            lengths,
            table,
            head_dim**-0.5,
            QLEN,
            parts,
            compute_type=key_cache.dtype,
            key_scale=scale,
            value_scale=scale,
            max_logits=pmax,
            exp_sums=psum,
            temporary_output=pout,
        )

    launch()
    torch.cuda.synchronize()
    first_result = output.clone()
    torch.testing.assert_close(
        output[0, :, 0].float(),
        torch.full((HEADS,), 1 / (context - QLEN + 1), device="cuda"),
        rtol=0.01,
        atol=1e-5,
    )
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch()
    lengths.fill_(context + 1)
    table.add_(pages_per_table)
    output.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    assert torch.count_nonzero(output[0, :, 0]) == 0
    torch.testing.assert_close(
        output[1, :, 0].float(),
        torch.full((HEADS,), 1 / (context - QLEN + 2), device="cuda"),
        rtol=0.01,
        atol=1e-5,
    )
    lengths.fill_(context)
    table.sub_(pages_per_table)
    output.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(output, first_result, rtol=0, atol=0)


def test_mimo_fp8_qlen8_wave_graph_replays_current_metadata():
    """The B*NP>64 wave schedule must use refreshed lengths and physical pages."""
    batch, head_dim, context, parts = 9, 192, 513, 8
    pages_per_table = (context + 1 + PAGE - 1) // PAGE
    pages_per_version = batch * pages_per_table
    query = torch.zeros(
        (batch * QLEN, HEADS, head_dim), dtype=torch.bfloat16, device="cuda"
    )
    k = torch.zeros((2 * pages_per_version, PAGE, 1, head_dim), device="cuda").to(
        torch.float8_e4m3fn
    )
    v = torch.zeros((2 * pages_per_version, PAGE, 1, VALUE_DIM), device="cuda").to(
        torch.bfloat16
    )
    new_marker = context + 1 - QLEN + 1
    for b in range(batch):
        v[b * pages_per_table, 0, 0, 0] = 1
        v[
            pages_per_version + b * pages_per_table + new_marker // PAGE,
            new_marker % PAGE,
            0,
            0,
        ] = 1
    key_cache, value_cache = _pack_caches(k, v.to(torch.float8_e4m3fn))
    output = torch.empty(
        (batch * QLEN, HEADS, VALUE_DIM), dtype=torch.bfloat16, device="cuda"
    )
    table = torch.arange(pages_per_version, device="cuda", dtype=torch.int32).view(
        batch, pages_per_table
    )
    lengths = torch.full((batch,), context, dtype=torch.int32, device="cuda")
    scale = torch.ones(1, dtype=torch.float32, device="cuda")
    pmax, psum, pout = _workspace(batch, parts)

    def launch():
        pa_decode(
            output,
            query,
            key_cache,
            value_cache,
            lengths,
            table,
            head_dim**-0.5,
            QLEN,
            parts,
            compute_type=key_cache.dtype,
            key_scale=scale,
            value_scale=scale,
            max_logits=pmax,
            exp_sums=psum,
            temporary_output=pout,
        )

    launch()
    torch.cuda.synchronize()
    first_result = output.clone()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch()
    lengths.fill_(context + 1)
    table.add_(pages_per_version)
    output.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    for b in range(batch):
        assert torch.count_nonzero(output[b * QLEN, :, 0]) == 0
        torch.testing.assert_close(
            output[b * QLEN + 1, :, 0].float(),
            torch.full((HEADS,), 1 / (context - QLEN + 2), device="cuda"),
            rtol=0.01,
            atol=1e-5,
        )
    lengths.fill_(context)
    table.sub_(pages_per_version)
    output.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(output, first_result, rtol=0, atol=0)


def test_mimo_draft_qlen8_window1024_sink_and_graph_replay():
    """Merged FlyDSL planned SWA retains the draft's current-inclusive window."""
    head_dim, context, window = 128, 1100, 1024
    pages = (context + 1 + PAGE - 1) // PAGE
    query = torch.zeros((QLEN, HEADS, head_dim), dtype=torch.bfloat16, device="cuda")
    k = torch.zeros((pages, PAGE, 1, head_dim), device="cuda").to(torch.float8_e4m3fn)
    v = torch.zeros((pages, PAGE, 1, VALUE_DIM), device="cuda").to(torch.bfloat16)
    for token in range(context - QLEN - window + 1, context - window + 1):
        v[token // PAGE, token % PAGE, 0, 0] = 1
    key_cache, value_cache = _pack_caches(k, v.to(torch.float8_e4m3fn))
    output = torch.empty((QLEN, HEADS, VALUE_DIM), dtype=torch.bfloat16, device="cuda")
    table = torch.arange(pages, device="cuda", dtype=torch.int32)[None, :]
    lengths = torch.tensor([context], dtype=torch.int32, device="cuda")
    scale = torch.ones(1, dtype=torch.float32, device="cuda")
    sinks = torch.full((HEADS,), math.log(window), dtype=torch.float32, device="cuda")
    plan = plan_pa_decode(
        lengths, 1, max_partitions=8, sliding_window=window, query_length=QLEN
    )
    scalar_shape = (1, plan.capacity, QLEN * HEADS)
    pmax = torch.empty(scalar_shape, dtype=torch.float32, device="cuda")
    psum = torch.empty_like(pmax)
    pout = torch.empty((*scalar_shape, VALUE_DIM), dtype=torch.bfloat16, device="cuda")

    def launch():
        plan_pa_decode(lengths, 1, sliding_window=window, query_length=QLEN, plan=plan)
        pa_decode(
            output,
            query,
            key_cache,
            value_cache,
            lengths,
            table,
            head_dim**-0.5,
            QLEN,
            plan.max_partitions,
            compute_type=key_cache.dtype,
            key_scale=scale,
            value_scale=scale,
            max_logits=pmax,
            exp_sums=psum,
            temporary_output=pout,
            sinks=sinks,
            sliding_window=window,
            work_plan=plan,
        )

    launch()
    torch.cuda.synchronize()
    torch.testing.assert_close(
        output[0, :, 0].float(),
        torch.full((HEADS,), 8 / (2 * window), device="cuda"),
        rtol=0,
        atol=0,
    )
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch()
    lengths.fill_(context + 1)
    output.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(
        output[0, :, 0].float(),
        torch.full((HEADS,), 7 / (2 * window), device="cuda"),
        rtol=0,
        atol=0,
    )
