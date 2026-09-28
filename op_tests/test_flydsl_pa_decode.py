# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Generic FlyDSL paged decode correctness test.

Run with: python -m pytest -q op_tests/test_flydsl_pa_decode.py
"""

import importlib.util

import pytest
import torch

from aiter import per_tensor_quant, pertoken_quant
from aiter.jit.utils.chip_info import get_gfx_runtime


@pytest.fixture
def device(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("ROCm is not available")
    if get_gfx_runtime() not in ("gfx942", "gfx950"):
        pytest.skip(f"pa_decode is unsupported on {get_gfx_runtime()}")
    if importlib.util.find_spec("flydsl") is None:
        pytest.skip("FlyDSL is not installed")
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    return torch.device("cuda")


def _reference(query, key, value, table, lengths, query_length, window, sinks):
    """FP32 attention over dequantized KV, independent of kernel Q/P rounding."""
    batch = lengths.numel()
    heads, dim = query.shape[1:]
    kv_heads, page_size = key.shape[1:3]
    group = heads // kv_heads
    queries = query.float().reshape(batch, query_length, kv_heads, group, dim)
    output = torch.zeros_like(queries)
    positions = torch.arange(query_length, device=query.device)

    for seq, length in enumerate(lengths.cpu().tolist()):
        if length == 0:
            continue
        tokens = torch.arange(length, device=query.device)
        pages = table[seq, tokens // page_size].long()
        offsets = tokens % page_size
        keys, values = key[pages, :, offsets], value[pages, :, offsets]
        scores = torch.einsum("qhgd,khd->qhgk", queries[seq], keys) * dim**-0.5
        visible = length - query_length + 1 + positions
        masked = tokens[None, :] >= visible[:, None]
        if window > 0:
            masked |= tokens[None, :] < (visible - window)[:, None]
        scores.masked_fill_(masked[:, None, None, :], float("-inf"))
        denominator = torch.logsumexp(scores, dim=-1, keepdim=True)
        if sinks is not None:
            denominator = torch.logaddexp(
                denominator, sinks.float().reshape(1, kv_heads, group, 1)
            )
        probs = torch.exp(scores - denominator)
        # Empty query rows have no attention mass, including with -inf sinks.
        probs.masked_fill_(visible[:, None, None, None] <= 0, 0)
        output[seq] = torch.einsum("qhgk,khd->qhgd", probs, values)
    return output.reshape_as(query)


def _make_inputs(
    query_length, kv_heads, group_size, dim, page_size, dtype, pattern, device
):
    lengths = (0, 1, 255, 256, 257, 1027) if pattern == "random" else (1024,)
    batch = len(lengths)
    pages_per_seq = (max(lengths) + page_size - 1) // page_size
    num_pages = batch * pages_per_seq
    table = torch.randperm(num_pages, device=device, dtype=torch.int32).reshape(
        batch, pages_per_seq
    )
    query = torch.empty(
        (batch * query_length, kv_heads * group_size, dim), dtype=dtype, device=device
    ).uniform_(-0.5, 0.5)
    key = torch.empty(
        (num_pages, kv_heads, page_size, dim), dtype=dtype, device=device
    ).uniform_(-0.5, 0.5)
    value = torch.empty_like(key).uniform_(-0.5, 0.5)

    if pattern == "small_query":
        # Q must be normalized per row before FP8 conversion in either KV mode.
        query.fill_(2**-12)
        key.fill_(14)
        key[:, :, 1::2] = -14
        value.fill_(3.5)
        value[:, :, 1::2] = 0
    elif pattern == "probability_tail":
        # Individually tiny probabilities carry substantial mass together.
        # Keep the maximum first and use one partition to retain all 1023 tails.
        query.zero_()
        query[..., 0] = 8
        query[..., 1] = 1
        key.zero_()
        key[..., 0] = -10
        key[..., 1] = -0.15625
        key[..., 2] = -14
        value.fill_(3.5)
        first_page = table[0, 0].item()
        key[first_page, :, 0] = 0
        value[first_page, :, 0] = 0
        # Give both quantization modes the same V absmax; the other channels
        # still isolate the probability-tail mass.
        value[first_page, :, 0, 0] = 3.5

    context = torch.tensor(lengths, dtype=torch.int32, device=device)
    return query, key, value, table, context


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
@pytest.mark.parametrize("per_token", [False, True], ids=["per-tensor", "per-token"])
@pytest.mark.parametrize("trans_v", [False, True], ids=["plain-v", "transposed-v"])
@pytest.mark.parametrize(
    "query_length,kv_heads,group_size,head_dim,block_size,"
    "execution,window,sink_dtype,pattern",
    [
        pytest.param(1, 1, 8, 128, 16, "direct", 0, None, "random", id="decode"),
        pytest.param(
            3, 2, 4, 64, 64, "partitioned", 0, torch.float16, "random", id="mtp"
        ),
        pytest.param(
            4, 1, 16, 128, 128, "planned", 257, torch.float32, "random", id="window"
        ),
        pytest.param(2, 2, 8, 256, 16, "planned", 0, None, "random", id="planned"),
        pytest.param(1, 1, 16, 128, 128, "auto", 0, None, "random", id="auto"),
        pytest.param(
            1, 1, 4, 1024, 128, "direct", 0, torch.bfloat16, "random", id="head1024"
        ),
        *[
            pytest.param(
                1,
                1,
                group,
                128,
                page,
                "direct",
                0,
                None,
                pattern,
                id=f"{pattern}-page{page}-g{group}",
            )
            for page, group in ((16, 8), (128, 16))
            for pattern in ("small_query", "probability_tail")
        ],
    ],
)
def test_pa_decode(
    query_length,
    kv_heads,
    group_size,
    head_dim,
    block_size,
    execution,
    window,
    sink_dtype,
    pattern,
    dtype,
    per_token,
    trans_v,
    device,
):
    from aiter.ops.flydsl.pa_decode import pa_decode, plan_pa_decode

    torch.manual_seed(0)
    query, key, value, table, context = _make_inputs(
        query_length, kv_heads, group_size, head_dim, block_size, dtype, pattern, device
    )
    fp8 = (
        torch.float8_e4m3fn if get_gfx_runtime() == "gfx950" else torch.float8_e4m3fnuz
    )
    quantize = pertoken_quant if per_token else per_tensor_quant
    key, key_scale = quantize(key, quant_dtype=fp8)
    value, value_scale = quantize(value, quant_dtype=fp8)
    sinks = None
    if sink_dtype is not None:
        sinks = torch.linspace(-2, 2, query.shape[1], dtype=sink_dtype, device=device)
        sinks[0], sinks[-1] = float("-inf"), float("inf")
    reference = _reference(
        query,
        key.float() * key_scale,
        value.float() * value_scale,
        table,
        context,
        query_length,
        window,
        sinks,
    )

    pages = key.shape[0]
    key_cache = (
        key.reshape(pages, kv_heads, block_size, head_dim // 16, 16)
        .permute(0, 1, 3, 2, 4)
        .contiguous()
    )
    value_cache = (
        value.reshape(pages, kv_heads, block_size // 16, 16, head_dim)
        .permute(0, 1, 2, 4, 3)
        .contiguous()
        if trans_v
        else value.permute(0, 1, 3, 2).contiguous()
    )
    partitions = {"direct": 1, "partitioned": 3, "planned": 7, "auto": None}[execution]
    plan = (
        plan_pa_decode(
            context,
            kv_heads,
            max_partitions=partitions,
            query_length=query_length,
            sliding_window=window,
        )
        if execution == "planned"
        else None
    )
    output = torch.full_like(query, float("nan"))
    pa_decode(
        output,
        query,
        key_cache,
        value_cache,
        context,
        table,
        softmax_scale=head_dim**-0.5,
        query_length=query_length,
        max_context_partition_num=partitions,
        compute_type=fp8,
        key_scale=key_scale,
        value_scale=value_scale,
        sinks=sinks,
        sliding_window=window,
        work_plan=plan,
        max_context_length=context.max().item(),
    )
    assert torch.isfinite(output).all()
    torch.testing.assert_close(output.float(), reference, atol=5e-3, rtol=5e-3)
