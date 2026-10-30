# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Generic FlyDSL paged decode correctness test.

Run with: python -m pytest -q op_tests/test_flydsl_pa_decode.py
"""

import importlib.util
import subprocess
import sys
from itertools import product

import pytest
import torch

from aiter import dtypes, per_tensor_quant, pertoken_quant
from aiter.jit.utils.chip_info import get_gfx_runtime


@pytest.fixture
def device(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("ROCm is not available")
    if get_gfx_runtime() not in ("gfx942", "gfx950", "gfx1250"):
        pytest.skip(f"pa_decode is unsupported on {get_gfx_runtime()}")
    if importlib.util.find_spec("flydsl") is None:
        pytest.skip("FlyDSL is not installed")
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    return torch.device("cuda")


@pytest.mark.parametrize(
    "batch,max_parts,budget,window,query_length",
    [
        (1, 1, 1, 0, 1),
        (6, 7, 160, 257, 4),
        (63, 7, 160, 0, 1),
        (65, 7, 512, 1, 4),
        (257, None, 4096, 8193, 4),
        (4096, None, 8192, 0, 1),
    ],
)
def test_pa_decode_plan(device, batch, max_parts, budget, window, query_length):
    """Packed tasks cover every visible tile and remain valid after graph refresh."""
    from aiter.ops.flydsl.kernels.pa_decode_plan import plan_pa_decode

    def check(plan, lengths):
        info = plan.reduce_info.cpu().tolist()
        work = plan.work_info.cpu().tolist()
        cursor = 0
        for seq, (ctx, (start, count)) in enumerate(zip(lengths, info)):
            assert start == cursor
            first = (
                max(max(ctx, 0) - query_length + 1 - window, 0) // 256 if window else 0
            )
            last = (max(ctx, 0) + 255) // 256
            assert 0 <= count <= min(last - first, plan.max_partitions)
            assert (count > 0) == (last > first)
            tile = first
            for record in work[start : start + count]:
                row, begin, end, original_length = record
                assert row == seq and original_length == ctx
                assert begin == tile and begin < end <= last
                tile = end
            assert tile == last
            cursor += count
        assert cursor <= plan.capacity
        assert all(record == [0, 0, 0, 0] for record in work[cursor:])

    pattern = (-1, 0, 1, 255, 256, 257, 65537, 2**31 - 1)
    lengths = [pattern[i % len(pattern)] for i in range(batch)]
    context = torch.tensor(lengths, dtype=torch.int32, device=device)
    plan = plan_pa_decode(
        context,
        2,
        max_partitions=max_parts,
        workgroup_budget=budget,
        sliding_window=window,
        query_length=query_length,
    )
    check(plan, lengths)
    pointers = (plan.work_info.data_ptr(), plan.reduce_info.data_ptr())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        refreshed = plan_pa_decode(
            context, 2, sliding_window=window, query_length=query_length, plan=plan
        )
    assert refreshed is plan
    # All-empty replay must erase old active records, then mixed lengths must
    # rebuild them using the same buffers without stale padding or readback.
    for values in ([0] * batch, list(reversed(lengths))):
        context.copy_(torch.tensor(values, dtype=torch.int32, device=device))
        graph.replay()
        check(plan, values)
        assert (plan.work_info.data_ptr(), plan.reduce_info.data_ptr()) == pointers


@pytest.mark.parametrize(
    "first",
    [
        "PADecodePlan",
        "pa_decode_default_workgroup_budget",
        "pa_decode_nhd_max_query_length",
        "plan_pa_decode",
        "prepare_pa_decode_plan",
    ],
)
def test_pa_decode_export_import_order(first):
    """Loading a sibling export first must not leave ``pa_decode`` a module."""
    if importlib.util.find_spec("flydsl") is None:
        pytest.skip("FlyDSL is not installed")
    code = (
        f"from aiter.ops.flydsl import {first}, pa_decode; assert callable(pa_decode)"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


_NHD_CONFIG = {
    "kv_dtype": torch.float8_e4m3fn,
    "head_dim": 256,
    "block_size": 64,
    "query_group_size": 8,
}


@pytest.mark.parametrize(
    "overrides,gfx950_max",
    [
        ({}, 4),
        ({"query_group_size": 16}, 2),
        ({"query_group_size": 32}, 1),
        ({"query_group_size": 33}, 0),
        ({"block_size": 16}, 0),
        ({"block_size": 128}, 0),
        ({"head_dim": 128}, 0),
        ({"kv_dtype": torch.bfloat16}, 0),
        ({"kv_dtype": torch.float8_e4m3fnuz}, 0),
    ],
    ids=[
        "gqa8",
        "gqa16",
        "gqa32",
        "gqa33",
        "page16",
        "page128",
        "head128",
        "bf16",
        "fnuz",
    ],
)
def test_pa_decode_nhd_max_query_length(overrides, gfx950_max, device):
    """NHD runs gfx950 FP8 head 256 page 64 with at most 32 query rows."""
    pa = importlib.import_module("aiter.ops.flydsl.pa_decode")
    expected = gfx950_max if get_gfx_runtime() == "gfx950" else 0
    config = {**_NHD_CONFIG, **overrides}
    assert pa.pa_decode_nhd_max_query_length(device, **config) == expected


@pytest.mark.parametrize(
    "overrides,one_cta_per_cu",
    [
        ({}, True),
        ({"query_length": 1}, True),
        ({"sliding_window": 4096}, False),
        ({"block_size": 16}, False),
        ({"head_dim": 128}, False),
        ({"query_group_size": 16}, False),
        ({"kv_dtype": torch.bfloat16}, False),
    ],
)
def test_pa_decode_default_workgroup_budget(overrides, one_cta_per_cu, device):
    """Dense page-64 head-256 FP8 pipe shapes get one CTA per CU, else two."""
    pa = importlib.import_module("aiter.ops.flydsl.pa_decode")
    config = {**_NHD_CONFIG, "query_length": 4, **overrides}
    num_cu = torch.cuda.get_device_properties(device).multi_processor_count
    one_cta_per_cu = one_cta_per_cu and get_gfx_runtime() == "gfx950"
    budget = pa.pa_decode_default_workgroup_budget(device, **config)
    assert budget == (num_cu if one_cta_per_cu else 2 * num_cu)


@pytest.mark.parametrize(
    "shape", ["page64", "nhd", "page16", "window", "head128", "rows64"]
)
def test_prepare_pa_decode_plan_default_budget(shape, device, tmp_path, monkeypatch):
    """A cache miss plans dense page-64 head-256 pipe shapes at the CU count."""
    from aiter.ops.flydsl.pa_decode_tuning import prepare_pa_decode_plan

    monkeypatch.delenv("FLYDSL_AUTOTUNE", raising=False)
    arch = get_gfx_runtime()
    fp8 = dtypes.fp8
    page = 16 if shape == "page16" else 64
    dim = 128 if shape == "head128" else 256
    group = 16 if shape == "rows64" else 8
    batch, query_length, hkv = 4, 4, 1
    window = 4096 if shape == "window" else 0
    pages = batch * 2
    query = torch.zeros(
        batch * query_length, hkv * group, dim, dtype=torch.bfloat16, device=device
    )
    if shape == "nhd":
        kv = torch.zeros(pages, page, hkv, 2, dim, dtype=fp8, device=device)
        key_cache, value_cache = kv[:, :, :, 0], kv[:, :, :, 1]
    else:
        key_cache = torch.zeros(
            pages, hkv, dim // 16, page, 16, dtype=fp8, device=device
        )
        value_cache = torch.zeros(pages, hkv, dim, page, dtype=fp8, device=device)
    context = torch.full((batch,), 2 * page, dtype=torch.int32, device=device)
    table = torch.arange(pages, dtype=torch.int32, device=device).view(batch, 2)
    plan = prepare_pa_decode_plan(
        query,
        key_cache,
        value_cache,
        context,
        table,
        dim**-0.5,
        query_length,
        compute_type=fp8,
        key_scale=1.0,
        value_scale=1.0,
        sliding_window=window,
        config_dir=tmp_path,
    )
    num_cu = torch.cuda.get_device_properties(device).multi_processor_count
    one_cta_per_cu = arch == "gfx950" and shape in ("page64", "nhd")
    assert plan.capacity == (num_cu if one_cta_per_cu else 2 * num_cu)


def _reference(query, key, value, table, lengths, query_length, window, sinks):
    """FP32 attention over logical KV, independent of kernel Q/P rounding."""
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
    query_length,
    kv_heads,
    group_size,
    dim,
    page_size,
    dtype,
    pattern,
    device,
    lengths,
    table_width=None,
):
    if lengths is None:
        lengths = (0, 1, 255, 256, 257, 1027) if pattern == "random" else (1024,)
    batch = len(lengths)
    pages_per_seq = (max(lengths) + page_size - 1) // page_size
    num_pages = batch * pages_per_seq
    table = torch.randperm(num_pages, device=device, dtype=torch.int32).reshape(
        batch, pages_per_seq
    )
    if table_width is not None:
        # Block-table padding beyond the live pages must not reach the kernel.
        table = torch.nn.functional.pad(table, (0, table_width - pages_per_seq))
    query = torch.empty(
        (batch * query_length, kv_heads * group_size, dim), dtype=dtype, device=device
    ).uniform_(-0.5, 0.5)
    key = torch.empty(
        (num_pages, kv_heads, page_size, dim), dtype=dtype, device=device
    ).uniform_(-0.5, 0.5)
    value = torch.empty_like(key).uniform_(-0.5, 0.5)

    if pattern == "constant":
        query.zero_()
        key.zero_()
        value.fill_(0.5)
    elif pattern == "small_query":
        # Tiny Q values must survive FP8 normalization or native BF16 loads.
        query.fill_(2**-12)
        key.fill_(14)
        key[:, :, 1::2] = -14
        value.fill_(3.5)
        value[:, :, 1::2] = 0
    elif pattern == "probability_tail":
        # Individually tiny probabilities carry substantial mass together.
        # Keep the maximum first and use one partition to retain all 1023 tails.
        # These Q/K values give the same scores in E4M3FN and E4M3FNUZ,
        # with tail probabilities near 2**-12. Scaling P preserves the tails
        # in both formats; converting unscaled P to FP8 rounds them to zero.
        query.zero_()
        query[..., 0] = 13.4375
        key.zero_()
        key[..., 0] = -7
        key[..., 2] = -14
        value.fill_(3.5)
        first_page = table[0, 0].item()
        key[first_page, :, 0] = 0
        value[first_page, :, 0] = 0
        # Give both quantization modes the same V absmax; the other channels
        # still isolate the probability-tail mass.
        value[first_page, :, 0, 0] = 3.5
    elif pattern == "varied":
        # Uniform draws quantize to nearly one per-token scale, so a shifted
        # scale still matches the oracle. Token magnitudes spanning about a
        # decade make a wrong scale index fail it.
        key.mul_(torch.randn_like(key[..., :1]).mul_(0.4).exp_())
        value.mul_(torch.randn_like(value[..., :1]).mul_(0.4).exp_())

    context = torch.tensor(lengths, dtype=torch.int32, device=device)
    return query, key, value, table, context


_COMMON_CASES = [
    ("decode", {"block_size": 16, "max_partitions": 1}),
    (
        "window",
        {
            "query_length": 4,
            "group_size": 16,
            "block_size": 128,
            "max_partitions": 7,
            "window": 257,
            "sink_dtype": torch.float32,
        },
    ),
    (
        "head1024",
        {
            "group_size": 4,
            "head_dim": 1024,
            "block_size": 128,
            "sink_dtype": torch.bfloat16,
        },
    ),
    *[
        (
            f"{pattern}-page{page}-g{group}",
            {"group_size": group, "block_size": page, "pattern": pattern},
        )
        for page, group in ((16, 8), (128, 16))
        for pattern in ("small_query", "probability_tail")
    ],
]
_FP8_CASES = _COMMON_CASES + [
    (
        "packed-mtp4",
        {
            "query_length": 4,
            "kv_heads": 2,
            "group_size": 8,
            "head_dim": 256,
            "block_size": 128,
            "max_partitions": 7,
            "sink_dtype": torch.float32,
        },
    ),
    (
        "mtp",
        {
            "query_length": 3,
            "kv_heads": 2,
            "group_size": 4,
            "head_dim": 64,
            "max_partitions": 3,
            "sink_dtype": torch.float16,
        },
    ),
    (
        "head256",
        {
            "query_length": 2,
            "kv_heads": 2,
            "head_dim": 256,
            "block_size": 16,
            "max_partitions": 7,
        },
    ),
    (
        "default-cap",
        {"group_size": 16, "block_size": 128, "max_partitions": None},
    ),
]
_BF16_CASES = _COMMON_CASES + [
    (
        "mtp",
        {
            "query_length": 4,
            "kv_heads": 2,
            "group_size": 4,
            "head_dim": 64,
            "max_partitions": 3,
            "sink_dtype": torch.float16,
        },
    ),
    (
        "head256",
        {"kv_heads": 2, "head_dim": 256, "block_size": 16, "max_partitions": 7},
    ),
    ("head192-decode", {"group_size": 16, "head_dim": 192, "max_partitions": 4}),
    (
        "head192-mtp",
        {"query_length": 4, "group_size": 16, "head_dim": 192, "max_partitions": 4},
    ),
    # Large BF16 Q/P operands must split MTP rows to fit the LDS budget.
    *[
        (
            f"head{dim}-mtp-split{splits}",
            {
                "query_length": length,
                "group_size": 16,
                "head_dim": dim,
                "max_partitions": 4,
            },
        )
        for length, dim, splits in ((4, 512, 2), (4, 1024, 4), (3, 1024, 3))
    ],
    ("graph-workspace", {"query_length": 4, "max_partitions": 7, "capture": True}),
    (
        "loose-bound",
        {
            "block_size": 16,
            "lengths": (0, 1, 257, 300, 512),
            "max_partitions": None,
            "table_width": 768,
            "max_context": None,
        },
    ),
    (
        "head192-np80",
        {
            "group_size": 16,
            "head_dim": 192,
            "lengths": (80 * 256,),
            "max_partitions": 80,
        },
    ),
    *[
        (
            f"head{dim}-mixed-np96",
            {
                "query_length": 4,
                "kv_heads": 2,
                "group_size": 16,
                "head_dim": dim,
                "lengths": (97 * 256,) + (0, 1, 257, 1027) * 10,
                "max_partitions": 96,
                "budget": 4096,
            },
        )
        for dim in (256, 128)
    ],
]
# FP8 head 256, page 64, plain V, uniform query length with at most 32 query
# rows: the gfx950 pipe, for per-tensor and per-token scales. A plan-independent
# staging bound selects its specialization;
# other architectures run the same cases on the generic path.
_PIPE_BASE = {
    "kv_dtype": "fp8",
    "scale_mode": "per-tensor",
    "trans_v": False,
    "head_dim": 256,
    "block_size": 64,
    "group_size": 8,
    "pipe": True,
}
_NHD_BASE = {**_PIPE_BASE, "layout": "nhd"}
_PIPE_CASES = [
    # At GQA 8, query length q gives 8 * q rows: one M-tile at 1 and 2 (1 pads
    # it), two at 3 and 4 (3 pads the second). Lengths put some row's causal
    # edge on a page or tile boundary, and lengths below q leave rows empty.
    # P is FP8, so random rows see 0, 1 or at least 6 tokens; the token
    # one-hot token pattern covers short contexts exactly.
    #
    # A task needing more than 4096 page ids cannot stage the whole list.
    (
        "d256-unstaged-q2",
        {
            **_PIPE_BASE,
            "query_length": 2,
            "lengths": (0, 1, 65, 257, 8192),
            "max_partitions": 4,
            "budget": 1,
            "table_width": 20000,
            "max_context": None,
            "staged": False,
        },
    ),
    # max_context_length below the real context used to index past the staged
    # page ids and fault. The tail must still match a full-context reference.
    (
        "d256-page64-short-bound",
        {
            **_PIPE_BASE,
            "lengths": (128 * 1024,),
            "max_partitions": 1,
            "max_context": 8192,
            "budget": 1,
        },
    ),
    # 81 and 130 partitions leave a short tail, and sequences with few or no
    # tasks must not change a channel.
    *(
        (
            f"d256-page64-q4-np{partitions}",
            {
                **_PIPE_BASE,
                "query_length": 4,
                "lengths": (0, 1, 257, 66532),
                "max_partitions": partitions,
                "budget": partitions,
                "pattern": "one_hot_v",
            },
        )
        for partitions in (81, 130)
    ),
    # A head-major pool of 2**15 + 1 blocks puts KV head 2 past 2**31 bytes, so
    # the host must select wide addressing and widen the head term. 288 CTAs
    # exceed the CU count, which selects one V buffer. The long sequence gives
    # each task four tiles that reuse that buffer, and its last tile ends
    # mid-page with two empty pages.
    (
        "nhd-d256-page64-hkv3-head-major-wide",
        {
            **_NHD_BASE,
            "kv_heads": 3,
            "lengths": (1, 65, 32600),
            "max_partitions": 32,
            "nhd_order": "head",
            "pool_blocks": 2**15 + 1,
            "wide": True,
            "nhd_bufs": 1,
        },
    ),
    (
        "nhd-d256-page64",
        {
            **_NHD_BASE,
            "lengths": (0, 1, 63, 64, 65, 255, 256, 257, 8192),
            "max_partitions": 8,
            "nhd_bufs": 2,
            "nhd_contracts": True,
        },
    ),
    # Every KV head in one token row: a head stride of one K/V row pair and a
    # token stride of all heads' rows.
    (
        "nhd-d256-page64-hkv2-token-rows",
        {
            **_NHD_BASE,
            "kv_heads": 2,
            "nhd_order": "token",
            "lengths": (1, 65, 257, 4097),
            "max_partitions": 8,
            "nhd_bufs": 2,
        },
    ),
    # Two M-tiles always take two V buffers. The longest sequence gives tasks
    # several tiles that alternate them, and the last task ends mid-page.
    (
        "nhd-d256-page64-q3",
        {
            **_NHD_BASE,
            "query_length": 3,
            "lengths": (0, 1, 2, 3, 5, 65, 257, 2049),
            "max_partitions": 8,
            "pattern": "one_hot_token",
        },
    ),
    (
        "nhd-d256-page64-q4",
        {
            **_NHD_BASE,
            "query_length": 4,
            "lengths": (0, 1, 65, 66, 257, 258, 8258),
            "max_partitions": 8,
            "nhd_bufs": 2,
            "capture": True,
        },
    ),
    # Per-token scales take the same pipe. The wide grid takes the one-buffer
    # NHD path, whose V gather must not be replaced by a plain-V reload.
    (
        "d256-page64-per-token",
        {
            **_PIPE_BASE,
            "scale_mode": "per-token",
            "lengths": (1, 65, 257, 1027),
            "max_partitions": 4,
            "pattern": "varied",
        },
    ),
    (
        "d256-page64-q4-per-token",
        {
            **_PIPE_BASE,
            "scale_mode": "per-token",
            "query_length": 4,
            "lengths": (0, 1, 65, 257, 2049),
            "max_partitions": 4,
            "pattern": "varied",
        },
    ),
    (
        "nhd-d256-page64-per-token",
        {
            **_NHD_BASE,
            "scale_mode": "per-token",
            "lengths": (1, 65, 257, 1027),
            "max_partitions": 4,
            "pattern": "varied",
            "nhd_bufs": 2,
        },
    ),
    (
        "nhd-d256-page64-q4-per-token",
        {
            **_NHD_BASE,
            "scale_mode": "per-token",
            "query_length": 4,
            "lengths": (0, 1, 65, 257, 2049),
            "max_partitions": 4,
            "pattern": "varied",
            "nhd_bufs": 2,
        },
    ),
    (
        "nhd-d256-page64-per-token-wide-grid",
        {
            **_NHD_BASE,
            "scale_mode": "per-token",
            "kv_heads": 3,
            "lengths": (1, 65, 32600),
            "max_partitions": 32,
            "pattern": "varied",
            "nhd_bufs": 1,
        },
    ),
    # A window-257 plan is single-tile and keeps one set of per-token K/V
    # scales. Window 8193 at budget 1 gives tasks that start mid-sequence and
    # loop over tiles with window-bounded page staging.
    (
        "nhd-d256-page64-q4-w257-per-token",
        {
            **_NHD_BASE,
            "scale_mode": "per-token",
            "query_length": 4,
            "window": 257,
            "lengths": (0, 1, 258, 1027, 8258),
            "max_partitions": 8,
            "pattern": "varied",
        },
    ),
    (
        "nhd-d256-page64-w8193",
        {
            **_NHD_BASE,
            "window": 8193,
            "lengths": (1, 257, 8193, 20000),
            "max_partitions": 2,
            "budget": 1,
        },
    ),
]
_INVALID_CASES = [
    ("fp16-query", NotImplementedError, "BF16 KV requires bfloat16 queries"),
    ("plain-v", ValueError, "BF16 KV requires the vectorized 5D value_cache layout"),
    (
        "scaled-kv",
        ValueError,
        "BF16 KV is unscaled; key_scale and value_scale must be None",
    ),
    ("vector16-k", ValueError, "key_cache shape must be"),
    ("vector16-v", ValueError, "transposed value_cache shape must be"),
]


def _check_output(output, reference):
    assert torch.isfinite(output).all()
    torch.testing.assert_close(output.float(), reference, atol=5e-3, rtol=5e-3)


def _check_unsupported_input(pa, arguments, options, plan, case, monkeypatch):
    query, key, value, *rest = arguments
    invalid = case["invalid_input"]
    if invalid == "fp16-query":
        query = query.to(torch.float16)
    elif invalid == "plain-v":
        value = torch.empty((1, 1, 128, 64), dtype=query.dtype, device=query.device)
    elif invalid == "scaled-kv":
        options.update(
            key_scale=torch.ones(1, dtype=torch.float32, device=query.device),
            value_scale=torch.ones(1, dtype=torch.float32, device=query.device),
        )
    elif invalid == "vector16-k":
        key = torch.empty((1, 1, 8, 64, 16), dtype=query.dtype, device=query.device)
    elif invalid == "vector16-v":
        value = torch.empty((1, 1, 4, 128, 16), dtype=query.dtype, device=query.device)

    def unexpected_compile(**_):
        pytest.fail("invalid BF16 inputs reached kernel compilation")

    monkeypatch.setattr(pa, "compile_pa_decode_tile", unexpected_compile)
    with pytest.raises(case["error"], match=case["match"]):
        pa.pa_decode(
            torch.empty_like(query),
            query,
            key,
            value,
            *rest,
            work_plan=plan,
            **options,
        )


def _one_hot_kv(key, value, table, context, compute_type, tokens_per_channel=256):
    """Zero K and one-hot V in channel (token // tokens_per_channel) % dim.

    With 256 tokens per channel each tile owns one output channel, so a task
    that drops or repeats a tile changes that channel's mean. With one, a row's
    output marks exactly the tokens it sees, and zero K keeps FP8 P exact.
    Scales are explicit because a zero K must not reach the quantizer.
    """
    dim, page = value.shape[-1], value.shape[2]
    dense = torch.zeros(value.shape, dtype=torch.float32, device=value.device)
    for seq, length in enumerate(context.tolist()):
        tokens = torch.arange(length, device=value.device)
        pages = table[seq, tokens // page].long()
        dense[pages, :, tokens % page, (tokens // tokens_per_channel) % dim] = 224
    scale = torch.ones(1, dtype=torch.float32, device=value.device)
    return (
        torch.zeros_like(key, dtype=compute_type),
        scale,
        dense.to(compute_type),
        scale * 0.25,
    )


def _nhd_caches(key, value, table, context, order, pool_blocks=None):
    """K and V views of one [page, token, KV head, 2 * dim] row store.

    ``order`` is the storage order outside the rows: "token" puts every KV
    head in one token row, "page" puts each head's page contiguous within a
    block, and "head" makes KV heads outermost. Unwritten K and V tails hold
    FP8 NaN: masking must drop K tails, and the kernel must zero-fill V tails
    because 0 * NaN would poison PV. Pool blocks past the table's pages stay
    unwritten.
    """
    pages, kv_heads, page, dim = key.shape
    pool = pool_blocks or pages
    options = {"dtype": key.dtype, "device": key.device}
    if order == "head":
        rows = torch.empty((kv_heads, pool, page, 2 * dim), **options).transpose(0, 1)
    elif order == "token":
        rows = torch.empty((pool, page, kv_heads, 2 * dim), **options).transpose(1, 2)
    else:
        rows = torch.empty((pool, kv_heads, page, 2 * dim), **options)
    rows[:pages, ..., :dim] = key
    rows[:pages, ..., dim:] = value
    nan = torch.tensor(0x7F, dtype=torch.uint8, device=key.device).view(key.dtype)
    for seq, length in enumerate(context.tolist()):
        if length % page:
            rows[table[seq, length // page], :, length % page :] = nan
    return rows.transpose(1, 2).split(dim, dim=-1)


def _check_pipe_policy(pa, case, arch, compile_kwargs):
    """Pin the selected specialization so a case cannot drift off its path."""
    from aiter.ops.flydsl.kernels.pa_decode.traits import (
        PaDecodeSchedule,
        PaDecodeTraits,
    )

    assert compile_kwargs[-1]["wide_kv_addressing"] == case["wide"]
    if case["pipe"]:
        schedule_kwargs = dict(compile_kwargs[-1])
        schedule_kwargs.pop("architecture", None)
        schedule_kwargs.pop("direct_output", None)
        traits = PaDecodeTraits.create(PaDecodeSchedule.select(**schedule_kwargs))
        assert traits.D256_PIPE == (arch == "gfx950")
        if traits.D256_PIPE:
            assert (traits.stage_page_capacity > 0) == (case["staged"] is not False)
            if case["nhd_bufs"]:
                assert traits.NHD_V_BUFS == case["nhd_bufs"]


def _check_nhd_contracts(pa, output, arguments, options, plan, check):
    """Reject NHD views that break the 16-byte loads or the V tail bound."""
    query, key, value, *rest = arguments
    block, token, head, _ = key.stride()

    def view(tensor, stride, offset=0):
        return tensor.as_strided(tensor.shape, stride, tensor.storage_offset() + offset)

    def run(out, bad_key, bad_value):
        pa.pa_decode(out, query, bad_key, bad_value, *rest, work_plan=plan, **options)

    for invalid, message in [
        (view(key, key.stride(), offset=1), "16-byte"),
        (view(key, (block, token - 8, head, 1)), "16-byte"),
        (view(key, (block, 128, head, 1)), "overlap"),
    ]:
        with pytest.raises(ValueError, match=message):
            run(torch.empty_like(query), invalid, value)
    if key.shape[2] == 1:
        # A single KV head is never stepped, so its stride may be anything.
        odd = (block, token, 2**40 + 8, 1)
        output.fill_(float("nan"))
        run(output, view(key, odd), view(value, odd))
        check()


@pytest.mark.parametrize(
    "case",
    [
        *[
            pytest.param(
                dict(
                    geometry,
                    kv_dtype="fp8",
                    query_dtype=dtype,
                    scale_mode=scale_mode,
                    trans_v=trans_v,
                ),
                id=f"fp8-{dtype}-{scale_mode}-{'transposed' if trans_v else 'plain'}-v-{name}",
            )
            for name, geometry in _FP8_CASES
            for dtype, scale_mode, trans_v in product(
                (torch.bfloat16, torch.float16),
                ("per-tensor", "per-token"),
                (False, True),
            )
        ],
        *[pytest.param(geometry, id=f"bf16-{name}") for name, geometry in _BF16_CASES],
        *[pytest.param(geometry, id=name) for name, geometry in _PIPE_CASES],
        *[
            pytest.param(
                {
                    "invalid_input": invalid,
                    "error": error,
                    "match": match,
                    "lengths": (1,),
                },
                id=f"invalid-bf16-{invalid}",
            )
            for invalid, error, match in _INVALID_CASES
        ],
    ],
)
def test_pa_decode(case, device, monkeypatch):
    """FP8/BF16 decode shares input, reference, planning and replay checks."""
    case = dict(
        {
            "query_length": 1,
            "kv_heads": 1,
            "group_size": 8,
            "head_dim": 128,
            "block_size": 64,
            "max_partitions": 1,
            "window": 0,
            "sink_dtype": None,
            "pattern": "random",
            "kv_dtype": "bf16",
            "query_dtype": torch.bfloat16,
            "scale_mode": "none",
            "trans_v": True,
            "capture": False,
            "lengths": None,
            "layout": "packed",
            "nhd_order": "page",
            "pool_blocks": None,
            "budget": None,
            "table_width": None,
            "max_context": "lengths",
            "wide": False,
            "pipe": False,
            "staged": None,
            "nhd_bufs": None,
            "nhd_contracts": False,
        },
        **case,
    )
    arch = get_gfx_runtime()
    num_cu = torch.cuda.get_device_properties(device).multi_processor_count
    if case["max_partitions"] is not None and case["max_partitions"] > num_cu:
        pytest.skip(f"plan_pa_decode caps max_partitions at {num_cu} CUs")
    if case["layout"] == "nhd" and arch != "gfx950":
        pytest.skip("NHD decode is gfx950 only")
    pa = importlib.import_module("aiter.ops.flydsl.pa_decode")
    if get_gfx_runtime() == "gfx1250":
        if case.get("invalid_input") in ("fp16-query", "plain-v"):
            pytest.skip("gfx1250 supports FP16 queries and plain BF16 V caches")
        if (
            case["kv_dtype"] == "bf16"
            and case["head_dim"] % 128
            and case["head_dim"] != 64
        ):
            pytest.skip("gfx1250 supports head_dim 64 or multiples of 128")
    query_length, kv_heads, group_size, dim, page, window = (
        case[name]
        for name in (
            "query_length",
            "kv_heads",
            "group_size",
            "head_dim",
            "block_size",
            "window",
        )
    )
    if case["pool_blocks"]:
        pool_bytes = case["pool_blocks"] * kv_heads * page * 2 * dim
        if torch.cuda.mem_get_info(device)[0] < pool_bytes + 2**30:
            pytest.skip(f"needs {pool_bytes} B of free GPU memory for the KV pool")
    torch.manual_seed(0)
    query, key, value, table, context = _make_inputs(
        query_length,
        kv_heads,
        group_size,
        dim,
        page,
        case["query_dtype"],
        case["pattern"],
        device,
        case["lengths"],
        case["table_width"],
    )
    nhd = case["layout"] == "nhd"
    compute_type = torch.bfloat16
    key_scale = value_scale = None
    vector = 8
    if case["kv_dtype"] == "fp8":
        compute_type = (
            torch.float8_e4m3fn
            if get_gfx_runtime() in ("gfx950", "gfx1250")
            else torch.float8_e4m3fnuz
        )
        vector = 16
        mode = case["scale_mode"]
        if case["pattern"] in ("one_hot_v", "one_hot_token"):
            tokens_per_channel = 1 if case["pattern"] == "one_hot_token" else 256
            key, key_scale, value, value_scale = _one_hot_kv(
                key, value, table, context, compute_type, tokens_per_channel
            )
        elif mode in ("per-tensor", "per-token"):
            quantize = pertoken_quant if mode == "per-token" else per_tensor_quant
            key, key_scale = quantize(key, quant_dtype=compute_type)
            value, value_scale = quantize(value, quant_dtype=compute_type)
        else:
            key, value = key.to(compute_type), value.to(compute_type)
            if mode == "python":
                key_scale, value_scale = 0.25, 2.0
    if nhd:
        assert query_length <= pa.pa_decode_nhd_max_query_length(
            device,
            kv_dtype=compute_type,
            head_dim=dim,
            block_size=page,
            query_group_size=group_size,
        )
    logical_key = key.float() * (1.0 if key_scale is None else key_scale)
    logical_value = value.float() * (1.0 if value_scale is None else value_scale)
    pages = key.shape[0]
    if nhd:
        key_cache, value_cache = _nhd_caches(
            key, value, table, context, case["nhd_order"], case["pool_blocks"]
        )
    else:
        key_cache = (
            key.reshape(pages, kv_heads, page, dim // vector, vector)
            .permute(0, 1, 3, 2, 4)
            .contiguous()
        )
        value_cache = (
            value.reshape(pages, kv_heads, page // vector, vector, dim)
            .permute(0, 1, 2, 4, 3)
            .contiguous()
            if case["trans_v"]
            else value.permute(0, 1, 3, 2).contiguous()
        )
    sinks = None
    if case["sink_dtype"] is not None:
        sinks = torch.linspace(
            -2, 2, query.shape[1], dtype=case["sink_dtype"], device=device
        )
        sinks[0], sinks[-1] = float("-inf"), float("inf")
    options = {
        "compute_type": compute_type,
        "key_scale": key_scale,
        "value_scale": value_scale,
        "sinks": sinks,
        "sliding_window": window,
        "max_context_length": (
            None
            if case["max_context"] is None
            else (
                context.max().item()
                if case["max_context"] == "lengths"
                else case["max_context"]
            )
        ),
    }
    arguments = (query, key_cache, value_cache, context, table, dim**-0.5, query_length)
    plan = pa.plan_pa_decode(
        context,
        kv_heads,
        max_partitions=case["max_partitions"],
        workgroup_budget=case["budget"],
        query_length=query_length,
        sliding_window=window,
    )
    if "invalid_input" in case:
        _check_unsupported_input(pa, arguments, options, plan, case, monkeypatch)
        return

    compile_kwargs = []
    compile_tile = pa.compile_pa_decode_tile

    def spy_tile(**kwargs):
        compile_kwargs.append(kwargs)
        return compile_tile(**kwargs)

    monkeypatch.setattr(pa, "compile_pa_decode_tile", spy_tile)

    output = torch.full_like(query, float("nan"))
    workspace = {}
    if case["capture"]:
        scalar_shape = (kv_heads, plan.capacity, query_length * group_size)
        workspace = {
            "exp_sums": torch.empty(scalar_shape, dtype=torch.float32, device=device),
            "max_logits": torch.empty(scalar_shape, dtype=torch.float32, device=device),
            "temporary_output": torch.empty(
                (*scalar_shape, dim), dtype=query.dtype, device=device
            ),
        }

    def launch():
        pa.pa_decode(output, *arguments, work_plan=plan, **options, **workspace)

    def check():
        reference = _reference(
            query,
            logical_key,
            logical_value,
            table,
            context,
            query_length,
            window,
            sinks,
        )
        _check_output(output, reference)

    launch()
    check()
    _check_pipe_policy(pa, case, arch, compile_kwargs)
    if case["nhd_contracts"]:
        _check_nhd_contracts(pa, output, arguments, options, plan, check)
    if case["capture"]:
        # Compile/allocate before capture; replay must observe changed query data.
        capture_stream = torch.cuda.Stream()
        capture_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(capture_stream):
            launch()
        torch.cuda.current_stream().wait_stream(capture_stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=capture_stream):
            launch()
        for _ in range(2):
            query.neg_()
            output.fill_(float("nan"))
            graph.replay()
            check()


@pytest.mark.parametrize("variant", ["arch", "bf16", "mtp", "page16", "page128"])
def test_pa_decode_nhd_rejected(variant, device, monkeypatch):
    """NHD requires gfx950 FP8 page 64 and at most 32 query rows."""
    arch = get_gfx_runtime()
    if (variant == "arch") == (arch == "gfx950"):
        pytest.skip(f"{variant} is not a rejected NHD variant on {arch}")
    pa = importlib.import_module("aiter.ops.flydsl.pa_decode")
    fp8 = dtypes.fp8

    def unexpected_compile(**_):
        pytest.fail("rejected NHD inputs reached kernel compilation")

    monkeypatch.setattr(pa, "compile_pa_decode_tile", unexpected_compile)
    page = {"page16": 16, "page128": 128}.get(variant, 64)
    # GQA 8: query length 5 is 40 rows, past the pipe's two M-tiles.
    query_length = 5 if variant == "mtp" else 1
    dtype = torch.bfloat16 if variant == "bf16" else fp8
    query = torch.zeros((query_length, 8, 256), dtype=torch.bfloat16, device=device)
    cache = torch.zeros((2, page, 1, 256), dtype=dtype, device=device)
    context = torch.tensor([page], dtype=torch.int32, device=device)
    table = torch.zeros((1, 1), dtype=torch.int32, device=device)
    scale = torch.ones(1, dtype=torch.float32, device=device)
    plan = pa.plan_pa_decode(context, 1, max_partitions=1, query_length=query_length)
    assert query_length > pa.pa_decode_nhd_max_query_length(
        device, kv_dtype=dtype, head_dim=256, block_size=page, query_group_size=8
    )
    with pytest.raises(NotImplementedError, match="NHD"):
        pa.pa_decode(
            torch.empty_like(query),
            query,
            cache,
            cache,
            context,
            table,
            256**-0.5,
            query_length,
            compute_type=dtype,
            key_scale=None if variant == "bf16" else scale,
            value_scale=None if variant == "bf16" else scale,
            work_plan=plan,
        )


@pytest.mark.parametrize(
    "softmax_scale", [-(256**-0.5), 0.0, float("nan"), float("inf")]
)
def test_pa_decode_softmax_scale_rejected(softmax_scale, device, monkeypatch):
    """Masked scores can be scaled after masking, so the scale must be positive."""
    pa = importlib.import_module("aiter.ops.flydsl.pa_decode")

    def unexpected_compile(**_):
        pytest.fail("a rejected softmax_scale reached kernel compilation")

    monkeypatch.setattr(pa, "compile_pa_decode_tile", unexpected_compile)
    query = torch.zeros((1, 8, 256), dtype=torch.bfloat16, device=device)
    cache = torch.zeros((2, 64, 1, 256), dtype=dtypes.fp8, device=device)
    context = torch.tensor([64], dtype=torch.int32, device=device)
    table = torch.zeros((1, 1), dtype=torch.int32, device=device)
    scale = torch.ones(1, dtype=torch.float32, device=device)
    plan = pa.plan_pa_decode(context, 1, max_partitions=1)
    with pytest.raises(ValueError, match="softmax_scale"):
        pa.pa_decode(
            torch.empty_like(query),
            query,
            cache,
            cache,
            context,
            table,
            softmax_scale,
            1,
            key_scale=scale,
            value_scale=scale,
            work_plan=plan,
        )
