# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""One parametrized correctness test and a CLI performance sweep for FlyDSL PA.

    python -m pytest -q op_tests/test_flydsl_pa_decode.py
    python op_tests/test_flydsl_pa_decode.py -d bf16 -b 200 -q 4 \
        -s 16,1,128,200000 --block-size 16 128 --trans-v 0 1 \
        --per-token 1 --num-partitions 3 5

Both entry points share input generation, the FP32 reference and kernel launch.
Context lengths include the MTP query tokens. Explicit partition counts override
automatic recommendations; the CLI times attention plus the native reducer.
"""

import argparse
import importlib
import itertools
from dataclasses import dataclass
from functools import partial

import pandas as pd
import pytest
import torch

import aiter
from aiter import dtypes, per_tensor_quant, pertoken_quant
from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.test_common import benchmark, run_perftest

try:
    from aiter.ops.flydsl.pa_decode import (
        MAX_CONTEXT_PARTITIONS,
        get_recommended_splits,
        pa_decode,
        plan_pa_decode,
    )
except (ImportError, AttributeError, RuntimeError, OSError):
    MAX_CONTEXT_PARTITIONS = 256
    get_recommended_splits = pa_decode = plan_pa_decode = None

SUPPORTED_GFX = ("gfx942", "gfx950")
KV_COMPUTE_BLOCK = 256
ACCURACY_TOLERANCE = 5e-3
DEFAULT_BATCH_SIZES = [3, 81, 128]
DEFAULT_SHAPES = [
    (8, 1, 128, 257),
    (4, 1, 128, 1027),
    (8, 1, 128, 1027),
    (8, 1, 256, 1027),
    (16, 1, 128, 8192),
]
BOUNDARY_LENGTHS = (
    0,
    1,
    2,
    3,
    4,
    255,
    256,
    257,
    258,
    259,
    511,
    512,
    513,
    514,
    515,
    2303,
    2304,
    2305,
    2306,
    2307,
    4099,
)


@dataclass(frozen=True)
class DecodeCase:
    lengths: tuple[int, ...] = BOUNDARY_LENGTHS
    query_length: int = 4
    num_kv_heads: int = 1
    query_group_size: int = 16
    head_dim: int = 128
    block_size: int = 128
    trans_v: bool = True
    dtype: torch.dtype = torch.bfloat16
    num_partitions: int | None = 7
    per_token: bool = True
    sliding_window: int = 0
    sink_dtype: torch.dtype | None = None
    max_partitions: int | None = None
    workgroup_budget: int | None = None
    sparse: bool = True
    masked_scale: bool = False
    query_splits: int | None = None
    wide_kv_addressing: bool | None = None
    expected_prefetch: bool | str | None = None
    expected_query_splits: int | None = None
    compact_reduce: bool | None = None
    check_work_hints: bool = False


def _require_gpu():
    if not torch.cuda.is_available():
        pytest.skip("ROCm is not available")
    if pa_decode is None:
        pytest.skip("FlyDSL is not available")
    if get_gfx_runtime() not in SUPPORTED_GFX:
        pytest.skip(f"pa_decode is unsupported on {get_gfx_runtime()}")


@pytest.fixture(autouse=True)
def _default_cuda_device():
    # Do not leak the default device into other files in the shared CI shard.
    _require_gpu()
    previous = torch.get_default_device()
    torch.set_default_device("cuda")
    try:
        yield
    finally:
        torch.set_default_device(previous)


def run_torch(
    query,
    key_cache,
    value_cache,
    block_tables,
    context_lengths,
    key_scale,
    value_scale,
    query_length=1,
    sliding_window=0,
    sinks=None,
):
    """Dequantized FP32 GQA reference, including empty rows and infinite sinks."""
    batch = context_lengths.numel()
    heads, dim = query.shape[1:]
    kv_heads, page_size = key_cache.shape[1:3]
    group = heads // kv_heads
    queries = query.float().reshape(batch, query_length, kv_heads, group, dim)
    output = torch.zeros_like(queries)
    positions = torch.arange(query_length, device=query.device)

    for seq, length in enumerate(context_lengths.cpu().tolist()):
        if length <= 0:
            continue
        tokens = torch.arange(length, device=query.device)
        pages = block_tables[seq, tokens // page_size].long()
        offsets = tokens % page_size
        keys = key_cache[pages, :, offsets, :].float()
        values = value_cache[pages, :, :, offsets].float()
        if key_scale.numel() == 1:
            keys *= key_scale
            values *= value_scale
        else:
            keys *= key_scale[pages, :, offsets, 0, None]
            values *= value_scale[pages, :, offsets, 0, None]
        scores = torch.einsum("qhgd,khd->qhgk", queries[seq], keys) * dim**-0.5
        visible = length - query_length + 1 + positions
        masked = tokens[None, :] >= visible[:, None]
        if sliding_window > 0:
            masked |= tokens[None, :] < (visible - sliding_window)[:, None]
        scores.masked_fill_(masked[:, None, None, :], float("-inf"))
        log_denominator = torch.logsumexp(scores, dim=-1, keepdim=True)
        if sinks is not None:
            log_denominator = torch.logaddexp(
                log_denominator, sinks.float().reshape(1, kv_heads, group, 1)
            )
        # A +inf sink suppresses all finite KV logits. Fully masked rows need
        # an explicit zero because -inf - -inf is undefined without a sink.
        probs = torch.exp(scores - log_denominator)
        probs.masked_fill_(visible[:, None, None, None] <= 0, 0)
        output[seq] = torch.einsum("qhgk,khd->qhgd", probs, values)
    return output.reshape_as(query)


def _make_inputs(case, planned=False):
    """Build one sparse/dense paged cache, launch arguments and reference call."""
    torch.manual_seed(37 if case.sparse else 0)
    batch, page, dim = len(case.lengths), case.block_size, case.head_dim
    kv_heads, ql = case.num_kv_heads, case.query_length
    heads = kv_heads * case.query_group_size
    counts = [max(1, (length + page - 1) // page) for length in case.lengths]
    num_pages = sum(counts)
    quant_dtype = (
        torch.float8_e4m3fn if get_gfx_runtime() == "gfx950" else torch.float8_e4m3fnuz
    )
    query = torch.empty((batch * ql, heads, dim), dtype=case.dtype).uniform_(-0.5, 0.5)
    if case.masked_scale:
        query.zero_()  # Also exercise online Q quantization with zero Q scale.
    key = torch.empty((num_pages, kv_heads, page, dim), dtype=case.dtype).uniform_(
        -0.5, 0.5
    )
    value = torch.empty_like(key).uniform_(-0.5, 0.5)
    quantize = pertoken_quant if case.per_token else per_tensor_quant
    key_quant, key_scale = quantize(key, quant_dtype=quant_dtype)
    value_quant, value_scale = quantize(value, quant_dtype=quant_dtype)
    del key, value

    if case.per_token and case.sparse:
        token = torch.arange(num_pages * page).reshape(num_pages, 1, page, 1)
        head = torch.arange(kv_heads).reshape(1, kv_heads, 1, 1)
        key_scale *= torch.exp2(((2 * token + head) % 4 - 2).float())
        factors = torch.exp2(((token + 2 * head) % 5 - 3).float())
        # Exact binary ratios isolate W=1 masking from existing FP8 P rounding.
        value_scale = (
            factors * 2**-10 if case.sliding_window == 1 else value_scale * factors
        )

    start = 0
    for length, count in zip(case.lengths, counts):
        end = start + count
        if 0 < length <= 4 or (case.sink_dtype is not None and length in (257, 513)):
            # Positive, periodic V covers short rows and makes repeated sink
            # mass across multiple partitions observable without cancellation.
            # Per-tensor cases keep their shared scale and use larger FP8 values.
            token_values = (torch.arange(count * page) % 4 + 1).float()
            token_values *= 0.25 if case.per_token else 32
            key_quant[start:end].zero_()
            value_quant[start:end] = token_values.reshape(count, 1, page, 1).to(
                quant_dtype
            )
            if case.per_token:
                key_scale[start:end].fill_(1)
                value_scale[start:end].fill_(1)
        if case.masked_scale:
            assert case.per_token and case.sliding_window == 1
            tokens = torch.arange(count * page)
            visible = (tokens >= max(0, length - ql)) & (tokens < length)
            key_quant[start:end].zero_()
            key_scale[start:end].fill_(1)
            value_scale[start:end] = torch.where(visible, 1.0, 1e9).reshape(
                count, 1, page, 1
            )
            values = torch.where(visible, (tokens % 4 + 1).float() * 0.25, 0)
            value_quant[start:end] = values.reshape(count, 1, page, 1).to(quant_dtype)
        start = end

    selected = (
        2 * torch.randperm(num_pages) + 1 if case.sparse else torch.arange(num_pages)
    )
    physical_pages = 2 * num_pages + 1 if case.sparse else num_pages

    def scatter(tensor):
        if not case.sparse:
            return tensor
        result = torch.zeros((physical_pages, *tensor.shape[1:]), dtype=tensor.dtype)
        result[selected] = tensor
        return result

    key_quant, value_quant = scatter(key_quant), scatter(value_quant)
    if case.per_token:
        key_scale, value_scale = scatter(key_scale), scatter(value_scale)
    key_cache = (
        key_quant.reshape(physical_pages, kv_heads, page, dim // 16, 16)
        .permute(0, 1, 3, 2, 4)
        .contiguous()
    )
    value_cache = (
        value_quant.reshape(physical_pages, kv_heads, page // 16, 16, dim)
        .permute(0, 1, 2, 4, 3)
        .contiguous()
        if case.trans_v
        else value_quant.permute(0, 1, 3, 2).contiguous()
    )
    table = torch.zeros((batch, max(counts)), dtype=torch.int32)
    start = 0
    for seq, count in enumerate(counts):
        table[seq, :count] = selected[start : start + count]
        start += count
    context = torch.tensor(case.lengths, dtype=torch.int32)
    sinks = None
    if case.sink_dtype is not None:
        logits = [float("-inf"), float("inf"), -1000, 1000, -2, 0, 0.5, 5]
        if case.sink_dtype == torch.float32:
            logits += [-3e38, 3e38]
        sinks = (
            torch.tensor(logits, dtype=case.sink_dtype)
            .repeat((heads + len(logits) - 1) // len(logits))[:heads]
            .contiguous()
        )
        # Distinguish heads even when GQA is a multiple of the sentinel period.
        offset = torch.arange(heads).to(case.sink_dtype) * 0.125
        sinks = torch.where(sinks.abs() < 10, sinks + offset, sinks)
    parts = case.num_partitions
    if parts is None:
        parts = get_recommended_splits(
            batch,
            kv_heads,
            KV_COMPUTE_BLOCK // page,
            case.max_partitions,
            max_context_length=max(case.lengths),
            query_length=ql,
        )
    plan = (
        plan_pa_decode(
            context,
            kv_heads,
            max_partitions=parts,
            workgroup_budget=case.workgroup_budget,
            sliding_window=case.sliding_window,
            query_length=ql,
            total_context_length=sum(max(0, length) for length in case.lengths),
        )
        if planned
        else None
    )
    rows = ql * case.query_group_size
    shape = (
        (kv_heads, plan.capacity, rows) if planned else (batch, kv_heads, parts, rows)
    )
    psum = torch.full(shape, float("nan"), dtype=torch.float32)
    pmax = torch.full_like(psum, float("nan"))
    pout = torch.full((*shape, dim), float("nan"), dtype=case.dtype)
    args = (
        torch.full_like(query, float("nan")),
        query,
        key_cache,
        value_cache,
        context,
        table,
        dim**-0.5,
        ql,
        parts,
        KV_COMPUTE_BLOCK,
        quant_dtype,
        None,
        key_scale,
        value_scale,
        psum,
        pmax,
        pout,
        None,
        sinks,
    )
    options = {"sliding_window": case.sliding_window, "work_plan": plan}
    reference = partial(
        run_torch,
        query,
        key_quant,
        value_quant.permute(0, 1, 3, 2),
        table,
        context,
        key_scale,
        value_scale,
        query_length=ql,
        sliding_window=case.sliding_window,
        sinks=sinks,
    )
    return args, options, reference


def _run_flydsl(*args, sliding_window=0, work_plan=None):
    if work_plan is None:
        torch.ops.aiter.pa_decode_flydsl(*args, sliding_window=sliding_window)
    else:
        plan_pa_decode(
            args[4],
            args[2].shape[1],
            query_length=args[7],
            sliding_window=sliding_window,
            plan=work_plan,
        )
        pa_decode(
            *args[:8],
            context_partition_size=args[9],
            compute_type=args[10],
            query_scale=args[11],
            key_scale=args[12],
            value_scale=args[13],
            exp_sums=args[14],
            max_logits=args[15],
            temporary_output=args[16],
            alibi_slopes=args[17],
            sinks=args[18],
            sliding_window=sliding_window,
            work_plan=work_plan,
        )
    return args[0]


def _assert_close(output, reference):
    assert torch.isfinite(output).all()
    torch.testing.assert_close(
        output.float(),
        reference.float(),
        atol=ACCURACY_TOLERANCE,
        rtol=ACCURACY_TOLERANCE,
    )


def _assert_plan(plan, lengths):
    """Independent integer oracle for absolute tiles, budgets and packed slots."""
    first = [
        (
            max(0, length - plan.query_length + 1 - plan.sliding_window) // 256
            if plan.sliding_window > 0
            else 0
        )
        for length in lengths
    ]
    last = [(max(0, length) + 255) // 256 for length in lengths]
    tiles = [end - begin for begin, end in zip(first, last)]
    remaining = plan.capacity - sum(count > 0 for count in tiles)
    total = max(sum(tiles), 1)
    prefix, work, reductions = 0, [], []
    for seq, (length, begin, count) in enumerate(zip(lengths, first, tiles)):
        extra = (prefix + count) * remaining // total - prefix * remaining // total
        parts = min(int(count > 0) + extra, count, plan.max_partitions)
        prefix += count
        reductions.append([len(work), parts])
        for part in range(parts):
            work.append(
                [
                    seq,
                    begin + part * count // parts,
                    begin + (part + 1) * count // parts,
                    length,
                ]
            )
    assert len(work) <= plan.capacity
    active = len(work)
    work += [[0, 0, 0, 0]] * (plan.capacity - active)
    assert plan.reduce_info.cpu().tolist() == reductions
    assert plan.work_info.cpu().tolist() == work
    return active


def _assert_contracts(args, options):
    """Reuse the valid inputs for API validation instead of separate fixtures."""
    heads = args[1].shape[1]
    for invalid, error in [
        ([0.0] * heads, TypeError),
        (torch.zeros(1, heads), ValueError),
        (torch.zeros(heads - 1), ValueError),
        (torch.zeros(heads, dtype=torch.int32), TypeError),
        (torch.zeros(heads, dtype=torch.float64), TypeError),
        (torch.zeros(heads, device="cpu"), ValueError),
        (torch.zeros(heads * 2)[::2], ValueError),
    ]:
        with pytest.raises(error, match="sinks"):
            pa_decode(*args[:-1], sinks=invalid, **options)
    for value, error in [(-2, ValueError), (1.5, TypeError)]:
        with pytest.raises(error, match="sliding_window"):
            pa_decode(*args, **{**options, "sliding_window": value})
        with pytest.raises(error, match="sliding_window"):
            plan_pa_decode(args[4], args[2].shape[1], sliding_window=value)
    for value, error in [(0, ValueError), (1.5, TypeError)]:
        with pytest.raises(error, match="query_length"):
            pa_decode(*args[:7], value, *args[8:], **options)
        with pytest.raises(error, match="query_length"):
            plan_pa_decode(args[4], args[2].shape[1], query_length=value)
    with pytest.raises(ValueError, match="required without work_plan"):
        pa_decode(*args[:8])
    with pytest.raises(TypeError, match="work_plan must be a PADecodePlan"):
        pa_decode(*args[:8], work_plan=object())
    plan = options["work_plan"]
    if plan is not None:
        expected = args[0].clone()
        for bound in (None, plan.max_partitions):
            pa_decode(*args[:8], bound, *args[9:], **options)
            torch.testing.assert_close(args[0], expected, atol=0, rtol=0)
        with pytest.raises(ValueError, match="must match work_plan.max_partitions"):
            pa_decode(
                *args[:8], 1 if plan.max_partitions != 1 else 2, *args[9:], **options
            )
        context, heads = args[4], args[2].shape[1]
        reuse = {
            "max_partitions": plan.max_partitions,
            "sliding_window": plan.sliding_window,
            "query_length": plan.query_length,
            "plan": plan,
        }
        changes = [
            (
                {"max_partitions": 1 if plan.max_partitions != 1 else 2},
                "max_partitions",
            ),
            ({"sliding_window": plan.sliding_window + 1}, "sliding_window"),
            ({"workgroup_budget": 1}, "workgroup_budget"),
        ]
        counts = plan.num_partitions
        plan_pa_decode(
            context,
            heads,
            plan=plan,
            sliding_window=plan.sliding_window,
            query_length=plan.query_length,
        )
        assert torch.equal(counts, plan.reduce_info[:, 1])
        with pytest.raises(ValueError, match="only used when creating"):
            plan_pa_decode(context, heads, total_context_length=1, **reuse)
        if plan.sliding_window > 0:
            changes.append(({"query_length": plan.query_length + 1}, "query_length"))
        for change, message in changes:
            with pytest.raises(ValueError, match=message):
                plan_pa_decode(context, heads, **{**reuse, **change})
        for lengths, kv_heads in [(context.repeat(2), heads), (context, heads + 1)]:
            with pytest.raises(ValueError):
                plan_pa_decode(lengths, kv_heads, **reuse)
        with pytest.raises(ValueError, match="sliding_window"):
            pa_decode(*args, **{**options, "sliding_window": plan.sliding_window + 1})
        if plan.sliding_window > 0:
            wrong_plan = plan_pa_decode(
                context,
                heads,
                max_partitions=plan.max_partitions,
                sliding_window=plan.sliding_window,
                query_length=plan.query_length + 1,
            )
            with pytest.raises(ValueError, match="query_length"):
                pa_decode(*args, **{**options, "work_plan": wrong_plan})


def _assert_work_hint_policy(monkeypatch):
    """Check host sizing and GPU coverage without allocating long KV caches."""
    with monkeypatch.context() as patch:
        patch.setattr(
            torch.cuda,
            "get_device_properties",
            lambda *_args, **_kwargs: type(
                "Props", (), {"multi_processor_count": 256}
            )(),
        )
        recommendations = [
            (1, 200000, 4, None, 256),
            (8, 200000, 4, None, 64),
            (16, 200000, 4, None, 32),
            (64, 100000, 4, None, 8),
            (200, 200000, 2, None, 8),
            (200, 200000, 3, None, 16),
            (200, 200000, 4, None, 16),
            (200, 200000, 4, 8, 8),
            (200, 200000, 4, 5, 5),
        ]
        for blocks in (2, 16):
            for batch, context, ql, clamp, expected in recommendations:
                assert (
                    get_recommended_splits(
                        batch,
                        1,
                        blocks,
                        clamp,
                        max_context_length=context,
                        query_length=ql,
                    )
                    == expected
                )
        configurations = [
            ([200000] * 200, 4, None, 256, 0, 3200),
            ([200000] + [1024] * 199, 4, None, 256, 0, 512),
            ([100000] * 64, 4, None, 256, 0, 512),
            ([200000], 4, None, 256, 0, 256),
            ([200000] * 200, 1, None, 256, 0, 512),
            ([200000] * 200, 4, 800, 256, 0, 800),
            ([200000] * 200, 4, None, 8, 0, 1600),
            ([200000] * 200, 4, None, 256, 1, 512),
            ([200000] * 200, 4, None, 256, 257, 512),
            ([200000] * 200, 4, None, 256, 65536, 1600),
        ]
        for lengths, ql, budget, limit, window, capacity in configurations:
            context = torch.tensor(lengths, dtype=torch.int32)
            plan = plan_pa_decode(
                context,
                1,
                max_partitions=limit,
                workgroup_budget=budget,
                total_context_length=sum(lengths),
                query_length=ql,
                sliding_window=window,
            )
            assert plan.capacity == capacity
            _assert_plan(plan, lengths)
            counts = plan.num_partitions
            updated = [0] * len(lengths)
            updated[-1] = 200003
            context.copy_(torch.tensor(updated, dtype=torch.int32))
            plan_pa_decode(
                context, 1, plan=plan, query_length=ql, sliding_window=window
            )
            _assert_plan(plan, updated)
            assert torch.equal(counts, plan.reduce_info[:, 1])


def _case(
    name,
    shape=(4, 1, 16, 128),
    cache=(128, 1, 1),
    parts=7,
    window=0,
    sink=None,
    **kwargs,
):
    # shape = (QL, KV heads, GQA, D); cache = (page size, transposed V, per-token).
    ql, heads, group, dim = shape
    page, trans_v, per_token = cache
    return pytest.param(
        DecodeCase(
            query_length=ql,
            num_kv_heads=heads,
            query_group_size=group,
            head_dim=dim,
            block_size=page,
            trans_v=bool(trans_v),
            per_token=bool(per_token),
            num_partitions=parts,
            sliding_window=window,
            sink_dtype=sink,
            **kwargs,
        ),
        id=name,
    )


# Each row uses the same eager/graph checks with static and packed plans.
BF16, FP16, FP32 = torch.bfloat16, torch.float16, torch.float32
CASES = [
    _case("scalar-direct", (1, 2, 8, 128), (16, 1, 0), 1),
    _case("scalar-window-sinks", (1, 2, 8, 128), (128, 1, 0), 7, 257, FP32),
    _case(
        "register-64-65",
        (1, 2, 4, 256),
        (64, 1, 0),
        86,
        sink=FP16,
        dtype=FP16,
        lengths=(0, 1, 257, 16384, 16385),
    ),
    _case("head64-fp16", (2, 2, 4, 64), (16, 0, 0), 7, 257, BF16, dtype=FP16),
    _case("head1024", (1, 2, 4, 1024), (128, 1, 0), 256, 8192, FP32),
    _case(
        "np1-fused-sink", cache=(16, 1, 1), parts=1, window=1, sink=FP32, query_splits=1
    ),
    _case("np1-split-sink", parts=1, window=1, sink=BF16, query_splits=4),
    _case("mtp3-window", (3, 1, 16, 128), (128, 0, 1), 7, 257, FP16),
    _case("mtp2-odd-parts", (2, 2, 16, 128), (16, 0, 1), 3),
    _case(
        "mtp2-query-split",
        (2, 1, 16, 128),
        parts=3,
        lengths=(257, 259),
        expected_query_splits=2,
        expected_prefetch=True,
    ),
    _case(
        "hkv2-prefetch-1wg",
        (1, 2, 8, 128),
        parts=8,
        window=1,
        sink=FP32,
        lengths=(257,) * 16,
        expected_prefetch="occupancy",
    ),
    _case(
        "hkv2-prefetch-2wg",
        (1, 2, 8, 128),
        parts=8,
        lengths=(257,) * 32,
        expected_prefetch="occupancy",
    ),
    _case(
        "hkv2-prefetch-page16",
        (1, 2, 8, 128),
        (16, 1, 1),
        8,
        257,
        FP16,
        lengths=(257,) * 32,
        expected_prefetch=True,
    ),
    _case(
        "hkv2-prefetch-plain-v",
        (1, 2, 16, 128),
        (128, 0, 1),
        8,
        sink=BF16,
        lengths=(257,) * 32,
        expected_prefetch=True,
    ),
    _case("hkv2-direct-sinks", (1, 2, 8, 128), parts=1, window=1, sink=BF16),
    _case(
        "long-200k",
        cache=(128, 0, 1),
        parts=256,
        sink=FP32,
        lengths=(0, 1, 3, 4, 255, 256, 257, 200003),
    ),
    _case("window-int64", cache=(16, 1, 1), window=2**40, wide_kv_addressing=True),
    _case(
        "masked-scale-decode",
        (1, 1, 16, 128),
        window=1,
        sink=FP32,
        lengths=(511,),
        masked_scale=True,
    ),
    _case("masked-scale-mtp", window=1, lengths=(511,), masked_scale=True),
    _case(
        "disabled-window-wide",
        cache=(16, 0, 1),
        parts=256,
        window=-1,
        sink=FP32,
        query_splits=1,
        wide_kv_addressing=True,
    ),
    _case(
        "large-batch-auto",
        cache=(128, 0, 1),
        parts=None,
        lengths=(257,) * 200,
        expected_query_splits=1,
    ),
    _case("small-batch-auto", (1, 1, 8, 128), (16, 0, 0), None, lengths=(257,) * 3),
    _case(
        "exact-parts-override",
        (1, 1, 16, 128),
        parts=5,
        lengths=(200000,),
        max_partitions=4,
    ),
    _case(
        "window255-budget",
        (2, 2, 16, 128),
        (64, 0, 1),
        64,
        255,
        BF16,
        workgroup_budget=17,
    ),
    _case("window256", (3, 2, 8, 128), (128, 0, 1), 7, 256),
    _case("window509", cache=(64, 0, 1), parts=86, window=509, sink=FP16),
    _case("window8192", (2, 1, 16, 128), parts=256, window=8192),
    _case("window-int32-max", cache=(16, 0, 1), parts=1, window=2**31 - 1),
    _case("window-int32-overflow", cache=(16, 0, 1), parts=1, window=2**31, sink=FP32),
    _case(
        "long-np64",
        cache=(16, 1, 1),
        parts=64,
        lengths=(0, 3, 257, 16384, 16385, 65537),
    ),
    _case("head64-mtp-window", (4, 2, 4, 64), (16, 1, 1), 256, 1, FP32),
    _case("scalar-fp16-mtp", cache=(128, 1, 0), parts=3, dtype=FP16),
    _case(
        "fused-narrow",
        cache=(16, 0, 1),
        parts=1,
        query_splits=1,
        wide_kv_addressing=False,
    ),
    _case("fused-wide", parts=8, query_splits=1, wide_kv_addressing=True),
    _case(
        "fused-hkv2-window",
        (4, 2, 16, 128),
        (128, 0, 1),
        8,
        257,
        BF16,
        query_splits=1,
        wide_kv_addressing=False,
    ),
    _case("empty", window=1, sink=FP16, lengths=(0,) * 4),
]


# Compact planned reduction: chunk boundaries, sinks, windows and both dtypes.
for parts, heads, page, trans, sink, dtype, window in (
    (65, 1, 16, False, None, BF16, 0),
    (86, 2, 128, True, FP32, BF16, 0),
    (128, 1, 128, True, BF16, BF16, 0),
    (256, 2, 16, False, FP32, FP16, 0),
    (256, 1, 128, True, FP32, BF16, 257),
    (256, 1, 16, True, FP32, BF16, 1),
):
    CASES.append(
        _case(
            f"compact-{parts}-h{heads}-p{page}-w{window}",
            (4, heads, 16, 128),
            (page, trans, 1),
            parts,
            window,
            sink,
            dtype=dtype,
            lengths=(0, 1, 3, 4, 257, 16387, 70003),
            workgroup_budget=7 * heads * parts,
            compact_reduce=True,
        )
    )
CASES.extend(
    [
        _case("compact-auto", parts=256, lengths=(257,) * 64),
        _case(
            "work-hints", parts=256, lengths=(0, 1, 257, 4099), check_work_hints=True
        ),
    ]
)


@pytest.mark.parametrize("planned", [False, True], ids=["static", "planned"])
@pytest.mark.parametrize("case", CASES)
def test_pa_decode(case, planned, monkeypatch):
    """One real-kernel path for all layouts, masks, sinks, plans and replays."""
    args, options, reference = _make_inputs(case, planned)
    output, query, _, _, context = args[:5]
    scratch, sinks, plan = args[14:17], args[-1], options["work_plan"]
    module = importlib.import_module("aiter.ops.flydsl.pa_decode")
    compile_tile = module.compile_pa_decode_tile
    selected = []

    def compile_checked(**kwargs):
        if case.query_splits is not None and not planned:
            kwargs["query_splits"] = case.query_splits
        if case.wide_kv_addressing is not None:
            kwargs["wide_kv_addressing"] = case.wide_kv_addressing
        assert kwargs["use_sinks"] == (
            sinks is not None and args[8] == 1 and not planned
        )
        selected.append(kwargs)
        return compile_tile(**kwargs)

    compile_reduce = module.compile_pa_decode_ps_reduce

    def compile_reduce_checked(**kwargs):
        num_cus = torch.cuda.get_device_properties(query.device).multi_processor_count
        output_rows = len(case.lengths) * case.query_length * query.shape[1]
        assert kwargs["compact_work_plan"] == (
            planned
            and case.head_dim == 128
            and args[8] > 64
            and output_rows >= 4 * num_cus
        )
        if planned and case.compact_reduce is not None:
            kwargs["compact_work_plan"] = case.compact_reduce
        return compile_reduce(**kwargs)

    monkeypatch.setattr(module, "compile_pa_decode_ps_reduce", compile_reduce_checked)

    def unexpected_reducer(*_args, **_kwargs):
        raise AssertionError("static NP=1 must not launch a reducer")

    monkeypatch.setattr(module, "compile_pa_decode_tile", compile_checked)
    if not planned and args[8] == 1:
        monkeypatch.setattr(module, "launch_pa_decode_ps_reduce", unexpected_reducer)

    def check(disabled_sink=False):
        _assert_close(output, reference(sinks=None) if disabled_sink else reference())
        if plan is not None:
            active = _assert_plan(plan, context.cpu().tolist())
            for tensor in scratch:
                assert torch.isnan(tensor[:, active:]).all()
        elif args[8] == 1:
            assert all(torch.isnan(tensor).all() for tensor in scratch)

    _run_flydsl(*args, **options)
    check()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _run_flydsl(*args, **options)
    original_sinks = sinks.clone() if sinks is not None else None
    for step, lengths in enumerate(
        (
            tuple(max(0, length - 7) for length in case.lengths),
            (0,) * len(case.lengths),
            case.lengths,
        )
    ):
        context.copy_(torch.tensor(lengths, dtype=torch.int32))
        if sinks is not None:
            if step == 2:
                sinks.fill_(float("-inf"))
            else:
                sinks.copy_(original_sinks.roll(step + 1))
        for tensor in (output, *scratch):
            tensor.fill_(float("nan"))
        graph.replay()
        check(disabled_sink=step == 2)

    if case.expected_prefetch is not None:
        expected = case.expected_prefetch
        if expected == "occupancy":
            workgroups = len(case.lengths) * case.num_kv_heads * args[8]
            expected = (
                workgroups
                <= torch.cuda.get_device_properties(query.device).multi_processor_count
            )
        expected = bool(expected) and not planned and get_gfx_runtime() == "gfx950"
        assert selected and all(config["prefetch_v"] == expected for config in selected)
    if case.expected_query_splits is not None:
        expected = (
            case.expected_query_splits
            if not planned and get_gfx_runtime() == "gfx950"
            else 1
        )
        assert selected and all(
            config["query_splits"] == expected for config in selected
        )

    _assert_contracts(args, options)
    if plan is not None:
        # Exercise planner integer limits without allocating an INT32_MAX KV cache.
        lengths = (-1, 0, 1, 257, 2**31 - 1)
        extreme_plan = plan_pa_decode(
            torch.tensor(lengths, dtype=torch.int32),
            case.num_kv_heads,
            max_partitions=args[8],
            sliding_window=case.sliding_window,
            query_length=case.query_length,
        )
        _assert_plan(extreme_plan, lengths)
        if case.check_work_hints:
            _assert_work_hint_policy(monkeypatch)


@benchmark()
def run_pa_decode_tile_case(
    batch_size,
    num_query_heads,
    num_kv_heads,
    head_dim,
    context_length,
    block_size,
    dtype,
    trans_v,
    max_partitions=None,
    per_token=False,
    query_length=1,
    num_partitions=None,
):
    """CLI-only timing wrapper around the same input/reference/launch helpers."""
    if min(batch_size, context_length, query_length, num_query_heads, num_kv_heads) < 1:
        raise ValueError(
            "batch, context, query length and head counts must be positive"
        )
    if num_query_heads % num_kv_heads:
        raise ValueError("num_query_heads must be divisible by num_kv_heads")
    if dtype not in (dtypes.fp16, dtypes.bf16):
        raise ValueError("pa_decode only supports fp16/bf16")
    if num_partitions is not None and not 1 <= num_partitions <= MAX_CONTEXT_PARTITIONS:
        raise ValueError(f"num_partitions must be in [1, {MAX_CONTEXT_PARTITIONS}]")
    case = DecodeCase(
        lengths=(context_length,) * batch_size,
        query_length=query_length,
        num_kv_heads=num_kv_heads,
        query_group_size=num_query_heads // num_kv_heads,
        head_dim=head_dim,
        block_size=block_size,
        dtype=dtype,
        trans_v=trans_v,
        num_partitions=num_partitions,
        max_partitions=max_partitions,
        per_token=per_token,
        sparse=False,
    )
    args, options, reference_call = _make_inputs(case)
    reference = reference_call()
    del reference_call
    # Keep tensors positional so allocation rotation accounts for their memory.
    output, us = run_perftest(_run_flydsl, *args, **options)
    _assert_close(output, reference)
    query, table = args[1], args[5]
    attended = sum(
        max(0, context_length - query_length + 1 + p) for p in range(query_length)
    )
    flops = 4 * batch_size * num_query_heads * attended * head_dim
    scale_elements = batch_size * num_kv_heads * context_length if per_token else 1
    nbytes = (
        2 * query.numel() * query.element_size()
        + 2
        * batch_size
        * num_kv_heads
        * context_length
        * head_dim
        * args[2].element_size()
        + table.numel() * table.element_size()
        + args[4].numel() * args[4].element_size()
        + scale_elements * (args[12].element_size() + args[13].element_size())
    )
    return {
        "gfx": get_gfx_runtime(),
        "partitions": args[8],
        "trans_v": trans_v,
        "per_token": per_token,
        "flydsl us": us,
        "flydsl us/token": us / (batch_size * query_length),
        "flydsl TFLOPS": flops / us / 1e6,
        "flydsl TB/s": nbytes / us / 1e6,
        "flydsl err": 0,
    }


def _positive_int(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="FlyDSL pa_decode correctness + perf sweep"
    )
    parser.add_argument(
        "-d", "--dtype", type=dtypes.str2Dtype, nargs="*", default=[dtypes.bf16]
    )
    parser.add_argument(
        "-b", "--batch", type=_positive_int, nargs="*", default=DEFAULT_BATCH_SIZES
    )
    parser.add_argument(
        "-q",
        "--query-length",
        type=_positive_int,
        nargs="+",
        default=[1],
        help="Query tokens per sequence; context lengths include them.",
    )
    parser.add_argument(
        "-s",
        "--shapes",
        type=dtypes.str2tuple,
        nargs="*",
        default=DEFAULT_SHAPES,
        help="num_query_heads,num_kv_heads,head_dim,context_length",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        nargs="*",
        choices=[16, 64, 128],
        default=[16, 64, 128],
    )
    parser.add_argument(
        "--trans-v",
        type=int,
        nargs="*",
        choices=[0, 1],
        default=[0, 1],
        help="0: plain V cache; 1: transposed V cache.",
    )
    parser.add_argument(
        "--per-token",
        type=int,
        nargs="*",
        choices=[0, 1],
        default=[0],
        help="0: per-tensor KV scales; 1: per-token KV scales.",
    )
    parser.add_argument(
        "--max-partitions",
        type=int,
        default=None,
        help="Upper clamp for automatic splits (4..256); 8 keeps the legacy clamp.",
    )
    parser.add_argument(
        "--num-partitions",
        type=_positive_int,
        nargs="+",
        default=[None],
        help="Exact split counts (1..256), overriding --max-partitions.",
    )
    args = parser.parse_args(argv)
    if (
        args.max_partitions is not None
        and not 4 <= args.max_partitions <= MAX_CONTEXT_PARTITIONS
    ):
        parser.error(f"--max-partitions must be in [4, {MAX_CONTEXT_PARTITIONS}]")
    if any(n is not None and n > MAX_CONTEXT_PARTITIONS for n in args.num_partitions):
        parser.error(f"--num-partitions must be in [1, {MAX_CONTEXT_PARTITIONS}]")
    for shape in args.shapes:
        if not isinstance(shape, tuple) or len(shape) != 4 or any(n < 1 for n in shape):
            parser.error("each --shapes value must contain four positive integers")
        if shape[0] % shape[1]:
            parser.error("num_query_heads must be divisible by num_kv_heads")
    return args


def main():
    # Parse before GPU checks so --help and invalid options remain usable.
    args = _parse_args()
    if not torch.cuda.is_available():
        aiter.logger.warning("ROCm is not available; skipping pa_decode")
        return
    if get_gfx_runtime() not in SUPPORTED_GFX or pa_decode is None:
        aiter.logger.warning("FlyDSL pa_decode is unavailable or unsupported; skipping")
        return
    torch.set_default_device("cuda")
    rows = []
    for dtype, batch, shape, page, trans_v, per_token, ql, parts in itertools.product(
        args.dtype,
        args.batch,
        args.shapes,
        args.block_size,
        args.trans_v,
        args.per_token,
        args.query_length,
        args.num_partitions,
    ):
        heads, kv_heads, dim, context = shape
        rows.append(
            run_pa_decode_tile_case(
                batch,
                heads,
                kv_heads,
                dim,
                context,
                page,
                dtype,
                bool(trans_v),
                args.max_partitions,
                bool(per_token),
                query_length=ql,
                num_partitions=parts,
            )
        )
    aiter.logger.info(
        "pa_decode summary (markdown):\n%s", pd.DataFrame(rows).to_markdown(index=False)
    )


if __name__ == "__main__":
    main()
