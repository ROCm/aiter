# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""One parametrized correctness/contract test and a CLI performance sweep.

    python -m pytest -q op_tests/test_flydsl_pa_decode.py
    python op_tests/test_flydsl_pa_decode.py -d bf16 -b 200 -q 4 \
        -s 16,1,128,200000 --block-size 16 128 --trans-v 0 1 \
        --per-token 1 --num-partitions 3 5

Both entry points share input generation, the FP32 reference and kernel launch.
Context lengths include the MTP query tokens. Explicit partition counts override
automatic recommendations; the CLI times attention plus the native reducer.
Positive sliding windows require a work plan: planned cases check numerics and
graph replays, while static cases check rejection by the core and wrapper APIs.
Disabled windows retain both static and planned numerical coverage.
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
from aiter.ops.attention import pa_decode_flydsl
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
    check_schedule: bool = False


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
        )
    plan = (
        plan_pa_decode(
            context,
            kv_heads,
            max_partitions=parts,
            workgroup_budget=case.workgroup_budget,
            sliding_window=case.sliding_window,
            query_length=ql if case.sliding_window > 0 else 1,
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
    # Fixtures use the core signature; the wrapper inserts ps before sinks.
    args = (*args[:-1], True, args[-1])
    if work_plan is None:
        torch.ops.aiter.pa_decode_flydsl(*args, sliding_window=sliding_window)
    else:
        plan_pa_decode(
            args[4],
            args[2].shape[1],
            max_partitions=args[8],
            query_length=args[7],
            sliding_window=sliding_window,
            plan=work_plan,
        )
        pa_decode_flydsl(*args, sliding_window=sliding_window, work_plan=work_plan)
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
    plan = options["work_plan"]
    if plan is not None:
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


def _max_window_tiles(window, query_length):
    # The union has W + QL - 1 tokens and can start at any offset.
    union_tokens = window + query_length - 1
    return 1 + (union_tokens + KV_COMPUTE_BLOCK - 2) // KV_COMPUTE_BLOCK


def _expected_single_tile_plan(kwargs):
    capacity, window = kwargs["work_capacity"], kwargs["sliding_window"]
    if not kwargs["use_work_plan"] or capacity is None or window <= 0:
        return False
    tiles = _max_window_tiles(window, kwargs["query_length"])
    return kwargs["num_partitions"] >= tiles and capacity >= kwargs["num_seqs"] * tiles


def _expected_batch_first_plan_grid(kwargs, query_splits):
    """Independent shape/occupancy oracle for the measured grid domain."""
    return (
        get_gfx_runtime() == "gfx950"
        and kwargs["head_dim"] == 128
        and kwargs["query_dtype"] == "bf16"
        and kwargs["per_token_kv"]
        and _expected_single_tile_plan(kwargs)
        and kwargs["num_partitions"] <= 64
        and kwargs["query_length"] == query_splits == 4
        and kwargs["num_kv_heads"] == 1
        and kwargs["query_group_size"] == 16
        and kwargs["block_size"] == 128
        and 4096 <= kwargs["sliding_window"] <= 8192
        and 4 <= kwargs["num_seqs"] <= 24
        and kwargs["work_capacity"]
        == kwargs["num_seqs"]
        * _max_window_tiles(kwargs["sliding_window"], kwargs["query_length"])
        and kwargs["num_compute_units"] >= (kwargs["work_capacity"] + 1) // 2
    )


def _assert_schedule_cache(compile_tile, kwargs):
    """Check schedule equivalence through handles, without launching fake grids."""
    workgroups = kwargs["num_seqs"] * kwargs["num_kv_heads"] * kwargs["num_partitions"]
    planned, ql = kwargs["use_work_plan"], kwargs["query_length"]
    cached = {}

    def check(cus, expected, **changes):
        config = {**kwargs, "num_compute_units": cus, **changes}
        compiled = compile_tile(**config)
        expected = (
            *expected,
            _expected_single_tile_plan(config),
            _expected_batch_first_plan_grid(config, expected[0]),
        )
        # Only these compile constants vary within this helper. Runtime sizes
        # must share handles iff they select the same expected schedule.
        compile_key = (
            config["num_partitions"],
            config["query_length"],
            config["trans_v"],
        )
        schedules = cached.setdefault(compile_key, {})
        for schedule, previous in schedules.items():
            same_schedule = schedule == expected
            assert (compiled is previous) == same_schedule
            assert (compiled["kernel"] is previous["kernel"]) == same_schedule
            assert (compiled["launch"] is previous["launch"]) == same_schedule
        schedules.setdefault(expected, compiled)

    def split_boundary(capacity, **changes):
        config = {**kwargs, **changes, "work_capacity": capacity}
        tasks = config["num_seqs"] * config["num_partitions"]
        if config["use_work_plan"]:
            if capacity is not None:
                tasks = capacity
            if config["sliding_window"] > 0:
                max_tiles = _max_window_tiles(config["sliding_window"], ql)
                tasks = min(tasks, config["num_seqs"] * max_tiles)
        queries_per_task = ql
        if (
            _expected_single_tile_plan(config)
            and ql == 4
            and config["block_size"] == 128
            and config["trans_v"]
        ):
            queries_per_task = 1
        return (queries_per_task * config["num_kv_heads"] * tasks + 1) // 2

    def ql1_points(**changes):
        config = {**kwargs, **changes}
        dense = config["num_seqs"] * config["num_kv_heads"] * config["num_partitions"]
        eligible = (
            config["head_dim"] == 128
            and config["query_dtype"] == "bf16"
            and config["block_size"] in (16, 128)
            and 8 <= config["query_group_size"] <= 16
        )
        # Express the independent expectation as the minimum CU count. A
        # proven one-tile plan has at most B * Hkv * window_tiles workgroups,
        # regardless of excess capacity or the nominal B * Hkv * NP grid.
        threshold = None
        if eligible:
            threshold = (
                dense if config["block_size"] == 128 and config["trans_v"] else 0
            )
            if _expected_single_tile_plan(config):
                tasks = (
                    config["num_seqs"]
                    * config["num_kv_heads"]
                    * _max_window_tiles(config["sliding_window"], 1)
                )
                threshold = min(threshold, (tasks + 1) // 2)
        cus = {max(1, dense - 1), dense, dense + 1}
        if threshold is not None:
            cus.update((max(1, threshold - 1), max(1, threshold), threshold + 1))
        return [
            (cu, (1, threshold is not None and cu >= threshold)) for cu in sorted(cus)
        ]

    if not kwargs["per_token_kv"]:
        # Scalar prefetch requires CU < workgroups <= 2 * CU.
        half = (workgroups + 1) // 2
        points = [
            (workgroups, (1, False)),
            (workgroups - 1, (1, True)),
            (half, (1, True)),
            (half - 1, (1, False)),
        ]
    elif ql == 1:
        points = ql1_points()
    else:
        # Ordinary grids count QL CTAs per task; one-tile QL4/page128/transV
        # plans permit two tasks per CU. Round up for an odd workgroup count.
        boundary = split_boundary(kwargs["work_capacity"])
        split = (ql, True)
        points = [(boundary - 1, (1, False)), (boundary, split), (boundary + 1, split)]
    for cus, expected in points:
        check(cus, expected)

    if kwargs["per_token_kv"] and ql == 1:
        # Head counts affect occupancy, not single-query M1 eligibility.
        for heads in (1, 2, 3):
            for cus, expected in ql1_points(num_kv_heads=heads):
                check(cus, expected, num_kv_heads=heads)
    elif kwargs["per_token_kv"]:
        # Explicit split overrides must also select matching prefetching.
        check(boundary - 1, (ql, True), query_splits=ql)
        check(boundary + 1, (1, False), query_splits=1)

    # Capacity selects splitting and the semantic one-tile guarantee, not a
    # raw cache key. The guarantee uses full QL even when queries are split.
    capacities = [
        None,
        kwargs["num_seqs"],
        kwargs["num_seqs"] * kwargs["num_partitions"],
    ]
    if kwargs["sliding_window"] > 0:
        full_window = kwargs["num_seqs"] * _max_window_tiles(
            kwargs["sliding_window"], ql
        )
        capacities += [full_window - 1, full_window, full_window + 1]
    for capacity in dict.fromkeys(capacities):
        capacity_points = points
        if kwargs["per_token_kv"] and ql > 1:
            boundary = split_boundary(capacity)
            capacity_points = [(boundary - 1, (1, False)), (boundary, (ql, True))]
        elif kwargs["per_token_kv"] and ql == 1:
            capacity_points = ql1_points(work_capacity=capacity)
        for cus, expected in capacity_points:
            check(cus, expected, work_capacity=capacity)

    if kwargs["sliding_window"] > 0 and kwargs["per_token_kv"] and ql == 1:
        # Isolate NP's single-tile guarantee from the already sufficient
        # capacity; hypothetical metadata is compiled but never launched.
        tiles = _max_window_tiles(kwargs["sliding_window"], 1)
        for parts in dict.fromkeys((max(1, tiles - 1), tiles)):
            for cus, expected in ql1_points(
                num_partitions=parts, work_capacity=full_window
            ):
                check(cus, expected, num_partitions=parts, work_capacity=full_window)

    if kwargs["sliding_window"] > 0 and kwargs["per_token_kv"] and ql > 1:
        # Hypothetical metadata isolates the NP guard from the capacity guard.
        # Real plans cannot exceed B * NP; these compile-only grids never launch.
        tiles = _max_window_tiles(kwargs["sliding_window"], ql)
        for parts in (tiles - 1, tiles, tiles + 1):
            boundary = split_boundary(full_window, num_partitions=parts)
            for cus, expected in [(boundary - 1, (1, False)), (boundary, (ql, True))]:
                check(
                    cus,
                    expected,
                    num_partitions=parts,
                    work_capacity=full_window,
                )
        if planned and kwargs["block_size"] == 128 and kwargs["trans_v"]:
            # Enough capacity alone must not widen plain-V or QL2 thresholds.
            wider_boundary = (full_window * kwargs["num_kv_heads"] + 1) // 2
            check(
                wider_boundary,
                (1, False),
                trans_v=False,
                work_capacity=full_window,
            )
            check(
                wider_boundary,
                (1, False),
                query_length=2,
                work_capacity=full_window,
            )

    if (
        planned
        and kwargs["per_token_kv"]
        and ql == 4
        and kwargs["num_kv_heads"] == 1
        and kwargs["block_size"] == 128
        and kwargs["sliding_window"] > 0
    ):
        # Probe both batch endpoints and just outside them for every window.
        # Out-of-range windows must stay flat even for an in-range batch.
        # Plain V retains the full QL split threshold in either address width.
        # These hypothetical grids compile but never launch against this plan.
        tiles = _max_window_tiles(kwargs["sliding_window"], ql)
        for batch in (3, 4, 5, 12, 24, 25):
            capacity = batch * tiles
            boundary = split_boundary(capacity, num_seqs=batch, num_partitions=tiles)
            for cus in (boundary, boundary + 1):
                check(
                    cus,
                    (4, True),
                    num_seqs=batch,
                    num_partitions=tiles,
                    work_capacity=capacity,
                )
        # NP is itself a compile constant. Compare B4/B25 within each cap:
        # only in-range windows with NP64 separate handles; NP65 stays flat.
        for parts in (64, 65):
            for batch in (4, 25):
                capacity = batch * tiles
                boundary = split_boundary(
                    capacity, num_seqs=batch, num_partitions=parts
                )
                check(
                    boundary,
                    (4, True),
                    num_seqs=batch,
                    num_partitions=parts,
                    work_capacity=capacity,
                )

    # Runtime size/CU metadata must not fragment the kernel specialization cache.
    cus, expected = points[-1]
    changes = {"num_seqs": kwargs["num_seqs"] * 2}
    if planned and kwargs["work_capacity"] is not None:
        changes["work_capacity"] = kwargs["work_capacity"] * 2
    multiplier = 2
    if not kwargs["per_token_kv"]:
        changes["num_kv_heads"] = kwargs["num_kv_heads"] * 2
        multiplier = 4
    check(cus * multiplier, expected, **changes)


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


# Positive-window rows reject static calls and run eager/graph checks with plans.
# Disabled-window rows retain the same numerical checks in both modes.
BF16, FP16, FP32 = torch.bfloat16, torch.float16, torch.float32
CASES = [
    _case("scalar-direct", (1, 2, 8, 128), (16, 1, 0), 1),
    _case(
        "scalar-window-sinks",
        (1, 2, 8, 128),
        (128, 1, 0),
        7,
        257,
        FP32,
        check_schedule=True,
    ),
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
    # QL4/W1 can span two tiles, including when explicitly split into four CTAs.
    # NP1 therefore retains the multi-tile fallback even with a full plan budget.
    _case(
        "np1-fused-sink", cache=(16, 1, 1), parts=1, window=1, sink=FP32, query_splits=1
    ),
    _case("np1-split-sink", parts=1, window=1, sink=BF16, query_splits=4),
    _case(
        "split2-hkv2-full-m1",
        (2, 2, 16, 128),
        parts=17,
        window=1024,
        sink=FP32,
        lengths=(0, 769, 1025, 2049),
        query_splits=2,
        wide_kv_addressing=True,
    ),
    _case(
        "split4-hkv2-full-m1",
        (4, 2, 16, 128),
        (16, 0, 1),
        parts=34,
        window=4096,
        sink=BF16,
        lengths=(1, 257, 4353, 0),
        query_splits=4,
    ),
    _case(
        "full-m1-fp16-d256",
        (2, 2, 8, 256),
        (64, 1, 1),
        parts=5,
        window=257,
        sink=FP16,
        dtype=FP16,
        lengths=(0, 1, 257, 513),
        query_splits=1,
    ),
    _case("mtp3-window", (3, 1, 16, 128), (128, 0, 1), 7, 257, FP16),
    _case("mtp2-odd-parts", (2, 2, 16, 128), (16, 0, 1), 3),
    _case(
        "mtp2-query-split",
        (2, 1, 16, 128),
        parts=3,
        lengths=(257, 259),
        check_schedule=True,
    ),
    _case(
        "mtp4-window-query-split",
        cache=(16, 1, 1),
        window=1024,
        lengths=(1023, 1024, 1025, 4099),
        check_schedule=True,
    ),
    _case(
        "mtp4-window-capacity",
        parts=256,
        window=1024,
        lengths=(0, 3, 1024, 4099),
        check_schedule=True,
    ),
    _case(
        "mtp4-dense-capacity",
        cache=(128, 0, 1),
        parts=256,
        lengths=(257, 259, 1027, 4099),
        workgroup_budget=17,
        check_schedule=True,
    ),
    _case(
        "mtp4-window-large-grid",
        window=1024,
        lengths=(1027,) * 200,
        check_schedule=True,
    ),
    # Exact one-tile budgets enter the measured (B, H*QS, C/B) grid on gfx950.
    # Ragged/empty owners are not physical x; each row also runs poisoned
    # scratch, changing sinks, minus7/all-empty/restored graph plan refreshes.
    _case(
        "batch-first-window4096",
        parts=18,
        window=4096,
        sink=FP32,
        lengths=(0, 1, 4097, 4353),
        workgroup_budget=72,
        check_schedule=True,
    ),
    _case(
        "batch-first-window8192",
        parts=34,
        window=8192,
        sink=BF16,
        lengths=(0, 1, 3, 4, 257, 8193, 8449, 0),
        workgroup_budget=272,
        check_schedule=True,
    ),
    _case(
        "batch-first-window8192-batch12",
        parts=34,
        window=8192,
        sink=FP32,
        lengths=(0, 1, 8193, 8449) * 3,
        workgroup_budget=408,
        check_schedule=True,
    ),
    # Interior window/odd batch: C125 reaches the QS4 threshold at CU63.
    _case(
        "batch-first-window6000-batch5",
        parts=25,
        window=6000,
        sink=FP32,
        lengths=(0, 1, 6001, 6145, 6257),
        workgroup_budget=125,
        check_schedule=True,
    ),
    # These plans satisfy the tile/capacity guards but miss the window range.
    _case(
        "batch-first-window4095-fallback",
        parts=18,
        window=4095,
        sink=FP32,
        lengths=(0, 1, 4097, 4353),
        workgroup_budget=72,
        check_schedule=True,
    ),
    _case(
        "batch-first-window8193-fallback",
        parts=34,
        window=8193,
        sink=FP32,
        lengths=(0, 1, 8193, 8449),
        workgroup_budget=136,
        check_schedule=True,
    ),
    _case(
        "batch-first-window1024-rejected",
        parts=6,
        window=1024,
        sink=FP32,
        lengths=(0, 1, 1025, 1281) * 2,
        workgroup_budget=48,
        check_schedule=True,
    ),
    _case(
        "batch-first-excess-partitions",
        parts=35,
        window=8192,
        sink=FP32,
        lengths=(0, 1, 8193, 8449),
        workgroup_budget=140,
        check_schedule=True,
    ),
    _case(
        "batch-first-excess-capacity",
        parts=35,
        window=8192,
        sink=BF16,
        lengths=(0, 1, 8193, 8449),
        workgroup_budget=137,
        check_schedule=True,
    ),
    _case(
        "batch-first-tight-capacity",
        parts=34,
        window=8192,
        sink=FP32,
        lengths=(0, 1, 8193, 8449),
        workgroup_budget=135,
        check_schedule=True,
    ),
    _case(
        "batch-first-window4096-wide",
        parts=18,
        window=4096,
        sink=FP32,
        lengths=(0, 1, 4097, 4353),
        workgroup_budget=72,
        wide_kv_addressing=True,
        check_schedule=True,
    ),
    # Plain V needs CU >= QL * C / 2 = 144 to split this B4/C72 plan.
    # Exercise both address widths and a larger partition cap with real inputs.
    _case(
        "batch-first-window4096-plain-v",
        cache=(128, 0, 1),
        parts=18,
        window=4096,
        sink=FP32,
        lengths=(0, 1, 4097, 4353),
        workgroup_budget=72,
        check_schedule=True,
    ),
    _case(
        "batch-first-window4096-plain-v-wide",
        cache=(128, 0, 1),
        parts=18,
        window=4096,
        sink=FP32,
        lengths=(0, 1, 4097, 4353),
        workgroup_budget=72,
        wide_kv_addressing=True,
        check_schedule=True,
    ),
    _case(
        "batch-first-window4096-np64-plain-v-wide",
        cache=(128, 0, 1),
        parts=64,
        window=4096,
        sink=FP32,
        lengths=(0, 1, 4097, 4353),
        workgroup_budget=72,
        wide_kv_addressing=True,
        check_schedule=True,
    ),
    # The unchanged plain-V split threshold is CU272: CU256 uses QS1/flat.
    _case(
        "batch-first-window8192-plain-v-auto",
        cache=(128, 0, 1),
        parts=34,
        window=8192,
        sink=FP32,
        lengths=(0, 1, 8193, 8449),
        workgroup_budget=136,
        check_schedule=True,
    ),
    _case(
        "batch-first-window4096-batch24",
        parts=18,
        window=4096,
        sink=FP32,
        lengths=(0, 1, 3, 4, 257, 1025, 2049, 4096, 4097, 4098, 4353, 4354) * 2,
        workgroup_budget=432,
        check_schedule=True,
    ),
    # NP need not equal the window bound: exact useful capacity still proves
    # one task per tile. Counts span 0/1/4/8/16/17/18 and poisoned padding.
    _case(
        "batch-first-window4096-np64",
        parts=64,
        window=4096,
        sink=FP32,
        lengths=(0, 1, 769, 1793, 4096, 4097, 4353, 4354),
        workgroup_budget=144,
        check_schedule=True,
    ),
    # Large NP keeps the flat grid. The 34-task rows still cross the NP256
    # reducer's 32-partition group boundary, including on graph replays.
    _case(
        "batch-first-window8192-np256-fallback",
        parts=256,
        window=8192,
        sink=FP32,
        lengths=(0, 1, 8449, 8450),
        workgroup_budget=136,
        check_schedule=True,
    ),
    _case(
        "ql1-window-prefetch",
        (1, 1, 16, 128),
        parts=64,
        window=8192,
        lengths=(8193,) * 8,
        workgroup_budget=512,
        check_schedule=True,
    ),
    _case(
        "ql1-window-prefetch-small-capacity",
        (1, 1, 16, 128),
        parts=64,
        window=8192,
        lengths=(8193,) * 8,
        workgroup_budget=263,
        check_schedule=True,
    ),
    _case(
        "ql1-window-prefetch-wide",
        (1, 1, 16, 128),
        parts=64,
        window=8192,
        lengths=(8193,) * 8,
        workgroup_budget=512,
        wide_kv_addressing=True,
        check_schedule=True,
    ),
    # G8 pads half an M-tile. Distinct per-head data/scales also cover planned
    # Hkv2; the Hkv1/NP256 row isolates G8 with a large nominal dense grid.
    _case(
        "ql1-hkv2-g8-window-prefetch",
        (1, 2, 8, 128),
        parts=64,
        window=8192,
        sink=FP32,
        lengths=(0, 1, 8193, 8449),
        workgroup_budget=264,
        check_schedule=True,
    ),
    _case(
        "ql1-hkv1-g8-large-np-prefetch",
        (1, 1, 8, 128),
        parts=256,
        window=8192,
        sink=FP32,
        lengths=(0, 1, 8193, 8449),
        workgroup_budget=132,
        check_schedule=True,
    ),
    # Hkv4 exercises dense M1 prefetch with multi-tile tasks in each KV layout.
    _case(
        "ql1-hkv4-page128-multi",
        (1, 4, 16, 128),
        parts=2,
        window=1024,
        sink=FP32,
        lengths=(0, 1, 1025, 1281),
        workgroup_budget=32,
        check_schedule=True,
    ),
    _case(
        "ql1-hkv4-page16-multi",
        (1, 4, 16, 128),
        (16, 1, 1),
        parts=2,
        window=1024,
        sink=FP32,
        lengths=(0, 1, 1025, 1281),
        workgroup_budget=32,
        check_schedule=True,
    ),
    _case(
        "ql1-hkv4-plain-v-multi",
        (1, 4, 16, 128),
        (128, 0, 1),
        parts=2,
        window=1024,
        sink=FP32,
        lengths=(0, 1, 1025, 1281),
        workgroup_budget=32,
        check_schedule=True,
    ),
    _case(
        "ql1-hkv4-page16-direct-sinks",
        (1, 4, 16, 128),
        (16, 1, 1),
        parts=1,
        sink=FP32,
        lengths=(0, 1, 257, 769),
        workgroup_budget=16,
        check_schedule=True,
    ),
    # Interior GQA sizes pad to one M-tile; G7/G17 exercise both range misses.
    # Fixed groups per fixture keep the handle-only cache checks unambiguous.
    *[
        _case(
            f"ql1-hkv2-g{group}-window",
            (1, 2, group, 128),
            parts=8,
            window=1024,
            sink=FP32,
            lengths=(0, 1, 1025, 1281),
            workgroup_budget=40,
            check_schedule=True,
        )
        for group in (7, *range(9, 16), 17)
    ],
    _case(
        "ql1-hkv2-g9-page16-direct-sinks",
        (1, 2, 9, 128),
        (16, 1, 1),
        parts=1,
        sink=FP32,
        lengths=(0, 1, 257, 769),
        workgroup_budget=8,
        check_schedule=True,
    ),
    _case(
        "ql1-hkv2-g15-plain-v-multi",
        (1, 2, 15, 128),
        (128, 0, 1),
        parts=2,
        window=1024,
        sink=FP32,
        lengths=(0, 1, 1025, 1281),
        workgroup_budget=16,
        check_schedule=True,
    ),
    # Explicit splits also use one padded M-tile for interior GQA sizes.
    _case(
        "split2-hkv2-g9-page16-direct-sinks",
        (2, 2, 9, 128),
        (16, 1, 1),
        parts=1,
        sink=FP32,
        lengths=(0, 1, 257, 769),
        workgroup_budget=8,
        query_splits=2,
    ),
    _case(
        "split4-hkv1-g15-plain-v-window-wide",
        (4, 1, 15, 128),
        (128, 0, 1),
        parts=8,
        window=1024,
        sink=FP32,
        lengths=(0, 1, 1025, 1281),
        workgroup_budget=24,
        query_splits=4,
        wide_kv_addressing=True,
    ),
    _case(
        "hkv2-prefetch-1wg",
        (1, 2, 8, 128),
        parts=8,
        window=1,
        sink=FP32,
        lengths=(257,) * 16,
        check_schedule=True,
    ),
    _case(
        "hkv2-prefetch-2wg",
        (1, 2, 8, 128),
        parts=8,
        lengths=(257,) * 32,
    ),
    _case(
        "hkv2-prefetch-page16",
        (1, 2, 8, 128),
        (16, 1, 1),
        8,
        257,
        FP16,
        lengths=(257,) * 32,
    ),
    _case(
        "hkv2-prefetch-plain-v",
        (1, 2, 16, 128),
        (128, 0, 1),
        8,
        sink=BF16,
        lengths=(257,) * 32,
    ),
    # QL1/W1 has M=1: NP=M and capacity=B*M both meet the exact boundary.
    _case("hkv2-direct-sinks", (1, 2, 8, 128), parts=1, window=1, sink=BF16),
    _case(
        "long-200k",
        cache=(128, 0, 1),
        parts=256,
        sink=FP32,
        lengths=(0, 1, 3, 4, 255, 256, 257, 200003),
    ),
    _case("window-int64", cache=(16, 1, 1), window=2**40, wide_kv_addressing=True),
    # The single active tile is absolute tile 1; padding stays NaN across replay.
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
    # QL3/G8 pads 24 query rows to 32; C<QL also has causally empty rows.
    # A huge planned window stays valid, and NP1 keeps C257 in a two-tile task.
    _case(
        "window-huge-padded-query",
        (3, 2, 8, 128),
        parts=1,
        window=2**31 - 1,
        sink=FP32,
        lengths=(0, 1, 2, 257),
    ),
    _case(
        "long-np64",
        cache=(16, 1, 1),
        parts=64,
        lengths=(0, 3, 257, 16384, 16385, 65537),
    ),
    # Counts [1, 2, 2, 0, 0] fill capacity=5: trailing empty rows start at
    # capacity, so a clamped partition index must also clamp the packed base.
    _case(
        "reduce-np33-tail-empty",
        parts=33,
        window=1,
        sink=FP32,
        lengths=(1, 257, 513, 0, 0),
        workgroup_budget=5,
    ),
    # Counts [1, 1, 2, 2, 2, 2, 0, 0] leave NaN padding in capacity=16.
    # Mask inactive loaded values before arithmetic, including all-empty replay.
    _case(
        "reduce-np64-tail-padding",
        (4, 2, 8, 128),
        (16, 0, 1),
        64,
        255,
        BF16,
        lengths=(1, 3, 257, 258, 514, 515, 0, 0),
        workgroup_budget=32,
    ),
    # Counts [0, 4, 5, 0] become [0, 3, 4, 0] after the minus7 graph replay.
    # Capacity=24=B*M guarantees single-tile tasks and leaves tail NaN padding.
    _case(
        "reduce-shortcount-boundary",
        parts=34,
        window=1024,
        sink=FP32,
        lengths=(0, 769, 1025, 0),
        workgroup_budget=24,
    ),
    # Single-tile counts [0, 8, 9, 0] become [0, 7, 8, 0] after refresh.
    # Exercise both sides of the eight-part reducer branch with poisoned
    # unused slots, then the empty and restored-count graph replays.
    _case(
        "reduce-eightcount-boundary",
        parts=34,
        window=2048,
        sink=FP32,
        lengths=(0, 1793, 2049, 0),
        workgroup_budget=48,
    ),
    _case("head64-mtp-window", (4, 2, 4, 64), (16, 1, 1), 256, 1, FP32),
    _case("scalar-fp16-mtp", cache=(128, 1, 0), parts=3, dtype=FP16),
    _case(
        "fused-narrow",
        cache=(16, 0, 1),
        parts=1,
        sink=FP32,
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


@pytest.mark.parametrize("planned", [False, True], ids=["static", "planned"])
@pytest.mark.parametrize("case", CASES)
def test_pa_decode(case, planned, monkeypatch):
    """Reject static windows; otherwise check real kernels, plans and replays."""
    args, options, reference = _make_inputs(case, planned)
    if not planned and case.sliding_window > 0:
        with pytest.raises(ValueError, match="work_plan"):
            pa_decode(*args, **options)
        with pytest.raises(ValueError, match="work_plan"):
            _run_flydsl(*args, **options)
        return
    output, query, _, _, context = args[:5]
    scratch, sinks, plan = args[14:17], args[-1], options["work_plan"]
    if plan is not None and case.workgroup_budget is not None:
        budget_slots = (
            case.workgroup_budget + case.num_kv_heads - 1
        ) // case.num_kv_heads
        assert plan.capacity == min(
            context.numel() * args[8], max(context.numel(), budget_slots)
        )
    module = importlib.import_module("aiter.ops.flydsl.pa_decode")
    build_tile = module.compile_pa_decode_tile
    compile_reduce = module.compile_pa_decode_ps_reduce
    num_compute_units = torch.cuda.get_device_properties(
        query.device
    ).multi_processor_count
    compile_args = None
    cached_reducer = None

    def compile_tile(**kwargs):
        # Schedule details remain internal; callers receive only launch handles.
        assert (
            not {
                "single_tile_plan",
                "batch_first_plan_grid",
                "buffer_plan_output",
            }
            & kwargs.keys()
        )
        compiled = build_tile(**kwargs)
        assert set(compiled) == {"launch", "kernel"}
        return compiled

    def compile_checked(**kwargs):
        nonlocal compile_args
        assert kwargs["num_seqs"] == context.numel()
        assert kwargs["num_kv_heads"] == case.num_kv_heads
        assert kwargs["num_compute_units"] == num_compute_units
        assert kwargs["work_capacity"] == (plan.capacity if planned else None)
        if case.query_splits is not None:
            kwargs["query_splits"] = case.query_splits
        if case.wide_kv_addressing is not None:
            kwargs["wide_kv_addressing"] = case.wide_kv_addressing
        assert kwargs["use_sinks"] == (
            sinks is not None and args[8] == 1 and not planned
        )
        compile_args = kwargs
        return compile_tile(**kwargs)

    def compile_reduce_checked(**kwargs):
        nonlocal cached_reducer
        assert kwargs["query_group_size"] == case.query_group_size
        assert kwargs["bounded_plan_logits"] == (planned and args[8] <= 64)
        assert kwargs["vectorize_plan_logits"] == (
            planned
            and args[8] <= 64
            and case.head_dim == 128
            and case.dtype in (BF16, FP16)
            and scratch[2].data_ptr() % 4 == 0
            and all(stride % 2 == 0 for stride in scratch[2].stride()[:-1])
        )
        compiled = compile_reduce(**kwargs)
        if cached_reducer is None:
            cached_reducer = compiled
        assert compiled is cached_reducer
        return compiled

    def unexpected_reducer(*_args, **_kwargs):
        raise AssertionError("static NP=1 must not launch a reducer")

    monkeypatch.setattr(module, "compile_pa_decode_tile", compile_checked)
    monkeypatch.setattr(module, "compile_pa_decode_ps_reduce", compile_reduce_checked)
    if not planned and args[8] == 1:
        monkeypatch.setattr(module, "launch_pa_decode_ps_reduce", unexpected_reducer)

    def check(disabled_sink=False):
        _assert_close(output, reference(sinks=None) if disabled_sink else reference())
        visible = (
            context[:, None]
            - case.query_length
            + 1
            + torch.arange(case.query_length, device=context.device)
        )
        assert (output[(visible <= 0).flatten()] == 0).all()
        if sinks is not None:
            assert (output[:, torch.isposinf(sinks)] == 0).all()
        if plan is not None:
            active = _assert_plan(plan, context.cpu().tolist())
            if _expected_single_tile_plan(compile_args):
                records = plan.work_info[:active]
                assert (records[:, 2] - records[:, 1] == 1).all()
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

    if case.check_schedule and get_gfx_runtime() == "gfx950":
        # Compile-only checks stay outside capture and never launch fake-CU results.
        _assert_schedule_cache(compile_tile, compile_args)

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
