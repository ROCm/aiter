# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Generic FlyDSL paged decode correctness test.

Run with: python -m pytest -q op_tests/test_flydsl_pa_decode.py
"""

import importlib.util
from dataclasses import replace
from functools import partial
from itertools import product

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


@pytest.fixture(autouse=True)
def _default_cuda_device(device):
    # Contract cases create device tensors implicitly; restore the caller's default.
    previous = torch.get_default_device()
    torch.set_default_device(device)
    try:
        yield
    finally:
        torch.set_default_device(previous)


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


def _reference(
    query,
    key,
    value,
    table,
    lengths,
    query_length,
    window,
    sinks,
    query_scale=None,
    key_scale=None,
    value_scale=None,
):
    """FP32 attention over logical KV, independent of kernel Q/P rounding."""
    batch = lengths.numel()
    heads, dim = query.shape[1:]
    kv_heads, page_size = key.shape[1:3]
    group = heads // kv_heads
    queries = query.float().reshape(batch, query_length, kv_heads, group, dim)
    if query_scale is not None:
        queries = queries * query_scale
    output = torch.zeros_like(queries)
    positions = torch.arange(query_length, device=query.device)

    for seq, length in enumerate(lengths.cpu().tolist()):
        if length <= 0:
            continue
        tokens = torch.arange(length, device=query.device)
        pages = table[seq, tokens // page_size].long()
        offsets = tokens % page_size
        keys, values = key[pages, :, offsets], value[pages, :, offsets]
        if key_scale is not None:
            if key_scale.numel() == 1:
                keys *= key_scale
                values *= value_scale
            else:
                keys *= key_scale[pages, :, offsets, 0, None]
                values *= value_scale[pages, :, offsets, 0, None]
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
    *,
    sparse=None,
    fp8_query=False,
    masked_scale=False,
):
    if lengths is None:
        lengths = (0, 1, 255, 256, 257, 1027) if pattern == "random" else (1024,)
    batch = len(lengths)
    torch.manual_seed(37 if sparse else 0)
    pages_per_seq = (max(lengths) + page_size - 1) // page_size
    num_pages = batch * pages_per_seq
    counts = None
    table = None
    if sparse is None:
        table = torch.randperm(num_pages, device=device, dtype=torch.int32).reshape(
            batch, pages_per_seq
        )
    else:
        counts = [max(1, (length + page_size - 1) // page_size) for length in lengths]
        num_pages = sum(counts)
    query = torch.empty(
        (batch * query_length, kv_heads * group_size, dim), dtype=dtype, device=device
    ).uniform_(-0.5, 0.5)
    query_scale = None
    if masked_scale:
        query.zero_()
    if fp8_query:
        quant_dtype = (
            torch.float8_e4m3fn
            if get_gfx_runtime() == "gfx950"
            else torch.float8_e4m3fnuz
        )
        query, query_scale = per_tensor_quant(query, quant_dtype=quant_dtype)
        query_scale *= 3.0
        if query_length == 1:
            query_scale = query_scale.reshape(())
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

    context = torch.tensor(lengths, dtype=torch.int32, device=device)
    return query, key, value, table, context, query_scale, counts


def _prepare_inputs(case, device, planned=False):
    pa = importlib.import_module("aiter.ops.flydsl.pa_decode")
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
    heads = kv_heads * group_size
    query, key, value, table, context, query_scale, counts = _make_inputs(
        query_length,
        kv_heads,
        group_size,
        dim,
        page,
        case["query_dtype"],
        case["pattern"],
        device,
        case["lengths"],
        sparse=case["sparse"],
        fp8_query=case["fp8_query"],
        masked_scale=case["masked_scale"],
    )
    num_pages = key.shape[0]
    compute_type = torch.bfloat16
    key_scale = value_scale = None
    quant_dtype = None
    per_token = case["scale_mode"] == "per-token"
    vector = 8
    if case["kv_dtype"] == "fp8":
        compute_type = (
            torch.float8_e4m3fn
            if get_gfx_runtime() == "gfx950"
            else torch.float8_e4m3fnuz
        )
        vector = 16
        quant_dtype = compute_type
        mode = case["scale_mode"]
        if mode in ("per-tensor", "per-token"):
            quantize = pertoken_quant if mode == "per-token" else per_tensor_quant
            key, key_scale = quantize(key, quant_dtype=compute_type)
            value, value_scale = quantize(value, quant_dtype=compute_type)
        else:
            key, value = key.to(compute_type), value.to(compute_type)
            if mode == "python":
                key_scale, value_scale = 0.25, 2.0
    if case["sparse"] is not None:
        if per_token and case["sparse"]:
            token = torch.arange(num_pages * page).reshape(num_pages, 1, page, 1)
            head = torch.arange(kv_heads).reshape(1, kv_heads, 1, 1)
            key_scale *= torch.exp2(((2 * token + head) % 4 - 2).float())
            factors = torch.exp2(((token + 2 * head) % 5 - 3).float())
            # Power-of-two scales isolate W=1 masking from FP8 P rounding.
            value_scale = factors * 2**-10 if window == 1 else value_scale * factors

        start = 0
        for length, count in zip(case["lengths"], counts):
            end = start + count
            if 0 < length <= 4 or (
                case["sink_dtype"] is not None and length in (257, 513)
            ):
                # Positive periodic V exposes repeated sink mass without cancellation;
                # per-tensor cases retain a shared scale and use larger FP8 values.
                token_values = (torch.arange(count * page) % 4 + 1).float()
                token_values *= 0.25 if per_token else 32
                key[start:end].zero_()
                value[start:end] = token_values.reshape(count, 1, page, 1).to(
                    quant_dtype
                )
                if per_token:
                    key_scale[start:end].fill_(1)
                    value_scale[start:end].fill_(1)
            if case["masked_scale"]:
                assert per_token and window == 1
                tokens = torch.arange(count * page)
                visible = (tokens >= max(0, length - query_length)) & (tokens < length)
                key[start:end].zero_()
                key_scale[start:end].fill_(1)
                value_scale[start:end] = torch.where(visible, 1.0, 1e9).reshape(
                    count, 1, page, 1
                )
                values = torch.where(visible, (tokens % 4 + 1).float() * 0.25, 0)
                value[start:end] = values.reshape(count, 1, page, 1).to(quant_dtype)
            start = end

        selected = (
            2 * torch.randperm(num_pages) + 1
            if case["sparse"]
            else torch.arange(num_pages)
        )
        physical_pages = 2 * num_pages + 1 if case["sparse"] else num_pages

        def scatter(tensor):
            if not case["sparse"]:
                return tensor
            result = torch.zeros(
                (physical_pages, *tensor.shape[1:]), dtype=tensor.dtype
            )
            result[selected] = tensor
            return result

        key, value = scatter(key), scatter(value)
        if per_token:
            key_scale, value_scale = scatter(key_scale), scatter(value_scale)
        table = torch.zeros(
            (len(counts), max(counts)), dtype=torch.int32, device=device
        )
        start = 0
        for seq, count in enumerate(counts):
            table[seq, :count] = selected[start : start + count]
            start += count
    logical_key = key.float() * (1.0 if key_scale is None else key_scale)
    logical_value = value.float() * (1.0 if value_scale is None else value_scale)
    reference_key = key.float() if case["branch_checks"] else logical_key
    reference_value = value.float() if case["branch_checks"] else logical_value
    pages = key.shape[0]
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
    if case["branch_checks"] and case["sink_dtype"] is not None:
        logits = [float("-inf"), float("inf"), -1000, 1000, -2, 0, 0.5, 5]
        if case["sink_dtype"] == torch.float32:
            logits += [-3e38, 3e38]
        sinks = (
            torch.tensor(logits, dtype=case["sink_dtype"])
            .repeat((heads + len(logits) - 1) // len(logits))[:heads]
            .contiguous()
        )
        # Distinguish heads even when GQA is a multiple of the sentinel period.
        offset = torch.arange(heads).to(case["sink_dtype"]) * 0.125
        sinks = torch.where(sinks.abs() < 10, sinks + offset, sinks)
    options = {
        "query_scale": query_scale,
        "compute_type": compute_type,
        "key_scale": key_scale,
        "value_scale": value_scale,
        "sinks": sinks,
        "sliding_window": window,
        **(
            {}
            if case["branch_checks"]
            else {"max_context_length": context.max().item()}
        ),
    }
    arguments = (query, key_cache, value_cache, context, table, dim**-0.5, query_length)
    budget = case["workgroup_budget"]
    if case["branch_checks"] and not planned:
        budget = (
            context.numel()
            * kv_heads
            * (
                case["max_partitions"]
                or torch.cuda.get_device_properties(
                    context.device
                ).multi_processor_count
            )
        )
    plan = pa.plan_pa_decode(
        context,
        kv_heads,
        max_partitions=case["max_partitions"],
        workgroup_budget=budget,
        query_length=query_length,
        sliding_window=window,
    )
    return (
        arguments,
        options,
        plan,
        partial(
            _reference,
            query,
            reference_key,
            reference_value,
            table,
            context,
            query_length,
            window,
            sinks=sinks,
            query_scale=query_scale,
            key_scale=key_scale if case["branch_checks"] else None,
            value_scale=value_scale if case["branch_checks"] else None,
        ),
    )


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


def _launch(pa, output, arguments, options, plan, workspace, *, refresh=False):
    if refresh:
        pa.plan_pa_decode(
            arguments[3],
            arguments[1].shape[1],
            query_length=arguments[6],
            sliding_window=options["sliding_window"],
            plan=plan,
        )
    pa.pa_decode(output, *arguments, work_plan=plan, **options, **workspace)


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


def _assert_contracts(pa, output, arguments, options, plan):
    """Reuse numerical inputs for API validation."""
    query, key, value_cache, context, table, scale, query_length = arguments

    def call(**changes):
        kwargs = dict(options, work_plan=plan)
        kwargs.update(changes)
        out = kwargs.pop("output", output)
        ql = kwargs.pop("query_length", query_length)
        pa.pa_decode(out, query, key, value_cache, context, table, scale, ql, **kwargs)

    heads = query.shape[1]
    if options["query_scale"] is not None:
        for invalid, error in [
            (None, TypeError),
            (1.0, TypeError),
            (torch.ones(1, dtype=torch.float16), TypeError),
            (torch.ones(2, dtype=torch.float32), ValueError),
            (torch.ones(1, dtype=torch.float32, device="cpu"), ValueError),
        ]:
            with pytest.raises(error, match="query_scale"):
                call(query_scale=invalid)
        for dtype in (query.dtype, torch.float32):
            with pytest.raises(TypeError, match="output"):
                call(output=torch.empty(output.shape, dtype=dtype))
    else:
        with pytest.raises(NotImplementedError, match="query_scale"):
            call(query_scale=torch.ones(1))
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
            call(sinks=invalid)
    for value, error in [(-2, ValueError), (1.5, TypeError)]:
        with pytest.raises(error, match="sliding_window"):
            call(sliding_window=value)
        with pytest.raises(error, match="sliding_window"):
            pa.plan_pa_decode(context, key.shape[1], sliding_window=value)
    for value, error in [(0, ValueError), (1.5, TypeError)]:
        with pytest.raises(error, match="query_length"):
            call(query_length=value)
        with pytest.raises(error, match="query_length"):
            pa.plan_pa_decode(context, key.shape[1], query_length=value)
    num_compute_units = torch.cuda.get_device_properties(
        context.device
    ).multi_processor_count
    for parts in (0, num_compute_units + 1):
        with pytest.raises(ValueError, match="max_partitions"):
            pa.plan_pa_decode(context, key.shape[1], max_partitions=parts)
        if plan is not None:
            with pytest.raises(ValueError, match="max_partitions"):
                replace(plan, max_partitions=parts).validate(
                    context.numel(), key.shape[1], context.device
                )
    with pytest.raises(TypeError, match="work_plan"):
        pa.pa_decode(output, *arguments, **options)
    if plan is not None:
        heads = key.shape[1]
        reuse = {
            "sliding_window": plan.sliding_window,
            "query_length": plan.query_length,
            "plan": plan,
        }
        metadata = plan.work_info, plan.reduce_info
        assert pa.plan_pa_decode(context, heads, **reuse) is plan
        assert plan.work_info is metadata[0] and plan.reduce_info is metadata[1]
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
                pa.plan_pa_decode(context, heads, **{**reuse, **change})
        for lengths, kv_heads in [(context.repeat(2), heads), (context, heads + 1)]:
            with pytest.raises(ValueError):
                pa.plan_pa_decode(lengths, kv_heads, **reuse)
        with pytest.raises(ValueError, match="sliding_window"):
            call(sliding_window=plan.sliding_window + 1)
        if plan.sliding_window > 0:
            wrong_plan = pa.plan_pa_decode(
                context,
                heads,
                max_partitions=plan.max_partitions,
                sliding_window=plan.sliding_window,
                query_length=plan.query_length + 1,
            )
            with pytest.raises(ValueError, match="query_length"):
                call(work_plan=wrong_plan)


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
        dict(
            query_length=ql,
            kv_heads=heads,
            group_size=group,
            head_dim=dim,
            block_size=page,
            trans_v=bool(trans_v),
            scale_mode="per-token" if per_token else "per-tensor",
            max_partitions=parts,
            window=window,
            sink_dtype=sink,
            query_dtype=kwargs.pop("dtype", torch.bfloat16),
            kv_dtype="fp8",
            sparse=True,
            branch_checks=True,
            lengths=kwargs.pop(
                "lengths",
                (
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
                ),
            ),
            **{key: value for key, value in kwargs.items() if key != "max_partitions"},
        ),
        id=name,
    )


def _cases(columns, rows, *, prefix="", **shared):
    """Expand parameter rows over shared defaults."""
    return [
        _case(
            prefix + name,
            **{**shared, **dict(zip(columns.split(), values, strict=True))},
        )
        for name, *values in rows
    ]


BF16, FP16, FP32 = torch.bfloat16, torch.float16, torch.float32
LENS_1024 = (0, 1, 1025, 1281)
LENS_4096 = (0, 1, 4097, 4353)
LENS_8192 = (0, 1, 8193, 8449)
CASES = [
    *_cases(
        "shape cache parts window sink",
        [
            ("scalar-direct", (1, 2, 8, 128), (16, 1, 0), 1, 0, None),
            ("scalar-window-sinks", (1, 2, 8, 128), (128, 1, 0), 7, 257, FP32),
            ("head1024", (1, 2, 4, 1024), (128, 1, 0), 256, 8192, FP32),
            ("mtp3-window", (3, 1, 16, 128), (128, 0, 1), 7, 257, FP16),
            ("mtp2-odd-parts", (2, 2, 16, 128), (16, 0, 1), 3, 0, None),
            ("window256", (3, 2, 8, 128), (128, 0, 1), 7, 256, None),
            ("window509", (4, 1, 16, 128), (64, 0, 1), 86, 509, FP16),
            ("window8192", (2, 1, 16, 128), (128, 1, 1), 256, 8192, None),
            ("head64-mtp-window", (4, 2, 4, 64), (16, 1, 1), 256, 1, FP32),
            ("hkv2-direct-sinks", (1, 2, 8, 128), (128, 1, 1), 1, 1, BF16),
        ],
    ),
    _case("head64-fp16", (2, 2, 4, 64), (16, 0, 0), 7, 257, BF16, dtype=FP16),
    _case("scalar-fp16-mtp", cache=(128, 1, 0), parts=3, dtype=FP16),
    _case(
        "register-64-65",
        (1, 2, 4, 256),
        (64, 1, 0),
        86,
        sink=FP16,
        dtype=FP16,
        lengths=(0, 1, 257, 16384, 16385),
    ),
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
        34,
        4096,
        BF16,
        lengths=(1, 257, 4353, 0),
        query_splits=4,
    ),
    _case(
        "full-m1-fp16-d256",
        (2, 2, 8, 256),
        (64, 1, 1),
        5,
        257,
        FP16,
        dtype=FP16,
        lengths=(0, 1, 257, 513),
        query_splits=1,
    ),
    _case(
        "split4-hkv1-g15-plain-v-window-wide",
        (4, 1, 15, 128),
        (128, 0, 1),
        8,
        1024,
        FP32,
        lengths=LENS_1024,
        workgroup_budget=24,
        query_splits=4,
        wide_kv_addressing=True,
    ),
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
    _case("mtp2-query-split", (2, 1, 16, 128), parts=3, lengths=(257, 259)),
    *_cases(
        "cache parts window lengths workgroup_budget",
        [
            ("window-query-split", (16, 1, 1), 7, 1024, (1023, 1024, 1025, 4099), None),
            ("window-capacity", (128, 1, 1), 256, 1024, (0, 3, 1024, 4099), None),
            ("dense-capacity", (128, 0, 1), 256, 0, (257, 259, 1027, 4099), 17),
            ("window-large-grid", (128, 1, 1), 7, 1024, (1027,) * 200, None),
        ],
        prefix="mtp4-",
    ),
    *_cases(
        "parts window lengths workgroup_budget",
        [
            ("window4096", 18, 4096, LENS_4096, 72),
            ("window8192-batch12", 34, 8192, LENS_8192 * 3, 408),
            ("window6000-batch5", 25, 6000, (0, 1, 6001, 6145, 6257), 125),
            ("window4095-fallback", 18, 4095, LENS_4096, 72),
            ("window8193-fallback", 34, 8193, LENS_8192, 136),
            ("window1024-rejected", 6, 1024, LENS_1024 * 2, 48),
            ("excess-partitions", 35, 8192, LENS_8192, 140),
            ("tight-capacity", 34, 8192, LENS_8192, 135),
            (
                "window4096-batch24",
                18,
                4096,
                (0, 1, 3, 4, 257, 1025, 2049, 4096, 4097, 4098, 4353, 4354) * 2,
                432,
            ),
            (
                "window4096-np64",
                64,
                4096,
                (0, 1, 769, 1793, 4096, 4097, 4353, 4354),
                144,
            ),
            ("window8192-np256-fallback", 256, 8192, (0, 1, 8449, 8450), 136),
        ],
        prefix="batch-first-",
        sink=FP32,
    ),
    *_cases(
        "parts lengths workgroup_budget",
        [
            ("window8192", 34, (0, 1, 3, 4, 257, 8193, 8449, 0), 272),
            ("excess-capacity", 35, LENS_8192, 137),
        ],
        prefix="batch-first-",
        window=8192,
        sink=BF16,
    ),
    *_cases(
        "cache parts wide_kv_addressing",
        [
            ("window4096-wide", (128, 1, 1), 18, True),
            ("window4096-plain-v", (128, 0, 1), 18, None),
            ("window4096-plain-v-wide", (128, 0, 1), 18, True),
            ("window4096-np64-plain-v-wide", (128, 0, 1), 64, True),
        ],
        prefix="batch-first-",
        window=4096,
        sink=FP32,
        lengths=LENS_4096,
        workgroup_budget=72,
    ),
    _case(
        "batch-first-window8192-plain-v-auto",
        cache=(128, 0, 1),
        parts=34,
        window=8192,
        sink=FP32,
        lengths=LENS_8192,
        workgroup_budget=136,
    ),
    *_cases(
        "workgroup_budget wide_kv_addressing",
        [
            ("prefetch", 512, None),
            ("prefetch-small-capacity", 263, None),
            ("prefetch-wide", 512, True),
        ],
        prefix="ql1-window-",
        shape=(1, 1, 16, 128),
        parts=64,
        window=8192,
        lengths=(8193,) * 8,
    ),
    *_cases(
        "shape parts workgroup_budget",
        [
            ("hkv2-g8-window-prefetch", (1, 2, 8, 128), 64, 264),
            ("hkv1-g8-large-np-prefetch", (1, 1, 8, 128), 256, 132),
        ],
        prefix="ql1-",
        window=8192,
        sink=FP32,
        lengths=LENS_8192,
    ),
    *_cases(
        "shape cache workgroup_budget",
        [
            ("hkv4-page128-multi", (1, 4, 16, 128), (128, 1, 1), 32),
            ("hkv4-page16-multi", (1, 4, 16, 128), (16, 1, 1), 32),
            ("hkv4-plain-v-multi", (1, 4, 16, 128), (128, 0, 1), 32),
            ("hkv2-g15-plain-v-multi", (1, 2, 15, 128), (128, 0, 1), 16),
        ],
        prefix="ql1-",
        parts=2,
        window=1024,
        sink=FP32,
        lengths=LENS_1024,
    ),
    *_cases(
        "shape workgroup_budget query_splits",
        [
            ("ql1-hkv4-page16-direct-sinks", (1, 4, 16, 128), 16, None),
            ("ql1-hkv2-g9-page16-direct-sinks", (1, 2, 9, 128), 8, None),
            ("split2-hkv2-g9-page16-direct-sinks", (2, 2, 9, 128), 8, 2),
        ],
        cache=(16, 1, 1),
        parts=1,
        sink=FP32,
        lengths=(0, 1, 257, 769),
    ),
    *_cases(
        "shape",
        [
            (f"ql1-hkv2-g{group}-window", (1, 2, group, 128))
            for group in (7, *range(9, 16), 17)
        ],
        parts=8,
        window=1024,
        sink=FP32,
        lengths=LENS_1024,
        workgroup_budget=40,
    ),
    *_cases(
        "shape cache window sink lengths",
        [
            ("1wg", (1, 2, 8, 128), (128, 1, 1), 1, FP32, (257,) * 16),
            ("2wg", (1, 2, 8, 128), (128, 1, 1), 0, None, (257,) * 32),
            ("page16", (1, 2, 8, 128), (16, 1, 1), 257, FP16, (257,) * 32),
            ("plain-v", (1, 2, 16, 128), (128, 0, 1), 0, BF16, (257,) * 32),
        ],
        prefix="hkv2-prefetch-",
        parts=8,
    ),
    _case(
        "long-200k",
        cache=(128, 0, 1),
        parts=256,
        sink=FP32,
        lengths=(0, 1, 3, 4, 255, 256, 257, 200003),
    ),
    _case(
        "long-np64",
        cache=(16, 1, 1),
        parts=64,
        lengths=(0, 3, 257, 16384, 16385, 65537),
    ),
    _case("large-batch-auto", cache=(128, 0, 1), parts=None, lengths=(257,) * 200),
    _case("small-batch-auto", (1, 1, 8, 128), (16, 0, 0), None, lengths=(257,) * 3),
    _case("default-cu-window", parts=None, window=1024, sink=FP32, lengths=LENS_1024),
    # CU > 256 gives the long row 257 reducer tasks.
    _case(
        "default-cu-head256",
        (2, 1, 8, 256),
        (64, 0, 1),
        parts=None,
        sink=FP16,
        dtype=FP16,
        lengths=(0, 1, 257, 65537),
    ),
    _case(
        "exact-parts-override",
        (1, 1, 16, 128),
        parts=5,
        lengths=(200000,),
        max_partitions=4,
    ),
    _case("window-int64", cache=(16, 1, 1), window=2**40, wide_kv_addressing=True),
    *_cases(
        "window sink",
        [("max", 2**31 - 1, None), ("overflow", 2**31, FP32)],
        prefix="window-int32-",
        cache=(16, 0, 1),
        parts=1,
    ),
    _case(
        "window-huge-padded-query",
        (3, 2, 8, 128),
        parts=1,
        window=2**31 - 1,
        sink=FP32,
        lengths=(0, 1, 2, 257),
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
    # Inactive scales and NaN scratch must not contaminate live rows.
    *_cases(
        "shape sink",
        [("decode", (1, 1, 16, 128), FP32), ("mtp", (4, 1, 16, 128), None)],
        prefix="masked-scale-",
        window=1,
        lengths=(511,),
        masked_scale=True,
    ),
    *_cases(
        "parts window lengths workgroup_budget",
        [
            ("np33-tail-empty", 33, 1, (1, 257, 513, 0, 0), 5),
            ("shortcount-boundary", 34, 1024, (0, 769, 1025, 0), 24),
            ("eightcount-boundary", 34, 2048, (0, 1793, 2049, 0), 48),
        ],
        prefix="reduce-",
        sink=FP32,
    ),
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
    _case("empty", window=1, sink=FP16, lengths=(0,) * 4),
]


CASES += [
    _case(
        f"fp8-q-p{page}-q{ql}-token{int(per_token)}",
        (ql, 1, 16, 128),
        (page, int(ql == 1), int(per_token)),
        parts=1 if ql == 1 else 3,
        lengths=(0, 1, 3, 259, 1027),
        dtype=BF16 if ql == 1 else FP16,
        fp8_query=True,
    )
    for page, ql, per_token in product((16, 64, 128), (1, 4), (False, True))
]
CASES += [
    _case(
        "fp8-q-sink",
        (2, 2, 12, 128),
        (16, 1, 1),
        parts=1,
        sink=FP32,
        fp8_query=True,
        lengths=(0, 1, 259, 1027),
    ),
    _case(
        "fp8-q-window",
        (4, 1, 16, 128),
        (128, 1, 1),
        parts=7,
        window=257,
        sink=BF16,
        fp8_query=True,
        lengths=(0, 3, 259, 1027),
    ),
    _case(
        "fp8-q-window-scalar",
        (2, 1, 8, 128),
        (64, 0, 0),
        parts=3,
        window=1,
        sink=FP16,
        fp8_query=True,
        lengths=(0, 1, 259, 1027),
    ),
    _case(
        "fp8-q-d64",
        (2, 1, 8, 64),
        (64, 1, 1),
        parts=3,
        fp8_query=True,
        lengths=(0, 259, 1027),
    ),
    _case(
        "fp8-q-d256",
        (4, 1, 16, 256),
        (128, 0, 0),
        parts=3,
        fp8_query=True,
        lengths=(0, 259, 1027),
    ),
]


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
        *[
            pytest.param(
                dict(param.values[0], planned=planned), id=f"{param.id}-{label}"
            )
            for param in CASES
            # With one partition both budget policies give capacity == batch, so the
            # planned variant would repeat the full-budget plan.
            for planned, label in (
                ((False, "full-budget"),)
                if param.values[0]["max_partitions"] == 1
                else ((False, "full-budget"), (True, "planned-budget"))
            )
        ],
        *[pytest.param(geometry, id=f"bf16-{name}") for name, geometry in _BF16_CASES],
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
            "sparse": None,
            "branch_checks": False,
            "fp8_query": False,
            "masked_scale": False,
            "workgroup_budget": None,
            "query_splits": None,
            "wide_kv_addressing": None,
            "planned": False,
        },
        **case,
    )
    pa = importlib.import_module("aiter.ops.flydsl.pa_decode")
    planned = case["planned"]
    if case["branch_checks"] and case["max_partitions"] is not None:
        context = torch.tensor(case["lengths"], dtype=torch.int32, device=device)
        units = torch.cuda.get_device_properties(context.device).multi_processor_count
        if case["max_partitions"] > units:
            with pytest.raises(ValueError, match="max_partitions"):
                pa.plan_pa_decode(
                    context, case["kv_heads"], max_partitions=case["max_partitions"]
                )
            case["max_partitions"] = units
    arguments, options, plan, reference = _prepare_inputs(case, device, planned)
    query, _key_cache, _value_cache, context, _table, _, query_length = arguments
    kv_heads, group_size, dim = case["kv_heads"], case["group_size"], case["head_dim"]
    sinks = options["sinks"]
    if "invalid_input" in case:
        _check_unsupported_input(pa, arguments, options, plan, case, monkeypatch)
        return

    output = torch.full(
        query.shape, float("nan"), dtype=case["query_dtype"], device=device
    )
    workspace = {}
    if case["capture"] or case["branch_checks"]:
        scalar_shape = (kv_heads, plan.capacity, query_length * group_size)
        workspace = {
            "exp_sums": torch.empty(scalar_shape, dtype=torch.float32, device=device),
            "max_logits": torch.empty(scalar_shape, dtype=torch.float32, device=device),
            "temporary_output": torch.empty(
                (*scalar_shape, dim), dtype=output.dtype, device=device
            ),
        }

    if case["branch_checks"]:
        for tensor in workspace.values():
            tensor.fill_(float("nan"))
        with pytest.raises(TypeError, match="work_plan"):
            pa.pa_decode(output, *arguments, **options, **workspace)
        units = torch.cuda.get_device_properties(context.device).multi_processor_count
        expected_parts = (
            units if case["max_partitions"] is None else case["max_partitions"]
        )
        assert plan.max_partitions == expected_parts
        budget = (
            (
                case["workgroup_budget"]
                if case["workgroup_budget"] is not None
                else 2 * units
            )
            if planned
            else context.numel() * kv_heads * expected_parts
        )
        slots = (budget + kv_heads - 1) // kv_heads
        assert plan.capacity == min(
            context.numel() * expected_parts, max(context.numel(), slots)
        )
    overrides = {
        name: case[name]
        for name in ("query_splits", "wide_kv_addressing")
        if case[name] is not None
    }
    if overrides:
        build_tile = pa.compile_pa_decode_tile

        def compile_tile(**kwargs):
            return build_tile(**{**kwargs, **overrides})

        monkeypatch.setattr(pa, "compile_pa_decode_tile", compile_tile)

    def launch():
        _launch(
            pa,
            output,
            arguments,
            options,
            plan,
            workspace,
            refresh=case["branch_checks"],
        )

    def check(disabled_sink=False):
        _check_output(output, reference(sinks=None) if disabled_sink else reference())
        if case["branch_checks"]:
            visible = (
                context[:, None]
                - query_length
                + 1
                + torch.arange(query_length, device=context.device)
            )
            assert (output[(visible <= 0).flatten()] == 0).all()
            if sinks is not None:
                assert (output[:, torch.isposinf(sinks)] == 0).all()
            active = _assert_plan(plan, context.cpu().tolist())
            for tensor in workspace.values():
                assert torch.isnan(tensor[:, active:]).all()

    launch()
    check()
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

    if case["branch_checks"]:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            launch()
        original_sinks = sinks.clone() if sinks is not None else None
        for step, lengths in enumerate(
            (
                tuple(max(0, length - 7) for length in case["lengths"]),
                (0,) * len(case["lengths"]),
                case["lengths"],
            )
        ):
            context.copy_(torch.tensor(lengths, dtype=torch.int32, device=device))
            if case["fp8_query"]:
                options["query_scale"].mul_(0.5)
            if sinks is not None:
                if step == 2:
                    sinks.fill_(float("-inf"))
                else:
                    sinks.copy_(original_sinks.roll(step + 1))
            for tensor in (output, *workspace.values()):
                tensor.fill_(float("nan"))
            graph.replay()
            check(disabled_sink=step == 2)
        _assert_contracts(pa, output, arguments, options, plan)
        lengths = (-1, 0, 1, 257, 2**31 - 1)
        extreme_plan = pa.plan_pa_decode(
            torch.tensor(lengths, dtype=torch.int32, device=device),
            kv_heads,
            max_partitions=plan.max_partitions,
            sliding_window=case["window"],
            query_length=query_length,
        )
        _assert_plan(extreme_plan, lengths)
