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

from aiter import dtypes, per_tensor_quant, pertoken_quant
from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.ops.quant import per_1x32_f4_quant
from aiter.test_common import checkAllclose
from aiter.utility import fp4_utils


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
    "batch,max_parts,budget,window,query_length,max_batch_size,bucket",
    [
        (1, 1, 1, 0, 1, None, 1),
        (1, 7, 160, 257, 4, None, 1),
        (1, None, 512, 0, 1, None, 1),
        (6, 7, 160, 257, 4, None, 8),
        (63, 7, 160, 0, 1, None, 64),
        (65, 7, 512, 1, 4, None, 128),
        (257, None, 4096, 8193, 4, None, 512),
        (4096, None, 8192, 0, 1, None, 4096),
        (1, 1, 1, 0, 1, 1, 1),
        (65, None, None, 0, 1, 100, 128),
        (100, None, None, 0, 1, 100, 128),
        (4097, None, None, 0, 1, 5000, 8192),
        pytest.param(None, None, None, 0, 1, None, None, id="lds-limit"),
    ],
)
@pytest.mark.parametrize("tile_tokens", [128, 256])
def test_pa_decode_plan(
    device,
    monkeypatch,
    batch,
    max_parts,
    budget,
    window,
    query_length,
    max_batch_size,
    bucket,
    tile_tokens,
):
    """Packed tasks cover visible tiles, refresh in graphs and reuse batch buckets."""
    from aiter.ops.flydsl.kernels import pa_decode_plan as planner

    properties = torch.cuda.get_device_properties(device)
    if tile_tokens == 128 and get_gfx_runtime() != "gfx1250":
        pytest.skip("128-token plans require gfx1250")
    lds_limit = batch is None
    if lds_limit:
        lds_bytes = getattr(properties, "shared_memory_per_block", None)
        if lds_bytes is None:
            lds_bytes = planner.get_lds_capacity_bytes(properties.gcnArchName)
        batch = bucket = (lds_bytes - planner._PLAN_SCAN_LDS_BYTES) // 16 * 4
        max_batch_size = batch + 1

    compile_plan = planner.compile_pa_decode_plan
    calls = []

    def spy_compile(*args):
        launch = compile_plan(*args)
        calls.append((args, launch))
        return launch

    monkeypatch.setattr(planner, "compile_pa_decode_plan", spy_compile)

    def check(plan, lengths):
        window, query_length = plan.sliding_window, plan.query_length
        info = plan.reduce_info.cpu().tolist()
        work = plan.work_info.cpu().tolist()
        ranges = [
            (
                (
                    max(max(ctx, 0) - query_length + 1 - window, 0) // plan.tile_tokens
                    if window
                    else 0
                ),
                (max(ctx, 0) + plan.tile_tokens - 1) // plan.tile_tokens,
            )
            for ctx in lengths
        ]
        total_tiles = max(sum(last - first for first, last in ranges), 1)
        remaining = plan.capacity - sum(last > first for first, last in ranges)
        cursor = prefix_tiles = 0
        for seq, (ctx, (start, count), (first, last)) in enumerate(
            zip(lengths, info, ranges)
        ):
            assert start == cursor
            tiles = last - first
            # Verify that the runtime capacity supplies the task budget used
            # to distribute slots by cumulative visible tile count.
            extra = (prefix_tiles + tiles) * remaining // total_tiles
            extra -= prefix_tiles * remaining // total_tiles
            assert count == min(int(tiles > 0) + extra, tiles, plan.max_partitions)
            tile = first
            for record in work[start : start + count]:
                row, begin, end, original_length = record
                assert row == seq and original_length == ctx
                assert begin == tile and begin < end <= last
                tile = end
            assert tile == last
            cursor += count
            prefix_tiles += tiles
        assert cursor <= plan.capacity
        assert all(record == [0, 0, 0, 0] for record in work[cursor:])
        if plan.cluster_info is not None:
            records = plan.cluster_info.cpu().tolist()
            for seq, ((start, count), row_records) in enumerate(zip(info, records)):
                for part, record in enumerate(row_records):
                    assert record[0] == seq
                    assert record[3:] == [lengths[seq], start, count, 0, 0]
                    if part < count:
                        assert record[:4] == work[start + part]
                    else:
                        assert record[1:3] == [0, 0]

    pattern = (-1, 0, 1, 255, 256, 257, 65537, 2**31 - 1)
    lengths = [pattern[i % len(pattern)] for i in range(batch)]
    context = torch.tensor(lengths, dtype=torch.int32, device=device)
    plan = planner.plan_pa_decode(
        context,
        2,
        tile_tokens=tile_tokens,
        max_partitions=max_parts,
        max_batch_size=max_batch_size,
        workgroup_budget=budget,
        sliding_window=window,
        query_length=query_length,
    )
    check(plan, lengths)
    config, launch = calls[-1]
    assert config[0] == bucket
    if max_batch_size is not None:
        assert plan.max_batch_size == max_batch_size

    def plan_pointers():
        return (
            plan.work_info.data_ptr(),
            plan.reduce_info.data_ptr(),
            plan.cluster_info.data_ptr() if plan.cluster_info is not None else None,
        )

    pointers = plan_pointers()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        refreshed = planner.plan_pa_decode(
            context, 2, sliding_window=window, query_length=query_length, plan=plan
        )
    assert refreshed is plan
    # All-empty replay must erase old active records, then mixed lengths must
    # rebuild them using the same buffers without stale padding or readback.
    replay_lengths = [[0] * batch, list(reversed(lengths))]
    if batch == 1:
        replay_lengths.extend([[length] for length in pattern])
    for values in replay_lengths:
        context.copy_(torch.tensor(values, dtype=torch.int32, device=device))
        graph.replay()
        check(plan, values)
        assert plan_pointers() == pointers

    with pytest.raises(ValueError, match="tile_tokens must match"):
        planner.plan_pa_decode(
            context,
            2,
            plan=plan,
            sliding_window=window,
            query_length=query_length,
            tile_tokens=384 - tile_tokens,
        )

    if bucket > 1:
        # Change every runtime value inside the same batch/block-size bucket
        # and window mode.
        if batch < min(bucket, plan.max_batch_size):
            other_batch = batch + 1
        else:
            other_batch = batch - 1
        # Move the capacity by one slot within the block size.
        threads = planner._block_threads(plan.capacity)
        other_capacity = plan.capacity - 1
        if (
            other_capacity < other_batch
            or planner._block_threads(other_capacity) != threads
        ):
            other_capacity = plan.capacity + 1
        # The smallest partition cap that admits it, so long rows reach the cap.
        other_parts = max(2, -(-other_capacity // other_batch))
        assert other_parts != plan.max_partitions
        other_window = window + 256 if window else 0
        values = [pattern[i % len(pattern)] for i in range(other_batch)]
        other_context = torch.tensor(values, dtype=torch.int32, device=device)
        other_plan = planner.plan_pa_decode(
            other_context,
            2,
            max_partitions=other_parts,
            tile_tokens=tile_tokens,
            max_batch_size=plan.max_batch_size,
            workgroup_budget=2 * other_capacity,
            sliding_window=other_window,
            query_length=query_length + 1,
        )
        assert other_plan.capacity == other_capacity
        check(other_plan, values)
        other_config, other_launch = calls[-1]
        assert other_config[:3] == config[:3]
        if other_config == config:
            assert other_launch is launch

    if max_batch_size is not None:
        invalid_batch = batch + 1 if lds_limit else max_batch_size + 1
        message = "exceeds planner LDS capacity" if lds_limit else "supports batches"
        with pytest.raises(ValueError, match=message):
            planner.plan_pa_decode(
                torch.zeros(invalid_batch, dtype=torch.int32, device=device),
                2,
                max_batch_size=max_batch_size,
            )


@pytest.mark.parametrize(
    "head_size,dtype,with_sinks",
    [
        (128, torch.bfloat16, False),
        (128, torch.bfloat16, True),
        (64, torch.float16, False),
        (192, torch.bfloat16, True),
        (256, torch.float16, False),
        (512, torch.bfloat16, False),
    ],
)
@pytest.mark.parametrize("output_offset", [0, 1])
def test_pa_decode_reduce_split_groups(
    device, head_size, dtype, with_sinks, output_offset
):
    """Rungs split across partition groups match the single-group ladder bit for bit.

    Register-only CTAs (24 and 32 partition bounds) and the register branch of
    dual-mode CTAs reduce the same packed partials with one group and with
    split groups; sequences above 32 partitions take the LDS branch and match
    a float64 reference. Slots past the live tasks hold NaN.
    """
    if get_gfx_runtime() != "gfx1250":
        pytest.skip("gfx1250 splits register rungs across partition groups")
    import flydsl.compiler as flyc
    import flydsl.expr as fx

    from aiter.ops.flydsl.kernels.pa_decode_reduce import compile_pa_decode_ps_reduce
    from aiter.ops.flydsl.kernels.tensor_shim import (
        _run_compiled,
        get_dtype_str,
        ptr_arg,
    )

    kv_heads, rows = 2, 16
    counts = (17, 0, 32, 1, 24, 16, 31, 20, 40, 256, 100)
    first = [sum(counts[:i]) for i in range(len(counts))]
    capacity = sum(counts) + 8
    torch.manual_seed(0)
    exp_sums = torch.full((kv_heads, capacity, rows), float("nan"), device=device)
    max_logits = torch.full_like(exp_sums, float("nan"))
    logits = torch.full(
        (kv_heads, capacity, rows, head_size), float("nan"), dtype=dtype, device=device
    )
    live = sum(counts)
    exp_sums[:, :live] = torch.rand(kv_heads, live, rows, device=device) * 50 + 0.5
    max_logits[:, :live] = torch.randn(kv_heads, live, rows, device=device) * 4
    logits[:, :live] = torch.randn(kv_heads, live, rows, head_size, device=device).to(
        dtype
    )
    reduce_info = torch.tensor(
        [[f, c] for f, c in zip(first, counts)], dtype=torch.int32, device=device
    )
    sinks = None
    if with_sinks:
        sinks = torch.linspace(-3, 3, kv_heads * rows, device=device).to(dtype)
        sinks[0] = float("-inf")

    def pairs_for(partitions):
        return head_size % 128 == 0 if partitions > 32 else head_size in (128, 256)

    def run(partitions, groups, store_pairs=False):
        pairs = pairs_for(partitions)
        compiled = compile_pa_decode_ps_reduce(
            architecture="gfx1250",
            max_context_partition_num=partitions,
            head_size=head_size,
            output_dtype_str=get_dtype_str(dtype),
            logits_dtype_str=get_dtype_str(dtype),
            sink_dtype_str=get_dtype_str(dtype),
            use_sinks=with_sinks,
            query_group_size=rows,
            pairs=pairs,
            parallel_groups=groups,
            paired_output=store_pairs and pairs and output_offset == 0,
        )
        storage = torch.full(
            (len(counts) * kv_heads * rows * head_size + output_offset,),
            float("nan"),
            dtype=dtype,
            device=device,
        )
        out = storage[output_offset:].view(len(counts), 1, kv_heads, rows, head_size)
        pointer = fx.BFloat16 if dtype == torch.bfloat16 else fx.Float16
        _run_compiled(
            compiled["launch"],
            ptr_arg(out, pointer),
            ptr_arg(exp_sums, fx.Float32),
            ptr_arg(max_logits, fx.Float32),
            ptr_arg(logits, pointer),
            (
                flyc.from_c_void_p(pointer, 0)
                if sinks is None
                else ptr_arg(sinks, pointer)
            ),
            *out.stride()[:4],
            *exp_sums.stride()[:2],
            *logits.stride()[:3],
            1,
            len(counts),
            kv_heads,
            ptr_arg(reduce_info, fx.Int32),
            -(-(1 << 30) // rows),
            -(-(1 << 30) // (rows * kv_heads)),
            torch.cuda.current_stream(),
        )
        return out

    reference = torch.zeros(len(counts), kv_heads, rows, head_size, device=device)
    for seq, (f, c) in enumerate(zip(first, counts)):
        if c == 0:
            continue
        maxima = max_logits[:, f : f + c].double()
        top = maxima.max(dim=1).values
        if sinks is not None:
            top = torch.maximum(top, sinks.double().view(kv_heads, rows))
        mass = exp_sums[:, f : f + c].double() * torch.exp(maxima - top[:, None])
        total = mass.sum(dim=1)
        if sinks is not None:
            total = total + torch.exp(sinks.double().view(kv_heads, rows) - top)
        weights = mass / total[:, None]
        reference[seq] = (
            (weights[..., None] * logits[:, f : f + c].double()).sum(1).float()
        )
    for partitions, split in ((32, 4), (24, 4), (32, 2), (64, 16), (256, 4), (256, 16)):
        if split * head_size // (2 if pairs_for(partitions) else 1) // 32 > 32:
            continue
        single, grouped = run(partitions, 1), run(partitions, split, store_pairs=True)
        register = [seq for seq, c in enumerate(counts) if c <= min(partitions, 32)]
        assert torch.equal(
            single[register].view(torch.int16), grouped[register].view(torch.int16)
        )
        # Plans keep every count within the partition bound.
        bounded = [seq for seq, c in enumerate(counts) if c <= partitions]
        err = checkAllclose(
            grouped[bounded, 0].float(),
            reference[bounded],
            atol=2e-2,
            rtol=2e-2,
            msg=f"split_groups partitions={partitions} split={split}",
        )
        assert err == 0, (
            f"split_groups partitions={partitions} split={split}: "
            f"mismatch ratio={err}"
        )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "mode,dim,scratch_offset",
    [
        ("last", 128, 0),
        ("last", 128, 1),
        ("last", 128, 2),
        ("last", 128, 3),
        ("cluster", 128, 1),
        ("cluster", 384, 1),
        ("reducer", 128, 1),
    ],
)
def test_pa_decode_merge_alignment(
    device, monkeypatch, mode, dim, scratch_offset, dtype
):
    """Offset scratch and odd query/output strides retain element alignment.

    Last-arriver atomics require eight-byte scratch alignment; unaligned
    scratch falls back to the cluster merge. Cluster loads and final stores
    support the otherwise valid offset views directly, as does the reducer,
    which a -inf sink selects without adding attention mass.
    """
    if get_gfx_runtime() != "gfx1250":
        pytest.skip("gfx1250 tile-kernel partition merges")
    pa = importlib.import_module("aiter.ops.flydsl.pa_decode")
    lengths = (1024,) if mode == "cluster" else (1024, 0, 0, 0)
    page, group = 16, 16
    query, key, value, table, context = _make_inputs(
        1, 1, group, dim, page, dtype, "random", device, lengths
    )
    # Both the base address and each head's stride may be only two-byte aligned.
    query_storage = torch.empty(
        query.numel() + query.shape[0] * group + 1, dtype=dtype, device=device
    )
    query_view = query_storage[1:].as_strided(
        query.shape, (group * (dim + 1), dim + 1, 1)
    )
    query_view.copy_(query)
    query = query_view
    output_storage = torch.full_like(query_storage, float("nan"))
    output = output_storage[1:].as_strided(query.shape, query.stride())
    key_cache = (
        key.to(torch.bfloat16)
        .reshape(-1, 1, page, dim // 8, 8)
        .permute(0, 1, 3, 2, 4)
        .contiguous()
    )
    value_cache = (
        value.to(torch.bfloat16)
        .reshape(-1, 1, page // 8, 8, dim)
        .permute(0, 1, 2, 4, 3)
        .contiguous()
    )
    plan = pa.plan_pa_decode(
        context, 1, max_partitions=4, workgroup_budget=None if mode == "cluster" else 1
    )
    sinks = (
        torch.full((group,), float("-inf"), dtype=dtype, device=device)
        if mode == "reducer"
        else None
    )
    scalar_shape = (1, plan.capacity, group)
    partial_shape = (*scalar_shape, dim)
    partials = torch.empty(
        plan.capacity * group * dim + scratch_offset, dtype=dtype, device=device
    )[scratch_offset:].reshape(partial_shape)
    workspace = {
        "exp_sums": torch.empty(scalar_shape, device=device),
        "max_logits": torch.empty(scalar_shape, device=device),
        "temporary_output": partials,
    }
    compile_kwargs = []
    compile_tile = pa.compile_pa_decode_tile

    def spy_tile(**kwargs):
        compile_kwargs.append(kwargs)
        return compile_tile(**kwargs)

    monkeypatch.setattr(pa, "compile_pa_decode_tile", spy_tile)
    reference = _reference(
        query,
        key.to(torch.bfloat16).float(),
        value.to(torch.bfloat16).float(),
        table,
        context,
        1,
        0,
        sinks,
    )
    for _ in range(3):
        output.fill_(float("nan"))
        for tensor in workspace.values():
            tensor.fill_(float("nan"))
        pa.pa_decode(
            output,
            query,
            key_cache,
            value_cache,
            context,
            table,
            dim**-0.5,
            1,
            sinks=sinks,
            work_plan=plan,
            **workspace,
        )
        _check_output(output, reference)
        assert torch.count_nonzero(plan.arrival_counters) == 0
    last_arriver = mode == "last" and scratch_offset == 0
    assert compile_kwargs[-1]["direct_output"] == last_arriver
    assert compile_kwargs[-1]["cluster_merge"] == (
        0 if last_arriver or mode == "reducer" else 4
    )


@pytest.mark.parametrize("architecture", ["gfx942", "gfx950"])
@pytest.mark.parametrize("invalid_input", [False, True])
def test_pa_decode_fp16_bf16_case_policy(
    device, monkeypatch, architecture, invalid_input
):
    """Skip positive FP16-query cases while retaining input rejection checks."""
    monkeypatch.setattr(
        importlib.import_module(__name__), "get_gfx_runtime", lambda: architecture
    )
    monkeypatch.setattr(
        importlib.import_module("aiter.ops.flydsl.pa_decode"),
        "get_gfx_runtime",
        lambda: architecture,
    )
    case = {
        "kv_dtype": "bf16",
        "query_dtype": torch.float16,
        "query_length": 4,
        "group_size": 16,
        "block_size": 128,
        "max_partitions": 3,
    }
    if invalid_input:
        case.update(
            invalid_input="fp16-query",
            error=NotImplementedError,
            match="BF16 KV requires bfloat16 queries",
        )
        test_pa_decode(case, device, monkeypatch)
    else:
        with pytest.raises(
            pytest.skip.Exception, match="BF16 KV requires bfloat16 queries"
        ):
            test_pa_decode(case, device, monkeypatch)


def _reference(
    query, key, value, table, lengths, query_length, window, sinks, dtype=torch.float32
):
    """Attention over logical KV in ``dtype``, independent of kernel Q/P rounding."""
    batch = lengths.numel()
    heads, dim = query.shape[1:]
    kv_heads, page_size = key.shape[1:3]
    group = heads // kv_heads
    queries = query.to(dtype).reshape(batch, query_length, kv_heads, group, dim)
    output = torch.zeros_like(queries)
    positions = torch.arange(query_length, device=query.device)

    for seq, length in enumerate(lengths.cpu().tolist()):
        if length == 0:
            continue
        tokens = torch.arange(length, device=query.device)
        pages = table[seq, tokens // page_size].long()
        offsets = tokens % page_size
        keys = key[pages, :, offsets].to(dtype)
        values = value[pages, :, offsets].to(dtype)
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
    return output.reshape_as(query).float()


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
    elif pattern in ("peaked", "peaked-extreme"):
        # One token per sequence leads every other by ``gap`` in logit (softmax
        # weights e**-gap): at 12 the 128k-1 others still carry most of the mass, at 90
        # their weights sit below 2**-113, the floor of the fixed P shift.
        gap = 12.0 if pattern == "peaked" else 90.0
        query.fill_(1.0)
        key.zero_()
        for seq, length in enumerate(lengths):
            peak = length // 3
            key[table[seq, peak // page_size].long(), :, peak % page_size] = (
                gap / dim**0.5
            )
    elif pattern == "varied":
        # Uniform draws quantize to nearly one per-token scale, so a shifted
        # scale still matches the oracle. Token magnitudes spanning about a
        # decade make a wrong scale index fail it.
        key.mul_(torch.randn_like(key[..., :1]).mul_(0.4).exp_())
        value.mul_(torch.randn_like(value[..., :1]).mul_(0.4).exp_())
    elif pattern == "block_scales":
        # MXFP4 scales are per 32 elements: unequal blocks expose a block/scale mix-up.
        blocks = torch.tensor([1.0, 0.5, 0.25, 0.125], device=device)
        key.mul_(blocks.repeat_interleave(dim // 4))
        value.mul_(blocks.repeat_interleave(dim // 4))

    context = torch.tensor(lengths, dtype=torch.int32, device=device)
    return query, key, value, table, context


def _mxfp4_dequant(pool):
    """f32 [pages, heads, 128, 128] of a pool of MXFP4 pages (see ``_mxfp4_pages``)."""
    pages, heads, _ = pool.shape
    data = pool[..., : 128 * 64].reshape(-1, 64).clone()
    scale = pool[..., 128 * 64 :].reshape(-1, 4)
    dequant = fp4_utils.mxfp4_to_f32(data) * fp4_utils.e8m0_to_f32(
        scale
    ).repeat_interleave(32, dim=-1)
    return dequant.reshape(pages, heads, 128, 128)


def _mxfp4_pages(x):
    """x [pages, kv_heads, 128, 128] -> (page pool, f32 dequantization).

    A pool row is one (page, head): [128 x 64 B e2m1][128 x 4 B e8m0].
    """
    pages, heads, _, dim = x.shape
    data, scale = per_1x32_f4_quant(x.reshape(-1, dim))
    # Bytes here: torch.cat packs data and scale into one pool (see _mxfp4_views).
    data = data.view(torch.uint8).reshape(-1, dim // 2)
    scale = scale.view(torch.uint8).reshape(-1, dim // 32)
    dequant = fp4_utils.mxfp4_to_f32(data.clone()) * fp4_utils.e8m0_to_f32(
        scale
    ).repeat_interleave(32, dim=-1)
    pool = torch.cat(
        [data.reshape(pages, heads, -1), scale.reshape(pages, heads, -1)], -1
    )
    return pool, dequant.reshape(x.shape)


def _round_query_e4m3(query):
    """A bf16 query as the MXFP4 kernel sees it: each head row rounded to e4m3."""
    scale = (query.float().abs().amax(-1, keepdim=True) / 448).clamp_min(1e-30)
    return (query.float() / scale).to(torch.float8_e4m3fn).float() * scale


def _mxfp4_views(pool):
    """The fp4x2 data and fp8_e8m0 scale views of a pool that pa_decode takes.

    The pool holds both per (page, head), so the two views share page and head strides.
    """
    pages, heads, _ = pool.shape
    page_head = pool.stride()[:2]
    data = pool.as_strided((pages, heads, 128, 64), (*page_head, 64, 1))
    scale = pool.as_strided((pages, heads, 128, 4), (*page_head, 4, 1), 128 * 64)
    return data.view(dtypes.fp4x2), scale.view(dtypes.fp8_e8m0)


# (tag, (q exponent, K e8m0 byte, K e2m1 bytes, V e2m1 bytes[, V e8m0 byte, tokens])).
# V byte and tokens default to 127 and len(k_codes). Logits stay inside f32; the query
# and K scale alone are extreme. With tokens, Q = K = 0: every token has weight 1, the
# codes cycle over the tokens, and the output is the mean of V, exact in bf16 and
# checked bitwise.
_MXFP4_POOL_SPECS = [
    ("mxfp4-tiny-query", (-80, 207, (0x22, 0xAA), (0x22, 0xAA))),
    ("mxfp4-tiny-query-k2e120", (-80, 247, (0x22, 0xAA), (0x22, 0xAA))),
    ("mxfp4-unit-query-k2e120", (0, 247, (0x22, 0xAA), (0x22, 0xAA))),
    ("mxfp4-subnormal-query-k2e127", (-133, 254, (0x22, 0xAA), (0x22, 0xAA))),
    ("mxfp4-huge-query-k2e-120", (120, 7, (0x22, 0xAA), (0x22, 0xAA))),
    # Logits 0.35, 0.71, 1.06: the softmax weights depend on the restored scale.
    ("mxfp4-weighted", (-104, 227, (0x11, 0x22, 0x33), (0x22, 0xAA, 0x11))),
    # Four equal weights of V = 2**126 (byte 253): output 2**126, but an unscaled f32 PV
    # sum reaches 2**128 before the division.
    ("mxfp4-pv-sum-2e128", (None, 127, (0x00,) * 4, (0x22,) * 4, 253)),
    # Two tokens at 2**127 (byte 254), two zeros: sum 2**128, output 2**126.
    (
        "mxfp4-pv-sum-2e127-mixed",
        (None, 127, (0x00,) * 4, (0x22, 0x00, 0x22, 0x00), 254),
    ),
    # Smallest valid magnitudes (e2m1 0.5 at e8m0 byte 1; byte 0 is unsupported): the PV
    # sum nears the f32 subnormal floor under the P shift and must stay exact, not zero.
    ("mxfp4-pv-1tok-2e-126", (None, 127, (0,), (0x22,), 1, 1)),
    ("mxfp4-pv-1tok-2e-127", (None, 127, (0,), (0x11,), 1, 1)),
    ("mxfp4-pv-4tok-2e-127", (None, 127, (0,), (0x11,), 1, 4)),
    ("mxfp4-pv-4tok-mixed-2e-126", (None, 127, (0,), (0x11, 0x33), 1, 4)),
    # Largest context one task owns (4096 staged pages) at the largest finite V
    # (1.5 * 2**127): an unscaled f32 PV sum would reach 1.5 * 2**146.
    ("mxfp4-pv-ctx2e19-v2e127", (None, 127, (0,), (0x33,), 254, 2**19)),
]


def _mxfp4_pool_inputs(
    q_exp, k_byte, k_codes, v_codes, v_byte=127, tokens=None, *, device
):
    """One-head decode over hand-built MXFP4 pages (every logical page is page 0).

    A bf16 query at 2**q_exp (zero for None) meets K scales of 2**(k_byte - 127): the
    logits are finite, but the e4m3-normalized query times such a K scale reaches f32
    range or beyond before the query scale is restored. V scales of 2**(v_byte - 127)
    test the PV sum: it must not overflow where the output is finite. With ``tokens``
    the context is that long, and tokens past it hold 6.0 so a missing mask shows.
    """
    length = tokens or len(k_codes)
    rows = min(length, 128)

    def page(codes, exponent, poison=False):
        pool = torch.zeros(1, 1, 128 * 64 + 128 * 4, dtype=torch.uint8, device=device)
        data, scale = pool[..., : 128 * 64], pool[..., 128 * 64 :]
        if poison:
            data[...] = 0x77
            scale[...] = exponent
        for token in range(rows):
            data.view(1, 1, 128, 64)[..., token, :] = codes[token % len(codes)]
        scale.view(1, 1, 128, 4)[..., :rows, :] = exponent
        return pool, _mxfp4_dequant(pool)

    query = torch.full(
        (1, 1, 128),
        0.0 if q_exp is None else 2.0**q_exp,
        dtype=torch.bfloat16,
        device=device,
    )
    table = torch.zeros(1, -(-length // 128), dtype=torch.int32, device=device)
    context = torch.tensor([length], dtype=torch.int32, device=device)
    return (
        query,
        page(k_codes, k_byte),
        page(v_codes, v_byte, poison=tokens is not None and length < 128),
        table,
        context,
    )


def _short_bound_pool(*, device):
    """313 pages, K = 0 (uniform softmax), V = 1 only on logical pages 256 and up: the
    output is the share of those tokens (0.18) unless their pages go through the table.
    """
    pages, length = 313, 40000
    table = torch.randperm(pages, dtype=torch.int32, device=device)[None]
    key_pool = torch.zeros(
        pages, 1, 128 * 64 + 128 * 4, dtype=torch.uint8, device=device
    )
    value_pool = torch.zeros_like(key_pool)
    key_pool[..., 128 * 64 :] = value_pool[..., 128 * 64 :] = 127
    value_pool[table[0, 256:].long(), :, : 128 * 64] = 0x22
    query = torch.ones(1, 16, 128, dtype=torch.bfloat16, device=device)
    context = torch.tensor([length], dtype=torch.int32, device=device)
    return (
        query,
        (key_pool, _mxfp4_dequant(key_pool)),
        (value_pool, _mxfp4_dequant(value_pool)),
        table,
        context,
    )


_COMMON_CASES = [
    (
        "fine-plan-window128",
        {
            "group_size": 16,
            "query_length": 4,
            "block_size": 128,
            "max_partitions": 7,
            "window": 257,
            "tile_tokens": 128,
            "sink_dtype": torch.float32,
            "capture": True,
        },
    ),
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
    # Sparse decode rows of up to 2048 tokens take 8 partitions, which gfx1250
    # merges within one workgroup cluster; replays must not see stale partials.
    (
        "cluster8",
        {
            "group_size": 16,
            "block_size": 16,
            "lengths": (2048,) * 6 + (0, 1, 257, 2047),
            "max_partitions": 8,
            "capture": True,
            "cluster": 8,
        },
    ),
    (
        "cluster16-record-fallback",
        {
            "group_size": 16,
            "block_size": 16,
            "lengths": (8192,) * 6 + (0, 1, 257, 8191),
            "max_partitions": 16,
            "capture": True,
            "cluster": 16,
        },
    ),
    (
        "mtp4-g16-page128",
        {
            "query_length": 4,
            "group_size": 16,
            "block_size": 128,
            "max_partitions": 7,
            "sink_dtype": torch.float32,
        },
    ),
    (
        "direct-context-cap1",
        {
            "direct_context": True,
            "pattern": "one_hot_token",
            "group_size": 16,
            "block_size": 16,
            "lengths": (0, 1, 15, 16, 17, 127, 128, 129, 2048),
            "max_partitions": 1,
            "capture": True,
        },
    ),
]
_FP8_CASES = _COMMON_CASES + [
    (
        "single-page-plan128",
        {
            "group_size": 16,
            "block_size": 128,
            "lengths": (0, 1, 31, 127, 128, 129, 8192),
            "max_partitions": 64,
            "max_context": 8192,
            "tile_tokens": 128,
            "single_page": True,
            "capture": True,
        },
    ),
    (
        "cluster8-scale-head-padding",
        {
            "kv_heads": 2,
            "group_size": 8,
            "block_size": 16,
            "lengths": (0, 1, 33, 129, 257, 1027, 2048),
            "max_partitions": 8,
            "scale_head_padding": 8,
            "cluster": 8,
            "capture": True,
        },
    ),
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
    # Four 16-row query tiles at D128 give each gfx1250 wave its own tile.
    # Varied token magnitudes make a misplaced per-token scale fail; rows past
    # one page run unmasked interior pages (two-token rows exceed FP8 P error).
    (
        "mtp4-g16-page128-direct",
        {"query_length": 4, "group_size": 16, "block_size": 128},
    ),
    (
        "mtp4-g16-page128-varied",
        {
            "query_length": 4,
            "group_size": 16,
            "block_size": 128,
            "max_partitions": None,
            "lengths": (65, 129, 257, 300, 1027, 4100),
            "pattern": "varied",
        },
    ),
    (
        "mtp4-g16-page128-probability_tail",
        {
            "query_length": 4,
            "group_size": 16,
            "block_size": 128,
            "pattern": "probability_tail",
        },
    ),
    # One task slot per sequence: slots of the empty rows give the long row
    # several tasks, which its last-arriving CTA merges across replays.
    (
        "mtp4-g16-page128-merge",
        {
            "query_length": 4,
            "group_size": 16,
            "block_size": 128,
            "lengths": (0, 0, 1027, 4100),
            "max_partitions": 4,
            "budget": 1,
            "capture": True,
        },
    ),
    # 60 rows pad the last wave's tile; GQA 8 puts two queries in each tile.
    (
        "mtp5-g12-page128",
        {"query_length": 5, "group_size": 12, "block_size": 128, "max_partitions": 7},
    ),
    ("mtp8-g8-page128", {"query_length": 8, "block_size": 128, "max_partitions": 7}),
]
_BF16_CASES = _COMMON_CASES + [
    # One query tile keeps four independent page-quarter states on gfx1250.
    # Exercise token/page tails, padded query rows and every output mode.
    *[
        (
            f"decode-wave-g{group}-direct",
            {
                "group_size": group,
                "block_size": 128,
                "lengths": (
                    0,
                    1,
                    31,
                    32,
                    33,
                    63,
                    64,
                    65,
                    127,
                    128,
                    129,
                    255,
                    256,
                    257,
                    8193,
                ),
                "capture": True,
            },
        )
        for group in (1, 8, 16)
    ],
    (
        "decode-wave-cluster",
        {
            "group_size": 16,
            "kv_heads": 2,
            "block_size": 128,
            "lengths": (0, 1, 33, 129, 257, 1027, 8193),
            "max_partitions": 8,
            "cluster": 8,
            "capture": True,
        },
    ),
    (
        "decode-wave-merge",
        {
            "group_size": 16,
            "block_size": 128,
            "lengths": (0, 0, 1027, 8193),
            "max_partitions": 4,
            "budget": 1,
            "capture": True,
        },
    ),
    (
        "decode-wave-reducer-fp16-query",
        {
            "group_size": 16,
            "block_size": 128,
            "query_dtype": torch.float16,
            "lengths": (0, 1, 65, 1027, 8193),
            "max_partitions": 7,
            "sink_dtype": torch.float32,
            "capture": True,
        },
    ),
    (
        "decode-wave-causal-window",
        {
            "query_length": 3,
            "group_size": 4,
            "block_size": 128,
            "lengths": (0, 1, 31, 33, 65, 129, 257, 1027, 8193),
            "window": 257,
            "sink_dtype": torch.float32,
            "capture": True,
        },
    ),
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
    (
        "cluster16",
        {
            "group_size": 16,
            "block_size": 16,
            "lengths": (4096,) * 3 + (0, 255),
            "max_partitions": 16,
            "cluster": 16,
        },
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
    # Four 16-row query tiles at D64/D128 over page128 caches: on gfx1250, wave
    # pairs share two tiles and split every page, in each output mode. One
    # task per sequence runs window-masked, unmasked and causal pages in turn;
    # one slot per sequence merges the long rows' tasks across replays.
    (
        "mtp4-g16-page128-direct",
        {"query_length": 4, "group_size": 16, "block_size": 128},
    ),
    (
        "mtp4-g16-page128-cluster",
        {
            "query_length": 4,
            "kv_heads": 2,
            "group_size": 16,
            "block_size": 128,
            "max_partitions": 7,
        },
    ),
    (
        "mtp4-g16-page128-merge",
        {
            "query_length": 4,
            "group_size": 16,
            "block_size": 128,
            "lengths": (0, 0, 1027, 4100),
            "max_partitions": 4,
            "budget": 1,
            "capture": True,
        },
    ),
    (
        "mtp4-g16-page128-fp16-query",
        {
            "query_length": 4,
            "group_size": 16,
            "block_size": 128,
            "query_dtype": torch.float16,
            "lengths": (65, 1027, 4100),
            "max_partitions": 3,
            "sink_dtype": torch.float16,
        },
    ),
    (
        "mtp4-g16-page128-window",
        {
            "query_length": 4,
            "group_size": 16,
            "block_size": 128,
            "lengths": (65, 300, 1027, 4100),
            "window": 700,
            "sink_dtype": torch.float32,
        },
    ),
    # 60 rows pad the last tile; GQA 8 puts two queries in each tile.
    (
        "mtp5-g12-page128",
        {"query_length": 5, "group_size": 12, "block_size": 128, "max_partitions": 7},
    ),
    ("mtp8-g8-page128", {"query_length": 8, "block_size": 128, "max_partitions": 7}),
    # D64 takes the same wave-pair pipeline with V chunks 1 KiB apart.
    *[
        (f"{name}-d64", dict(geometry, head_dim=64))
        for name, geometry in (
            (
                "mtp4-g16-page128-direct",
                {"query_length": 4, "group_size": 16, "block_size": 128},
            ),
            (
                "mtp4-g16-page128-cluster",
                {
                    "query_length": 4,
                    "kv_heads": 2,
                    "group_size": 16,
                    "block_size": 128,
                    "max_partitions": 7,
                },
            ),
            (
                "mtp4-g16-page128-merge",
                {
                    "query_length": 4,
                    "group_size": 16,
                    "block_size": 128,
                    "lengths": (0, 0, 1027, 4100),
                    "max_partitions": 4,
                    "budget": 1,
                    "capture": True,
                },
            ),
            (
                "mtp4-g16-page128-window",
                {
                    "query_length": 4,
                    "group_size": 16,
                    "block_size": 128,
                    "lengths": (65, 300, 1027, 4100),
                    "window": 700,
                    "sink_dtype": torch.float32,
                },
            ),
            (
                "mtp5-g12-page128",
                {
                    "query_length": 5,
                    "group_size": 12,
                    "block_size": 128,
                    "max_partitions": 7,
                },
            ),
        )
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
]
# MXFP4 K/V on gfx950: head_dim 128, page 128, one query token a row, an e4m3 or a
# bf16 query. Plan-free runs one task per row; unequal 32-element blocks make a
# block/scale mix-up visible.
_E4M3 = torch.float8_e4m3fn
_MXFP4_SWEEP = (1, 15, 16, 17, 127, 128, 129, 255, 256, 257, 511, 2047, 2048)
_MXFP4_BASE = {
    "kv_dtype": "mxfp4",
    "block_size": 128,
    "group_size": 16,
    "max_partitions": None,
    "pattern": "block_scales",
}
_MXFP4_CASES = [
    *[
        (
            f"mxfp4-g{group}-{'e4m3' if dtype == _E4M3 else 'bf16'}-q",
            {
                **_MXFP4_BASE,
                "group_size": group,
                "query_dtype": dtype,
                "lengths": _MXFP4_SWEEP,
            },
        )
        for group in (16, 8)
        for dtype in (torch.bfloat16, _E4M3)
    ],
    *[
        (
            f"mxfp4-plan-free-{'e4m3' if dtype == _E4M3 else 'bf16'}-q",
            {
                **_MXFP4_BASE,
                "query_dtype": dtype,
                "plan_free": True,
                "lengths": (1, 16, 255, 256, 257, 1000, 2048, 4097, 5000),
            },
        )
        for dtype in (torch.bfloat16, _E4M3)
    ],
    *[
        (
            f"mxfp4-kv{kv_heads}-g{group}",
            {
                **_MXFP4_BASE,
                "query_dtype": _E4M3,
                "kv_heads": kv_heads,
                "group_size": group,
                "lengths": (1, 17, 300, 1025),
            },
        )
        for kv_heads, group in ((2, 16), (4, 4), (1, 1), (1, 3))
    ],
    *[
        (
            f"mxfp4-np{partitions}",
            {
                **_MXFP4_BASE,
                "query_dtype": _E4M3,
                "max_partitions": partitions,
                "lengths": (16, 255, 1000, 4097, 5000),
            },
        )
        for partitions in (1, 2, 3, 8)
    ],
    *[
        (
            f"mxfp4-{pattern}-128k{'-plan-free' if free else ''}",
            {
                **_MXFP4_BASE,
                "query_dtype": _E4M3,
                "pattern": pattern,
                "max_partitions": 8,
                "plan_free": free,
                "lengths": (131072,),
            },
        )
        for pattern, free in (
            ("peaked", False),
            ("peaked", True),
            ("peaked-extreme", False),
        )
    ],
    (
        "mxfp4-one-task-per-row",
        {
            **_MXFP4_BASE,
            "query_dtype": _E4M3,
            "max_partitions": 1,
            "lengths": (2048, 1, 129, 777, 1000, 17, 2048, 300) * 8,
        },
    ),
    *[
        (
            f"mxfp4-{'plan-free-' if free else ''}graph",
            {
                **_MXFP4_BASE,
                "query_dtype": torch.bfloat16,
                "max_partitions": 7,
                "plan_free": free,
                "capture": True,
                "lengths": (2048, 700, 2048, 33),
            },
        )
        for free in (False, True)
    ],
    *[
        (
            tag,
            {
                **_MXFP4_BASE,
                "group_size": 1,
                "query_dtype": torch.bfloat16,
                "pattern": "pool",
                "plan_free": True,
                "pool": partial(_mxfp4_pool_inputs, *spec),
                "exact": len(spec) > 5,
            },
        )
        for tag, spec in _MXFP4_POOL_SPECS
    ],
    # A two-partition plan over 1024 tokens with max_context_length 128: the reducer
    # must merge both partials, whatever the bound.
    (
        "mxfp4-short-bound-np2",
        {
            **_MXFP4_BASE,
            "query_dtype": _E4M3,
            "max_partitions": 2,
            "max_context": 128,
            "lengths": (1024,),
        },
    ),
    # One task per row three ways: plan-free, a one-partition plan, a strided output (the
    # reducer); the same bytes as uint8 views; a context-0 row is zero-filled.
    (
        "mxfp4-one-task-variants",
        {
            **_MXFP4_BASE,
            "query_dtype": _E4M3,
            "max_partitions": 1,
            "plan_free": True,
            "variants": True,
            "lengths": (300, 0, 2048, 5),
        },
    ),
    # max_context_length (128 tokens) far below the context (40000): the page ids past
    # the staged window come from the block table. One task owns all 313 pages:
    # through the reducer (a one-partition plan with a strided output), directly (the
    # same plan, contiguous output) or plan-free.
    *[
        (
            f"mxfp4-short-bound-{mode}",
            {
                **_MXFP4_BASE,
                "query_dtype": _E4M3,
                "max_partitions": 1,
                "max_context": 128,
                "pattern": "pool",
                "pool": _short_bound_pool,
                "plan_free": mode == "plan-free",
                "strided_output": mode == "reducer",
                "ref_mean_min": 0.1,
            },
        )
        for mode in ("reducer", "direct", "plan-free")
    ],
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
_MXFP4_INVALID_CASES = [
    ("fp16-output", TypeError, "bf16 output"),
    ("fp16-query", TypeError, "e4m3 or bf16 query"),
    ("page-64", ValueError, "uint8 .* view"),
    ("17-heads", NotImplementedError, "1..16 query heads"),
    ("sinks", NotImplementedError, "no sinks"),
    ("sliding-window", NotImplementedError, "no sliding window"),
    ("query-length-2", NotImplementedError, "query_length 1, got 2"),
    ("compute-type", NotImplementedError, "compute_type None, uint8 or fp4x2"),
    ("plan-free-strides", ValueError, "output needs the query's strides"),
    ("oversized-task", ValueError, "PAGE_ID_STAGE_LIMIT"),
    # Operands the kernel would read with the wrong layout or plan: each used to
    # produce wrong output (or an AttributeError) instead of an error.
    ("non-contiguous-table", ValueError, "block_tables must be contiguous"),
    ("context-on-cpu", ValueError, "context_lengths must be contiguous on the query"),
    ("context-length-mismatch", ValueError, r"query.shape\[0\] .* must match"),
    ("table-rows-mismatch", ValueError, r"query.shape\[0\] .* must match"),
    ("query-stride-2", ValueError, "contiguous head_dim axis"),
    ("output-stride-2", ValueError, "contiguous head_dim axis"),
    ("query-span", ValueError, "query must span at most"),
    ("output-span", ValueError, "output must span at most"),
    ("output-on-cpu", ValueError, "output must be on the query device"),
    ("scratch-on-cpu", ValueError, "exp_sums must be contiguous .* on cuda"),
    ("key-scale-none", TypeError, "key_scale must be a uint8 torch.Tensor"),
    ("scale-pages-mismatch", ValueError, "key_scale must be a uint8 .* view"),
    ("misaligned-key", ValueError, "key_cache base pointer must be 16 B aligned"),
    ("misaligned-scale", ValueError, "key_scale base pointer must be 4 B aligned"),
    # A windowed plan is refused with one partition (served without it) and more.
    ("windowed-plan-1", ValueError, "sliding_window must match work_plan"),
    ("windowed-plan-2", ValueError, "sliding_window must match work_plan"),
]
# name -> (arch FlyDSL reports, kv_dtype, query_dtype) of a PaDecodeSchedule.select call.
_SELECT_INVALID = {
    "mxfp4-on-gfx942": ("gfx942", "mxfp4", "bf16"),
    "fp8-query-fp8-kv": ("gfx950", "fp8", "fp8"),
    "fp8-query-bf16-kv": ("gfx950", "bf16", "fp8"),
}
_SELECT_INVALID_CASES = [
    ("mxfp4-on-gfx942", ValueError, "MXFP4 PA decode requires gfx950"),
    ("fp8-query-fp8-kv", ValueError, "An FP8 query needs MXFP4 KV"),
    ("fp8-query-bf16-kv", ValueError, "BF16 KV requires a BF16 query"),
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


def _mxfp4_inputs(device, lengths=(300, 1, 2048, 5)):
    """e4m3 query, MXFP4 K/V views and the dequantized reference, one query token."""
    torch.manual_seed(0)
    query, key, value, table, context = _make_inputs(
        1, 1, 16, 128, 128, torch.bfloat16, "block_scales", device, lengths
    )
    query = query.to(torch.float8_e4m3fn)
    (key_pool, logical_key), (value_pool, logical_value) = (
        _mxfp4_pages(key),
        _mxfp4_pages(value),
    )
    (kd, ks), (vd, vs) = _mxfp4_views(key_pool), _mxfp4_views(value_pool)
    return query, kd, ks, vd, vs, table, context, (logical_key, logical_value)


def _check_unsupported_mxfp4(invalid, error, match, device):
    """An MXFP4 input the wrapper refuses before any kernel compiles."""
    pa = importlib.import_module("aiter.ops.flydsl.pa_decode")
    query, kd, ks, vd, vs, table, context, _ = _mxfp4_inputs(device)
    output = torch.empty(query.shape, dtype=torch.bfloat16, device=device)
    options = {"key_scale": ks, "value_scale": vs, "work_plan": None}
    query_length = 1
    if invalid == "fp16-output":
        output = output.half()
    elif invalid == "fp16-query":
        query = query.to(torch.float16)
    elif invalid == "page-64":
        kd = kd[:, :, :64]
    elif invalid == "17-heads":
        query = torch.cat([query, query[:, :1]], 1)
        output = torch.empty(query.shape, dtype=torch.bfloat16, device=device)
    elif invalid == "sinks":
        options["sinks"] = torch.zeros(16, device=device)
    elif invalid == "sliding-window":
        options["sliding_window"] = 256
    elif invalid == "query-length-2":
        query_length = 2
    elif invalid == "compute-type":
        options["compute_type"] = torch.float8_e4m3fn
    elif invalid == "plan-free-strides":
        output = torch.empty(4, 32, 128, dtype=torch.bfloat16, device=device)[:, :16]
    elif invalid == "oversized-task":
        table = torch.zeros(4, 4097, dtype=torch.int32, device=device)
    elif invalid == "non-contiguous-table":
        table = torch.cat([table, table], 1)[:, ::2]
    elif invalid == "context-on-cpu":
        context = context.cpu()
    elif invalid == "context-length-mismatch":
        context = context[:3]
    elif invalid == "table-rows-mismatch":
        table = table[:3]
    elif invalid == "query-stride-2":
        strided = torch.empty(4, 16, 256, dtype=query.dtype, device=device)[..., ::2]
        query = strided.copy_(query)
    elif invalid == "output-stride-2":
        output = torch.empty(4, 16, 256, dtype=torch.bfloat16, device=device)[..., ::2]
    elif invalid == "query-span":
        # Just over 2**31 - 1 bytes between the first and the last row's end.
        rows = query.shape[0] - 1
        base = torch.empty(rows * 720_000_000 + 2048, dtype=query.dtype, device=device)
        query = base.as_strided(query.shape, (720_000_000, *query.stride()[1:]))
    elif invalid == "output-span":
        rows = output.shape[0] - 1
        base = torch.empty(rows * 400_000_000 + 2048, dtype=output.dtype, device=device)
        output = base.as_strided(output.shape, (400_000_000, *output.stride()[1:]))
    elif invalid == "output-on-cpu":
        output = output.cpu()
    elif invalid == "scratch-on-cpu":
        # Scratch is only used (and checked) when the plan has more than one partition.
        options["work_plan"] = pa.plan_pa_decode(context, 1, max_partitions=2)
        options["exp_sums"] = torch.empty(
            1, options["work_plan"].capacity, 16, device="cpu"
        )
    elif invalid == "key-scale-none":
        options["key_scale"] = None
    elif invalid == "scale-pages-mismatch":
        ks = ks[:1]
        options["key_scale"] = ks
    elif invalid == "misaligned-key":
        # One byte into the pool: in bounds, with the shape and strides unchanged.
        kd = kd.as_strided(kd.shape, kd.stride(), kd.storage_offset() + 1)
    elif invalid == "misaligned-scale":
        ks = ks.as_strided(ks.shape, ks.stride(), ks.storage_offset() - 1)
        options["key_scale"] = ks
    elif invalid.startswith("windowed-plan"):
        options["work_plan"] = pa.plan_pa_decode(
            context, 1, max_partitions=int(invalid[-1]), sliding_window=256
        )
    with pytest.raises(error, match=match):
        pa.pa_decode(
            output, query, kd, vd, context, table, 1.0, query_length, **options
        )


def _check_select_rejected(monkeypatch, invalid, error, match):
    """The schedule refuses these itself, whatever the Python wrapper checks first."""
    traits = importlib.import_module("aiter.ops.flydsl.kernels.pa_decode.traits")
    arch, kv_dtype, query_dtype = _SELECT_INVALID[invalid]
    monkeypatch.setattr(traits, "get_rocm_arch", lambda: arch)
    with pytest.raises(error, match=match):
        traits.PaDecodeSchedule.select(
            head_dim=128,
            query_group_size=16,
            block_size=128,
            num_seqs=1,
            num_kv_heads=1,
            num_compute_units=256,
            work_capacity=1,
            query_dtype=query_dtype,
            kv_dtype=kv_dtype,
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


def _check_pipe_policy(pa, case, arch, compile_kwargs):
    """Pin the selected specialization so a case cannot drift off its path."""
    from aiter.ops.flydsl.kernels.pa_decode.traits import PaDecodeSchedule

    assert compile_kwargs[-1]["wide_kv_addressing"] == case["wide"]
    if (case.get("single_page") or case.get("direct_context")) and arch == "gfx1250":
        from aiter.ops.flydsl.kernels import pa_decode_kernel as kernels

        selected = compile_kwargs[-1]
        if case.get("single_page"):
            assert selected["plan_tile_tokens"] == 128
        compiled = pa.compile_pa_decode_tile(**selected)
        traits = [
            t
            for t, entry in kernels._GFX1250_PA_DECODE_CACHE.items()
            if entry is compiled
        ]
        assert len(traits) == 1
        if case.get("single_page"):
            assert traits[0].single_task_tile and traits[0].buffers == 1
        if case.get("direct_context"):
            assert traits[0].direct_context
    if case["cluster"] is not None:
        expected = case["cluster"] if arch == "gfx1250" else 0
        assert compile_kwargs[-1]["cluster_merge"] == expected
    if case["pipe"]:
        # The cache key ends with (D256_PIPE, staged page ids).
        schedule_kwargs = dict(compile_kwargs[-1])
        schedule_kwargs.pop("architecture", None)
        schedule_kwargs.pop("direct_output", None)
        schedule_kwargs.pop("cluster_merge", None)
        schedule_kwargs.pop("packed_scale_heads", None)
        schedule_kwargs.pop("cluster_task_stride", None)
        schedule_kwargs.pop("plan_tile_tokens", None)
        pipe, staged = PaDecodeSchedule.select(**schedule_kwargs).cache_key[-2:]
        assert pipe == (arch == "gfx950")
        if pipe:
            assert (staged > 0) == (case["staged"] is not False)


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
        *[pytest.param(geometry, id=name) for name, geometry in _MXFP4_CASES],
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
        *[
            pytest.param(
                {
                    "invalid_input": invalid,
                    "invalid_kind": "mxfp4",
                    "kv_dtype": "mxfp4",
                    "error": error,
                    "match": match,
                },
                id=f"invalid-mxfp4-{invalid}",
            )
            for invalid, error, match in _MXFP4_INVALID_CASES
        ],
        *[
            pytest.param(
                {
                    "invalid_input": invalid,
                    "invalid_kind": "select",
                    "error": error,
                    "match": match,
                },
                id=f"invalid-select-{invalid}",
            )
            for invalid, error, match in _SELECT_INVALID_CASES
        ],
    ],
)
def test_pa_decode(case, device, monkeypatch):
    """FP8/BF16/MXFP4 decode shares input, reference, planning and replay checks.

    A case with ``invalid_input`` is an input the wrapper must refuse before any
    kernel compiles.
    """
    if case.get("tile_tokens") == 128 and get_gfx_runtime() != "gfx1250":
        pytest.skip("128-token plans require gfx1250")
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
            "budget": None,
            "table_width": None,
            "max_context": "lengths",
            "wide": False,
            "pipe": False,
            "staged": None,
            "invalid_kind": "bf16",
            "plan_free": False,
            "pool": None,
            "exact": False,
            "strided_output": False,
            "ref_mean_min": None,
            "variants": False,
            "cluster": None,
        },
        **case,
    )
    arch = get_gfx_runtime()
    if case["kv_dtype"] == "mxfp4" and arch != "gfx950":
        pytest.skip("MXFP4 pa_decode runs on gfx950")
    if (
        arch != "gfx1250"
        and case["kv_dtype"] == "bf16"
        and case["query_dtype"] == torch.float16
        and "invalid_input" not in case
    ):
        pytest.skip("BF16 KV requires bfloat16 queries on gfx942/gfx950")
    num_cu = torch.cuda.get_device_properties(device).multi_processor_count
    if case["max_partitions"] is not None and case["max_partitions"] > num_cu:
        pytest.skip(f"plan_pa_decode caps max_partitions at {num_cu} CUs")
    pa = importlib.import_module("aiter.ops.flydsl.pa_decode")
    if get_gfx_runtime() == "gfx1250":
        if case["invalid_kind"] == "bf16" and case.get("invalid_input") in (
            "fp16-query",
            "plain-v",
        ):
            pytest.skip("gfx1250 supports FP16 queries and plain BF16 V caches")
        if (
            case["kv_dtype"] == "bf16"
            and case["head_dim"] % 128
            and case["head_dim"] != 64
        ):
            pytest.skip("gfx1250 supports head_dim 64 or multiples of 128")
    if case["invalid_kind"] == "mxfp4":
        _check_unsupported_mxfp4(
            case["invalid_input"], case["error"], case["match"], device
        )
        return
    if case["invalid_kind"] == "select":
        _check_select_rejected(
            monkeypatch, case["invalid_input"], case["error"], case["match"]
        )
        return
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
    torch.manual_seed(0)
    # An e4m3 query is the bf16 one rounded.
    query, key, value, table, context = _make_inputs(
        query_length,
        kv_heads,
        group_size,
        dim,
        page,
        (
            torch.bfloat16
            if case["query_dtype"] == torch.float8_e4m3fn
            else case["query_dtype"]
        ),
        case["pattern"],
        device,
        case["lengths"],
        case["table_width"],
    )
    pools = None
    if case["pool"]:
        # Hand-built pages: the pool, table, context and query replace the random ones.
        query, *pools, table, context = case["pool"](device=device)
    query = query.to(case["query_dtype"])
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
            if mode == "per-token" and case.get("scale_head_padding", 0):
                # NaN padding makes an assumed packed head stride observable
                # on KV head 1, while the token dimension remains contiguous.
                def pad_scale_heads(scale):
                    shape = scale.shape[:-1]
                    padded = torch.full(
                        (*shape[:-1], shape[-1] + case["scale_head_padding"]),
                        float("nan"),
                        dtype=scale.dtype,
                        device=scale.device,
                    )
                    padded[..., : shape[-1]].copy_(scale.squeeze(-1))
                    return padded[..., : shape[-1]].unsqueeze(-1)

                key_scale = pad_scale_heads(key_scale)
                value_scale = pad_scale_heads(value_scale)
        else:
            key, value = key.to(compute_type), value.to(compute_type)
            if mode == "python":
                key_scale, value_scale = 0.25, 2.0
    elif case["kv_dtype"] == "bf16":
        # FP16 queries read BF16 caches.
        key, value = key.to(compute_type), value.to(compute_type)
    if case["kv_dtype"] == "mxfp4":
        compute_type = torch.uint8
        (key_pool, logical_key), (value_pool, logical_value) = pools or (
            _mxfp4_pages(key),
            _mxfp4_pages(value),
        )
        (key_cache, key_scale), (value_cache, value_scale) = (
            _mxfp4_views(key_pool),
            _mxfp4_views(value_pool),
        )
    else:
        logical_key = key.float() * (1.0 if key_scale is None else key_scale)
        logical_value = value.float() * (1.0 if value_scale is None else value_scale)
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
        tile_tokens=case.get("tile_tokens"),
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

    output = torch.full_like(
        query,
        float("nan"),
        dtype=torch.bfloat16 if query.dtype == torch.float8_e4m3fn else query.dtype,
    )
    if case["strided_output"]:
        # Rows of a wider tensor: the output cannot be written directly (the reducer).
        output = torch.full(
            (query.shape[0], 2 * query.shape[1], dim),
            float("nan"),
            dtype=output.dtype,
            device=device,
        )[:, : query.shape[1]]
    workspace = {}
    if case["capture"]:
        scalar_shape = (kv_heads, plan.capacity, query_length * group_size)
        workspace = {
            "exp_sums": torch.empty(scalar_shape, dtype=torch.float32, device=device),
            "max_logits": torch.empty(scalar_shape, dtype=torch.float32, device=device),
            "temporary_output": torch.empty(
                (*scalar_shape, dim), dtype=output.dtype, device=device
            ),
        }

    def launch():
        pa.pa_decode(
            output,
            *arguments,
            work_plan=None if case["plan_free"] else plan,
            **options,
            **workspace,
        )

    def check(out=None):
        out = output if out is None else out
        if case["exact"]:
            # K = 0, Q = 0: the output is the mean of V, bit for bit and nonzero (the 5e-3
            # atol would pass a flushed result at 2**-126).
            rows = min(context[0].item(), 128)
            expected = logical_value[0, 0, :rows].double().mean(0).to(torch.bfloat16)
            assert (expected != 0).all()
            assert torch.equal(out[0, 0], expected), (out[0, 0, :4], expected[:4])
            return
        reference = _reference(
            # The kernel quantizes a bf16 query; an e4m3 query is used as stored (a
            # hand-built pool's query is used as given, its logits are what is extreme).
            (
                _round_query_e4m3(query)
                if case["kv_dtype"] == "mxfp4"
                and query.dtype == torch.bfloat16
                and not case["pool"]
                else query
            ),
            logical_key,
            logical_value,
            table,
            context,
            query_length,
            window,
            sinks,
            # The long peaked cases are the accuracy check of the P shift: fp64.
            torch.float64 if case["pattern"].startswith("peaked") else torch.float32,
        )
        if case["pool"]:
            assert torch.isfinite(reference).all() and reference.abs().max() > 0.1
        if case["ref_mean_min"]:
            assert reference.mean() > case["ref_mean_min"]
        _check_output(out, reference)

    launch()
    check()
    cluster_size = compile_kwargs[-1]["cluster_merge"]
    use_records = (
        plan.cluster_info is not None and 0 < cluster_size <= plan.cluster_task_stride
    )
    if use_records:
        assert compile_kwargs[-1]["cluster_task_stride"] == plan.cluster_task_stride
        packed = output.clone()
        original_plan = plan
        plan = replace(plan, cluster_info=None)
        launch()
        check()
        assert compile_kwargs[-1]["cluster_task_stride"] == 0
        assert torch.equal(output, packed)
        plan = original_plan
    elif cluster_size:
        assert compile_kwargs[-1]["cluster_task_stride"] == 0
    if case["variants"]:
        # Plan-free, a one-partition plan and uint8 views of the same bytes are one
        # task; a strided output takes the reducer, and a context-0 row is zero-filled
        # on every path (7.0 would stay otherwise).
        def run(out, caches=(key_cache, value_cache, key_scale, value_scale), **opts):
            key, value, ks, vs = caches
            pa.pa_decode(
                out,
                query,
                key,
                value,
                context,
                table,
                dim**-0.5,
                query_length,
                key_scale=ks,
                value_scale=vs,
                **opts,
            )
            return out

        free = run(torch.full_like(output, 7.0), work_plan=None)
        planned = run(torch.full_like(output, 7.0), work_plan=plan)
        as_bytes = run(
            torch.full_like(output, 7.0),
            caches=tuple(
                t.view(torch.uint8)
                for t in (key_cache, value_cache, key_scale, value_scale)
            ),
            work_plan=None,
        )
        assert torch.equal(free, planned) and torch.equal(free, as_bytes)
        check(free)
        padded = torch.full(
            (query.shape[0], 32, dim), 7.0, dtype=output.dtype, device=device
        )
        check(run(padded[:, : query.shape[1]], work_plan=plan))
        assert (padded[:, query.shape[1] :] == 7.0).all()
    if case["kv_dtype"] != "mxfp4":
        _check_pipe_policy(pa, case, arch, compile_kwargs)
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


def _poison_unowned(cache, scale, owned, fill):
    """Overwrite K/V rows no sequence owns (and their per-token scales)."""
    rows = ~owned[:, None, :].expand(-1, cache.shape[1], -1)
    if cache.dtype == torch.bfloat16:
        bits = {"zero": 0, "nan": 0x7FC0, "inf": 0x7F80}[fill]
        cache.view(torch.int16)[rows] = bits
    else:
        # E4M3 has no infinity; its largest finite magnitude stands in.
        nan = 0x80 if cache.dtype == torch.float8_e4m3fnuz else 0x7F
        bits = {"zero": 0, "nan": nan, "inf": 0x7E}[fill]
        cache.view(torch.uint8)[rows] = bits
    if scale is not None and scale.dtype in (torch.uint8, dtypes.fp8_e8m0):
        # e8m0: 255 is NaN, 254 is 2**127.
        scale.view(torch.uint8)[rows] = {"zero": 0, "nan": 255}.get(fill, 254)
    elif scale is not None and scale.numel() > 1:
        scale.view(scale.shape[:3])[rows] = {"zero": 0.0, "nan": float("nan")}.get(
            fill, float("inf")
        )


@pytest.mark.parametrize("fill", ["nan", "inf"])
@pytest.mark.parametrize(
    "query_length,kv_heads,group_size,head_dim",
    [(1, 1, 8, 128), (4, 2, 8, 256), (4, 1, 16, 128)],
)
@pytest.mark.parametrize("block_size", [16, 64, 128])
@pytest.mark.parametrize(
    "kv",
    ["fp8-per-tensor", "fp8-per-tensor-plain-v", "fp8-per-token", "bf16", "mxfp4"],
)
def test_pa_decode_unowned_kv_rows(
    device, kv, block_size, query_length, kv_heads, group_size, head_dim, fill
):
    """Rows no sequence owns contribute exactly zero, even when NaN or Inf.

    Padded block-table lanes read block 0 and the last page's tail past the
    context is read as is. Both may hold NaN (a recycled page, a checkpoint
    image); P is zero there but 0 * NaN is NaN in PV.
    ``mask_unowned_kv=False`` skips the masking and matches it on clean input.
    """
    if get_gfx_runtime() == "gfx1250":
        pytest.skip("gfx1250 pipelines still require finite unowned V rows")
    if kv == "mxfp4" and (
        get_gfx_runtime() != "gfx950"
        or (block_size, query_length, head_dim) != (128, 1, 128)
    ):
        pytest.skip("MXFP4 pa_decode: gfx950, page 128, head_dim 128, one query token")
    pa = importlib.import_module("aiter.ops.flydsl.pa_decode")
    torch.manual_seed(0)
    lengths = tuple(max(n, query_length) for n in (1, 17, 255, 257, 1000))
    query, key, value, table, context = _make_inputs(
        query_length,
        kv_heads,
        group_size,
        head_dim,
        block_size,
        torch.bfloat16,
        "random",
        device,
        lengths,
    )
    # Page 0 is owned by no sequence; padded block-table lanes point at it.
    spare = torch.zeros_like(key[:1])
    key, value = torch.cat([spare, key]), torch.cat([spare, value])
    table = torch.nn.functional.pad(table + 1, (0, 2))
    owned = torch.zeros(key.shape[0], block_size, dtype=torch.bool, device=device)
    for seq, length in enumerate(lengths):
        tokens = torch.arange(length, device=device)
        owned[table[seq, tokens // block_size].long(), tokens % block_size] = True

    key_scale = value_scale = None
    vector = 8
    if kv.startswith("fp8"):
        compute_type = (
            torch.float8_e4m3fn
            if get_gfx_runtime() == "gfx950"
            else torch.float8_e4m3fnuz
        )
        vector = 16
        quantize = pertoken_quant if kv == "fp8-per-token" else per_tensor_quant
        key, key_scale = quantize(key, quant_dtype=compute_type)
        value, value_scale = quantize(value, quant_dtype=compute_type)
    if kv == "mxfp4":
        (key_pool, logical_key), (value_pool, logical_value) = (
            _mxfp4_pages(key),
            _mxfp4_pages(value),
        )
    else:
        logical_key = key.float() * (1.0 if key_scale is None else key_scale)
        logical_value = value.float() * (1.0 if value_scale is None else value_scale)
    reference = _reference(
        query, logical_key, logical_value, table, context, query_length, 0, None
    )
    plan = pa.plan_pa_decode(context, kv_heads, query_length=query_length)

    def run(fill, mask_unowned_kv=True):
        if kv == "mxfp4":
            (k, ks), (v, vs) = (
                _mxfp4_views(key_pool.clone()),
                _mxfp4_views(value_pool.clone()),
            )
        else:
            k, v = key.clone(), value.clone()
            ks = None if key_scale is None else key_scale.clone()
            vs = None if value_scale is None else value_scale.clone()
        _poison_unowned(k, ks, owned, fill)
        _poison_unowned(v, vs, owned, fill)
        key_cache, value_cache = k, v
        if kv != "mxfp4":
            pages = k.shape[0]
            key_cache = (
                k.reshape(pages, kv_heads, block_size, head_dim // vector, vector)
                .permute(0, 1, 3, 2, 4)
                .contiguous()
            )
            if kv == "fp8-per-tensor-plain-v":
                value_cache = v.permute(0, 1, 3, 2).contiguous()
            else:
                value_cache = (
                    v.reshape(pages, kv_heads, block_size // vector, vector, head_dim)
                    .permute(0, 1, 2, 4, 3)
                    .contiguous()
                )
        output = torch.full_like(query, float("nan"))
        pa.pa_decode(
            output,
            query,
            key_cache,
            value_cache,
            context,
            table,
            head_dim**-0.5,
            query_length,
            compute_type=key_cache.dtype,
            key_scale=ks,
            value_scale=vs,
            work_plan=plan,
            max_context_length=max(lengths),
            mask_unowned_kv=mask_unowned_kv,
        )
        return output

    clean, poisoned = run("zero"), run(fill)
    assert torch.isfinite(poisoned).all()
    # Short FP8 contexts round P coarsely; exactness is the bitwise check.
    tol = 5e-3 if kv == "bf16" else 2.5e-2
    torch.testing.assert_close(poisoned.float(), reference, atol=tol, rtol=tol)
    assert torch.equal(poisoned, clean)
    if fill == "nan":
        assert torch.equal(run("zero", mask_unowned_kv=False), clean)
