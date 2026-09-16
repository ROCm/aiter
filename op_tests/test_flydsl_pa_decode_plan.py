# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Dynamic PA plans: coverage, packed reduction, and graph replay correctness."""

import importlib

import pytest
import torch

from aiter.ops.flydsl.pa_decode import get_recommended_splits, pa_decode, plan_pa_decode
from op_tests import flydsl_pa_decode_test_utils as reference_tests


@pytest.fixture(autouse=True)
def cuda_device():
    reference_tests._require_gpu()
    previous = torch.get_default_device()
    torch.set_default_device("cuda")
    yield
    torch.set_default_device(previous)


def _assert_plan(plan, lengths):
    work = plan.work_info.cpu().tolist()
    reductions = plan.reduce_info.cpu().tolist()
    tiles_per_sequence = [(max(length, 0) + 255) // 256 for length in lengths]
    nonempty = [int(tiles > 0) for tiles in tiles_per_sequence]
    total_tiles = max(sum(tiles_per_sequence), 1)
    remaining = plan.capacity - sum(nonempty)
    cumulative_tiles = 0
    expected_counts = []
    for tiles, is_nonempty in zip(tiles_per_sequence, nonempty):
        lower = cumulative_tiles * remaining // total_tiles
        cumulative_tiles += tiles
        upper = cumulative_tiles * remaining // total_tiles
        expected_counts.append(
            min(is_nonempty + upper - lower, tiles, plan.max_partitions)
        )

    expected_reductions = []
    expected_work = []
    offset = 0
    for seq, (length, tiles, count) in enumerate(
        zip(lengths, tiles_per_sequence, expected_counts)
    ):
        start = offset
        expected_reductions.append([start, count])
        for part in range(count):
            expected_work.append(
                [
                    seq,
                    part * tiles // count,
                    (part + 1) * tiles // count,
                    length,
                ]
            )

        actual_start, actual_count = reductions[seq]
        assert (actual_start, actual_count) == (start, count)
        assert 0 <= actual_count <= min(tiles, plan.max_partitions)
        assert (actual_count > 0) == (length > 0)
        previous_end = 0
        for task in work[actual_start : actual_start + actual_count]:
            assert task[0] == seq and task[3] == length
            assert task[1] == previous_end and task[1] < task[2] <= tiles
            previous_end = task[2]
        assert previous_end == tiles
        offset += actual_count
    expected_work.extend([[0, 0, 0, 0]] * (plan.capacity - offset))
    assert reductions == expected_reductions
    assert work == expected_work
    assert offset <= plan.capacity
    assert all(row == [0, 0, 0, 0] for row in work[offset:])


@pytest.mark.parametrize(
    "heads,max_parts,budget", [(1, 1, 512), (1, 7, 17), (1, 256, 512), (2, 256, 512)]
)
@pytest.mark.parametrize(
    "lengths",
    [
        [0],
        [1],
        [257],
        [200003],
        [0] * 8,
        [0, 1, 3, 4, 257, 4095, 16385, 200003],
        [100000] * 8,
        [i * 7919 % 200003 for i in range(200)],
    ],
)
def test_plan_covers_each_token_once(lengths, heads, max_parts, budget):
    context = torch.tensor(lengths, dtype=torch.int32)
    plan = plan_pa_decode(
        context, heads, max_partitions=max_parts, workgroup_budget=budget
    )
    _assert_plan(plan, lengths)


def test_plan_graph_refresh_overwrites_old_metadata():
    lengths = [200003] * 8
    context = torch.tensor(lengths, dtype=torch.int32)
    plan = plan_pa_decode(context, 1)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        plan_pa_decode(context, 1, plan=plan)
    for lengths in ([0] * 8, [0, 1, 3, 4, 257, 4096, 16385, 200003], [200003] * 8):
        context.copy_(torch.tensor(lengths, dtype=torch.int32))
        graph.replay()
        _assert_plan(plan, lengths)


def _planned_call(*args, **kwargs):
    plan = plan_pa_decode(args[4], args[2].shape[1], max_partitions=args[8])
    pa_decode(*args, work_plan=plan, **kwargs)


@pytest.mark.parametrize("block_size", [16, 128])
@pytest.mark.parametrize("trans_v", [False, True])
@pytest.mark.parametrize("max_parts", [1, 7, 64, 86, 256])
@pytest.mark.parametrize("heads", [1, 2])
def test_planned_mtp4_sparse_causal_reference(
    monkeypatch, block_size, trans_v, max_parts, heads
):
    monkeypatch.setattr(torch.ops.aiter, "pa_decode_flydsl", _planned_call)
    reference_tests._run_mtp4_fused_reference_case(
        [0, 1, 2, 3, 4, 255, 256, 257, 2305, 16387],
        heads,
        block_size,
        trans_v,
        max_parts,
    )


@pytest.mark.parametrize(
    "head_dim,block_size,parts,dtype",
    [
        (64, 16, 8, torch.bfloat16),
        (256, 64, 86, torch.float16),
        (1024, 128, 256, torch.bfloat16),
        (128, 128, 256, torch.float16),
    ],
)
def test_planned_decode_other_reducer_paths(
    monkeypatch, head_dim, block_size, parts, dtype
):
    monkeypatch.setattr(torch.ops.aiter, "pa_decode_flydsl", _planned_call)
    reference_tests._run_accuracy_case(
        [0, 1, 257, 4099],
        num_query_heads=8,
        num_kv_heads=2,
        head_dim=head_dim,
        block_size=block_size,
        query_dtype=dtype,
        num_partitions=parts,
        tolerance=0.005,
        trans_v=True,
    )


@pytest.mark.parametrize("long_context", [16384, 16385])
def test_planned_decode_register_reducer_chunk_boundary(monkeypatch, long_context):
    """Exercise the generic reducer with 64 and 65 active partitions."""
    monkeypatch.setattr(torch.ops.aiter, "pa_decode_flydsl", _planned_call)
    reference_tests._run_accuracy_case(
        [0, 1, 257, long_context],
        num_query_heads=8,
        num_kv_heads=2,
        head_dim=256,
        block_size=64,
        query_dtype=torch.float16,
        num_partitions=86,
        tolerance=0.005,
        trans_v=True,
    )


def test_planned_decode_graph_replay_with_poisoned_scratch(monkeypatch):
    def capture_and_replay(*args, **kwargs):
        output, query, key, _value, context = args[:5]
        heads, rows = (
            key.shape[1],
            query.shape[0] // context.numel() * query.shape[1] // key.shape[1],
        )
        plan = plan_pa_decode(context, heads, max_partitions=args[8])
        shape = (heads, plan.capacity, rows)
        psum = torch.full(shape, float("nan"), dtype=torch.float32)
        pmax = torch.full_like(psum, float("nan"))
        pout = torch.full((*shape, query.shape[2]), float("nan"), dtype=query.dtype)

        def run():
            plan_pa_decode(context, heads, max_partitions=args[8], plan=plan)
            pa_decode(
                *args,
                **kwargs,
                work_plan=plan,
                exp_sums=psum,
                max_logits=pmax,
                temporary_output=pout
            )

        run()  # Compile before capture.
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run()
        active_tasks = int(plan.reduce_info[:, 1].sum().item())
        assert active_tasks < plan.capacity
        for scratch in (pmax, psum, pout):
            assert torch.isnan(scratch[:, active_tasks:]).all()
        original = context.clone()
        for scratch in (pmax, psum, pout):
            scratch.fill_(float("nan"))
        context.zero_()
        graph.replay()
        assert torch.equal(output, torch.zeros_like(output))
        for scratch in (pmax, psum, pout):
            assert torch.isnan(scratch).all()
        context.copy_(original)
        graph.replay()
        for scratch in (pmax, psum, pout):
            assert torch.isnan(scratch[:, active_tasks:]).all()

    monkeypatch.setattr(torch.ops.aiter, "pa_decode_flydsl", capture_and_replay)
    reference_tests._run_mtp4_fused_reference_case(
        [0, 257, 4096, 200003], 1, 128, True, 256
    )


def test_plan_rejects_incompatible_reuse():
    context = torch.tensor([1, 257], dtype=torch.int32)
    plan = plan_pa_decode(context, 1, max_partitions=7)
    with pytest.raises(ValueError, match="max_partitions"):
        plan_pa_decode(context, 1, plan=plan)
    with pytest.raises(ValueError, match="KV head"):
        plan_pa_decode(context, 2, max_partitions=7, plan=plan)
    with pytest.raises(ValueError, match="shape"):
        plan_pa_decode(context[:1], 1, max_partitions=7, plan=plan)


@pytest.mark.parametrize("split_kv_blocks", [2, 16])
@pytest.mark.parametrize(
    "batch_size,context_length,max_partitions,expected",
    [
        (1, None, None, 8),
        (1, 0, None, 8),
        (1, 257, None, 8),
        (1, 4096, None, 16),
        (1, 16384, None, 64),
        (1, 100000, None, 256),
        (2, 200000, None, 256),
        (3, 200000, None, 171),
        (5, 200000, None, 103),
        (6, 200000, None, 86),
        (7, 200000, None, 74),
        (4, 200000, None, 128),
        (8, 200000, None, 64),
        (1, 200000, 8, 8),
        (4, 200000, 5, 5),
        (8, 200000, 32, 32),
    ],
)
def test_recommended_splits_uses_context_work(
    monkeypatch, split_kv_blocks, batch_size, context_length, max_partitions, expected
):
    if get_recommended_splits is None:
        pytest.skip("FlyDSL is not available")
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda *_args, **_kwargs: type("Props", (), {"multi_processor_count": 256})(),
    )
    assert (
        get_recommended_splits(
            batch_size,
            1,
            split_kv_blocks,
            max_partitions,
            max_context_length=context_length,
        )
        == expected
    )


@pytest.mark.parametrize("split_kv_blocks,expected", [(2, 4), (16, 8)])
def test_recommended_splits_keeps_large_batch_policy(
    monkeypatch, split_kv_blocks, expected
):
    if get_recommended_splits is None:
        pytest.skip("FlyDSL is not available")
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda *_args, **_kwargs: type("Props", (), {"multi_processor_count": 256})(),
    )
    assert (
        get_recommended_splits(200, 1, split_kv_blocks, max_context_length=200000)
        == expected
    )


@pytest.mark.parametrize("num_partitions", [64, 74, 86, 103, 128, 171, 256])
@pytest.mark.parametrize("trans_v", [False, True])
@pytest.mark.parametrize("block_size", [16, 128])
def test_mtp4_large_partition_mixed_contexts(num_partitions, trans_v, block_size):
    """Long, sparse MTP contexts and empty partitions share the high-NP reducer."""
    reference_tests._require_gpu()
    reference_tests._run_mtp4_fused_reference_case(
        [0, 1, 3, 4, 255, 256, 257, 200003],
        1,
        block_size,
        trans_v,
        num_partitions,
    )


@pytest.mark.parametrize("wide_kv_addressing", [False, True])
@pytest.mark.parametrize("num_partitions", [1, 8])
@pytest.mark.parametrize("trans_v", [False, True])
@pytest.mark.parametrize("block_size", [16, 128])
@pytest.mark.parametrize(
    "lengths,num_kv_heads",
    [
        pytest.param([4], 1, id="c4-hkv1"),
        pytest.param([257], 1, id="c257-hkv1"),
        pytest.param([259], 1, id="c259-hkv1"),
        pytest.param([2305], 1, id="c2305-hkv1"),
        pytest.param([2307], 1, id="c2307-hkv1"),
        pytest.param(
            [0, 1, 2, 3, 4, 255, 256, 257, 258, 259, 2303, 2304, 2305, 2306, 2307],
            1,
            id="mixed-hkv1",
        ),
        pytest.param([257], 2, id="c257-hkv2"),
        pytest.param([259], 2, id="c259-hkv2"),
    ],
)
def test_mtp4_fused_phase_b_accuracy(
    monkeypatch,
    lengths,
    num_kv_heads,
    block_size,
    trans_v,
    num_partitions,
    wide_kv_addressing,
):
    """Cover fused MTP boundaries under both narrow and wide V-load schedules."""
    reference_tests._require_gpu()
    pa_decode_module = importlib.import_module("aiter.ops.flydsl.pa_decode")
    compile_tile = pa_decode_module.compile_pa_decode_tile

    def compile_fused_mtp4(**kwargs):
        kwargs["query_splits"] = 1
        # Exercise wide-address scheduling with sparse/empty/tail cases without
        # allocating a multi-GiB cache for every boundary in this matrix.
        kwargs["wide_kv_addressing"] = wide_kv_addressing
        return compile_tile(**kwargs)

    monkeypatch.setattr(pa_decode_module, "compile_pa_decode_tile", compile_fused_mtp4)
    reference_tests._run_mtp4_fused_reference_case(
        lengths, num_kv_heads, block_size, trans_v, num_partitions
    )
