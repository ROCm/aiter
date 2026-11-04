# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Partition-policy safety checks that do not import GPU libraries."""

import importlib.util
import sys
from itertools import product
from pathlib import Path

import pytest

_PATH = Path(__file__).parents[1] / "aiter/ops/flydsl/pa_decode_policy.py"
_SPEC = importlib.util.spec_from_file_location("pa_partition_policy_host", _PATH)
policy = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = policy
_SPEC.loader.exec_module(policy)


def select(**changes):
    arguments = {
        "arch": "gfx1250",
        "num_compute_units": 256,
        "num_seqs": 16,
        "num_kv_heads": 1,
        "query_group_size": 16,
        "head_dim": 128,
        "query_length": 1,
        "kv_dtype": "fp8",
        "max_context_length": 8192,
    }
    arguments.update(changes)
    return policy.select_pa_decode_partition_config(**arguments)


@pytest.mark.parametrize("batch,context", product((1, 4), (8192, 32768, 131072)))
def test_small_batch_long_context_keeps_parallel_plan(batch, context):
    config = select(num_seqs=batch, max_context_length=context)
    assert (config.max_partitions, config.workgroup_budget) == (256, 512)


@pytest.mark.parametrize("dtype,qlen,batch", product(("fp8", "bf16"), (1, 4), (16, 64)))
def test_long_serial_tasks_keep_original_plan(dtype, qlen, batch):
    config = select(
        kv_dtype=dtype, query_length=qlen, num_seqs=batch, max_context_length=32768
    )
    budget = 1024 if dtype == "fp8" and qlen == 1 and batch == 64 else 512
    assert (config.max_partitions, config.workgroup_budget) == (256, budget)


@pytest.mark.parametrize("context", (32000, 32768, 131072))
def test_long_fp8_single_tile_work_adds_reducer_parallelism(context):
    config = select(num_seqs=64, max_context_length=context)
    assert (config.max_partitions, config.workgroup_budget) == (256, 1024)


@pytest.mark.parametrize("context", (32000, 32768))
def test_bounded_fp8_default_tasks_keep_budget(context):
    config = select(num_seqs=16, max_context_length=context)
    assert (config.max_partitions, config.workgroup_budget) == (256, 512)


@pytest.mark.parametrize("limit", (12, 16, 32))
def test_reducer_budget_fallback_preserves_cap_and_dispatch(limit):
    config = select(num_seqs=64, max_context_length=32768, max_partitions=limit)
    assert config.max_partitions == limit
    assert config.workgroup_budget == (1024 if limit > 16 else 512)


def test_existing_short_context_cluster_choice_is_preserved():
    config = select(num_seqs=64)
    assert (config.max_partitions, config.workgroup_budget) == (16, 1024)


@pytest.mark.parametrize("changes", ({"kv_dtype": "bf16"}, {"query_length": 4}))
def test_reducer_budget_fallback_does_not_apply_to_other_pipelines(changes):
    config = select(num_seqs=64, max_context_length=32768, **changes)
    assert (config.max_partitions, config.workgroup_budget) == (256, 512)


def test_reducer_budget_uses_query_geometry_instead_of_query_length():
    config = select(
        num_seqs=64,
        query_group_size=8,
        query_length=2,
        max_context_length=32768,
    )
    assert (config.max_partitions, config.workgroup_budget) == (256, 1024)


@pytest.mark.parametrize(
    "changes",
    [
        {"arch": "gfx942"},
        {"arch": "gfx950"},
        {"head_dim": 1024},
        {"has_sinks": True},
        {"output_layout_matches": False},
        {"sliding_window": 4096},
        {"num_seqs": 256},
    ],
)
def test_existing_fallbacks_remain_original(changes):
    config = select(**changes)
    assert (config.max_partitions, config.workgroup_budget) == (256, 512)


def test_no_runtime_search_or_configuration_io(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("host policy attempted configuration I/O")

    monkeypatch.setattr(Path, "read_text", forbidden)
    monkeypatch.setenv("FLYDSL_AUTOTUNE", "1")
    monkeypatch.setenv("FLYDSL_AUTOTUNE_CACHE_DIR", "/nonexistent/partition-policy")
    assert select().max_partitions == 16
    assert select(num_seqs=64, max_context_length=32768).workgroup_budget == 1024


def test_low_caps_respect_caller_and_preserve_useful_parallelism():
    for batch, heads, limit, qlen, dim, dtype, context in product(
        (1, 4, 16, 32, 64, 128, 256),
        (1, 2, 4),
        (1, 8, 12, 16, 64, 256),
        (1, 4),
        (64, 128, 256, 512),
        ("fp8", "bf16"),
        (1024, 4096, 8192, 32768),
    ):
        config = select(
            num_seqs=batch,
            num_kv_heads=heads,
            max_partitions=limit,
            query_length=qlen,
            head_dim=dim,
            kv_dtype=dtype,
            max_context_length=context,
        )
        assert 1 <= config.max_partitions <= limit
        assert config.workgroup_budget >= 512
        if config.max_partitions != limit or (
            config.workgroup_budget != 512 and config.max_partitions <= 16
        ):
            tiles = (context + 255) // 256
            assert batch * heads * min(tiles, config.max_partitions) >= 256
            assert (tiles + config.max_partitions - 1) // config.max_partitions <= 4
        elif config.workgroup_budget != 512:
            assert config.workgroup_budget == 1024
            assert dtype == "fp8" and qlen == 1
            assert config.max_partitions > 16


@pytest.mark.parametrize(
    "changes",
    [
        {"num_seqs": 0},
        {"num_compute_units": True},
        {"max_partitions": 0},
        {"max_partitions": 257},
        {"max_context_length": -1},
        {"max_context_length": True},
        {"query_length": 0},
        {"kv_dtype": "fp16"},
    ],
)
def test_invalid_geometry_rejected(changes):
    with pytest.raises(ValueError):
        select(**changes)


@pytest.mark.parametrize("kv_dtype", ("fp8", "bf16"))
@pytest.mark.parametrize(
    "batch,heads,group,dim,qlen,page,context,window,layout,sinks",
    [
        (32, 1, 16, 128, 1, 64, 6144, 0, False, False),
        (16, 2, 8, 128, 1, 16, 8192, 0, False, False),
        (8, 4, 16, 256, 1, 128, 8192, 0, False, False),
        (32, 1, 8, 512, 1, 64, 8192, 0, False, False),
        (64, 1, 16, 64, 4, 128, 8192, 0, False, False),
        (96, 1, 16, 128, 1, 128, 4096, 0, False, False),
        (240, 1, 16, 128, 1, 64, 4096, 0, False, False),
        (241, 1, 16, 128, 1, 64, 4096, 0, False, False),
        (64, 4, 4, 128, 1, 128, 8192, 0, False, False),
        (1, 1, 16, 1024, 4, 128, 1024, 0, False, False),
        (32, 1, 16, 128, 4, 128, 8192, 2048, False, False),
        (16, 1, 16, 128, 1, 128, 8192, 0, True, False),
        (16, 1, 16, 128, 1, 128, 8192, 0, False, True),
        (32, 1, 32, 128, 1, 64, 8192, 0, False, False),
    ],
)
def test_policy_graph_decode(
    kv_dtype,
    batch,
    heads,
    group,
    dim,
    qlen,
    page,
    context,
    window,
    layout,
    sinks,
    monkeypatch,
    tmp_path,
):
    """Generic geometry, mixed lengths and changed-input graph replay match an oracle."""
    import torch

    if not torch.cuda.is_available() or importlib.util.find_spec("flydsl") is None:
        pytest.skip("ROCm and FlyDSL are required")
    from aiter.jit.utils.chip_info import get_gfx_runtime

    if get_gfx_runtime() != "gfx1250":
        pytest.skip("the new partition policy targets gfx1250")
    from aiter.ops.flydsl import prepare_pa_decode_plan_with_policy
    from aiter.ops.flydsl.pa_decode import pa_decode, plan_pa_decode

    spec = importlib.util.spec_from_file_location(
        "canonical_pa_policy_gpu",
        _PATH.parents[3] / "op_tests/test_flydsl_pa_decode.py",
    )
    canonical = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(canonical)
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    # A forced-tune setting and malformed cached result must have no effect.
    monkeypatch.setenv("FLYDSL_AUTOTUNE", "1")
    monkeypatch.setenv("FLYDSL_AUTOTUNE_CACHE_DIR", str(tmp_path))
    (tmp_path / "_pa_decode_autotuner.json").write_text("invalid tuning JSON")
    torch.manual_seed(37)
    query, key, value, table, lengths = canonical._make_inputs(
        qlen,
        heads,
        group,
        dim,
        page,
        torch.bfloat16,
        "varied",
        "cuda",
        (context,) * batch,
        (context + page - 1) // page + 16,
    )
    key_scale = value_scale = None
    vector = 8
    if kv_dtype == "fp8":
        vector = 16
        key_scale = key.float().abs().amax(dim=-1, keepdim=True).clamp(min=1e-20) / 448
        value_scale = (
            value.float().abs().amax(dim=-1, keepdim=True).clamp(min=1e-20) / 448
        )
        key = (key.float() / key_scale).to(torch.float8_e4m3fn)
        value = (value.float() / value_scale).to(torch.float8_e4m3fn)
    logical_key = key.float() if key_scale is None else key.float() * key_scale
    logical_value = (
        value.float() if value_scale is None else value.float() * value_scale
    )
    pages = key.shape[0]
    key = (
        key.reshape(pages, heads, page, dim // vector, vector)
        .permute(0, 1, 3, 2, 4)
        .contiguous()
    )
    value = (
        value.reshape(pages, heads, page // vector, vector, dim)
        .permute(0, 1, 2, 4, 3)
        .contiguous()
    )
    output = torch.empty_like(query)
    if layout:
        output = torch.empty_strided(
            query.shape,
            (heads * group * (dim + 1) + 3, dim + 1, 1),
            dtype=query.dtype,
            device=query.device,
        )
    sink_tensor = None
    if sinks:
        sink_tensor = torch.linspace(-2, 2, heads * group, device=query.device).to(
            query.dtype
        )
    plan = prepare_pa_decode_plan_with_policy(
        query,
        key,
        value,
        lengths,
        table,
        qlen,
        output=output,
        sinks=sink_tensor,
        sliding_window=window,
        max_context_length=context,
    )
    scalar_shape = (heads, plan.capacity, qlen * group)
    workspace = {
        "exp_sums": torch.empty(scalar_shape, device=query.device),
        "max_logits": torch.empty(scalar_shape, device=query.device),
        "temporary_output": torch.empty(
            (*scalar_shape, dim),
            device=query.device,
            dtype=query.dtype,
        ),
    }

    def launch():
        pa_decode(
            output,
            query,
            key,
            value,
            lengths,
            table,
            dim**-0.5,
            qlen,
            work_plan=plan,
            key_scale=key_scale,
            value_scale=value_scale,
            sinks=sink_tensor,
            sliding_window=window,
            max_context_length=context,
            **workspace,
        )

    def check():
        reference = canonical._reference(
            query, logical_key, logical_value, table, lengths, qlen, window, sink_tensor
        )
        canonical._check_output(output, reference)

    launch()
    check()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch()
        launch()
    pattern = (0, 1, 127, 128, 129, 255, 256, 257, 2047, context)
    mixed = [min(context, pattern[i % len(pattern)]) for i in range(batch)]
    query.neg_()
    lengths.copy_(torch.tensor(mixed, dtype=torch.int32, device=query.device))
    plan_pa_decode(lengths, heads, sliding_window=window, query_length=qlen, plan=plan)
    output.fill_(float("nan"))
    graph.replay()
    check()
    lengths.fill_(context)
    plan_pa_decode(lengths, heads, sliding_window=window, query_length=qlen, plan=plan)
    graph.replay()
    check()
