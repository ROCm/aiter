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
    return query, key, value, table, context


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
@pytest.mark.parametrize("per_token", [False, True], ids=["per-tensor", "per-token"])
@pytest.mark.parametrize("trans_v", [False, True], ids=["plain-v", "transposed-v"])
@pytest.mark.parametrize(
    "query_length,kv_heads,group_size,head_dim,block_size,"
    "max_partitions,window,sink_dtype,pattern",
    [
        pytest.param(1, 1, 8, 128, 16, 1, 0, None, "random", id="decode"),
        pytest.param(3, 2, 4, 64, 64, 3, 0, torch.float16, "random", id="mtp"),
        pytest.param(4, 1, 16, 128, 128, 7, 257, torch.float32, "random", id="window"),
        pytest.param(2, 2, 8, 256, 16, 7, 0, None, "random", id="head256"),
        pytest.param(1, 1, 16, 128, 128, None, 0, None, "random", id="default-cap"),
        pytest.param(1, 1, 4, 1024, 128, 1, 0, torch.bfloat16, "random", id="head1024"),
        *[
            pytest.param(
                1,
                1,
                group,
                128,
                page,
                1,
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
    max_partitions,
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
    plan = plan_pa_decode(
        context,
        kv_heads,
        max_partitions=max_partitions,
        query_length=query_length,
        sliding_window=window,
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


@pytest.mark.parametrize(
    "query_length,kv_heads,group_size,head_dim,block_size,"
    "max_partitions,window,sink_dtype,pattern,capture",
    [
        pytest.param(1, 1, 8, 128, 16, 1, 0, None, "random", False, id="decode"),
        pytest.param(4, 2, 4, 64, 64, 3, 0, torch.float16, "random", False, id="mtp"),
        pytest.param(
            4, 1, 16, 128, 128, 7, 257, torch.float32, "random", False, id="window"
        ),
        pytest.param(
            1, 1, 16, 192, 64, 4, 0, None, "random", False, id="head192-decode"
        ),
        pytest.param(4, 1, 16, 192, 64, 4, 0, None, "random", False, id="head192-mtp"),
        pytest.param(1, 2, 8, 256, 16, 7, 0, None, "random", False, id="head256"),
        pytest.param(
            1, 1, 4, 1024, 128, 1, 0, torch.bfloat16, "random", False, id="head1024"
        ),
        # Large BF16 Q/P operands must split MTP rows to fit the LDS budget.
        pytest.param(
            4, 1, 16, 512, 64, 4, 0, None, "random", False, id="head512-mtp-split2"
        ),
        pytest.param(
            4, 1, 16, 1024, 64, 4, 0, None, "random", False, id="head1024-mtp-split4"
        ),
        pytest.param(
            3, 1, 16, 1024, 64, 4, 0, None, "random", False, id="head1024-mtp-split3"
        ),
        pytest.param(
            4, 1, 8, 128, 64, 7, 0, None, "random", True, id="graph-workspace"
        ),
        *[
            pytest.param(
                1,
                1,
                group,
                128,
                page,
                1,
                0,
                None,
                pattern,
                False,
                id=f"{pattern}-page{page}-g{group}",
            )
            for page, group in ((16, 8), (128, 16))
            for pattern in ("small_query", "probability_tail")
        ],
    ],
)
def test_pa_decode_bf16_kv(
    query_length,
    kv_heads,
    group_size,
    head_dim,
    block_size,
    max_partitions,
    window,
    sink_dtype,
    pattern,
    capture,
    device,
):
    from aiter.ops.flydsl.pa_decode import pa_decode, plan_pa_decode

    torch.manual_seed(0)
    query, key, value, table, context = _make_inputs(
        query_length,
        kv_heads,
        group_size,
        head_dim,
        block_size,
        torch.bfloat16,
        pattern,
        device,
    )
    # Native BF16 operands use vector-8 K and transposed V, without KV scales.
    pages = key.shape[0]
    key_cache = (
        key.reshape(pages, kv_heads, block_size, head_dim // 8, 8)
        .permute(0, 1, 3, 2, 4)
        .contiguous()
    )
    value_cache = (
        value.reshape(pages, kv_heads, block_size // 8, 8, head_dim)
        .permute(0, 1, 2, 4, 3)
        .contiguous()
    )
    sinks = None
    if sink_dtype is not None:
        sinks = torch.linspace(-2, 2, query.shape[1], dtype=sink_dtype, device=device)
        sinks[0], sinks[-1] = float("-inf"), float("inf")
    plan = plan_pa_decode(
        context,
        kv_heads,
        max_partitions=max_partitions,
        query_length=query_length,
        sliding_window=window,
    )
    output = torch.full_like(query, float("nan"))
    workspace = {}
    if capture:
        scalar_shape = (kv_heads, plan.capacity, query_length * group_size)
        workspace = {
            "exp_sums": torch.empty(scalar_shape, dtype=torch.float32, device=device),
            "max_logits": torch.empty(scalar_shape, dtype=torch.float32, device=device),
            "temporary_output": torch.empty(
                (*scalar_shape, head_dim), dtype=query.dtype, device=device
            ),
        }

    def launch():
        pa_decode(
            output,
            query,
            key_cache,
            value_cache,
            context,
            table,
            softmax_scale=head_dim**-0.5,
            query_length=query_length,
            sinks=sinks,
            sliding_window=window,
            work_plan=plan,
            max_context_length=1027 if pattern == "random" else 1024,
            **workspace,
        )

    launch()
    if capture:
        # Compile and allocate the plan/workspace before capture. Reusing the
        # captured query pointer must also observe new data on every replay.
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
            reference = _reference(
                query,
                key.float(),
                value.float(),
                table,
                context,
                query_length,
                window,
                sinks,
            )
            assert torch.isfinite(output).all()
            torch.testing.assert_close(output.float(), reference, atol=5e-3, rtol=5e-3)
    else:
        reference = _reference(
            query,
            key.float(),
            value.float(),
            table,
            context,
            query_length,
            window,
            sinks,
        )
        assert torch.isfinite(output).all()
        torch.testing.assert_close(output.float(), reference, atol=5e-3, rtol=5e-3)


@pytest.mark.parametrize(
    "invalid_input,error,match",
    [
        pytest.param(
            "fp16-query",
            NotImplementedError,
            "BF16 KV requires bfloat16 queries",
            id="fp16-query",
        ),
        pytest.param(
            "plain-v",
            ValueError,
            "BF16 KV requires the vectorized 5D value_cache layout",
            id="plain-v",
        ),
        pytest.param(
            "scaled-kv",
            ValueError,
            "BF16 KV is unscaled; key_scale and value_scale must be None",
            id="scaled-kv",
        ),
        pytest.param(
            "vector16-k", ValueError, "key_cache shape must be", id="vector16-k"
        ),
        pytest.param(
            "vector16-v",
            ValueError,
            "transposed value_cache shape must be",
            id="vector16-v",
        ),
    ],
)
def test_pa_decode_bf16_kv_rejects_unsupported_inputs(
    invalid_input, error, match, device, monkeypatch
):
    pa = importlib.import_module("aiter.ops.flydsl.pa_decode")

    query = torch.empty((1, 8, 128), dtype=torch.bfloat16, device=device)
    key_cache = torch.empty((1, 1, 16, 64, 8), dtype=query.dtype, device=device)
    value_cache = torch.empty((1, 1, 8, 128, 8), dtype=query.dtype, device=device)
    context = torch.ones(1, dtype=torch.int32, device=device)
    table = torch.zeros((1, 1), dtype=torch.int32, device=device)
    kwargs = {}
    if invalid_input == "fp16-query":
        query = query.to(torch.float16)
    elif invalid_input == "plain-v":
        value_cache = torch.empty((1, 1, 128, 64), dtype=query.dtype, device=device)
    elif invalid_input == "scaled-kv":
        kwargs = {
            "key_scale": torch.ones(1, dtype=torch.float32, device=device),
            "value_scale": torch.ones(1, dtype=torch.float32, device=device),
        }
    elif invalid_input == "vector16-k":
        key_cache = torch.empty((1, 1, 8, 64, 16), dtype=query.dtype, device=device)
    elif invalid_input == "vector16-v":
        value_cache = torch.empty((1, 1, 4, 128, 16), dtype=query.dtype, device=device)

    def unexpected_compile(**_):
        pytest.fail("invalid BF16 inputs reached kernel compilation")

    monkeypatch.setattr(pa, "compile_pa_decode_tile", unexpected_compile)
    plan = pa.plan_pa_decode(context, 1, max_partitions=1)
    with pytest.raises(error, match=match):
        pa.pa_decode(
            torch.empty_like(query),
            query,
            key_cache,
            value_cache,
            context,
            table,
            softmax_scale=128**-0.5,
            query_length=1,
            work_plan=plan,
            **kwargs,
        )


def _isolated_pa_decode_autotuner(pa, cache_file):
    """Use FlyDSL's own cache loader while keeping user tuning data untouched."""
    from flydsl.autotune import Autotuner

    original = pa._pa_decode_autotuner
    tuner = Autotuner(
        original.fn,
        configs=original.configs,
        key=original.key,
        warmup=original.warmup,
        rep=original.rep,
        prune_configs_by=original.prune_configs_by,
        reset_to_zero=original.reset_to_zero,
        restore_value=original.restore_value,
        pre_hook=original.pre_hook,
        post_hook=original.post_hook,
        default=original.default,
        artifact_name=original.artifact_name,
        validate_hook=original.validate_hook,
        select_config=original.select_config,
    )
    tuner._cache_file = cache_file
    tuner.cache.clear()
    tuner._load_disk_cache()
    return tuner


@pytest.mark.parametrize(
    "kv_dtype,query_dtype,per_token,trans_v,head_dim,query_length,window,use_sinks",
    [
        pytest.param(
            "fp8", torch.bfloat16, True, True, 128, 1, 0, False, id="fp8-per-token"
        ),
        pytest.param(
            "fp8", torch.float16, False, False, 64, 1, 0, False, id="fp8-per-tensor"
        ),
        pytest.param(
            "bf16", torch.bfloat16, False, True, 192, 4, 0, False, id="bf16-mtp"
        ),
        pytest.param(
            "bf16", torch.bfloat16, False, True, 128, 4, 257, True, id="bf16-window"
        ),
    ],
)
def test_prepare_pa_decode_plan_native_autotune(
    kv_dtype,
    query_dtype,
    per_token,
    trans_v,
    head_dim,
    query_length,
    window,
    use_sinks,
    device,
    monkeypatch,
    tmp_path,
):
    from contextlib import contextmanager

    from flydsl.autotune import do_bench

    pa = importlib.import_module("aiter.ops.flydsl.pa_decode")
    monkeypatch.delenv("FLYDSL_AUTOTUNE_CONFIG_DIR", raising=False)
    monkeypatch.setenv("FLYDSL_AUTOTUNE_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("FLYDSL_AUTOTUNE", "1")
    cache_file = tmp_path / "_pa_decode_autotuner.json"
    tuner = _isolated_pa_decode_autotuner(pa, cache_file)
    monkeypatch.setattr(pa, "_pa_decode_autotuner", tuner)
    # Time the real graphs with FlyDSL's default benchmark and fastest selection.
    assert tuner._do_bench is do_bench
    assert tuner.select_config is None

    torch.manual_seed(0)
    kv_heads, group_size, page_size = 2, 4, 64
    query, key, value, table, context = _make_inputs(
        query_length,
        kv_heads,
        group_size,
        head_dim,
        page_size,
        query_dtype,
        "random",
        device,
    )
    compute_type = torch.bfloat16
    key_scale = value_scale = None
    vector = 8
    if kv_dtype == "fp8":
        compute_type = (
            torch.float8_e4m3fn
            if get_gfx_runtime() == "gfx950"
            else torch.float8_e4m3fnuz
        )
        quantize = pertoken_quant if per_token else per_tensor_quant
        key, key_scale = quantize(key, quant_dtype=compute_type)
        value, value_scale = quantize(value, quant_dtype=compute_type)
        vector = 16
    logical_key = key.float() if key_scale is None else key.float() * key_scale
    logical_value = (
        value.float() if value_scale is None else value.float() * value_scale
    )
    pages = key.shape[0]
    key_cache = (
        key.reshape(pages, kv_heads, page_size, head_dim // vector, vector)
        .permute(0, 1, 3, 2, 4)
        .contiguous()
    )
    value_cache = (
        value.reshape(pages, kv_heads, page_size // vector, vector, head_dim)
        .permute(0, 1, 2, 4, 3)
        .contiguous()
        if trans_v
        else value.permute(0, 1, 3, 2).contiguous()
    )
    sinks = None
    if use_sinks:
        sinks = torch.linspace(
            -2, 2, query.shape[1], dtype=torch.float32, device=device
        )
        sinks[0], sinks[-1] = float("-inf"), float("inf")
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
    options = {
        "compute_type": compute_type,
        "key_scale": key_scale,
        "value_scale": value_scale,
        "sinks": sinks,
        "sliding_window": window,
        "max_context_length": 1027,
    }
    arguments = (
        query,
        key_cache,
        value_cache,
        context,
        table,
        head_dim**-0.5,
        query_length,
    )
    input_snapshots = [
        (tensor, tensor.clone())
        for tensor in (
            query,
            key_cache,
            value_cache,
            context,
            table,
            key_scale,
            value_scale,
            sinks,
        )
        if tensor is not None
    ]
    prepared = {}
    validated = set()
    original_prepare = pa._PADecodeAutotuneResources.prepare
    original_validate = tuner.validate_hook

    def record_prepare(resources, workgroup_budget):
        result = original_prepare(resources, workgroup_budget)
        prepared[workgroup_budget] = resources
        return result

    @contextmanager
    def validate_candidate(sig_args):
        with original_validate(sig_args):
            yield
        budget = sig_args["workgroup_budget"]
        output = prepared[budget].outputs[budget]
        assert torch.isfinite(output).all()
        torch.testing.assert_close(output.float(), reference, atol=5e-3, rtol=5e-3)
        validated.add(budget)

    monkeypatch.setattr(pa._PADecodeAutotuneResources, "prepare", record_prepare)
    monkeypatch.setattr(tuner, "validate_hook", validate_candidate)
    plan = pa.prepare_pa_decode_plan(*arguments, **options)
    assert validated
    selected = next(iter(tuner.cache.values()))
    selected_budget = selected.kwargs["workgroup_budget"]
    assert selected_budget in validated
    resources = prepared[selected_budget]
    # Capacity aliases may be pruned, but every graph retained for search must
    # pass the independent FP32 reference before native timing selects a winner.
    assert validated == set(resources.graphs)
    assert cache_file.is_file()
    num_cu = torch.cuda.get_device_properties(device).multi_processor_count
    expected_budgets = {}
    for budget in (2 * num_cu, 128, 256, 512, 1024, 2048, 4096):
        capacity = min(
            context.numel() * num_cu,
            max(context.numel(), (budget + kv_heads - 1) // kv_heads),
        )
        expected_budgets.setdefault(capacity, budget)
    assert validated == set(expected_budgets.values())
    assert plan.capacity == min(
        context.numel() * plan.max_partitions,
        max(context.numel(), (selected_budget + kv_heads - 1) // kv_heads),
    )
    for tensor, snapshot in input_snapshots:
        torch.testing.assert_close(tensor, snapshot, atol=0, rtol=0)

    def forbidden_prepare(*_, **__):
        pytest.fail("cache/default lookup prepared autotune benchmark candidates")

    def forbidden_benchmark(*_, **__):
        pytest.fail("cache/default lookup benchmarked an autotune candidate")

    monkeypatch.setenv("FLYDSL_AUTOTUNE", "0")
    reloaded = _isolated_pa_decode_autotuner(pa, cache_file)
    assert reloaded.cache
    monkeypatch.setattr(pa, "_pa_decode_autotuner", reloaded)
    monkeypatch.setattr(pa._PADecodeAutotuneResources, "prepare", forbidden_prepare)
    monkeypatch.setattr(reloaded, "_do_bench", forbidden_benchmark)
    cached_plan = pa.prepare_pa_decode_plan(*arguments, **options)
    assert cached_plan.capacity == plan.capacity
    assert cached_plan.max_partitions == plan.max_partitions
    torch.testing.assert_close(cached_plan.work_info, plan.work_info, atol=0, rtol=0)
    torch.testing.assert_close(
        cached_plan.reduce_info, plan.reduce_info, atol=0, rtol=0
    )

    # A fresh native-cache miss uses 2*CU without preparing or timing graphs.
    miss = _isolated_pa_decode_autotuner(pa, tmp_path / "empty.json")
    monkeypatch.setattr(pa, "_pa_decode_autotuner", miss)
    monkeypatch.setattr(miss, "_do_bench", forbidden_benchmark)
    default_plan = pa.prepare_pa_decode_plan(*arguments, **options)
    assert default_plan.capacity == min(
        context.numel() * default_plan.max_partitions,
        max(context.numel(), (2 * num_cu + kv_heads - 1) // kv_heads),
    )
    assert not miss.cache

    # The public selected plan remains reusable with caller-owned scratch during
    # graph capture; decode must not consult the tuner or prepare more candidates.
    output = torch.full_like(query, float("nan"))
    scalar_shape = (kv_heads, cached_plan.capacity, query_length * group_size)
    workspace = {
        "exp_sums": torch.empty(scalar_shape, dtype=torch.float32, device=device),
        "max_logits": torch.empty(scalar_shape, dtype=torch.float32, device=device),
        "temporary_output": torch.empty(
            (*scalar_shape, head_dim), dtype=query.dtype, device=device
        ),
    }

    def launch():
        pa.pa_decode(
            output,
            query,
            key_cache,
            value_cache,
            context,
            table,
            head_dim**-0.5,
            query_length,
            work_plan=cached_plan,
            **options,
            **workspace,
        )

    launch()
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
        assert torch.isfinite(output).all()
        torch.testing.assert_close(output.float(), reference, atol=5e-3, rtol=5e-3)


@pytest.mark.parametrize(
    "key_scale,value_scale",
    [(0.25, 2.0), (None, None)],
    ids=["python-scalars", "no-scales"],
)
def test_prepare_pa_decode_plan_fp8_scalar_scales(
    key_scale, value_scale, device, monkeypatch, tmp_path
):
    from contextlib import contextmanager

    pa = importlib.import_module("aiter.ops.flydsl.pa_decode")
    monkeypatch.delenv("FLYDSL_AUTOTUNE_CONFIG_DIR", raising=False)
    monkeypatch.setenv("FLYDSL_AUTOTUNE", "1")
    tuner = _isolated_pa_decode_autotuner(pa, tmp_path / "scalar_scales.json")
    monkeypatch.setattr(pa, "_pa_decode_autotuner", tuner)
    fp8 = (
        torch.float8_e4m3fn if get_gfx_runtime() == "gfx950" else torch.float8_e4m3fnuz
    )
    query = torch.zeros((2, 4, 64), dtype=torch.bfloat16, device=device)
    key = torch.zeros((2, 1, 64, 64), dtype=torch.bfloat16, device=device).to(fp8)
    value = torch.full_like(key, 0.5)
    key_cache = key.reshape(2, 1, 64, 4, 16).permute(0, 1, 3, 2, 4).contiguous()
    value_cache = value.reshape(2, 1, 4, 16, 64).permute(0, 1, 2, 4, 3).contiguous()
    context = torch.tensor([1, 64], dtype=torch.int32, device=device)
    table = torch.tensor([[0], [1]], dtype=torch.int32, device=device)
    reference = _reference(
        query,
        key.float() * (1.0 if key_scale is None else key_scale),
        value.float() * (1.0 if value_scale is None else value_scale),
        table,
        context,
        1,
        0,
        None,
    )
    original_validate = tuner.validate_hook
    resources_seen = []

    @contextmanager
    def validate_candidate(sig_args):
        resources = sig_args["resources"]
        with original_validate(sig_args):
            yield
        output = resources.outputs[sig_args["workgroup_budget"]]
        assert torch.isfinite(output).all()
        torch.testing.assert_close(output.float(), reference, atol=5e-3, rtol=5e-3)
        resources_seen.append(resources)
        # Scalar copies and unit scales must be materialized before capture and
        # retained with the graphs that hold their device pointers.
        for name, expected in (("key_scale", key_scale), ("value_scale", value_scale)):
            scale = resources.options[name]
            assert isinstance(scale, torch.Tensor)
            assert scale.device == query.device
            assert scale.dtype == torch.float32
            assert scale.shape == (1,)
            assert scale.item() == (1.0 if expected is None else expected)

    monkeypatch.setattr(tuner, "validate_hook", validate_candidate)
    plan = pa.prepare_pa_decode_plan(
        query,
        key_cache,
        value_cache,
        context,
        table,
        64**-0.5,
        1,
        compute_type=fp8,
        key_scale=key_scale,
        value_scale=value_scale,
        max_partitions=2,
        max_context_length=64,
    )
    assert resources_seen
    resources = resources_seen[-1]
    assert len(resources.graphs) == 1  # Every budget has the same capped capacity.
    assert len(tuner.cache) == 1

    output = torch.full_like(query, float("nan"))
    scalar_shape = (1, plan.capacity, 4)
    workspace = {
        "exp_sums": torch.empty(scalar_shape, dtype=torch.float32, device=device),
        "max_logits": torch.empty(scalar_shape, dtype=torch.float32, device=device),
        "temporary_output": torch.empty(
            (*scalar_shape, 64), dtype=query.dtype, device=device
        ),
    }

    def launch():
        pa.pa_decode(
            output,
            query,
            key_cache,
            value_cache,
            context,
            table,
            64**-0.5,
            1,
            work_plan=plan,
            **resources.options,
            **workspace,
        )

    capture_stream = torch.cuda.Stream()
    capture_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(capture_stream):
        launch()
    torch.cuda.current_stream().wait_stream(capture_stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=capture_stream):
        launch()
    graph.replay()
    assert torch.isfinite(output).all()
    torch.testing.assert_close(output.float(), reference, atol=5e-3, rtol=5e-3)
