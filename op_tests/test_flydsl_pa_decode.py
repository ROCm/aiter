# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Generic FlyDSL paged decode correctness test.

Run with: python -m pytest -q op_tests/test_flydsl_pa_decode.py
"""

import importlib.util
from contextlib import contextmanager
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
    query_length, kv_heads, group_size, dim, page_size, dtype, pattern, device, lengths
):
    if lengths is None:
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
_AUTOTUNE_CASES = [
    (
        "fp8-per-token",
        {"kv_dtype": "fp8", "scale_mode": "per-token", "head_dim": 128},
    ),
    (
        "fp8-per-tensor",
        {
            "kv_dtype": "fp8",
            "query_dtype": torch.float16,
            "scale_mode": "per-tensor",
            "trans_v": False,
            "head_dim": 64,
        },
    ),
    ("bf16-mtp", {"head_dim": 192, "query_length": 4}),
    ("bf16-window", {"query_length": 4, "window": 257, "sink_dtype": torch.float32}),
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


def _check_same_plan(actual, expected):
    assert actual.capacity == expected.capacity
    assert actual.max_partitions == expected.max_partitions
    torch.testing.assert_close(actual.work_info, expected.work_info, atol=0, rtol=0)
    torch.testing.assert_close(actual.reduce_info, expected.reduce_info, atol=0, rtol=0)


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


def _check_cache_reuse(pa, tuner, arguments, options, plan, reference, monkeypatch):
    """Allocation geometry and scheduling bounds must reuse the native cache."""
    query, key, value, lengths, table, scale, query_length = arguments
    dim, page = query.shape[-1], key.shape[3]
    padded_query = torch.empty(
        (*query.shape[:-1], dim + 16), dtype=query.dtype, device=query.device
    )
    strided_query = padded_query[..., :dim]
    strided_query.copy_(query)
    extended_arguments = (
        strided_query,
        torch.cat((key, key[:1])),
        torch.cat((value, value[:1])),
        lengths,
        torch.cat((table, torch.zeros_like(table[:, :1])), dim=1),
        scale,
        query_length,
    )
    extended_options = dict(options)
    extended_options["max_context_length"] = table.shape[1] * page + 1
    for name in ("key_scale", "value_scale"):
        kv_scale = options[name]
        if isinstance(kv_scale, torch.Tensor):
            extended_options[name] = (
                torch.cat((kv_scale, kv_scale[:1])).squeeze(-1)
                if kv_scale.numel() > 1
                else kv_scale.reshape(1, 1)
            )

    def forbidden_default(*_, **__):
        pytest.fail("equivalent attention geometry missed the native tuning cache")

    with monkeypatch.context() as cached_only:
        cached_only.setattr(tuner, "default", forbidden_default)
        reused_plan = pa.prepare_pa_decode_plan(
            *extended_arguments, max_partitions=plan.max_partitions, **extended_options
        )
    assert len(tuner.cache) == 1
    _check_same_plan(reused_plan, plan)
    output = torch.full_like(query, float("nan"))
    pa.pa_decode(output, *extended_arguments, work_plan=reused_plan, **extended_options)
    _check_output(output, reference)


def _prepare_autotuned_plan(
    pa, arguments, options, reference, case, monkeypatch, tmp_path
):
    """Check native search, candidate accuracy, persistence and default lookup."""
    from flydsl.autotune import do_bench

    query, key, value, lengths, table, _, _ = arguments
    kv_heads = key.shape[1]
    num_cu = torch.cuda.get_device_properties(query.device).multi_processor_count
    limit = num_cu if case["max_partitions"] is None else case["max_partitions"]
    monkeypatch.delenv("FLYDSL_AUTOTUNE_CONFIG_DIR", raising=False)
    monkeypatch.delenv("FLYDSL_AUTOTUNE_CACHE_DIR", raising=False)
    monkeypatch.setenv("FLYDSL_AUTOTUNE", "1")
    config_root = tmp_path / "configs" / "pa"
    monkeypatch.setattr(pa, "_PA_DECODE_CONFIGS_PATH", config_root)
    arch = torch.cuda.get_device_properties(query.device).gcnArchName.split(":", 1)[0]
    # Use the production path selection without retaining temporary test tuners
    # in the process-wide cache after monkeypatch restores the config root.
    get_autotuner = pa._get_pa_decode_autotuner.__wrapped__
    tuner = get_autotuner(arch)
    cache_file = config_root / arch / "_pa_decode_autotuner.json"
    assert tuner._cache_file == cache_file
    monkeypatch.setattr(pa, "_get_pa_decode_autotuner", {arch: tuner}.__getitem__)
    assert tuner._do_bench is do_bench
    assert tuner.select_config is None
    snapshots = [
        (tensor, tensor.clone())
        for tensor in (query, key, value, lengths, table, *options.values())
        if isinstance(tensor, torch.Tensor)
    ]
    # Scalar host-to-device copies must precede capture, with device tensors
    # retained alongside the graphs holding their pointers.
    host_scales = {
        name: options[name]
        for name in ("key_scale", "value_scale")
        if key.dtype != torch.bfloat16 and not isinstance(options[name], torch.Tensor)
    }
    validated = {}
    original_validate = tuner.validate_hook

    @contextmanager
    def validate_candidate(sig_args):
        resources = sig_args["resources"]
        with original_validate(sig_args):
            yield
        budget = sig_args["workgroup_budget"]
        _check_output(resources.outputs[budget], reference)
        validated[budget] = resources
        for name, expected in host_scales.items():
            scale = resources.options[name]
            assert isinstance(scale, torch.Tensor)
            assert scale.device == query.device
            assert scale.dtype == torch.float32
            assert scale.shape == (1,)
            assert scale.item() == (1.0 if expected is None else expected)

    monkeypatch.setattr(tuner, "validate_hook", validate_candidate)
    plan = pa.prepare_pa_decode_plan(*arguments, max_partitions=limit, **options)
    assert validated
    assert len(tuner.cache) == 1
    assert cache_file.is_file()
    selected = next(iter(tuner.cache.values())).kwargs["workgroup_budget"]
    assert selected in validated
    resources = validated[selected]
    assert set(validated) == set(resources.graphs)
    expected_budgets = {}
    for budget in (2 * num_cu, 128, 256, 512, 1024, 2048, 4096):
        capacity = min(
            lengths.numel() * limit,
            max(lengths.numel(), (budget + kv_heads - 1) // kv_heads),
        )
        expected_budgets.setdefault(capacity, budget)
    assert set(validated) == set(expected_budgets.values())
    assert plan.capacity == min(
        lengths.numel() * limit,
        max(lengths.numel(), (selected + kv_heads - 1) // kv_heads),
    )
    for tensor, snapshot in snapshots:
        torch.testing.assert_close(tensor, snapshot, atol=0, rtol=0)

    def forbidden_prepare(*_, **__):
        pytest.fail("cache/default lookup prepared autotune benchmark candidates")

    def forbidden_benchmark(*_, **__):
        pytest.fail("cache/default lookup benchmarked an autotune candidate")

    monkeypatch.setenv("FLYDSL_AUTOTUNE", "0")
    reloaded = get_autotuner(arch)
    assert reloaded.cache
    other_arch = "gfx950" if arch == "gfx942" else "gfx942"
    other = get_autotuner(other_arch)
    assert other._cache_file == config_root / other_arch / cache_file.name
    assert not other.cache
    assert other.cache is not reloaded.cache
    with monkeypatch.context() as custom_cache:
        custom_cache.setenv("FLYDSL_AUTOTUNE_CACHE_DIR", str(tmp_path / "custom-cache"))
        # An override keeps the original native tuner's cache/path, selected
        # when the module was imported, shared across device architectures.
        assert get_autotuner(arch) is pa._pa_decode_autotuner
        assert get_autotuner(other_arch) is pa._pa_decode_autotuner
    monkeypatch.setattr(pa, "_get_pa_decode_autotuner", {arch: reloaded}.__getitem__)
    monkeypatch.setattr(pa._PADecodeAutotuneResources, "prepare", forbidden_prepare)
    monkeypatch.setattr(reloaded, "_do_bench", forbidden_benchmark)
    cached_plan = pa.prepare_pa_decode_plan(*arguments, max_partitions=limit, **options)
    _check_same_plan(cached_plan, plan)
    if case["cache_reuse"]:
        _check_cache_reuse(
            pa, reloaded, arguments, options, cached_plan, reference, monkeypatch
        )

    # A fresh native-cache miss uses 2*CU without preparing or timing graphs.
    miss = _isolated_pa_decode_autotuner(pa, tmp_path / "empty.json")
    monkeypatch.setattr(pa, "_get_pa_decode_autotuner", {arch: miss}.__getitem__)
    monkeypatch.setattr(miss, "_do_bench", forbidden_benchmark)
    default_plan = pa.prepare_pa_decode_plan(
        *arguments, max_partitions=limit, **options
    )
    assert default_plan.capacity == min(
        lengths.numel() * limit,
        max(lengths.numel(), (2 * num_cu + kv_heads - 1) // kv_heads),
    )
    assert not miss.cache
    options.update(resources.options)
    return cached_plan


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
        *[
            pytest.param(
                dict(
                    kv_heads=2,
                    group_size=4,
                    max_partitions=None,
                    autotune=True,
                    cache_reuse=True,
                    capture=True,
                    **geometry,
                ),
                id=f"autotune-{name}",
            )
            for name, geometry in _AUTOTUNE_CASES
        ],
        *[
            pytest.param(
                {
                    "kv_dtype": "fp8",
                    "group_size": 4,
                    "head_dim": 64,
                    "max_partitions": 2,
                    "pattern": "constant",
                    "lengths": (1, 64),
                    "scale_mode": mode,
                    "autotune": True,
                    "capture": True,
                },
                id=f"autotune-{mode}-scales",
            )
            for mode in ("python", "none")
        ],
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
def test_pa_decode(case, device, monkeypatch, tmp_path):
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
            "autotune": False,
            "cache_reuse": False,
            "lengths": None,
        },
        **case,
    )
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
    )
    compute_type = torch.bfloat16
    key_scale = value_scale = None
    vector = 8
    if case["kv_dtype"] == "fp8":
        compute_type = (
            torch.float8_e4m3fn
            if get_gfx_runtime() == "gfx950"
            else torch.float8_e4m3fnuz
        )
        vector = 16
        mode = case["scale_mode"]
        if mode in ("per-tensor", "per-token"):
            quantize = pertoken_quant if mode == "per-token" else per_tensor_quant
            key, key_scale = quantize(key, quant_dtype=compute_type)
            value, value_scale = quantize(value, quant_dtype=compute_type)
        else:
            key, value = key.to(compute_type), value.to(compute_type)
            if mode == "python":
                key_scale, value_scale = 0.25, 2.0
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
        "max_context_length": context.max().item(),
    }
    arguments = (query, key_cache, value_cache, context, table, dim**-0.5, query_length)
    if case["autotune"]:
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
        plan = _prepare_autotuned_plan(
            pa, arguments, options, reference, case, monkeypatch, tmp_path
        )
    else:
        plan = pa.plan_pa_decode(
            context,
            kv_heads,
            max_partitions=case["max_partitions"],
            query_length=query_length,
            sliding_window=window,
        )
    if "invalid_input" in case:
        _check_unsupported_input(pa, arguments, options, plan, case, monkeypatch)
        return

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
