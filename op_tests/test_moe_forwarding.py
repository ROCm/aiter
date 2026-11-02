# SPDX-License-Identifier: MIT
"""The public MoE wrapper's positional dispatcher call must follow its schema."""

import ast
import importlib
import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

fm = importlib.import_module("aiter.fused_moe")


def _registered_signature():
    # _eager_body exists only if this module registered the implementation.
    return inspect.signature(fm.fused_moe_._eager_body)


def _inputs():
    x = torch.zeros((3, 8))
    return {
        "hidden_states": x,
        "w1": torch.empty((2, 8, 8)),
        "w2": torch.empty((2, 8, 4)),
        "topk_weight": torch.empty((3, 2)),
        "topk_ids": torch.empty((3, 2), dtype=torch.int32),
    }


def test_forwarded_values_follow_registered_schema(monkeypatch):
    signature = _registered_signature()
    seen = []

    def record(*args, **kwargs):
        assert not kwargs
        assert len(args) == len(signature.parameters)
        bound = signature.bind(*args)
        seen.append(bound.arguments)
        return bound.arguments["hidden_states"]

    monkeypatch.setattr(fm, "fused_moe_", record)
    inputs = _inputs()
    defaults = inspect.signature(fm.fused_moe).bind(**inputs)
    defaults.apply_defaults()
    public = defaults.arguments
    assert fm.fused_moe(**inputs) is inputs["hidden_states"]
    actual = seen[-1]
    for name, value in actual.items():
        if name.startswith("ep_"):
            assert value is None if name == "ep_source_token_map" else value == 0
        elif name == "block_size_M":
            assert value == -1
        elif name in ("activation", "quant_type"):
            assert value == public[name].value
        else:
            assert value is public[name] or value == public[name]


@pytest.mark.parametrize("scatter", [False, True])
def test_all_optional_arguments_keep_their_identity(monkeypatch, scatter):
    signature = _registered_signature()
    seen = {}

    def record(*args, **kwargs):
        seen.update(signature.bind(*args, **kwargs).arguments)
        return seen["output"] if seen["output"] is not None else seen["hidden_states"]

    monkeypatch.setattr(fm, "fused_moe_", record)
    inputs = _inputs()
    tensor_names = (
        "expert_mask",
        "w1_scale",
        "w2_scale",
        "a1_scale",
        "a2_scale",
        "num_local_tokens",
        "bias1",
        "bias2",
        "output",
    )
    options = {
        name: torch.empty((index + 1,)) for index, name in enumerate(tensor_names)
    }
    options.update(
        activation=fm.ActivationType.Swiglu,
        quant_type=fm.QuantType.per_1x32,
        doweight_stage1=True,
        block_size_M=32,
        moe_sorting_dispatch_policy=2,
        dtype=torch.float16,
        hidden_pad=11,
        intermediate_pad=13,
        swiglu_limit=7.25,
        beta=4.5,
        linear_beta=25.5,
        gate_mode="interleave",
        quant_type_a=fm.QuantType.per_Token,
        quant_dtype_a=torch.float8_e4m3fn,
        quant_dtype_a2=torch.bfloat16,
        herd_topk=3,
        herd_routed_scaling_factor=1.75,
    )
    ctx = SimpleNamespace(
        arena_handle=101,
        combine_input_offset=103,
        slot_stride_bytes=107,
        max_tokens_per_rank=109,
        world_size=4,
        source_token_map=torch.empty((3, 2)),
        combine_quant_bits=8,
    )
    if scatter:
        options["stage2_scatter"] = ctx
    assert fm.fused_moe(**inputs, **options) is options["output"]
    for name, value in {**inputs, **options}.items():
        if name == "stage2_scatter":
            continue
        want = (
            value.value
            if name in ("activation", "quant_type", "quant_type_a")
            else value
        )
        assert seen[name] is want or seen[name] == want
    for name, source in (
        ("ep_arena_handle", "arena_handle"),
        ("ep_combine_input_offset", "combine_input_offset"),
        ("ep_slot_stride_bytes", "slot_stride_bytes"),
        ("ep_max_tokens_per_rank", "max_tokens_per_rank"),
        ("ep_world_size", "world_size"),
        ("ep_combine_quant", "combine_quant_bits"),
        ("ep_source_token_map", "source_token_map"),
    ):
        want = (
            getattr(ctx, source)
            if scatter
            else (None if name == "ep_source_token_map" else 0)
        )
        assert seen[name] is want or seen[name] == want


def test_internal_positional_order_matches_parameter_names():
    source = Path(fm.__file__).read_text()
    tree = ast.parse(source)
    public = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "fused_moe"
    )
    internal = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "fused_moe_"
    )
    call = next(
        n
        for n in ast.walk(public)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name)
        and n.func.id == "fused_moe_"
    )
    names = [n.arg for n in internal.args.args]
    assert not call.keywords and len(call.args) == len(names)
    for name, value in zip(names, call.args):
        if name.startswith("ep_") or name == "quant_type_a":
            continue  # runtime identity tested above
        if name in ("activation", "quant_type"):
            assert isinstance(value, ast.Attribute) and value.attr == "value"
            assert value.value.id == name
        else:
            assert isinstance(value, ast.Name) and value.id == name


def test_registered_op_and_compile_boundary_remain(monkeypatch):
    seen = []

    def inner(**kwargs):
        seen.append(kwargs)
        return kwargs["hidden_states"] + 1

    monkeypatch.setattr(fm, "_fused_moe_impl", inner)
    values = _inputs()
    expected = values["hidden_states"] + 1
    eager = fm.fused_moe(**values)
    torch.testing.assert_close(eager, expected)

    graphs = []

    def backend(gm, _inputs):
        graphs.append(gm)
        return gm.forward

    def public(x, w1, w2, weights, ids):
        return fm.fused_moe(x, w1, w2, weights, ids)

    compiled = torch.compile(public, backend=backend, fullgraph=True)
    result = compiled(*values.values())
    torch.testing.assert_close(result, expected)
    assert graphs
    assert any(
        "aiter.fused_moe_" in str(n.target) for gm in graphs for n in gm.graph.nodes
    )
    assert len(seen) >= 2


@pytest.mark.parametrize("data_format", ["a4w4", "a8w4"])
@pytest.mark.parametrize("tokens", [1, 16, 257])
@pytest.mark.parametrize(
    "activation",
    [fm.ActivationType.Silu, fm.ActivationType.Swiglu, fm.ActivationType.Situv2],
)
def test_positional_dispatch_matches_keywords_on_gpu(data_format, tokens, activation):
    if not torch.cuda.is_available():
        pytest.skip("requires ROCm")
    from aiter.jit.utils.chip_info import get_gfx_runtime

    if get_gfx_runtime() != "gfx1250":
        pytest.skip("grouped MoE reference requires gfx1250")
    import importlib.util

    path = Path(__file__).parent / "flydsl_tests" / "test_flydsl_grouped_gemm.py"
    spec = importlib.util.spec_from_file_location("_forwarding_grouped_reference", path)
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    registered = fm.fused_moe_
    signature = inspect.signature(registered._eager_body)
    captured = []
    public = reference.fused_moe

    def capture(*args, **kwargs):
        kwargs["quant_dtype_a"] = (
            fm.dtypes.fp8 if data_format == "a8w4" else fm.dtypes.fp4x2
        )
        captured.append((args, kwargs))
        return public(*args, **kwargs)

    reference.fused_moe = capture
    try:
        output, expected, _, _ = reference._run_grouped_via_fused_moe(
            experts=8,
            tokens=tokens,
            topk=2,
            model_dim=256,
            inter_dim=256,
            data_format=data_format,
            activation=activation,
            use_bias=True,
            seed=42,
            bench=False,
        )
    finally:
        reference.fused_moe = public
    assert reference._logits_diff(output, expected) < reference.LOGITS_DIFF_TOL
    args, kwargs = captured[-1]

    def keyword_dispatch(*pos, **kw):
        return registered(**signature.bind(*pos, **kw).arguments)

    def keyword_call():
        # Both arms still enter the real PyTorch registered operator.
        fm.fused_moe_ = keyword_dispatch
        try:
            return fm.fused_moe(*args, **kwargs)
        finally:
            fm.fused_moe_ = registered

    assert torch.equal(fm.fused_moe(*args, **kwargs), keyword_call())
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        replayed = fm.fused_moe(*args, **kwargs)
    torch.cuda.current_stream().wait_stream(stream)
    for _ in range(2):
        args[0].neg_()
        graph.replay()
        assert torch.equal(replayed, keyword_call())
