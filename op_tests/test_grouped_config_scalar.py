# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Scalar CSV parsing is cached, not mutable configuration rows or defaults."""

import csv
import gc
import importlib
import weakref
from pathlib import Path

import pytest

pytest.importorskip("flydsl")
gm = importlib.import_module("aiter.ops.flydsl.grouped_moe_gfx1250")
ROOT = Path(__file__).resolve().parents[1]


def previous(value, default):
    """The pre-change conversion/error contract, not a second tuning policy."""
    text = "" if value is None else str(value).strip()
    if text == "" or text.lower() in ("nan", "none"):
        return default
    number = float(text)
    if not number.is_integer():
        raise ValueError(f"expected an integer config value, got {text!r}")
    return int(number)


def outcome(fn, value, default):
    try:
        return ("ok", fn(value, default))
    except (ValueError, TypeError, OverflowError) as error:
        return ("err", type(error).__name__, str(error))


@pytest.fixture(autouse=True)
def clear_cache():
    gm._parse_config_int.cache_clear()
    yield
    gm._parse_config_int.cache_clear()


@pytest.mark.parametrize(
    "value",
    [
        None,
        "",
        "  ",
        "nan",
        "NaN",
        "none",
        "None",
        "  NONE ",
        " 16 ",
        "16.0",
        "3.5",
        "inf",
        "-inf",
        "bad",
        "1e20",
        "0x10",
        "-0.0",
        "-32",
        0,
        1,
        16,
        -1,
        2**65,
        True,
        False,
        16.0,
        3.5,
        float("nan"),
        float("inf"),
        b"16",
        [],
        {},
        "1" * 1000,
    ],
)
@pytest.mark.parametrize("default", [None, -1, 0, 64, "sentinel"])
def test_semantics(value, default):
    assert outcome(previous, value, default) == outcome(gm._as_int, value, default)


def test_current_value_and_default():
    class Mutable:
        def __init__(self):
            self.value = "3"

        def __str__(self):
            return self.value

    v = Mutable()
    for value in ("3", "4", "nan", "17.5", "None", "8"):
        v.value = value
        for d in (None, 31, 64):
            assert outcome(previous, v, d) == outcome(gm._as_int, v, d)


def test_bounded_text_cache():
    gm._parse_config_int.cache_clear()
    for i in range(400):
        assert gm._as_int(str(i), None) == i
    assert gm._parse_config_int.cache_info().currsize == 128


def test_all_shipped_csv_cells():
    count = 0
    for p in (ROOT / "aiter/configs").rglob("*grouped*csv"):
        for row in csv.DictReader(p.open()):
            for v in row.values():
                assert outcome(previous, v, 64) == outcome(gm._as_int, v, 64)
                count += 1
    assert count > 1000


def test_errors_are_not_cached():
    for text in ("bad", "3.5", "inf"):
        before = gm._parse_config_int.cache_info()
        for _ in range(2):
            assert outcome(gm._as_int, text, None) == outcome(previous, text, None)
        after = gm._parse_config_int.cache_info()
        assert after.misses == before.misses + 2
        assert after.currsize == before.currsize


def test_cache_retains_neither_value_object_nor_default():
    class Scalar:
        def __str__(self):
            return "16.0"

    value = Scalar()
    ref = weakref.ref(value)
    assert gm._as_int(value, None) == 16
    del value
    gc.collect()
    assert ref() is None
    default = Scalar()
    ref = weakref.ref(default)
    assert gm._as_int("nan", default) is default
    del default
    gc.collect()
    assert ref() is None


def test_cached_values_reuse_but_changed_row_still_applies():
    row = {"tile_m": "16", "num_buffers": "3"}
    for _ in range(2):
        assert gm._as_int(row["tile_m"], 64) == 16
        assert gm._as_int(row["num_buffers"], 2) == 3
    assert gm._parse_config_int.cache_info().hits >= 2
    row["tile_m"] = "32.0"
    row["num_buffers"] = "nan"
    assert gm._as_int(row["tile_m"], 64) == 32
    assert gm._as_int(row["num_buffers"], 4) == 4
    assert gm._as_int(row["num_buffers"], 2) == 2


@pytest.mark.parametrize("fmt", ["a4w4", "a8w4"])
@pytest.mark.parametrize("tokens", [1, 8, 16, 256, 2517])
def test_public_moe_previous_parser_and_graph(fmt, tokens, monkeypatch):
    import importlib.util

    import torch

    if not torch.cuda.is_available():
        pytest.skip("requires gfx1250 GPU")
    from aiter.jit.utils.chip_info import get_gfx_runtime

    if get_gfx_runtime() != "gfx1250":
        pytest.skip("gfx1250 grouped MoE")
    fm = importlib.import_module("aiter.fused_moe")
    gemm = importlib.import_module("aiter.ops.flydsl.grouped_gemm_mxfp4")
    path = ROOT / "op_tests/flydsl_tests/test_flydsl_grouped_gemm.py"
    spec = importlib.util.spec_from_file_location("_scalar_config_reference", path)
    ref = importlib.util.module_from_spec(spec)
    default = torch.get_default_device()
    captured = []
    try:
        spec.loader.exec_module(ref)
        public = ref.fused_moe

        def capture(*args, **kwargs):
            kwargs["quant_dtype_a"] = (
                fm.dtypes.fp8 if fmt == "a8w4" else fm.dtypes.fp4x2
            )
            captured.append((args, kwargs))
            return public(*args, **kwargs)

        ref.fused_moe = capture
        output, reference, _, _ = ref._run_grouped_via_fused_moe(
            experts=129,
            tokens=tokens,
            topk=5,
            model_dim=6144,
            inter_dim=768,
            data_format=fmt,
            activation=fm.ActivationType.Swiglu,
            use_bias=True,
            seed=305,
        )
        assert ref._logits_diff(output, reference) < ref.LOGITS_DIFF_TOL
    finally:
        torch.set_default_device(default)
    args, kwargs = captured[-1]
    launch = gemm.flydsl_grouped_gemm_a8w4_masked
    calls = []

    def record(*a, **kw):
        calls.append(
            {k: v for k, v in kw.items() if isinstance(v, (int, float, bool, str))}
        )
        return launch(*a, **kw)

    monkeypatch.setattr(gemm, "flydsl_grouped_gemm_a8w4_masked", record)
    calls.clear()
    got = fm.fused_moe(*args, **kwargs)
    new_calls = calls.copy()
    calls.clear()
    with monkeypatch.context() as old:
        old.setattr(gm, "_as_int", previous)
        expected = fm.fused_moe(*args, **kwargs)
    assert len(calls) == len(new_calls) == 2
    assert calls == new_calls
    assert torch.equal(got, expected)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            replay = fm.fused_moe(*args, **kwargs)
        for _ in range(2):
            args[0].neg_()
            args[4].add_(1).remainder_(129)
            graph.replay()
            with monkeypatch.context() as old:
                old.setattr(gm, "_as_int", previous)
                expected = fm.fused_moe(*args, **kwargs)
            assert torch.equal(replay, expected)
    torch.cuda.current_stream().wait_stream(stream)

    # Environment overrides are still read after CSV parsing. Use the exact
    # tested tile geometry so this checks precedence, not another kernel choice.
    keys = ("tile_m", "tile_n", "tile_k", "m_warp", "n_warp", "num_buffers")
    env_names = ("TILE_M", "TILE_N", "TILE_K", "M_WARP", "N_WARP", "NUM_BUFFERS")
    with monkeypatch.context() as env:
        for stage in (0, 1):
            for key, name in zip(keys, env_names):
                if stage:
                    name = name + "2"
                env.setenv("AITER_TDM_" + name, str(new_calls[stage][key]))
        overridden = fm.fused_moe(*args, **kwargs)
        with monkeypatch.context() as old:
            old.setattr(gm, "_as_int", previous)
            baseline_override = fm.fused_moe(*args, **kwargs)
        assert torch.equal(overridden, baseline_override)
