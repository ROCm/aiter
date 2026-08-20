# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""CPU-only contract tests for the OPUS MXFP8 BMM offline tuner.

Only selected definitions are compiled from the tuner source. This avoids
importing ROCm bindings, enumerating GPUs, building JIT modules, or launching a
kernel while still exercising the actual data-layout and preprocessing code.
"""

from __future__ import annotations

import argparse
import ast
import logging
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar
from unittest.mock import patch

import pandas as pd
import pytest
import torch


_ROOT = Path(__file__).resolve().parents[2]
_TUNER = _ROOT / "csrc" / "opus_gemm" / "opus_bmm_mxscale_tune.py"


def _selected_definitions(*names: str):
    tree = ast.parse(_TUNER.read_text())
    wanted = set(names)
    nodes = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef))
        and node.name in wanted
    ]
    assert {node.name for node in nodes} == wanted
    return ast.Module(body=nodes, type_ignores=[])


def _layout_namespace():
    def gen_varied(shape, _k, _device):
        numel = torch.tensor(shape).prod().item()
        return torch.arange(numel, dtype=torch.float32).view(shape)

    def quant_per_token(x):
        g, m, k = x.shape
        scale = torch.ones((g, m, k // 128), dtype=torch.uint8)
        return x.clone(), scale, scale.to(torch.float32)

    def quant_block(weight):
        g, n, k = weight.shape
        scale = torch.ones((g, n // 128, k // 128), dtype=torch.uint8)
        return weight.clone(), scale, scale.to(torch.float32)

    def reference(x, weight, _x_scale, _w_scale):
        return torch.zeros((x.shape[0], x.shape[1], weight.shape[1]))

    namespace = {
        "torch": torch,
        "_gen_varied": gen_varied,
        "_quant_per_token_e8m0": quant_per_token,
        "_quant_block_e8m0": quant_block,
        "run_torch": reference,
        "_workspace_numel": lambda *_args: 0,
        "opus_bmm": None,
    }
    module = _selected_definitions(
        "gen_bmm_mxscale_data",
        "run_bmm_mxscale_bench",
    )
    exec(compile(module, _TUNER, "exec"), namespace)
    return namespace


class _StubTunerCommon:
    ARG_DEFAULTS = {
        "verbose": False,
        "tune_file": "",
        "untune_file": "",
        "errRatio": 0.05,
        "batch": 100,
        "profile_file": "",
        "timeout": 1800,
        "warmup": 5,
        "iters": 101,
        "min_improvement_pct": 3.0,
        "sort": True,
    }
    INVALID_TIME = -1

    def __init__(self, name, keys, results, description=None):
        self.name = name
        self.keys = keys
        self.columns = keys + results
        self.parser = argparse.ArgumentParser(description=description)
        defaults = self.get_arg_defaults()
        self.parser.add_argument("--verbose", "-v", action="store_true")
        self.parser.add_argument("-i", "--untune_file", default=defaults["untune_file"])
        self.parser.add_argument("-o", "--tune_file", default=defaults["tune_file"])
        self.parser.add_argument("--mp", type=int, default=0)
        self.parser.add_argument("-k", "--splitK", action="store_true")
        self.parser.add_argument("--shape_grouped", action="store_true")
        self.parser.add_argument("--sort", action="store_true")
        self.parser.add_argument("--errRatio", type=float, default=defaults["errRatio"])
        self.parser.add_argument("--batch", type=int, default=defaults["batch"])
        self.parser.add_argument("--all", action="store_true")
        self.parser.add_argument("-o2", "--profile_file", default="")
        self.parser.add_argument("--warmup", type=int, default=defaults["warmup"])
        self.parser.add_argument("--iters", type=int, default=defaults["iters"])
        self.parser.add_argument("--timeout", type=int, default=defaults["timeout"])
        self.parser.add_argument(
            "--run_config", nargs="?", const=True, default=False
        )
        self.parser.add_argument("--compare", action="store_true")
        self.parser.add_argument("--update_improved", action="store_true")
        self.parser.add_argument(
            "--min_improvement_pct",
            type=float,
            default=defaults["min_improvement_pct"],
        )
        self._setup_specific_arguments()

    def get_arg_defaults(self):
        return self.ARG_DEFAULTS.copy()

    def parse_args(self):
        args = self.parser.parse_args()
        if args.update_improved and not args.compare:
            self.parser.error("--update_improved requires --compare")
        return args

    def get_gfx(self):
        raise AssertionError("test must provide a synthetic architecture")

    def get_tuned_gemm_list(self, path):
        if os.path.exists(path):
            return pd.read_csv(path)
        return pd.DataFrame(columns=self.columns)


class _StubGemmCommonTuner(_StubTunerCommon):
    pass


def _tuner_namespace():
    namespace = {
        "Any": Any,
        "ClassVar": ClassVar,
        "DEFAULT_OUT": "/mock/dsv4_bmm_mxscale_retuned.csv",
        "SHIPPED_CSV": "/mock/shipped_mxscale.csv",
        "GROUP": 128,
        "GemmCommonTuner": _StubGemmCommonTuner,
        "TunerCommon": _StubTunerCommon,
        "_CODEGEN_BMM": {},
        "dtypes": SimpleNamespace(bf16=torch.bfloat16),
        "logger": logging.getLogger("opus-bmm-mxscale-tuner-test"),
        "mp_tuner": None,
        "os": os,
        "pd": pd,
    }
    module = _selected_definitions(
        "_read_shape_csv",
        "_validate_tune_shapes",
        "OpusBmmMxscaleTuner",
    )
    exec(compile(module, _TUNER, "exec"), namespace)
    return namespace


_TUNER_NS = _tuner_namespace()
OpusBmmMxscaleTuner = _TUNER_NS["OpusBmmMxscaleTuner"]


def _parse(tuner, argv):
    with patch.object(sys, "argv", ["opus_bmm_mxscale_tune.py", *argv]):
        return tuner.parse_args()


def test_generated_output_restores_production_mmajor_raw_layout():
    namespace = _layout_namespace()
    batch, m, n, k = 2, 3, 128, 256
    data = namespace["gen_bmm_mxscale_data"](
        batch,
        m,
        n,
        k,
        1,
        torch.float32,
        8032,
        1,
        device="cpu",
    )
    x, weight, out, x_scale, w_scale, workspace, ref = data

    assert x.shape == (batch, m, k)
    assert x.is_contiguous()
    assert x.stride() == (m * k, k, 1)
    assert x.transpose(0, 1).stride() == (k, m * k, 1)

    assert out.shape == (m, batch, n)
    assert out.is_contiguous()
    assert out.stride() == (batch * n, n, 1)

    groups_k = k // 128
    assert x_scale.is_contiguous()
    assert x_scale.stride() == (m * groups_k, groups_k, 1)
    assert x_scale.transpose(0, 1).stride() == (groups_k, m * groups_k, 1)
    assert weight.is_contiguous()
    assert w_scale.is_contiguous()
    assert workspace is None
    assert ref.shape == out.shape

    captured = {}

    def fake_opus_bmm(public_x, public_weight, public_out, **kwargs):
        captured.update(kwargs)
        assert public_x is x
        assert public_weight is weight
        assert kwargs["x_scale"] is x_scale
        assert kwargs["w_scale"] is w_scale
        assert public_out.shape == (batch, m, n)
        raw_out = public_out.transpose(0, 1)
        assert raw_out.data_ptr() == out.data_ptr()
        assert raw_out.is_contiguous()
        assert raw_out.stride() == (batch * n, n, 1)
        public_out.zero_()
        return public_out

    namespace["opus_bmm"] = fake_opus_bmm
    actual = namespace["run_bmm_mxscale_bench"](
        x, weight, out, x_scale, w_scale, workspace, 8032, 1
    )
    assert actual is out
    assert captured["kid"] == 8032
    assert captured["layout"] == "mxscale_bmm"
    assert captured["split_k"] == 1


def test_apply_implies_all_and_does_not_filter_every_shipped_shape(
    tmp_path, monkeypatch
):
    shipped = tmp_path / "shipped.csv"
    pd.DataFrame(
        [
            {
                "gfx": "gfx950",
                "b": 2,
                "m": 3,
                "n": 128,
                "k": 256,
                "libtype": "opus",
                "kernelId": 8032,
                "splitK": 1,
                "us": 1.0,
                "kernelName": "mock",
                "tflops": 1.0,
                "bw": 1.0,
                "errRatio": 0.0,
            }
        ]
    ).to_csv(shipped, index=False)
    monkeypatch.setitem(_TUNER_NS, "SHIPPED_CSV", str(shipped))

    tuner = OpusBmmMxscaleTuner()
    tuner.get_gfx = lambda: "gfx950"
    args = _parse(tuner, ["--apply"])
    assert not args.all

    tuner.pre_process(args)

    assert args.all
    assert args.tune_file == str(shipped)
    assert tuner.untunedf[["b", "m", "n", "k"]].to_dict("records") == [
        {"b": 2, "m": 3, "n": 128, "k": 256}
    ]


@pytest.mark.parametrize(
    "shape, message",
    [
        ((0, 1, 128, 128), "positive"),
        ((2, 1, 64, 128), "multiples of 128"),
        ((2, 1, 128, 192), "multiples of 128"),
    ],
)
def test_shape_validation_rejects_values_outside_production_contract(
    shape, message
):
    with pytest.raises(ValueError, match=message):
        _TUNER_NS["_validate_tune_shapes"]([shape])


def test_shape_validation_deduplicates_valid_rows():
    validate = _TUNER_NS["_validate_tune_shapes"]
    assert validate([(2, 1, 128, 256), (2, 1, 128, 256)]) == [
        (2, 1, 128, 256)
    ]


def test_preprocess_rejects_non_gfx950_before_creating_tasks():
    tuner = OpusBmmMxscaleTuner()
    tuner.get_gfx = lambda: "gfx942"
    args = _parse(tuner, ["-g", "2", "-m", "3", "-n", "128", "-k", "256"])

    with pytest.raises(RuntimeError, match="gfx950-only"):
        tuner.pre_process(args)


def test_partial_manual_shape_cli_is_rejected():
    tuner = OpusBmmMxscaleTuner()
    tuner.get_gfx = lambda: "gfx950"
    args = _parse(tuner, ["-g", "2"])

    with pytest.raises(ValueError, match="must be provided together"):
        tuner.pre_process(args)


def test_missing_explicit_shape_csv_does_not_fall_back_to_shipped(tmp_path):
    tuner = OpusBmmMxscaleTuner()
    tuner.get_gfx = lambda: "gfx950"
    missing = tmp_path / "missing.csv"
    args = _parse(tuner, ["-i", str(missing)])

    with pytest.raises(FileNotFoundError, match="does not exist"):
        tuner.pre_process(args)


@pytest.mark.parametrize("option", ["--run_config", "--compare"])
def test_unimplemented_production_benchmark_modes_fail_explicitly(option):
    tuner = OpusBmmMxscaleTuner()
    with pytest.raises(SystemExit):
        _parse(tuner, [option])
