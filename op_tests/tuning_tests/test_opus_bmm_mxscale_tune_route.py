# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Minimal CPU regression for the MXFP8 BMM offline-tune route."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
import torch


_ROOT = Path(__file__).resolve().parents[2]
_TUNER = _ROOT / "csrc/opus_gemm/opus_bmm_mxscale_tune.py"


def _definitions(*names):
    tree = ast.parse(_TUNER.read_text())
    wanted = set(names)
    nodes = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef))
        and node.name in wanted
    ]
    assert {node.name for node in nodes} == wanted
    return ast.Module(nodes, type_ignores=[])


def _layout_namespace():
    def varied(shape, _k, _device):
        return torch.arange(
            torch.tensor(shape).prod().item(), dtype=torch.float32
        ).view(shape)

    def quant_token(x):
        g, m, k = x.shape
        scale = torch.ones((g, m, k // 128), dtype=torch.uint8)
        return x.clone(), scale, scale.float()

    def quant_block(x):
        g, n, k = x.shape
        scale = torch.ones((g, n // 128, k // 128), dtype=torch.uint8)
        return x.clone(), scale, scale.float()

    namespace = {
        "torch": torch,
        "_gen_varied": varied,
        "_quant_per_token_e8m0": quant_token,
        "_quant_block_e8m0": quant_block,
        "_workspace_numel": lambda *_args: 0,
        "run_torch": lambda x, w, _xs, _ws: torch.zeros(
            (x.shape[0], x.shape[1], w.shape[1])
        ),
        "opus_bmm": None,
    }
    exec(
        compile(
            _definitions("gen_bmm_mxscale_data", "run_bmm_mxscale_bench"),
            _TUNER,
            "exec",
        ),
        namespace,
    )
    return namespace


def test_tuner_preserves_production_layout_and_calls_public_bmm():
    namespace = _layout_namespace()
    data = namespace["gen_bmm_mxscale_data"](
        2, 3, 128, 256, 1, torch.float32, 8032, 1, device="cpu"
    )
    x, weight, out, x_scale, w_scale, workspace, ref = data

    assert x.shape == (2, 3, 256)
    assert out.shape == (3, 2, 128)
    assert out.stride() == (256, 128, 1)
    assert x_scale.shape == (2, 3, 2)
    assert weight.shape == (2, 128, 256)
    assert workspace is None
    assert ref.shape == out.shape
    captured = {}

    def fake_opus_bmm(public_x, public_weight, public_out, **kwargs):
        captured.update(kwargs)
        assert public_x is x and public_weight is weight
        assert public_out.shape == (2, 3, 128)
        assert public_out.transpose(0, 1).data_ptr() == out.data_ptr()
        return public_out

    namespace["opus_bmm"] = fake_opus_bmm
    assert namespace["run_bmm_mxscale_bench"](
        x, weight, out, x_scale, w_scale, workspace, 8032, 1
    ) is out
    assert captured["kid"] == 8032
    assert captured["layout"] == "mxscale_bmm"
    assert captured["x_scale"] is x_scale
    assert captured["w_scale"] is w_scale
    assert captured["split_k"] == 1
    assert captured["workspace"] is None


def test_tuner_shape_validation_is_deduplicated_and_gfx950_scoped():
    namespace = {"GROUP": 128}
    exec(compile(_definitions("_validate_tune_shapes"), _TUNER, "exec"), namespace)
    validate = namespace["_validate_tune_shapes"]

    assert validate([(2, 1, 128, 256), (2, 1, 128, 256)]) == [
        (2, 1, 128, 256)
    ]
    with pytest.raises(ValueError, match="positive"):
        validate([(0, 1, 128, 256)])
    with pytest.raises(ValueError, match="multiples of 128"):
        validate([(2, 1, 64, 256)])

    source = _TUNER.read_text()
    assert 'if gfx != "gfx950"' in source
    assert "MXFP8 BMM tuning is gfx950-only" in source
