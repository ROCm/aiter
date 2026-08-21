# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""CPU-only regression for the OPUS logical-2D GEMM tune runner."""

from __future__ import annotations

import ast
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn.functional as F


_ROOT = Path(__file__).resolve().parents[2]
_TUNER = _ROOT / "csrc" / "gemm_a16w16" / "gemm_a16w16_tune.py"


def _isolated_runner():
    """Load only run_opus_gemm_bf16, without importing the ROCm tuner."""
    tree = ast.parse(_TUNER.read_text())
    runner = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "run_opus_gemm_bf16"
    )
    namespace = {
        "F": F,
        "torch": torch,
        "_opus_max_delta_checked": set(),
    }
    exec(compile(ast.Module([runner], type_ignores=[]), _TUNER, "exec"), namespace)
    return namespace


def test_opus_gemm_tune_runner_uses_logical_2d_tensors_for_launch_and_check():
    namespace = _isolated_runner()
    inp = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    weight = torch.tensor([[2.0, 0.0], [0.0, 3.0], [1.0, 1.0]])
    bias = torch.tensor([0.5, -0.5, 1.0])
    out = torch.empty(2, 3)

    def fake_opus_gemm(x, w, y, *, bias, kid, split_k):
        assert x.dim() == w.dim() == y.dim() == 2
        assert kid == 1206
        assert split_k == 3
        y.copy_(F.linear(x, w, bias))

    namespace["_opus_gemm"] = fake_opus_gemm
    with patch.object(
        torch.cuda, "is_current_stream_capturing", return_value=False
    ):
        actual = namespace["run_opus_gemm_bf16"](
            inp,
            weight,
            out,
            bias=bias,
            kid=1206,
            splitK=3,
        )

    assert actual is out
    torch.testing.assert_close(out, F.linear(inp, weight, bias))
