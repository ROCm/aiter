# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""aiter.moev2 against the legacy MoE code.

The CPU test compares dispatch plans and log lines on every snapshot key a v2 path
claims. The GPU test runs tests/moe_dispatch/gpu_differential.py for each registered
path: same kernels on the same inputs, same outputs and log lines as legacy.
"""

import pathlib
import subprocess
import sys
from functools import partial
from types import SimpleNamespace

import pytest

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))


def test_whole_graph_launches_are_compared_without_a_stage1_record(monkeypatch):
    import torch
    from moe_dispatch import record
    from moe_dispatch.differential import compare_launches

    import aiter.fused_moe_registry as registry
    from aiter.fused_moe_registry import FusedMoeRequest

    def whole_graph(request, config):
        assert config == "unit_config"
        return request.hidden_states

    def request():
        return FusedMoeRequest(
            hidden_states=torch.ones(1, 2),
            w1=torch.ones(1, 2, 2),
            w2=torch.ones(1, 2, 2),
            topk_weight=torch.ones(1, 1),
            topk_ids=torch.zeros(1, 1, dtype=torch.int32),
        )

    legacy_request, v2_request = request(), request()
    monkeypatch.setitem(registry._IMPLEMENTATIONS, "differential_test", whole_graph)
    kernel_name = registry.make_fused_moe_impl_kernel_name(
        "differential_test", "unit_config"
    )
    legacy_impl = registry.resolve_fused_moe_impl(kernel_name)
    v2_impl = registry.resolve_fused_moe_impl(kernel_name)
    binding = record.normalize(legacy_impl)
    assert binding["config"] == "unit_config"
    assert binding["implementation"].endswith(".<locals>.whole_graph")
    call = {"topk_ids": legacy_request.topk_ids, "expert_mask": None}
    problems = compare_launches(
        [("whole_graph", partial(legacy_impl, legacy_request))],
        [("whole_graph", partial(v2_impl, v2_request))],
        call,
        torch.empty(1, 2),
        torch.empty(1, 2),
    )
    assert not problems


def test_whole_graph_gpu_case_only_runs_on_its_tuned_architecture():
    from moe_dispatch import gpu_differential

    whole_graph = (1, 4096, 192, 192, 8, "Silu", False, "bf16", False)
    assert whole_graph in gpu_differential.cases("a8w8_pt", gfx="gfx942", cu_num=80)
    assert whole_graph not in gpu_differential.cases(
        "a8w8_pt", gfx="gfx950", cu_num=304
    )


def test_perf_baseline_rejects_partial_token_results(monkeypatch):
    from moe_dispatch import perf_baseline

    result = SimpleNamespace(stdout='@@RESULT@@[{"token": 1, "us": 2.5}]\n', stderr="")
    monkeypatch.setattr(perf_baseline.subprocess, "run", lambda *a, **k: result)
    with pytest.raises(RuntimeError, match=r"missing=\['16'\]"):
        perf_baseline.run_case(perf_baseline.CASES[0], (1, 16))


def test_dispatch_plans_match_legacy():
    proc = subprocess.run(
        [sys.executable, str(_HERE / "moe_dispatch" / "differential.py"), "--check"],
        cwd=_HERE.parent,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, (
        proc.stdout[-6000:]
        + proc.stderr[-3000:]
        + "\nv2 must make the same choice as legacy for every key it claims."
    )


def _registered():
    from aiter.moev2 import paths

    return sorted(paths.registered())


@pytest.mark.parametrize("path_name", _registered())
def test_outputs_match_legacy(path_name):
    import torch

    if not torch.cuda.is_available():
        pytest.skip("needs a GPU")
    proc = subprocess.run(
        [
            sys.executable,
            str(_HERE / "moe_dispatch" / "gpu_differential.py"),
            "--path",
            path_name,
        ],
        cwd=_HERE.parent,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout[-6000:] + proc.stderr[-3000:]
