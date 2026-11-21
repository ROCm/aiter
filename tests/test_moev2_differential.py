# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""aiter.moev2 against the legacy MoE code.

The CPU test compares dispatch plans and log lines on every snapshot key a v2 path
claims. The GPU test runs tests/moe_dispatch/gpu_differential.py for each registered
path: same kernels on the same inputs, same outputs and log lines as legacy.
"""

import json
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


@pytest.fixture
def perf_check(monkeypatch, tmp_path, capsys):
    """Run `perf_baseline.main` on a fake gfx942 with baselines under tmp_path; the
    measurement returns `measured` and records which cases it was asked for."""
    from moe_dispatch import perf_baseline

    measured, asked = {}, []

    def measure(gfx, names=None):
        asked.append(names)
        return dict(measured)

    monkeypatch.setattr(perf_baseline, "_gfx", lambda: "gfx942")
    monkeypatch.setattr(perf_baseline, "GOLDEN_DIR", tmp_path)
    monkeypatch.setattr(perf_baseline, "measure", measure)

    def check(*argv, baseline=None):
        if baseline is not None:
            (tmp_path / "perf_gfx942.json").write_text(
                json.dumps({"gfx": "gfx942", "us": baseline})
            )
        code = perf_baseline.main(["--check", *argv])
        return code, capsys.readouterr().out

    return SimpleNamespace(check=check, measured=measured, asked=asked)


def test_perf_check_rejects_requested_cases_without_a_baseline(perf_check):
    code, out = perf_check.check(
        "--case", "qwen3_235b_tp8_bf16", baseline={"dsv3_tp8_pt": {"1": 10.0}}
    )
    assert code == 1 and perf_check.asked == []
    assert "['qwen3_235b_tp8_bf16']" in out
    assert "--record --case qwen3_235b_tp8_bf16" in out


def test_perf_baseline_refuses_an_arch_without_cases(
    perf_check, monkeypatch, tmp_path, capsys
):
    from moe_dispatch import perf_baseline

    assert not any("gfx1250" in case[-1] for case in perf_baseline.CASES)
    monkeypatch.setattr(perf_baseline, "_gfx", lambda: "gfx1250")
    for mode in ("--record", "--check"):
        assert perf_baseline.main([mode]) == 1
        assert "no perf case runs on gfx1250" in capsys.readouterr().out
    assert perf_check.asked == [] and not list(tmp_path.iterdir())

    monkeypatch.setattr(perf_baseline, "_gfx", lambda: "gfx942")
    assert perf_baseline.main(["--record", "--case", "dsv3_tp8_a4w4"]) == 1
    assert "do not run on gfx942" in capsys.readouterr().out
    assert perf_check.asked == [] and not list(tmp_path.iterdir())


def test_perf_check_rejects_a_missing_file_and_other_arch_cases(perf_check):
    code, out = perf_check.check("--case", "dsv3_tp8_pt")
    assert code == 1 and "record one first" in out
    code, out = perf_check.check("--case", "dsv3_tp8_a4w4", baseline={})
    assert code == 1 and "do not run on gfx942" in out
    assert perf_check.asked == []


def test_perf_check_without_cases_needs_every_case_of_the_arch(perf_check):
    from moe_dispatch import perf_baseline

    gfx942 = [c[0] for c in perf_baseline.CASES if "gfx942" in c[-1]]
    code, out = perf_check.check(baseline={gfx942[0]: {"1": 10.0}})
    assert code == 1 and perf_check.asked == []
    assert str(gfx942[1:]) in out


def test_perf_check_compares_the_requested_cases(perf_check):
    baseline = {"dsv3_tp8_pt": {"1": 10.0}, "qwen3_235b_tp8_bf16": {"1": 10.0}}
    perf_check.measured["qwen3_235b_tp8_bf16"] = {"1": 10.5}
    code, _ = perf_check.check("--case", "qwen3_235b_tp8_bf16", baseline=baseline)
    assert code == 0 and perf_check.asked == [["qwen3_235b_tp8_bf16"]]
    perf_check.measured["qwen3_235b_tp8_bf16"] = {"error": "boom"}
    code, out = perf_check.check("--case", "qwen3_235b_tp8_bf16")
    assert code == 1 and "MISSING qwen3_235b_tp8_bf16 M=1" in out


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


def _path_supports_device(path_name, gfx=None):
    from aiter.jit.utils.chip_info import get_gfx_runtime
    from aiter.moev2 import paths

    path = paths.registered()[path_name]
    gfx = get_gfx_runtime() if gfx is None else gfx
    return not path.supported_archs or gfx in path.supported_archs


def test_gpu_differential_architecture_filter():
    assert _path_supports_device("a8w8_pt", "gfx950")
    assert not _path_supports_device("a8w8_pt", "gfx1250")


@pytest.mark.parametrize("path_name", _registered())
def test_outputs_match_legacy(path_name):
    import torch

    if not torch.cuda.is_available():
        pytest.skip("needs a GPU")
    if not _path_supports_device(path_name):
        pytest.skip(f"{path_name} does not support this GPU architecture")
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
