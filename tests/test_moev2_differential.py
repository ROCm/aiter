# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""aiter.moev2 against the legacy MoE code.

The CPU test compares dispatch plans and log lines on every snapshot key a v2 path
claims. The GPU test runs tests/moe_dispatch/gpu_differential.py for each registered
path: same kernels on the same inputs, same outputs and log lines as legacy. It runs
only with AITER_MOEV2_GPU_DIFFERENTIAL=1, so CI (which runs this file) skips it.

NOTE: part of the aiter.moev2 migration (under active development, not graduated).
Please coordinate changes with the moev2 owners (@qli88). See aiter/moev2/README.md.

Golden snapshots: after an intended change to legacy MoE dispatch, run
`python tests/moe_dispatch/snapshot.py --update` and paste the changes it prints into
the PR.
"""

import json
import os
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


def _kernel(*args, **kwargs):
    pass


def _stage2(a2, sorted_ids, block_m=32):
    return [("stage2", partial(_kernel, a2, sorted_ids, block_m=block_m))]


def _compare(legacy, v2, legacy2=None):
    import torch
    from moe_dispatch.differential import compare_launches

    call = {"topk_ids": torch.zeros(4, 2, dtype=torch.int32), "expert_mask": None}
    out = torch.empty(1)
    return compare_launches(legacy, v2, call, out, out, legacy2)


@pytest.mark.parametrize("dtype", ["bfloat16", "float8_e4m3fn"])
def test_launch_floats_may_differ_by_rounding_only(dtype):
    import torch

    dtype = getattr(torch, dtype)
    ids = torch.arange(8, dtype=torch.int32)
    a2 = torch.linspace(0.5, 4, 8).to(dtype)
    # The next representable values up (positive, so the raw bits plus one).
    bits = torch.uint8 if a2.element_size() == 1 else torch.int16
    one_ulp = (a2.view(bits) + 1).view(dtype)
    assert not _compare(_stage2(a2, ids), _stage2(one_ulp, ids))
    far = (a2.float() * 1.5).to(dtype)
    problems = _compare(_stage2(a2, ids), _stage2(far, ids))
    assert list(problems) == [("stage2", 0)], problems
    # Legacy's own second run spreads as far: rounding of a non-repeatable kernel.
    assert not _compare(_stage2(a2, ids), _stage2(far, ids), _stage2(far, ids))


def test_launch_integers_and_arguments_stay_exact():
    import torch

    a2 = torch.ones(8, dtype=torch.bfloat16)
    ids = torch.arange(8, dtype=torch.int32)
    shifted = ids.clone()
    shifted[3] += 1
    spread = _stage2(a2, shifted)
    assert list(_compare(_stage2(a2, ids), _stage2(a2, shifted), spread)) == [
        ("stage2", 1)
    ]
    assert list(_compare(_stage2(a2, ids), _stage2(a2, ids, block_m=64))) == [
        ("stage2", "block_m")
    ]


def test_launch_non_finite_values_must_match():
    import torch

    ids = torch.arange(4, dtype=torch.int32)
    a2 = torch.tensor([1.0, 2.0, float("inf"), 4.0], dtype=torch.bfloat16)
    nan = a2.clone()
    nan[2] = float("nan")
    assert list(_compare(_stage2(a2, ids), _stage2(nan, ids))) == [("stage2", 0)]


def test_launch_equal_nans_pass_and_legacy_noise_excuses_no_new_ones():
    import torch

    ids = torch.arange(4, dtype=torch.int32)
    legacy = torch.tensor([1.0, 2.0, float("nan"), 4.0], dtype=torch.bfloat16)
    # Same NaN, and a rounding flip elsewhere so the tensors are not bitwise equal.
    flipped = legacy.clone()
    flipped[0] = (legacy[:1].view(torch.int16) + 1).view(torch.bfloat16)
    assert not _compare(_stage2(legacy, ids), _stage2(flipped, ids))
    # Legacy's second run has inf where the first has 3: v2 may match either run there,
    # but a NaN in another slot is still a difference.
    first = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.bfloat16)
    second = torch.tensor([1.0, 2.0, float("inf"), 4.0], dtype=torch.bfloat16)
    assert not _compare(_stage2(first, ids), _stage2(second, ids), _stage2(second, ids))
    new_nan = second.clone()
    new_nan[0] = float("nan")
    assert list(
        _compare(_stage2(first, ids), _stage2(new_nan, ids), _stage2(second, ids))
    ) == [("stage2", 0)]


def test_every_gpu_case_variant_is_defined():
    from moe_dispatch import gpu_differential

    variants = {
        case[9]
        for case in gpu_differential.cases("a8w8_pt", gfx="gfx942", cu_num=80)
        if len(case) > 9
    }
    assert variants <= set(gpu_differential._VARIANTS) | {"g1u0"}
    assert set(gpu_differential._RAISES) <= variants


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

    if os.environ.get("AITER_MOEV2_GPU_DIFFERENTIAL") != "1":
        pytest.skip(
            "manual: AITER_MOEV2_GPU_DIFFERENTIAL=1 (non-repeatable kernels, "
            "arch-specific cases)"
        )
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
