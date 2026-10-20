# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""MXFP4 tuner tests: lifecycle, configuration snapshot and timeouts.

CPU only; devices and measurements are simulated.
"""

import argparse
import importlib
import multiprocessing
import os
import signal
import subprocess
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from multiprocessing.connection import Connection
from multiprocessing.context import BaseContext
from multiprocessing.process import BaseProcess
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest import mock

import pandas as pd
import pytest
import torch

from op_tests.tuning_tests._mxfp4_test_process import (
    require_gpu_hidden_process,
    use_default_device,
)

require_gpu_hidden_process()


def import_moe_tuner() -> "ModuleType":
    run = subprocess.run
    set_default_device = torch.set_default_device

    def discover(
        command: "list[str]", *args: "object", **kwargs: "object"
    ) -> "subprocess.CompletedProcess[str]":
        if isinstance(command, (list, tuple)) and Path(command[0]).name in (
            "rocminfo",
            "hipinfo",
        ):
            output = (
                "Agent 1\n  Name: gfx950\n  Device Type: GPU\n"
                "  Compute Unit: 256\n  ASIC Revision: 1\n"
            )
            return subprocess.CompletedProcess(command, 0, stdout=output, stderr="")
        return run(command, *args, **kwargs)

    # Restored last, after the set_default_device patch is gone.
    with (
        use_default_device("cpu"),
        mock.patch.dict(
            os.environ,
            {
                "ROCR_VISIBLE_DEVICES": "",
                "HIP_VISIBLE_DEVICES": "",
                "CUDA_VISIBLE_DEVICES": "",
                "GPU_ARCHS": "gfx950",
                "CU_NUM": "256",
            },
        ),
        mock.patch.object(subprocess, "run", side_effect=discover),
        mock.patch.object(
            torch,
            "set_default_device",
            side_effect=lambda device: set_default_device("cpu"),
        ),
    ):
        tuner = importlib.import_module(
            "csrc.ck_gemm_moe_2stages_codegen.gemm_moe_tune"
        )
    # Synthetic import discovery must not become runtime evidence.
    for name in ("chip_info", "aiter.jit.utils.chip_info"):
        module = sys.modules.get(name)
        if module is not None:
            for function in (
                "_gpu_info_output",
                "_detect_native",
                "get_gfx_runtime",
                "get_gfx",
                "get_gfx_custom_op_core",
                "_rocminfo_gpu_agents",
                "get_gfx_list",
            ):
                clear = getattr(getattr(module, function), "cache_clear", None)
                if clear is not None:
                    clear()
    return tuner


moe = import_moe_tuner()

KEYS = [
    "gfx",
    "cu_num",
    "token",
    "model_dim",
    "inter_dim",
    "expert",
    "topk",
    "act_type",
    "dtype",
    "q_dtype_a",
    "q_dtype_w",
    "q_type",
    "use_g1u1",
    "doweight_stage1",
]
RESULTS = [
    "block_m",
    "ksplit",
    "us1",
    "kernelName1",
    "err1",
    "us2",
    "kernelName2",
    "err2",
    "us",
    "run_1stage",
    "xbf16",
    "flat",
    "tflops",
    "bw",
]


def block_real_work(monkeypatch: "pytest.MonkeyPatch") -> "None":
    # Spawned workers do not inherit mocks; multi-slot tests must stub scheduling.
    def forbidden(*args: "object", **kwargs: "object") -> "None":
        pytest.fail("controlled lifecycle test reached real GPU work")

    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "")
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setattr(multiprocessing, "get_context", lambda *args: None)
    monkeypatch.setattr(BaseProcess, "start", forbidden)
    monkeypatch.setattr(moe, "_run_shapes_isolated", forbidden)
    monkeypatch.setattr(moe.Mxfp4FlydslTuner, "_prepare_case", forbidden)
    monkeypatch.setattr(moe.Mxfp4FlydslTuner, "_run_candidate", forbidden)
    monkeypatch.setattr(moe.Mxfp4FlydslTuner, "_port_e2e", forbidden)
    monkeypatch.setattr(moe, "mp_tuner", forbidden)
    for name in ("generate_data", "generate_data_1stage", "generate_data_2stages"):
        monkeypatch.setattr(moe.FmoeTuner, name, forbidden)
    monkeypatch.setattr("aiter.test_common.run_perftest", forbidden)
    monkeypatch.setattr(torch.cuda, "_lazy_init", forbidden)
    monkeypatch.setattr(torch.cuda, "synchronize", forbidden)
    monkeypatch.setattr(torch.cuda, "Event", forbidden)
    monkeypatch.setattr(torch.cuda, "CUDAGraph", forbidden)
    for name in (
        "empty",
        "zeros",
        "ones",
        "tensor",
        "arange",
        "rand",
        "randn",
        "randint",
    ):
        monkeypatch.setattr(torch, name, forbidden)


@pytest.fixture(autouse=True)
def prohibit_real_shape_workers_and_gpu_data(
    monkeypatch: "pytest.MonkeyPatch",
) -> "Iterator[None]":
    block_real_work(monkeypatch)
    with use_default_device("cpu"):
        yield


def shape(token: "int" = 4, **updates: "object") -> "dict[str, object]":
    row = dict(
        zip(
            KEYS,
            [
                "gfx950",
                256,
                token,
                3072,
                512,
                128,
                4,
                "ActivationType.Swiglu",
                "torch.bfloat16",
                "torch.float4_e2m1fn_x2",
                "torch.float4_e2m1fn_x2",
                "QuantType.per_1x32",
                1,
                0,
            ],
        )
    )
    row.update(updates)
    return row


def tuner_args(
    tuner: "moe.Mxfp4FlydslTuner", source: "Path", output: "Path", *options: "object"
) -> "argparse.Namespace":
    return tuner.parser.parse_args(
        [
            "-i",
            str(source),
            "-o",
            str(output),
            "--mp",
            "1",
            "--timeout",
            "0",
            "--sort",
            "True",
            *options,
        ]
    )


@contextmanager
def devices(*identities: "tuple[str, int]", current: "int" = 0) -> "Iterator[None]":
    properties = [
        SimpleNamespace(gcnArchName=gfx, multi_processor_count=cu)
        for gfx, cu in identities
    ]
    with (
        mock.patch.object(torch.cuda, "device_count", return_value=len(properties)),
        mock.patch.object(torch.cuda, "current_device", return_value=current),
        mock.patch.object(
            torch.cuda, "get_device_properties", side_effect=lambda i: properties[i]
        ),
        mock.patch.object(torch.cuda, "set_device"),
        mock.patch.object(
            moe.TunerCommon, "get_gfx", return_value=identities[current][0]
        ),
    ):
        yield


def controlled_mxfp4(
    tuner: "moe.Mxfp4FlydslTuner",
    monkeypatch: "pytest.MonkeyPatch",
    attempts: "list[dict[str, object]]",
) -> "None":
    def candidates(
        row: "dict[str, object]", full_search: "bool" = False
    ) -> "list[dict[str, object]]":
        return [tuner._candidate_row(row, 16, "g1-controlled", "g2-controlled")]

    def measure(
        row: "dict[str, object]",
        candidate: "dict[str, object]",
        args: "SimpleNamespace",
        **options: "object",
    ) -> "float":
        attempts.append(dict(row))
        candidate.update(us=10.25, us1=10.25, us2=0, err1="0.0%", err2="0.0%")
        return 10.25

    monkeypatch.setattr(tuner, "_candidate_rows", candidates)
    monkeypatch.setattr(tuner, "_run_candidate", measure)


@pytest.mark.parametrize("same_file", [False, True])
def test_all_preserves_gate_metadata_through_winner_summary(
    tmp_path: "Path", monkeypatch: "pytest.MonkeyPatch", same_file: "bool"
) -> "None":
    source = tmp_path / "input.csv"
    output = source if same_file else tmp_path / "winner.csv"
    pd.DataFrame([shape(gate_mode="GateMode.SEPARATED")]).to_csv(source, index=False)
    attempts = []
    with devices(("gfx950", 256)):
        tuner = moe.Mxfp4FlydslTuner("mxfp4", KEYS, RESULTS)
        controlled_mxfp4(tuner, monkeypatch, attempts)
        tuner.run(tuner_args(tuner, source, output, "--all"))
    assert attempts[0]["gate_mode"] == "GateMode.SEPARATED"
    winner = pd.read_csv(output)
    assert winner.loc[0, "us"] == 10.25
    assert "gate_mode" not in winner.columns
    assert tuner.failed.empty and tuner.remain_untuned.empty


def test_retune_replaces_duplicate_old_winners(
    tmp_path: "Path", monkeypatch: "pytest.MonkeyPatch"
) -> "None":
    source, output = tmp_path / "input.csv", tmp_path / "winner.csv"
    pd.DataFrame([shape()]).to_csv(source, index=False)
    with devices(("gfx950", 256)):
        tuner = moe.Mxfp4FlydslTuner("mxfp4", KEYS, RESULTS)
        old = tuner._candidate_row(
            shape(), 16, "flydsl_mxmoe_g1_old", "flydsl_mxmoe_g2_old"
        )
        pd.DataFrame([dict(old, us=20), dict(old, us=21)]).to_csv(output, index=False)
        controlled_mxfp4(tuner, monkeypatch, [])
        tuner.run(tuner_args(tuner, source, output, "--all"))
    winner = pd.read_csv(output)
    assert winner["us"].tolist() == [10.25]
    assert winner["kernelName1"].tolist() == ["g1-controlled"]


def test_same_file_all_without_device_identity_retains_gate_metadata(
    tmp_path: "Path", monkeypatch: "pytest.MonkeyPatch"
) -> "None":
    file = tmp_path / "same.csv"
    pd.DataFrame([shape(gate_mode="GateMode.SEPARATED")]).drop(
        columns=["gfx", "cu_num"]
    ).to_csv(file, index=False)
    attempts = []
    with devices(("gfx950", 256)):
        tuner = moe.Mxfp4FlydslTuner("mxfp4", KEYS, RESULTS)
        controlled_mxfp4(tuner, monkeypatch, attempts)
        tuner.run(tuner_args(tuner, file, file, "--all"))
    assert attempts[0]["gate_mode"] == "GateMode.SEPARATED"
    assert pd.read_csv(file).us.tolist() == [10.25]


@pytest.mark.parametrize("empty_input", [False, True])
def test_shared_csv_preserves_valid_rows_and_normalizes_legacy_defaults(
    tmp_path: "Path", monkeypatch: "pytest.MonkeyPatch", empty_input: "bool"
) -> "None":
    source, output = tmp_path / "input.csv", tmp_path / "shared.csv"
    with devices(("gfx950", 256)):
        tuner = moe.Mxfp4FlydslTuner("mxfp4", KEYS, RESULTS)
        legacy = tuner._candidate_row(
            shape(64, cu_num=304), 32, "ordinary-g1", "ordinary-g2"
        )
        legacy.update(
            gfx=None, us=15.75, us1=6.5, us2=9.25, nt=1, xbf16=None, flat=None
        )
        explicit = dict(legacy, token=128, gfx="gfx1250", cu_num=256, nt=0)
        fallback = dict(legacy, token=32, gfx="gfx942", _tag="flydsl_fallback")
        pd.DataFrame([legacy, explicit, fallback]).to_csv(output, index=False)
        pd.DataFrame([] if empty_input else [shape()], columns=KEYS).to_csv(
            source, index=False
        )
        attempts = []
        controlled_mxfp4(tuner, monkeypatch, attempts)
        tuner.run(tuner_args(tuner, source, output))
    csv = pd.read_csv(output)
    assert list(csv.token) == ([128, 64] if empty_input else [128, 64, 4])
    assert csv.loc[csv.token == 64, "gfx"].item() == "gfx942"
    assert csv.loc[csv.token == 128, "gfx"].item() == "gfx1250"
    assert csv.loc[csv.token == 64, "nt"].item() == 1
    assert csv.loc[csv.token == 64, ["us1", "us2", "us"]].values.tolist() == [
        [6.5, 9.25, 15.75]
    ]
    assert csv[["xbf16", "flat"]].notna().all().all()
    assert csv["_tag"].isna().all() and "gate_mode" not in csv.columns


@pytest.mark.parametrize("empty_input", [False, True])
def test_shared_csv_keeps_every_row_the_runtime_still_loads(
    tmp_path: "Path", monkeypatch: "pytest.MonkeyPatch", empty_input: "bool"
) -> "None":
    source, output = tmp_path / "input.csv", tmp_path / "shared.csv"
    with devices(("gfx950", 256)):
        tuner = moe.Mxfp4FlydslTuner("mxfp4", KEYS, RESULTS)
        row = tuner._candidate_row(shape(64), 32, "live-g1", "live-g2")
        row.update(us=15.75)
        rows = [
            dict(row, token=8, _tag="0"),
            dict(row, token=16, _tag="glm5_ptpc", comm_fused_configs="cfg-16"),
            dict(row, token=32, us=None, us1=None, us2=None),
            dict(row, token=64, _tag="flydsl_fallback", kernelName1="stale-g1"),
            dict(row, token=64),
            dict(row, token=128, _tag="flydsl_fallback"),
            # A later MXFP4 duplicate is unreachable; a non-MXFP4 one may
            # still serve the runtime activation fallback.
            dict(row, token=256, kernelName1="flydsl_mxmoe_g1_first"),
            dict(row, token=256, kernelName1="flydsl_mxmoe_g1_second"),
            dict(row, token=512, kernelName1="flydsl_mxmoe_g1_first"),
            dict(row, token=512, kernelName1="fmoe_1stage_fallback", kernelName2=""),
        ]
        pd.DataFrame(rows).to_csv(output, index=False)
        pd.DataFrame([] if empty_input else [shape()], columns=KEYS).to_csv(
            source, index=False
        )
        controlled_mxfp4(tuner, monkeypatch, [])
        tuner.run(tuner_args(tuner, source, output))
    csv = pd.read_csv(output, dtype={"_tag": str})
    assert csv.loc[csv.token == 512, "kernelName1"].tolist() == [
        "flydsl_mxmoe_g1_first",
        "fmoe_1stage_fallback",
    ]
    csv = csv[csv.token != 512].set_index("token")
    tuned = [8, 16, 32, 64, 256]
    assert sorted(csv.index) == (tuned if empty_input else [4, *tuned])
    assert csv.loc[8, "_tag"] == "0" and csv.loc[16, "_tag"] == "glm5_ptpc"
    assert csv.loc[16, "comm_fused_configs"] == "cfg-16"
    assert pd.isna(csv.loc[32, "us"]) and csv.loc[32, "kernelName1"] == "live-g1"
    assert csv.loc[64, "kernelName1"] == "live-g1" and pd.isna(csv.loc[64, "_tag"])
    assert csv.loc[256, "kernelName1"] == "flydsl_mxmoe_g1_first"


def test_retune_updates_the_row_runtime_loads_and_keeps_its_tag(
    tmp_path: "Path", monkeypatch: "pytest.MonkeyPatch"
) -> "None":
    source, output = tmp_path / "input.csv", tmp_path / "winner.csv"
    pd.DataFrame([shape()]).to_csv(source, index=False)
    with devices(("gfx950", 256)):
        tuner = moe.Mxfp4FlydslTuner("mxfp4", KEYS, RESULTS)
        old = tuner._candidate_row(
            shape(), 16, "flydsl_mxmoe_g1_old", "flydsl_mxmoe_g2_old"
        )
        pd.DataFrame(
            [
                dict(old, us=20, _tag="first"),
                dict(old, us=21, _tag="second"),
                dict(old, us=22, _tag="first"),
            ]
        ).to_csv(output, index=False)
        controlled_mxfp4(tuner, monkeypatch, [])
        tuner.run(tuner_args(tuner, source, output, "--all"))
    winner = pd.read_csv(output)
    assert winner[["us", "kernelName1", "_tag"]].values.tolist() == [
        [10.25, "g1-controlled", "first"]
    ]


@pytest.mark.parametrize("same_file", [False, True])
def test_failed_retune_keeps_old_winner_behind_a_placeholder_row(
    tmp_path: "Path", monkeypatch: "pytest.MonkeyPatch", same_file: "bool"
) -> "None":
    source = tmp_path / "input.csv"
    output = source if same_file else tmp_path / "winner.csv"
    with devices(("gfx950", 256)):
        tuner = moe.Mxfp4FlydslTuner("mxfp4", KEYS, RESULTS)
        old = tuner._candidate_row(
            shape(), 16, "flydsl_mxmoe_g1_old", "flydsl_mxmoe_g2_old"
        )
        pd.DataFrame([shape(), dict(old, us=20)]).to_csv(output, index=False)
        if not same_file:
            pd.DataFrame([shape()]).to_csv(source, index=False)

        def measure(*args: "object", **kwargs: "object") -> "float":
            return float("nan")

        controlled_mxfp4(tuner, monkeypatch, [])
        monkeypatch.setattr(tuner, "_run_candidate", measure)
        with pytest.raises(SystemExit):
            tuner.run(tuner_args(tuner, source, output, "--all"))
    csv = pd.read_csv(output)
    assert "flydsl_mxmoe_g1_old" in csv.kernelName1.tolist()


def test_compare_merge_keeps_rows_the_runtime_fallback_can_load(
    tmp_path: "Path",
) -> "None":
    base, candidate = tmp_path / "base.csv", tmp_path / "candidate.csv"
    with devices(("gfx950", 256)):
        tuner = moe.Mxfp4FlydslTuner("mxfp4", KEYS, RESULTS)
        old = tuner._candidate_row(shape(), 16, "flydsl_mxmoe_g1_old", "g2")
        improved = tuner._candidate_row(shape(8), 16, "flydsl_mxmoe_g1_new", "g2")
        pd.DataFrame(
            [
                dict(old, us=20),
                dict(old, us=30, kernelName1="fmoe_1stage_fallback"),
                dict(improved, us=20),
            ]
        ).to_csv(base, index=False)
        pd.DataFrame([dict(improved, us=10)]).to_csv(candidate, index=False)
        comparison = pd.DataFrame([dict(shape(8), update=True)])
        merged = tuner._merge_compare_filtered_results(
            str(base), str(candidate), comparison
        )
    assert merged[["token", "kernelName1"]].values.tolist() == [
        [4, "flydsl_mxmoe_g1_old"],
        [4, "fmoe_1stage_fallback"],
        [8, "flydsl_mxmoe_g1_new"],
    ]


def test_same_file_drops_only_untuned_input_rows(
    tmp_path: "Path", monkeypatch: "pytest.MonkeyPatch"
) -> "None":
    file = tmp_path / "same.csv"
    with devices(("gfx950", 256)):
        tuner = moe.Mxfp4FlydslTuner("mxfp4", KEYS, RESULTS)
        untimed = tuner._candidate_row(shape(32), 16, "live-g1", "live-g2")
        untimed.update(us=None)
        pd.DataFrame([shape(), untimed]).to_csv(file, index=False)
        attempts = []
        controlled_mxfp4(tuner, monkeypatch, attempts)
        tuner.run(tuner_args(tuner, file, file))
    assert [row["token"] for row in attempts] == [4]
    csv = pd.read_csv(file).set_index("token")
    assert sorted(csv.index) == [4, 32]
    assert csv.loc[4, "us"] == 10.25 and csv.loc[32, "kernelName1"] == "live-g1"


def test_separate_output_keeps_rows_without_kernels(
    tmp_path: "Path", monkeypatch: "pytest.MonkeyPatch"
) -> "None":
    source, output = tmp_path / "input.csv", tmp_path / "manifest.csv"
    pd.DataFrame([shape()]).to_csv(source, index=False)
    pd.DataFrame([shape(32), shape(64)]).to_csv(output, index=False)
    with devices(("gfx950", 256)):
        tuner = moe.Mxfp4FlydslTuner("mxfp4", KEYS, RESULTS)
        controlled_mxfp4(tuner, monkeypatch, [])
        tuner.run(tuner_args(tuner, source, output))
    assert sorted(pd.read_csv(output).token) == [4, 32, 64]


def test_default_device_helper_leaves_a_fresh_process_usable() -> "None":
    # In-process checks would see the fixture's DeviceContext mode instead.
    code = (
        "import torch\n"
        "from op_tests.tuning_tests._mxfp4_test_process import use_default_device\n"
        "with use_default_device('cpu'):\n"
        "    pass\n"
        "assert not hasattr(torch._GLOBAL_DEVICE_CONTEXT, 'device_context')\n"
        "print(torch.get_default_device())\n"
    )
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "cpu"


def test_unsaved_winner_is_not_reported_tuned_and_exits_nonzero(
    tmp_path: "Path", monkeypatch: "pytest.MonkeyPatch"
) -> "None":
    source, output = tmp_path / "input.csv", tmp_path / "winner.csv"
    pd.DataFrame([shape()]).to_csv(source, index=False)
    to_csv = pd.DataFrame.to_csv

    def failing_to_csv(df: "pd.DataFrame", path: "object" = None, **kwargs: "object"):
        if str(path) == str(output):
            raise OSError("No space left on device")
        return to_csv(df, path, **kwargs)

    monkeypatch.setattr(pd.DataFrame, "to_csv", failing_to_csv)
    with devices(("gfx950", 256)):
        tuner = moe.Mxfp4FlydslTuner("mxfp4", KEYS, RESULTS)
        controlled_mxfp4(tuner, monkeypatch, [])
        with pytest.raises(SystemExit) as status:
            tuner.run(tuner_args(tuner, source, output))
    assert status.value.code == 1
    assert tuner.success.empty and list(tuner.remain_untuned.token) == [4]


@pytest.mark.parametrize("modern_second", [False, True])
def test_multiple_shared_files_migrate_identity_before_merge(
    tmp_path: "Path", monkeypatch: "pytest.MonkeyPatch", modern_second: "bool"
) -> "None":
    source, first, second = (
        tmp_path / "input.csv",
        tmp_path / "first.csv",
        tmp_path / "second.csv",
    )
    pd.DataFrame([shape()]).to_csv(source, index=False)
    with devices(("gfx950", 256)):
        tuner = moe.Mxfp4FlydslTuner("mxfp4", KEYS, RESULTS)
        old1 = tuner._candidate_row(shape(64, cu_num=304), 16, "legacy-g1", "legacy-g2")
        old1.update(us=20)
        old2 = dict(
            old1, token=128, gfx="gfx1250" if modern_second else "gfx950", cu_num=256
        )
        pd.DataFrame([old1]).drop(columns="gfx").to_csv(first, index=False)
        other = pd.DataFrame([old2])
        if not modern_second:
            other = other.drop(columns="gfx")
        other.to_csv(second, index=False)
        before = second.read_bytes()
        combined = tuner.get_tuned_gemm_list(f"{first}:{second}")
        assert combined.loc[combined.token == 64, "gfx"].item() == "gfx942"
        assert combined.loc[combined.token == 128, "gfx"].item() == (
            "gfx1250" if modern_second else "gfx950"
        )
        controlled_mxfp4(tuner, monkeypatch, [])
        tuner.run(tuner_args(tuner, source, f"{first}:{second}"))
    merged = pd.read_csv(first)
    assert merged.loc[merged.token == 64, "gfx"].item() == "gfx942"
    assert second.read_bytes() == before
    assert merged.loc[merged.token == 4, "us"].item() == 10.25


@pytest.mark.parametrize("mode", ["prune", "full"])
def test_runtime_preflight_rejects_participating_device_before_work(
    monkeypatch: "pytest.MonkeyPatch", mode: "str | None"
) -> "None":
    with devices(("gfx950", 256), ("gfx942", 304)):
        tuner = moe.Mxfp4FlydslTuner("mxfp4", KEYS, RESULTS)
        attempts = []
        controlled_mxfp4(tuner, monkeypatch, attempts)
        args = SimpleNamespace(mp=2, timeout=0, mxfp4_search_mode=mode)

        def forbid_workers(*args: "object", **kwargs: "object") -> "None":
            pytest.fail("unsupported participants reached worker launch")

        monkeypatch.setattr(moe, "_run_shapes_isolated", forbid_workers)
        with pytest.raises(ValueError, match="GPU1.*gfx942"):
            tuner.tune(pd.DataFrame([shape(), shape(8)]), None, args)
    assert attempts == []


@pytest.mark.parametrize("mode", ["prune", "full"])
def test_runtime_identity_uses_selected_logical_gpu_and_ignores_unused_device(
    monkeypatch: "pytest.MonkeyPatch", mode: "str | None"
) -> "None":
    with devices(("gfx950:sramecc+", 224), ("gfx942", 304)):
        tuner = moe.Mxfp4FlydslTuner("mxfp4", KEYS, RESULTS)
        attempts = []
        controlled_mxfp4(tuner, monkeypatch, attempts)
        monkeypatch.setattr(moe, "get_gfx", lambda: "gfx942")
        results = tuner.tune(
            pd.DataFrame([shape(gfx="gfx942", cu_num=304)]),
            None,
            SimpleNamespace(mp=1, timeout=0, mxfp4_search_mode=mode),
        )
    assert results[0]["gfx"] == "gfx950"
    assert results[0]["cu_num"] == 224


def test_supported_mixed_cu_runtime_rows_finish_summary(
    tmp_path: "Path", monkeypatch: "pytest.MonkeyPatch"
) -> "None":
    source, output = tmp_path / "input.csv", tmp_path / "winner.csv"
    pd.DataFrame([shape(), shape(8)]).to_csv(source, index=False)
    with devices(("gfx950", 256), ("gfx950", 224)):
        tuner = moe.Mxfp4FlydslTuner("mxfp4", KEYS, RESULTS)

        def simulated_workers(
            payloads: "object", slots: "int", ctx: "BaseContext"
        ) -> "list[dict[str, object]]":
            results = []
            for index, (keys, row, args, _) in enumerate(payloads):
                row = dict(row, gfx="gfx950", cu_num=256 if index == 0 else 224)
                result = tuner._candidate_row(row, 16, "g1-controlled", "g2-controlled")
                result.update(us=10.25, us1=10.25)
                results.append(result)
            return results

        monkeypatch.setattr(moe, "_run_shapes_isolated", simulated_workers)
        args = tuner_args(tuner, source, output)
        args.mp = 2
        tuner.run(args)
    csv = pd.read_csv(output)
    assert csv.loc[csv.token == 8, "cu_num"].item() == 224
    assert tuner.remain_untuned.empty


@pytest.mark.parametrize("mode", ["prune", "full"])
def test_no_visible_gpu_fails_nonempty_but_empty_batch_starts_no_work(
    monkeypatch: "pytest.MonkeyPatch", mode: "str | None"
) -> "None":
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)
    tuner = moe.Mxfp4FlydslTuner("mxfp4", KEYS, RESULTS)
    args = SimpleNamespace(mp=0, timeout=0, mxfp4_search_mode=mode)
    assert tuner.tune(pd.DataFrame(columns=KEYS), None, args) == []
    with pytest.raises(ValueError, match="visible runtime GPU"):
        tuner.tune(pd.DataFrame([shape()]), None, args)


@pytest.mark.parametrize("all_rows", [False, True])
def test_unused_gpu0_does_not_reject_single_slot_current_supported_gpu(
    tmp_path: "Path", monkeypatch: "pytest.MonkeyPatch", all_rows: "bool"
) -> "None":
    source, output = tmp_path / "input.csv", tmp_path / "winner.csv"
    pd.DataFrame([shape()]).to_csv(source, index=False)
    with devices(("gfx942", 304), ("gfx950", 224), current=1):
        tuner = moe.Mxfp4FlydslTuner("mxfp4", KEYS, RESULTS)
        controlled_mxfp4(tuner, monkeypatch, [])
        # Omit input device identity to request tuning on the current device.
        pd.DataFrame([shape()]).drop(columns=["gfx", "cu_num"]).to_csv(
            source, index=False
        )
        tuner.run(tuner_args(tuner, source, output, *("--all",) if all_rows else ()))
    winner = pd.read_csv(output)
    assert winner.loc[0, "gfx"] == "gfx950"
    assert winner.loc[0, "cu_num"] == 224


@pytest.mark.parametrize("mode", ["prune", "full"])
def test_single_slot_rejects_current_unsupported_gpu_even_if_gpu0_supported(
    monkeypatch: "pytest.MonkeyPatch", mode: "str | None"
) -> "None":
    with devices(("gfx950", 256), ("gfx942", 304), current=1):
        tuner = moe.Mxfp4FlydslTuner("mxfp4", KEYS, RESULTS)
        with pytest.raises(ValueError, match="GPU1.*gfx942"):
            tuner.tune(
                pd.DataFrame([shape()]),
                None,
                SimpleNamespace(mp=1, timeout=0, mxfp4_search_mode=mode),
            )


@pytest.mark.parametrize(
    "failure", ["empty", "exception", "nan", "inf", "zero", "negative"]
)
def test_failed_retune_preserves_old_winner_and_continues_later_shape(
    tmp_path: "Path", monkeypatch: "pytest.MonkeyPatch", failure: "str"
) -> "None":
    source, output = tmp_path / "input.csv", tmp_path / "winner.csv"
    pd.DataFrame([shape(), shape(8)]).to_csv(source, index=False)
    with devices(("gfx950", 256)):
        tuner = moe.Mxfp4FlydslTuner("mxfp4", KEYS, RESULTS)
        old = tuner._candidate_row(shape(), 16, "old-g1", "old-g2")
        old.update(us=20, us1=20)
        pd.DataFrame([old]).to_csv(output, index=False)

        def candidates(
            row: "dict[str, object]", full_search: "bool" = False
        ) -> "list[dict[str, object]]":
            if row["token"] == 4 and failure == "empty":
                return []
            return [tuner._candidate_row(row, 16, "g1-controlled", "g2-controlled")]

        def measure(
            row: "dict[str, object]",
            candidate: "dict[str, object]",
            args: "SimpleNamespace",
            **options: "object",
        ) -> "float":
            if row["token"] == 4:
                if failure == "exception":
                    raise RuntimeError("controlled measurement failure")
                us = {
                    "nan": float("nan"),
                    "inf": float("inf"),
                    "zero": 0,
                    "negative": -10,
                }.get(failure, 10.25)
            else:
                us = 10.25
            candidate.update(us=us, us1=us)
            return us

        monkeypatch.setattr(tuner, "_candidate_rows", candidates)
        monkeypatch.setattr(tuner, "_run_candidate", measure)
        with pytest.raises(SystemExit) as status:
            tuner.run(tuner_args(tuner, source, output, "--all"))
    assert status.value.code == 1
    assert tuner.remain_untuned.empty
    winner = pd.read_csv(output)
    assert winner.loc[winner.token == 4, ["kernelName1", "us"]].values.tolist() == [
        ["old-g1", 20.0]
    ]
    assert winner.loc[winner.token == 8, "us"].item() == 10.25
    assert list(tuner.failed.token) == [4]


@pytest.mark.parametrize("mode", ["prune", "full"])
def test_candidate_order_and_finite_winner_survive_bad_first_timing(
    monkeypatch: "pytest.MonkeyPatch", mode: "str | None"
) -> "None":
    with devices(("gfx950", 256)):
        tuner = moe.Mxfp4FlydslTuner("mxfp4", KEYS, RESULTS)
        observed = []

        def candidates(
            row: "dict[str, object]", full_search: "bool" = False
        ) -> "list[dict[str, object]]":
            observed.append(("universe", full_search))
            return [
                tuner._candidate_row(row, 16, name, "g2")
                for name in ("bad", "valid", "slower")
            ]

        def measure(
            row: "dict[str, object]",
            candidate: "dict[str, object]",
            args: "SimpleNamespace",
            **options: "object",
        ) -> "float":
            name = candidate["kernelName1"]
            observed.append(name)
            us = {"bad": float("nan"), "valid": 8.5, "slower": 12.0}[name]
            candidate.update(us=us, us1=us)
            return us

        monkeypatch.setattr(tuner, "_candidate_rows", candidates)
        monkeypatch.setattr(tuner, "_run_candidate", measure)
        result = tuner.tune(
            pd.DataFrame([shape()]),
            None,
            SimpleNamespace(mp=1, timeout=0, mxfp4_search_mode=mode),
        )[0]
    assert observed == [("universe", mode == "full"), "bad", "valid", "slower"]
    assert result["kernelName1"] == "valid"
    assert result["us"] == 8.5


@pytest.mark.parametrize("same_file", [False, True])
def test_all_empty_device_selection_does_not_restore_input_rows(
    tmp_path: "Path", monkeypatch: "pytest.MonkeyPatch", same_file: "bool"
) -> "None":
    source, output = tmp_path / "input.csv", tmp_path / "winner.csv"
    if same_file:
        output = source
    with devices(("gfx950", 256)):
        tuner = moe.Mxfp4FlydslTuner("mxfp4", KEYS, RESULTS)
        old = tuner._candidate_row(
            shape(4, gfx="gfx942", cu_num=304), 16, "old-g1", "old-g2"
        )
        old.update(us=20, gate_mode="GateMode.INTERLEAVE")
        pd.DataFrame([old]).to_csv(source, index=False)
        attempts = []
        controlled_mxfp4(tuner, monkeypatch, attempts)
        tuner.run(tuner_args(tuner, source, output, "--all"))
    assert attempts == []
    assert tuner.untunedf.empty
    if same_file:
        csv = pd.read_csv(output)
        assert list(csv.token) == [4]
        assert csv.loc[0, "gfx"] == "gfx942"


@pytest.mark.parametrize(
    "mp,shape_count,gpu_count,expected",
    [(0, 3, 3, 1), (-1, 3, 3, 1), (8, 1, 3, 1), (8, 3, 2, 2), (2, 3, 3, 2)],
)
def test_prune_full_parallelism_selection_is_preserved(
    monkeypatch: "pytest.MonkeyPatch",
    mp: "int",
    shape_count: "int",
    gpu_count: "int",
    expected: "int",
) -> "None":
    with devices(*[("gfx950", 256)] * gpu_count):
        tuner = moe.Mxfp4FlydslTuner("mxfp4", KEYS, RESULTS)
        attempts = []
        controlled_mxfp4(tuner, monkeypatch, attempts)

        def scheduled(
            payloads: "object", slots: "int", ctx: "BaseContext"
        ) -> "list[dict[str, object]]":
            assert slots == expected
            return [tuner._tune_one_shape(row, args) for _, row, args, _ in payloads]

        monkeypatch.setattr(moe, "_run_shapes_isolated", scheduled)
        results = tuner.tune(
            pd.DataFrame([shape(i + 1) for i in range(shape_count)]),
            None,
            SimpleNamespace(mp=mp, timeout=0, mxfp4_search_mode="prune"),
        )
    assert len(results) == shape_count and len(attempts) == shape_count


@pytest.mark.parametrize("all_rows", [False, True])
@pytest.mark.parametrize("same_file", [False, True])
def test_unsupported_already_tuned_row_fails_and_later_winner_persists(
    tmp_path: "Path",
    monkeypatch: "pytest.MonkeyPatch",
    all_rows: "bool",
    same_file: "bool",
) -> "None":
    source = tmp_path / "input.csv"
    output = source if same_file else tmp_path / "winner.csv"
    requested = [
        shape(gate_mode="GateMode.INTERLEAVE"),
        shape(8, gate_mode="GateMode.SEPARATED"),
    ]
    with devices(("gfx950", 256)):
        tuner = moe.Mxfp4FlydslTuner("mxfp4", KEYS, RESULTS)
        old = tuner._candidate_row(requested[0], 16, "old-g1", "old-g2")
        old.update(us=20, us1=20)
        pd.DataFrame([old]).to_csv(output, index=False)
        pd.DataFrame(requested).to_csv(source, index=False)
        if same_file:
            pd.DataFrame(
                [dict(old, gate_mode="GateMode.INTERLEAVE"), requested[1]]
            ).to_csv(source, index=False)
        attempts = []
        controlled_mxfp4(tuner, monkeypatch, attempts)
        args = tuner_args(tuner, source, output, *("--all",) if all_rows else ())
        with pytest.raises(SystemExit) as status:
            tuner.run(args)
    assert status.value.code == 1
    assert [row["token"] for row in attempts] == [8]
    assert list(tuner.failed["token"]) == [4]
    assert tuner.remain_untuned.empty
    winner = pd.read_csv(output)
    assert winner.loc[winner.token == 8, "us"].item() == 10.25
    assert "gate_mode" not in winner.columns


PORT_E2E = moe.Mxfp4FlydslTuner._port_e2e


def test_invocation_captures_limit_and_stage2_settings_for_every_shape(
    monkeypatch: "pytest.MonkeyPatch",
) -> "None":
    monkeypatch.setenv("AITER_MXFP4_TUNE_SWIGLU_LIMIT", "5")
    monkeypatch.setenv("AITER_FLYDSL_STAGE2_FP8", "1")
    monkeypatch.setenv("AITER_MXFP4_INTERMEDIATE", "1")
    with devices(("gfx950", 256)):
        tuner = moe.Mxfp4FlydslTuner("mxfp4", KEYS, RESULTS)
        monkeypatch.setattr("sys.argv", ["tuner", "--mxfp4-flydsl"])
        args = tuner.parse_args()
        snapshots = []

        def tune_shape(
            row: "dict[str, object]", args: "SimpleNamespace"
        ) -> "dict[str, object]":
            snapshots.append(args.mxfp4_configuration)
            return {**row, "us": 1.0}

        monkeypatch.setattr(tuner, "_tune_one_shape", tune_shape)
        tuner.tune(pd.DataFrame([shape(128), shape(2048)]), None, args)
    assert (
        snapshots
        == [{"swiglu_limit": 5.0, "stage2_fp8": True, "fp4_scatter": True}] * 2
    )
    assert snapshots[0] is snapshots[1]


@pytest.mark.parametrize("value", ["0", "-5", "nan", "inf"])
def test_invalid_activation_limit_rejects_invocation_before_gpu_work(
    monkeypatch: "pytest.MonkeyPatch", value: "object"
) -> "None":
    monkeypatch.setenv("AITER_MXFP4_TUNE_SWIGLU_LIMIT", value)
    with devices(("gfx950", 256)):
        tuner = moe.Mxfp4FlydslTuner("mxfp4", KEYS, RESULTS)
        monkeypatch.setattr("sys.argv", ["tuner", "--mxfp4-flydsl"])
        with pytest.raises(ValueError, match="finite and positive"):
            tuner.parse_args()


def test_custom_limit_rejects_situv2_even_if_same_shape_is_already_tuned(
    monkeypatch: "pytest.MonkeyPatch",
) -> "None":
    monkeypatch.setenv("AITER_MXFP4_TUNE_SWIGLU_LIMIT", "5")
    with devices(("gfx950", 256)):
        tuner = moe.Mxfp4FlydslTuner("mxfp4", KEYS, RESULTS)
    row = shape(act_type="ActivationType.Situv2")
    tuner.untunedf = pd.DataFrame([row])
    tuner.tunedf = pd.DataFrame([{**row, "us": 1.0}])
    assert not tuner._already_tuned_mask(KEYS).iloc[0]
    result = tuner._tune_one_shape(row, SimpleNamespace(timeout=0))
    assert result["us"] == -1 and "Situv2" in result["kernelName1"]


def test_separate_invocations_on_one_tuner_capture_current_settings(
    monkeypatch: "pytest.MonkeyPatch",
) -> "None":
    with devices(("gfx950", 256)):
        tuner = moe.Mxfp4FlydslTuner("mxfp4", KEYS, RESULTS)
    snapshots = []
    monkeypatch.setattr(
        tuner,
        "_tune_one_shape",
        lambda row, args: snapshots.append(args.mxfp4_configuration)
        or {**row, "us": 1.0},
    )
    for limit in ("7", "5"):
        monkeypatch.setenv("AITER_MXFP4_TUNE_SWIGLU_LIMIT", limit)
        with devices(("gfx950", 256)):
            tuner.tune(pd.DataFrame([shape()]), None, SimpleNamespace(mp=1))
    assert [item["swiglu_limit"] for item in snapshots] == [7.0, 5.0]


def test_fp8_stage2_pair_and_screening_receive_original_route_weights(
    monkeypatch: "pytest.MonkeyPatch",
) -> "None":
    from op_tests.tuning_tests.test_mxfp4_staged import fake_case

    data, sorting = fake_case()
    monkeypatch.setattr(moe, "moe_sorting", lambda *args, **kwargs: sorting)
    monkeypatch.setattr(
        moe, "_mxfp4_a4w4_stage1_fw", lambda *args, **kwargs: (object(), object())
    )
    monkeypatch.setattr(
        moe.aiter,
        "fused_dynamic_mxfp4_quant_moe_sort",
        lambda **kwargs: (object(), object()),
    )
    observed = []

    def stage2(*args: "object", **kwargs: "object") -> "object":
        observed.append(kwargs["topk_weights"])
        return sorting[4]

    monkeypatch.setattr(moe, "_mxfp4_a4w4_stage2_fw", stage2)
    name = "flydsl_moe2_layout_afp4_wfp4_bf16_t128x128x128_reduce_sbm128"
    PORT_E2E(
        data,
        "flydsl_mxmoe_g1_a4w4_128x128x256_swiglu",
        name,
        4,
        128,
        3072,
        moe.dtypes.bf16,
        swiglu_limit=5,
    )
    launch, tensors = moe.Mxfp4FlydslTuner._gemm2_screening_call(
        data, sorting, object(), object(), name, 4
    )
    launch(*tensors)
    assert observed == [data["topk_weights"], data["topk_weights"]]


@pytest.mark.parametrize(
    "kn1,kn2",
    [
        (
            "flydsl_mxmoe_g1_a4w4_16x128x256_f16in_nt",
            "flydsl_moe2_layout_afp4_wfp4_bf16_t16x128x128_atomic_sbm16",
        ),
        (
            "flydsl_mxmoe_g1_a4w4_128x128x256_swiglu",
            "flydsl_moe2_layout_afp4_wfp4_bf16_t128x128x128_reduce_sbm128",
        ),
    ],
)
def test_port_e2e_reads_exactly_the_rotated_pipeline_inputs(
    monkeypatch: "pytest.MonkeyPatch", kn1: "str", kn2: "str"
) -> "None":
    from op_tests.tuning_tests.test_mxfp4_staged import fake_case

    class Recording(dict):
        def __init__(self, items: "dict[str, object]") -> "None":
            super().__init__(items)
            self.read: set[str] = set()

        def __getitem__(self, key: "str") -> "object":
            self.read.add(key)
            return super().__getitem__(key)

        def get(self, key: "str", default: "object" = None) -> "object":
            self.read.add(key)
            return super().get(key, default)

    data, sorting = fake_case()
    monkeypatch.setattr(moe, "moe_sorting", lambda *args, **kwargs: sorting)
    monkeypatch.setattr(
        moe, "_mxfp4_a4w4_stage1_fw", lambda *args, **kwargs: (object(), object())
    )
    monkeypatch.setattr(
        moe.aiter,
        "fused_dynamic_mxfp4_quant_moe_sort",
        lambda **kwargs: (object(), object()),
    )
    monkeypatch.setattr(
        moe, "_mxfp4_a4w4_stage2_fw", lambda *args, **kwargs: sorting[4]
    )
    recording = Recording(data)
    PORT_E2E(recording, kn1, kn2, 4, 128, 3072, moe.dtypes.bf16)
    # Timing rebuilds the pipeline input from exactly these keys.
    assert recording.read == set(moe.Mxfp4FlydslTuner._PORT_E2E_INPUTS)


@pytest.mark.parametrize(
    "fp8,fp4", [(None, None), ("", ""), ("1", None), (None, "1"), ("1", "1")]
)
@pytest.mark.parametrize(
    "epilog,bn", [("atomic", 128), ("reduce", 128), ("scatter", 128), ("scatter", 256)]
)
def test_production_stage2_path_matches_invocation_configuration(
    monkeypatch: "pytest.MonkeyPatch",
    fp8: "str | None",
    fp4: "str | None",
    epilog: "str",
    bn: "int",
) -> "None":
    import importlib

    from op_tests.tuning_tests.test_mxfp4_staged import FakeTensor, fake_case

    monkeypatch.setattr(FakeTensor, "element_size", lambda self: 1, raising=False)
    fm = importlib.import_module("aiter.fused_moe")
    dispatcher = importlib.import_module("aiter.ops.flydsl.kernels.mxmoe_dispatcher")
    kernels = importlib.import_module("aiter.ops.flydsl.moe_kernels")
    for name, value in (
        ("AITER_FLYDSL_STAGE2_FP8", fp8),
        ("AITER_MXFP4_INTERMEDIATE", fp4),
    ):
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)
    with devices(("gfx950", 256)):
        tuner = moe.Mxfp4FlydslTuner("mxfp4", KEYS, RESULTS)
    data, sorting = fake_case()
    config = tuner._invocation_configuration()
    expected = (
        "fp8"
        if epilog == "reduce" and config["stage2_fp8"]
        else "fp4" if epilog == "scatter" and config["fp4_scatter"] else "bf16"
    )
    observed = {}
    finished = []
    monkeypatch.setattr(
        moe.torch,
        "empty",
        lambda dimensions, **kwargs: FakeTensor(dimensions, kwargs["dtype"]),
    )

    def gemm(**kwargs: "object") -> "None":
        observed.update(kwargs)

    def reduction(
        target: "torch.Tensor",
        out: "torch.Tensor",
        token: "int",
        topk: "int",
        hidden: "int",
        **kwargs: "object",
    ) -> "None":
        assert kwargs["is_fp8"] == (expected == "fp8")
        assert kwargs["topk_weights"] is (
            data["topk_weights"] if expected == "fp8" else None
        )
        finished.append("reduce")

    def scatter(**kwargs: "object") -> "None":
        assert kwargs["out"] is sorting[4]
        finished.append("scatter")

    def quantized_scatter(**kwargs: "object") -> "None":
        assert kwargs["flat_out_q"].shape == (256, 1536)
        assert kwargs["flat_out_scale"].shape == (256, 96)
        assert (
            kwargs["flat_out_q"].dtype
            == kwargs["flat_out_scale"].dtype
            == moe.torch.uint8
        )
        finished.append("fp4-scatter")

    monkeypatch.setattr(dispatcher, "mxfp4_moe_gemm2", gemm)
    monkeypatch.setattr(kernels, "_run_moe_reduction", reduction)
    monkeypatch.setattr(moe.aiter, "mxfp4_moe_scatter_reduce", scatter)
    monkeypatch.setattr(moe.aiter, "mxfp4_moe_scatter_reduce_q", quantized_scatter)
    result = fm._flydsl_v2_stage2_wrapper(
        FakeTensor((256, 256), moe.torch.uint8),
        None,
        data["w2_a16"],
        sorting[0],
        sorting[2],
        sorting[3],
        sorting[4],
        4,
        kernelName=f"flydsl_moe2_layout_afp4_wfp4_bf16_t128x{bn}x128_{epilog}_sbm128",
        model_dim=3072,
        inter_dim=512,
        num_experts=128,
        a2_scale=FakeTensor((256, 16), moe.torch.uint8),
        w2_scale=data["w2s_a16"],
        sorted_weights=sorting[1],
        topk_weights=data["topk_weights"],
        reverse_sorted=sorting[6],
    )
    assert result is sorting[4] and observed["out_dtype"] == expected
    assert finished == (
        []
        if epilog == "atomic"
        else (
            ["reduce"]
            if epilog == "reduce"
            else ["fp4-scatter"] if expected == "fp4" else ["scatter"]
        )
    )


_GET_CONTEXT = multiprocessing.get_context
_PROCESS_START = BaseProcess.start


def cpu_configuration_worker(
    payload: "tuple[tuple[str, ...], dict[str, object], SimpleNamespace, int]",
    result_tx: "Connection",
) -> "None":
    import importlib

    from op_tests.tuning_tests.test_mxfp4_tuner import import_moe_tuner

    imported = import_moe_tuner()
    fm = importlib.import_module("aiter.fused_moe")
    keys, row, args, gpu = payload
    del keys, gpu
    try:
        result_tx.send(
            {
                **row,
                "snapshot": args.mxfp4_configuration,
                "production_settings": {
                    "stage2_fp8": fm._flydsl_stage2_fp8_enabled(),
                    "fp4_scatter": fm._mxfp4_intermediate_enabled(),
                    "swiglu_limit": imported.runtime_swiglu_limit(
                        args.mxfp4_configuration["swiglu_limit"], "swiglu"
                    ),
                },
                "environment": [
                    os.environ.get(name)
                    for name in (
                        "AITER_MXFP4_TUNE_SWIGLU_LIMIT",
                        "AITER_FLYDSL_STAGE2_FP8",
                        "AITER_MXFP4_INTERMEDIATE",
                    )
                ],
                "us": 1.0,
            }
        )
    finally:
        result_tx.close()


@pytest.mark.parametrize(
    "limit,fp8,fp4",
    [
        (None, None, None),
        ("", "", ""),
        (None, "1", None),
        (None, None, "1"),
        (None, "1", "1"),
        ("5", "1", "1"),
    ],
)
def test_spawned_shape_inherits_snapshot_and_stable_production_settings(
    monkeypatch: "pytest.MonkeyPatch",
    limit: "str | None",
    fp8: "str | None",
    fp4: "str | None",
) -> "None":
    for name, value in zip(
        (
            "AITER_MXFP4_TUNE_SWIGLU_LIMIT",
            "AITER_FLYDSL_STAGE2_FP8",
            "AITER_MXFP4_INTERMEDIATE",
        ),
        (limit, fp8, fp4),
    ):
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)
    with devices(("gfx950", 256)):
        tuner = moe.Mxfp4FlydslTuner("mxfp4", KEYS, RESULTS)
        monkeypatch.setattr(
            "sys.argv", ["tuner", "--mxfp4-flydsl", "--mxfp4-search-mode", "staged"]
        )
        args = tuner.parse_args()

        def cpu_only_start(process: "BaseProcess") -> "None":
            assert process._target is cpu_configuration_worker
            return _PROCESS_START(process)

        monkeypatch.setattr(BaseProcess, "start", cpu_only_start)
        monkeypatch.setattr(multiprocessing, "get_context", _GET_CONTEXT)
        isolated = moe._run_staged_shapes_isolated
        monkeypatch.setattr(
            moe,
            "_run_staged_shapes_isolated",
            lambda payloads, slots, ctx: isolated(
                payloads, slots, ctx, entry=cpu_configuration_worker
            ),
        )
        results = tuner.tune(pd.DataFrame([shape(128), shape(2048)]), None, args)
    for result in results:
        assert result["environment"] == [limit, fp8, fp4]
        assert result["snapshot"] == args.mxfp4_configuration
        assert result["production_settings"] == {
            "stage2_fp8": fp8 == "1",
            "fp4_scatter": fp4 == "1",
            "swiglu_limit": 5.0 if limit == "5" else 7.0,
        }


def test_staged_operation_restores_existing_handler_and_timer() -> "None":
    tuner = moe.Mxfp4FlydslTuner.__new__(moe.Mxfp4FlydslTuner)
    tuner.keys = KEYS
    owned_handler = object()
    restored = []
    with (
        mock.patch.object(signal, "getsignal", return_value=owned_handler),
        mock.patch.object(signal, "getitimer", return_value=(25.0, 3.0)),
        mock.patch.object(
            signal, "signal", side_effect=lambda sig, handler: restored.append(handler)
        ),
        mock.patch.object(signal, "setitimer") as timer,
        tuner._staged_timeout(4),
    ):
        pass
    assert restored[-1] is owned_handler
    assert timer.call_args_list[-1] == mock.call(signal.ITIMER_REAL, 25.0, 3.0)


def test_staged_soft_timeout_is_returned_as_screen_failure_and_restores_timer() -> (
    "None"
):
    tuner = moe.Mxfp4FlydslTuner.__new__(moe.Mxfp4FlydslTuner)
    installed = []
    owned_handler = object()

    def interrupted() -> "float":
        installed[0](signal.SIGALRM, None)

    with (
        mock.patch.object(signal, "getsignal", return_value=owned_handler),
        mock.patch.object(signal, "getitimer", return_value=(13.0, 0.0)),
        mock.patch.object(
            signal, "signal", side_effect=lambda sig, handler: installed.append(handler)
        ),
        mock.patch.object(signal, "setitimer") as timer,
    ):
        assert tuner._screen_candidate(interrupted, "fixture", 4) < 0
    assert installed[-1] is owned_handler
    assert timer.call_args_list[-1] == mock.call(signal.ITIMER_REAL, 13.0, 0.0)


def test_staged_soft_timeout_unavailable_handler_runs_without_touching_timer() -> (
    "None"
):
    tuner = moe.Mxfp4FlydslTuner.__new__(moe.Mxfp4FlydslTuner)
    with (
        mock.patch.object(signal, "getsignal", return_value=None),
        mock.patch.object(signal, "setitimer") as timer,
    ):
        assert tuner._screen_candidate(lambda: 3.0, "fixture", 4) == 3.0
    timer.assert_not_called()


def test_staged_soft_timeout_outside_main_thread_runs_without_touching_timer() -> (
    "None"
):
    tuner = moe.Mxfp4FlydslTuner.__new__(moe.Mxfp4FlydslTuner)
    with (
        mock.patch.object(signal, "signal", side_effect=ValueError("main thread")),
        mock.patch.object(signal, "setitimer") as timer,
    ):
        assert tuner._screen_candidate(lambda: 3.0, "fixture", 4) == 3.0
    timer.assert_not_called()
