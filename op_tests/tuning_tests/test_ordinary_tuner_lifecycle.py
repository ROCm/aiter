# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Ordinary fMoE lifecycle regression with controlled measurements.

No GPU kernels run. Full-module imports still require AITER's ROCm/FlyDSL
environment. CK candidate enumeration, result processing, profiles, CSV writes,
sorting, and summary use the real ordinary tuner; only the measurement dispatch
and runtime device properties are controlled.
"""

import argparse
import os
import subprocess
import sys
import textwrap
from collections.abc import Callable, Iterator
from contextlib import ExitStack, contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pandas as pd
import pytest
import torch

from op_tests.tuning_tests._controlled_tuner_import import import_moe_tuner

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
    "nt",
]


@contextmanager
def prohibit_gpu_work() -> "Iterator[None]":
    """Fail closed, including when a subprocess imports a fresh tuner class."""

    def forbidden(*args: "object", **kwargs: "object") -> "None":
        # pytest.fail escapes the tuner's catch(Exception) recovery path.
        pytest.fail("controlled ordinary lifecycle test reached real GPU work")

    with ExitStack() as guards:
        torch.set_default_device("cpu")
        guards.enter_context(
            mock.patch.dict(
                os.environ,
                {
                    "ROCR_VISIBLE_DEVICES": "",
                    "HIP_VISIBLE_DEVICES": "",
                    "CUDA_VISIBLE_DEVICES": "",
                },
            )
        )
        guards.enter_context(
            mock.patch("multiprocessing.get_context", return_value=None)
        )
        for target in (
            "multiprocessing.process.BaseProcess.start",
            "aiter.test_common.run_perftest",
            "aiter.utility.mp_tuner.mp_tuner",
            "aiter.utility.mp_tuner.worker",
            "aiter.utility.mp_tuner.work_group",
            "torch.cuda._lazy_init",
            "torch.cuda.set_device",
            "torch.cuda.synchronize",
            "torch.cuda.Event",
            "torch.cuda.CUDAGraph",
        ):
            guards.enter_context(mock.patch(target, side_effect=forbidden))
        guards.enter_context(mock.patch.object(moe, "mp_tuner", side_effect=forbidden))
        for name in (
            "empty",
            "zeros",
            "ones",
            "tensor",
            "arange",
            "rand",
            "randn",
            "randint",
            "empty_like",
            "zeros_like",
            "ones_like",
            "full",
            "full_like",
        ):
            factory = getattr(torch, name)

            def cpu_only(
                *args: "object",
                _factory: "Callable[..., torch.Tensor]" = factory,
                **kwargs: "object"
            ) -> "torch.Tensor":
                device = kwargs.get("device")
                if device is not None and torch.device(device).type == "cpu":
                    return _factory(*args, **kwargs)
                return forbidden(*args, **kwargs)

            guards.enter_context(mock.patch.object(torch, name, side_effect=cpu_only))
        for name in ("generate_data", "generate_data_1stage", "generate_data_2stages"):
            guards.enter_context(
                mock.patch.object(moe.FmoeTuner, name, side_effect=forbidden)
            )
        yield


@pytest.fixture(autouse=True)
def forbid_uncontrolled_gpu_work() -> "Iterator[None]":
    with prohibit_gpu_work():
        yield


def shape(token: "int" = 4, **updates: "object") -> "dict[str, object]":
    row = {
        "gfx": "gfx942",
        "cu_num": 304,
        "token": token,
        "model_dim": 512,
        "inter_dim": 256,
        "expert": 2,
        "topk": 2,
        "act_type": "ActivationType.Silu",
        "dtype": "torch.bfloat16",
        "q_dtype_a": str(moe.dtypes.fp8),
        "q_dtype_w": str(moe.dtypes.fp8),
        "q_type": "QuantType.per_1x128",
        "use_g1u1": 1,
        "doweight_stage1": 0,
    }
    row.update(updates)
    return row


def historical_result(token: "int" = 2, **updates: "object") -> "dict[str, object]":
    row = dict(
        shape(token),
        block_m=16,
        ksplit=0,
        us1=6.25,
        kernelName1="historical-g1",
        err1="2.0%",
        us2=3.5,
        kernelName2="historical-g2",
        err2="1.0%",
        us=9.75,
        run_1stage=0,
        xbf16=0,
        flat=0,
        tflops=0.32,
        bw=81.08,
        nt=1,
    )
    row.update(updates)
    return row


@contextmanager
def controlled_environment(
    root: "Path",
    outcomes: "dict[int, str] | None" = None,
    dispatched: "list[object] | None" = None,
) -> "Iterator[None]":
    """Keep real CK tasks and return measurements without executing their funcs."""
    outcomes = outcomes or {}
    (root / "aiter" / "configs").mkdir(parents=True, exist_ok=True)

    def measure(
        tasks: "list[object]", *args: "object", **kwargs: "object"
    ) -> "list[tuple[object, float, float]]":
        results = []
        for task in tasks:
            tag = task[0]
            info, stage, _name, block_m, _flat, _v2, nt = tag
            if dispatched is not None:
                dispatched.append(tag)
            if outcomes.get(info[2]) == "unfinished":
                continue
            if outcomes.get(info[2]) in {"failed", "timeout", "incorrect"}:
                us, err = {
                    "failed": (-1, 1.0),
                    "timeout": (float("inf"), 1.0),
                    "incorrect": (0.5, 0.5),
                }[outcomes[info[2]]]
                results.append((tag, us, err))
                continue
            us = {"stage1": 2.25, "stage2": 3.5}[stage]
            if not nt:
                us += 1
            if block_m != 16:
                us += 20
            results.append((tag, us, 0.02 if stage == "stage1" else 0.01))
        return results

    properties = SimpleNamespace(gcnArchName="gfx942", multi_processor_count=304)
    with (
        # This context is also entered inside the command subprocess.
        prohibit_gpu_work(),
        mock.patch.dict("os.environ", {"TUNE_ONLY": "cktile"}),
        mock.patch.object(torch.cuda, "device_count", return_value=1),
        mock.patch.object(torch.cuda, "current_device", return_value=0),
        mock.patch.object(torch.cuda, "get_device_properties", return_value=properties),
        mock.patch.object(moe, "get_gfx", return_value="gfx942"),
        mock.patch.object(moe.TunerCommon, "get_gfx", return_value="gfx942"),
        mock.patch.object(moe, "AITER_ROOT_DIR", str(root)),
        mock.patch.object(moe, "mp_tuner", side_effect=measure),
        # The script imports these functions again under __main__.
        mock.patch("aiter.jit.utils.chip_info.get_gfx", return_value="gfx942"),
        mock.patch("aiter.jit.utils.chip_info.get_gfx_runtime", return_value="gfx942"),
        mock.patch("aiter.jit.core.AITER_ROOT_DIR", str(root)),
        mock.patch("aiter.utility.mp_tuner.mp_tuner", side_effect=measure),
    ):
        yield


def args_for(
    tuner: "moe.FmoeTuner", source: "Path", output: "Path", *options: "object"
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


def test_ordinary_fmoe_preserves_schema_nt_stage_timings_and_profiles(
    tmp_path: "Path",
) -> "None":
    source, output = tmp_path / "input.csv", tmp_path / "winner.csv"
    profile = tmp_path / "paired-profile.csv"
    pd.DataFrame([shape(8), shape(), shape()]).to_csv(source, index=False)
    pd.DataFrame([historical_result(), historical_result(us=9.5)]).to_csv(
        output, index=False
    )
    dispatched = []
    with controlled_environment(tmp_path, dispatched=dispatched):
        tuner = moe.FmoeTuner("ordinary-fmoe", KEYS, RESULTS)
        tuner.run(args_for(tuner, source, output, "--profile_file", str(profile)))

    winner = pd.read_csv(output)
    assert list(winner.columns) == KEYS + RESULTS
    assert list(winner.token) == [2, 4, 8]
    assert winner.loc[winner.token == 2, "us"].item() == 9.5
    historical = winner[winner.token == 2].iloc[0]
    assert (historical.us1, historical.us2, historical.nt) == (6.25, 3.5, 1)
    assert (historical.tflops, historical.bw) == (0.32, 81.08)
    row = winner[winner.token == 4].iloc[0]
    assert (row.gfx, row.cu_num, row.block_m, row.nt) == ("gfx942", 304, 16, 1)
    assert (row.us1, row.us2, row.us) == (2.25, 3.5, 5.75)
    assert (row.err1, row.err2, row.tflops, row.bw) == ("2.0%", "1.0%", 1.09, 137.84)
    assert (row.run_1stage, row.xbf16, row.flat, row.ksplit) == (0, 0, 0, 0)
    assert row.kernelName1.startswith("moe_ck2stages_gemm1_")
    assert row.kernelName2.startswith("moe_ck2stages_gemm2_")
    assert all(
        pd.api.types.is_numeric_dtype(winner[c])
        for c in ["us1", "us2", "us", "nt", "tflops", "bw"]
    )
    assert tuner.failed.empty and tuner.remain_untuned.empty

    paired = pd.read_csv(profile)
    assert set(paired.token) == {4, 8}
    assert set(paired.nt) == {0, 1}
    assert len(paired) > 2
    assert (paired.us == paired.us1 + paired.us2).all()
    raw = pd.read_csv(tmp_path / "aiter" / "configs" / "profile_fmoe.csv")
    assert set(raw.stage) == {"stage1", "stage2"}
    assert set(raw.nt) == {0, 1}
    assert set(raw.token) == {4, 8}
    assert len(raw) == len(dispatched)


def test_ordinary_retune_replaces_duplicate_old_winners(tmp_path: "Path") -> "None":
    source, output = tmp_path / "input.csv", tmp_path / "winner.csv"
    pd.DataFrame([shape()]).to_csv(source, index=False)
    pd.DataFrame([historical_result(4, us=20), historical_result(4, us=21)]).to_csv(
        output, index=False
    )
    with controlled_environment(tmp_path):
        tuner = moe.FmoeTuner("ordinary-fmoe", KEYS, RESULTS)
        tuner.run(args_for(tuner, source, output, "--all"))
    winner = pd.read_csv(output)
    assert winner.us.tolist() == [5.75]
    assert winner.kernelName1.str.startswith("moe_ck2stages_gemm1_").all()


def test_ordinary_fmoe_migrates_legacy_shared_csv_without_losing_nt_or_numbers(
    tmp_path: "Path",
) -> "None":
    source, output = tmp_path / "input.csv", tmp_path / "winner.csv"
    pd.DataFrame([shape()]).to_csv(source, index=False)
    old = pd.DataFrame(
        [
            historical_result(),
            historical_result(8, cu_num=256),
            historical_result(10, _tag="obsolete"),
        ]
    ).drop(columns=["gfx", "xbf16", "flat"])
    old.to_csv(output, index=False)
    with controlled_environment(tmp_path):
        tuner = moe.FmoeTuner("ordinary-fmoe", KEYS, RESULTS)
        tuner.run(args_for(tuner, source, output))

    winner = pd.read_csv(output)
    assert list(winner.columns) == KEYS + RESULTS
    assert list(winner.token) == [2, 4, 8]
    assert list(winner.gfx) == ["gfx942", "gfx942", "gfx950"]
    assert list(winner.nt) == [1, 1, 1]
    assert list(winner.xbf16) == [0, 0, 0]
    assert list(winner.flat) == [0, 0, 0]
    assert list(winner.us) == [9.75, 5.75, 9.75]
    assert winner.loc[winner.token == 8, "cu_num"].item() == 256


@pytest.mark.parametrize("outcome", ["failed", "timeout", "incorrect"])
def test_ordinary_fmoe_reports_failed_and_unfinished_rows_and_saves_later_winner(
    tmp_path: "Path", outcome: "object"
) -> "None":
    source, output = tmp_path / "input.csv", tmp_path / "winner.csv"
    profile = tmp_path / "paired-profile.csv"
    pd.DataFrame([shape(), shape(6, use_g1u1=0), shape(8)]).to_csv(source, index=False)
    pd.DataFrame([historical_result()]).to_csv(output, index=False)
    dispatched = []
    with controlled_environment(tmp_path, outcomes={4: outcome}, dispatched=dispatched):
        tuner = moe.FmoeTuner("ordinary-fmoe", KEYS, RESULTS)
        with pytest.raises(SystemExit) as status:
            tuner.run(
                args_for(
                    tuner,
                    source,
                    output,
                    "--batch",
                    "1",
                    "--profile_file",
                    str(profile),
                )
            )

    assert status.value.code == 1
    assert list(tuner.failed.token) == [4]
    assert list(tuner.remain_untuned.token) == [6]
    assert list(tuner.success.token) == [8]
    winner = pd.read_csv(output)
    assert list(winner.token) == [2, 8]
    assert winner.loc[winner.token == 8, "us"].item() == 5.75
    assert {tag[0][2] for tag in dispatched} == {4, 8}
    paired = pd.read_csv(profile)
    assert set(paired.token) == {8}
    raw = pd.read_csv(tmp_path / "aiter" / "configs" / "profile_fmoe.csv")
    assert set(raw.token) == {4, 8}
    failed_times = raw.loc[raw.token == 4, "us"]
    assert (
        set(failed_times)
        == {
            "failed": {-1},
            "timeout": {float("inf")},
            "incorrect": {0.5},
        }[outcome]
    )


@pytest.mark.parametrize("output_exists", [False, True])
@pytest.mark.parametrize("all_rows", [False, True])
def test_ordinary_fmoe_empty_input_finishes_without_measurement(
    tmp_path: "Path", output_exists: "bool", all_rows: "bool"
) -> "None":
    source, output = tmp_path / "input.csv", tmp_path / "winner.csv"
    pd.DataFrame(columns=KEYS).to_csv(source, index=False)
    if output_exists:
        pd.DataFrame([historical_result(), historical_result(us=9.5)]).to_csv(
            output, index=False
        )
    dispatched = []
    with controlled_environment(tmp_path, dispatched=dispatched):
        tuner = moe.FmoeTuner("ordinary-fmoe", KEYS, RESULTS)
        options = ["--all"] if all_rows else []
        tuner.run(args_for(tuner, source, output, *options))

    assert dispatched == []
    assert tuner.failed.empty and tuner.remain_untuned.empty
    if output_exists:
        winner = pd.read_csv(output)
        assert list(winner.token) == [2]
        assert winner.us.item() == 9.5
    else:
        assert not output.exists()


@pytest.mark.parametrize("same_file", [False, True])
@pytest.mark.parametrize("all_rows", [False, True])
def test_ordinary_fmoe_all_and_non_all_preserve_history_and_select_requested_rows(
    tmp_path: "Path", same_file: "bool", all_rows: "bool"
) -> "None":
    source = tmp_path / "input.csv"
    output = source if same_file else tmp_path / "winner.csv"
    old = [historical_result(8), historical_result()]
    requested = old if same_file else [shape(8), shape(4)]
    pd.DataFrame(old).to_csv(output, index=False)
    pd.DataFrame(requested).to_csv(source, index=False)
    dispatched = []
    with controlled_environment(tmp_path, dispatched=dispatched):
        tuner = moe.FmoeTuner("ordinary-fmoe", KEYS, RESULTS)
        options = ["--all"] if all_rows else []
        tuner.run(args_for(tuner, source, output, *options))

    expected_tokens = {
        (False, False): {4},
        (False, True): {4, 8},
        (True, False): set(),
        (True, True): {2, 8},
    }[same_file, all_rows]
    assert {tag[0][2] for tag in dispatched} == expected_tokens
    winner = pd.read_csv(output)
    assert list(winner.token) == ([2, 8] if same_file else [2, 4, 8])
    assert list(winner.columns) == KEYS + RESULTS
    assert winner.loc[winner.token == 8, "us"].item() == (5.75 if all_rows else 9.75)
    assert tuner.failed.empty and tuner.remain_untuned.empty


@pytest.mark.parametrize(
    "outcome,expected_status",
    [
        ("success", 0),
        ("failed", 1),
        ("unfinished", 1),
    ],
)
def test_ordinary_fmoe_command_returns_summary_exit_status(
    tmp_path: "Path", outcome: "object", expected_status: "int"
) -> "None":
    source, output = tmp_path / "input.csv", tmp_path / "winner.csv"
    pd.DataFrame([shape(), shape(8)]).to_csv(source, index=False)
    command = textwrap.dedent("""\
        from pathlib import Path
        import runpy
        import sys
        from op_tests.tuning_tests.test_ordinary_tuner_lifecycle import (
            controlled_environment,
        )
        root, outcome = sys.argv[1:3]
        sys.argv = [sys.argv[0], *sys.argv[3:]]
        with controlled_environment(Path(root), outcomes={4: outcome}):
            runpy.run_path("csrc/ck_gemm_moe_2stages_codegen/gemm_moe_tune.py",
                           run_name="__main__")
    """)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            command,
            str(tmp_path),
            outcome,
            "-i",
            str(source),
            "-o",
            str(output),
            "--mp",
            "1",
            "--timeout",
            "0",
            "--batch",
            "1",
            "--sort",
            "True",
        ],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        env=dict(
            os.environ,
            ROCR_VISIBLE_DEVICES="",
            HIP_VISIBLE_DEVICES="",
            CUDA_VISIBLE_DEVICES="",
        ),
        timeout=30,
        check=False,
    )

    assert result.returncode == expected_status, result.stdout + result.stderr
    winner = pd.read_csv(output)
    assert list(winner.token) == ([4, 8] if expected_status == 0 else [8])
    assert "Tuning results Summary" in result.stderr
    if expected_status:
        assert "Tuning not Finished" in result.stderr
