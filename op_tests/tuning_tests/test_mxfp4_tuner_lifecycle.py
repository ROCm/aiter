# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Controlled tuner lifecycle checks; no GPU kernels or measurements run.

Full-module imports require the normal AITER ROCm/FlyDSL toolchain. Devices and
measurement outcomes are simulated at the approved tuning boundaries.
"""

import multiprocessing
from contextlib import contextmanager
from multiprocessing.process import BaseProcess
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
]


@pytest.fixture(autouse=True)
def prohibit_real_shape_workers_and_gpu_data(monkeypatch):
    # Spawn does not inherit parent mocks. Every multi-slot fixture must replace
    # this boundary with controlled scheduling, even while the code is red.
    def forbidden(*args, **kwargs):
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
    torch.set_default_device("cpu")


def shape(token=4, **updates):
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


def tuner_args(tuner, source, output, *options):
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
def devices(*identities, current=0):
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


def controlled_mxfp4(tuner, monkeypatch, attempts):
    def candidates(row, full_search=False):
        return [tuner._candidate_row(row, 16, "g1-controlled", "g2-controlled")]

    def measure(row, candidate, args):
        attempts.append(dict(row))
        candidate.update(us=10.25, us1=10.25, us2=0, err1="0.0%", err2="0.0%")
        return 10.25

    monkeypatch.setattr(tuner, "_candidate_rows", candidates)
    monkeypatch.setattr(tuner, "_run_candidate", measure)


@pytest.mark.parametrize("same_file", [False, True])
def test_all_preserves_gate_metadata_through_winner_summary(
    tmp_path, monkeypatch, same_file
):
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


def test_retune_replaces_duplicate_old_winners(tmp_path, monkeypatch):
    source, output = tmp_path / "input.csv", tmp_path / "winner.csv"
    pd.DataFrame([shape()]).to_csv(source, index=False)
    with devices(("gfx950", 256)):
        tuner = moe.Mxfp4FlydslTuner("mxfp4", KEYS, RESULTS)
        old = tuner._candidate_row(shape(), 16, "old-g1", "old-g2")
        pd.DataFrame([dict(old, us=20), dict(old, us=21)]).to_csv(output, index=False)
        controlled_mxfp4(tuner, monkeypatch, [])
        tuner.run(tuner_args(tuner, source, output, "--all"))
    winner = pd.read_csv(output)
    assert winner["us"].tolist() == [10.25]
    assert winner["kernelName1"].tolist() == ["g1-controlled"]


def test_same_file_all_without_device_identity_retains_gate_metadata(
    tmp_path, monkeypatch
):
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
    tmp_path, monkeypatch, empty_input
):
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
        tagged = dict(legacy, token=32, gfx="gfx942", _tag="fallback")
        pd.DataFrame([legacy, explicit, tagged]).to_csv(output, index=False)
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
    assert "_tag" not in csv.columns and "gate_mode" not in csv.columns


@pytest.mark.parametrize("modern_second", [False, True])
def test_multiple_shared_files_migrate_identity_before_merge(
    tmp_path, monkeypatch, modern_second
):
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
def test_runtime_preflight_rejects_participating_device_before_work(monkeypatch, mode):
    with devices(("gfx950", 256), ("gfx942", 304)):
        tuner = moe.Mxfp4FlydslTuner("mxfp4", KEYS, RESULTS)
        attempts = []
        controlled_mxfp4(tuner, monkeypatch, attempts)
        args = SimpleNamespace(mp=2, timeout=0, mxfp4_search_mode=mode)

        def forbid_workers(*args, **kwargs):
            pytest.fail("unsupported participants reached worker launch")

        monkeypatch.setattr(moe, "_run_shapes_isolated", forbid_workers)
        with pytest.raises(ValueError, match="GPU1.*gfx942"):
            tuner.tune(pd.DataFrame([shape(), shape(8)]), None, args)
    assert attempts == []


@pytest.mark.parametrize("mode", ["prune", "full"])
def test_runtime_identity_uses_selected_logical_gpu_and_ignores_unused_device(
    monkeypatch, mode
):
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


def test_supported_mixed_cu_runtime_rows_finish_summary(tmp_path, monkeypatch):
    source, output = tmp_path / "input.csv", tmp_path / "winner.csv"
    pd.DataFrame([shape(), shape(8)]).to_csv(source, index=False)
    with devices(("gfx950", 256), ("gfx950", 224)):
        tuner = moe.Mxfp4FlydslTuner("mxfp4", KEYS, RESULTS)

        def simulated_workers(payloads, slots, ctx):
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
    monkeypatch, mode
):
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)
    tuner = moe.Mxfp4FlydslTuner("mxfp4", KEYS, RESULTS)
    args = SimpleNamespace(mp=0, timeout=0, mxfp4_search_mode=mode)
    assert tuner.tune(pd.DataFrame(columns=KEYS), None, args) == []
    with pytest.raises(ValueError, match="visible runtime GPU"):
        tuner.tune(pd.DataFrame([shape()]), None, args)


@pytest.mark.parametrize("all_rows", [False, True])
def test_unused_gpu0_does_not_reject_single_slot_current_supported_gpu(
    tmp_path, monkeypatch, all_rows
):
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
    monkeypatch, mode
):
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
    tmp_path, monkeypatch, failure
):
    source, output = tmp_path / "input.csv", tmp_path / "winner.csv"
    pd.DataFrame([shape(), shape(8)]).to_csv(source, index=False)
    with devices(("gfx950", 256)):
        tuner = moe.Mxfp4FlydslTuner("mxfp4", KEYS, RESULTS)
        old = tuner._candidate_row(shape(), 16, "old-g1", "old-g2")
        old.update(us=20, us1=20)
        pd.DataFrame([old]).to_csv(output, index=False)

        def candidates(row, full_search=False):
            if row["token"] == 4 and failure == "empty":
                return []
            return [tuner._candidate_row(row, 16, "g1-controlled", "g2-controlled")]

        def measure(row, candidate, args):
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
def test_candidate_order_and_finite_winner_survive_bad_first_timing(monkeypatch, mode):
    with devices(("gfx950", 256)):
        tuner = moe.Mxfp4FlydslTuner("mxfp4", KEYS, RESULTS)
        observed = []

        def candidates(row, full_search=False):
            observed.append(("universe", full_search))
            return [
                tuner._candidate_row(row, 16, name, "g2")
                for name in ("bad", "valid", "slower")
            ]

        def measure(row, candidate, args):
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
    tmp_path, monkeypatch, same_file
):
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
    monkeypatch, mp, shape_count, gpu_count, expected
):
    with devices(*[("gfx950", 256)] * gpu_count):
        tuner = moe.Mxfp4FlydslTuner("mxfp4", KEYS, RESULTS)
        attempts = []
        controlled_mxfp4(tuner, monkeypatch, attempts)

        def scheduled(payloads, slots, ctx):
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
    tmp_path, monkeypatch, all_rows, same_file
):
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
