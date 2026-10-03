# SPDX-License-Identifier: MIT

import argparse
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from op_tests.tuners import tune_fhmoe
from op_tests.tuners.tune_fhmoe import (
    Contract,
    _matches,
    _parse_tokens,
    _read_rows,
    _sample_expert_ids,
    _sample_rows,
    _select_rows,
    _validate_sampled_reference,
    _write_rows,
    stage1_candidates,
    stage2_candidates,
)


def test_parse_tokens():
    assert _parse_tokens("32,1,32,16") == [1, 16, 32]
    with pytest.raises(argparse.ArgumentTypeError):
        _parse_tokens("0")


def test_large_m_samples_cover_boundary_experts():
    rows = _sample_rows(131072)
    assert rows == (0, 1, 15, 16, 65536, 131071)
    expert_ids = _sample_expert_ids(Contract(), rows)
    assert {0, 7, 8, 127, 128, 255} <= set(expert_ids)


def test_hy4_rows_and_candidate_kernels_are_complete():
    root = Path(__file__).resolve().parents[2]
    fields, rows = _read_rows(root / "aiter/configs/tuned_fhmoe.csv")
    contract = Contract()
    expected_tokens = {
        1,
        2,
        4,
        8,
        16,
        32,
        64,
        128,
        256,
        512,
        1024,
        2048,
        4096,
        8192,
        16384,
        32768,
        131072,
    }
    actual_tokens = {
        int(row["token"]) for row in rows if _matches(row, contract, int(row["token"]))
    }
    assert actual_tokens == expected_tokens
    assert fields[-2:] == ["kernelName1", "kernelName2"]

    for block_m in (32, 64, 128):
        assert stage1_candidates(block_m, quick=True)
        assert stage2_candidates(block_m, quick=True)


def test_matches_requires_complete_hy4_contract():
    root = Path(__file__).resolve().parents[2]
    _, rows = _read_rows(root / "aiter/configs/tuned_fhmoe.csv")
    contract = Contract()
    row = next(
        row for row in rows if row["model_dim"] == "6144" and row["token"] == "32"
    )
    assert _matches(row, contract, 32)

    mismatches = {
        "gfx": "gfx942",
        "cu_num": "304",
        "token": "64",
        "model_dim": "4096",
        "inter_dim": "128",
        "expert": "256",
        "topk": "8",
        "shared_expert_id": "255",
        "act_type": "ActivationType.Swiglu",
        "dtype": "torch.float16",
        "q_dtype_a": "torch.float8_e4m3fnuz",
        "q_dtype_w": "torch.float4_e2m1fn_x2",
        "q_type": "QuantType.per_Token",
        "use_g1u1": "0",
        "doweight_stage1": "1",
        "hidden_pad": "1",
        "intermediate_pad": "1",
        "gate_mode": "GateMode.SEPARATED",
        "ksplit": "1",
    }
    for field, value in mismatches.items():
        assert not _matches(row | {field: value}, contract, 32), field


def test_select_rows_rejects_duplicate_complete_contract():
    root = Path(__file__).resolve().parents[2]
    _, rows = _read_rows(root / "aiter/configs/tuned_fhmoe.csv")
    row = next(
        row for row in rows if row["model_dim"] == "6144" and row["token"] == "32"
    )
    with pytest.raises(ValueError, match="multiple HY4 FHMoE rows.*M=32"):
        _select_rows([row, dict(row)], Contract(), [32])


def test_csv_roundtrip_preserves_unrelated_rows(tmp_path: Path):
    root = Path(__file__).resolve().parents[2]
    fields, rows = _read_rows(root / "aiter/configs/tuned_fhmoe.csv")
    output = tmp_path / "tuned_fhmoe.csv"
    _write_rows(output, fields, rows)
    roundtrip_fields, roundtrip_rows = _read_rows(output)
    assert roundtrip_fields == fields
    assert roundtrip_rows == rows


def test_baseline_only_preserves_current_kernels(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    row = {
        "token": "32",
        "block_m": "32",
        "kernelName1": "stage1",
        "kernelName2": "stage2",
    }
    monkeypatch.setattr(tune_fhmoe, "_write_candidate", lambda *_args: None)
    monkeypatch.setattr(tune_fhmoe, "_runner", lambda *_args: object())
    monkeypatch.setattr(
        tune_fhmoe,
        "_time_candidate",
        lambda *_args: (1.25, torch.ones(1)),
    )

    tuned, profile = tune_fhmoe.tune_row(
        row,
        tensors=object(),
        contract=Contract(),
        config_file=tmp_path / "candidate.csv",
        quick=True,
        warmup_ms=1,
        rep_ms=1,
        tolerance=0,
        swiglu_limit=10,
        min_improvement_pct=1,
        baseline_only=True,
    )

    assert tuned == row
    assert profile == [{"stage": "baseline", "kernel": "stage1|stage2", "ms": 1.25}]


@pytest.mark.parametrize("baseline_only", (False, True))
def test_nonfinite_baseline_fails_before_tuning(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    baseline_only: bool,
):
    row = {
        "token": "32",
        "block_m": "32",
        "kernelName1": "stage1",
        "kernelName2": "stage2",
    }
    monkeypatch.setattr(tune_fhmoe, "_write_candidate", lambda *_args: None)
    monkeypatch.setattr(tune_fhmoe, "_runner", lambda *_args: object())
    monkeypatch.setattr(
        tune_fhmoe,
        "_time_candidate",
        lambda *_args: (float("inf"), torch.tensor([float("nan")])),
    )

    with pytest.raises(RuntimeError, match="baseline FHMoE kernels.*non-finite"):
        tune_fhmoe.tune_row(
            row,
            tensors=object(),
            contract=Contract(),
            config_file=tmp_path / "candidate.csv",
            quick=True,
            warmup_ms=1,
            rep_ms=1,
            tolerance=0,
            swiglu_limit=10,
            min_improvement_pct=1,
            baseline_only=baseline_only,
        )


def test_tune_row_selects_only_fast_valid_candidates(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    row = {
        "token": "32",
        "block_m": "32",
        "kernelName1": "stage1",
        "kernelName2": "stage2",
    }
    active: dict[str, str] = {}

    def write_candidate(_path, candidate):
        active.clear()
        active.update(candidate)

    timings = {
        ("stage1", "stage2"): (1.0, torch.ones(2)),
        ("stage1", "stage2_fast"): (0.8, torch.ones(2)),
        ("stage1", "stage2_wrong"): (0.5, torch.zeros(2)),
        ("stage1", "stage2_nan"): (
            float("inf"),
            torch.tensor([float("nan"), 1.0]),
        ),
        ("stage1_fast", "stage2_fast"): (0.6, torch.ones(2)),
    }
    monkeypatch.setattr(tune_fhmoe, "_write_candidate", write_candidate)
    monkeypatch.setattr(
        tune_fhmoe,
        "_runner",
        lambda *_args: (active["kernelName1"], active["kernelName2"]),
    )
    monkeypatch.setattr(tune_fhmoe, "_time_candidate", lambda run, *_: timings[run])
    monkeypatch.setattr(
        tune_fhmoe,
        "stage2_candidates",
        lambda *_args: ["stage2_fast", "stage2_wrong", "stage2_nan"],
    )
    monkeypatch.setattr(tune_fhmoe, "stage1_candidates", lambda *_args: ["stage1_fast"])

    tuned, _ = tune_fhmoe.tune_row(
        row,
        tensors=object(),
        contract=Contract(),
        config_file=tmp_path / "candidate.csv",
        quick=True,
        warmup_ms=1,
        rep_ms=1,
        tolerance=0.1,
        swiglu_limit=10,
        min_improvement_pct=1,
    )

    assert tuned["kernelName1"] == "stage1_fast"
    assert tuned["kernelName2"] == "stage2_fast"


def test_tune_row_keeps_original_below_minimum_improvement(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    row = {
        "token": "32",
        "block_m": "32",
        "kernelName1": "stage1",
        "kernelName2": "stage2",
    }
    active: dict[str, str] = {}

    def write_candidate(_path, candidate):
        active.clear()
        active.update(candidate)

    monkeypatch.setattr(tune_fhmoe, "_write_candidate", write_candidate)
    monkeypatch.setattr(
        tune_fhmoe,
        "_runner",
        lambda *_args: (active["kernelName1"], active["kernelName2"]),
    )
    monkeypatch.setattr(
        tune_fhmoe,
        "_time_candidate",
        lambda run, *_: (
            (0.995 if run == ("stage1", "candidate") else 1.0),
            torch.ones(1),
        ),
    )
    monkeypatch.setattr(tune_fhmoe, "stage2_candidates", lambda *_args: ["candidate"])
    monkeypatch.setattr(tune_fhmoe, "stage1_candidates", lambda *_args: [])

    tuned, _ = tune_fhmoe.tune_row(
        row,
        tensors=object(),
        contract=Contract(),
        config_file=tmp_path / "candidate.csv",
        quick=True,
        warmup_ms=1,
        rep_ms=1,
        tolerance=0,
        swiglu_limit=10,
        min_improvement_pct=1,
    )

    assert tuned == row


@pytest.mark.parametrize("expected_value,should_pass", ((1.0, True), (0.0, False)))
def test_sampled_reference_controls_final_acceptance(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    expected_value: float,
    should_pass: bool,
):
    tensors = SimpleNamespace(
        sample_rows=(0, 1),
        hidden=torch.zeros((2, 2)),
    )
    monkeypatch.setattr(tune_fhmoe, "_write_candidate", lambda *_args: None)
    monkeypatch.setattr(
        tune_fhmoe,
        "_runner",
        lambda *_args: lambda: torch.ones((2, 2)),
    )
    monkeypatch.setattr(
        tune_fhmoe,
        "_sampled_fp32_reference",
        lambda *_args: torch.full((2, 2), expected_value),
    )
    kwargs = dict(
        row={"token": "32"},
        tensors=tensors,
        contract=Contract(),
        config_file=tmp_path / "candidate.csv",
        swiglu_limit=10,
        tolerance=0.1,
    )

    if should_pass:
        assert _validate_sampled_reference(**kwargs) == 0
    else:
        with pytest.raises(RuntimeError, match="sampled FP32 validation failed"):
            _validate_sampled_reference(**kwargs)
