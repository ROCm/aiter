# SPDX-License-Identifier: MIT

import argparse
import multiprocessing
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from aiter.utility import fp4_utils
from op_tests.tuners import tune_fhmoe
from op_tests.tuners.tune_fhmoe import (
    PROFILE_FIELDS,
    Contract,
    _fp8_group_quant_dequant,
    _matches,
    _parse_nonnegative_float,
    _parse_positive_float,
    _parse_tokens,
    _read_rows,
    _routed_silu,
    _sample_expert_ids,
    _sample_rows,
    _select_rows,
    _time_candidate,
    _validate_benchmark_token,
    _validate_existing_output,
    _validate_output_paths,
    _validate_sampled_reference,
    _validate_tuning_environment,
    _varying_e8m0_scales,
    _write_profile_rows,
    _write_rows,
    _write_tuned_output,
    stage1_candidates,
    stage2_candidates,
)


def _concurrent_tuned_output_update(
    barrier,
    input_path,
    output_path,
    fields,
    original,
    token,
    index,
    kernel,
):
    stale = [dict(row) for row in original]
    stale[index]["kernelName1"] = kernel
    barrier.wait(timeout=60)
    _write_tuned_output(
        input_path,
        output_path,
        fields,
        stale,
        {token: index},
        {token: dict(original[index])},
        Contract(),
    )


def test_parse_tokens():
    assert _parse_tokens("32,1,32,16") == [1, 16, 32]
    with pytest.raises(argparse.ArgumentTypeError):
        _parse_tokens("0")


@pytest.mark.parametrize("value", ("nan", "inf", "-1"))
def test_nonnegative_float_rejects_invalid_values(value: str):
    with pytest.raises(argparse.ArgumentTypeError, match="finite and nonnegative"):
        _parse_nonnegative_float(value)


@pytest.mark.parametrize("value", ("nan", "inf", "0", "-1"))
def test_positive_float_rejects_invalid_values(value: str):
    with pytest.raises(argparse.ArgumentTypeError, match="finite and positive"):
        _parse_positive_float(value)


def test_benchmark_token_respects_hy4_limit():
    _validate_benchmark_token(38836, [32768])
    with pytest.raises(ValueError, match="must not exceed 38836"):
        _validate_benchmark_token(38837, [32768])
    _validate_benchmark_token(None, [32768])
    with pytest.raises(ValueError, match="tokens must not exceed 38836"):
        _validate_benchmark_token(None, [32768, 131072])


@pytest.mark.parametrize(
    "name", ("AITER_FLYDSL_FORCE_REDUCE", "AITER_FLYDSL_STAGE2_FP8")
)
def test_canonical_tuning_rejects_runtime_overrides(
    monkeypatch: pytest.MonkeyPatch, name: str
):
    monkeypatch.setenv(name, "1")
    with pytest.raises(RuntimeError, match=f"{name}=0"):
        _validate_tuning_environment()


def test_large_m_samples_cover_boundary_experts():
    rows = _sample_rows(38836)
    assert rows[:32] == tuple(range(32))
    assert rows[-2:] == (19418, 38835)
    expert_ids = _sample_expert_ids(Contract(), rows)
    assert expert_ids == tuple(range(256))


def test_reference_scales_vary_across_all_weight_axes():
    scales = _varying_e8m0_scales((2, 3, 4), torch.device("cpu"))
    assert set(scales.unique().tolist()) == {0x7D, 0x7E, 0x7F, 0x80}
    assert scales[1, 0, 0] != scales[0, 0, 0]
    assert scales[0, 1, 0] != scales[0, 0, 0]
    assert scales[0, 0, 1] != scales[0, 0, 0]


def test_zero_swiglu_limit_keeps_routed_reference_unclamped():
    gate = torch.tensor([[2.0, -2.0]])
    up = torch.tensor([[3.0, -3.0]])
    expected = torch.nn.functional.silu(gate) * up
    assert torch.equal(_routed_silu(gate, up, 0), expected)


def test_sampled_reference_uses_fused_moe_mxfp8_rounding():
    amax = torch.tensor([383.0, 384.0, 400.0, 448.0])
    scales = fp4_utils.f32_to_fused_moe_mxfp8_scale(amax).view(torch.uint8)
    assert scales.tolist() == [0x7F, 0x80, 0x80, 0x80]


def test_input_and_fused_intermediate_use_distinct_mxfp8_rounding(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(tune_fhmoe.dtypes, "fp8", torch.float8_e4m3fnuz)
    values = torch.zeros(32)
    values[0] = 1.5
    values[1] = 2**-17

    input_qdq = _fp8_group_quant_dequant(values)
    fused_qdq = _fp8_group_quant_dequant(values, fused_intermediate=True)

    assert input_qdq[1] != 0
    assert fused_qdq[1] == 0


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
    }
    actual_tokens = {
        int(row["token"]) for row in rows if _matches(row, contract, int(row["token"]))
    }
    assert actual_tokens == expected_tokens
    assert fields[-2:] == ["kernelName1", "kernelName2"]

    for block_m in (32, 64, 128):
        assert stage1_candidates(block_m, quick=True)
        full_stage2 = stage2_candidates(block_m, quick=False)
        assert full_stage2
        assert stage2_candidates(block_m, quick=True)
        assert all(
            tune_fhmoe.get_flydsl_kernel_params(name).get("persist") is not True
            and int(tune_fhmoe.get_flydsl_kernel_params(name).get("b_nt", 0)) == 0
            for name in full_stage2
        )


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
    assert b"\r\n" not in output.read_bytes()


def test_tuned_output_merges_concurrent_row_updates(tmp_path: Path):
    root = Path(__file__).resolve().parents[2]
    fields, all_rows = _read_rows(root / "aiter/configs/tuned_fhmoe.csv")
    contract = Contract()
    original = [
        dict(row)
        for row in all_rows
        if _matches(row, contract, int(row["token"])) and row["token"] in ("1", "2")
    ]
    input_path = tmp_path / "input.csv"
    output_path = tmp_path / "output.csv"
    _write_rows(input_path, fields, original)

    context = multiprocessing.get_context("spawn")
    barrier = context.Barrier(2)
    common = (barrier, input_path, output_path, fields, original)
    first = context.Process(
        target=_concurrent_tuned_output_update,
        args=(*common, 1, 0, "first_update"),
    )
    second = context.Process(
        target=_concurrent_tuned_output_update,
        args=(*common, 2, 1, "second_update"),
    )
    first.start()
    second.start()
    processes = (first, second)
    try:
        for process in processes:
            process.join(120)
        assert not any(
            process.is_alive() for process in processes
        ), "concurrent tuner update workers timed out: " + ", ".join(
            process.name for process in processes if process.is_alive()
        )
        assert first.exitcode == 0
        assert second.exitcode == 0
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
        for process in processes:
            process.join(5)

    _, merged = _read_rows(output_path)
    assert merged[0]["kernelName1"] == "first_update"
    assert merged[1]["kernelName1"] == "second_update"


def test_tuner_main_commits_completed_tokens_and_resumes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    import sys

    root = Path(__file__).resolve().parents[2]
    fields, rows = _read_rows(root / "aiter/configs/tuned_fhmoe.csv")
    input_path = tmp_path / "input.csv"
    output_path = tmp_path / "output.csv"
    _write_rows(input_path, fields, rows)

    fail_token_two = True

    def fake_tune_row(row, *_args, **_kwargs):
        token = int(row["token"])
        if fail_token_two and token == 2:
            raise RuntimeError("simulated interruption")
        tuned = dict(row)
        tuned["kernelName1"] = f"tuned_token_{token}"
        return tuned, []

    monkeypatch.setattr(tune_fhmoe, "get_gfx_runtime", lambda: "gfx950")
    monkeypatch.setattr(tune_fhmoe, "get_cu_num", lambda: Contract().cu_num)
    monkeypatch.setattr(tune_fhmoe, "_make_tensors", lambda *_args: object())
    monkeypatch.setattr(tune_fhmoe, "tune_row", fake_tune_row)
    monkeypatch.setattr(
        tune_fhmoe,
        "_validate_sampled_reference",
        lambda *_args, **_kwargs: 0.0,
    )
    monkeypatch.setattr(tune_fhmoe.torch.cuda, "empty_cache", lambda: None)
    monkeypatch.delenv("AITER_FLYDSL_FORCE_REDUCE", raising=False)
    monkeypatch.delenv("AITER_FLYDSL_STAGE2_FP8", raising=False)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "tune_fhmoe.py",
            "-i",
            str(input_path),
            "-o",
            str(output_path),
            "--tokens",
            "1,2",
            "--quick",
        ],
    )
    with pytest.raises(RuntimeError, match="simulated interruption"):
        tune_fhmoe.main()

    _, partial_rows = _read_rows(output_path)
    token_one = next(row for row in partial_rows if _matches(row, Contract(), 1))
    token_two = next(row for row in partial_rows if _matches(row, Contract(), 2))
    assert token_one["kernelName1"] == "tuned_token_1"
    assert token_two["kernelName1"] != "tuned_token_2"

    fail_token_two = False
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "tune_fhmoe.py",
            "-i",
            str(output_path),
            "-o",
            str(output_path),
            "--tokens",
            "2",
            "--quick",
        ],
    )
    tune_fhmoe.main()

    _, resumed_rows = _read_rows(output_path)
    token_one = next(row for row in resumed_rows if _matches(row, Contract(), 1))
    token_two = next(row for row in resumed_rows if _matches(row, Contract(), 2))
    assert token_one["kernelName1"] == "tuned_token_1"
    assert token_two["kernelName1"] == "tuned_token_2"


def test_tuner_main_uses_representative_runtime_token(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    import sys

    root = Path(__file__).resolve().parents[2]
    fields, rows = _read_rows(root / "aiter/configs/tuned_fhmoe.csv")
    input_path = tmp_path / "input.csv"
    output_path = tmp_path / "output.csv"
    profile_path = tmp_path / "profile.csv"
    _write_rows(input_path, fields, rows)
    observed_tokens = []

    def make_tensors(_contract, token, *_args):
        observed_tokens.append(token)
        return object()

    monkeypatch.setattr(tune_fhmoe, "get_gfx_runtime", lambda: "gfx950")
    monkeypatch.setattr(tune_fhmoe, "get_cu_num", lambda: Contract().cu_num)
    monkeypatch.setattr(tune_fhmoe, "_make_tensors", make_tensors)
    monkeypatch.setattr(
        tune_fhmoe,
        "tune_row",
        lambda row, *_args, **_kwargs: (
            dict(row),
            [{"stage": "baseline", "kernel": "stage1|stage2", "ms": 1.0}],
        ),
    )
    monkeypatch.setattr(
        tune_fhmoe,
        "_validate_sampled_reference",
        lambda *_args, **_kwargs: 0.0,
    )
    monkeypatch.setattr(tune_fhmoe.torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "tune_fhmoe.py",
            "-i",
            str(input_path),
            "-o",
            str(output_path),
            "--tokens",
            "32768",
            "--benchmark-token",
            "38836",
            "--profile",
            str(profile_path),
            "--quick",
        ],
    )

    tune_fhmoe.main()

    assert observed_tokens == [38836]
    _, output_rows = _read_rows(output_path)
    assert any(_matches(row, Contract(), 32768) for row in output_rows)
    profile_fields, profile_rows = _read_rows(profile_path)
    assert profile_fields == [
        "token",
        "runtime_token",
        "stage",
        "kernel",
        "ms",
        "error",
    ]
    assert profile_rows[0]["token"] == "32768"
    assert profile_rows[0]["runtime_token"] == "38836"


def test_tuned_output_rejects_stale_same_token_update(tmp_path: Path):
    root = Path(__file__).resolve().parents[2]
    fields, all_rows = _read_rows(root / "aiter/configs/tuned_fhmoe.csv")
    contract = Contract()
    original = [
        dict(row)
        for row in all_rows
        if row["token"] == "32" and _matches(row, contract, 32)
    ]
    input_path = tmp_path / "input.csv"
    output_path = tmp_path / "output.csv"
    _write_rows(input_path, fields, original)
    baseline = {32: dict(original[0])}

    first = [dict(row) for row in original]
    first[0]["kernelName1"] = "first_winner"
    _write_tuned_output(
        input_path, output_path, fields, first, {32: 0}, baseline, contract
    )

    stale = [dict(row) for row in original]
    stale[0]["kernelName1"] = "stale_winner"
    with pytest.raises(RuntimeError, match="changed during tuning"):
        _write_tuned_output(
            input_path, output_path, fields, stale, {32: 0}, baseline, contract
        )

    _, preserved = _read_rows(output_path)
    assert preserved[0]["kernelName1"] == "first_winner"


def test_existing_output_must_match_input_baseline(tmp_path: Path):
    root = Path(__file__).resolve().parents[2]
    fields, all_rows = _read_rows(root / "aiter/configs/tuned_fhmoe.csv")
    contract = Contract()
    baseline = next(
        dict(row)
        for row in all_rows
        if row["token"] == "32" and _matches(row, contract, 32)
    )
    output_path = tmp_path / "output.csv"
    _write_rows(output_path, fields, [baseline])
    _validate_existing_output(output_path, fields, {32: baseline}, contract)

    changed = dict(baseline)
    changed["kernelName1"] = "newer_output"
    _write_rows(output_path, fields, [changed])
    with pytest.raises(RuntimeError, match="differs from input"):
        _validate_existing_output(output_path, fields, {32: baseline}, contract)


def test_profile_path_must_not_alias_input_or_output(tmp_path: Path):
    input_path = tmp_path / "input.csv"
    output_path = tmp_path / "output.csv"
    input_path.write_text("x\n")
    output_path.write_text("x\n")

    with pytest.raises(ValueError, match="must differ"):
        _validate_output_paths(input_path, output_path, input_path)
    with pytest.raises(ValueError, match="must differ"):
        _validate_output_paths(input_path, output_path, output_path)

    hardlink = tmp_path / "profile-hardlink.csv"
    hardlink.hardlink_to(output_path)
    with pytest.raises(ValueError, match="must differ"):
        _validate_output_paths(input_path, output_path, hardlink)


def test_profile_rows_merge_and_replace_runtime_measurement(tmp_path: Path):
    profile_path = tmp_path / "profile.csv"
    first = {
        "token": 32,
        "runtime_token": 32,
        "stage": "baseline",
        "kernel": "first",
        "ms": 1.0,
    }
    second = {
        "token": 32768,
        "runtime_token": 38836,
        "stage": "baseline",
        "kernel": "second",
        "ms": 2.0,
    }
    replacement = dict(first, kernel="replacement", ms=0.5)

    _write_profile_rows(profile_path, [first])
    _write_profile_rows(profile_path, [second])
    _write_profile_rows(profile_path, [replacement])

    fields, rows = _read_rows(profile_path)
    assert fields == list(PROFILE_FIELDS)
    assert {(row["token"], row["runtime_token"]) for row in rows} == {
        ("32", "32"),
        ("32768", "38836"),
    }
    token_32 = next(row for row in rows if row["token"] == "32")
    assert token_32["kernel"] == "replacement"
    assert token_32["ms"] == "0.5"


def test_profile_rows_migrate_legacy_schema(tmp_path: Path):
    profile_path = tmp_path / "profile.csv"
    _write_rows(
        profile_path,
        ["token", "stage", "kernel", "ms", "error"],
        [
            {
                "token": "32",
                "stage": "baseline",
                "kernel": "legacy",
                "ms": "1.0",
                "error": "",
            }
        ],
    )

    _write_profile_rows(
        profile_path,
        [
            {
                "token": 32768,
                "runtime_token": 38836,
                "stage": "baseline",
                "kernel": "new",
                "ms": 2.0,
            }
        ],
    )

    fields, rows = _read_rows(profile_path)
    assert fields == list(PROFILE_FIELDS)
    legacy = next(row for row in rows if row["token"] == "32")
    assert legacy["runtime_token"] == "32"


def test_output_symlink_is_rejected(tmp_path: Path):
    input_path = tmp_path / "input.csv"
    target = tmp_path / "target.csv"
    output_link = tmp_path / "output.csv"
    input_path.write_text("x\n")
    target.write_text("x\n")
    output_link.symlink_to(target)

    with pytest.raises(ValueError, match="must not be a symlink"):
        _validate_output_paths(input_path, output_link, None)


def test_baseline_only_empty_selection_preserves_same_token_update(tmp_path: Path):
    root = Path(__file__).resolve().parents[2]
    fields, all_rows = _read_rows(root / "aiter/configs/tuned_fhmoe.csv")
    contract = Contract()
    original = [
        dict(row)
        for row in all_rows
        if row["token"] == "32" and _matches(row, contract, 32)
    ]
    path = tmp_path / "in_place.csv"
    _write_rows(path, fields, original)
    stale = [dict(row) for row in original]

    concurrent = [dict(row) for row in original]
    concurrent[0]["kernelName1"] = "concurrent_update"
    _write_rows(path, fields, concurrent)

    _write_tuned_output(path, path, fields, stale, {}, {}, contract)

    _, preserved = _read_rows(path)
    assert preserved[0]["kernelName1"] == "concurrent_update"


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


def test_tune_row_rejects_aliased_stage2_baseline(tmp_path: Path):
    row = {
        "token": "32",
        "block_m": "32",
        "kernelName1": "flydsl_moe1_afp8_wfp8_bf16_t32x64x256_w3_gui_fp8",
        "kernelName2": "flydsl_moe2_afp8_wfp8_bf16_t32x128x128_atomic_bnt2",
    }
    with pytest.raises(ValueError, match="invalid baseline stage2"):
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
            baseline_only=True,
        )


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


@pytest.mark.parametrize("baseline_only", (False, True))
def test_unstable_baseline_timing_fails_before_tuning(
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
        lambda *_args: (float("inf"), torch.ones(1)),
    )

    with pytest.raises(RuntimeError, match="unstable or non-finite timing"):
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
        ("stage1_fast", "stage2"): (0.9, torch.ones(2)),
        ("stage1_fast", "stage2_fast"): (0.6, torch.ones(2)),
    }
    monkeypatch.setattr(tune_fhmoe, "_write_candidate", write_candidate)
    monkeypatch.setattr(
        tune_fhmoe,
        "_runner",
        lambda *_args: (active["kernelName1"], active["kernelName2"]),
    )

    def time_candidate(run, *_args):
        if run == ("stage1", "stage2_broken"):
            raise RuntimeError("compile failed")
        return timings[run]

    monkeypatch.setattr(tune_fhmoe, "_time_candidate", time_candidate)
    monkeypatch.setattr(
        tune_fhmoe,
        "stage2_candidates",
        lambda *_args: [
            "stage2_broken",
            "stage2_fast",
            "stage2_wrong",
            "stage2_nan",
        ],
    )
    monkeypatch.setattr(tune_fhmoe, "stage1_candidates", lambda *_args: ["stage1_fast"])

    tuned, profile = tune_fhmoe.tune_row(
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
    broken = next(row for row in profile if row["kernel"] == "stage2_broken")
    assert broken["ms"] == float("inf")
    assert broken["error"] == float("inf")


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


def test_pair_beam_finds_joint_kernel_improvement(
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
        ("stage1", "stage2"): 1.0,
        ("stage1", "stage2_joint"): 1.1,
        ("stage1_joint", "stage2"): 1.1,
        ("stage1_joint", "stage2_joint"): 0.5,
    }
    monkeypatch.setattr(tune_fhmoe, "_write_candidate", write_candidate)
    monkeypatch.setattr(
        tune_fhmoe,
        "_runner",
        lambda *_args: (active["kernelName1"], active["kernelName2"]),
    )
    monkeypatch.setattr(
        tune_fhmoe,
        "_time_candidate",
        lambda run, *_args: (timings[run], torch.ones(1)),
    )
    monkeypatch.setattr(
        tune_fhmoe, "stage2_candidates", lambda *_args: ["stage2_joint"]
    )
    monkeypatch.setattr(
        tune_fhmoe, "stage1_candidates", lambda *_args: ["stage1_joint"]
    )

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

    assert tuned["kernelName1"] == "stage1_joint"
    assert tuned["kernelName2"] == "stage2_joint"


def test_full_search_checks_pairs_outside_quick_beam(
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
    stage1 = [f"stage1_{index}" for index in range(5)]
    stage2 = [f"stage2_{index}" for index in range(5)]

    def write_candidate(_path, candidate):
        active.clear()
        active.update(candidate)

    def timing(run):
        s1, s2 = run
        if (s1, s2) == (stage1[-1], stage2[-1]):
            return 0.1
        if s1 == "stage1" and s2 == "stage2":
            return 10.0
        if s1 == "stage1":
            return 1.0 + stage2.index(s2)
        if s2 == "stage2":
            return 1.0 + stage1.index(s1)
        return 9.0

    monkeypatch.setattr(tune_fhmoe, "_write_candidate", write_candidate)
    monkeypatch.setattr(
        tune_fhmoe,
        "_runner",
        lambda *_args: (active["kernelName1"], active["kernelName2"]),
    )
    monkeypatch.setattr(
        tune_fhmoe,
        "_time_candidate",
        lambda run, *_args: (timing(run), torch.ones(1)),
    )
    monkeypatch.setattr(tune_fhmoe, "stage1_candidates", lambda *_args: stage1)
    monkeypatch.setattr(tune_fhmoe, "stage2_candidates", lambda *_args: stage2)

    tuned, _ = tune_fhmoe.tune_row(
        row,
        tensors=object(),
        contract=Contract(),
        config_file=tmp_path / "candidate.csv",
        quick=False,
        warmup_ms=1,
        rep_ms=1,
        tolerance=0,
        swiglu_limit=10,
        min_improvement_pct=1,
    )

    assert tuned["kernelName1"] == stage1[-1]
    assert tuned["kernelName2"] == stage2[-1]


def test_time_candidate_rejects_nonfinite_repeat(
    monkeypatch: pytest.MonkeyPatch,
):
    outputs = iter((torch.ones(1), torch.tensor([float("nan")])))
    monkeypatch.setattr(
        tune_fhmoe.triton.testing,
        "do_bench",
        lambda *_args, **_kwargs: 1.0,
    )

    timing, _ = _time_candidate(lambda: next(outputs), warmup_ms=1, rep_ms=1)

    assert timing == float("inf")


def test_time_candidate_uses_independent_stability_tolerance(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        tune_fhmoe.triton.testing,
        "do_bench",
        lambda *_args, **_kwargs: 1.0,
    )

    def measure(stability_tolerance: float) -> float:
        outputs = iter(
            (
                torch.ones(1),
                torch.full((1,), 1.01),
                torch.full((1,), 1.01),
                torch.full((1,), 1.01),
            )
        )
        timing, _ = _time_candidate(
            lambda: next(outputs),
            warmup_ms=1,
            rep_ms=1,
            stability_tolerance=stability_tolerance,
        )
        return timing

    assert measure(2e-2) == 1.0
    assert measure(5e-3) == float("inf")


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
    kwargs = {
        "row": {"token": "32"},
        "tensors": tensors,
        "contract": Contract(),
        "config_file": tmp_path / "candidate.csv",
        "swiglu_limit": 10,
        "tolerance": 0.1,
    }

    if should_pass:
        assert _validate_sampled_reference(**kwargs) == 0
    else:
        with pytest.raises(RuntimeError, match="sampled FP32 validation failed"):
            _validate_sampled_reference(**kwargs)
