# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import os
from types import SimpleNamespace

import pandas as pd
import pytest

from aiter import test_common
from aiter.ops.flydsl.moe_kernels import get_flydsl_kernel_params
from aiter.ops.flydsl.mxfp4_kname import _parse_mxfp4_g1_kname
from csrc.ck_gemm_moe_2stages_codegen import gemm_moe_tune
from csrc.ck_gemm_moe_2stages_codegen.gemm_moe_tune import Mxfp4FlydslTuner

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

RESULT_COLUMNS = [
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


def _shape(token):
    return {
        "gfx": "gfx950",
        "cu_num": 256,
        "token": token,
        "model_dim": 6144,
        "inter_dim": 512,
        "expert": 257,
        "topk": 9,
        "act_type": "ActivationType.Silu",
        "dtype": "torch.bfloat16",
        "q_dtype_a": "torch.float4_e2m1fn_x2",
        "q_dtype_w": "torch.float4_e2m1fn_x2",
        "q_type": "QuantType.per_1x32",
        "use_g1u1": 1,
        "doweight_stage1": 0,
    }


def _tuner():
    tuner = Mxfp4FlydslTuner.__new__(Mxfp4FlydslTuner)
    tuner.keys = KEYS
    return tuner


def test_candidate_rows_cover_every_gemm2_family():
    candidates = _tuner()._candidate_rows(_shape(1024))

    assert candidates
    assert all(
        candidate["kernelName1"].startswith("flydsl_mxmoe_g1_a4w4_")
        for candidate in candidates
    )
    families = (
        # registry family: what the tuned configs actually ship
        "flydsl_moe2_afp4_wfp4_bf16_",
        # native mxmoe family
        "flydsl_mxmoe_g2_a4w4_",
        # layout (v2) family
        "flydsl_moe2_layout_",
    )
    assert all(
        candidate["kernelName2"].startswith(families) for candidate in candidates
    )
    for family in families:
        assert any(
            candidate["kernelName2"].startswith(family) for candidate in candidates
        ), family
    parsed = [
        _parse_mxfp4_g1_kname(candidate["kernelName1"]) for candidate in candidates
    ]
    assert any(
        candidate["BN"] == 128 and candidate["interleave"] for candidate in parsed
    )
    assert any(
        candidate["BN"] == 128 and not candidate["interleave"] for candidate in parsed
    )
    assert all(
        not candidate["interleave"] for candidate in parsed if candidate["BN"] == 256
    )


def test_token_one_excludes_inaccurate_bm16_inline_quant():
    candidates = _tuner()._candidate_rows(_shape(1))

    assert candidates
    assert all(candidate["block_m"] != 16 for candidate in candidates)


def test_a4w4_stage1_inputs_match_candidate_layout():
    separated_weight = object()
    separated_scale = object()
    interleaved_weight = object()
    interleaved_scale = object()
    data = {
        "w1_a16": separated_weight,
        "w1s_a16": separated_scale,
        "w1_a16_interleaved": interleaved_weight,
        "w1s_a16_interleaved": interleaved_scale,
    }

    assert Mxfp4FlydslTuner._a4w4_stage1_inputs(data, False) == (
        separated_weight,
        separated_scale,
    )
    assert Mxfp4FlydslTuner._a4w4_stage1_inputs(data, True) == (
        interleaved_weight,
        interleaved_scale,
    )


def test_a8w4_candidates_lock_gemm2_and_block_m():
    row = {
        **_shape(8),
        "model_dim": 3584,
        "expert": 896,
        "topk": 16,
        "act_type": "ActivationType.Situv2",
        "q_dtype_a": "torch.float8_e4m3fn",
        "block_m": 32,
        "kernelName2": "flydsl_moe2_afp8_wfp4_bf16_t32x256x256_reduce_persist",
    }

    candidates = _tuner()._candidate_rows(row)

    assert len(candidates) == 6
    assert {candidate["block_m"] for candidate in candidates} == {32}
    assert {candidate["kernelName2"] for candidate in candidates} == {
        row["kernelName2"]
    }
    assert all(
        candidate["kernelName1"].startswith("flydsl_mxmoe_g1_a8w4_32x256x256")
        and "_il_" in candidate["kernelName1"]
        and "_fp8out_" in candidate["kernelName1"]
        and "_situv2" in candidate["kernelName1"]
        for candidate in candidates
    )


def _a8w4_row():
    return {
        **_shape(8),
        "model_dim": 3584,
        "expert": 896,
        "topk": 16,
        "act_type": "ActivationType.Situv2",
        "q_dtype_a": "torch.float8_e4m3fn",
        "block_m": 32,
        "kernelName2": "flydsl_moe2_afp8_wfp4_bf16_t32x256x256_reduce_persist",
    }


@pytest.mark.parametrize(
    ("row_extra", "args_stage", "expected"),
    [
        # A tuned row naming a replacement GEMM2 is a GEMM1 sweep by default.
        (
            {"block_m": 32, "kernelName2": "flydsl_moe2_afp4_wfp4_bf16_t32"},
            None,
            "gemm1",
        ),
        # A shape-only row has nothing locked, so both stages are searched.
        ({}, None, "both"),
        # An explicit --tune-stage always wins over the inference.
        (
            {"block_m": 32, "kernelName2": "flydsl_moe2_afp4_wfp4_bf16_t32"},
            "both",
            "both",
        ),
        ({}, "gemm2", "gemm2"),
    ],
)
def test_resolve_tune_stage(row_extra, args_stage, expected):
    row = {**_shape(1024), **row_extra}
    args = (
        SimpleNamespace()
        if args_stage is None
        else SimpleNamespace(tune_stage=args_stage)
    )

    assert Mxfp4FlydslTuner._resolve_tune_stage(row, args) == expected


def test_gemm2_stage_locks_gemm1_and_sweeps_a4w4_gemm2():
    row = {
        **_shape(1024),
        "block_m": 32,
        "kernelName1": "flydsl_mxmoe_g1_a4w4_32x256x256",
    }

    candidates = _tuner()._candidate_rows(row, "gemm2")

    assert candidates
    assert {candidate["kernelName1"] for candidate in candidates} == {
        row["kernelName1"]
    }
    assert {candidate["block_m"] for candidate in candidates} == {32}
    names = [candidate["kernelName2"] for candidate in candidates]
    assert len(set(names)) == len(names)
    assert any(name.startswith("flydsl_mxmoe_g2_a4w4_") for name in names)
    assert any(name.startswith("flydsl_moe2_layout_") for name in names)


def test_gemm2_stage_sweeps_fp8_gemm2_for_a8w4_gemm1():
    """A8W4 GEMM1 emits fp8, so its GEMM2 partners are the afp8 family."""
    row = {
        **_a8w4_row(),
        "kernelName1": ("flydsl_mxmoe_g1_a8w4_32x256x256_il_fp8out_situv2"),
    }
    row.pop("kernelName2")
    tuner = _tuner()
    kn1 = next(
        name
        for name in tuner._a8w4_gemm1_knames(row, 32)
        if "_xcd" not in name.rsplit("_", 1)[-1]
    )
    row["kernelName1"] = kn1

    candidates = tuner._candidate_rows(row, "gemm2")

    assert candidates
    assert {candidate["kernelName1"] for candidate in candidates} == {kn1}
    assert all(
        candidate["kernelName2"].startswith("flydsl_moe2_afp8_wfp4_bf16_t32x")
        for candidate in candidates
    )
    # Only tiles that divide the shape survive.
    params = [
        get_flydsl_kernel_params(candidate["kernelName2"]) for candidate in candidates
    ]
    assert all(param is not None for param in params)
    assert all(
        row["model_dim"] % param["tile_n"] == 0
        and row["inter_dim"] % param["tile_k"] == 0
        for param in params
    )


def test_gemm2_stage_drops_tiles_that_do_not_divide_the_shape():
    # inter_dim=384 is 128-aligned but not 256-aligned, so tile_k=256 is illegal.
    row = {
        **_a8w4_row(),
        "inter_dim": 384,
        "kernelName1": "flydsl_mxmoe_g1_a8w4_32x256x256_il_fp8out_situv2",
    }
    row.pop("kernelName2")
    tuner = _tuner()
    row["kernelName1"] = tuner._a8w4_gemm1_knames(row, 32)[0]

    candidates = tuner._candidate_rows(row, "gemm2")

    assert candidates
    assert all(
        get_flydsl_kernel_params(candidate["kernelName2"])["tile_k"] == 128
        for candidate in candidates
    )


def test_gemm2_stage_requires_a_locked_gemm1():
    row = {**_shape(1024), "block_m": 32}

    with pytest.raises(ValueError, match="locked kernelName1"):
        _tuner()._candidate_rows(row, "gemm2")


def test_baseline_config_requires_a_single_stage_sweep():
    tuner = _tuner()

    with pytest.raises(ValueError, match="--tune-stage gemm1 or --tune-stage gemm2"):
        tuner.run(
            SimpleNamespace(tune_stage="both", baseline_config="/tmp/baseline.csv")
        )


def test_gemm2_stage_rejects_a_non_replacement_gemm1():
    row = {
        **_shape(1024),
        "block_m": 32,
        "kernelName1": "moe_ck2stages_gemm1_test",
    }

    with pytest.raises(ValueError, match="replacement GEMM1"):
        _tuner()._candidate_rows(row, "gemm2")


def test_a8w4_both_stage_sweeps_gemm2_as_well():
    row = _a8w4_row()
    tuner = _tuner()

    locked = tuner._candidate_rows(row, "gemm1")
    joint = tuner._candidate_rows(row, "both")

    assert {candidate["kernelName2"] for candidate in locked} == {row["kernelName2"]}
    assert len({candidate["kernelName2"] for candidate in joint}) > 1
    assert {candidate["kernelName1"] for candidate in joint} == {
        candidate["kernelName1"] for candidate in locked
    }


def test_extract_stage_kernel_times_requires_only_the_swept_stage():
    kernel_times = {"gemm1_a4w4_port_kernel": 12.5, "unrelated_kernel": 3.0}

    us1, us2 = Mxfp4FlydslTuner._extract_stage_kernel_times(kernel_times, ("GEMM1",))
    assert (us1, us2) == (12.5, 0)

    with pytest.raises(RuntimeError, match="GEMM2"):
        Mxfp4FlydslTuner._extract_stage_kernel_times(kernel_times)


@pytest.mark.parametrize(
    "q_dtype_a",
    ["torch.float4_e2m1fn_x2", "torch.float8_e4m3fn"],
)
def test_ck_gemm2_rows_keep_original_gemm1(q_dtype_a):
    row = {
        **_shape(1),
        "q_dtype_a": q_dtype_a,
        "block_m": 32,
        "kernelName2": "moe_ck2stages_gemm2_test",
        "run_1stage": 0,
    }

    assert _tuner()._candidate_rows(row) == []


def test_multiple_input_csvs_are_merged_and_deduplicated(tmp_path):
    untuned_path = tmp_path / "untuned.csv"
    tuned_path = tmp_path / "tuned.csv"
    untuned_row = {
        **{
            key: value
            for key, value in _shape(1).items()
            if key not in ("gfx", "cu_num")
        },
        "block_m": 32,
        "kernelName2": "flydsl_moe2_afp4_wfp4_bf16_t32x128x256_reduce",
        "run_1stage": 0,
    }
    pd.DataFrame([untuned_row]).to_csv(untuned_path, index=False)
    tuned_row = {
        **_shape(1),
        "block_m": 32,
        "kernelName1": "old",
        "kernelName2": "flydsl_moe2_afp4_wfp4_bf16_t32x128x256_reduce",
        "run_1stage": 0,
        "us": 1.0,
    }
    pd.DataFrame([tuned_row, {**tuned_row, **_shape(2)}]).to_csv(
        tuned_path, index=False
    )

    tuner = _tuner()
    tuner.get_gfx = lambda: "gfx950"
    tuner.get_cu_num = lambda: 256
    merged = tuner.get_untuned_gemm_list(
        os.pathsep.join((str(untuned_path), str(tuned_path)))
    )

    assert list(merged["token"]) == [1, 2]
    assert list(merged.columns) == KEYS + ["block_m", "kernelName2", "run_1stage"]


def test_trace_perf_can_return_per_kernel_gpu_times():
    class DeviceType:
        def __str__(self):
            return "DeviceType.CUDA"

    class Profiler:
        def events(self):
            events = []
            samples = [
                (10.0, 4.0, 2.0),
                (20.0, 6.0, 2.0),
                (30.0, 8.0, 2.0),
            ]
            for g1_us, g2_us, aux_us in samples:
                for name, us in (
                    ("gemm1_a4w4_port_test", g1_us),
                    ("mfma_moe2_test", g2_us),
                    ("moe_sorting_test", aux_us),
                ):
                    events.append(
                        SimpleNamespace(
                            name=name,
                            self_cpu_time_total=0.0,
                            self_device_time_total=us,
                            device_type=DeviceType(),
                            device_index=0,
                        )
                    )
            return events

    avg_us, kernel_times = test_common.get_trace_perf(
        Profiler(), 3, return_kernel_times=True
    )

    assert avg_us == pytest.approx(34.0)
    assert kernel_times == pytest.approx(
        {
            "gemm1_a4w4_port_test": 25.0,
            "mfma_moe2_test": 7.0,
            "moe_sorting_test": 2.0,
        }
    )


def test_extract_stage_kernel_times_ignores_auxiliary_kernels():
    us1, us2 = Mxfp4FlydslTuner._extract_stage_kernel_times(
        {
            "gemm1_a4w4_port_fp4_test": 26.00556,
            "mfma_moe2_fp4_test": 16.29006,
            "mxfp4_moe_quant": 80.0,
            "moe_sorting": 40.0,
        }
    )

    assert us1 == 26.0056
    assert us2 == 16.2901


def test_extract_stage_kernel_times_rejects_missing_target():
    with pytest.raises(RuntimeError, match="GEMM2"):
        Mxfp4FlydslTuner._extract_stage_kernel_times({"gemm1_a4w4_port_fp4_test": 26.0})


def test_run_candidate_records_stage_performance(monkeypatch):
    tuner = _tuner()
    row = _shape(1)
    candidate = tuner._candidate_row(
        row,
        16,
        "flydsl_mxmoe_g1_a4w4_16x256x256_f16in_nt",
        "flydsl_moe2_afp4_wfp4_bf16_t16x128x256_atomic",
    )
    tuner._port_e2e = lambda *_args: object()
    tuner._calculate_candidate_performance = lambda *_args: (16.06, 229371.86)
    monkeypatch.setattr(
        gemm_moe_tune,
        "cosine_diff_compare",
        lambda *_args, **_kwargs: 0.0123,
    )

    def fake_run_perftest(fn, **kwargs):
        assert kwargs["return_kernel_times"] is True
        return (
            fn(),
            145.8825,
            {
                "gemm1_a4w4_port_fp4_test": 26.00556,
                "mfma_moe2_fp4_test": 16.29006,
                "mxfp4_moe_quant": 80.0,
            },
        )

    monkeypatch.setattr(test_common, "run_perftest", fake_run_perftest)
    e2e_us = tuner._run_candidate(
        row,
        candidate,
        SimpleNamespace(errRatio=0.1, warmup=5, iters=101),
        data={},
        ref=object(),
    )

    assert e2e_us == 145.8825
    assert candidate["us1"] == 26.0056
    assert candidate["us2"] == 16.2901
    assert candidate["us"] == 42.2957
    assert candidate["tflops"] == 16.06
    assert candidate["bw"] == 229371.86
    assert candidate["err1"] == candidate["err2"] == "1.2%"


def test_run_candidate_rejects_nonfinite_cosine_error(monkeypatch):
    tuner = _tuner()
    row = _shape(1)
    candidate = tuner._candidate_row(
        row,
        32,
        "flydsl_mxmoe_g1_a4w4_32x256x256",
        "flydsl_moe2_afp4_wfp4_bf16_t32x128x256_reduce",
    )
    tuner._port_e2e = lambda *_args: object()
    monkeypatch.setattr(
        gemm_moe_tune,
        "cosine_diff_compare",
        lambda *_args, **_kwargs: float("nan"),
    )

    with pytest.raises(RuntimeError, match="cosine err_ratio nan"):
        tuner._run_candidate(
            row,
            candidate,
            SimpleNamespace(errRatio=0.1, warmup=1, iters=1),
            data={},
            ref=object(),
        )


def _stage_selection_tuner(row, us1_by_kernel, us2_by_kernel, e2e_by_kernel):
    """A tuner whose two candidates report fixed per-stage and e2e times."""
    tuner = _tuner()
    candidate_a = tuner._candidate_row(row, 16, "g1_a", "g2_a")
    candidate_b = tuner._candidate_row(row, 32, "g1_b", "g2_b")
    tuner._candidate_rows = lambda _row, _stage="auto": [candidate_a, candidate_b]
    tuner._prepare_case = lambda *_args: {}
    tuner._torch_ref = lambda *_args: object()

    def fake_run_candidate(_row, candidate, _args, **_kwargs):
        name = candidate["kernelName1"]
        candidate["us1"] = us1_by_kernel[name]
        candidate["us2"] = us2_by_kernel[name]
        candidate["us"] = candidate["us1"] + candidate["us2"]
        return e2e_by_kernel[name]

    tuner._run_candidate = fake_run_candidate
    return tuner


def test_tune_one_shape_keeps_e2e_selection_metric():
    row = _shape(1)
    tuner = _stage_selection_tuner(
        row,
        us1_by_kernel={"g1_a": 4.0, "g1_b": 15.0},
        us2_by_kernel={"g1_a": 6.0, "g1_b": 5.0},
        e2e_by_kernel={"g1_a": 100.0, "g1_b": 50.0},
    )

    best, profiles = tuner._tune_one_shape(row, SimpleNamespace(timeout=0))

    # Joint tuning ranks by end-to-end latency, not by either stage alone.
    assert best["kernelName1"] == "g1_b"
    assert [profile["e2e_us"] for profile in profiles] == [100.0, 50.0]


def test_errors_are_percent_strings_like_the_rest_of_the_csv():
    tuner = _tuner()

    assert tuner._candidate_errors(_shape(1), 0.011706, "both") == {
        "err1": "1.2%",
        "err2": "1.2%",
    }


@pytest.mark.parametrize(
    ("stage", "baseline_column", "expected"),
    [
        ("gemm1", "_baseline_err2", {"err1": "1.2%", "err2": "0.4%"}),
        ("gemm2", "_baseline_err1", {"err1": "0.4%", "err2": "1.2%"}),
    ],
)
def test_locked_stage_keeps_its_baseline_error(stage, baseline_column, expected):
    """A single-stage sweep must not overwrite the locked stage's own error."""
    row = {**_shape(1), baseline_column: "0.4%"}

    assert _tuner()._candidate_errors(row, 0.011706, stage) == expected


def test_locked_stage_error_falls_back_to_the_measurement():
    # No --baseline-config: the e2e cosine is the only number available.
    assert _tuner()._candidate_errors(_shape(1), 0.011706, "gemm2") == {
        "err1": "1.2%",
        "err2": "1.2%",
    }


def test_first_candidate_gets_jit_build_headroom(monkeypatch):
    """A kernel-sized --timeout must not abort the one-time JIT module build."""
    row = _shape(1)
    tuner = _stage_selection_tuner(
        row,
        us1_by_kernel={"g1_a": 4.0, "g1_b": 15.0},
        us2_by_kernel={"g1_a": 6.0, "g1_b": 5.0},
        e2e_by_kernel={"g1_a": 100.0, "g1_b": 50.0},
    )
    import signal

    alarms = []
    monkeypatch.setattr(signal, "signal", lambda *_args: None)
    monkeypatch.setattr(signal, "alarm", lambda seconds: alarms.append(seconds) or 0)

    tuner._tune_one_shape(row, SimpleNamespace(timeout=90))

    armed = [seconds for seconds in alarms if seconds]
    assert armed == [90 + Mxfp4FlydslTuner.JIT_BUILD_GRACE_SECONDS, 90]


def test_tune_one_shape_gemm1_stage_ranks_by_us1():
    row = _shape(1)
    tuner = _stage_selection_tuner(
        row,
        us1_by_kernel={"g1_a": 4.0, "g1_b": 15.0},
        us2_by_kernel={"g1_a": 6.0, "g1_b": 5.0},
        e2e_by_kernel={"g1_a": 100.0, "g1_b": 50.0},
    )

    best, _profiles = tuner._tune_one_shape(
        row, SimpleNamespace(timeout=0, tune_stage="gemm1")
    )

    assert best["kernelName1"] == "g1_a"


def test_tune_one_shape_gemm2_stage_ranks_by_us2():
    row = _shape(1)
    tuner = _stage_selection_tuner(
        row,
        us1_by_kernel={"g1_a": 4.0, "g1_b": 15.0},
        us2_by_kernel={"g1_a": 6.0, "g1_b": 5.0},
        e2e_by_kernel={"g1_a": 100.0, "g1_b": 50.0},
    )

    best, _profiles = tuner._tune_one_shape(
        row, SimpleNamespace(timeout=0, tune_stage="gemm2")
    )

    assert best["kernelName2"] == "g2_b"


def test_calculate_candidate_performance_accepts_csv_dtype_strings():
    tuner = _tuner()
    row = _shape(1)
    candidate = tuner._candidate_row(
        row,
        16,
        "flydsl_mxmoe_g1_a4w4_16x256x256_f16in_nt_xcd4",
        "flydsl_moe2_afp4_wfp4_bf16_t16x256x256_atomic_xcd4",
    )

    tflops, bw = tuner._calculate_candidate_performance(row, candidate, 27.9554)

    assert tflops == 6.08
    assert bw == 86758.72


def test_post_process_writes_candidate_performance_profile(tmp_path):
    tuner = _tuner()
    tuner.columns = KEYS + RESULT_COLUMNS
    candidate = tuner._candidate_row(
        _shape(1),
        16,
        "flydsl_mxmoe_g1_a4w4_16x256x256_f16in_nt",
        "flydsl_moe2_afp4_wfp4_bf16_t16x128x256_atomic",
    )
    candidate.update(
        {
            "us1": 26.0056,
            "us2": 16.2901,
            "us": 42.2957,
            "tflops": 16.06,
            "bw": 229371.86,
        }
    )
    tuner._profile_rows = [{**candidate, "e2e_us": 145.8825}]
    profile_file = tmp_path / "profile.csv"

    result = tuner.post_process(
        [candidate],
        SimpleNamespace(profile_file=str(profile_file)),
    )
    profile = pd.read_csv(profile_file)

    assert list(result.columns) == tuner.columns
    assert len(profile) == 1
    assert profile.loc[0, "us1"] == 26.0056
    assert profile.loc[0, "us2"] == 16.2901
    assert profile.loc[0, "us"] == 42.2957
    assert profile.loc[0, "e2e_us"] == 145.8825


def test_two_wave_specialization_is_restricted_to_bn64():
    # Mirrors mxfp4_gemm1_kernels._assert_supported: num_waves==2 needs the
    # effective BN64 tile, which is BM32 A4W4 non-inline separated.
    assert Mxfp4FlydslTuner._g1_num_waves(32, 64, False, False) == (2, 4)
    assert Mxfp4FlydslTuner._g1_num_waves(32, 128, False, False) == (4,)
    assert Mxfp4FlydslTuner._g1_num_waves(32, 256, False, False) == (4,)
    assert Mxfp4FlydslTuner._g1_num_waves(128, 64, False, False) == (4,)
    assert Mxfp4FlydslTuner._g1_num_waves(32, 64, True, False) == (4,)
    assert Mxfp4FlydslTuner._g1_num_waves(32, 64, False, True) == (4,)


def test_k_wave_ceiling_follows_the_wave_budget():
    # The kernel bound is num_waves * k_wave <= 8, so k_wave=4 is reachable
    # only from the two-wave form. A hardcoded 4 * kw <= 8 used to cap every
    # candidate at k_wave=2.
    row = _shape(1)
    assert Mxfp4FlydslTuner._g1_k_waves(row, 32, False, False, num_waves=4) == (1, 2)
    assert Mxfp4FlydslTuner._g1_k_waves(row, 32, False, False, num_waves=2) == (
        1,
        2,
        4,
    )


def test_decode_tail_sweeps_bn64_two_wave_candidates():
    tuner = _tuner()
    small = _shape(1)
    names = tuner._a4w4_gemm1_knames(small, 32)

    # (num_waves=2, k_wave=4) measured fastest of every BN64 form at tok<8.
    assert "flydsl_mxmoe_g1_a4w4_32x64x256_kw4_w2" in names
    assert any(name.endswith("_w2") for name in names)

    # Above the bound BN64 stays behind G1_TRY_BN64, so the candidate set is
    # unchanged and the extra axis costs nothing.
    large = _shape(2048)
    assert Mxfp4FlydslTuner._g1_bns(large, 32, False, False) == (128, 256)
    assert not any(name.endswith("_w2") for name in tuner._a4w4_gemm1_knames(large, 32))
