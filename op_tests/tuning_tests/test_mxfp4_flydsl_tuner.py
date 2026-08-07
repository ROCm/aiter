# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import os
from types import SimpleNamespace

import pandas as pd
import pytest

from aiter import test_common
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


def test_candidate_rows_cover_native_and_layout_gemm2_kernels():
    candidates = _tuner()._candidate_rows(_shape(1024))

    assert candidates
    assert all(
        candidate["kernelName1"].startswith("flydsl_mxmoe_g1_a4w4_")
        for candidate in candidates
    )
    assert all(
        candidate["kernelName2"].startswith(
            ("flydsl_mxmoe_g2_a4w4_", "flydsl_moe2_layout_")
        )
        for candidate in candidates
    )
    assert any(
        candidate["kernelName2"].startswith("flydsl_mxmoe_g2_a4w4_")
        for candidate in candidates
    )
    assert any(
        candidate["kernelName2"].startswith("flydsl_moe2_layout_")
        for candidate in candidates
    )
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
    assert candidate["err1"] == candidate["err2"] == 0.0123


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


def test_tune_one_shape_keeps_e2e_selection_metric():
    tuner = _tuner()
    row = _shape(1)
    candidate_a = tuner._candidate_row(row, 16, "g1_a", "g2_a")
    candidate_b = tuner._candidate_row(row, 32, "g1_b", "g2_b")
    tuner._candidate_rows = lambda _row: [candidate_a, candidate_b]
    tuner._prepare_case = lambda *_args: {}
    tuner._torch_ref = lambda *_args: object()

    def fake_run_candidate(_row, candidate, _args, **_kwargs):
        if candidate["kernelName1"] == "g1_a":
            candidate["us"] = 10.0
            return 100.0
        candidate["us"] = 20.0
        return 50.0

    tuner._run_candidate = fake_run_candidate
    best, profiles = tuner._tune_one_shape(
        row,
        SimpleNamespace(timeout=0),
    )

    assert best["kernelName1"] == "g1_b"
    assert [profile["e2e_us"] for profile in profiles] == [100.0, 50.0]


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
