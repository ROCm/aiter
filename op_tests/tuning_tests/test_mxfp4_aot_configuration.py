# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""MXFP4 AOT job and compile tests; the FlyDSL compiler is stubbed."""

import csv
import os
import sys
from pathlib import Path

import pytest
import torch

from op_tests.tuning_tests._mxfp4_test_process import require_gpu_hidden_process

require_gpu_hidden_process()

from op_tests.tuning_tests.test_mxfp4_tuner import import_moe_tuner

import_moe_tuner()

from aiter.aot.flydsl import mxfp4_moe as aot


@pytest.fixture(autouse=True)
def offline_environment(monkeypatch: "pytest.MonkeyPatch") -> "None":
    for name in ("ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"):
        assert os.environ.get(name) == "", name
    for name in (
        "AITER_MXFP4_TUNE_SWIGLU_LIMIT",
        "AITER_FLYDSL_STAGE2_FP8",
        "AITER_MXFP4_INTERMEDIATE",
    ):
        monkeypatch.delenv(name, raising=False)

    def forbidden(*args: "object", **kwargs: "object") -> "None":
        pytest.fail("offline AOT check reached GPU work")

    monkeypatch.setattr(torch.cuda, "_lazy_init", forbidden)
    monkeypatch.setattr(torch.cuda, "synchronize", forbidden)
    monkeypatch.setattr(torch.cuda, "Event", forbidden)
    monkeypatch.setattr(torch.cuda, "CUDAGraph", forbidden)


def winner_csv(
    tmp_path: "Path",
    *,
    activation: "str" = "swiglu",
    epilog: "str" = "scatter",
    bn: "int" = 128,
    bm: "int" = 128,
) -> "Path":
    suffix = "" if activation == "silu" else f"_{activation}"
    row = {
        "token": 128,
        "model_dim": 3072,
        "inter_dim": 512,
        "expert": 128,
        "topk": 4,
        "cu_num": 256,
        "dtype": "torch.bfloat16",
        "q_dtype_w": "torch.float4_e2m1fn_x2",
        "q_type": "QuantType.per_1x32",
        "kernelName1": f"flydsl_mxmoe_g1_a4w4_{bm}x128x256{suffix}",
        "kernelName2": f"flydsl_moe2_layout_afp4_wfp4_bf16_t{bm}x{bn}x128_{epilog}",
    }
    path = tmp_path / "winner.csv"
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=row)
        writer.writeheader()
        writer.writerow(row)
    return path


def test_custom_limit_changes_stage1_compile_jobs_without_changing_names(
    tmp_path: "Path",
) -> "None":
    path = winner_csv(tmp_path)
    defaults = [job for job in aot.parse_csv(str(path)) if job["stage"] == 1]
    custom = [
        job for job in aot.parse_csv(str(path), swiglu_limit=5) if job["stage"] == 1
    ]
    assert [job["swiglu_limit"] for job in defaults] == [7.0, 7.0]
    assert [job["swiglu_limit"] for job in custom] == [5.0, 5.0]
    assert [job["kernel_name"] for job in custom] == [
        job["kernel_name"] for job in defaults
    ]


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf"), "invalid"])
def test_invalid_limits_are_rejected_before_aot_jobs_are_returned(
    tmp_path: "Path", value: "object"
) -> "None":
    path = winner_csv(tmp_path)
    with pytest.raises(ValueError, match="finite positive"):
        aot.parse_csv(str(path), swiglu_limit=value)


def test_situv2_retains_default_parameter_variants_and_rejects_a_custom_limit(
    tmp_path: "Path",
) -> "None":
    path = winner_csv(tmp_path, activation="situv2")
    jobs = [job for job in aot.parse_csv(str(path)) if job["stage"] == 1]
    assert {
        (job["situ_beta"], job["situ_linear_beta"], job["swiglu_limit"]) for job in jobs
    } == {(1.0, 1.0, float("inf")), (4.0, 25.0, float("inf"))}
    with pytest.raises(ValueError, match="situv2"):
        aot.parse_csv(str(path), swiglu_limit=5)


def test_situv2_limit_error_names_the_csv_row_and_limit_source(
    monkeypatch: "pytest.MonkeyPatch", tmp_path: "Path"
) -> "None":
    path = winner_csv(tmp_path, activation="situv2")
    monkeypatch.setenv("AITER_MXFP4_TUNE_SWIGLU_LIMIT", "5")
    with pytest.raises(ValueError) as error:
        aot.parse_csv(str(path))
    message = str(error.value)
    assert str(path) in message and "_situv2" in message
    assert "AITER_MXFP4_TUNE_SWIGLU_LIMIT=5" in message


def test_batch_aot_ignores_a_stale_tuning_limit_for_the_default_corpus(
    monkeypatch: "pytest.MonkeyPatch", tmp_path: "Path"
) -> "None":
    from aiter.aot.flydsl.common import OpKind, _collect_aot_jobs_for

    (tmp_path / "situv2").mkdir()
    (tmp_path / "swiglu").mkdir()
    situv2 = winner_csv(tmp_path / "situv2", activation="situv2")
    swiglu = winner_csv(tmp_path / "swiglu")
    monkeypatch.setattr(aot, "DEFAULT_CSVS", [str(situv2), str(swiglu)])
    monkeypatch.setenv("AITER_MXFP4_TUNE_SWIGLU_LIMIT", "5")
    jobs = _collect_aot_jobs_for(OpKind.MXFP4_MOE)
    limits = {
        job["swiglu_limit"]
        for job in jobs
        if job["stage"] == 1 and "_swiglu" in job["kernel_name"]
    }
    assert limits == {7.0}
    assert os.environ["AITER_MXFP4_TUNE_SWIGLU_LIMIT"] == "5"


@pytest.mark.parametrize(
    "activation,default", [("swiglu", 7.0), ("silu", float("inf"))]
)
@pytest.mark.parametrize("value", [None, "", "5"])
def test_omitted_limit_matches_batch_environment_and_explicit_limit_takes_precedence(
    monkeypatch: "pytest.MonkeyPatch",
    tmp_path: "Path",
    activation: "str",
    default: "float",
    value: "object",
) -> "None":
    if value is not None:
        monkeypatch.setenv("AITER_MXFP4_TUNE_SWIGLU_LIMIT", value)
    path = winner_csv(tmp_path, activation=activation)
    jobs = [job for job in aot.parse_csv(str(path)) if job["stage"] == 1]
    expected = 5.0 if value == "5" else default
    assert {job["swiglu_limit"] for job in jobs} == {expected}
    explicit = [
        job for job in aot.parse_csv(str(path), swiglu_limit=3) if job["stage"] == 1
    ]
    assert {job["swiglu_limit"] for job in explicit} == {3.0}


@pytest.fixture
def external_compiler(monkeypatch: "pytest.MonkeyPatch") -> "list[dict[str, object]]":
    import flydsl.compiler

    invocations = []

    def compile_without_launch(*args: "object", **kwargs: "object") -> "None":
        assert os.environ["COMPILE_ONLY"] == "1"
        assert os.environ["FLYDSL_GPU_ARCH"] == "gfx950"
        invocations.append(args)

    monkeypatch.setattr(flydsl.compiler, "compile", compile_without_launch)
    return invocations


def test_standalone_accepts_custom_limit_and_compiles_the_discovered_jobs(
    monkeypatch: "pytest.MonkeyPatch",
    tmp_path: "Path",
    external_compiler: "list[dict[str, object]]",
    capsys: "pytest.CaptureFixture[str]",
) -> "None":
    path = winner_csv(tmp_path, epilog="atomic")
    monkeypatch.setattr(
        sys, "argv", ["mxfp4_moe", "--csv", str(path), "--swiglu-limit", "5"]
    )
    with pytest.raises(SystemExit) as exited:
        aot.main()
    assert exited.value.code == 0
    # 2 GEMM1 layouts + 2 bias x 2 valid-token-count GEMM2 variants.
    assert "Compiled:     6 ok, 0 failed" in capsys.readouterr().out
    assert external_compiler


@pytest.mark.parametrize("bn", [128, 256])
def test_enabled_fp4_scatter_jobs_match_production_and_preserve_bias_fallback(
    monkeypatch: "pytest.MonkeyPatch", tmp_path: "Path", bn: "int"
) -> "None":
    monkeypatch.setenv("AITER_MXFP4_INTERMEDIATE", "1")
    path = winner_csv(tmp_path, bn=bn)
    jobs = [job for job in aot.parse_csv(str(path)) if job["stage"] == 2]
    assert {(job["out_dtype"], job["enable_bias"], job["persist"]) for job in jobs} == {
        ("fp4", False, True),
        ("bf16", True, True),
    }


@pytest.mark.parametrize("bn", [128, 256])
def test_fp4_scatter_compile_entry_supplies_valid_packed_payload_and_e8m0_scales(
    monkeypatch: "pytest.MonkeyPatch",
    tmp_path: "Path",
    bn: "int",
    external_compiler: "list[dict[str, object]]",
) -> "None":
    monkeypatch.setenv("AITER_MXFP4_INTERMEDIATE", "1")
    path = winner_csv(tmp_path, bn=bn)
    job = next(
        job
        for job in aot.parse_csv(str(path))
        if job.get("out_dtype") == "fp4" and job["stage"] == 2
    )
    result = aot.compile_one_config(**job)
    assert result["compile_time"] is not None, result
    assert external_compiler


@pytest.mark.parametrize(
    "fp8,fp4,reduce_dtype,scatter_dtype",
    [
        ("", "", "bf16", "bf16"),
        ("1", "0", "fp8", "bf16"),
        ("0", "1", "bf16", "fp4"),
        ("1", "1", "fp8", "fp4"),
        ("true", "2", "bf16", "bf16"),
    ],
)
@pytest.mark.parametrize("limit", [None, 5])
def test_stage2_flags_select_matching_compile_jobs_and_leave_atomic_bf16(
    monkeypatch: "pytest.MonkeyPatch",
    tmp_path: "Path",
    external_compiler: "list[dict[str, object]]",
    fp8: "str",
    fp4: "str",
    reduce_dtype: "str",
    scatter_dtype: "str",
    limit: "float | None",
) -> "None":
    monkeypatch.setenv("AITER_FLYDSL_STAGE2_FP8", fp8)
    monkeypatch.setenv("AITER_MXFP4_INTERMEDIATE", fp4)
    for epilog, expected in (
        ("atomic", "bf16"),
        ("reduce", reduce_dtype),
        ("scatter", scatter_dtype),
    ):
        path = winner_csv(tmp_path, epilog=epilog)
        jobs = [
            job
            for job in aot.parse_csv(str(path), swiglu_limit=limit)
            if job["stage"] == 2
        ]
        no_bias = next(job for job in jobs if not job["enable_bias"])
        with_bias = next(job for job in jobs if job["enable_bias"])
        assert no_bias["out_dtype"] == expected
        assert with_bias["out_dtype"] == ("bf16" if epilog == "scatter" else expected)
        assert all(
            aot.compile_one_config(**job)["compile_time"] is not None for job in jobs
        )
    assert external_compiler


def test_fp4_flag_keeps_bn512_scatter_on_supported_bf16_path(
    monkeypatch: "pytest.MonkeyPatch", tmp_path: "Path"
) -> "None":
    monkeypatch.setenv("AITER_MXFP4_INTERMEDIATE", "1")
    path = winner_csv(tmp_path, bn=512)
    jobs = [job for job in aot.parse_csv(str(path)) if job["stage"] == 2]
    assert {job["out_dtype"] for job in jobs} == {"bf16"}
