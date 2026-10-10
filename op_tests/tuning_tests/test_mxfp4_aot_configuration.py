# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""AOT discovery/compile checks with GPU visibility empty.

Normal ROCm/FlyDSL imports are required. Import discovery is synthetic; the
external FlyDSL compiler is replaced so no GPU kernel compiles or launches.
"""

import csv
import os
import sys

import pytest
import torch

from op_tests.tuning_tests._controlled_tuner_import import import_moe_tuner

import_moe_tuner()

from aiter.aot.flydsl import mxfp4_moe as aot


@pytest.fixture(autouse=True)
def offline_environment(monkeypatch):
    for name in ("ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"):
        assert os.environ.get(name) == "", name
    for name in (
        "AITER_MXFP4_TUNE_SWIGLU_LIMIT",
        "AITER_FLYDSL_STAGE2_FP8",
        "AITER_MXFP4_INTERMEDIATE",
    ):
        monkeypatch.delenv(name, raising=False)

    def forbidden(*args, **kwargs):
        pytest.fail("offline AOT check reached GPU work")

    monkeypatch.setattr(torch.cuda, "_lazy_init", forbidden)
    monkeypatch.setattr(torch.cuda, "synchronize", forbidden)
    monkeypatch.setattr(torch.cuda, "Event", forbidden)
    monkeypatch.setattr(torch.cuda, "CUDAGraph", forbidden)


def winner_csv(tmp_path, *, activation="swiglu", epilog="scatter", bn=128, bm=128):
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


def test_custom_limit_changes_stage1_compile_jobs_without_changing_names(tmp_path):
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
def test_invalid_limits_are_rejected_before_aot_jobs_are_returned(tmp_path, value):
    path = winner_csv(tmp_path)
    with pytest.raises(ValueError, match="finite positive"):
        aot.parse_csv(str(path), swiglu_limit=value)


def test_situv2_retains_default_parameter_variants_and_rejects_a_custom_limit(tmp_path):
    path = winner_csv(tmp_path, activation="situv2")
    jobs = [job for job in aot.parse_csv(str(path)) if job["stage"] == 1]
    assert {
        (job["situ_beta"], job["situ_linear_beta"], job["swiglu_limit"]) for job in jobs
    } == {(1.0, 1.0, float("inf")), (4.0, 25.0, float("inf"))}
    with pytest.raises(ValueError, match="situv2"):
        aot.parse_csv(str(path), swiglu_limit=5)


@pytest.mark.parametrize(
    "activation,default", [("swiglu", 7.0), ("silu", float("inf"))]
)
@pytest.mark.parametrize("value", [None, "", "5"])
def test_omitted_limit_matches_batch_environment_and_explicit_limit_takes_precedence(
    monkeypatch, tmp_path, activation, default, value
):
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
def external_compiler(monkeypatch):
    import flydsl.compiler

    invocations = []

    def compile_without_launch(*args, **kwargs):
        assert os.environ["COMPILE_ONLY"] == "1"
        assert os.environ["FLYDSL_GPU_ARCH"] == "gfx950"
        invocations.append(args)

    monkeypatch.setattr(flydsl.compiler, "compile", compile_without_launch)
    return invocations


def test_standalone_accepts_custom_limit_and_compiles_the_discovered_jobs(
    monkeypatch, tmp_path, external_compiler, capsys
):
    path = winner_csv(tmp_path, epilog="atomic")
    monkeypatch.setattr(
        sys, "argv", ["mxfp4_moe", "--csv", str(path), "--swiglu-limit", "5"]
    )
    with pytest.raises(SystemExit) as exited:
        aot.main()
    assert exited.value.code == 0
    assert "Compiled:     4 ok, 0 failed" in capsys.readouterr().out
    assert external_compiler


@pytest.mark.parametrize("bn", [128, 256])
def test_enabled_fp4_scatter_jobs_match_production_and_preserve_bias_fallback(
    monkeypatch, tmp_path, bn
):
    monkeypatch.setenv("AITER_MXFP4_INTERMEDIATE", "1")
    path = winner_csv(tmp_path, bn=bn)
    jobs = [job for job in aot.parse_csv(str(path)) if job["stage"] == 2]
    assert {(job["out_dtype"], job["enable_bias"], job["persist"]) for job in jobs} == {
        ("fp4", False, True),
        ("bf16", True, True),
    }


@pytest.mark.parametrize("bn", [128, 256])
def test_fp4_scatter_compile_entry_supplies_valid_packed_payload_and_e8m0_scales(
    monkeypatch, tmp_path, bn, external_compiler
):
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
    monkeypatch,
    tmp_path,
    external_compiler,
    fp8,
    fp4,
    reduce_dtype,
    scatter_dtype,
    limit,
):
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


def test_fp4_flag_keeps_bn512_scatter_on_supported_bf16_path(monkeypatch, tmp_path):
    monkeypatch.setenv("AITER_MXFP4_INTERMEDIATE", "1")
    path = winner_csv(tmp_path, bn=512)
    jobs = [job for job in aot.parse_csv(str(path)) if job["stage"] == 2]
    assert {job["out_dtype"] for job in jobs} == {"bf16"}


def test_selected_csv_batch_collection_uses_matching_environment_and_excludes_other_csvs(
    monkeypatch, tmp_path, external_compiler
):
    from aiter.aot.flydsl.common import collect_aot_jobs

    unused_dir = tmp_path / "unused"
    unused_dir.mkdir()
    winner_csv(unused_dir, activation="situv2")
    selected = winner_csv(tmp_path)
    monkeypatch.setenv("AITER_MXFP4_TUNE_SWIGLU_LIMIT", "5")
    monkeypatch.setenv("AITER_FLYDSL_STAGE2_FP8", "1")
    monkeypatch.setenv("AITER_MXFP4_INTERMEDIATE", "1")
    jobs = collect_aot_jobs([str(selected), str(selected)], aot.parse_csv)
    assert len(jobs) == 4
    assert {job["swiglu_limit"] for job in jobs if job["stage"] == 1} == {5.0}
    assert {
        (job["out_dtype"], job["enable_bias"]) for job in jobs if job["stage"] == 2
    } == {
        ("fp4", False),
        ("bf16", True),
    }
    assert all(
        aot.compile_one_config(**job)["compile_time"] is not None for job in jobs
    )
    assert external_compiler
