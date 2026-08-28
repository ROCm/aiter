# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import csv
from pathlib import Path

from aiter.aot.flydsl.mxfp4_moe import parse_csv
from aiter.ops.flydsl.moe_common import (
    DEFAULT_SITUV2_BETA,
    DEFAULT_SITUV2_LINEAR_BETA,
)

MODEL_CONFIGS = (
    Path(__file__).resolve().parents[2] / "aiter" / "configs" / "model_configs"
)


def test_aot_only_precompiles_configured_mxmoe_gemm1():
    config = MODEL_CONFIGS / "dsv3_fp4_tuned_fmoe.csv"
    with config.open(newline="") as stream:
        configured = {
            row["kernelName1"]
            for row in csv.DictReader(stream)
            if row.get("kernelName1", "").startswith("flydsl_mxmoe_g1_")
        }

    generated = {
        job["kernel_name"] for job in parse_csv(str(config)) if job["stage"] == 1
    }
    assert generated <= configured


def test_situv2_aot_covers_runtime_beta_contracts():
    config = MODEL_CONFIGS / "kimik3_a4w4_tuned_fmoe.csv"
    jobs = [
        job
        for job in parse_csv(str(config))
        if job["stage"] == 1 and job["act"] == "situv2"
    ]

    assert jobs
    assert {(job["situ_beta"], job["situ_linear_beta"]) for job in jobs} == {
        (1.0, 1.0),
        (DEFAULT_SITUV2_BETA, DEFAULT_SITUV2_LINEAR_BETA),
    }


def test_bm16_aot_matches_runtime_native_scale_layout():
    config = MODEL_CONFIGS / "glm5_fp4_tuned_fmoe.csv"
    jobs = [
        job for job in parse_csv(str(config)) if job["stage"] == 1 and job["BM"] == 16
    ]

    assert jobs
    assert all(job["native_scale_layout"] for job in jobs)
