# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

from pathlib import Path
from unittest import mock

import pandas as pd

from aiter.aot.flydsl.common import OpKind, _collect_aot_jobs_for
from aiter.aot.flydsl.mxscale_preshuffle import parse_csv
from aiter.ops.flydsl.gemm_tune.flydsl_gemm_mxscale_preshuffle_common import (
    instance_valid,
    kernelInstance,
    parse_kernel_name,
)
from aiter.ops.flydsl.mxscale_preshuffle_kernels import (
    _TUNED_CACHE,
    get_mxscale_preshuffle_config,
)

ROOT = Path(__file__).resolve().parents[2]
MODEL_CONFIG = (
    ROOT
    / "aiter/configs/model_configs/"
    "minimax_m3_dense_mxfp8_mxscale_preshuffle_tuned_gemm.csv"
)
EP4_TUNED = (
    ROOT
    / "aiter/configs/model_configs/"
    "minimax_m3_ep4_mxfp8_tuned_fmoe.csv"
)
EP4_UNTUNED = (
    ROOT
    / "aiter/configs/model_configs/"
    "minimax_m3_ep4_mxfp8_untuned_fmoe.csv"
)


def test_minimax_m3_dense_manifest_is_structurally_legal():
    frame = pd.read_csv(MODEL_CONFIG)
    assert len(frame) == 147
    assert set(frame["M"]) == {
        1,
        2,
        4,
        8,
        12,
        16,
        24,
        32,
        40,
        48,
        56,
        64,
        128,
        256,
        512,
        1024,
        2048,
        4096,
        8192,
        8320,
        16384,
    }
    assert len(frame.drop_duplicates(["M", "N", "K", "a_dtype", "b_dtype"])) == 147

    for row in frame.itertuples():
        parsed = parse_kernel_name(row.kernelName)
        assert parsed is not None
        instance = kernelInstance(
            tile_m=parsed["tile_m"],
            tile_n=parsed["tile_n"],
            tile_k=parsed["tile_k"],
            a_dtype=parsed["a_dtype"],
            b_dtype=parsed["b_dtype"],
            out_dtype=parsed["out_dtype"],
            waves_per_eu=parsed["waves_per_eu"],
            xcd_swizzle=parsed["xcd_swizzle"],
            split_k=parsed["split_k"],
        )
        assert instance_valid(instance)
        assert int(row.splitK) == instance.split_k
        assert int(row.N) % instance.tile_n == 0
        assert int(row.K) % instance.tile_k == 0
        if instance.split_k > 1:
            k_per_split = int(row.K) // instance.split_k
            assert int(row.K) % instance.split_k == 0
            assert k_per_split % instance.tile_k == 0
            assert k_per_split % 256 == 0
            assert int(row.M) * int(row.N) * instance.split_k * 4 < (1 << 32)


def test_aot_parser_preserves_every_exact_signature():
    frame = pd.read_csv(MODEL_CONFIG)
    jobs = parse_csv(str(MODEL_CONFIG))

    assert len(jobs) == len(frame) == 147
    assert {
        (job["m"], job["n"], job["k"], job["a_dtype"], job["b_dtype"])
        for job in jobs
    } == {
        (int(row.M), int(row.N), int(row.K), row.a_dtype, row.b_dtype)
        for row in frame.itertuples()
    }


def test_shared_aot_driver_collects_mxscale_jobs():
    jobs = _collect_aot_jobs_for(OpKind.MXSCALE_PRESHUFFLE)
    assert len(jobs) == 147


def test_runtime_lookup_is_exact_signature_only(tmp_path):
    row = pd.read_csv(MODEL_CONFIG).iloc[[0]]
    tune_file = tmp_path / "mxscale.csv"
    row.to_csv(tune_file, index=False)
    _TUNED_CACHE.clear()
    try:
        with (
            mock.patch(
                "aiter.jit.utils.chip_info.get_gfx_runtime",
                return_value="gfx950",
            ),
            mock.patch("aiter.jit.utils.chip_info.get_cu_num", return_value=256),
        ):
            exact = get_mxscale_preshuffle_config(
                int(row.iloc[0]["M"]),
                int(row.iloc[0]["N"]),
                int(row.iloc[0]["K"]),
                a_dtype="fp8",
                b_dtype="fp8",
                tuned_file=str(tune_file),
            )
            missing = get_mxscale_preshuffle_config(
                int(row.iloc[0]["M"]) + 1,
                int(row.iloc[0]["N"]),
                int(row.iloc[0]["K"]),
                a_dtype="fp8",
                b_dtype="fp8",
                tuned_file=str(tune_file),
            )
        assert exact is not None
        assert exact["kernelName"] == row.iloc[0]["kernelName"]
        assert missing is None
    finally:
        _TUNED_CACHE.clear()


def test_ep4_post2_rows_are_tuned_and_cover_requested_tokens():
    tuned = pd.read_csv(EP4_TUNED)
    untuned = pd.read_csv(EP4_UNTUNED)

    assert len(tuned) == 9
    assert set(tuned["_tag"]) == {"post2_tuned"}
    assert set(tuned["token"]) == {1, 2, 4, 8, 16, 32, 64, 8192, 16384}
    assert set(tuned["topk"]) == {4}
    assert tuned["kernelName1"].str.endswith("_fp8").all()
    assert set(untuned["token"]) == {1, 2, 4, 8, 16, 32, 64, 8192, 16384}
    assert set(untuned["topk"]) == {4}
