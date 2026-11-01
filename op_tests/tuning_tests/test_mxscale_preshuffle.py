# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

from pathlib import Path
from unittest import mock

import pandas as pd
import torch

from aiter.aot.flydsl import mxscale_preshuffle as mxscale_aot
from aiter.aot.flydsl.common import OpKind, _collect_aot_jobs_for
from aiter.ops.flydsl.gemm_tune.flydsl_gemm_mxscale_preshuffle_common import (
    instance_valid,
    kernelInstance,
    mxscale_preshuffle_1x32_params,
    parse_kernel_name,
)
from aiter.ops.flydsl.mxscale_preshuffle_kernels import (
    _TUNED_CACHE,
    _heuristic_tile,
    _mx_scale_nbytes,
    flydsl_mxscale_preshuffle_gemm,
    gemm_a8w8_mxscale_preshuffle_1x32_flydsl,
    get_mxscale_preshuffle_config,
)
from aiter.ops.shuffle import shuffle_scale_a16w4

ROOT = Path(__file__).resolve().parents[2]
MODEL_CONFIG = (
    ROOT / "aiter/configs/model_configs/"
    "minimax_m3_dense_mxfp8_mxscale_preshuffle_tuned_gemm.csv"
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
        assert not parsed["fused_split_k"]
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
    jobs = mxscale_aot.parse_csv(str(MODEL_CONFIG))

    assert len(jobs) == len(frame) == 147
    assert {
        (job["m"], job["n"], job["k"], job["a_dtype"], job["b_dtype"]) for job in jobs
    } == {
        (int(row.M), int(row.N), int(row.K), row.a_dtype, row.b_dtype)
        for row in frame.itertuples()
    }


def test_aot_compile_one_config_accepts_parsed_jobs():
    job = mxscale_aot.parse_csv(str(MODEL_CONFIG))[0]
    assert job["fused_split_k"] is False
    with mock.patch(
        "aiter.ops.flydsl.mxscale_preshuffle_kernels.flydsl_mxscale_preshuffle_gemm"
    ) as gemm:
        result = mxscale_aot.compile_one_config(**job)
    assert result["compile_time"] is not None
    gemm.assert_called_once()


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


def test_heuristic_keeps_split_k_one_when_the_grid_fills_the_cus():
    scales = (
        ("a_scale_block", "1x32"),
        ("a_scale_layout", "mfma_tile"),
        ("b_scale_block", "1x32"),
        ("b_scale_layout", "mfma_tile"),
    )
    instance = _heuristic_tile("fp8", "fp8", 8192, 6144, 6144, scales)
    assert instance is not None
    assert instance.split_k == 1


def test_unknown_kernel_name_is_not_parsed():
    assert parse_kernel_name("flydsl_mxpsh_not_a_tile") is None
    parsed = parse_kernel_name("flydsl_mxpsh_32x32x256_F6_F4_B16_w0_x0_sk1")
    assert parsed is not None
    assert parsed["a_dtype"] == "fp6"


def test_rejects_unknown_dtype():
    dummy = torch.empty((32, 256), dtype=torch.uint8)
    out = torch.empty((32, 64), dtype=torch.bfloat16)
    try:
        flydsl_mxscale_preshuffle_gemm(
            dummy,
            dummy,
            dummy,
            dummy,
            out,
            a_dtype="fp7",
            b_dtype="fp8",
            tile_m=32,
            tile_n=16,
            tile_k=256,
            a_scale_block="1x32",
            b_scale_block="1x32",
            a_scale_layout="mfma_tile",
            b_scale_layout="mfma_tile",
        )
    except ValueError as err:
        assert "fp7" in str(err)
        return
    raise AssertionError("unknown a_dtype must be rejected")


def test_per_1x32_rejects_blockscale_tile_m():
    M, N, K = 16, 128, 256
    A = torch.empty((M, K), dtype=torch.uint8)
    B = torch.empty((N, K), dtype=torch.uint8)
    a_scale = torch.empty(_mx_scale_nbytes(M, K), dtype=torch.uint8)
    b_scale = torch.empty(_mx_scale_nbytes(N, K), dtype=torch.uint8)
    out = torch.empty((M, N), dtype=torch.bfloat16)
    try:
        gemm_a8w8_mxscale_preshuffle_1x32_flydsl(
            A,
            B,
            a_scale,
            b_scale,
            out,
            a_dtype="fp8",
            b_dtype="fp8",
            tile_m=16,
            tile_n=128,
            tile_k=256,
        )
    except ValueError as err:
        assert "multiple of 32" in str(err)
        return
    raise AssertionError("tile_m=16 must be rejected on the per-1x32 path")


def test_mx_scale_nbytes_matches_shuffle_scale_a16w4():
    for rows, k in ((32, 128), (32, 256), (32, 384), (64, 768)):
        raw = torch.empty((rows, k // 32), dtype=torch.uint8)
        assert shuffle_scale_a16w4(raw, 1, False).numel() == _mx_scale_nbytes(rows, k)


def test_rejects_unshuffled_scale_nbytes():
    M, N, K = 32, 64, 256
    A = torch.empty((M, K), dtype=torch.uint8)
    B = torch.empty((N, K), dtype=torch.uint8)
    a_scale = torch.empty((1, K // 32), dtype=torch.uint8)
    b_scale = torch.empty((32, K // 32), dtype=torch.uint8)
    out = torch.empty((M, N), dtype=torch.bfloat16)
    try:
        gemm_a8w8_mxscale_preshuffle_1x32_flydsl(
            A,
            B,
            a_scale,
            b_scale,
            out,
            a_dtype="fp8",
            b_dtype="fp8",
            tile_m=32,
            tile_n=16,
            tile_k=256,
        )
    except ValueError as err:
        assert "a_scale nbytes" in str(err)
        return
    raise AssertionError("mismatched a_scale must be rejected")


def _uint8_mnk(M, N, K, out_dtype=torch.bfloat16):
    A = torch.empty((M, K), dtype=torch.uint8)
    B = torch.empty((N, K), dtype=torch.uint8)
    a_scale = torch.empty(_mx_scale_nbytes(M, K), dtype=torch.uint8)
    b_scale = torch.empty(_mx_scale_nbytes(N, K), dtype=torch.uint8)
    out = torch.empty((M, N), dtype=out_dtype)
    return A, B, a_scale, b_scale, out


def test_require_tuned_miss_raises():
    A, B, a_scale, b_scale, out = _uint8_mnk(32, 64, 256)
    with mock.patch(
        "aiter.ops.flydsl.mxscale_preshuffle_kernels.get_mxscale_preshuffle_config",
        return_value=None,
    ):
        try:
            gemm_a8w8_mxscale_preshuffle_1x32_flydsl(
                A,
                B,
                a_scale,
                b_scale,
                out,
                a_dtype="fp8",
                b_dtype="fp8",
                require_tuned=True,
            )
        except RuntimeError as err:
            assert "no exact mxscale_preshuffle tune" in str(err)
            return
    raise AssertionError("require_tuned miss must raise")


def test_out_dtype_mismatch_is_tuned_miss():
    A, B, a_scale, b_scale, out = _uint8_mnk(32, 64, 256, out_dtype=torch.float16)
    try:
        gemm_a8w8_mxscale_preshuffle_1x32_flydsl(
            A,
            B,
            a_scale,
            b_scale,
            out,
            a_dtype="fp8",
            b_dtype="fp8",
            config={"kernelName": "flydsl_mxpsh_32x16x256_F8_F8_B16_w0_x0_sk1"},
            require_tuned=True,
        )
    except RuntimeError as err:
        assert "no exact mxscale_preshuffle tune" in str(err)
        assert "fp16" in str(err)
        return
    raise AssertionError("bf16 kernelName vs fp16 Out must be a tuned miss")


def test_rejects_noncontiguous_out():
    M, N, K = 32, 64, 256
    A = torch.empty((M, K), dtype=torch.uint8)
    B = torch.empty((N, K), dtype=torch.uint8)
    a_scale = torch.empty(_mx_scale_nbytes(M, K), dtype=torch.uint8)
    b_scale = torch.empty(_mx_scale_nbytes(N, K), dtype=torch.uint8)
    out = torch.empty((M, N * 2), dtype=torch.bfloat16)[:, ::2]
    try:
        gemm_a8w8_mxscale_preshuffle_1x32_flydsl(
            A,
            B,
            a_scale,
            b_scale,
            out,
            a_dtype="fp8",
            b_dtype="fp8",
            tile_m=32,
            tile_n=16,
            tile_k=256,
        )
    except ValueError as err:
        assert "Out must be contiguous" in str(err)
        return
    raise AssertionError("noncontiguous Out must be rejected")


def test_rejects_bad_splitk_workspace():
    M, N, K = 32, 64, 512
    A = torch.empty((M, K), dtype=torch.uint8)
    B = torch.empty((N, K), dtype=torch.uint8)
    a_scale = torch.empty((1,), dtype=torch.uint8)
    b_scale = torch.empty((1,), dtype=torch.uint8)
    out = torch.empty((M, N), dtype=torch.bfloat16)
    bad = torch.empty((1, M, N), dtype=torch.float32)
    with (
        mock.patch(
            "aiter.ops.flydsl.mxscale_preshuffle_kernels._gemm_exe",
            return_value=object(),
        ),
        mock.patch(
            "aiter.ops.flydsl.kernels.tensor_shim.ptr_arg",
            side_effect=lambda t: t,
        ),
    ):
        try:
            flydsl_mxscale_preshuffle_gemm(
                A,
                B,
                a_scale,
                b_scale,
                out,
                a_dtype="fp8",
                b_dtype="fp8",
                tile_m=32,
                tile_n=16,
                tile_k=256,
                split_k=2,
                splitk_workspace=bad,
                stream=mock.Mock(),
                a_scale_block="1x32",
                b_scale_block="1x32",
                a_scale_layout="mfma_tile",
                b_scale_layout="mfma_tile",
            )
        except ValueError as err:
            assert "splitk_workspace" in str(err)
            return
    raise AssertionError("bad splitk_workspace must be rejected")


def test_shared_row_helper_rejects_fused_split_k():
    try:
        mxscale_preshuffle_1x32_params(
            "flydsl_mxpsh_32x16x256_F8_F8_B16_w0_x0_sk2_fsk",
            a_dtype="fp8",
            b_dtype="fp8",
        )
    except ValueError as err:
        assert "fused split-K" in str(err)
        return
    raise AssertionError("_fsk names must be rejected for mfma_tile 1x32")
