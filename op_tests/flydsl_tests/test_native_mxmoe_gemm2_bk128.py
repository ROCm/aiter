# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import csv

import pytest
import torch

from aiter.ops.flydsl.mxfp4_gemm2_kernels import _assert_supported
from aiter.ops.flydsl.mxfp4_kname import (
    _parse_mxfp4_g2_kname,
    parse_g2_kname_any,
)


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        (
            "flydsl_mxmoe_g2_a4w4_16x256x128_atomic_nt",
            {
                "BM": 16,
                "BN": 256,
                "BK": 128,
                "atomic": True,
                "use_nt": True,
                "mxfp4out": False,
                "cshuffle": False,
            },
        ),
        (
            "flydsl_mxmoe_g2_a4w4_128x256x256_f4out",
            {
                "BM": 128,
                "BN": 256,
                "BK": 256,
                "atomic": False,
                "use_nt": False,
                "mxfp4out": True,
                "cshuffle": False,
            },
        ),
        (
            "flydsl_mxmoe_g2_a4w4_64x256x128_cshuffle",
            {
                "BM": 64,
                "BN": 256,
                "BK": 128,
                "atomic": False,
                "use_nt": False,
                "mxfp4out": False,
                "cshuffle": True,
            },
        ),
    ],
)
def test_native_g2_parser_preserves_tiles_and_flags(name, expected):
    parsed = _parse_mxfp4_g2_kname(name)
    for key, value in expected.items():
        assert parsed[key] == value
    unified = parse_g2_kname_any(name)
    assert unified["v2"] is False
    for key, value in expected.items():
        assert unified[key] == value


def test_stage2_fw_forwards_native_tiles(monkeypatch):
    from aiter import fused_moe

    called = {}

    def capture(*args, **kwargs):
        called.update(kwargs)
        return args[9]

    monkeypatch.setattr(fused_moe, "_mxfp4_a4w4_stage2", capture)

    inter = torch.empty((32, 192), dtype=torch.uint8)
    w1 = torch.empty((2, 768, 128), dtype=torch.uint8)
    w2 = torch.empty((2, 256, 192), dtype=torch.uint8)
    ids = torch.zeros(32, dtype=torch.int32)
    weights = torch.ones(32, dtype=torch.float32)
    scale = torch.empty(1, dtype=torch.uint8)
    out = torch.empty((1, 256), dtype=torch.bfloat16)

    result = fused_moe._mxfp4_a4w4_stage2_fw(
        inter,
        w1,
        w2,
        ids,
        ids,
        ids,
        out,
        2,
        w2_scale=scale,
        a2_scale=scale,
        block_m=32,
        sorted_weights=weights,
        kernelName2="flydsl_mxmoe_g2_a4w4_32x256x128_atomic_nt",
        reverse_sorted=ids,
    )

    assert result is out
    assert (called["BM"], called["BN"], called["BK"]) == (32, 256, 128)
    assert called["atomic"] is True
    assert called["use_nt"] is True
    assert called["D_INTER"] == 384


def _validation_kwargs(**overrides):
    kwargs = {
        "NE": 2,
        "D_HIDDEN": 256,
        "D_INTER": 384,
        "topk": 2,
        "BM": 32,
        "use_nt": False,
        "atomic": True,
        "mxfp4out": False,
        "cshuffle": False,
        "BN": 256,
        "BK": 128,
    }
    kwargs.update(overrides)
    return kwargs


def test_native_validation_accepts_k384_bk128():
    _assert_supported(**_validation_kwargs())


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"BK": 256}, "multiple of 256"),
        ({"BK": 64}, "BK must be one of"),
        ({"BN": 128}, "BN=256"),
    ],
)
def test_native_validation_rejects_wrong_tile_contract(overrides, message):
    with pytest.raises(NotImplementedError, match=message):
        _assert_supported(**_validation_kwargs(**overrides))


def _write_native_csv(path, *, inter_dim, kernel_name2):
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "topk",
                "model_dim",
                "expert",
                "inter_dim",
                "kernelName1",
                "kernelName2",
                "cu_num",
                "act_type",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "topk": 2,
                "model_dim": 256,
                "expert": 2,
                "inter_dim": inter_dim,
                "kernelName1": "flydsl_mxmoe_g1_a4w4_32x256x256",
                "kernelName2": kernel_name2,
                "cu_num": 256,
                "act_type": "ActivationType.Silu",
            }
        )


def test_native_aot_preserves_bk128_and_stage1_k384(tmp_path):
    from aiter.aot.flydsl.mxfp4_moe import _job_key, parse_csv

    csv_path = tmp_path / "native_bk128.csv"
    _write_native_csv(
        csv_path,
        inter_dim=384,
        kernel_name2="flydsl_mxmoe_g2_a4w4_32x256x128_atomic",
    )
    jobs = parse_csv(str(csv_path))
    stage1 = next(job for job in jobs if job["stage"] == 1)
    stage2 = next(job for job in jobs if job["stage"] == 2)

    assert stage1["D_INTER"] == 384
    assert (stage2["BN"], stage2["BK"]) == (256, 128)
    assert stage2["D_INTER"] == 384
    assert stage2["D_INTER_REAL"] is None
    assert _job_key(stage2) != _job_key({**stage2, "BK": 256})


def test_native_aot_compile_forwards_tiles(monkeypatch):
    from aiter.aot.flydsl import mxfp4_moe
    from aiter.ops.flydsl import mxfp4_gemm2_kernels

    called = {}
    monkeypatch.setattr(
        mxfp4_gemm2_kernels,
        "flydsl_mxfp4_gemm2",
        lambda **kwargs: called.update(kwargs),
    )
    mxfp4_moe._compile_stage2(
        {
            "stage": 2,
            "kernel_name": "flydsl_mxmoe_g2_a4w4_32x256x128_atomic",
            "BM": 32,
            "BN": 256,
            "BK": 128,
            "use_nt": False,
            "NE": 2,
            "N_OUT": 256,
            "epilog": "atomic",
            "D_INTER": 384,
            "D_INTER_REAL": None,
            "topk": 2,
            "xcd_swizzle": 0,
        }
    )
    assert (called["BN"], called["BK"]) == (256, 128)


_TUNER_KEYS = [
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


def _tuner_row(inter_dim):
    from aiter import ActivationType, QuantType, dtypes

    return {
        "gfx": "gfx950",
        "cu_num": 256,
        "token": 4,
        "model_dim": 256,
        "inter_dim": inter_dim,
        "expert": 2,
        "topk": 2,
        "act_type": ActivationType.Silu,
        "dtype": dtypes.bf16,
        "q_dtype_a": dtypes.fp4x2,
        "q_dtype_w": dtypes.fp4x2,
        "q_type": QuantType.per_1x32,
        "use_g1u1": True,
        "doweight_stage1": False,
    }


@pytest.mark.parametrize(
    ("inter_dim", "expected_bks"),
    [(384, {128}), (512, {128, 256}), (320, set())],
)
def test_native_tuner_candidate_bks(monkeypatch, inter_dim, expected_bks):
    from csrc.ck_gemm_moe_2stages_codegen import gemm_moe_tune

    monkeypatch.setattr(
        gemm_moe_tune,
        "get_flydsl_stage2_v2_kernels",
        lambda *args, **kwargs: {},
    )
    tuner = gemm_moe_tune.Mxfp4FlydslTuner.__new__(
        gemm_moe_tune.Mxfp4FlydslTuner
    )
    tuner.keys = _TUNER_KEYS
    candidates = tuner._candidate_rows(_tuner_row(inter_dim))
    native_names = [
        candidate["kernelName2"]
        for candidate in candidates
        if candidate["kernelName2"].startswith("flydsl_mxmoe_g2_")
    ]
    bks = {_parse_mxfp4_g2_kname(name)["BK"] for name in native_names}
    assert bks == expected_bks


def test_native_core_cache_epoch_changes_launcher_key(monkeypatch):
    from flydsl.compiler.jit_function import _jit_function_cache_key

    from aiter.ops.flydsl.kernels import mxfp4_gemm2

    original_epoch = mxfp4_gemm2.NATIVE_GEMM2_CORE_CACHE_EPOCH

    def launcher_key(epoch):
        monkeypatch.setattr(mxfp4_gemm2, "NATIVE_GEMM2_CORE_CACHE_EPOCH", epoch)
        launch = mxfp4_gemm2.compile_gemm2_a4w4_port(
            BM=32,
            use_nt=False,
            NE=2,
            N_OUT=256,
            epilog="atomic",
            D_INTER=512,
            BN=256,
            BK=256,
        )
        return _jit_function_cache_key(launch.func)

    try:
        key_n = launcher_key(original_epoch)
        key_next = launcher_key(original_epoch + 1)
    finally:
        monkeypatch.setattr(
            mxfp4_gemm2, "NATIVE_GEMM2_CORE_CACHE_EPOCH", original_epoch
        )

    assert key_n != key_next, (key_n, key_next)
