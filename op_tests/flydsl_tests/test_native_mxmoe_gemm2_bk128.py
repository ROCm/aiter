# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

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
