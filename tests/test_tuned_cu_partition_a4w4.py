# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""A4W4 tuned-row lookups on compute partitions, with the partition mocked."""

import pytest

from aiter.jit.utils import chip_info
from aiter.ops import gemm_op_a4w4

MI355X_DPX = ("gfx950", (128, 256))
MI355X_QPX = ("gfx950", (64, 256))
MI350P = ("gfx950", (128,))
GFX1250_HALF = ("gfx1250", (128, 256))
GFX1250 = ("gfx1250", (256,))


@pytest.fixture
def device(monkeypatch):
    """Make the current GPU a gfx with the given tuned CU counts."""

    def install(gpu):
        gfx, cu_nums = gpu
        monkeypatch.setattr(chip_info, "_tuned_cu_nums_of_device", lambda _: cu_nums)
        monkeypatch.setattr(chip_info, "_shared_tuned_cu_nums", lambda: cu_nums)
        monkeypatch.setattr(gemm_op_a4w4, "get_gfx", lambda: gfx)
        monkeypatch.setattr(gemm_op_a4w4, "get_cu_num", lambda: cu_nums[0])

    return install


A4W4_TABLE = {
    ("gfx950", 256, 1024, 3072, 1536): {"kernelName": "full_chip", "splitK": 0},
    ("gfx950", 128, 1024, 7168, 512): {"kernelName": "own_128", "splitK": 0},
    ("gfx950", 256, 1024, 7168, 512): {"kernelName": "full_chip", "splitK": 0},
}


@pytest.mark.parametrize(
    "gpu, n, k, expected",
    [
        (MI355X_DPX, 3072, 1536, "full_chip"),
        (MI355X_QPX, 3072, 1536, "full_chip"),
        (MI355X_DPX, 7168, 512, "own_128"),
        (MI350P, 3072, 1536, None),  # MI350P is no partition
    ],
)
def test_a4w4_lookup(device, monkeypatch, gpu, n, k, expected):
    device(gpu)
    lookup = gemm_op_a4w4.get_GEMM_config
    monkeypatch.setattr(lookup, "gemm_dict", A4W4_TABLE, False)
    monkeypatch.setattr(lookup, "has_gfx", True, False)
    lookup.cache_clear()
    config = lookup(1024, n, k)
    assert (config and config["kernelName"]) == expected


@pytest.mark.parametrize(
    "gpu, m, expected",
    [
        (GFX1250_HALF, 1024, ("asm", "own_128")),
        (GFX1250_HALF, 1000, ("asm", "full_chip")),  # exact M before padded M
        (GFX1250, 1024, ("asm", "full_chip_m1024")),
        (("gfx1250", (128,)), 1000, ("asm", "own_128")),  # no partition
    ],
)
def test_a4w4_mxscale_lookup(device, monkeypatch, gpu, m, expected):
    device(gpu)
    rows = {
        ("gfx1250", 256, 1024, 4096, 4096): ("asm", "full_chip_m1024"),
        ("gfx1250", 256, 1000, 4096, 4096): ("asm", "full_chip"),
        ("gfx1250", 128, 1024, 4096, 4096): ("asm", "own_128"),
    }
    monkeypatch.setattr(gemm_op_a4w4, "_load_a4w4_mxscale_tuned", lambda _: rows)
    monkeypatch.setattr(gemm_op_a4w4, "get_padded_m", lambda m, n, k, gl: 1024)
    monkeypatch.setattr(gemm_op_a4w4, "flydsl_a4w4_fallback_kernel", lambda *_: None)
    gemm_op_a4w4._a4w4_mxscale_config.cache_clear()
    assert gemm_op_a4w4._a4w4_mxscale_config(m, 4096, 4096) == expected
