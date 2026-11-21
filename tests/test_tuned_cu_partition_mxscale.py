# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""MXFP8 and mxscale GEMM tuned-row lookups on compute partitions, partition mocked."""

import pytest

from aiter.jit.utils import chip_info
from aiter.ops import gemm_op_a8w8, gemm_op_common

MI355X_DPX = ("gfx950", (128, 256))
MI355X = ("gfx950", (256,))
MI350P = ("gfx950", (128,))
GFX1250_HALF = ("gfx1250", (128, 256))


@pytest.fixture
def device(monkeypatch):
    """Make the current GPU a gfx with the given tuned CU counts."""

    def install(gpu):
        gfx, cu_nums = gpu
        monkeypatch.setattr(chip_info, "_tuned_cu_nums_of_device", lambda _: cu_nums)
        monkeypatch.setattr(chip_info, "_shared_tuned_cu_nums", lambda: cu_nums)
        monkeypatch.setattr(gemm_op_a8w8, "get_gfx", lambda: gfx)
        monkeypatch.setattr(gemm_op_a8w8, "get_cu_num", lambda: cu_nums[0])

    return install


def mxscale_row(name):
    return {"libtype": "flydsl", "kernelId": "bmm", "kernelName": name}


@pytest.mark.parametrize(
    "gpu, m, expected",
    [
        (MI355X_DPX, 200, "full_chip"),
        (MI355X_DPX, 512, "own_128"),
        (MI355X_DPX, 100, "full_chip_m100"),  # exact M before padded M
        (MI350P, 200, "own_128"),
        (MI355X, 200, "full_chip"),
    ],
)
def test_mxscale_bpreshuffle_lookup(device, monkeypatch, gpu, m, expected):
    device(gpu)
    table = {
        ("gfx950", 256, 256, 4096, 7168, "1x32"): mxscale_row("full_chip"),
        ("gfx950", 256, 100, 4096, 7168, "1x32"): mxscale_row("full_chip_m100"),
        ("gfx950", 128, 512, 4096, 7168, "1x32"): mxscale_row("own_128"),
    }
    monkeypatch.setattr(gemm_op_a8w8, "_load_mxscale_bpreshuffle_tuned", lambda: table)
    monkeypatch.setattr(
        gemm_op_a8w8,
        "_mxscale_bpreshuffle_tuned_nk",
        lambda: frozenset({key[:2] + key[3:] for key in table}),
    )
    monkeypatch.setattr(
        gemm_op_common, "get_padded_m", lambda m, n, k, gl: (256, 512)[gl]
    )
    gemm_op_a8w8.get_mxscale_bpreshuffle_config.cache_clear()
    row = gemm_op_a8w8.get_mxscale_bpreshuffle_config(m, 4096, 7168, "1x32")
    assert (row and row["kernelName"]) == expected


@pytest.mark.parametrize(
    "gpu, expected", [(GFX1250_HALF, True), (("gfx1250", (128,)), False)]
)
def test_mxfp8_bpreshuffle_tuned(device, monkeypatch, gpu, expected):
    """vLLM asks this before it picks the MXFP8 bpreshuffle GEMM."""
    device(gpu)
    monkeypatch.setattr(
        gemm_op_a8w8,
        "_mxfp8_bpreshuffle_tuned_nk",
        lambda _: frozenset({("gfx1250", 256, 4096, 7168)}),
    )
    assert gemm_op_a8w8.mxfp8_bpreshuffle_tuned(4096, 7168) is expected
