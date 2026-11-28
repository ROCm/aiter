# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""MXFP8 BMM tuned-row lookup on compute partitions, with the partition mocked."""

import pytest

from aiter.jit.utils import chip_info
from aiter.ops import batched_gemm_op_a8w8, gemm_op_common
from aiter.ops.opus import policy

TABLE = {
    ("gfx950", 256, 8, 256, 512, 7168, "128x128"): {"kernelId": "full_chip"},
    ("gfx950", 256, 8, 100, 512, 7168, "128x128"): {"kernelId": "full_chip_m100"},
    ("gfx950", 128, 8, 512, 512, 7168, "128x128"): {"kernelId": "own_128"},
}
# HIP device 0 is an MI355X in SPX mode, device 1 a partition of an MI355X in DPX.
# The node has two kinds of GPUs. The current device is device 1.
DEVICES = {0: (256,), 1: (128, 256)}


@pytest.fixture(autouse=True)
def node(monkeypatch):
    monkeypatch.setattr(chip_info, "_tuned_cu_nums_of_device", DEVICES.get)
    monkeypatch.setattr(chip_info, "_shared_tuned_cu_nums", lambda: None)
    monkeypatch.setattr(chip_info, "_current_device", lambda: 1)
    monkeypatch.setattr(policy, "_load_mxscale_bmm_tuned", lambda *_: TABLE)
    monkeypatch.setattr(
        gemm_op_common, "get_padded_m", lambda m, n, k, gl: (256, 512)[gl]
    )
    policy.lookup_mxscale_bmm_config.cache_clear()
    yield
    policy.lookup_mxscale_bmm_config.cache_clear()


@pytest.mark.parametrize(
    "m, cu_num, device, expected",
    [
        (200, 128, None, "full_chip"),
        (100, 128, None, "full_chip_m100"),  # exact M before padded M
        (512, 128, None, "own_128"),
        (200, 128, 1, "full_chip"),
        (200, 128, 0, "own_128"),  # device 0 is no partition
        (200, 256, 0, "full_chip"),
    ],
)
def test_mxscale_bmm_lookup(m, cu_num, device, expected):
    row = batched_gemm_op_a8w8.lookup_mxscale_bmm_config(
        8, m, 512, 7168, gfx="gfx950", cu_num=cu_num, device=device
    )
    assert row["kernelId"] == expected
