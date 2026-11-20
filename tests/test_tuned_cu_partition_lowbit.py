# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""A6W6, A6W4, A4W6 and mixed MXFP tuned-row lookups on compute partitions.

The partition info is mocked: HIP device 0 is an MI355X in SPX mode (256 CUs)
and device 1 a partition of an MI355X in DPX mode (128 CUs).
"""

import pytest
import torch

from aiter.jit.utils import chip_info
from aiter.ops import gemm_op_a6w6
from aiter.ops.gemm_op_mixed_mxfp import _find_mixed_mxfp_config, _tuned_rows_snapshot

# The node has two kinds of GPUs. The current device is device 1.
DEVICES = {0: (256,), 1: (128, 256)}
TABLE = {
    ("gfx950", 256, 256, 4096, 4096): {"kernelName": "full_chip"},
    ("gfx950", 128, 256, 1024, 4096): {"kernelName": "own_128"},
    ("gfx950", 256, 256, 1024, 4096): {"kernelName": "full_chip"},
    ("gfx950", 128, 256, 2048, 4096): {"kernelName": "own_128_padded"},
    ("gfx950", 256, 200, 2048, 4096): {"kernelName": "full_chip_exact"},
}


@pytest.fixture(autouse=True)
def node(monkeypatch):
    monkeypatch.setattr(chip_info, "_tuned_cu_nums_of_device", DEVICES.get)
    monkeypatch.setattr(chip_info, "_shared_tuned_cu_nums", lambda: None)
    monkeypatch.setattr(chip_info, "_current_device", lambda: 1)


CASES = [
    (256, 4096, 128, None, "full_chip"),
    (256, 1024, 128, None, "own_128"),
    (200, 2048, 128, None, "full_chip_exact"),  # exact shape before padded
    (256, 4096, 128, 1, "full_chip"),
    (256, 4096, 128, 0, None),  # device 0 is no partition
    (256, 4096, 256, 0, "full_chip"),
]


@pytest.mark.parametrize("m, n, cu_num, device, expected", CASES)
def test_mixed_mxfp_lookup(m, n, cu_num, device, expected):
    """A4W6, A6W4 and the mixed MXFP GEMMs share this lookup."""
    hit = _find_mixed_mxfp_config(
        TABLE, "gfx950", cu_num, m, n, 4096, 256, n, 4096, device
    )
    assert (hit and hit[0]["kernelName"]) == expected


@pytest.mark.parametrize("m, n, cu_num, device, expected", CASES)
def test_compiled_snapshot_keeps_the_eager_order(m, n, cu_num, device, expected):
    """torch.compile takes the first exact, then the first padded snapshot row."""
    rows = _tuned_rows_snapshot(TABLE, "gfx950", cu_num, device)
    hits = [
        row[3]
        for want in ((m, n, 4096), (256, n, 4096))
        for row in rows
        if row[:3] == want
    ]
    assert (hits[0] if hits else None) == expected


@pytest.mark.parametrize("cu_num, expected", [(128, "full_chip"), (80, None)])
def test_a6w6_lookup(monkeypatch, cu_num, expected):
    monkeypatch.setattr(gemm_op_a6w6, "_load_gemm_a6w6_configs", lambda _: TABLE)
    monkeypatch.setattr(gemm_op_a6w6, "get_gfx", lambda: "gfx950")
    monkeypatch.setattr(gemm_op_a6w6, "get_cu_num", lambda: cu_num)
    gemm_op_a6w6.get_GEMM_A6W6_config.cache_clear()
    config = gemm_op_a6w6.get_GEMM_A6W6_config(256, 4096, 4096, "a6w6.csv")
    assert (config and config["kernelName"]) == expected
    gemm_op_a6w6.get_GEMM_A6W6_config.cache_clear()


@pytest.mark.parametrize("device, expected", [(0, "full_chip"), (1, "own_128")])
def test_a6w6_eager_lookup_uses_the_device_of_the_input(monkeypatch, device, expected):
    """The current device is the partition (device 1); the input decides."""
    gpus = {0: ("gfx950", 256), 1: ("gfx950", 128)}
    monkeypatch.setattr(gemm_op_a6w6, "_load_gemm_a6w6_configs", lambda _: TABLE)
    monkeypatch.setattr(gemm_op_a6w6, "_get_device_gfx_cu", gpus.get)
    monkeypatch.setattr(gemm_op_a6w6, "get_gfx", lambda: "gfx950")
    monkeypatch.setattr(gemm_op_a6w6, "get_cu_num", lambda: 128)
    gemm_op_a6w6.get_GEMM_A6W6_config.cache_clear()
    kernel = gemm_op_a6w6._select_gemm_a6w6_kernel(
        256, 1024, 4096, None, torch.device("cuda", device)
    )
    assert kernel == expected
    gemm_op_a6w6.get_GEMM_A6W6_config.cache_clear()
