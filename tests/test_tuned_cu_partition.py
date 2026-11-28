# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Lookup order of the M walk on a compute partition, with the partition mocked."""

import pytest

from aiter.jit.utils import chip_info
from aiter.ops import gemm_op_common

TABLE = {
    ("gfx942", 152, 128, 1024): "own_exact",
    ("gfx942", 304, 128, 1024): "full_exact",
    ("gfx942", 152, 128, 2048): "own_padded",
    ("gfx942", 304, 100, 2048): "full_exact",
    ("gfx942", 304, 128, 4096): "full_padded",
}


@pytest.fixture
def mi300x_dpx(monkeypatch):
    monkeypatch.setattr(chip_info, "_tuned_cu_nums_of_device", lambda _: (152, 304))
    monkeypatch.setattr(chip_info, "_shared_tuned_cu_nums", lambda: (152, 304))
    monkeypatch.setattr(gemm_op_common, "get_padded_m", lambda m, n, k, gl: 128)


@pytest.mark.parametrize(
    "m, n, cu_num, expected",
    [
        (128, 1024, 152, ("own_exact", 128)),
        (100, 2048, 152, ("full_exact", 100)),  # exact shape before padded M
        (100, 4096, 152, ("full_padded", 128)),
        (100, 4096, 80, (None, 128)),  # another CU count: exact CU count only
    ],
)
def test_exact_shape_comes_before_padded_m(mi300x_dpx, m, n, cu_num, expected):
    row = gemm_op_common.find_tuned_row(
        TABLE, lambda cu, pm: ("gfx942", cu, pm, n), "gfx942", cu_num, m, n, 64, "t"
    )
    assert row == expected
