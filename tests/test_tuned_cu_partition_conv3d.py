# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Conv3d tuned-tile lookup on compute partitions, with the partition mocked.

HIP device 0 is an MI355X in SPX mode (256 CUs) and device 1 a partition of an
MI355X in DPX mode (128 CUs).
"""

import types

import pytest

from aiter.jit.utils import chip_info
from aiter.ops.flydsl import conv_kernels

DEVICES = {0: (256,), 1: (128, 256)}
KEY = (1, 2, 3)


@pytest.mark.parametrize(
    "exact, borrow, cu_num, device, expected",
    [
        ({256: "full_chip"}, {}, 128, 1, "full_chip"),
        ({128: "own"}, {}, 128, 1, "own"),
        ({256: "full_chip"}, {128: "own_borrowed"}, 128, 1, "full_chip"),
        ({}, {128: "own_borrowed", 256: "full_borrowed"}, 128, 1, "own_borrowed"),
        ({}, {256: "full_borrowed"}, 128, 1, "full_borrowed"),
        ({256: "full_chip"}, {}, 128, 0, None),  # device 0 is no partition
    ],
)
def test_conv3d_lookup(monkeypatch, exact, borrow, cu_num, device, expected):
    monkeypatch.setattr(chip_info, "_tuned_cu_nums_of_device", DEVICES.get)
    monkeypatch.setattr(chip_info, "get_gfx", lambda: "gfx950")
    monkeypatch.setattr(chip_info, "get_cu_num", lambda: cu_num)
    table = {("gfx950", cu, *KEY): hit for cu, hit in exact.items()}
    monkeypatch.setattr(conv_kernels, "_load_tuned_table", lambda: table)
    borrowed = []

    def borrow_tile(dev, key, cu):
        borrowed.append(cu)
        return borrow.get(dev[1])

    monkeypatch.setattr(conv_kernels, "_borrow_tuned_tile", borrow_tile)
    monkeypatch.setattr(conv_kernels, "_log_tuned_lookup", lambda *_, **__: None)
    hit = conv_kernels._lookup_tuned_tile(KEY, types.SimpleNamespace(index=device))
    assert hit == expected
    assert set(borrowed) <= {cu_num}  # a borrowed tile must fill the partition
