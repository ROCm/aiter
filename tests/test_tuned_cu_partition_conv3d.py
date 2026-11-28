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


@pytest.fixture
def lookup(monkeypatch):
    """Run the lookup on device with current as the current device."""

    def run(exact, borrow, device, current):
        monkeypatch.setattr(chip_info, "_tuned_cu_nums_of_device", DEVICES.get)
        monkeypatch.setattr(chip_info, "_shared_tuned_cu_nums", lambda: None)
        monkeypatch.setattr(chip_info, "_current_device", lambda: current)
        monkeypatch.setattr(chip_info, "get_gfx", lambda: "gfx950")
        monkeypatch.setattr(chip_info, "get_cu_num", lambda: DEVICES[current][0])
        table = {("gfx950", cu, *KEY): hit for cu, hit in exact.items()}
        monkeypatch.setattr(conv_kernels, "_load_tuned_table", lambda: table)
        borrowed, logged = [], []

        def borrow_tile(dev, key, cu):
            borrowed.append(cu)
            return borrow.get(dev[1])

        monkeypatch.setattr(conv_kernels, "_borrow_tuned_tile", borrow_tile)
        monkeypatch.setattr(
            conv_kernels,
            "_log_tuned_lookup",
            lambda _, dev, *__, **___: logged.append(dev),
        )
        hit = conv_kernels._lookup_tuned_tile(KEY, types.SimpleNamespace(index=device))
        # A borrowed tile must fill the CUs of the device of the lookup.
        assert set(borrowed) <= {DEVICES[device][0]}
        return hit, logged

    return run


@pytest.mark.parametrize(
    "exact, borrow, device, current, expected",
    [
        ({256: "full_chip"}, {}, 1, 1, "full_chip"),
        ({128: "own"}, {}, 1, 1, "own"),
        ({256: "full_chip"}, {128: "own_borrowed"}, 1, 1, "full_chip"),
        ({}, {128: "own_borrowed", 256: "full_borrowed"}, 1, 1, "own_borrowed"),
        ({}, {256: "full_borrowed"}, 1, 1, "full_borrowed"),
        ({128: "own"}, {}, 0, 0, None),  # device 0 is no partition
        # The CU counts come from the device argument, not the current device.
        ({128: "own", 256: "full_chip"}, {}, 1, 0, "own"),
        ({}, {256: "full_borrowed"}, 1, 0, "full_borrowed"),
        ({256: "full_chip"}, {}, 0, 1, "full_chip"),
    ],
)
def test_conv3d_lookup(lookup, exact, borrow, device, current, expected):
    assert lookup(exact, borrow, device, current)[0] == expected


def test_conv3d_lookup_logs_the_cu_count_of_the_row(lookup):
    hit, logged = lookup({256: "full_chip"}, {}, 1, 1)
    assert (hit, logged) == ("full_chip", [("gfx950", 256)])
