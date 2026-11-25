# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Which router calls take the 8-row launch (``moe_router8_gfx1250``); every other call keeps the split-K kernel."""

import pytest

moe_front = pytest.importorskip("aiter.ops.flydsl.moe_front_gfx1250")

# (m, k, split_k, topk, has_bias, quant_mode, with_counts)
DECODE = (8, 6144, 8, 4, True, 1, False)


@pytest.mark.parametrize(
    "change, takes",
    [
        ({}, True),
        ({0: 7}, False),  # not a full 8-row tile
        ({0: 16}, False),
        ({2: 4}, False),  # another split
        ({3: 5}, False),  # top-k > 4
        ({1: 6016}, False),  # K not a multiple of 8 x 32
        ({4: False}, False),  # no bias
        ({5: 0}, False),  # no MXFP4 rows
        ({5: 2}, False),  # MXFP8 rows too
        ({6: True}, False),  # group counts
    ],
)
def test_router8_takes_only_the_decode_tile(change, takes):
    args = list(DECODE)
    for i, v in change.items():
        args[i] = v
    assert moe_front._router8_takes(*args) is takes
