# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest

from aiter.ops.triton.utils.mxfp4 import reserve_mxfp4_philox_streams


@pytest.mark.parametrize(
    "enabled",
    [(True,), (False,), (True, True), (True, False), (False, True), (False, False)],
)
def test_philox_stream_reservation(enabled):
    seed = 7 if any(enabled) else None
    offset = 11 if any(enabled) else 0
    actual_seed, offsets = reserve_mxfp4_philox_streams(1024, enabled, seed, offset)
    assert actual_seed == (7 if any(enabled) else 0)
    expected = []
    for active in enabled:
        expected.append(offset if active else 0)
        if active:
            offset += 128
    assert offsets == tuple(expected)
