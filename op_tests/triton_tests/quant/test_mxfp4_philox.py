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


@pytest.mark.parametrize("enabled", [(True,), (True, True), (False, True)])
def test_philox_inclusive_last_counter(enabled):
    last_start = (1 << 63) - 128 * sum(enabled)
    reserve_mxfp4_philox_streams(1024, enabled, 1, last_start)
    with pytest.raises(ValueError, match="leave room"):
        reserve_mxfp4_philox_streams(1024, enabled, 1, last_start + 1)


@pytest.mark.parametrize(
    "seed,offset,error",
    [
        (None, 0, ValueError),
        (True, 0, TypeError),
        (1, False, TypeError),
        (1.5, 0, TypeError),
        (-1, 0, ValueError),
        (1 << 63, 0, ValueError),
        (1, -1, ValueError),
    ],
)
def test_philox_invalid_metadata(seed, offset, error):
    with pytest.raises(error):
        reserve_mxfp4_philox_streams(1024, (True,), seed, offset)


@pytest.mark.parametrize("seed,offset", [(1, 0), (None, 1)])
def test_philox_disabled_rejects_metadata(seed, offset):
    with pytest.raises(ValueError, match="only valid"):
        reserve_mxfp4_philox_streams(1024, (False,), seed, offset)
