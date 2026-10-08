# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest
from packaging.version import Version

from aiter.ops.triton.gather_kv_b_proj import (
    _needs_gfx1250_grid_stride_workaround,
)


@pytest.mark.parametrize(
    "arch,triton_version,expected",
    [
        ("gfx1250", "3.8.0", True),
        ("gfx1250", "3.8.0.dev20260903+git9551a57b", True),
        ("gfx1250", "3.9.0", False),
        ("gfx1250", "3.10.0", False),
        ("gfx950", "3.8.0", False),
    ],
)
def test_gfx1250_grid_stride_workaround_dispatch(arch, triton_version, expected):
    version = Version(Version(triton_version).base_version)
    assert _needs_gfx1250_grid_stride_workaround(arch, version) is expected
