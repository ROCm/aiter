# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest
import triton

from aiter.ops.triton.utils._triton import arch_info


@pytest.mark.parametrize("arch, expect_off", [("gfx1250", True), ("gfx950", False)])
def test_no_async_copy_on_gfx1250(monkeypatch, arch, expect_off):
    amd_knobs = getattr(triton.knobs, "amd", None)
    if not hasattr(amd_knobs, "use_async_copy"):
        pytest.skip("Triton has no use_async_copy knob")
    monkeypatch.setattr(arch_info, "get_arch", lambda: arch)

    before = amd_knobs.use_async_copy
    with arch_info.no_async_copy_on_gfx1250():
        inside = amd_knobs.use_async_copy
    assert (inside is False) == expect_off
    assert amd_knobs.use_async_copy == before
