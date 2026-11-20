# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

from types import SimpleNamespace

import pytest

from aiter.ops.triton.utils._triton import arch_info


def test_get_arch_keeps_cached_default(monkeypatch):
    monkeypatch.setattr(arch_info, "_CACHED_ARCH", "gfx942")
    assert arch_info.get_arch() == "gfx942"


@pytest.mark.parametrize("arch", ["gfx950", "gfx950:sramecc+:xnack-"])
def test_get_arch_refreshes_current_target(monkeypatch, arch):
    monkeypatch.setattr(arch_info, "_CACHED_ARCH", "gfx942")
    monkeypatch.setattr(
        arch_info.triton.runtime.driver.active,
        "get_current_target",
        lambda: SimpleNamespace(arch=arch),
    )
    assert arch_info.get_arch(refresh=True) == "gfx950"
    assert arch_info.get_arch() == "gfx942"
