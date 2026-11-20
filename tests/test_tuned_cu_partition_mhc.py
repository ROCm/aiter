# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Fused MHC GEMM tuned policy on compute partitions, with the partition mocked."""

import pytest

from aiter.jit.utils import chip_info
from aiter.ops import mhc


@pytest.mark.parametrize(
    "cu_nums, policy",
    [
        ((128, 256), "gfx950_256"),  # MI355X DPX
        ((128,), "default"),  # MI350P is no partition
    ],
)
def test_mhc_policy_lookup(monkeypatch, cu_nums, policy):
    monkeypatch.setattr(chip_info, "_tuned_cu_nums_of_device", lambda _: cu_nums)
    monkeypatch.setattr(chip_info, "_shared_tuned_cu_nums", lambda: cu_nums)
    monkeypatch.setattr(mhc, "get_gfx_runtime", lambda: "gfx950")
    monkeypatch.setattr(mhc, "get_cu_num", lambda: cu_nums[0])
    mhc.get_mhc_fused_post_pre_config.cache_clear()
    config = mhc.get_mhc_fused_post_pre_config(512, 7168)
    mhc.get_mhc_fused_post_pre_config.cache_clear()
    # The policy of the full chip gets the CU count of the partition.
    assert config == getattr(mhc, f"_mhc_fused_config_{policy}")(512, 7168, 128)
