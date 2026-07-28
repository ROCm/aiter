# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

from aiter.ops.flydsl.moe_kernels import (
    requires_flydsl_stage2_reduce,
    resolve_flydsl_grid_y_persist_m,
)


def test_stage1_persist_m_keeps_grid_y_within_hip_limit():
    assert resolve_flydsl_grid_y_persist_m(0) == 1
    assert resolve_flydsl_grid_y_persist_m(65535) == 1
    assert resolve_flydsl_grid_y_persist_m(65536) == 2
    assert resolve_flydsl_grid_y_persist_m(131070) == 2
    assert resolve_flydsl_grid_y_persist_m(131071) == 3


def test_stage1_persist_m_preserves_larger_requested_value():
    assert resolve_flydsl_grid_y_persist_m(65536, requested_persist_m=4) == 4


def test_stage2_large_output_avoids_32bit_atomic_offsets():
    assert not requires_flydsl_stage2_reduce(349525, 6144, 2)
    assert requires_flydsl_stage2_reduce(349526, 6144, 2)
