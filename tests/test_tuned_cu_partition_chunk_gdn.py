# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Chunk GDN tuned-BV lookup on compute partitions, with the partition mocked."""

import pytest

from aiter.jit.utils import chip_info

try:
    from aiter.ops.flydsl import linear_attention_prefill_kernels as gdn
except (ImportError, RuntimeError) as e:
    # No FlyDSL >= 0.5.0, or no visible GPU for the Triton driver.
    pytest.skip(f"chunk GDN module not importable: {e}", allow_module_level=True)

SHAPE = {
    "H": 32,
    "Hg": 16,
    "V": 128,
    "is_varlen": False,
    "use_h0": True,
    "store_fs": True,
    "snapshot_bf16": True,
    "state_bf16": False,
    "total_chunks": 8,
    "max_seq_chunks": 8,
}


@pytest.mark.parametrize(
    "cu_nums, rows, expected",
    [
        ((76, 304), {304: 64}, 64),  # MI300X QPX
        ((76, 304), {76: 32, 304: 64}, 32),
        ((80,), {304: 64}, None),  # MI308X is no partition
    ],
)
def test_chunk_gdn_bv_lookup(monkeypatch, cu_nums, rows, expected):
    monkeypatch.setattr(chip_info, "_tuned_cu_nums_of_device", lambda _: cu_nums)
    monkeypatch.setattr(chip_info, "_shared_tuned_cu_nums", lambda: cu_nums)
    monkeypatch.setattr(chip_info, "get_gfx_runtime", lambda: "gfx942")
    monkeypatch.setattr(chip_info, "get_cu_num", lambda: cu_nums[0])
    monkeypatch.setattr(gdn, "_TUNED_BV_RUNTIME", None)
    table = {("gfx942", cu, *SHAPE.values()): bv for cu, bv in rows.items()}
    monkeypatch.setattr(gdn, "_BV_TUNED_TABLE", table)
    assert gdn._tuned_bv(**SHAPE) == expected
