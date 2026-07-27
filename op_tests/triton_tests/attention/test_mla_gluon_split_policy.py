# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest

from aiter.ops.triton.gluon.mla_gluon import _resolve_num_kv_splits


@pytest.mark.parametrize("auto_num_kv_splits", [1, 4, 32, 256])
def test_default_num_kv_splits_uses_auto_policy(auto_num_kv_splits):
    assert _resolve_num_kv_splits(auto_num_kv_splits, None) == auto_num_kv_splits


@pytest.mark.parametrize("num_kv_splits", [1, 16, 32, 256])
def test_num_kv_splits_override(num_kv_splits):
    assert _resolve_num_kv_splits(1, num_kv_splits) == num_kv_splits


@pytest.mark.parametrize("num_kv_splits", [0, 257, -1])
def test_num_kv_splits_override_rejects_out_of_range(num_kv_splits):
    with pytest.raises(ValueError, match=r"must be in \[1, 256\]"):
        _resolve_num_kv_splits(1, num_kv_splits)


@pytest.mark.parametrize("num_kv_splits", [True, 32.0, "32"])
def test_num_kv_splits_override_rejects_non_integer(num_kv_splits):
    with pytest.raises(TypeError, match="must be an int or None"):
        _resolve_num_kv_splits(1, num_kv_splits)
