# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""MoE v2 tuned-row lookups on compute partitions, with the partition mocked."""

import pytest

from aiter import ActivationType, QuantType, dtypes
from aiter.jit.utils import chip_info
from aiter.moev2.dispatch import tuned
from aiter.moev2.problem import MoeProblem

MI355X_DPX = (128, 256)
MI355X_SPX = (256,)


def problem(cu_num, token):
    fp8 = dtypes.fp8
    args = (token, 7168, 256, 256, 8, dtypes.bf16, fp8, fp8, QuantType.per_Tensor)
    args += (True, ActivationType.Silu, False, 0, 0)
    return MoeProblem.from_legacy_key(*args, arch="gfx950", cu_num=cu_num)


def table(p, rows, disabled=False):
    keys, keys_disabled = tuned.lookup_keys(p)
    if not disabled:
        keys = keys[:7] + (str(p.activation.activation),) + keys[8:]
    else:
        keys = keys_disabled
    return {keys[:1] + (cu, token) + keys[3:]: (cu, token) for cu, token in rows}


@pytest.mark.parametrize(
    "cu_nums, token, rows, expected",
    [
        (MI355X_DPX, 64, [(256, 64)], (256, 64)),
        (MI355X_DPX, 64, [(128, 64), (256, 64)], (128, 64)),
        # The exact token of the full chip comes before a smaller token tier.
        (MI355X_DPX, 131072, [(128, 32768), (256, 131072)], (256, 131072)),
        (MI355X_DPX, 131072, [(256, 32768)], (256, 32768)),
        (MI355X_SPX, 64, [(256, 64)], (256, 64)),
        (MI355X_SPX, 64, [(128, 64)], None),
    ],
)
def test_moev2_lookup_order(monkeypatch, cu_nums, token, rows, expected):
    monkeypatch.setattr(chip_info, "_tuned_cu_nums_of_device", lambda _: cu_nums)
    monkeypatch.setattr(chip_info, "_shared_tuned_cu_nums", lambda: cu_nums)
    p = problem(cu_nums[0], token)
    assert tuned.lookup(p, (table(p, rows), {})) == expected
    assert tuned.lookup(p, ({}, table(p, rows, disabled=True))) == expected
