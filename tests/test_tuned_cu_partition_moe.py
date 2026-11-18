# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Fused MoE tuned-row lookups on compute partitions, with the partition mocked."""

import importlib

import pytest
import torch

from aiter import ActivationType, QuantType
from aiter.jit.utils import chip_info
from aiter.ops.flydsl import moe_a16w4_policy

fused_moe = importlib.import_module("aiter.fused_moe")

MI300X_DPX = (152, 304)
MI308X = (80,)
BF16 = str(torch.bfloat16)


class Hit(Exception):
    pass


class Table(dict):
    """A tuned table that stops get_2stage_cfgs at the first row it finds."""

    def get(self, key, default=None):
        if key in self:
            raise Hit(key[1:3])
        return default


def moe_key(cu, token, act=str(ActivationType.Silu)):
    return ("gfx942", cu, token, 4096, 384, 128, 8, act, BF16, BF16, BF16) + (
        str(QuantType.No),
        True,
        False,
    )


@pytest.mark.parametrize(
    "cu_nums, token, rows, expected",
    [
        (MI300X_DPX, 64, [(304, 64)], (304, 64)),
        (MI300X_DPX, 64, [(152, 64), (304, 64)], (152, 64)),
        # The exact token of the full chip comes before a smaller token tier.
        (MI300X_DPX, 131072, [(152, 32768), (304, 131072)], (304, 131072)),
        (MI300X_DPX, 131072, [(304, 32768)], (304, 32768)),
        (MI308X, 64, [(80, 64), (304, 64)], (80, 64)),
    ],
)
def test_fused_moe_lookup_order(monkeypatch, cu_nums, token, rows, expected):
    monkeypatch.setattr(chip_info, "_tuned_cu_nums_of_device", lambda _: cu_nums)
    monkeypatch.setattr(chip_info, "_shared_tuned_cu_nums", lambda: cu_nums)
    monkeypatch.setattr(fused_moe, "get_cu_num", lambda: cu_nums[0])
    monkeypatch.setattr(fused_moe, "get_gfx_runtime", lambda: "gfx942")
    primary = Table({moe_key(cu, t): {} for cu, t in rows})
    monkeypatch.setattr(fused_moe, "cfg_2stages", (primary, Table()))
    fused_moe.get_2stage_cfgs.cache_clear()
    with pytest.raises(Hit) as hit:
        fused_moe.get_2stage_cfgs(
            token,
            4096,
            384,
            128,
            8,
            torch.bfloat16,
            torch.bfloat16,
            torch.bfloat16,
            QuantType.No,
            True,
            ActivationType.Silu,
            False,
            0,
            0,
        )
    fused_moe.get_2stage_cfgs.cache_clear()
    assert hit.value.args[0] == expected


@pytest.mark.parametrize(
    "cu_nums, expected", [(MI300X_DPX, "full_chip"), (MI308X, None)]
)
def test_moe_a16w4_lookup(monkeypatch, cu_nums, expected):
    monkeypatch.setattr(chip_info, "_tuned_cu_nums_of_device", lambda _: cu_nums)
    monkeypatch.setattr(chip_info, "_shared_tuned_cu_nums", lambda: cu_nums)
    monkeypatch.setattr(chip_info, "get_gfx_runtime", lambda: "gfx942")
    monkeypatch.setattr(chip_info, "get_cu_num", lambda: cu_nums[0])
    key = ("gfx942", 304, 8, 5120, 384, 288, 384, 6, 10.0)
    monkeypatch.setattr(moe_a16w4_policy, "_TUNED_TILES", {key: "full_chip"})
    tiles = moe_a16w4_policy.get_a16w4_tuned_tiles(
        8,
        5120,
        384,
        288,
        384,
        6,
        act="silu_clamped",
        activation_bf16=True,
        swiglu_limit=10.0,
    )
    assert tiles == expected
