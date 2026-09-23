# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""No-GPU tests for CK fmha factory-key mapping."""

import os
import sys
from unittest.mock import patch

import pytest

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO_ROOT, "aiter", "jit", "utils"))

from build_targets import (
    ck_fmha_batch_prefill_gen_targets,
    ck_fmha_factory_key,
    ck_fmha_targets,
    map_gpu_archs_to_ck_fmha_targets,
)


@pytest.mark.parametrize(
    "archs, expected",
    [
        (["gfx1100"], ["gfx11"]),
        (["gfx1101"], ["gfx11"]),
        (["gfx1102"], ["gfx11"]),
        (["gfx1103"], ["gfx11"]),
        (["gfx1150"], ["gfx115"]),
        (["gfx1151"], ["gfx115"]),
        (["gfx1200"], ["gfx12"]),
        (["gfx1201"], ["gfx12"]),
        (["gfx1250"], ["gfx125"]),
        (["gfx942"], ["gfx9"]),
        (["gfx90a"], ["gfx9"]),
        (["gfx908"], ["gfx9"]),
        (["gfx950"], ["gfx950"]),
        (["gfx942", "gfx1100"], ["gfx9", "gfx11"]),
        (["gfx1100", "gfx942"], ["gfx11", "gfx9"]),
        (["gfx1100", "gfx1101"], ["gfx11"]),
        (["gfx950", "gfx942"], ["gfx950", "gfx9"]),
        (["gfx1030"], []),
        ([], []),
        (["cpu"], []),
        (["gfx12"], []),  # bare gfx12 is not a prefix of gfx120/gfx125
    ],
)
def test_map_gpu_archs_to_ck_fmha_targets(archs, expected):
    assert map_gpu_archs_to_ck_fmha_targets(archs) == expected


def test_ck_fmha_targets_joins_and_falls_back():
    with patch("chip_info.get_gfx_list", return_value=["gfx1100"]):
        assert ck_fmha_targets() == "gfx11"
    with patch("chip_info.get_gfx_list", return_value=["gfx942", "gfx1100"]):
        assert ck_fmha_targets() == "gfx9,gfx11"
    for unmapped in (["gfx1030"], ["cpu"], []):
        with patch("chip_info.get_gfx_list", return_value=unmapped):
            assert ck_fmha_targets() == "gfx9,gfx950"


def test_ck_fmha_factory_key():
    assert ck_fmha_factory_key("gfx1100") == "gfx11"
    assert ck_fmha_factory_key("gfx1201") == "gfx12"
    assert ck_fmha_factory_key("gfx1151") == "gfx115"
    assert ck_fmha_factory_key("gfx942") == "gfx9"
    assert ck_fmha_factory_key("gfx1030") == ""


@pytest.mark.parametrize(
    "gfx_list, expected",
    [
        (["gfx942", "gfx1100"], "gfx9"),
        (["gfx950"], "gfx950"),
        (["gfx950", "gfx1100"], "gfx950"),
        (["gfx1100"], "gfx11"),
        (["gfx1201"], "gfx12"),
        (["cpu"], "gfx9,gfx950"),
    ],
)
def test_ck_fmha_batch_prefill_gen_targets(gfx_list, expected):
    with patch("chip_info.get_gfx_list", return_value=gfx_list):
        assert ck_fmha_batch_prefill_gen_targets() == expected
