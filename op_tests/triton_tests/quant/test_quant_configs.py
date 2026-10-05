# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""CPU-only tests for the bucketed config lookup and the Gluon quant config tables."""

import glob
import json
import os
import re

import pytest

from aiter.ops.triton.utils.config_utils import (
    AITER_TRITON_CONFIGS_PATH,
    select_leq_config,
)
from aiter.ops.triton.utils.gmm_common import is_power_of_2


def test_select_leq_config_one_axis():
    table = {"N_LEQ_64": {"v": 1}, "N_LEQ_1024": {"v": 2}, "any": {"v": 3}}
    assert select_leq_config(table, 1)["v"] == 1
    assert select_leq_config(table, 64)["v"] == 1
    assert select_leq_config(table, 65)["v"] == 2
    assert select_leq_config(table, 5000)["v"] == 3


def test_select_leq_config_many_axes():
    table = {
        "M_LEQ_32": {"v": "small_m"},
        "M_LEQ_32.N_LEQ_1024": {"v": "small_m_small_n"},
        "M_GEQ_33.N_LEQ_1024": {"v": "big_m_small_n"},
        "N_LEQ_16384": {"v": "any_m_mid_n"},
        "any": {"v": "any"},
    }

    def pick(m, n):
        return select_leq_config(table, axes=("M", "N"), M=m, N=n)["v"]

    assert pick(8, 512) == "small_m_small_n"
    assert pick(8, 4096) == "small_m"  # leftmost axis wins the tie
    assert pick(100, 512) == "big_m_small_n"
    assert pick(100, 4096) == "any_m_mid_n"
    assert pick(100, 20000) == "any"


def test_select_leq_config_many_axes_needs_any():
    with pytest.raises(KeyError):
        select_leq_config({"M_LEQ_32": {}}, axes=("M",), M=100)


def test_select_leq_config_returns_copy():
    table = {"M_LEQ_32": {"v": 1}, "any": {"v": 2}}
    select_leq_config(table, axes=("M",), M=8)["v"] = 99
    assert table["M_LEQ_32"]["v"] == 1


def test_select_leq_config_needs_value_or_axes():
    with pytest.raises(TypeError):
        select_leq_config({"any": {}})


@pytest.mark.parametrize(
    "kwargs",
    [
        {"value": 5, "prefx": "M_LEQ_"},  # misspelled keyword, one axis
        {"value": 5, "M": 5},  # axis value without axes
        {"value": 5, "axes": ("M",), "M": 5},  # value and axes together
        {"axes": ("M", "N"), "M": 5},  # missing axis value
        {"axes": ("M",), "M": 5, "N": 7},  # unexpected axis value
    ],
)
def test_select_leq_config_rejects_bad_calls(kwargs):
    with pytest.raises(TypeError):
        select_leq_config({"any": {}}, **kwargs)


_QUANT_JSONS = sorted(
    glob.glob(f"{AITER_TRITON_CONFIGS_PATH}/*/gluon/quant/*/DEFAULT.json")
)
_KEY_RE = re.compile(r"^(any|[A-Z]+_(LEQ|GEQ)_\d+(\.[A-Z]+_(LEQ|GEQ)_\d+)*)$")
_REQUIRED = {"NUM_ITER", "BLOCK_SIZE_M", "BLOCK_SIZE_N", "NUM_WARPS"}
_OPTIONAL = {"NUM_STAGES", "NUM_BUFFERS", "waves_per_eu"}


def _ids(paths):
    return [os.path.relpath(p, AITER_TRITON_CONFIGS_PATH) for p in paths]


def test_quant_config_files_exist():
    assert _QUANT_JSONS, "no Gluon quant config files found"


@pytest.mark.parametrize("path", _QUANT_JSONS, ids=_ids(_QUANT_JSONS))
def test_quant_config_table(path):
    with open(path) as f:
        table = json.load(f)
    assert "any" in table
    required = set(_REQUIRED)
    if "/gfx950/" in path and path.endswith("/mxfp4/DEFAULT.json"):
        required.add("NUM_STAGES")
    for key, cfg in table.items():
        assert _KEY_RE.match(key), f"bad bucket key {key!r}"
        keys = set(cfg)
        assert required <= keys <= required | _OPTIONAL, f"{key}: keys {sorted(keys)}"
        assert all(isinstance(v, int) for v in cfg.values()), key
        assert is_power_of_2(cfg["BLOCK_SIZE_M"]), key
        assert is_power_of_2(cfg["BLOCK_SIZE_N"]), key
        assert cfg["BLOCK_SIZE_N"] % 32 == 0, key
        assert cfg["NUM_WARPS"] in (1, 2, 4, 8), key


@pytest.mark.parametrize("path", _QUANT_JSONS, ids=_ids(_QUANT_JSONS))
def test_quant_config_resolves_every_shape(path):
    with open(path) as f:
        table = json.load(f)
    axis_names = {
        part.split("_")[0] for key in table if key != "any" for part in key.split(".")
    }
    other = ({"N", "K"} & axis_names or {"N"}).pop()
    for m in (1, 2, 3, 8, 17, 32, 33, 100, 1024, 1025, 8192, 131072):
        for v in (32, 64, 96, 128, 1024, 1025, 3072, 7168, 16384, 16385, 53248):
            cfg = select_leq_config(table, axes=("M", other), M=m, **{other: v})
            assert _REQUIRED <= set(cfg)
