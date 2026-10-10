# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Dedicated test suite for Mixture of Experts (MoE) kernel configurations.

Validates all 8 MoE configuration files across architectures:
1. Schema conformity: valid dispatch and kernel entries.
2. Production loader resolution: get_moe_dispatch across published architectures and backends.
3. Fallback behavior: un-shipped architectures fall back cleanly to empty dictionary.
4. SonicMoE configuration resolution: direct loader resolution and architecture fallback.
"""

from __future__ import annotations

import json
import pytest

from aiter.ops.triton.utils import (
    moe_config_utils,
)
from op_tests.triton_tests.config_tests.common import (
    check_no_mixed_geq_leq,
    filter_configs_by_op,
    is_power_of_two,
    rel_path,
    set_test_arch,
)

MOE_FILES = filter_configs_by_op("moe")


@pytest.mark.parametrize("path", MOE_FILES, ids=rel_path)
def test_moe_config_schema(path: str):
    """Verify MOE config structure contains valid dispatch or kernel entries and power-of-2 tile sizes."""
    with open(path, "r", encoding="utf-8") as f:
        table = json.load(f)

    for key, cfg in table.items():
        if key.startswith("_") or key == "comment":
            continue
        assert isinstance(cfg, dict), f"{rel_path(path)}: {key} must map to a dict"
        assert len(cfg) > 0, f"{rel_path(path)} [{key}]: config entry cannot be empty"

        for bkey in ("block_n", "block_k", "BLOCK_SIZE_K", "BLOCK_SIZE_N", "BLOCK_M", "BLOCK_N"):
            if bkey in cfg:
                val = cfg[bkey]
                assert is_power_of_two(val), (
                    f"{rel_path(path)} [{key}]: {bkey}={val} must be a positive power of 2"
                )
        if "num_warps" in cfg:
            warps = cfg["num_warps"]
            assert is_power_of_two(warps), (
                f"{rel_path(path)} [{key}]: num_warps={warps} must be a positive power of 2"
            )
    check_no_mixed_geq_leq(table, path)


@pytest.mark.parametrize(
    "config_name,arch,backend",
    [
        ("A16W4", "gfx942", "triton"),
        ("A8W4", "gfx950", "triton"),
        ("A8W4", "gfx1250", "triton"),
        ("A4W4", "gfx1250", "gluon"),
        ("A8W4", "gfx1250", "gluon"),
    ],
)
def test_moe_dispatch_loader_resolves(config_name: str, arch: str, backend: str):
    """Test get_moe_dispatch resolution for all published MOE dispatch tables."""
    with set_test_arch(arch):
        dispatch = moe_config_utils.get_moe_dispatch(config_name, arch=arch, backend=backend)
        assert isinstance(dispatch, dict)
        assert len(dispatch) > 0, f"MOE {config_name} {backend} dispatch empty on {arch}"


def test_moe_unsupported_arch_returns_empty():
    """Verify that requesting an un-shipped MOE table falls back cleanly to empty dict."""
    with set_test_arch("gfx942"):
        dispatch = moe_config_utils.get_moe_dispatch("A8W4", arch="gfx942", backend="triton")
        assert dispatch == {}


def test_sonicmoe_loader_resolves():
    """Test SonicMoE config resolution across supported and fallback architectures."""
    from aiter.ops.triton.utils import sonicmoe_config_utils

    with set_test_arch("gfx942"):
        cfg_942 = sonicmoe_config_utils.load_sonicmoe_configs()
        assert isinstance(cfg_942, dict)
        assert len(cfg_942) > 0

    # gfx950 falls back to gfx942 SonicMoE table
    with set_test_arch("gfx950"):
        cfg_950 = sonicmoe_config_utils.load_sonicmoe_configs()
        assert isinstance(cfg_950, dict)
        assert len(cfg_950) > 0
