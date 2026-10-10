# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Dedicated test suite for Convolution kernel configurations.

Validates all 107 Convolution configuration files across architectures:
1. Schema conformity: table contains 'any', 'shapes', or M-bounds.
2. Production loader resolution: has_conv_config and get_conv_config resolve family defaults.
3. Exact shape-key resolution when shapes dictionary is published.
"""

from __future__ import annotations

import json
import os
import pytest

from aiter.ops.triton.utils import (
    config_utils,
    conv_config_utils,
)
from op_tests.triton_tests.config_tests.common import (
    check_no_mixed_geq_leq,
    discover_architectures,
    filter_configs_by_op,
    is_power_of_two,
    rel_path,
    set_test_arch,
)

CONV_FILES = filter_configs_by_op("conv")
DISCOVERED_ARCHS = discover_architectures()


@pytest.mark.parametrize("path", CONV_FILES, ids=rel_path)
def test_conv_config_schema(path: str):
    """Verify Conv table structure contains 'any', shapes, or M-bounds and valid tile blocks."""
    with open(path, "r", encoding="utf-8") as f:
        table = json.load(f)

    has_any = "any" in table
    has_shapes = any(k in table for k in ("shapes", "shapes_nhwc", "shapes_nchw"))
    has_m_bounds = any(k.startswith("M_LEQ_") for k in table)
    assert has_any or has_shapes or has_m_bounds, (
        f"{rel_path(path)}: Conv table must have shapes, M_LEQ, or any"
    )
    check_no_mixed_geq_leq(table, path)

    # Validate tile parameters in entries
    entries = []
    for k, v in table.items():
        if isinstance(v, dict):
            if k in ("shapes", "shapes_nhwc", "shapes_nchw"):
                for sk, sv in v.items():
                    if isinstance(sv, dict):
                        entries.append((f"{k}.{sk}", sv))
            else:
                entries.append((k, v))

    for name, cfg in entries:
        for bkey in ("BLOCK_M", "BLOCK_N", "BLOCK_K", "BLOCK_SIZE_M", "BLOCK_SIZE_N", "BLOCK_SIZE_K"):
            if bkey in cfg:
                val = cfg[bkey]
                assert is_power_of_two(val), (
                    f"{rel_path(path)} [{name}]: {bkey}={val} must be a positive power of 2"
                )
        if "num_warps" in cfg:
            warps = cfg["num_warps"]
            assert warps in (1, 2, 4, 8, 16, 32), (
                f"{rel_path(path)} [{name}]: unexpected num_warps={warps}"
            )


def _get_conv_families():
    families = []
    for arch in DISCOVERED_ARCHS:
        conv_dir = os.path.join(config_utils.AITER_TRITON_CONFIGS_PATH, arch, "triton", "conv")
        if not os.path.isdir(conv_dir):
            continue
        for dtype_dir in os.listdir(conv_dir):
            fdir = os.path.join(conv_dir, dtype_dir)
            if os.path.isdir(fdir) and os.path.exists(os.path.join(fdir, "DEFAULT.json")):
                config_name = dtype_dir.upper().replace("_", "-")
                families.append((arch, config_name, os.path.join(fdir, "DEFAULT.json")))
    return families


CONV_FAMILIES = _get_conv_families()


@pytest.mark.parametrize(
    "arch,config_name,fpath",
    CONV_FAMILIES,
    ids=[f"{x[0]}-{x[1]}" for x in CONV_FAMILIES],
)
def test_conv_loader_resolves_family(arch, config_name, fpath):
    """Test get_conv_config and has_conv_config resolution across all Conv families."""
    with set_test_arch(arch):
        assert conv_config_utils.has_conv_config(config_name) is True
        cfg = conv_config_utils.get_conv_config(config_name, M=1)
        assert isinstance(cfg, dict)
        assert len(cfg) > 0

        # If file publishes exact shapes, test exact-shape resolution
        file_data = config_utils.load_config_json(fpath)
        shapes = file_data.get("shapes", {})
        if shapes:
            shape_key = next(iter(shapes))
            exact_cfg = conv_config_utils.get_conv_config(config_name, shape_key=shape_key)
            assert exact_cfg == shapes[shape_key]
