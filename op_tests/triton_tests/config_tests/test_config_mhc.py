# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Dedicated test suite for Multi-Head Convolution / Attention (MHC) kernel configurations.

Validates all 17 MHC configuration files across architectures:
1. Schema conformity: valid M_LEQ, C_, or fallback keys; no mixed GEQ/LEQ.
2. Specialized file resolution: exercises 100% of the 11 C-specialized configuration files.
3. Default loader resolution: get_mhc_config, get_mhc_post_config, and fused post-pre delayed RMSNorm.
"""

from __future__ import annotations

import json
import os
import re
import pytest

from aiter.ops.triton.utils import (
    config_utils,
    mhc_config_utils,
)
from op_tests.triton_tests.config_tests.common import (
    check_no_mixed_geq_leq,
    discover_architectures,
    filter_configs_by_op,
    is_power_of_two,
    rel_path,
    set_test_arch,
)

MHC_FILES = filter_configs_by_op("mhc")
DISCOVERED_ARCHS = discover_architectures()


@pytest.mark.parametrize("path", MHC_FILES, ids=rel_path)
def test_mhc_config_schema(path: str):
    """Verify MHC config structure contains valid keys and power-of-2 tile blocks."""
    with open(path, "r", encoding="utf-8") as f:
        table = json.load(f)

    has_valid_keys = any(
        k in ("any", "default")
        or k.startswith("M_LEQ_")
        or k.startswith("C_")
        or k.startswith("_mhc_")
        for k in table
    )
    assert has_valid_keys, f"{rel_path(path)}: MHC table missing standard keys"

    for k, v in table.items():
        if isinstance(v, dict):
            assert len(v) > 0, f"{rel_path(path)} [{k}]: entry cannot be empty"
            for bkey in ("BLOCK_M", "BLOCK_K", "BLOCK_C", "TILE_K"):
                if bkey in v:
                    val = v[bkey]
                    assert is_power_of_two(val), (
                        f"{rel_path(path)} [{k}]: {bkey}={val} must be a positive power of 2"
                    )
            if "num_warps" in v:
                warps = v["num_warps"]
                assert is_power_of_two(warps), (
                    f"{rel_path(path)} [{k}]: num_warps={warps} must be a positive power of 2"
                )
    check_no_mixed_geq_leq(table, path)


def _get_mhc_c_specialized_files():
    files = []
    for arch in DISCOVERED_ARCHS:
        mhc_dir = os.path.join(config_utils.AITER_TRITON_CONFIGS_PATH, arch, "triton", "mhc")
        if not os.path.isdir(mhc_dir):
            continue
        for root, _, fnames in os.walk(mhc_dir):
            for fname in fnames:
                m = re.search(r"-C=(\d+)\.json$", fname)
                if m:
                    files.append((arch, int(m.group(1)), os.path.join(root, fname)))
    return sorted(files)


MHC_C_FILES = _get_mhc_c_specialized_files()


@pytest.mark.parametrize(
    "arch,c_val,fpath",
    MHC_C_FILES,
    ids=[f"{x[0]}-C={x[1]}" for x in MHC_C_FILES],
)
def test_mhc_c_specialized_resolution(arch: str, c_val: int, fpath: str):
    """Test get_mhc_config resolves ALL C-specialized MHC configuration files."""
    with set_test_arch(arch):
        cfg, used_spec = mhc_config_utils.get_mhc_config(
            "MHC_FUSED", M=1, C=c_val, mode="sinkhorn"
        )
        assert used_spec is True
        assert isinstance(cfg, dict)
        assert len(cfg) > 0


@pytest.mark.parametrize("arch", ["gfx942", "gfx950"])
def test_mhc_loaders_default_resolution(arch: str):
    """Test get_mhc_config, get_mhc_post_config, and get_mhc_fused_post_pre_delayed_rmsnorm_config."""
    with set_test_arch(arch):
        cfg, _ = mhc_config_utils.get_mhc_config(
            "MHC_FUSED", M=1, C=128, mode="sinkhorn"
        )
        assert isinstance(cfg, dict)

        post_cfg = mhc_config_utils.get_mhc_post_config(M=1, C=512)
        assert isinstance(post_cfg, dict)

        fused_post_cfg = mhc_config_utils.get_mhc_fused_post_pre_delayed_rmsnorm_config(M=1)
        assert isinstance(fused_post_cfg, dict)
