# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Dedicated test suite for Attention and Multi-Head Attention kernel configurations.

Validates all 44 Attention configuration files across architectures:
1. Schema conformity: non-empty configuration dictionaries, power-of-2 tile bounds.
2. Unified attention loader resolution: _load resolves table and axes without GPU compilation.
3. Tuned kernel attention loader resolution: _get_tuned_kernel_entry.
4. Legacy configuration validation: explicit handling of gluon mha.json.
"""

from __future__ import annotations

import json
import os
import pytest

from aiter.ops.triton.utils import (
    config_utils,
    tuned_config_utils,
    unified_attention_utils,
)
from op_tests.triton_tests.config_tests.common import (
    check_no_mixed_geq_leq,
    filter_configs_by_op,
    is_power_of_two,
    rel_path,
    set_test_arch,
)

ATTENTION_FILES = filter_configs_by_op("attention")


@pytest.mark.parametrize("path", ATTENTION_FILES, ids=rel_path)
def test_attention_config_schema(path: str):
    """Verify Attention config tables have non-empty dictionary content and power-of-2 tile blocks."""
    with open(path, "r", encoding="utf-8") as f:
        table = json.load(f)
    assert isinstance(table, dict)
    assert len(table) > 0, f"{rel_path(path)}: empty table"
    check_no_mixed_geq_leq(table, path)

    # Check tile parameters when present in entries
    for k, v in table.items():
        if isinstance(v, dict):
            assert len(v) > 0, f"{rel_path(path)} [{k}]: entry cannot be empty"
            for param in ("BLOCK_M", "BLOCK_N", "BLOCK_SIZE_M", "BLOCK_SIZE_N"):
                if param in v:
                    val = v[param]
                    assert is_power_of_two(val), (
                        f"{rel_path(path)} [{k}]: {param}={val} must be a power of 2"
                    )
            if "num_warps" in v:
                warps = v["num_warps"]
                assert warps in (1, 2, 4, 8, 16, 32), (
                    f"{rel_path(path)} [{k}]: unexpected num_warps={warps}"
                )


@pytest.mark.parametrize("arch", ["gfx942", "gfx950", "gfx1250"])
def test_unified_attention_loader_resolves(arch: str):
    """Test unified attention _load resolves schema and config tables."""
    with set_test_arch(arch):
        table, axes, cfg_dir = unified_attention_utils._load(
            "attn_2d", backend="triton", arch=arch
        )
        assert isinstance(table, dict)
        assert len(axes) > 0
        assert os.path.isdir(cfg_dir)


def test_tuned_kernel_loader_resolves():
    """Test _get_tuned_kernel_entry loads published kernel configs."""
    with set_test_arch("gfx950"):
        fpath, entry = tuned_config_utils._get_tuned_kernel_entry(
            "attention",
            "CHUNK_DELTA_ATTN",
            "chunk_gla_fwd_kernel_o",
            "triton",
        )
        assert entry is not None
        assert "BLOCK_M" in entry or "BLOCK_N" in entry or "num_warps" in entry


def test_legacy_mha_json_validity():
    """Verify legacy mha.json file parses and conforms to top-level dict schema."""
    legacy_path = os.path.join(
        config_utils.AITER_TRITON_CONFIGS_PATH,
        "gfx950",
        "gluon",
        "attention",
        "mha",
        "mha.json",
    )
    assert os.path.isfile(legacy_path), f"Legacy config not found: {legacy_path}"
    with open(legacy_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert isinstance(data, dict)
    assert len(data) > 0
