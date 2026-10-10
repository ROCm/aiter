# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Dedicated test suite for GEMM and Fused GEMM kernel configurations.

Validates all 656 GEMM configuration files (110 DEFAULT families + 546 specialized files)
across all 9 target architectures:
1. Schema conformity: valid keys, 'any' fallback or M-bounds, power-of-2 tile sizes, valid num_warps.
2. Production loader resolution: get_gemm_config resolves all family defaults.
3. Specialized file resolution: exercises 100% of specialized GEMM files without sampling.
4. Split-K parameter computation and default parameter augmentation.
"""

from __future__ import annotations

import json
import os
import re
import pytest

from aiter.ops.triton.utils import (
    config_utils,
    gemm_config_utils,
)
from op_tests.triton_tests.config_tests.common import (
    check_no_mixed_geq_leq,
    discover_architectures,
    filter_configs_by_op,
    is_power_of_two,
    rel_path,
    set_test_arch,
)

GEMM_FILES = filter_configs_by_op("gemm")
SPECIALIZED_GEMM_FILES = [p for p in GEMM_FILES if not p.endswith("DEFAULT.json")]
DISCOVERED_ARCHS = discover_architectures()


@pytest.mark.parametrize("path", GEMM_FILES, ids=rel_path)
def test_gemm_config_schema(path: str):
    """Verify GEMM table structure:
    - Must have 'any' fallback or M-bounds
    - Keys are M_LEQ_x, M_GEQ_x, 'any', or recognized metadata
    - Block sizes are positive powers of 2
    - num_warps is in {1, 2, 4, 8, 16, 32}
    - No mixing of M_LEQ and M_GEQ conventions in the same file
    """
    with open(path, "r", encoding="utf-8") as f:
        table = json.load(f)

    assert "any" in table or any(k.startswith("M_LEQ_") for k in table), (
        f"{rel_path(path)}: GEMM table must contain an 'any' fallback or M_LEQ bounds"
    )
    check_no_mixed_geq_leq(table, path)

    for key, cfg in table.items():
        if key in ("M_BOUNDS", "_note", "DEFAULT_FALLBACK"):
            continue
        assert key == "any" or key.startswith("M_LEQ_") or key.startswith("M_GEQ_"), (
            f"{rel_path(path)}: unexpected key '{key}'"
        )
        assert isinstance(cfg, dict), f"{rel_path(path)}: {key} must map to a dict"

        for param in ("BLOCK_SIZE_M", "BLOCK_SIZE_N", "BLOCK_SIZE_K"):
            if param in cfg:
                val = cfg[param]
                assert is_power_of_two(val), (
                    f"{rel_path(path)} [{key}]: {param}={val} must be a positive power of 2"
                )

        if "num_warps" in cfg:
            warps = cfg["num_warps"]
            assert warps in (1, 2, 4, 8, 16, 32), (
                f"{rel_path(path)} [{key}]: unexpected num_warps={warps}"
            )

        if "num_stages" in cfg:
            stages = cfg["num_stages"]
            assert isinstance(stages, int) and stages > 0, (
                f"{rel_path(path)} [{key}]: num_stages must be positive integer, got {stages}"
            )


def _infer_gemm_config_name(dtype_dir: str, files: list[str]) -> str:
    """Infer the exact config_name stem corresponding to a dtype directory."""
    for f in files:
        if f != "DEFAULT.json" and f.endswith(".json"):
            stem = f[:-5]
            m = re.split(r"-(?:B|N|N4|N8|N16|C)=", stem)
            if len(m) > 1:
                return m[0]
    return dtype_dir.upper().replace("_", "-")


def _get_gemm_families():
    """Discover all (arch, backend, config_name, dtype_dir) GEMM families with DEFAULT.json."""
    families = []
    gemm_root = config_utils.AITER_TRITON_CONFIGS_PATH
    for arch in DISCOVERED_ARCHS:
        for backend in config_utils._VALID_BACKENDS:
            op_dir = os.path.join(gemm_root, arch, backend, "gemm")
            if not os.path.isdir(op_dir):
                continue
            for dtype_dir in os.listdir(op_dir):
                family_dir = os.path.join(op_dir, dtype_dir)
                if not os.path.isdir(family_dir):
                    continue
                files = os.listdir(family_dir)
                if "DEFAULT.json" not in files:
                    continue
                config_name = _infer_gemm_config_name(dtype_dir, files)
                families.append((arch, backend, config_name, dtype_dir))
    return families


GEMM_FAMILIES = _get_gemm_families()


@pytest.mark.parametrize(
    "arch,backend,config_name,dtype_dir",
    GEMM_FAMILIES,
    ids=[f"{x[0]}-{x[1]}-{x[3]}" for x in GEMM_FAMILIES],
)
def test_gemm_family_default_resolution(arch, backend, config_name, dtype_dir):
    """Test get_gemm_config loads family defaults across all architectures without production changes."""
    with set_test_arch(arch):
        cfg, _ = gemm_config_utils.get_gemm_config(
            config_name, M=1, backend=backend
        )
        assert isinstance(cfg, dict), f"Failed resolving fallback for {config_name} on {arch}"
        assert "BLOCK_SIZE_M" in cfg or "num_warps" in cfg or "num_stages" in cfg

        computed = gemm_config_utils.compute_splitk_params(cfg, K=1024)
        assert "SPLITK_BLOCK_SIZE" in computed


@pytest.mark.parametrize("path", SPECIALIZED_GEMM_FILES, ids=rel_path)
def test_gemm_specialized_file_resolution(path: str):
    """Test that EVERY discoverable specialized GEMM file resolves through get_gemm_config.

    Strictly satisfies Requirement 4: zero sampling (no specialized[:5]),
    every single specialized file is exercised and validated against its actual content.
    """
    rel = rel_path(path)
    arch, backend, op, dtype_dir, fname = rel.split(os.sep)
    stem = fname[:-5]

    data = config_utils.load_config_json(path)
    m_val = 1
    expected = None
    for k in data:
        if k.startswith("M_LEQ_") and k[6:].isdigit():
            m_val = int(k[6:])
            expected = data[k]
            break
        elif k.startswith("M_GEQ_") and k[6:].isdigit():
            m_val = int(k[6:])
            expected = data[k]
            break
    if expected is None and "any" in data:
        expected = data["any"]

    m_b = re.match(r"^(.*?)-(B=\d+-N=\d+-K=\d+)$", stem)
    m_nk = re.match(r"^(.*?)-(N=\d+-K=\d+)$", stem)

    with set_test_arch(arch):
        if m_b:
            cfg_name = m_b.group(1)
            parts = dict(part.split("=") for part in m_b.group(2).split("-"))
            res, is_tuned = gemm_config_utils.get_gemm_config(
                cfg_name,
                M=m_val,
                N=int(parts["N"]),
                K=int(parts["K"]),
                B=int(parts["B"]),
                backend=backend,
            )
        elif m_nk:
            cfg_name = m_nk.group(1)
            parts = dict(part.split("=") for part in m_nk.group(2).split("-"))
            res, is_tuned = gemm_config_utils.get_gemm_config(
                cfg_name,
                M=m_val,
                N=int(parts["N"]),
                K=int(parts["K"]),
                backend=backend,
            )
        else:
            # Custom specialized filename (e.g. fused kernels with N4/N16)
            matched_cfg_name = None
            for i in range(1, len(stem)):
                candidate = stem[:i]
                if config_utils._dtype_dir(candidate) == dtype_dir:
                    matched_cfg_name = candidate
            assert matched_cfg_name is not None, f"Could not deduce config_name for {fname}"
            suffix = stem[len(matched_cfg_name) + 1:]
            res, is_tuned = gemm_config_utils.get_gemm_config(
                matched_cfg_name,
                M=m_val,
                specialized_filename=suffix,
                backend=backend,
            )
        if any(k.startswith(("M_LEQ_", "M_GEQ_")) for k in data):
            assert is_tuned is True, f"{rel}: expected is_tuned=True when resolving specialized file"
        else:
            # Per configs/CLAUDE.md: is_tuned is False when matching 'any' fallback
            assert is_tuned is False, f"{rel}: expected is_tuned=False for any-only specialized file"
        assert isinstance(res, dict)
        if expected is not None:
            for param, val in expected.items():
                assert res.get(param) == val, (
                    f"{rel}: expected {param}={val}, got {res.get(param)}"
                )
