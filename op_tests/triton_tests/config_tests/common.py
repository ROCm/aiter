# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Shared utilities, discovery helpers, schema assertions, and architecture
virtualization context manager for the modular configuration test suite.

Addresses ROCm/aiter#6105.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import sys
from unittest.mock import MagicMock

# On systems where Triton is not installed (e.g. CPU test runners, developer laptops),
# install a minimal arithmetic shim in sys.modules so imports of arch_info and
# config_utils succeed. Kernel execution is NOT faked: this suite only validates
# host-side configuration discovery, JSON parsing, schemas, and loader resolution.
if "triton" not in sys.modules:
    try:
        import triton  # noqa: F401
    except (ImportError, ModuleNotFoundError):
        _mock_triton = MagicMock()
        _mock_triton.cdiv = lambda a, b: (a + b - 1) // b
        _mock_triton.next_power_of_2 = lambda n: 1 if n <= 1 else 2 ** (n - 1).bit_length()
        sys.modules.setdefault("triton", _mock_triton)
        sys.modules.setdefault("triton.language", MagicMock())
        sys.modules.setdefault("triton.runtime", MagicMock())

# Ensure math helpers are callable integers even if triton was pre-mocked by runner
_t = sys.modules.get("triton", None)
if _t is not None:
    if not hasattr(_t, "cdiv") or isinstance(_t.cdiv, MagicMock):
        _t.cdiv = lambda a, b: (a + b - 1) // b
    if not hasattr(_t, "next_power_of_2") or isinstance(_t.next_power_of_2, MagicMock):
        _t.next_power_of_2 = lambda n: 1 if n <= 1 else 2 ** (n - 1).bit_length()
_rocm_home = os.getenv("ROCM_PATH", "/opt/rocm")
if not os.path.exists(_rocm_home) and not os.path.exists(f"{_rocm_home}/.info/version"):
    os.environ.setdefault("AITER_TRITON_ONLY", "1")

from aiter.ops.triton.utils._triton import arch_info  # noqa: E402
from aiter.ops.triton.utils import (  # noqa: E402
    config_utils,
    conv_config_utils,
    gemm_config_utils,
    mhc_config_utils,
    moe_config_utils,
    normalization_config_utils,
    quant_config_utils,
    sonicmoe_config_utils,
    tuned_config_utils,
    unified_attention_utils,
)
import aiter.ops.triton.gmm as gmm  # noqa: E402

CONFIGS_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../../aiter/ops/triton/configs")
)


def rel_path(path: str) -> str:
    """Return path relative to the configs root directory."""
    return os.path.relpath(path, CONFIGS_PATH)


def discover_all_config_files() -> list[str]:
    """Authoritative discovery of all JSON configuration files in tree."""
    files = []
    for root, _, filenames in os.walk(CONFIGS_PATH):
        for fname in filenames:
            if fname.endswith(".json"):
                files.append(os.path.join(root, fname))
    return sorted(files)


def discover_architectures() -> list[str]:
    """Discover all valid target GPU architectures present in config tree."""
    return sorted(
        d for d in os.listdir(CONFIGS_PATH)
        if os.path.isdir(os.path.join(CONFIGS_PATH, d)) and not d.startswith(".")
    )


def filter_configs_by_op(op: str) -> list[str]:
    """Return all configuration files belonging to a specific operator family."""
    all_files = discover_all_config_files()
    target = f"{os.sep}{op}{os.sep}"
    return [p for p in all_files if target in p]


def is_power_of_two(n: int) -> bool:
    """Check if n is a positive power of two."""
    return isinstance(n, int) and n > 0 and (n & (n - 1)) == 0


def check_no_mixed_geq_leq(data: dict, filepath: str) -> None:
    """Assert that threshold keys adhere to canonical LEQ/GEQ naming and partitioned bounds.

    1. No legacy non-canonical tokens (e.g. '_LE_' or '_GE_') are used instead of 'LEQ'/'GEQ'.
    2. If a 1D table specifies both LEQ and GEQ bounds for the same axis (e.g. M_LEQ_* and M_GEQ_*),
       they must form non-overlapping partitions (max(LEQ) <= min(GEQ)), following the canonical
       resolution order (LEQ ascending followed by GEQ descending) documented in configs/CLAUDE.md.
    """
    legacy_keys = [
        k for k in data
        if "_LE_" in k or "_GE_" in k or k.startswith("LE_") or k.startswith("GE_")
    ]
    assert not legacy_keys, (
        f"{rel_path(filepath)}: Non-canonical legacy LE/GE key format found: {legacy_keys}"
    )

    m_leq = [
        int(k.split("_")[-1])
        for k in data
        if "." not in k and k.startswith("M_LEQ_") and k.split("_")[-1].isdigit()
    ]
    m_geq = [
        int(k.split("_")[-1])
        for k in data
        if "." not in k and k.startswith("M_GEQ_") and k.split("_")[-1].isdigit()
    ]
    if m_leq and m_geq:
        assert max(m_leq) <= min(m_geq), (
            f"{rel_path(filepath)}: Overlapping M_LEQ ({max(m_leq)}) and M_GEQ ({min(m_geq)}) ranges"
        )


def load_json_dict(filepath: str) -> dict:
    """Load JSON file and assert it is a non-empty dictionary."""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert isinstance(data, dict), f"{rel_path(filepath)}: root JSON object must be a dict"
    assert len(data) > 0, f"{rel_path(filepath)}: configuration dictionary is empty"
    return data


def clear_all_config_caches() -> None:
    """Flush all 16 production LRU caches across all config loaders in AITER."""
    config_utils.load_config_json.cache_clear()
    config_utils._bucket_index.cache_clear()
    gemm_config_utils._get_gemm_config_cached.cache_clear()
    conv_config_utils._get_conv_config_cached.cache_clear()
    conv_config_utils.has_conv_config.cache_clear()
    moe_config_utils.get_moe_dispatch.cache_clear()
    mhc_config_utils._c_thresholds.cache_clear()
    mhc_config_utils.get_mhc_config.cache_clear()
    mhc_config_utils.get_mhc_post_config.cache_clear()
    mhc_config_utils.get_mhc_fused_post_pre_delayed_rmsnorm_config.cache_clear()
    quant_config_utils._axes_from_keys.cache_clear()
    normalization_config_utils.get_normalization_config.cache_clear()
    tuned_config_utils._get_tuned_kernel_entry.cache_clear()
    unified_attention_utils._get_unified_attention_config_cached.cache_clear()
    unified_attention_utils._index.cache_clear()
    gmm.get_config.cache_clear()


@contextlib.contextmanager
def set_test_arch(arch: str):
    """Context manager to test config loaders for any target GPU architecture.

    Dynamically virtualizes arch_info.get_arch() and arch_info._CACHED_ARCH,
    and flushes all LRU caches before and after to guarantee 100% cache isolation.
    Restores original functions and cache states on exit.
    """
    orig_cached = getattr(arch_info, "_CACHED_ARCH", None)
    orig_func = arch_info.get_arch

    arch_info._CACHED_ARCH = arch
    arch_info.get_arch = lambda: arch

    orig_sonic_func = getattr(sonicmoe_config_utils, "get_arch", None)
    sonicmoe_config_utils.get_arch = lambda: arch

    clear_all_config_caches()
    try:
        yield
    finally:
        arch_info._CACHED_ARCH = orig_cached
        arch_info.get_arch = orig_func
        if orig_sonic_func is not None:
            sonicmoe_config_utils.get_arch = orig_sonic_func
        clear_all_config_caches()


_SPECIALIZED_CONFIG_RE = re.compile(
    r"^([A-Za-z0-9][A-Za-z0-9_-]*)-([A-Za-z0-9](?:[A-Za-z0-9_=-]*[A-Za-z0-9])?)\.json$"
)


def validate_config_file_layout_and_naming(path: str) -> None:
    """Verify that a config file strictly follows configs/CLAUDE.md rules:
    - Path matches <configs>/<arch>/<backend>/<op>/<dtype>/<filename> (exactly 5 components)
    - <arch> matches config_utils._ARCH_SAFE_RE
    - <backend> is in _VALID_BACKENDS ('triton', 'gluon')
    - <op> matches config_utils._OP_RE
    - <dtype> is lowercase with underscores, no hyphens
    - Filename has no architecture prefix
    - Filename is either DEFAULT.json or <CONFIG_NAME>-<suffix>.json (or legacy mha.json)
    """
    rel = rel_path(path)
    parts = rel.split(os.sep)
    assert len(parts) == 5, (
        f"Path must have exactly 5 components (<arch>/<backend>/<op>/<dtype>/<filename>): {rel}"
    )

    arch, backend, op, dtype = parts[0], parts[1], parts[2], parts[3]
    fname = parts[4]

    assert config_utils._ARCH_SAFE_RE.fullmatch(arch), f"{rel}: invalid arch identifier '{arch}'"
    assert backend in config_utils._VALID_BACKENDS, f"{rel}: invalid backend '{backend}'"
    assert config_utils._OP_RE.fullmatch(op), f"{rel}: invalid op identifier '{op}'"
    assert dtype == dtype.lower(), f"{rel}: dtype directory must be lowercase"
    assert "-" not in dtype, f"{rel}: dtype directory must not contain hyphens"

    # No architecture prefix in filenames
    for a in discover_architectures():
        assert not fname.startswith(f"{a}_") and not fname.startswith(f"{a}-"), (
            f"{rel}: filename must not have arch prefix '{a}'"
        )

    # Valid filename formats
    if fname not in ("DEFAULT.json", "mha.json"):
        assert fname.endswith(".json"), f"{rel}: specialized file must end with '.json'"
        assert _SPECIALIZED_CONFIG_RE.fullmatch(fname), (
            f"{rel}: specialized file '{fname}' must follow <CONFIG_NAME>-<suffix>.json format"
        )
        stem = fname[:-5]
        parts = stem.split("-")
        matching_splits = [
            ("-".join(parts[:i]), "-".join(parts[i:]))
            for i in range(1, len(parts))
            if config_utils._dtype_dir("-".join(parts[:i])) == dtype and len("-".join(parts[i:])) > 0
        ]
        assert len(matching_splits) == 1, (
            f"{rel}: specialized file '{fname}' must have exactly one valid decomposition "
            f"matching parent dtype directory '{dtype}' under <CONFIG_NAME>-<suffix>.json convention, "
            f"found {len(matching_splits)}"
        )


def validate_gemm_config_table(
    table: dict,
    path: str = "dummy.json",
    backend: str | None = None,
) -> None:
    """Validate GEMM configuration table structure:
    - Must contain an 'any' fallback
    - No mixing of M_LEQ and M_GEQ conventions in the same file
    - Keys are M_LEQ_x, M_GEQ_x, 'any', or recognized metadata
    - Backend-specific required parameter set (triton vs gluon)
    - Block sizes are positive powers of 2
    - num_warps is in {1, 2, 4, 8, 16, 32}
    - num_stages is a positive integer
    """
    display_path = rel_path(path) if os.path.isabs(path) else path
    assert "any" in table, (
        f"{display_path}: GEMM table must contain an 'any' fallback"
    )
    check_no_mixed_geq_leq(table, path)

    if backend is None:
        try:
            rel = rel_path(path)
            parts = rel.split(os.sep)
            if len(parts) >= 2 and parts[1] in config_utils._VALID_BACKENDS:
                backend = parts[1]
            else:
                backend = "triton"
        except Exception:
            backend = "triton"

    for key, cfg in table.items():
        if key in ("M_BOUNDS", "_note", "DEFAULT_FALLBACK"):
            continue
        assert key == "any" or key.startswith("M_LEQ_") or key.startswith("M_GEQ_"), (
            f"{display_path}: unexpected key '{key}'"
        )
        assert isinstance(cfg, dict), f"{display_path}: {key} must map to a dict"
        assert len(cfg) > 0, f"{display_path} [{key}]: config entry cannot be empty"

        # Validate backend-specific required parameter set
        if backend == "triton":
            assert "BLOCK_SIZE_M" in cfg, f"{display_path} [{key}]: missing required 'BLOCK_SIZE_M'"
            assert "BLOCK_SIZE_N" in cfg, f"{display_path} [{key}]: missing required 'BLOCK_SIZE_N'"
            assert "num_warps" in cfg or "waves_per_eu" in cfg, (
                f"{display_path} [{key}]: missing required 'num_warps' or 'waves_per_eu'"
            )
        elif backend == "gluon":
            has_m = "BLOCK_SIZE_M" in cfg or "BLOCK_M" in cfg
            has_n = "BLOCK_SIZE_N" in cfg or "BLOCK_N" in cfg
            has_warps = "num_warps" in cfg or "NUM_WARPS" in cfg
            assert has_m, f"{display_path} [{key}]: Gluon entry missing 'BLOCK_SIZE_M' or 'BLOCK_M'"
            assert has_n, f"{display_path} [{key}]: Gluon entry missing 'BLOCK_SIZE_N' or 'BLOCK_N'"
            assert has_warps, f"{display_path} [{key}]: Gluon entry missing 'num_warps' or 'NUM_WARPS'"

        for param in ("BLOCK_SIZE_M", "BLOCK_SIZE_N", "BLOCK_SIZE_K", "BLOCK_M", "BLOCK_N", "BLOCK_K"):
            if param in cfg:
                val = cfg[param]
                assert is_power_of_two(val), (
                    f"{display_path} [{key}]: {param}={val} must be a positive power of 2"
                )

        for w_param in ("num_warps", "NUM_WARPS"):
            if w_param in cfg:
                warps = cfg[w_param]
                assert warps in (1, 2, 4, 8, 16, 32), (
                    f"{display_path} [{key}]: unexpected {w_param}={warps}"
                )

        for s_param in ("num_stages", "NUM_STAGES"):
            if s_param in cfg:
                stages = cfg[s_param]
                assert isinstance(stages, int) and stages > 0, (
                    f"{display_path} [{key}]: {s_param} must be positive integer, got {stages}"
                )
