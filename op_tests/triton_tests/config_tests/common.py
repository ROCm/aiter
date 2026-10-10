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
import aiter.ops.triton._triton_kernels.gmm as gmm_kernels  # noqa: E402

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
    gmm_kernels.get_config.cache_clear()


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
