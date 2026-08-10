# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import json
import os

import pytest

from aiter.ops.triton._triton_kernels.gemm.basic.gemm_a8w8 import _get_config
from aiter.ops.triton.utils._triton import arch_info
from aiter.ops.triton.utils.core import AITER_TRITON_CONFIGS_PATH
from aiter.ops.triton.utils.gemm_config_utils import (
    _get_gemm_config_cached,
    compute_splitk_params,
    get_gemm_config,
)

# gfx1100 (RDNA3) tuning bands from vLLM#51136 reference patch. Decode M<=32 and
# the 33..63 gap (via "any") share one band; prefill M>=64 gets the other.
GFX1100_DECODE = {
    "BLOCK_SIZE_M": 16,
    "BLOCK_SIZE_N": 128,
    "BLOCK_SIZE_K": 128,
    "GROUP_SIZE_M": 4,
    "num_warps": 4,
    "num_stages": 3,
    "waves_per_eu": 2,
    "matrix_instr_nonkdim": 16,
    "kpack": 1,
    "cache_modifier": None,
    "NUM_KSPLIT": 1,
}

GFX1100_PREFILL = {
    "BLOCK_SIZE_M": 64,
    "BLOCK_SIZE_N": 256,
    "BLOCK_SIZE_K": 128,
    "GROUP_SIZE_M": 4,
    "num_warps": 8,
    "num_stages": 2,
    "waves_per_eu": 2,
    "matrix_instr_nonkdim": 16,
    "kpack": 1,
    "cache_modifier": None,
    "NUM_KSPLIT": 1,
}

# Keys the kernel launch consumes (after compute_splitk_params backfills
# SPLITK_BLOCK_SIZE and add_default_gemm_config_params defaults).
KERNEL_CONFIG_KEYS = (
    "BLOCK_SIZE_M",
    "BLOCK_SIZE_N",
    "BLOCK_SIZE_K",
    "GROUP_SIZE_M",
    "num_warps",
    "num_stages",
    "cache_modifier",
    "NUM_KSPLIT",
    "SPLITK_BLOCK_SIZE",
)

# Full schema required for newly shipped configs (see configs/CLAUDE.md).
GFX1100_SCHEMA_KEYS = (
    "BLOCK_SIZE_M",
    "BLOCK_SIZE_N",
    "BLOCK_SIZE_K",
    "GROUP_SIZE_M",
    "num_warps",
    "num_stages",
    "waves_per_eu",
    "matrix_instr_nonkdim",
    "kpack",
    "cache_modifier",
    "NUM_KSPLIT",
)

# Archs that ship a default GEMM-A8W8 config in the legacy flat layout.
GEMM_A8W8_ARCHS = sorted(
    name.split("-")[0]
    for name in os.listdir(os.path.join(AITER_TRITON_CONFIGS_PATH, "gemm"))
    if name.endswith("-GEMM-A8W8.json")
)


@pytest.fixture(autouse=True)
def _fresh_config_caches():
    # _get_gemm_config_cached is lru-cached on a key that does not include the
    # arch, so cross-arch lookups inside one test would otherwise collide.
    _get_gemm_config_cached.cache_clear()
    _get_gemm_config_cached._config_cache = {}
    yield


@pytest.fixture
def set_arch(monkeypatch):
    def _set(arch: str):
        monkeypatch.setattr(arch_info, "_CACHED_ARCH", arch)

    return _set


def test_gfx1100_default_config_file_present():
    fpath = os.path.join(AITER_TRITON_CONFIGS_PATH, "gemm", "gfx1100-GEMM-A8W8.json")
    assert os.path.isfile(fpath), f"missing {fpath}"
    with open(fpath, "r") as f:
        data = json.load(f)
    assert data["M_LEQ_32"] == GFX1100_DECODE
    assert data["M_GEQ_64"] == GFX1100_PREFILL
    assert data["any"] == GFX1100_DECODE


def test_gfx1100_default_config_resolves(set_arch):
    set_arch("gfx1100")
    cfg, is_tuned = get_gemm_config("GEMM-A8W8", 32, 4096, 4096)
    assert is_tuned is False
    assert cfg == GFX1100_DECODE


def test_gfx1100_m_band_selectors(set_arch):
    set_arch("gfx1100")
    assert get_gemm_config("GEMM-A8W8", 1, 4096, 4096)[0] == GFX1100_DECODE
    assert get_gemm_config("GEMM-A8W8", 32, 4096, 4096)[0] == GFX1100_DECODE
    assert get_gemm_config("GEMM-A8W8", 33, 4096, 4096)[0] == GFX1100_DECODE
    assert get_gemm_config("GEMM-A8W8", 63, 4096, 4096)[0] == GFX1100_DECODE
    assert get_gemm_config("GEMM-A8W8", 64, 4096, 4096)[0] == GFX1100_PREFILL
    assert get_gemm_config("GEMM-A8W8", 2048, 4096, 4096)[0] == GFX1100_PREFILL


def test_gfx1100_schema_and_splitk(set_arch):
    set_arch("gfx1100")
    for m in (1, 16, 32, 64, 128, 2048, 8192):
        cfg, _ = get_gemm_config("GEMM-A8W8", m, 4096, 4096)
        for key in GFX1100_SCHEMA_KEYS:
            assert key in cfg, f"gfx1100 M={m}: missing schema key {key!r}"
        assert cfg["NUM_KSPLIT"] == 1
        kernel_cfg = compute_splitk_params(cfg, 4096)
        for key in KERNEL_CONFIG_KEYS:
            assert key in kernel_cfg, f"gfx1100 M={m}: missing kernel key {key!r}"
        assert kernel_cfg["SPLITK_BLOCK_SIZE"] == 4096


def test_gfx1100_kernel_config_path(set_arch):
    set_arch("gfx1100")
    decode_cfg, decode_tuned = _get_config(32, 4096, 4096)
    assert decode_tuned is False
    for key in KERNEL_CONFIG_KEYS:
        assert key in decode_cfg
    assert decode_cfg["BLOCK_SIZE_M"] == GFX1100_DECODE["BLOCK_SIZE_M"]

    prefill_cfg, _ = _get_config(2048, 4096, 4096)
    assert prefill_cfg["BLOCK_SIZE_M"] == GFX1100_PREFILL["BLOCK_SIZE_M"]
    assert prefill_cfg["BLOCK_SIZE_N"] == GFX1100_PREFILL["BLOCK_SIZE_N"]
    assert prefill_cfg["num_warps"] == GFX1100_PREFILL["num_warps"]


def test_existing_arch_configs_preserved(set_arch):
    assert "gfx1100" in GEMM_A8W8_ARCHS
    for arch in GEMM_A8W8_ARCHS:
        set_arch(arch)
        for m in (32, 2048):
            _get_gemm_config_cached.cache_clear()
            cfg, _ = get_gemm_config("GEMM-A8W8", m, 4096, 4096)
            assert cfg, f"{arch} M={m}: resolved config is empty"
            kernel_cfg = compute_splitk_params(cfg, 4096)
            for key in KERNEL_CONFIG_KEYS:
                assert key in kernel_cfg, f"{arch} M={m}: missing kernel key {key!r}"
