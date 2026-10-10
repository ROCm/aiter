# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Dedicated test suite for GMM, Quantization, Normalization, Fusions, and Cache Isolation.

Validates:
1. GMM: Schema, loader resolution across architectures, and gfx950 dispatch threshold switching.
2. Quantization: Composite dynamic table axes and get_quant_config resolution.
3. Normalization: get_normalization_config across architectures.
4. Fusions: Specialized N-dimension file resolution.
5. Cache isolation: Zero cross-architecture cache contamination.
6. Fail-fast error handling: Missing files, invalid backends, and unsafe paths.
"""

from __future__ import annotations

import json
import os
import re
import pytest

from aiter.ops.triton.utils import (
    config_utils,
    mhc_config_utils,
    normalization_config_utils,
    quant_config_utils,
)
import aiter.ops.triton.gmm as gmm
from op_tests.triton_tests.config_tests.common import (
    CONFIGS_PATH,
    check_no_mixed_geq_leq,
    clear_all_config_caches,
    discover_architectures,
    filter_configs_by_op,
    is_power_of_two,
    load_json_dict,
    rel_path,
    set_test_arch,
    validate_config_file_layout_and_naming,
    validate_gemm_config_table,
)

GMM_FILES = filter_configs_by_op("gmm")
QUANT_FILES = filter_configs_by_op("quant")
NORMALIZATION_FILES = filter_configs_by_op("normalization")
FUSIONS_FILES = filter_configs_by_op("fusions")
DISCOVERED_ARCHS = discover_architectures()


# --- GMM ---

@pytest.mark.parametrize("path", GMM_FILES, ids=rel_path)
def test_gmm_config_schema(path: str):
    """Verify GMM table structure across architectures."""
    with open(path, "r", encoding="utf-8") as f:
        table = json.load(f)

    for variant in ("gmm", "ptgmm", "nptgmm"):
        assert variant in table, f"{rel_path(path)}: missing GMM variant '{variant}'"
        v_dict = table[variant]
        assert "default" in v_dict, f"{rel_path(path)}: '{variant}' missing 'default' config"

        if "dispatch" in v_dict:
            for rule in v_dict["dispatch"]:
                assert "config" in rule, f"{rel_path(path)}: rule missing 'config'"
                assert rule["config"] in v_dict, (
                    f"{rel_path(path)}: rule config '{rule['config']}' not in variant"
                )
    check_no_mixed_geq_leq(table, path)


@pytest.mark.parametrize("arch", ["gfx942", "gfx950", "gfx1250"])
def test_gmm_loader_resolves(arch: str):
    """Test GMM get_config loader across architectures."""
    with set_test_arch(arch):
        for variant in ("gmm", "ptgmm", "nptgmm"):
            cfg = gmm.get_config(variant, M=1024, K=2048, N=2048, G=4)
            assert isinstance(cfg, dict)
            assert "BLOCK_SIZE_M" in cfg or "num_warps" in cfg


def test_gmm_dispatch_thresholds_gfx950():
    """Verify GMM dispatch rules switch correctly on gfx950 (addressing #6105)."""
    with set_test_arch("gfx950"):
        cfg_large = gmm.get_config("gmm", M=8192, K=4096, N=4096, G=32, accumulate=False)
        cfg_default = gmm.get_config("gmm", M=8191, K=4096, N=4096, G=32, accumulate=False)
        assert cfg_large != cfg_default, "GMM dispatch rule failed to select large_kn config"


# --- Quantization ---

@pytest.mark.parametrize("path", QUANT_FILES, ids=rel_path)
def test_quant_config_schema(path: str):
    """Verify Quant table structure has uniform composite axes."""
    with open(path, "r", encoding="utf-8") as f:
        table = json.load(f)

    axes = quant_config_utils.table_axes(table)
    assert len(axes) > 0, f"{rel_path(path)}: no axes found"
    assert "any" in table, f"{rel_path(path)}: table must contain 'any' fallback"
    check_no_mixed_geq_leq(table, path)


@pytest.mark.parametrize("arch", ["gfx950", "gfx1250"])
def test_quant_loader_resolves(arch: str):
    """Test get_quant_config for Gluon MXFP4 and MXFP8 using dynamic table axes."""
    with set_test_arch(arch):
        for name in ("MXFP4", "MXFP8"):
            cfg_dir = config_utils.resolve_config_dir("quant", name, backend="gluon")
            table = config_utils.load_config_json(f"{cfg_dir}/DEFAULT.json")
            axes = quant_config_utils.table_axes(table)
            vals = {a: 128 for a in axes}
            cfg = quant_config_utils.get_quant_config(name, **vals)
            assert isinstance(cfg, dict)
            assert "BLOCK_SIZE_M" in cfg


# --- Normalization ---

@pytest.mark.parametrize("path", NORMALIZATION_FILES, ids=rel_path)
def test_normalization_config_schema(path: str):
    """Verify Normalization config structure."""
    with open(path, "r", encoding="utf-8") as f:
        table = json.load(f)

    assert "num_warps" in table or "BLOCK_SIZE" in table or any(isinstance(v, dict) for v in table.values()), (
        f"{rel_path(path)}: invalid normalization table"
    )
    check_no_mixed_geq_leq(table, path)


@pytest.mark.parametrize("arch", ["gfx942", "gfx950"])
def test_normalization_loader_resolves(arch: str):
    """Test get_normalization_config across architectures that publish it."""
    with set_test_arch(arch):
        cfg = normalization_config_utils.get_normalization_config("RMSNORM_LARGE_M_SMALL_N", arch=arch)
        assert isinstance(cfg, dict)
        assert len(cfg) > 0


# --- Fusions ---

@pytest.mark.parametrize("path", FUSIONS_FILES, ids=rel_path)
def test_fusions_config_schema(path: str):
    """Verify Fusions config tables and power-of-2 tile bounds."""
    with open(path, "r", encoding="utf-8") as f:
        table = json.load(f)
    assert len(table) > 0, f"{rel_path(path)}: empty table"
    check_no_mixed_geq_leq(table, path)

    for k, v in table.items():
        if isinstance(v, dict):
            for bkey in ("BLOCK_SIZE_M", "BLOCK_SIZE_N", "BLOCK_SIZE", "BLOCK_M", "BLOCK_N"):
                if bkey in v and v[bkey] is not None:
                    val = v[bkey]
                    assert is_power_of_two(val), (
                        f"{rel_path(path)} [{k}]: {bkey}={val} must be a positive power of 2"
                    )
            if "num_warps" in v and v["num_warps"] is not None:
                warps = v["num_warps"]
                assert is_power_of_two(warps), (
                    f"{rel_path(path)} [{k}]: num_warps={warps} must be a positive power of 2"
                )


def _get_fusions_specialized_files():
    files = []
    for arch in DISCOVERED_ARCHS:
        for backend in config_utils._VALID_BACKENDS:
            fdir = os.path.join(config_utils.AITER_TRITON_CONFIGS_PATH, arch, backend, "fusions")
            if not os.path.isdir(fdir):
                continue
            for root, _, fnames in os.walk(fdir):
                for fname in fnames:
                    if fname != "DEFAULT.json" and fname.endswith(".json"):
                        m = re.search(r"-N=(\d+)\.json$", fname)
                        if m:
                            files.append((arch, backend, int(m.group(1)), os.path.join(root, fname)))
    return sorted(files)


FUSIONS_SPECIALIZED_FILES = _get_fusions_specialized_files()


@pytest.mark.parametrize(
    "arch,backend,n_val,fpath",
    FUSIONS_SPECIALIZED_FILES,
    ids=[f"{x[0]}-{x[1]}-N={x[2]}" for x in FUSIONS_SPECIALIZED_FILES],
)
def test_fusions_specialized_resolution(arch: str, backend: str, n_val: int, fpath: str):
    """Test standard resolver path (resolve_config_dir + load_config_json) for specialized Fusion configs."""
    with set_test_arch(arch):
        cfg_dir = config_utils.resolve_config_dir("fusions", "FUSED_CLAMP_ACT_MUL", backend=backend)
        spec_file = f"{cfg_dir}/FUSED_CLAMP_ACT_MUL-N={n_val}.json"
        loaded = config_utils.load_config_json(spec_file, required=True)
        assert isinstance(loaded, dict)
        assert len(loaded) > 0


# --- Cache Isolation & Fail-Fast ---

def test_cache_isolation_across_architectures():
    """Verify that looking up configs on different arches does not suffer cache pollution.

    Verifies that the underlying cached loader (load_config_json) preserves distinct
    configs across architectures without clearing the cache between lookups, and that
    cached results match their architecture-specific expected configurations.
    """
    clear_all_config_caches()
    try:
        fpath_942 = os.path.join(CONFIGS_PATH, "gfx942/triton/gemm/gemm_a16w16/DEFAULT.json")
        fpath_950 = os.path.join(CONFIGS_PATH, "gfx950/triton/gemm/gemm_a16w16/DEFAULT.json")

        cfg_942 = config_utils.load_config_json(fpath_942)
        assert config_utils.load_config_json.cache_info().currsize >= 1

        # DO NOT clear cache between lookups: query gfx950 while gfx942 is cached
        cfg_950 = config_utils.load_config_json(fpath_950)
        assert config_utils.load_config_json.cache_info().currsize >= 2

        # Assert architecture-specific expected parameters differ and match expected values
        assert cfg_942["any"]["num_stages"] == 2
        assert cfg_950["any"]["num_stages"] == 3
        assert cfg_942["any"]["BLOCK_SIZE_N"] == 256
        assert cfg_950["any"]["BLOCK_SIZE_N"] == 128
        assert cfg_942 != cfg_950

        # Query gfx942 again from warm cache: must still match gfx942, not gfx950
        cfg_942_cached = config_utils.load_config_json(fpath_942)
        assert cfg_942_cached == cfg_942

        # Verify multi-arch cached function _c_thresholds preserves distinct arch results
        t_942 = mhc_config_utils._c_thresholds("gfx942", "MHC_FUSED_SINKHORN")
        t_950 = mhc_config_utils._c_thresholds("gfx950", "MHC_FUSED_SINKHORN")
        assert mhc_config_utils._c_thresholds.cache_info().currsize >= 2
        assert 7168 not in t_942
        assert 7168 in t_950
        assert t_942 != t_950

        # Query gfx942 again from warm cache: must still match original gfx942 thresholds
        t_942_cached = mhc_config_utils._c_thresholds("gfx942", "MHC_FUSED_SINKHORN")
        assert t_942_cached == t_942
    finally:
        clear_all_config_caches()


def test_missing_required_config_file_raises():
    """Verify load_config_json raises FileNotFoundError when required file is missing."""
    with pytest.raises(FileNotFoundError):
        config_utils.load_config_json("/nonexistent/path/DEFAULT.json", required=True)


def test_invalid_backend_rejected():
    """Verify resolve_config_dir rejects backends outside ('triton', 'gluon')."""
    with pytest.raises(AssertionError, match="unknown backend"):
        config_utils.resolve_config_dir("gemm", "GEMM-A16W16", backend="invalid")


def test_unsafe_arch_override_rejected():
    """Verify resolve_config_dir rejects path-unsafe arch overrides."""
    with pytest.raises(AssertionError, match="arch override must match"):
        config_utils.resolve_config_dir("gemm", "GEMM-A16W16", arch="../../bad_path")


def test_uncovered_shape_without_fallback_raises():
    """Verify select_leq_config raises KeyError when value is out of bounds and no 'any' key."""
    table = {"M_LEQ_32": {"v": 1}}
    with pytest.raises(KeyError):
        config_utils.select_leq_config(table, axes=("M",), M=100)


# --- Dedicated Negative Diagnostics Tests ---

def test_negative_power_of_two_validator():
    """Verify is_power_of_two reliably accepts only positive powers of two."""
    for valid in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024):
        assert is_power_of_two(valid) is True

    for invalid in (0, -1, -16, 3, 5, 7, 37, 63, 100, "16", None, 3.14):
        assert is_power_of_two(invalid) is False


def test_negative_empty_and_non_dict_json(tmp_path):
    """Verify load_json_dict rejects empty JSON dictionaries and non-dict JSON roots."""
    empty_file = tmp_path / "empty.json"
    empty_file.write_text("{}", encoding="utf-8")
    with pytest.raises(AssertionError, match="configuration dictionary is empty"):
        load_json_dict(str(empty_file))

    list_file = tmp_path / "list.json"
    list_file.write_text("[1, 2, 3]", encoding="utf-8")
    with pytest.raises(AssertionError, match="root JSON object must be a dict"):
        load_json_dict(str(list_file))


def test_negative_malformed_json_syntax(tmp_path):
    """Verify json.loads fails on syntax-corrupted JSON."""
    bad_file = tmp_path / "syntax_error.json"
    bad_file.write_text('{"BLOCK_SIZE_M": 32, }', encoding="utf-8")  # trailing comma
    with pytest.raises(json.JSONDecodeError):
        with open(str(bad_file), "r", encoding="utf-8") as f:
            json.load(f)


def test_negative_legacy_naming_tokens_rejected():
    """Verify check_no_mixed_geq_leq detects and rejects non-canonical legacy _LE_/_GE_ tokens."""
    bad_table_le = {"M_LE_16": {"BLOCK_SIZE_M": 32}}
    with pytest.raises(AssertionError, match="Non-canonical legacy LE/GE key format found"):
        check_no_mixed_geq_leq(bad_table_le, "dummy/path.json")

    bad_table_ge = {"M_GE_32": {"BLOCK_SIZE_M": 32}}
    with pytest.raises(AssertionError, match="Non-canonical legacy LE/GE key format found"):
        check_no_mixed_geq_leq(bad_table_ge, "dummy/path.json")


def test_negative_overlapping_bounds_rejected():
    """Verify check_no_mixed_geq_leq rejects overlapping 1D LEQ and GEQ threshold bounds."""
    overlapping_table = {
        "M_LEQ_64": {"BLOCK_SIZE_M": 32},
        "M_GEQ_32": {"BLOCK_SIZE_M": 64},
    }
    with pytest.raises(AssertionError, match="Overlapping M_LEQ .* and M_GEQ .* ranges"):
        check_no_mixed_geq_leq(overlapping_table, "dummy/path.json")


def test_negative_filename_and_layout_rejected():
    """Verify validate_config_file_layout_and_naming rejects malformed paths and names."""
    # 1. Reject architecture-prefixed filenames in nested layout
    for arch in ("gfx942", "gfx950", "gfx1250"):
        bad_name = os.path.join(CONFIGS_PATH, f"{arch}/triton/gemm/gemm_a16w16/{arch}_DEFAULT.json")
        with pytest.raises(AssertionError, match="filename must not have arch prefix"):
            validate_config_file_layout_and_naming(bad_name)

    # 2. Reject paths that are too shallow (depth 4)
    bad_shallow = os.path.join(CONFIGS_PATH, "gfx942/triton/gemm/DEFAULT.json")
    with pytest.raises(AssertionError, match="Path must have exactly 5 components"):
        validate_config_file_layout_and_naming(bad_shallow)

    # 3. Reject paths that are too deep (depth 6)
    bad_deep = os.path.join(CONFIGS_PATH, "gfx942/triton/gemm/gemm_a16w16/extra/DEFAULT.json")
    with pytest.raises(AssertionError, match="Path must have exactly 5 components"):
        validate_config_file_layout_and_naming(bad_deep)

    # 4. Reject invalid backends
    bad_backend = os.path.join(CONFIGS_PATH, "gfx942/invalid_backend/gemm/gemm_a16w16/DEFAULT.json")
    with pytest.raises(AssertionError, match="invalid backend"):
        validate_config_file_layout_and_naming(bad_backend)


def test_negative_gemm_missing_any_fallback_rejected():
    """Verify GEMM table schema rejects tables without the required 'any' fallback."""
    table_without_any = {
        "M_LEQ_32": {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 32, "num_warps": 4},
        "M_LEQ_64": {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "num_warps": 4},
    }
    with pytest.raises(AssertionError, match="GEMM table must contain an 'any' fallback"):
        validate_gemm_config_table(table_without_any, path="dummy.json", backend="triton")


def test_negative_specialized_filename_rejected():
    """Verify validate_config_file_layout_and_naming rejects malformed specialized filenames."""
    # 1. Reject specialized files missing hyphens / suffixes entirely
    bad_no_suffix = os.path.join(CONFIGS_PATH, "gfx942/triton/gemm/gemm_a16w16/specialized.json")
    with pytest.raises(AssertionError, match="must follow <CONFIG_NAME>-<suffix>.json format"):
        validate_config_file_layout_and_naming(bad_no_suffix)

    bad_no_hyphen = os.path.join(CONFIGS_PATH, "gfx942/triton/gemm/gemm_a16w16/GEMM_A16W16.json")
    with pytest.raises(AssertionError, match="must follow <CONFIG_NAME>-<suffix>.json format"):
        validate_config_file_layout_and_naming(bad_no_hyphen)

    # 2. Reject empty suffix (trailing hyphen)
    bad_empty_suffix = os.path.join(CONFIGS_PATH, "gfx942/triton/gemm/gemm_a16w16/GEMM-A16W16-.json")
    with pytest.raises(AssertionError, match="must follow <CONFIG_NAME>-<suffix>.json format"):
        validate_config_file_layout_and_naming(bad_empty_suffix)

    # 3. Reject missing suffix when hyphen is only within CONFIG_NAME
    bad_stem_only = os.path.join(CONFIGS_PATH, "gfx942/triton/gemm/gemm_a16w16/GEMM-A16W16.json")
    with pytest.raises(AssertionError, match="must have exactly one valid decomposition"):
        validate_config_file_layout_and_naming(bad_stem_only)

    # 4. Reject mismatched CONFIG_NAME stem vs parent dtype directory
    bad_mismatch = os.path.join(CONFIGS_PATH, "gfx942/triton/gemm/gemm_a16w16/OTHER_OP-N=1024-K=1024.json")
    with pytest.raises(AssertionError, match="must have exactly one valid decomposition"):
        validate_config_file_layout_and_naming(bad_mismatch)

    # 5. Reject non-JSON extension
    bad_ext = os.path.join(CONFIGS_PATH, "gfx942/triton/gemm/gemm_a16w16/GEMM-A16W16-N=1024.txt")
    with pytest.raises(AssertionError, match="must end with.*json"):
        validate_config_file_layout_and_naming(bad_ext)


def test_negative_unanchored_specialized_decomposition_ambiguous():
    """Verify that specialized filenames with hyphens in CONFIG_NAME are ambiguous
    in isolation without parent dtype anchoring, proving why directory-anchored
    decomposition (validate_config_file_layout_and_naming) is required.
    """
    fname = "GEMM-A16W16-N=1024-K=4096.json"
    stem = fname[:-5]
    parts = stem.split("-")
    # Without anchoring to dtype, multiple valid (config_name, suffix) splits exist:
    unanchored_splits = [
        ("-".join(parts[:i]), "-".join(parts[i:]))
        for i in range(1, len(parts))
        if config_utils._CONFIG_NAME_RE.fullmatch("-".join(parts[:i]))
        and len("-".join(parts[i:])) > 0
    ]
    # In isolation: multiple decompositions exist (e.g. ('GEMM', ...) and ('GEMM-A16W16', ...))
    assert len(unanchored_splits) > 1, f"Expected ambiguous splits in isolation, got {unanchored_splits}"

    # Anchored to parent dtype 'gemm_a16w16': exactly one unique decomposition exists
    anchored_splits = [
        s for s in unanchored_splits
        if config_utils._dtype_dir(s[0]) == "gemm_a16w16"
    ]
    assert len(anchored_splits) == 1, f"Expected unique anchored split, got {anchored_splits}"
    assert anchored_splits[0] == ("GEMM-A16W16", "N=1024-K=4096")
