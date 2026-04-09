# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""
Unit tests for aiter/utility/pretune.py.

No GPU or torch required — tests exercise file-system and config resolution
logic only.  Run with:
    python op_tests/test_pretune.py
"""

import json
import os
import sys
import tempfile

import pandas as pd

REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CSRC_DIR = os.path.join(REPO_DIR, "csrc")
CFG_PATH = os.path.join(REPO_DIR, "aiter", "jit", "optCompilerConfig.json")

sys.path.insert(0, os.path.join(REPO_DIR, "aiter"))

from utility.pretune import (  # noqa: E402
    _all_tune_modules,
    _make_untune_csv,
    _resolve,
    _SCRIPT_FALLBACK,
)

with open(CFG_PATH) as f:
    CFG = json.load(f)

# ── expected resolution table ────────────────────────────────────────────────
# (tune_module, expected_script_suffix_from_repo_root, expected_config_attr, expect_skip)
EXPECTED = [
    (
        "module_batched_gemm_a8w8_tune",
        "csrc/ck_batched_gemm_a8w8/batched_gemm_a8w8_tune.py",
        "AITER_CONFIG_A8W8_BATCHED_GEMM_FILE",
        False,
    ),
    (
        "module_batched_gemm_bf16_tune",
        "csrc/ck_batched_gemm_bf16/batched_gemm_bf16_tune.py",
        "AITER_CONFIG_BF16_BATCHED_GEMM_FILE",
        False,
    ),
    (
        "module_gemm_a4w4_blockscale_tune",
        "csrc/ck_gemm_a4w4_blockscale/gemm_a4w4_blockscale_tune.py",
        "AITER_CONFIG_GEMM_A4W4_FILE",
        False,
    ),
    (
        "module_gemm_a8w8_tune",
        "csrc/ck_gemm_a8w8/gemm_a8w8_tune.py",
        "AITER_CONFIG_GEMM_A8W8_FILE",
        False,
    ),
    (
        "module_gemm_a8w8_blockscale_tune",
        "csrc/ck_gemm_a8w8_blockscale/gemm_a8w8_blockscale_tune.py",
        "AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_FILE",
        False,
    ),
    (
        # cktile variant: falls back to blockscale parent script
        "module_gemm_a8w8_blockscale_cktile_tune",
        "csrc/ck_gemm_a8w8_blockscale/gemm_a8w8_blockscale_tune.py",
        "AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_FILE",
        False,
    ),
    (
        "module_gemm_a8w8_bpreshuffle_tune",
        "csrc/ck_gemm_a8w8_bpreshuffle/gemm_a8w8_bpreshuffle_tune.py",
        "AITER_CONFIG_GEMM_A8W8_BPRESHUFFLE_FILE",
        False,
    ),
    (
        # cktile variant: falls back to bpreshuffle parent script
        "module_gemm_a8w8_bpreshuffle_cktile_tune",
        "csrc/ck_gemm_a8w8_bpreshuffle/gemm_a8w8_bpreshuffle_tune.py",
        "AITER_CONFIG_GEMM_A8W8_BPRESHUFFLE_FILE",
        False,
    ),
    (
        # no tune script writes to the bpreshuffle blockscale CSV
        "module_gemm_a8w8_blockscale_bpreshuffle_tune",
        None,
        "AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_BPRESHUFFLE_FILE",
        True,
    ),
    (
        "module_gemm_a8w8_blockscale_bpreshuffle_cktile_tune",
        None,
        "AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_BPRESHUFFLE_FILE",
        True,
    ),
]

# AITER_CONFIGS tune CSVs with expected shape key columns after _make_untune_csv.
# FMOE and BF16 GEMM are excluded — they have no tune module.
TUNE_CONFIGS = [
    ("a8w8_blockscale_tuned_gemm.csv", ["M", "N", "K"]),
    ("a8w8_tuned_gemm.csv", ["M", "N", "K"]),
    ("a4w4_blockscale_tuned_gemm.csv", ["M", "N", "K"]),
    ("a8w8_bpreshuffle_tuned_gemm.csv", ["M", "N", "K"]),
    ("a8w8_blockscale_bpreshuffle_tuned_gemm.csv", ["M", "N", "K"]),
    ("a8w8_tuned_batched_gemm.csv", ["B", "M", "N", "K"]),
    ("bf16_tuned_batched_gemm.csv", ["B", "M", "N", "K"]),
]

CONFIGS_DIR = os.path.join(REPO_DIR, "aiter", "configs")


# ── test helpers ──────────────────────────────────────────────────────────────

_passed = 0
_failed = 0


def check(name: str, condition: bool, msg: str = ""):
    global _passed, _failed
    if condition:
        print(f"  [PASS] {name}")
        _passed += 1
    else:
        print(f"  [FAIL] {name}{': ' + msg if msg else ''}")
        _failed += 1


# ── tests ─────────────────────────────────────────────────────────────────────


def test_resolve():
    """_resolve returns correct script path and config attr for every tune module."""
    print("\n=== test_resolve ===")
    for module, script_suffix, config_attr, expect_skip in EXPECTED:
        script, attr = _resolve(module, CFG, CSRC_DIR)

        if expect_skip:
            check(
                f"{module} script=None", script is None, f"expected None, got {script}"
            )
        else:
            expected_abs = os.path.join(REPO_DIR, script_suffix)
            check(
                f"{module} script path",
                script == expected_abs,
                f"\n    got:      {script}\n    expected: {expected_abs}",
            )
            check(
                f"{module} script exists",
                script is not None and os.path.exists(script),
                f"not on disk: {script}",
            )

        check(
            f"{module} config_attr",
            attr == config_attr,
            f"got {attr!r}, expected {config_attr!r}",
        )


def test_all_tune_modules_covered():
    """Every _tune module in optCompilerConfig.json appears in EXPECTED."""
    print("\n=== test_all_tune_modules_covered ===")
    cfg_modules = set(_all_tune_modules(CFG))
    expected_modules = {e[0] for e in EXPECTED}
    missing_from_expected = cfg_modules - expected_modules
    extra_in_expected = expected_modules - cfg_modules
    check(
        "no modules missing from EXPECTED",
        not missing_from_expected,
        f"{missing_from_expected}",
    )
    check("no extra modules in EXPECTED", not extra_in_expected, f"{extra_in_expected}")


def test_script_fallback_keys_in_config():
    """Every key in _SCRIPT_FALLBACK is a known tune module in optCompilerConfig.json."""
    print("\n=== test_script_fallback_keys_in_config ===")
    cfg_modules = set(_all_tune_modules(CFG))
    unknown = set(_SCRIPT_FALLBACK) - cfg_modules
    check(
        "all _SCRIPT_FALLBACK keys are valid tune modules",
        not unknown,
        f"unknown: {unknown}",
    )


def test_make_untune_csv():
    """_make_untune_csv retains shape keys, drops metadata columns, deduplicates."""
    print("\n=== test_make_untune_csv ===")
    for csv_name, expected_shape_keys in TUNE_CONFIGS:
        csv_path = os.path.join(CONFIGS_DIR, csv_name)
        if not os.path.exists(csv_path):
            print(f"  [SKIP] {csv_name}: file not present")
            continue

        orig = pd.read_csv(csv_path)
        present_keys = [k for k in expected_shape_keys if k in orig.columns]
        expected_unique = orig[present_keys].drop_duplicates().shape[0]

        tmp = _make_untune_csv(csv_path, ["B", "M", "N", "K"])
        try:
            df = pd.read_csv(tmp)
        finally:
            os.unlink(tmp)

        check(
            f"{csv_name} columns={expected_shape_keys}",
            list(df.columns) == expected_shape_keys,
            f"got {df.columns.tolist()}",
        )
        check(
            f"{csv_name} no metadata columns leaked",
            not any(
                c in df.columns for c in ["gfx", "cu_num", "kernelId", "us", "libtype"]
            ),
        )
        check(f"{csv_name} non-empty", len(df) > 0)
        check(
            f"{csv_name} deduplicated ({expected_unique} unique shapes)",
            len(df) == expected_unique,
            f"got {len(df)}",
        )


def test_make_untune_csv_multi_path():
    """_make_untune_csv handles colon-separated multi-path tune_file correctly."""
    print("\n=== test_make_untune_csv_multi_path ===")
    rows_a = pd.DataFrame({"M": [128, 256], "N": [512, 512], "K": [1024, 1024]})
    rows_b = pd.DataFrame(
        {"M": [256, 512], "N": [512, 512], "K": [1024, 1024]}
    )  # 256 overlaps

    with (
        tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as fa,
        tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as fb,
    ):
        rows_a.to_csv(fa.name, index=False)
        rows_b.to_csv(fb.name, index=False)
        tune_file = f"{fa.name}{os.pathsep}{fb.name}"

    try:
        tmp = _make_untune_csv(tune_file, ["B", "M", "N", "K"])
        df = pd.read_csv(tmp)
        os.unlink(tmp)
        # 3 unique (M,N,K): (128,512,1024), (256,512,1024), (512,512,1024)
        check("multi-path: 3 unique shapes after dedup", len(df) == 3, f"got {len(df)}")
        check("multi-path: columns=[M,N,K]", list(df.columns) == ["M", "N", "K"])
    finally:
        os.unlink(fa.name)
        os.unlink(fb.name)


def test_make_untune_csv_missing_raises():
    """_make_untune_csv raises FileNotFoundError when no CSV path exists."""
    print("\n=== test_make_untune_csv_missing_raises ===")
    raised = False
    try:
        _make_untune_csv("/nonexistent/path.csv", ["M", "N", "K"])
    except FileNotFoundError:
        raised = True
    check("raises FileNotFoundError for missing path", raised)


def test_parse_pretune_modules():
    """PRETUNE_MODULES env values parse to the correct module lists."""
    print("\n=== test_parse_pretune_modules ===")
    cases = [
        (
            "module_gemm_a8w8_blockscale_tune",
            ["module_gemm_a8w8_blockscale_tune"],
        ),
        (
            "module_gemm_a8w8_tune,module_gemm_a8w8_blockscale_tune",
            ["module_gemm_a8w8_tune", "module_gemm_a8w8_blockscale_tune"],
        ),
        (
            " module_gemm_a8w8_tune , module_gemm_a8w8_blockscale_tune ",
            ["module_gemm_a8w8_tune", "module_gemm_a8w8_blockscale_tune"],
        ),
    ]
    for env_value, expected in cases:
        modules = [m.strip() for m in env_value.split(",") if m.strip()]
        check(f"parse {env_value.strip()!r}", modules == expected, f"got {modules}")

    # "all" expands to every _tune module
    all_modules = _all_tune_modules(CFG)
    check("all: count matches EXPECTED", len(all_modules) == len(EXPECTED))
    check("all: set matches EXPECTED", set(all_modules) == {e[0] for e in EXPECTED})


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_resolve()
    test_all_tune_modules_covered()
    test_script_fallback_keys_in_config()
    test_make_untune_csv()
    test_make_untune_csv_multi_path()
    test_make_untune_csv_missing_raises()
    test_parse_pretune_modules()

    print(f"\n{'='*50}")
    print(f"Results: {_passed} passed, {_failed} failed")
    if _failed:
        sys.exit(1)
