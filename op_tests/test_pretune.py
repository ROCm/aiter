# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""
Unit tests for aiter/utility/pretune.py.

No GPU or torch required — tests exercise file-system and config resolution
logic only.  Run with:
    python op_tests/test_pretune.py
"""

import contextlib
import io
import json
import logging
import os
import subprocess
import sys
import tempfile
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

import pandas as pd

REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CSRC_DIR = os.path.join(REPO_DIR, "csrc")
CFG_PATH = os.path.join(REPO_DIR, "aiter", "jit", "optCompilerConfig.json")

sys.path.insert(0, os.path.join(REPO_DIR, "aiter"))

# Import through the lightweight utility package to avoid GPU/torch initialization.
from utility import pretune
from utility.pretune import (
    _SCRIPT_FALLBACK,
    _SHAPE_KEYS,
    _all_tune_modules,
    _inference_modules,
    _make_untune_csv,
    _parse_module_list,
    _resolve,
    run_pretune,
    run_pretune_modules,
)

with open(CFG_PATH) as f:
    CFG = json.load(f)

logging.getLogger("aiter").addHandler(logging.StreamHandler(sys.stdout))

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

# AITER_CONFIGS tune CSVs paired with the untuned catalogue whose columns the
# family's tuner indexes; _make_untune_csv must produce exactly those columns.
# FMOE and BF16 GEMM are excluded — they have no tune module.
TUNE_CONFIGS = [
    ("a8w8_blockscale_tuned_gemm.csv", "a8w8_blockscale_untuned_gemm.csv"),
    ("a8w8_tuned_gemm.csv", "a8w8_untuned_gemm.csv"),
    ("a4w4_blockscale_tuned_gemm.csv", "a4w4_blockscale_untuned_gemm.csv"),
    ("a8w8_bpreshuffle_tuned_gemm.csv", "a8w8_bpreshuffle_untuned_gemm.csv"),
    (
        "a8w8_blockscale_bpreshuffle_tuned_gemm.csv",
        "a8w8_blockscale_bpreshuffle_untuned_gemm.csv",
    ),
    ("a8w8_tuned_batched_gemm.csv", "a8w8_untuned_batched_gemm.csv"),
    ("bf16_tuned_batched_gemm.csv", "bf16_untuned_batched_gemm.csv"),
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
    for csv_name, untuned_name in TUNE_CONFIGS:
        csv_path = os.path.join(CONFIGS_DIR, csv_name)
        if not os.path.exists(csv_path):
            print(f"  [SKIP] {csv_name}: file not present")
            continue
        expected_shape_keys = list(
            pd.read_csv(os.path.join(CONFIGS_DIR, untuned_name), nrows=0).columns
        )

        orig = pd.read_csv(csv_path)
        present_keys = [k for k in expected_shape_keys if k in orig.columns]
        expected_unique = orig[present_keys].drop_duplicates().shape[0]

        tmp = _make_untune_csv(csv_path, _SHAPE_KEYS)
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


def test_write_tune_file_resolution():
    """In standalone mode, write_tune_file must resolve to the source CSV (in aiter/configs/),
    not the ephemeral /tmp merged path.

    The logic in run_pretune() strips the _FILE suffix from config_attr to derive a
    module-level variable name in core (e.g. AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_FILE →
    AITER_CONFIG_GEMM_A8W8_BLOCKSCALE), then calls getattr(core, source_attr).
    This test verifies that all supported modules have a matching module-level variable
    in core.py whose value points inside aiter/configs/ (not /tmp/).
    """
    print("\n=== test_write_tune_file_resolution ===")
    core_py = os.path.join(REPO_DIR, "aiter", "jit", "core.py")
    with open(core_py, encoding="utf-8") as f:
        core_src = f.read()

    for module, _, config_attr, expect_skip in EXPECTED:
        if expect_skip or config_attr is None:
            continue
        source_attr = config_attr.removesuffix("_FILE")
        if source_attr == config_attr:
            check(
                f"{module} config_attr has _FILE suffix",
                False,
                f"config_attr {config_attr!r} has no _FILE suffix — cannot derive source_attr",
            )
            continue
        # Verify the module-level variable exists in core.py
        check(
            f"{module} source_attr '{source_attr}' defined in core.py",
            f"{source_attr} " in core_src or f"{source_attr}=" in core_src,
            f"not found in core.py — getattr(core, {source_attr!r}) would return None",
        )
        # Verify it maps to aiter/configs/ (not /tmp/)
        import re

        m = re.search(
            rf'{source_attr}\s*=\s*os\.getenv\(\s*["\'][\w]+["\'],\s*([^\)]+)\)',
            core_src,
        )
        if m:
            default_val = m.group(1).strip().strip('"').strip("'")
            check(
                f"{module} default write_tune_file is in aiter/configs/",
                "aiter/configs" in default_val or "aiter_meta/configs" in default_val,
                f"default points to: {default_val!r}",
            )


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
        modules = _parse_module_list(env_value, CFG)
        check(f"parse {env_value.strip()!r}", modules == expected, f"got {modules}")

    # "all" expands to every _tune module, excluding _unsupported entries
    all_modules = _all_tune_modules(CFG)
    check("all: count matches EXPECTED", len(all_modules) == len(EXPECTED))
    check("all: set matches EXPECTED", set(all_modules) == {e[0] for e in EXPECTED})
    _unsupported = {m for m, v in _SCRIPT_FALLBACK.items() if v is None}
    all_parsed = _parse_module_list("all", CFG)
    check(
        "all: _unsupported modules excluded",
        not any(m in _unsupported for m in all_parsed),
        f"unsupported in result: {[m for m in all_parsed if m in _unsupported]}",
    )


def _test_pretune_module_names(use_callback):
    """Both build paths must supply usable Python extension module names."""
    config_attr = "AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_FILE"
    inf_modules = _inference_modules(config_attr, CFG)
    for module_name in (
        "module_gemm_a8w8_blockscale_tune",
        "module_gemm_a8w8_blockscale_cktile_tune",
    ):
        for name_args in (
            {"md_name": ""},
            {},
            {"md_name": "custom_inference_module"},
        ):
            with tempfile.TemporaryDirectory() as tmp:
                tune_file = os.path.join(tmp, "tuned.csv")
                pd.DataFrame({"M": [1], "N": [128], "K": [128]}).to_csv(
                    tune_file, index=False
                )
                build_args = {
                    "srcs": ["inference.cu"],
                    "flags_extra_cc": [],
                    "flags_extra_hip": [],
                    "blob_gen_cmd": "",
                    "extra_include": [],
                    "third_party": [],
                    **name_args,
                }
                original_build_args = {**build_args}
                core = SimpleNamespace(
                    AITER_CONFIGS=SimpleNamespace(
                        AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_FILE=tune_file
                    ),
                    AITER_CONFIG_GEMM_A8W8_BLOCKSCALE=tune_file,
                    get_args_of_build=Mock(return_value=build_args),
                    rm_module=Mock(),
                    clear_build=Mock(),
                    build_module=Mock(),
                )
                build_one_module = Mock() if use_callback else None
                with patch(
                    "utility.pretune.subprocess.run",
                    return_value=SimpleNamespace(returncode=0),
                ):
                    run_pretune(
                        module_name, CFG, core, CSRC_DIR, REPO_DIR, build_one_module
                    )
                assert build_args == original_build_args
                custom_name = name_args.get("md_name") or None
                if custom_name:
                    expected_inf_names = [custom_name] * len(inf_modules)
                else:
                    expected_inf_names = list(inf_modules)
                ops_names = [
                    call.kwargs["ops_name"]
                    for call in core.get_args_of_build.call_args_list
                ]
                if use_callback:
                    assert ops_names == [module_name, *inf_modules]
                    core.build_module.assert_not_called()
                    expected_tune_name = custom_name or module_name
                    expected_names = [expected_tune_name, *expected_inf_names]
                    assert build_one_module.call_count == len(expected_names)
                    for build_call, expected in zip(
                        build_one_module.call_args_list,
                        expected_names,
                    ):
                        assert build_call.args[0] is not build_args
                        actual = build_call.args[0]["md_name"]
                        check(
                            f"{module_name} {name_args!r} callback name {expected}",
                            actual == expected,
                            f"got {actual!r}",
                        )
                else:
                    assert ops_names == inf_modules
                    assert core.build_module.call_count == len(inf_modules)
                    actual_names = [
                        call.kwargs["md_name"]
                        for call in core.build_module.call_args_list
                    ]
                    check(
                        f"{module_name} {name_args!r} rebuild names",
                        actual_names == expected_inf_names,
                        f"got {actual_names!r}",
                    )


def test_direct_inference_module_name():
    print("\n=== test_direct_inference_module_name ===")
    _test_pretune_module_names(use_callback=False)


def test_callback_module_names():
    print("\n=== test_callback_module_names ===")
    _test_pretune_module_names(use_callback=True)


def test_inference_modules():
    """Every supported tune module's own inference module is rebuilt, and CK and
    CKTile variants sharing a CSV are rebuilt together."""
    print("\n=== test_inference_modules ===")
    for module, _, config_attr, expect_skip in EXPECTED:
        if expect_skip:
            continue
        inf = _inference_modules(config_attr, CFG)
        own = module.removesuffix("_tune")
        check(f"{module} rebuilds {own}", own in inf, f"got {inf}")
        check(
            f"{module} rebuilds no tune module",
            not any(m.endswith("_tune") for m in inf),
        )

    for config_attr, expected in [
        (
            "AITER_CONFIG_GEMM_A8W8_BPRESHUFFLE_FILE",
            {"module_gemm_a8w8_bpreshuffle", "module_gemm_a8w8_bpreshuffle_cktile"},
        ),
        (
            "AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_FILE",
            {"module_gemm_a8w8_blockscale", "module_gemm_a8w8_blockscale_cktile"},
        ),
    ]:
        inf = set(_inference_modules(config_attr, CFG))
        check(f"{config_attr} → CK and CKTile", inf == expected, f"got {inf}")


class _FakeCore:
    """Stands in for aiter.jit.core: records rebuilds instead of compiling."""

    def __init__(self, csv_paths: dict[str, str]):
        self.AITER_CONFIGS = SimpleNamespace(**csv_paths)
        for config_attr, csv_path in csv_paths.items():
            setattr(self, config_attr.removesuffix("_FILE"), csv_path)
        self.removed: list[str] = []
        self.built: list[str] = []

    def rm_module(self, name):
        self.removed.append(name)

    def clear_build(self, name):
        pass

    def get_args_of_build(self, ops_name):
        # Like the real one for a single module: md_name is left empty.
        return {
            "md_name": "",
            "srcs": ["fake.cu"],
            "flags_extra_cc": [],
            "flags_extra_hip": [],
            "blob_gen_cmd": "",
            "extra_include": [],
            "third_party": [],
        }

    def build_module(self, md_name, **kwargs):
        self.built.append(md_name)


def _with_tuner_exit(returncode, fn):
    """Run fn() with every tuner subprocess reporting returncode.

    returncode is an int, or a callable that takes the tuner command and
    returns the exit code for it.
    """
    exit_for = returncode if callable(returncode) else lambda cmd: returncode
    with patch(
        "utility.pretune.subprocess.run",
        side_effect=lambda cmd, **kw: subprocess.CompletedProcess(cmd, exit_for(cmd)),
    ):
        return fn()


def _shape_csv() -> str:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        pd.DataFrame({"M": [128], "N": [512], "K": [1024]}).to_csv(f.name, index=False)
        return f.name


def test_run_pretune_tuner_failure():
    """A failing tuner raises and leaves every inference module untouched."""
    print("\n=== test_run_pretune_tuner_failure ===")
    module = "module_gemm_a8w8_bpreshuffle_cktile_tune"
    attr = "AITER_CONFIG_GEMM_A8W8_BPRESHUFFLE_FILE"
    csv_path = _shape_csv()
    core = _FakeCore({attr: csv_path})
    try:
        raised = False
        try:
            _with_tuner_exit(
                1, lambda: run_pretune(module, CFG, core, CSRC_DIR, REPO_DIR)
            )
        except RuntimeError:
            raised = True
        check("tuner exit 1 raises RuntimeError", raised)
        check("no inference module removed", core.removed == [], f"got {core.removed}")
        check("no inference module built", core.built == [], f"got {core.built}")
    finally:
        os.unlink(csv_path)


def test_run_pretune_rebuilds_cktile():
    """A successful CKTile pretune rebuilds the CKTile module, not only its CK parent."""
    print("\n=== test_run_pretune_rebuilds_cktile ===")
    module = "module_gemm_a8w8_bpreshuffle_cktile_tune"
    attr = "AITER_CONFIG_GEMM_A8W8_BPRESHUFFLE_FILE"
    csv_path = _shape_csv()
    core = _FakeCore({attr: csv_path})
    try:
        _with_tuner_exit(0, lambda: run_pretune(module, CFG, core, CSRC_DIR, REPO_DIR))
        expected = {
            "module_gemm_a8w8_bpreshuffle",
            "module_gemm_a8w8_bpreshuffle_cktile",
        }
        check(
            "removed CK and CKTile",
            set(core.removed) == expected,
            f"got {core.removed}",
        )
        check(
            "built CK and CKTile under their own names",
            sorted(core.built) == sorted(expected),
            f"got {core.built}",
        )
    finally:
        os.unlink(csv_path)


def test_run_pretune_modules_reports_failures():
    """run_pretune_modules keeps going after a failed module, then raises naming
    only the failures."""
    print("\n=== test_run_pretune_modules_reports_failures ===")
    failing = "module_gemm_a8w8_bpreshuffle_tune"
    passing = "module_gemm_a8w8_blockscale_tune"
    failing_script, failing_attr = _resolve(failing, CFG, CSRC_DIR)
    passing_script, passing_attr = _resolve(passing, CFG, CSRC_DIR)
    passing_inf = _inference_modules(passing_attr, CFG)
    csv_path = _shape_csv()
    core = _FakeCore({failing_attr: csv_path, passing_attr: csv_path})
    built: list[str] = []
    tuned: list[str] = []

    def tuner_exit(cmd):
        tuned.append(cmd[1])
        return 1 if cmd[1] == failing_script else 0

    try:
        message = ""
        try:
            _with_tuner_exit(
                tuner_exit,
                lambda: run_pretune_modules(
                    f"{failing},{passing}",
                    CFG,
                    core,
                    lambda args: built.append(args["md_name"]),
                    CSRC_DIR,
                    REPO_DIR,
                ),
            )
        except RuntimeError as exc:
            message = str(exc)
        check(
            "both tuners run, the failing one first",
            tuned == [failing_script, passing_script],
            f"got {tuned}",
        )
        check(
            "raises naming only the failed module",
            message == f"[pretune] failed modules: {failing}",
            f"got {message!r}",
        )
        check(
            "only the passing module's inference modules are rebuilt",
            core.removed == passing_inf,
            f"got {core.removed}",
        )
        check(
            "tune modules and passing inference modules built under their own names",
            built == [failing, passing, *passing_inf],
            f"got {built}",
        )
    finally:
        os.unlink(csv_path)


def test_main_exits_nonzero_on_failure():
    """_main keeps going after a failed module and exits 1 naming it; with no
    failure it returns normally."""
    print("\n=== test_main_exits_nonzero_on_failure ===")
    failing = "module_gemm_a8w8_bpreshuffle_tune"
    passing = "module_gemm_a8w8_blockscale_tune"
    # _main imports jit.core, which needs torch; run_pretune is faked, so any
    # object will do.
    fake_jit = ModuleType("jit")
    fake_jit.core = object()
    calls: list[str] = []

    def fake_run_pretune(mod, *args, **kwargs):
        calls.append(mod)
        if mod == failing:
            raise RuntimeError("tuner exited 1")

    for modules, expected_calls, expected_code in (
        (f"{failing},{passing}", [failing, passing], 1),
        (passing, [passing], None),
    ):
        calls.clear()
        code = None
        out = io.StringIO()
        with (
            patch.object(sys, "argv", ["pretune.py", modules]),
            patch.object(sys, "path", list(sys.path)),
            patch.dict(sys.modules, {"jit": fake_jit}),
            patch.object(pretune, "run_pretune", fake_run_pretune),
            contextlib.redirect_stdout(out),
        ):
            try:
                pretune._main()
            except SystemExit as exc:
                code = exc.code
        check(f"{modules}: runs every module", calls == expected_calls, f"got {calls}")
        check(f"{modules}: exit code", code == expected_code, f"got {code!r}")
        named = f"[pretune] failed modules: {failing}" in out.getvalue()
        check(
            f"{modules}: failed modules listed only on failure",
            named == (expected_code == 1),
            f"output: {out.getvalue()!r}",
        )


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_resolve()
    test_all_tune_modules_covered()
    test_script_fallback_keys_in_config()
    test_make_untune_csv()
    test_make_untune_csv_multi_path()
    test_make_untune_csv_missing_raises()
    test_write_tune_file_resolution()
    test_parse_pretune_modules()
    test_direct_inference_module_name()
    test_callback_module_names()
    test_inference_modules()
    test_run_pretune_tuner_failure()
    test_run_pretune_rebuilds_cktile()
    test_run_pretune_modules_reports_failures()
    test_main_exits_nonzero_on_failure()

    print(f"\n{'='*50}")
    print(f"Results: {_passed} passed, {_failed} failed")
    if _failed:
        sys.exit(1)
