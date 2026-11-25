# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Compare merge-target defaults and execute the production exact-key lookup."""

import argparse
import ast
import csv
import functools
import importlib.util
import io
import json
import logging
import subprocess
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def git(*args):
    return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True)


def keyed_rows(payload):
    return {
        (r["gfx"], *(int(r[k]) for k in ("cu_num", "M", "N", "K"))): (
            r["kernelName"],
            int(r["splitK"]),
        )
        for r in csv.DictReader(io.StringIO(payload))
    }


def lookup_function(path):
    tree = ast.parse((ROOT / "aiter/ops/gemm_op_a4w4.py").read_text())
    function = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "get_GEMM_config"
    )
    namespace = {
        "functools": functools,
        "pd": pd,
        "logger": logging.getLogger("audit"),
        "AITER_CONFIGS": SimpleNamespace(AITER_CONFIG_GEMM_A4W4_FILE=str(path)),
        "AITER_LOG_TUNED_CONFIG": False,
    }
    # Execute only the checked-out production getter, without device bootstrap.
    exec(  # noqa: S102
        compile(
            ast.Module(body=[function], type_ignores=[]), "gemm_op_a4w4.py", "exec"
        ),
        namespace,
    )
    return namespace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-ref", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    baseline_ref = git("rev-parse", args.baseline_ref).strip()
    paths = ["aiter/configs/a4w4_blockscale_tuned_gemm.csv"]
    paths.extend(
        p
        for p in git(
            "ls-tree", "-r", "--name-only", baseline_ref, "aiter/configs/model_configs"
        ).splitlines()
        if "a4w4_blockscale_tuned_gemm" in p
        and p.endswith(".csv")
        and "untuned" not in p
    )
    expected = {}
    for path in paths:
        rows = keyed_rows(git("show", f"{baseline_ref}:{path}"))
        assert not (expected.keys() & rows.keys()), f"overlapping baseline keys: {path}"
        expected.update(rows)
    spec = importlib.util.spec_from_file_location(
        "audit_mxfp4", ROOT / "aiter/utility/mxfp4_tuning.py"
    )
    api = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(api)
    actual = keyed_rows(api._merged_csv(api._default_sources()).decode())
    assert expected == actual, (len(expected), len(actual))
    namespace = lookup_function(api.get_mxfp4_config_file())
    getter = namespace["get_GEMM_config"]
    for key, selected in expected.items():
        gfx, cu, m, n, k = key
        namespace["get_gfx"] = lambda gfx=gfx: gfx
        namespace["get_cu_num"] = lambda cu=cu: cu
        getter.cache_clear()
        row = getter(m, n, k)
        assert (row["kernelName"], int(row["splitK"])) == selected, key
    with tempfile.TemporaryDirectory() as tmp:
        legacy = Path(tmp) / "legacy.csv"
        legacy.write_text("cu_num,M,N,K,kernelName,splitK\n256,32,64,128,legacy,0\n")
        derived = Path(tmp) / "derived.csv"
        derived.write_bytes(api._merged_csv((str(legacy),)))
        for gfx in ("gfx950", "gfx1250"):
            namespace = lookup_function(derived)
            namespace["get_gfx"] = lambda gfx=gfx: gfx
            namespace["get_cu_num"] = lambda: 256
            assert namespace["get_GEMM_config"](32, 64, 128)["kernelName"] == "legacy"
            assert not namespace["get_GEMM_config"].has_gfx
    record = {
        "baseline_revision": baseline_ref,
        "baseline_sources": paths,
        "default_keys": len(expected),
        "exact_lookup_checks": len(expected),
        "default_differences": 0,
        "gfx950_128_cu_exact_checks": sum(k[:2] == ("gfx950", 128) for k in expected),
        "legacy_cu_only_checks": 2,
        "scope": "Production get_GEMM_config function executed with pandas and substituted device metadata; exact-key checks only, no GPU execution or padded-M validation.",
    }
    Path(args.output).write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
