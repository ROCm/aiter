# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""
test_gemm_codegen.py — unit tests for the gfx-aware GEMM codegen fix.

Tests the fix in fix/gemm_codegen_gfx_build_targets that adds gfx to:
  - gen_instances.py build-time filter (get_build_targets in chip_info.py)
  - Python runtime dispatch keys in gemm_op_a8w8.py et al.

No GPU kernel execution or .so compilation required.  All tests run on CPU
using only pandas and the chip_info / gemm_op_a8w8 Python layers.

Scenarios:
  1. get_build_targets() — env-driven target selection
  2. gen_instances filter simulation — CSV row selection matches target GPU
  3. Runtime dispatch key selection — (gfx, cu_num, M, N, K) lookup

Usage:
    python op_tests/test_gemm_codegen.py
    GPU_ARCHS=gfx942 python op_tests/test_gemm_codegen.py
"""

import os
import sys
import tempfile
import textwrap

# Ensure the repo-local aiter is imported, not any system/site-packages install.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)
# Import arch constants directly from build_targets — no torch dependency.
sys.path.insert(0, os.path.join(_REPO_ROOT, "aiter", "jit", "utils"))
from build_targets import (  # noqa: E402
    GFX_CU_NUM_MAP,
    filter_tune_df,
    get_build_targets_env,
)

import pandas as pd  # noqa: E402

REPRO_CSV = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "configs",
    "gemm_codegen_gfx_filter.csv",
)
REPRO_BPRESHUFFLE_CSV = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "configs",
    "gemm_codegen_gfx_filter_bpreshuffle.csv",
)

# GPU targets used throughout this test.  cu_num values match GFX_CU_NUM_MAP
# in aiter/jit/utils/build_targets.py (re-exported via chip_info.py) — update
# here if that mapping changes.
TARGET_A = ("gfx942", 304)  # MI300X
TARGET_B = ("gfx950", 256)  # MI350
TARGET_C = ("gfx942", 80)  # MI308X — gfx942 with CU_NUM override

# ---------------------------------------------------------------------------
# Minimal test harness (no external test framework required)
# ---------------------------------------------------------------------------

_passed = _failed = 0


def _check(name: str, condition: bool, detail: str = "") -> None:
    global _passed, _failed
    if condition:
        _passed += 1
        print(f"  PASS  {name}")
    else:
        _failed += 1
        msg = f"  FAIL  {name}"
        if detail:
            msg += f"\n        {detail}"
        print(msg)


def _section(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Section 1: get_build_targets()
# ---------------------------------------------------------------------------


def test_get_build_targets():
    _section("1. get_build_targets() — env-driven target selection")

    orig_archs = os.environ.pop("GPU_ARCHS", None)
    orig_cu = os.environ.pop("CU_NUM", None)

    try:
        # 1.1 Single known arch
        os.environ["GPU_ARCHS"] = TARGET_A[0]
        t = get_build_targets_env()
        _check(f"GPU_ARCHS={TARGET_A[0]} → [{TARGET_A}]", t == [TARGET_A], str(t))

        # 1.2 CU_NUM override (MI308X: gfx942 but cu_num=80)
        os.environ["GPU_ARCHS"] = TARGET_C[0]
        os.environ["CU_NUM"] = str(TARGET_C[1])
        t = get_build_targets_env()
        _check(
            f"GPU_ARCHS={TARGET_C[0]} + CU_NUM={TARGET_C[1]} → [{TARGET_C}]",
            t == [TARGET_C],
            str(t),
        )
        del os.environ["CU_NUM"]

        # 1.3 Second known arch
        os.environ["GPU_ARCHS"] = TARGET_B[0]
        t = get_build_targets_env()
        _check(f"GPU_ARCHS={TARGET_B[0]} → [{TARGET_B}]", t == [TARGET_B], str(t))

        # 1.4 Multi-arch (semicolon-separated)
        os.environ["GPU_ARCHS"] = f"{TARGET_A[0]};{TARGET_B[0]}"
        t = get_build_targets_env()
        _check(
            f"GPU_ARCHS={TARGET_A[0]};{TARGET_B[0]} → two targets",
            t == [TARGET_A, TARGET_B],
            str(t),
        )

        # 1.5 Unknown arch raises RuntimeError
        os.environ["GPU_ARCHS"] = "gfx999"
        raised = False
        try:
            get_build_targets_env()
        except RuntimeError:
            raised = True
        _check("GPU_ARCHS=gfx999 → RuntimeError", raised)

        # 1.6 GFX_CU_NUM_MAP covers at least the two known production targets
        _check(
            "GFX_CU_NUM_MAP contains gfx942 and gfx950",
            "gfx942" in GFX_CU_NUM_MAP and "gfx950" in GFX_CU_NUM_MAP,
        )

        # 1.7 Live GPU fallback — requires torch and a GPU; skipped otherwise
        del os.environ["GPU_ARCHS"]
        try:
            from aiter.jit.utils.chip_info import get_build_targets

            t = get_build_targets()
            _check(
                "No GPU_ARCHS + live GPU → single (gfx, cu_num) pair",
                len(t) == 1 and isinstance(t[0], tuple) and len(t[0]) == 2,
                str(t),
            )
        except (ImportError, ModuleNotFoundError):
            print("  SKIP  No GPU_ARCHS + live GPU (torch not available)")
        except RuntimeError:
            print("  SKIP  No GPU_ARCHS + live GPU (no GPU detected — expected in CI)")

    finally:
        if orig_archs is not None:
            os.environ["GPU_ARCHS"] = orig_archs
        elif "GPU_ARCHS" in os.environ:
            del os.environ["GPU_ARCHS"]
        if orig_cu is not None:
            os.environ["CU_NUM"] = orig_cu
        elif "CU_NUM" in os.environ:
            del os.environ["CU_NUM"]


# ---------------------------------------------------------------------------
# Section 2: gen_instances filter — uses filter_tune_df from build_targets
# ---------------------------------------------------------------------------


def test_gen_instances_filter(
    csv_path=None, target_a=TARGET_A, target_b=TARGET_B, label=""
):
    """Verify gen_instances filter behaviour against a repro CSV."""
    if csv_path is None:
        csv_path = REPRO_CSV
    pfx = f"[{label}] " if label else ""

    _section(
        f"2. gen_instances filter — CSV row selection per target{' (' + label + ')' if label else ''}"
    )

    if not os.path.exists(csv_path):
        print(f"  SKIP  repro CSV not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    gfx_a, cu_a = target_a
    gfx_b, cu_b = target_b

    # 2.1 gfx column present (fix applied to CSV)
    _check(f"{pfx}repro CSV has 'gfx' column", "gfx" in df.columns)

    # 2.2 Bug scenario: no filter compiles all rows (last-writer-wins)
    _check(
        f"{pfx}unfiltered CSV has rows for multiple gfx targets (bug: all compiled)",
        df["gfx"].nunique() > 1,
        f"gfx targets found: {df['gfx'].unique().tolist()}",
    )

    # 2.3 Fix: filter for target_a selects only those rows
    filtered = filter_tune_df(df, [target_a])
    _check(
        f"{pfx}{gfx_a}/cu_num={cu_a} filter keeps only {gfx_a} rows",
        len(filtered) > 0
        and all(filtered["gfx"] == gfx_a)
        and all(filtered["cu_num"] == cu_a),
        f"rows={len(filtered)}, gfx={filtered['gfx'].unique().tolist()}",
    )

    # 2.4 Fix: filter for target_b selects only those rows
    filtered = filter_tune_df(df, [target_b])
    _check(
        f"{pfx}{gfx_b}/cu_num={cu_b} filter keeps only {gfx_b} rows",
        len(filtered) > 0
        and all(filtered["gfx"] == gfx_b)
        and all(filtered["cu_num"] == cu_b),
        f"rows={len(filtered)}",
    )

    # 2.5 Multi-arch filter is the union of per-arch filters
    n_a = len(filter_tune_df(df, [target_a]))
    n_b = len(filter_tune_df(df, [target_b]))
    n_multi = len(filter_tune_df(df, [target_a, target_b]))
    _check(
        f"{pfx}multi-arch filter row count equals sum of individual filters",
        n_multi == n_a + n_b,
        f"multi={n_multi}, {gfx_a}/{cu_a}={n_a}, {gfx_b}/{cu_b}={n_b}",
    )

    # 2.6 All MNK shapes in the repro CSV have different kernelIds across gfx targets
    grp = df.groupby(["M", "N", "K"])["kernelId"].nunique()
    shapes_with_diff = grp[grp > 1]
    _check(
        f"{pfx}repro CSV has shapes with different kernelIds across gfx targets",
        len(shapes_with_diff) > 0,
        f"shapes with diverging kernelIds: {len(shapes_with_diff)}/{len(grp)}",
    )

    # 2.7 Contamination: the two targets share MNK shapes with different kernelIds
    d_a = filter_tune_df(df, [target_a]).set_index(["M", "N", "K"])
    d_b = filter_tune_df(df, [target_b]).set_index(["M", "N", "K"])
    common = d_a.index.intersection(d_b.index)
    if len(common) > 0:
        n_diff = sum(
            d_a.loc[idx, "kernelId"] != d_b.loc[idx, "kernelId"] for idx in common
        )
        _check(
            f"{pfx}shared MNK shapes have different kernelIds across {gfx_a}/{cu_a} and {gfx_b}/{cu_b}",
            n_diff > 0,
            f"{n_diff}/{len(common)} shared shapes have diverging kernelIds",
        )
    else:
        print(
            f"  SKIP  no MNK overlap between {gfx_a}/{cu_a} and {gfx_b}/{cu_b} in repro CSV"
        )


# ---------------------------------------------------------------------------
# Section 3: Python runtime dispatch key selection
# Tests get_CKGEMM_config() using unique temp CSV files to avoid polluting
# the module-level cache used by the real config files.
# ---------------------------------------------------------------------------


def _make_temp_csv(content: str) -> str:
    f = tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False, prefix="test_gemm_codegen_"
    )
    f.write(textwrap.dedent(content).strip() + "\n")
    f.close()
    return f.name


def test_runtime_dispatch_key():
    _section("3. Runtime dispatch — (gfx, cu_num, M, N, K) lookup key")

    try:
        from aiter.ops.gemm_op_a8w8 import get_CKGEMM_config
        import aiter.ops.gemm_op_a8w8 as _mod
    except Exception as e:
        print(f"  SKIP  could not import get_CKGEMM_config ({e})")
        return

    # get_CKGEMM_config() uses get_gfx_runtime() which always detects the live GPU
    # via rocminfo — GPU_ARCHS is intentionally ignored at runtime.  Derive the
    # test CSV rows from the actual live GPU so the test is correct on any runner.
    try:
        from aiter.jit.utils.chip_info import get_gfx_runtime, get_cu_num

        gfx = get_gfx_runtime()
        cu_num = get_cu_num()
    except Exception as e:
        print(f"  SKIP  runtime dispatch tests require a live GPU ({e})")
        return

    # Pick a "wrong" target that is guaranteed to differ from the live GPU.
    wrong_target = TARGET_B if gfx != TARGET_B[0] else TARGET_A
    wrong_gfx, wrong_cu_num = wrong_target

    csv_with_gfx = wrong_gfx_csv = old_csv = None
    try:
        # 3.1 New CSV schema (gfx column present) — correct target is found
        csv_with_gfx = _make_temp_csv(f"""
            gfx,cu_num,M,N,K,kernelId,splitK,us,kernelName,tflops,bw,errRatio
            {gfx},{cu_num},128,1280,8192,42,0,10.0,correct_kernel,100.0,500.0,0.0
            {wrong_gfx},{wrong_cu_num},128,1280,8192,99,0,10.0,wrong_kernel,100.0,500.0,0.0
        """)
        _mod._CKGEMM_CONFIG_CACHE = {}
        cfg = get_CKGEMM_config(128, 1280, 8192, tuned_file=csv_with_gfx)
        _check(
            "new CSV (gfx column): shape tuned for this gfx is found",
            cfg is not None,
            "returned None",
        )
        if cfg is not None:
            _check(
                "new CSV: kernelId matches this gfx target, not the other",
                cfg.get("kernelId") == 42,
                f"expected kernelId=42, got {cfg.get('kernelId')}",
            )

        # 3.2 Shape tuned only for a different gfx returns None on this target
        wrong_gfx_csv = _make_temp_csv(f"""
            gfx,cu_num,M,N,K,kernelId,splitK,us,kernelName,tflops,bw,errRatio
            {wrong_gfx},{wrong_cu_num},128,1280,8192,99,0,10.0,wrong_kernel,100.0,500.0,0.0
        """)
        _mod._CKGEMM_CONFIG_CACHE = {}
        cfg = get_CKGEMM_config(128, 1280, 8192, tuned_file=wrong_gfx_csv)
        _check(
            f"new CSV: shape tuned only for {wrong_gfx} returns None on {gfx}",
            cfg is None,
            f"expected None, got {cfg}",
        )

        # 3.3 Old CSV (no gfx column) falls back to cu_num-only key with a warning
        old_csv = _make_temp_csv(f"""
            cu_num,M,N,K,kernelId,splitK,us,kernelName,tflops,bw,errRatio
            {cu_num},128,1280,8192,7,0,10.0,old_kernel,100.0,500.0,0.0
        """)
        import logging
        import io

        buf = io.StringIO()
        handler = logging.StreamHandler(buf)
        logging.getLogger("aiter").addHandler(handler)
        _mod._CKGEMM_CONFIG_CACHE = {}
        cfg = get_CKGEMM_config(128, 1280, 8192, tuned_file=old_csv)
        logging.getLogger("aiter").removeHandler(handler)

        _check(
            "old CSV (no gfx column): shape still found via cu_num fallback",
            cfg is not None and cfg.get("kernelId") == 7,
            f"cfg={cfg}",
        )
        _check(
            "old CSV (no gfx column): deprecation warning is logged",
            "gfx" in buf.getvalue().lower(),
            f"log output: {buf.getvalue()!r}",
        )

    finally:
        _mod._CKGEMM_CONFIG_CACHE = {}
        _mod._CKGEMM_HAS_GFX = {}
        for path in [csv_with_gfx, wrong_gfx_csv, old_csv]:
            if path:
                try:
                    os.unlink(path)
                except Exception:
                    pass


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_get_build_targets()
    test_gen_instances_filter(
        csv_path=REPRO_CSV,
        target_a=TARGET_C,
        target_b=TARGET_B,
        label="module_gemm_a8w8",
    )
    test_gen_instances_filter(
        csv_path=REPRO_BPRESHUFFLE_CSV,
        target_a=TARGET_A,
        target_b=TARGET_B,
        label="module_gemm_a8w8_bpreshuffle",
    )
    test_runtime_dispatch_key()

    print(f"\n{'='*60}")
    print(f"  Results: {_passed} passed, {_failed} failed")
    print("=" * 60)
    sys.exit(0 if _failed == 0 else 1)
