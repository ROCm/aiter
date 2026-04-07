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
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

REPRO_CSV = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "configs", "gemm_codegen_gfx_filter.csv",
)
REPRO_BPRESHUFFLE_CSV = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "configs", "gemm_codegen_gfx_filter_bpreshuffle.csv",
)

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

    from aiter.jit.utils.chip_info import get_build_targets, GFX_CU_NUM_MAP

    orig_archs = os.environ.pop("GPU_ARCHS", None)
    orig_cu    = os.environ.pop("CU_NUM",    None)

    try:
        # 1.1 Single known arch
        os.environ["GPU_ARCHS"] = "gfx942"
        t = get_build_targets()
        _check("GPU_ARCHS=gfx942 → [('gfx942', 304)]",
               t == [("gfx942", 304)], str(t))

        # 1.2 CU_NUM override (MI308X: gfx942 but cu_num=80)
        os.environ["GPU_ARCHS"] = "gfx942"
        os.environ["CU_NUM"] = "80"
        t = get_build_targets()
        _check("GPU_ARCHS=gfx942 + CU_NUM=80 → [('gfx942', 80)]",
               t == [("gfx942", 80)], str(t))
        del os.environ["CU_NUM"]

        # 1.3 Second known arch
        os.environ["GPU_ARCHS"] = "gfx950"
        t = get_build_targets()
        _check("GPU_ARCHS=gfx950 → [('gfx950', 256)]",
               t == [("gfx950", 256)], str(t))

        # 1.4 Multi-arch (semicolon-separated)
        os.environ["GPU_ARCHS"] = "gfx942;gfx950"
        t = get_build_targets()
        _check("GPU_ARCHS=gfx942;gfx950 → two targets",
               t == [("gfx942", 304), ("gfx950", 256)], str(t))

        # 1.5 Unknown arch raises RuntimeError
        os.environ["GPU_ARCHS"] = "gfx999"
        raised = False
        try:
            get_build_targets()
        except RuntimeError:
            raised = True
        _check("GPU_ARCHS=gfx999 → RuntimeError", raised)

        # 1.6 GFX_CU_NUM_MAP covers at least the two known production targets
        _check("GFX_CU_NUM_MAP contains gfx942 and gfx950",
               "gfx942" in GFX_CU_NUM_MAP and "gfx950" in GFX_CU_NUM_MAP)

        # 1.7 Live GPU fallback (only meaningful when a GPU is present)
        del os.environ["GPU_ARCHS"]
        try:
            t = get_build_targets()
            _check("No GPU_ARCHS + live GPU → single (gfx, cu_num) pair",
                   len(t) == 1 and isinstance(t[0], tuple) and len(t[0]) == 2,
                   str(t))
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
# Section 2: gen_instances filter simulation
# Replicates the exact filter block from csrc/*/gen_instances.py so we can
# verify CSV row selection without running a build.
# ---------------------------------------------------------------------------

def _apply_filter(tune_df: pd.DataFrame, targets: list) -> pd.DataFrame:
    """Mirror of the filter block in the fixed gen_instances.py files."""
    mask = pd.Series([False] * len(tune_df), index=tune_df.index)
    for gfx, cu_num in targets:
        mask |= (tune_df["gfx"] == gfx) & (tune_df["cu_num"] == cu_num)
    return tune_df[mask].reset_index(drop=True)


def test_gen_instances_filter(
    csv_path=None,
    target_a=("gfx942", 80),
    target_b=("gfx950", 256),
    label="",
):
    """
    Verify gen_instances filter behaviour against a repro CSV.

    target_a / target_b: (gfx, cu_num) pairs for the two GPU targets in the CSV.
      main CSV:         target_a=("gfx942", 80),  target_b=("gfx950", 256)
      bpreshuffle CSV:  target_a=("gfx942", 304), target_b=("gfx950", 256)
    """
    if csv_path is None:
        csv_path = REPRO_CSV
    pfx = f"[{label}] " if label else ""

    _section(f"2. gen_instances filter — CSV row selection per target{' (' + label + ')' if label else ''}")

    if not os.path.exists(csv_path):
        print(f"  SKIP  repro CSV not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    gfx_a, cu_a = target_a
    gfx_b, cu_b = target_b

    # 2.1 gfx column present (fix applied to CSV)
    _check(f"{pfx}repro CSV has 'gfx' column", "gfx" in df.columns)

    # 2.2 Bug scenario: no filter compiles all rows (last-writer-wins)
    _check(f"{pfx}unfiltered CSV has rows for multiple gfx targets (bug: all compiled)",
           df["gfx"].nunique() > 1,
           f"gfx targets found: {df['gfx'].unique().tolist()}")

    # 2.3 Fix: filter for target_a selects only those rows
    filtered = _apply_filter(df, [target_a])
    _check(f"{pfx}{gfx_a}/cu_num={cu_a} filter keeps only {gfx_a} rows",
           len(filtered) > 0
           and all(filtered["gfx"] == gfx_a)
           and all(filtered["cu_num"] == cu_a),
           f"rows={len(filtered)}, gfx={filtered['gfx'].unique().tolist()}")

    # 2.4 Fix: filter for target_b selects only those rows
    filtered = _apply_filter(df, [target_b])
    _check(f"{pfx}{gfx_b}/cu_num={cu_b} filter keeps only {gfx_b} rows",
           len(filtered) > 0
           and all(filtered["gfx"] == gfx_b)
           and all(filtered["cu_num"] == cu_b),
           f"rows={len(filtered)}")

    # 2.5 Multi-arch filter is the union of per-arch filters
    n_a = len(_apply_filter(df, [target_a]))
    n_b = len(_apply_filter(df, [target_b]))
    n_multi = len(_apply_filter(df, [target_a, target_b]))
    _check(f"{pfx}multi-arch filter row count equals sum of individual filters",
           n_multi == n_a + n_b,
           f"multi={n_multi}, {gfx_a}/{cu_a}={n_a}, {gfx_b}/{cu_b}={n_b}")

    # 2.6 All MNK shapes in the repro CSV have different kernelIds across gfx targets
    grp = df.groupby(["M", "N", "K"])["kernelId"].nunique()
    shapes_with_diff = grp[grp > 1]
    _check(f"{pfx}repro CSV has shapes with different kernelIds across gfx targets",
           len(shapes_with_diff) > 0,
           f"shapes with diverging kernelIds: {len(shapes_with_diff)}/{len(grp)}")

    # 2.7 Contamination: the two targets share MNK shapes with different kernelIds
    d_a = _apply_filter(df, [target_a]).set_index(["M", "N", "K"])
    d_b = _apply_filter(df, [target_b]).set_index(["M", "N", "K"])
    common = d_a.index.intersection(d_b.index)
    if len(common) > 0:
        n_diff = sum(
            d_a.loc[idx, "kernelId"] != d_b.loc[idx, "kernelId"]
            for idx in common
        )
        _check(
            f"{pfx}shared MNK shapes have different kernelIds across {gfx_a}/{cu_a} and {gfx_b}/{cu_b}",
            n_diff > 0,
            f"{n_diff}/{len(common)} shared shapes have diverging kernelIds",
        )
    else:
        print(f"  SKIP  no MNK overlap between {gfx_a}/{cu_a} and {gfx_b}/{cu_b} in repro CSV")


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

    from aiter.jit.utils.chip_info import GFX_CU_NUM_MAP
    # Use GPU_ARCHS so the test is deterministic even without a live GPU.
    orig_archs = os.environ.get("GPU_ARCHS")
    os.environ["GPU_ARCHS"] = "gfx942"
    gfx    = "gfx942"
    cu_num = GFX_CU_NUM_MAP["gfx942"]   # 304

    csv_with_gfx = wrong_gfx_csv = old_csv = None
    try:
        # 3.1 New CSV schema (gfx column present) — correct target is found
        csv_with_gfx = _make_temp_csv(f"""
            gfx,cu_num,M,N,K,kernelId,splitK,us,kernelName,tflops,bw,errRatio
            {gfx},{cu_num},128,1280,8192,42,0,10.0,correct_kernel,100.0,500.0,0.0
            gfx950,256,128,1280,8192,99,0,10.0,wrong_kernel,100.0,500.0,0.0
        """)
        _mod._CKGEMM_CONFIG_CACHE = {}
        cfg = get_CKGEMM_config(128, 1280, 8192, tuned_file=csv_with_gfx)
        _check("new CSV (gfx column): shape tuned for this gfx is found",
               cfg is not None, "returned None")
        if cfg is not None:
            _check("new CSV: kernelId matches this gfx target, not the other",
                   cfg.get("kernelId") == 42,
                   f"expected kernelId=42, got {cfg.get('kernelId')}")

        # 3.2 Shape tuned only for a different gfx returns None on this target
        wrong_gfx_csv = _make_temp_csv(f"""
            gfx,cu_num,M,N,K,kernelId,splitK,us,kernelName,tflops,bw,errRatio
            gfx950,256,128,1280,8192,99,0,10.0,wrong_kernel,100.0,500.0,0.0
        """)
        _mod._CKGEMM_CONFIG_CACHE = {}
        cfg = get_CKGEMM_config(128, 1280, 8192, tuned_file=wrong_gfx_csv)
        _check("new CSV: shape tuned only for gfx950 returns None on gfx942",
               cfg is None, f"expected None, got {cfg}")

        # 3.3 Old CSV (no gfx column) falls back to cu_num-only key with a warning
        old_csv = _make_temp_csv(f"""
            cu_num,M,N,K,kernelId,splitK,us,kernelName,tflops,bw,errRatio
            {cu_num},128,1280,8192,7,0,10.0,old_kernel,100.0,500.0,0.0
        """)
        import logging, io
        buf = io.StringIO()
        handler = logging.StreamHandler(buf)
        logging.getLogger("aiter").addHandler(handler)
        _mod._CKGEMM_CONFIG_CACHE = {}
        cfg = get_CKGEMM_config(128, 1280, 8192, tuned_file=old_csv)
        logging.getLogger("aiter").removeHandler(handler)

        _check("old CSV (no gfx column): shape still found via cu_num fallback",
               cfg is not None and cfg.get("kernelId") == 7,
               f"cfg={cfg}")
        _check("old CSV (no gfx column): deprecation warning is logged",
               "gfx" in buf.getvalue().lower(),
               f"log output: {buf.getvalue()!r}")

    finally:
        _mod._CKGEMM_CONFIG_CACHE = {}
        if orig_archs is not None:
            os.environ["GPU_ARCHS"] = orig_archs
        elif "GPU_ARCHS" in os.environ:
            del os.environ["GPU_ARCHS"]
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
        target_a=("gfx942", 80),
        target_b=("gfx950", 256),
        label="module_gemm_a8w8",
    )
    test_gen_instances_filter(
        csv_path=REPRO_BPRESHUFFLE_CSV,
        target_a=("gfx942", 304),
        target_b=("gfx950", 256),
        label="module_gemm_a8w8_bpreshuffle",
    )
    test_runtime_dispatch_key()

    print(f"\n{'='*60}")
    print(f"  Results: {_passed} passed, {_failed} failed")
    print("=" * 60)
    sys.exit(0 if _failed == 0 else 1)
