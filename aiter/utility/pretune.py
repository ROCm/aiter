# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""
pretune.py — run GEMM tuners for all CSV shapes on the live GPU.

Two entry points:

1. Via PRETUNE_MODULES during setup.py build (full build + retune + .so rebuild):

    PREBUILD_KERNELS=1 PRETUNE_MODULES=module_gemm_a8w8_blockscale_tune \
    python setup.py develop

2. As a standalone script on an already-installed aiter (tune only, no full rebuild):

    python3 aiter/utility/pretune.py module_gemm_a8w8_blockscale_tune
    python3 aiter/utility/pretune.py module_gemm_a8w8_tune,module_gemm_a8w8_blockscale_tune
    python3 aiter/utility/pretune.py all
    python3 aiter/utility/pretune.py --list          # show available tune modules

   After standalone tuning, rebuild the inference .so with:
    AITER_REBUILD=1 python3 op_tests/test_gemm_a8w8_blockscale.py

Both modes accept a single module name, a comma-separated list, or "all".
Requires a live GPU matching the target architecture.
All shapes in the merged tune CSV are (re-)tuned for the live GPU.

Flow per module (PRETUNE_MODULES / setup.py path):
  gen_instances.py --tune       →  build tune .so  (all candidate kernels)
  <tune_script>.py --all        →  benchmark on live GPU, update CSV
  gen_instances.py --tune_file  →  rebuild inference .so  (winners only)

Flow per module (standalone / direct path):
  <tune_script>.py --all        →  JIT-builds tune .so on first run, then
                                   benchmarks all shapes and writes winners
                                   back to the primary source CSV
"""

import json
import logging
import os
import re
import subprocess
import sys
import tempfile

logger = logging.getLogger("aiter")


# ---------------------------------------------------------------------------
# Tune module script fallback table
#
# Some _tune modules share a tune script with a parent module.  The parent
# tuner covers the child's kernel family via --libtype all.
# Value None means no viable tune script exists — the module is skipped with
# a warning.
#
# Background:
#   gemm_a8w8_blockscale_tune.py covers cktile and standard bpreshuffle
#   variants via --libtype all, but it writes to AITER_CONFIG_GEMM_A8W8_BLOCKSCALE.
#   The blockscale_bpreshuffle family uses a separate CSV
#   (AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_BPRESHUFFLE_FILE) that no existing
#   .py script writes to — those modules cannot be pretuned until a dedicated
#   tune script is added.
# ---------------------------------------------------------------------------
_SCRIPT_FALLBACK: dict = {
    # cktile variant: covered by blockscale parent tuner (--libtype all)
    "module_gemm_a8w8_blockscale_cktile_tune": "module_gemm_a8w8_blockscale_tune",
    # bpreshuffle_cktile: covered by bpreshuffle parent tuner
    "module_gemm_a8w8_bpreshuffle_cktile_tune": "module_gemm_a8w8_bpreshuffle_tune",
    # blockscale_bpreshuffle variants: no tune script writes to the bpreshuffle CSV
    "module_gemm_a8w8_blockscale_bpreshuffle_tune": None,
    "module_gemm_a8w8_blockscale_bpreshuffle_cktile_tune": None,
}

_SENTINEL = object()  # distinct from None: "not in fallback table"


def _get_tune_script(entry: dict, csrc_dir: str):
    """Derive the tune .py path from the non-pybind _tune.cu src entry."""
    AITER_CSRC_DIR = csrc_dir  # noqa: N806 — referenced by eval()
    for src_expr in entry.get("srcs", []):
        if "_tune" in src_expr and "pybind" not in src_expr:
            try:
                return eval(src_expr).replace(".cu", ".py")  # noqa: S307
            except Exception:
                pass
    return None


def _get_config_attr(cfg: dict, tune_module_name: str):
    """
    Find the AITER_CONFIGS.<ATTR> property name used by the inference module
    that corresponds to this tune module.

    Strips _cktile_tune / _tune suffixes to derive candidate inference module
    names and searches their blob_gen_cmd for AITER_CONFIGS.<ATTR>.
    """
    candidates = [
        tune_module_name.replace("_cktile_tune", "").replace("_tune", ""),
        tune_module_name.replace("_tune", ""),
    ]
    for inf_name in candidates:
        cmd = cfg.get(inf_name, {}).get("blob_gen_cmd", "")
        m = re.search(r"AITER_CONFIGS\.(\w+)", cmd)
        if m:
            return m.group(1)
    return None


def _resolve(module_name: str, cfg: dict, csrc_dir: str):
    """
    Return (tune_script_path, config_attr) for a tune module.

    Looks up the module's own tune script; if absent or missing on disk,
    consults _SCRIPT_FALLBACK.  Returns (None, config_attr) when no script
    is available.
    """
    entry = cfg.get(module_name, {})
    tune_script = _get_tune_script(entry, csrc_dir)
    config_attr = _get_config_attr(cfg, module_name)

    if tune_script and not os.path.exists(tune_script):
        tune_script = None

    if tune_script is None:
        fallback_key = _SCRIPT_FALLBACK.get(module_name, _SENTINEL)
        if fallback_key is _SENTINEL:
            # Not in table: auto-derive parent by stripping _cktile suffix
            parent = module_name.replace("_cktile_tune", "_tune")
            fallback_key = parent if (parent != module_name and parent in cfg) else None
        if fallback_key is not None:
            fb_script = _get_tune_script(cfg.get(fallback_key, {}), csrc_dir)
            if fb_script and os.path.exists(fb_script):
                tune_script = fb_script

    return tune_script, config_attr


def _all_tune_modules(cfg: dict) -> list:
    return [k for k in cfg if k.endswith("_tune")]


def _make_untune_csv(tune_file: str, shape_keys: list) -> str:
    """
    Read all paths in tune_file (colon-separated AITER multi-config format),
    concatenate, extract all unique rows for shape_keys columns (absent columns
    silently ignored), and write to a named temp file.

    Returns the temp file path — caller must delete it.
    """
    import pandas as pd  # deferred: absent during CI metadata-only phase

    paths = [p for p in tune_file.split(os.pathsep) if p]
    dfs = []
    for p in paths:
        if os.path.exists(p):
            dfs.append(pd.read_csv(p))
        else:
            logger.warning(f"[pretune] CSV not found, skipping: {p}")

    if not dfs:
        raise FileNotFoundError(f"[pretune] No CSV files found for: {tune_file}")

    merged = pd.concat(dfs, ignore_index=True)
    present = [k for k in shape_keys if k in merged.columns]
    if not present:
        raise ValueError(
            f"[pretune] None of {shape_keys} found in CSV columns: "
            f"{merged.columns.tolist()}"
        )

    shapes = merged[present].drop_duplicates().reset_index(drop=True)
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", prefix="aiter_pretune_", delete=False
    )
    shapes.to_csv(tmp.name, index=False)
    tmp.close()
    logger.info(f"[pretune] {len(shapes)} unique shapes → {tmp.name}")
    return tmp.name


def run_pretune(
    module_name: str,
    cfg: dict,
    core,
    build_one_module,
    csrc_dir: str,
    repo_dir: str,
) -> None:
    """
    Full pretune cycle for one tune module:

      1. Resolve tune script and CSV config attr from optCompilerConfig.json.
      2. Build the tune .so  (gen_instances.py --tune → all candidate kernels).
      3. Write a temp untune CSV: all unique shape keys from the merged tune CSV,
         no gfx/cu_num — the tuner auto-fills those from the live GPU.
      4. Run the tune script with --all so every shape, including those already
         tuned for the live GPU, is re-benchmarked with the current kernel set.
      5. Rebuild the inference .so  (gen_instances.py --tune_file → winners only).
    """
    tune_script, config_attr = _resolve(module_name, cfg, csrc_dir)

    if not tune_script:
        logger.warning(f"[pretune] {module_name}: no tune script available. Skipping.")
        return
    if not config_attr:
        logger.warning(
            f"[pretune] {module_name}: cannot determine CSV config attr. Skipping."
        )
        return

    tune_file = getattr(core.AITER_CONFIGS, config_attr)
    logger.info(
        f"[pretune] {module_name}: "
        f"script={os.path.relpath(tune_script, repo_dir)}, "
        f"tune_file={tune_file}"
    )

    # ── 1. Build tune .so ──────────────────────────────────────────────────
    tune_args = core.get_args_of_build(ops_name=module_name)
    if isinstance(tune_args, dict) and tune_args.get("srcs"):
        logger.info(f"[pretune] building {module_name}")
        build_one_module(tune_args)
    else:
        logger.warning(
            f"[pretune] get_args_of_build({module_name!r}) returned no srcs. "
            "Tune .so may already exist or module is unknown."
        )

    # ── 2. Write untune CSV ────────────────────────────────────────────────
    # Shape key columns only (no gfx/cu_num).  B included for batched GEMM;
    # silently dropped if absent.
    # With --all + untune_file != tune_file, get_retune_gemm_list() else-branch
    # auto-tags rows with live GPU's gfx/cu_num, re-benchmarks shapes already
    # in tune_file, and tunes shapes not yet present for this GPU.
    shape_keys = ["B", "M", "N", "K"]
    untune_csv = _make_untune_csv(tune_file, shape_keys)

    try:
        # ── 3. Run tuner ───────────────────────────────────────────────────
        env = {
            **os.environ,
            "PYTHONPATH": f"{repo_dir}{os.pathsep}{os.environ.get('PYTHONPATH', '')}",
        }
        cmd = [
            sys.executable,
            tune_script,
            "--untune_file",
            untune_csv,
            "--tune_file",
            tune_file,
            "--libtype",
            "all",
            "--all",
        ]
        logger.info(f"[pretune] {' '.join(cmd)}")
        result = subprocess.run(cmd, env=env)
        if result.returncode != 0:
            logger.warning(
                f"[pretune] tuner exited {result.returncode} for {module_name}. "
                "Inference module will still be rebuilt with whatever was written."
            )
    finally:
        try:
            os.unlink(untune_csv)
        except OSError:
            pass

    # ── 4. Rebuild inference .so ───────────────────────────────────────────
    inf_module = re.sub(r"_cktile_tune$|_tune$", "", module_name)
    logger.info(f"[pretune] rebuilding inference module {inf_module}")
    core.rm_module(inf_module)
    core.clear_build(inf_module)
    inf_args = core.get_args_of_build(ops_name=inf_module)
    if isinstance(inf_args, dict) and inf_args.get("srcs"):
        build_one_module(inf_args)
    else:
        logger.warning(
            f"[pretune] get_args_of_build({inf_module!r}) returned no srcs. "
            "Inference module not rebuilt."
        )


def run_pretune_modules(
    pretune_env: str,
    cfg: dict,
    core,
    build_one_module,
    csrc_dir: str,
    repo_dir: str,
) -> None:
    """
    Parse PRETUNE_MODULES and dispatch run_pretune() for each requested module.

    pretune_env values:
      "all"                                          → every _tune module in config
      "module_gemm_a8w8_blockscale_tune"             → single module
      "module_gemm_a8w8_tune,module_gemm_a8w8_blockscale_tune"  → comma list
    """
    value = pretune_env.strip()
    if value.lower() == "all":
        modules = _all_tune_modules(cfg)
        logger.info(f"[pretune] PRETUNE_MODULES=all → {len(modules)} tune modules")
    else:
        modules = [m.strip() for m in value.split(",") if m.strip()]

    for mod in modules:
        try:
            run_pretune(mod, cfg, core, build_one_module, csrc_dir, repo_dir)
        except Exception as exc:
            logger.warning(
                f"[pretune] {mod} failed: {exc}. Continuing with remaining modules."
            )


def run_tune_direct(
    module_name: str,
    cfg: dict,
    csrc_dir: str,
    repo_dir: str,
    libtype: str = "all",
) -> None:
    """
    Run the GEMM tuner directly for a single module without a full setup.py build.

    Suitable for interactive retuning on an already-installed aiter.  The tune
    .so is JIT-built on the first run if not already present.  Results are
    written back to the primary source CSV (e.g. aiter/configs/a8w8_blockscale_tuned_gemm.csv).

    After this call, rebuild the inference .so with:
        AITER_REBUILD=1 python3 op_tests/test_gemm_a8w8_blockscale.py

    Args:
        module_name: tune module name from optCompilerConfig.json
        cfg:         parsed optCompilerConfig.json dict
        csrc_dir:    absolute path to aiter/csrc/
        repo_dir:    absolute path to aiter repo root
        libtype:     kernel families to tune — "ck", "cktile", or "all"
    """
    # Deferred import: core requires torch, not available during CI metadata phase
    sys.path.insert(0, os.path.join(repo_dir, "aiter"))
    from jit import core  # noqa: PLC0415

    tune_script, config_attr = _resolve(module_name, cfg, csrc_dir)

    if not tune_script:
        print(f"[pretune] {module_name}: no tune script available. Skipping.")
        return
    if not config_attr:
        print(f"[pretune] {module_name}: cannot determine CSV config attr. Skipping.")
        return

    # Merged temp path — used as shape source (includes model_config CSVs)
    merged_tune_file = getattr(core.AITER_CONFIGS, config_attr)

    # Primary source CSV — strip _FILE to get the module-level env var name,
    # then read the first colon-separated path as the write-back target.
    source_attr = config_attr.removesuffix("_FILE")
    source_paths_str = getattr(core.AITER_CONFIGS, source_attr, None)
    if source_paths_str:
        write_tune_file = source_paths_str.split(os.pathsep)[0]
    else:
        write_tune_file = merged_tune_file

    print(
        f"[pretune] {module_name}\n"
        f"  script : {os.path.relpath(tune_script, repo_dir)}\n"
        f"  shapes : {merged_tune_file}\n"
        f"  output : {write_tune_file}\n"
        f"  libtype: {libtype}",
        flush=True,
    )

    shape_keys = ["B", "M", "N", "K"]
    untune_csv = _make_untune_csv(merged_tune_file, shape_keys)

    try:
        env = {
            **os.environ,
            "PYTHONPATH": f"{repo_dir}{os.pathsep}{os.environ.get('PYTHONPATH', '')}",
        }
        cmd = [
            sys.executable,
            tune_script,
            "--untune_file",
            untune_csv,
            "--tune_file",
            write_tune_file,
            "--libtype",
            libtype,
            "--all",
        ]
        print(f"[pretune] running: {' '.join(cmd)}", flush=True)
        result = subprocess.run(cmd, env=env)
        if result.returncode != 0:
            print(
                f"[pretune] tuner exited {result.returncode} for {module_name}.",
                flush=True,
            )
    finally:
        try:
            os.unlink(untune_csv)
        except OSError:
            pass

    print(
        f"[pretune] done. Rebuild inference .so with:\n"
        f"  AITER_REBUILD=1 python3 op_tests/test_gemm_a8w8_blockscale.py",
        flush=True,
    )


def _main() -> None:
    import argparse

    # Auto-detect repo root from this file's location: utility/pretune.py → repo root
    _this_dir = os.path.dirname(os.path.abspath(__file__))
    _default_repo_dir = os.path.dirname(os.path.dirname(_this_dir))

    parser = argparse.ArgumentParser(
        description=(
            "Tune GEMM shapes for the live GPU on an already-installed aiter.\n\n"
            "Examples:\n"
            "  python3 aiter/utility/pretune.py module_gemm_a8w8_blockscale_tune\n"
            "  python3 aiter/utility/pretune.py module_gemm_a8w8_tune,module_gemm_a8w8_blockscale_tune\n"
            "  python3 aiter/utility/pretune.py all\n"
            "  python3 aiter/utility/pretune.py --list"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "modules",
        nargs="?",
        help="Module name, comma-separated list, or 'all'.",
    )
    parser.add_argument(
        "--libtype",
        default="all",
        choices=["ck", "cktile", "all"],
        help="Kernel families to tune (default: all).",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Print available tune modules and exit.",
    )
    parser.add_argument(
        "--repo_dir",
        default=_default_repo_dir,
        help="Path to aiter repo root (auto-detected by default).",
    )
    args = parser.parse_args()

    repo_dir = os.path.abspath(args.repo_dir)
    csrc_dir = os.path.join(repo_dir, "csrc")
    cfg_path = os.path.join(repo_dir, "aiter", "jit", "optCompilerConfig.json")

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    if args.list:
        modules = _all_tune_modules(cfg)
        print(f"Available tune modules ({len(modules)}):")
        for m in sorted(modules):
            _, config_attr = _resolve(m, cfg, csrc_dir)
            skip = m in _SCRIPT_FALLBACK and _SCRIPT_FALLBACK[m] is None
            status = (
                "skip (no tune script)" if skip else config_attr or "unknown config"
            )
            print(f"  {m:<55} {status}")
        return

    if not args.modules:
        parser.print_help()
        return

    value = args.modules.strip()
    if value.lower() == "all":
        modules = _all_tune_modules(cfg)
        print(f"[pretune] tuning all {len(modules)} tune modules")
    else:
        modules = [m.strip() for m in value.split(",") if m.strip()]

    for mod in modules:
        try:
            run_tune_direct(mod, cfg, csrc_dir, repo_dir, libtype=args.libtype)
        except Exception as exc:
            print(f"[pretune] {mod} failed: {exc}. Continuing.")


if __name__ == "__main__":
    _main()
