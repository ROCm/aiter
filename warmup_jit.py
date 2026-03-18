#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Parallel JIT warmup for AITER.

Pre-builds all (or selected) JIT modules in parallel using a thread pool,
so that subsequent `import aiter` and model loading incur zero JIT latency.

Typical usage — run once before starting an inference server::

    # Build 4 modules at a time (default), skip expensive optional ones
    python warmup_jit.py --exclude module_aiter_operator

    # Then start the server as usual
    ENABLE_CK=0 AITER_USE_OPUS_MOE_SORTING=1 python -m atom.entrypoints.openai_server ...

The script leverages the existing ``PREBUILD_THREAD_NUM`` mechanism in
AITER's ``_get_num_workers()`` so that each parallel ninja invocation
gets a fair share of CPU cores (MAX_JOBS / parallel).
"""
import argparse
import glob
import logging
import os
import shutil
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger("aiter")
logging.basicConfig(level=logging.INFO, format="[aiter] %(message)s")


def warmup_build_all(max_parallel_modules=4, exclude=None, clean=False):
    """Pre-build all needed JIT modules in parallel.

    Args:
        max_parallel_modules: number of modules to compile concurrently.
            CPU budget is split via PREBUILD_THREAD_NUM so each ninja
            invocation gets MAX_JOBS / max_parallel_modules workers.
        exclude: optional list of module names to skip.
        clean: if True, remove all cached .so files before building.

    Returns:
        tuple: (list of (name, elapsed, success), total_elapsed)
    """
    from aiter.jit.core import get_user_jit_dir, get_args_of_build, build_module

    jit_dir = get_user_jit_dir()
    bd_dir = f"{jit_dir}/build"

    if clean:
        logger.info("[warmup] Cleaning compiled modules...")
        for so in glob.glob(f"{jit_dir}/*.so"):
            os.remove(so)
        if os.path.exists(bd_dir):
            shutil.rmtree(bd_dir)
        os.makedirs(bd_dir, exist_ok=True)

    # Gather all buildable modules
    all_opts, _ = get_args_of_build("all")

    # Apply exclusions
    exclude_set = set(exclude) if exclude else set()
    all_opts = [opt for opt in all_opts if opt["md_name"] not in exclude_set]

    # Filter to modules whose .so does not yet exist
    needed = [
        opt
        for opt in all_opts
        if not os.path.exists(os.path.join(jit_dir, f"{opt['md_name']}.so"))
    ]

    existing = len(glob.glob(f"{jit_dir}/*.so"))
    logger.info(f"[warmup] Existing .so: {existing}, to build: {len(needed)}")

    if not needed:
        logger.info("[warmup] All modules already compiled — nothing to do")
        return [], 0.0

    # Set PREBUILD_THREAD_NUM so _get_num_workers() divides MAX_JOBS
    orig_ptn = os.environ.get("PREBUILD_THREAD_NUM")
    os.environ["PREBUILD_THREAD_NUM"] = str(max_parallel_modules)

    def _build_one(opt):
        md_name = opt["md_name"]
        t0 = time.perf_counter()
        try:
            build_module(
                md_name=md_name,
                srcs=opt["srcs"],
                flags_extra_cc=opt["flags_extra_cc"],
                flags_extra_hip=opt["flags_extra_hip"],
                blob_gen_cmd=opt["blob_gen_cmd"],
                extra_include=opt["extra_include"],
                extra_ldflags=None,
                verbose=False,
                is_python_module=True,
                is_standalone=False,
                torch_exclude=False,
                third_party=opt.get("third_party", []),
            )
        except Exception:
            elapsed = time.perf_counter() - t0
            logger.warning(f"[warmup] {md_name} FAILED after {elapsed:.1f}s")
            traceback.print_exc()
            return md_name, elapsed, False
        elapsed = time.perf_counter() - t0
        logger.info(f"[warmup] {md_name} done in {elapsed:.1f}s")
        return md_name, elapsed, True

    logger.info(
        f"[warmup] Building {len(needed)} modules "
        f"(max_parallel={max_parallel_modules})"
    )
    t_start = time.perf_counter()

    with ThreadPoolExecutor(max_workers=max_parallel_modules) as executor:
        futures = {
            executor.submit(_build_one, opt): opt["md_name"] for opt in needed
        }
        results = []
        for future in as_completed(futures):
            results.append(future.result())

    # Restore
    if orig_ptn is not None:
        os.environ["PREBUILD_THREAD_NUM"] = orig_ptn
    else:
        os.environ.pop("PREBUILD_THREAD_NUM", None)

    total = time.perf_counter() - t_start
    ok = sum(1 for r in results if r[2])
    fail = sum(1 for r in results if not r[2])
    logger.info(
        f"[warmup] Done in {total:.1f}s — built: {ok}, failed: {fail}, "
        f"total .so: {len(glob.glob(f'{jit_dir}/*.so'))}"
    )
    return results, total


def main():
    parser = argparse.ArgumentParser(
        description="Pre-build AITER JIT modules in parallel"
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=4,
        help="Number of parallel module builds (default: 4)",
    )
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="Build sequentially (equivalent to --parallel 1)",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove all cached .so files before building",
    )
    parser.add_argument(
        "--exclude",
        nargs="*",
        default=[],
        help="Module names to skip (e.g. module_aiter_operator)",
    )
    args = parser.parse_args()

    if args.sequential:
        args.parallel = 1

    results, total = warmup_build_all(
        max_parallel_modules=args.parallel,
        exclude=args.exclude,
        clean=args.clean,
    )

    if results:
        results.sort(key=lambda x: -x[1])
        print("\nPer-module times:")
        for name, elapsed, success in results:
            status = "OK" if success else "FAIL"
            print(f"  {name}: {elapsed:.1f}s [{status}]")


if __name__ == "__main__":
    main()
