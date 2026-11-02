#!/usr/bin/env python3

# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""AOT pre-compilation for the FlyDSL paged-attention decode planner.

``plan_pa_decode`` compiles one kernel per batch bucket, block size and
dense/windowed mode; batch, capacity, partition cap and window are runtime
values. The kernels for batches up to a ``max_batch_size`` therefore form a
finite set per arch, derived here from the planner's own bucket and block-size
rules and the arch's LDS capacity. This script is standalone, not part of
``run_aot``.

Usage:
    python -m aiter.aot.flydsl.pa_decode_plan                        # 4096
    python -m aiter.aot.flydsl.pa_decode_plan --max-batch-size 32768 --arch gfx950
"""

from __future__ import annotations

import argparse
import time

from aiter.aot.flydsl.common import (
    aot_target_archs,
    compile_only_env,
    dedupe_jobs,
    override_env,
    run_jobs_parallel,
)

# Archs that pa_decode dispatches to.
AOT_ARCHS = ("gfx942", "gfx950", "gfx1250")


def plan_variants(max_batch_size: int, lds_bytes: int) -> list[tuple[int, int, bool]]:
    """Every ``compile_pa_decode_plan`` key that ``plan_pa_decode`` can reach
    for batches up to ``max_batch_size`` on a device with ``lds_bytes``.

    Buckets are every power of two below the cap and the cap itself, which the
    runtime clips to ``max_batch_size`` and the LDS limit. Capacity lies in
    [batch, batch * max_partitions], so a bucket's smallest batch fixes its
    smallest block and every larger block up to 1024 remains reachable for
    some partition cap.
    """
    from aiter.ops.flydsl.kernels.pa_decode_plan import _block_threads, _lds_max_batch

    max_batch = min(max_batch_size, _lds_max_batch(lds_bytes))
    buckets = [1 << bit for bit in range((max_batch - 1).bit_length())]
    buckets.append(max_batch)
    variants = []
    for bucket in buckets:
        # The previous bucket is the largest power of two below this one.
        threads = _block_threads((1 << (bucket - 1).bit_length()) // 2 + 1)
        while threads <= 1024:
            for windowed in (False, True):
                variants.append((bucket, threads, windowed))
            threads *= 2
    return variants


def jobs_for_arch(arch: str, max_batch_size: int) -> list[dict]:
    from aiter.jit.utils.chip_info import get_lds_capacity_bytes

    return [
        {
            "kernel_name": (
                f"pa_decode_plan_{arch}_b{max_batch}_t{threads}"
                f"_{'window' if windowed else 'dense'}"
            ),
            "arch": arch,
            "max_batch": max_batch,
            "threads": threads,
            "windowed": windowed,
        }
        for max_batch, threads, windowed in plan_variants(
            max_batch_size, get_lds_capacity_bytes(arch)
        )
    ]


def default_jobs(max_batch_sizes: list[int] | None = None) -> list[dict]:
    """Every variant of the archs ``GPU_ARCHS`` names (all supported archs when
    unset) for each ``max_batch_size``, by default the planner's default."""
    from aiter.ops.flydsl.kernels.pa_decode_plan import DEFAULT_MAX_BATCH_SIZE

    targets = aot_target_archs()
    archs = [arch for arch in AOT_ARCHS if targets is None or arch in targets]
    jobs = []
    for max_batch_size in max_batch_sizes or [DEFAULT_MAX_BATCH_SIZE]:
        for arch in archs:
            jobs.extend(jobs_for_arch(arch, max_batch_size))
    return dedupe_jobs(jobs)


def _compile_to_cache(*, max_batch, threads, windowed):
    import flydsl.compiler as flyc
    import flydsl.expr as fx

    from aiter.ops.flydsl.kernels.pa_decode_plan import compile_pa_decode_plan

    # Mirror the launch at the end of plan_pa_decode: pointer dtypes, the four
    # Int32 runtime values and the stream argument form the cache key, not
    # their values.
    null = flyc.from_c_void_p(fx.Int32, 0)
    flyc.compile(
        compile_pa_decode_plan(max_batch, threads, windowed),
        null,
        null,
        null,
        1,
        1,
        1,
        0,
        None,
    )


def compile_one_config(**job) -> dict:
    result = {**job, "compile_time": None}
    params = {k: v for k, v in job.items() if k not in ("kernel_name", "arch")}
    started = time.time()
    try:
        # COMPILE_ONLY is mandatory: without it flyc.compile also launches.
        with override_env("FLYDSL_GPU_ARCH", job["arch"]), compile_only_env():
            _compile_to_cache(**params)
        result["compile_time"] = time.time() - started
    except Exception as error:  # noqa: BLE001
        print(f"  [FAIL] {job['kernel_name']}: {error}")
    return result


def _positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--max-batch-size",
        type=_positive_int,
        nargs="+",
        help="planner batch caps to cover (default: the planner's default, 4096)",
    )
    parser.add_argument(
        "--arch",
        nargs="+",
        choices=AOT_ARCHS,
        help="target archs (default: GPU_ARCHS, or every supported arch)",
    )
    parser.add_argument("--list", action="store_true", help="print the jobs and exit")
    args = parser.parse_args()
    if args.arch:
        with override_env("GPU_ARCHS", ";".join(args.arch)):
            jobs = default_jobs(args.max_batch_size)
    else:
        jobs = default_jobs(args.max_batch_size)
    if args.list:
        for job in jobs:
            print(job["kernel_name"])
        print(f"{len(jobs)} jobs")
        return
    if not jobs:
        print("no supported arch selected; nothing to compile")
        return
    results = run_jobs_parallel(compile_one_config, jobs)
    failed = sum(result["compile_time"] is None for result in results)
    print(f"Compiled: {len(results) - failed} ok, {failed} failed")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
