#!/usr/bin/env python3

# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""AOT pre-compilation for the FlyDSL paged-attention decode planner.

``plan_pa_decode`` compiles one kernel per batch bucket, block size and
dense/windowed mode; batch, capacity, partition cap and window are runtime
values, and ``max_batch_size`` only bounds the batch. The kernels therefore
form a finite set per arch. This module replays ``plan_pa_decode`` itself on
FakeTensors inside :func:`aiter.aot.flydsl.dry_run.dry_run` (see
``replay.py``): for each bucket, at its largest batch, and at each task
capacity where the block size can change, dense and windowed.

Buckets are 1, 2, 4, ... clipped to the arch's LDS limit (16256 on gfx942,
40832 on gfx950, 81792 on gfx1250), and a bucket's largest batch reaches every
block size its smaller ones do. The block size follows the capacity up to 1024
threads, and the capacity stops at the batch times the CU count (the partition
cap's bound), so each arch replays with the most CUs any of its devices has.

Usage:
    python -m aiter.aot.flydsl.pa_decode_plan
    GPU_ARCHS=gfx950 python -m aiter.aot.flydsl.pa_decode_plan
    python -m aiter.aot.flydsl.pa_decode_plan --check
"""

from __future__ import annotations

import argparse

from aiter.aot.flydsl import replay as _replay

LABEL = "PA_DECODE_PLAN"
# No tuned table: the jobs come from the planner's bucket and block-size steps.
DEFAULT_CSVS: list[str] = []
# Archs that pa_decode dispatches to, with the most CUs of any of their devices.
AOT_ARCHS = {"gfx942": 304, "gfx950": 256, "gfx1250": 256}
# Capacities at which the block size can change (64 to 1024 threads).
_BLOCK_STEPS = (64, 128, 256, 512, 1024)


def jobs_for_arch(arch: str) -> list[dict]:
    """A job per bucket, block-size step and window mode of ``arch``."""
    from aiter.jit.utils.chip_info import get_lds_capacity_bytes
    from aiter.ops.flydsl.kernels.pa_decode_plan import _lds_max_batch

    cu_num = AOT_ARCHS[arch]
    max_batch = _lds_max_batch(get_lds_capacity_bytes(arch))
    batches = [1 << bit for bit in range((max_batch - 1).bit_length())]
    batches.append(max_batch)
    jobs = []
    for batch in batches:
        capacities = {min(max(step, batch), batch * cu_num) for step in _BLOCK_STEPS}
        for capacity in sorted(capacities):
            for windowed in (False, True):
                jobs.append(
                    {
                        "kernel_name": (
                            f"pa_decode_plan_b{batch}_c{capacity}"
                            f"_{'window' if windowed else 'dense'}"
                        ),
                        "gfx": arch,
                        "cu_num": cu_num,
                        "batch": batch,
                        "capacity": capacity,
                        "windowed": windowed,
                    }
                )
    return jobs


def default_jobs() -> list[dict]:
    """Every job of each arch; ``replay.run`` drops archs ``GPU_ARCHS`` omits."""
    return [job for arch in AOT_ARCHS for job in jobs_for_arch(arch)]


def replay(job: dict) -> None:
    import torch

    from aiter.ops.flydsl.kernels.pa_decode_plan import plan_pa_decode

    batch, capacity = job["batch"], job["capacity"]
    # One KV head: the budget is the capacity, which the partition cap admits.
    plan_pa_decode(
        torch.empty(batch, dtype=torch.int32, device="cuda"),
        1,
        max_partitions=-(-capacity // batch),
        max_batch_size=batch,
        workgroup_budget=capacity,
        sliding_window=int(job["windowed"]),
    )


def describe(job: dict) -> str:
    return f"{job['kernel_name']} cu_num={job['cu_num']}"


def collect_jobs(_csv_paths: list[str], _args: argparse.Namespace | None) -> list:
    return default_jobs()


def main() -> None:
    _replay.main(
        "aiter.aot.flydsl.pa_decode_plan",
        "AOT pre-compile the FlyDSL paged-attention decode planner",
    )


if __name__ == "__main__":
    main()
