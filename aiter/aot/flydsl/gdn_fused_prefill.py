#!/usr/bin/env python3

# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""AOT pre-compilation for the gfx950 single-kernel FlyDSL GDN prefill.

``gdn_fused_prefill_fwd`` compiles one kernel per (H, Hg, BV, HAS_TRACK,
HAS_INIT_FLAG, scale). BV is picked per batch from N * H and the CU count, and
the two flags follow the optional ``track_state`` / ``has_initial_state``
arguments, so a serving process can reach every combination of a head shape
and would JIT it on the request path. They are enumerated here for each head
shape in ``DEFAULT_SHAPES``, at the default scale only.

Usage:
    python -m aiter.aot.flydsl.gdn_fused_prefill               # DEFAULT_SHAPES
    python -m aiter.aot.flydsl.gdn_fused_prefill --shape 48:16  # ad hoc H:Hg
    python -m aiter.aot.flydsl.gdn_fused_prefill --list
"""

from __future__ import annotations

import argparse
import itertools
import sys
import time

import flydsl.expr as fx

from aiter.aot.flydsl.common import (
    compile_only_env,
    dedupe_jobs,
    override_env,
    run_jobs_parallel,
)
from aiter_worker_limits import adopt_legacy_max_jobs

AOT_ARCH = "gfx950"
# Value/key head counts per rank, (H, Hg), at TP1/2/4/8. Jobs dedupe by kernel
# name, so two models may share a head shape.
DEFAULT_SHAPES = {
    "qwen3_8_flash_next": [(48, 16), (24, 8), (12, 4), (6, 2)],
    "qwen3_5_397b": [(64, 16), (32, 8), (16, 4), (8, 2)],
    "qwen3_5_35b": [(32, 16), (16, 8), (8, 4), (4, 2)],
}
# Every BV ``_pick_bv`` can return.
_BV_CANDIDATES = (16, 32)


def parse_shape(spec: str) -> tuple[int, int]:
    """``"H:Hg"`` -> ``(H, Hg)``."""
    h, hg = (int(x) for x in spec.strip().split(":"))
    return h, hg


def unsupported_reason(h: int, hg: int) -> str | None:
    """Why the kernel cannot serve this head shape, or None."""
    if h <= 0 or hg <= 0 or h % hg:
        return f"H={h} must be a positive multiple of Hg={hg}"
    return None


def jobs_for_shape(h: int, hg: int) -> list[dict]:
    return [
        {
            "kernel_name": (
                f"gdn_fused_prefill_bv{bv}_h{h}_hg{hg}"
                f"_t{int(has_track)}_i{int(has_init_flag)}"
            ),
            "H": h,
            "Hg": hg,
            "BV": bv,
            "has_track": has_track,
            "has_init_flag": has_init_flag,
        }
        for bv, has_track, has_init_flag in itertools.product(
            _BV_CANDIDATES, (False, True), (False, True)
        )
    ]


def default_jobs(shapes: dict[str, list] = DEFAULT_SHAPES) -> list[dict]:
    """Every variant ``DEFAULT_SHAPES`` can reach; unsupported shapes warn."""
    jobs = []
    for model, entries in shapes.items():
        for h, hg in entries:
            reason = unsupported_reason(h, hg)
            if reason:
                print(f"  [WARN] {model}: {h}:{hg} unsupported ({reason}), skipping")
                continue
            jobs.extend(jobs_for_shape(h, hg))
    return dedupe_jobs(jobs)


def _compile_to_cache(*, H, Hg, BV, has_track, has_init_flag):
    import torch

    from aiter.ops.flydsl.gdn_fused_prefill import _KD, _VD
    from aiter.ops.flydsl.kernels.gdr_prefill import compile_gdn_fused_prefill
    from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled

    exe = compile_gdn_fused_prefill(
        H=H,
        Hg=Hg,
        BV=BV,
        HAS_TRACK=has_track,
        HAS_INIT_FLAG=has_init_flag,
        scale=_KD**-0.5,
    )

    # Mirror the launch at the end of gdn_fused_prefill_fwd: flat tensors of
    # the same dtypes in the same slots. Sizes and the int values never reach
    # the cache key; a wrong dtype silently would.
    cpu = torch.device("cpu")

    def t(n, dtype):
        return torch.empty(n, dtype=dtype, device=cpu)

    bf16, f32, i32 = torch.bfloat16, torch.float32, torch.int32
    N, T = 1, 64
    q_stride = 2 * Hg * _KD + H * _VD
    pool = t(N * H * _VD * _KD, bf16)
    idx = t(N, i32)
    # Without tracking the launcher passes the pool and cache indices in the
    # track slots, and the indices again for has_initial_state.
    trs = t(N * H * _VD * _KD, f32) if has_track else pool
    tri = t(N, i32) if has_track else idx
    init = t(N, i32) if has_init_flag else idx
    with compile_only_env():
        _run_compiled(
            exe,
            t(T * q_stride, bf16),  # q
            t(T * q_stride, bf16),  # k
            t(T * q_stride, bf16),  # v
            t(T * H, f32),  # g
            t(T * H, f32),  # beta
            t(N + 1, i32),  # cu_seqlens
            idx,
            pool,
            t(T * H * _VD, bf16),  # o
            trs,
            tri,
            init,
            q_stride,
            q_stride,
            q_stride,
            H,
            H,
            T,
            N * H,
            fx.Stream(0),
        )


def compile_one_config(**job) -> dict:
    result = {**job, "compile_time": None}
    params = {k: v for k, v in job.items() if k != "kernel_name"}
    from torch._subclasses.fake_tensor import FakeTensorMode

    started = time.time()
    try:
        with override_env("FLYDSL_GPU_ARCH", AOT_ARCH), FakeTensorMode():
            _compile_to_cache(**params)
        result["compile_time"] = time.time() - started
    except Exception as error:  # noqa: BLE001
        print(f"  [FAIL] {job['kernel_name']}: {error}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--shape",
        action="append",
        default=[],
        help="H:Hg, repeatable; replaces DEFAULT_SHAPES",
    )
    parser.add_argument("--list", action="store_true", help="print the jobs and exit")
    args = parser.parse_args()
    if args.shape:
        jobs = []
        for spec in args.shape:
            h, hg = parse_shape(spec)
            reason = unsupported_reason(h, hg)
            if reason:
                print(f"Error: {spec} is unsupported: {reason}")
                sys.exit(1)
            jobs.extend(jobs_for_shape(h, hg))
        jobs = dedupe_jobs(jobs)
    else:
        jobs = default_jobs()
    if args.list:
        for job in jobs:
            print(job["kernel_name"])
        print(f"{len(jobs)} jobs")
        return
    results = run_jobs_parallel(compile_one_config, jobs)
    failed = sum(result["compile_time"] is None for result in results)
    print(f"Compiled: {len(results) - failed} ok, {failed} failed")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    adopt_legacy_max_jobs()
    main()
