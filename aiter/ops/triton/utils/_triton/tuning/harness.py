# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Worker: plans the search space (--plan) or benchmarks candidate configs on one GPU.

The kernel's benchmark (op_tests/op_benchmarks/triton/bench_<op>.py, make_bench_case) makes the
inputs and the launch, and RotatingBufferBench (op_tests/op_benchmarks/triton/utils/
rotating_buffer_bench.py) times it. This file only feeds it the candidates and writes the
records.

Flow: get the kernel's (make_case, should_skip), then
  --plan     build the space (space.build_space) and write it as JSON for the driver
  --configs  build the rotating-buffer pool once (the RotatingBufferBench defaults), then time
             each candidate with TUNE_SCREEN, or with --confirm TUNE_CONFIRM -> append one
             record to --out

--configs is a JSON list of config dicts; null is the baseline, the wrapper called with
config=None (the installed config). Records are appended as soon as a candidate finishes.

Record fields:
  status    ready (worker is set up), start (candidate began), ok, error
  baseline  true for the installed config; every GPU of a sweep times it once
  config    the raw candidate, never the mutated copy; for the baseline the installed config of
            the tree the sweep installs into (the default tree, or with --cu-dir the <cu_num>cu/
            tree), launched through the wrapper's own config=None path unless that path reads
            another tree on this device or keys DEFAULT.json does not have had to be stripped
            first (dropped_keys lists them)
  gpu       HIP_VISIBLE_DEVICES of the worker; the baseline record of each GPU is keyed
            baseline@<gpu> (_utils.candidate_key)
  cu_num    CU count of the worker's device (get_config_cu_num()); null when it cannot be read
  is_tuned  baseline only: a specialized file served this shape
  us, us_min, us_max, tflops   per-launch time over the timed graph replays (median, min, max)
  error     the exception text when status is error
  dur_s     wall time of the candidate
The ready record carries the pool and timing settings (pool, timing, n_bufs); a results file is
only resumed with the same settings.
"""

import argparse
import functools
import json
import os
import statistics
import sys
import time

from _utils import add_shape_args, append_record, iter_records, shape_from_args
from kernels import (
    check_backend,
    ensure_repo_on_path,
    family_bounds,
    get_spec,
    installed_baseline,
)
from space import UnknownConfigKey, build_space, load_defaults


def gpu_alive():
    import torch

    try:
        torch.cuda.synchronize()
        (torch.ones(1, device="cuda", dtype=torch.float32) + 1).item()
        return True
    except Exception:  # noqa: BLE001
        return False


def plan(spec, shape, backend, arch, should_skip, space_out, cu_num=None):
    """cu_num: the --cu-dir scope; the plan also records the device's CU count."""
    from aiter.ops.triton.utils.config_utils import get_config_cu_num

    try:
        configs, report = build_space(spec, shape, backend, arch, should_skip, cu_num)
    except UnknownConfigKey as e:
        print(e, file=sys.stderr)
        return 2
    plan_ = dict(
        report,
        arch=arch,
        backend=backend,
        kernel=spec.name,
        shape=shape,
        bounds=list(family_bounds(spec, backend, shape, cu_num)),
        configs=configs,
        cu_num=get_config_cu_num(),
        cu_dir=cu_num is not None,
    )
    with open(space_out, "w") as f:
        json.dump(plan_, f)
    return 0


def check_settings(out, settings):
    """Records can only be resumed when they were timed the same way."""
    ready = next((r for r in iter_records(out) if r["status"] == "ready"), None)
    if ready is not None and {k: ready.get(k) for k in settings} != settings:
        sys.exit(
            f"{out} was timed with pool={ready.get('pool')} timing={ready.get('timing')}; "
            "rerun with --fresh"
        )


def benchmark(spec, shape, backend, arch, make_case, candidates, out, confirm, cu_num):
    from aiter.ops.triton.utils.config_utils import get_config_cu_num
    from op_tests.op_benchmarks.triton.utils.rotating_buffer_bench import (
        TUNE_CONFIRM,
        TUNE_SCREEN,
        RotatingBufferBench,
    )

    bencher = RotatingBufferBench()
    timing = TUNE_CONFIRM if confirm else TUNE_SCREEN
    pool = {
        "target_rot_pool_sz_mb": bencher.target_rot_pool_sz_mb,
        "min_pool_len": bencher.min_pool_len,
        "max_pool_len": bencher.max_pool_len,
    }
    settings = {"pool": pool, "timing": timing}
    check_settings(out, settings)
    keys, _, _ = load_defaults(spec, backend, cu_num)
    launch, rot_bufs, flops, _ = make_case(bencher)
    common = {
        "kernel": spec.name,
        "backend": backend,
        "arch": arch,
        "shape": shape,
        "gpu": os.environ.get("HIP_VISIBLE_DEVICES"),
        "cu_num": get_config_cu_num(),
    }
    append_record(out, dict(common, status="ready", n_bufs=len(rot_bufs), **settings))
    print("ready", flush=True)  # the driver's watchdog reads the worker's log
    for candidate in candidates:
        is_baseline = candidate is None
        append_record(
            out, dict(common, status="start", baseline=is_baseline, config=candidate)
        )
        record = dict(common, status="ok", baseline=is_baseline, config=candidate)
        t0 = time.time()
        try:
            if is_baseline:
                record["config"], record["is_tuned"], dropped, explicit = (
                    installed_baseline(spec, shape, backend, keys, cu_num)
                )
                if dropped:  # keys DEFAULT.json does not have are never launched
                    record["dropped_keys"] = dropped
                if explicit:
                    candidate = record["config"]
                # otherwise candidate stays None: the wrapper's own installed-config path
            times_ms = bencher.bench_times(
                functools.partial(launch, candidate), **timing
            )
            median = statistics.median(times_ms) * 1e3
            record.update(
                us=round(median, 3),
                us_min=round(min(times_ms) * 1e3, 3),
                us_max=round(max(times_ms) * 1e3, 3),
                tflops=round(flops / median / 1e6, 2),
            )
        except Exception as e:  # noqa: BLE001
            # a failing candidate is a result, not a crash
            record.update(
                status="error",
                error=f"{type(e).__name__}: {' '.join(str(e).split())[:300]}",
            )
        record["dur_s"] = round(time.time() - t0, 1)
        append_record(out, record)
        print(record["status"], json.dumps(record["config"]), flush=True)
        if record["status"] == "error" and not gpu_alive():
            print(
                f"GPU unusable after: {record['error']}; exiting for a clean restart",
                file=sys.stderr,
            )
            os._exit(3)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("kernel", type=get_spec)
    add_shape_args(parser)
    parser.add_argument("--backend", choices=("triton", "gluon"))
    parser.add_argument(
        "--plan", action="store_true", help="only build the search space"
    )
    parser.add_argument("--space-out", help="where --plan writes the space (JSON)")
    parser.add_argument(
        "--cu-dir",
        action="store_true",
        help="plan from and time the baseline of the <arch>/<cu_num>cu/ tree of this device",
    )
    parser.add_argument("--configs", help="JSON list of candidates, null = baseline")
    parser.add_argument("--out", help="JSONL results file, appended")
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="final-round timing (TUNE_CONFIRM)",
    )
    args = parser.parse_args(argv)

    spec = args.kernel
    shape = shape_from_args(args, spec.dims)
    ensure_repo_on_path()
    from aiter.ops.triton.utils._triton import arch_info

    arch = arch_info.get_arch()
    backend = args.backend or spec.default_backend(arch)
    check_backend(spec, arch, backend)
    make_case, should_skip = spec.setup(shape, backend)
    cu_num = None
    if args.cu_dir:
        from aiter.ops.triton.utils.config_utils import get_config_cu_num

        cu_num = get_config_cu_num()
        if cu_num is None:
            sys.exit("--cu-dir: the CU count of this device cannot be read")

    if args.plan:
        return plan(spec, shape, backend, arch, should_skip, args.space_out, cu_num)
    with open(args.configs) as f:
        candidates = json.load(f)
    benchmark(
        spec,
        shape,
        backend,
        arch,
        make_case,
        candidates,
        args.out,
        args.confirm,
        cu_num,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
