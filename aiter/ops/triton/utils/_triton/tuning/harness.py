# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Worker: plans the search space (--plan) or benchmarks candidate configs on one GPU.

Flow: set the kernel up once (inputs, call, should_skip), then
  --plan     build the space (space.build_space) and write it as JSON for the driver
  --configs  for each candidate: warm up (compiles) -> capture a CUDA graph over cold input
             copies -> replay -> append one record to --out

--configs is a JSON list of config dicts; null is the baseline, the wrapper called with
config=None (the installed config). Records are appended as soon as a candidate finishes.

Record fields:
  status    ready (worker is set up), start (candidate began), ok, error
  baseline  true for the installed config
  config    the raw candidate, never the mutated copy; for the baseline the installed config of
            the tree the sweep installs into (the default tree, or with --cu-dir the <cu_num>cu/
            tree), launched through the wrapper's own config=None path unless that path reads
            another tree on this device or keys DEFAULT.json does not have had to be stripped
            first (dropped_keys lists them)
  gpu       HIP_VISIBLE_DEVICES of the worker
  cu_num    CU count of the worker's device (get_config_cu_num()); null when it cannot be read
  is_tuned  baseline only: a specialized file served this shape
  us, us_min, us_max, tflops   per-launch time from graph replay (median, min, max)
  error     the exception text when status is error
  dur_s     wall time of the candidate
"""

import argparse
import copy
import gc
import json
import os
import sys
import time

from _utils import add_shape_args, append_record, shape_from_args
from kernels import (
    check_backend,
    ensure_repo_on_path,
    family_bounds,
    flops,
    get_spec,
    installed_baseline,
)
from space import UnknownConfigKey, build_space, load_defaults


def make_cold_copies(inputs, cold_mb):
    """copies[0] is the original input tuple; the rest are clones, within the byte budget."""
    import torch

    def clone(a):
        if not isinstance(a, torch.Tensor):
            return a
        return torch.empty_strided(
            a.shape, a.stride(), dtype=a.dtype, device=a.device
        ).copy_(a)

    per_copy = sum(
        a.numel() * a.element_size() for a in inputs if isinstance(a, torch.Tensor)
    )
    budget = min(cold_mb << 20, torch.cuda.mem_get_info()[0] // 4)
    n = max(1, min(64, budget // max(per_copy, 1)))
    return [tuple(inputs)] + [tuple(clone(a) for a in inputs) for _ in range(n - 1)]


def time_with_cuda_graph(call, config, copies, calls=24, replays=25):
    """Median/min/max microseconds per launch: one eager warm-up (compiles), then graph replay.

    Every launch gets a deep copy of config because the wrappers mutate it.
    """
    import torch

    stream = torch.cuda.Stream()
    n_calls = max(calls, len(copies))
    with torch.cuda.stream(stream):
        call(copy.deepcopy(config), *copies[0])
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        gc.disable()
        try:
            with torch.cuda.graph(graph, stream=stream):
                for i in range(n_calls):
                    call(copy.deepcopy(config), *copies[i % len(copies)])
        finally:
            gc.enable()
    torch.cuda.synchronize()
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    times = []
    for _ in range(replays):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
            enable_timing=True
        )
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) * 1e3 / n_calls)
    times.sort()
    return times[len(times) // 2], times[0], times[-1]


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


def benchmark(
    spec,
    shape,
    backend,
    arch,
    call,
    inputs,
    candidates,
    out,
    calls,
    replays,
    cold_mb,
    cu_num=None,
):
    from aiter.ops.triton.utils.config_utils import get_config_cu_num

    keys, _, _ = load_defaults(spec, backend, cu_num)
    copies = make_cold_copies(inputs, cold_mb)
    common = {
        "kernel": spec.name,
        "backend": backend,
        "arch": arch,
        "shape": shape,
        "gpu": os.environ.get("HIP_VISIBLE_DEVICES"),
        "cu_num": get_config_cu_num(),
    }
    append_record(
        out,
        dict(
            common,
            status="ready",
            n_copies=len(copies),
            calls=calls,
            replays=replays,
            cold_mb=cold_mb,
        ),
    )
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
            median, low, high = time_with_cuda_graph(
                call, candidate, copies, calls, replays
            )
            record.update(
                us=round(median, 3),
                us_min=round(low, 3),
                us_max=round(high, 3),
                tflops=round(flops(shape) / median / 1e6, 2),
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
    parser.add_argument("--replays", type=int, default=25)
    parser.add_argument(
        "--calls", type=int, default=24, help="launches captured per graph"
    )
    parser.add_argument(
        "--cold-mb", type=int, default=1024, help="byte budget for cold input copies"
    )
    args = parser.parse_args(argv)

    spec = args.kernel
    shape = shape_from_args(args, spec.dims)
    ensure_repo_on_path()
    from aiter.ops.triton.utils._triton import arch_info

    arch = arch_info.get_arch()
    backend = args.backend or spec.default_backend(arch)
    check_backend(spec, arch, backend)
    call, inputs, should_skip = spec.setup(shape, backend)
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
        call,
        inputs,
        candidates,
        args.out,
        args.calls,
        args.replays,
        args.cold_mb,
        cu_num,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
