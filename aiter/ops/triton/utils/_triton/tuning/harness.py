# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Worker: plans the search space (--plan) or benchmarks candidate configs on one GPU.

Flow: set the kernel up once (inputs, call, should_skip), then
  --plan     build the space (space.build_space) and write it as JSON for the driver
  --configs  for each candidate: compile without launching -> skip it when the compiled
             kernels overflow LDS or spill registers -> warm up -> capture a CUDA graph over
             cold input copies -> replay -> append one record to --out

--configs is a JSON list of config dicts; null is the baseline, the wrapper called with
config=None (the installed config). Records are appended as soon as a candidate finishes.

Record fields:
  status    ready (worker is set up), reference (us: the installed config on this GPU, timed
            before the candidates), start (candidate began), ok, error, skipped (the compiled
            kernels overflow LDS or spill registers; reason says which)
  baseline  true for the installed config
  config    the raw candidate, never the mutated copy; for the baseline the installed config
            (launched through the wrapper's own config=None path unless keys DEFAULT.json does
            not have had to be stripped first; dropped_keys lists them)
  gpu       HIP_VISIBLE_DEVICES of the worker
  is_tuned  baseline only: a specialized file served this shape
  us, us_min, us_max, tflops   per-launch time from graph replay (median, min, max)
  rel       us / the reference (the installed config on the same GPU): GPUs differ by a few
            percent, so screening and the final-round shortlist compare rel, not us
  screened  true when one replay was already above --screen x the lowest rel of this M (the
            reference is 1): the candidate stopped there and us is that replay
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
from collections import namedtuple

from _utils import add_shape_args, append_record, read_records, shape_from_args
from kernels import (
    check_backend,
    ensure_repo_on_path,
    family_bounds,
    flops,
    get_spec,
    resolve_installed,
)
from space import (
    UnknownConfigKey,
    build_space,
    exceeds_compiled_resources,
    load_defaults,
)


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


Timing = namedtuple("Timing", "median low high screened")


def time_with_cuda_graph(call, config, copies, calls=24, replays=25, abort_above=None):
    """Median/min/max microseconds per launch: one eager warm-up (compiles), then graph replay.

    Every launch gets a deep copy of config because the wrappers mutate it. With abort_above
    (microseconds per launch), one replay after the first warm-up replay decides: a candidate
    slower than that stops there (screened), so a far-off candidate costs a few launches. A
    candidate whose eager warm-up alone takes over 1 ms and 1.25x abort_above stops right
    there: at that length the launch overhead in the eager time does not matter, and the
    margin covers the cold caches of a first launch.
    """
    import torch

    stream = torch.cuda.Stream()
    n_calls = max(calls, len(copies))
    with torch.cuda.stream(stream):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
            enable_timing=True
        )
        start.record()
        call(copy.deepcopy(config), *copies[0])
        end.record()
        torch.cuda.synchronize()
        eager = start.elapsed_time(end) * 1e3
        if abort_above is not None and eager > max(1.25 * abort_above, 1000):
            return Timing(eager, eager, eager, True)
        graph = torch.cuda.CUDAGraph()
        gc.disable()
        try:
            with torch.cuda.graph(graph, stream=stream):
                for i in range(n_calls):
                    call(copy.deepcopy(config), *copies[i % len(copies)])
        finally:
            gc.enable()
    torch.cuda.synchronize()

    def timed_replay():
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
            enable_timing=True
        )
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        return start.elapsed_time(end) * 1e3 / n_calls

    graph.replay()
    if abort_above is not None:
        first = timed_replay()
        if first > abort_above:
            return Timing(first, first, first, True)
    for _ in range(2):
        graph.replay()
    torch.cuda.synchronize()
    times = sorted(timed_replay() for _ in range(replays))
    return Timing(times[len(times) // 2], times[0], times[-1], False)


def compile_only(call, config, inputs):
    """Run the wrapper with every kernel launch turned into a compile; returns the compiled kernels."""
    from triton.runtime import jit

    compiled = []
    launch = jit.JITFunction.run

    def compile_instead(self, *args, grid, warmup, **kwargs):
        kernel = launch(self, *args, grid=grid, warmup=True, **kwargs)
        compiled.append(kernel)
        return kernel

    jit.JITFunction.run = compile_instead
    try:
        call(copy.deepcopy(config), *inputs)
    finally:
        jit.JITFunction.run = launch
    return [kernel for kernel in compiled if kernel is not None]


def gpu_alive():
    import torch

    try:
        torch.cuda.synchronize()
        (torch.ones(1, device="cuda") + 1).item()
        return True
    except Exception:  # noqa: BLE001
        return False


def plan(spec, shape, backend, arch, should_skip, space_out):
    try:
        configs, report = build_space(spec, shape, backend, arch, should_skip)
    except UnknownConfigKey as e:
        print(e, file=sys.stderr)
        return 2
    plan_ = dict(
        report,
        arch=arch,
        backend=backend,
        kernel=spec.name,
        shape=shape,
        bounds=list(family_bounds(spec, backend, shape)),
        configs=configs,
    )
    with open(space_out, "w") as f:
        json.dump(plan_, f)
    return 0


def fastest_rel(out):
    """Lowest rel of the fully timed ok records of this results file (all GPUs), or None."""
    rels = [
        r["rel"]
        for r in read_records(out).values()
        if r["status"] == "ok" and not r.get("screened") and "rel" in r
    ]
    return min(rels, default=None)


def reference_time(spec, shape, backend, keys, call, copies, calls, replays):
    """This GPU's time for the installed config, or None when it does not run."""
    installed, _ = resolve_installed(spec, shape, backend)
    config = None
    if any(k not in keys for k in installed):  # launch it as the baseline record does
        config = {k: v for k, v in installed.items() if k in keys}
    try:
        return time_with_cuda_graph(call, config, copies, calls, replays).median
    except Exception:  # noqa: BLE001
        return None


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
    screen,
):
    keys, _, _ = load_defaults(spec, backend)
    copies = make_cold_copies(inputs, cold_mb)
    common = {
        "kernel": spec.name,
        "backend": backend,
        "arch": arch,
        "shape": shape,
        "gpu": os.environ.get("HIP_VISIBLE_DEVICES"),
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
    reference = reference_time(spec, shape, backend, keys, call, copies, calls, replays)
    append_record(out, dict(common, status="reference", us=reference))
    # the best rel so far: the installed config is 1, and rel makes every GPU's records count
    best_rel = min(fastest_rel(out) or 1.0, 1.0) if screen and reference else None
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
                installed, record["is_tuned"] = resolve_installed(spec, shape, backend)
                dropped = sorted(k for k in installed if k not in keys)
                record["config"] = {k: v for k, v in installed.items() if k in keys}
                if dropped:  # keys DEFAULT.json does not have are never launched
                    record["dropped_keys"] = dropped
                    candidate = record["config"]
                # otherwise candidate stays None: the wrapper's own installed-config path
            else:
                # the baseline is always timed; a candidate only when its kernels fit
                reason = exceeds_compiled_resources(
                    compile_only(call, candidate, copies[0])
                )
                if reason:
                    record.update(status="skipped", reason=reason)
            if record["status"] == "ok":
                # the baseline is never screened
                abort_above = None
                if best_rel and not is_baseline:
                    abort_above = screen * best_rel * reference  # in this GPU's us
                timing = time_with_cuda_graph(
                    call, candidate, copies, calls, replays, abort_above
                )
                record.update(
                    us=round(timing.median, 3),
                    us_min=round(timing.low, 3),
                    us_max=round(timing.high, 3),
                    tflops=round(flops(shape) / timing.median / 1e6, 2),
                )
                if reference:
                    record["rel"] = round(timing.median / reference, 4)
                if timing.screened:
                    record["screened"] = True
                elif best_rel:
                    best_rel = min(best_rel, record["rel"])
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
    parser.add_argument("--configs", help="JSON list of candidates, null = baseline")
    parser.add_argument("--out", help="JSONL results file, appended")
    parser.add_argument("--replays", type=int, default=25)
    parser.add_argument(
        "--calls", type=int, default=24, help="launches captured per graph"
    )
    parser.add_argument(
        "--cold-mb", type=int, default=1024, help="byte budget for cold input copies"
    )
    parser.add_argument(
        "--screen",
        type=float,
        default=0,
        help="stop a candidate after one replay when its rel is above this x the lowest rel "
        "so far (0: off)",
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

    if args.plan:
        return plan(spec, shape, backend, arch, should_skip, args.space_out)
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
        args.screen,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
