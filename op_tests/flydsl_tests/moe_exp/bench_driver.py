# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Run test_flydsl_grouped_gemm.py's main() unchanged, with three additions.

1. ``[moe-exp] <name>: profiler_us = ...`` for every launch callable the test's
   kernel-bench times (quant_a1, gemm1, gemm2). These are the values the test
   computes with ``run_perftest``; the test itself prints only gemm1 and gemm2.
2. A second, profiler-free kernel time: after the test's kernel-bench has timed
   all callables, each callable is captured ``MOE_EXP_GRAPH_ITERS`` times into
   one CUDA graph, the graph is replayed ``MOE_EXP_GRAPH_REPS`` times and each
   replay is timed with CUDA events. Printed as ``[moe-exp] <name>: graph_us =
   <median per launch>``; it includes the gaps between back-to-back kernels.
   The GPU idles ``MOE_EXP_GRAPH_GAP_S`` seconds (default 1) before each
   callable's graph timing, so one kernel's back-to-back run does not heat the
   GPU for the next kernel's timing. The graph timing runs only after all of
   the test's own timing, so it cannot change the test's numbers.
3. ``MOE_EXP_SAVE_OUT=<file.pt>`` saves the fused-MoE output that the test checks
   against its torch reference, for a bit-exact comparison between runs.

usage (from the repository root):
    python op_tests/flydsl_tests/moe_exp/bench_driver.py <test_flydsl_grouped_gemm.py args>
"""

import importlib.util
import os
import statistics
import sys
import time

import torch

from aiter import test_common

HERE = os.path.dirname(os.path.abspath(__file__))
TEST = os.path.join(os.path.dirname(HERE), "test_flydsl_grouped_gemm.py")
GRAPH_ITERS = int(os.environ.get("MOE_EXP_GRAPH_ITERS", "128"))
GRAPH_REPS = int(os.environ.get("MOE_EXP_GRAPH_REPS", "10"))
GRAPH_GAP_S = float(os.environ.get("MOE_EXP_GRAPH_GAP_S", "1"))
SAVE_OUT = os.environ.get("MOE_EXP_SAVE_OUT")

_timed = []  # launch callables of the kernel-bench loop, in run_perftest call order


def graph_time_us(fn, iters, reps):
    """Median per-launch time of ``iters`` launches of ``fn`` in one CUDA graph."""
    torch.cuda.synchronize()
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(2):
            fn()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(iters):
            fn()
    graph.replay()
    torch.cuda.synchronize()
    times = []
    for _ in range(reps):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) * 1000.0 / iters)
    del graph
    torch.cuda.synchronize()
    return statistics.median(times), times


_orig_run_perftest = test_common.run_perftest


def run_perftest(func, *args, **kwargs):
    # The kernel-bench loop passes a bare launch callable with testGraph=False.
    if not args and not kwargs.get("testGraph", False):
        _timed.append(func)
    return _orig_run_perftest(func, *args, **kwargs)


test_common.run_perftest = run_perftest

sys.path.insert(0, os.path.dirname(TEST))
spec = importlib.util.spec_from_file_location("test_flydsl_grouped_gemm", TEST)
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)
_orig_run = mod._run_grouped_via_fused_moe


def _run_grouped_via_fused_moe(*args, **kwargs):
    first = len(_timed)
    out, ref, us, kernel_us = _orig_run(*args, **kwargs)
    funcs = _timed[first:]
    names = list(kernel_us or {})
    # kernel_us is filled in the same order as the run_perftest calls above.
    if len(names) != len(funcs):
        print(
            f"[moe-exp] {len(funcs)} timed callables but kernel_us has {names}; "
            "labelling graph times by index",
            flush=True,
        )
        names = [f"callable{i}" for i in range(len(funcs))]
    for name in names:
        if name in (kernel_us or {}):
            v = float(kernel_us[name])
            # run_perftest returns 0 when the profiler trace has no events for the callable's kernels
            text = f"{v:.2f}" if v > 0 else "n/a (no kernel events in the profiler trace)"
            print(f"[moe-exp] {name}: profiler_us = {text}", flush=True)
    if GRAPH_ITERS > 0:
        for name, func in zip(names, funcs):
            torch.cuda.synchronize()
            time.sleep(GRAPH_GAP_S)
            try:
                median, times = graph_time_us(func, GRAPH_ITERS, GRAPH_REPS)
            except RuntimeError as exc:  # report and keep the test's own result
                print(f"[moe-exp] {name}: graph timing failed: {exc!r}", flush=True)
                continue
            print(
                f"[moe-exp] {name}: graph_us = {median:.2f} "
                f"(median of {len(times)} replays x {GRAPH_ITERS} launches, "
                f"range {min(times):.2f}-{max(times):.2f}, after {GRAPH_GAP_S:g} s idle)",
                flush=True,
            )
    if SAVE_OUT:
        torch.save(out.detach().cpu(), SAVE_OUT)
        print(f"[moe-exp] saved output {SAVE_OUT} {tuple(out.shape)}", flush=True)
    return out, ref, us, kernel_us


mod._run_grouped_via_fused_moe = _run_grouped_via_fused_moe
sys.argv = [TEST] + sys.argv[1:]
mod.main()
