# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Run test_flydsl_grouped_gemm.py's main() unchanged, with two additions.

1. A second, profiler-free kernel time: every gemm1/gemm2 launch callable that the
   test's kernel-bench times with ``run_perftest`` is also captured
   ``MOE_EXP_GRAPH_ITERS`` times into one CUDA graph; the graph is replayed
   ``MOE_EXP_GRAPH_REPS`` times and each replay is timed with CUDA events. Printed
   as ``[moe-exp] <name>: graph_us = <median per launch>``; it includes the gaps
   between back-to-back kernels.
2. ``MOE_EXP_SAVE_OUT=<file.pt>`` saves the fused-MoE output that the test checks
   against its torch reference, for a bit-exact comparison between runs.

usage (from the repository root):
    python op_tests/flydsl_tests/moe_exp/bench_driver.py <test_flydsl_grouped_gemm.py args>
"""

import importlib.util
import os
import statistics
import sys

import torch

from aiter import test_common

HERE = os.path.dirname(os.path.abspath(__file__))
TEST = os.path.join(os.path.dirname(HERE), "test_flydsl_grouped_gemm.py")
GRAPH_ITERS = int(os.environ.get("MOE_EXP_GRAPH_ITERS", "128"))
GRAPH_REPS = int(os.environ.get("MOE_EXP_GRAPH_REPS", "10"))
SAVE_OUT = os.environ.get("MOE_EXP_SAVE_OUT")

_graph_us = []  # per run_perftest call of the kernel-bench loop, in call order


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
    result = _orig_run_perftest(func, *args, **kwargs)
    # The kernel-bench loop passes a bare launch callable with testGraph=False.
    if not args and not kwargs.get("testGraph", False) and GRAPH_ITERS > 0:
        try:
            _graph_us.append(graph_time_us(func, GRAPH_ITERS, GRAPH_REPS))
        except RuntimeError as exc:  # report and keep the test's own result
            print(f"[moe-exp] graph timing failed: {exc!r}", flush=True)
            _graph_us.append(None)
    return result


test_common.run_perftest = run_perftest

sys.path.insert(0, os.path.dirname(TEST))
spec = importlib.util.spec_from_file_location("test_flydsl_grouped_gemm", TEST)
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)
_orig_run = mod._run_grouped_via_fused_moe


def _run_grouped_via_fused_moe(*args, **kwargs):
    first = len(_graph_us)
    out, ref, us, kernel_us = _orig_run(*args, **kwargs)
    # kernel_us is filled in the same order as the run_perftest calls above.
    for name, timed in zip(kernel_us or {}, _graph_us[first:]):
        if timed is not None:
            median, times = timed
            spread = f"{min(times):.2f}-{max(times):.2f}"
            print(
                f"[moe-exp] {name}: graph_us = {median:.2f} "
                f"(median of {len(times)} replays x {GRAPH_ITERS} launches, "
                f"range {spread})",
                flush=True,
            )
    if SAVE_OUT:
        torch.save(out.detach().cpu(), SAVE_OUT)
        print(f"[moe-exp] saved output {SAVE_OUT} {tuple(out.shape)}", flush=True)
    return out, ref, us, kernel_us


mod._run_grouped_via_fused_moe = _run_grouped_via_fused_moe
sys.argv = [TEST] + sys.argv[1:]
mod.main()
