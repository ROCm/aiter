# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Device timing: hardware kernel timestamps from torch.profiler.

Each timed call runs inside a ``record_function`` range. A GPU operation
(kernel, memcpy, memset) belongs to call ``i`` when the runtime call that
launched it -- linked by correlation id -- was issued inside range ``i``. No
kernel-name or repeating-pattern guess is involved, so repeated kernels and a
data-dependent kernel count are handled. Work launched outside the ranges
(the cache flush, profiler internals) is ignored.

Each sample is the busy time of one call: the sum of its GPU operations'
durations. ``span_us`` is the median first-start to last-end time per call;
``span_us - median busy`` is time spent in gaps between the call's kernels.
"""

import bisect
import statistics
from collections import defaultdict

import torch
from torch.profiler import ProfilerActivity, profile, record_function

from . import TimerOutput

_MARKER = "__aiter_bench_call_"
# Calls profiled but discarded: the profiler's first ranges are slow to start.
PROFILER_WARMUP = 2
_RUNTIME_ACTIVITY = ("cuda_runtime", "cuda_driver", "privateuse1_runtime")
_ANNOTATION_ACTIVITY = ("gpu_user_annotation",)


def _activity(event):
    return str(event.activity_type())


def _is_device(event):
    return event.device_type() != torch.autograd.DeviceType.CPU


def _split_calls(events, n_calls):
    """Assign device operations to calls. Returns (per_call, unmatched)."""
    ranges = []  # (start_ns, end_ns, call index)
    runtime_start = {}  # correlation id -> launch time
    device_ops = []
    for e in events:
        activity = _activity(e)
        if _is_device(e):
            if activity not in _ANNOTATION_ACTIVITY:
                device_ops.append(e)
        elif activity in _RUNTIME_ACTIVITY:
            runtime_start[e.correlation_id()] = e.start_ns()
        elif e.name().startswith(_MARKER):
            ranges.append((e.start_ns(), e.end_ns(), int(e.name()[len(_MARKER) :])))
    ranges.sort()
    starts = [r[0] for r in ranges]

    per_call = [[] for _ in range(n_calls)]
    unmatched = 0
    for op in device_ops:
        launched = runtime_start.get(op.correlation_id())
        if launched is None:
            unmatched += 1
            continue
        pos = bisect.bisect_right(starts, launched) - 1
        if pos >= 0 and launched <= ranges[pos][1]:
            per_call[ranges[pos][2]].append(op)
    return per_call, unmatched


def time_device(calls, iters, *, before_call=None) -> TimerOutput:
    n_calls = PROFILER_WARMUP + iters
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for i in range(n_calls):
            if before_call is not None:
                before_call()
            with record_function(f"{_MARKER}{i}"):
                calls[i % len(calls)]()
        torch.cuda.synchronize()

    per_call, unmatched = _split_calls(prof.profiler.kineto_results.events(), n_calls)
    per_call = per_call[PROFILER_WARMUP:]
    empty = sum(1 for ops in per_call if not ops)
    if empty:
        raise RuntimeError(
            f"device timing found no GPU work for {empty} of {iters} calls. The "
            "callable launched nothing, or the profiler could not link its "
            "kernels to their launches."
        )

    samples, spans, counts = [], [], []
    per_kernel = defaultdict(list)
    for ops in per_call:
        samples.append(sum(op.duration_ns() for op in ops) / 1000.0)
        spans.append(
            (max(op.end_ns() for op in ops) - min(op.start_ns() for op in ops)) / 1000.0
        )
        counts.append(len(ops))
        by_name = defaultdict(float)
        for op in ops:
            by_name[op.name()] += op.duration_ns() / 1000.0
        for name, us in by_name.items():
            per_kernel[name].append(us)

    notes = []
    if min(counts) != max(counts):
        notes.append(
            f"kernel count per call varies from {min(counts)} to {max(counts)}"
        )
    if unmatched:
        notes.append(
            f"{unmatched} GPU operations could not be linked to a launch and "
            "were not counted"
        )
    info = {
        # A kernel missing from some calls contributes its median over the
        # calls where it ran.
        "per_kernel_us": {name: statistics.median(v) for name, v in per_kernel.items()},
        "span_us": statistics.median(spans),
        "kernel_count": statistics.median_low(counts),
        "all_kernels_counted": unmatched == 0,
    }
    return TimerOutput(samples_us=samples, info=info, notes=notes)
