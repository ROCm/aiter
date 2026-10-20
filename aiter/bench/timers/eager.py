# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Eager timing: one GPU event pair around each call.

Default is the triton.testing.do_bench pattern: events around every call, one
synchronize at the end, so the host runs ahead and per-launch dispatch cost is
amortized the way a steady non-graph serving loop amortizes it. With
``sync_each`` every call is drained before the next one, so each sample also
contains the host launch round trip.
"""

import torch

from . import TimerOutput


def time_eager(calls, iters, *, before_call=None, sync_each=False) -> TimerOutput:
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    torch.cuda.synchronize()
    for i in range(iters):
        if before_call is not None:
            # Issued before the start event, so it is never inside a sample.
            before_call()
        starts[i].record()
        calls[i % len(calls)]()
        ends[i].record()
        if sync_each:
            torch.cuda.synchronize()
    torch.cuda.synchronize()
    samples = [s.elapsed_time(e) * 1000.0 for s, e in zip(starts, ends)]
    return TimerOutput(samples_us=samples, info={"eager_sync_each": sync_each})
