# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Graph timing: replay K calls recorded into one HIP graph.

Each ``g.replay()`` costs the host a few microseconds (about 6 us on MI350P).
With one call per graph, any kernel shorter than that measures the host's
replay rate instead of device work. Recording K calls per graph, with K chosen
so one replay takes at least ``min_replay_us``, keeps the GPU ahead of the
host; each sample is one replay divided by K.

All samples are enqueued back to back with a single synchronize at the end,
so no sample contains a host-side gap.
"""

import math
import warnings

import torch

from . import TimerOutput


class EmptyGraphCaptureError(RuntimeError):
    def __init__(self):
        super().__init__(
            "graph capture recorded no GPU work. The callable launched its "
            "kernels outside the capture stream (for example on the NULL "
            "stream) or launched nothing. Pass torch.cuda.current_stream() to "
            "the kernel launch (FlyDSL: a 'stream: fx.Stream' launcher "
            "argument) so it can be captured."
        )


class GraphCaptureError(RuntimeError):
    pass


def _capture(calls, k, stream):
    graph = torch.cuda.CUDAGraph()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            with torch.cuda.graph(graph, stream=stream):
                for j in range(k):
                    calls[j % len(calls)]()
        except Exception as error:
            raise GraphCaptureError(
                f"callable is not graph-capturable ({type(error).__name__}: "
                f"{error}). Common causes: host synchronization, .item(), "
                "host-to-device copies or allocations outside the caching "
                "allocator inside the call."
            ) from error
    if any("empty" in str(w.message).lower() for w in caught):
        raise EmptyGraphCaptureError()
    return graph


def _replay_us(graph, replays):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(replays):
        graph.replay()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1000.0 / replays


def _estimate_call_us(calls, probes=5):
    times = []
    for i in range(probes):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        calls[i % len(calls)]()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) * 1000.0)
    return max(sorted(times)[len(times) // 2], 1e-3)


def _calls_for(target_us, call_us, n_copies, max_k):
    """Calls per replay: a whole number of passes over the input copies."""
    passes = math.ceil(target_us / call_us / n_copies)
    passes = max(1, min(passes, max_k // n_copies))
    return passes * n_copies


def time_graph(calls, iters, *, min_replay_us, max_calls) -> TimerOutput:
    notes = []
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    # Warm on the capture stream: lazy initialization and allocator growth are
    # not allowed during capture.
    with torch.cuda.stream(stream):
        for call in calls:
            call()
    torch.cuda.synchronize()

    # Every rotated copy appears equally often per replay, otherwise the
    # replay would favour a subset of the copies.
    n_copies = min(len(calls), max_calls)
    # The eager estimate includes host overhead, so it overestimates small
    # kernels and K comes out too small; one refinement from the measured
    # replay corrects that.
    k = _calls_for(min_replay_us, _estimate_call_us(calls), n_copies, max_calls)
    graph = _capture(calls, k, stream)
    replay_us = _replay_us(graph, 3)
    if replay_us < min_replay_us / 2 and k < max_calls:
        refined = _calls_for(min_replay_us, replay_us / k, n_copies, max_calls)
        if refined > k:
            graph.reset()
            k = refined
            graph = _capture(calls, k, stream)
            replay_us = _replay_us(graph, 3)
    if replay_us < min_replay_us / 2:
        notes.append(
            f"graph replay is only {replay_us:.1f} us with K={k} calls "
            f"(graph_max_calls={max_calls}); per-replay host cost may be "
            "included. Raise graph_max_calls."
        )

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    graph.replay()
    for i in range(iters):
        starts[i].record()
        graph.replay()
        ends[i].record()
    torch.cuda.synchronize()
    samples = [s.elapsed_time(e) * 1000.0 / k for s, e in zip(starts, ends)]
    graph.reset()
    info = {"graph_calls_per_replay": k, "captured_nodes": None}
    return TimerOutput(samples_us=samples, info=info, notes=notes)
