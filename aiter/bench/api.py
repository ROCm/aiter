# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Public entry points: ``time_fn``, ``run_bench`` and ``time_all``."""

import statistics
import time

import torch

from .cache import flush_cache, input_bytes, make_rotation
from .config import MODES, MeasureConfig
from .env import gpu_state, system_info
from .result import BenchResult
from .stats import summarize
from .telemetry import telemetry_window
from .timers.device import PROFILER_WARMUP, time_device
from .timers.eager import time_eager
from .timers.graph import time_graph

# auto cache policy: treat the kernel as cache-sensitive when a cold call is
# this much slower than a warm one.
_AUTO_COLD_THRESHOLD = 0.05
_AUTO_PROBES = 5
# Device mode profiles a few extra calls that it discards.
_DEVICE_EXTRA_CALLS = PROFILER_WARMUP
# Notes flag a GPU that was already busy before timing started.
_BUSY_ACTIVITY_PCT = 5


def _probe_us(call, flush):
    times = []
    for _ in range(_AUTO_PROBES):
        if flush:
            flush_cache()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        call()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return statistics.median(times)


def _resolve_auto(call, mode, has_inputs, notes):
    warm = _probe_us(call, flush=False)
    cold = _probe_us(call, flush=True)
    delta = (cold - warm) / cold if cold > 0 else 0.0
    if delta <= _AUTO_COLD_THRESHOLD:
        return "warm", delta
    if mode != "graph":
        return "flush", delta
    if has_inputs:
        return "rotate", delta
    notes.append(
        "auto: kernel is cache-sensitive, but graph mode can only run cold by "
        "rotating inputs and none were passed; timed warm"
    )
    return "warm", delta


def _call(fn, args, kwargs):
    return lambda: fn(*args, **kwargs)


def _measure(fn, args, kwargs, cfg, name, flops, nbytes):
    device = torch.cuda.current_device()
    notes = []
    state_before = gpu_state(device) if cfg.record_gpu_state else None
    activity = (state_before or {}).get("gfx_activity_pct")
    if activity is not None and activity > _BUSY_ACTIVITY_PCT:
        notes.append(f"GPU was already busy before timing (gfx_activity={activity}%)")

    torch.cuda.synchronize()
    start = time.perf_counter()
    data = fn(*args, **kwargs)
    torch.cuda.synchronize()
    compile_ms = (time.perf_counter() - start) * 1000.0
    for _ in range(cfg.warmup_iters):
        fn(*args, **kwargs)
    torch.cuda.synchronize()

    cache = cfg.resolved_cache
    delta = None
    if cache == "auto":
        has_inputs = input_bytes(args, kwargs) > 0
        cache, delta = _resolve_auto(
            _call(fn, args, kwargs), cfg.mode, has_inputs, notes
        )
    if cache == "flush" and cfg.mode == "graph":
        raise ValueError(
            "graph mode cannot flush the cache between recorded calls; use "
            "cache='rotate' to time graph replay with cold inputs"
        )
    rotation = None
    copies = [(args, kwargs)]
    if cache == "rotate":
        # Graph replays repeat the recorded calls, so copies are reused.
        timed_calls = None if cfg.mode == "graph" else cfg.iters + _DEVICE_EXTRA_CALLS
        rotation = make_rotation(
            args, kwargs, cfg.rotate_max_copies, timed_calls=timed_calls, device=device
        )
        copies = rotation.copies
        if not rotation.cold:
            notes.append(
                f"rotate: only {len(copies)} input copies fit (covers "
                f"{rotation.llc_coverage:.2f}x the last-level cache); timing "
                "may be partly warm"
            )
    calls = [_call(fn, a, k) for a, k in copies]
    before_call = flush_cache if cache == "flush" else None

    with telemetry_window(device, cfg.telemetry) as window:
        if cfg.mode == "device":
            out = time_device(calls, cfg.iters, before_call=before_call)
        elif cfg.mode == "graph":
            out = time_graph(
                calls,
                cfg.iters,
                min_replay_us=cfg.graph_min_replay_us,
                max_calls=cfg.graph_max_calls,
            )
        else:
            out = time_eager(
                calls, cfg.iters, before_call=before_call, sync_each=cfg.eager_sync_each
            )
    notes += out.notes
    if window.error:
        notes.append(window.error)
    state_after = gpu_state(device) if cfg.record_gpu_state else None

    stats = summarize(out.samples_us)
    result = BenchResult(
        name=name or getattr(fn, "__name__", None),
        mode=cfg.mode,
        metric=cfg.metric,
        us=stats.median if cfg.metric == "median" else stats.mean,
        samples_us=out.samples_us,
        median_us=stats.median,
        mean_us=stats.mean,
        min_us=stats.min,
        max_us=stats.max,
        std_us=stats.std,
        outliers=stats.outliers,
        cache_requested=cfg.cache,
        cache_policy=cache,
        cache_probe_delta=delta,
        rotate_copies=len(copies) if rotation else None,
        rotate_llc_coverage=rotation.llc_coverage if rotation else None,
        compile_ms=compile_ms,
        warmup_iters=cfg.warmup_iters,
        iters=cfg.iters,
        flops=flops,
        nbytes=nbytes,
        tflops=flops / stats.median / 1e6 if flops else None,
        gb_s=nbytes / stats.median / 1e3 if nbytes else None,
        system=system_info(device).to_dict(),
        gpu_state_before=state_before,
        gpu_state_after=state_after,
        telemetry=window.summary,
        notes=notes,
        **out.info,
    )
    return result, data


def time_fn(
    fn, measure=None, *, args=(), kwargs=None, name=None, flops=None, nbytes=None
) -> BenchResult:
    """Time ``fn(*args, **kwargs)``. Pass inputs via ``args`` for cache='rotate'.

    ``flops`` and ``nbytes`` per call give ``tflops`` and ``gb_s``.
    """
    result, _ = _measure(
        fn,
        tuple(args),
        dict(kwargs or {}),
        measure or MeasureConfig(),
        name,
        flops,
        nbytes,
    )
    return result


def run_bench(fn, *args, measure=None, name=None, flops=None, nbytes=None, **kwargs):
    """``run_perftest``-style runner: ``run_bench(fn, *args, **kwargs)``.

    Returns ``(data, BenchResult)``. ``data`` is the output of the first call,
    made on the caller's original inputs before any timing. ``measure``,
    ``name``, ``flops`` and ``nbytes`` are reserved keyword names.
    """
    result, data = _measure(
        fn, args, kwargs, measure or MeasureConfig(), name, flops, nbytes
    )
    return data, result


def time_all(
    fn,
    measure=None,
    *,
    modes=MODES,
    args=(),
    kwargs=None,
    name=None,
    flops=None,
    nbytes=None,
) -> dict:
    """Time ``fn`` in each of ``modes``; returns ``{mode: BenchResult}``.

    ``eager - graph`` is the host launch overhead per call; a device-mode
    ``span_us - median_us`` is the time between the call's kernels.
    """
    base = measure or MeasureConfig()
    results = {}
    for mode in modes:
        cfg = base.replace(mode=mode)
        if base.cache == "flush" and mode == "graph":
            cfg = cfg.replace(cache="rotate")
        results[mode] = time_fn(
            fn, cfg, args=args, kwargs=kwargs, name=name, flops=flops, nbytes=nbytes
        )
    first = results[modes[0]].compile_ms
    for mode in modes[1:]:
        results[mode].compile_ms = first
    return results
