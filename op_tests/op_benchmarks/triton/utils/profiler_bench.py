# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Torch-profiler based timing, filtered by GPU kernel name.

triton.testing.do_bench times the callable from the host, so everything fn does
besides the kernel of interest ends up in the number.
Here only the device time of the named kernels is summed,
which also works when an op launches several kernels or sits inside a larger
wrapper.
"""

import math
import re
import statistics
import warnings
from collections.abc import Callable, Sequence
from typing import Any

import torch
import torch.profiler as tpf

_REDUCE = {
    "mean": statistics.fmean,
    "median": statistics.median,
    "min": min,
    "max": max,
}
_PERCENTILE = re.compile(r"p(\d+(?:\.\d+)?)")

# Usable as argparse choices; any other "pNN" percentile is accepted too.
RETURN_MODES = (*_REDUCE, "p25", "p50", "p75", "p90", "p99")


def _percentile(values, q):
    """Linear interpolation between the closest ranks, as numpy.percentile does."""
    values = sorted(values)
    idx = (len(values) - 1) * q / 100
    lo, hi = math.floor(idx), math.ceil(idx)
    return values[lo] + (values[hi] - values[lo]) * (idx - lo)


def _reducer(return_mode):
    if return_mode in _REDUCE:
        return _REDUCE[return_mode]
    match = _PERCENTILE.fullmatch(str(return_mode))
    if match and 0 <= float(match.group(1)) <= 100:
        return lambda values: _percentile(values, float(match.group(1)))
    raise ValueError(
        f"unknown return_mode {return_mode!r}: expected one of {list(RETURN_MODES)}, "
        "or any percentile 'pNN'"
    )


def _is_device_event(ev) -> bool:
    return "CUDA" in str(getattr(ev, "device_type", ""))


def _call(fn, data):
    if data is None:
        return fn()
    if isinstance(data, tuple):
        return fn(*data)
    if isinstance(data, dict):
        return fn(**data)
    return fn(data)


def _reduce(times_us, num_iters, reduce_fns, single):
    """Sum the launches belonging to one iteration, then reduce across iterations."""
    if len(times_us) % num_iters != 0:
        warnings.warn(
            f"{len(times_us)} launches over {num_iters} iterations: the kernel count "
            "per iteration is not constant, falling back to mean.",
            stacklevel=3,
        )
        values = [sum(times_us) / num_iters / 1000] * len(reduce_fns)
    else:
        per_iter = len(times_us) // num_iters
        totals = [
            sum(times_us[i * per_iter : (i + 1) * per_iter]) for i in range(num_iters)
        ]
        values = [fn(totals) / 1000 for fn in reduce_fns]
    return values[0] if single else values


def do_bench_profiler(
    fn: Callable,
    kernel_names: str | Sequence[str],
    num_warmup: int = 25,
    num_iters: int = 100,
    flush_l2: bool = True,
    cache_size_mb: int = 512,
    data_gen: Callable[[], Any] | None = None,
    num_copies: int = 1,
    return_mode: str | Sequence[str] = "mean",
    per_kernel: bool = False,
):
    """Time the kernels named in kernel_names while running fn.

    A kernel is selected when one of kernel_names is a substring of its profiler
    name, so launch overhead, cache flushes and everything else fn does are left
    out of the measurement.

    The flush is itself a fill kernel over an int8 buffer, so keep kernel_names
    narrow enough not to match it.

    Pass data_gen to rotate over num_copies input sets built up front instead of
    flushing: iteration i gets copy i % num_copies, called as fn(*data) for a
    tuple, fn(**data) for a dict, else fn(data).

    return_mode is one of RETURN_MODES: mean, median, min, max, or a percentile
    such as p25/p50/p75 (any pNN works). Pass a list of modes, e.g.
    ["mean", "p50", "p90"], to get one value back per mode.

    Returns the per-iteration time in ms, or, with per_kernel=True, one
    {name: ms} entry per name in kernel_names. Each ms is a list of ms when
    return_mode is a list.
    """
    if isinstance(kernel_names, str):
        kernel_names = [kernel_names]
    assert num_iters > 0, "num_iters must be > 0"
    single = isinstance(return_mode, str)
    modes = [return_mode] if single else list(return_mode)
    assert modes, "return_mode must not be empty"
    reduce_fns = [_reducer(mode) for mode in modes]

    copies = [data_gen() for _ in range(max(num_copies, 1))] if data_gen else None
    flush_buf = (
        torch.empty(cache_size_mb * 1024 * 1024, dtype=torch.int8, device="cuda")
        if flush_l2
        else None
    )

    def run(i):
        if flush_buf is not None:
            flush_buf.zero_()
        _call(fn, copies[i % len(copies)] if copies else None)

    for i in range(num_warmup):
        run(i)
    torch.cuda.synchronize()

    with tpf.profile(
        activities=[tpf.ProfilerActivity.CPU, tpf.ProfilerActivity.CUDA]
    ) as prof:
        for i in range(num_iters):
            run(i)
        torch.cuda.synchronize()

    events = [ev for ev in prof.events() if _is_device_event(ev)]
    matched = []
    for ev in sorted(events, key=lambda ev: ev.time_range.start):
        key = next((k for k in kernel_names if k in ev.name), None)
        if key is not None:
            matched.append((key, ev.self_device_time_total))
    if not matched:
        raise ValueError(
            f"no kernel matched {list(kernel_names)}. Kernels seen: "
            f"{sorted({ev.name for ev in events})}"
        )

    if per_kernel:
        return {
            key: _reduce(
                [t for k, t in matched if k == key], num_iters, reduce_fns, single
            )
            for key in dict.fromkeys(k for k, _ in matched)
        }
    return _reduce([t for _, t in matched], num_iters, reduce_fns, single)
