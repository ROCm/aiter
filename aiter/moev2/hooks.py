# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Explicit test and benchmark hooks for aiter.moev2.

The legacy code is driven by assigning module globals (`kernel_bench_callable` and
`cfg_2stages` in aiter.fused_moe, `_moe_sorting_impl` in aiter.moev2.segments.sort,
which legacy and v2 share). These context managers replace that: they restore
the previous state on exit, so a failing test cannot leak its override into the next.

NOTE: aiter.moev2 is under active development (MoE v2 migration, not graduated).
Please do not modify this code without coordinating with the moev2 owners (@qli88). See
aiter/moev2/README.md.
"""

import contextlib

_kernel_calls: list | None = None
_sort_impl = None


@contextlib.contextmanager
def set_kernel_bench_callable(calls: list | None = None):
    """Collect `(stage name, zero-argument callable)` for every kernel v2 launches.

    Benchmarks re-run the callables to time each kernel alone. Yields the list.
    """
    global _kernel_calls
    previous, _kernel_calls = _kernel_calls, [] if calls is None else calls
    try:
        yield _kernel_calls
    finally:
        _kernel_calls = previous


def legacy_kernel_bench(calls: list | None):
    """`set_kernel_bench_callable(calls)`, or nothing when `calls` is None.

    The legacy code wraps v2 runs in this with its `kernel_bench_callable` global, so
    benchmarks that set that global see the kernels v2 launches too.
    """
    if calls is None:
        return contextlib.nullcontext()
    return set_kernel_bench_callable(calls)


def record_kernel(stage: str, call):
    """Called by the executor after building each kernel launch."""
    if _kernel_calls is not None:
        _kernel_calls.append((stage, call))


@contextlib.contextmanager
def set_sort_impl(impl):
    """Replace the sorting backend v2 calls; `impl` takes the backend's arguments."""
    global _sort_impl
    previous, _sort_impl = _sort_impl, impl
    try:
        yield impl
    finally:
        _sort_impl = previous


def sort_impl(default):
    """The sorting backend to call: the override if one is set, else `default`."""
    return default if _sort_impl is None else _sort_impl
