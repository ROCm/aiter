#!/usr/bin/env python3

# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

from concurrent.futures import Future, ProcessPoolExecutor
from contextlib import contextmanager
import functools
import inspect
from dataclasses import dataclass
import enum
import multiprocessing
import os
from typing import Any, Callable, Iterator

# Cap on the entries embedded in the AssertionError message -- beyond
# this we append a "(... N more)" suffix. Per-kernel diagnostics still
# go to stdout from compile_one_config's [FAIL] prints; the exception
# text just needs enough to point at the problem.
_MAX_ERRORS_IN_MSG = 10

# Default ceiling for the FlyDSL AOT process pool, applied on top of
# affinity-aware CPU count. Each worker re-imports torch + FlyDSL
# (~1.5-2.5 GB RSS), so 64 x 2 GB ~= 128 GB -- fits comfortably on a
# typical 64+ core build host with >=256 GB RAM, but containers with
# tighter cgroup memory caps may want to lower via AITER_FLYDSL_AOT_WORKERS.
_DEFAULT_MAX_WORKERS = 64


class OpKind(enum.Enum):
    """FlyDSL AOT kernel categories -- enum so typos at call sites become
    construction errors instead of silently routing to the wrong code path."""

    MOE = "moe"
    GEMM = "gemm"
    CHUNK_GDN_H = "chunk_gdn_h"


@dataclass(frozen=True)
class JobLabel:
    """Diagnostic label attached to a submitted future. Replaces the
    earlier string-formatted label that wait_aot had to parse back into
    a kind via ``label.startswith(OpKind.MOE.name)`` -- a heuristic that
    silently misattributed crashes if a future OpKind member was added."""

    kind: OpKind
    kernel_name: str

    def __str__(self) -> str:
        return f"{self.kind.name} {self.kernel_name}"


_CU_NUM_TO_ARCH = {
    80: "gfx942",
    304: "gfx942",
    256: "gfx950",
}


def cu_num_to_arch(cu_num: int, default: str = "gfx950") -> str:
    """Map compute-unit count to GPU architecture string."""
    return _CU_NUM_TO_ARCH.get(cu_num, default)


def job_identity(job: dict[str, Any]) -> tuple:
    return tuple(sorted(job.items()))


def dedupe_jobs(jobs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    unique_jobs = []
    seen = set()
    for job in jobs:
        key = job_identity(job)
        if key in seen:
            continue
        seen.add(key)
        unique_jobs.append(job)
    return unique_jobs


def collect_aot_jobs(
    csv_paths: list[str],
    parse_csv: Callable[[str], list[dict[str, Any]]],
    on_missing_csv: Callable[[str], None] | None = None,
) -> list[dict[str, Any]]:
    jobs = []
    for csv_path in csv_paths:
        if os.path.isfile(csv_path):
            jobs.extend(parse_csv(csv_path))
        elif on_missing_csv is not None:
            on_missing_csv(csv_path)
    return dedupe_jobs(jobs)


def raise_if_aot_cache_miss(
    case_kwargs: dict[str, Any],
    cache_misses: list[tuple[str, int, Any, Any, int]],
    last_cache_key: dict[int, Any],
) -> None:
    if not cache_misses:
        return

    details = []
    for name, jf_id, manager_key, cache_dir, miss_count in cache_misses:
        exists = cache_dir.exists() if cache_dir is not None else False
        pkl_count = sum(1 for _ in cache_dir.glob("*.pkl")) if exists else 0
        cache_key = last_cache_key.get(jf_id)
        cache_key_str = (
            "\n".join(f"      {item!r}" for item in cache_key)
            if cache_key
            else "<unknown>"
        )
        details.append(
            f"  {name}: +{miss_count} miss, manager_key={manager_key}\n"
            f"    cache_dir={cache_dir} (exists={exists}, pkl_count={pkl_count})\n"
            f"    looked-up cache_key:\n{cache_key_str}"
        )

    raise RuntimeError(
        "AOT cache miss for case " + repr(case_kwargs) + ":\n" + "\n".join(details)
    )


def fail_on_aot_cache_miss(
    run_compiled_module: Any,
    run_compiled_name: str = "_run_compiled",
) -> Callable:
    """Fail a wrapped test when a patched FlyDSL run helper reports cache misses."""

    def decorator(func: Callable) -> Callable:
        jit_fns_seen = []
        last_cache_key = {}

        def case_arguments(args, kwargs):
            try:
                bound = inspect.signature(func).bind_partial(*args, **kwargs)
                bound.apply_defaults()
                return dict(bound.arguments)
            except Exception:
                case_kwargs = dict(kwargs)
                if args:
                    case_kwargs["args"] = args
                return case_kwargs

        def aot_cache_misses():
            misses = []
            for jf in jit_fns_seen:
                info = jf.cache_info()
                if info is None or info.misses == 0:
                    continue
                cache_dir = getattr(jf.cache_manager, "cache_dir", None)
                misses.append(
                    (jf.func.__name__, id(jf), jf.manager_key, cache_dir, info.misses)
                )
            return misses

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            orig_run_compiled = getattr(run_compiled_module, run_compiled_name)

            def run_compiled_tracked(exe, compile_args):
                if exe not in jit_fns_seen:
                    jit_fns_seen.append(exe)
                try:
                    exe._ensure_sig()
                    bound = exe._sig.bind(*compile_args)
                    bound.apply_defaults()
                    last_cache_key[id(exe)] = exe._make_cache_key(bound.arguments)
                except Exception:
                    pass
                return orig_run_compiled(exe, compile_args)

            setattr(run_compiled_module, run_compiled_name, run_compiled_tracked)
            try:
                ret = func(*args, **kwargs)
                raise_if_aot_cache_miss(
                    case_arguments(args, kwargs), aot_cache_misses(), last_cache_key
                )
                return ret
            finally:
                setattr(run_compiled_module, run_compiled_name, orig_run_compiled)

        return wrapper

    return decorator


@contextmanager
def compile_only_env() -> Iterator[None]:
    prev = os.environ.get("COMPILE_ONLY")
    os.environ["COMPILE_ONLY"] = "1"
    try:
        yield
    finally:
        if prev is None:
            os.environ.pop("COMPILE_ONLY", None)
        else:
            os.environ["COMPILE_ONLY"] = prev


@contextmanager
def override_env(var_name: str, value: str | None) -> Iterator[None]:
    prev = os.environ.get(var_name)
    if value is None:
        os.environ.pop(var_name, None)
    else:
        os.environ[var_name] = value
    try:
        yield
    finally:
        if prev is None:
            os.environ.pop(var_name, None)
        else:
            os.environ[var_name] = prev


def _collect_aot_jobs_for(kind: OpKind) -> list[dict[str, Any]]:
    """Load DEFAULT_CSVS + parse_csv for the named kind and return its
    job list. Note: importing .gemm / .moe / .chunk_gdn_h here also
    runs their module-level imports, which pull in FlyDSL (e.g.
    ``flydsl.expr``). Job collection is therefore not free in the
    parent process, just shifted once out of every child."""
    if kind is OpKind.MOE:
        from .moe import DEFAULT_CSVS, parse_csv
    elif kind is OpKind.GEMM:
        from .gemm import DEFAULT_CSVS, parse_csv
    elif kind is OpKind.CHUNK_GDN_H:
        from .chunk_gdn_h import DEFAULT_CSVS, parse_csv
    else:
        raise ValueError(f"unknown FlyDSL AOT kind: {kind!r}")
    return collect_aot_jobs(DEFAULT_CSVS, parse_csv)


def _compile_one(kind: OpKind, job: dict[str, Any]) -> tuple[OpKind, dict[str, Any]]:
    """Per-kernel worker -- runs in a ProcessPoolExecutor child process.
    Top-level so it's picklable. Imports compile_one_config lazily so
    the pickle wire payload is just (kind, job-dict)."""
    if kind is OpKind.MOE:
        from .moe import compile_one_config
    elif kind is OpKind.GEMM:
        from .gemm import compile_one_config
    elif kind is OpKind.CHUNK_GDN_H:
        from .chunk_gdn_h import compile_one_config
    else:
        raise ValueError(f"unknown FlyDSL AOT kind: {kind!r}")
    return kind, compile_one_config(**job)


def _affinity_aware_cpu_count() -> int:
    """Return the number of CPUs this process is actually allowed to
    use. ``os.cpu_count()`` reports host CPUs and ignores cgroup /
    cpuset constraints common in CI containers; ``sched_getaffinity``
    is the right answer where available (Linux). Fallback to
    ``cpu_count`` otherwise. Clamped to >=1 -- empty-affinity-set or
    None-from-cpu_count would otherwise yield 0 and break the
    ``ProcessPoolExecutor(max_workers=0)`` call site."""
    try:
        n = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        n = os.cpu_count() or 0
    return max(n, 1)


def start_aot(
    cache_dir: str,
) -> tuple[ProcessPoolExecutor | None, dict[Future, JobLabel]]:
    """Start FlyDSL AOT compilation in background processes.

    Submits one task per kernel (across all OpKind members) to a single
    shared ProcessPoolExecutor. Pool size is configurable via env:

      AITER_FLYDSL_AOT_WORKERS -- explicit worker count. Non-integer
                                 values raise ValueError; "0" / negatives
                                 are clamped to 1.
                                 default: min(_affinity_aware_cpu_count(),
                                 _DEFAULT_MAX_WORKERS) -- affinity/cpuset-
                                 aware count capped at the module
                                 constant.

    Returns (pool, futures_dict) -- caller must call ``wait_aot``
    to collect results and raise on failure. If there are no jobs to
    compile, returns (None, {}) and ``wait_aot`` becomes a no-op.
    """
    os.makedirs(cache_dir, exist_ok=True)
    os.environ["FLYDSL_RUNTIME_CACHE_DIR"] = cache_dir

    workers_env = os.environ.get("AITER_FLYDSL_AOT_WORKERS")
    if workers_env is not None:
        try:
            max_workers = max(int(workers_env), 1)
        except ValueError as e:
            raise ValueError(
                f"AITER_FLYDSL_AOT_WORKERS must be an integer, got {workers_env!r}"
            ) from e
    else:
        max_workers = min(_affinity_aware_cpu_count(), _DEFAULT_MAX_WORKERS)

    all_jobs: list[tuple[OpKind, dict[str, Any]]] = []
    for kind in OpKind:
        for job in _collect_aot_jobs_for(kind):
            all_jobs.append((kind, job))

    if not all_jobs:
        print("[aiter] FlyDSL AOT: no kernels to compile, skipping")
        return None, {}

    max_workers = min(max_workers, len(all_jobs))
    print(
        f"[aiter] FlyDSL AOT: {len(all_jobs)} kernels "
        f"({'+'.join(k.name for k in OpKind)}), "
        f"{max_workers} worker processes (cache: {cache_dir})"
    )

    # Use spawn so that workers start a fresh interpreter without inheriting
    # the parent CUDA context. When AITER is installed inside a live GPU
    # container (e.g. SGLang/vLLM), the parent process has an active CUDA
    # context at fork time; forked workers that subsequently import torch
    # trigger CUDA re-initialization errors. Spawn avoids this entirely.
    # The upstream fork-safe analysis (mutex/deadlock) remains valid under
    # spawn as well — workers shell out to the FlyDSL compiler subprocess.
    pool = ProcessPoolExecutor(
        max_workers=max_workers,
        mp_context=multiprocessing.get_context("spawn"),
    )
    futures: dict[Future, JobLabel] = {}
    for kind, job in all_jobs:
        f = pool.submit(_compile_one, kind, job)
        futures[f] = JobLabel(kind=kind, kernel_name=str(job.get("kernel_name", "?")))
    return pool, futures


def wait_aot(pool: ProcessPoolExecutor | None, futures: dict[Future, JobLabel]) -> None:
    """Wait for FlyDSL AOT workers and raise on any failure.

    Aggregates per-kernel results back to per-kind tallies for log
    parity with the previous run_aot_worker output."""
    if pool is None or not futures:
        return
    try:
        ok_by_kind: dict[OpKind, int] = {k: 0 for k in OpKind}
        fail_by_kind: dict[OpKind, int] = {k: 0 for k in OpKind}
        errors: list[str] = []
        for future in futures:
            label = futures[future]
            try:
                kind, result = future.result()
                if result.get("compile_time") is not None:
                    ok_by_kind[kind] += 1
                else:
                    fail_by_kind[kind] += 1
                    # A None compile_time means compile_one_config returned
                    # cleanly but didn't produce a kernel -- still a
                    # failure that the original wait_aot raised on.
                    errors.append(f"FlyDSL {label} produced no kernel")
            except Exception as worker_err:
                # Use the JobLabel's kind directly -- no string parsing,
                # so a future OpKind addition won't silently misattribute.
                fail_by_kind[label.kind] += 1
                errors.append(f"FlyDSL {label} AOT worker crashed: {worker_err}")
        for kind in OpKind:
            print(
                f"[aiter] FlyDSL {kind.name} AOT: "
                f"compiled {ok_by_kind[kind]} ok, {fail_by_kind[kind]} failed"
            )
        if errors:
            # Dedupe before truncating: a BrokenProcessPool cascades to
            # every remaining future.result() call with the SAME message,
            # which would otherwise fill the cap with copies of one
            # symptom and bury the actual first crash.
            seen: set[str] = set()
            unique_errors = [e for e in errors if not (e in seen or seen.add(e))]
            head = unique_errors[:_MAX_ERRORS_IN_MSG]
            suffix = ""
            if len(unique_errors) > _MAX_ERRORS_IN_MSG:
                suffix = (
                    f"; ... ({len(unique_errors) - _MAX_ERRORS_IN_MSG} more unique)"
                )
            tally = ", ".join(f"{k.name}: {fail_by_kind[k]} failed" for k in OpKind)
            raise AssertionError(
                f"[aiter] FlyDSL AOT failures ({tally}): " + "; ".join(head) + suffix
            )
    finally:
        pool.shutdown(wait=False)
