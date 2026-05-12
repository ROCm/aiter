#!/usr/bin/env python3

# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

from contextlib import contextmanager
import enum
import os
from typing import Any, Callable, Iterator


class OpKind(enum.Enum):
    """FlyDSL AOT kernel categories. Members are picklable across
    ProcessPoolExecutor process boundaries; using an enum (rather than
    raw strings) means typos at call sites become construction errors
    instead of silently routing to the wrong code path."""

    MOE = "moe"
    GEMM = "gemm"


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


def _collect_aot_jobs_for(kind: OpKind):
    """Helper: load DEFAULT_CSVS + parse_csv for the named kind and
    return its job list. Note that importing .gemm / .moe here also
    executes their module-level imports, which include FlyDSL itself
    (e.g. ``flydsl.expr`` in gemm.py). Job collection is therefore not
    truly free in the parent process — but it's identical to what the
    pre-refactor ``run_aot_worker`` paid in each child, just shifted
    once into the parent."""
    if kind is OpKind.MOE:
        from .moe import DEFAULT_CSVS, parse_csv
    elif kind is OpKind.GEMM:
        from .gemm import DEFAULT_CSVS, parse_csv
    else:
        raise ValueError(f"unknown FlyDSL AOT kind: {kind!r}")
    return collect_aot_jobs(DEFAULT_CSVS, parse_csv)


def _compile_one(kind: OpKind, job):
    """Per-kernel worker — runs in a ProcessPoolExecutor child process.
    Top-level so it's picklable. Imports compile_one_config lazily so
    the pickle wire payload is just (kind, job-dict)."""
    if kind is OpKind.MOE:
        from .moe import compile_one_config
    elif kind is OpKind.GEMM:
        from .gemm import compile_one_config
    else:
        raise ValueError(f"unknown FlyDSL AOT kind: {kind!r}")
    return kind, compile_one_config(**job)


def _affinity_aware_cpu_count() -> int:
    """Return the number of CPUs this process is actually allowed to
    use. ``os.cpu_count()`` reports host CPUs and ignores cgroup /
    cpuset constraints common in CI containers; ``sched_getaffinity``
    is the right answer where available (Linux). Fallback to
    ``cpu_count`` otherwise."""
    try:
        return len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        return os.cpu_count() or 4


def start_aot(cache_dir: str):
    """Start FlyDSL AOT compilation in background processes.

    Submits one task per kernel (across MoE + GEMM) to a single shared
    ProcessPoolExecutor. Pool size is configurable via env:

      AITER_FLYDSL_AOT_WORKERS — explicit cap (positive int).
                                 Invalid values (non-int, "0", negatives,
                                 surrounding whitespace) raise ValueError
                                 rather than silently falling back.
                                 default: min(_affinity_aware_cpu_count(), 64)
                                 — affinity/cpuset-aware count (handles
                                 cgroup-constrained CI containers) capped
                                 at 64 so we don't blow past typical
                                 FlyDSL/torch import RAM budget on very
                                 wide hosts (192+ cores).

    Previously hardcoded to max_workers=2 (one process per kind, each
    serializing all of its kernels). On a 64+ core build host that left
    almost the entire machine idle for the FlyDSL stage. With one task
    per kernel + a shared pool, the per-kernel compiles fan out across
    all workers.

    Returns (pool, futures_dict) — caller must call ``wait_aot``
    to collect results and raise on failure. If there are no jobs to
    compile, returns (None, {}) and ``wait_aot`` becomes a no-op.
    """
    from concurrent.futures import ProcessPoolExecutor

    os.makedirs(cache_dir, exist_ok=True)
    os.environ["FLYDSL_RUNTIME_CACHE_DIR"] = cache_dir

    workers_env = os.environ.get("AITER_FLYDSL_AOT_WORKERS")
    if workers_env is not None:
        # Strict validation: silent fallback on a misspelled override is
        # exactly the kind of CI surprise we want to avoid.
        try:
            requested = int(workers_env)
        except ValueError as e:
            raise ValueError(
                f"AITER_FLYDSL_AOT_WORKERS must be a positive integer, got {workers_env!r}"
            ) from e
        if requested <= 0:
            raise ValueError(
                f"AITER_FLYDSL_AOT_WORKERS must be > 0, got {workers_env!r}"
            )
        max_workers = requested
    else:
        max_workers = min(_affinity_aware_cpu_count(), 64)

    # Flatten all kernels from both kinds into a single submission list.
    all_jobs: list[tuple[OpKind, dict]] = []
    for kind in OpKind:
        for job in _collect_aot_jobs_for(kind):
            all_jobs.append((kind, job))

    if not all_jobs:
        print("[aiter] FlyDSL AOT: no kernels to compile, skipping")
        return None, {}

    # Cap pool at the actual job count so we don't spin up idle workers.
    max_workers = min(max_workers, len(all_jobs))
    print(
        f"[aiter] FlyDSL AOT: {len(all_jobs)} kernels (MoE+GEMM), "
        f"{max_workers} worker processes (cache: {cache_dir})"
    )

    pool = ProcessPoolExecutor(max_workers=max_workers)
    futures = {}
    for kind, job in all_jobs:
        f = pool.submit(_compile_one, kind, job)
        # Store a small label string for crash diagnostics in wait_aot.
        futures[f] = f"{kind.name} {job.get('kernel_name', '?')}"
    return pool, futures


def wait_aot(pool, futures):
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
            try:
                kind, result = future.result()
                if result.get("compile_time") is not None:
                    ok_by_kind[kind] += 1
                else:
                    fail_by_kind[kind] += 1
                    # A None compile_time means compile_one_config returned
                    # cleanly but didn't produce a kernel — still a
                    # failure that the original wait_aot raised on.
                    label = futures[future]
                    errors.append(f"FlyDSL {label} compile failed (compile_time=None)")
            except Exception as worker_err:
                # Crashes don't tell us which kind — best-effort attribute
                # to whatever the label string starts with.
                label = futures[future]
                kind = OpKind.MOE if label.startswith(OpKind.MOE.name) else OpKind.GEMM
                fail_by_kind[kind] += 1
                errors.append(f"FlyDSL {label} AOT worker crashed: {worker_err}")
        for kind in OpKind:
            print(
                f"[aiter] FlyDSL {kind.name} AOT: "
                f"compiled {ok_by_kind[kind]} ok, {fail_by_kind[kind]} failed"
            )
        if errors:
            # Cap the message body — with per-kernel tasks the failure
            # list can grow to hundreds of entries (vs. ~2 in the
            # pre-refactor design), bloating CI logs and the exception
            # text. The full per-kernel diagnostics live in stdout from
            # the FAIL: lines compile_one_config already prints; the
            # exception text just needs enough to point at the problem.
            _MAX_ERRORS_IN_MSG = 10
            head = errors[:_MAX_ERRORS_IN_MSG]
            suffix = ""
            if len(errors) > _MAX_ERRORS_IN_MSG:
                suffix = f"; ... ({len(errors) - _MAX_ERRORS_IN_MSG} more)"
            tally = ", ".join(f"{k.name}: {fail_by_kind[k]} failed" for k in OpKind)
            raise AssertionError(
                f"[aiter] FlyDSL AOT failures ({tally}): " + "; ".join(head) + suffix
            )
    finally:
        pool.shutdown(wait=False)
