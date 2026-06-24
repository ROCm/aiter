#!/usr/bin/env python3

# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

from contextlib import contextmanager
import functools
import inspect
import json
import multiprocessing
from multiprocessing.connection import wait as wait_for_sentinels
import shutil
import time
from dataclasses import dataclass
import enum
import os
from typing import Any, Callable, Iterator

_DEFAULT_KERNEL_TIMEOUT = 1200.0
_DEFAULT_MAX_WORKERS = 64
_DEFAULT_MAX_RETRIES = 2
_DEFAULT_MEM_PER_WORKER_GB = 2.0
_MAX_ERRORS_IN_MSG = 10


class OpKind(enum.Enum):
    """FlyDSL AOT kernel categories -- enum so typos at call sites become
    construction errors instead of silently routing to the wrong code path."""

    MOE = "moe"
    GEMM = "gemm"
    GROUPED_MOE = "grouped_moe"
    CHUNK_GDN_H = "chunk_gdn_h"


@dataclass(frozen=True)
class JobLabel:
    """Diagnostic label attached to a submitted future."""

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
                    last_cache_key[id(exe)] = exe._build_full_cache_key(bound.arguments)
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
    elif kind is OpKind.GROUPED_MOE:
        # from .grouped_moe import DEFAULT_CSVS, parse_csv
        return []
    elif kind is OpKind.CHUNK_GDN_H:
        from .chunk_gdn_h import DEFAULT_CSVS, parse_csv
    else:
        raise ValueError(f"unknown FlyDSL AOT kind: {kind!r}")
    return collect_aot_jobs(DEFAULT_CSVS, parse_csv)


def _compile_one(kind: OpKind, job: dict[str, Any]) -> tuple[OpKind, dict[str, Any]]:
    """Per-kernel worker -- runs in a forked child process. Imports
    compile_one_config lazily so the import cost is paid once in the
    child rather than in the parent."""
    if kind is OpKind.MOE:
        from .moe import compile_one_config
    elif kind is OpKind.GEMM:
        from .gemm import compile_one_config
    elif kind is OpKind.GROUPED_MOE:
        # grouped_moe AOT not wired up yet; return trivial result so no
        # job is ever actually compiled (no jobs are collected either).
        return kind, {}
    elif kind is OpKind.CHUNK_GDN_H:
        from .chunk_gdn_h import compile_one_config
    else:
        raise ValueError(f"unknown FlyDSL AOT kind: {kind!r}")
    return kind, compile_one_config(**job)


def _compile_one_to_file(kind: OpKind, job: dict[str, Any], out_path: str) -> None:
    """Child-process entry point. Runs the compile and writes its result
    dict to ``out_path`` as JSON -- the child's *private* return channel.

    This is the crux of the deadlock-free design: results never travel
    through a result queue shared with sibling workers, so there is no
    cross-process write lock (POSIX semaphore) to leak when this process is
    OOM-killed mid-write. Each child owns its own file; siblings are
    completely decoupled. A crash here leaves the file absent or partial,
    which the parent treats as a failure (it only trusts the file on a clean
    exit code) -- never a hang.

    Write-then-rename so the parent never observes a torn file even if we
    are killed between open() and the final bytes."""
    _, result = _compile_one(kind, job)
    tmp_path = out_path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(result, f)
    os.replace(tmp_path, out_path)


def _affinity_aware_cpu_count() -> int:
    """Number of CPUs this process may actually use (respects cgroup /
    cpuset limits via ``sched_getaffinity``, unlike ``os.cpu_count()``).
    Falls back to ``cpu_count`` and is clamped to >=1."""
    try:
        n = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        n = os.cpu_count() or 0
    return max(n, 1)


@dataclass
class AotPlan:
    """Everything ``wait_aot`` needs to run the compile fleet, produced by
    ``start_aot``. No worker processes are live yet -- they are spawned and
    reaped entirely inside ``wait_aot``, so there is no shared pool object
    (and crucially no shared result queue) that could outlive an exception
    or wedge the parent."""

    jobs: list[tuple[OpKind, dict[str, Any]]]
    max_workers: int
    kernel_timeout: float
    max_retries: int
    result_dir: str


def get_kernel_timeout() -> float:
    """Per-kernel wall-clock cap in seconds. Env override is validated;
    "0" (or negative) disables the cap. See ``_DEFAULT_KERNEL_TIMEOUT``."""
    env = os.environ.get("AITER_FLYDSL_AOT_TIMEOUT")
    if env is None:
        return _DEFAULT_KERNEL_TIMEOUT
    try:
        return max(float(env), 0.0)
    except ValueError as e:
        raise ValueError(
            f"AITER_FLYDSL_AOT_TIMEOUT must be a number of seconds, got {env!r}"
        ) from e


def get_max_retries() -> int:
    """Retries for a worker that died abnormally. Env override validated;
    "0" disables retries. See ``_DEFAULT_MAX_RETRIES``."""
    env = os.environ.get("AITER_FLYDSL_AOT_MAX_RETRIES")
    if env is None:
        return _DEFAULT_MAX_RETRIES
    try:
        return max(int(env), 0)
    except ValueError as e:
        raise ValueError(
            f"AITER_FLYDSL_AOT_MAX_RETRIES must be an integer, got {env!r}"
        ) from e


def _memory_worker_cap(default_workers: int) -> int:
    """Cap concurrency by available memory so the OOM-killer never fires.

    A worker SIGKILLed mid-result by the OOM-killer was the original deadlock
    trigger; the cleanest defense is to not run out of memory at all. Assumes
    ``AITER_FLYDSL_AOT_MEM_PER_WORKER_GB`` (default
    ``_DEFAULT_MEM_PER_WORKER_GB``) resident per worker. If psutil is
    unavailable the cap is skipped."""
    env = os.environ.get("AITER_FLYDSL_AOT_MEM_PER_WORKER_GB")
    try:
        per_gb = float(env) if env else _DEFAULT_MEM_PER_WORKER_GB
    except ValueError as e:
        raise ValueError(
            f"AITER_FLYDSL_AOT_MEM_PER_WORKER_GB must be a number, got {env!r}"
        ) from e
    if per_gb <= 0:
        return default_workers
    try:
        import psutil

        avail_gb = psutil.virtual_memory().available / (1024**3)
    except Exception:
        return default_workers
    return min(default_workers, max(1, int(avail_gb / per_gb)))


def get_max_workers(num_jobs: int) -> int:
    """Resolve the concurrent AOT worker-process cap.

    ``AITER_FLYDSL_AOT_WORKERS``, if set, is honored verbatim (non-integer ->
    ValueError; "0"/negative clamped to 1) and bypasses the memory cap -- an
    explicit setting is the caller's deliberate choice. Otherwise the cap is
    ``min(affinity-aware CPU count, _DEFAULT_MAX_WORKERS)`` further bounded by
    available memory so the OOM-killer never fires. Either way the result never
    exceeds ``num_jobs`` -- spawning more workers than kernels is pointless."""
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
        # Auto path only: also bound by memory so we never trip the OOM-killer.
        max_workers = _memory_worker_cap(max_workers)
    return min(max_workers, num_jobs)


def start_aot(cache_dir: str) -> AotPlan | None:
    """Plan FlyDSL AOT compilation. Collects one job per kernel (across all
    OpKind members) and resolves the run configuration; ``wait_aot`` does the
    actual spawning. Returns None (and ``wait_aot`` becomes a no-op) when
    there are no kernels to compile.

    Env:
      AITER_FLYDSL_AOT_WORKERS         -- max concurrent worker processes.
                                          Non-integer -> ValueError;
                                          "0"/negative clamped to 1. Default:
                                          min(affinity-aware CPU count,
                                          _DEFAULT_MAX_WORKERS), then capped by
                                          available memory (see below).
      AITER_FLYDSL_AOT_MEM_PER_WORKER_GB -- assumed GiB/worker for the auto
                                          memory cap ("0" disables it). Only
                                          applies when AITER_FLYDSL_AOT_WORKERS
                                          is not set explicitly.
      AITER_FLYDSL_AOT_TIMEOUT         -- per-kernel wall-clock cap, seconds
                                          ("0" disables). See
                                          _DEFAULT_KERNEL_TIMEOUT.
      AITER_FLYDSL_AOT_MAX_RETRIES     -- retries for an abnormally-dead worker
                                          ("0" disables). See
                                          _DEFAULT_MAX_RETRIES.

    Workers are always created with the fork start method: fork+COW lets each
    of the (one-per-kernel) children share the parent's already-imported torch
    / FlyDSL for free, which is what makes process-per-kernel affordable.
    """
    os.makedirs(cache_dir, exist_ok=True)
    os.environ["FLYDSL_RUNTIME_CACHE_DIR"] = cache_dir

    all_jobs: list[tuple[OpKind, dict[str, Any]]] = []
    for kind in OpKind:
        for job in _collect_aot_jobs_for(kind):
            all_jobs.append((kind, job))

    if not all_jobs:
        print("[aiter] FlyDSL AOT: no kernels to compile, skipping")
        return None

    max_workers = get_max_workers(len(all_jobs))

    # Per-child result files live here -- recreated fresh so stale results
    # from a previous (e.g. crashed) build can never be mistaken for this
    # run's output.
    result_dir = os.path.join(cache_dir, ".aot_results")
    shutil.rmtree(result_dir, ignore_errors=True)
    os.makedirs(result_dir, exist_ok=True)

    print(
        f"[aiter] FlyDSL AOT: {len(all_jobs)} kernels "
        f"({'+'.join(k.name for k in OpKind)}), "
        f"{max_workers} worker processes (cache: {cache_dir})"
    )
    return AotPlan(
        jobs=all_jobs,
        max_workers=max_workers,
        kernel_timeout=get_kernel_timeout(),
        max_retries=get_max_retries(),
        result_dir=result_dir,
    )


def wait_aot(plan: AotPlan | None) -> None:
    """Run the AOT compile fleet and raise on any failure.

    Deadlock-free by construction. Each kernel runs in its *own* independent
    worker process that returns its result through its *own* file -- there is
    no result queue shared across workers, hence no cross-process write lock
    (POSIX semaphore) that an OOM-killed worker could leak to its siblings.
    That leaked-semaphore wedge was the root cause of the multi-hour hang;
    removing the shared channel removes the failure mode entirely.

    A worker that *dies* (OOM-kill -> exitcode -9, segfault -> -11) is noticed
    immediately via its process sentinel; a worker stuck *alive* is bounded by a
    generous per-kernel timeout. Either abnormal exit is *retried* up to
    ``plan.max_retries`` times -- OOM-kills and probabilistic fork wedges are
    transient, and the FlyDSL cache persists each artifact as produced, so a
    retry re-runs almost nothing. A clean compile error (exit 0, no kernel) is
    deterministic and never retried."""
    if plan is None or not plan.jobs:
        return

    ctx = multiprocessing.get_context("fork")
    timeout = plan.kernel_timeout
    max_retries = plan.max_retries

    ok_by_kind: dict[OpKind, int] = {k: 0 for k in OpKind}
    fail_by_kind: dict[OpKind, int] = {k: 0 for k in OpKind}
    errors: list[str] = []
    retries_used = 0

    # work item: (idx, attempt, kind, job). Submission order preserved: pop()
    # takes from the tail, so reverse. Retries are appended and run next.
    queue: list[tuple[int, int, OpKind, dict[str, Any]]] = [
        (idx, 0, kind, job) for idx, (kind, job) in enumerate(plan.jobs)
    ]
    queue.reverse()
    # proc -> (idx, attempt, kind, job, label, out_path, deadline|None)
    running: dict[Any, tuple] = {}

    def launch() -> None:
        while queue and len(running) < plan.max_workers:
            idx, attempt, kind, job = queue.pop()
            label = JobLabel(kind=kind, kernel_name=str(job.get("kernel_name", "?")))
            out_path = os.path.join(plan.result_dir, f"k{idx}.json")
            proc = ctx.Process(target=_compile_one_to_file, args=(kind, job, out_path))
            proc.start()
            deadline = (time.monotonic() + timeout) if timeout > 0 else None
            running[proc] = (idx, attempt, kind, job, label, out_path, deadline)

    def on_abnormal(
        idx: int, attempt: int, kind: OpKind, job: dict, label: JobLabel, reason: str
    ) -> None:
        """An abnormal worker death (crash or timeout-kill): retry or record."""
        nonlocal retries_used
        if attempt < max_retries:
            retries_used += 1
            queue.append((idx, attempt + 1, kind, job))
            print(f"[aiter] FlyDSL {label} {reason}; retry {attempt + 1}/{max_retries}")
        else:
            fail_by_kind[kind] += 1
            tries = f" after {attempt + 1} attempts" if attempt else ""
            errors.append(f"FlyDSL {label} {reason}{tries}")

    def reap(proc: Any) -> None:
        idx, attempt, kind, job, label, out_path, _ = running.pop(proc)
        if proc.exitcode == 0 and os.path.isfile(out_path):
            try:
                with open(out_path) as f:
                    result = json.load(f)
            except Exception:
                result = None
            if result is not None and result.get("compile_time") is not None:
                ok_by_kind[kind] += 1
            else:
                # Clean exit but no kernel produced (compile_one_config caught a
                # compile error and returned compile_time=None). Deterministic,
                # so not retried.
                fail_by_kind[kind] += 1
                errors.append(f"FlyDSL {label} produced no kernel")
        else:
            # Non-zero/None exit or missing file == the worker died (OOM-kill,
            # segfault) or could not write its result. exitcode < 0 is the
            # negated signal number (-9 = SIGKILL, -11 = SIGSEGV). Transient ->
            # retry.
            on_abnormal(
                idx,
                attempt,
                kind,
                job,
                label,
                f"worker crashed (exitcode={proc.exitcode})",
            )

    try:
        launch()
        while running:
            # Block until a worker exits, or until the nearest per-kernel
            # deadline so a stuck-but-alive worker can be killed.
            if timeout > 0:
                nearest = min(d for (*_, d) in running.values() if d is not None)
                wait_timeout: float | None = max(0.0, nearest - time.monotonic())
            else:
                wait_timeout = None
            wait_for_sentinels([p.sentinel for p in running], timeout=wait_timeout)

            for proc in list(running):
                if not proc.is_alive():
                    proc.join()
                    reap(proc)

            if timeout > 0:
                now = time.monotonic()
                for proc in list(running):
                    idx, attempt, kind, job, label, _, deadline = running[proc]
                    if deadline is not None and now > deadline and proc.is_alive():
                        proc.kill()
                        proc.join()
                        running.pop(proc)
                        on_abnormal(
                            idx,
                            attempt,
                            kind,
                            job,
                            label,
                            f"exceeded per-kernel timeout ({timeout:.0f}s); killed",
                        )

            launch()  # refill freed slots (including any just-requeued retries)

        if retries_used:
            print(
                f"[aiter] FlyDSL AOT: {retries_used} retr"
                f"{'y' if retries_used == 1 else 'ies'} after abnormal worker exits"
            )
        for kind in OpKind:
            print(
                f"[aiter] FlyDSL {kind.name} AOT: "
                f"compiled {ok_by_kind[kind]} ok, {fail_by_kind[kind]} failed"
            )
        if errors:
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
        # Kill any survivors (e.g. on an unexpected exception) so we never
        # leave orphaned compilers blocking the build's exit, then drop the
        # per-child result files.
        for proc in list(running):
            try:
                if proc.is_alive():
                    proc.kill()
            except Exception:
                pass
        shutil.rmtree(plan.result_dir, ignore_errors=True)
