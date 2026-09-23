#!/usr/bin/env python3

# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

import json
import multiprocessing
import os
import shutil
import tempfile
import time
import traceback
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from multiprocessing.connection import wait as wait_for_sentinels
from typing import Any

from .spec import DEFAULT_AOT_SPEC_REGISTRY, AotSpec, register_default_specs

_DEFAULT_KERNEL_TIMEOUT = 1200.0
_DEFAULT_MAX_WORKERS = 64
_DEFAULT_MAX_RETRIES = 2
_DEFAULT_MEM_PER_WORKER_GB = 2.0
_MAX_ERRORS_IN_MSG = 10


@dataclass(frozen=True)
class JobLabel:
    """Diagnostic label attached to a submitted future."""

    spec_name: str
    kernel_name: str

    def __str__(self) -> str:
        return f"{self.spec_name} {self.kernel_name}"


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
def run_only_env() -> Iterator[None]:
    """Force FlyDSL run-only mode: load AOT artifacts, never JIT-compile.

    Any kernel without a usable AOT cache raises RuntimeError at the call
    site (with manager_key/cache_key/cache_dir details) instead of silently
    masking missing precompiled coverage.
    """
    with override_env("FLYDSL_RUNTIME_RUN_ONLY", "1"):
        yield


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


def _compile_aot_job(spec: AotSpec, job: dict[str, Any]) -> dict[str, Any]:
    """Resolve and compile one declarative AOT job inside a fresh worker."""

    return spec.compile(job)


def _run_one_to_file(
    worker: Callable[..., dict[str, Any]],
    kwargs: dict[str, Any],
    label: Any,
    out_path: str,
) -> None:
    try:
        result = worker(**kwargs)
    except Exception as error:  # noqa: BLE001
        # A normal Python/compiler failure is deterministic and must not be
        # mistaken for an OOM/segfault and retried.  Truly abnormal exits never
        # reach this handler and are still retried by the parent.
        traceback.print_exc()
        kernel_name = label.kernel_name if isinstance(label, JobLabel) else str(label)
        result = {
            "kernel_name": kernel_name,
            "compile_time": None,
            "error": f"{type(error).__name__}: {error}",
        }
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


def get_kernel_timeout() -> float:
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
    except Exception:  # noqa: BLE001
        return default_workers
    return min(default_workers, max(1, int(avail_gb / per_gb)))


def get_max_workers(num_jobs: int) -> int:
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


def _run_file_pool(
    specs: list[tuple[Callable[..., dict[str, Any]], dict[str, Any], Any]],
    max_workers: int,
    kernel_timeout: float,
    max_retries: int,
    result_dir: str,
    *,
    start_method: str,
) -> list[dict[str, Any] | None]:
    ctx = multiprocessing.get_context(start_method)
    n = len(specs)
    results: list[dict[str, Any] | None] = [None] * n
    attempts = [0] * n
    retries_used = 0
    completed = 0
    progress_stride = max(1, n // 20)

    queue = list(range(n))
    queue.reverse()  # pop() from the tail -> submission order; retries appended
    running: dict[Any, tuple[int, float | None]] = {}  # proc -> (idx, deadline)

    def launch() -> None:
        while queue and len(running) < max_workers:
            idx = queue.pop()
            worker, kwargs, label = specs[idx]
            out_path = os.path.join(result_dir, f"k{idx}.json")
            try:
                os.remove(out_path)  # clear any stale file from a prior attempt
            except OSError:
                pass
            proc = ctx.Process(
                target=_run_one_to_file,
                args=(worker, kwargs, label, out_path),
            )
            proc.start()
            deadline = (
                (time.monotonic() + kernel_timeout) if kernel_timeout > 0 else None
            )
            running[proc] = (idx, deadline)

    def note_done() -> None:
        """Mark one spec as finalized (succeeded or definitively failed) and emit
        a throttled overall-progress line."""
        nonlocal completed
        completed += 1
        if completed % progress_stride == 0 or completed == n:
            print(f"  ... {completed}/{n} kernels done", flush=True)

    def retry_or_drop(idx: int, reason: str) -> None:
        """Abnormal exit: requeue for retry, or -- once retries are exhausted --
        give up, leaving results[idx]=None to mark the worker as dead/killed."""
        nonlocal retries_used
        if attempts[idx] < max_retries:
            attempts[idx] += 1
            retries_used += 1
            queue.append(idx)
            print(
                f"[aiter] FlyDSL {specs[idx][2]} {reason}; "
                f"retry {attempts[idx]}/{max_retries}",
                flush=True,
            )
        else:
            note_done()  # terminal: results[idx] stays None (== died/killed)

    def reap(proc: Any) -> None:
        idx, _ = running.pop(proc)
        out_path = os.path.join(result_dir, f"k{idx}.json")
        if proc.exitcode != 0:
            # Worker died abnormally (OOM-kill -9, segfault -11, ...) before
            # writing its result. Transient -> retry (terminal once retries run
            # out, leaving results[idx]=None).
            retry_or_drop(idx, f"worker crashed (exitcode={proc.exitcode})")
            proc.close()
            return
        # Clean exit (exitcode 0): deterministic, never retried.
        result: dict[str, Any] | None = None
        if os.path.isfile(out_path):
            try:
                with open(out_path) as f:
                    result = json.load(f)
            except Exception:  # noqa: BLE001
                result = None
        if result is None:
            label = specs[idx][2]
            name = label.kernel_name if isinstance(label, JobLabel) else str(label)
            result = {"kernel_name": name, "compile_time": None}
        results[idx] = result
        note_done()
        proc.close()

    try:
        launch()
        while running:
            if kernel_timeout > 0:
                nearest = min(d for (_, d) in running.values() if d is not None)
                wait_timeout: float | None = max(0.0, nearest - time.monotonic())
            else:
                wait_timeout = None
            wait_for_sentinels([p.sentinel for p in running], timeout=wait_timeout)

            for proc in list(running):
                if not proc.is_alive():
                    proc.join()
                    reap(proc)

            if kernel_timeout > 0:
                now = time.monotonic()
                for proc in list(running):
                    idx, deadline = running[proc]
                    if deadline is not None and now > deadline and proc.is_alive():
                        proc.kill()
                        proc.join()
                        running.pop(proc)
                        proc.close()
                        retry_or_drop(
                            idx,
                            f"exceeded per-kernel timeout ({kernel_timeout:.0f}s); killed",
                        )

            launch()  # refill freed slots (including any just-requeued retries)
    finally:
        # Kill any survivors (e.g. on an unexpected exception) so we never leave
        # orphaned compilers blocking the caller's exit.
        for proc in list(running):
            try:
                if proc.is_alive():
                    proc.kill()
                proc.join()
                proc.close()
            except Exception:  # noqa: BLE001,S110
                pass

    if retries_used:
        print(
            f"[aiter] FlyDSL AOT: {retries_used} retr"
            f"{'y' if retries_used == 1 else 'ies'} after abnormal worker exits",
            flush=True,
        )
    return results


def run_jobs_parallel(
    worker: Callable[..., dict[str, Any]],
    jobs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not jobs:
        return []
    max_workers = get_max_workers(len(jobs))
    print(
        f"[aiter] FlyDSL AOT: {len(jobs)} kernels, {max_workers} worker processes",
        flush=True,
    )
    result_dir = tempfile.mkdtemp(prefix="aot_results_")
    try:
        specs = [(worker, job, str(job.get("kernel_name", "?"))) for job in jobs]
        with override_env("AITER_AOT_IMPORT", "1"):
            raw = _run_file_pool(
                specs,
                max_workers,
                get_kernel_timeout(),
                get_max_retries(),
                result_dir,
                start_method=get_start_method(),
            )
    finally:
        shutil.rmtree(result_dir, ignore_errors=True)
    out: list[dict[str, Any]] = []
    for job, res in zip(jobs, raw):
        if res is None:
            out.append(
                {"kernel_name": job.get("kernel_name", "?"), "compile_time": None}
            )
        else:
            out.append(res)
    return out


def get_start_method() -> str:
    """Return the configured multiprocessing start method.

    ``spawn`` is the safe default for compiler/GPU state.  ``fork`` remains an
    opt-in escape hatch for environments where process startup dominates.
    """

    method = os.environ.get("AITER_FLYDSL_AOT_START_METHOD", "spawn")
    available = multiprocessing.get_all_start_methods()
    if method not in available:
        raise ValueError(
            "AITER_FLYDSL_AOT_START_METHOD must be one of "
            f"{available}, got {method!r}"
        )
    return method


def run_aot(cache_dir: str, *, specs: tuple[AotSpec, ...] | None = None) -> None:
    os.makedirs(cache_dir, exist_ok=True)
    if specs is None:
        register_default_specs()
        specs = DEFAULT_AOT_SPEC_REGISTRY.get_all_specs()

    with (
        override_env("AITER_AOT_IMPORT", "1"),
        override_env("FLYDSL_RUNTIME_CACHE_DIR", cache_dir),
    ):
        all_jobs: list[tuple[AotSpec, dict[str, Any]]] = []
        for spec in specs:
            all_jobs.extend((spec, job) for job in spec.collect_jobs())

        if not all_jobs:
            print("[aiter] FlyDSL AOT: no kernels to compile, skipping")
            return

        max_workers = get_max_workers(len(all_jobs))

        # Per-child result files live here -- recreated fresh so stale results
        # from a previous (e.g. crashed) build can never be mistaken for this
        # run's output.
        result_dir = os.path.join(cache_dir, ".aot_results")
        shutil.rmtree(result_dir, ignore_errors=True)
        os.makedirs(result_dir, exist_ok=True)

        spec_names = "+".join(spec.name for spec in specs)
        start_method = get_start_method()
        print(
            f"[aiter] FlyDSL AOT: {len(all_jobs)} kernels ({spec_names}), "
            f"{max_workers} {start_method} worker processes (cache: {cache_dir})"
        )

        # Every worker receives a small importable spec plus plain job metadata;
        # operator modules and FlyDSL are imported only in that fresh process.
        worker_specs = [
            (
                _compile_aot_job,
                {"spec": spec, "job": job},
                JobLabel(
                    spec_name=spec.name,
                    kernel_name=str(job.get("kernel_name", "?")),
                ),
            )
            for spec, job in all_jobs
        ]

        try:
            raw = _run_file_pool(
                worker_specs,
                max_workers,
                get_kernel_timeout(),
                get_max_retries(),
                result_dir,
                start_method=start_method,
            )

            ok_by_spec = {spec.name: 0 for spec in specs}
            fail_by_spec = {spec.name: 0 for spec in specs}
            errors: list[str] = []
            for (aot_spec, _job), result, worker_spec in zip(
                all_jobs, raw, worker_specs
            ):
                label = worker_spec[2]
                if result is not None and result.get("compile_time") is not None:
                    ok_by_spec[aot_spec.name] += 1
                elif result is None:
                    # Died (crash/OOM) or timed out, even after retries.
                    fail_by_spec[aot_spec.name] += 1
                    errors.append(f"FlyDSL {label} worker died or timed out")
                else:
                    # Clean exit but no kernel (deterministic compile error).
                    fail_by_spec[aot_spec.name] += 1
                    detail = result.get("error")
                    suffix = f": {detail}" if detail else ""
                    errors.append(f"FlyDSL {label} produced no kernel{suffix}")

            for spec in specs:
                print(
                    f"[aiter] FlyDSL {spec.name} AOT: "
                    f"compiled {ok_by_spec[spec.name]} ok, "
                    f"{fail_by_spec[spec.name]} failed"
                )
            if errors:
                seen: set[str] = set()
                unique_errors = [
                    error for error in errors if not (error in seen or seen.add(error))
                ]
                head = unique_errors[:_MAX_ERRORS_IN_MSG]
                suffix = ""
                if len(unique_errors) > _MAX_ERRORS_IN_MSG:
                    suffix = (
                        f"; ... ({len(unique_errors) - _MAX_ERRORS_IN_MSG} more unique)"
                    )
                tally = ", ".join(
                    f"{spec.name}: {fail_by_spec[spec.name]} failed" for spec in specs
                )
                raise AssertionError(
                    f"[aiter] FlyDSL AOT failures ({tally}): "
                    + "; ".join(head)
                    + suffix
                )
        finally:
            shutil.rmtree(result_dir, ignore_errors=True)
