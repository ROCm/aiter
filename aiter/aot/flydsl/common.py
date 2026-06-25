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
import tempfile
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


def _compile_one_config_for(kind: OpKind) -> Callable[..., dict[str, Any]]:
    if kind is OpKind.MOE:
        from .moe import compile_one_config
    elif kind is OpKind.GEMM:
        from .gemm import compile_one_config
    elif kind is OpKind.CHUNK_GDN_H:
        from .chunk_gdn_h import compile_one_config
    elif kind is OpKind.GROUPED_MOE:
        # grouped_moe AOT not wired up yet (no jobs are ever collected); keep a
        # trivial stub so the dispatch is total.
        return lambda **_kw: {}
    else:
        raise ValueError(f"unknown FlyDSL AOT kind: {kind!r}")
    return compile_one_config


def _run_one_to_file(
    worker: Callable[..., dict[str, Any]], kwargs: dict[str, Any], out_path: str
) -> None:
    result = worker(**kwargs)
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
    except Exception:
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
) -> list[dict[str, Any] | None]:
    ctx = multiprocessing.get_context("fork")
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
            worker, kwargs, _label = specs[idx]
            out_path = os.path.join(result_dir, f"k{idx}.json")
            try:
                os.remove(out_path)  # clear any stale file from a prior attempt
            except OSError:
                pass
            proc = ctx.Process(target=_run_one_to_file, args=(worker, kwargs, out_path))
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
            return
        # Clean exit (exitcode 0): deterministic, never retried.
        result: dict[str, Any] | None = None
        if os.path.isfile(out_path):
            try:
                with open(out_path) as f:
                    result = json.load(f)
            except Exception:
                result = None
        if result is None:
            label = specs[idx][2]
            name = label.kernel_name if isinstance(label, JobLabel) else str(label)
            result = {"kernel_name": name, "compile_time": None}
        results[idx] = result
        note_done()

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
            except Exception:
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
        raw = _run_file_pool(
            specs, max_workers, get_kernel_timeout(), get_max_retries(), result_dir
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


def run_aot(cache_dir: str) -> None:
    os.makedirs(cache_dir, exist_ok=True)
    os.environ["FLYDSL_RUNTIME_CACHE_DIR"] = cache_dir

    all_jobs: list[tuple[OpKind, dict[str, Any]]] = []
    for kind in OpKind:
        for job in _collect_aot_jobs_for(kind):
            all_jobs.append((kind, job))

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

    print(
        f"[aiter] FlyDSL AOT: {len(all_jobs)} kernels "
        f"({'+'.join(k.name for k in OpKind)}), "
        f"{max_workers} worker processes (cache: {cache_dir})"
    )

    # One uniform task per kernel: worker = the kind's compile_one_config.
    specs = [
        (
            _compile_one_config_for(kind),
            job,
            JobLabel(kind=kind, kernel_name=str(job.get("kernel_name", "?"))),
        )
        for kind, job in all_jobs
    ]

    try:
        raw = _run_file_pool(
            specs,
            max_workers,
            get_kernel_timeout(),
            get_max_retries(),
            result_dir,
        )

        ok_by_kind: dict[OpKind, int] = {k: 0 for k in OpKind}
        fail_by_kind: dict[OpKind, int] = {k: 0 for k in OpKind}
        errors: list[str] = []
        for (kind, _job), result, spec in zip(all_jobs, raw, specs):
            label = spec[2]
            if result is not None and result.get("compile_time") is not None:
                ok_by_kind[kind] += 1
            elif result is None:
                # Died (crash/OOM) or timed out, even after retries.
                fail_by_kind[kind] += 1
                errors.append(f"FlyDSL {label} worker died or timed out")
            else:
                # Clean exit but no kernel (deterministic compile error).
                fail_by_kind[kind] += 1
                errors.append(f"FlyDSL {label} produced no kernel")

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
        shutil.rmtree(result_dir, ignore_errors=True)
