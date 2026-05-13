#!/usr/bin/env python3

# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

from contextlib import contextmanager
import functools
import inspect
import os
from typing import Any, Callable, Iterator

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


def run_aot_worker(kind):
    """Worker for ProcessPoolExecutor — runs in a child process."""
    if kind == "moe":
        from .moe import (
            DEFAULT_CSVS,
            compile_one_config,
            parse_csv,
        )
    else:
        from .gemm import (
            DEFAULT_CSVS,
            compile_one_config,
            parse_csv,
        )

    label = f"FlyDSL {kind.upper()} AOT"
    jobs = collect_aot_jobs(DEFAULT_CSVS, parse_csv)
    if not jobs:
        return label, 0, 0
    cache_dir = os.environ.get("FLYDSL_RUNTIME_CACHE_DIR", "~/.flydsl/cache")
    print(f"[aiter] {label}: {len(jobs)} kernels to compile (cache: {cache_dir})")
    results = [compile_one_config(**job) for job in jobs]
    ok = sum(1 for r in results if r["compile_time"] is not None)
    fail = len(results) - ok
    print(f"[aiter] {label}: compiled {ok} ok, {fail} failed")
    return label, ok, fail


def start_aot(cache_dir: str):
    """Start FlyDSL AOT compilation in background processes.

    Returns (pool, futures_dict) — caller must call ``wait_aot``
    to collect results and raise on failure.
    """
    from concurrent.futures import ProcessPoolExecutor

    os.makedirs(cache_dir, exist_ok=True)
    os.environ["FLYDSL_RUNTIME_CACHE_DIR"] = cache_dir

    pool = ProcessPoolExecutor(max_workers=2)
    futures = {
        pool.submit(run_aot_worker, "moe"): "MoE",
        pool.submit(run_aot_worker, "gemm"): "GEMM",
    }
    return pool, futures


def wait_aot(pool, futures):
    """Wait for FlyDSL AOT workers and raise on any failure."""
    try:
        errors = []
        for future in futures:
            try:
                label, ok, fail = future.result()
                if fail > 0:
                    errors.append(f"{label}: {fail} compile failure(s)")
            except Exception as worker_err:
                errors.append(
                    f"FlyDSL {futures[future]} AOT worker crashed: {worker_err}"
                )
        if errors:
            raise AssertionError("[aiter] FlyDSL AOT failures: " + "; ".join(errors))
    finally:
        pool.shutdown(wait=False)
