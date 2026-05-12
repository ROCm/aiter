#!/usr/bin/env python3

# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

from contextlib import contextmanager
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


# Registry of AOT kinds. Each entry maps a short kind name to
#   (module_path, human_label).
# Each module must export ``DEFAULT_CSVS`` (or alias), ``parse_csv`` (or
# alias), and ``compile_one_config`` -- the contract consumed by
# ``run_aot_worker`` below.
_AOT_KINDS: dict[str, tuple[str, str]] = {
    "moe": (".moe", "MoE"),
    "gemm": (".gemm", "GEMM"),
    "chunk_gdn_h": (".chunk_gdn_h", "chunk-gdn-h"),
}


def run_aot_worker(kind: str):
    """Worker for ProcessPoolExecutor -- runs in a child process."""
    if kind not in _AOT_KINDS:
        raise ValueError(f"Unknown AOT kind {kind!r}; supported: {sorted(_AOT_KINDS)}")
    module_path, human_label = _AOT_KINDS[kind]

    from importlib import import_module

    mod = import_module(module_path, package=__package__)
    DEFAULT_CSVS = mod.DEFAULT_CSVS
    parse_csv = mod.parse_csv
    compile_one_config = mod.compile_one_config

    label = f"FlyDSL {human_label} AOT"
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


def start_aot(cache_dir: str, kinds: list[str] | None = None):
    """Start FlyDSL AOT compilation in background processes.

    Args:
        cache_dir: FlyDSL JIT cache directory. Exported to children via
            ``FLYDSL_RUNTIME_CACHE_DIR``.
        kinds: Optional subset of AOT kinds to launch (default: all
            registered kinds).

    Returns (pool, futures_dict) -- caller must call ``wait_aot``
    to collect results and raise on failure.
    """
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor

    os.makedirs(cache_dir, exist_ok=True)
    os.environ["FLYDSL_RUNTIME_CACHE_DIR"] = cache_dir

    if kinds is None:
        kinds = list(_AOT_KINDS)
    else:
        unknown = [k for k in kinds if k not in _AOT_KINDS]
        if unknown:
            raise ValueError(
                f"Unknown AOT kind(s) {unknown}; " f"supported: {sorted(_AOT_KINDS)}"
            )

    # Use ``spawn`` to avoid inheriting a half-initialised CUDA context from
    # the parent process; ``fork`` would crash with
    # "Cannot re-initialize CUDA in forked subprocess" the moment any child
    # touches ``torch.cuda``.
    mp_ctx = multiprocessing.get_context("spawn")
    pool = ProcessPoolExecutor(max_workers=max(1, len(kinds)), mp_context=mp_ctx)
    futures = {pool.submit(run_aot_worker, kind): _AOT_KINDS[kind][1] for kind in kinds}
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
