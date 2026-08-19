#!/usr/bin/env python3

# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

import enum
import os
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

from flydsl.utils.parallel import run_jobs_parallel

_MAX_ERRORS_IN_MSG = 10


class OpKind(enum.Enum):
    """FlyDSL AOT kernel categories -- enum so typos at call sites become
    construction errors instead of silently routing to the wrong code path."""

    MOE = "moe"
    MXFP4_MOE = "mxfp4_moe"
    GEMM = "gemm"
    GROUPED_MOE = "grouped_moe"
    CHUNK_GDN_H = "chunk_gdn_h"


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


def _collect_aot_jobs_for(kind: OpKind) -> list[dict[str, Any]]:
    """Load DEFAULT_CSVS + parse_csv for the named kind and return its
    job list. Note: importing .gemm / .moe / .chunk_gdn_h here also
    runs their module-level imports, which pull in FlyDSL (e.g.
    ``flydsl.expr``). Job collection is therefore not free in the
    parent process, just shifted once out of every child."""
    if kind is OpKind.MOE:
        from .moe import DEFAULT_CSVS, parse_csv
    elif kind is OpKind.MXFP4_MOE:
        from .mxfp4_moe import DEFAULT_CSVS, parse_csv
    elif kind is OpKind.GEMM:
        from .gemm import DEFAULT_CSVS, parse_csv
    elif kind is OpKind.GROUPED_MOE:
        from .grouped_moe import DEFAULT_CSVS, parse_csv
    elif kind is OpKind.CHUNK_GDN_H:
        from .chunk_gdn_h import DEFAULT_CSVS, parse_csv
    else:
        raise ValueError(f"unknown FlyDSL AOT kind: {kind!r}")
    return collect_aot_jobs(DEFAULT_CSVS, parse_csv)


def _compile_one_config_for(kind: OpKind) -> Callable[..., dict[str, Any]]:
    if kind is OpKind.MOE:
        from .moe import compile_one_config
    elif kind is OpKind.MXFP4_MOE:
        from .mxfp4_moe import compile_one_config
    elif kind is OpKind.GEMM:
        from .gemm import compile_one_config
    elif kind is OpKind.GROUPED_MOE:
        from .grouped_moe import compile_one_config
    elif kind is OpKind.CHUNK_GDN_H:
        from .chunk_gdn_h import compile_one_config
    else:
        raise ValueError(f"unknown FlyDSL AOT kind: {kind!r}")
    return compile_one_config


def _compile_aot_job(
    kind: str,
    job: dict[str, Any],
    kernel_name: str,
) -> dict[str, Any]:
    """Dispatch one uniform parallel job to its aiter AOT compiler."""
    del kernel_name
    return _compile_one_config_for(OpKind(kind))(**job)


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

    print(
        f"[aiter] FlyDSL AOT: {len(all_jobs)} kernels "
        f"({'+'.join(k.name for k in OpKind)}), "
        f"cache: {cache_dir}"
    )

    parallel_jobs = [
        {
            "kind": kind.value,
            "job": job,
            "kernel_name": str(job.get("kernel_name", "?")),
        }
        for kind, job in all_jobs
    ]
    results = run_jobs_parallel(_compile_aot_job, parallel_jobs)

    ok_by_kind: dict[OpKind, int] = {kind: 0 for kind in OpKind}
    fail_by_kind: dict[OpKind, int] = {kind: 0 for kind in OpKind}
    errors: list[str] = []
    for (kind, job), result in zip(all_jobs, results):
        if result.get("compile_time") is not None:
            ok_by_kind[kind] += 1
        else:
            fail_by_kind[kind] += 1
            kernel_name = str(job.get("kernel_name", "?"))
            errors.append(
                f"FlyDSL {kind.name} {kernel_name} failed to produce a kernel "
                "(compile error, worker crash, or timeout)"
            )

    for kind in OpKind:
        print(
            f"[aiter] FlyDSL {kind.name} AOT: "
            f"compiled {ok_by_kind[kind]} ok, {fail_by_kind[kind]} failed"
        )
    if errors:
        seen: set[str] = set()
        unique_errors = [
            error for error in errors if not (error in seen or seen.add(error))
        ]
        head = unique_errors[:_MAX_ERRORS_IN_MSG]
        suffix = ""
        if len(unique_errors) > _MAX_ERRORS_IN_MSG:
            suffix = f"; ... ({len(unique_errors) - _MAX_ERRORS_IN_MSG} more unique)"
        tally = ", ".join(
            f"{kind.name}: {fail_by_kind[kind]} failed" for kind in OpKind
        )
        raise AssertionError(
            f"[aiter] FlyDSL AOT failures ({tally}): " + "; ".join(head) + suffix
        )
