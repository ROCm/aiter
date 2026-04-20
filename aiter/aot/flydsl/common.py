#!/usr/bin/env python3

# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

import os
import sys
import time
from contextlib import contextmanager
from typing import Any, Callable, Iterator, Mapping, TypedDict, Tuple

CompileBundle = Tuple[Any, Tuple[Any, ...]]


class ProblemShape(TypedDict, total=False):
    """Logical workload shape carried by an AOT compile job."""


class KernelSpec(TypedDict, total=False):
    """Static kernel configuration carried by an AOT compile job."""


class AotJob(TypedDict):
    """Normalized AOT job shared by different FlyDSL kernel families."""

    kernel_name: str
    problem: ProblemShape
    spec: KernelSpec


def make_aot_job(
    *,
    kernel_name: str,
    problem: Mapping[str, Any],
    spec: Mapping[str, Any],
) -> AotJob:
    return {
        "kernel_name": kernel_name,
        "problem": dict(problem),
        "spec": dict(spec),
    }


def _freeze_identity(value: Any) -> Any:
    if isinstance(value, Mapping):
        return tuple(sorted((k, _freeze_identity(v)) for k, v in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_identity(v) for v in value)
    if isinstance(value, set):
        return tuple(sorted(_freeze_identity(v) for v in value))
    return value


def job_identity(job: AotJob) -> tuple[Any, ...]:
    return (
        job["kernel_name"],
        _freeze_identity(job["problem"]),
        _freeze_identity(job["spec"]),
    )


def dedupe_jobs(jobs: list[AotJob]) -> list[AotJob]:
    unique_jobs: list[AotJob] = []
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
    parse_csv: Callable[[str], list[AotJob]],
    on_missing_csv: Callable[[str], None] | None = None,
) -> list[AotJob]:
    jobs: list[AotJob] = []
    for csv_path in csv_paths:
        if os.path.isfile(csv_path):
            jobs.extend(parse_csv(csv_path))
        elif on_missing_csv is not None:
            on_missing_csv(csv_path)
    return dedupe_jobs(jobs)


def resolve_csv_paths(csv_paths: list[str]) -> list[str]:
    resolved = [os.path.abspath(p) for p in csv_paths]
    for csv_path in resolved:
        if not os.path.isfile(csv_path):
            print(f"Error: CSV file not found: {csv_path}")
            sys.exit(1)
    return resolved


def torch_dtype_for_kernel(dtype_name: str):
    import torch

    mapping = {
        "bf16": torch.bfloat16,
        "f16": torch.float16,
        "fp16": torch.float16,
    }
    if dtype_name not in mapping:
        raise ValueError(f"Unsupported torch dtype name for GEMM AOT: {dtype_name!r}")
    return mapping[dtype_name]


def compile_job(
    *,
    shape_str: str,
    result: dict[str, Any],
    compile_fn: Callable[[], None],
) -> dict[str, Any]:
    t0 = time.time()
    try:
        compile_fn()
        elapsed = time.time() - t0
        result["compile_time"] = elapsed
        print(f"  [OK] compile  {elapsed:6.1f}s  {shape_str}")
    except Exception as e:
        print(f"  [FAIL] compile  {shape_str}: {e}")
    return result


def make_compile_one_config(
    *,
    shape_builder: Callable[[AotJob], str],
    compile_action: Callable[[AotJob], None],
    result_builder: Callable[[AotJob], dict[str, Any]] | None = None,
) -> Callable[[AotJob], dict[str, Any]]:
    def compile_one_config(job: AotJob) -> dict[str, Any]:
        shape_str = shape_builder(job)
        result = {"shape": shape_str, "compile_time": None}
        if result_builder is not None:
            result.update(result_builder(job))
        return compile_job(
            shape_str=shape_str,
            result=result,
            compile_fn=lambda: compile_action(job),
        )

    return compile_one_config


def run_job_sections(
    sections: list[tuple[str, list[AotJob]]],
    compile_one_config: Callable[[AotJob], dict[str, Any]],
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for title, jobs in sections:
        if not jobs:
            continue
        print(f"\n--- {title} ({len(jobs)} kernels) ---")
        for i, job in enumerate(jobs, 1):
            print(f"\n[{i}/{len(jobs)}] ", end="")
            results.append(compile_one_config(job))
    return results


def print_summary(
    *,
    total_elapsed: float,
    results: list[dict[str, Any]],
    cache_dir: str,
) -> int:
    ok = sum(1 for r in results if r["compile_time"] is not None)
    fail = sum(1 for r in results if r["compile_time"] is None)

    print("\n" + "=" * 72)
    print("Summary")
    print("=" * 72)
    print(f"  Total time:   {total_elapsed:.1f}s")
    print(f"  Compiled:     {ok} ok, {fail} failed")
    print(f"  Cache dir:    {cache_dir}")
    print()

    if fail > 0:
        print("Some compilations failed. Check output above for details.")
        return 1

    print("All compilations succeeded. Cache is ready.")
    return 0


def _cleanup_leaked_ir_context() -> None:
    """Best-effort cleanup for leaked MLIR contexts after compilation failure."""
    try:
        from flydsl._mlir import ir

        while ir.Context.current is not None:
            ir.Context.current.__exit__(None, None, None)
    except Exception:
        pass


def compile_bundle_to_cache(
    bundle: CompileBundle,
    runner: Callable[[Any, Tuple[Any, ...]], None] | None = None,
) -> None:
    exe, args = bundle
    with compile_only_env():
        try:
            if runner is not None:
                runner(exe, args)
                return

            compile_fn = getattr(exe, "compile", None)
            if compile_fn is None:
                import flydsl.compiler as flyc

                compile_fn = flyc.compile
                args = (exe, *args)
            compile_fn(*args)
        except Exception:
            _cleanup_leaked_ir_context()
            raise


def execute_bundle(exe: Any, args: Tuple[Any, ...]) -> None:
    """Trigger JIT compilation/execution by calling the executable directly."""
    exe(*args)


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
