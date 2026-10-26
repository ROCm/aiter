# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Shared driver of the FlyDSL AOT modules that replay aiter's runtime.

An AOT module of this kind restates no launch logic. It turns each tuned row
(or configured shape) into a *job*: the arguments a caller passes the op's
public entry, as FakeTensors. :func:`replay_job` calls that entry inside
:func:`aiter.aot.flydsl.dry_run.dry_run`, where the runtime looks the row up,
picks and parameterizes its kernels and launches them as at inference, and
every FlyDSL launch it reaches is compiled into the cache instead. A kernel or
wrapper change therefore needs no AOT change.

A module provides

- ``LABEL`` (its name in the log) and ``DEFAULT_CSVS`` (the tuned tables the
  runtime reads, as ``AITER_CONFIGS`` resolves them; may be empty).
- ``parse_csv(path) -> list[job]``: one job per point to replay. A job carries
  ``kernel_name`` (a label), ``gfx`` or ``cu_num`` (the arch it compiles for),
  and, when the runtime's choice follows a size the row covers a range of, that
  size under ``m`` and the smallest one the row serves under ``m_low``.
  ``collect_jobs(csv_paths, args)`` replaces it for a module whose jobs come
  from elsewhere (``args`` is None for the defaults).
- ``replay(job)``: build the caller's inputs and call the public entry.
- ``describe(job) -> str``: one line naming the job in the log.
- optionally ``config_env(csv_paths)``, the environment that points the
  runtime's tuned-config lookups at the given tables, and
  ``configure(parser)`` for options of its own.

Its ``python -m`` entry is :func:`main`, and ``setup.py``'s build runs
:func:`run_default`. Both run one process per target arch (``replay_env``),
each replaying its jobs in a pool of workers.

Where a job's two ends request different FlyDSL kernels, the sizes between
them are bisected down to every size at which the runtime switches kernels
(as in ``moe.py``). Each choice the runtime makes from the size is monotonic,
so two sizes that request the same kernels bound a run that requests nothing
else. An end that requests no FlyDSL kernel at all (another backend serves
it) bounds nothing, and is not bisected against.
"""

from __future__ import annotations

import argparse
import importlib
import os
import re
import subprocess
import sys
import time
import traceback
import zlib
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from aiter.aot.flydsl.common import (
    aot_target_archs,
    check_replay_process,
    collect_aot_jobs,
    cu_num_to_arch,
    get_max_workers,
    replay_env,
    run_jobs_parallel,
)

# Jobs one worker replays in turn: they share tuned-table loads and kernels.
_JOBS_PER_WORKER = 16
_MAX_REASONS = 10


def job_arch(job: dict, default: str = "gfx950") -> str:
    gfx = str(job.get("gfx") or "").strip()
    return gfx or cu_num_to_arch(int(job.get("cu_num") or 0), default)


@contextmanager
def requested_kernels() -> Iterator[set]:
    """Collect the cache identity of every FlyDSL launch a replay reaches."""
    from unittest import mock

    from flydsl.compiler.jit_function import JitFunction

    kernels = set()
    build_key = JitFunction._build_full_cache_key

    def record(self, *args, **kwargs):
        key = build_key(self, *args, **kwargs)
        kernels.add((self.manager_key, key))
        return key

    with mock.patch.object(JitFunction, "_build_full_cache_key", record):
        yield kernels


def replay_job(
    module: Any, job: dict, probe: bool = False
) -> tuple[dict[str, Any], frozenset | None]:
    """Replay one job of ``module``; returns its result and the kernels it
    requested (None when it failed). A ``NotImplementedError`` is the runtime
    refusing the call, which is reported, not failed. So is any error of the
    runtime's host code before it reaches FlyDSL at a ``probe`` (a size inside
    a row's range rather than the row's own), where the runtime meets a limit
    of its own (e.g. no kernel for an odd M below the smallest tuned one)."""
    from aiter.aot.flydsl.dry_run import GpuRuntimeReached, dry_run

    arch = job_arch(job, getattr(module, "AOT_ARCH_DEFAULT", "gfx950"))
    description = module.describe(job)
    result: dict[str, Any] = {
        "kernel_name": job["kernel_name"],
        "shape": description,
        "compile_time": None,
        "compile_arch": arch,
    }
    started = time.time()
    with requested_kernels() as kernels:
        try:
            with dry_run(arch, int(job.get("cu_num") or 0), python_ops=True):
                module.replay(job)
            result["compile_time"] = time.time() - started
        except NotImplementedError as error:
            result["compile_time"] = 0.0
            result["unsupported"] = str(error) or type(error).__name__
        except Exception as error:  # noqa: BLE001
            if (
                not isinstance(error, GpuRuntimeReached)
                and _in_other_backend(error)
                and not _in_flydsl(error)
                and not kernels
            ):
                # The runtime handed the call to another backend (Triton, or
                # asm the arch gate refuses), whose launch the dry run cannot
                # carry out; it has no FlyDSL kernel to compile. Any other
                # error, and reaching the GPU runtime, is a gap of the dry run
                # or a broken path, and fails.
                result["compile_time"] = 0.0
                result["other_backend"] = f"{type(error).__name__}: {error}"
                result["kernels"] = 0
                return result, frozenset()
            if (
                probe
                and not isinstance(error, GpuRuntimeReached)
                and not _in_flydsl(error)
                and not kernels
            ):
                result["compile_time"] = 0.0
                result["refused"] = f"{type(error).__name__}: {error}"
                result["kernels"] = 0
                return result, frozenset()
            print(
                f"  [FAIL] {description}  arch={arch}: "
                f"{type(error).__name__}: {error}",
                flush=True,
            )
            if os.environ.get("AITER_FLYDSL_AOT_TRACEBACK") == "1":
                traceback.print_exc()
    if result["compile_time"] is None:
        return result, None
    result["kernels"] = len(kernels)
    return result, frozenset(kernels)


_AOT_DIR = os.path.dirname(os.path.abspath(__file__))


# Code of the backends a replay hands calls to without reaching FlyDSL: Triton
# (the package and aiter's Triton ops) and the gate in front of asm kernels.
_OTHER_BACKENDS = (f"{os.sep}triton{os.sep}", f"{os.sep}asm_guard.py")


def _in_other_backend(error: BaseException) -> bool:
    return any(
        marker in frame.filename
        for frame in traceback.extract_tb(error.__traceback__)
        for marker in _OTHER_BACKENDS
    )


def _in_flydsl(error: BaseException) -> bool:
    """Whether ``error`` was raised on the way through FlyDSL code (FlyDSL
    itself or aiter's FlyDSL ops), not only in host code or other backends."""
    marker = f"{os.sep}flydsl{os.sep}"
    return any(
        marker in frame.filename and not frame.filename.startswith(_AOT_DIR)
        for frame in traceback.extract_tb(error.__traceback__)
    )


def _bisect(module, job: dict, kernels: dict, low: int, high: int) -> list[dict]:
    if high - low < 2 or kernels[low] == kernels[high]:
        return []
    middle = (low + high) // 2
    result, kernels[middle] = replay_job(module, {**job, "m": middle}, probe=True)
    if not kernels[middle]:
        return [result]
    return [
        result,
        *_bisect(module, job, kernels, low, middle),
        *_bisect(module, job, kernels, middle, high),
    ]


def _replay_batch(module_name: str, jobs: list[dict], kernel_name: str) -> dict:
    """Replay each job at its size and at the smallest size it serves, and
    bisect between the two where they request different kernels."""
    module = importlib.import_module(module_name)
    results, probes = [], []
    for job in jobs:
        result, high = replay_job(module, job)
        result["row"] = True
        results.append(result)
        low_m = job.get("m_low")
        if low_m is None or low_m >= job["m"] or high is None:
            continue
        low_result, low = replay_job(module, {**job, "m": low_m}, probe=True)
        probes.append(low_result)
        if high and low:
            kernels = {low_m: low, job["m"]: high}
            probes += _bisect(module, job, kernels, low_m, job["m"])
    return {
        "kernel_name": kernel_name,
        "compile_time": 0.0,
        "results": results,
        "probes": probes,
    }


def _batches(jobs: list[dict]) -> list[list[dict]]:
    """Jobs in runs of at most ``_JOBS_PER_WORKER``, neighbours (which share
    kernels and tuned-table lookups) together, runs in a stable shuffle so
    that no two workers start on the same kernels."""
    ordered = sorted(jobs, key=lambda job: repr(sorted(job.items())))
    batches = [
        ordered[start : start + _JOBS_PER_WORKER]
        for start in range(0, len(ordered), _JOBS_PER_WORKER)
    ]
    batches.sort(key=lambda batch: zlib.crc32(repr(batch[0]).encode()))
    return batches


def replay_arch(module_name: str, jobs: list[dict], arch: str, label: str) -> int:
    """Replay ``jobs`` in this process, which :func:`run` started for ``arch``.
    Returns the number of failed replays."""
    check_replay_process(arch)
    module = importlib.import_module(module_name)
    prepare = getattr(module, "prepare", None)
    if prepare is not None:
        prepare(jobs)
    batches = _batches(jobs)
    outcomes = run_jobs_parallel(
        lambda **kwargs: _replay_batch(module_name, **kwargs),
        [{"jobs": batch, "kernel_name": batch[0]["kernel_name"]} for batch in batches],
    )
    results, probes = [], []
    for batch, outcome in zip(batches, outcomes):
        # A worker that died takes its whole batch with it.
        results += outcome.get("results") or [{"compile_time": None}] * len(batch)
        probes += outcome.get("probes") or []
    failed = sum(1 for r in results if r["compile_time"] is None)
    # Grouped by reason, the sizes it quotes masked.
    refused = Counter(
        re.sub(r"\d+", "N", r["unsupported"]) for r in results if r.get("unsupported")
    )
    unreached = [
        r for r in results if r.get("kernels") == 0 and not r.get("unsupported")
    ]
    print(
        f"[aiter] FlyDSL {label} replay {arch}: "
        f"{len(results) - failed - sum(refused.values())} ok, {failed} failed, "
        f"{sum(refused.values())} refused by the runtime",
        flush=True,
    )
    for reason, count in refused.most_common(_MAX_REASONS):
        print(f"  [SKIP] {count} x {reason}", flush=True)
    if len(refused) > _MAX_REASONS:
        print(f"  [SKIP] ... {len(refused) - _MAX_REASONS} more reasons", flush=True)
    if unreached:
        # The runtime serves these rows with another backend (e.g. a duplicate
        # shape won by a faster row, or a path that ignores the row's libtype).
        print(
            f"  [UNREACHED] {len(unreached)} rows request no FlyDSL kernel, e.g. "
            f"{unreached[0]['shape']}",
            flush=True,
        )
        backends = Counter(
            re.sub(r"\d+", "N", r["other_backend"])
            for r in unreached
            if r.get("other_backend")
        )
        for reason, count in backends.most_common(3):
            print(f"  [UNREACHED] {count} x handed to: {reason}", flush=True)
    probes_failed = sum(1 for r in probes if r["compile_time"] is None)
    if probes:
        print(
            f"  [BISECT] {len(probes)} sizes inside rows' ranges, "
            f"{probes_failed} failed",
            flush=True,
        )
    limits = Counter(
        re.sub(r"\d+", "N", r["refused"]) for r in probes if r.get("refused")
    )
    for reason, count in limits.most_common(_MAX_REASONS):
        print(f"  [REFUSED] {count} sizes inside rows' ranges: {reason}", flush=True)
    return failed + probes_failed


def run(
    module_name: str,
    jobs: list[dict],
    argv: list[str],
    *,
    env: dict[str, str] | None = None,
) -> int:
    """Replay ``jobs`` of every arch ``GPU_ARCHS`` names (every arch the jobs
    have when it names none), one process per arch, running
    ``python -m module_name --arch <arch> *argv``. The processes run side by
    side and split the worker budget by job count. Returns the number of archs
    whose replay failed."""
    module = importlib.import_module(module_name)
    arch_of = Counter(job_arch(job, _default_arch(module)) for job in jobs)
    targets = aot_target_archs()
    for arch, count in sorted(arch_of.items()):
        if targets and arch not in targets:
            print(
                f"  [SKIP] {count} {module.LABEL} jobs for {arch}, which GPU_ARCHS omits"
            )
    archs = sorted(arch for arch in arch_of if not targets or arch in targets)
    total = sum(arch_of[arch] for arch in archs)
    print(
        f"[aiter] FlyDSL {module.LABEL} replay: {total} jobs, "
        f"archs {', '.join(archs) or '-'}",
        flush=True,
    )
    if not archs:
        return 0
    workers = get_max_workers(total)
    replays = []
    for arch in archs:
        command = [sys.executable, "-m", module_name, "--arch", arch, *argv]
        share = max(1, round(workers * arch_of[arch] / total))
        process_env = replay_env(arch, share)
        process_env.update(env or {})
        replays.append(subprocess.Popen(command, env=process_env))
    return sum(replay.wait() != 0 for replay in replays)


def run_default(module_name: str) -> int:
    """Replay a module's default jobs (its ``DEFAULT_CSVS`` that exist, or its
    ``default_jobs()``), as ``setup.py``'s :func:`~.common.run_aot` does.
    Returns the number of archs whose replay failed."""
    module = importlib.import_module(module_name)
    csv_paths = [p for p in module.DEFAULT_CSVS if os.path.isfile(p)]
    return run(
        module_name,
        _jobs(module, csv_paths, None),
        ["--csv", *csv_paths] if csv_paths else [],
        env=_child_env(module, csv_paths),
    )


def _default_arch(module: Any) -> str:
    return getattr(module, "AOT_ARCH_DEFAULT", "gfx950")


def _jobs(module: Any, csv_paths: list[str], args: argparse.Namespace | None) -> list:
    """``module.collect_jobs(csv_paths, args)`` when it has one (args None for
    the defaults), else ``parse_csv`` over every path."""
    collect = getattr(module, "collect_jobs", None)
    if collect is not None:
        return collect(csv_paths, args)
    return collect_aot_jobs(csv_paths, module.parse_csv)


def _child_env(module: Any, csv_paths: list[str]) -> dict[str, str] | None:
    """What points the runtime's tuned-config lookups of the replay processes
    at ``csv_paths`` (``module.config_env``)."""
    config_env = getattr(module, "config_env", None)
    return config_env(csv_paths) if config_env is not None and csv_paths else None


def main(module_name: str, description: str) -> None:
    """The command line of a replaying AOT module: ``--csv`` (tuned tables,
    default ``DEFAULT_CSVS``), ``--check`` (compile nothing; fail on any
    kernel the cache lacks), the module's own options (``configure(parser)``)
    and the internal ``--arch`` a per-arch replay process runs with."""
    from aiter_worker_limits import adopt_legacy_max_jobs

    adopt_legacy_max_jobs()
    module = importlib.import_module(module_name)
    parser = argparse.ArgumentParser(
        description=description, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--csv",
        type=str,
        nargs="+",
        default=list(module.DEFAULT_CSVS),
        help="Path(s) to tuned CSV config file(s); defaults come from AITER_CONFIGS",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="compile nothing; fail on any kernel missing from the cache",
    )
    parser.add_argument("--arch", help=argparse.SUPPRESS)
    configure = getattr(module, "configure", None)
    if configure is not None:
        configure(parser)
    args = parser.parse_args()

    csv_paths = [os.path.abspath(p) for p in args.csv]
    for csv_path in csv_paths:
        if not os.path.isfile(csv_path):
            print(f"Error: CSV file not found: {csv_path}")
            sys.exit(1)
    if args.check:
        os.environ["FLYDSL_RUNTIME_RUN_ONLY"] = "1"
    jobs = _jobs(module, csv_paths, args)

    if args.arch:
        jobs = [
            job for job in jobs if job_arch(job, _default_arch(module)) == args.arch
        ]
        failed = jobs and replay_arch(module_name, jobs, args.arch, module.LABEL)
        sys.exit(1 if failed else 0)

    cache_dir = os.path.expanduser(
        os.environ.get("FLYDSL_RUNTIME_CACHE_DIR", "~/.flydsl/cache")
    )
    print("=" * 72)
    action = "cache check" if args.check else "pre-compilation"
    print(f"FlyDSL {module.LABEL} AOT {action}")
    print("=" * 72)
    for csv_path in csv_paths:
        print(f"  CSV:          {csv_path}")
    print(f"  Cache dir:    {cache_dir}")
    print("=" * 72, flush=True)

    argv = (["--csv", *csv_paths] if csv_paths else []) + _forwarded(parser, args)
    if args.check:
        argv.append("--check")
    started = time.time()
    failed_archs = run(module_name, jobs, argv, env=_child_env(module, csv_paths))
    print(f"\n  Total time:   {time.time() - started:.1f}s")
    if failed_archs:
        print("Some replays failed. Check output above for details.")
        sys.exit(1)
    print("All replays succeeded. Cache is ready.")
    sys.exit(0)


def _forwarded(parser: argparse.ArgumentParser, args: argparse.Namespace) -> list:
    """The module's own options, as the per-arch processes must see them."""
    argv = []
    for action in parser._actions:
        if action.dest in ("help", "csv", "check", "arch"):
            continue
        value = getattr(args, action.dest)
        if value == action.default:
            continue
        flag = action.option_strings[0]
        if isinstance(action, argparse._StoreTrueAction):
            argv.append(flag)
        elif isinstance(action, argparse._AppendAction):
            for item in value:
                argv += [flag, str(item)]
        elif isinstance(value, list):
            argv += [flag, *map(str, value)]
        else:
            argv += [flag, str(value)]
    return argv
