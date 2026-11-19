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
  runtime's tuned-config lookups at the given tables,
  ``configure(parser)`` for options of its own, ``SCAN_ARCHS``, the archs
  whose jobs have their sizes identified, ``scan_sizes(job, low)``, the
  sizes of a job's range to identify there (default: every one), and
  ``REFUSALS``, the limits the runtime meets at sizes inside a row's range
  (all below).

Its ``python -m`` entry is :func:`main`, and ``setup.py``'s build runs
:func:`run_default`. Both run one process per target arch (``replay_env``),
each replaying its jobs in a pool of workers.

For an arch in ``SCAN_ARCHS``, the sizes of a job's range that
``scan_sizes`` names (every size without it) are identified: replayed with
their launches stopped at their cache identity (:func:`identified_kernels`,
nothing compiled), and one size per set of kernels not yet built is replayed
for real. A runtime whose choices are not monotonic in the size needs that,
as a gfx1250 GEMM's are: it fits cluster_m to the number of M tiles and
specializes a ragged last tile, so M=5120 and M=6144 can request one kernel
and M=5632 another.

Otherwise, where a job's two ends request different FlyDSL kernels, the sizes
between them are bisected down to every size at which the runtime switches
kernels (as in ``moe.py``). That holds only while each choice the runtime
makes from the size is monotonic, so that two sizes that request the same
kernels bound a run that requests nothing else. An end that requests no
FlyDSL kernel at all (another backend serves it, or the runtime refuses the
size) bounds nothing, and is not bisected against.

At a size inside a row's range the runtime may meet a limit of its own, e.g.
no FlyDSL kernel that fits an untuned M. A module names each such limit in
``REFUSALS``, patterns searched in the error's ``"Type: message"``, and a
size that fails with one is reported as refused. Any other error there fails
the replay, as it does at the row itself: a dry-run stand-in that an upgrade
broke must not pass for the runtime refusing a size.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import os
import re
import subprocess
import sys
import time
import traceback
import zlib
from collections import Counter
from collections.abc import Iterable, Iterator
from contextlib import AbstractContextManager, contextmanager
from typing import Any

from aiter.aot.flydsl.common import (
    STRICT_ERRATA_TABLES,
    aot_target_archs,
    assert_targets_covered,
    check_replay_process,
    collect_aot_jobs,
    cu_num_to_arch,
    fail_build_on_missing_target_feature,
    get_max_workers,
    is_strict_identity,
    replay_env,
    resolve_target_arch,
    run_jobs_parallel,
    worker_waves,
)

# Jobs one worker replays in turn: they share tuned-table loads and kernels.
_JOBS_PER_WORKER = 16
_MAX_REASONS = 10


def job_arch(job: dict, default: str = "gfx950") -> str:
    gfx = str(job.get("gfx") or "").strip()
    return gfx or cu_num_to_arch(int(job.get("cu_num") or 0), default)


class ReplayHookBroken(RuntimeError):
    """A FlyDSL launch derived no cache key in
    ``jit_function._artifact_cache_key``: the FlyDSL internals the replay
    hooks into have changed, and it would miss the launch's kernel."""


class _Identified(BaseException):
    """Ends a launch at its cache identity. A BaseException, so that no
    ``except Exception`` on the way out of FlyDSL takes it for an error."""


def _frozen(value) -> tuple:
    """``value`` with its type, by value where hashable, else by repr."""
    try:
        hash(value)
    except TypeError:
        return type(value), repr(value)
    return type(value), value


def _trace_identity(jit, sig, bound, bound_self) -> tuple:
    """What a FlyDSL trace of ``jit`` depends on besides the launcher itself:
    the object it is bound to, the IR types of its runtime arguments (static
    shape, stride and dtype included), its Constexpr values, the compile hints
    and the target. Two launches of one ``jit`` that agree on these trace to
    the same IR, provided the trace reads no other state that changes between
    them (as FlyDSL's own key assumed before 0.5.0)."""
    from flydsl.compiler import jit_function

    _, jit_args, dsl_types, constexprs = jit_function.convert_to_jit_arguments(
        sig, bound
    )
    hints = jit._effective_compile_hints()
    return (
        jit,
        _frozen(bound_self),
        tuple(str(t) for t in jit_function.get_ir_types(jit_args)),
        tuple(dsl_types),
        tuple(sorted((k, _frozen(v)) for k, v in constexprs.items())),
        tuple(sorted((k, _frozen(v)) for k, v in hints.items())),
        _frozen(jit_function.get_backend().target),
    )


@contextmanager
def requested_kernels(*, stop: bool = False, memo: dict | None = None) -> Iterator[set]:
    """Collect the cache identity of every FlyDSL launch a replay reaches, as
    FlyDSL itself derives it for its cache lookup. With ``stop``, a launch
    ends there: it is traced, but nothing is compiled or read from the cache,
    and the launcher gets the None a COMPILE_ONLY launch returns. A ``memo``
    (with ``stop``) maps :func:`_trace_identity` to the key it traced to, so
    that a launch whose trace inputs were seen before is not traced again: a
    scan identifies thousands of sizes that request a handful of kernels.
    Share one only among launches under one environment (one dry run).

    Hooks private names of FlyDSL's ``jit_function`` module:
    ``JitFunction._compile_or_execute``, which every launch (``__call__``,
    ``flyc.compile``) goes through, ``_artifact_cache_key``, which it calls
    once after tracing, and, for a ``memo``, ``JitFunction._trace`` and the
    argument conversion it starts with. Should one be renamed, entering this
    fails; should a launch stop deriving its key there, it raises
    :class:`ReplayHookBroken`."""
    from unittest import mock

    from flydsl.compiler import jit_function
    from flydsl.compiler.jit_function import JitFunction

    kernels = set()
    build_key = jit_function._artifact_cache_key
    dispatch = JitFunction._compile_or_execute
    trace = JitFunction._trace
    derived = depth = 0
    pending = None

    def record(*args, **kwargs):
        nonlocal derived, pending
        key = build_key(*args, **kwargs)
        kernels.add(key)
        derived += 1
        if pending is not None:
            memo[pending] = key
            pending = None
        # Only a launch is stopped; host code may derive a key itself.
        if stop and depth:
            raise _Identified
        return key

    def traced(self, ctx, sig, bound, bound_self, *args, **kwargs):
        nonlocal derived, pending
        identity = _trace_identity(self, sig, bound, bound_self)
        key = memo.get(identity)
        if key is not None:
            kernels.add(key)
            derived += 1
            raise _Identified
        pending = identity
        return trace(self, ctx, sig, bound, bound_self, *args, **kwargs)

    def launch(self, *args, **kwargs):
        nonlocal depth, pending
        before = derived
        depth += 1
        try:
            result = dispatch(self, *args, **kwargs)
        except _Identified:
            return None
        finally:
            depth -= 1
            # A launch that failed between its trace and its key memoizes
            # nothing.
            pending = None
        # A launch inside another one's trace is inlined and derives no key.
        if not depth and derived == before:
            raise ReplayHookBroken(
                f"FlyDSL launched {self.func.__name__} without deriving its "
                "cache key in jit_function._artifact_cache_key"
            )
        return result

    patches = [
        mock.patch.object(jit_function, "_artifact_cache_key", record),
        mock.patch.object(JitFunction, "_compile_or_execute", launch),
    ]
    if stop and memo is not None:
        patches.append(mock.patch.object(JitFunction, "_trace", traced))
    with contextlib.ExitStack() as patched:
        for patch in patches:
            patched.enter_context(patch)
        yield kernels


def identified_kernels(memo: dict | None = None) -> AbstractContextManager[set]:
    """:func:`requested_kernels` with every launch stopped at its cache key."""
    return requested_kernels(stop=True, memo=memo)


def replay_job(
    module: Any, job: dict, probe: bool = False
) -> tuple[dict[str, Any], frozenset | None]:
    """Replay one job of ``module``; returns its result and the kernels it
    requested (None when it failed). A ``NotImplementedError`` is the runtime
    refusing the call, which is reported, not failed. So, at a ``probe`` (a
    size inside a row's range rather than the row's own), is an error the
    module names in ``REFUSALS``: a limit the runtime meets there (e.g. no
    FlyDSL kernel that fits an untuned M)."""
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
                not isinstance(error, (GpuRuntimeReached, ReplayHookBroken))
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
            reason = f"{type(error).__name__}: {error}"
            if probe and not kernels and _refused(module, reason):
                result["compile_time"] = 0.0
                result["refused"] = reason
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


def _refused(module: Any, reason: str) -> bool:
    """Whether ``reason``, an error's ``"Type: message"``, is one of the
    limits ``module.REFUSALS`` says the runtime meets inside a row's range."""
    return any(re.search(p, reason) for p in getattr(module, "REFUSALS", ()))


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


def _kernel_sets(module: Any, job: dict, sizes: Iterable[int]) -> dict:
    """The FlyDSL kernels the runtime requests at each of ``sizes``, identified
    without compiling: each distinct set, or for a size the call fails at its
    error with the numbers masked, mapped to the first size to request it."""
    import logging

    from aiter.aot.flydsl.dry_run import dry_run

    arch = job_arch(job, _default_arch(module))
    found: dict = {}
    memo: dict = {}
    logger = logging.getLogger("aiter")
    level = logger.level
    # The runtime warns at every untuned size, and a scan visits thousands.
    logger.setLevel(logging.ERROR)
    try:
        with dry_run(arch, int(job.get("cu_num") or 0), python_ops=True):
            for size in sizes:
                with identified_kernels(memo) as kernels:
                    try:
                        module.replay({**job, "m": size})
                        outcome = frozenset(kernels)
                    except ReplayHookBroken:
                        raise
                    except Exception as error:  # noqa: BLE001
                        outcome = (
                            type(error).__name__,
                            re.sub(r"\d+", "N", str(error)),
                        )
                found.setdefault(outcome, size)
    finally:
        logger.setLevel(level)
    return found


def _scan_sizes(module: Any, job: dict, low: int) -> Iterator[int]:
    """The sizes of a job's range to identify: those ``module.scan_sizes``
    names, else every one. Lazy, so that they are named inside the dry run
    that identifies them, as the runtime helpers they may use (e.g.
    ``get_padded_m``) need its fake device."""
    scan_sizes = getattr(module, "scan_sizes", None)
    yield from scan_sizes(job, low) if scan_sizes else range(low, job["m"])


def _scan(module: Any, job: dict, low: int, built: frozenset) -> list[dict]:
    """Identify the sizes from ``low`` up to the job's own (those
    ``module.scan_sizes(job, low)`` names, else every one), and replay one
    size per set of kernels ``built`` lacks, and one per kind of failure (which
    :func:`replay_job` then reports as it reports any probe)."""
    built = set(built)
    probes = []
    sizes = _scan_sizes(module, job, low)
    for outcome, size in _kernel_sets(module, job, sizes).items():
        if isinstance(outcome, frozenset) and outcome <= built:
            continue
        result, kernels = replay_job(module, {**job, "m": size}, probe=True)
        probes.append(result)
        built |= kernels or set()
    return probes


def _scans(module: Any, arch: str) -> bool:
    return arch in getattr(module, "SCAN_ARCHS", ())


def _replay_batch(module_name: str, jobs: list[dict], kernel_name: str) -> dict:
    """Replay each job at its size, then the smaller sizes it serves: every
    one of them identified where the module scans the job's arch
    (``SCAN_ARCHS``), else the smallest, bisecting between the two where they
    request different kernels."""
    module = importlib.import_module(module_name)
    results, probes = [], []
    for job in jobs:
        result, high = replay_job(module, job)
        result["row"] = True
        results.append(result)
        low_m = job.get("m_low")
        if low_m is None or low_m >= job["m"] or high is None:
            continue
        if _scans(module, job_arch(job, _default_arch(module))):
            probes += _scan(module, job, low_m, high)
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
        tag = "SCAN" if _scans(module, arch) else "BISECT"
        print(
            f"  [{tag}] {len(probes)} sizes inside rows' ranges, "
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
    side as far as the worker budget gives each a worker, and split it by job
    count (:func:`~.common.worker_waves`). Returns the number of archs whose
    replay failed."""
    module = importlib.import_module(module_name)
    arch_of = Counter(job_arch(job, _default_arch(module)) for job in jobs)
    targets = aot_target_archs()
    # Map each tuned arch to the exact requested target in its kernel family
    # (gfx1250 -> gfx1250-strict for a strict build), so a gfx1250-tuned row is
    # replayed and compiled for the exact strict target (ELF mach 0xeb, not
    # gfx1250 0x49). A tuned arch whose family GPU_ARCHS does not request is
    # skipped; a requested family present in the tables but dropped is a hard
    # error (an empty/partial wheel reported as success).
    resolved = {arch: resolve_target_arch(arch, targets) for arch in arch_of}
    for arch, count in sorted(arch_of.items()):
        if resolved[arch] is None:
            print(
                f"  [SKIP] {count} {module.LABEL} jobs for {arch}, which GPU_ARCHS omits"
            )
    target_of: Counter = Counter()
    for arch, count in arch_of.items():
        if resolved[arch] is not None:
            target_of[resolved[arch]] += count
    archs = sorted(target_of)
    assert_targets_covered(targets, list(arch_of), archs, f"FlyDSL {module.LABEL} AOT")
    total = sum(target_of.values())
    print(
        f"[aiter] FlyDSL {module.LABEL} replay: {total} jobs, "
        f"archs {', '.join(archs) or '-'}",
        flush=True,
    )
    if not archs:
        return 0
    failed = 0
    waves = worker_waves(get_max_workers(total), dict(target_of))
    for wave in waves:
        replays = []
        for arch, share in wave.items():
            command = [sys.executable, "-m", module_name, "--arch", arch, *argv]
            process_env = replay_env(arch, share)
            process_env.update(env or {})
            replays.append(subprocess.Popen(command, env=process_env))
        failed += sum(replay.wait() != 0 for replay in replays)
    return failed


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
    and the internal ``--arch`` a per-arch replay process runs with. The
    per-arch processes get the same command line, plus ``--arch``."""
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
    argv = sys.argv[1:]
    args = parser.parse_args(argv)

    csv_paths = [os.path.abspath(p) for p in args.csv]
    for csv_path in csv_paths:
        if not os.path.isfile(csv_path):
            print(f"Error: CSV file not found: {csv_path}")
            sys.exit(1)
    if args.check:
        os.environ["FLYDSL_RUNTIME_RUN_ONLY"] = "1"
    jobs = _jobs(module, csv_paths, args)

    if args.arch:
        # args.arch is the exact requested target; keep rows whose tuned arch
        # shares its family and retarget each so it compiles for that arch
        # (a gfx1250 row -> gfx1250-strict -> 0xeb).
        kept = []
        for job in jobs:
            cur = job_arch(job, _default_arch(module))
            if resolve_target_arch(cur, {args.arch}) != args.arch:
                continue
            # gfx1250-strict disables the WMMA features these tables need
            # (A0 errata); compiled for strict they are hollow, so skip them.
            if (
                is_strict_identity(args.arch)
                and job.get("table") in STRICT_ERRATA_TABLES
            ):
                continue
            if cur != args.arch:
                job = {**job, "gfx": args.arch}
            kept.append(job)
        jobs = kept
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

    started = time.time()
    with fail_build_on_missing_target_feature():
        failed_archs = run(module_name, jobs, argv, env=_child_env(module, csv_paths))
    print(f"\n  Total time:   {time.time() - started:.1f}s")
    if failed_archs:
        print("Some replays failed. Check output above for details.")
        sys.exit(1)
    print("All replays succeeded. Cache is ready.")
    sys.exit(0)
