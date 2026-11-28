#!/usr/bin/env python3

# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

import ctypes
import fcntl
import functools
import gc
import importlib
import json
import multiprocessing
import os
import random
import re
import shutil
import signal
import sys
import tempfile
import threading
import time
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import wait as wait_for_stages
from contextlib import contextmanager
from multiprocessing.connection import wait as wait_for_sentinels
from typing import Any

from aiter.jit.utils.build_targets import base_gfx_arch
from aiter_worker_limits import (
    configure_worker_subprocesses,
    get_fork_worker_count_for,
    get_worker_count_for,
)

_DEFAULT_KERNEL_TIMEOUT = 1200.0
_DEFAULT_MAX_RETRIES = 2
# The directory of the worker slots that the pools of one build share
# (:func:`worker_slots`). Unset, each pool runs its own budget of workers.
_SLOTS_ENV = "AITER_FLYDSL_AOT_SLOTS"
# How often a pool that has work, but found every slot taken, looks again.
_SLOT_POLL_SECONDS = 0.25
# Workers of the shared budget per stage that replays side by side. A stage
# brings a replay process per arch (about 0.5 GB private each, up to three),
# which the worker estimate's margin over what a worker holds (about half of
# it) pays for at eight workers. Below sixteen workers the stages therefore
# run one after another.
_WORKERS_PER_STAGE = 8


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
    site (naming the missing artifact and its cache dir) instead of silently
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


def aot_target_archs() -> set[str] | None:
    """The archs ``GPU_ARCHS`` names, as the wheel build sets it; ``None``
    when it is unset or ``native``, meaning every arch the tables have."""
    value = os.environ.get("GPU_ARCHS", "").strip()
    if not value or value.lower() == "native":
        return None
    return {arch.strip() for arch in value.split(";") if arch.strip()}


def resolve_target_arch(job_arch: str, targets: set[str] | None) -> str | None:
    """Exact requested target arch in the same kernel family as ``job_arch``, or
    ``None`` when no requested target shares its family.

    ``gfx1250-strict`` (MI450 A0) and ``gfx1250`` (B0) are one source/kernel
    family, so a ``gfx1250``-tuned row is selected for a ``gfx1250-strict`` build
    and **compiled for the exact requested target** (producing the strict
    ``0xeb`` code object, not ``gfx1250`` ``0x49``). ``targets`` is ``None``
    (GPU_ARCHS unset/native) keeps ``job_arch`` as-is.
    """
    if not targets:
        return job_arch
    fam = base_gfx_arch(job_arch)
    for t in sorted(targets):
        if base_gfx_arch(t) == fam:
            return t
    return None


def assert_targets_covered(
    targets: set[str] | None, available_archs, matched_archs, label: str
) -> None:
    """Raise when a requested arch's kernel family exists in the tuned data but
    produced no job -- the strict-AOT filter bug that silently emitted an empty
    wheel while reporting success (``GPU_ARCHS=gfx1250-strict`` -> 0 artifacts,
    "succeeded"). A family genuinely absent from the tables (the op has no
    kernels for that arch) is fine and not an error.
    """
    if not targets:
        return
    avail_fams = {base_gfx_arch(a) for a in available_archs}
    matched_fams = {base_gfx_arch(a) for a in matched_archs}
    dropped = sorted(
        t
        for t in targets
        if base_gfx_arch(t) in avail_fams and base_gfx_arch(t) not in matched_fams
    )
    if dropped:
        raise RuntimeError(
            f"[aiter] {label}: tuned rows exist for {dropped}'s kernel family but "
            f"none were selected for the requested target; refusing to emit a "
            f"partial wheel (strict-AOT filter bug)"
        )


def replay_env(arch: str, workers: int | None = None) -> dict[str, str]:
    """Environment of a process that replays the runtime for ``arch``. aiter
    fixes some arch-dependent values at import (e.g. ``dtypes.fp8``), so the
    process starts with ``GPU_ARCHS`` naming ``arch`` and with
    ``AITER_AOT_REPLAY=1``, under which aiter's runtime arch queries answer
    ``arch`` instead of probing a device. The GPUs stay hidden, and aiter is
    imported in full rather than in AOT mode."""
    env = dict(os.environ)
    env.pop("AITER_AOT_IMPORT", None)
    env.update(
        GPU_ARCHS=arch,
        AITER_AOT_REPLAY="1",
        # An empty list hides every GPU. "-1" would too, but ROCm 10's
        # rocprofiler-sdk aborts any process it is loaded into on "-1".
        HIP_VISIBLE_DEVICES="",
        ROCR_VISIBLE_DEVICES="",
        CUDA_VISIBLE_DEVICES="",
    )
    if workers is not None:
        env["AITER_MAX_JOBS"] = str(workers)
    return env


def check_replay_process(arch: str) -> None:
    """Raise unless this process started with ``replay_env(arch)``, so that
    what aiter fixed at import belongs to ``arch``. Such a process is started
    by a build, and ends with it (:func:`die_with_parent`)."""
    die_with_parent()
    from aiter.utility import dtypes

    if os.environ.get("AITER_AOT_REPLAY") != "1" or os.environ.get("GPU_ARCHS") != arch:
        raise RuntimeError(
            f"replaying {arch} needs AITER_AOT_REPLAY=1 and GPU_ARCHS={arch} set "
            "before aiter is imported (see replay_env)"
        )
    expected = dtypes.defaultDtypes[arch]["fp8"]
    if dtypes.fp8 is not expected:
        raise RuntimeError(
            f"aiter was imported for another arch: dtypes.fp8 is {dtypes.fp8}, "
            f"{arch} needs {expected}"
        )


def die_with_parent(sig: int = signal.SIGTERM, parent_pid: int | None = None) -> None:
    """Have the kernel send this process ``sig`` when its parent dies (Linux;
    a no-op elsewhere), and exit now if ``parent_pid`` is given and is already
    gone.

    A replay process supervises its workers (deadlines, retries) and a build
    its replay processes. Without this, killing one of them (an OOM kill, a
    cancelled build) leaves what it started running unsupervised, and a worker
    left so keeps its worker slot from every other pool of the build."""
    try:
        ctypes.CDLL(None, use_errno=True).prctl(
            1, int(sig), 0, 0, 0
        )  # PR_SET_PDEATHSIG
    except (AttributeError, OSError):
        return
    if parent_pid is not None and os.getppid() != parent_pid:
        os._exit(1)


def _run_one_to_file(
    worker: Callable[..., dict[str, Any]],
    kwargs: dict[str, Any],
    out_path: str,
    parent_pid: int | None = None,
) -> None:
    if parent_pid is not None:  # forked by the pool, which supervises it
        die_with_parent(signal.SIGKILL, parent_pid)
    # A pool this worker starts is its own: it must not wait for a slot of the
    # build's, of which the worker holds one.
    os.environ.pop(_SLOTS_ENV, None)
    configure_worker_subprocesses()
    result = worker(**kwargs)
    tmp_path = out_path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(result, f)
    os.replace(tmp_path, out_path)


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


def get_max_workers(num_jobs: int, start_method: str = "fork") -> int:
    """Workers of a FlyDSL AOT pool. Forked ones are budgeted by what each adds
    to the process it shares its loaded modules and jobs with; those of another
    start method load their own, and take the general per-worker estimate.

    Where the pools of a build share worker slots (:func:`worker_slots`), a
    fork pool may run as many workers as there are slots: the slots, which
    were budgeted once for all of them, are what bounds it."""
    if start_method == "fork":
        slots = _slot_count()
        if slots:
            return max(1, min(int(num_jobs), slots))
        return get_fork_worker_count_for(num_jobs)
    return get_worker_count_for(num_jobs)


def _slot_dir() -> str | None:
    return os.environ.get(_SLOTS_ENV) or None


def _slot_count() -> int:
    """The worker slots this process shares with other pools; 0 for none."""
    slot_dir = _slot_dir()
    if slot_dir is None:
        return 0
    try:
        return len(os.listdir(slot_dir))
    except OSError as e:
        raise RuntimeError(
            f"{_SLOTS_ENV}={slot_dir} names no directory of worker slots: {e}"
        ) from e


def _acquire_slot(slot_dir: str, count: int) -> int | None:
    """Take a free worker slot: an open file holding its lock, or None when
    every slot is taken. The lock lasts as long as a process holds the file
    open, so a worker forked with it keeps the slot until it exits, however it
    exits, once this process has closed its own copy."""
    first = random.randrange(count)
    for offset in range(count):
        path = os.path.join(slot_dir, str((first + offset) % count))
        try:
            fd = os.open(path, os.O_RDWR)
        except OSError as e:
            raise RuntimeError(
                f"worker slot {path} is gone; was the build that owns "
                f"{_SLOTS_ENV} stopped?"
            ) from e
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(fd)
            continue
        except BaseException:
            os.close(fd)
            raise
        return fd
    return None


@contextmanager
def worker_slots(count: int) -> Iterator[str]:
    """Share ``count`` workers among every fork pool started below, in this
    process or a child: each worker holds one slot while it runs. Pools then
    run side by side within one budget, and a pool that finishes leaves its
    workers to the others, where a split made up front would leave them idle."""
    slot_dir = tempfile.mkdtemp(prefix="aot_slots_")
    previous = os.environ.get(_SLOTS_ENV)
    try:
        for slot in range(max(1, int(count))):
            with open(os.path.join(slot_dir, str(slot)), "w"):
                pass
        os.environ[_SLOTS_ENV] = slot_dir
        yield slot_dir
    finally:
        if previous is None:
            os.environ.pop(_SLOTS_ENV, None)
        else:
            os.environ[_SLOTS_ENV] = previous
        shutil.rmtree(slot_dir, ignore_errors=True)


@contextmanager
def shared_worker_slots(count: int) -> Iterator[str]:
    """The worker slots this process already shares (an enclosing build's),
    else ``count`` new ones for what it starts."""
    slot_dir = _slot_dir()
    if slot_dir is not None:
        yield slot_dir
        return
    with worker_slots(count) as slot_dir:
        yield slot_dir


def stages_side_by_side(workers: int, stages: int) -> int:
    """How many of ``stages`` replay at once on a budget of ``workers``."""
    return max(1, min(int(stages), int(workers) // _WORKERS_PER_STAGE))


def worker_waves(workers: int, jobs_per_arch: dict[str, int]) -> list[dict[str, int]]:
    """The per-arch replay processes to run side by side, wave after wave, and
    the workers each runs, such that no wave runs more than ``workers`` in all:
    as many archs as the budget gives a worker each, the largest first, the
    budget split among them by job count (largest remainder)."""
    archs = sorted(jobs_per_arch, key=lambda arch: (-jobs_per_arch[arch], arch))
    workers = max(1, workers)
    waves = []
    for start in range(0, len(archs), workers):
        wave = archs[start : start + workers]
        total = sum(jobs_per_arch[arch] for arch in wave)
        extra = {
            arch: (workers - len(wave)) * jobs_per_arch[arch] / total for arch in wave
        }
        shares = {arch: 1 + int(extra[arch]) for arch in wave}
        rest = workers - sum(shares.values())
        by_remainder = sorted(wave, key=lambda arch: int(extra[arch]) - extra[arch])
        for arch in by_remainder[:rest]:
            shares[arch] += 1
        waves.append(shares)
    return waves


def _trim_heap() -> None:
    """Return the allocator's free memory to the OS (glibc; a no-op elsewhere)."""
    try:
        ctypes.CDLL(None).malloc_trim(0)
    except (AttributeError, OSError):
        pass


@contextmanager
def _lean_fork_parent(start_method: str) -> Iterator[None]:
    """Keep what forked workers inherit shared with this process, and small.

    A worker copies every page it writes to. A garbage collection writes to
    the header of each object it tracks, so the first one in a worker copies
    most of the inherited heap: freezing the objects alive now keeps them out
    of the workers' collections (a worker's median private memory from 0.34 GB
    to 0.23 GB for MegaMoE, from 0.17 GB to 0.09 GB for MoE). The heap also
    still holds what collecting and batching the jobs freed, which each
    worker's RSS would count again: trimming returns it (GEMM: 0.6 GB off this
    process, 0.4 GB off each worker's RSS).

    The fork-pool worker estimate (``EST_FORK_WORKER_BYTES``) was measured
    with both in place."""
    if start_method != "fork":
        yield
        return
    gc.collect()
    _trim_heap()
    if gc.get_freeze_count():
        # The caller froze objects of its own. Unfreezing is all or nothing,
        # so freezing here could not be undone without undoing the caller's.
        yield
        return
    gc.freeze()
    try:
        yield
    finally:
        gc.unfreeze()


def _run_file_pool(
    specs: list[tuple[Callable[..., dict[str, Any]], dict[str, Any], Any]],
    max_workers: int,
    kernel_timeout: float,
    max_retries: int,
    result_dir: str,
    start_method: str = "fork",
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
    # Slots shared with other pools; only a forked worker inherits one.
    forked = start_method == "fork"
    slot_dir = _slot_dir() if forked else None
    slot_count = _slot_count() if slot_dir else 0

    def launch() -> bool:
        """Start workers; False if work is left only for want of a free slot."""
        while queue and len(running) < max_workers:
            slot = _acquire_slot(slot_dir, slot_count) if slot_dir else None
            if slot_dir and slot is None:
                return False
            try:
                start(queue.pop())
            finally:
                if slot is not None:
                    os.close(slot)  # the worker holds it now
        return True

    def start(idx: int) -> None:
        worker, kwargs, _label = specs[idx]
        out_path = os.path.join(result_dir, f"k{idx}.json")
        try:
            os.remove(out_path)  # clear any stale file from a prior attempt
        except OSError:
            pass
        proc = ctx.Process(
            target=_run_one_to_file,
            args=(worker, kwargs, out_path, os.getpid() if forked else None),
        )
        proc.start()
        deadline = (time.monotonic() + kernel_timeout) if kernel_timeout > 0 else None
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
            except Exception:  # noqa: BLE001
                result = None
        if result is None:
            label = specs[idx][2]
            result = {"kernel_name": str(label), "compile_time": None}
        results[idx] = result
        note_done()

    try:
        launched = launch()
        while running or queue:
            wait_timeout: float | None = None
            if kernel_timeout > 0 and running:
                nearest = min(d for (_, d) in running.values() if d is not None)
                wait_timeout = max(0.0, nearest - time.monotonic())
            if not launched:
                # No worker's exit tells that another pool freed a slot.
                wait_timeout = (
                    _SLOT_POLL_SECONDS
                    if wait_timeout is None
                    else min(wait_timeout, _SLOT_POLL_SECONDS)
                )
            if running:
                wait_for_sentinels([p.sentinel for p in running], timeout=wait_timeout)
            else:
                time.sleep(wait_timeout or 0.0)

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

            # Refill freed slots (including any just-requeued retries).
            launched = launch()
    finally:
        # Kill any survivors (e.g. on an unexpected exception) so we never leave
        # orphaned compilers blocking the caller's exit.
        for proc in list(running):
            try:
                if proc.is_alive():
                    proc.kill()
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
    *,
    start_method: str = "fork",
) -> list[dict[str, Any]]:
    if not jobs:
        return []
    max_workers = get_max_workers(len(jobs), start_method)
    print(
        f"[aiter] FlyDSL AOT: {len(jobs)} kernels, {max_workers} worker processes",
        flush=True,
    )
    result_dir = tempfile.mkdtemp(prefix="aot_results_")
    try:
        specs = [(worker, job, str(job.get("kernel_name", "?"))) for job in jobs]
        # For the whole run: workers start as others finish.
        with _lean_fork_parent(start_method):
            raw = _run_file_pool(
                specs,
                max_workers,
                get_kernel_timeout(),
                get_max_retries(),
                result_dir,
                start_method,
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


# The FlyDSL AOT modules setup.py runs, each replaying the runtime of its ops.
AOT_MODULES = (
    "aiter.aot.flydsl.gemm",
    "aiter.aot.flydsl.conv",
    "aiter.aot.flydsl.chunk_gdn_h",
    "aiter.aot.flydsl.gdn_fused_prefill",
    "aiter.aot.flydsl.fmha_fp8",
    "aiter.aot.flydsl.pa_decode_plan",
)


STRICT_ERRATA_TABLES = frozenset(
    {"a8w8_bpreshuffle_tuned_gemm", "a4w4_mxscale_bpreshuffle_tuned_gemm"}
)


def is_strict_identity(arch: str) -> bool:
    """True for a gfx1250-strict-family arch (not the plain gfx1250 base)."""
    return base_gfx_arch(arch) != arch


@contextmanager
def fail_build_on_missing_target_feature():
    """Fail an AOT build that dropped a matmul for a disabled target feature.

    gfx1250-strict disables some WMMA features (F4, fp8 16x16x128) for A0 errata;
    a kernel that needs one compiles to a hollow, compute-less object with rc 0
    and the LLVM diagnostic 'requires target feature ...'. That line is written
    to the real stderr fd by the C++ backend (a Python-level redirect misses it),
    so this captures fd 2, tees it back, and raises if the marker appears --
    refusing to ship a hollow wheel. An errata op reaching here was not gated off
    the strict build (see STRICT_ERRATA_TABLES / the MoE mxfp4 skip)."""
    import sys
    import threading

    if os.environ.get("AITER_AOT_FEATURE_GUARD") == "1":
        # An enclosing build already captures fd 2 (this process or a parent,
        # via the inherited env); a second capture would double-handle it.
        yield
        return

    marker = b"requires target feature"
    r_fd, w_fd = os.pipe()
    real_err = os.dup(2)
    hit = []

    def pump():
        buf = b""
        while True:
            try:
                chunk = os.read(r_fd, 65536)
            except OSError:
                break
            if not chunk:
                break
            os.write(real_err, chunk)
            buf = (buf + chunk)[-(len(marker) + 4096) :]
            if marker in buf:
                hit.append(True)

    t = threading.Thread(target=pump, daemon=True)
    t.start()
    os.environ["AITER_AOT_FEATURE_GUARD"] = "1"
    sys.stderr.flush()
    os.dup2(w_fd, 2)
    os.close(w_fd)
    try:
        yield
    finally:
        os.environ.pop("AITER_AOT_FEATURE_GUARD", None)
        sys.stderr.flush()
        os.dup2(real_err, 2)
        os.close(real_err)
        t.join(timeout=10)
        os.close(r_fd)
    if hit:
        raise AssertionError(
            "[aiter] AOT build dropped a matmul for a missing target feature "
            "('requires target feature' in the compiler output): a gfx1250-strict "
            "build produced hollow, compute-less kernels. An errata op was not "
            "gated off the strict build. Refusing to ship a hollow wheel."
        )


_STRICT_ERRATA_KERNEL_RE = re.compile(
    r"@((?:gemm_a4w4|gemm_a8w8_ptpc|mxfp4_tdm_fp4)\w*)"
)
_ROCDL_CHIP_RE = re.compile(r'#rocdl\.target<[^>]*chip\s*=\s*"([^"]+)"')


def assert_no_strict_errata_products(cache_dir: str) -> None:
    """For a strict build, fail if the cache still holds an errata-family kernel.

    A stale hollow object left by an earlier build survives in the persisted
    FlyDSL cache, and the compile-time guard cannot see it (a cache hit recompiles
    nothing and prints no diagnostic). Cache paths carry the launcher name and a
    hash, not the kernel (ptpc shares launch_gemm_a8w8 with mx128, fp4 MoE shares
    launch_gemm_a8w4_tdm with fp8), so read the kernels inside each object; and
    judge only objects built for a strict target, so a mixed gfx1250 + strict
    build keeps its legitimate gfx1250 ones."""
    import glob
    import pickle

    targets = aot_target_archs()
    if not targets or not any(is_strict_identity(t) for t in targets):
        return
    found: list[str] = []
    for path in glob.glob(os.path.join(cache_dir, "*", "*.pkl")):
        try:
            with open(path, "rb") as fh:
                ir = getattr(pickle.load(fh), "_ir_text", "") or ""
        except Exception:  # noqa: BLE001, S112  not a readable FlyDSL object
            continue
        if not any(is_strict_identity(chip) for chip in _ROCDL_CHIP_RE.findall(ir)):
            continue
        match = _STRICT_ERRATA_KERNEL_RE.search(ir)
        if match:
            found.append(f"{os.path.basename(os.path.dirname(path))}:{match.group(1)}")
    if found:
        raise AssertionError(
            f"[aiter] gfx1250-strict AOT cache holds {len(found)} errata-family "
            "kernel object(s) that strict gates off -- likely stale hollow kernels "
            "a cache hit left from an earlier build. Clear the FlyDSL cache before a "
            f"strict wheel build. Offenders (first 3): {sorted(found)[:3]}"
        )


def run_aot(cache_dir: str) -> None:
    """Build the FlyDSL AOT cache of every module, as ``setup.py`` does. Each
    replays aiter's runtime, which needs aiter fully imported for the target
    arch, so each arch replays in a process of its own.

    The modules are stages of one build. They replay side by side, as many
    as the worker budget carries (:func:`stages_side_by_side`), and their
    pools share that budget worker by worker (:func:`worker_slots`). The MoE
    stage first builds the C++ modules every replay imports, so the others
    start once it has."""
    os.makedirs(cache_dir, exist_ok=True)
    os.environ["FLYDSL_RUNTIME_CACHE_DIR"] = cache_dir

    from .mega_moe import run as run_mega_moe_replay
    from .moe import DEFAULT_CSVS as MOE_CSVS
    from .moe import run as run_moe_replay
    from .replay import run_default

    # Imported here, in turn: the stage threads below would import them at once.
    for module_name in AOT_MODULES:
        importlib.import_module(module_name)

    host_modules_built = threading.Event()

    def moe_stage() -> str | None:
        try:
            failed = run_moe_replay(
                [p for p in MOE_CSVS if os.path.isfile(p)],
                prepared=host_modules_built.set,
            )
        finally:
            host_modules_built.set()  # also when it failed, or had no jobs
        return f"FlyDSL MOE replay failed for {failed} arch(es)" if failed else None

    def mega_moe_stage() -> str | None:
        host_modules_built.wait()
        return "FlyDSL MEGA_MOE replay failed" if run_mega_moe_replay() else None

    def module_stage(module_name: str) -> str | None:
        host_modules_built.wait()
        failed = run_default(module_name)
        label = module_name.rsplit(".", 1)[-1].upper()
        return f"FlyDSL {label} replay failed for {failed} arch(es)" if failed else None

    stages = [moe_stage, mega_moe_stage]
    stages += [functools.partial(module_stage, name) for name in AOT_MODULES]
    workers = get_max_workers(sys.maxsize)
    side_by_side = stages_side_by_side(workers, len(stages))
    print(
        f"[aiter] FlyDSL AOT: {len(stages)} stages, {side_by_side} side by side, "
        f"{workers} workers in all",
        flush=True,
    )
    with fail_build_on_missing_target_feature():
        # Every stage runs to its end, also after another failed: the pools of
        # a stage cannot be stopped from here. Then every failure is reported.
        with shared_worker_slots(workers):
            pool = ThreadPoolExecutor(side_by_side)
            try:
                outcomes = [pool.submit(stage) for stage in stages]
                wait_for_stages(outcomes)
            finally:
                # Reached before all are done only when interrupted: the stages
                # not yet started must not start then.
                pool.shutdown(wait=True, cancel_futures=True)
        errors, first_error = [], None
        for outcome in outcomes:
            try:
                error = outcome.result()
            except Exception as e:  # noqa: BLE001
                error = f"{type(e).__name__}: {e}"
                first_error = first_error or e
            if error:
                errors.append(error)
        if errors:
            raise AssertionError(
                f"[aiter] FlyDSL AOT failures: {'; '.join(errors)}"
            ) from first_error

    assert_no_strict_errata_products(cache_dir)
