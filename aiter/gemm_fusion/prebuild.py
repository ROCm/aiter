# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Ahead-of-time build of every kernel a gemm_fusion profile needs, in ONE parallel pass (torch-free, no GPU).

``prebuild_profile(profile, *, arch=None, cu=None, world=None)`` enumerates the profile's rows, lets each variant class turn its rows into build UNITS (a unit = one cache directory
= one cache key, made of TASKS: compile steps with dependencies) and runs all tasks of all units in one scheduler, so the wall is the longest dependency chain (or the total
CPU time / cores), not the sum of per-call builds.

Rank coordination (one process builds, the others wait): every unit has an atomic lock directory (mkdir) under <jit root>/.locks. A process CLAIMS a unit only when it has an
idle worker slot (demand-driven), so concurrent ranks split the units between them instead of one rank taking everything or all of them compiling everything. A process that
does not hold a unit's lock polls for the unit's completion marker; the lock holder touches its lock about once a second while it runs, and a lock untouched for STALE_S seconds
is stolen (a crashed rank cannot hang the others). A finished unit directory is published with an atomic rename, so a reader never sees a half-built cache entry.

Idempotent: a unit whose marker exists is "cached" and costs one stat().
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import threading
import time
import uuid
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

log = logging.getLogger("aiter.gemm_fusion")

STALE_S = 120.0  # a lock not touched for this long belongs to a dead process
WAIT_TIMEOUT_S = 3600.0  # a unit that neither completes nor frees its lock for this long fails the wait
Runner = Callable[[Sequence[str]], str]


class BuildError(RuntimeError):
    pass


# ----------------------------------------------------------------------------------------------------------------------------- model
@dataclass
class Task:
    """One schedulable step. ``run(tmp, run_cmd)`` does the work inside the unit's temporary directory; ``run_cmd(argv)`` executes one command and returns its output.
    ``cost`` is a relative wall estimate used only for scheduling priority (critical path first)."""

    name: str
    run: Callable[[str, Runner], None]
    deps: Tuple[str, ...] = ()
    cost: float = 1.0


@dataclass
class Unit:
    """One cache entry: <root>/<dirname>/ made by `tasks`, complete when <marker> exists in it. finalize(tmp, command_log) runs after the last task (writes the marker)."""

    kind: str
    label: str
    key: str
    dirname: str
    root: str
    tasks: List[Task]
    marker: str = "BUILD_OK"
    finalize: Optional[Callable[[str, List[str]], None]] = None
    result: object = None
    meta: dict = field(default_factory=dict)  # the planner's inputs (defs, source, ...) for tools that rebuild one task

    @property
    def final(self) -> str:
        return os.path.join(self.root, self.dirname)

    @property
    def ok_path(self) -> str:
        return os.path.join(self.final, self.marker)

    def priorities(self) -> Dict[str, float]:
        """Critical-path length from each task to the end of the unit (own cost + the most expensive chain of its successors)."""
        succ: Dict[str, List[str]] = {t.name: [] for t in self.tasks}
        for t in self.tasks:
            for d in t.deps:
                succ[d].append(t.name)
        by = {t.name: t for t in self.tasks}
        memo: Dict[str, float] = {}

        def prio(n: str) -> float:
            if n not in memo:
                memo[n] = by[n].cost + max((prio(s) for s in succ[n]), default=0.0)
            return memo[n]

        return {t.name: prio(t.name) for t in self.tasks}

    def critical_cost(self) -> float:
        return max(self.priorities().values(), default=0.0)


@dataclass
class UnitReport:
    name: str
    kind: str
    label: str
    key: str
    status: str  # cached | built | waited | failed
    wall_s: float = 0.0
    cpu_s: float = 0.0


@dataclass
class Report:
    wall_s: float = 0.0
    jobs: int = 1
    units: List[UnitReport] = field(default_factory=list)
    tasks: Dict[str, float] = field(default_factory=dict)  # "<unit dirname>/<task>" -> wall seconds

    def keys(self) -> List[str]:
        return [u.key for u in self.units if u.status != "failed"]

    def summary(self, top: int = 12) -> str:
        built = [u for u in self.units if u.status == "built"]
        cpu = sum(self.tasks.values())
        crit = max((v for v in self.tasks.values()), default=0.0)
        lines = [f"prebuild wall={self.wall_s:.1f}s jobs={self.jobs} units={len(self.units)} (built={len(built)} cached={sum(u.status == 'cached' for u in self.units)} "
                 f"waited={sum(u.status == 'waited' for u in self.units)}) task_cpu={cpu:.1f}s slowest_task={crit:.1f}s lower_bound={max(crit, cpu / max(1, self.jobs)):.1f}s"]
        for n, s in sorted(self.tasks.items(), key=lambda kv: -kv[1])[:top]:
            lines.append(f"  {s:7.2f}s  {n}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {"wall_s": self.wall_s, "jobs": self.jobs, "tasks": dict(self.tasks), "units": [vars(u) for u in self.units]}


# ----------------------------------------------------------------------------------------------------------------------------- jobs / runner
def default_jobs(env=None) -> int:
    """Workers for this process: MAX_JOBS (the aiter convention) else the usable cores, divided by the processes that share the node (LOCAL_WORLD_SIZE, else WORLD_SIZE), so
    the ranks together use about all cores instead of oversubscribing them."""
    env = os.environ if env is None else env
    try:
        cores = len(os.sched_getaffinity(0))
    except AttributeError:
        cores = os.cpu_count() or 1
    n = int(env["MAX_JOBS"]) if env.get("MAX_JOBS") else cores
    share = int(env.get("LOCAL_WORLD_SIZE") or env.get("WORLD_SIZE") or 1)
    return max(1, n // max(1, share))


class SubprocessRunner:
    """Runs argv lists; keeps the live processes so an abort can terminate them (fail fast)."""

    def __init__(self):
        self._lock = threading.Lock()
        self._procs: set = set()
        self.aborted = False

    def __call__(self, argv: Sequence[str]) -> str:
        if self.aborted:
            raise BuildError("aborted")
        p = subprocess.Popen(list(argv), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        with self._lock:
            self._procs.add(p)
        try:
            out, _ = p.communicate()
        finally:
            with self._lock:
                self._procs.discard(p)
        if p.returncode != 0:
            raise BuildError(f"command failed ({p.returncode}): {' '.join(argv)}\n{(out or '')[-4000:]}")
        return out

    def abort(self):
        self.aborted = True
        with self._lock:
            for p in list(self._procs):
                try:
                    p.terminate()
                except OSError:
                    pass


# ----------------------------------------------------------------------------------------------------------------------------- locks
class _Lock:
    """Atomic directory lock with a heartbeat (mtime) so a dead holder is detected."""

    def __init__(self, path: str):
        self.path = path
        self.held = False

    def try_acquire(self, stale_s: float) -> bool:
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        for _ in range(2):
            try:
                os.mkdir(self.path)
            except FileExistsError:
                try:
                    age = time.time() - os.path.getmtime(self.path)
                except OSError:
                    continue  # released between mkdir and stat: retry
                if age <= stale_s:
                    return False
                log.warning("event=prebuild stale_lock path=%s age_s=%.0f: stealing", self.path, age)
                shutil.rmtree(self.path, ignore_errors=True)
                continue
            try:
                with open(os.path.join(self.path, "owner"), "w") as f:
                    f.write(f"pid={os.getpid()}\n")
            except OSError:
                pass
            self.held = True
            return True
        return False

    def touch(self):
        try:
            os.utime(self.path, None)
        except OSError:
            pass

    def release(self):
        if self.held:
            shutil.rmtree(self.path, ignore_errors=True)
            self.held = False


class _State:
    def __init__(self, unit: Unit):
        self.unit = unit
        self.status = "peer"  # peer (not claimed) | mine | cached | waited | built | failed
        self.lock = _Lock(os.path.join(unit.root, ".locks", unit.dirname + ".lock"))
        self.tmp: Optional[str] = None
        self.pending: Dict[str, Task] = {}
        self.started: set = set()
        self.done: set = set()
        self.log: List[str] = []
        self.log_lock = threading.Lock()
        self.prio: Dict[str, float] = {}
        self.t0 = time.monotonic()
        self.cpu = 0.0


# ----------------------------------------------------------------------------------------------------------------------------- scheduler
def execute(units: Sequence[Unit], jobs: Optional[int] = None, *, runner: Optional[Runner] = None, stale_s: float = STALE_S,
            wait_timeout_s: float = WAIT_TIMEOUT_S, poll_s: float = 0.2) -> Report:
    """Build every unit whose marker is missing, all tasks in one pool of `jobs` workers, critical path first. Raises BuildError on the first failure (running commands are
    terminated, this process's temporary directories and locks are removed; another process may then take the units over)."""
    t_start = time.monotonic()
    jobs = jobs or default_jobs()
    runner = runner or SubprocessRunner()
    states = [_State(u) for u in sorted(units, key=lambda u: -u.critical_cost())]  # biggest units are claimed first (LPT)
    report = Report(jobs=jobs)
    for st in states:
        if os.path.exists(st.unit.ok_path):
            st.status = "cached"
    running: Dict[object, Tuple[_State, Task, float]] = {}
    first_err: Optional[BaseException] = None
    last_beat = 0.0
    wait0 = time.monotonic()
    pool = ThreadPoolExecutor(max_workers=jobs)

    def claim_one() -> bool:
        for st in states:
            if st.status != "peer":
                continue
            if os.path.exists(st.unit.ok_path):
                st.status = "waited"
                continue
            if not st.lock.try_acquire(stale_s):
                continue
            if os.path.exists(st.unit.ok_path):  # finished by a peer between our check and the lock
                st.lock.release()
                st.status = "waited"
                continue
            os.makedirs(st.unit.root, exist_ok=True)
            st.tmp = os.path.join(st.unit.root, f"{st.unit.dirname}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp")  # unique; plain mkdir keeps the inherited permissions
            os.mkdir(st.tmp)
            st.pending = {t.name: t for t in st.unit.tasks}
            st.prio = st.unit.priorities()
            st.status, st.t0 = "mine", time.monotonic()
            log.info("event=prebuild claim unit=%s kind=%s label=%s", st.unit.dirname, st.unit.kind, st.unit.label)
            return True
        return False

    def ready() -> List[Tuple[float, _State, Task]]:
        out = []
        for st in states:
            if st.status != "mine":
                continue
            for n, t in st.pending.items():
                if n not in st.started and all(d in st.done for d in t.deps):
                    out.append((st.prio[n], st, t))
        out.sort(key=lambda x: -x[0])
        return out

    def run_task(st: _State, t: Task):
        def run_cmd(argv: Sequence[str]) -> str:
            with st.log_lock:
                st.log.append(" ".join(argv))
            return runner(argv)

        t0 = time.monotonic()
        t.run(st.tmp, run_cmd)
        return time.monotonic() - t0

    def publish(st: _State):
        u = st.unit
        if u.finalize:
            u.finalize(st.tmp, st.log)
        with open(os.path.join(st.tmp, "build-command.txt"), "w") as f:
            f.write("\n".join(st.log) + "\n")
        try:
            os.rename(st.tmp, u.final)
        except OSError:  # another process published first
            if not os.path.exists(u.ok_path):
                raise
            shutil.rmtree(st.tmp, ignore_errors=True)
        st.tmp = None
        st.lock.release()
        st.status = "built"
        report.units.append(UnitReport(u.dirname, u.kind, u.label, u.key, "built", time.monotonic() - st.t0, st.cpu))
        log.info("event=prebuild built unit=%s kind=%s wall_s=%.1f cpu_s=%.1f", u.dirname, u.kind, time.monotonic() - st.t0, st.cpu)

    try:
        while True:
            # 1. fill the free worker slots: ready tasks first, then claim another unit (demand-driven: ranks split the units)
            while len(running) < jobs and first_err is None:
                rd = [r for r in ready()]
                if not rd:
                    if claim_one():
                        continue
                    break
                _prio, st, t = rd[0]
                st.started.add(t.name)
                t_sub = time.monotonic()
                running[pool.submit(run_task, st, t)] = (st, t, t_sub)
            # 2. done?
            if not running and all(st.status in ("cached", "waited", "built", "failed") for st in states):
                break
            # 3. wait for a task (or poll the peers)
            if running:
                finished, _ = wait(list(running), timeout=poll_s, return_when=FIRST_COMPLETED)
            else:
                time.sleep(poll_s)
                finished = set()
            now = time.monotonic()
            if now - last_beat >= 1.0:
                last_beat = now
                for st in states:
                    if st.status == "mine":
                        st.lock.touch()
            for fut in finished:
                st, t, _ts = running.pop(fut)
                try:
                    wall = fut.result()
                except BaseException as e:  # noqa: BLE001
                    if first_err is None:
                        first_err = e if isinstance(e, BuildError) else BuildError(f"{st.unit.dirname}/{t.name}: {e!r}")
                        if hasattr(runner, "abort"):
                            runner.abort()
                    st.status = "failed"
                    continue
                st.done.add(t.name)
                st.cpu += wall
                report.tasks[f"{st.unit.dirname}/{t.name}"] = wall
                if len(st.done) == len(st.pending) and first_err is None:
                    publish(st)
            if first_err is not None:
                if not running:
                    break
                continue
            # 4. peers: complete, stale, or timed out
            if now - wait0 > wait_timeout_s and any(st.status == "peer" for st in states):
                raise BuildError("timed out waiting for units held by another process: " + ", ".join(st.unit.dirname for st in states if st.status == "peer"))
            for st in states:
                if st.status == "peer" and os.path.exists(st.unit.ok_path):
                    st.status = "waited"
    except BaseException as e:  # noqa: BLE001
        if first_err is None:
            first_err = e
        if hasattr(runner, "abort"):
            runner.abort()
    finally:
        pool.shutdown(wait=True)
        for st in states:
            if st.status in ("mine", "failed") and st.tmp:
                shutil.rmtree(st.tmp, ignore_errors=True)
            st.lock.release()
    if first_err is not None:
        for st in states:
            if st.status != "built":
                report.units.append(UnitReport(st.unit.dirname, st.unit.kind, st.unit.label, st.unit.key, "failed" if st.status in ("mine", "failed") else st.status))
        if isinstance(first_err, BuildError):
            raise first_err
        raise BuildError(f"prebuild failed: {first_err!r}") from first_err
    for st in states:
        if st.status in ("cached", "waited"):
            report.units.append(UnitReport(st.unit.dirname, st.unit.kind, st.unit.label, st.unit.key, st.status))
    report.wall_s = time.monotonic() - t_start
    return report


# ----------------------------------------------------------------------------------------------------------------------------- public API
def plan_profile(profile, *, arch=None, cu=None, world=None, recipe=None, tool=None) -> Tuple[List[Unit], List[str]]:
    """(units, skipped): the build units of every row that passes the filters. Each variant class contributes through
    ``factory.prebuild_units(rows, recipe=, tool=)``; rows whose variant cannot be loaded or has no such hook are skipped with a reason."""
    from . import catalog
    from .match import _load_builtin
    from .profile import Profile, ProfileError, load_profile

    if not isinstance(profile, Profile):
        profile = load_profile(profile)
    if profile is None:
        raise ProfileError("prebuild_profile: no profile (pass a Profile, a name or a path)")
    rows = [r for r in profile.rows() if (arch is None or r.arch == arch) and (cu is None or int(r.cu) == int(cu)) and (world is None or int(r.world) == int(world))]
    by_vid: Dict[str, list] = {}
    for r in rows:
        by_vid.setdefault(r.vid, []).append(r)
    units: Dict[Tuple[str, str], Unit] = {}
    skipped: List[str] = []
    for vid, vrows in sorted(by_vid.items()):
        if vid not in catalog._CATALOG:
            _load_builtin(vid)
        info = catalog._CATALOG.get(vid)
        hook = getattr(info.factory, "prebuild_units", None) if info else None
        if hook is None:
            skipped.append(f"{vid}: no registered variant with a prebuild hook ({len(vrows)} row(s))")
            continue
        us, sk = hook(vrows, recipe=recipe, tool=tool)
        skipped += sk
        for u in us:
            units.setdefault((u.kind, u.dirname), u)  # dedupe: rows that share a build share a unit
    return list(units.values()), skipped


def prebuild_profile(profile, *, arch=None, cu=None, world=None, jobs=None, recipe=None) -> List[str]:
    """Build (or find cached) every kernel the profile's rows need, in one parallel pass; returns the keys of the units that are ready as "<kind>/<key>" strings.
    Idempotent; safe under concurrent processes (one builds a unit, the others wait for it). Raises ProfileError (bad profile) or ContractError (a build failed)."""
    from .match import ContractError

    units, skipped = plan_profile(profile, arch=arch, cu=cu, world=world, recipe=recipe)
    for s in skipped:
        log.warning("event=prebuild skipped %s", s)
    try:
        rep = execute(units, jobs)
    except BuildError as e:
        raise ContractError(f"gemm_fusion prebuild failed: {e}") from e
    log.info("event=prebuild done %s", rep.summary(top=6).replace("\n", " | "))
    prebuild_profile.last_report = rep  # for tools/tests: the per-task wall table
    by_dir = {u.dirname: u for u in units}
    return [f"{by_dir[u.name].kind}/{u.key}" for u in rep.units if u.name in by_dir and u.status != "failed"]


prebuild_profile.last_report = None
