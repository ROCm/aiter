# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Measuring configs: the backend protocol and the evaluator that enforces
the measurement rules.

A :class:`Backend` knows how to compile, check and time one config of one
problem. The :class:`Evaluator` owns everything strategies must be able to
rely on, whatever the backend:

- the numeric check is mandatory and tied to the artifact: every artifact
  compiled in this process is checked before its first timing, so a config
  that the current compiler, kernel sources or device miscompiles can never be
  timed. A config that fails is ``numeric_fail`` with value ``inf`` --
  never "fast";
- configs whose compiled resources show register spills or scratch use
  are ``spill`` (``reject_spill``), since they never win and distort a
  search;
- compile, check and measure calls can carry a wall-clock limit;
- measurements of one round at one fidelity go through ``measure_batch``
  when the backend has it, so samples are interleaved and paired by round
  (:class:`~.stats.Race` relies on it);
- results are cached per (config key, fidelity) and the cache can persist
  to JSON, so a resumed search spends nothing twice -- but only while the
  environment is the same (see below).

Environment. A persistent cache is only valid for the toolchain and device
that produced it. :meth:`Evaluator.fingerprint` covers the space, the
fidelity ladder and the environment: :func:`toolchain_environment` (the
FlyDSL and torch versions, a hash of the loaded kernel-builder sources, the
codegen environment variables, the device's arch, name and CU count when
torch has initialized HIP) plus ``backend.environment()`` and the
``environment`` argument. A cache written under another toolchain is
dropped; one written on another machine (same toolchain, other device or
host) keeps its structural failures (compile errors, spills, numeric
failures) and drops its timings. Transient failures (timeouts, run errors)
are never persisted, and every config is re-checked in every process.

Fidelities. A fidelity with a ``proxy`` is a different (smaller) problem:
its artifact, numeric check and structural failures are keyed by
(config, proxy). A backend serving proxy rungs takes the fidelity as an
extra argument of ``compile`` and ``check``.

Timeouts. Backend calls with a limit run on one dedicated daemon worker
thread (the same thread for every call, so thread-local GPU state such as a
current stream persists between calls; set it inside the backend). A call
that exceeds its limit cannot be killed in-process: it keeps running and is
abandoned. After a check or measure timeout the evaluator is *dirty*: work
may still be in flight on the device and would contaminate later timings,
so it calls ``backend.recover()`` (synchronize / reset) when the backend
has it and measures nothing more until the abandoned call has finished. The
driver stops a dirty search. A backend that must survive hung kernels or
compiles should run them in a subprocess and enforce its own limits there;
in-process soft limits are a last line of defence.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import os
import platform
import queue
import random
import sys
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from ._hash import stable_json_hash
from .params import SearchSpace
from .stats import NoiseModel, median
from .trial import (
    STRUCTURAL_STATUSES,
    Fidelity,
    Observation,
    Status,
    Trial,
)

__all__ = [
    "CACHE_VERSION",
    "DEFAULT_SPILL_KEYS",
    "Backend",
    "CallableBackend",
    "CompileOutcome",
    "Evaluator",
    "SyntheticBackend",
    "TableBackend",
    "loaded_builders_id",
    "toolchain_environment",
]

CACHE_VERSION = 2

DEFAULT_SPILL_KEYS = (
    "scratch_bytes",
    "spill_vgpr",
    "spill_sgpr",
    "private_segment_size",
)

# Environment variables that change the code a compile produces.
_CODEGEN_ENV = (
    "FLYDSL_COMPILE_BACKEND",
    "FLYDSL_COMPILE_OPT_LEVEL",
    "FLYDSL_COMPILE_LLVM_DIR",
    "FLYDSL_COMPILE_ARCH",
    "FLYDSL_GPU_ARCH",
    "FLYDSL_DEBUG_ENABLE_DEBUG_INFO",
    "HSA_OVERRIDE_GFX_VERSION",
    "ROCM_PATH",
)
# Distributions whose version decides the emitted code.
_TOOLCHAIN_DISTS = ("flydsl", "torch")
# Bound on the evaluator's in-memory maps (a very long search on a huge
# space must not grow without limit); the oldest entries go first.
_MAP_LIMIT = 100_000


@dataclass(frozen=True)
class CompileOutcome:
    """What compiling one config produced.

    ``artifact`` is opaque to the search (a loaded kernel, an HSACO path);
    ``resources`` are numbers the evaluator can gate on (VGPRs, LDS bytes,
    scratch/spill counts); ``seconds`` is the compile cost (measured by the
    evaluator when the backend leaves it 0).
    """

    ok: bool
    artifact: Any = None
    seconds: float = 0.0
    resources: Mapping[str, float] = field(default_factory=dict)
    detail: str = ""


@runtime_checkable
class Backend(Protocol):
    """Compile, check and time configs of one problem.

    ``config`` is the decoded knob dict and ``key`` its canonical key.
    ``check`` must compare the kernel's output against a reference (the
    evaluator calls it before the first timing of every artifact) and return
    ``(ok, why)``. ``measure`` returns per-rep times (seconds or ms; one
    unit per backend).

    Optional members:

    - ``measure_batch(items, fidelity)`` takes ``[(config, key, artifact),
      ...]`` and returns one sample list per item, interleaving reps across
      the items;
    - ``environment() -> Mapping`` names the toolchain and device the
      backend compiles and times on (JSON data); it enters the evaluator's
      cache fingerprint;
    - ``recover()`` drains or resets the device after a timed-out call;
    - ``compile(config, key, fidelity)`` / ``check(config, key, artifact,
      fidelity)``: a backend whose ladder has proxy rungs takes the
      fidelity as an extra argument (the evaluator detects it).
    """

    def compile(self, config: Mapping[str, object], key: str) -> CompileOutcome: ...

    def check(
        self, config: Mapping[str, object], key: str, artifact: Any
    ) -> tuple[bool, str]: ...

    def measure(
        self, config: Mapping[str, object], key: str, artifact: Any, fidelity: Fidelity
    ) -> Sequence[float]: ...


class _Timeout(Exception):
    pass


class _Dirty(Exception):
    pass


# ---------------------------------------------------------------- environment
# Module-name prefixes of the kernel builders whose loaded sources decide the
# emitted code.
_BUILDER_PACKAGES = ("aiter.ops.flydsl",)


def loaded_builders_id() -> str:
    """A hash of the sources of the kernel-builder modules loaded in this
    process (:data:`_BUILDER_PACKAGES`): changes whenever an edit could change
    the code a config compiles to. ``""`` when none is loaded. Builders
    imported after an evaluator was created are not covered; a search with a
    persistent cache imports its builders first."""
    h = hashlib.sha256()
    n = 0
    for name in sorted(sys.modules):
        if not any(name == p or name.startswith(p + ".") for p in _BUILDER_PACKAGES):
            continue
        path = getattr(sys.modules[name], "__file__", None)
        if not path or not path.endswith(".py"):
            continue
        try:
            data = Path(path).read_bytes()
        except OSError:
            continue
        h.update(name.encode())
        h.update(b"\0")
        h.update(hashlib.sha256(data.replace(b"\r\n", b"\n")).digest())
        n += 1
    return h.hexdigest()[:16] if n else ""


def _dist_versions() -> dict:
    """Installed versions of :data:`_TOOLCHAIN_DISTS` (``None`` when absent)."""
    from importlib import metadata

    out: dict[str, object] = {}
    for dist in _TOOLCHAIN_DISTS:
        try:
            out[dist] = metadata.version(dist)
        except metadata.PackageNotFoundError:
            out[dist] = None
    return out


def _device_environment() -> dict:
    """The current device, only when torch has already initialized HIP
    (never loads a GPU runtime as a side effect). A GPU backend should also
    report its device in ``environment()``."""
    torch = sys.modules.get("torch")
    if torch is None:
        return {}
    try:
        if not torch.cuda.is_initialized():
            return {}
        props = torch.cuda.get_device_properties(torch.cuda.current_device())
    except Exception:  # noqa: BLE001 - discovery is best effort
        return {}
    return {
        "target_id": getattr(props, "gcnArchName", None),
        "device_name": props.name,
        "num_cus": props.multi_processor_count,
    }


def toolchain_environment() -> dict:
    """``{"toolchain": {...}, "machine": {...}}`` of this process.

    ``toolchain`` decides what code a config compiles to (FlyDSL and torch
    versions, kernel-builder sources, codegen environment); ``machine`` adds
    where it runs (device, host). Never imports torch or initializes HIP.
    """
    tool = {
        **_dist_versions(),
        "builders": loaded_builders_id(),
        "env": {k: os.environ.get(k) for k in _CODEGEN_ENV},
    }
    dev = _device_environment()
    if dev.get("target_id"):
        tool["arch"] = dev["target_id"]
    machine = {"host": platform.node(), **dev}
    return {"toolchain": tool, "machine": machine}


# ------------------------------------------------------------------- workers
class _Worker:
    """One daemon thread that runs submitted calls in order."""

    def __init__(self) -> None:
        self._jobs: queue.Queue = queue.Queue()
        self.thread = threading.Thread(
            target=self._loop, name="tuning-search-worker", daemon=True
        )
        self.thread.start()

    def _loop(self) -> None:
        while True:
            job = self._jobs.get()
            if job is None:
                return
            fn, box, done = job
            try:
                box.append((True, fn()))
            except BaseException as e:  # noqa: BLE001 - re-raised by the caller
                box.append((False, e))
            done.set()

    def run(self, fn: Callable[[], Any], timeout: float) -> Any:
        box: list = []
        done = threading.Event()
        self._jobs.put((fn, box, done))
        if not done.wait(timeout):
            raise _Timeout(f"exceeded {timeout:g} s")
        ok, value = box[0]
        if ok:
            return value
        raise value

    def stop(self) -> None:
        self._jobs.put(None)


def _takes_fidelity(fn: Callable, n_base: int) -> bool:
    """Whether ``fn`` accepts a fidelity after its ``n_base`` arguments."""
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return False
    params = list(sig.parameters.values())
    if any(p.kind == p.VAR_POSITIONAL for p in params):
        return True
    if any(p.name == "fidelity" for p in params):
        return True
    positional = [
        p for p in params if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
    ]
    return len(positional) > n_base


def _bounded_set(d: dict, key, value) -> None:
    if key not in d and len(d) >= _MAP_LIMIT:
        d.pop(next(iter(d)))
    d[key] = value


class Evaluator:
    """Turns trials into observations under the measurement rules above.

    ``ladder`` is the fidelity ladder, cheapest first; its last entry is the
    full fidelity and trial fidelity ``-1`` means it. ``cache_path`` makes
    the result cache persistent (see the module docstring for when a cache
    is reused). ``environment`` is extra JSON data for the fingerprint
    (e.g. a library build id the default detection cannot see).
    ``compile_timeout`` and ``measure_timeout`` are per-config limits; an
    interleaved batch of ``n`` configs gets ``n`` times the measure limit.
    """

    def __init__(
        self,
        space: SearchSpace,
        backend: Backend,
        *,
        ladder: Sequence[Fidelity] = (Fidelity(),),
        reject_spill: bool = True,
        spill_keys: Sequence[str] = DEFAULT_SPILL_KEYS,
        compile_timeout: float | None = None,
        measure_timeout: float | None = None,
        cache_path: str | os.PathLike | None = None,
        clock: Callable[[], float] = time.perf_counter,
        environment: Mapping[str, object] | None = None,
    ):
        if not ladder:
            raise ValueError("Evaluator ladder - len=0, expected >= 1 Fidelity")
        for name in ("compile", "check", "measure"):
            if not callable(getattr(backend, name, None)):
                raise TypeError(
                    f"backend {type(backend).__name__} has no callable {name}(); "
                    "the numeric check is mandatory"
                )
        self.space = space
        self.backend = backend
        self.ladder = tuple(ladder)
        self._compile_fid = _takes_fidelity(backend.compile, 2)
        self._check_fid = _takes_fidelity(backend.check, 3)
        proxies = sorted({f.proxy for f in self.ladder if f.proxy})
        if proxies and not (self._compile_fid and self._check_fid):
            raise ValueError(
                f"Evaluator ladder - proxy={proxies[0]!r}, expected a backend whose "
                "compile(config, key, fidelity) and check(config, key, artifact, "
                "fidelity) take the fidelity (a proxy rung is another problem)"
            )
        self.reject_spill = bool(reject_spill)
        self.spill_keys = tuple(spill_keys)
        self.compile_timeout = compile_timeout
        self.measure_timeout = measure_timeout
        self.clock = clock
        self.noise = NoiseModel()
        self.cache_path = Path(cache_path) if cache_path else None
        self.environment = self._environment(environment)
        self._results: dict[str, Observation] = {}
        self._compiled: dict[str, CompileOutcome] = {}
        self._failed: dict[str, Observation] = {}
        self._checked: set[str] = set()
        self._calls: dict[str, list[float]] = {}
        self._batch = 0
        self._pending_compile_s: dict[str, float] = {}
        self._worker: _Worker | None = None
        self._zombies: list[threading.Thread] = []
        self.dirty = ""
        self.compiles = 0
        self.checks = 0
        if self.cache_path is not None and self.cache_path.exists():
            self._load_file()

    # ------------------------------------------------------------- helpers
    def _environment(self, extra: Mapping[str, object] | None) -> dict:
        env = toolchain_environment()
        own = getattr(self.backend, "environment", None)
        if callable(own):
            got = dict(own() or {})
            env["toolchain"]["backend"] = got.pop("toolchain", None)
            env["machine"]["backend"] = got
        if extra:
            env["toolchain"]["extra"] = dict(extra)
        return json.loads(json.dumps(env, sort_keys=True, default=repr))

    def resolve(self, fidelity: int) -> int:
        n = len(self.ladder)
        idx = n - 1 if fidelity < 0 else int(fidelity)
        if not 0 <= idx < n:
            raise ValueError(f"trial fidelity={fidelity}, expected -1 or 0..{n - 1}")
        return idx

    def _rkey(self, key: str, fid: int) -> str:
        f = self.ladder[fid]
        return f"{key}|{f.name}|{f.reps}|{f.pool_id}"

    def _akey(self, key: str, fid: int) -> str:
        """Artifact key: one compile, check and structural verdict per
        (config, proxy problem)."""
        return f"{key}|{self.ladder[fid].proxy}"

    def fingerprint(self) -> dict:
        """What a persistent cache must match: ``tool`` (space, ladder,
        toolchain) for structural failures, ``machine`` for timings."""
        tool = {
            "version": CACHE_VERSION,
            "space": self.space.fingerprint(),
            "ladder": [f.to_json() for f in self.ladder],
            "toolchain": stable_json_hash(self.environment["toolchain"]),
        }
        return {
            "tool": stable_json_hash(tool),
            "machine": stable_json_hash(self.environment["machine"]),
        }

    def known(self, key: str, fidelity: int = -1) -> Observation | None:
        """The cached observation of ``key`` at ``fidelity``, if any: a
        measurement at that fidelity, a structural failure of its artifact,
        or a transient failure at that fidelity."""
        fid = self.resolve(fidelity)
        rkey = self._rkey(key, fid)
        return (
            self._results.get(rkey)
            or self._failed.get(self._akey(key, fid))
            or self._failed.get(rkey)
        )

    def needs_compile(self, key: str, fidelity: int = -1) -> bool:
        """Whether measuring ``key`` would pay a compile (cost awareness)."""
        fid = self.resolve(fidelity)
        akey = self._akey(key, fid)
        return akey not in self._compiled and self.known(key, fid) is None

    # ------------------------------------------------------------ evaluate
    def evaluate(
        self,
        trials: Sequence[Trial],
        *,
        max_fresh: int | None = None,
        deadline: float | None = None,
    ) -> tuple[list[Observation], list[Trial]]:
        """Observations of ``trials`` (in order) and the trials left
        unevaluated: past ``max_fresh`` fresh measurements, past the
        ``deadline`` (a :attr:`clock` time), or while the evaluator is
        dirty.

        Each trial is resolved from the caches when possible; the rest are
        compiled and checked one by one, then timed in one interleaved
        batch per fidelity.
        """
        out: dict[int, Observation] = {}
        to_measure: dict[int, list[tuple[int, Trial, str, dict, Any]]] = {}
        leftover: list[int] = []
        fresh = 0
        claimed: dict[tuple[str, int], int] = {}
        top = len(self.ladder) - 1
        for i, t in enumerate(trials):
            fid = self.resolve(t.fidelity)
            key = self.space.key(t.genome)
            if key is None:
                out[i] = Observation(
                    trial=t,
                    key=None,
                    status=Status.INVALID,
                    fidelity=fid,
                    full=fid == top,
                    detail="refused by the space",
                    pool=self.ladder[fid].pool_id,
                )
                continue
            hit = self.known(key, fid)
            if hit is not None:
                out[i] = hit.replace_trial(
                    t, cached=True, fidelity=fid, full=fid == top
                )
                continue
            if (key, fid) in claimed:
                # Same config twice in one round: measured once, shared below.
                continue
            if (
                (max_fresh is not None and fresh >= max_fresh)
                or (deadline is not None and self.clock() >= deadline)
                or self._is_dirty()
            ):
                leftover.append(i)
                continue
            fresh += 1
            claimed[(key, fid)] = i
            config = self.space.decode(t.genome)
            ob, artifact = self._prepare(t, key, fid, config)
            if ob is not None:
                out[i] = ob
                continue
            to_measure.setdefault(fid, []).append((i, t, key, config, artifact))
        for fid in sorted(to_measure):
            for i, ob in self._measure(fid, to_measure[fid]):
                out[i] = ob
        for i, t in enumerate(trials):
            if i in out or i in leftover:
                continue
            fid = self.resolve(t.fidelity)
            key = self.space.key(t.genome)
            first = claimed.get((key, fid))
            if first is None or first not in out:
                leftover.append(i)
                continue
            out[i] = out[first].replace_trial(t, cached=True)
        self._save_file()
        return [out[i] for i in sorted(out)], [trials[i] for i in sorted(leftover)]

    # ------------------------------------------------------------ backend
    def _call(self, fn: Callable[[], Any], timeout: float | None, kind: str):
        if timeout is None:
            return fn()
        if self._worker is None or not self._worker.thread.is_alive():
            self._worker = _Worker()
        worker = self._worker
        try:
            return worker.run(fn, timeout)
        except _Timeout:
            # The call cannot be killed; abandon the worker. A hung check or
            # measure may still have work in flight on the device.
            self._worker = None
            worker.stop()
            if kind != "compile":
                self._zombies.append(worker.thread)
                self.dirty = f"a {kind} call exceeded {timeout:g} s"
                if kind != "recover":
                    self._recover()
            raise

    def _recover(self) -> None:
        fn = getattr(self.backend, "recover", None)
        if not callable(fn):
            return
        try:
            self._call(fn, self.measure_timeout, "recover")
        except Exception:  # noqa: BLE001 - stays dirty
            return

    def _is_dirty(self) -> bool:
        if not self.dirty:
            return False
        self._zombies = [z for z in self._zombies if z.is_alive()]
        if not self._zombies:
            # The abandoned calls finished: nothing of theirs is in flight.
            self.dirty = ""
        return bool(self.dirty)

    def _compile(self, config, key, fid):
        if self._compile_fid:
            fidelity = self.ladder[fid]
            return lambda: self.backend.compile(config, key, fidelity)
        return lambda: self.backend.compile(config, key)

    def _check(self, config, key, fid, artifact):
        if self._check_fid:
            fidelity = self.ladder[fid]
            return lambda: self.backend.check(config, key, artifact, fidelity)
        return lambda: self.backend.check(config, key, artifact)

    def _prepare(self, t: Trial, key: str, fid: int, config: dict):
        """Compile (once per process) and check (once per artifact) ``key``
        at ``fid``'s problem: ``(failure_observation, None)`` or ``(None,
        artifact)``."""
        akey = self._akey(key, fid)
        comp = self._compiled.get(akey)
        compile_s = 0.0
        if comp is None:
            t0 = self.clock()
            self.compiles += 1
            try:
                comp = self._call(
                    self._compile(config, key, fid), self.compile_timeout, "compile"
                )
            except _Timeout as e:
                return (
                    self._fail(
                        t, key, fid, Status.TIMEOUT, f"compile {e}", self.clock() - t0
                    ),
                    None,
                )
            except Exception as e:  # noqa: BLE001 - a backend failure is an outcome
                return (
                    self._fail(
                        t,
                        key,
                        fid,
                        Status.COMPILE_ERROR,
                        f"{type(e).__name__}: {e}",
                        self.clock() - t0,
                    ),
                    None,
                )
            compile_s = comp.seconds or (self.clock() - t0)
            if not comp.ok:
                return (
                    self._fail(
                        t,
                        key,
                        fid,
                        Status.COMPILE_ERROR,
                        comp.detail or "compile failed",
                        compile_s,
                    ),
                    None,
                )
            if self.reject_spill:
                spilled = {
                    k: comp.resources[k]
                    for k in self.spill_keys
                    if float(comp.resources.get(k, 0) or 0) > 0
                }
                if spilled:
                    return (
                        self._fail(
                            t,
                            key,
                            fid,
                            Status.SPILL,
                            f"spills {spilled}",
                            compile_s,
                            comp.resources,
                        ),
                        None,
                    )
            _bounded_set(self._compiled, akey, comp)
            # A fresh artifact is a new binary: it is checked, whatever an
            # earlier process concluded about the same config.
            self._checked.discard(akey)
        if akey not in self._checked:
            fail = self._run_check(t, key, fid, config, comp, compile_s)
            if fail is not None:
                return fail, None
        self._pending_compile_s[key] = compile_s
        return None, comp.artifact

    def _run_check(self, t, key, fid, config, comp, compile_s) -> Observation | None:
        akey = self._akey(key, fid)
        self.checks += 1
        try:
            ok, why = self._call(
                self._check(config, key, fid, comp.artifact),
                self.measure_timeout,
                "check",
            )
        except _Timeout as e:
            return self._fail(
                t, key, fid, Status.TIMEOUT, f"check {e}", compile_s, comp.resources
            )
        except Exception as e:  # noqa: BLE001 - a backend failure is an outcome
            return self._fail(
                t,
                key,
                fid,
                Status.RUN_ERROR,
                f"check {type(e).__name__}: {e}",
                compile_s,
                comp.resources,
            )
        if not ok:
            self._compiled.pop(akey, None)
            return self._fail(
                t,
                key,
                fid,
                Status.NUMERIC_FAIL,
                why or "mismatch",
                compile_s,
                comp.resources,
            )
        self._checked.add(akey)
        return None

    def _fail(
        self, t, key, fid, status, detail, compile_s, resources=None
    ) -> Observation:
        ob = Observation(
            trial=t,
            key=key,
            status=status,
            fidelity=fid,
            full=fid == len(self.ladder) - 1,
            compile_s=float(compile_s),
            detail=detail,
            resources=dict(resources or {}),
            pool=self.ladder[fid].pool_id,
        )
        self._remember_failure(key, fid, ob)
        return ob

    def _remember_failure(self, key: str, fid: int, ob: Observation) -> None:
        # A config that fails to build, spills or computes wrong values fails
        # at every fidelity of the same problem; a timeout or run error is
        # remembered for this fidelity and this process only (re-running it
        # now is what a budget cannot afford, but it may not repeat).
        if ob.status in STRUCTURAL_STATUSES:
            _bounded_set(self._failed, self._akey(key, fid), ob)
        else:
            _bounded_set(self._failed, self._rkey(key, fid), ob)

    # -------------------------------------------------------------- measure
    def _measure_one(self, cfg, key, art, fidelity):
        return self._call(
            lambda: self.backend.measure(cfg, key, art, fidelity),
            self.measure_timeout,
            "measure",
        )

    def _measure(self, fid: int, items):
        fidelity = self.ladder[fid]
        full = fid == len(self.ladder) - 1
        self._batch += 1
        batch = self._batch
        pending_cs = self._pending_compile_s
        t0 = self.clock()
        results: list[Sequence[float] | Exception] = []
        batch_fn = getattr(self.backend, "measure_batch", None)
        batched = callable(batch_fn) and len(items) > 1
        if batched:
            payload = [(cfg, key, art) for _i, _t, key, cfg, art in items]
            limit = (
                None
                if self.measure_timeout is None
                else self.measure_timeout * len(items)
            )
            try:
                got = list(
                    self._call(lambda: batch_fn(payload, fidelity), limit, "measure")
                )
                if len(got) != len(items):
                    raise ValueError(
                        f"measure_batch - len(result)={len(got)}, expected "
                        f"{len(items)} (one sample list per config)"
                    )
                results = got
            except Exception:  # noqa: BLE001 - isolate the culprit below
                # One config's failure must not fail its batch-mates: measure
                # each alone before blaming any of them.
                batched = False
                results = []
        if not batched:
            for _i, _t, key, cfg, art in items:
                if self._is_dirty():
                    results.append(_Dirty(self.dirty))
                    continue
                try:
                    results.append(self._measure_one(cfg, key, art, fidelity))
                except Exception as e:  # noqa: BLE001 - a backend failure is an outcome
                    results.append(e)
            batch = -1  # measured one by one: not paired
        per = (self.clock() - t0) / max(1, len(items))
        for (i, t, key, _cfg, _art), res in zip(items, results):
            cs = float(pending_cs.pop(key, 0.0))
            akey = self._akey(key, fid)
            resources = (
                dict(self._compiled[akey].resources) if akey in self._compiled else {}
            )
            common = {
                "trial": t,
                "key": key,
                "fidelity": fid,
                "full": full,
                "compile_s": cs,
                "measure_s": per,
                "resources": resources,
                "pool": fidelity.pool_id,
            }
            if isinstance(res, Exception):
                if isinstance(res, _Timeout):
                    status, detail = Status.TIMEOUT, f"measure {res}"
                elif isinstance(res, _Dirty):
                    status, detail = Status.RUN_ERROR, f"evaluator dirty: {res}"
                else:
                    status, detail = Status.RUN_ERROR, f"{type(res).__name__}: {res}"
                ob = Observation(status=status, detail=detail, **common)
                self._remember_failure(key, fid, ob)
                yield i, ob
                continue
            samples = tuple(float(x) for x in res)
            if not samples or not all(math.isfinite(x) and x > 0 for x in samples):
                ob = Observation(
                    status=Status.RUN_ERROR,
                    detail="non-positive or non-finite samples",
                    **common,
                )
                self._remember_failure(key, fid, ob)
                yield i, ob
                continue
            self.noise.update(samples)
            value = median(samples)
            self._learn_call(key, fidelity, value)
            ob = Observation(
                status=Status.OK, value=value, samples=samples, batch=batch, **common
            )
            _bounded_set(self._results, self._rkey(key, fid), ob)
            yield i, ob

    def _learn_call(self, key: str, fidelity: Fidelity, value: float) -> None:
        """Feed the call-to-call spread of one config's medians (same pool)
        to the noise model: the run-to-run part of timing noise that more
        reps do not average away."""
        ck = f"{key}|{fidelity.pool_id}"
        prev = self._calls.get(ck)
        if prev:
            self.noise.update_call(prev[-1], value)
        lst = prev or []
        lst.append(value)
        _bounded_set(self._calls, ck, lst[-4:])

    def remeasure(
        self,
        keys: Sequence[str],
        genomes: Mapping[str, Sequence[int]],
        *,
        fidelity: int = -1,
        rounds: int = 1,
    ) -> dict[str, list[list[float]]]:
        """Fresh interleaved samples of configs, bypassing the cache:
        ``rounds`` batches; per key, one sample list per round (empty when
        that round failed for it). Used to confirm the finalists of a search
        on equal footing; round ``r`` of every key is one interleaved
        batch, so rounds pair across keys.

        A key without a live artifact (measured before a resume, or never
        measured, like a baseline) is compiled and checked first; one that
        fails is left out."""
        fid = self.resolve(fidelity)
        out: dict[str, list[list[float]]] = {k: [] for k in keys}
        live = []
        for k in keys:
            if k not in genomes:
                continue
            akey = self._akey(k, fid)
            if akey in self._compiled and akey in self._checked:
                live.append(k)
                continue
            if self._is_dirty():
                continue
            g = tuple(genomes[k])
            t = Trial(genome=g, uid=f"confirm:{k}", origin="confirm", fidelity=fid)
            ob, _art = self._prepare(t, k, fid, self.space.decode(g))
            if ob is None:
                self._pending_compile_s.pop(k, None)
                live.append(k)
        for _ in range(int(rounds)):
            if self._is_dirty():
                break
            items = []
            for k in live:
                t = Trial(
                    genome=tuple(genomes[k]),
                    uid=f"confirm:{k}",
                    origin="confirm",
                    fidelity=fid,
                )
                items.append(
                    (
                        0,
                        t,
                        k,
                        self.space.decode(genomes[k]),
                        self._compiled[self._akey(k, fid)].artifact,
                    )
                )
            if not items:
                break
            stash, failed = dict(self._results), dict(self._failed)
            got = {k: [] for k in live}
            for _i, ob in self._measure(fid, items):
                if ob.ok:
                    got[ob.key] = list(ob.samples)
            self._results, self._failed = stash, failed
            for k in live:
                out[k].append(got[k])
        return out

    # ---------------------------------------------------------------- state
    def state_dict(self) -> dict:
        """Persistable state: timings and structural failures. Transient
        failures and check verdicts are per process and never saved."""
        failed = {
            k: ob.to_json()
            for k, ob in self._failed.items()
            if ob.status in STRUCTURAL_STATUSES
        }
        return {
            "fingerprint": self.fingerprint(),
            "results": {k: ob.to_json() for k, ob in self._results.items()},
            "failed": failed,
            "batch": self._batch,
            "noise": self.noise.to_json(),
        }

    def load_state_dict(self, d: Mapping) -> None:
        """Restore :meth:`state_dict` output. Refused (``ValueError``) when
        the space, ladder or toolchain differ; on another machine only the
        structural failures are restored."""
        fp, mine = d.get("fingerprint") or {}, self.fingerprint()
        if fp.get("tool") != mine["tool"]:
            raise ValueError(
                "evaluator state - fingerprint differs, expected the same space, "
                "fidelity ladder and toolchain; refusing to reuse its results"
            )
        self._failed = {
            k: Observation.from_json(v)
            for k, v in d["failed"].items()
            if v.get("status") in STRUCTURAL_STATUSES
        }
        same_machine = fp.get("machine") == mine["machine"]
        self._results = (
            {k: Observation.from_json(v) for k, v in d["results"].items()}
            if same_machine
            else {}
        )
        self._checked = set()
        self._batch = int(d.get("batch", 0))
        self.noise = NoiseModel.from_json(d["noise"]) if same_machine else NoiseModel()

    def _load_file(self) -> None:
        try:
            data = json.loads(self.cache_path.read_text())
            self.load_state_dict(data)
        except (ValueError, KeyError, OSError, TypeError, AttributeError):
            # A cache for another space or toolchain (or a corrupt file) is
            # ignored and overwritten; it is a cache, not a record.
            self._results, self._failed, self._checked = {}, {}, set()

    def _save_file(self) -> None:
        if self.cache_path is None:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.cache_path.with_suffix(self.cache_path.suffix + ".tmp")
        tmp.write_text(json.dumps(self.state_dict(), sort_keys=True))
        tmp.replace(self.cache_path)

    def close(self) -> None:
        """Stop the worker thread (a timed-out call keeps running)."""
        if self._worker is not None:
            self._worker.stop()
            self._worker = None


# --------------------------------------------------------------- backends
class CallableBackend:
    """A backend from three callables (and an optional batch timer and
    environment)."""

    def __init__(
        self,
        compile: Callable[[Mapping[str, object], str], CompileOutcome],
        check: Callable[[Mapping[str, object], str, Any], tuple[bool, str]],
        measure: Callable[[Mapping[str, object], str, Any, Fidelity], Sequence[float]],
        measure_batch: (
            Callable[[Sequence, Fidelity], Sequence[Sequence[float]]] | None
        ) = None,
        environment: Mapping[str, object] | None = None,
    ):
        self.compile = compile
        self.check = check
        self.measure = measure
        if measure_batch is not None:
            self.measure_batch = measure_batch
        self._environment = dict(environment or {})

    def environment(self) -> dict:
        return dict(self._environment)


class SyntheticBackend:
    """A deterministic stand-in for a GPU: ``objective(config)`` is the true
    time, timed with multiplicative log-normal noise.

    ``noise`` is the per-rep relative sigma; ``call_noise`` one factor per
    measurement call and config (run-to-run noise that more reps do not
    average away); ``drift`` a per-rep factor shared by every config of one
    interleaved batch (what pairing cancels). ``fail(config)`` returns a
    status (``"compile_error"``, ``"numeric_fail"``, ``"spill"``) or None.
    ``objective`` may take a second argument, the :class:`Fidelity`, to
    model proxies. Noise is seeded by (seed, key, call count), so a run is
    reproducible.
    """

    def __init__(
        self,
        objective: Callable[..., float],
        *,
        noise: float = 0.0,
        drift: float = 0.0,
        seed: int = 0,
        fail: Callable[[Mapping[str, object]], str | None] | None = None,
        compile_seconds: float | Callable[[Mapping[str, object]], float] = 0.0,
        call_noise: float = 0.0,
    ):
        self.objective = objective
        self.noise = float(noise)
        self.drift = float(drift)
        self.call_noise = float(call_noise)
        self.seed = int(seed)
        self.fail = fail
        self.compile_seconds = compile_seconds
        self.compiles = 0
        self.checks = 0
        self.measures = 0
        self._calls: dict[str, int] = {}
        try:
            self._two_args = len(inspect.signature(objective).parameters) >= 2
        except (TypeError, ValueError):
            self._two_args = False

    def environment(self) -> dict:
        return {"backend": "synthetic"}

    def _truth(self, config: Mapping[str, object], fidelity: Fidelity) -> float:
        if self._two_args:
            return float(self.objective(config, fidelity))
        return float(self.objective(config))

    def compile(self, config: Mapping[str, object], key: str) -> CompileOutcome:
        self.compiles += 1
        status = self.fail(config) if self.fail else None
        cs = self.compile_seconds
        seconds = float(cs(config) if callable(cs) else cs)
        if status == Status.COMPILE_ERROR:
            return CompileOutcome(
                False, detail="synthetic compile error", seconds=seconds
            )
        resources = {"scratch_bytes": 64.0} if status == Status.SPILL else {}
        return CompileOutcome(
            True, artifact=status, seconds=seconds, resources=resources
        )

    def check(
        self, config: Mapping[str, object], key: str, artifact: Any
    ) -> tuple[bool, str]:
        self.checks += 1
        if artifact == Status.NUMERIC_FAIL:
            return False, "synthetic mismatch"
        return True, "ok"

    def _rng(self, key: str) -> random.Random:
        n = self._calls.get(key, 0)
        self._calls[key] = n + 1
        return random.Random(f"{self.seed}:{key}:{n}")

    def _call_factor(self, rng: random.Random) -> float:
        if self.call_noise <= 0.0:
            return 1.0
        return math.exp(rng.gauss(0.0, self.call_noise))

    def measure(
        self,
        config: Mapping[str, object],
        key: str,
        artifact: Any,
        fidelity: Fidelity,
    ) -> list[float]:
        self.measures += 1
        rng = self._rng(key)
        truth = self._truth(config, fidelity) * self._call_factor(rng)
        return [
            truth * math.exp(rng.gauss(0.0, self.noise)) for _ in range(fidelity.reps)
        ]

    def measure_batch(
        self, items: Sequence[tuple[Mapping[str, object], str, Any]], fidelity: Fidelity
    ) -> list[list[float]]:
        self.measures += len(items)
        drift_rng = self._rng("|".join(k for _c, k, _a in items))
        factors = [
            math.exp(drift_rng.gauss(0.0, self.drift)) for _ in range(fidelity.reps)
        ]
        out = []
        for config, key, _art in items:
            rng = self._rng(key)
            truth = self._truth(config, fidelity) * self._call_factor(rng)
            out.append(
                [truth * f * math.exp(rng.gauss(0.0, self.noise)) for f in factors]
            )
        return out


class TableBackend:
    """A recorded table as a backend, for offline replay.

    ``table`` maps a config key (``SearchSpace.key``) to its recorded time
    or recorded samples. Configs missing from it fail to compile (they were
    never measured, so they cannot be scored). With ``noise > 0`` each rep
    is the recorded median times log-normal noise, seeded by (seed, key,
    call), so racing and halving can be exercised on recorded data.
    ``call_noise`` adds one log-normal factor per measurement call, shared
    by its reps: the run-to-run part of timing noise (clock and thermal
    state, placement) that more reps do not average away. Repeated
    measurements of the same configs in recorded GEMM sweeps show both
    parts: the spread shrinks with more reps but does not vanish.

    Only measured values are hidden from a strategy (read through
    ``measure`` for asked configs). The replay *space* is built from the
    recorded rows, so which configs exist -- and therefore where repair and
    sampling land -- is known to the search: see :mod:`.replay`.
    """

    def __init__(
        self,
        table: Mapping[str, float | Sequence[float]],
        *,
        noise: float = 0.0,
        seed: int = 0,
        failures: Mapping[str, str] | None = None,
        call_noise: float = 0.0,
    ):
        self.table = dict(table)
        self.noise = float(noise)
        self.call_noise = float(call_noise)
        self.seed = int(seed)
        self.failures = dict(failures or {})
        self._calls: dict[str, int] = {}
        self.measures = 0

    def environment(self) -> dict:
        return {"backend": "table"}

    def compile(self, config: Mapping[str, object], key: str) -> CompileOutcome:
        if key in self.failures and self.failures[key] == Status.COMPILE_ERROR:
            return CompileOutcome(False, detail="recorded compile error")
        if key not in self.table and key not in self.failures:
            return CompileOutcome(False, detail="not in the recorded table")
        return CompileOutcome(True, artifact=key)

    def check(
        self, config: Mapping[str, object], key: str, artifact: Any
    ) -> tuple[bool, str]:
        if self.failures.get(key) == Status.NUMERIC_FAIL:
            return False, "recorded numeric failure"
        return True, "ok"

    def measure(
        self,
        config: Mapping[str, object],
        key: str,
        artifact: Any,
        fidelity: Fidelity,
    ) -> list[float]:
        self.measures += 1
        if key in self.failures and key not in self.table:
            raise RuntimeError(f"recorded {self.failures[key]}")
        rec = self.table[key]
        if isinstance(rec, (int, float)):
            base = [float(rec)]
        else:
            base = [float(x) for x in rec]
        if self.noise <= 0.0 and self.call_noise <= 0.0:
            reps = fidelity.reps
            return [base[i % len(base)] for i in range(reps)]
        n = self._calls.get(key, 0)
        self._calls[key] = n + 1
        rng = random.Random(f"{self.seed}:{key}:{n}")
        m = median(base)
        if self.call_noise > 0.0:
            m *= math.exp(
                random.Random(f"run:{self.seed}:{key}:{n}").gauss(0.0, self.call_noise)
            )
        return [m * math.exp(rng.gauss(0.0, self.noise)) for _ in range(fidelity.reps)]
