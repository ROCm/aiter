# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The tuning DB (what dispatch serves) and search traces (what happened).

A :class:`TunedRecord` is one tuned pick: for a candidate, an arch and a
problem, the winning ``tuning_id`` (``stem@config_key``, so the defaults
fingerprint is in it) and the knobs it replays from. Dispatch serves it by
pinning a request (:meth:`TuningDB.pin`): the candidate rebuilds the spec
from the knobs and refuses the pin when the knobs no longer reproduce the
key -- a stale record is reported, never silently replaced.

The DB file is one JSON document with records sorted by
``(family, arch, problem, candidate)``, written atomically under a lock
file: every :meth:`TuningDB.put` re-reads the file, merges, and writes, so
concurrent tuners (cluster jobs sharing one DB) do not lose each other's
records.

Replacement policy (:meth:`TuningDB.put`). Raw times from different
sessions, nodes and clock states are not comparable, so a new record does
not replace an old one by being faster. A record carries its confirmed,
*paired* speedup over the dispatcher's default (measured interleaved in its
own session) and that speedup's one-sided p-value. A new record replaces
the old one when the slot is empty, when it re-measures the same config (a
refresh), when the old record was made with another toolchain, when the old
one is stale (its knobs no longer reproduce its key; the caller says so),
when ``force`` is set, or when its own speedup is significant and larger
than the old record's. Only *significant* records (speedup above
``min_speedup`` with p below ``alpha``) are served
(:meth:`TuningDB.serve`): a pick within noise of the default is recorded
but never pinned.

Transfer: :meth:`TuningDB.neighbors` finds the records of the nearest
problems (Euclidean distance over log2 of the numeric problem features) and
:meth:`TuningDB.other_arch` the same problem tuned on other archs; both
become :class:`~.portfolio.WarmStart` lists via :meth:`TuningDB.warm_starts`.

Traces are JSON lines: a header (space, strategy, seed, budget) then one
line per observation, with the incumbent after it.
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

from ._hash import stable_json_hash
from .portfolio import WarmStart
from .trial import Observation

__all__ = [
    "DB_VERSION",
    "TraceWriter",
    "TunedRecord",
    "TuningDB",
    "problem_key",
    "read_trace",
    "records_from_rows",
]

DB_VERSION = 2
_TUNING_FIELDS = ("algorithm", "spec_id", "tuning_id", "tuning_knobs")


def problem_key(request: object) -> str:
    """A stable key for a request's problem: its normalized fields minus the
    selector and tuning pins (so the pinned and unpinned request agree)."""
    norm = getattr(request, "normalized", None)
    if callable(norm):
        payload = dict(norm())
    elif isinstance(request, Mapping):
        payload = dict(request)
    else:
        payload = {"repr": repr(request)}
    for f in _TUNING_FIELDS:
        payload.pop(f, None)
    return stable_json_hash({"problem": _jsonable(payload)})


def _jsonable(v):
    if isinstance(v, Mapping):
        return {str(k): _jsonable(x) for k, x in sorted(v.items())}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, (bool, int, float, str)) or v is None:
        return v
    return repr(v)


@dataclass(frozen=True)
class TunedRecord:
    """One tuned pick. ``value`` is the confirmed time (lower is better) in
    ``unit``; ``features`` are numeric problem features used for transfer.
    ``knobs`` are the canonical knobs the ``tuning_id`` replays from.

    ``speedup`` is the confirmed paired speedup over the dispatcher default
    ``baseline_key`` (``None``: no baseline was confirmed), ``p_value`` its
    one-sided p-value over ``rounds`` interleaved rounds. ``samples`` and
    ``spread`` describe the confirmation samples ``value`` comes from.
    ``toolchain`` / ``machine`` fingerprint where it was measured."""

    family: str
    candidate: str
    arch: str
    problem: str
    tuning_id: str
    config_key: str
    knobs: Mapping[str, object]
    value: float
    unit: str = "ms"
    algorithm: str = ""
    spec_id: str = ""
    features: Mapping[str, float] = field(default_factory=dict)
    samples: int = 0
    spread: float = 0.0
    strategy: str = ""
    evaluations: int = 0
    seed: int = 0
    trace: str = ""
    created: str = ""
    speedup: float | None = None
    p_value: float | None = None
    baseline_key: str = ""
    rounds: int = 0
    toolchain: str = ""
    machine: str = ""

    def __post_init__(self):
        if not self.tuning_id.endswith(f"@{self.config_key}"):
            raise ValueError(
                f"TunedRecord tuning_id={self.tuning_id!r}, expected it to end in "
                f"'@{self.config_key}'"
            )
        if not (math.isfinite(self.value) and self.value > 0):
            raise ValueError(
                f"TunedRecord value={self.value}, expected a finite time > 0"
            )
        for k, v in self.knobs.items():
            if not isinstance(v, (bool, int, float, str, type(None))):
                raise TypeError(f"TunedRecord knob - {k}={v!r}, expected a JSON scalar")

    @property
    def slot(self) -> tuple[str, str, str, str]:
        return (self.family, self.arch, self.problem, self.candidate)

    def significant(self, *, alpha: float = 0.05, min_speedup: float = 1.0) -> bool:
        """Whether the confirmed speedup over the default is real: above
        ``min_speedup`` with p below ``alpha``."""
        return (
            self.speedup is not None
            and self.p_value is not None
            and self.speedup > min_speedup
            and self.p_value < alpha
        )

    def to_json(self) -> dict:
        d = asdict(self)
        d["knobs"] = dict(self.knobs)
        d["features"] = dict(self.features)
        return d

    @classmethod
    def from_json(cls, d: Mapping) -> TunedRecord:
        return cls(**dict(d))


class TuningDB:
    """Tuned picks keyed by ``(family, arch, problem, candidate)``; an
    in-memory store when ``path`` is None. ``alpha`` and ``min_speedup``
    decide which records are significant (served, and allowed to displace
    another record); ``lock_timeout`` bounds the wait for the file lock."""

    def __init__(
        self,
        path: str | os.PathLike | None = None,
        *,
        alpha: float = 0.05,
        min_speedup: float = 1.01,
        lock_timeout: float = 120.0,
    ):
        self.path = Path(path) if path else None
        self.alpha = float(alpha)
        self.min_speedup = float(min_speedup)
        self.lock_timeout = float(lock_timeout)
        self._records: dict[tuple, TunedRecord] = {}
        if self.path is not None and self.path.exists():
            with self._locked():
                self._reload()

    def _reload(self) -> None:
        if self.path is None or not self.path.exists():
            return
        data = json.loads(self.path.read_text())
        if int(data.get("version", 0)) not in (1, DB_VERSION):
            raise ValueError(
                f"tuning DB {self.path} - version={data.get('version')!r}, "
                f"expected {DB_VERSION}"
            )
        self._records = {}
        for d in data["records"]:
            r = TunedRecord.from_json(d)
            self._records[r.slot] = r

    def __len__(self) -> int:
        return len(self._records)

    def records(self) -> list[TunedRecord]:
        return [self._records[s] for s in sorted(self._records)]

    def significant(self, record: TunedRecord) -> bool:
        return record.significant(alpha=self.alpha, min_speedup=self.min_speedup)

    def _replaces(self, new: TunedRecord, old: TunedRecord | None, stale: bool) -> str:
        """Why ``new`` replaces ``old`` ("" when it does not)."""
        if old is None:
            return "empty"
        if stale:
            return "stale"
        if new.config_key == old.config_key:
            return "refresh"
        if new.toolchain != old.toolchain:
            return "toolchain"
        if self.significant(new) and (
            not self.significant(old) or new.speedup > (old.speedup or 0.0)
        ):
            return "faster"
        return ""

    def put(
        self, record: TunedRecord, *, force: bool = False, stale: bool = False
    ) -> bool:
        """Store ``record`` under the policy of the module docstring
        (``stale``: the caller found the slot's record no longer reproduces
        its key). Returns whether it was stored."""
        if not record.created:
            record = replace(record, created=time.strftime("%Y-%m-%dT%H:%M:%S"))
        with self._locked():
            self._reload()
            old = self._records.get(record.slot)
            if not (force or self._replaces(record, old, stale)):
                return False
            self._records[record.slot] = record
            self._write()
            return True

    def get(
        self,
        family: str,
        arch: str,
        problem: str,
        candidate: str | None = None,
        *,
        servable: bool = False,
    ) -> TunedRecord | None:
        """The record for the slot; with no ``candidate``, the record of any
        candidate with the largest confirmed speedup over its default (then
        the lowest time). ``servable`` keeps only significant records."""
        if candidate is not None:
            r = self._records.get((family, arch, problem, candidate))
            return None if r is None or (servable and not self.significant(r)) else r
        rows = [
            r
            for s, r in self._records.items()
            if s[:3] == (family, arch, problem)
            and (not servable or self.significant(r))
        ]
        if not rows:
            return None
        return min(rows, key=lambda r: (-(r.speedup or 0.0), r.value, r.candidate))

    def serve(
        self, request, family: str, arch: str, problem: str, candidate: str
    ) -> tuple[object, TunedRecord | None]:
        """``(request pinned to the slot's significant record, record)``, or
        ``(request, None)`` when there is none: a pick that did not beat the
        default beyond noise is never pinned."""
        rec = self.get(family, arch, problem, candidate, servable=True)
        if rec is None:
            return request, None
        return self.pin(request, rec), rec

    def pin(self, request, record: TunedRecord):
        """``request`` pinned to ``record`` (its algorithm, spec_id, tuning_id
        and knobs); the candidate validates the pin when it resolves it."""
        if not record.algorithm or not record.spec_id:
            raise ValueError(
                f"TunedRecord {record.candidate!r} - algorithm={record.algorithm!r}, "
                f"spec_id={record.spec_id!r}, expected both set to pin a request"
            )
        return replace(
            request,
            algorithm=record.algorithm,
            spec_id=record.spec_id,
            tuning_id=record.tuning_id,
            tuning_knobs=dict(record.knobs),
        )

    def neighbors(
        self,
        family: str,
        arch: str,
        features: Mapping[str, float],
        *,
        k: int = 3,
        exclude_problem: str | None = None,
        candidate: str | None = None,
    ) -> list[tuple[float, TunedRecord]]:
        """The ``k`` records of the nearest other problems on ``arch``."""
        out = []
        for r in self._records.values():
            if r.family != family or r.arch != arch or r.problem == exclude_problem:
                continue
            if candidate is not None and r.candidate != candidate:
                continue
            d = _feature_distance(features, r.features)
            if d is not None:
                out.append((d, r))
        out.sort(key=lambda t: (t[0], t[1].problem, t[1].candidate))
        return out[: int(k)]

    def other_arch(
        self, family: str, arch: str, problem: str, candidate: str | None = None
    ) -> list[TunedRecord]:
        return [
            r
            for s, r in sorted(self._records.items())
            if r.family == family
            and r.arch != arch
            and r.problem == problem
            and (candidate is None or r.candidate == candidate)
        ]

    def warm_starts(
        self,
        family: str,
        arch: str,
        problem: str,
        features: Mapping[str, float],
        *,
        candidate: str | None = None,
        k: int = 3,
        cross_arch: bool = True,
    ) -> list[WarmStart]:
        """Warm starts for a new search: this slot's record (a re-tune), the
        ``k`` nearest problems' winners, and other archs' winners."""
        out: list[WarmStart] = []
        own = self.get(family, arch, problem, candidate)
        if own is not None:
            out.append(WarmStart(dict(own.knobs), "db"))
        for _d, r in self.neighbors(
            family, arch, features, k=k, exclude_problem=problem, candidate=candidate
        ):
            out.append(WarmStart(dict(r.knobs), f"neighbor:{r.problem}"))
        if cross_arch:
            for r in self.other_arch(family, arch, problem, candidate):
                out.append(WarmStart(dict(r.knobs), f"arch:{r.arch}"))
        return out

    def save(self) -> None:
        """Merge this store into the file (records of the file whose slot
        this store does not hold are kept)."""
        if self.path is None:
            return
        with self._locked():
            mine = dict(self._records)
            self._reload()
            self._records.update(mine)
            self._write()

    def _write(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = {"version": DB_VERSION, "records": [r.to_json() for r in self.records()]}
        tmp = self.path.with_name(f"{self.path.name}.{os.getpid()}.tmp")
        tmp.write_text(json.dumps(data, indent=1, sort_keys=True))
        tmp.replace(self.path)

    @contextlib.contextmanager
    def _locked(self):
        """An exclusive lock file next to the DB (``O_CREAT | O_EXCL``,
        which NFS honours); a lock older than ``10 * lock_timeout`` is
        considered abandoned and broken."""
        if self.path is None:
            yield
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        lock = self.path.with_name(self.path.name + ".lock")
        t0 = time.monotonic()
        while True:
            try:
                fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(fd, f"{os.getpid()}\n".encode())
                os.close(fd)
                break
            except (FileExistsError, PermissionError):
                # (Windows reports a lock file being deleted as EACCES)
                try:
                    age = time.time() - lock.stat().st_mtime
                except OSError:
                    time.sleep(0.01)
                    continue
                if age > 10 * self.lock_timeout:
                    with contextlib.suppress(OSError):
                        lock.unlink()
                    continue
                if time.monotonic() - t0 > self.lock_timeout:
                    raise TimeoutError(
                        f"tuning DB lock {lock} - held for {age:.0f} s, expected "
                        f"release within lock_timeout={self.lock_timeout:g} s"
                    ) from None
                time.sleep(0.05)
        try:
            yield
        finally:
            with contextlib.suppress(OSError):
                lock.unlink()


def _feature_distance(a: Mapping[str, float], b: Mapping[str, float]) -> float | None:
    keys = sorted(set(a) & set(b))
    if not keys:
        return None
    s = 0.0
    for k in keys:
        x, y = float(a[k]), float(b[k])
        if x > 0 and y > 0:
            s += (math.log2(x) - math.log2(y)) ** 2
        else:
            s += (x - y) ** 2
    return math.sqrt(s)


class TraceWriter:
    """Appends a search trace as JSON lines (``path``), or keeps it in
    memory (``path=None``; read :attr:`rows`)."""

    def __init__(self, path: str | os.PathLike | None = None, *, append: bool = False):
        self.path = Path(path) if path else None
        self.rows: list[dict] = []
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            if not append and self.path.exists():
                self.path.unlink()

    def write(self, row: Mapping) -> None:
        row = dict(row)
        self.rows.append(row)
        if self.path is not None:
            with self.path.open("a") as f:
                f.write(json.dumps(row, sort_keys=True) + "\n")

    def header(self, **fields) -> None:
        self.write({"type": "header", **fields})

    def observation(
        self,
        ob: Observation,
        *,
        config: Mapping,
        elapsed: float,
        incumbent: float,
        n: int,
    ) -> None:
        d = ob.to_json()
        d.update(
            type="obs",
            n=n,
            config=dict(config),
            elapsed=elapsed,
            incumbent=incumbent if math.isfinite(incumbent) else None,
        )
        self.write(d)

    def __len__(self) -> int:
        return len(self.rows)


def read_trace(path: str | os.PathLike) -> list[dict]:
    rows = []
    with Path(path).open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def records_from_rows(rows: Sequence[Mapping]) -> list[TunedRecord]:
    """Records from JSON rows (``TunedRecord.to_json`` output)."""
    return [TunedRecord.from_json(r) for r in rows]
