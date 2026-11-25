# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Verify what a fresh process dispatches for a tuned row.

``verify_dispatch`` starts a child that runs one row through the public
operator with the family's config file pointed at the candidate CSV, and,
when the row names an expected selection, checks the operator recorded it:

- the operator calls ``record_selection(family, ...)`` where it dispatches;
- the family's tuner module runs ``run_selection_probe(probe)`` when started
  with ``--_selection_probe``;
- ``TunerCommon.verify_dispatch`` starts that child and checks the result.

Standard library only: operators import it on their dispatch path, and tests
load it by file path without importing aiter.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Mapping
from typing import Any, NamedTuple

DISPATCH_TRACE_ENV = "AITER_DISPATCH_TRACE_FILE"
SELECTION_PROBE_REQUEST_ENV = "AITER_SELECTION_PROBE_REQUEST"
SELECTION_PROBE_FLAG = "--_selection_probe"
_RESULT_PREFIX = "SELECTION_PROBE_RESULT="
_RECORD_METADATA = ("family", "pid")

_recorded: set[tuple[str, tuple[tuple[str, Any], ...]]] = set()


def record_selection(family: str, **selection: Any) -> None:
    """Append what the operator dispatched to; a no-op outside a probe child.

    Each distinct selection is written once per process, so timing loops do
    not grow the file. Values must be hashable scalars.
    """
    path = os.environ.get(DISPATCH_TRACE_ENV)
    if not path:
        return
    identity = (family, tuple(sorted(selection.items())))
    if identity in _recorded:
        return
    _recorded.add(identity)
    payload = json.dumps(
        {"family": family, "pid": os.getpid(), **selection},
        sort_keys=True,
        separators=(",", ":"),
    )
    descriptor = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
    try:
        os.write(descriptor, f"{payload}\n".encode())
    finally:
        os.close(descriptor)


def run_selection_probe(probe: Callable[..., Mapping[str, Any]]) -> int:
    """Child side of a verification: run the requested row through ``probe``.

    ``probe(row, warmup=..., iters=...)`` must call the public operator the
    way serving does, raise if its output disagrees with the reference, and
    return at least ``latency_us``.
    """
    request = json.loads(os.environ[SELECTION_PROBE_REQUEST_ENV])
    result = dict(
        probe(request["row"], warmup=request["warmup"], iters=request["iters"])
    )
    print(_RESULT_PREFIX + json.dumps(result, sort_keys=True), flush=True)
    return 0


class DispatchVerification(NamedTuple):
    """Outcome of one fresh-process verification.

    ``status``: ``verified`` (correct, recorded only the expected selection),
    ``unchecked`` (correct, nothing expected), ``wrong_selection`` (correct,
    dispatched elsewhere) or ``failed`` (the child failed or was unreadable).
    """

    status: str
    expected: dict[str, Any] | None
    observed: list[dict[str, Any]]
    latency_us: float | None
    detail: str
    elapsed_s: float


def read_selections(path: str, family: str) -> list[dict[str, Any]]:
    """The distinct selections ``family`` recorded in ``path``, in order."""
    if not os.path.exists(path):
        return []
    selections: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("family") != family:
                continue
            selection = {
                key: value
                for key, value in record.items()
                if key not in _RECORD_METADATA
            }
            if selection not in selections:
                selections.append(selection)
    return selections


def _value_text(value: Any) -> str:
    # Same as base_tuner._key_text: a CSV cell may come back as "32" or 32.0.
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    return str(value).strip()


def _check_expected_selection(
    expected: Mapping[str, Any] | None, observed: list[dict[str, Any]]
) -> tuple[str, str]:
    if expected is None:
        return "unchecked", ""
    if not observed:
        return "wrong_selection", "the operator recorded no selection"
    stray = [
        selection
        for selection in observed
        if any(
            _value_text(selection.get(key)) != _value_text(value)
            for key, value in expected.items()
        )
    ]
    if stray:
        return "wrong_selection", f"expected {dict(expected)}, observed {observed}"
    return "verified", ""


def _usable_latency(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value > 0
    )


def verify_dispatch(
    *,
    family: str,
    probe_module: str,
    row: Mapping[str, Any],
    expected: Mapping[str, Any] | None,
    config_env: str | None,
    config_file: str | None,
    warmup: int,
    iters: int,
    timeout: float,
    repository_root: str,
) -> DispatchVerification:
    """Run ``row`` in a fresh process and check what it dispatched to.

    The child runs from ``repository_root``, so ``import aiter`` resolves to
    the checkout being tuned rather than an installed copy.
    ``config_file=None`` keeps the caller's config.
    """
    started = time.time()
    descriptor, trace_path = tempfile.mkstemp(prefix="dispatch-trace-", suffix=".jsonl")
    os.close(descriptor)
    os.unlink(trace_path)
    environment = os.environ.copy()
    if config_env and config_file:
        environment[config_env] = os.pathsep.join(
            os.path.abspath(path) for path in config_file.split(os.pathsep)
        )
    environment[DISPATCH_TRACE_ENV] = trace_path
    environment[SELECTION_PROBE_REQUEST_ENV] = json.dumps(
        {"row": dict(row), "warmup": int(warmup), "iters": int(iters)}
    )
    environment["PYTHONPATH"] = os.pathsep.join(
        path for path in (repository_root, os.environ.get("PYTHONPATH", "")) if path
    )
    expected_dict = dict(expected) if expected is not None else None

    def outcome(status, observed=(), latency_us=None, detail=""):
        return DispatchVerification(
            status=status,
            expected=expected_dict,
            observed=list(observed),
            latency_us=latency_us,
            detail=detail,
            elapsed_s=time.time() - started,
        )

    if config_file and not config_env:
        return outcome(
            "failed",
            detail="no config env var to point the probe at the candidate CSV",
        )
    try:
        completed = subprocess.run(
            [sys.executable, "-m", probe_module, SELECTION_PROBE_FLAG],
            cwd=repository_root,
            env=environment,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        observed = read_selections(trace_path, family)
    except subprocess.TimeoutExpired:
        return outcome("failed", detail=f"probe timed out after {timeout} s")
    except (OSError, ValueError) as error:
        return outcome("failed", detail=f"probe could not be checked: {error}")
    finally:
        if os.path.exists(trace_path):
            os.unlink(trace_path)

    result_line = next(
        (
            line.removeprefix(_RESULT_PREFIX)
            for line in completed.stdout.splitlines()
            if line.startswith(_RESULT_PREFIX)
        ),
        None,
    )
    if completed.returncode != 0 or result_line is None:
        return outcome(
            "failed",
            observed,
            detail=(
                f"probe exited with {completed.returncode}: "
                f"{completed.stderr[-2000:].strip()}"
            ),
        )
    try:
        result = json.loads(result_line)
    except ValueError as error:
        return outcome("failed", observed, detail=f"unreadable probe result: {error}")
    latency_us = result.get("latency_us") if isinstance(result, dict) else None
    if not _usable_latency(latency_us):
        return outcome(
            "failed",
            observed,
            detail=f"probe returned no usable latency_us: {latency_us!r}",
        )
    status, detail = _check_expected_selection(expected_dict, observed)
    return outcome(status, observed, latency_us, detail)
