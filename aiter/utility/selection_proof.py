# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Prove that a tuned row is what a fresh process actually dispatches.

A tuner measures a candidate in its own worker, but serving reads the tuned
CSV through the public operator in a new process. ``prove_selection`` closes
that gap: it starts a child that runs one row through the public operator
with the family's config file redirected to the candidate CSV, and checks
that the operator recorded the selection the row names.

Three pieces, each owned by a different side:

- the operator calls ``record_selection(family, ...)`` where it dispatches;
- the family's tuner module runs ``run_selection_probe(probe)`` when started
  with ``--_selection_probe``, with a ``probe`` that calls the public
  operator, checks it against a reference and times it;
- ``TunerCommon.prove_selection`` starts that child and judges the result.

Standard library only: operators import it on their dispatch path, and tests
load it by file path without importing aiter.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Mapping
from typing import Any, NamedTuple

SELECTION_PROOF_ENV = "AITER_SELECTION_PROOF_FILE"
SELECTION_PROBE_REQUEST_ENV = "AITER_SELECTION_PROBE_REQUEST"
SELECTION_PROBE_FLAG = "--_selection_probe"
_RESULT_PREFIX = "SELECTION_PROBE_RESULT="
_RECORD_METADATA = ("family", "pid")

_recorded: set[tuple[str, tuple[tuple[str, Any], ...]]] = set()


def record_selection(family: str, **selection: Any) -> None:
    """Append what the operator dispatched to, when a probe asks for it.

    A no-op unless ``AITER_SELECTION_PROOF_FILE`` is set, which only a probe
    child does. Each distinct selection is written once per process, so
    timing loops do not grow the file. Values must be hashable scalars.
    """

    path = os.environ.get(SELECTION_PROOF_ENV)
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


def run_selection_probe(
    probe: Callable[..., Mapping[str, Any]],
) -> int:
    """Child side of a proof: run the requested row through ``probe``.

    ``probe(row, warmup=..., iters=...)`` must call the public operator the
    way serving does, raise if its output disagrees with the reference, and
    return at least ``latency_us``. A raised exception fails the proof.
    """

    request = json.loads(os.environ[SELECTION_PROBE_REQUEST_ENV])
    result = dict(
        probe(request["row"], warmup=request["warmup"], iters=request["iters"])
    )
    result.setdefault("correctness", "ok")
    print(_RESULT_PREFIX + json.dumps(result, sort_keys=True), flush=True)
    return 0


class SelectionProof(NamedTuple):
    """Outcome of one fresh-process proof.

    ``status`` is ``verified`` when the child ran, matched the reference and
    recorded only the expected selection; ``wrong_selection`` when it ran
    correctly but dispatched elsewhere; ``failed`` when the child itself
    failed or could not be judged.
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


def _judge(
    expected: Mapping[str, Any] | None, observed: list[dict[str, Any]]
) -> tuple[str, str]:
    if expected is None:
        return "verified", ""
    if not observed:
        return "wrong_selection", "the operator recorded no selection"
    stray = [
        selection
        for selection in observed
        if any(selection.get(key) != value for key, value in expected.items())
    ]
    if stray:
        return "wrong_selection", f"expected {dict(expected)}, observed {observed}"
    return "verified", ""


def prove_selection(
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
    extra_env: Mapping[str, str] | None = None,
) -> SelectionProof:
    """Run ``row`` in a fresh process and check what it dispatched to.

    The child is started as ``python -m probe_module --_selection_probe``
    from ``repository_root``, so ``import aiter`` resolves to the checkout
    being tuned rather than an installed copy. ``config_file=None`` keeps the
    caller's config. ``expected=None`` checks only that the public operator
    runs and matches its reference.
    """

    started = time.time()
    descriptor, proof_path = tempfile.mkstemp(prefix="selection-", suffix=".jsonl")
    os.close(descriptor)
    os.unlink(proof_path)
    environment = os.environ.copy()
    environment.update(extra_env or {})
    if config_env and config_file:
        environment[config_env] = os.path.abspath(config_file)
    environment[SELECTION_PROOF_ENV] = proof_path
    environment[SELECTION_PROBE_REQUEST_ENV] = json.dumps(
        {"row": dict(row), "warmup": int(warmup), "iters": int(iters)}
    )
    environment["PYTHONPATH"] = os.pathsep.join(
        path for path in (repository_root, os.environ.get("PYTHONPATH", "")) if path
    )
    expected_dict = dict(expected) if expected is not None else None

    def outcome(status, observed=(), latency_us=None, detail=""):
        return SelectionProof(
            status=status,
            expected=expected_dict,
            observed=list(observed),
            latency_us=latency_us,
            detail=detail,
            elapsed_s=time.time() - started,
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
        observed = read_selections(proof_path, family)
    except subprocess.TimeoutExpired:
        return outcome("failed", detail=f"probe timed out after {timeout} s")
    except (OSError, ValueError) as error:
        return outcome("failed", detail=f"probe could not be judged: {error}")
    finally:
        if os.path.exists(proof_path):
            os.unlink(proof_path)

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
    result = json.loads(result_line)
    if result.get("correctness") != "ok":
        return outcome("failed", observed, detail=f"correctness: {result}")
    status, detail = _judge(expected_dict, observed)
    return outcome(status, observed, result.get("latency_us"), detail)
