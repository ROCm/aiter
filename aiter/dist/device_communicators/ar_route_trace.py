# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Which backend served each fused all-reduce call.

Off unless ``AITER_AR_ROUTE_TRACE`` is set; then ``note`` counts one entry per
``(op, backend, rows, hidden, dtype, gemma, captured)`` and rewrites a JSON file
whenever a new key appears and every 512 calls (so a killed server still
leaves its table).

* ``AITER_AR_ROUTE_TRACE=1``: counts are logged at exit.
* ``AITER_AR_ROUTE_TRACE=<path>``: also written to ``<path>.rank<N>.json``.

A call recorded inside a CUDA graph capture runs its Python once, at capture,
so ``captured=True`` entries count graph recordings (one per captured batch
size and call site), not replays; eager calls count every call.
"""

from __future__ import annotations

import atexit
import json
import logging
import os
import threading

import torch

logger = logging.getLogger("aiter")

ENV = "AITER_AR_ROUTE_TRACE"
_raw = os.environ.get(ENV, "").strip()
ENABLED = _raw not in ("", "0")
_PATH = _raw if ENABLED and _raw != "1" else None

_lock = threading.Lock()
_counts: dict[tuple, int] = {}


def _rank() -> int:
    try:
        import torch.distributed as dist

        return dist.get_rank() if dist.is_initialized() else 0
    except Exception:  # noqa: BLE001
        return 0


def note(op: str, backend: str, inp: torch.Tensor, *, gemma: bool | None = None):
    """Count one call of *op* served by *backend*. No-op unless enabled."""
    if not ENABLED:
        return
    try:
        captured = torch.cuda.is_current_stream_capturing()
    except Exception:  # noqa: BLE001
        captured = False
    key = (
        op,
        backend,
        int(inp.numel() // inp.shape[-1]),
        int(inp.shape[-1]),
        str(inp.dtype).replace("torch.", ""),
        gemma,
        bool(captured),
    )
    with _lock:
        new = key not in _counts
        _counts[key] = _counts.get(key, 0) + 1
        total = sum(_counts.values())
    # A killed worker gets no exit hook: write on every new key, then every
    # 512 calls.
    if _PATH and (new or total % 512 == 0):
        dump()


def table() -> list[dict]:
    """The counts as rows, sorted by op, backend and rows."""
    fields = ("op", "backend", "rows", "hidden", "dtype", "gemma", "captured")
    with _lock:
        items = sorted(_counts.items(), key=lambda kv: tuple(map(str, kv[0])))
    return [
        {**dict(zip(fields, key)), "bytes": key[2] * key[3] * 2, "calls": n}
        for key, n in items
    ]


def dump(path: str | None = None) -> None:
    path = path or (f"{_PATH}.rank{_rank()}.json" if _PATH else None)
    if path is None:
        return
    tmp = f"{path}.tmp"
    try:
        with open(tmp, "w") as f:
            json.dump(table(), f, indent=1)
        os.replace(tmp, path)
    except OSError:
        logger.warning("AITER_AR_ROUTE_TRACE: could not write %s", path)


def _at_exit() -> None:
    if not _counts:
        return
    logger.info("AITER_AR_ROUTE_TRACE rank %d: %s", _rank(), json.dumps(table()))
    dump()


if ENABLED:
    atexit.register(_at_exit)
