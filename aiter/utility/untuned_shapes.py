# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Record the shapes that miss a tuned table, in the schema its tuner consumes.

``AITER_TUNE_GEMM=1`` has long done this for the bf16 path (``tuned_gemm.py``),
which is the only reason tuning a real bf16 deployment is self-service: run the
server, collect ``bf16_untuned_gemm.csv``, feed it straight back to the tuner.
The fp8 / block-scale / mxfp4 families log their misses to INFO and write
nothing, so their shape lists have to be scraped out of server logs by hand --
which is also how shape-key mistakes creep in.

This module gives every family the same behaviour. Call :func:`record` from a
lookup's miss path with the row its tuner expects; the file name is derived
from the tuned table's own name (``*_tuned_*`` -> ``*_untuned_*``) so a family
never has to name its untuned file twice.

Environment:
    AITER_TUNE_GEMM=1        enable recording (same switch as the bf16 path)
    AITER_TUNE_GEMM_DIR=DIR  write there instead of ``aiter/configs``; useful
                             when the package directory is read-only or lives
                             inside a container you would rather not reach into
"""

import fcntl
import os
import threading

from aiter import logger

_ENABLED = None
_LOCK = threading.Lock()
# file path -> {ordered column names, set of row tuples already written}
_SEEN: dict = {}
_THIS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def enabled() -> bool:
    global _ENABLED
    if _ENABLED is None:
        _ENABLED = os.environ.get("AITER_TUNE_GEMM", "0") not in ("0", "", "false")
    return _ENABLED


def untuned_path_for(tuned_file: str) -> str:
    """``.../a8w8_bpreshuffle_tuned_gemm.csv`` -> ``<dir>/a8w8_bpreshuffle_untuned_gemm.csv``

    The runtime may be reading a merged copy out of ``/tmp/aiter_configs``, so
    only the base name is reused; the destination directory is always the
    configs dir (or ``AITER_TUNE_GEMM_DIR``).
    """
    base = os.path.basename(tuned_file)
    if "_tuned_" in base:
        base = base.replace("_tuned_", "_untuned_", 1)
    else:
        base = "untuned_" + base
    out_dir = os.environ.get("AITER_TUNE_GEMM_DIR") or os.path.join(
        _THIS_DIR, "configs"
    )
    return os.path.join(out_dir, base)


def record(tuned_file: str, row: dict) -> None:
    """Append one missed shape, de-duplicated, in the tuner's input schema.

    Cheap enough for a dispatch path: a set lookup when the shape has been seen
    before (the common case -- a serving run repeats the same shapes), and one
    small append otherwise. Never raises: a read-only configs directory or a
    full disk must not take down inference.
    """
    if not enabled():
        return
    try:
        path = untuned_path_for(tuned_file)
        key = tuple(str(v) for v in row.values())
        with _LOCK:
            state = _SEEN.get(path)
            if state is not None and key in state["rows"]:
                return

            cols = list(row.keys())
            first_use = state is None
            os.makedirs(os.path.dirname(path), exist_ok=True)
            # _LOCK only covers threads in this process. Serving commonly has
            # several worker processes writing the same collection, so protect
            # initialization, the disk-level duplicate check, and append with
            # an advisory interprocess lock as one transaction.
            with open(path + ".lock", "a") as lock_fh:
                fcntl.flock(lock_fh, fcntl.LOCK_EX)
                existing = set()
                if os.path.exists(path):
                    with open(path) as fh:
                        header = fh.readline().strip().split(",")
                        if header == cols:
                            for line in fh:
                                line = line.strip()
                                if line:
                                    existing.add(tuple(line.split(",")))
                        else:  # different schema on disk: start a fresh file
                            os.replace(path, path + ".bak")
                if not os.path.exists(path):
                    with open(path, "w") as fh:
                        fh.write(",".join(cols) + "\n")
                # Publish initialization state only after the directory and a
                # valid CSV header exist, so a transient failure can recover.
                state = _SEEN[path] = {"cols": cols, "rows": existing}
                if first_use:
                    logger.info(f"[AITER_TUNE_GEMM] recording untuned shapes to {path}")
                if key in state["rows"]:
                    return
                needs_separator = os.path.getsize(path) > 0
                if needs_separator:
                    with open(path, "rb") as fh:
                        fh.seek(-1, os.SEEK_END)
                        needs_separator = fh.read(1) not in (b"\n", b"\r")
                original_size = os.path.getsize(path)
                try:
                    with open(path, "a") as fh:
                        if needs_separator:
                            fh.write("\n")
                        fh.write(",".join(key) + "\n")
                except Exception:
                    # write() and close() may fail after flushing only a prefix.
                    # Restore the last known-good boundary while holding the
                    # interprocess lock so the next dispatch can retry cleanly.
                    try:
                        with open(path, "r+b") as fh:
                            fh.truncate(original_size)
                    except OSError:
                        pass
                    raise
                # Only cache a row after its append succeeds. A transient write
                # failure must remain retryable on the next dispatch.
                state["rows"].add(key)
    except Exception as e:  # noqa: BLE001 - never break dispatch over telemetry
        logger.warning(f"[AITER_TUNE_GEMM] could not record untuned shape: {e}")
