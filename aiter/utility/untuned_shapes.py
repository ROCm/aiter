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
                             inside a container you would rather not reach into.
                             Set this to a model-specific directory such as
                             ``/tuning/glm-5.2`` to collect every GEMM family in
                             one place.
"""

import os
import tempfile
import threading

from aiter import logger

_ENABLED = None
_LOCK = threading.Lock()
# file path -> {ordered column names, process-local rows, separator state}
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
    """Append one missed shape in the tuner's input schema.

    Rows are de-duplicated within this process. Different workers may append the
    same row; the tuners already drop duplicates when reading their inputs. Each
    new row is one ``O_APPEND`` write, so workers never need an interprocess lock
    or a full-file duplicate scan. Never raises: a read-only output directory or
    a full disk must not take down inference.
    """
    if not enabled():
        return
    try:
        path = untuned_path_for(tuned_file)
        cols = list(row.keys())
        key = tuple(str(v) for v in row.values())
        with _LOCK:
            state = _SEEN.get(path)
            if state is not None and key in state["rows"]:
                return

            first_use = state is None
            if first_use:
                os.makedirs(os.path.dirname(path), exist_ok=True)
                needs_separator = _ensure_header(path, cols)
                # Publish initialization state only after the directory and a
                # valid CSV header exist, so a transient failure can recover.
                state = _SEEN[path] = {
                    "cols": cols,
                    "rows": set(),
                    "needs_separator": needs_separator,
                }
            elif state["cols"] != cols:
                raise ValueError(
                    f"schema for {path} changed from {state['cols']} to {cols}"
                )

            if first_use:
                logger.info(f"[AITER_TUNE_GEMM] recording untuned shapes to {path}")

            prefix = "\n" if state["needs_separator"] else ""
            _append_line(path, (prefix + ",".join(key) + "\n").encode())
            state["needs_separator"] = False
            # Only cache a row after its append succeeds. A transient write
            # failure must remain retryable on the next dispatch.
            state["rows"].add(key)
    except Exception as e:  # noqa: BLE001 - never break dispatch over telemetry
        logger.warning(f"[AITER_TUNE_GEMM] could not record untuned shape: {e}")


def _ensure_header(path: str, cols: list[str]) -> bool:
    """Atomically publish a new header and validate an existing one once.

    A temporary file plus ``link`` ensures another worker cannot observe an
    empty destination between file creation and header write. Returns whether
    the existing file needs a newline before its first appended row.
    """
    try:
        fh = open(path, "rb")
    except FileNotFoundError:
        header = (",".join(cols) + "\n").encode()
        fd, temp_path = tempfile.mkstemp(
            dir=os.path.dirname(path),
            prefix=f".{os.path.basename(path)}.",
            suffix=".tmp",
        )
        try:
            with os.fdopen(fd, "wb") as temp_fh:
                temp_fh.write(header)
            try:
                os.link(temp_path, path)
            except FileExistsError:
                pass
        finally:
            try:
                os.unlink(temp_path)
            except FileNotFoundError:
                pass
        fh = open(path, "rb")

    with fh:
        disk_cols = fh.readline().rstrip(b"\r\n").decode().split(",")
        if disk_cols != cols:
            raise ValueError(
                f"schema for {path} is {disk_cols}, expected {cols}; refusing to append"
            )
        fh.seek(0, os.SEEK_END)
        if fh.tell() == 0:
            return False
        fh.seek(-1, os.SEEK_END)
        return fh.read(1) not in (b"\n", b"\r")


def _append_line(path: str, payload: bytes) -> None:
    """Append a complete CSV row with one operating-system write."""
    fd = os.open(path, os.O_WRONLY | os.O_APPEND)
    try:
        written = os.write(fd, payload)
        if written != len(payload):
            raise OSError(
                f"short append to {path}: wrote {written}/{len(payload)} bytes"
            )
    finally:
        os.close(fd)
