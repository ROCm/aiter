# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Record untuned GEMM shapes in process-local CSV shards.

Set ``AITER_TUNE_GEMM=1`` to enable recording. ``AITER_TUNE_GEMM_DIR`` selects
the output directory (for example ``/tuning/glm-5.2``); otherwise shards are
written below the current working directory, never inside the installed package.
Tuners already de-duplicate their inputs, so a later merge of ``*.pid.csv``
shards avoids interprocess coordination in dispatch paths.
"""

import os
import threading

from aiter import logger

_ENABLED = None
_LOCK = threading.Lock()
_SEEN: dict[str, set[tuple[str, ...]]] = {}


def enabled() -> bool:
    global _ENABLED
    if _ENABLED is None:
        _ENABLED = os.environ.get("AITER_TUNE_GEMM", "0") not in ("0", "", "false")
    return _ENABLED


def untuned_path_for(tuned_file: str) -> str:
    """Return this process's shard for ``tuned_file``'s untuned counterpart."""
    base = os.path.basename(tuned_file).replace("_tuned_", "_untuned_", 1)
    if base == os.path.basename(tuned_file):
        base = "untuned_" + base
    stem, extension = os.path.splitext(base)
    out_dir = os.environ.get("AITER_TUNE_GEMM_DIR") or os.getcwd()
    return os.path.join(out_dir, f"{stem}.{os.getpid()}{extension}")


def _write_all(fd: int, data: bytes) -> None:
    while data:
        written = os.write(fd, data)
        if written <= 0:
            raise OSError("short write while recording untuned GEMM shape")
        data = data[written:]


def record(tuned_file: str, row: dict) -> None:
    """Append a locally unique shape to this worker's shard without raising."""
    if not enabled():
        return
    try:
        path = untuned_path_for(tuned_file)
        cols, values = list(row), tuple(str(value) for value in row.values())
        with _LOCK:
            seen = _SEEN.setdefault(path, set())
            if values in seen:
                return
            os.makedirs(os.path.dirname(path), exist_ok=True)
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
            try:
                if os.fstat(fd).st_size == 0:
                    _write_all(fd, (",".join(cols) + "\n").encode())
                _write_all(fd, (",".join(values) + "\n").encode())
            finally:
                os.close(fd)
            seen.add(values)
            logger.info(f"[AITER_TUNE_GEMM] recorded untuned shape in {path}")
    except Exception as error:  # noqa: BLE001 - telemetry must not break dispatch
        logger.warning(f"[AITER_TUNE_GEMM] could not record untuned shape: {error}")
