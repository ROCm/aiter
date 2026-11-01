# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Record untuned GEMM shapes in process-local CSV shards.

Set ``AITER_TUNE_GEMM=1`` and ``AITER_TUNE_GEMM_DIR=/tuning/<model-or-run>``
to enable recording. A directory is mandatory: recording model-unscoped shapes
in a process working directory makes later merges ambiguous. Each worker writes
``<family>_untuned_gemm.<shard-id>.csv``. Set ``AITER_TUNE_GEMM_SHARD_ID`` to a
rank or pod ID; otherwise the hostname and PID provide the worker identifier.
Tuners already de-duplicate their inputs, so a later merge of shards avoids
interprocess coordination in dispatch paths.
"""

import os
import re
import socket
import threading

from aiter import logger

_ENABLED = None
_MISSING_DIR_WARNING_EMITTED = False
_LOCK = threading.Lock()
_SEEN: dict[str, set[tuple[str, ...]]] = {}


def enabled() -> bool:
    global _ENABLED, _MISSING_DIR_WARNING_EMITTED
    if _ENABLED is None:
        _ENABLED = os.environ.get("AITER_TUNE_GEMM", "0") not in ("0", "", "false")
        if _ENABLED and not os.environ.get("AITER_TUNE_GEMM_DIR"):
            if not _MISSING_DIR_WARNING_EMITTED:
                logger.warning(
                    "[AITER_TUNE_GEMM] recording is disabled: set "
                    "AITER_TUNE_GEMM_DIR to a model- or run-specific directory"
                )
                _MISSING_DIR_WARNING_EMITTED = True
            _ENABLED = False
    return _ENABLED


def untuned_path_for(tuned_file: str) -> str:
    """Return this process's shard for ``tuned_file``'s untuned counterpart."""
    base = os.path.basename(tuned_file).replace("_tuned_", "_untuned_", 1)
    if base == os.path.basename(tuned_file):
        base = "untuned_" + base
    stem, extension = os.path.splitext(base)
    out_dir = os.environ.get("AITER_TUNE_GEMM_DIR")
    if not out_dir:
        raise ValueError("AITER_TUNE_GEMM_DIR is required when recording untuned GEMMs")
    shard_id = os.environ.get("AITER_TUNE_GEMM_SHARD_ID")
    if not shard_id:
        shard_id = f"{socket.gethostname()}.{os.getpid()}"
    # Keep the ID one filename component even when it comes from a pod name or
    # another external launcher value.
    shard_id = re.sub(r"[^A-Za-z0-9_.-]", "_", shard_id)
    if not shard_id or shard_id in {".", ".."}:
        raise ValueError("AITER_TUNE_GEMM_SHARD_ID must contain a filename-safe value")
    return os.path.join(out_dir, f"{stem}.{shard_id}{extension}")


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
