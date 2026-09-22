# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Optional lossless SMI sample export, independent of summary plotting."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import threading
from pathlib import Path

TRACE_PREFIX = "AITER_SMI_TRACE "
_lock = threading.Lock()
_output_path = None
_failed = False


def emit_smi_trace(result, samples, *, start_s, end_s, start_unix_s):
    """Persist every collected sample after the monitored workload completes.

    timestamp_s is the original perf_counter timestamp. Monotonic boundaries
    retain gaps between windows on this host; window_start_unix_s gives an
    approximate wall-clock anchor. No readings are averaged or filtered.
    Ordinary summary records and their destinations remain untouched.
    """
    global _output_path, _failed

    if os.environ.get("AITER_SMI_TRACE", "0") != "1":
        return None
    record = {
        **result,
        "schema_version": 1,
        "sample_count": len(samples),
        "window_start_monotonic_s": start_s,
        "window_end_monotonic_s": end_s,
        "window_start_unix_s": start_unix_s,
        "samples": samples,
    }
    with _lock:
        if _failed:
            return None
        try:
            if _output_path is None:
                root = Path(
                    os.environ.get("AITER_SMI_TRACE_DIR") or "smi_traces"
                ).resolve()
                root.mkdir(parents=True, exist_ok=True)
                directory = Path(tempfile.mkdtemp(prefix="run-", dir=root))
                _output_path = directory / "samples.jsonl"
                print(f"AITER SMI trace: {_output_path}", file=sys.stderr, flush=True)
            with _output_path.open("a", encoding="utf-8") as output:
                output.write(TRACE_PREFIX + json.dumps(record, sort_keys=True) + "\n")
        except (OSError, TypeError, ValueError) as error:
            _failed = True
            print(
                f"AITER SMI trace export failed: {error}", file=sys.stderr, flush=True
            )
            return None
    return _output_path


def _reset_after_fork():
    global _lock, _output_path, _failed

    _lock = threading.Lock()
    _output_path = None
    _failed = False


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reset_after_fork)
