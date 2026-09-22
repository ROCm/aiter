# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Optional per-process plot capture. This module has no GPU/plot dependencies."""

from __future__ import annotations

import atexit
import os
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

_lock = threading.Lock()
_session = None
_capture_failed = False


def _warning(message):
    print(f"AITER SMI plotting: {message}", file=sys.stderr, flush=True)


class _PlotSession:
    def __init__(self):
        self.pid = os.getpid()
        root = Path(os.environ.get("AITER_SMI_PLOT_DIR") or "smi_plots").resolve()
        root.mkdir(parents=True, exist_ok=True)
        self.directory = Path(tempfile.mkdtemp(prefix="run-", dir=root))
        self.input_path = self.directory / "smi_results.log"
        self.report = self.directory / "report"
        self.records = 0
        self.finished = False

    def append(self, line):
        # Close on every emission so records remain usable after abnormal exit.
        with self.input_path.open("a", encoding="utf-8") as output:
            output.write(line + "\n")
        self.records += 1

    def finish(self):
        if self.finished or self.pid != os.getpid() or not self.records:
            return None
        self.finished = True
        log_path = self.directory / "plotter.log"
        try:
            with log_path.open("w", encoding="utf-8") as output:
                result = subprocess.run(
                    [
                        sys.executable,
                        str(Path(__file__).with_name("smi_plot.py")),
                        str(self.input_path),
                        "--output",
                        str(self.report),
                    ],
                    stdout=output,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            if result.returncode:
                _warning(
                    f"plotter exited with status {result.returncode}; "
                    f"see {log_path}. Records preserved at {self.input_path}"
                )
                return None
        except OSError as error:
            _warning(f"could not render report: {error}. Records: {self.input_path}")
            return None
        index = self.report / "index.html"
        _warning(f"report ready: {index}")
        return index


def record_smi_result(line: str) -> None:
    """Copy a serialized record only when AITER_SMI_PLOT is exactly '1'."""
    global _session, _capture_failed

    if os.environ.get("AITER_SMI_PLOT", "0") != "1":
        return
    with _lock:
        if _capture_failed:
            return
        try:
            if _session is None:
                _session = _PlotSession()
            _session.append(line)
        except OSError as error:
            # Warn once, leaving ordinary benchmark output unaffected.
            _capture_failed = True
            _warning(f"could not capture plot input: {error}")


def flush_smi_plots():
    """Render pending records once. Explicit calls also work before shutdown."""
    global _session

    with _lock:
        session, _session = _session, None
    if session is None:
        return None
    return session.finish()


def _reset_after_fork():
    # A fork may inherit a held lock and the parent's partially written report.
    global _lock, _session, _capture_failed

    _lock = threading.Lock()
    _session = None
    _capture_failed = False


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reset_after_fork)

atexit.register(flush_smi_plots)
