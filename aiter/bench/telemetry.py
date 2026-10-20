# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""GPU telemetry sampled during the timed window (not in a separate replay)."""

import contextlib


class _Window:
    def __init__(self):
        self.summary = None
        self.error = None


@contextlib.contextmanager
def telemetry_window(device_index, enabled, interval_s=0.01):
    """Poll GpuMonitor while the body runs; ``window.summary`` afterwards.

    When the amdsmi binding is unavailable, ``window.error`` says why and the
    body still runs. Windows shorter than ``interval_s`` yield no samples.
    """
    window = _Window()
    if not enabled:
        yield window
        return
    try:
        from aiter.smi_monitor import GpuMonitor

        monitor = GpuMonitor(device_index=device_index, interval_s=interval_s)
        monitor.start()
    except Exception as error:  # noqa: BLE001 - reported in the result
        window.error = f"telemetry unavailable: {error}"
        yield window
        return
    try:
        yield window
    finally:
        monitor.stop()
        window.summary = monitor.summary()
        if not window.summary:
            window.error = "telemetry window too short: no samples"
