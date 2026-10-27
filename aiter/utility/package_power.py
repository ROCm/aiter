# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Package power from the same hwmon file rocm-smi -P reads (power1_input, label PPT)."""

import ctypes
import glob
import logging
import os
import threading
import time

logger = logging.getLogger(__name__)


def _hip_pci_bus_id(device_index: int) -> str | None:
    candidates = [
        os.environ.get("AITER_HIP_LIB", ""),
        "libamdhip64.so",
        "/opt/rocm/lib/libamdhip64.so",
    ]
    lib = None
    for name in candidates:
        if not name:
            continue
        try:
            lib = ctypes.CDLL(name)
            break
        except OSError:
            continue
    if lib is None:
        return None
    buf = ctypes.create_string_buffer(64)
    status = lib.hipDeviceGetPCIBusId(buf, 64, ctypes.c_int(int(device_index)))
    if status != 0:
        return None
    return buf.value.decode().strip()


def power_input_path(device_index: int = 0) -> str | None:
    """hwmon power1_input for a HIP device ordinal in this process."""
    override = os.environ.get("AITER_POWER_PATH", "").strip()
    if override:
        return override
    bdf = _hip_pci_bus_id(device_index)
    if not bdf:
        return None
    bdf = bdf.lower()
    matches = glob.glob(f"/sys/bus/pci/devices/{bdf}/hwmon/hwmon*/power1_input")
    if matches:
        return matches[0]
    for dev in glob.glob("/sys/class/drm/card*/device"):
        try:
            real = os.path.realpath(dev)
        except OSError:
            continue
        if os.path.basename(real).lower() != bdf:
            continue
        hits = glob.glob(dev + "/hwmon/hwmon*/power1_input")
        if hits:
            return hits[0]
    return None


def _read_watts(path: str) -> float | None:
    try:
        with open(path, encoding="ascii") as handle:
            raw = handle.read().strip()
        uw = int(raw)
    except (OSError, ValueError):
        return None
    if uw <= 0:
        return None
    return uw / 1e6


def measure_package_watts(
    fn, seconds: float | None = None, device_index: int = 0
) -> float:
    """Run fn until `seconds` elapse and return the mean package power in watts.

    The timed GEMM loop in the tuner is often shorter than the PPT update, so callers
    that rank by power should pass a window of a few hundred milliseconds.
    """
    if seconds is None:
        seconds = float(os.environ.get("AITER_POWER_WINDOW_S", "0.4"))
    path = power_input_path(device_index)
    if path is None or seconds <= 0:
        return float("nan")

    samples: list[float] = []
    stop = threading.Event()

    def loop():
        while not stop.is_set():
            watts = _read_watts(path)
            if watts is not None:
                samples.append(watts)
            stop.wait(0.005)

    fn()
    _sync_cuda()
    thread = threading.Thread(target=loop, daemon=True)
    thread.start()
    deadline = time.perf_counter() + seconds
    try:
        while time.perf_counter() < deadline:
            fn()
    finally:
        _sync_cuda()
        stop.set()
        thread.join()
    if not samples:
        return float("nan")
    return sum(samples) / len(samples)


def _sync_cuda() -> None:
    try:
        import torch

        torch.cuda.synchronize()
    except Exception:
        logger.debug("CUDA synchronize skipped", exc_info=True)


def record_package_watts(fn, device_index: int | None = None) -> float:
    """Sample PPT for fn. Returns NaN when the sensor cannot be read."""
    if device_index is None:
        try:
            import torch

            device_index = int(torch.cuda.current_device())
        except Exception:  # noqa: BLE001
            device_index = 0
    try:
        return float(measure_package_watts(fn, device_index=device_index))
    except Exception:  # noqa: BLE001
        return float("nan")
