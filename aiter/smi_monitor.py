# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
# ruff: noqa: BLE001, PYI034, S110, UP035, UP037
"""AMD GPU metrics monitor using amdsmi.

ROCm ships the binding without a setup.py. If the normal import fails, this
module temporarily searches ``/opt/rocm/share/amd_smi`` while importing it.
Neither ``PYTHONPATH`` nor the caller's lasting ``sys.path`` is changed.
The amdsmi session and HIP-device handle mapping are initialized lazily once
per process and reused by every monitor instance.

Usage (context manager):
    with GpuMonitor(device_index=0, interval_s=0.05) as mon:
        run_workload()
    samples = mon.samples   # list[dict]

Usage (explicit start/stop):
    mon = GpuMonitor(device_index=0, interval_s=0.05)
    mon.start()
    run_workload()
    mon.stop()
    samples = mon.samples
"""

from __future__ import annotations

import atexit
import ctypes
import importlib
import inspect
import json
import os
import sys
import threading
import time
from contextlib import contextmanager
from contextvars import ContextVar
from enum import Enum
from functools import cache
from typing import Generator

import numpy as np
import torch

SMI_RESULT_PREFIX = "AITER_SMI_RESULT "
_ROCM_AMDSMI_PATH = "/opt/rocm/share/amd_smi"
_SMI_LABEL_COUNTS = {}
_SMI_CALL_LABEL = ContextVar("aiter_smi_call_label", default=None)


def _smi_label_value(value):
    """Return a compact, stable label value, or None for opaque arguments."""
    if isinstance(value, torch.Tensor):
        shape = "x".join(map(str, value.shape)) or "scalar"
        return f"{shape}:{str(value.dtype).removeprefix('torch.')}"
    if isinstance(value, torch.dtype):
        return str(value).removeprefix("torch.")
    if isinstance(value, Enum):
        return str(value.value)
    if value is None or isinstance(value, (str, bool, int, float, np.generic)):
        return str(value)
    if isinstance(value, (tuple, list)):
        items = [_smi_label_value(item) for item in value]
        if all(item is not None for item in items):
            return ",".join(items)
    return None


def _smi_call_tag(func, callargs):
    """Build a call-local SMI label from @benchmark's named arguments."""
    source = os.path.splitext(os.path.basename(func.__code__.co_filename))[0]
    parts = [f"{source}.{func.__name__}"]
    aliases = {"m": "M", "n": "N", "k": "K", "t": "T", "h": "H", "d": "D"}
    for name, value in callargs.items():
        formatted = _smi_label_value(value)
        if formatted is None:
            continue
        formatted = formatted.replace("/", "_").replace("\n", "")
        parts.append(f"{aliases.get(name, name)}={formatted}")
    return "/".join(parts)


def _smi_perftest_tag(func, args, kwargs):
    """Return scalar call details that distinguish one perftest invocation."""
    try:
        signature = inspect.signature(func)
        callargs = signature.bind(*args, **kwargs)
        callargs.apply_defaults()
    except (TypeError, ValueError):
        return None

    parts = []
    for name, parameter in signature.parameters.items():
        if parameter.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue
        value = callargs.arguments.get(name)
        if value is None or isinstance(value, torch.Tensor) or callable(value):
            continue
        formatted = _smi_label_value(value)
        if formatted is None:
            continue
        formatted = formatted.replace("/", "_").replace("\n", "")
        parts.append(f"{name}={formatted}")
    return "/".join(parts) or None


@contextmanager
def benchmark_call_context(func, callargs):
    """Expose one benchmark call's metadata to an inner SMI replay."""
    token = _SMI_CALL_LABEL.set(_smi_call_tag(func, callargs))
    try:
        yield
    finally:
        _SMI_CALL_LABEL.reset(token)


def replay_with_smi_metadata(
    func,
    args,
    kwargs,
    replay,
    *,
    synchronize,
    estimated_us: float | None = None,
):
    """Replay one callable with stable benchmark metadata and skip filtering."""
    if not smi_replay_enabled():
        return None

    fn_name = getattr(func, "__name__", "kernel")
    skipped = {
        name.strip()
        for name in os.environ.get("AITER_SMI_SKIP_FUNCTIONS", "").split(",")
        if name.strip()
    }
    if fn_name in skipped:
        return None

    case_label = _SMI_CALL_LABEL.get() or os.environ.get(
        "AITER_SMI_LABEL", "benchmark_case"
    )
    perftest_tag = _smi_perftest_tag(func, args, kwargs)
    if perftest_tag:
        case_label = f"{case_label}/{perftest_tag}"
    label_key = (case_label, fn_name)
    occurrence = _SMI_LABEL_COUNTS.get(label_key, 0) + 1
    _SMI_LABEL_COUNTS[label_key] = occurrence
    return replay_with_smi(
        replay,
        label=f"{case_label}/{fn_name}#{occurrence}",
        synchronize=synchronize,
        estimated_us=estimated_us,
    )


def _import_amdsmi():
    """Import amdsmi, temporarily searching ROCm's unpackaged binding."""
    try:
        return importlib.import_module("amdsmi")
    except ImportError:
        if not os.path.isdir(_ROCM_AMDSMI_PATH):
            raise

    added_path = _ROCM_AMDSMI_PATH not in sys.path
    if added_path:
        sys.path.insert(0, _ROCM_AMDSMI_PATH)
    try:
        return importlib.import_module("amdsmi")
    finally:
        if added_path:
            sys.path.remove(_ROCM_AMDSMI_PATH)


try:
    amdsmi = _import_amdsmi()

    _AMDSMI_AVAILABLE = True
except ImportError:
    _AMDSMI_AVAILABLE = False


_AMDSMI_LOCK = threading.Lock()
_AMDSMI_INITIALIZED = False
_AMDSMI_HANDLES_BY_BDF = {}
_AMDSMI_HANDLES_BY_HIP_DEVICE = {}


# ------------------------------------------------------------------
# HIP device -> amdsmi handle via PCIe BDF
# ------------------------------------------------------------------


@cache
def _hip_runtime_library():
    """Load and cache the HIP runtime used for device-to-BDF lookup."""
    import glob as _glob

    candidates = ["libamdhip64.so"] + sorted(
        _glob.glob("/opt/rocm/lib/libamdhip64.so.*"), reverse=True
    )
    for name in candidates:
        try:
            return ctypes.CDLL(name)
        except OSError:
            continue
    raise RuntimeError(
        "libamdhip64.so not found (tried unversioned and /opt/rocm/lib/libamdhip64.so.*); "
        "is ROCm installed?"
    )


@cache
def _hip_device_bdf(hip_device: int) -> str:
    """Return the PCIe BDF string for a HIP device index, e.g. '0000:03:00.0'.

    Calls ``hipDeviceGetPCIBusId`` via ctypes so there is no hard dependency on
    PyTorch or the hip-python package.
    """
    libhip = _hip_runtime_library()
    buf = ctypes.create_string_buffer(64)
    ret = libhip.hipDeviceGetPCIBusId(buf, ctypes.c_int(64), ctypes.c_int(hip_device))
    if ret != 0:
        raise RuntimeError(f"hipDeviceGetPCIBusId failed with error code {ret}")
    return buf.value.decode().lower().strip()


def _amdsmi_bdf_str(handle) -> str:
    """Normalise the BDF returned by amdsmi into 'dddd:bb:dd.f' lowercase."""
    raw = amdsmi.amdsmi_get_gpu_device_bdf(handle)
    if isinstance(raw, str):
        return raw.lower().strip()
    # Some amdsmi versions return a dict: {'domain': 0, 'bus': 3, 'device': 0, 'function': 0}
    return (
        f"{raw['domain']:04x}:{raw['bus']:02x}:{raw['device']:02x}.{raw['function']:x}"
    )


def _ensure_amdsmi_initialized() -> None:
    """Initialize amdsmi and enumerate processor handles once per process."""
    global _AMDSMI_INITIALIZED, _AMDSMI_HANDLES_BY_BDF

    if not _AMDSMI_AVAILABLE:
        raise ImportError("amdsmi is not installed or not importable")
    with _AMDSMI_LOCK:
        if _AMDSMI_INITIALIZED:
            return
        amdsmi.amdsmi_init()
        try:
            _AMDSMI_HANDLES_BY_BDF = {
                _amdsmi_bdf_str(handle): handle
                for handle in amdsmi.amdsmi_get_processor_handles()
            }
        except BaseException:
            amdsmi.amdsmi_shut_down()
            raise
        _AMDSMI_INITIALIZED = True


def _shutdown_amdsmi() -> None:
    """Release the process-wide amdsmi session during interpreter shutdown."""
    global _AMDSMI_INITIALIZED

    with _AMDSMI_LOCK:
        if not _AMDSMI_INITIALIZED:
            return
        try:
            amdsmi.amdsmi_shut_down()
        finally:
            _AMDSMI_INITIALIZED = False
            _AMDSMI_HANDLES_BY_BDF.clear()
            _AMDSMI_HANDLES_BY_HIP_DEVICE.clear()


atexit.register(_shutdown_amdsmi)


def hip_device_to_amdsmi_handle(hip_device: int):
    """Return the amdsmi processor handle that corresponds to a HIP device index.

    Uses PCIe BDF as the stable identifier linking the two numbering schemes.

    Args:
        hip_device: HIP device ordinal (as used by ``torch.cuda`` / HIP runtime).

    Returns:
        The amdsmi processor handle for that GPU.

    Raises:
        RuntimeError: if no amdsmi handle matches the HIP device's BDF.
        ImportError: if amdsmi is not available.
    """
    _ensure_amdsmi_initialized()
    with _AMDSMI_LOCK:
        cached = _AMDSMI_HANDLES_BY_HIP_DEVICE.get(hip_device)
        if cached is not None:
            return cached
        target_bdf = _hip_device_bdf(hip_device)
        handle = _AMDSMI_HANDLES_BY_BDF.get(target_bdf)
        if handle is None:
            raise RuntimeError(
                f"No amdsmi handle found with BDF {target_bdf!r} "
                f"(HIP device {hip_device})"
            )
        _AMDSMI_HANDLES_BY_HIP_DEVICE[hip_device] = handle
        return handle


def _collect_sample(handle) -> dict:
    """Collect one snapshot from a single GPU handle."""
    sample: dict = {"timestamp_s": time.perf_counter()}
    try:
        metrics = amdsmi.amdsmi_get_gpu_metrics_info(handle)
        sample["gfx_clk_mhz"] = metrics.get("current_gfxclk", None)
        sample["soc_clk_mhz"] = metrics.get("current_socclk", None)
        sample["power_w"] = metrics.get("current_socket_power", None)
        sample["temp_hotspot_c"] = metrics.get("temperature_hotspot", None)
    except Exception:
        pass
    try:
        info = amdsmi.amdsmi_get_gpu_activity(handle)
        sample["gfx_activity_pct"] = info.get("gfx_activity", None)
        sample["umc_activity_pct"] = info.get("umc_activity", None)
    except Exception:
        pass
    try:
        mem = amdsmi.amdsmi_get_gpu_memory_usage(handle, amdsmi.AmdSmiMemoryType.VRAM)
        sample["vram_used_mb"] = mem / 1024 / 1024
    except Exception:
        pass
    return sample


class GpuMonitor:
    """Poll AMD GPU metrics on a background thread.

    Args:
        device_index: Integer ordinal of the GPU to monitor (default 0), or a
                      pre-resolved amdsmi processor handle (e.g. from
                      ``hip_device_to_amdsmi_handle``).
        interval_s:   Polling interval in seconds (default 0.05 = 50 ms).
    """

    def __init__(self, device_index: int = 0, interval_s: float = 0.05) -> None:
        if not _AMDSMI_AVAILABLE:
            raise ImportError("amdsmi is not installed or not importable")
        self._device_index = device_index
        self._interval_s = interval_s
        self._samples: list[dict] = []
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._ready_event = threading.Event()
        self._error: BaseException | None = None
        self._handle = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Begin background polling. Safe to call only once per instance."""
        if self._thread is not None and self._thread.is_alive():
            raise RuntimeError("GpuMonitor is already running")
        self._samples = []
        self._error = None
        self._stop_event.clear()
        self._ready_event.clear()
        try:
            if isinstance(self._device_index, int):
                self._handle = hip_device_to_amdsmi_handle(self._device_index)
            else:
                _ensure_amdsmi_initialized()
                self._handle = self._device_index
        except BaseException as error:
            raise RuntimeError(
                f"failed to initialize amdsmi monitor: {error}"
            ) from error
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()
        if not self._ready_event.wait(timeout=10.0):
            self._stop_event.set()
            raise RuntimeError("timed out while initializing amdsmi monitor")
        if self._error is not None:
            error = self._error
            self.stop()
            raise RuntimeError(f"failed to initialize amdsmi monitor: {error}")

    def stop(self) -> None:
        """Stop background polling and wait for the thread to finish."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join()
            self._thread = None

    @property
    def samples(self) -> list[dict]:
        """Collected samples; each is a dict with 'timestamp_s' plus metric keys."""
        return list(self._samples)

    def summary(
        self, *, start_s: float | None = None, end_s: float | None = None
    ) -> dict:
        """Return metric summaries, optionally restricted to a timestamp window."""
        samples = [
            sample
            for sample in self._samples
            if (start_s is None or sample["timestamp_s"] >= start_s)
            and (end_s is None or sample["timestamp_s"] <= end_s)
        ]
        if not samples:
            return {}
        keys = {key for sample in samples for key in sample if key != "timestamp_s"}
        result: dict = {}
        for key in sorted(keys):
            vals = sorted(
                s[key] for s in samples if s.get(key) is not None and s[key] != "N/A"
            )
            if not vals:
                continue
            n = len(vals)
            mid = n // 2
            median = vals[mid] if n % 2 else (vals[mid - 1] + vals[mid]) / 2
            result[key] = {
                "min": vals[0],
                "mean": sum(vals) / n,
                "median": median,
                "max": vals[-1],
                "n": n,
            }
        return result

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "GpuMonitor":
        self.start()
        return self

    def __exit__(self, *_) -> None:
        self.stop()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _poll_loop(self) -> None:
        try:
            self._ready_event.set()
            while not self._stop_event.is_set():
                t0 = time.perf_counter()
                self._samples.append(_collect_sample(self._handle))
                elapsed = time.perf_counter() - t0
                remaining = self._interval_s - elapsed
                if remaining > 0:
                    self._stop_event.wait(timeout=remaining)
        except BaseException as error:
            self._error = error
            self._ready_event.set()
        finally:
            self._ready_event.set()


# ------------------------------------------------------------------
# Module-level convenience functions
# ------------------------------------------------------------------


@contextmanager
def monitor_gpu(
    device_index: int = 0, interval_s: float = 0.05
) -> Generator[GpuMonitor, None, None]:
    """Context manager that yields a running GpuMonitor.

    Example::

        with monitor_gpu(device_index=0) as mon:
            run_workload()
        print(mon.summary())
    """
    mon = GpuMonitor(device_index=device_index, interval_s=interval_s)
    with mon:
        yield mon


def smi_replay_enabled() -> bool:
    """Whether the benchmark requested an isolated SMI replay window."""
    return os.environ.get("AITER_SMI_MONITOR", "0") == "1"


def emit_smi_result(result: dict) -> None:
    """Write one structured result to the combo JSONL sink or stdout."""
    line = SMI_RESULT_PREFIX + json.dumps(result, sort_keys=True)
    output_path = os.environ.get("AITER_SMI_OUTPUT_PATH")
    if output_path:
        with open(output_path, "a", encoding="utf-8") as output:
            output.write(line + "\n")
    else:
        print(line, flush=True)


def replay_with_smi(
    fn,
    *,
    label: str,
    synchronize,
    estimated_us: float | None = None,
) -> dict | None:
    """Repeat one already-prepared benchmark case under the GPU monitor.

    Input creation, compilation, correctness and the latency measurement happen
    before this function is called.  Batching launches between synchronizations
    keeps short kernels busy while still checking the wall-clock deadline often
    enough for slow kernels.
    """
    if not smi_replay_enabled():
        return None

    device = int(os.environ.get("AITER_SMI_DEVICE", "0"))
    interval_s = float(os.environ.get("AITER_SMI_INTERVAL", "0.05"))
    duration_s = float(os.environ.get("AITER_SMI_DURATION", "1.0"))
    if interval_s <= 0 or duration_s <= 0:
        raise ValueError("AITER_SMI_INTERVAL and AITER_SMI_DURATION must be positive")

    # Aim for roughly one synchronization per monitor tick.  The cap prevents a
    # near-zero/invalid latency estimate from enqueueing an unbounded amount of
    # work, while a slow case is synchronized after every launch.
    if estimated_us is not None and estimated_us > 0:
        batch_iters = max(1, min(1024, int(interval_s * 1e6 / estimated_us)))
    else:
        batch_iters = 1

    synchronize()
    launches = 0
    start = time.perf_counter()
    with monitor_gpu(device_index=device, interval_s=interval_s) as monitor:
        while launches == 0 or time.perf_counter() - start < duration_s:
            for _ in range(batch_iters):
                fn()
            launches += batch_iters
            synchronize()
    elapsed_s = time.perf_counter() - start

    result = {
        "label": label,
        "device": device,
        "interval_s": interval_s,
        "duration_s": elapsed_s,
        "launches": launches,
        "samples": len(monitor.samples),
        "metrics": monitor.summary(),
    }
    expected_samples = max(1, int(duration_s / interval_s))
    result["sample_status"] = (
        "ok"
        if len(monitor.samples) >= max(2, expected_samples // 2)
        else "insufficient"
    )
    # A shared JSONL sink survives fd silencing and child processes. Standalone
    # UT runs without a sink still get a machine-readable stdout record.
    emit_smi_result(result)
    return result
