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
    """hwmon power1_input for a HIP device ordinal in this process.

    ``AITER_POWER_PATH`` replaces ``{device}`` with that ordinal. A path
    without ``{device}`` is one file and is only safe for a single tuner
    process.
    """
    override = os.environ.get("AITER_POWER_PATH", "").strip()
    if override:
        if "{device}" in override:
            return override.replace("{device}", str(int(device_index)))
        return override
    device_dir = _device_sysfs_dir(device_index)
    if device_dir is None:
        return None
    hits = glob.glob(f"{device_dir}/hwmon/hwmon*/power1_input")
    return hits[0] if hits else None


def _device_sysfs_dir(device_index: int) -> str | None:
    """PCI sysfs directory that has the hwmon node for a HIP device ordinal."""
    bdf = _hip_pci_bus_id(device_index)
    if not bdf:
        return None
    bdf = bdf.lower()
    direct = f"/sys/bus/pci/devices/{bdf}"
    if glob.glob(f"{direct}/hwmon/hwmon*/power1_input"):
        return direct
    for dev in glob.glob("/sys/class/drm/card*/device"):
        try:
            real = os.path.realpath(dev)
        except OSError:
            continue
        if os.path.basename(real).lower() != bdf:
            continue
        if glob.glob(dev + "/hwmon/hwmon*/power1_input"):
            return dev
    return None


def _read_text(path: str) -> str | None:
    try:
        with open(path, encoding="ascii") as handle:
            return handle.read()
    except (OSError, UnicodeError):
        return None


def _device_conditions(device_index: int) -> dict[str, str]:
    """Power-management state a power tune ran under, as strings for a CSV row.

    Reads only: the DPM performance level, the package power cap, and the top
    of the overdrive SCLK range. A perf-determinism clock clamp is not always
    visible in sysfs, so ``perf_level`` is the record of that mode.
    """
    from aiter.utility.tune_result import UNKNOWN_CONDITION

    conditions = {
        "perf_level": UNKNOWN_CONDITION,
        "power_cap_w": UNKNOWN_CONDITION,
        "sclk_max_mhz": UNKNOWN_CONDITION,
    }
    device_dir = _device_sysfs_dir(device_index)
    if device_dir is None:
        return conditions
    level = _read_text(f"{device_dir}/power_dpm_force_performance_level")
    if level and level.strip():
        conditions["perf_level"] = level.strip()
    caps = glob.glob(f"{device_dir}/hwmon/hwmon*/power1_cap")
    cap = _read_text(caps[0]) if caps else None
    try:
        conditions["power_cap_w"] = str(round(int(cap.strip()) / 1e6))
    except (AttributeError, ValueError):
        pass
    od = _read_text(f"{device_dir}/pp_od_clk_voltage")
    if od:
        in_sclk = False
        top = None
        for line in od.splitlines():
            stripped = line.strip()
            if stripped.endswith(":"):
                in_sclk = stripped.upper().startswith("OD_SCLK")
                continue
            if in_sclk and ":" in stripped:
                value = stripped.split(":", 1)[1].strip().lower()
                if value.endswith("mhz"):
                    top = value[: -len("mhz")].strip()
        if top:
            conditions["sclk_max_mhz"] = top
    return conditions


def tune_conditions(device_indices) -> dict[str, str]:
    """Conditions shared by every tuning device, or ``mixed`` where they differ."""
    from aiter.utility.tune_result import MIXED_CONDITION, UNKNOWN_CONDITION

    per_device = [_device_conditions(int(index)) for index in device_indices]
    if not per_device:
        return {
            "perf_level": UNKNOWN_CONDITION,
            "power_cap_w": UNKNOWN_CONDITION,
            "sclk_max_mhz": UNKNOWN_CONDITION,
        }
    merged = {}
    for name in per_device[0]:
        values = {device[name] for device in per_device}
        merged[name] = values.pop() if len(values) == 1 else MIXED_CONDITION
    return merged


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


def shared_power_override_rejected() -> str | None:
    """Why a multi-process tune must not use this AITER_POWER_PATH, or None."""
    override = os.environ.get("AITER_POWER_PATH", "").strip()
    if override and "{device}" not in override:
        return (
            "AITER_POWER_PATH points at one hwmon file. With more than one "
            "tuner process that file would be read for every GPU. Put "
            "'{device}' in the path for the HIP ordinal, or run with --mp 1."
        )
    return None


def _perftest_option_names() -> frozenset:
    import inspect

    from aiter.test_common import run_perftest

    params = inspect.signature(run_perftest).parameters
    return frozenset(
        name
        for name, param in params.items()
        if param.kind is inspect.Parameter.KEYWORD_ONLY
    )


def kernel_call(func, args, kwargs):
    """The kernel launch the timer measured, without the ``run_perftest`` options.

    ``mp_tuner`` tasks carry ``run_perftest`` options (``num_warmup``,
    ``num_iters``, and the rest) in the same kwargs as the kernel's own. The
    kernel must not receive them, and replaying ``run_perftest`` itself would
    spend most of the power window in profiler setup with the GPU idle.
    """
    options = _perftest_option_names()
    kernel_kwargs = {k: v for k, v in (kwargs or {}).items() if k not in options}
    args = tuple(args or ())
    return lambda: func(*args, **kernel_kwargs)


def measure_kernel(call, device_index: int | None = None) -> float:
    """Mean package watts while ``call`` launches back to back.

    This is the one place a tune reads package power. ``call`` launches the
    kernel the timer measured and nothing else; NaN means the sensor could not
    be read or the call raised.
    """
    if device_index is None:
        try:
            import torch

            device_index = int(torch.cuda.current_device())
        except Exception:  # noqa: BLE001
            device_index = 0
    try:
        return float(measure_package_watts(call, device_index=device_index))
    except Exception as exc:  # noqa: BLE001
        logger.warning("package power measurement failed: %s", exc)
        return float("nan")
