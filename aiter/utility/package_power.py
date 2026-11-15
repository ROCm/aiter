# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Package power from the same hwmon file rocm-smi -P reads (power1_input, label PPT)."""

import collections
import ctypes
import glob
import logging
import math
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
    fn,
    seconds: float | None = None,
    device_index: int = 0,
    kernel_us: float | None = None,
) -> float:
    """Run fn until `seconds` elapse and return package power in watts.

    The timed GEMM loop in the tuner is often shorter than the PPT update, so callers
    that rank by power should pass a window of a few hundred milliseconds.

    Without ``kernel_us`` this is the mean reading over the window. With it,
    the reading is the power while the kernel runs: launching ``fn`` from
    Python can leave the GPU idle between calls, and a kernel whose launch
    costs more host time than its GPU time would otherwise read close to idle
    and look efficient. See ``active_watts``.
    """
    if seconds is None:
        seconds = float(os.environ.get("AITER_POWER_WINDOW_S", "0.4"))
    path = power_input_path(device_index)
    if path is None or seconds <= 0:
        return float("nan")
    idle_w = idle_watts(path) if kernel_us is not None else None

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
    started = time.perf_counter()
    fn()
    _sync_cuda()
    in_flight = _InFlight(time.perf_counter() - started)
    calls = 0
    thread = threading.Thread(target=loop, daemon=True)
    thread.start()
    window_start = time.perf_counter()
    deadline = window_start + seconds
    try:
        while time.perf_counter() < deadline:
            fn()
            calls += 1
            in_flight.mark()
    finally:
        _sync_cuda()
        window_s = time.perf_counter() - window_start
        stop.set()
        thread.join()
    if not samples:
        return float("nan")
    mean_w = sum(samples) / len(samples)
    if kernel_us is None:
        return mean_w
    if in_flight.wait_s > SATURATED_WAIT * window_s:
        # The loop had to wait for the GPU, so the queue never ran dry. The
        # timed us can undercount a sustained window (clocks settle lower),
        # which would otherwise read as idle time.
        return mean_w
    return active_watts(mean_w, idle_w, calls * kernel_us * 1e-6 / window_s)


# Below this GPU-busy fraction the window is mostly idle and scaling the
# reading up would mostly amplify sensor noise.
MIN_BUSY = 0.2
# At or above this the window kept the GPU busy and the reading stands.
FULL_BUSY = 0.95
# Waiting on the GPU for this share of the window means launches outran it.
SATURATED_WAIT = 0.02
_idle_watts: dict[str, float] = {}


def active_watts(mean_w: float, idle_w: float | None, busy: float) -> float:
    """Package power while the kernel runs, from a window that was ``busy`` busy.

    The window's energy is idle power for the whole window plus the kernel's
    extra power while it ran, so ``mean = idle + busy * (active - idle)``.
    NaN when the window was too idle to tell, which ranks the kernel as
    unmeasured.
    """
    if not math.isfinite(busy) or busy <= 0:
        return float("nan")
    if busy >= FULL_BUSY or idle_w is None:
        return mean_w
    if busy < MIN_BUSY:
        logger.warning(
            "package power: the GPU was busy %.0f%% of the window "
            "(launch-bound), so this kernel's power is not recorded",
            busy * 100,
        )
        return float("nan")
    return idle_w + (mean_w - idle_w) / busy


def idle_watts(path: str) -> float | None:
    """Package power between launches, read once per process and sensor.

    Between back-to-back launches the GPU stays awake at high clocks, which
    on MI355X reads ~300 W against ~240 W fully idle. A loop of one-element
    kernels reproduces that state, so it is the baseline the launch gaps are
    charged at; a fully idle baseline would overstate short kernels.
    ``AITER_POWER_IDLE_W`` overrides the reading.
    """
    override = os.environ.get("AITER_POWER_IDLE_W", "").strip()
    if override:
        return float(override)
    if path not in _idle_watts:
        try:
            import torch

            device = torch.cuda.current_device()
            tick = torch.zeros(1, device=f"cuda:{device}")
        except Exception:  # noqa: BLE001
            return None
        samples = []
        end = time.perf_counter() + 0.3
        next_read = 0.0
        while time.perf_counter() < end:
            tick.add_(0)
            now = time.perf_counter()
            if now >= next_read:
                watts = _read_watts(path)
                if watts is not None:
                    samples.append(watts)
                next_read = now + 0.005
        _sync_cuda()
        # Drop the first third while the clocks ramp.
        samples = samples[len(samples) // 3 :]
        if not samples:
            return None
        _idle_watts[path] = sum(samples) / len(samples)
    return _idle_watts[path]


class _InFlight:
    """Keep a bounded amount of work queued on the GPU.

    An unbounded queue of large graph replays crashes some kernels (IQ2R MoE)
    inside the HIP runtime. Recording an event after every launch costs host
    time that starves short kernels, so one event covers about ``SPAN_S`` of
    work, sized from one synchronized call, and the loop waits on the event
    ``DEPTH`` back so the GPU never drains.
    """

    DEPTH = 4
    SPAN_S = 200e-6

    def __init__(self, call_s: float = 0.0):
        self._events = collections.deque()
        self._every = max(1, int(self.SPAN_S / call_s)) if call_s > 0 else 1
        self._calls = 0
        self.wait_s = 0.0
        try:
            import torch

            self._event = torch.cuda.Event if torch.cuda.is_available() else None
        except Exception:  # noqa: BLE001
            self._event = None

    def mark(self) -> None:
        if self._event is None:
            return
        self._calls += 1
        if self._calls % self._every:
            return
        done = self._event()
        done.record()
        self._events.append(done)
        if len(self._events) > self.DEPTH:
            started = time.perf_counter()
            self._events.popleft().synchronize()
            self.wait_s += time.perf_counter() - started


def _positive(value) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) and number > 0 else None


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


def measure_kernel(
    call, device_index: int | None = None, kernel_us: float | None = None
) -> float:
    """Package watts while ``call`` launches back to back.

    This is the one place a tune reads package power. ``call`` launches the
    kernel the timer measured and nothing else. ``kernel_us`` is that call's
    timed GPU time; with it the reading is corrected for host launch gaps
    (``active_watts``). NaN means the sensor could not be read, the call
    raised, or the window was too idle to tell.
    """
    if device_index is None:
        try:
            import torch

            device_index = int(torch.cuda.current_device())
        except Exception:  # noqa: BLE001
            device_index = 0
    try:
        return float(
            measure_package_watts(
                call, device_index=device_index, kernel_us=_positive(kernel_us)
            )
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("package power measurement failed: %s", exc)
        return float("nan")
