# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Environment metadata recorded with every result.

Read-only: nothing here changes GPU settings. Clocks, performance level and
activity come from the amdsmi Python binding when it imports, otherwise from
the ``amd-smi`` CLI; when neither works the snapshot carries the reason.
"""

import functools
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
from dataclasses import asdict, dataclass

import torch

_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
_SMI_TIMEOUT_S = 5


@dataclass(frozen=True)
class SystemInfo:
    gpu_name: str
    gfx: str
    cu_count: int
    device_index: int
    pci_bdf: str
    hip_version: str | None
    torch_version: str
    triton_version: str | None
    flydsl_version: str | None
    python_version: str
    aiter_git_sha: str | None
    hostname: str

    def to_dict(self):
        return asdict(self)


def _package_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _git_sha():
    try:
        out = subprocess.run(
            ["git", "-C", _REPO_ROOT, "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=_SMI_TIMEOUT_S,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    sha = out.stdout.strip()
    dirty = subprocess.run(
        ["git", "-C", _REPO_ROOT, "status", "--porcelain", "--untracked-files=no"],
        capture_output=True,
        text=True,
        timeout=_SMI_TIMEOUT_S,
        check=False,
    ).stdout.strip()
    return f"{sha}-dirty" if dirty else sha


def pci_bdf(device_index) -> str:
    props = torch.cuda.get_device_properties(device_index)
    return (
        f"{props.pci_domain_id:04x}:{props.pci_bus_id:02x}:{props.pci_device_id:02x}.0"
    )


@functools.cache
def system_info(device_index) -> SystemInfo:
    props = torch.cuda.get_device_properties(device_index)
    return SystemInfo(
        gpu_name=props.name,
        gfx=getattr(props, "gcnArchName", "").split(":", 1)[0],
        cu_count=props.multi_processor_count,
        device_index=device_index,
        pci_bdf=pci_bdf(device_index),
        hip_version=getattr(torch.version, "hip", None),
        torch_version=torch.__version__,
        triton_version=_package_version("triton"),
        flydsl_version=_package_version("flydsl"),
        python_version=platform.python_version(),
        aiter_git_sha=_git_sha(),
        hostname=platform.node(),
    )


# --------------------------------------------------------------------------- #
# GPU state snapshot: clocks, performance level, activity, power, temperature.
# --------------------------------------------------------------------------- #


def _value(entry):
    return entry.get("value") if isinstance(entry, dict) else None


def _run_smi(*args):
    out = subprocess.run(
        ["amd-smi", *args, "--json"],
        capture_output=True,
        text=True,
        timeout=_SMI_TIMEOUT_S,
        check=True,
    )
    return json.loads(out.stdout)


@functools.cache
def _smi_index(bdf):
    for gpu in _run_smi("list"):
        if gpu.get("bdf", "").lower() == bdf.lower():
            return gpu["gpu"]
    raise LookupError(f"amd-smi lists no GPU with BDF {bdf}")


def _snapshot_cli(device_index):
    idx = _smi_index(pci_bdf(device_index))
    data = _run_smi(
        "metric",
        "-g",
        str(idx),
        "--clock",
        "--usage",
        "--perf-level",
        "--power",
        "--temperature",
    )["gpu_data"][0]
    clocks = data.get("clock", {})
    gfx = [
        _value(v.get("clk"))
        for k, v in clocks.items()
        if k.startswith("gfx_") and isinstance(v, dict)
    ]
    gfx = [c for c in gfx if c is not None]
    locked = {
        v.get("clk_locked")
        for k, v in clocks.items()
        if k.startswith("gfx_") and isinstance(v, dict)
    }
    usage = data.get("usage", {})
    return {
        "source": "amd-smi",
        "gfx_clk_mhz": max(gfx) if gfx else None,
        "mem_clk_mhz": _value(clocks.get("mem_0", {}).get("clk")),
        "gfx_clk_locked": sorted(x for x in locked if x is not None) or None,
        "perf_level": data.get("perf_level"),
        "gfx_activity_pct": _value(usage.get("gfx_activity")),
        "umc_activity_pct": _value(usage.get("umc_activity")),
        "power_w": _value(data.get("power", {}).get("socket_power")),
        "temp_hotspot_c": _value(data.get("temperature", {}).get("hotspot")),
    }


def _snapshot_binding(device_index):
    from aiter import smi_monitor

    handle = smi_monitor.hip_device_to_amdsmi_handle(device_index)
    sample = smi_monitor._collect_sample(handle)
    sample.pop("timestamp_s", None)
    try:
        sample["perf_level"] = str(smi_monitor.amdsmi.amdsmi_get_gpu_perf_level(handle))
    except Exception:  # noqa: BLE001 - optional field
        sample["perf_level"] = None
    return {"source": "amdsmi", **sample}


def gpu_state(device_index) -> dict:
    """One read-only snapshot of the GPU's clocks and activity."""
    errors = []
    try:
        return _snapshot_binding(device_index)
    except Exception as error:  # noqa: BLE001 - fall back to the CLI
        errors.append(f"amdsmi: {error}")
    if shutil.which("amd-smi"):
        try:
            return _snapshot_cli(device_index)
        except Exception as error:  # noqa: BLE001 - reported in the result
            errors.append(f"amd-smi: {error}")
    else:
        errors.append("amd-smi: not on PATH")
    return {"source": None, "error": "; ".join(errors)}
