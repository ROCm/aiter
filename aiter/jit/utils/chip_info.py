# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
import functools
import os
import re
import shutil
import subprocess
import sys

from typing import Optional
from cpp_extension import executable_path
from torch_guard import torch_compile_guard

GFX_MAP = {
    0: "native",
    1: "gfx90a",
    2: "gfx908",
    3: "gfx940",
    4: "gfx941",
    5: "gfx942",
    6: "gfx945",
    7: "gfx1100",
    8: "gfx950",
    9: "gfx1101",
    10: "gfx1102",
    11: "gfx1103",
    12: "gfx1150",
    13: "gfx1151",
    14: "gfx1152",
    15: "gfx1153",
    16: "gfx1200",
    17: "gfx1201",
}


def _run_gpu_info_tool() -> str:
    """
    Run rocminfo (Linux) or hipinfo (Windows) and return stdout.
    rocminfo is not shipped with the Windows ROCm SDK; hipinfo is the
    equivalent tool available there.
    """
    # Try rocminfo first (Linux / full ROCm installs)
    rocminfo_path = shutil.which("rocminfo")
    if rocminfo_path is None and sys.platform != "win32":
        # On Linux also try via executable_path which searches ROCM_HOME/bin
        try:
            rocminfo_path = executable_path("rocminfo")
        except Exception:
            pass

    if rocminfo_path is not None:
        result = subprocess.run(
            [rocminfo_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout

    # Fall back to hipinfo (Windows ROCm SDK)
    hipinfo_path = shutil.which("hipinfo")
    if hipinfo_path is not None:
        result = subprocess.run(
            [hipinfo_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout

    raise RuntimeError(
        "Could not find rocminfo or hipinfo in PATH. "
        "Please ensure the ROCm/HIP SDK bin directory is on your PATH "
        "(e.g. C:\\Program Files\\AMD\\ROCm\\<ver>\\bin on Windows)."
    )


def _extract_gfx_from_output(output: str) -> Optional[str]:
    """
    Parse a gfx architecture string out of rocminfo or hipinfo output.
    """
    for line in output.splitlines():
        match = re.search(r"\b(gfx\w+)\b", line, re.IGNORECASE)
        if match:
            return match.group(1).lower()
    return None


def _extract_cu_from_output(output: str) -> Optional[int]:
    """
    Parse the Compute Unit count out of rocminfo or hipinfo output.
    """
    for line in output.splitlines():
        match = re.search(r"Compute Units?\s*:+\s*(\d+)", line, re.IGNORECASE)
        if match:
            return int(match.group(1))
    return None


@functools.lru_cache(maxsize=1)
def _detect_native() -> list[str]:
    try:
        output = _run_gpu_info_tool()
        gfx = _extract_gfx_from_output(output)
        if gfx is not None:
            return [gfx]
    except Exception as e:
        raise RuntimeError(f"Get GPU arch from rocminfo/hipinfo failed: {e}") from e
    raise RuntimeError("No gfx arch found in rocminfo/hipinfo output.")


@torch_compile_guard()
def get_gfx_custom_op() -> int:
    return get_gfx_custom_op_core()


@functools.lru_cache(maxsize=10)
def get_gfx_custom_op_core() -> int:
    gfx = os.getenv("GPU_ARCHS", "native")
    gfx_mapping = {v: k for k, v in GFX_MAP.items()}

    if gfx == "native":
        try:
            output = _run_gpu_info_tool()
            gfx_arch = _extract_gfx_from_output(output)
            if gfx_arch is None:
                raise RuntimeError("No gfx arch found in rocminfo/hipinfo output.")
            try:
                return gfx_mapping[gfx_arch]
            except KeyError:
                raise KeyError(
                    f"Unknown GPU architecture: {gfx_arch}. "
                    f"Supported architectures: {list(gfx_mapping.keys())}"
                )
        except Exception as e:
            raise RuntimeError(f"Get GPU arch from rocminfo/hipinfo failed: {e}")
    elif ";" in gfx:
        gfx = gfx.split(";")[-1]

    try:
        return gfx_mapping[gfx]
    except KeyError:
        raise KeyError(
            f"Unknown GPU architecture: {gfx}. "
            f"Supported architectures: {list(gfx_mapping.keys())}"
        )


@functools.lru_cache(maxsize=1)
def get_gfx():
    gfx_num = get_gfx_custom_op()
    return GFX_MAP.get(gfx_num, "unknown")


@functools.lru_cache(maxsize=1)
def get_gfx_list() -> list[str]:
    gfx_env = os.getenv("GPU_ARCHS", "native")
    if gfx_env == "native":
        try:
            gfxs = _detect_native()
        except RuntimeError:
            gfxs = ["cpu"]
    else:
        gfxs = [g.strip() for g in gfx_env.split(";") if g.strip()]
    os.environ["AITER_GPU_ARCHS"] = ";".join(gfxs)
    return gfxs


@torch_compile_guard()
def get_cu_num_custom_op() -> int:
    cu_num = int(os.getenv("CU_NUM", 0))
    if cu_num == 0:
        try:
            output = _run_gpu_info_tool()
            # rocminfo groups by Agent; collect CU counts for GPU agents only
            devices = re.split(r"Agent\s*\d+", output)
            gpu_compute_units = []
            for device in devices:
                if "Device Type" in device and "GPU" in device:
                    cu = _extract_cu_from_output(device)
                    if cu is not None:
                        gpu_compute_units.append(cu)
            # hipinfo doesn't have the Agent split — fall back to global scan
            if not gpu_compute_units:
                cu = _extract_cu_from_output(output)
                if cu is not None:
                    gpu_compute_units.append(cu)
        except Exception as e:
            raise RuntimeError(f"Get GPU Compute Unit from rocminfo/hipinfo failed: {e}")

        if not gpu_compute_units:
            raise RuntimeError("Could not determine Compute Unit count from GPU info output.")
        assert len(set(gpu_compute_units)) == 1, (
            f"Multiple different CU counts found: {gpu_compute_units}"
        )
        cu_num = gpu_compute_units[0]
    return cu_num


@functools.lru_cache(maxsize=1)
def get_cu_num():
    cu_num = get_cu_num_custom_op()
    return cu_num


def _get_pci_chip_id(device_id=0):
    import ctypes

    libhip = ctypes.CDLL("libamdhip64.so")
    chip_id = ctypes.c_int(0)
    hipDeviceAttributePciChipId = 10019
    err = libhip.hipDeviceGetAttribute(
        ctypes.byref(chip_id),
        hipDeviceAttributePciChipId,
        device_id,
    )
    if err != 0:
        raise RuntimeError(f"hipDeviceGetAttribute(PciChipId) failed with error {err}")
    return chip_id.value


MI308_CHIP_IDS = {0x74A2, 0x74A8, 0x74B6, 0x74BC}


def get_device_name():
    gfx = get_gfx()

    if gfx == "gfx942":
        if sys.platform == "win32":
            # libamdhip64.so ctypes probe is Linux-only; identify the SKU by
            # Compute Unit count instead.  Known gfx942 SKU map:
            #   304 CU -> MI300X
            #   228 CU -> MI300A
            #    80 CU -> MI308X
            #    64 CU -> MI308  (smaller variant)
            cu = get_cu_num()
            if cu == 304:
                return "MI300X"
            elif cu == 228:
                return "MI300A"
            elif cu in (80, 64):
                return "MI308"
            else:
                return f"MI300-family (gfx942, {cu} CUs)"
        else:
            chip_id = _get_pci_chip_id()
            if chip_id in MI308_CHIP_IDS:
                return "MI308"
            return "MI300"
    elif gfx == "gfx950":
        return "MI350"
    else:
        raise RuntimeError("Unsupported gfx")
