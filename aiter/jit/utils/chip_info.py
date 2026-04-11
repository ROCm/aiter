# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
import functools
import os
import re
import subprocess
import sys

from cpp_extension import executable_path
from torch_guard import torch_compile_guard

IS_WINDOWS = sys.platform == "win32"
# On Windows, ROCm ships `hipinfo.exe` instead of `rocminfo`. The output
# format differs, so we keep two separate parsers below.
_GPU_INFO_TOOL = "hipinfo" if IS_WINDOWS else "rocminfo"

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
    18: "gfx1250",
}


def _run_gpu_info_tool() -> str:
    """Run the platform GPU info tool (rocminfo / hipinfo) and return stdout."""
    tool = executable_path(_GPU_INFO_TOOL)
    result = subprocess.run(
        [tool],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=True,
    )
    return result.stdout


@functools.lru_cache(maxsize=1)
def _detect_native() -> list[str]:
    try:
        output = _run_gpu_info_tool()
        for line in output.splitlines():
            # rocminfo:  "Name: gfx942"
            # hipinfo:   "gcnArchName:                  gfx1100"
            match = re.search(r"\b(gfx\w+)\b", line, re.IGNORECASE)
            if match:
                return [match.group(1).lower()]
    except Exception as e:
        raise RuntimeError(f"Get GPU arch from {_GPU_INFO_TOOL} failed: {e}") from e
    raise RuntimeError(f"No gfx arch found in {_GPU_INFO_TOOL} output.")


@torch_compile_guard()
def get_gfx_custom_op() -> int:
    return get_gfx_custom_op_core()


@functools.lru_cache(maxsize=10)
def get_gfx_custom_op_core() -> int:
    gfx = os.getenv("GPU_ARCHS", "native")
    gfx_mapping = {v: k for k, v in GFX_MAP.items()}
    # gfx = os.getenv("GPU_ARCHS", "native")
    if gfx == "native":
        try:
            output = _run_gpu_info_tool()
            for line in output.split("\n"):
                match = re.search(r"\b(gfx\w+)\b", line, re.IGNORECASE)
                if match:
                    gfx_arch = match.group(1).lower()
                    try:
                        return gfx_mapping[gfx_arch]
                    except KeyError:
                        raise KeyError(
                            f"Unknown GPU architecture: {gfx_arch}. "
                            f"Supported architectures: {list(gfx_mapping.keys())}"
                        )

        except Exception as e:
            raise RuntimeError(f"Get GPU arch from {_GPU_INFO_TOOL} failed {str(e)}")
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


def _parse_cu_num_rocminfo(output: str) -> list[int]:
    devices = re.split(r"Agent\s*\d+", output)
    gpu_compute_units: list[int] = []
    for device in devices:
        for line in device.split("\n"):
            if "Device Type" in line and line.find("GPU") != -1:
                match = re.search(r"Compute Unit\s*:\s*(\d+)", device)
                if match:
                    gpu_compute_units.append(int(match.group(1)))
                break
    return gpu_compute_units


def _parse_cu_num_hipinfo(output: str) -> list[int]:
    # hipinfo prints one block per device with a line like:
    #   "multiProcessorCount:           80"
    # On AMD GPUs this is the compute-unit count.
    return [
        int(m.group(1)) for m in re.finditer(r"multiProcessorCount\s*:\s*(\d+)", output)
    ]


@torch_compile_guard()
def get_cu_num_custom_op() -> int:
    cu_num = int(os.getenv("CU_NUM", 0))
    if cu_num == 0:
        try:
            output = _run_gpu_info_tool()
            if IS_WINDOWS:
                gpu_compute_units = _parse_cu_num_hipinfo(output)
            else:
                gpu_compute_units = _parse_cu_num_rocminfo(output)
        except Exception as e:
            raise RuntimeError(
                f"Get GPU Compute Unit from {_GPU_INFO_TOOL} failed {str(e)}"
            )
        if not gpu_compute_units:
            raise RuntimeError(f"No GPU Compute Unit found in {_GPU_INFO_TOOL} output.")
        assert len(set(gpu_compute_units)) == 1
        cu_num = gpu_compute_units[0]
    return cu_num


@functools.lru_cache(maxsize=1)
def get_cu_num():
    cu_num = get_cu_num_custom_op()
    return cu_num


def _get_pci_chip_id(device_id=0):
    import ctypes

    # On Linux ROCm ships `libamdhip64.so`; on Windows ROCm 7 ships
    # `amdhip64_7.dll`.
    if IS_WINDOWS:
        candidates = ("amdhip64_7.dll", "amdhip64.dll")
    else:
        candidates = ("libamdhip64.so",)
    libhip = None
    last_err: Exception | None = None
    for name in candidates:
        try:
            libhip = ctypes.CDLL(name)
            break
        except OSError as e:
            last_err = e
    if libhip is None:
        raise RuntimeError(
            f"Could not load AMD HIP runtime ({', '.join(candidates)}): {last_err}"
        )
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
        chip_id = _get_pci_chip_id()
        if chip_id in MI308_CHIP_IDS:
            return "MI308"
        return "MI300"
    elif gfx == "gfx950":
        return "MI350"
    else:
        raise RuntimeError("Unsupported gfx")
