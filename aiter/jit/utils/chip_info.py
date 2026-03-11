# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
import functools
import os
import re
import subprocess

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


@functools.lru_cache(maxsize=1)
def _detect_native() -> list[str]:
    try:
        rocminfo = executable_path("rocminfo")
        result = subprocess.run(
            [rocminfo],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
        for line in result.stdout.splitlines():
            if "gfx" in line.lower():
                return [line.split(":", 1)[-1].strip()]
    except Exception as e:
        raise RuntimeError(f"Get GPU arch from rocminfo failed: {e}") from e
    raise RuntimeError("No gfx arch found in rocminfo output.")


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
            rocminfo = executable_path("rocminfo")
            result = subprocess.run(
                [rocminfo], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )
            output = result.stdout
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
            raise RuntimeError(f"Get GPU arch from rocminfo failed {str(e)}")
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
            rocminfo = executable_path("rocminfo")
            result = subprocess.run(
                [rocminfo], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )
            output = result.stdout
            devices = re.split(r"Agent\s*\d+", output)
            gpu_compute_units = []
            for device in devices:
                for line in device.split("\n"):
                    if "Device Type" in line and line.find("GPU") != -1:
                        match = re.search(r"Compute Unit\s*:\s*(\d+)", device)
                        if match:
                            gpu_compute_units.append(int(match.group(1)))
                        break
        except Exception as e:
            raise RuntimeError(f"Get GPU Compute Unit from rocminfo failed {str(e)}")
        assert len(set(gpu_compute_units)) == 1
        cu_num = gpu_compute_units[0]
    return cu_num


@functools.lru_cache(maxsize=1)
def get_cu_num():
    cu_num = get_cu_num_custom_op()
    return cu_num


@torch_compile_guard()
def get_lds_size_per_cu_custom_op() -> int:
    """Return the LDS (shared memory) size per CU in bytes.

    Parses the GROUP segment pool size from ``rocminfo`` output.
    The value corresponds to ``hipDeviceProp_t.sharedMemPerMultiprocessor``.
    """
    try:
        rocminfo = executable_path("rocminfo")
        result = subprocess.run(
            [rocminfo], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        output = result.stdout
        devices = re.split(r"Agent\s*\d+", output)
        lds_sizes = []
        for device in devices:
            is_gpu = False
            for line in device.split("\n"):
                if "Device Type" in line and line.find("GPU") != -1:
                    is_gpu = True
                    break
            if not is_gpu:
                continue
            # Find GROUP segment pool size within this GPU agent section.
            lines = device.split("\n")
            for i, line in enumerate(lines):
                if re.search(r"Segment\s*:\s*GROUP", line):
                    # The Size line immediately follows the Segment line.
                    if i + 1 < len(lines):
                        m = re.search(r"Size\s*:\s*(\d+)", lines[i + 1])
                        if m:
                            lds_sizes.append(int(m.group(1)) * 1024)  # KB to bytes
                    break
    except Exception as e:
        raise RuntimeError(f"Get LDS size per CU from rocminfo failed {str(e)}")
    if not lds_sizes:
        raise RuntimeError("No GPU GROUP segment found in rocminfo output")
    assert len(set(lds_sizes)) == 1, f"Inconsistent LDS sizes across GPUs: {lds_sizes}"
    return lds_sizes[0]


@functools.lru_cache(maxsize=1)
def get_lds_size_per_cu() -> int:
    return get_lds_size_per_cu_custom_op()


def get_device_name():
    gfx = get_gfx()

    if gfx == "gfx942":
        cu = get_cu_num()
        if cu == 304:
            return "MI300"
        elif cu == 80 or cu == 64:
            return "MI308"
    elif gfx == "gfx950":
        return "MI350"
    else:
        raise RuntimeError("Unsupported gfx")
