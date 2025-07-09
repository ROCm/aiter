# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
import functools
import os
import re
import subprocess

from cpp_extension import executable_path


@functools.lru_cache(maxsize=1)
def get_gfx():
    gfx = os.getenv("GPU_ARCHS", "native")
    if gfx == "native":
        try:
            rocminfo = executable_path("rocminfo")
            result = subprocess.run(
                [rocminfo], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )
            output = result.stdout
            for line in output.split("\n"):
                if "gfx" in line.lower():
                    return line.split(":")[-1].strip()
        except Exception as e:
            raise RuntimeError(f"Get GPU arch from rcominfo failed {str(e)}")
    elif ";" in gfx:
        gfx = gfx.split(";")[-1]
    return gfx


@functools.lru_cache(maxsize=1)
def get_cu_num():
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
        raise RuntimeError(f"Get GPU Compute Unit from rcominfo failed {str(e)}")
    assert len(set(gpu_compute_units)) == 1
    return gpu_compute_units[0]


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
