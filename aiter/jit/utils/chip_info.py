# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
import functools
import os
import re
import subprocess

from cpp_extension import executable_path
from torch_guard import torch_compile_guard

from aiter.jit.utils.build_targets import (  # noqa: F401 — re-exported for callers
    GFX_MAP,
    GFX_CU_NUM_MAP,
    filter_tune_df,
    get_build_targets_env,
)


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


def get_build_targets() -> list[tuple[str, int]]:
    """Return (gfx, cu_num) pairs to compile kernels for.

    Used by gen_instances.py in all CK GEMM modules to filter the tuning CSV
    to exactly the right set of kernels for the target GPU(s).

    Priority:
      1. GPU_ARCHS env var set → delegates to get_build_targets_env() (no GPU needed).
      2. Live GPU detected → use actual (gfx, cu_num) from rocminfo, which
         correctly reflects partition mode and binned variants.
      3. Neither → raise RuntimeError with a clear message.
    """
    if os.getenv("GPU_ARCHS"):
        return get_build_targets_env()

    try:
        return [(get_gfx(), get_cu_num())]
    except Exception as e:
        raise RuntimeError(
            "No GPU detected and GPU_ARCHS is not set. "
            "Set GPU_ARCHS=gfx942 (or similar) to build without a GPU."
        ) from e


def build_tune_dict(tune_df, default_dict, kernels_list, libtype=None, kernels_by_name=None):
    """Filter tune_df to rows matching the current build targets and return a
    (cu_num, M, N, K)-keyed dispatch dict, starting from a copy of default_dict.

    Replaces the duplicated get_tune_dict filtering loop in each gen_instances.py.
    Modules keep their own default_dict and kernels_list; only the CSV filtering
    and key construction are shared here.

    Args:
        tune_df:          pandas DataFrame already loaded from the tuning CSV.
        default_dict:     module-level fallback dict (negative-int keys) to start from.
        kernels_list:     module-level dict mapping kernelId → kernelInstance.
        libtype:          Optional string to filter the "libtype" column (e.g. "ck").
                          Required for CSVs that mix multiple library types (e.g.
                          a8w8_bpreshuffle_tuned_gemm.csv mixes "ck" and "cktile").
                          If None, no libtype filtering is applied.
        kernels_by_name:  Optional dict mapping kernelName string → kernelInstance.
                          When provided and the CSV has a "kernelName" column, kernel
                          lookup uses the name instead of kernelId. Falls back to
                          kernelId if the name is not found or the column is absent.

    Returns:
        dict with mixed keys: negative ints (from default_dict) and
        (cu_num, M, N, K) 4-tuples (from the filtered CSV rows).
    """
    import pandas as pd

    tune_dict = dict(default_dict)
    targets = get_build_targets()
    filtered = filter_tune_df(tune_df, targets)
    if libtype is not None and "libtype" in tune_df.columns:
        filtered = filtered[filtered["libtype"] == libtype]
    use_name = kernels_by_name is not None and "kernelName" in tune_df.columns
    if kernels_by_name is not None and not use_name:
        print("[Warning]: kernels_by_name provided but CSV has no kernelName column, falling back to kernelId.")
    for _, row in filtered.iterrows():
        key = (int(row["cu_num"]), int(row["M"]), int(row["N"]), int(row["K"]))
        if use_name:
            kname = str(row["kernelName"])
            if kname in kernels_by_name:
                tune_dict[key] = kernels_by_name[kname]
            else:
                print(f"[Warning]: kernelName '{kname}' not found, skip it")
        else:
            tune_dict[key] = kernels_list[int(row["kernelId"])]
    return tune_dict


def build_tune_dict_batched(tune_df, default_dict, kernels_list, libtype=None):
    """Like build_tune_dict, but for batched GEMM modules whose dispatch key
    includes the batch dimension B.

    Builds a (cu_num, B, M, N, K) 5-tuple keyed dict suitable for use with
    BatchedGemmDispatchMap in the C++ dispatch layer.

    Args:
        tune_df:      pandas DataFrame loaded from the batched tuning CSV.
        default_dict: module-level fallback dict (negative-int keys) to start from.
        kernels_list: module-level dict mapping kernelId → kernelInstance.
        libtype:      Optional string to filter the "libtype" column (same semantics
                      as build_tune_dict).

    Returns:
        dict with mixed keys: negative ints (from default_dict) and
        (cu_num, B, M, N, K) 5-tuples (from the filtered CSV rows).
    """
    tune_dict = dict(default_dict)
    targets = get_build_targets()
    filtered = filter_tune_df(tune_df, targets)
    if libtype is not None and "libtype" in tune_df.columns:
        filtered = filtered[filtered["libtype"] == libtype]
    for _, row in filtered.iterrows():
        key = (int(row["cu_num"]), int(row["B"]), int(row["M"]), int(row["N"]), int(row["K"]))
        tune_dict[key] = kernels_list[int(row["kernelId"])]
    return tune_dict


def write_lookup_header(
    output_path, kernels_dict, lookup_head, lookup_template, lookup_end, istune=False
):
    """Write a C++ GEMM dispatch lookup header from a kernels_dict.

    Replaces the duplicated gen_lookup_dict loop in each gen_instances.py codegen
    class.  Each module still defines its own lookup_head / lookup_template /
    lookup_end strings (they embed the module-specific GENERATE_LOOKUP_TABLE macro
    type parameters), but the iteration and key-formatting logic is shared here.

    Key layout in kernels_dict:
      - Negative ints   (default_dict entries)  → skipped in non-tune mode.
      - (cu_num,M,N,K) 4-tuples (tuned entries) → written as {cu_num,M,N,K} C++ key.
      - (cu_num,B,M,N,K) 5-tuples (batched)     → written as {cu_num,B,M,N,K} C++ key.
      - Non-negative ints (tune mode only)       → written as plain integer kernel ID.

    Args:
        output_path:     Full path of the .h file to write.
        kernels_dict:    Dict returned by build_tune_dict (or get_tune_dict).
        lookup_head:     String written before the loop (defines the macro header).
        lookup_template: String with {MNK} and {kernel_name} placeholders.
        lookup_end:      String written after the loop (closes the macro / #endif).
        istune:          True when generating the tune-mode lookup (int kernelId keys).
    """
    with open(output_path, "w") as f:
        f.write(lookup_head)
        for key, k in kernels_dict.items():
            if not istune and (isinstance(key, tuple) and key[1] > 0):
                # 4-tuple key: (cu_num, M, N, K) — key[1] = M > 0 for real shapes
                # 5-tuple key: (cu_num, B, M, N, K) — key[1] = B >= 1 for batched shapes
                f.write(lookup_template.format(
                    MNK="{" + ", ".join(str(x) for x in key) + "}",
                    kernel_name=k.name,
                ))
            elif istune and isinstance(key, int) and key >= 0:
                f.write(lookup_template.format(MNK=key, kernel_name=k.name))
        f.write(lookup_end)


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
        chip_id = _get_pci_chip_id()
        if chip_id in MI308_CHIP_IDS:
            return "MI308"
        return "MI300"
    elif gfx == "gfx950":
        return "MI350"
    else:
        raise RuntimeError("Unsupported gfx")
