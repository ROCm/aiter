# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
import functools
import logging
import os
import re
import subprocess

from cpp_extension import executable_path
from torch_guard import torch_compile_guard

from build_targets import (  # noqa: F401 — re-exported for callers
    GFX_MAP,
    filter_tune_df,
    get_build_targets_env,
)

logger = logging.getLogger("aiter")


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
            match = re.search(r"\b(gfx\w+)\b", line, re.IGNORECASE)
            if match:
                return [match.group(1).lower()]
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
    if gfx == "native":
        gfx = _detect_native()[0]
    elif ";" in gfx:
        # TODO: multi-arch GPU_ARCHS (e.g. "gfx942;gfx950") — picking the
        # last entry is a known limitation for build-time codegen callers.
        # For runtime dispatch, prefer get_gfx_runtime().
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
def get_gfx_runtime() -> str:
    """Return the arch of the live GPU, always via rocminfo.

    Unlike get_gfx(), ignores GPU_ARCHS — always detects the actual running
    GPU.  Use for runtime dispatch decisions (selecting tuned kernels, picking
    code paths).  Use get_gfx() for build-time codegen paths (gen_instances,
    csrc module-level arch selection) where no GPU may be available.
    """
    gfx_arch = _detect_native()[0]
    supported = set(GFX_MAP.values())
    if gfx_arch not in supported:
        raise KeyError(
            f"Unknown GPU architecture: {gfx_arch}. "
            f"Supported architectures: {sorted(supported)}"
        )
    return gfx_arch


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
      1. GPU_ARCHS set to an explicit non-empty target list → delegate to
         get_build_targets_env() (no GPU needed).
      2. GPU_ARCHS unset, empty/whitespace, or "native" → detect the live GPU
         and use actual (gfx, cu_num) from rocminfo, which correctly reflects
         partition mode and binned variants.
      3. Neither → raise RuntimeError with a clear message.
    """
    gpu_archs = os.getenv("GPU_ARCHS")
    gpu_archs_normalized = gpu_archs.strip() if gpu_archs is not None else ""
    if gpu_archs_normalized and gpu_archs_normalized.lower() != "native":
        return get_build_targets_env()

    try:
        # get_gfx() is intentional here — this is a build-time path; get_gfx_runtime()
        # would fail in CI environments without a live GPU.
        return [(get_gfx(), get_cu_num())]
    except Exception as e:
        raise RuntimeError(
            "No GPU detected and GPU_ARCHS is not set to an explicit target. "
            "Set GPU_ARCHS=gfx942 (or similar) to build without a GPU."
        ) from e


def build_tune_dict(
    tune_df, default_dict, kernels_list, libtype=None, kernels_by_name=None
):
    """Filter tune_df to rows matching the current build targets and return a
    (gfx, cu_num, M, N, K)-keyed dispatch dict, starting from a copy of default_dict.

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
                          lookup uses the name instead of kernelId. If the name is not
                          found in kernels_by_name, the entry is skipped (heuristic
                          default used) and a warning is logged — no kernelId fallback
                          is attempted, because kernelIds are not stable across kernel
                          list reorderings. Falls back to kernelId if the kernelName
                          column is absent from the CSV.

    Returns:
        dict with mixed keys: negative ints (from default_dict) and
        (gfx, cu_num, M, N, K) 5-tuples (from the filtered CSV rows).
    """
    tune_dict = dict(default_dict)
    targets = get_build_targets()
    filtered = filter_tune_df(tune_df, targets)
    if libtype is not None and "libtype" in tune_df.columns:
        filtered = filtered[filtered["libtype"] == libtype]
    use_name = kernels_by_name is not None and "kernelName" in tune_df.columns
    if kernels_by_name is not None and not use_name:
        logger.warning(
            "kernels_by_name provided but CSV has no kernelName column, falling back to kernelId."
        )
    for _, row in filtered.iterrows():
        key = (
            str(row["gfx"]),
            int(row["cu_num"]),
            int(row["M"]),
            int(row["N"]),
            int(row["K"]),
        )
        if use_name:
            kname = str(row["kernelName"])
            kernel = kernels_by_name.get(kname)
            if kernel is not None:
                tune_dict[key] = kernel
            else:
                logger.warning(
                    f"kernelName '{kname}' not found in kernels_by_name "
                    f"(gfx={key[0]}, cu_num={key[1]}, M={key[2]}, N={key[3]}, K={key[4]}); "
                    f"falling back to heuristic default."
                )
        else:
            kid = int(row["kernelId"])
            kernel = kernels_list.get(kid)
            if kernel is not None:
                tune_dict[key] = kernel
            else:
                logger.warning(
                    f"kernelId {kid} not in kernels_list "
                    f"(gfx={key[0]}, cu_num={key[1]}, M={key[2]}, N={key[3]}, K={key[4]}, "
                    f"kernels_list size={len(kernels_list)}); falling back to heuristic default."
                )
    return tune_dict


def build_tune_dict_batched(tune_df, default_dict, kernels_list, libtype=None):
    """Like build_tune_dict, but for batched GEMM modules whose dispatch key
    includes the batch dimension B.

    Builds a (gfx, cu_num, B, M, N, K) 6-tuple keyed dict suitable for use with
    BatchedGemmDispatchMap in the C++ dispatch layer.

    Args:
        tune_df:      pandas DataFrame loaded from the batched tuning CSV.
        default_dict: module-level fallback dict (negative-int keys) to start from.
        kernels_list: module-level dict mapping kernelId → kernelInstance.
        libtype:      Optional string to filter the "libtype" column (same semantics
                      as build_tune_dict).

    Returns:
        dict with mixed keys: negative ints (from default_dict) and
        (gfx, cu_num, B, M, N, K) 6-tuples (from the filtered CSV rows).
    """
    tune_dict = dict(default_dict)
    targets = get_build_targets()
    filtered = filter_tune_df(tune_df, targets)
    if libtype is not None and "libtype" in tune_df.columns:
        filtered = filtered[filtered["libtype"] == libtype]
    for _, row in filtered.iterrows():
        key = (
            str(row["gfx"]),
            int(row["cu_num"]),
            int(row["B"]),
            int(row["M"]),
            int(row["N"]),
            int(row["K"]),
        )
        kid = int(row["kernelId"])
        kernel = kernels_list.get(kid)
        if kernel is not None:
            tune_dict[key] = kernel
        else:
            logger.warning(
                f"kernelId {kid} not in kernels_list "
                f"(gfx={key[0]}, cu_num={key[1]}, B={key[2]}, M={key[3]}, N={key[4]}, K={key[5]}, "
                f"kernels_list size={len(kernels_list)}); falling back to heuristic default."
            )
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
      - Negative ints          (default_dict entries) → skipped in non-tune mode.
      - (gfx,cu_num,M,N,K) 5-tuples (tuned entries)  → written as {"gfx",cu_num,M,N,K} C++ key.
      - (gfx,cu_num,B,M,N,K) 6-tuples (batched)      → written as {"gfx",cu_num,B,M,N,K} C++ key.
      - Non-negative ints (tune mode only)            → written as plain integer kernel ID.

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
            if not istune and (isinstance(key, tuple) and isinstance(key[0], str)):
                # 5-tuple key: (gfx, cu_num, M, N, K)
                # 6-tuple key: (gfx, cu_num, B, M, N, K)
                # key[0] is the gfx arch string; the remaining elements are ints.
                cpp_key = (
                    '{"' + key[0] + '", ' + ", ".join(str(x) for x in key[1:]) + "}"
                )
                f.write(
                    lookup_template.format(
                        MNK=cpp_key,
                        kernel_name=k.name,
                    )
                )
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
