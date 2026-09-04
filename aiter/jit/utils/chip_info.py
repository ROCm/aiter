# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
import functools
import logging
import os
import re
import subprocess

from build_targets import (
    GFX_MAP,
    _parse_gpu_archs_env,
    filter_tune_df,
    get_build_targets_env,
)
from cpp_extension import executable_path
from torch_guard import torch_compile_guard

logger = logging.getLogger("aiter")


@functools.lru_cache(maxsize=1)
def _detect_native() -> list[str]:
    try:
        rocminfo = executable_path("rocminfo")
        result = subprocess.run(
            [rocminfo],
            capture_output=True,
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
        # TODO: multi-arch GPU_ARCHS (e.g. "gfx942;gfx950") -- picking the
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


_LDS_CAPACITY_BYTES = {
    "gfx90a": 64 * 1024,
    "gfx942": 64 * 1024,
    "gfx950": 160 * 1024,
    "gfx1100": 64 * 1024,
    "gfx1151": 64 * 1024,
    "gfx1201": 64 * 1024,
    "gfx1250": 320 * 1024,
}


def get_lds_capacity_bytes(gfx: str | None = None) -> int:
    """Return the architectural LDS capacity for one workgroup."""
    arch = (gfx or get_gfx()).split(":", 1)[0].lower()
    try:
        return _LDS_CAPACITY_BYTES[arch]
    except KeyError as exc:
        raise ValueError(f"Unknown LDS capacity for architecture {arch!r}") from exc


@functools.lru_cache(maxsize=1)
def get_gfx_runtime() -> str:
    """Return the arch of the live GPU, always via rocminfo.

    Unlike get_gfx(), ignores GPU_ARCHS -- always detects the actual running
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


# Backfill map for legacy tuned configs that predate the `gfx` column.
# These cu_num values were only ever tuned on a single arch historically:
#   256 -> gfx950, 80/228/304 -> gfx942 (MI308X / MI300A / MI300X).
# Newer archs that happen to share a cu_num (e.g. gfx1250 also reports 256)
# are always written with their real arch by the tuner, so they never rely on
# this backfill.
_LEGACY_CU_NUM_TO_GFX = {
    256: "gfx950",
    80: "gfx942",
    228: "gfx942",
    304: "gfx942",
}


def gfx_from_cu_num(cu_num) -> str:
    """Infer the gfx arch for a legacy config row that has no `gfx` column.

    Used to migrate old tuned CSVs (keyed on cu_num only) to the new
    (gfx, cu_num, ...) schema. Unknown cu_num falls back to the live GPU arch.
    """
    try:
        cu_num = int(cu_num)
    except (TypeError, ValueError):
        return get_gfx_runtime()
    gfx = _LEGACY_CU_NUM_TO_GFX.get(cu_num)
    if gfx is not None:
        return gfx
    try:
        return get_gfx_runtime()
    except Exception:  # noqa: BLE001
        return "gfx942"


@functools.lru_cache(maxsize=1)
def get_gfx_list() -> list[str]:

    gfx_env = os.getenv("GPU_ARCHS", "native")
    if gfx_env == "native":
        try:
            gfxs = _detect_native()
        except RuntimeError:
            gfxs = ["cpu"]
    else:
        gfxs = _parse_gpu_archs_env(gfx_env)
    os.environ["AITER_GPU_ARCHS"] = ";".join(gfxs)

    return gfxs


@torch_compile_guard()
def get_cu_num_custom_op() -> int:
    cu_num = int(os.getenv("CU_NUM", "0"))
    if cu_num == 0:
        try:
            rocminfo = executable_path("rocminfo")
            result = subprocess.run(
                [rocminfo], capture_output=True, text=True, check=False
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
        except Exception as e:  # noqa: BLE001  blanket catch is intentional here
            raise RuntimeError(f"Get GPU Compute Unit from rocminfo failed {e!s}")
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
      1. GPU_ARCHS set to an explicit non-empty target list -> delegate to
         get_build_targets_env() (no GPU needed), then refine the cu_num of any
         target matching the live GPU (see below).
      2. GPU_ARCHS unset, empty/whitespace, or "native" -> call get_gfx()
         (GPU_ARCHS-aware; falls back to rocminfo when GPU_ARCHS is unset) and
         get_cu_num(), which correctly reflect partition mode and binned variants.
      3. Neither -> raise RuntimeError with a clear message.

    On the GPU_ARCHS path, GFX_CU_NUM_MAP supplies one canonical cu_num per arch
    (gfx942 -> 304, the MI300X SPX value). That is wrong for every other gfx942
    SKU and partition mode: on a 228-CU MI300A, `GPU_ARCHS=gfx942` alone yields
    ('gfx942', 304), gen_instances then filters out every (gfx942, 228) tuned row,
    and dispatch fails at runtime with "not present in the compiled registry".
    So when the live GPU's arch matches a requested target and CU_NUM was not set
    explicitly, prefer the live CU count. An explicit CU_NUM still wins, and
    cross-compiling for an arch that is not the live GPU is unaffected.
    """
    gpu_archs = os.getenv("GPU_ARCHS")
    gpu_archs_normalized = gpu_archs.strip() if gpu_archs is not None else ""
    if gpu_archs_normalized and gpu_archs_normalized.lower() != "native":
        targets = get_build_targets_env()
        if os.getenv("CU_NUM"):
            return targets
        try:
            live_gfx, live_cu = get_gfx_runtime(), get_cu_num()
        except Exception:  # noqa: BLE001  no live GPU: keep the table defaults
            return targets
        return [
            (gfx, live_cu if gfx == live_gfx else cu_num) for gfx, cu_num in targets
        ]

    try:
        # get_gfx() is intentional here -- this is a build-time path; get_gfx_runtime()
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
        kernels_list:     module-level dict mapping kernelId -> kernelInstance.
        libtype:          Optional string to filter the "libtype" column (e.g. "ck").
                          Required for CSVs that mix multiple library types (e.g.
                          a8w8_bpreshuffle_tuned_gemm.csv mixes "ck" and "cktile").
                          If None, no libtype filtering is applied.
        kernels_by_name:  Optional dict mapping kernelName string -> kernelInstance.
                          When provided and the CSV has a "kernelName" column, kernel
                          lookup uses the name instead of kernelId. Falls back to
                          kernelId if the kernelName column is absent from the CSV.

    Strict on stale tuned-CSV rows: any row whose kernelName (or kernelId, in the
    fallback path) is not present in the registry will raise RuntimeError listing
    every offending row. A row that codegen silently drops would otherwise compile
    into a .so guaranteed to TORCH_CHECK(false, ...) at runtime for that shape.

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
    bad_rows: list[str] = []
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
                bad_rows.append(
                    f"  kernelName={kname!r} not in kernels_by_name "
                    f"(gfx={key[0]}, cu_num={key[1]}, M={key[2]}, N={key[3]}, K={key[4]})"
                )
        else:
            kid = int(row["kernelId"])
            kernel = kernels_list.get(kid)
            if kernel is not None:
                tune_dict[key] = kernel
            else:
                bad_rows.append(
                    f"  kernelId={kid} not in kernels_list "
                    f"(gfx={key[0]}, cu_num={key[1]}, M={key[2]}, N={key[3]}, K={key[4]}, "
                    f"kernels_list size={len(kernels_list)})"
                )
    if bad_rows:
        raise RuntimeError(
            "build_tune_dict: tuned CSV references kernels not in the build registry. "
            "Either re-tune the CSV against the current kernel list or restore the "
            "missing kernel definition; the build refuses to produce a .so that would "
            "TORCH_CHECK(false, ...) at runtime for these shapes:\n"
            + "\n".join(bad_rows)
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
        kernels_list: module-level dict mapping kernelId -> kernelInstance.
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
    bad_rows: list[str] = []
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
            bad_rows.append(
                f"  kernelId={kid} not in kernels_list "
                f"(gfx={key[0]}, cu_num={key[1]}, B={key[2]}, M={key[3]}, N={key[4]}, K={key[5]}, "
                f"kernels_list size={len(kernels_list)})"
            )
    if bad_rows:
        raise RuntimeError(
            "build_tune_dict_batched: tuned CSV references kernels not in the build "
            "registry. Either re-tune the CSV against the current kernel list or "
            "restore the missing kernel definition; the build refuses to produce a "
            ".so that would TORCH_CHECK(false, ...) at runtime for these shapes:\n"
            + "\n".join(bad_rows)
        )
    return tune_dict


def write_name_keyed_lookup_header(
    output_path, kernels_dict, lookup_head, lookup_template, lookup_end
):
    """Write a name-keyed C++ GEMM dispatch lookup header from a kernels_dict.

    Sister of write_lookup_header(), but emits {"<kernel_name>", &kernel<...>}
    entries instead of (gfx,cu_num,M,N,K) tuple keys.  Used by the blockscale
    GEMM modules whose runtime dispatch is now driven by Python-resolved
    kernel name strings (read from the tuned CSV) rather than a build-time
    tuple-keyed lookup.  The kernels_dict may contain duplicate entries for
    the same kernel (multiple shapes mapping to the same kernel.name); we
    dedupe by name so each kernel is registered exactly once.

    Skips negative-int default_dict keys (heuristic fallbacks the dispatch
    layer references directly by symbol).

    Args:
        output_path:     Full path of the .h file to write.
        kernels_dict:    Dict returned by build_tune_dict.
        lookup_head:     String written before the loop (defines the macro header).
        lookup_template: String with {kernel_name} placeholder (used twice:
                          once for the C++ string key, once for the symbol).
        lookup_end:      String written after the loop (closes the macro / #endif).
    """
    seen = set()
    with open(output_path, "w") as f:
        f.write(lookup_head)
        for key, k in kernels_dict.items():
            if isinstance(key, int) and key < 0:
                # default_dict heuristic-fallback entries; the dispatch layer
                # references the heuristic kernel by symbol, not via the table.
                continue
            if k.name in seen:
                continue
            seen.add(k.name)
            f.write(lookup_template.format(kernel_name=k.name))
        f.write(lookup_end)


def write_lookup_header(
    output_path, kernels_dict, lookup_head, lookup_template, lookup_end, istune=False
):
    """Write a C++ GEMM dispatch lookup header from a kernels_dict.

    Replaces the duplicated gen_lookup_dict loop in each gen_instances.py codegen
    class.  Each module still defines its own lookup_head / lookup_template /
    lookup_end strings (they embed the module-specific GENERATE_LOOKUP_TABLE macro
    type parameters), but the iteration and key-formatting logic is shared here.

    Key layout in kernels_dict:
      - Negative ints          (default_dict entries) -> skipped in non-tune mode.
      - (gfx,cu_num,M,N,K) 5-tuples (tuned entries)  -> written as {"gfx",cu_num,M,N,K} C++ key.
      - (gfx,cu_num,B,M,N,K) 6-tuples (batched)      -> written as {"gfx",cu_num,B,M,N,K} C++ key.
      - Non-negative ints (tune mode only)            -> written as plain integer kernel ID.

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
    """Return the PCI device id of a GPU (e.g. 0x74A0), or None if unavailable.

    Resolved via hipDeviceGetPCIBusId plus sysfs rather than
    hipDeviceGetAttribute(hipDeviceAttributePciChipId). The AMD-specific block
    of hipDeviceAttribute_t is an unnumbered enum, so the ordinal of
    PciChipId shifts whenever an attribute is inserted ahead of it. On
    ROCm 7.2 the ordinal this used to hardcode (10019) resolves to
    hipDeviceAttributeMaxAvailableVgprsPerThread and returns 512 on every
    device, which silently defeated the MI308 check below.
    """
    import ctypes

    try:
        libhip = ctypes.CDLL("libamdhip64.so")
        buf = ctypes.create_string_buffer(64)
        if libhip.hipDeviceGetPCIBusId(buf, len(buf), device_id) != 0:
            return None
        bdf = buf.value.decode()
        with open(f"/sys/bus/pci/devices/{bdf}/device") as f:
            return int(f.read().strip(), 16)
    except (OSError, ValueError, UnicodeDecodeError, AttributeError):
        return None


MI308_CHIP_IDS = {0x74A2, 0x74A8, 0x74B6, 0x74BC}
# MI300A is the gfx942 APU: 228 CUs and 6 XCCs, against MI300X's 304 and 8.
MI300A_CHIP_IDS = {0x74A0}


def get_device_name():
    gfx = get_gfx()

    if gfx == "gfx942":
        chip_id = _get_pci_chip_id()
        if chip_id in MI308_CHIP_IDS:
            return "MI308"
        if chip_id in MI300A_CHIP_IDS:
            return "MI300A"
        return "MI300"
    elif gfx == "gfx950":
        return "MI350"
    elif gfx == "gfx1250":
        return "MI400"
    else:
        raise RuntimeError("Unsupported gfx")
