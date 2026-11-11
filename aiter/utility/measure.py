# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Device facts that decide how a kernel timing reads its inputs."""

import functools

import torch

from aiter import logger

MiB = 1 << 20

# Used only when amdsmi cannot answer. Each entry is an upper bound for its arch
# (MI350P has 128 MiB): an LLC taken too small times inputs warm, one taken too
# large only costs extra copies.
_LLC_BY_ARCH = {
    "gfx90a": 8 * MiB,
    "gfx942": 256 * MiB,
    "gfx950": 256 * MiB,
}
_LLC_UNKNOWN = 256 * MiB


def llc_from_cache_info(caches) -> int:
    """Bytes in the highest cache level of an ``amdsmi_get_gpu_cache_info`` list."""
    top = max(c["cache_level"] for c in caches)
    return max(
        c["cache_size"] * 1024 * c["num_cache_instance"]
        for c in caches
        if c["cache_level"] == top
    )


def _amdsmi_llc(device: int) -> int:
    from aiter.smi_monitor import amdsmi, hip_device_to_amdsmi_handle

    try:
        handle = hip_device_to_amdsmi_handle(device)
        caches = amdsmi.amdsmi_get_gpu_cache_info(handle)["cache"]
    except amdsmi.AmdSmiException as e:
        raise RuntimeError(f"{type(e).__name__}: {e}") from e
    return llc_from_cache_info(caches)


@functools.cache
def llc_bytes(device: int) -> int:
    """Last-level cache size of HIP device ``device``, in bytes."""
    try:
        return _amdsmi_llc(device)
    except (ImportError, RuntimeError, KeyError, ValueError) as e:
        reason = f"{type(e).__name__}: {e}"
    arch = torch.cuda.get_device_properties(device).gcnArchName.split(":")[0]
    llc = _LLC_BY_ARCH.get(arch, _LLC_UNKNOWN)
    logger.warning(
        f"amdsmi gave no LLC size for device {device} ({reason}); "
        f"using {llc // MiB} MiB for {arch}"
    )
    return llc
