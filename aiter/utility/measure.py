# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Device facts that decide how a kernel timing reads its inputs."""

import functools

import torch
import triton
import triton.language as tl

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


@functools.cache
def llc_flush_buffer(device: int) -> torch.Tensor:
    """The 2 x LLC buffer that `flush_llc` reads, allocated once per device."""
    return torch.empty(2 * llc_bytes(device) // 8, dtype=torch.int64, device=device)


@triton.jit
def _flush_kernel(buf_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    acc = tl.zeros([BLOCK], dtype=tl.int64)
    for start in range(pid * BLOCK, n, tl.num_programs(0) * BLOCK):
        offs = start + tl.arange(0, BLOCK)
        acc += tl.load(buf_ptr + offs, mask=offs < n, other=0)
    tl.store(out_ptr + pid, tl.sum(acc))


@functools.cache
def _flush_partials(device: int) -> torch.Tensor:
    """One int64 sum per `_flush_kernel` workgroup, four workgroups per CU."""
    cus = torch.cuda.get_device_properties(device).multi_processor_count
    return torch.empty(4 * cus, dtype=torch.int64, device=device)


def flush_llc(device: int) -> None:
    """Push every earlier input out of the L2s and the LLC of ``device``."""
    # Our own kernel rather than a torch reduction: on gfx942 torch's float sum of
    # this buffer leaves the inputs cached, so the load must not be torch's choice.
    # Workgroups go round-robin over the XCDs, so every XCD's L2 is swept.
    buf, out = llc_flush_buffer(device), _flush_partials(device)
    _flush_kernel[(out.numel(),)](buf, out, buf.numel(), BLOCK=1024)
