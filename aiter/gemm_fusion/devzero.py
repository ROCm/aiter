# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Device-side zeroing of the comm windows for reset().

The H2D memcpy zero (MORI's own method for ccoMemAlloc windows, kept for the first zeroing in finalize()) is executed by the DMA engines BEHIND the GPU's L2. After an error
the pages hold values written by kernels (ERR through system-scope atomics, STEP through torch's add_ with plain stores): a dirty or stale cache line that is written back after the
DMA zero would resurrect the old content (the "rank 3's ERR word came back after reset()" symptom, hypothesis B). A DEVICE-side fill goes through the same L2 as every kernel
access, so it overwrites such a line coherently; the caller follows it with device syncs (and the verification re-reads ERR / STEP / DIAG after the final barrier).

The default fill is torch's own fill kernel over a zero-copy view of the window (no build step, works on any mapped VA). NOTE: its stores are plain stores, not explicit system-scope
stores; a hand-written kernel with system-scope stores would be a csrc / build change (the comm library is a host-only clang++ build) and is not part of this module.
"""

from __future__ import annotations

from .errpage import DevView

CHUNK_BYTES = (
    1 << 30
)  # one fill launch per GiB: keeps the int32 element count of a view small and the tensors cheap


def torch_available() -> bool:
    """True when torch can launch a device fill (torch importable with a usable CUDA/HIP device). A missing torch, a fake torch without `cuda` or a driver error all mean no."""
    try:
        import torch

        return bool(torch.cuda.is_available())
    except (ImportError, AttributeError, RuntimeError):
        return False


def torch_fill_zero(device: int, local: int, nbytes: int) -> None:
    """Zero [local, local + nbytes) of HIP device `device` with a device-side fill kernel, then wait for it. nbytes must be a multiple of 4 (windows are 4 MiB multiples)."""
    import torch

    if nbytes % 4 or local % 4:
        raise ValueError(
            f"devzero needs a 4-byte aligned range, got {local:#x} + {nbytes}"
        )
    off = 0
    while off < nbytes:
        n = min(CHUNK_BYTES, nbytes - off)
        view = torch.as_tensor(
            DevView(local + off, (n // 4,), "<i4"), device=f"cuda:{device}"
        )
        view.zero_()
        off += n
    torch.cuda.synchronize(device)
