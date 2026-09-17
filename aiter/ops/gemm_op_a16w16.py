# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import functools

import torch
from torch import Tensor

from ..jit.core import compile_ops


@compile_ops(
    "module_gemm_a16w16_asm",
    fc_name="gemm_a16w16_asm",
    ffi_type="ctypes",
)
def _gemm_a16w16_asm(
    A: Tensor,
    B: Tensor,
    out: Tensor,
    semaphore: Tensor,
    bias: Tensor | None = None,
    splitK: int | None = None,
    kernelName: str | None = None,
    bpreshuffle: bool = False,
) -> None: ...


# Semaphore workspace shape for ASM SplitK kernels.
# The kernel indexes into a flat array of size rows*cols; candidates whose
# grid (gdx*gdy) exceeds this limit must be skipped to avoid out-of-bounds writes.
_SEMA_SHAPE = (16, 64)
ASM_SPLITK_MAX_GRID = _SEMA_SHAPE[0] * _SEMA_SHAPE[1]


@functools.lru_cache(maxsize=64)
def _get_semaphore_workspace_keyed(device: torch.device, stream_id: int) -> Tensor:
    return torch.zeros(_SEMA_SHAPE, dtype=torch.uint32, device=device)


# A graph records launches, not the allocation that zeroed the counter. Give
# each recorded launch its own slot and record the zero-fill inside the graph.
# Public: a host that records more than this raises it before its first launch.
CAPTURE_SEMAPHORE_POOL_SLOTS = 4096
_capture_rings: dict[torch.device, Tensor] = {}
_capture_ring_next: dict[torch.device, int] = {}


def _prime_capture_pool(device: torch.device) -> Tensor:
    """Allocate the capture pool. Never called from inside a capture."""
    ring = _capture_rings.get(device)
    if ring is None:
        ring = torch.zeros(
            (CAPTURE_SEMAPHORE_POOL_SLOTS, *_SEMA_SHAPE),
            dtype=torch.uint32,
            device=device,
        )
        _capture_rings[device] = ring
    return ring


def _get_captured_semaphore_workspace(device: torch.device) -> Tensor:
    ring = _capture_rings.get(device)
    if ring is None:
        raise RuntimeError(
            f"no split-K a16w16 semaphore pool on {device}: the pool is "
            "allocated by an eager call that takes the splitK path (splitK "
            "None or > 1); make one on this device before capturing a graph."
        )
    idx = _capture_ring_next.get(device, 0)
    # Bound against the pool as allocated, not the constant: raising the
    # constant after allocation must not let idx past the end of the tensor.
    if idx >= ring.shape[0]:
        raise RuntimeError(
            f"split-K a16w16 capture semaphore pool exhausted on {device}: "
            f"{ring.shape[0]} slots recorded. Raise "
            "aiter.ops.gemm_op_a16w16.CAPTURE_SEMAPHORE_POOL_SLOTS before the "
            "first split-K launch; the pool is sized when it is allocated."
        )
    _capture_ring_next[device] = idx + 1
    sema = ring[idx]
    sema.zero_()  # a graph node, so every replay starts from zero
    return sema


def get_semaphore_workspace(device: torch.device) -> Tensor:
    """Return a per-(device, stream) zero-initialized semaphore workspace.

    SplitK a16w16 ASM kernels use an atomic-counter protocol where the last
    workgroup performs the reduction phase. Concurrent launches on different
    streams must not share the same atomic counter, or the counts get mixed
    and the reduction phase never fires (deadlock).

    Reuse across launches on the same stream relies on the kernel resetting
    the counter to zero after the reduction completes; do not call this from
    callers that violate that invariant.

    Workspace size is small (~4 KB) and stream count per process is typically
    < 8, so the LRU cap of 64 leaves plenty of headroom before any in-flight
    workspace risks being evicted.

    Under capture this returns a slot from a per-device pool instead, with the
    zero-fill recorded as a graph node so replay restores counter == 0. That
    pool has to exist before capture starts, so the first eager splitK call on
    a device allocates it: a fixed CAPTURE_SEMAPHORE_POOL_SLOTS * 4 KiB, paid
    once per device even by a process that never captures a graph.
    """
    # torch.device("cuda") means "the current device", which is not a stable
    # key: resolve it now so the pool is keyed and allocated per GPU.
    if device.index is None:
        device = torch.device(device.type, torch.cuda.current_device())

    # is_current_stream_capturing() answers about the current device; only pay
    # the device switch (~1us, and this is a per-GEMM path) when it differs.
    if device.index == torch.cuda.current_device():
        capturing = torch.cuda.is_current_stream_capturing()
    else:
        with torch.cuda.device(device):
            capturing = torch.cuda.is_current_stream_capturing()
    if capturing:
        return _get_captured_semaphore_workspace(device)

    # Allocate here, never under capture: graph-pool memory cannot be freed.
    _prime_capture_pool(device)
    stream = torch.cuda.current_stream(device)
    return _get_semaphore_workspace_keyed(device, stream.cuda_stream)


def gemm_a16w16_asm(
    A: Tensor,
    B: Tensor,
    out: Tensor,
    bias: Tensor | None = None,
    splitK: int | None = None,
    kernelName: str | None = None,
    bpreshuffle: bool = False,
):
    if splitK is None or splitK > 1:
        sema = get_semaphore_workspace(out.device)
    else:
        sema = torch.empty((0,), dtype=torch.uint32, device=out.device)

    _gemm_a16w16_asm(A, B, out, sema, bias, splitK, kernelName, bpreshuffle)
    return out
