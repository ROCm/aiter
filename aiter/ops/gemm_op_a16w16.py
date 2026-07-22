# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import torch
from torch import Tensor
from typing import Optional
from ..jit.core import compile_ops
import functools


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
    bias: Optional[Tensor] = None,
    splitK: Optional[int] = None,
    kernelName: Optional[str] = None,
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
    """
    stream = torch.cuda.current_stream(device)
    return _get_semaphore_workspace_keyed(device, stream.cuda_stream)


def gemm_a16w16_asm(
    A: Tensor,
    B: Tensor,
    out: Tensor,
    bias: Optional[Tensor] = None,
    splitK: Optional[int] = None,
    kernelName: Optional[str] = None,
    bpreshuffle: bool = False,
):
    if splitK is None or splitK > 1:
        sema = get_semaphore_workspace(out.device)
    else:
        sema = torch.empty((0,), dtype=torch.uint32, device=out.device)

    _gemm_a16w16_asm(A, B, out, sema, bias, splitK, kernelName, bpreshuffle)
    return out
