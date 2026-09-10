# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import torch
from torch import Tensor

from ..jit.core import compile_ops
from ..utility.graph_buffers import per_stream_buffers_shared_by_graphs


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


@per_stream_buffers_shared_by_graphs
def get_semaphore_workspace(device: torch.device) -> Tensor:
    """Return a zero-initialized semaphore workspace for SplitK a16w16 ASM.

    The kernels use an atomic-counter protocol where the last workgroup runs
    the reduction phase and resets the counter to zero, so a cached workspace
    can be reused; do not call from paths that violate that invariant.
    """
    return torch.zeros(_SEMA_SHAPE, dtype=torch.uint32, device=device)


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
