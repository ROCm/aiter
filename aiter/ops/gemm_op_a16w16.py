# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.


import torch
from torch import Tensor

from ..jit.core import compile_ops
from ..utility.graph_alloc import capture_scoped_cache, persistent_alloc


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


# The ASM kernels reach A, B and the output only through buffer descriptors:
# the per-lane offset is an unsigned 32-bit byte offset and `num_records` is
# 32 bits as well, so one launch cannot span an operand of 2**32 bytes or more.
# The kernels do not check this and silently return wrong values (#4743).
ASM_BUFFER_LIMIT_BYTES = 1 << 32
# Rows are padded up to the kernel's tileM (at most 256). Chunks are a multiple
# of the lcm of every shipped tileM (32/48/64/80/96/128/160/256), so only the
# last chunk is ever padded and it is never longer than a full chunk.
_ASM_MAX_TILE_M = 256
_ASM_TILE_M_LCM = 3840


def asm_m_chunk_rows(
    M: int, a_row_bytes: int, out_row_bytes: int, b_bytes: int
) -> int | None:
    """Rows per launch that keep A and the output addressable, or None.

    None means a single launch is safe, so the call is left untouched. Raises
    ValueError when no row split can help: B cannot be split along N because
    the output row stride is derived from N, and a single row may be too large.
    """
    limit = ASM_BUFFER_LIMIT_BYTES
    if b_bytes >= limit:
        raise ValueError(
            f"gemm_a16w16_asm: B spans {b_bytes} bytes, which exceeds the 32-bit "
            f"buffer addressing limit of {limit} bytes and cannot be split"
        )
    row_bytes = max(a_row_bytes, out_row_bytes)
    if (M + _ASM_MAX_TILE_M - 1) * row_bytes < limit:
        return None
    rows = (limit - 1) // row_bytes // _ASM_TILE_M_LCM * _ASM_TILE_M_LCM
    if rows == 0:
        raise ValueError(
            f"gemm_a16w16_asm: a row of A ({a_row_bytes} bytes) or of the output "
            f"({out_row_bytes} bytes) is too large to split M within the 32-bit "
            f"buffer addressing limit of {limit} bytes"
        )
    return rows


@capture_scoped_cache
def _get_semaphore_workspace_keyed(device: torch.device, stream_id: int) -> Tensor:
    with persistent_alloc(device):
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
    bias: Tensor | None = None,
    splitK: int | None = None,
    kernelName: str | None = None,
    bpreshuffle: bool = False,
):
    if splitK is None or splitK > 1:
        sema = get_semaphore_workspace(out.device)
    else:
        sema = torch.empty((0,), dtype=torch.uint32, device=out.device)

    M, N = A.shape[0], B.shape[0]
    rows = asm_m_chunk_rows(
        M,
        A.stride(0) * A.element_size(),
        N * out.element_size(),
        N * B.stride(0) * B.element_size(),
    )
    if rows is None:
        _gemm_a16w16_asm(A, B, out, sema, bias, splitK, kernelName, bpreshuffle)
        return out

    # A row slice of a row-major tensor starts the kernel's offsets at zero.
    # Launches are stream-ordered, so the split-K semaphore, which the kernel
    # resets after each reduction, is reused as is.
    for i in range(0, M, rows):
        j = min(i + rows, M)
        _gemm_a16w16_asm(
            A[i:j], B, out[i:j], sema, bias, splitK, kernelName, bpreshuffle
        )
    return out
