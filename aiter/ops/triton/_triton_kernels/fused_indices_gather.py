import functools
import json
import os
import triton
import triton.language as tl
from ..utils._triton import arch_info
from ..utils.core import AITER_TRITON_CONFIGS_PATH

@triton.jit
def _fused_indices_and_gather_kernel(
    x2d,
    gather_out,
    idx1d,
    E: tl.constexpr,
    a: tl.constexpr,
    D: tl.constexpr,
    stridex0: tl.int32,
    stridex1: tl.int32,
    strideop0: tl.int32,
    strideop1: tl.int32,
    strideidx: tl.int32,:
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # Figure out which tile we are in
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    # Compute the absolute indices covered by this tile
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # rows r
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # cols c

    # Compute masks to prevent the last tiles from falling off the edge.
    mask_m = offs_m < (E * a)
    mask_n = offs_n < D

    # Compute the source row for each output row.
    row_vals = offs_m % a

    # Compute write addresses using the strides.
    # We need this for the gather output, the input tensor X, and the indices tensor idx2d
    src_x = row_vals[:, None] * stridex0 + offs_n[None, :] * stridex1
    dst_gather_out = offs_m[:, None] * strideop0 + offs_n[None, :] * strideop1

    # Now load the input tile from x2d
    tile = tl.load(x2d + src_x, mask=mask_m[:, None] & mask_n[None, :])
    # Store to out
    tl.store(gather_out + dst_gather_out, tile, mask=mask_m[:, None] & mask_n[None, :])
    if pid_n == 0:
        dst_idx = offs_m * strideidx
        tl.store(idx1d + dst_idx, row_vals.to(tl.int32), mask=mask_m)
