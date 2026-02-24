# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

from triton.experimental import gluon
import triton.experimental.gluon.language as gl


@gluon.jit
def issue_l2_prefetch(desc, offsets, pred=True):
    """
    Prefetch a single 2D descriptor tile at the given [row, col] offsets.
    """
    gl.amd.gfx1250.tdm.prefetch(desc, offsets, pred=pred)


@gluon.jit
def gemm_l2_prefetch(distance, load_idx, a_desc, b_desc, off_am, off_bn, BLOCK_K: gl.constexpr,
                     TRANSPOSE_A: gl.constexpr, TRANSPOSE_B: gl.constexpr, pred=True):
    """
    Creates L2 prefetch for iteration `load_idx + distance` for a GEMM's A and B descriptors.
    """
    if distance == 0:
        return

    off_k = (load_idx + distance) * BLOCK_K
    if not TRANSPOSE_A:
        issue_l2_prefetch(a_desc, [off_am, off_k], pred=pred)
    else:
        issue_l2_prefetch(a_desc, [off_k, off_am], pred=pred)
    if not TRANSPOSE_B:
        issue_l2_prefetch(b_desc, [off_k, off_bn], pred=pred)
    else:
        issue_l2_prefetch(b_desc, [off_bn, off_k], pred=pred)


@gluon.jit
def gemm_l2_prefetch_prologue(distance, load_idx, a_desc, b_desc, off_am, off_bn, BLOCK_K: gl.constexpr,
                              NUM_BUFFERS: gl.constexpr, TRANSPOSE_A: gl.constexpr,
                              TRANSPOSE_B: gl.constexpr, pred=True):
    """
    Creates prefetches for iterations [NUM_BUFFERS, distance - NUM_BUFFERS) or no prefetches if distance <= NUM_BUFFERS.
    Skips iterations preloaded in the prologue since prefetching them is redundant.
    """
    if distance <= NUM_BUFFERS:
        return

    for i in gl.static_range(NUM_BUFFERS - distance):
        gemm_l2_prefetch(NUM_BUFFERS + i, load_idx, a_desc, b_desc, off_am, off_bn, BLOCK_K,
                         TRANSPOSE_A, TRANSPOSE_B, pred)
