# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

from triton.experimental import gluon
import triton.experimental.gluon.language as gl


@gluon.jit
def issue_l2_prefetches(distance, load_idx, a_desc, b_desc, off_am, off_bn, BLOCK_K: gl.constexpr,
                        TRANSPOSE_A: gl.constexpr, TRANSPOSE_B: gl.constexpr, pred=True):
    """
    Creates L2 prefetch for iteration `load_idx + distance`.
    """
    if distance == 0:
        return

    prefetch_iteration = load_idx + distance
    if not TRANSPOSE_A:
        gl.amd.gfx1250.tdm.prefetch(a_desc, [off_am, prefetch_iteration * BLOCK_K], pred=pred)
    else:
        gl.amd.gfx1250.tdm.prefetch(a_desc, [prefetch_iteration * BLOCK_K, off_am], pred=pred)
    if not TRANSPOSE_B:
        gl.amd.gfx1250.tdm.prefetch(b_desc, [prefetch_iteration * BLOCK_K, off_bn], pred=pred)
    else:
        gl.amd.gfx1250.tdm.prefetch(b_desc, [off_bn, prefetch_iteration * BLOCK_K], pred=pred)


@gluon.jit
def issue_l2_prefetches_prologue(distance, load_idx, a_desc, b_desc, off_am, off_bn, BLOCK_K: gl.constexpr,
                                 NUM_BUFFERS: gl.constexpr, TRANSPOSE_A: gl.constexpr,
                                 TRANSPOSE_B: gl.constexpr, pred=True):
    """
    Creates prefetches for iterations beyond what the prologue loads.
    No prefetches are issued if distance <= NUM_BUFFERS.
    """
    if distance <= NUM_BUFFERS:
        return

    for i in gl.static_range(distance - NUM_BUFFERS):
        issue_l2_prefetches(NUM_BUFFERS + i, load_idx, a_desc, b_desc, off_am, off_bn, BLOCK_K,
                            TRANSPOSE_A, TRANSPOSE_B, pred)
