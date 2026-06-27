# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Gluon grouped row-reduce for the gfx1250 MoE scatter-combine (replaces the Triton ``_reduce_grouped``).

One workgroup per group sums the ``K*B`` rows ``indx[g, :]`` (TDM-loaded, summed
in-register, no cross-wave communication) into ``out[g, :N]``, with optional
external residual fold-in. Swiglu stays on the Triton path.
"""

import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl


@gluon.jit
def reduce_grouped_gluon(
    X,  # [B, M, N] (flattened to [B*M, N] in the descriptor)
    Out,  # [num_groups, N]
    InIndx,  # [num_groups, K] int
    Residual,  # [num_groups, N] external residual to fold in (dummy ptr if unused)
    stride_xm,
    stride_om,
    stride_on,
    stride_res_m,
    stride_res_n,
    M,
    N: gl.constexpr,
    NPAD: gl.constexpr,  # next_pow2(N)
    B: gl.constexpr,
    K: gl.constexpr,
    NUM_WARPS: gl.constexpr,
    HAS_EXT_RESIDUAL: gl.constexpr,
):
    group = gl.program_id(0)

    # Load a power-of-2 column tile NPAD>=N (TDM block dims must be pow2) while the descriptor shape stays at true N, so TDM zero-pads cols [N:NPAD) (masked off on store).
    SIZE_N: gl.constexpr = NPAD // (NUM_WARPS * 32)
    BLKN: gl.constexpr = gl.BlockedLayout([1, SIZE_N], [1, 32], [1, NUM_WARPS], [1, 0])
    SH: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, order=[1, 0])

    x_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        X, [B * M, N], [stride_xm, 1], [1, NPAD], SH
    )
    smem = gl.allocate_shared_memory(X.dtype.element_ty, [K * B, 1, NPAD], SH)

    # issue all K*B row loads (overlapped), then reduce
    buf = 0
    for i in gl.static_range(K):
        idx_i = gl.load(InIndx + group * K + i)
        for b in gl.static_range(B):
            row = b * M + idx_i
            gl.amd.gfx1250.tdm.async_load(x_desc, [row, 0], smem.index(buf))
            buf += 1
    gl.amd.gfx1250.tdm.async_wait(0)

    acc = gl.zeros([1, NPAD], dtype=gl.float32, layout=BLKN)
    buf = 0
    for i in gl.static_range(K):
        for b in gl.static_range(B):
            acc += smem.index(buf).load(BLKN).to(gl.float32)
            buf += 1

    offs_n = gl.arange(0, NPAD, layout=gl.SliceLayout(0, BLKN))
    o_offs = group * stride_om + offs_n[None, :] * stride_on
    o_mask = offs_n[None, :] < N

    # Fold in the external residual before writeback (matches the Triton HAS_EXT_RESIDUAL path).
    if HAS_EXT_RESIDUAL:
        r_offs = group * stride_res_m + offs_n[None, :] * stride_res_n
        res = gl.amd.gfx1250.buffer_load(Residual, r_offs, mask=o_mask, other=0.0)
        acc += res.to(gl.float32)

    gl.amd.gfx1250.buffer_store(acc.to(Out.dtype.element_ty), Out, o_offs, mask=o_mask)


def reduce_grouped_gluon_num_warps(npad: int) -> int:
    """Pick the largest wave count W in {8,4,2,1} with ``npad % (W*32) == 0``."""
    for w in (8, 4, 2, 1):
        if npad % (w * 32) == 0:
            return w
    return 1
