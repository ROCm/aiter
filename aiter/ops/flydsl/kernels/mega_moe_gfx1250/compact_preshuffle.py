# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Repack an already-quantized compact wire without changing FP4 or scales.

The separate launch serves ordinary compact GEMM and the S1F configuration
probe. An active S1F consumer loads the ready wire in the GEMM's LDS layout.
"""

import torch
import triton as tr
import triton.language as tl


@tr.jit
def _compact_fp4_preshuffle(
    A,
    SA,
    OUT,
    OUT_SA,
    PSUM,
    K: tl.constexpr,
    PITCH: tl.constexpr,
    SCALE_PITCH: tl.constexpr,
    EXPERTS: tl.constexpr,
    ALIGN: tl.constexpr,
    WMMA_REP: tl.constexpr,
    BLOCK: tl.constexpr,
    BISECT: tl.constexpr,
):
    group = tl.program_id(0)
    kt = tl.program_id(1)
    row0 = group * 16
    lo = tl.full((), 0, tl.int32)
    hi = tl.full((), EXPERTS, tl.int32)
    for _ in tl.static_range(BISECT):
        mid = (lo + hi) // 2
        value = tl.load(PSUM + tl.minimum(mid, EXPERTS - 1))
        right = value <= row0
        lo = tl.where(right, mid + 1, lo)
        hi = tl.where(right, hi, mid)
    expert = tl.minimum(lo, EXPERTS - 1)
    end = tl.load(PSUM + expert)
    prev = tl.load(PSUM + tl.maximum(expert - 1, 0))
    begin = tl.where(expert > 0, tl.cdiv(prev, ALIGN) * ALIGN, 0)
    x = tl.arange(0, BLOCK)
    row = row0 + x // 64
    col = x % 64
    live = (lo < EXPERTS) & (row >= begin) & (row < end)
    payload = tl.load(A + row * PITCH + kt * 64 + col, mask=live, other=0)
    dst = (
        group * (K // 2 * 16)
        + kt * 1024
        + (col // 16) * 256
        + (row % 16) * 16
        + col % 16
    )
    tl.store(OUT + dst, payload)
    s = tl.arange(0, 64)
    sr = row0 + s // 4
    sc = kt * 4 + s % 4
    scale = tl.load(
        SA + sr * SCALE_PITCH + sc,
        mask=(lo < EXPERTS) & (sr >= begin) & (sr < end),
        other=0,
    )
    mspan = 16 * WMMA_REP
    soff = (
        (sr // mspan) * (K // 32 * mspan) + kt * (mspan * 4) + (sr % mspan) * 4 + s % 4
    )
    tl.store(OUT_SA + soff, scale)


def compact_fp4_preshuffle(payload, scales, psum, *, rows, k, experts, align, wmma_rep):
    if rows % (16 * wmma_rep) or k % 128:
        raise ValueError(
            "compact A preshuffle requires complete M16/scale and K128 groups"
        )
    out = torch.empty((1, rows, k // 2), device=payload.device, dtype=torch.uint8)
    out_scales = torch.empty(
        (1, rows // wmma_rep, k // 32 * wmma_rep),
        device=payload.device,
        dtype=torch.uint8,
    )
    _compact_fp4_preshuffle[(rows // 16, k // 128)](
        payload,
        scales,
        out,
        out_scales,
        psum,
        K=k,
        PITCH=payload.stride(-2),
        SCALE_PITCH=scales.stride(-2),
        EXPERTS=experts,
        ALIGN=align,
        WMMA_REP=wmma_rep,
        BLOCK=1024,
        BISECT=experts.bit_length() + 1,
        num_warps=4,
    )
    return out, out_scales
