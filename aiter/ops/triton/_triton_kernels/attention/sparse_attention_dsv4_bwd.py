# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Triton kernels for the DeepSeek-V4 sparse-MLA training BACKWARD (gfx950 / CDNA4).

The two memory-bound phases; the MFMA phases are Gluon and live in
``aiter.ops.triton._gluon_kernels.gfx950.attention.sparse_attention_dsv4_bwd``.

``_delta_v4_kernel``
    ``delta = rowsum(O * dO)`` -- the standard flash-attention "o_dot_do" preamble. Streams the
    bf16 inputs and accumulates in fp32, so it moves exactly the working set.

``_bwd_dkv_gather_acc_v4``
    Reduces ``interm[t, slot, :]`` into ``dkv[kv_row, :]`` over the top-k mapping. The scatter is
    inverted into a CSR gather (each output KV row collects its own contributors), so no atomics
    are needed.

Launchers live in ``aiter.ops.triton.attention.sparse_attention_dsv4_bwd``; this module stays
free of torch so the kernels can be called without it.
"""

import triton
import triton.language as tl

from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr

_delta_v4_kernel_repr = make_kernel_repr(
    "_delta_v4_kernel",
    [
        "D",
        "BLOCK_R",
    ],
)


@triton.jit(repr=_delta_v4_kernel_repr)
def _delta_v4_kernel(
    O_ptr,  # [n_rows, D] bf16   (rows = T*H, contiguous)
    dO_ptr,  # [n_rows, D] bf16
    Delta_ptr,  # [n_rows]    fp32
    n_rows,
    D: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    """Grid (cdiv(n_rows, BLOCK_R),) — each program reduces BLOCK_R rows of width D."""
    pid = tl.program_id(0)
    rows = pid * BLOCK_R + tl.arange(0, BLOCK_R)
    mask = rows < n_rows
    offs = rows.to(tl.int64)[:, None] * D + tl.arange(0, D)[None, :]
    o = tl.load(O_ptr + offs, mask=mask[:, None], other=0.0).to(tl.float32)
    d = tl.load(dO_ptr + offs, mask=mask[:, None], other=0.0).to(tl.float32)
    tl.store(Delta_ptr + rows, tl.sum(o * d, axis=1), mask=mask)


_bwd_dkv_gather_acc_v4_repr = make_kernel_repr(
    "_bwd_dkv_gather_acc_v4",
    [
        "D",
        "BLOCK_E",
        "ACCUMULATE",
    ],
)


@triton.jit(repr=_bwd_dkv_gather_acc_v4_repr)
def _bwd_dkv_gather_acc_v4(
    Interm_ptr,  # [T, R_CHUNK, D] bf16, flat [T*R_CHUNK, D]
    InvPtr_ptr,  # [num_kv+1] int32 — CSR row pointers
    InvData_ptr,  # [valid] int32 — encoded q*R_CHUNK+local_r, sorted by KV token
    dKV_acc_ptr,  # [num_kv, D] fp32 — accumulator
    stride_interm_r: tl.int64,
    stride_acc_t: tl.int64,
    D: tl.constexpr,
    BLOCK_E: tl.constexpr,
    ACCUMULATE: tl.constexpr,
):
    """Grid (num_kv,) — one CTA per KV token, BLOCK_E CSR entries in flight.

    ``BLOCK_E`` entries are carried per iteration, which the gather needs for two reasons:

      * **load width.** A bare ``tl.arange(0, D)`` block over 256 threads is 2 bf16 = 4 B per
        lane -- a dword. The [BLOCK_E, D] block gives ``BLOCK_E*D/threads`` elements per lane,
        so the loads become dwordx4. The gather is issue-bound, so this dominates.
      * **trip count.** ``tl.sum`` folds the entry axis, so the run is consumed BLOCK_E at a
        time. A realistic top-k gives run lengths up to ~3000 on the pool rows.

    ``ACCUMULATE=False`` writes the destination instead of reading it back first. The caller
    uses it for the first chunk, where the accumulator is still zero.
    """
    k = tl.program_id(0)
    offs_d = tl.arange(0, D)
    offs_e = tl.arange(0, BLOCK_E)
    start = tl.load(InvPtr_ptr + k)
    end = tl.load(InvPtr_ptr + k + 1)
    acc_base = k.to(tl.int64) * stride_acc_t

    if ACCUMULATE:
        acc = tl.load(dKV_acc_ptr + acc_base + offs_d).to(tl.float32)
    else:
        acc = tl.zeros([D], dtype=tl.float32)

    for i0 in range(start, end, BLOCK_E):
        idx = i0 + offs_e
        m = idx < end
        entry = tl.load(InvData_ptr + idx, mask=m, other=0).to(tl.int64)
        vals = tl.load(
            Interm_ptr + entry[:, None] * stride_interm_r + offs_d[None, :],
            mask=m[:, None],
            other=0.0,
        )
        acc += tl.sum(vals.to(tl.float32), axis=0)

    tl.store(dKV_acc_ptr + acc_base + offs_d, acc)
