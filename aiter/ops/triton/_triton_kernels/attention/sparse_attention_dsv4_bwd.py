# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Triton kernels for the DeepSeek-V4 sparse-MLA training BACKWARD (gfx950 / CDNA4).

``_delta_v4_kernel``
    ``delta = rowsum(O * dO)`` -- the standard flash-attention "o_dot_do" preamble. Streams the
    bf16 inputs and accumulates in fp32, so it moves exactly the working set.

``_bwd_dkv_gather_acc_v4_be`` + ``build_inverted_topk_fast``
    Reduce ``interm[t, slot, :]`` into ``dkv[kv_row, :]`` over the top-k mapping. The scatter is
    inverted into a CSR gather (each output KV row collects its own contributors), so no atomics
    are needed. ``BLOCK_E`` entries are carried per loop iteration, which both widens the load
    and cuts the trip count on the long runs a realistic top-k produces.

Public entry: ``aiter.ops.triton.attention.sparse_attention_dsv4_bwd.sparse_mla_bwd_dsv4``.
"""

import torch
import triton
import triton.language as tl


@triton.jit
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


def delta_v4(o, do, out=None, BLOCK_R=8, num_warps=8):
    # BLOCK_R=8 / num_warps=8 measured best (0.173 ms, 6.21 TB/s = 78% peak at T4096 H128);
    # the whole sweep plateaus at 0.173-0.187 once a lane loads >= 8 bf16, i.e. once the load
    # is a dwordx4. Below that (BLOCK_R=2 nw=8, 2 bf16/lane) it falls off a cliff to 3.12 TB/s.
    """o[T,H,D] bf16, do[T,H,D] bf16 -> delta[T,H] fp32 = sum_d o*do.

    ``do`` must already be the D-wide (lora) slice, contiguous — same contract as the dQ kernel.
    """
    assert o.shape == do.shape and o.is_contiguous() and do.is_contiguous()
    T, H, D = o.shape
    n_rows = T * H
    if out is None:
        out = torch.empty(T, H, dtype=torch.float32, device=o.device)
    _delta_v4_kernel[(triton.cdiv(n_rows, BLOCK_R),)](
        o,
        do,
        out,
        n_rows,
        D=D,
        BLOCK_R=BLOCK_R,
        num_warps=num_warps,
    )
    return out


@triton.jit
def _bwd_dkv_gather_acc_v4_be(
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

    Fixes two things about ``_bwd_dkv_gather_acc_v4``, which walks the run one entry at a time
    with a bare ``tl.arange(0, D)``:

      * **load width.** A [D] block over 256 threads is 2 bf16 = 4 B per lane -- a dword. The
        [BLOCK_E, D] block gives ``BLOCK_E*D/threads`` elements per lane instead, so the loads
        become dwordx4. The gather is issue-bound, so this is the dominant term.
      * **loop trip count.** The run is consumed BLOCK_E entries at a time rather than one, and
        ``tl.sum`` over the entry axis folds them. The realistic topk gives run lengths up to
        ~3000 (pool rows), so the serial walk was the other half of the problem.

    ``ACCUMULATE=False`` skips the read-modify-write of the destination, valid when the caller
    does not chunk (each KV row is then written by exactly one CTA).
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


def build_inverted_topk_fast(topk_indices_slice, num_kv):
    """CSR inverted index over ``num_kv`` KV rows. Bit-identical to the reference below.

    One stable sort yields both the permutation (``inv_data``) and the sorted keys;
    ``inv_ptr[k] = searchsorted(sorted, k, 'left')`` = the number of entries with value < k,
    which is exactly what ``cumsum(bincount(flat+1))`` computes. Invalid (-1) entries sort to
    the front, so ``inv_ptr[0]`` starts past them and they are never visited.

    Two things make this ~3x faster than the reference:
      * the sort key is narrowed to int16 when ``num_kv`` fits, so the radix sort makes 2
        byte-passes instead of 8;
      * ``searchsorted`` replaces the separate ``bincount`` + ``cumsum`` passes.

    Returns ``inv_ptr[num_kv+1]`` int32, ``inv_data[T*R]`` int32.
    """
    flat_kv = topk_indices_slice.reshape(-1)  # [T*R] int32; -1 = invalid
    if num_kv < 32767:  # int16 range, -1 included
        keys = flat_kv.to(torch.int16)
        ar = torch.arange(num_kv + 1, device=flat_kv.device, dtype=torch.int16)
    else:
        keys = flat_kv.to(torch.int32)
        ar = torch.arange(num_kv + 1, device=flat_kv.device, dtype=torch.int32)
    sorted_vals, inv_data = torch.sort(keys, stable=True)
    inv_ptr = torch.searchsorted(sorted_vals, ar).to(torch.int32)
    return inv_ptr, inv_data.to(torch.int32)


def dkv_gather_acc_be(
    interm, inv_ptr, inv_data, dkv_acc, BLOCK_E=64, num_warps=8, accumulate=True
):
    # BLOCK_E=64 / num_warps=8 measured best (0.345 ms, 6.29 TB/s = 79% peak at T4096 H128
    # topk512 SWA+pool). Time falls monotonically with BLOCK_E across the whole sweep
    # (4->64: 1.021, 0.715, 0.505, 0.407, 0.345), i.e. both the load width AND the trip count
    # on the ~3000-entry pool runs were binding. Reference was 1.491 ms at 1.45 TB/s.
    """interm[T,R,D] bf16 -> dkv_acc[num_kv,D] fp32 via the entry-blocked CSR gather.

    Grid is ``num_kv`` (from ``dkv_acc``), not ``T``, so a compressed-pool KV works.
    """
    _, _, D = interm.shape
    num_kv = dkv_acc.shape[0]
    _bwd_dkv_gather_acc_v4_be[(num_kv,)](
        interm,
        inv_ptr,
        inv_data,
        dkv_acc,
        interm.stride(1),
        dkv_acc.stride(0),
        D=D,
        BLOCK_E=BLOCK_E,
        ACCUMULATE=accumulate,
        num_warps=num_warps,
    )
