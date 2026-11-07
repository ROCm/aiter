# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import triton
import triton.language as tl


@triton.jit
def _split_chunk(
    kv_len, NSPLIT: tl.constexpr, BLOCK_N: tl.constexpr, MIN_CHUNK: tl.constexpr
):
    per = tl.cdiv(kv_len, NSPLIT)
    per = tl.cdiv(per, BLOCK_N) * BLOCK_N
    return tl.maximum(per, MIN_CHUNK)


@triton.jit
def _mla_verify_reduce_kernel(
    KV_INDPTR,
    O_PART,
    LSE_PART,
    O,
    stride_o_tok,
    stride_o_h,
    H: tl.constexpr,
    QLEN: tl.constexpr,
    D_V: tl.constexpr,
    NSPLIT: tl.constexpr,
    NSPLIT_P2: tl.constexpr,
    BLOCK_N: tl.constexpr,
    MIN_CHUNK: tl.constexpr,
    S_BLK: tl.constexpr,
    D_BLK: tl.constexpr,
):
    """Base-2 LSE merge of hk_mla_verify_fwd splits.

    One program per (request, row = q_pos * H + h, D_BLK columns). Uses the same
    split formula as the HIP kernel, so only the active splits are read.
    """
    ROWS: tl.constexpr = H * QLEN
    b = tl.program_id(0)
    r = tl.program_id(1)
    offs_d = tl.program_id(2) * D_BLK + tl.arange(0, D_BLK)
    o_off = (b * QLEN + r // H).to(tl.int64) * stride_o_tok + (r % H) * stride_o_h
    kv_len = tl.load(KV_INDPTR + b + 1) - tl.load(KV_INDPTR + b)
    chunk = _split_chunk(kv_len, NSPLIT, BLOCK_N, MIN_CHUNK)
    nact = tl.minimum(tl.cdiv(kv_len, chunk), NSPLIT)
    base = (b * NSPLIT) * ROWS + r
    offs_all = tl.arange(0, NSPLIT_P2)
    lse_all = tl.load(
        LSE_PART + base + offs_all * ROWS, mask=offs_all < nact, other=float("-inf")
    )
    m = tl.max(lse_all, axis=0)
    m = tl.where(m == float("-inf"), 0.0, m)
    wsum = tl.sum(tl.where(offs_all < nact, tl.math.exp2(lse_all - m), 0.0), axis=0)
    offs_s = tl.arange(0, S_BLK)
    acc = tl.zeros([D_BLK], dtype=tl.float32)
    for s0 in range(0, nact, S_BLK):
        sid = s0 + offs_s
        sm = sid < nact
        lse = tl.load(LSE_PART + base + sid * ROWS, mask=sm, other=float("-inf"))
        w = tl.where(sm, tl.math.exp2(lse - m), 0.0)
        po = tl.load(
            O_PART + (base + sid * ROWS)[:, None].to(tl.int64) * D_V + offs_d[None, :],
            mask=sm[:, None],
            other=0.0,
        ).to(tl.float32)
        acc += tl.sum(w[:, None] * po, axis=0)
    o = acc / tl.where(wsum > 0.0, wsum, 1.0)
    tl.store(O + o_off + offs_d, o.to(O.dtype.element_ty))
