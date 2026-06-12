# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#
# Block-level mean-pooling Triton kernel.
#
# ``triton_bmm_pool_sim_simmean`` performs per-block mean pooling plus an
# optional intra-block self-similarity test (the "meansim" predicate that
# decides whether a block is internally coherent enough to be summarized by its
# mean). It is the cheap O(nblock^2) block-attention proxy used by the
# SpargeAttn / VFA block-sparse mask construction in
# ``aiter/ops/triton/pool.py`` and ``aiter/ops/triton/attention/fav3_sage.py``.
#
# Reference: "SpargeAttn: Accurate Sparse Attention Accelerating Any Model
# Inference" (https://arxiv.org/abs/2502.18137).

import triton
import triton.language as tl


@triton.jit
def triton_bmm_pool_sim_simmean(
    x_ptr,
    pool_ptr,
    sim_ptr,
    simthreshd1_ptr,
    N: tl.constexpr,
    D: tl.constexpr,
    BS: tl.constexpr,
    SKIP_SIM: tl.constexpr = False,
):
    b, h, nb = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    B, H, NB = tl.num_programs(0), tl.num_programs(1), tl.num_programs(2)

    block_offset = b * H * N * D + h * N * D + nb * BS * D
    xmask = (nb * BS + tl.arange(0, BS)[:, None]) < N
    x_ptrs = x_ptr + block_offset + tl.arange(0, BS)[:, None] * D + tl.arange(0, D)[None, :]
    # Load the input block, xmask will return nan for out-of-bound elements
    x = tl.load(x_ptrs, mask=xmask)
    BS_ = BS if (N - nb * BS) >= BS else (N - nb * BS)

    x_fp32 = x.to(tl.float32)
    # Check for NaN values
    is_nan = x_fp32 != x_fp32
    x_fp32 = tl.where(is_nan, 0.0, x_fp32)

    pool = tl.sum(x_fp32, axis=0) / BS_
    pool_block_offset = b * H * NB * D + h * NB * D + nb * D
    tl.store(pool_ptr + pool_block_offset + tl.arange(0, D), pool)

    if not SKIP_SIM:
        cur_h1 = tl.load(simthreshd1_ptr + h)
        x_norm = tl.sqrt(tl.sum(x_fp32 * x_fp32, axis=1, keep_dims=True))
        x = (x / x_norm).to(tl.float16)  # norm at D dim
        # Check for NaN values after normalization
        is_nan = x != x
        x = tl.where(is_nan, 0.0, x)

        grams = tl.dot(x, tl.trans(x))
        sum_value = tl.sum(grams).to(tl.float32)
        cur_sim = (sum_value / (BS_ * BS_)) > cur_h1

        sim_offset = b * H * NB + h * NB + nb
        tl.store(sim_ptr + sim_offset, cur_sim)
