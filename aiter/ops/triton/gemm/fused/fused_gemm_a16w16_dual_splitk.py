# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import torch
import triton

from aiter.ops.triton._triton_kernels.gemm.fused.fused_gemm_a16w16_dual_splitk import (
    _fused_gemm_a16w16_dual_splitk_kernel,
    _fused_gemm_a16w16_dual_splitk_reduce_kernel,
    _get_config,
)
from aiter.ops.triton.utils.gemm_config_utils import compute_splitk_params
from aiter.ops.triton.utils.logger import AiterTritonLogger

_LOGGER = AiterTritonLogger()


def _part_splitk_params(config: dict, part: int, K: int, exact: bool = False):
    """compute_splitk_params on one part's BLOCK_SIZE_K / NUM_KSPLIT keys; with
    exact, NUM_KSPLIT is kept as given (trailing splits past K are zero)."""
    bk, ks = config[f"BLOCK_SIZE_K{part}"], config[f"NUM_KSPLIT{part}"]
    if exact:
        per_split = triton.cdiv(K, ks)
        while bk > 16 and bk > per_split:
            bk //= 2
        config[f"BLOCK_SIZE_K{part}"] = bk
        config[f"SPLITK_BLOCK_SIZE{part}"] = triton.cdiv(per_split, bk) * bk
        return
    sub = compute_splitk_params({"BLOCK_SIZE_K": bk, "NUM_KSPLIT": ks}, K)
    config[f"BLOCK_SIZE_K{part}"] = sub["BLOCK_SIZE_K"]
    config[f"NUM_KSPLIT{part}"] = sub["NUM_KSPLIT"]
    config[f"SPLITK_BLOCK_SIZE{part}"] = sub["SPLITK_BLOCK_SIZE"]


def fused_gemm_a16w16_dual_splitk(
    x: torch.Tensor,
    w: torch.Tensor,
    n2: int,
    dtype: torch.dtype | None = torch.bfloat16,
    y: torch.Tensor | None = None,
    skip_reduce: bool | None = False,
    config: dict | None = None,
    num_ksplit2: int | None = None,
):
    """
    Computes the 16 bit matmul Y = X x WT for W = [W1; W2] in one launch, with a
    separate tile size and split-K for each part: W1 is the first N1 = N - n2 rows,
    W2 the last n2 rows. Aimed at a wide output part fused with a skinny one, where
    only the skinny part needs split-K to fill the GPU.

    Key parameters:
    - x: Matrix X with shape (M, K).
    - w: Matrix W with shape (N1 + n2, K).
    - n2: Rows of W that form the second part.
    - skip_reduce: Return the second part's split-K partials instead of reducing
      them (requires NUM_KSPLIT1 == 1).
    - num_ksplit2: Overrides the config's NUM_KSPLIT2 and is kept exactly (splits
      past K hold zeros), e.g. for a consumer that sums a fixed number of partials.

    Returns:
    - y: The output matrix with shape (M, N1 + n2), or with skip_reduce and
      NUM_KSPLIT2 > 1, (y, y2_pp): y holds the first part in its first N1 columns
      and y2_pp the second part's fp32 partials, shape (NUM_KSPLIT2, M, n2).
    """
    _LOGGER.info(
        "FUSED_GEMM_A16W16_DUAL_SPLITK: x=%s w=%s n2=%d",
        tuple(x.shape),
        tuple(w.shape),
        n2,
    )

    M, K = x.shape
    N, _ = w.shape
    N1 = N - n2
    assert 0 < n2 < N, "n2 must leave both parts non-empty"
    w = w.T

    if config is None:
        config, _ = _get_config(M, N1, n2, K)
    if num_ksplit2 is not None:
        config["NUM_KSPLIT2"] = num_ksplit2
    _part_splitk_params(config, 1, K)
    _part_splitk_params(config, 2, K, exact=num_ksplit2 is not None)
    num_ksplit1, num_ksplit2 = config["NUM_KSPLIT1"], config["NUM_KSPLIT2"]
    assert not skip_reduce or num_ksplit1 == 1, "skip_reduce needs NUM_KSPLIT1 == 1"

    grid_m = triton.cdiv(M, config["BLOCK_SIZE_M"])
    grid1 = grid_m * triton.cdiv(N1, config["BLOCK_SIZE_N1"]) * num_ksplit1
    grid2 = grid_m * triton.cdiv(n2, config["BLOCK_SIZE_N2"]) * num_ksplit2

    if y is None:
        y = torch.empty((M, N), dtype=dtype, device=x.device)
    y1, y2 = y[:, :N1], y[:, N1:]
    y1_pp = (
        torch.empty((num_ksplit1, M, N1), dtype=torch.float32, device=x.device)
        if num_ksplit1 > 1
        else None
    )
    y2_pp = (
        torch.empty((num_ksplit2, M, n2), dtype=torch.float32, device=x.device)
        if num_ksplit2 > 1
        else None
    )
    c1 = y1 if y1_pp is None else y1_pp
    c2 = y2 if y2_pp is None else y2_pp

    grid = (max(grid1, grid2),)
    _fused_gemm_a16w16_dual_splitk_kernel[grid](
        x,
        w,
        c1,
        c2,
        M,
        N1,
        n2,
        K,
        x.stride(0),
        x.stride(1),
        w.stride(0),
        w.stride(1),
        0 if y1_pp is None else y1_pp.stride(0),
        c1.stride(-2),
        c1.stride(-1),
        0 if y2_pp is None else y2_pp.stride(0),
        c2.stride(-2),
        c2.stride(-1),
        **config,
    )

    if num_ksplit1 == 1 and num_ksplit2 == 1:
        return y
    if skip_reduce:
        return y, y2_pp

    REDUCE_BLOCK_SIZE_M = 32
    REDUCE_BLOCK_SIZE_N = 32
    grid_reduce = (
        triton.cdiv(M, REDUCE_BLOCK_SIZE_M),
        (triton.cdiv(N1, REDUCE_BLOCK_SIZE_N) if num_ksplit1 > 1 else 0)
        + (triton.cdiv(n2, REDUCE_BLOCK_SIZE_N) if num_ksplit2 > 1 else 0),
    )
    c1_in = c1 if y1_pp is not None else c2
    c2_in = c2 if y2_pp is not None else c1
    _fused_gemm_a16w16_dual_splitk_reduce_kernel[grid_reduce](
        c1_in,
        y1,
        c2_in,
        y2,
        M,
        N1,
        n2,
        c1_in.stride(0),
        c1_in.stride(1),
        c1_in.stride(2),
        y1.stride(0),
        y1.stride(1),
        c2_in.stride(0),
        c2_in.stride(1),
        c2_in.stride(2),
        y2.stride(0),
        y2.stride(1),
        REDUCE_BLOCK_SIZE_M,
        REDUCE_BLOCK_SIZE_N,
        num_ksplit1,
        triton.next_power_of_2(num_ksplit1),
        num_ksplit2,
        triton.next_power_of_2(num_ksplit2),
    )
    return y
