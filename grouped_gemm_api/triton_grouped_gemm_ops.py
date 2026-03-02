# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""
Triton-based Grouped GEMM API for training framework.

Provides fprop / dgrad / wgrad as separate functions built on top of
AITER's Triton GMM kernels.

Tensor conventions:
    G  – number of expert groups
    GM – total tokens (sum of split_sizes)
    K  – input dimension
    N  – output dimension
    All tensors are contiguous.

Kernel mapping:
    fprop  uses gmm:    (GM, K) x (G, K, N) -> (GM, N)
                        w[G,N,K] is passed column-major so TRANS_RHS=True,
                        which fuses the per-group transposition w[g]^T.
    dgrad  uses gmm:    (GM, N) x (G, N, K) -> (GM, K)
                        w[G,N,K] is treated as (G, K_in=N, N_out=K) row-major,
                        no transposition needed.
    wgrad  uses nptgmm: (N, GM) x (GM, K) -> (G, N, K)
                        dy is reinterpreted as column-major (N, GM), i.e.
                        TRANS_LHS=True fuses dy[g]^T in the kernel.
"""

from typing import Optional

import torch

from aiter.ops.triton.gmm import gmm, nptgmm


def _group_sizes_int32(split_sizes: torch.Tensor) -> torch.Tensor:
    """Return split_sizes as a contiguous int32 tensor on the same device."""
    return split_sizes.to(dtype=torch.int32, device=split_sizes.device).contiguous()


def grouped_gemm_fprop(
    x: torch.Tensor,            # [GM, K]
    w: torch.Tensor,            # [G, N, K]
    split_sizes: torch.Tensor,  # [G], int32 or int64
) -> torch.Tensor:              # [GM, N]
    """Forward pass:  out[g] = x[g] @ w[g].T  for each group g.

    w has shape [G, N, K].  Passing it directly to gmm as column-major
    [G, K, N] (stride (N*K, 1, N)) sets TRANS_RHS=True in the kernel,
    which fuses the transposition w[g]^T without a copy.
    """
    group_sizes = _group_sizes_int32(split_sizes)

    # w is [G, N, K]; reinterpret as [G, K, N] column-major by transposing dims 1 and 2.
    # gmm expects rhs shape [G, K, N]; when col-major it sets TRANS_RHS=True.
    rhs = w.transpose(1, 2)  # logical [G, K, N], stride (N*K, 1, N) — col-major

    return gmm(
        lhs=x,
        rhs=rhs,
        group_sizes=group_sizes,
        preferred_element_type=x.dtype,
    )


def grouped_gemm_dgrad(
    dy: torch.Tensor,           # [GM, N]
    w: torch.Tensor,            # [G, N, K]
    split_sizes: torch.Tensor,  # [G]
) -> torch.Tensor:              # [GM, K]
    """Data gradient:  dx[g] = dy[g] @ w[g]  for each group g.

    w has shape [G, N, K] and is already row-major [G, N, K].
    gmm treats rhs as [G, K_in, N_out]; here K_in=N (weight rows) and
    N_out=K (weight cols), so we pass w directly with TRANS_RHS=False.
    """
    group_sizes = _group_sizes_int32(split_sizes)

    # w is [G, N, K], row-major. gmm sees it as (G, K_inner=N_w, N_out=K_w).
    # The contiguous check: w.stride() must be (N*K, K, 1).
    assert w.is_contiguous(), "w must be contiguous for dgrad"

    return gmm(
        lhs=dy,
        rhs=w,
        group_sizes=group_sizes,
        preferred_element_type=dy.dtype,
    )


def grouped_gemm_wgrad(
    dy: torch.Tensor,                         # [GM, N]
    x: torch.Tensor,                          # [GM, K]
    split_sizes: torch.Tensor,                # [G]
    wgrad: Optional[torch.Tensor] = None,     # [G, N, K]
    output_accum: bool = False,
) -> torch.Tensor:                            # [G, N, K]
    """Weight gradient:  dw[g] = dy[g].T @ x[g]  for each group g.

    nptgmm computes  out[g] = lhs[:, group_g] @ rhs[group_g, :]
    where lhs is [K_out, GM] and rhs is [GM, N_out], producing [G, K_out, N_out].

    We want dw[g] = dy[g].T @ x[g], i.e. [N, gm] x [gm, K] -> [G, N, K].
    So  lhs = dy.T  (shape [N, GM], col-major view of dy)
        rhs = x     (shape [GM, K], row-major)
    and the output is [G, N, K].

    Passing dy.T directly sets lhs stride to (1, N), which is col-major,
    so the kernel uses TRANS_LHS=True and fuses the transposition.

    When output_accum is True and wgrad is not None, the fresh gradient is
    added into wgrad in-place and wgrad is returned.
    """
    group_sizes = _group_sizes_int32(split_sizes)

    # dy.T has shape [N, GM] with stride (1, N) — column-major.
    # nptgmm accepts col-major lhs and sets TRANS_LHS=True.
    lhs = dy.T  # [N, GM], no data copy

    existing_out = None
    if output_accum and wgrad is not None:
        existing_out = wgrad

    new_wgrad = nptgmm(
        lhs=lhs,
        rhs=x,
        group_sizes=group_sizes,
        preferred_element_type=dy.dtype,
        existing_out=existing_out,
        accumulate=output_accum and wgrad is not None,
    )

    if output_accum and wgrad is not None:
        return wgrad
    return new_wgrad
