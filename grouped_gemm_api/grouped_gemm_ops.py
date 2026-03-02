"""
Grouped GEMM API for training framework.

Provides fprop / dgrad / wgrad as separate functions built on top of
Primus-Turbo's grouped GEMM kernels (CK + hipBLASLt backends).

Tensor conventions:
    G  – number of expert groups
    GM – total tokens (sum of split_sizes)
    K  – input dimension
    N  – output dimension
    All tensors are contiguous.
"""

from typing import Optional

import torch

from primus_turbo.pytorch.core.backend import BackendType
from primus_turbo.pytorch.kernels.grouped_gemm.grouped_gemm_csrc_impl import (
    grouped_gemm_impl,
    grouped_gemm_variable_k_impl,
)
from primus_turbo.pytorch.kernels.grouped_gemm.grouped_gemm_fp8_impl import (
    grouped_gemm_compute_offs,
)


def _prepare(split_sizes: torch.Tensor):
    """Normalize split_sizes to int64 and compute prefix-sum offsets."""
    group_lens = split_sizes.to(dtype=torch.int64, device=split_sizes.device)
    group_offs = grouped_gemm_compute_offs(group_lens)
    return group_lens, group_offs


def grouped_gemm_fprop(
    x: torch.Tensor,            # [GM, K]
    w: torch.Tensor,            # [G, N, K]
    split_sizes: torch.Tensor,  # [G], int32 or int64
) -> torch.Tensor:              # [GM, N]
    """Forward pass:  out[g] = x[g] @ w[g]^T  for each group g."""
    group_lens, group_offs = _prepare(split_sizes)
    return grouped_gemm_impl(
        a=x,
        b=w,
        group_lens=group_lens,
        group_offs=group_offs,
        trans_a=False,
        trans_b=True,
        num_cu=None,
        default_backend=BackendType.CK.value,
        maybe_pre_sync=True,
    )


def grouped_gemm_dgrad(
    dy: torch.Tensor,           # [GM, N]
    w: torch.Tensor,            # [G, N, K]
    split_sizes: torch.Tensor,  # [G]
) -> torch.Tensor:              # [GM, K]
    """Data gradient:  dx[g] = dy[g] @ w[g]  for each group g."""
    group_lens, group_offs = _prepare(split_sizes)
    return grouped_gemm_impl(
        a=dy,
        b=w,
        group_lens=group_lens,
        group_offs=group_offs,
        trans_a=False,
        trans_b=False,
        num_cu=None,
        default_backend=BackendType.CK.value,
    )


def grouped_gemm_wgrad(
    dy: torch.Tensor,                         # [GM, N]
    x: torch.Tensor,                          # [GM, K]
    split_sizes: torch.Tensor,                # [G]
    wgrad: Optional[torch.Tensor] = None,     # [G, N, K]
    output_accum: bool = False,
) -> torch.Tensor:                            # [G, N, K]
    """Weight gradient:  dw[g] = dy[g]^T @ x[g]  for each group g.

    When *output_accum* is True and *wgrad* is not None the freshly-computed
    gradient is **added** to *wgrad* in-place and *wgrad* is returned.
    """
    group_lens, group_offs = _prepare(split_sizes)
    new_wgrad = grouped_gemm_variable_k_impl(
        a=x,
        b=dy,
        group_lens=group_lens,
        group_offs=group_offs,
        trans_a=True,
        trans_b=False,
        trans_c=True,
        num_cu=None,
        default_backend=BackendType.CK.value,
    )
    if output_accum and wgrad is not None:
        wgrad.add_(new_wgrad)
        return wgrad
    return new_wgrad
