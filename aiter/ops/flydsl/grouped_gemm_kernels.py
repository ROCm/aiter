# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""High-level FlyDSL Grouped FP8 GEMM APIs.

Two variants matching DeepGEMM:
  - **contiguous**: A is [M_total, K] with a grouped_layout mapping rows to groups.
  - **masked**: A is [G, expected_m, K] with a masked_m array tracking valid tokens.
"""

from __future__ import annotations

from typing import Optional

import torch

from .kernels.group_gemm_blockscale_contiguous import compile_grouped_fp8_gemm
from .kernels.group_gemm_blockscale_masked import compile_masked_grouped_fp8_gemm

__all__ = [
    "flydsl_grouped_gemm_contiguous",
    "flydsl_grouped_gemm_masked",
]


# ────────────────────────────────────────────────────────────────────
# Validation helpers
# ────────────────────────────────────────────────────────────────────

def _validate_contiguous_inputs(
    a: torch.Tensor,
    b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    grouped_layout: torch.Tensor,
    out: Optional[torch.Tensor],
):
    """Validate inputs for contiguous grouped GEMM."""
    if a.dim() != 2:
        raise ValueError(f"`a` must be 2D [M, K], got {a.dim()}D")
    if b.dim() != 3:
        raise ValueError(f"`b` must be 3D [num_groups, N, K], got {b.dim()}D")
    if a.device.type != "cuda":
        raise ValueError("Only CUDA/ROCm tensors are supported")
    if a.device != b.device:
        raise ValueError(f"`a` and `b` must be on same device: {a.device} vs {b.device}")

    M, K = a.shape
    num_groups, N, bK = b.shape
    if K != bK:
        raise ValueError(f"K mismatch: a.shape[1]={K}, b.shape[2]={bK}")
    if grouped_layout.shape != (M,):
        raise ValueError(
            f"`grouped_layout` must have shape [{M}], got {list(grouped_layout.shape)}"
        )
    if out is not None:
        if out.shape != (M, N):
            raise ValueError(f"`out` must have shape [{M}, {N}], got {list(out.shape)}")
        if not out.is_contiguous():
            raise ValueError("`out` must be contiguous")

    return M, N, K, num_groups


def _validate_masked_inputs(
    a: torch.Tensor,
    b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    masked_m: torch.Tensor,
    out: Optional[torch.Tensor],
):
    """Validate inputs for masked grouped GEMM."""
    if a.dim() != 3:
        raise ValueError(f"`a` must be 3D [G, expected_m, K], got {a.dim()}D")
    if b.dim() != 3:
        raise ValueError(f"`b` must be 3D [G, N, K], got {b.dim()}D")
    if a.device.type != "cuda":
        raise ValueError("Only CUDA/ROCm tensors are supported")
    if a.device != b.device:
        raise ValueError(f"`a` and `b` must be on same device: {a.device} vs {b.device}")

    G, expected_m, K = a.shape
    bG, N, bK = b.shape
    if G != bG:
        raise ValueError(f"Group count mismatch: a.shape[0]={G}, b.shape[0]={bG}")
    if K != bK:
        raise ValueError(f"K mismatch: a.shape[2]={K}, b.shape[2]={bK}")
    if masked_m.shape != (G,):
        raise ValueError(f"`masked_m` must have shape [{G}], got {list(masked_m.shape)}")
    if out is not None:
        if out.shape != (G, expected_m, N):
            raise ValueError(
                f"`out` must have shape [{G}, {expected_m}, {N}], got {list(out.shape)}"
            )
        if not out.is_contiguous():
            raise ValueError("`out` must be contiguous")

    return G, expected_m, N, K


# ────────────────────────────────────────────────────────────────────
# Internal compile + launch wrappers
# ────────────────────────────────────────────────────────────────────

def _compile_contiguous(
    n: int,
    k: int,
    num_groups: int,
    *,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    scale_block_k: int,
    scale_block_n: int,
    out_dtype: str,
):
    """Compile contiguous grouped FP8 GEMM, return a launcher closure."""
    exe = compile_grouped_fp8_gemm(
        n=n,
        k=k,
        num_groups=num_groups,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        scale_block_k=scale_block_k,
        scale_block_n=scale_block_n,
        out_dtype=out_dtype,
    )

    def launcher(
        out: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        scale_a: torch.Tensor,
        scale_b: torch.Tensor,
        grouped_layout: torch.Tensor,
        m: int,
        stream=None,
    ):
        launch_stream = (
            torch.cuda.current_stream(device=a.device) if stream is None else stream
        )
        exe(
            out.contiguous().view(-1),
            a.contiguous().view(torch.int8).view(-1),
            b.contiguous().view(torch.int8).view(-1),
            scale_a.contiguous().view(-1),
            scale_b.contiguous().view(-1),
            grouped_layout.contiguous(),
            m, n, k, num_groups,
            launch_stream,
        )

    return launcher


def _compile_masked(
    n: int,
    k: int,
    num_groups: int,
    *,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    scale_block_k: int,
    scale_block_n: int,
    out_dtype: str,
):
    """Compile masked grouped FP8 GEMM, return a launcher closure."""
    exe = compile_masked_grouped_fp8_gemm(
        n=n,
        k=k,
        num_groups=num_groups,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        scale_block_k=scale_block_k,
        scale_block_n=scale_block_n,
        out_dtype=out_dtype,
    )

    def launcher(
        out: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        scale_a: torch.Tensor,
        scale_b: torch.Tensor,
        masked_m: torch.Tensor,
        expected_m: int,
        stream=None,
    ):
        launch_stream = (
            torch.cuda.current_stream(device=a.device) if stream is None else stream
        )
        exe(
            out.contiguous().view(-1),
            a.contiguous().view(torch.int8).view(-1),
            b.contiguous().view(torch.int8).view(-1),
            scale_a.contiguous().view(-1),
            scale_b.contiguous().view(-1),
            masked_m.contiguous(),
            expected_m, n, k, num_groups,
            launch_stream,
        )

    return launcher


# ────────────────────────────────────────────────────────────────────
# Public API
# ────────────────────────────────────────────────────────────────────

def flydsl_grouped_gemm_contiguous(
    a: torch.Tensor,
    b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    grouped_layout: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    *,
    tile_m: int = 128,
    tile_n: int = 128,
    tile_k: int = 128,
    scale_block_k: int = 128,
    scale_block_n: int = 128,
    out_dtype: str = "bf16",
) -> torch.Tensor:
    """Grouped FP8 GEMM with contiguous M layout.

    Matches DeepGEMM's ``m_grouped_fp8_gemm_nt_contiguous`` API.

    ``b`` **must** be pre-shuffled once at model-load time via
    ``shuffle_weight(b, layout=(16, 16))`` — the kernel expects the
    preshuffle memory layout.  Do **not** shuffle on every call; weight
    matrices are static and the shuffle only needs to happen once.

    Args:
        a: [M_total, K] FP8 activation tensor (concatenated rows from all groups).
        b: [num_groups, N, K] FP8 weight tensor, **pre-shuffled** via
           ``shuffle_weight(b, layout=(16, 16))``.
        scale_a: [scale_k, M_total] FP32 per-token, per-128K scales (transposed).
        scale_b: [num_groups, scale_n, scale_k] FP32 per-block scales.
        grouped_layout: [M_total] INT32 mapping each row to a group ID (-1 for padding).
        out: Optional pre-allocated [M_total, N] BF16/FP16 output tensor.
        tile_m, tile_n, tile_k: Tiling dimensions (default 128).
        scale_block_k, scale_block_n: Scale block sizes (default 128).
        out_dtype: Output data type, ``"bf16"`` or ``"f16"``.

    Returns:
        [M_total, N] output tensor.
    """
    M, N, K, num_groups = _validate_contiguous_inputs(
        a, b, scale_a, scale_b, grouped_layout, out,
    )

    if not a.is_contiguous():
        a = a.contiguous()
    if not b.is_contiguous():
        b = b.contiguous()

    torch_out_dtype = torch.bfloat16 if out_dtype == "bf16" else torch.float16
    if out is None:
        out = torch.empty((M, N), dtype=torch_out_dtype, device=a.device)

    launcher = _compile_contiguous(
        n=N, k=K, num_groups=num_groups,
        tile_m=tile_m, tile_n=tile_n, tile_k=tile_k,
        scale_block_k=scale_block_k, scale_block_n=scale_block_n,
        out_dtype=out_dtype,
    )
    launcher(out, a, b, scale_a, scale_b, grouped_layout, M)

    return out


def flydsl_grouped_gemm_masked(
    a: torch.Tensor,
    b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    masked_m: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    *,
    tile_m: int = 128,
    tile_n: int = 128,
    tile_k: int = 128,
    scale_block_k: int = 128,
    scale_block_n: int = 128,
    out_dtype: str = "bf16",
) -> torch.Tensor:
    """Masked Grouped FP8 GEMM with padded per-group layout.

    Matches DeepGEMM's ``m_grouped_fp8_gemm_nt_masked`` API.

    ``b`` **must** be pre-shuffled once at model-load time via
    ``shuffle_weight(b, layout=(16, 16))`` — the kernel expects the
    preshuffle memory layout.  Do **not** shuffle on every call; weight
    matrices are static and the shuffle only needs to happen once.

    Args:
        a: [G, expected_m, K] FP8 activation tensor (padded per group).
        b: [G, N, K] FP8 weight tensor, **pre-shuffled** via
           ``shuffle_weight(b, layout=(16, 16))``.
        scale_a: [G, scale_k, expected_m] FP32 per-token scales (transposed).
        scale_b: [G, scale_n, scale_k] FP32 per-block scales.
        masked_m: [G] INT32 actual valid token count per group.
        out: Optional pre-allocated [G, expected_m, N] BF16/FP16 output tensor.
        tile_m, tile_n, tile_k: Tiling dimensions (default 128).
        scale_block_k, scale_block_n: Scale block sizes (default 128).
        out_dtype: Output data type, ``"bf16"`` or ``"f16"``.

    Returns:
        [G, expected_m, N] output tensor (padded rows beyond masked_m are undefined).
    """
    G, expected_m, N, K = _validate_masked_inputs(
        a, b, scale_a, scale_b, masked_m, out,
    )

    if not a.is_contiguous():
        a = a.contiguous()
    if not b.is_contiguous():
        b = b.contiguous()

    torch_out_dtype = torch.bfloat16 if out_dtype == "bf16" else torch.float16
    if out is None:
        out = torch.empty((G, expected_m, N), dtype=torch_out_dtype, device=a.device)

    launcher = _compile_masked(
        n=N, k=K, num_groups=G,
        tile_m=tile_m, tile_n=tile_n, tile_k=tile_k,
        scale_block_k=scale_block_k, scale_block_n=scale_block_n,
        out_dtype=out_dtype,
    )
    launcher(out, a, b, scale_a, scale_b, masked_m, expected_m)

    return out
