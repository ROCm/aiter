# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import os

import torch
import triton

from aiter.ops.triton._triton_kernels.moe.sonicmoe.grouped_gemm import (
    _grouped_gemm_dw_kernel,
    _grouped_gemm_kernel,
)
from aiter.ops.triton.utils.sonicmoe_config_utils import (
    get_grouped_gemm_dw_config,
    get_grouped_gemm_fwd_config,
    split_launch_config,
)


def _local_tensor(tensor: torch.Tensor | None) -> torch.Tensor | None:
    if tensor is not None and hasattr(tensor, "to_local"):
        return tensor.to_local()
    return tensor


def _use_qwen3_tuned_configs() -> bool:
    return os.environ.get("SONIC_MOE_USE_QWEN3_TUNED_GEMM", "0") == "1"


def grouped_gemm(
    A: torch.Tensor,
    B: torch.Tensor,
    cu_seqlens: torch.Tensor,
    out: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    A_idx: torch.Tensor | None = None,
    scatter_idx: torch.Tensor | None = None,
    A_is_transposed: bool = False,
    B_is_transposed: bool = False,
    A_scale: torch.Tensor | None = None,
    B_scale: torch.Tensor | None = None,
    block_size: int = 128,
    out_dtype: torch.dtype | None = None,
):
    """Run grouped GEMM, optionally with 1x128 activation and 128x128 weight scales."""
    if (A_scale is None) != (B_scale is None):
        raise ValueError("A_scale and B_scale must be provided together")
    if A_scale is not None and block_size != 128:
        raise ValueError("Sonic blockwise FP8 requires block_size=128")
    if A_is_transposed:
        if B_is_transposed:
            raise ValueError("a grouped wgrad does not support a transposed B")
        if bias is not None:
            raise ValueError("bias is invalid for a grouped wgrad")
        if scatter_idx is not None:
            raise ValueError("scatter_idx is invalid for a grouped wgrad")

    local_out = _local_tensor(out)
    local_b = _local_tensor(B)
    triton_b = local_b.transpose(1, 2) if B_is_transposed else local_b
    local_b_scale = _local_tensor(B_scale)
    triton_b_scale = (
        local_b_scale.transpose(1, 2)
        if B_is_transposed and local_b_scale is not None
        else local_b_scale
    )
    result = _grouped_gemm_triton(
        _local_tensor(A),
        triton_b,
        _local_tensor(cu_seqlens),
        local_out,
        _local_tensor(bias),
        _local_tensor(A_idx),
        _local_tensor(scatter_idx),
        A_is_transposed,
        _local_tensor(A_scale),
        triton_b_scale,
        block_size,
        out_dtype,
    )
    return out if out is not None else result


def _grouped_gemm_triton(
    A: torch.Tensor,
    B: torch.Tensor,
    cu_seqlens: torch.Tensor,
    out: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    A_idx: torch.Tensor | None = None,
    scatter_idx: torch.Tensor | None = None,
    A_is_transposed: bool = False,
    A_scale: torch.Tensor | None = None,
    B_scale: torch.Tensor | None = None,
    block_size: int = 128,
    out_dtype: torch.dtype | None = None,
):
    if A_is_transposed and B.dim() == 2:
        return _grouped_gemm_dw(
            A, B, cu_seqlens, out, A_idx, A_scale, B_scale, block_size, out_dtype
        )

    E = B.shape[0]
    K_dim = B.shape[1]
    N = B.shape[2]

    TK = A.shape[0] if A_idx is None else A_idx.numel()

    if out is None:
        out = torch.empty(
            TK,
            N,
            dtype=out_dtype if out_dtype is not None else A.dtype,
            device=A.device,
        )

    blockwise_fp8 = A_scale is not None
    if blockwise_fp8:
        if B_scale is None:
            raise ValueError("B_scale is required when A_scale is provided")
        expected_a_scale = (A.shape[0], triton.cdiv(K_dim, block_size))
        expected_b_scale = (
            E,
            triton.cdiv(K_dim, block_size),
            triton.cdiv(N, block_size),
        )
        if tuple(A_scale.shape) != expected_a_scale:
            raise ValueError(
                f"A_scale must have shape {expected_a_scale}, got {tuple(A_scale.shape)}"
            )
        if tuple(B_scale.shape) != expected_b_scale:
            raise ValueError(
                f"B_scale must have shape {expected_b_scale}, got {tuple(B_scale.shape)}"
            )

    def grid(META):
        max_m_blocks = triton.cdiv(TK, META["BLOCK_M"]) + E - 1
        return (max_m_blocks * triton.cdiv(N, META["BLOCK_N"]),)

    launch_args = (
        A,
        B,
        A_scale if A_scale is not None else A,
        B_scale if B_scale is not None else B,
        out,
        cu_seqlens,
        bias if bias is not None else A,
        A_idx if A_idx is not None else cu_seqlens,
        scatter_idx if scatter_idx is not None else cu_seqlens,
        A.stride(0),
        A.stride(1),
        B.stride(0),
        B.stride(1),
        B.stride(2),
        A_scale.stride(0) if A_scale is not None else 0,
        A_scale.stride(1) if A_scale is not None else 0,
        B_scale.stride(0) if B_scale is not None else 0,
        B_scale.stride(1) if B_scale is not None else 0,
        B_scale.stride(2) if B_scale is not None else 0,
        out.stride(0),
        out.stride(1),
        bias.stride(0) if bias is not None else 0,
        bias.stride(1) if bias is not None else 0,
    )
    launch_meta = {
        "N": N,
        "K": K_dim,
        "E": E,
        "SCALE_BLOCK_SIZE": block_size,
        "BLOCKWISE_FP8": blockwise_fp8,
        "HAS_BIAS": (bias is not None),
        "HAS_GATHER_IDX": (A_idx is not None),
        "HAS_SCATTER_IDX": (scatter_idx is not None),
    }
    fwd_cfg = get_grouped_gemm_fwd_config(
        N, K_dim, E, A_idx is not None, _use_qwen3_tuned_configs()
    )
    constexprs, launch = split_launch_config(fwd_cfg)
    _grouped_gemm_kernel[grid](*launch_args, **launch_meta, **constexprs, **launch)
    return out


def _grouped_gemm_dw(
    A: torch.Tensor,
    B: torch.Tensor,
    cu_seqlens: torch.Tensor,
    out: torch.Tensor | None,
    A_idx: torch.Tensor | None,
    A_scale: torch.Tensor | None = None,
    B_scale: torch.Tensor | None = None,
    block_size: int = 128,
    out_dtype: torch.dtype | None = None,
):
    K_dim = A.shape[1]
    N = B.shape[1]
    E = cu_seqlens.shape[0] - 1

    if out is None:
        out = torch.empty(
            E,
            K_dim,
            N,
            dtype=out_dtype if out_dtype is not None else A.dtype,
            device=A.device,
        )

    blockwise_fp8 = A_scale is not None
    if blockwise_fp8:
        if B_scale is None:
            raise ValueError("B_scale is required when A_scale is provided")
        expert_rows = cu_seqlens[1:] - cu_seqlens[:-1]
        expected_scale_rows = (
            torch.div(
                expert_rows + block_size - 1,
                block_size,
                rounding_mode="floor",
            )
            .sum()
            .item()
        )
        if (
            A_scale.dim() != 2
            or A_scale.shape[0] != expected_scale_rows
            or A_scale.shape[1] != K_dim
        ):
            raise ValueError(
                f"A_scale must have shape [{expected_scale_rows}, {K_dim}], "
                f"got {tuple(A_scale.shape)}"
            )
        if (
            B_scale.dim() != 2
            or B_scale.shape[0] != expected_scale_rows
            or B_scale.shape[1] != N
        ):
            raise ValueError(
                f"B_scale must have shape [{expected_scale_rows}, {N}], "
                f"got {tuple(B_scale.shape)}"
            )
        if A_idx is not None:
            raise ValueError("blockwise FP8 grouped wgrad does not support A_idx")

    def grid(META):
        num_k_blocks = triton.cdiv(K_dim, META["BLOCK_K"])
        num_n_blocks = triton.cdiv(N, META["BLOCK_N"])
        return (E * num_k_blocks * num_n_blocks,)

    launch_args = (
        A,
        B,
        A_scale if A_scale is not None else A,
        B_scale if B_scale is not None else B,
        out,
        cu_seqlens,
        A_idx if A_idx is not None else cu_seqlens,
        A.stride(0),
        A.stride(1),
        B.stride(0),
        B.stride(1),
        A_scale.stride(0) if A_scale is not None else 0,
        A_scale.stride(1) if A_scale is not None else 0,
        B_scale.stride(0) if B_scale is not None else 0,
        B_scale.stride(1) if B_scale is not None else 0,
        out.stride(0),
        out.stride(1),
        out.stride(2),
    )
    launch_meta = {
        "N": N,
        "K": K_dim,
        "E": E,
        "SCALE_BLOCK_SIZE": block_size,
        "BLOCKWISE_FP8": blockwise_fp8,
        "HAS_GATHER_IDX": A_idx is not None,
    }
    dw_cfg = get_grouped_gemm_dw_config(
        N, K_dim, E, A_idx is not None, _use_qwen3_tuned_configs()
    )
    constexprs, launch = split_launch_config(dw_cfg)
    _grouped_gemm_dw_kernel[grid](*launch_args, **launch_meta, **constexprs, **launch)
    return out
