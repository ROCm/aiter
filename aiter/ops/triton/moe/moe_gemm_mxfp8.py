# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Fused MXFP8 grouped (MoE) GEMM.

Performs all expert GEMMs in a single Triton kernel launch using MXFP8
quantisation.  E8M0 microscales (one uint8 per ``quant_block_size`` elements
along K) are converted to FP32 power-of-two factors inside the kernel.

Convention (TN layout):
    ``out[tokens_for_e] = lhs[tokens_for_e] @ rhs[e]^T``
where both lhs and rhs are stored in FP8 with per-block E8M0 scales.
"""

import torch
import triton

from aiter.ops.triton._triton_kernels.moe.moe_gemm_mxfp8 import (
    _moe_gemm_mxfp8_kernel,
)
from aiter.ops.triton.utils.logger import AiterTritonLogger

_LOGGER = AiterTritonLogger()

BLOCK_M = 64
BLOCK_N = 128


def _build_block_mapping(
    group_sizes: torch.Tensor,
    block_m: int,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Pre-compute per-M-block expert IDs and token offsets."""
    expert_ids_list = []
    offsets_list = []
    offset = 0
    for e in range(group_sizes.shape[0]):
        size = int(group_sizes[e].item())
        n_blocks = triton.cdiv(size, block_m)
        for b in range(n_blocks):
            expert_ids_list.append(e)
            offsets_list.append(offset + b * block_m)
        offset += size

    if not expert_ids_list:
        return (
            torch.empty(0, dtype=torch.int32, device=group_sizes.device),
            torch.empty(0, dtype=torch.int32, device=group_sizes.device),
            0,
        )

    block_expert_ids = torch.tensor(
        expert_ids_list,
        dtype=torch.int32,
        device=group_sizes.device,
    )
    block_token_offsets = torch.tensor(
        offsets_list,
        dtype=torch.int32,
        device=group_sizes.device,
    )
    return block_expert_ids, block_token_offsets, len(expert_ids_list)


def moe_gemm_mxfp8(
    lhs: torch.Tensor,
    rhs: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    group_sizes: torch.Tensor,
    quant_block_size: int = 32,
    bias: torch.Tensor | None = None,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Fused MXFP8 grouped GEMM for MoE.

    Args:
        lhs: FP8 activation ``[total_tokens, K]``.
        rhs: FP8 weight ``[E, N, K]``.
        x_scale: E8M0 uint8 activation scales ``[total_tokens, K // qbs]``.
        w_scale: E8M0 uint8 weight scales ``[E, N, K // qbs]``.
        group_sizes: Expert token counts ``[E]`` (int).
        quant_block_size: MXFP block size (default 32).
        bias: Optional per-expert bias ``[E, N]``.
        out_dtype: Output dtype (default BF16).

    Returns:
        Output tensor ``[total_tokens, N]``.
    """
    _LOGGER.info(
        f"MOE_GEMM_MXFP8: lhs={tuple(lhs.shape)} rhs={tuple(rhs.shape)} "
        f"x_scale={tuple(x_scale.shape)} w_scale={tuple(w_scale.shape)} "
        f"block_size={quant_block_size}"
    )

    total_tokens = lhs.shape[0]
    rhs.shape[0]
    N = rhs.shape[1]
    K = rhs.shape[2]

    assert lhs.shape[1] == K, "K dimension mismatch"

    BLOCK_K = quant_block_size
    out = torch.empty(total_tokens, N, dtype=out_dtype, device=lhs.device)

    if total_tokens == 0:
        return out

    block_expert_ids, block_token_offsets, total_m_blocks = _build_block_mapping(
        group_sizes,
        BLOCK_M,
    )

    if total_m_blocks == 0:
        return out

    num_n_blocks = triton.cdiv(N, BLOCK_N)
    grid = (total_m_blocks * num_n_blocks,)

    _moe_gemm_mxfp8_kernel[grid](
        lhs,
        rhs,
        out,
        x_scale,
        w_scale,
        bias,
        block_expert_ids,
        block_token_offsets,
        total_tokens,
        N,
        K,
        lhs.stride(0),
        lhs.stride(1),
        rhs.stride(0),
        rhs.stride(1),
        rhs.stride(2),
        out.stride(0),
        out.stride(1),
        x_scale.stride(0),
        x_scale.stride(1),
        w_scale.stride(0),
        w_scale.stride(1),
        w_scale.stride(2),
        HAS_BIAS=bias is not None,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        QUANT_BLOCK_SIZE=quant_block_size,
    )
    return out
