# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import torch
import triton
import triton.language as tl
from typing import Any, Dict
from aiter.ops.triton.utils.logger import AiterTritonLogger
from aiter.ops.triton._triton_kernels.moe_op_mxfp4_silu_fused import (
    _fused_moe_kernel_mxfp4_silu,
)

_LOGGER = AiterTritonLogger()


def fused_moe_mxfp4_silu(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    A_scale: torch.Tensor,
    B_scale: torch.Tensor,
    A_mx_scale: torch.Tensor,
    B_mx_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    mul_routed_weight: bool,
    top_k: int,
    swizzle_mx_a: bool,
    swizzle_mx_b: bool,
    config: Dict[str, Any],
    compute_type: tl.dtype,
) -> None:
    """
    #TODO: Add doc
    """
    _LOGGER.info(
        f"MOE_OP_MXFP4:  A={tuple(A.shape)}  B={tuple(B.shape)}  C={tuple(C.shape)} "
        + "A_scale={tuple(A_scale.shape)}  B_scale={tuple(B_scale.shape)} "
        + "A_mx_scale={tuple(A_mx_scale.shape)}  B_mx_scale={tuple(B_mx_scale.shape)} "
        + "topk_weights={tuple(topk_weights.shape)} sorted_token_ids={tuple(sorted_token_ids.shape)} "
        + "expert_ids={tuple(expert_ids.shape)} num_tokens_post_padded={tuple(num_tokens_post_padded.shape)} "
        + "top_k={top_k}"
    )
    assert topk_weights.stride(1) == 1
    assert sorted_token_ids.stride(0) == 1

    assert A_scale is not None
    assert B_scale is not None
    if A.dtype == torch.uint8:
        assert A_mx_scale is not None, "A_mx_scale should exist when A is mxfp4"
        A_mx_scale_strid_m, A_mx_scale_strid_k = A_mx_scale.stride()
    else:
        assert A_mx_scale is None, "A_mx_scale should not exist when A is not mxfp4"
        A_mx_scale_strid_m, A_mx_scale_strid_k = None, None
    # NOTE: Only supports B_mx_scale
    assert B_mx_scale is not None

    EM = sorted_token_ids.shape[0]
    if A.shape[0] < config["BLOCK_SIZE_M"]:
        # optimize for small batch_size.
        # We assume that top_ids of each token is unique, so
        # so num_valid_experts <= batch_size <= BLOCK_SIZE_M,
        # and we can skip some invalid blocks.
        EM = min(sorted_token_ids.shape[0], A.shape[0] * top_k * config["BLOCK_SIZE_M"])

    grid = lambda META: (  # noqa: E731
        triton.cdiv(EM, META["BLOCK_SIZE_M"])
        * triton.cdiv(B.shape[1], META["BLOCK_SIZE_N"]),
    )
    _fused_moe_kernel_mxfp4_silu[grid](
        A,
        B,
        C,
        A_scale,
        B_scale,
        A_mx_scale,
        B_mx_scale,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        B.shape[1],
        A.shape[1],
        EM,
        topk_ids.numel(),
        A.stride(0),
        A.stride(1),
        B.stride(0),
        B.stride(2),
        B.stride(1),
        C.stride(0),
        C.stride(1),
        A_mx_scale_strid_m,
        A_mx_scale_strid_k,
        B_mx_scale.stride(0),
        B_mx_scale.stride(2),
        B_mx_scale.stride(1),
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        top_k=top_k,
        compute_type=compute_type,
        SWIZZLE_MX_A=swizzle_mx_a,  # TODO add swizzle support
        SWIZZLE_MX_B=swizzle_mx_b,  # TODO add swizzle support
        **config,
    )
