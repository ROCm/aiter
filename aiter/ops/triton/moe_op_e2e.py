# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import torch
import triton
import triton.language as tl
from typing import Any, Dict, Optional, List

from aiter.ops.triton.quant import dynamic_per_tensor_quant_fp8_i8
from aiter.ops.triton.utils.types import torch_to_triton_dtype
from aiter.ops.triton.utils.logger import AiterTritonLogger
from aiter.ops.triton.utils.device_info import get_num_xcds
from aiter.ops.triton._triton_kernels.moe_op_e2e import e2e_moe_kernel

_LOGGER = AiterTritonLogger()

_PADDING_SIZE = 0

_MOE_A_QUANT_FUNC = dynamic_per_tensor_quant_fp8_i8


def moe_set_padding_size(size: int):
    """
    Override padding size
    """
    global _PADDING_SIZE
    _PADDING_SIZE = size


def moe_set_quant_func(func):
    """
    Override 'A' matrix ie activations quantization function.
    Default function does dynamic quantization.
    """
    global _MOE_A_QUANT_FUNC
    _MOE_A_QUANT_FUNC = func


def e2e_moe(
    A: torch.Tensor,
    W1: torch.Tensor,
    W2: torch.Tensor,
    Out: torch.Tensor,
    A_scale: Optional[torch.Tensor],
    W1_scale: Optional[torch.Tensor],
    W2_scale: Optional[torch.Tensor],
    topk_weights: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    topk_ids,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    mul_routed_weight: bool,
    top_k: int,
    use_fp8_w8a8: bool,
    use_int8_w8a16: bool,
    block_shape: Optional[List[int]] = None,
    config: Optional[Dict[str, Any]] = None,
) -> None:
    """
    #TODO: Add doc
    """
    _LOGGER.info(
        f"MOE_E2E:  A={tuple(A.shape)}  W1={tuple(W1.shape)}  W2={tuple(W2.shape)}  topk_weights={tuple(topk_weights.shape)}"
        + f" sorted_token_ids={tuple(sorted_token_ids.shape)} expert_ids={tuple(expert_ids.shape)}"
        + f" num_tokens_post_padded={tuple(num_tokens_post_padded.shape)} top_k={top_k} "
    )
    assert topk_weights.stride(1) == 1
    assert sorted_token_ids.stride(0) == 1

    if use_fp8_w8a8:
        assert W1_scale is not None
        assert W2_scale is not None
        if block_shape is None:
            output = torch.zeros(A.shape, device=A.device, dtype=torch.float8_e4m3fnuz)
            A_scale = torch.zeros(1, device=A.device, dtype=torch.float32)
            A, A_scale = _MOE_A_QUANT_FUNC(output, A, A_scale)
        else:
            # TODO: Add support for per token group quantization
            assert len(block_shape) == 2
            block_n, block_k = block_shape[0], block_shape[1]

            assert (
                config["BLOCK_SIZE_K1"] <= block_k
            ), "BLOCK_SIZE_K1 must be <= group_k when using fp8"

            # A, A_scale = per_token_group_quant_fp8(A, block_k)
            assert triton.cdiv(A.shape[-1], block_k) == A_scale.shape[-1]
            assert triton.cdiv(W1.shape[-2], block_n) == W1_scale.shape[-2]
            assert triton.cdiv(W1.shape[-1], block_k) == W1_scale.shape[-1]
            assert triton.cdiv(W2.shape[-2], block_n) == W2_scale.shape[-2]
            assert triton.cdiv(W2.shape[-1], block_k) == W2_scale.shape[-1]
    else:
        assert A_scale is None
        assert W1_scale is None
        assert W2_scale is None

    EM = sorted_token_ids.shape[0]
    if A.shape[0] < config["BLOCK_SIZE_M"]:
        # optimize for small batch_size.
        # We assume that top_ids of each token is unique, so
        # so num_valid_experts <= batch_size <= BLOCK_SIZE_M,
        # and we can skip some invalid blocks.
        EM = min(sorted_token_ids.shape[0], A.shape[0] * top_k * config["BLOCK_SIZE_M"])

    N = W1.shape[1]
    K = A.shape[1] - _PADDING_SIZE
    EVEN_K = K % config["BLOCK_SIZE_K1"] == 0

    stride_om = Out.stride(1)

    grid = lambda META: (  # noqa: E731
        triton.cdiv(EM, META["BLOCK_SIZE_M"])
        * triton.cdiv(W1.shape[1], META["BLOCK_SIZE_N"]),
    )
    dtype = W1.dtype  # input dtype
    out_dtype = Out.dtype
    # if the intermediate token dimension is small enough, we can try to fit the whole thing in shared memory
    SKINNY = config["BLOCK_SIZE_N"] >= N

    if block_shape is not None:
        if len(block_shape) == 2:
            group_n = block_shape[0]
            group_k = block_shape[1]
        elif len(block_shape) == 1:
            group_n = block_shape[0]
            group_k = group_n
        else:
            raise ValueError("block_shape must be of length 1 or 2")
    else:
        group_n = 0
        group_k = 0

    assert config["BLOCK_SIZE_N"] % 2 == 0, "BLOCK_SIZE_N must be even"
    BLOCK_SIZE_HALF = config["BLOCK_SIZE_N"] // 2
    if use_fp8_w8a8 and block_shape is not None:
        assert (BLOCK_SIZE_HALF <= group_n) or (
            BLOCK_SIZE_HALF % group_n == 0
        ), "BLOCK_SIZE_N//2 must be multiple of group_n or <= group_n"
        assert (
            config["BLOCK_SIZE_K1"] <= group_k
        ), "BLOCK_SIZE_K1 must strictly be <= group_k"
        assert (config["BLOCK_SIZE_K2"] <= group_k) or (
            config["BLOCK_SIZE_K2"] % group_k == 0
        ), "BLOCK_SIZE_K2 must be multiple of group_k or <= group_k"

    e2e_moe_kernel[grid](
        A,
        W1,
        W2,
        Out,
        A_scale,
        W1_scale,
        W2_scale,
        A.stride(0),
        A.stride(1),
        W1.stride(0),
        W1.stride(1),
        W1.stride(2),
        W2.stride(0),
        W2.stride(2),
        W2.stride(1),
        stride_om,
        A_scale.stride(0) if A_scale is not None else 0,
        A_scale.stride(1) if A_scale is not None else 0,
        W1_scale.stride(0) if W1_scale is not None and W1_scale.ndim >= 2 else 0,
        W1_scale.stride(1) if W1_scale is not None and W1_scale.ndim >= 2 else 0,
        W1_scale.stride(2) if W1_scale is not None and W1_scale.ndim >= 2 else 0,
        W2_scale.stride(0) if W2_scale is not None and W2_scale.ndim >= 2 else 0,
        W1_scale.stride(1) if W2_scale is not None and W2_scale.ndim >= 2 else 0,
        W2_scale.stride(2) if W2_scale is not None and W2_scale.ndim >= 2 else 0,
        top_k,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        topk_ids.numel(),
        group_n,
        group_k,
        EM,
        N,
        K,
        EVEN_K,
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        use_fp8_w8a8=use_fp8_w8a8,
        NUM_XCDS=get_num_xcds(),
        SKINNY=SKINNY,
        dtype=torch_to_triton_dtype[dtype],  # input dtype, mma dtype
        out_dtype=torch_to_triton_dtype[out_dtype],
        **config,
    )

    return Out.to(out_dtype)
