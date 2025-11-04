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
    block_shape: Optional[List[int]] = None,
    config: Optional[Dict[str, Any]] = None,
    return_intermediate: Optional[bool] = False,
    pertoken_quant_a: Optional[bool] = False,
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
        assert A_scale is not None
        assert W1_scale is not None
        assert W2_scale is not None
        if block_shape is None:
            block_n, block_k = 0, 0
        else:
            assert len(block_shape) == 2
            block_n, block_k = block_shape[0], block_shape[1]
            assert (
                config["BLOCK_SIZE_K1"] <= block_k
            ), "BLOCK_SIZE_K1 must be <= group_k when using fp8"
            assert triton.cdiv(A.shape[-1], block_k) == A_scale.shape[-1]
            assert triton.cdiv(W1.shape[-2], block_n) == W1_scale.shape[-2]
            assert triton.cdiv(W1.shape[-1], block_k) == W1_scale.shape[-1]
            assert triton.cdiv(W2.shape[-1], block_n) == W2_scale.shape[-1]
            assert triton.cdiv(W2.shape[-2], block_k) == W2_scale.shape[-2]
        
    else:
        assert A_scale is None
        assert W1_scale is None
        assert W2_scale is None
        block_n, block_k = 0, 0

    EM = sorted_token_ids.shape[0]
    if A.shape[0] < config["BLOCK_SIZE_M"]:
        # optimize for small batch_size.
        # We assume that top_ids of each token is unique, so
        # so num_valid_experts <= batch_size <= BLOCK_SIZE_M,
        # and we can skip some invalid blocks.
        EM = min(sorted_token_ids.shape[0], A.shape[0] * top_k * config["BLOCK_SIZE_M"])

    M = A.shape[0]
    N = W1.shape[1]
    K = A.shape[1] - _PADDING_SIZE
    EVEN_K1 = K % config["BLOCK_SIZE_K1"] == 0
    EVEN_K2 = K % config["BLOCK_SIZE_K2"] == 0

    pid_m = (EM + config["BLOCK_SIZE_M"] - 1) // config["BLOCK_SIZE_M"]
    pid_n = (N + config["BLOCK_SIZE_N"] - 1) // config["BLOCK_SIZE_N"]
    grid = (pid_m * pid_n,)

    dtype = W1.dtype  # input dtype
    out_dtype = Out.dtype
    # if the intermediate token dimension is small enough, we can try to fit the whole thing in shared memory
    SKINNY = pid_n == 1

    if not SKINNY:
        Out = Out.to(torch.float32)  # atomics need to be done in fp32

    if return_intermediate:
        Intermediate = torch.zeros(
            (M, top_k, N // 2), dtype=torch.float32, device="cuda"
        )
    else:
        Intermediate = None

    e2e_moe_kernel[grid](
        A,
        W1,
        W2,
        Out,
        Intermediate,
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
        Out.stride(1),
        Out.stride(2),
        Intermediate.stride(1) if return_intermediate else 0,
        A_scale.stride(0) if A_scale is not None and block_shape is not None else 0,
        A_scale.stride(1) if A_scale is not None and block_shape is not None else 0,
        W1_scale.stride(0) if W1_scale is not None and W1_scale.ndim >= 2 else 0,
        W1_scale.stride(1) if W1_scale is not None and W1_scale.ndim >= 2 else 0,
        W1_scale.stride(2) if W1_scale is not None and W1_scale.ndim >= 2 else 0,
        W2_scale.stride(0) if W2_scale is not None and W2_scale.ndim >= 2 else 0,
        W2_scale.stride(1) if W2_scale is not None and W2_scale.ndim >= 2 else 0,
        W2_scale.stride(2) if W2_scale is not None and W2_scale.ndim >= 2 else 0,
        top_k,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        topk_ids.numel(),
        block_n,
        block_k,
        EM,
        N,
        K,
        EVEN_K1,
        EVEN_K2,
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        use_fp8_w8a8=use_fp8_w8a8,
        use_block_scale=block_shape is not None,
        NUM_XCDS=get_num_xcds(),
        SKINNY=SKINNY,
        dtype=torch_to_triton_dtype[dtype],  # mma dtype
        out_dtype=(
            torch_to_triton_dtype[out_dtype]
            if SKINNY
            else torch_to_triton_dtype[torch.float32]
        ),  # atomics need to be done in fp32
        **config,
        return_intermediate=return_intermediate,
        PER_TOKEN_QUANT_A=pertoken_quant_a,
    )

    if return_intermediate:
        return Out.to(out_dtype), Intermediate
    else:
        return Out.to(out_dtype)
