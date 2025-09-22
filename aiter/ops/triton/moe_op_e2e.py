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
from aiter.ops.triton._triton_kernels.moe_op_e2e import (
    e2e_moe_kernel,
    e2e_moe_persistent_kernel,
)

_LOGGER = AiterTritonLogger()

# Source:
# MoE Kernel adapted from VLLM

_PADDING_SIZE = 0

_MOE_A_QUANT_FUNC = dynamic_per_tensor_quant_fp8_i8

_USE_MOE_PERSISTENT_KERNEL = False


def moe_set_use_persistent_kernel(value: bool):
    global _USE_MOE_PERSISTENT_KERNEL
    _USE_MOE_PERSISTENT_KERNEL = value


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
    Intermediate: torch.Tensor,
    C: torch.Tensor,
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
            #TODO: Add support for per token group quantization
            assert len(block_shape) == 2
            block_n, block_k = block_shape[0], block_shape[1]
            #A, A_scale = per_token_group_quant_fp8(A, block_k)
            assert triton.cdiv(A.shape[-1], block_k) == A_scale.shape[-1]
            assert triton.cdiv(W1.shape[-2], block_n) == B_scale.shape[-2]
            assert triton.cdiv(W1.shape[-1], block_k) == B_scale.shape[-1]
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

    if EM > 1024:
        atomic_num_stages = 2
    else:
        atomic_num_stages = 1

    stride_cm = C.stride(1)
    if _USE_MOE_PERSISTENT_KERNEL:
        NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count * 2
        # TODO add N_split support to get more parallelism
        grid = lambda META: (  # noqa: E731
            min(NUM_SMS, triton.cdiv(sorted_token_ids.shape[0], META["BLOCK_SIZE_M"])),
        )
        stride_im = Intermediate.stride(0)
        EVEN_N = (N // 2) % config["BLOCK_SIZE_N2"] == 0

        e2e_moe_persistent_kernel[grid](
            A,
            W1,
            W2,
            Intermediate,
            C,
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
            stride_cm,
            W1_scale.stride(0) if W1_scale is not None and W1_scale.ndim >= 2 else 0,
            W1_scale.stride(1) if W1_scale is not None and W1_scale.ndim >= 2 else 0,
            W2_scale.stride(0) if W2_scale is not None and W2_scale.ndim >= 2 else 0,
            W1_scale.stride(1) if W2_scale is not None and W2_scale.ndim >= 2 else 0,
            stride_im,
            top_k,
            topk_weights,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            topk_ids.numel(),
            EM,
            N,
            K,
            EVEN_K,
            EVEN_N,
            MUL_ROUTED_WEIGHT=mul_routed_weight,
            use_fp8_w8a8=use_fp8_w8a8,
            use_int8_w8a16=use_int8_w8a16,
            NUM_SMS=NUM_SMS,
            NUM_XCDS=get_num_xcds(),
            **config,
        )

        return C
    else:
        grid = lambda META: (  # noqa: E731
            triton.cdiv(EM, META["BLOCK_SIZE_M"])
            * triton.cdiv(W1.shape[1], META["BLOCK_SIZE_N"]),
        )
        dtype = C.dtype
        Out = C.to(torch.float32) if dtype == torch.bfloat16 else C

        SKINNY = config["BLOCK_SIZE_N"] >= N

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
            stride_cm,
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
            0,0, # group n and k
            EM,
            N,
            K,
            EVEN_K,
            MUL_ROUTED_WEIGHT=mul_routed_weight,
            use_fp8_w8a8=use_fp8_w8a8,
            atomic_num_stages=atomic_num_stages,
            dtype=torch_to_triton_dtype[dtype],
            NUM_XCDS=get_num_xcds(),
            SKINNY=SKINNY,
            compute_type=torch_to_triton_dtype[torch.float32],
            **config,
        )

    return Out.to(dtype)
