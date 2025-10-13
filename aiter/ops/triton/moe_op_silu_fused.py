# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import torch
import triton
import triton.language as tl
from typing import Any, Dict, Optional, List

from aiter.ops.triton.quant import dynamic_per_tensor_quant_fp8_i8
from aiter.ops.triton.utils.logger import AiterTritonLogger
from aiter.ops.triton.utils.device_info import get_num_xcds
from aiter.ops.triton._triton_kernels.moe_op_silu_fused import (
    _fused_moe_silu_kernel_gptq_awq,
    _fused_moe_persistent_silu_kernel_gptq_awq,
    _fused_moe_silu_kernel,
    _fused_moe_persistent_silu_kernel,
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


def fused_moe_silu(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    A_scale: Optional[torch.Tensor],
    B_scale: Optional[torch.Tensor],
    B_zp: Optional[torch.Tensor],
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    mul_routed_weight: bool,
    top_k: int,
    compute_type: tl.dtype,
    use_fp8_w8a8: bool,
    use_int8_w8a16: bool,
    use_int4_w4a16: bool,
    block_shape: Optional[List[int]] = None,
    config: Optional[Dict[str, Any]] = None,
) -> None:
    """
    #TODO: Add doc
    """
    _LOGGER.info(
        f"FUSED_MOE_SILU:  A={tuple(A.shape)}  B={tuple(B.shape)}  C={tuple(C.shape)} "
        + f"topk_weights={tuple(topk_weights.shape)} sorted_token_ids={tuple(sorted_token_ids.shape)} "
        + f"expert_ids={tuple(expert_ids.shape)} num_tokens_post_padded={tuple(num_tokens_post_padded.shape)} "
        + f"top_k={top_k} "
    )
    assert topk_weights.stride(1) == 1
    assert sorted_token_ids.stride(0) == 1

    if use_fp8_w8a8:
        assert B_scale is not None
        if block_shape is None:
            output = torch.zeros(A.shape, device=A.device, dtype=B.dtype)
            A_scale = torch.zeros(1, device=A.device, dtype=torch.float32)
            A, A_scale = _MOE_A_QUANT_FUNC(output, A, A_scale)
        else:
            # TODO: Add support for per token group quantization
            assert len(block_shape) == 2
            block_n, block_k = block_shape[0], block_shape[1]
            # A, A_scale = per_token_group_quant_fp8(A, block_k)
            assert triton.cdiv(A.shape[-1], block_k) == A_scale.shape[-1]
            assert triton.cdiv(B.shape[-2], block_n) == B_scale.shape[-2]
            assert triton.cdiv(B.shape[-1], block_k) == B_scale.shape[-1]
    elif use_int8_w8a16 or use_int4_w4a16:
        assert B_scale is not None
        assert block_shape is None or block_shape[0] == 0
    else:
        assert A_scale is None
        assert B_scale is None

    EM = sorted_token_ids.shape[0]
    if A.shape[0] < config["BLOCK_SIZE_M"]:
        # optimize for small batch_size.
        # We assume that top_ids of each token is unique, so
        # so num_valid_experts <= batch_size <= BLOCK_SIZE_M,
        # and we can skip some invalid blocks.
        EM = min(sorted_token_ids.shape[0], A.shape[0] * top_k * config["BLOCK_SIZE_M"])

    if (
        (use_int8_w8a16 or use_int4_w4a16)
        and block_shape is not None
        and block_shape[1] > 0
    ):
        assert B_scale is not None and B_scale.ndim == 3
        assert B_zp is None or B_zp.ndim == 3
        if _USE_MOE_PERSISTENT_KERNEL:
            NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count * 2
            grid = lambda META: (  # noqa: E731
                min(
                    NUM_SMS,
                    triton.cdiv(sorted_token_ids.shape[0], META["BLOCK_SIZE_M"])
                    * triton.cdiv(B.shape[1], META["BLOCK_SIZE_N"]),
                ),
            )

            _fused_moe_persistent_silu_kernel_gptq_awq[grid](
                A,
                B,
                C,
                B_scale,
                B_zp,
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
                B_scale.stride(0),
                B_scale.stride(2),
                B_scale.stride(1),
                B_zp.stride(0) if B_zp is not None else 0,
                B_zp.stride(2) if B_zp is not None else 0,
                B_zp.stride(1) if B_zp is not None else 0,
                block_k_diviable=A.shape[1] % config["BLOCK_SIZE_K"] == 0,
                group_size=block_shape[1],
                NUM_SMS=NUM_SMS,
                MUL_ROUTED_WEIGHT=mul_routed_weight,
                top_k=top_k,
                compute_type=compute_type,
                has_zp=B_zp is not None,
                use_int4_w4a16=use_int4_w4a16,
                use_int8_w8a16=use_int8_w8a16,
                NUM_XCDS=get_num_xcds(),
                **config,
            )
        else:
            grid = lambda META: (  # noqa: E731
                triton.cdiv(EM, META["BLOCK_SIZE_M"])
                * triton.cdiv(B.shape[1], META["BLOCK_SIZE_N"]),
            )
            _fused_moe_silu_kernel_gptq_awq[grid](
                A,
                B,
                C,
                B_scale,
                B_zp,
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
                B_scale.stride(0),
                B_scale.stride(2),
                B_scale.stride(1),
                B_zp.stride(0) if B_zp is not None else 0,
                B_zp.stride(2) if B_zp is not None else 0,
                B_zp.stride(1) if B_zp is not None else 0,
                block_k_diviable=A.shape[1] % config["BLOCK_SIZE_K"] == 0,
                group_size=block_shape[1],
                MUL_ROUTED_WEIGHT=mul_routed_weight,
                top_k=top_k,
                compute_type=compute_type,
                has_zp=B_zp is not None,
                use_int4_w4a16=use_int4_w4a16,
                use_int8_w8a16=use_int8_w8a16,
                NUM_XCDS=get_num_xcds(),
                **config,
            )

    else:
        if _USE_MOE_PERSISTENT_KERNEL:
            NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count * 2
            grid = lambda META: (  # noqa: E731
                min(
                    NUM_SMS,
                    triton.cdiv(sorted_token_ids.shape[0], META["BLOCK_SIZE_M"])
                    * triton.cdiv(B.shape[1], META["BLOCK_SIZE_N"]),
                ),
            )

            _fused_moe_persistent_silu_kernel[grid](
                A,
                B,
                C,
                A_scale,
                B_scale,
                topk_weights,
                sorted_token_ids,
                expert_ids,
                num_tokens_post_padded,
                B.shape[1],
                A.shape[1] - _PADDING_SIZE,
                sorted_token_ids.shape[0],
                topk_ids.numel(),
                A.stride(0),
                A.stride(1),
                B.stride(0),
                B.stride(2),
                B.stride(1),
                C.stride(0),
                C.stride(1),
                A_scale.stride(0) if A_scale is not None and A_scale.ndim == 2 else 0,
                A_scale.stride(1) if A_scale is not None and A_scale.ndim == 2 else 0,
                B_scale.stride(0) if B_scale is not None and B_scale.ndim >= 2 else 0,
                B_scale.stride(2) if B_scale is not None and B_scale.ndim == 3 else 0,
                B_scale.stride(1) if B_scale is not None and B_scale.ndim >= 2 else 0,
                0 if block_shape is None else block_shape[0],
                0 if block_shape is None else block_shape[1],
                NUM_SMS=NUM_SMS,
                MUL_ROUTED_WEIGHT=mul_routed_weight,
                top_k=top_k,
                compute_type=compute_type,
                use_fp8_w8a8=use_fp8_w8a8,
                use_int8_w8a16=use_int8_w8a16,
                NUM_XCDS=get_num_xcds(),
                **config,
            )
        else:
            grid = lambda META: (  # noqa: E731
                triton.cdiv(EM, META["BLOCK_SIZE_M"])
                * triton.cdiv(B.shape[1], META["BLOCK_SIZE_N"]),
            )
            _fused_moe_silu_kernel[grid](
                A,
                B,
                C,
                A_scale,
                B_scale,
                topk_weights,
                sorted_token_ids,
                expert_ids,
                num_tokens_post_padded,
                B.shape[1],
                A.shape[1] - _PADDING_SIZE,
                EM,
                topk_ids.numel(),
                A.stride(0),
                A.stride(1),
                B.stride(0),
                B.stride(2),
                B.stride(1),
                C.stride(0),
                C.stride(1),
                A_scale.stride(0) if A_scale is not None and A_scale.ndim == 2 else 0,
                A_scale.stride(1) if A_scale is not None and A_scale.ndim == 2 else 0,
                B_scale.stride(0) if B_scale is not None and B_scale.ndim >= 2 else 0,
                B_scale.stride(2) if B_scale is not None and B_scale.ndim == 3 else 0,
                B_scale.stride(1) if B_scale is not None and B_scale.ndim >= 2 else 0,
                0 if block_shape is None else block_shape[0],
                0 if block_shape is None else block_shape[1],
                MUL_ROUTED_WEIGHT=mul_routed_weight,
                top_k=top_k,
                compute_type=compute_type,
                use_fp8_w8a8=use_fp8_w8a8,
                use_int8_w8a16=use_int8_w8a16,
                NUM_XCDS=get_num_xcds(),
                **config,
            )
