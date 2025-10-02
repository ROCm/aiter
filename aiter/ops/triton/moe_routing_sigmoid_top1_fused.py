# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

from typing import Optional
import torch
import triton
import triton.language as tl
from aiter.ops.triton.utils.logger import AiterTritonLogger
from aiter.ops.triton._triton_kernels.moe_routing_sigmoid_top1_fused import (
    _routing_sigmoid_top1_kernel,
    _get_config,
)

_LOGGER = AiterTritonLogger()


def routing_sigmoid_top1(
    x, w, topk, fused_shared_experts=False, config: Optional[dict[str, any]] = None
):
    _LOGGER.info(
        f"ROUTING_SIGMOID_TOP1:  x={tuple(x.shape)}  w={tuple(w.shape)}  topk={topk} "
    )
    x = x.view(-1, x.shape[-1])

    assert topk == 1

    # M: batch_size x seq_len, K: hidden_dim, N: num_experts
    M, K = x.shape
    Kb, N = w.shape
    assert K == Kb

    _topk = topk
    if fused_shared_experts:
        _topk += 1

    # Output tensor
    topk_ids = torch.empty((M, _topk), device=x.device, dtype=torch.int32)
    topk_weights = torch.empty((M, _topk), device=x.device, dtype=torch.float32)

    config = _get_config(M, N, K)

    # Grid size
    def grid(META):
        return (triton.cdiv(M, META["BLOCK_M"]),)

    _routing_sigmoid_top1_kernel[grid](
        x,
        w,
        topk_ids,
        topk_weights,
        M,
        N,
        K,
        x.stride(0),
        x.stride(1),
        w.stride(0),
        w.stride(1),
        topk_ids.stride(0),
        topk_ids.stride(1),
        topk_weights.stride(0),
        topk_weights.stride(1),
        BLOCK_N=N,  # Set BLOCK_N to N
        TOPK=topk,
        FUSED_SHARED_EXPERTS=fused_shared_experts,
        **config,
    )

    return topk_ids, topk_weights
