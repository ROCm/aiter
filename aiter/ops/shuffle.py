# SPDX-License-Identifier: MIT
# Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

import torch


def shuffle_weight(x: torch.Tensor, layout=(16, 16), use_int4=False) -> torch.Tensor:
    # Hardcode BLOCK_K and BLOCK_N
    x_type = x.dtype
    if hasattr(torch, "float4_e2m1fn_x2") and x_type == torch.float4_e2m1fn_x2:
        x = x.view(torch.uint8)

    IN, IK = layout
    BK = IK * 2
    K = 16 // x.element_size() if not use_int4 else 32
    BN = IN
    assert x.shape[-2] % BN == 0, f"{x.shape[-2]} % {BN} == {x.shape[-2] % BN }"
    assert x.shape[-1] % BK == 0, f"{x.shape[-1]} % {BK} == {x.shape[-1] % BK }"

    x_ = x
    x_ = x_.view(-1, x.shape[-2] // BN, BN, x.shape[-1] // BK, BK // K, K)
    x_ = x_.permute(0, 1, 3, 4, 2, 5)
    x_ = x_.contiguous()
    x_ = x_.view(*x.shape)
    return x_.view(x_type)


def shuffle_weight_NK(x: torch.Tensor, inst_N: int, inst_K: int, use_int4=False) -> torch.Tensor:
    kPerLane = inst_K // (64 //  inst_N)
    if(use_int4):
        kPerLane *= 2
    assert x.shape[-2] % inst_N == 0, f"{x.shape[-2]} % {inst_N} == {x.shape[-2] % N_WARP_TILE }"
    assert x.shape[-1] % inst_K == 0, f"{x.shape[-1]} % {inst_K} == {x.shape[-1] % K_WARP_TILE }"

    x_ = x
    x_ = x_.view(-1, x.shape[-2] // inst_N, inst_N, x.shape[-1] // inst_K, 64 // inst_N, kPerLane)
    x_ = x_.permute(0, 1, 3, 4, 2, 5).contiguous()
    return x_.view(*x.shape)
