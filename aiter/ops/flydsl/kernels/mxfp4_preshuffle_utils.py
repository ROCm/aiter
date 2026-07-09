# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Host-side CK preshuffle helpers for the FlyDSL MXFP4/MXFP8 preshuffle GEMM.

Verbatim from FlyDSL tests/kernels/utils/fp4_utils.py so the vendored kernel reads
weights/scales in exactly the layout it was validated against.
"""

import torch


def shuffle_weight_w4(
    src: torch.Tensor, NLane: int, gate_up: bool, moe_gemm: bool
) -> torch.Tensor:
    """src: [N, K_pk] (K_pk = K // 2) -> CK-preshuffled weight the kernel reads."""
    src_type = src.dtype
    if hasattr(torch, "float4_e2m1fn_x2") and src_type == torch.float4_e2m1fn_x2:
        src = src.view(torch.uint8)
    if moe_gemm:
        experts_cnt, N, K_pk = src.shape
        if gate_up:
            N = N // 2
        KPack = 16
        KLane = 64 // NLane  # 4
        N0 = N // NLane
        K0 = K_pk // (KLane * KPack)
        if gate_up:
            src_reshaped = src.view(experts_cnt, 2, N0, NLane, K0, KLane, KPack)
            src_reshaped = src_reshaped.permute(0, 2, 1, 4, 5, 3, 6).contiguous()
            interleaved = src_reshaped.view(*src.shape)
        else:
            src_reshaped = src.view(experts_cnt, N0, NLane, K0, KLane, KPack)
            interleaved = (
                src_reshaped.permute(0, 1, 3, 4, 2, 5).contiguous().view(*src.shape)
            )
        return interleaved.contiguous().view(src_type)
    else:
        N, K_pk = src.shape
        KPack = 16
        KLane = 64 // NLane  # 4
        N0 = N // NLane
        K0 = K_pk // (KLane * KPack)
        src_reshaped = src.view(N0, NLane, K0, KLane, KPack)
        interleaved = src_reshaped.permute(0, 2, 3, 1, 4).contiguous().view(*src.shape)
        return interleaved.contiguous().view(src_type)


def shuffle_scale_w4(
    src: torch.Tensor, experts_cnt: int, gate_up: bool
) -> torch.Tensor:
    """src: [n, K//32] e8m0 -> CK-preshuffled scale the kernel reads."""
    n_experts, k_ = src.shape
    n_ = n_experts // experts_cnt
    K_Pack = 2
    N_Pack = 2
    N_Lane = 16
    K_Lane = 64 // N_Lane  # 4

    K1 = k_ // K_Pack // K_Lane  # k_ // 8
    N1 = n_ // N_Lane // N_Pack  # n_ // 32
    real_k = 32 * k_ * K_Pack * K_Lane  # 1x32 quant
    assert real_k >= 256, f"K {real_k} must be larger than Tile_K(256)"
    if gate_up:
        shfl_scale = src.view(experts_cnt, N_Pack, N1, N_Lane, K1, K_Pack, K_Lane)
        shfl_scale = shfl_scale.permute(0, 2, 4, 6, 3, 5, 1).contiguous()
    else:
        shfl_scale = src.view(experts_cnt, N1, N_Pack, N_Lane, K1, K_Pack, K_Lane)
        shfl_scale = shfl_scale.permute(0, 1, 4, 6, 3, 5, 2).contiguous()
    return shfl_scale.view(*src.shape).contiguous()
