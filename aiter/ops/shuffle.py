# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import torch


def shuffle_weight(
    x: torch.Tensor,
    layout=(16, 16),
    use_int4=False,
    is_guinterleave=False,
    NLane: int = 16,
    gate_up: bool = False,
) -> torch.Tensor:
    x_type = x.dtype
    if hasattr(torch, "float4_e2m1fn_x2") and x_type == torch.float4_e2m1fn_x2:
        x = x.view(torch.uint8)

    if is_guinterleave:
        experts_cnt, N, K_pk = x.shape
        if gate_up:
            N = N // 2
        KPack = 16
        KLane = 64 // NLane
        N0 = N // NLane
        K0 = K_pk // (KLane * KPack)
        if gate_up:
            x_ = x.view(experts_cnt, 2, N0, NLane, K0, KLane, KPack)
            x_ = x_.permute(0, 2, 1, 4, 5, 3, 6).contiguous()
        else:
            x_ = x.view(experts_cnt, N0, NLane, K0, KLane, KPack)
            x_ = x_.permute(0, 1, 3, 4, 2, 5).contiguous()
        x_ = x_.view(*x.shape).contiguous().view(x_type)
        x_.is_shuffled = True
        return x_

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
    x_ = x_.view(x_type)
    x_.is_shuffled = True
    return x_


def shuffle_weight_a16w4(src: torch.Tensor, NLane: int, gate_up: bool) -> torch.Tensor:
    """Backward-compatible wrapper around `shuffle_weight(..., is_guinterleave=True)`."""
    return shuffle_weight(src, is_guinterleave=True, NLane=NLane, gate_up=gate_up)


def shuffle_weight_NK(
    x: torch.Tensor, inst_N: int, inst_K: int, use_int4=False
) -> torch.Tensor:
    kPerLane = inst_K // (64 // inst_N)
    if use_int4:
        kPerLane *= 2
    assert (
        x.shape[-2] % inst_N == 0
    ), f"{x.shape[-2]} % {inst_N} == {x.shape[-2] % N_WARP_TILE }"
    assert (
        x.shape[-1] % inst_K == 0
    ), f"{x.shape[-1]} % {inst_K} == {x.shape[-1] % K_WARP_TILE }"

    x_ = x
    x_ = x_.view(
        -1, x.shape[-2] // inst_N, inst_N, x.shape[-1] // inst_K, 64 // inst_N, kPerLane
    )
    x_ = x_.permute(0, 1, 3, 4, 2, 5).contiguous()
    return x_.view(*x.shape)


def shuffle_scale_a16w4(
    src: torch.Tensor, experts_cnt: int, gate_up: bool
) -> torch.Tensor:
    n_experts, k_ = src.shape
    n_ = n_experts // experts_cnt
    # MXFP4 constants
    K_Pack = 2
    N_Pack = 2
    N_Lane = 16
    K_Lane = 64 // N_Lane  # 4

    # Basic dimensions
    K1 = k_ // K_Pack // K_Lane  # k_ // 8
    N1 = n_ // N_Lane // N_Pack  # n_ // 32
    real_k = 32 * k_ * K_Pack * K_Lane  # 1x32 quant
    assert real_k >= 256, f"K {real_k} must be larger than Tile_K(256)"
    # print("src shape", src.shape)
    # Reshape based on moe_kind
    if gate_up:
        # Reshape to: [E, N_Pack, N1, N_Lane, K1, K_Pack, K_Lane]
        shfl_scale = src.view(experts_cnt, N_Pack, N1, N_Lane, K1, K_Pack, K_Lane)
        # Permute to: [E, N1, K1, K_Lane, N_Lane, K_Pack, N_Pack]
        shfl_scale = shfl_scale.permute(0, 2, 4, 6, 3, 5, 1).contiguous()
    else:
        # Reshape to: [E, K1, K_Pack, K_Lane, N1, N_Pack, N_Lane]
        shfl_scale = src.view(experts_cnt, N1, N_Pack, N_Lane, K1, K_Pack, K_Lane)
        # Permute to: [E, N1, K1, K_Lane, N_Lane, K_Pack, N_Pack]
        shfl_scale = shfl_scale.permute(0, 1, 4, 6, 3, 5, 2).contiguous()
    # print("shf_scale shape:", shfl_scale.shape)
    return shfl_scale.view(*src.shape).contiguous()
