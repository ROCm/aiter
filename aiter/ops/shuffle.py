# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

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
    x_ = x_.view(x_type)
    x_.is_shuffled = True
    return x_


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


def shuffle_weight_a16w4(src: torch.Tensor, NLane: int, gate_up: bool) -> torch.Tensor:
    """
    src: shape [experts_cnt, N, K_pk], where K_pk = K // 2
    Returns: shuffled tensor of shape [experts_cnt, N0*2, K0, KLane, NLane, KPack]
    """
    # print("gemm shape:", src.shape)
    src_type = src.dtype
    if hasattr(torch, "float4_e2m1fn_x2") and src_type == torch.float4_e2m1fn_x2:
        src = src.view(torch.uint8)
    experts_cnt, N, K_pk = src.shape
    if gate_up:
        N = N // 2
    KPack = 16
    KLane = 64 // NLane  # 4
    N0 = N // NLane
    K0 = K_pk // (KLane * KPack)
    if gate_up:
        src_reshaped = src.view(
            experts_cnt, 2, N0, NLane, K0, KLane, KPack
        )  # [E,2, N0, NLane ,K0, KLane, KPack]
        src_reshaped = src_reshaped.permute(
            0, 2, 1, 4, 5, 3, 6
        ).contiguous()  # [E, N0, 2, K0, KLane, NLane, KPack]
        interleaved = src_reshaped.view(*src.shape)
    else:
        src_reshaped = src.view(experts_cnt, N0, NLane, K0, KLane, KPack)
        interleaved = (
            src_reshaped.permute(0, 1, 3, 4, 2, 5).contiguous().view(*src.shape)
        )
    # print("interleaved shape:", interleaved.shape)
    return interleaved.contiguous().view(src_type)


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


def shuffle_weight_cktile(
    x: torch.Tensor, layout=(16, 16), use_int4=False
) -> torch.Tensor:
    # Hardcode BLOCK_K and BLOCK_N
    x_type = x.dtype

    IN, IK = layout
    divisor = 4 if IN == 32 else 2

    x_ = x
    x_ = x_.view(x.shape[-2] // IN, IN, x.shape[-1] // IK, divisor, IK // divisor)
    x_ = x_.permute(0, 2, 3, 1, 4)
    x_ = x_.contiguous()
    x_ = x_.view(*x.shape)
    x_ = x_.view(x_type)
    x_.is_shuffled = True
    return x_

def shuffle_bq(
    scale: torch.Tensor,
    block_bq_k: int,
) -> torch.Tensor:
    """
    Shuffle B quantization scale tensor for preshuffleQuant optimization.
    
    This is a direct port of the shuffle_bq function from Composable Kernel.
    It rearranges the scale tensor to match the memory access patterns of
    the GPU warps when using preshuffleQuant optimization.
    
    Args:
        scale: Input scale tensor
               - 2D: [bqk, n] where bqk = K // quant_group_k
               - 5D: [n, nrepeat, nwarp, n_warp_tile, bqk] (TilePermuteN case)
        block_bq_k: Block size for B quantization along K dimension.
                    This is typically derived from the warp tile configuration.
    
    Returns:
        Shuffled scale tensor with the same shape but rearranged for optimal access.
    
    Note:
        The shuffle pattern matches the TileGemmQuantTraits with PreshuffleQuant=true
        in CK Tile. The exact permutation depends on the tensor rank:
        - 2D: permute {1, 0, 2} after reshaping
        - 5D: permute {4, 0, 1, 2, 3, 5} after reshaping
    
    Reference:
        Composable Kernel PR #3629: [CK_Tile] Adding support for preshuffleQuant
        in AB quant Block Scale Gemm
    """
    scale_type = scale.dtype
    original_shape = scale.shape
    rank = scale.dim()
    
    if rank == 2:
        # Input: [bqk, n] where bqk = K // quant_group_k
        bqk, n = scale.shape
        
        if bqk % block_bq_k != 0:
            raise ValueError(
                f"shuffle_bq needs bqk dimension ({bqk}) to be a multiple of "
                f"block_bq_k ({block_bq_k})."
            )
        
        # Step 1: Transpose to [n, bqk]
        # Step 2: Reshape to [n, bqk // block_bq_k, block_bq_k]
        scale_view = scale.t().contiguous().view(n, bqk // block_bq_k, block_bq_k)
        
        # Step 3: Permute with {1, 0, 2} -> [bqk // block_bq_k, n, block_bq_k]
        scale_shuffled = scale_view.permute(1, 0, 2).contiguous()
        
        # Reshape back to original shape [bqk, n]
        scale_shuffled = scale_shuffled.view(original_shape)
        
    elif rank == 5:
        # Input: [n, nrepeat, nwarp, n_warp_tile, bqk] (TilePermuteN case)
        n, nrepeat, nwarp, n_warp_tile, bqk = scale.shape
        
        if bqk % block_bq_k != 0:
            raise ValueError(
                f"shuffle_bq needs bqk dimension ({bqk}) to be a multiple of "
                f"block_bq_k ({block_bq_k})."
            )
        
        # Reshape: [n, nrepeat, nwarp, n_warp_tile, bqk // block_bq_k, block_bq_k]
        scale_view = scale.view(
            n, nrepeat, nwarp, n_warp_tile, bqk // block_bq_k, block_bq_k
        )
        
        # Permute with {4, 0, 1, 2, 3, 5}
        # -> [bqk // block_bq_k, n, nrepeat, nwarp, n_warp_tile, block_bq_k]
        scale_shuffled = scale_view.permute(4, 0, 1, 2, 3, 5).contiguous()
        
        # Reshape back to original shape [n, nrepeat, nwarp, n_warp_tile, bqk]
        scale_shuffled = scale_shuffled.view(original_shape)
        
    else:
        raise ValueError(
            f"shuffle_bq expects either rank-2 or rank-5 tensor, got rank {rank}"
        )
    
    # Mark as shuffled for downstream checks
    scale_shuffled.is_bq_shuffled = True
    
    return scale_shuffled.to(scale_type)


def shuffle_b_scale_blockscale(
    b_scale: torch.Tensor,
    block_bq_k: int,
) -> torch.Tensor:
    """
    Pre-shuffle B (weight) quantization scales for blockscale GEMM.
    
    This is a convenience wrapper around shuffle_bq for the common case
    where the B scale tensor is 2D with shape [N, K // quant_group_k].
    
    Args:
        b_scale: Input B scale tensor of shape [N, K // quant_group_k].
                 Note: This expects the scale in [N, bqk] format, which
                 needs to be transposed to [bqk, N] before calling shuffle_bq.
        block_bq_k: Block size for B quantization along K dimension.
    
    Returns:
        Shuffled B scale tensor optimized for preshuffleQuant access pattern.
    
    Example:
        >>> # For a GEMM with K=4096, quant_group_k=128, N=8192
        >>> # b_scale shape: [8192, 32] (N, K // quant_group_k)
        >>> b_scale_shuffled = shuffle_b_scale_blockscale(b_scale, block_bq_k=4)
    """
    # b_scale is typically [N, bqk], but shuffle_bq expects [bqk, n]
    # So we transpose first
    b_scale_t = b_scale.t().contiguous()
    
    # Apply shuffle_bq
    b_scale_shuffled = shuffle_bq(b_scale_t, block_bq_k)
    
    # Transpose back to [N, bqk]
    return b_scale_shuffled.t().contiguous()

