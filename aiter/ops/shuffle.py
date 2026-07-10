# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import torch
import torch.nn.functional as F
from aiter.jit.utils.chip_info import get_gfx


def _moe_tile_shuffle(
    src: torch.Tensor, tile_minor: int, tile_major: int
) -> torch.Tensor:
    """Row-major ``[M, N]`` -> tiled layout, matching the POC ``moe_shuffle_one``
    (majorInN=true): the buffer is split into ``[M/tile_minor, N/tile_major]``
    tiles laid out tile-row-major, and within each tile the ``tile_minor`` (M/row)
    index is outer and the ``tile_major`` (N/col) index is inner.

    This is the single permutation shared by the gfx1250 mxfp8fp4 A / B / scale
    preshuffles; only the tile sizes differ.
    """
    M, N = src.shape
    assert M % tile_minor == 0, f"rows={M} must be divisible by {tile_minor}"
    assert N % tile_major == 0, f"cols={N} must be divisible by {tile_major}"
    out = src.view(M // tile_minor, tile_minor, N // tile_major, tile_major)
    out = out.permute(0, 2, 1, 3).contiguous()
    return out.view(M, N)


def shuffle_mxfp8fp4_a(src: torch.Tensor) -> torch.Tensor:
    """gfx1250 mxfp8fp4 GEMM activation (A) preshuffle (a_preshuffle=1).

    A is mxfp8 (e4m3, 1 byte/elem), row-major ``[M, K]``. The shader expects the
    ``(m, k) -> (m/2, k/128, 2, 128)`` tiling (POC ``moe_shuffle(A, ..., 128, 2)``:
    tileSizeMinor=2 over rows, tileSizeMajor=128 over K).
    """
    x_type = src.dtype
    s = src.view(torch.uint8)
    out = _moe_tile_shuffle(s, tile_minor=2, tile_major=128)
    return out.view(x_type)


def shuffle_mxfp8fp4_b(src: torch.Tensor) -> torch.Tensor:
    """gfx1250 mxfp8fp4 GEMM weight (B) preshuffle (always applied).

    Plain 16x16 tile transpose on the packed byte buffer (POC
    ``moe_shuffle(B, ..., LAYOUT_16X16)``: tileSizeMajor=tileSizeMinor=16). Works
    for both mxfp8 (``[N, K]`` 1 byte/elem) and mxfp4 (``[N, K/2]`` 2 elems/byte).
    """
    x_type = src.dtype
    if hasattr(torch, "float4_e2m1fn_x2") and x_type == torch.float4_e2m1fn_x2:
        s = src.view(torch.uint8)
    else:
        s = src.view(torch.uint8)
    out = _moe_tile_shuffle(s, tile_minor=16, tile_major=16)
    return out.view(x_type)


def shuffle_mxfp8fp4_scale(src: torch.Tensor) -> torch.Tensor:
    """gfx1250 mxfp8fp4 GEMM e8m0 block-scale preshuffle.

    Scale buffer is row-major ``[rows, K/32]`` (e8m0, one byte per 32-K block).
    The shader expects ``(m, k) -> (m/32, k/4, 32, 4)`` (POC
    ``moe_shuffle_one(scale, ..., tileSizeMajor=4, tileSizeMinor=32)``). Same
    layout for the A and B scales.
    """
    x_type = src.dtype
    s = src.view(torch.uint8)
    # The shader loads scales in 32-row super-rows, so the row count must be a
    # multiple of 32. Pad a short buffer (small M) up to the next multiple with
    # the neutral e8m0 scale 0x7F (2^0 == 1.0), matching the POC host's
    # ScaleA_M = (M + 31) & ~31 padding.
    pad = (-s.shape[0]) % 32
    if pad:
        s = F.pad(s, (0, 0, 0, pad), value=0x7F)
    out = _moe_tile_shuffle(s, tile_minor=32, tile_major=4)
    return out.view(x_type)


def shuffle_weight_gfx1250(w: torch.Tensor) -> torch.Tensor:
    """
    Preshuffle weights for gfx1250 WMMA.

    For 2D input (N, K): view as (N//16, 16, K//32, 2, 16) ->
        permute(0, 2, 3, 1, 4) -> reshape (N//16, K*16).
    For 3D input (E, N, K) or (E, K, N): transpose to (E, N, K) first,
        then apply the same pattern per-expert.

    The result is reshaped to (N//16, K*16) for TDM-optimal loading.
    """
    x_type = w.dtype
    if hasattr(torch, "float4_e2m1fn_x2") and x_type == torch.float4_e2m1fn_x2:
        w = w.view(torch.uint8)

    if w.ndim == 2:
        N, K = w.shape
        assert N % 16 == 0, f"N={N} must be divisible by 16"
        assert K % 32 == 0, f"K={K} must be divisible by 32"
        w = w.view(N // 16, 16, K // 32, 2, 16)
        w = w.permute(0, 2, 3, 1, 4).contiguous()
        w = w.view(N // 16, K * 16)
    elif w.ndim == 3:
        E, K, N = w.shape
        assert K % 32 == 0, f"K={K} must be divisible by 32"
        assert N % 16 == 0, f"N={N} must be divisible by 16"
        w = w.transpose(-1, -2)  # (E, N, K)
        w = w.view(E, N // 16, 16, K // 32, 2, 16)
        w = w.permute(0, 1, 3, 4, 2, 5).contiguous()
        w = w.view(E, N // 16, K * 16)
        w = w.transpose(-1, -2)  # (E, K*16, N//16)
    else:
        raise ValueError(f"Expected 2D or 3D tensor, got {w.ndim}D")

    w = w.view(x_type)
    return w


def shuffle_weight(
    x: torch.Tensor,
    layout=(16, 16),
    use_int4=False,
    is_guinterleave=False,
    gate_up: bool = False,
    pad_k_to: int = 0,
) -> torch.Tensor:
    x_type = x.dtype
    if hasattr(torch, "float4_e2m1fn_x2") and x_type == torch.float4_e2m1fn_x2:
        x = x.view(torch.uint8)

    original_k = x.shape[-1]
    if pad_k_to:
        if pad_k_to < 0:
            raise ValueError(f"pad_k_to must be non-negative, got {pad_k_to}")
        if use_int4:
            raise NotImplementedError("pad_k_to is not supported with use_int4=True")
        if is_guinterleave:
            raise NotImplementedError(
                "pad_k_to is not supported with is_guinterleave=True"
            )
        padded_k = ((original_k + pad_k_to - 1) // pad_k_to) * pad_k_to
        if padded_k != original_k:
            x = F.pad(x.contiguous(), (0, padded_k - original_k), value=0)

    if is_guinterleave:
        experts_cnt, N, K_pk = x.shape
        if gate_up:
            N = N // 2
        NLane, KPack = layout
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
    if pad_k_to:
        x_.aiter_original_k = original_k
        x_.aiter_padded_k = x.shape[-1]
    return x_


def shuffle_weight_a16w4(src: torch.Tensor, NLane: int, gate_up: bool) -> torch.Tensor:
    """Backward-compatible wrapper around `shuffle_weight(..., is_guinterleave=True)`."""
    return shuffle_weight(
        src, layout=(NLane, 16), is_guinterleave=True, gate_up=gate_up
    )


def shuffle_weight_NK(
    x: torch.Tensor, inst_N: int, inst_K: int, use_int4=False
) -> torch.Tensor:
    kPerLane = inst_K // (64 // inst_N)
    if use_int4:
        kPerLane *= 2
    assert (
        x.shape[-2] % inst_N == 0
    ), f"{x.shape[-2]} % {inst_N} == {x.shape[-2] % inst_N }"
    assert (
        x.shape[-1] % inst_K == 0
    ), f"{x.shape[-1]} % {inst_K} == {x.shape[-1] % inst_K }"

    x_ = x
    x_ = x_.view(
        -1, x.shape[-2] // inst_N, inst_N, x.shape[-1] // inst_K, 64 // inst_N, kPerLane
    )
    x_ = x_.permute(0, 1, 3, 4, 2, 5).contiguous()
    return x_.view(*x.shape)


def shuffle_scale_n32k4(
    src: torch.Tensor,
    experts_cnt: int = None,
    is_guinterleave: bool = False,
    gate_up: bool = False,
) -> torch.Tensor:
    """Shuffle a raw per-expert e8m0 weight (B) scale into the n32k4 layout.

    Input: ``(E, N, K//32)`` (3D) or ``(E*N, K//32)`` (2D, needs ``experts_cnt``).
    Output: ``(E, N//32, (K//32)*32)`` uint8.

    Within a 32-row super-row the column is ``remain_k*128 + row32*4 + r`` so each
    lane reads its full WMMA scaleB operand (4 e8m0 of one WMMA-K=128 step) with
    one contiguous ds_load_b32.  Consumed by the gfx1250 grouped MoE GEMM
    (see kernels/gemm_mxscale_gfx1250.py).

    ``is_guinterleave`` selects the stage1 gate/up packing (``N == 2*inter_dim``):

    * ``False`` (GGUU / ``GateMode.SEPARATED``): rows are ``[g0..g_{I-1},
      u0..u_{I-1}]``; the n32k4 super-rows are taken as-is.
    * ``True`` (GUGU / ``GateMode.INTERLEAVE``): the raw GGUU rows are first
      interleaved to ``[g0,u0,g1,u1,...]`` (matching the INTERLEAVE stage1
      weight layout produced for the fused gemm1) and then folded into n32k4.

    ``gate_up`` is accepted for signature parity with ``shuffle_scale`` /
    ``moe_shuffle_scale``; the gate/up interleave is driven by
    ``is_guinterleave`` (the n32k4 fold is identical for both).
    """
    s = src.view(torch.uint8).contiguous()
    if s.ndim == 2:
        if experts_cnt is None:
            raise ValueError("experts_cnt is required for a 2D n32k4 scale")
        s = s.view(experts_cnt, -1, s.shape[-1])
    elif s.ndim != 3:
        raise ValueError(f"n32k4 scale must be 2D or 3D, got {s.ndim}D")
    E, N, k_scale = s.shape
    if is_guinterleave:
        # GUGU: interleave gate/up rows [g..,u..] -> [g0,u0,g1,u1,...] so the
        # n32k4 super-rows line up with the INTERLEAVE stage1 weight layout.
        if N % 2 != 0:
            raise ValueError(
                f"GUGU n32k4 scale needs N=2*inter_dim (even rows), got N={N}"
            )
        # (E, [g..,u..], k) -> (E, 2, N/2, k) -> (E, N/2, 2, k) -> (E, N, k)
        s = s.view(E, 2, N // 2, k_scale).permute(0, 2, 1, 3).reshape(E, N, k_scale)
    if N % 32 != 0:
        raise ValueError(f"B-scale rows must be divisible by 32, got {N}")
    if k_scale % 4 != 0:
        raise ValueError(
            f"B-scale K//32 must be divisible by 4 (K%128==0), got {k_scale}"
        )
    g = s.view(E, N // 32, 32, k_scale // 4, 4).permute(0, 1, 3, 2, 4).contiguous()
    return g.reshape(E, N // 32, k_scale * 32)


def shuffle_scale_f4(
    src: torch.Tensor,
    intype: int = 7,
) -> torch.Tensor:
    """gfx1250 F4GEMM scale preshuffle.

    Tiles the [M, N] scale buffer with (majorInN=True):
      NVFP4 (intype=8): tileSizeMajor=8, tileSizeMinor=32
      MXFP4 (intype=7): tileSizeMajor=4, tileSizeMinor=32
    Each destination tile is ordered [tileM, tileN, m, k] (m, the M/row dir,
    outer; k, the N/col dir, inner).
    """
    tile_major = 8 if intype == 8 else 4
    tile_minor = 32
    M, N = src.shape

    tiles_m = M // tile_minor
    tiles_n = N // tile_major

    # src[tileM*minor + m, tileN*major + k] -> [tileM, m, tileN, k]
    out = src.view(tiles_m, tile_minor, tiles_n, tile_major)
    # -> [tileM, tileN, m, k] (m outer, k inner) per moe_shuffle_one
    out = out.permute(0, 2, 1, 3).contiguous()
    return out.view(M, N)


def shuffle_weight_f4(src: torch.Tensor) -> torch.Tensor:
    """gfx1250 F4GEMM weight (A/B) preshuffle.

    Input is packed fp4 ``[rows, K/2]`` (uint8, two nibbles per byte). Applies a
    plain 16x16 tile transpose on the packed byte buffer (tileSizeMajor=16 over
    the K dir, tileSizeMinor=16 over the row dir). Same layout for MXFP4/NVFP4.
    """
    x_type = src.dtype
    if hasattr(torch, "float4_e2m1fn_x2") and x_type == torch.float4_e2m1fn_x2:
        src = src.view(torch.uint8)
    rows, kp = src.shape  # kp = K/2 packed bytes
    assert rows % 16 == 0, f"rows={rows} must be divisible by 16"
    assert kp % 16 == 0, f"packed K dim={kp} must be divisible by 16"
    # src[tileM*16 + m, tileK*16 + k] -> [tileM, m, tileK, k]
    out = src.view(rows // 16, 16, kp // 16, 16)
    # -> [tileM, tileK, m, k] (tilePtr[m*16 + k]) per moe_shuffle_one
    out = out.permute(0, 2, 1, 3).contiguous()
    return out.view(rows, kp).view(x_type)


def shuffle_scale(
    src: torch.Tensor,
    experts_cnt: int = None,
    is_guinterleave: bool = False,
    gate_up: bool = False,
) -> torch.Tensor:
    if src is None:
        return src
    if src.dtype == torch.float32:
        return src
    assert src.ndim == 2, "scale must be a 2D tensor"

    if not is_guinterleave:
        m, n = src.shape
        scale_padded = torch.empty(
            (m + 255) // 256 * 256,
            (n + 7) // 8 * 8,
            dtype=src.dtype,
            device=src.device,
        )

        scale_padded[:m, :n] = src
        scale = scale_padded
        sm, sn = scale.shape
        scale = scale.view(sm // 32, 2, 16, sn // 8, 2, 4)
        scale = scale.permute(0, 3, 5, 2, 4, 1).contiguous()
        return scale.view(sm, sn)

    if experts_cnt is None:
        raise ValueError("experts_cnt is required when is_guinterleave=True")

    n_experts, k_ = src.shape
    if k_ % 8 != 0:
        k_padded = (k_ + 7) // 8 * 8
        scale_padded = torch.empty(
            n_experts, k_padded, dtype=src.dtype, device=src.device
        )
        if scale_padded.element_size() == 1:
            scale_padded.view(torch.uint8).fill_(0x7F)
        else:
            scale_padded.fill_(1)
        scale_padded[:, :k_] = src
        src = scale_padded
        k_ = k_padded
    n_ = n_experts // experts_cnt
    # MXFP4 constants.  The scale layout packs two 4-scale dwords per tile-K
    # group, so shapes with K//32 not divisible by 8 are padded above.
    K_Pack = 2
    N_Pack = 2
    N_Lane = 16
    K_Lane = 64 // N_Lane  # 4

    # Basic dimensions
    K1 = k_ // K_Pack // K_Lane
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


def moe_shuffle_scale(
    src: torch.Tensor,
    experts_cnt: int = None,
    is_guinterleave: bool = False,
    gate_up: bool = False,
) -> torch.Tensor:
    """Arch-aware MoE weight (B) scale shuffle."""

    if get_gfx() == "gfx1250":
        # n32k4 grouped-MoE B-scale. GGUU (is_guinterleave=False) folds rows
        # as-is; GUGU (is_guinterleave=True) interleaves gate/up rows first.
        return shuffle_scale_n32k4(
            src,
            experts_cnt,
            is_guinterleave=is_guinterleave,
            gate_up=gate_up,
        )
    return shuffle_scale(
        src, experts_cnt=experts_cnt, is_guinterleave=is_guinterleave, gate_up=gate_up
    )


def shuffle_scale_a16w4(
    src: torch.Tensor, experts_cnt: int, gate_up: bool
) -> torch.Tensor:
    """Backward-compatible wrapper around `shuffle_scale(..., is_guinterleave=True)`."""
    return shuffle_scale(
        src, experts_cnt=experts_cnt, is_guinterleave=True, gate_up=gate_up
    )


def pack_int8_to_packed_int4(x_shuf_i8: torch.Tensor) -> torch.Tensor:
    """Pack a preshuffled int8 tensor (values in [-8, 7]) into packed int4 bytes.

    Each contiguous 8-value block [v0..v7] -> 4 bytes:
      b0=(v4<<4)|v0, b1=(v5<<4)|v1, b2=(v6<<4)|v2, b3=(v7<<4)|v3.

    This matches the 7-op in-kernel unpack sequence used by FlyDSL int4_bf16.
    """
    flat = x_shuf_i8.contiguous().view(-1).to(torch.int16)
    assert flat.numel() % 8 == 0
    u = (flat & 0xF).to(torch.uint8).view(-1, 8)
    out = torch.empty((u.shape[0], 4), device=u.device, dtype=torch.uint8)
    out[:, 0] = u[:, 0] | (u[:, 4] << 4)
    out[:, 1] = u[:, 1] | (u[:, 5] << 4)
    out[:, 2] = u[:, 2] | (u[:, 6] << 4)
    out[:, 3] = u[:, 3] | (u[:, 7] << 4)
    return out.view(-1).to(torch.int8)


def shuffle_scale_for_int4(scale: torch.Tensor, group_size: int = 32) -> torch.Tensor:
    """Prepare groupwise scale tensor for W4A16 int4 kernel.

    Input: scale tensor of shape ``[E, num_groups, N]``.

    For **f32** scales the kernel uses ``(E, G, N)`` layout directly.

    For **bf16** scales the kernel uses ``(E, G//2, N, 2)`` layout -- two
    adjacent groups for the same N position are packed into one dword.

    Only group_size=32 is supported due to int4 preshuffle layout constraints.
    """
    if group_size != 32:
        raise ValueError(
            f"shuffle_scale_for_int4 only supports group_size=32, got {group_size}. "
            f"This is due to int4 preshuffle layout constraints."
        )

    if scale.dtype == torch.bfloat16:
        E, G, N = scale.shape
        return scale.view(E, G // 2, 2, N).permute(0, 1, 3, 2).contiguous()

    return scale.contiguous()
