# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import os
from typing import Optional

import torch


# Tensor attribute names used by ``pad_weight_for_bpreshuffle`` and propagated
# by ``shuffle_weight`` so downstream callers (e.g. SGLang's compressed-tensors
# FP8 path) can detect padded weights without inspecting raw shapes.
_BPRESHUFFLE_PADDING_ATTRS = ("aiter_original_k", "aiter_padded_k", "aiter_k_padding")

# Default K alignment used when padding weights for the bpreshuffle GEMM. 256
# covers all currently-shipping CK bpreshuffle BLOCK_K values for FP8 on
# gfx950. Override at process start with ``AITER_BPRESHUFFLE_PAD_ALIGNMENT``
# and rerun the bpreshuffle tuner if a padded shape is still rejected by
# ``IsSupportedArgument``.
_DEFAULT_BPRESHUFFLE_PAD_ALIGNMENT = int(
    os.environ.get("AITER_BPRESHUFFLE_PAD_ALIGNMENT", "256")
)


def pad_weight_for_bpreshuffle(
    x: torch.Tensor,
    alignment: int = _DEFAULT_BPRESHUFFLE_PAD_ALIGNMENT,
    layout=(16, 16),
) -> torch.Tensor:
    """Right-pad ``K`` (last dim) to a multiple of ``alignment`` with zeros.

    Returns a contiguous tensor tagged with ``aiter_original_k``,
    ``aiter_padded_k`` and ``aiter_k_padding`` metadata so downstream wrappers
    can detect padding without inspecting shapes. The caller is responsible
    for matching activation padding at runtime.

    The padded tail is exactly zero, which keeps the FP8 GEMM result
    mathematically identical for per-token / per-channel quantization
    (``x_scale`` is per token and ``w_scale`` is per output channel, neither
    is affected by padding the K dimension).

    ``alignment=256`` covers all currently-shipping CK bpreshuffle ``BLOCK_K``
    values for FP8 on gfx950. If a padded shape is still rejected by
    ``IsSupportedArgument``, bump via ``AITER_BPRESHUFFLE_PAD_ALIGNMENT=512``
    and rerun the tuner.

    Memory peak during this helper is ``2 * sizeof(weight)`` because both ``x``
    and ``out`` are live until return. Callers that load weights eagerly
    should ``del`` the original tensor and call ``torch.cuda.empty_cache()``
    once the returned tensor has been wrapped in a ``Parameter``.
    """
    if alignment <= 0:
        raise ValueError(
            f"pad_weight_for_bpreshuffle: alignment must be positive, got "
            f"{alignment}"
        )

    original_k = x.shape[-1]
    padded_k = ((original_k + alignment - 1) // alignment) * alignment
    if padded_k == original_k:
        out = x.contiguous()
    else:
        # ``torch.empty`` + targeted tail zero is cheaper than ``torch.zeros``
        # because we avoid an unnecessary full zero pass over ``original_k``
        # elements before overwriting them.
        out = torch.empty(
            *x.shape[:-1], padded_k, dtype=x.dtype, device=x.device
        )
        out[..., :original_k].copy_(x)
        out[..., original_k:].zero_()
    out.aiter_original_k = original_k
    out.aiter_padded_k = padded_k
    out.aiter_k_padding = padded_k - original_k
    return out


def _propagate_bpreshuffle_padding_attrs(
    src: torch.Tensor, dst: torch.Tensor
) -> None:
    for attr in _BPRESHUFFLE_PADDING_ATTRS:
        if hasattr(src, attr):
            setattr(dst, attr, getattr(src, attr))


def shuffle_weight(
    x: torch.Tensor,
    layout=(16, 16),
    use_int4=False,
    is_guinterleave=False,
    gate_up: bool = False,
) -> torch.Tensor:
    x_type = x.dtype
    if hasattr(torch, "float4_e2m1fn_x2") and x_type == torch.float4_e2m1fn_x2:
        x = x.view(torch.uint8)

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
        _propagate_bpreshuffle_padding_attrs(x, x_)
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
    _propagate_bpreshuffle_padding_attrs(x, x_)
    return x_


def is_bpreshuffle_kernel_tuned(
    n: int,
    k: int,
    dtype: torch.dtype,
    libtype: Optional[str] = None,
    m_candidates: Optional[list] = None,
) -> bool:
    """Return True if the bpreshuffle tuned CSV has at least one entry that
    matches ``(N=n, K=k, q_dtype_w=dtype)`` for the active GFX/CU.

    Used by SGLang at weight-load time to decide whether to pad-and-shuffle
    a weight (fast path) or fall back to the unshuffled
    ``gemm_a8w8_CK`` kernel for that layer. Probing a representative set of
    M values mirrors the M sweep used during tuning so layers whose tuned M
    range is sparse still resolve to True.

    ``libtype`` filters by the ``libtype`` column (e.g. ``"ck"``,
    ``"cktile"``, ``"flydsl"``). Default of ``None`` returns True for any
    libtype.
    """
    # Imported lazily to avoid an import cycle: ``gemm_op_a8w8`` imports from
    # ``..jit.core`` which transitively imports many AIter ops.
    from .gemm_op_a8w8 import get_GEMM_config_with_quant_type
    from ..jit.core import AITER_CONFIGS

    if m_candidates is None:
        # Mirrors the SGLang graph-capture batch-size sweep documented in the
        # GLM-4.6V FP8 padding plan.
        m_candidates = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]

    for m in m_candidates:
        config = get_GEMM_config_with_quant_type(
            m,
            n,
            k,
            dtype,
            AITER_CONFIGS.AITER_CONFIG_GEMM_A8W8_BPRESHUFFLE_FILE,
        )
        if config is None:
            continue
        if libtype is None or config.get("libtype") == libtype:
            return True
    return False


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
