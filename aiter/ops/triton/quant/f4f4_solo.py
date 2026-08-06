# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Production GPU packers for the dedicated f4f4-solo FMHA tensor ABI."""

import torch

from aiter.ops.triton._triton_kernels.quant.f4f4_solo import (
    pack_f4f4_solo_k_lds_kernel,
    quantize_f4f4_solo_v_lds_kernel,
)
from aiter.ops.triton.moe.quant_moe import downcast_to_mxfp

_TILE = 128
_TILE_BYTES = 8192
_V_SCALE_TILE_BYTES = 512
_K_SCALE_SLACK_BYTES = 64
_V_BYTE_MAP_CACHE = {}


def _swap_token_bits_2_5(token: int) -> int:
    bit2 = (token >> 2) & 1
    bit5 = (token >> 5) & 1
    return (token & 0b1011011) | (bit5 << 2) | (bit2 << 5)


def _v_source_token(selector: int, nibble: int) -> int:
    """Mirror one nibble of ``_v_lds_order_gather_index_fp4``."""
    k_half = selector // 32
    lane_half = (selector // 16) % 2
    byte_in_lane = selector % 16
    elem = (byte_in_lane // 4) * 8 + (byte_in_lane % 4) * 2 + nibble
    token = k_half * 64 + (elem // 4) * 8 + lane_half * 4 + elem % 4
    return _swap_token_bits_2_5(token)


def _make_v_byte_map() -> tuple[int, ...]:
    """Select the 16 LDS bytes owned by each natural 32-token MX block."""
    result = []
    for kv_block in range(4):
        selected = []
        for selector in range(64):
            token_lo = _v_source_token(selector, 0)
            token_hi = _v_source_token(selector, 1)
            if token_lo // 32 == kv_block:
                if token_hi // 32 != kv_block:
                    raise AssertionError("an LDS byte crosses V MX blocks")
                selected.append(selector)
        if len(selected) != 16:
            raise AssertionError(f"V MX block {kv_block} owns {len(selected)} bytes")
        result.extend(selected)
    return tuple(result)


_V_BYTE_MAP = _make_v_byte_map()


def _v_byte_map(device: torch.device) -> torch.Tensor:
    byte_map = _V_BYTE_MAP_CACHE.get(device)
    if byte_map is None:
        byte_map = torch.tensor(_V_BYTE_MAP, dtype=torch.int32, device=device)
        _V_BYTE_MAP_CACHE[device] = byte_map
    return byte_map


def _check_input(x: torch.Tensor, name: str) -> tuple[int, int, int]:
    if x.ndim != 4 or x.shape[-1] != 128:
        raise ValueError(f"{name} must have shape [b, s, h, 128], got {tuple(x.shape)}")
    if x.shape[1] <= 0:
        raise ValueError(f"{name} sequence length must be positive")
    if not x.is_cuda:
        raise ValueError(f"{name} must be on a GPU")
    if x.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError(f"{name} must be float16 or bfloat16, got {x.dtype}")
    return x.shape[0], x.shape[1], x.shape[2]


def quantize_f4f4_solo_k(
    k_bshd: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize and pack K for the solo C0-only coalesced load.

    Returns:
      * ``k_view``: uint8 ``[b, sk, h_kv, 64]`` with strides
        ``(h_kv*nT*8192, 64, nT*8192, 1)``.
      * ``k_scale``: uint8 E8M0 ``[b, sk, h_kv, 4]``.

    MX arithmetic is delegated to the same ``downcast_to_mxfp`` path used by
    ``sage_quant_f4f4``. The Triton kernel only performs the within-32 channel
    permutation and fused nibble repack/scatter into the 8192-byte tile image.
    """
    b, sk, h_kv = _check_input(k_bshd, "k")
    n_tiles = (sk + _TILE - 1) // _TILE

    canonical, scale = downcast_to_mxfp(k_bshd, torch.uint8, axis=-1)
    data_bytes = b * h_kv * n_tiles * _TILE_BYTES
    data = torch.empty(data_bytes, dtype=torch.uint8, device=k_bshd.device)
    pack_f4f4_solo_k_lds_kernel[(b * h_kv * n_tiles * 4,)](
        canonical,
        data,
        canonical.stride(0),
        canonical.stride(1),
        canonical.stride(2),
        canonical.stride(3),
        h_kv,
        n_tiles,
        sk,
        num_warps=4,
        num_stages=1,
    )

    head_stride = n_tiles * _TILE_BYTES
    k_view = torch.as_strided(
        data,
        (b, sk, h_kv, 64),
        (h_kv * head_stride, 64, head_stride, 1),
    )

    # The solo K-scale gather may issue its final dword at a +1-byte source
    # offset. Keep mapped trailing storage while preserving the exact API shape.
    scale_storage = torch.empty(
        scale.numel() + _K_SCALE_SLACK_BYTES,
        dtype=torch.uint8,
        device=scale.device,
    )
    scale_storage[: scale.numel()].copy_(scale.reshape(-1))
    scale_storage[scale.numel() :].zero_()
    k_scale = scale_storage[: scale.numel()].view(b, sk, h_kv, 4)
    return k_view, k_scale


def quantize_f4f4_solo_v(
    v_bshd: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused MXFP4 quantization and solo LDS-order packing for V.

    Returns:
      * ``v_view``: uint8 descriptor ``[b, sk, h_kv, 128]`` with strides
        ``(h_kv*nT*8192, 64, nT*8192, 1)`` over 8192 bytes per 128-token tile.
      * ``v_descale``: uint8 E8M0 image ``[b, h_kv, nT*512]``.

    Ragged tiles edge-pad with the final valid token. Both backing allocations
    include one trailing tile so the solo software pipeline's safe, unconsumed
    over-prefetch remains mapped.
    """
    b, sk, h_kv = _check_input(v_bshd, "v")
    n_tiles = (sk + _TILE - 1) // _TILE
    head_stride = n_tiles * _TILE_BYTES
    data_bytes = b * h_kv * head_stride

    data = torch.empty(
        data_bytes + _TILE_BYTES,
        dtype=torch.uint8,
        device=v_bshd.device,
    )
    data[data_bytes:].zero_()

    scale_bytes = b * h_kv * n_tiles * _V_SCALE_TILE_BYTES
    scale_storage = torch.empty(
        scale_bytes + _V_SCALE_TILE_BYTES,
        dtype=torch.uint8,
        device=v_bshd.device,
    )
    scale_storage[scale_bytes:].zero_()
    v_descale = scale_storage[:scale_bytes].view(b, h_kv, n_tiles * _V_SCALE_TILE_BYTES)

    quantize_f4f4_solo_v_lds_kernel[(b * h_kv * n_tiles * 16,)](
        v_bshd,
        data,
        v_descale,
        _v_byte_map(v_bshd.device),
        v_bshd.stride(0),
        v_bshd.stride(1),
        v_bshd.stride(2),
        v_bshd.stride(3),
        h_kv,
        n_tiles,
        sk,
        num_warps=4,
        num_stages=1,
    )

    v_view = torch.as_strided(
        data,
        (b, sk, h_kv, 128),
        (h_kv * head_stride, 64, head_stride, 1),
    )
    return v_view, v_descale


__all__ = ["quantize_f4f4_solo_k", "quantize_f4f4_solo_v"]
