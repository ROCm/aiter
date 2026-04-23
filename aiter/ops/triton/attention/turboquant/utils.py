# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""
Bit-packing utilities for TurboQuant.

Codebook indices must be packed tightly into uint8 tensors to achieve
the target compression ratios:

  bits=2 → 4 indices per byte  (2× vs uint8 storage)
  bits=3 → ~2.67 indices/byte  (stored in 3-byte groups of 8 indices)
  bits=4 → 2 indices per byte  (2× vs uint8 storage)

API:
    packed = pack_indices(indices, bits)
    indices = unpack_indices(packed, bits, original_d)

Shapes:
    indices : (..., d)          torch.uint8 or int64  (values in [0, 2^bits))
    packed  : (..., d_packed)   torch.uint8
    d_packed:
        bits=2 → ceil(d/4)
        bits=3 → ceil(d/8)*3   (8 indices → 3 bytes)
        bits=4 → ceil(d/2)

All operations are vectorized (no Python loops over elements).
"""

from __future__ import annotations

import math
import torch

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def pack_indices(indices: torch.Tensor, bits: int) -> torch.Tensor:
    """
    Pack integer indices into a uint8 tensor.

    Args:
        indices: (..., d) tensor with values in [0, 2**bits).
                 Accepted dtypes: uint8, int32, int64.
        bits:    Bits per index (2, 3, or 4).

    Returns:
        packed: (..., d_packed) uint8 tensor.
    """
    if bits == 2:
        return _pack_2bit(indices)
    elif bits == 3:
        return _pack_3bit(indices)
    elif bits == 4:
        return _pack_4bit(indices)
    else:
        raise ValueError(f"pack_indices: bits must be 2, 3, or 4; got {bits}")


def unpack_indices(packed: torch.Tensor, bits: int, d: int) -> torch.Tensor:
    """
    Unpack a uint8 tensor back into integer indices.

    Args:
        packed: (..., d_packed) uint8 tensor produced by pack_indices.
        bits:   Bits per index (must match what was used to pack).
        d:      Original number of indices (needed for 3-bit to trim padding).

    Returns:
        indices: (..., d) int64 tensor with values in [0, 2**bits).
    """
    if bits == 2:
        return _unpack_2bit(packed, d)
    elif bits == 3:
        return _unpack_3bit(packed, d)
    elif bits == 4:
        return _unpack_4bit(packed, d)
    else:
        raise ValueError(f"unpack_indices: bits must be 2, 3, or 4; got {bits}")


def packed_size(d: int, bits: int) -> int:
    """Return the number of uint8 bytes needed to store d indices at `bits` each."""
    if bits == 2:
        return math.ceil(d / 4)
    elif bits == 3:
        return math.ceil(d / 8) * 3
    elif bits == 4:
        return math.ceil(d / 2)
    else:
        raise ValueError(f"packed_size: bits must be 2, 3, or 4; got {bits}")


def compression_ratio(head_dim: int, key_bits: int, value_bits: int) -> float:
    """
    Estimate the KV compression ratio vs bf16 baseline.

    Accounts for:
      - MSE indices (key_bits per coord)
      - 1 fp16 norm per vector for keys
      - value indices (value_bits per coord)
      - 2 fp16 per group (scale + zero) for values

    Does NOT include the shared Π/S matrices (amortized across sequence).

    Args:
        head_dim:    d
        key_bits:    bits used for key MSE indices
        value_bits:  bits used for value group quantization

    Returns:
        ratio: (original bytes) / (compressed bytes), e.g. 4.4 means 4.4× smaller.
    """
    bf16_bytes_per_vector = head_dim * 2  # 2 bytes per bf16

    # Keys: (key_bits * d) bits + 2 bytes norm
    key_compressed = math.ceil(head_dim * key_bits / 8) + 2

    # Values: (value_bits * d) bits + 2*(d//32) bytes for scales+zeros
    group_size = 32
    n_groups = head_dim // group_size
    val_compressed = (
        math.ceil(head_dim * value_bits / 8) + n_groups * 2 * 2
    )  # scale+zero fp16

    total_compressed = key_compressed + val_compressed
    total_original = 2 * bf16_bytes_per_vector  # K + V
    return total_original / total_compressed


# ---------------------------------------------------------------------------
# 2-bit packing: 4 indices per byte
# ---------------------------------------------------------------------------


def _pack_2bit(indices: torch.Tensor) -> torch.Tensor:
    """Pack 2-bit indices: 4 per byte, LSB-first."""
    d = indices.shape[-1]
    batch_shape = indices.shape[:-1]
    idx = indices.to(torch.uint8)

    # Pad to multiple of 4
    pad = (4 - d % 4) % 4
    if pad:
        idx = torch.nn.functional.pad(idx, (0, pad))
    d_padded = idx.shape[-1]

    # Reshape to (..., d//4, 4) and pack
    idx = idx.reshape(*batch_shape, d_padded // 4, 4)
    # bits 0-1, 2-3, 4-5, 6-7
    packed = (
        (idx[..., 0] & 0x03)
        | ((idx[..., 1] & 0x03) << 2)
        | ((idx[..., 2] & 0x03) << 4)
        | ((idx[..., 3] & 0x03) << 6)
    )
    return packed.to(torch.uint8)


def _unpack_2bit(packed: torch.Tensor, d: int) -> torch.Tensor:
    """Unpack 2-bit indices from uint8 bytes."""
    batch_shape = packed.shape[:-1]
    p = packed.to(torch.int64)
    # Extract 4 indices per byte
    i0 = p & 0x03
    i1 = (p >> 2) & 0x03
    i2 = (p >> 4) & 0x03
    i3 = (p >> 6) & 0x03
    # Stack: (..., n_bytes, 4) → (..., n_bytes*4)
    indices = torch.stack([i0, i1, i2, i3], dim=-1).reshape(*batch_shape, -1)
    return indices[..., :d]


# ---------------------------------------------------------------------------
# 4-bit packing: 2 indices per byte
# ---------------------------------------------------------------------------


def _pack_4bit(indices: torch.Tensor) -> torch.Tensor:
    """Pack 4-bit indices: 2 per byte, lower nibble first."""
    d = indices.shape[-1]
    batch_shape = indices.shape[:-1]
    idx = indices.to(torch.uint8)

    # Pad to multiple of 2
    pad = d % 2
    if pad:
        idx = torch.nn.functional.pad(idx, (0, 1))
    d_padded = idx.shape[-1]

    idx = idx.reshape(*batch_shape, d_padded // 2, 2)
    packed = (idx[..., 0] & 0x0F) | ((idx[..., 1] & 0x0F) << 4)
    return packed.to(torch.uint8)


def _unpack_4bit(packed: torch.Tensor, d: int) -> torch.Tensor:
    """Unpack 4-bit indices from uint8 bytes."""
    batch_shape = packed.shape[:-1]
    p = packed.to(torch.int64)
    lo = p & 0x0F
    hi = (p >> 4) & 0x0F
    indices = torch.stack([lo, hi], dim=-1).reshape(*batch_shape, -1)
    return indices[..., :d]


# ---------------------------------------------------------------------------
# 3-bit packing: 8 indices → 3 bytes
# ---------------------------------------------------------------------------
# Layout (8 indices a0..a7, each 3 bits):
#   Byte 0: a0[2:0] | a1[2:0]<<3 | a2[1:0]<<6
#   Byte 1: a2[2:2] | a3[2:0]<<1 | a4[2:0]<<4 | a5[1:0]<<7
#   Byte 2: a5[2:2] | a6[2:0]<<1 | a7[2:0]<<4
# This is the standard 3-bit tight packing across a group of 8.


def _pack_3bit(indices: torch.Tensor) -> torch.Tensor:
    """Pack 3-bit indices: 8 per 3 bytes."""
    d = indices.shape[-1]
    batch_shape = indices.shape[:-1]
    idx = indices.to(torch.int32)

    # Pad to multiple of 8
    pad = (8 - d % 8) % 8
    if pad:
        idx = torch.nn.functional.pad(idx, (0, pad))
    d_padded = idx.shape[-1]

    # Reshape to (..., groups_of_8, 8)
    idx = idx.reshape(*batch_shape, d_padded // 8, 8)
    a = [idx[..., i] for i in range(8)]

    b0 = (a[0] & 0x07) | ((a[1] & 0x07) << 3) | ((a[2] & 0x03) << 6)
    b1 = (
        ((a[2] >> 2) & 0x01)
        | ((a[3] & 0x07) << 1)
        | ((a[4] & 0x07) << 4)
        | ((a[5] & 0x01) << 7)
    )
    b2 = ((a[5] >> 1) & 0x03) | ((a[6] & 0x07) << 2) | ((a[7] & 0x07) << 5)

    # Stack as (..., groups, 3) then flatten to (..., groups*3)
    packed = torch.stack([b0, b1, b2], dim=-1).to(torch.uint8)
    return packed.reshape(*batch_shape, -1)


def _unpack_3bit(packed: torch.Tensor, d: int) -> torch.Tensor:
    """Unpack 3-bit indices from uint8 bytes (groups of 3 bytes → 8 indices)."""
    batch_shape = packed.shape[:-1]
    n_bytes = packed.shape[-1]
    assert n_bytes % 3 == 0, "3-bit packed tensor must have length divisible by 3"
    n_groups = n_bytes // 3

    p = packed.to(torch.int64).reshape(*batch_shape, n_groups, 3)
    b0, b1, b2 = p[..., 0], p[..., 1], p[..., 2]

    a0 = b0 & 0x07
    a1 = (b0 >> 3) & 0x07
    a2 = ((b0 >> 6) & 0x03) | ((b1 & 0x01) << 2)
    a3 = (b1 >> 1) & 0x07
    a4 = (b1 >> 4) & 0x07
    a5 = ((b1 >> 7) & 0x01) | ((b2 & 0x03) << 1)
    a6 = (b2 >> 2) & 0x07
    a7 = (b2 >> 5) & 0x07

    indices = torch.stack([a0, a1, a2, a3, a4, a5, a6, a7], dim=-1)
    indices = indices.reshape(*batch_shape, n_groups * 8)
    return indices[..., :d]
