# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Pure-Torch correctness oracle for AITER IQ2R weights.

The implementation follows the physical byte layout consumed by the gfx950
scaled-FP8 MFMA operand.  It is deliberately explicit and optimized for audit
clarity rather than serving speed.
"""

from __future__ import annotations

import torch
from torch import Tensor

from .iq2r_format import (
    IQ2R_ATOMS_PER_TRIPLET,
    IQ2R_ATOM_TWO_METADATA_OFFSET,
    IQ2R_ATOM_TWO_RECORD_BYTES,
    IQ2R_ATOM_TWO_RECORDS_OFFSET,
    IQ2R_CODEBOOK_BYTES,
    IQ2R_CODEBOOK_ENTRIES,
    IQ2R_LANE_RECORD_BYTES,
    IQ2R_LANES_PER_TILE,
    IQ2R_N_BLOCKS_PER_GROUP,
    IQ2R_SCALE_BLOCK,
    IQ2R_TILE_K,
    IQ2R_TILE_N,
    IQ2R_TRIPLET_BYTES,
    IQ2R_VECTOR_SIZE,
    IQ2R_GROUP_BYTES,
    IQ2RMetadata,
    iq2r_validate_expert_weights,
)


def iq2r_physical_tile(n_block: int, k_tile: int, k_tiles: int) -> int:
    """Map one logical ``(N/16,K/128)`` tile to physical six-block order."""

    if not (0 <= n_block):
        raise ValueError(f"n_block must be non-negative, got {n_block}")
    if not (0 <= k_tile < k_tiles):
        raise ValueError(f"k_tile {k_tile} is outside [0,{k_tiles})")
    return (
        (n_block // IQ2R_N_BLOCKS_PER_GROUP) * k_tiles + k_tile
    ) * IQ2R_N_BLOCKS_PER_GROUP + n_block % IQ2R_N_BLOCKS_PER_GROUP


def iq2r_tile_views(
    packed: Tensor, n_block: int, k_tile: int, k_tiles: int
) -> tuple[Tensor, Tensor]:
    """Return the lane records and shared triplet metadata for one tile."""

    if packed.dtype != torch.uint8 or packed.ndim != 1 or not packed.is_contiguous():
        raise ValueError(
            "packed data must be a contiguous one-dimensional uint8 tensor"
        )
    tile = iq2r_physical_tile(n_block, k_tile, k_tiles)
    block_in_group = tile % IQ2R_N_BLOCKS_PER_GROUP
    triplet = block_in_group // IQ2R_ATOMS_PER_TRIPLET
    atom = block_in_group % IQ2R_ATOMS_PER_TRIPLET
    triplet_base = (
        tile // IQ2R_N_BLOCKS_PER_GROUP
    ) * IQ2R_GROUP_BYTES + triplet * IQ2R_TRIPLET_BYTES
    if atom < 2:
        record_base = triplet_base + atom * IQ2R_LANE_RECORD_BYTES
        record_stride = 2 * IQ2R_LANE_RECORD_BYTES
    else:
        record_base = triplet_base + IQ2R_ATOM_TWO_RECORDS_OFFSET
        record_stride = IQ2R_ATOM_TWO_RECORD_BYTES
    records = torch.as_strided(
        packed,
        size=(IQ2R_LANES_PER_TILE, IQ2R_LANE_RECORD_BYTES),
        stride=(record_stride, 1),
        storage_offset=record_base,
    )
    metadata = torch.as_strided(
        packed,
        size=(IQ2R_LANES_PER_TILE, IQ2R_ATOMS_PER_TRIPLET),
        stride=(IQ2R_ATOM_TWO_RECORD_BYTES, 1),
        storage_offset=(
            triplet_base + IQ2R_ATOM_TWO_RECORDS_OFFSET + IQ2R_ATOM_TWO_METADATA_OFFSET
        ),
    )
    return records, metadata


def iq2r_metadata_word(metadata: Tensor, lane: int) -> int:
    """Load one lane's three atom metadata bytes as a little-endian word."""

    return (
        int(metadata[lane, 0])
        | (int(metadata[lane, 1]) << 8)
        | (int(metadata[lane, 2]) << 16)
    )


def iq2r_set_metadata_word(metadata: Tensor, lane: int, value: int) -> None:
    """Store the low 24 bits of a metadata word into its strided view."""

    if value < 0 or value >= (1 << 24):
        raise ValueError(f"metadata word must fit in 24 bits, got {value}")
    metadata[lane, 0] = value & 0xFF
    metadata[lane, 1] = (value >> 8) & 0xFF
    metadata[lane, 2] = (value >> 16) & 0xFF


def _materialize_one(
    data: Tensor,
    auxiliary: Tensor,
    metadata: IQ2RMetadata,
) -> Tensor:
    data_cpu = data.detach().to(device="cpu", dtype=torch.uint8).contiguous()
    auxiliary_cpu = auxiliary.detach().to(device="cpu", dtype=torch.uint8).contiguous()
    codebook = (
        auxiliary_cpu[:IQ2R_CODEBOOK_BYTES]
        .view(torch.float8_e4m3fn)
        .float()
        .reshape(IQ2R_CODEBOOK_ENTRIES, IQ2R_VECTOR_SIZE)
    )
    base_exponents = auxiliary_cpu[IQ2R_CODEBOOK_BYTES:]
    output = torch.empty((metadata.logical_n, metadata.padded_k), dtype=torch.float32)

    for n_block in range(metadata.n_blocks):
        atom = n_block % IQ2R_ATOMS_PER_TRIPLET
        for k_tile in range(metadata.k_tiles):
            records, atom_metadata = iq2r_tile_views(
                data_cpu, n_block, k_tile, metadata.k_tiles
            )
            indices = records[:, :4].to(torch.int64).clone()
            signs = records[:, 4:]
            for lane in range(IQ2R_LANES_PER_TILE):
                word = iq2r_metadata_word(atom_metadata, lane)
                high_bits = (word >> (atom * 8)) & 0x0F
                for codeword in range(4):
                    indices[lane, codeword] |= ((high_bits >> codeword) & 1) << 8

            for lane in range(IQ2R_LANES_PER_TILE):
                row = n_block * IQ2R_TILE_N + lane % IQ2R_TILE_N
                group = lane // IQ2R_TILE_N
                fragment = codebook[indices[lane]].reshape(4 * IQ2R_VECTOR_SIZE).clone()
                for codeword in range(4):
                    sign_byte = int(signs[lane, codeword])
                    for bit in range(IQ2R_VECTOR_SIZE):
                        if sign_byte & (1 << bit):
                            fragment[codeword * IQ2R_VECTOR_SIZE + bit].neg_()

                for half in range(2):
                    logical_block = group // 2 + half * 2
                    scale_lane = logical_block * IQ2R_TILE_N + lane % IQ2R_TILE_N
                    word = iq2r_metadata_word(atom_metadata, scale_lane)
                    delta = (word >> (atom * 8 + 4)) & 0x0F
                    exponent = int(base_exponents[n_block]) + delta
                    scale = 0.0 if exponent == 0 else 2.0 ** (exponent - 127)
                    start = (
                        k_tile * IQ2R_TILE_K
                        + logical_block * IQ2R_SCALE_BLOCK
                        + (group % 2) * 16
                    )
                    output[row, start : start + 16] = (
                        fragment[half * 16 : (half + 1) * 16] * scale
                    )
    return output[:, : metadata.logical_k]


@torch.no_grad()
def iq2r_materialize(
    data: Tensor,
    auxiliary: Tensor,
    metadata: IQ2RMetadata,
    *,
    output_dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Materialize stacked IQ2R weights as dense ``[E,N,K]`` on the CPU."""

    if not output_dtype.is_floating_point:
        raise TypeError(f"output_dtype must be floating point, got {output_dtype}")
    iq2r_validate_expert_weights(data, auxiliary, metadata)
    decoded = torch.stack(
        [
            _materialize_one(data[expert], auxiliary[expert], metadata)
            for expert in range(data.shape[0])
        ]
    )
    return decoded.to(output_dtype)


@torch.no_grad()
def iq2r_dense_reference(
    activations: Tensor,
    data: Tensor,
    auxiliary: Tensor,
    metadata: IQ2RMetadata,
    *,
    expert: int = 0,
    bias: Tensor | None = None,
    output_dtype: torch.dtype | None = None,
) -> Tensor:
    """Dense multiplication oracle for one expert's materialized weights."""

    if activations.ndim != 2 or activations.shape[1] != metadata.logical_k:
        raise ValueError(
            f"activations must have shape [M,{metadata.logical_k}], "
            f"got {tuple(activations.shape)}"
        )
    if not (0 <= expert < data.shape[0]):
        raise ValueError(f"expert {expert} is outside [0,{data.shape[0]})")
    weights = iq2r_materialize(data, auxiliary, metadata)[expert]
    result = activations.float().cpu() @ weights.T
    if bias is not None:
        if bias.ndim == 2:
            bias = bias[expert]
        if tuple(bias.shape) != (metadata.logical_n,):
            raise ValueError(
                f"bias must have shape [{metadata.logical_n}] or [E,{metadata.logical_n}]"
            )
        result += bias.float().cpu()
    return result.to(output_dtype or activations.dtype)


__all__ = [
    "iq2r_dense_reference",
    "iq2r_materialize",
    "iq2r_metadata_word",
    "iq2r_physical_tile",
    "iq2r_set_metadata_word",
    "iq2r_tile_views",
]
