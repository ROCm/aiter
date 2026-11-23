# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Storage layout of AITER's native-basis IQ2R weights.

CPU-only, with no compiled extension, so checkpoint tooling can size and
slice IQ2R byte buffers before touching a GPU.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

IQ2R_ACTIVATION_BASIS = "native"

IQ2R_CODEBOOK_ENTRIES = 512
IQ2R_VECTOR_SIZE = 8
IQ2R_SCALE_BLOCK = 32
IQ2R_TILE_N = 16
IQ2R_TILE_K = 128
IQ2R_LANE_RECORD_BYTES = 8
IQ2R_LANES_PER_TILE = 64
IQ2R_N_BLOCKS_PER_GROUP = 6
IQ2R_ATOMS_PER_TRIPLET = 3
IQ2R_TRIPLETS_PER_GROUP = IQ2R_N_BLOCKS_PER_GROUP // IQ2R_ATOMS_PER_TRIPLET
IQ2R_ATOM_PAIR_RECORDS_BYTES = IQ2R_LANES_PER_TILE * 2 * IQ2R_LANE_RECORD_BYTES
IQ2R_ATOM_TWO_RECORD_BYTES = 12
IQ2R_ATOM_TWO_RECORDS_OFFSET = IQ2R_ATOM_PAIR_RECORDS_BYTES
IQ2R_ATOM_TWO_METADATA_OFFSET = IQ2R_LANE_RECORD_BYTES
IQ2R_TRIPLET_BYTES = IQ2R_ATOM_PAIR_RECORDS_BYTES + (
    IQ2R_LANES_PER_TILE * IQ2R_ATOM_TWO_RECORD_BYTES
)
IQ2R_GROUP_BYTES = IQ2R_TRIPLETS_PER_GROUP * IQ2R_TRIPLET_BYTES
IQ2R_CODEBOOK_BYTES = IQ2R_CODEBOOK_ENTRIES * IQ2R_VECTOR_SIZE

# A 1,024-entry codebook takes 10-bit indices: each 8-weight index gains a
# tenth bit, four per atom and lane. Atoms 0 and 1 keep theirs in the atom-two
# record's spare fourth metadata byte (atom a's codeword j at bit 4a + j).
# Atom 2's go to a byte per lane after every two K tiles (256 columns, the
# tensor-parallel slicing unit): records are stored in slabs of two K tiles'
# six-block groups, then 64 bytes per triplet (slab tile t, codeword j at bit
# 4t + j). A 512-entry slab is one K tile.
IQ2R_WIDE_CODEBOOK_ENTRIES = 1024
IQ2R_CODEBOOK_SIZES = (IQ2R_CODEBOOK_ENTRIES, IQ2R_WIDE_CODEBOOK_ENTRIES)
IQ2R_WIDE_SLAB_K_TILES = 2
IQ2R_WIDE_TENTH_BYTE = IQ2R_ATOM_TWO_METADATA_OFFSET + IQ2R_ATOMS_PER_TRIPLET
IQ2R_WIDE_EXT_OFFSET = IQ2R_WIDE_SLAB_K_TILES * IQ2R_GROUP_BYTES
IQ2R_WIDE_SLAB_BYTES = IQ2R_WIDE_EXT_OFFSET + (
    IQ2R_TRIPLETS_PER_GROUP * IQ2R_LANES_PER_TILE
)


def _require_int(name: str, value: int, *, positive: bool = False) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int, got {type(value).__name__}")
    if positive and value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")


def iq2r_index_bits(codebook_entries: int) -> int:
    """Index width of a codebook size: 9 bits for 512 entries, 10 for 1,024."""

    if codebook_entries not in IQ2R_CODEBOOK_SIZES:
        raise ValueError(
            f"IQ2R codebooks have {IQ2R_CODEBOOK_SIZES} entries, got {codebook_entries}"
        )
    return codebook_entries.bit_length() - 1


def iq2r_slab_k_tiles(codebook_entries: int = IQ2R_CODEBOOK_ENTRIES) -> int:
    """K tiles per stored slab of a six-block group: 1 (512 entries) or 2."""

    iq2r_index_bits(codebook_entries)
    if codebook_entries == IQ2R_WIDE_CODEBOOK_ENTRIES:
        return IQ2R_WIDE_SLAB_K_TILES
    return 1


def iq2r_slab_bytes(codebook_entries: int = IQ2R_CODEBOOK_ENTRIES) -> int:
    """Bytes of one slab: a six-block group over iq2r_slab_k_tiles K tiles."""

    if iq2r_slab_k_tiles(codebook_entries) == 1:
        return IQ2R_GROUP_BYTES
    return IQ2R_WIDE_SLAB_BYTES


def iq2r_k_tiles(k: int) -> int:
    """Number of stored K=128 tiles for a logical contraction width."""

    _require_int("k", k, positive=True)
    if k % IQ2R_SCALE_BLOCK:
        raise ValueError(f"IQ2R requires K divisible by {IQ2R_SCALE_BLOCK}, got {k}")
    return (k + IQ2R_TILE_K - 1) // IQ2R_TILE_K


def iq2r_n_blocks(n: int) -> int:
    """Number of logical N=16 blocks."""

    _require_int("n", n, positive=True)
    if n % IQ2R_TILE_N:
        raise ValueError(f"IQ2R requires N divisible by {IQ2R_TILE_N}, got {n}")
    return n // IQ2R_TILE_N


def iq2r_physical_n_blocks(n: int) -> int:
    """N-block count rounded to the six-block physical storage group."""

    logical = iq2r_n_blocks(n)
    return (
        (logical + IQ2R_N_BLOCKS_PER_GROUP - 1) // IQ2R_N_BLOCKS_PER_GROUP
    ) * IQ2R_N_BLOCKS_PER_GROUP


def iq2r_data_bytes(
    n: int, k: int, codebook_entries: int = IQ2R_CODEBOOK_ENTRIES
) -> int:
    physical_n_blocks = iq2r_physical_n_blocks(n)
    slab_k_tiles = iq2r_slab_k_tiles(codebook_entries)
    k_tiles = iq2r_k_tiles(k)
    if k_tiles % slab_k_tiles:
        raise ValueError(
            f"IQ2R with {codebook_entries} codebook entries requires K divisible by "
            f"{slab_k_tiles * IQ2R_TILE_K}, got {k}"
        )
    return (
        physical_n_blocks
        // IQ2R_N_BLOCKS_PER_GROUP
        * (k_tiles // slab_k_tiles)
        * iq2r_slab_bytes(codebook_entries)
    )


@dataclass(frozen=True, slots=True)
class IQ2RMetadata:
    """Shape of one IQ2R expert projection (``N`` outputs, ``K`` inputs) and
    its codebook size."""

    logical_n: int
    logical_k: int
    codebook_entries: int = IQ2R_CODEBOOK_ENTRIES

    def __post_init__(self) -> None:
        # Applies the type, divisibility and codebook-size checks.
        iq2r_data_bytes(self.logical_n, self.logical_k, self.codebook_entries)

    @property
    def n_blocks(self) -> int:
        return iq2r_n_blocks(self.logical_n)

    @property
    def physical_n_blocks(self) -> int:
        return iq2r_physical_n_blocks(self.logical_n)

    @property
    def k_tiles(self) -> int:
        return iq2r_k_tiles(self.logical_k)

    @property
    def slab_k_tiles(self) -> int:
        return iq2r_slab_k_tiles(self.codebook_entries)

    @property
    def slab_bytes(self) -> int:
        return iq2r_slab_bytes(self.codebook_entries)

    @property
    def data_bytes(self) -> int:
        return iq2r_data_bytes(self.logical_n, self.logical_k, self.codebook_entries)


def iq2r_slice_input_data(
    data: Tensor,
    metadata: IQ2RMetadata,
    start: int,
    length: int,
) -> Tensor:
    """Extract an exact contiguous K-tile slice of stacked IQ2R data (whole slabs:
    256-column aligned for a 1,024-entry codebook)."""

    _validate_byte_matrix("data", data, metadata.data_bytes)
    _require_int("start", start)
    _require_int("length", length, positive=True)
    align = IQ2R_TILE_K * metadata.slab_k_tiles
    if (
        start < 0
        or start % align
        or length % align
        or start + length > metadata.logical_k
    ):
        raise ValueError(
            f"IQ2R input slice must be in range and aligned to {align} columns"
        )
    target = IQ2RMetadata(
        logical_n=metadata.logical_n,
        logical_k=length,
        codebook_entries=metadata.codebook_entries,
    )
    experts = data.shape[0]
    groups = metadata.physical_n_blocks // IQ2R_N_BLOCKS_PER_GROUP
    slabs = metadata.k_tiles // metadata.slab_k_tiles
    source = data.view(experts, groups, slabs, metadata.slab_bytes)
    slab_start = start // align
    # ``contiguous()`` may return the original narrow view when its strides are
    # already dense, retaining a non-zero storage offset.  IQ2R consumers treat
    # byte zero as the first packed tile, so force a fresh zero-offset storage.
    return (
        source[:, :, slab_start : slab_start + length // align]
        .clone(memory_format=torch.contiguous_format)
        .view(experts, target.data_bytes)
    )


def _validate_byte_matrix(name: str, value: Tensor, expected_bytes: int) -> None:
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.dtype != torch.uint8:
        raise TypeError(f"{name} must have dtype uint8, got {value.dtype}")
    if value.ndim != 2:
        raise ValueError(
            f"{name} must have shape [experts,{expected_bytes}], got {tuple(value.shape)}"
        )
    if value.shape[1] != expected_bytes:
        raise ValueError(
            f"{name} has {value.shape[1]} bytes per expert, expected {expected_bytes}"
        )
    if not value.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


__all__ = [name for name in globals() if name.startswith("IQ2R_")] + [
    "IQ2RMetadata",
    "iq2r_data_bytes",
    "iq2r_index_bits",
    "iq2r_k_tiles",
    "iq2r_n_blocks",
    "iq2r_physical_n_blocks",
    "iq2r_slab_bytes",
    "iq2r_slab_k_tiles",
    "iq2r_slice_input_data",
]
