# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch

from aiter.ops.iq2r_format import (
    IQ2R_ATOMS_PER_TRIPLET,
    IQ2R_CODEBOOK_BYTES,
    IQ2R_GROUP_BYTES,
    IQ2R_TRIPLET_BYTES,
    IQ2R_VECTOR_SIZE,
    IQ2RMetadata,
)
from aiter.ops.iq2r_reference import (
    iq2r_materialize,
    iq2r_metadata_word,
    iq2r_physical_tile,
    iq2r_set_metadata_word,
    iq2r_tile_views,
)


def _empty_fixture(n=96, k=128):
    metadata = IQ2RMetadata(n, k)
    data = torch.zeros((1, metadata.data_bytes), dtype=torch.uint8)
    auxiliary = torch.zeros((1, metadata.auxiliary_bytes), dtype=torch.uint8)
    auxiliary[:, IQ2R_CODEBOOK_BYTES:] = 127
    return metadata, data, auxiliary


def test_six_block_physical_grouping_and_triplet_offsets():
    metadata, data, _ = _empty_fixture(n=192, k=256)
    assert [iq2r_physical_tile(nb, 0, 2) for nb in range(12)] == [
        0,
        1,
        2,
        3,
        4,
        5,
        12,
        13,
        14,
        15,
        16,
        17,
    ]
    assert [iq2r_physical_tile(nb, 1, 2) for nb in range(6)] == [6, 7, 8, 9, 10, 11]

    flat = data[0]
    records0, shared0 = iq2r_tile_views(flat, 0, 0, metadata.k_tiles)
    records1, shared1 = iq2r_tile_views(flat, 1, 0, metadata.k_tiles)
    records2, shared2 = iq2r_tile_views(flat, 2, 0, metadata.k_tiles)
    assert records1.storage_offset() - records0.storage_offset() == 8
    assert records2.storage_offset() - records0.storage_offset() == 1024
    assert (
        shared0.storage_offset() == shared1.storage_offset() == shared2.storage_offset()
    )
    records3, shared3 = iq2r_tile_views(flat, 3, 0, metadata.k_tiles)
    assert records3.storage_offset() == IQ2R_TRIPLET_BYTES
    assert shared3.storage_offset() == IQ2R_TRIPLET_BYTES + 1032
    records6, _ = iq2r_tile_views(flat, 6, 0, metadata.k_tiles)
    assert records6.storage_offset() == 2 * IQ2R_GROUP_BYTES


def test_all_atom_records_preserve_nine_bit_indices_signs_and_scale_deltas():
    metadata, data, _ = _empty_fixture(n=96)
    flat = data[0]
    for atom in range(IQ2R_ATOMS_PER_TRIPLET):
        indices = (torch.arange(256, dtype=torch.int64).reshape(64, 4) * 2 + atom) % 512
        signs = torch.arange(256, dtype=torch.uint8).reshape(64, 4) + atom
        records, atom_metadata = iq2r_tile_views(flat, atom, 0, metadata.k_tiles)
        records[:, :4] = indices.to(torch.uint8)
        records[:, 4:] = signs
        for lane in range(64):
            high_nibble = 0
            for codeword in range(4):
                high_nibble |= int((indices[lane, codeword] >> 8) & 1) << codeword
            scale_delta = 0 if lane == 0 else 15 if lane == 1 else (lane + atom) % 16
            word = iq2r_metadata_word(atom_metadata, lane)
            word |= high_nibble << (atom * 8)
            word |= scale_delta << (atom * 8 + 4)
            iq2r_set_metadata_word(atom_metadata, lane, word)

        for lane in range(64):
            word = iq2r_metadata_word(atom_metadata, lane)
            high_bits = (word >> (atom * 8)) & 0x0F
            decoded = records[lane, :4].to(torch.int64)
            for codeword in range(4):
                decoded[codeword] |= ((high_bits >> codeword) & 1) << 8
            torch.testing.assert_close(decoded, indices[lane], rtol=0, atol=0)
            torch.testing.assert_close(records[lane, 4:], signs[lane], rtol=0, atol=0)
        assert (iq2r_metadata_word(atom_metadata, 0) >> (atom * 8 + 4)) & 0xF == 0
        assert (iq2r_metadata_word(atom_metadata, 1) >> (atom * 8 + 4)) & 0xF == 15


def test_materializer_decodes_codebook_sign_index_and_e8m0_scale():
    metadata, data, auxiliary = _empty_fixture(n=96)
    codebook = torch.zeros((512, 8), dtype=torch.float32)
    codebook[1] = torch.arange(1, 9, dtype=torch.float32)
    codebook[257] = torch.arange(11, 19, dtype=torch.float32)
    auxiliary[0, :IQ2R_CODEBOOK_BYTES] = (
        codebook.to(torch.float8_e4m3fn).view(torch.uint8).reshape(-1)
    )

    for atom in range(3):
        records, atom_metadata = iq2r_tile_views(data[0], atom, 0, metadata.k_tiles)
        records[:, :4] = 1
        records[:, 4:] = 0
        records[0, 0] = 1
        records[0, 1] = 1
        records[0, 2] = 1
        records[0, 3] = 1
        records[0, 4] = 0b00000001
        word = iq2r_metadata_word(atom_metadata, 0)
        word |= 0b0001 << (atom * 8)  # first index is 257
        word |= 1 << (atom * 8 + 4)  # scale 2 for logical K block zero
        iq2r_set_metadata_word(atom_metadata, 0, word)

    decoded = iq2r_materialize(data, auxiliary, metadata)
    assert decoded.shape == (1, 96, 128)
    expected = (
        torch.tensor(
            [-11, 12, 13, 14, 15, 16, 16, 18, 1, 2, 3, 4, 5, 6, 7, 8],
            dtype=torch.float32,
        )
        * 2
    )
    torch.testing.assert_close(decoded[0, 0, :16], expected, rtol=0, atol=0)
    assert torch.isfinite(decoded).all()


def test_reserved_zero_codeword_makes_padded_k_exact_zero():
    metadata, data, auxiliary = _empty_fixture(n=96, k=2880)
    # Entry one is non-zero while every record remains reserved entry zero.
    codebook = torch.zeros((512, IQ2R_VECTOR_SIZE), dtype=torch.float32)
    codebook[1] = 3
    auxiliary[0, :IQ2R_CODEBOOK_BYTES] = (
        codebook.to(torch.float8_e4m3fn).view(torch.uint8).reshape(-1)
    )
    padded_metadata = IQ2RMetadata(96, 2944)
    padded = iq2r_materialize(data, auxiliary, padded_metadata)
    torch.testing.assert_close(padded[..., 2880:], torch.zeros_like(padded[..., 2880:]))


def test_materializer_rejects_nonzero_reserved_codeword():
    metadata, data, auxiliary = _empty_fixture()
    auxiliary[0, 0] = 1
    with pytest.raises(ValueError, match="all-zero"):
        iq2r_materialize(data, auxiliary, metadata)
