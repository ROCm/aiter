# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch

from aiter.ops.iq2r_encoder import (
    iq2r_dequantize_mxfp4,
    iq2r_encode_reference,
    iq2r_initial_codebook,
    iq2r_learn_codebook,
    iq2r_reserve_zero_codeword,
)
from aiter.ops.iq2r_format import IQ2RMetadata
from aiter.ops.iq2r_reference import iq2r_materialize


def _inputs(n=64, k=128):
    generator = torch.Generator().manual_seed(0x1022 + n + k)
    weight = torch.randn((n, k), generator=generator) * 0.08
    importance = torch.linspace(0.2, 2.0, k)
    return weight, importance


def test_mxfp4_dequantization_preserves_nibble_and_row_order():
    blocks = torch.tensor(
        [
            [[0x21, 0xF8], [0x40, 0xB3]],
            [[0x56, 0xA9], [0x87, 0xCD]],
        ],
        dtype=torch.uint8,
    )
    scales = torch.tensor([[127, 129], [126, 127]], dtype=torch.uint8)
    actual = iq2r_dequantize_mxfp4(blocks, scales, output_dtype=torch.float32)
    expected = torch.tensor(
        [
            [0.5, 1.0, -0.0, -6.0, 0.0, 8.0, 6.0, -6.0],
            [2.0, 1.5, -0.25, -0.5, 6.0, -0.0, -3.0, -2.0],
        ],
        dtype=torch.float32,
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_initial_codebook_has_exact_standard_seed_values():
    codebook = iq2r_initial_codebook()
    assert codebook.shape == (512, 8)
    torch.testing.assert_close(codebook[0], torch.zeros(8), rtol=0, atol=0)
    torch.testing.assert_close(
        codebook[1], torch.tensor([5.5, 1, 1, 1, 1, 1, 1, 1]), rtol=0, atol=0
    )
    torch.testing.assert_close(codebook[256], torch.full((8,), 1.5), rtol=0, atol=0)


def test_reserving_zero_does_not_mutate_input():
    codebook = iq2r_initial_codebook()
    codebook[0, 0] = 1
    original = codebook.clone()
    reserved = iq2r_reserve_zero_codeword(codebook)
    assert torch.count_nonzero(reserved[0]).item() == 0
    torch.testing.assert_close(codebook, original, rtol=0, atol=0)


def test_learned_codebook_reserves_zero():
    weight, importance = _inputs()
    codebook = iq2r_learn_codebook(
        weight, importance, iterations=2, sample_vectors=4096
    )
    assert torch.count_nonzero(codebook[0]).item() == 0


@pytest.mark.parametrize("n", [16, 48, 64, 96])
def test_reference_encode_decode_round_trip_and_physical_padding(n):
    weight, importance = _inputs(n=n)
    codebook = iq2r_initial_codebook()
    data, auxiliary = iq2r_encode_reference(weight, importance, codebook)
    metadata = IQ2RMetadata(n, 128)
    decoded = iq2r_materialize(data.unsqueeze(0), auxiliary.unsqueeze(0), metadata)[0]
    assert data.numel() == metadata.data_bytes
    assert auxiliary.numel() == metadata.auxiliary_bytes
    assert torch.isfinite(decoded).all()
    relative_l2 = torch.linalg.vector_norm(decoded - weight) / torch.linalg.vector_norm(
        weight
    )
    assert relative_l2.item() < 0.7


def test_reference_encoder_preserves_zero_padded_k():
    weight, importance = _inputs(n=16, k=2880)
    data, auxiliary = iq2r_encode_reference(weight, importance, iq2r_initial_codebook())
    padded = iq2r_materialize(
        data.unsqueeze(0), auxiliary.unsqueeze(0), IQ2RMetadata(16, 2944)
    )
    torch.testing.assert_close(padded[..., 2880:], torch.zeros_like(padded[..., 2880:]))
