# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import dataclasses

import pytest
import torch

from aiter.ops.iq2r_encoder import iq2r_encode_reference, iq2r_initial_codebook
from aiter.ops.iq2r_format import (
    IQ2R_ACTIVATION_BASIS,
    IQ2R_CODEBOOK_BYTES,
    IQ2R_FORMAT_NAME,
    IQ2R_FORMAT_VERSION,
    IQ2R_GPT_OSS_DOWN_N,
    IQ2R_GPT_OSS_EXPERTS,
    IQ2R_GPT_OSS_GATE_UP_N,
    IQ2R_GPT_OSS_K,
    IQ2RMetadata,
    iq2r_gpt_oss_metadata,
    iq2r_packed_sizes,
    iq2r_slice_input_data,
    iq2r_slice_output_auxiliary,
    iq2r_slice_output_data,
    iq2r_storage_bits_per_weight,
    iq2r_validate_expert_weights,
)
from aiter.ops.iq2r_reference import iq2r_materialize


def test_gpt_oss_exact_packed_sizes_and_storage_rate():
    gate_up = iq2r_gpt_oss_metadata("gate_up")
    down = iq2r_gpt_oss_metadata("down")

    assert (gate_up.data_bytes, gate_up.auxiliary_bytes) == (4_945_920, 4_459)
    assert (down.data_bytes, down.auxiliary_bytes) == (2_472_960, 4_279)
    assert gate_up.physical_n_blocks == 360
    assert down.physical_n_blocks == 180
    assert gate_up.k_tiles == down.k_tiles == 23
    assert gate_up.padded_k == down.padded_k == 2944
    assert gate_up.storage_bits_per_weight == pytest.approx(2.3873356006944446)
    assert down.storage_bits_per_weight == pytest.approx(2.3893123070987656)

    combined_bits = (
        8
        * (
            gate_up.data_bytes
            + gate_up.auxiliary_bytes
            + down.data_bytes
            + down.auxiliary_bytes
        )
        / (gate_up.logical_n * gate_up.logical_k + down.logical_n * down.logical_k)
    )
    assert combined_bits == pytest.approx(2.387994470164609)
    assert iq2r_storage_bits_per_weight(5760, 2880) == gate_up.storage_bits_per_weight


def test_metadata_round_trip_and_identity_fail_closed():
    metadata = iq2r_gpt_oss_metadata(
        "gate_up",
        source_model_fingerprint="model-sha",
        calibration_fingerprint="calibration-sha",
        calibration_scheme="iq2r-diagonal-second-moment",
    )
    serialized = metadata.to_dict()
    serialized["future_optional_field"] = "ignored"
    assert IQ2RMetadata.from_dict(serialized) == metadata

    for field in (
        "format_name",
        "format_version",
        "activation_basis",
        "architecture",
        "reserved_zero_codeword",
    ):
        corrupt = dict(serialized)
        corrupt[field] = "wrong" if field != "format_version" else 3
        with pytest.raises(ValueError, match=field):
            IQ2RMetadata.from_dict(corrupt)

    missing = dict(serialized)
    del missing["activation_basis"]
    with pytest.raises(ValueError, match="activation_basis"):
        IQ2RMetadata.from_dict(missing)

    assert metadata.format_name == IQ2R_FORMAT_NAME
    assert metadata.format_version == IQ2R_FORMAT_VERSION
    assert metadata.activation_basis == IQ2R_ACTIVATION_BASIS


def test_shape_and_projection_validation():
    assert iq2r_packed_sizes(64, 128) == (3584, 4105)
    assert iq2r_packed_sizes(16, 32) == (3584, 4105)
    with pytest.raises(ValueError, match="K divisible"):
        IQ2RMetadata(64, 127)
    with pytest.raises(ValueError, match="N divisible"):
        IQ2RMetadata(63, 128)
    with pytest.raises(ValueError, match="projection"):
        iq2r_gpt_oss_metadata("sideways")
    with pytest.raises(ValueError, match="GPT-OSS"):
        IQ2RMetadata(64, 128).validate_gpt_oss()


def test_stacked_buffer_and_reserved_zero_validation():
    metadata = IQ2RMetadata(64, 128)
    data = torch.zeros((2, metadata.data_bytes), dtype=torch.uint8)
    auxiliary = torch.zeros((2, metadata.auxiliary_bytes), dtype=torch.uint8)
    auxiliary[:, IQ2R_CODEBOOK_BYTES:] = 127
    iq2r_validate_expert_weights(data, auxiliary, metadata, expert_count=2)

    with pytest.raises(TypeError, match="uint8"):
        iq2r_validate_expert_weights(data.float(), auxiliary, metadata)
    with pytest.raises(ValueError, match="bytes per expert"):
        iq2r_validate_expert_weights(data[:, :-1].contiguous(), auxiliary, metadata)
    with pytest.raises(ValueError, match="expert counts differ"):
        iq2r_validate_expert_weights(data, auxiliary[:1], metadata)
    with pytest.raises(ValueError, match="expected 3"):
        iq2r_validate_expert_weights(data, auxiliary, metadata, expert_count=3)

    noncontiguous = torch.zeros((2, metadata.data_bytes, 2), dtype=torch.uint8)[..., 0]
    assert not noncontiguous.is_contiguous()
    with pytest.raises(ValueError, match="contiguous"):
        iq2r_validate_expert_weights(noncontiguous, auxiliary, metadata)

    corrupt = auxiliary.clone()
    corrupt[1, 0] = 1
    with pytest.raises(ValueError, match="all-zero"):
        iq2r_validate_expert_weights(data, corrupt, metadata)


def test_gpt_oss_metadata_constants():
    gate_up = iq2r_gpt_oss_metadata("gate_up")
    down = iq2r_gpt_oss_metadata("down")
    assert gate_up.logical_n == IQ2R_GPT_OSS_GATE_UP_N
    assert down.logical_n == IQ2R_GPT_OSS_DOWN_N
    assert gate_up.logical_k == down.logical_k == IQ2R_GPT_OSS_K
    assert IQ2R_GPT_OSS_EXPERTS == 128


def test_dataclass_replacement_cannot_relabel_o0():
    metadata = iq2r_gpt_oss_metadata("down")
    with pytest.raises(ValueError, match="activation_basis"):
        dataclasses.replace(metadata, activation_basis="hadamard")


def test_packed_output_and_input_slices_are_bit_exact():
    generator = torch.Generator().manual_seed(0x53_08)
    weight = torch.randn((96, 256), generator=generator)
    data, auxiliary = iq2r_encode_reference(
        weight,
        torch.ones((256,), dtype=torch.float32),
        iq2r_initial_codebook(),
        exponent_radius=8,
    )
    data = data.unsqueeze(0)
    auxiliary = auxiliary.unsqueeze(0)
    metadata = IQ2RMetadata(logical_n=96, logical_k=256)
    dense = iq2r_materialize(data, auxiliary, metadata)

    output_metadata = IQ2RMetadata(logical_n=64, logical_k=256)
    output_data = iq2r_slice_output_data(data, metadata, start=16, length=64)
    output_auxiliary = iq2r_slice_output_auxiliary(
        auxiliary, metadata, start=16, length=64
    )
    torch.testing.assert_close(
        iq2r_materialize(output_data, output_auxiliary, output_metadata),
        dense[:, 16:80],
        rtol=0,
        atol=0,
    )

    input_metadata = IQ2RMetadata(logical_n=96, logical_k=128)
    input_data = iq2r_slice_input_data(data, metadata, start=128, length=128)
    torch.testing.assert_close(
        iq2r_materialize(input_data, auxiliary, input_metadata),
        dense[:, :, 128:],
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize(
    ("function", "start", "length", "match"),
    [
        (iq2r_slice_output_data, 1, 32, "output slice"),
        (iq2r_slice_output_auxiliary, 0, 17, "output slice"),
        (iq2r_slice_input_data, 32, 128, "input slice"),
        (iq2r_slice_input_data, 128, 256, "input slice"),
    ],
)
def test_packed_slice_rejects_unaligned_or_out_of_range_requests(
    function, start, length, match
):
    metadata = IQ2RMetadata(logical_n=96, logical_k=256)
    width = (
        metadata.auxiliary_bytes
        if function is iq2r_slice_output_auxiliary
        else metadata.data_bytes
    )
    value = torch.zeros((1, width), dtype=torch.uint8)
    with pytest.raises(ValueError, match=match):
        function(value, metadata, start=start, length=length)
