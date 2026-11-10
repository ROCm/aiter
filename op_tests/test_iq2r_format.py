# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch

from aiter.ops.iq2r_format import (
    IQ2R_CODEBOOK_BYTES,
    IQ2RMetadata,
    iq2r_packed_sizes,
    iq2r_validate_expert_weights,
)

# GLM-5.3 expert projections: hidden 6144, moe_intermediate 2048.
_GLM53_GATE_UP = (4096, 6144)
_GLM53_DOWN = (6144, 2048)


def test_glm53_exact_packed_sizes_and_storage_rate():
    gate_up = IQ2RMetadata(*_GLM53_GATE_UP)
    down = IQ2RMetadata(*_GLM53_DOWN)

    assert (gate_up.data_bytes, gate_up.auxiliary_bytes) == (7_397_376, 4_357)
    assert (down.data_bytes, down.auxiliary_bytes) == (3_670_016, 4_483)
    assert gate_up.physical_n_blocks == 258
    assert down.physical_n_blocks == 384
    assert (gate_up.k_tiles, down.k_tiles) == (48, 16)
    assert (gate_up.padded_k, down.padded_k) == (6144, 2048)
    assert gate_up.storage_bits_per_weight == pytest.approx(2.352947552998861)
    assert down.storage_bits_per_weight == pytest.approx(2.336183547973633)


def test_shape_and_projection_validation():
    assert iq2r_packed_sizes(64, 128) == (3584, 4105)
    assert iq2r_packed_sizes(16, 32) == (3584, 4105)
    with pytest.raises(ValueError, match="K divisible"):
        IQ2RMetadata(64, 127)
    with pytest.raises(ValueError, match="N divisible"):
        IQ2RMetadata(63, 128)


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


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
