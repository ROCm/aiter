# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch

from aiter.ops.iq2r_format import IQ2RMetadata, iq2r_slice_input_data

# GLM-5.3 expert projections: hidden 6144, moe_intermediate 2048.
_GLM53_GATE_UP = (4096, 6144)
_GLM53_DOWN = (6144, 2048)


def test_glm53_data_sizes():
    gate_up = IQ2RMetadata(*_GLM53_GATE_UP)
    down = IQ2RMetadata(*_GLM53_DOWN)

    assert (gate_up.data_bytes, down.data_bytes) == (7_397_376, 3_670_016)
    assert (gate_up.physical_n_blocks, down.physical_n_blocks) == (258, 384)
    assert (gate_up.k_tiles, down.k_tiles) == (48, 16)


def test_shape_validation():
    assert IQ2RMetadata(64, 128).data_bytes == 3584
    assert IQ2RMetadata(16, 32).data_bytes == 3584
    with pytest.raises(ValueError, match="K divisible"):
        IQ2RMetadata(64, 127)
    with pytest.raises(ValueError, match="N divisible"):
        IQ2RMetadata(63, 128)
    with pytest.raises(TypeError, match="must be an int"):
        IQ2RMetadata(64.0, 128)


def test_input_slice_validation():
    metadata = IQ2RMetadata(96, 256)
    data = torch.arange(2 * metadata.data_bytes).to(torch.uint8).view(2, -1)
    half = iq2r_slice_input_data(data, metadata, 128, 128)
    assert half.shape == (2, IQ2RMetadata(96, 128).data_bytes)
    assert half.storage_offset() == 0

    with pytest.raises(TypeError, match="uint8"):
        iq2r_slice_input_data(data.float(), metadata, 0, 128)
    with pytest.raises(ValueError, match="bytes per expert"):
        iq2r_slice_input_data(data[:, :-1].contiguous(), metadata, 0, 128)
    with pytest.raises(ValueError, match="aligned"):
        iq2r_slice_input_data(data, metadata, 64, 128)
    with pytest.raises(ValueError, match="aligned"):
        iq2r_slice_input_data(data, metadata, 128, 256)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
