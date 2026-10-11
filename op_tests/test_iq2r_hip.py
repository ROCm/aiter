# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch

from aiter.ops.iq2r import iq2r_encode_device, iq2r_materialize_device
from aiter.ops.iq2r_encoder import iq2r_encode_reference, iq2r_initial_codebook
from aiter.ops.iq2r_format import IQ2RMetadata
from aiter.ops.iq2r_reference import iq2r_materialize


def _has_gfx950() -> bool:
    return torch.cuda.is_available() and "gfx950" in (
        torch.cuda.get_device_properties(0).gcnArchName
    )


pytestmark = pytest.mark.skipif(not _has_gfx950(), reason="requires gfx950")


def _weights(n: int, k: int, seed: int, device="cuda"):
    generator = torch.Generator(device=device).manual_seed(seed)
    weight = (torch.randn((n, k), generator=generator, device=device) * 0.08).float()
    importance = torch.linspace(0.2, 2.0, k, device=device)
    return weight, importance


@pytest.mark.parametrize("k", [128, 2048])
def test_device_encoder_matches_reference(k):
    weight, importance = _weights(64, k, 0x1022 + k)
    codebook = iq2r_initial_codebook("cuda")
    expected_data, expected_auxiliary = iq2r_encode_reference(
        weight, importance, codebook
    )
    actual_data, actual_auxiliary = iq2r_encode_device(weight, importance, codebook)
    torch.testing.assert_close(actual_data, expected_data, rtol=0, atol=0)
    torch.testing.assert_close(actual_auxiliary, expected_auxiliary, rtol=0, atol=0)


def test_device_kernels_honor_noncurrent_gpu():
    if torch.cuda.device_count() < 2:
        pytest.skip("requires at least two visible GPUs")

    torch.cuda.set_device(0)
    target = torch.device("cuda:1")
    weight, importance = _weights(64, 128, 0xD3A1CE, device=target)
    codebook = iq2r_initial_codebook(target)
    expected_data, expected_auxiliary = iq2r_encode_reference(
        weight, importance, codebook
    )

    actual_data, actual_auxiliary = iq2r_encode_device(weight, importance, codebook)

    assert torch.cuda.current_device() == 0
    torch.testing.assert_close(actual_data, expected_data, rtol=0, atol=0)
    torch.testing.assert_close(actual_auxiliary, expected_auxiliary, rtol=0, atol=0)

    metadata = IQ2RMetadata(64, 128)
    data = actual_data.reshape(1, -1)
    auxiliary = actual_auxiliary.reshape(1, -1)
    expected = iq2r_materialize(data.cpu(), auxiliary.cpu(), metadata)[0]
    actual = iq2r_materialize_device(data, auxiliary, metadata)
    assert actual.device == target and torch.cuda.current_device() == 0
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)


# N=64 fills four of a six-block group and N=96 fills one group; K=2048 is
# the GLM-5.3 TP1 down-projection depth.
@pytest.mark.parametrize("n", [64, 96])
def test_device_materializer_matches_independent_host_decoder(n):
    metadata = IQ2RMetadata(n, 2048)
    weight, importance = _weights(n, 2048, 0x3A7E + n)
    data, auxiliary = iq2r_encode_device(
        weight, importance, iq2r_initial_codebook("cuda")
    )
    data = data.reshape(1, -1)
    auxiliary = auxiliary.reshape(1, -1)
    expected = iq2r_materialize(data.cpu(), auxiliary.cpu(), metadata)[0]
    actual = iq2r_materialize_device(data, auxiliary, metadata).cpu()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
