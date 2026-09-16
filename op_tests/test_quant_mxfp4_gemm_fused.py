# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch

from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.ops.gemm_op_a6w4 import (
    _quant_mxfp4_gemm_torch_legacy,
    mxfp4_gemm_pack_size,
    quant_mxfp4_gemm,
    quant_mxfp4_gemm_hip_out,
)
from aiter.utility import dtypes, fp4_utils


def _is_gfx950() -> bool:
    try:
        return torch.cuda.is_available() and get_gfx_runtime() == "gfx950"
    except (KeyError, RuntimeError):
        return False


pytestmark = pytest.mark.skipif(
    not _is_gfx950(), reason="fused MXFP4 GEMM packing requires gfx950"
)

_TILE_ROWS = 256
_K_TILE = 128
_GUARD_TILES = 2
_PACKED_TILE_BYTES = 16384
_SCALE_TILE_BYTES = 1024


def _ceil(value: int, multiple: int) -> int:
    return (value + multiple - 1) // multiple * multiple


def _physical_indices(
    rows: int, K: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return logical gathers and masks for initialized physical output slots."""
    pad_rows = _ceil(rows, _TILE_ROWS)
    pad_k = _ceil(K, _K_TILE)
    nk = pad_k // _K_TILE
    nk_pad = nk + _GUARD_TILES
    packed_size, scale_size = mxfp4_gemm_pack_size(rows, K)
    packed_indices = torch.empty((rows, pad_k // 2), dtype=torch.int64)
    scale_indices = torch.empty((rows, pad_k // 32), dtype=torch.int64)
    packed_written = torch.zeros(packed_size, dtype=torch.bool)
    scale_written = torch.zeros(scale_size, dtype=torch.bool)

    for row in range(pad_rows):
        tile_row = row // _TILE_ROWS
        rem = row % _TILE_ROWS
        row_block, row16 = divmod(rem, 16)
        for step in range(nk):
            tile_base = (tile_row * nk_pad + step) * _PACKED_TILE_BYTES
            scale_tile_base = (tile_row * nk_pad + step) * _SCALE_TILE_BYTES
            for k_group in range(4):
                block = row_block * 64 + k_group * 16 + row16
                logical_byte = step * 64 + k_group * 16
                physical_bytes = torch.arange(
                    tile_base + block * 16, tile_base + block * 16 + 16
                )
                packed_written[physical_bytes] = True
                logical_group = step * 4 + k_group
                scale_index = (
                    scale_tile_base
                    + (rem // 128) * 512
                    + k_group * 128
                    + row16 * 8
                    + (rem % 128) // 16
                )
                scale_written[scale_index] = True
                if row < rows:
                    packed_indices[row, logical_byte : logical_byte + 16] = (
                        physical_bytes
                    )
                    scale_indices[row, logical_group] = scale_index

    assert packed_written.sum().item() == pad_rows * pad_k // 2
    assert scale_written.sum().item() == pad_rows * pad_k // 32
    return (
        packed_indices.to(device),
        scale_indices.to(device),
        packed_written.to(device),
        scale_written.to(device),
    )


def _logical_payload(
    packed: torch.Tensor,
    packed_scale: torch.Tensor,
    packed_indices: torch.Tensor,
    scale_indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return packed[packed_indices], packed_scale[scale_indices]


def _dequantize_logical(packed: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    values = fp4_utils.mxfp4_to_f32(packed.contiguous().view(dtypes.fp4x2))
    scale_f32 = fp4_utils.e8m0_to_f32(scales.contiguous())
    return values * scale_f32.repeat_interleave(32, dim=1)


def _edge_enriched_input(rows: int, K: int, dtype: torch.dtype) -> torch.Tensor:
    torch.manual_seed(rows * 1009 + K * 17 + dtype.itemsize)
    result = torch.randn((rows, K), dtype=torch.float32, device="cuda")
    edges = torch.tensor(
        [
            0.0,
            -0.0,
            2.0**-126,
            -(2.0**-126),
            0.25,
            -0.75,
            1.25,
            -2.5,
            3.5,
            -5.0,
            6.0,
            -6.0,
            448.0,
            -448.0,
        ],
        dtype=torch.float32,
        device="cuda",
    )
    count = min(K, edges.numel())
    result[0, :count] = edges[:count]
    return result.to(dtype)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("round_mode", range(4))
@torch.no_grad()
def test_fused_payload_and_scale_match_legacy(dtype, round_mode):
    rows, K = 17, 129
    x = _edge_enriched_input(rows, K, dtype)
    fused_packed, fused_scale = quant_mxfp4_gemm(x, round_mode=round_mode)
    legacy_packed, legacy_scale = _quant_mxfp4_gemm_torch_legacy(
        x, round_mode=round_mode
    )
    packed_indices, scale_indices, _, _ = _physical_indices(rows, K, x.device)
    fused_logical, fused_logical_scale = _logical_payload(
        fused_packed, fused_scale, packed_indices, scale_indices
    )
    legacy_logical, legacy_logical_scale = _logical_payload(
        legacy_packed, legacy_scale, packed_indices, scale_indices
    )

    packed_mismatches = (fused_logical != legacy_logical).sum().item()
    scale_mismatches = (fused_logical_scale != legacy_logical_scale).sum().item()
    packed_rate = packed_mismatches / fused_logical.numel()
    scale_rate = scale_mismatches / fused_logical_scale.numel()
    fused_values = _dequantize_logical(fused_logical, fused_logical_scale)
    legacy_values = _dequantize_logical(legacy_logical, legacy_logical_scale)
    relative_l2 = (
        (fused_values - legacy_values).norm()
        / legacy_values.norm().clamp_min(torch.finfo(torch.float32).tiny)
    ).item()
    print(
        f"dtype={dtype} mode={round_mode} "
        f"payload={packed_mismatches}/{fused_logical.numel()} ({packed_rate:.8%}) "
        f"scale={scale_mismatches}/{fused_logical_scale.numel()} ({scale_rate:.8%}) "
        f"dequant_relative_l2={relative_l2:.8g}"
    )

    assert scale_mismatches == 0
    assert packed_mismatches == 0
    assert relative_l2 == 0.0


@torch.no_grad()
def test_fused_payload_matches_legacy_at_model_scale():
    rows, K = 512, 5120
    torch.manual_seed(20260828)
    x = torch.randn((rows, K), dtype=torch.bfloat16, device="cuda")
    fused_packed, fused_scale = quant_mxfp4_gemm(x)
    legacy_packed, legacy_scale = _quant_mxfp4_gemm_torch_legacy(x)
    nk = K // _K_TILE
    nt = rows // _TILE_ROWS

    fused_payload = fused_packed.view(nt, nk + _GUARD_TILES, -1)[:, :nk]
    legacy_payload = legacy_packed.view(nt, nk + _GUARD_TILES, -1)[:, :nk]
    fused_scale_payload = fused_scale.view(nt, nk + _GUARD_TILES, -1)[:, :nk]
    legacy_scale_payload = legacy_scale.view(nt, nk + _GUARD_TILES, -1)[:, :nk]
    assert torch.equal(fused_payload, legacy_payload)
    assert torch.equal(fused_scale_payload, legacy_scale_payload)


@torch.no_grad()
def test_fused_hadamard_avoids_finite_intermediate_overflow():
    rows, K = 1, 32
    value = torch.finfo(torch.bfloat16).max / 8
    x = torch.full((rows, K), value, dtype=torch.bfloat16, device="cuda")
    fused_packed, fused_scale = quant_mxfp4_gemm(x, round_mode=2)
    legacy_packed, legacy_scale = _quant_mxfp4_gemm_torch_legacy(x, round_mode=2)
    packed_indices, scale_indices, _, _ = _physical_indices(rows, K, x.device)
    fused_payload, fused_scale_payload = _logical_payload(
        fused_packed, fused_scale, packed_indices, scale_indices
    )
    legacy_payload, legacy_scale_payload = _logical_payload(
        legacy_packed, legacy_scale, packed_indices, scale_indices
    )
    assert torch.equal(fused_payload, legacy_payload)
    assert torch.equal(fused_scale_payload, legacy_scale_payload)


@pytest.mark.parametrize("rows,K", [(1, 1), (16, 31), (17, 32), (255, 127), (257, 128)])
@torch.no_grad()
def test_fused_tail_layout_and_unwritten_canaries(rows, K):
    x = torch.zeros((rows, K), dtype=torch.bfloat16, device="cuda")
    packed_size, scale_size = mxfp4_gemm_pack_size(rows, K)
    packed = torch.full((packed_size,), 0xA5, dtype=torch.uint8, device="cuda")
    packed_scale = torch.full((scale_size,), 0x5A, dtype=torch.uint8, device="cuda")
    _, _, packed_written, scale_written = _physical_indices(rows, K, x.device)

    quant_mxfp4_gemm_hip_out(x, packed, packed_scale, 1)
    torch.cuda.synchronize()

    assert torch.count_nonzero(packed[packed_written]).item() == 0
    # Legacy RoundUp starts amax at 1e-10, so an all-zero group stores
    # ceil_pow2(1e-10 / 6) = 2^-35, biased E8M0 byte 92.
    assert torch.all(packed_scale[scale_written] == 92)
    assert torch.all(packed[~packed_written] == 0xA5)
    assert torch.all(packed_scale[~scale_written] == 0x5A)


@torch.no_grad()
def test_fused_api_contract_and_noncontiguous_input():
    base = torch.randn((9, 66), dtype=torch.float32, device="cuda")
    noncontiguous = base[:, ::2]
    assert not noncontiguous.is_contiguous()
    actual = quant_mxfp4_gemm(noncontiguous, round_mode=3)
    expected = quant_mxfp4_gemm(noncontiguous.contiguous(), round_mode=3)
    packed_indices, scale_indices, _, _ = _physical_indices(
        *noncontiguous.shape, noncontiguous.device
    )
    assert actual[0].dtype == torch.uint8 and actual[1].dtype == torch.uint8
    assert actual[0].ndim == actual[1].ndim == 1
    assert torch.equal(actual[0][packed_indices], expected[0][packed_indices])
    assert torch.equal(actual[1][scale_indices], expected[1][scale_indices])

    with pytest.raises(ValueError, match="round_mode"):
        quant_mxfp4_gemm(noncontiguous, round_mode=4)
    with pytest.raises(ValueError, match="integral"):
        quant_mxfp4_gemm(noncontiguous, round_mode=1.5)
    with pytest.raises(ValueError, match="2D"):
        quant_mxfp4_gemm(noncontiguous[0], round_mode=1)
    huge_view = torch.empty(1, dtype=torch.bfloat16, device="cuda").expand(65536, 65536)
    with pytest.raises(ValueError, match="2 GiB"):
        quant_mxfp4_gemm(huge_view, round_mode=1)


@torch.no_grad()
def test_fused_packer_handles_misaligned_contiguous_input():
    rows, K = 17, 128
    storage = torch.randn(rows * K + 1, dtype=torch.bfloat16, device="cuda")
    misaligned = storage[1:].view(rows, K)
    assert misaligned.is_contiguous()
    assert misaligned.data_ptr() % 16 != 0

    actual = quant_mxfp4_gemm(misaligned)
    expected = quant_mxfp4_gemm(misaligned.clone())
    packed_indices, scale_indices, _, _ = _physical_indices(rows, K, misaligned.device)
    actual_payload = _logical_payload(*actual, packed_indices, scale_indices)
    expected_payload = _logical_payload(*expected, packed_indices, scale_indices)
    assert torch.equal(actual_payload[0], expected_payload[0])
    assert torch.equal(actual_payload[1], expected_payload[1])


@torch.no_grad()
@pytest.mark.parametrize("target", ["packed", "scale"])
def test_fused_packer_rejects_misaligned_output(target):
    rows, K = 17, 128
    x = torch.randn((rows, K), dtype=torch.bfloat16, device="cuda")
    packed_size, scale_size = mxfp4_gemm_pack_size(rows, K)
    packed = torch.empty(packed_size, dtype=torch.uint8, device="cuda")
    packed_scale = torch.empty(scale_size, dtype=torch.uint8, device="cuda")
    if target == "packed":
        storage = torch.empty(packed_size + 1, dtype=torch.uint8, device="cuda")
        packed = storage[1:]
    else:
        storage = torch.empty(scale_size + 1, dtype=torch.uint8, device="cuda")
        packed_scale = storage[1:]

    with pytest.raises(ValueError, match="16-byte-aligned"):
        quant_mxfp4_gemm_hip_out(x, packed, packed_scale)


def test_fused_packer_preserves_public_torch_operator_name():
    x = torch.randn((16, 128), dtype=torch.bfloat16, device="cuda")
    quant_mxfp4_gemm(x)

    assert hasattr(torch.ops.aiter, "quant_mxfp4_gemm_hip_out")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "-s"]))
