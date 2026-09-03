# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch

import aiter.ops.gemm_op_a6w6 as mxfp6
from aiter.jit.utils.chip_info import get_gfx_runtime as get_gfx

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or get_gfx() != "gfx950" or not mxfp6._HAS_TRITON,
    reason="gfx950 hardware FP6 conversion and Triton are required",
)


def _pack_out(x: torch.Tensor, backend: str) -> tuple[torch.Tensor, torch.Tensor]:
    packed_size, scale_size = mxfp6.mxfp6_gemm_pack_size(*x.shape)
    packed = torch.full((packed_size,), 0xA5, dtype=torch.uint8, device=x.device)
    packed_scale = torch.full((scale_size,), 0x5A, dtype=torch.uint8, device=x.device)
    mxfp6._QUANT_BACKEND = backend
    return mxfp6.quant_mxfp6_gemm_out(x, packed, packed_scale)


@pytest.mark.parametrize("shape", [(4,), (2, 3, 4)])
def test_quant_mxfp6_gemm_rejects_non_matrix(shape: tuple[int, ...]):
    x = torch.empty(shape, dtype=torch.bfloat16, device="cuda")
    packed = torch.empty(1, dtype=torch.uint8, device=x.device)
    packed_scale = torch.empty(1, dtype=torch.uint8, device=x.device)

    with pytest.raises(ValueError, match=r"expects a 2D \[rows, K\] tensor"):
        mxfp6.quant_mxfp6_gemm(x)
    with pytest.raises(ValueError, match=r"expects a 2D \[rows, K\] tensor"):
        mxfp6.quant_mxfp6_gemm_out(x, packed, packed_scale)


def _unpack_first_block(packed: torch.Tensor) -> torch.Tensor:
    block = torch.cat((packed[:16], packed[16384:16392])).to(torch.int32)
    triplets = block.reshape(8, 3)
    b0, b1, b2 = triplets.unbind(dim=1)
    return torch.stack(
        (
            b0 & 0x3F,
            ((b0 >> 6) | (b1 << 2)) & 0x3F,
            ((b1 >> 4) | (b2 << 4)) & 0x3F,
            (b2 >> 2) & 0x3F,
        ),
        dim=1,
    ).reshape(32)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    ("rows", "cols"),
    [(1, 1), (1, 33), (17, 129), (255, 511), (257, 513)],
)
def test_hip_packer_matches_triton(
    monkeypatch: pytest.MonkeyPatch,
    dtype: torch.dtype,
    rows: int,
    cols: int,
):
    torch.manual_seed(rows * 1000 + cols)
    x = torch.randn((rows, cols), dtype=dtype, device="cuda")

    monkeypatch.setattr(mxfp6, "_QUANT_BACKEND", "hip")
    hip_packed, hip_scale = _pack_out(x, "hip")
    triton_packed, triton_scale = _pack_out(x, "triton")

    torch.testing.assert_close(hip_packed, triton_packed, rtol=0, atol=0)
    torch.testing.assert_close(hip_scale, triton_scale, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_hip_packer_canonicalizes_signed_zero(
    monkeypatch: pytest.MonkeyPatch,
    dtype: torch.dtype,
):
    x = torch.full((17, 129), -0.0, dtype=dtype, device="cuda")

    monkeypatch.setattr(mxfp6, "_QUANT_BACKEND", "hip")
    hip_packed, hip_scale = _pack_out(x, "hip")
    triton_packed, triton_scale = _pack_out(x, "triton")

    torch.testing.assert_close(hip_packed, triton_packed, rtol=0, atol=0)
    torch.testing.assert_close(hip_scale, triton_scale, rtol=0, atol=0)


def test_hip_packer_rounding_boundary_is_adjacent_to_triton(
    monkeypatch: pytest.MonkeyPatch,
):
    # This BF16 block previously separated the hardware conversion and Triton
    # fallback at the midpoint between adjacent E2M3 codes.
    values = [
        1.703125,
        -0.92578125,
        -0.0859375,
        0.87890625,
        0.59765625,
        -0.0004253387451171875,
        -2.671875,
        -0.306640625,
        -0.447265625,
        -0.28125,
        0.2314453125,
        -0.043212890625,
        0.1630859375,
        1.0546875,
        1.765625,
        1.09375,
        -1.1015625,
        -1.5,
        -0.119140625,
        -1.328125,
        -0.349609375,
        0.92578125,
        0.388671875,
        0.58203125,
        0.85546875,
        -0.84765625,
        -0.9140625,
        0.5625,
        0.4296875,
        -1.171875,
        1.6953125,
        -0.63671875,
    ]
    x = torch.tensor([values], dtype=torch.bfloat16, device="cuda")

    monkeypatch.setattr(mxfp6, "_QUANT_BACKEND", "hip")
    hip_packed, hip_scale = _pack_out(x, "hip")
    triton_packed, triton_scale = _pack_out(x, "triton")

    torch.testing.assert_close(hip_scale, triton_scale, rtol=0, atol=0)
    hip_codes = _unpack_first_block(hip_packed)
    triton_codes = _unpack_first_block(triton_packed)
    torch.testing.assert_close(hip_codes >> 5, triton_codes >> 5, rtol=0, atol=0)
    assert int(((hip_codes & 0x1F) - (triton_codes & 0x1F)).abs().max()) <= 1


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
