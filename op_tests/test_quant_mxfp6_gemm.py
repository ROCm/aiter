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
    previous_backend = mxfp6._QUANT_BACKEND
    try:
        mxfp6._QUANT_BACKEND = backend
        return mxfp6.quant_mxfp6_gemm_out(x, packed, packed_scale)
    finally:
        mxfp6._QUANT_BACKEND = previous_backend


@pytest.mark.parametrize("shape", [(4,), (2, 3, 4)])
def test_quant_mxfp6_gemm_rejects_non_matrix(shape: tuple[int, ...]):
    x = torch.empty(shape, dtype=torch.bfloat16, device="cuda")
    packed = torch.empty(1, dtype=torch.uint8, device=x.device)
    packed_scale = torch.empty(1, dtype=torch.uint8, device=x.device)

    with pytest.raises(ValueError, match=r"expects a 2D \[rows, K\] tensor"):
        mxfp6.quant_mxfp6_gemm(x)
    with pytest.raises(ValueError, match=r"expects a 2D \[rows, K\] tensor"):
        mxfp6.quant_mxfp6_gemm_out(x, packed, packed_scale)


def test_quant_mxfp6_gemm_out_rejects_wrong_output_dtype():
    x = torch.empty((16, 128), dtype=torch.bfloat16, device="cuda")
    packed_size, scale_size = mxfp6.mxfp6_gemm_pack_size(*x.shape)
    packed = torch.empty(packed_size, dtype=torch.int8, device=x.device)
    packed_scale = torch.empty(scale_size, dtype=torch.uint8, device=x.device)

    with pytest.raises(ValueError, match="uint8"):
        mxfp6.quant_mxfp6_gemm_out(x, packed, packed_scale)


def test_quant_mxfp6_gemm_rejects_oversized_shape_before_allocation():
    huge_view = torch.empty(1, dtype=torch.bfloat16).expand(65536, 65536)
    with pytest.raises(ValueError, match="2 GiB"):
        mxfp6.quant_mxfp6_gemm(huge_view)


def test_torch_pack_helpers_validate_physical_layout():
    with pytest.raises(ValueError, match="rows%256"):
        mxfp6.pack_big_torch(torch.zeros((255, 128), dtype=torch.uint8))
    with pytest.raises(ValueError, match="K%128"):
        mxfp6.pack_big_torch(torch.zeros((256, 127), dtype=torch.uint8))
    with pytest.raises(ValueError, match="matching positive rows"):
        mxfp6.pack_scale_torch(torch.zeros((255, 4), dtype=torch.uint8), rows=256)
    with pytest.raises(ValueError, match="positive multiple of 4"):
        mxfp6.pack_scale_torch(torch.zeros((256, 3), dtype=torch.uint8), rows=256)


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


def test_hip_packer_avoids_hadamard_intermediate_overflow(
    monkeypatch: pytest.MonkeyPatch,
):
    max_bf16 = torch.finfo(torch.bfloat16).max

    # This transform is finite, but delaying normalization until after the
    # butterfly overflowed its intermediate H8 sums.
    finite = torch.full((1, 32), max_bf16 / 8, dtype=torch.bfloat16, device="cuda")
    monkeypatch.setattr(mxfp6, "_QUANT_BACKEND", "hip")
    hip_packed, hip_scale = _pack_out(finite, "hip")
    triton_packed, triton_scale = _pack_out(finite, "triton")
    assert torch.equal(
        _unpack_first_block(hip_packed), _unpack_first_block(triton_packed)
    )
    assert hip_scale[0] == triton_scale[0]

    # The DC coefficient of an all-max block exceeds fp32/MXFP6 range. It must
    # saturate positively without inf-inf cancellation creating spurious signs.
    extreme = torch.full((1, 32), max_bf16, dtype=torch.bfloat16, device="cuda")
    for backend in ("hip", "triton"):
        packed, scale = _pack_out(extreme, backend)
        codes = _unpack_first_block(packed)
        assert int(scale[0]) == 254, backend
        assert int(codes[0]) == 31, backend
        assert torch.count_nonzero(codes[1:]).item() == 0, backend

    torch_codes, torch_scales = mxfp6.quant_mxfp6_torch(extreme)
    assert int(torch_scales[0, 0]) == 254
    assert int(torch_codes[0, 0]) == 31
    assert torch.count_nonzero(torch_codes[0, 1:]).item() == 0


@pytest.mark.parametrize("exponent", [-119, -120, -121, -122])
def test_hip_packer_preserves_low_e8m0_scales(
    monkeypatch: pytest.MonkeyPatch,
    exponent: int,
):
    x = torch.zeros((1, 32), dtype=torch.bfloat16, device="cuda")
    x[0, 0] = 2.0**exponent
    monkeypatch.setattr(mxfp6, "_QUANT_BACKEND", "hip")
    for backend in ("hip", "triton"):
        packed, scale = _pack_out(x, backend)
        assert torch.all(_unpack_first_block(packed) == 27), backend
        assert int(scale[0]) == exponent + 122, backend

    torch_codes, torch_scales = mxfp6.quant_mxfp6_torch(x)
    assert torch.all(torch_codes[0] == 27)
    assert int(torch_scales[0, 0]) == exponent + 122


def test_hip_packer_handles_misaligned_contiguous_input(
    monkeypatch: pytest.MonkeyPatch,
):
    rows, cols = 256, 128
    storage = torch.randn(rows * cols + 1, dtype=torch.bfloat16, device="cuda")
    misaligned = storage[1:].view(rows, cols)
    assert misaligned.is_contiguous()
    assert misaligned.data_ptr() % 16 != 0

    monkeypatch.setattr(mxfp6, "_QUANT_BACKEND", "hip")
    actual_packed, actual_scale = mxfp6.quant_mxfp6_gemm(misaligned)
    expected_packed, expected_scale = mxfp6.quant_mxfp6_gemm(misaligned.clone())

    assert torch.equal(
        actual_packed.view(1, 3, -1)[:, :1],
        expected_packed.view(1, 3, -1)[:, :1],
    )
    assert torch.equal(
        actual_scale.view(1, 3, -1)[:, :1],
        expected_scale.view(1, 3, -1)[:, :1],
    )


@pytest.mark.parametrize("target", ["packed", "scale"])
@pytest.mark.parametrize("backend", ["hip", "triton"])
def test_packer_rejects_misaligned_output(
    monkeypatch: pytest.MonkeyPatch,
    target: str,
    backend: str,
):
    rows, cols = 17, 128
    x = torch.randn((rows, cols), dtype=torch.bfloat16, device="cuda")
    packed_size, scale_size = mxfp6.mxfp6_gemm_pack_size(rows, cols)
    packed = torch.empty(packed_size, dtype=torch.uint8, device="cuda")
    packed_scale = torch.empty(scale_size, dtype=torch.uint8, device="cuda")
    if target == "packed":
        storage = torch.empty(packed_size + 1, dtype=torch.uint8, device="cuda")
        packed = storage[1:]
    else:
        storage = torch.empty(scale_size + 1, dtype=torch.uint8, device="cuda")
        packed_scale = storage[1:]

    monkeypatch.setattr(mxfp6, "_QUANT_BACKEND", backend)
    with pytest.raises(ValueError, match="16-byte-aligned"):
        mxfp6.quant_mxfp6_gemm_out(x, packed, packed_scale)


def test_hip_packer_preserves_public_torch_operator_name(
    monkeypatch: pytest.MonkeyPatch,
):
    x = torch.randn((16, 128), dtype=torch.bfloat16, device="cuda")
    monkeypatch.setattr(mxfp6, "_QUANT_BACKEND", "hip")
    mxfp6.quant_mxfp6_gemm(x)

    assert hasattr(torch.ops.aiter, "quant_mxfp6_gemm_hip_out")


def test_backend_architecture_check_is_device_specific(
    monkeypatch: pytest.MonkeyPatch,
):
    class Properties:
        def __init__(self, arch):
            self.gcnArchName = arch

    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda index: Properties("gfx950" if index == 0 else "gfx942"),
    )
    mxfp6._is_gfx950_device_index.cache_clear()
    try:
        assert mxfp6._is_gfx950_device(torch.device("cuda:0"))
        assert not mxfp6._is_gfx950_device(torch.device("cuda:1"))
    finally:
        mxfp6._is_gfx950_device_index.cache_clear()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
