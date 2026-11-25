# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch

from aiter.ops.triton.quant import (
    dequant_hadamard_quant_mxfp4,
    dynamic_mxfp4_quant,
)
from aiter.ops.triton.utils._triton import arch_info
from aiter.ops.triton.utils.shuffle import shuffle_scale_gemm, shuffle_weight
from aiter.utility import dtypes
from aiter.utility.fp4_utils import e8m0_to_f32, mxfp4_to_f32

_E2M1_VALUES = (
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    -0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
)


requires_gfx950 = pytest.mark.skipif(
    not torch.cuda.is_available() or arch_info.get_arch() != "gfx950",
    reason="dequant-H16 MXFP4 requantization requires gfx950",
)
pytestmark = requires_gfx950


def _pack_codes(codes: torch.Tensor) -> torch.Tensor:
    return (codes[:, 0::2] | (codes[:, 1::2] << 4)).to(torch.uint8)


def _unpack_codes(packed: torch.Tensor) -> torch.Tensor:
    codes = torch.empty(
        (packed.shape[0], packed.shape[1] * 2), dtype=torch.uint8, device="cpu"
    )
    codes[:, 0::2] = packed & 0x0F
    codes[:, 1::2] = packed >> 4
    return codes


def _decode_e8m0(raw: torch.Tensor) -> torch.Tensor:
    raw_i32 = raw.to(torch.int32)
    bits = raw_i32 << 23
    bits = torch.where(raw_i32 == 0, 0x00400000, bits)
    bits = torch.where(raw_i32 == 255, 0x7FC00000, bits)
    return bits.view(torch.float32)


def _hadamard16() -> torch.Tensor:
    matrix = torch.ones((1, 1), dtype=torch.float32, device="cpu")
    while matrix.shape[0] < 16:
        matrix = torch.cat(
            (
                torch.cat((matrix, matrix), dim=1),
                torch.cat((matrix, -matrix), dim=1),
            ),
            dim=0,
        )
    return matrix * 0.25


def _quantize_even_rne(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    invalid = ~torch.isfinite(x).all(dim=1)
    finite_x = torch.where(torch.isfinite(x), x, 0.0)
    amax = finite_x.abs().amax(dim=1)

    amax_bits = amax.view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    rounded_bits = (amax_bits + 0x00200000) & 0xFF800000
    rounded_exponent = rounded_bits >> 23
    raw_scale_i64 = (rounded_exponent - 2).clamp(0, 254)
    raw_scale_i64 = torch.where(
        rounded_exponent == 255,
        torch.full_like(raw_scale_i64, 254),
        raw_scale_i64,
    )
    raw_scale_i64 = torch.where(
        invalid,
        torch.full_like(raw_scale_i64, 255),
        raw_scale_i64,
    )

    safe_exponent = torch.where(
        invalid, torch.zeros_like(raw_scale_i64), raw_scale_i64 - 127
    )
    scale = torch.ldexp(
        torch.ones_like(safe_exponent, dtype=torch.float64),
        safe_exponent,
    )
    normalized = finite_x.to(torch.float64) / scale[:, None]
    normalized[invalid] = 0.0

    magnitude = normalized.abs()
    code = torch.zeros_like(magnitude, dtype=torch.uint8, device="cpu")
    code[magnitude > 0.25] = 1
    code[magnitude >= 0.75] = 2
    code[magnitude > 1.25] = 3
    code[magnitude >= 1.75] = 4
    code[magnitude > 2.50] = 5
    code[magnitude >= 3.50] = 6
    code[magnitude > 5.00] = 7
    code |= torch.signbit(normalized).to(torch.uint8) << 3
    return _pack_codes(code), raw_scale_i64.to(torch.uint8).unsqueeze(1)


def _reference(
    packed: torch.Tensor,
    scales: torch.Tensor,
    sign: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    codes = _unpack_codes(packed.cpu())
    table = torch.tensor(_E2M1_VALUES, dtype=torch.float32, device="cpu")
    values = table[codes.to(torch.int64)]
    scale = _decode_e8m0(scales.cpu()).repeat_interleave(32, dim=1)
    dequantized = values * scale

    transposed = dequantized.to(torch.bfloat16).t().contiguous().float()
    K, M = transposed.shape
    blocked = transposed.reshape(K, M // 16, 16)
    blocked = blocked * sign.cpu().float().reshape(1, 1, 16)
    rotated = (blocked @ _hadamard16()).reshape(K, M)

    packed_out = torch.empty((K, M // 2), dtype=torch.uint8, device="cpu")
    scales_out = torch.empty((K, M // 32), dtype=torch.uint8, device="cpu")
    for block in range(M // 32):
        start = block * 32
        packed_block, scale_block = _quantize_even_rne(rotated[:, start : start + 32])
        packed_out[:, start // 2 : (start + 32) // 2] = packed_block
        scales_out[:, block : block + 1] = scale_block
    return packed_out, scales_out


def _alternating_sign() -> torch.Tensor:
    sign = torch.ones(16, dtype=torch.float32, device="cuda")
    sign[1::2] = -1
    return sign


@pytest.mark.parametrize("shape", [(32, 32), (64, 64), (64, 256)])
@pytest.mark.parametrize("alternating", [False, True])
def test_dequant_hadamard_quant_matches_independent_reference(shape, alternating):
    M, K = shape
    generator = torch.Generator(device="cpu").manual_seed(17)
    codes = torch.randint(
        0, 16, (M, K), dtype=torch.uint8, device="cpu", generator=generator
    )
    packed = _pack_codes(codes).cuda()
    scales = torch.randint(
        119,
        136,
        (M, K // 32),
        dtype=torch.uint8,
        device="cpu",
        generator=generator,
    ).cuda()
    sign = _alternating_sign() if alternating else torch.ones(16, device="cuda")

    output, output_scales = dequant_hadamard_quant_mxfp4(packed, scales, sign)
    expected, expected_scales = _reference(packed, scales, sign)

    torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)
    torch.testing.assert_close(output_scales.cpu(), expected_scales, rtol=0, atol=0)
    assert output.shape == (K, M // 2)
    assert output_scales.shape == (K, M // 32)
    assert output.is_contiguous()
    assert output_scales.is_contiguous()


def test_dequant_hadamard_quant_matches_lumen_fused_semantics():
    torch.manual_seed(29)
    x = torch.randn((64, 256), dtype=torch.bfloat16, device="cuda")
    packed, scales = dynamic_mxfp4_quant(x)
    sign = _alternating_sign()

    dequantized = mxfp4_to_f32(packed)
    dequantized *= e8m0_to_f32(scales).repeat_interleave(32, dim=1)
    transposed_bf16 = dequantized.to(torch.bfloat16).t().contiguous()
    K, M = transposed_bf16.shape
    rotated = (
        transposed_bf16.float().reshape(K, M // 16, 16) * sign.reshape(1, 1, 16)
    ) @ _hadamard16().cuda()
    expected, expected_scales = dynamic_mxfp4_quant(rotated.reshape(K, M))

    output, output_scales = dequant_hadamard_quant_mxfp4(packed, scales, sign)

    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    torch.testing.assert_close(output_scales, expected_scales, rtol=0, atol=0)


def test_dequant_hadamard_quant_matches_lumen_fused_wide_scales():
    M = K = 256
    generator = torch.Generator(device="cpu").manual_seed(314159)
    codes = torch.randint(
        0, 16, (M, K), dtype=torch.uint8, device="cpu", generator=generator
    )
    packed = _pack_codes(codes).cuda()
    scales = torch.randint(
        100,
        151,
        (M, K // 32),
        dtype=torch.uint8,
        device="cpu",
        generator=generator,
    ).cuda()
    sign = _alternating_sign()

    dequantized = mxfp4_to_f32(packed)
    dequantized *= e8m0_to_f32(scales).repeat_interleave(32, dim=1)
    transposed_bf16 = dequantized.to(torch.bfloat16).t().contiguous()
    rotated = (
        transposed_bf16.float().reshape(K, M // 16, 16) * sign.reshape(1, 1, 16)
    ) @ _hadamard16().cuda()
    expected, expected_scales = dynamic_mxfp4_quant(rotated.reshape(K, M))

    output, output_scales = dequant_hadamard_quant_mxfp4(packed, scales, sign)

    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    torch.testing.assert_close(output_scales, expected_scales, rtol=0, atol=0)


def test_dequant_hadamard_quant_e8m0_raw_zero_byte_oracle():
    M = K = 32
    codes = torch.full((M, K), 2, dtype=torch.uint8, device="cpu")
    packed = _pack_codes(codes).cuda()
    scales = torch.zeros((M, 1), dtype=torch.uint8, device="cuda")
    sign = torch.ones(16, dtype=torch.float32, device="cuda")

    output, output_scales = dequant_hadamard_quant_mxfp4(packed, scales, sign)

    expected_row = torch.zeros(16, dtype=torch.uint8, device="cuda")
    expected_row[[0, 8]] = 0x06
    torch.testing.assert_close(output, expected_row.expand(M, -1), rtol=0, atol=0)
    assert torch.all(output_scales == 0)


def test_dequant_hadamard_quant_e8m0_raw_254_byte_oracle():
    M = K = 32
    codes = torch.zeros((M, K), dtype=torch.uint8, device="cpu")
    codes[:5] = 3
    codes[16:21] = 3
    packed = _pack_codes(codes).cuda()
    scales = torch.full((M, 1), 254, dtype=torch.uint8, device="cuda")
    sign = torch.ones(16, dtype=torch.float32, device="cuda")

    output, output_scales = dequant_hadamard_quant_mxfp4(packed, scales, sign)

    expected_row = torch.tensor(
        [0x14, 0x11, 0x92, 0x99, 0x14, 0x11, 0x92, 0x99] * 2,
        dtype=torch.uint8,
        device="cuda",
    )
    torch.testing.assert_close(output, expected_row.expand(M, -1), rtol=0, atol=0)
    assert torch.all(output_scales == 254)


def test_dequant_hadamard_quant_e8m0_raw_255_byte_oracle():
    M = K = 32
    packed = _pack_codes(torch.full((M, K), 2, dtype=torch.uint8, device="cpu")).cuda()
    scales = torch.full((M, 1), 255, dtype=torch.uint8, device="cuda")
    sign = torch.ones(16, dtype=torch.float32, device="cuda")

    output, output_scales = dequant_hadamard_quant_mxfp4(packed, scales, sign)

    assert torch.count_nonzero(output).item() == 0
    assert torch.all(output_scales == 255)


def test_raw_255_poisons_its_complete_output_scale_block():
    codes = torch.full((32, 32), 2, dtype=torch.uint8, device="cpu")
    packed = _pack_codes(codes).cuda()
    scales = torch.full((32, 1), 127, dtype=torch.uint8, device="cuda")
    scales[3, 0] = 255
    sign = torch.ones(16, dtype=torch.float32, device="cuda")

    output, output_scales = dequant_hadamard_quant_mxfp4(packed, scales, sign)

    # One NaN-scaled input row reaches an entire H16 half-block. Since the
    # requantized scale covers both H16 groups, its complete 32-value output
    # block is invalid; raw 255 plus zero payload is deterministic.
    assert torch.all(output_scales == 255)
    assert torch.count_nonzero(output).item() == 0


@pytest.mark.parametrize("mixed", [False, True])
def test_signed_zero_matches_lumen_fused_bytes(mixed):
    if mixed:
        codes = (
            torch.arange(32 * 32, dtype=torch.int64, device="cpu").reshape(32, 32)
            % 2
            * 8
        ).to(torch.uint8)
    else:
        codes = torch.full((32, 32), 8, dtype=torch.uint8, device="cpu")
    packed = _pack_codes(codes).cuda()
    scales = torch.full((32, 1), 127, dtype=torch.uint8, device="cuda")
    sign = _alternating_sign()

    dequantized = mxfp4_to_f32(packed)
    dequantized *= e8m0_to_f32(scales).repeat_interleave(32, dim=1)
    transposed_bf16 = dequantized.to(torch.bfloat16).t().contiguous()
    rotated = (
        transposed_bf16.float().reshape(32, 2, 16) * sign.reshape(1, 1, 16)
    ) @ _hadamard16().cuda()
    expected, expected_scales = dynamic_mxfp4_quant(rotated.reshape(32, 32))

    output, output_scales = dequant_hadamard_quant_mxfp4(packed, scales, sign)

    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    torch.testing.assert_close(output_scales, expected_scales, rtol=0, atol=0)


def test_hadamard_normalization_prevents_finite_raw_254_overflow():
    codes = torch.zeros((32, 32), dtype=torch.uint8, device="cpu")
    codes[:5] = 3
    codes[16:21] = 3
    packed = _pack_codes(codes).cuda()
    scales = torch.full((32, 1), 254, dtype=torch.uint8, device="cuda")
    sign = torch.ones(16, dtype=torch.float32, device="cuda")

    _, output_scales = dequant_hadamard_quant_mxfp4(packed, scales, sign)

    # Five 1.5*2^127 terms produce a finite normalized DC coefficient of
    # 1.875*2^127. Summing before the 0.25 normalization would overflow.
    assert torch.all(output_scales == 254)


def test_dequant_hadamard_quant_accepts_canonical_dtypes():
    generator = torch.Generator(device="cpu").manual_seed(19)
    packed = torch.randint(
        0, 256, (32, 32), dtype=torch.uint8, device="cpu", generator=generator
    ).cuda()
    scales = torch.randint(
        120, 135, (32, 2), dtype=torch.uint8, device="cpu", generator=generator
    ).cuda()
    sign = _alternating_sign()

    expected = dequant_hadamard_quant_mxfp4(packed, scales, sign)
    actual = dequant_hadamard_quant_mxfp4(
        packed.view(dtypes.fp4x2), scales.view(dtypes.fp8_e8m0), sign
    )

    for result, reference in zip(actual, expected):
        torch.testing.assert_close(result, reference, rtol=0, atol=0)


@pytest.mark.parametrize("invalid", [0.0, float("inf"), float("nan")])
def test_dequant_hadamard_quant_rejects_invalid_sign_values(invalid):
    packed = torch.zeros((32, 16), dtype=torch.uint8, device="cuda")
    scales = torch.zeros((32, 1), dtype=torch.uint8, device="cuda")
    sign = torch.ones(16, dtype=torch.float32, device="cuda")
    sign[3] = invalid

    with pytest.raises(ValueError, match="finite and equal to"):
        dequant_hadamard_quant_mxfp4(packed, scales, sign)


def test_dequant_hadamard_quant_revalidates_mutated_sign_vector():
    packed = torch.zeros((32, 16), dtype=torch.uint8, device="cuda")
    scales = torch.zeros((32, 1), dtype=torch.uint8, device="cuda")
    sign = torch.ones(16, dtype=torch.float32, device="cuda")

    dequant_hadamard_quant_mxfp4(packed, scales, sign)
    sign[3] = 0

    with pytest.raises(ValueError, match="finite and equal to"):
        dequant_hadamard_quant_mxfp4(packed, scales, sign)


def test_dequant_hadamard_quant_accepts_inference_tensor_sign():
    packed = torch.zeros((32, 16), dtype=torch.uint8, device="cuda")
    scales = torch.zeros((32, 1), dtype=torch.uint8, device="cuda")
    with torch.inference_mode():
        sign = torch.ones(16, dtype=torch.float32, device="cuda")

    output, output_scales = dequant_hadamard_quant_mxfp4(packed, scales, sign)

    assert output.shape == (32, 16)
    assert output_scales.shape == (32, 1)


def test_dequant_hadamard_quant_validates_during_first_graph_capture():
    packed = torch.zeros((32, 16), dtype=torch.uint8, device="cuda")
    scales = torch.zeros((32, 1), dtype=torch.uint8, device="cuda")
    dequant_hadamard_quant_mxfp4(
        packed, scales, torch.ones(16, dtype=torch.float32, device="cuda")
    )
    torch.cuda.synchronize()

    uncached_sign = torch.ones(16, dtype=torch.float32, device="cuda")
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output, output_scales = dequant_hadamard_quant_mxfp4(
            packed, scales, uncached_sign
        )
    graph.replay()

    assert output.shape == (32, 16)
    assert output_scales.shape == (32, 1)


def test_dequant_hadamard_quant_validates_metadata():
    packed = torch.zeros((32, 16), dtype=torch.uint8, device="cuda")
    scales = torch.zeros((32, 1), dtype=torch.uint8, device="cuda")
    sign = torch.ones(16, dtype=torch.float32, device="cuda")

    with pytest.raises(ValueError, match="block_size must be 32"):
        dequant_hadamard_quant_mxfp4(packed, scales, sign, block_size=16)
    with pytest.raises(ValueError, match="g must be 16"):
        dequant_hadamard_quant_mxfp4(packed, scales, sign, g=32)
    with pytest.raises(ValueError, match="scales shape"):
        dequant_hadamard_quant_mxfp4(packed, scales.expand(32, 2), sign)
    with pytest.raises(ValueError, match="same device"):
        dequant_hadamard_quant_mxfp4(packed, scales, sign.cpu())
    with pytest.raises(TypeError, match="data_fp4 must have dtype"):
        dequant_hadamard_quant_mxfp4(packed.float(), scales, sign)
    with pytest.raises(TypeError, match="scales must have dtype"):
        dequant_hadamard_quant_mxfp4(packed, scales.float(), sign)


def test_dequant_hadamard_quant_validates_production_metadata():
    packed = torch.zeros((32, 16), dtype=torch.uint8, device="cuda")
    scales = torch.zeros((32, 1), dtype=torch.uint8, device="cuda")
    sign = torch.ones(16, dtype=torch.float32, device="cuda")

    with pytest.raises(ValueError, match="philox_seed is required"):
        dequant_hadamard_quant_mxfp4(packed, scales, sign, use_sr=True)
    with pytest.raises(ValueError, match="only valid"):
        dequant_hadamard_quant_mxfp4(packed, scales, sign, philox_seed=1)
    with pytest.raises(TypeError, match="must be bool"):
        dequant_hadamard_quant_mxfp4(packed, scales, sign, use_sr=1)
    # (32, 32) logical input: one output scale column cannot fill an 8-wide
    # swizzle chunk; 16 packed payload bytes cannot fill a 32-byte group.
    with pytest.raises(ValueError, match="cannot be swizzled"):
        dequant_hadamard_quant_mxfp4(packed, scales, sign, swizzle_scale=True)
    with pytest.raises(ValueError, match="cannot be shuffled"):
        dequant_hadamard_quant_mxfp4(packed, scales, sign, shuffle_data=True)
    with pytest.raises(ValueError, match="cannot be swizzled"):
        dequant_hadamard_quant_mxfp4(packed, scales, sign, in_scale_swizzled=True)


def test_dequant_hadamard_quant_accepts_strided_inputs():
    generator = torch.Generator(device="cpu").manual_seed(23)
    packed_storage = torch.randint(
        0, 256, (32, 32), dtype=torch.uint8, device="cpu", generator=generator
    ).cuda()
    scale_storage = torch.randint(
        120, 135, (32, 2), dtype=torch.uint8, device="cpu", generator=generator
    ).cuda()
    packed = packed_storage[:, ::2]
    scales = scale_storage[:, ::2]
    sign = _alternating_sign()

    expected = dequant_hadamard_quant_mxfp4(
        packed.contiguous(), scales.contiguous(), sign
    )
    actual = dequant_hadamard_quant_mxfp4(packed, scales, sign)

    for result, reference in zip(actual, expected):
        torch.testing.assert_close(result, reference, rtol=0, atol=0)


def test_dequant_hadamard_quant_fused_layout_matches_shared_shuffles():
    torch.manual_seed(37)
    x = torch.randn((256, 256), dtype=torch.bfloat16, device="cuda")
    packed, scales = dynamic_mxfp4_quant(x)
    sign = _alternating_sign()

    output, output_scales = dequant_hadamard_quant_mxfp4(packed, scales, sign)
    fused, fused_scales = dequant_hadamard_quant_mxfp4(
        packed,
        scales,
        sign,
        swizzle_scale=True,
        shuffle_data=True,
    )

    torch.testing.assert_close(
        fused,
        shuffle_weight(output, arch="gfx950", layout=(16, 16)),
        atol=0,
        rtol=0,
    )
    torch.testing.assert_close(
        fused_scales,
        shuffle_scale_gemm(
            output_scales,
            arch="gfx950",
            preshuffle_factor=32,
            scale_kwidth=8,
        ),
        atol=0,
        rtol=0,
    )


def test_dequant_hadamard_quant_swizzled_input_scales_match_row_major():
    torch.manual_seed(41)
    x = torch.randn((256, 256), dtype=torch.bfloat16, device="cuda")
    packed, scales = dynamic_mxfp4_quant(x)
    sign = _alternating_sign()

    expected = dequant_hadamard_quant_mxfp4(packed, scales, sign)
    swizzled_scales = shuffle_scale_gemm(
        scales.view(torch.uint8),
        arch="gfx950",
        preshuffle_factor=32,
        scale_kwidth=8,
    )
    actual = dequant_hadamard_quant_mxfp4(
        packed, swizzled_scales, sign, in_scale_swizzled=True
    )

    torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
    torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)


def test_dequant_hadamard_quant_sr_is_reproducible_and_reuses_rtn_scales():
    torch.manual_seed(31)
    x = torch.randn((64, 256), dtype=torch.bfloat16, device="cuda")
    packed, scales = dynamic_mxfp4_quant(x)
    sign = _alternating_sign()
    kwargs = {"use_sr": True, "philox_seed": 1234, "philox_offset": 0}

    actual = dequant_hadamard_quant_mxfp4(packed, scales, sign, **kwargs)
    repeated = dequant_hadamard_quant_mxfp4(packed, scales, sign, **kwargs)
    rtn = dequant_hadamard_quant_mxfp4(packed, scales, sign)

    torch.testing.assert_close(actual[0], repeated[0], rtol=0, atol=0)
    torch.testing.assert_close(actual[1], repeated[1], rtol=0, atol=0)
    # RTN and SR share the same E8M0 scale calculation.
    torch.testing.assert_close(actual[1], rtn[1], rtol=0, atol=0)
    assert not torch.equal(actual[0], rtn[0]), "SR must not silently return RTN payload"


def test_dequant_hadamard_quant_rejects_unsupported_arch(monkeypatch):
    monkeypatch.setattr(arch_info, "get_arch", lambda: "gfx942")
    packed = torch.zeros((32, 16), dtype=torch.uint8, device="cpu")
    scales = torch.zeros((32, 1), dtype=torch.uint8, device="cpu")
    sign = torch.ones(16, dtype=torch.float32, device="cpu")

    with pytest.raises(RuntimeError, match="requires gfx950"):
        dequant_hadamard_quant_mxfp4(packed, scales, sign)


def test_dequant_hadamard_quant_requires_published_config(monkeypatch):
    def missing_config(path, required=True):
        assert required
        assert path.endswith(
            "/gfx950/triton/fusions/dequant_hadamard_quant_mxfp4/DEFAULT.json"
        )
        raise FileNotFoundError(path)

    # The public function shares its name with the module; patch the module
    # object directly so the string form cannot resolve to the function.
    import importlib

    _module = importlib.import_module(
        "aiter.ops.triton.quant.dequant_hadamard_quant_mxfp4"
    )
    monkeypatch.setattr(_module, "load_config_json", missing_config)
    packed = torch.zeros((32, 16), dtype=torch.uint8, device="cuda")
    scales = torch.zeros((32, 1), dtype=torch.uint8, device="cuda")
    sign = torch.ones(16, dtype=torch.float32, device="cuda")

    with pytest.raises(
        FileNotFoundError, match="dequant_hadamard_quant_mxfp4/DEFAULT.json"
    ):
        dequant_hadamard_quant_mxfp4(packed, scales, sign)


def test_dequant_hadamard_quant_rejects_noncontiguous_swizzled_scales():
    packed = torch.zeros((256, 128), dtype=torch.uint8, device="cuda")
    storage = torch.zeros((8, 512), dtype=torch.uint8, device="cuda")
    scales = storage[:, ::2]
    assert scales.shape == (8, 256) and not scales.is_contiguous()
    sign = _alternating_sign()
    with pytest.raises(ValueError, match="swizzled scales must be contiguous"):
        dequant_hadamard_quant_mxfp4(packed, scales, sign, in_scale_swizzled=True)


def test_cached_sign_validation_is_recorded_during_graph_capture(monkeypatch):
    packed = torch.zeros((32, 16), dtype=torch.uint8, device="cuda")
    scales = torch.zeros((32, 1), dtype=torch.uint8, device="cuda")
    sign = _alternating_sign()
    dequant_hadamard_quant_mxfp4(packed, scales, sign)
    calls = []
    original = torch._assert_async

    def record_assert(condition, message):
        calls.append(message)
        return original(condition, message)

    monkeypatch.setattr(torch, "_assert_async", record_assert)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        dequant_hadamard_quant_mxfp4(packed, scales, sign)
    graph.replay()
    torch.cuda.synchronize()
    assert calls, "Cached eager validation must not suppress capture assertion"
