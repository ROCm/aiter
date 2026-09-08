# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import math
import os
from typing import NamedTuple

import pytest
import torch
import torch._dynamo

from aiter import dtypes
from aiter.jit.core import AITER_ROOT_DIR
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.mha_v4 import (
    MHA_V4_LOG2E,
    AttentionFormat,
    AttentionScaleMode,
    mha_v4,
    mha_v4_kv_tile,
    mha_v4_mxfp8,
    mha_v4_packed,
    mha_v4_q_multiplier,
    mha_v4_sparse_work_table,
    mxfp4_k_view,
    mxfp4_v_view,
    mxfp6_k_view,
    native_fp8_format,
    quantize_fp8,
    quantize_fp8_rotated,
    quantize_int8,
    quantize_mxfp4_k,
    quantize_mxfp4_q,
    quantize_mxfp6_k,
    quantize_mxfp8_k,
    quantize_mxfp8_q,
    quantize_v_mxfp4,
    quantize_v_mxfp6,
    rotate_activation_hd128,
    rotate_activation_mxfp6_quant,
    scale_modes_for_formats,
)
from aiter.ops.triton.attention.utils import block_attn_mask_to_ragged_lut
from aiter.ops.triton.quant.mxfp6_fmha_pack import (
    _v_direct_kvtab,
    fp6_k_raw_buffer_sizes,
    quantize_fp6_v_clean_triton,
    quantize_fp6_v_data_scale_triton,
    reorder_fp6_k_lds_order_triton,
)
from aiter.ops.triton.quant.quant import dynamic_mxfp8_quant
from aiter.ops.triton.quant.sage_attention_quant_wrappers import (
    fp4_v_padded_sequence,
    fp4_v_raw_buffer_size,
)


def _e2m1_code_ties_low(value):
    magnitude = value.abs()
    code = sum(
        magnitude > midpoint for midpoint in (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)
    ).to(torch.uint8)
    return code | ((value < 0).to(torch.uint8) << 3)


def _rotate_hd128_reference(value):
    rotated = value.float()
    group_size = 1
    while group_size < 128:
        pairs = rotated.reshape(*value.shape[:-1], -1, 2, group_size)
        left = pairs[..., 0, :]
        right = pairs[..., 1, :]
        rotated = torch.cat((left + right, left - right), dim=-1).reshape(value.shape)
        group_size *= 2
    return (rotated / 128**0.5).to(value.dtype)


def _reference_mxfp4_v(value):
    batch, sequence, heads, _ = value.shape
    padded_sequence = fp4_v_padded_sequence(sequence)
    tiles = padded_sequence // 128
    padded = torch.nn.functional.pad(
        value.float(), (0, 0, 0, 0, 0, padded_sequence - sequence)
    )
    padded = padded.permute(0, 2, 1, 3)

    column = torch.arange(64, device=value.device)
    lane = column % 32
    permutation = 4 * (lane // 8) + 16 * ((lane // 4) % 2) + lane % 4
    tau64 = 32 * (column // 32) + permutation
    kperm = torch.empty(64, dtype=torch.long, device=value.device)
    kperm[tau64] = column

    raw = torch.zeros(
        fp4_v_raw_buffer_size(batch, sequence, heads),
        dtype=torch.uint8,
        device=value.device,
    )
    payload = raw[:-64].view(batch, heads, tiles * 8192)
    scale = torch.empty(
        (batch, heads, tiles * 512), dtype=torch.uint8, device=value.device
    )
    for tile in range(tiles):
        for channel_block in range(4):
            for token_half in range(2):
                unit = 2 * channel_block + token_half
                tokens = tile * 128 + token_half * 64 + kperm
                channels = slice(channel_block * 32, (channel_block + 1) * 32)
                block = padded[:, :, tokens, channels]
                exponents = []
                normalized = torch.empty_like(block)
                for token_block in range(2):
                    columns = slice(token_block * 32, (token_block + 1) * 32)
                    amax = block[:, :, columns].abs().amax(dim=2)
                    exponent = torch.ceil(
                        torch.log2(torch.clamp_min(amax, 1e-12) / 6.0)
                    )
                    exponents.append(exponent)
                    normalized[:, :, columns] = block[:, :, columns] / torch.exp2(
                        exponent[:, :, None]
                    )

                code = _e2m1_code_ties_low(normalized)
                packed = code[..., 0::2] | (code[..., 1::2] << 4)
                payload[
                    :, :, tile * 8192 + unit * 1024 : tile * 8192 + (unit + 1) * 1024
                ] = packed.flatten(2)

                scale_base = tile * 512 + token_half * 256
                for token_block, exponent in enumerate(exponents):
                    encoded = (exponent + 127).clamp(0, 255).to(torch.uint8)
                    for pair in range(16):
                        offset = (
                            scale_base + token_block * 128 + 8 * pair + channel_block
                        )
                        scale[:, :, offset] = encoded[:, :, 2 * pair]
                        scale[:, :, offset + 4] = encoded[:, :, 2 * pair + 1]
    return raw, scale


@pytest.fixture(autouse=True)
def isolate_dynamo_cache():
    """Keep each test's ``torch.compile`` behaviour independent of the tests that ran before it.

    Dynamo caches compiled entries per code object, and this file compiles ``mha_v4`` under many
    format combinations. Sharing that cache across tests means the suite drifts toward the recompile
    limit and whichever test compiles last fails under ``fullgraph=True`` -- a failure that reports
    against a kernel while actually depending on how many earlier tests got far enough to compile.
    """
    torch._dynamo.reset()
    yield


def test_attention_format_ids_are_stable():
    assert int(AttentionFormat.FP32) == 0
    assert int(AttentionFormat.FP16) == 1
    assert int(AttentionFormat.BF16) == 2
    assert int(AttentionFormat.FP8_E4M3) == 3
    assert AttentionFormat.FP8 is AttentionFormat.FP8_E4M3
    assert int(AttentionFormat.FP8_E4M3_FNUZ) == 4
    assert int(AttentionFormat.FP8_E5M2) == 5
    assert int(AttentionFormat.FP8_E5M2_FNUZ) == 6
    assert int(AttentionFormat.FP6_E2M3) == 7
    assert AttentionFormat.MXFP6 is AttentionFormat.FP6_E2M3
    assert int(AttentionFormat.FP6_E3M2) == 8
    assert AttentionFormat.MXBF6 is AttentionFormat.FP6_E3M2
    assert int(AttentionFormat.FP4_E2M1) == 9
    assert AttentionFormat.MXFP4 is AttentionFormat.FP4_E2M1
    assert int(AttentionFormat.INT8) == 10
    assert int(AttentionFormat.UINT8) == 11
    assert int(AttentionFormat.INT4) == 12
    assert int(AttentionFormat.UINT4) == 13


def test_mha_v4_q_multiplier_recipe():
    softmax_scale = 128**-0.5
    assert mha_v4_q_multiplier(softmax_scale) == softmax_scale * MHA_V4_LOG2E


def test_mha_v4_f8f6_scale_recipe():
    assert scale_modes_for_formats(
        AttentionFormat.FP8, AttentionFormat.FP8, AttentionFormat.MXFP6
    ) == (
        AttentionScaleMode.F32_PER_TENSOR,
        AttentionScaleMode.F32_PER_TENSOR,
        AttentionScaleMode.E8M0_PER_1X32,
    )


def test_mha_v4_bf16_scale_recipe():
    assert scale_modes_for_formats(
        AttentionFormat.BF16, AttentionFormat.BF16, AttentionFormat.BF16
    ) == (
        AttentionScaleMode.NONE,
        AttentionScaleMode.NONE,
        AttentionScaleMode.NONE,
    )


def test_mha_v4_rejects_f8f4_format_pair():
    with pytest.raises(ValueError, match="matching FP8 or MXFP6 V"):
        scale_modes_for_formats(
            AttentionFormat.FP8, AttentionFormat.FP8, AttentionFormat.MXFP4
        )


def test_mha_v4_f8f6_v_kv_table_matches_live_p_pack():
    expected = []
    for lane in range(64):
        row = []
        for field in range(32):
            physical = 32 * (lane // 32) + field
            paired = (
                (physical & 0x0F) | ((physical & 0x10) << 1) | ((physical & 0x20) >> 1)
            )
            group, byte = divmod(paired, 32)
            row.append(
                32 * (byte // 16) + 8 * ((byte % 16) // 4) + byte % 4 + 4 * group
            )
        expected.append(row)

    assert _v_direct_kvtab().tolist() == expected


@pytest.mark.skipif(get_gfx() != "gfx950", reason="gfx950 MXFP6 V packing")
def test_mha_v4_mxfp6_v_layout_contract():
    value = torch.randn((1, 256, 2, 128), device="cuda", dtype=torch.bfloat16)

    packed, scale = quantize_v_mxfp6(value)

    assert packed.shape == value.shape
    assert packed.dtype == torch.uint8
    assert packed.stride() == (2 * 2 * 12288, 96, 2 * 12288, 1)
    assert scale.shape == (1, 2, 2 * 512)
    assert scale.dtype == torch.uint8


@pytest.mark.skipif(get_gfx() != "gfx950", reason="gfx950 MXFP6 V packing")
@pytest.mark.parametrize("sequence", [256, 257])
def test_mha_v4_mxfp6_v_direct_buffers_match_combined_reference(sequence):
    torch.manual_seed(sequence)
    value = torch.randn((1, sequence, 2, 128), device="cuda", dtype=torch.bfloat16)
    tiles = (sequence + 127) // 128
    if sequence % 128:
        reference_value = torch.cat(
            [value, value[:, -1:].expand(-1, tiles * 128 - sequence, -1, -1)],
            dim=1,
        )
    else:
        reference_value = value

    combined = quantize_fp6_v_clean_triton(reference_value, direct_p=True).view(
        1, 2, tiles, 12800
    )
    data, scale = quantize_fp6_v_data_scale_triton(value)
    expected_data = combined[..., :12288].contiguous().view(-1)

    scale_tail = combined[..., 12288:].view(1, 2, tiles, 128, 4)
    lane = torch.arange(64, device=value.device)
    channel = lane[:, None] % 32 + 32 * torch.arange(4, device=value.device)[None, :]
    expected_scale = torch.stack(
        [scale_tail[..., channel, 2 * half + lane[:, None] // 32] for half in range(2)],
        dim=-3,
    ).contiguous()

    assert torch.equal(data[: expected_data.numel()], expected_data)
    assert torch.count_nonzero(data[expected_data.numel() :]) == 0
    assert torch.equal(scale, expected_scale.view(-1))


@pytest.mark.skipif(get_gfx() != "gfx950", reason="gfx950 per-tensor quantization")
@pytest.mark.parametrize("clip", [1.0, 0.9])
def test_mha_v4_int8_quantization_matches_torch(clip):
    torch.manual_seed(17)
    value = torch.randn((2, 257, 3, 128), device="cuda", dtype=torch.bfloat16)
    expected_scale = value.float().abs().max() * clip / 127.0
    expected = torch.clamp(torch.round(value.float() / expected_scale), -128, 127).to(
        torch.int8
    )

    actual, scale = quantize_int8(value, clip)

    assert torch.equal(actual, expected)
    assert torch.equal(scale, expected_scale.reshape(1))


@pytest.mark.skipif(get_gfx() != "gfx950", reason="gfx950 per-tensor quantization")
def test_mha_v4_fp8_quantization_matches_torch():
    torch.manual_seed(19)
    value = torch.randn((2, 257, 3, 128), device="cuda", dtype=torch.bfloat16)
    expected_scale = value.float().abs().max() / torch.finfo(torch.float8_e4m3fn).max
    expected = (value.float() / expected_scale).to(torch.float8_e4m3fn)

    actual, scale = quantize_fp8(value)

    assert torch.equal(actual, expected)
    assert torch.equal(scale, expected_scale.reshape(1))


@pytest.mark.skipif(
    get_gfx() not in ("gfx942", "gfx950"),
    reason="gfx942/gfx950 hd128 rotation",
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("sequence,heads", [(1, 1), (129, 5), (2048, 1)])
def test_mha_v4_rotation_matches_an_explicit_hadamard_matrix(dtype, sequence, heads):
    """The rotation is the orthonormal hd128 Walsh-Hadamard transform, not merely self-consistent.

    Checked against a Sylvester matrix applied as an fp32 matmul, which shares no code with the
    kernel's butterfly. That pins the transform and its 1/sqrt(128) normalization, where comparing
    against another kernel would only pin the two against each other -- and the recipe test below
    cannot help, because it evaluates the same rotation on both sides.

    The comparison is exact rather than toleranced: 128 values of one 16-bit dtype summed in fp32
    lose nothing, so the only rounding is the final cast back to that dtype.
    """
    torch.manual_seed(31)
    value = torch.randn((1, sequence, heads, 128), device="cuda", dtype=dtype)
    rotated = torch.empty_like(value)
    rotate_activation_hd128(rotated, value)

    index = torch.arange(128, device=value.device)
    overlap = index.view(-1, 1) & index.view(1, -1)
    parity = torch.zeros_like(overlap)
    for bit in range(7):
        parity ^= (overlap >> bit) & 1
    hadamard = torch.where(parity.bool(), -1.0, 1.0)
    assert torch.equal(
        hadamard @ hadamard.T / 128, torch.eye(128, device=value.device)
    ), "the reference matrix is not an orthonormal Hadamard matrix"

    expected = (value.float().reshape(-1, 128) @ hadamard) / math.sqrt(128)
    assert torch.equal(rotated, expected.to(dtype).reshape(value.shape))


@pytest.mark.skipif(
    get_gfx() not in ("gfx942", "gfx950"),
    reason="gfx942/gfx950 rotated FP8 quantization",
)
@pytest.mark.parametrize("sequence,heads", [(257, 3), (512, 1), (2048, 1)])
def test_mha_v4_rotated_fp8_quantization_matches_reference(sequence, heads):
    torch.manual_seed(23)
    value = torch.randn((1, heads, sequence, 128), device="cuda", dtype=torch.bfloat16)
    value = value.permute(0, 2, 1, 3).contiguous()
    expected_rotated = _rotate_hd128_reference(value)
    rotated = torch.empty_like(value)
    rotate_activation_hd128(rotated, value)
    expected, expected_scale = quantize_fp8(expected_rotated)

    actual, scale = quantize_fp8_rotated(value)

    assert torch.equal(rotated, expected_rotated)
    assert torch.equal(actual, expected)
    assert torch.equal(scale, expected_scale)


def test_mha_v4_rotated_fp8_quantization_rejects_noncontiguous_input():
    value = torch.randn((1, 1, 128, 2), device="cuda", dtype=torch.bfloat16)
    value = value.transpose(-1, -2)

    with pytest.raises(ValueError, match="requires contiguous hd128 input"):
        quantize_fp8_rotated(value)


@pytest.mark.skipif(
    get_gfx() not in ("gfx942", "gfx950"),
    reason="gfx942/gfx950 activation rotation",
)
def test_mha_v4_rotate_activation_hd128_accepts_empty_input():
    value = torch.empty((1, 0, 1, 128), device="cuda", dtype=torch.bfloat16)
    rotated = torch.empty_like(value)

    rotate_activation_hd128(rotated, value)

    assert rotated.shape == value.shape
    assert rotated.numel() == 0


@pytest.mark.skipif(
    get_gfx() not in ("gfx942", "gfx950"),
    reason="gfx942/gfx950 FP8 recipe validation",
)
def test_mha_v4_fp8_raw_recipe_matches_rotated_packed():
    torch.manual_seed(29)
    q = torch.randn((1, 512, 5, 128), device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    fp8_format = native_fp8_format()

    q_quantized, q_descale = quantize_fp8_rotated(q)
    k_quantized, k_descale = quantize_fp8_rotated(k)
    v_quantized, v_descale = quantize_fp8(v)
    expected = mha_v4_packed(
        q_quantized,
        k_quantized,
        v_quantized,
        q_descale,
        k_descale,
        v_descale,
        fp8_format,
        fp8_format,
        fp8_format,
        *scale_modes_for_formats(fp8_format, fp8_format, fp8_format),
    )

    actual = mha_v4(q, k, v, fp8_format, fp8_format, fp8_format)
    compiled = torch.compile(mha_v4, fullgraph=True)(
        q, k, v, fp8_format, fp8_format, fp8_format
    )
    torch.cuda.synchronize()

    assert torch.equal(actual, expected)
    assert torch.equal(compiled, expected)


@pytest.mark.skipif(get_gfx() != "gfx950", reason="gfx950 MXFP8 quantization")
@pytest.mark.parametrize("case", ["random", "zero", "powers", "extreme"])
def test_mha_v4_mxfp8_q_matches_unfused_pipeline(case):
    from aiter import dtypes

    if case == "random":
        torch.manual_seed(29)
        value = torch.randn((1, 129, 5, 128), device="cuda", dtype=torch.bfloat16)
    elif case == "zero":
        value = torch.zeros((1, 7, 5, 128), device="cuda", dtype=torch.bfloat16)
    elif case == "powers":
        powers = torch.tensor(
            [0.0] + [2.0**exponent for exponent in range(-12, 13)],
            device="cuda",
            dtype=torch.bfloat16,
        )
        value = powers.repeat((128 + powers.numel() - 1) // powers.numel())[:128]
        value = value.reshape(1, 1, 1, 128)
    else:
        value = torch.full(
            (1, 1, 1, 128),
            torch.finfo(torch.bfloat16).max,
            device="cuda",
            dtype=torch.bfloat16,
        )

    multiplier = mha_v4_q_multiplier(128**-0.5)
    rotated = torch.empty_like(value)
    rotate_activation_hd128(rotated, value)
    expected, expected_scale = dynamic_mxfp8_quant(
        rotated * multiplier, quant_dtype=dtypes.fp8
    )

    actual, scale = quantize_mxfp8_q(value, multiplier)

    assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))
    assert torch.equal(scale, expected_scale)


@pytest.mark.skipif(get_gfx() != "gfx950", reason="gfx950 MXFP8 quantization")
@pytest.mark.parametrize("case", ["random", "zero", "extreme"])
def test_mha_v4_mxfp8_k_matches_unfused_pipeline(case):
    from aiter import dtypes

    if case == "random":
        torch.manual_seed(41)
        value = torch.randn((1, 129, 5, 128), device="cuda", dtype=torch.bfloat16)
    elif case == "zero":
        value = torch.zeros((1, 7, 5, 128), device="cuda", dtype=torch.bfloat16)
    else:
        value = torch.full(
            (1, 1, 1, 128),
            torch.finfo(torch.bfloat16).max,
            device="cuda",
            dtype=torch.bfloat16,
        )

    rotated = torch.empty_like(value)
    rotate_activation_hd128(rotated, value)
    expected, expected_scale = dynamic_mxfp8_quant(rotated, quant_dtype=dtypes.fp8)

    actual, scale = quantize_mxfp8_k(value)

    assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))
    assert torch.equal(scale, expected_scale)


@pytest.mark.skipif(get_gfx() != "gfx950", reason="gfx950 per-tensor quantization")
@pytest.mark.parametrize(
    "quantize", [quantize_int8, quantize_fp8, quantize_fp8_rotated]
)
def test_mha_v4_per_tensor_quantization_handles_zero(quantize):
    value = torch.zeros((1, 128, 2, 128), device="cuda", dtype=torch.bfloat16)

    actual, scale = quantize(value)

    assert torch.count_nonzero(actual) == 0
    assert torch.equal(scale, torch.ones_like(scale))


def test_mha_v4_raw_buffer_sizes_are_stable():
    assert fp6_k_raw_buffer_sizes(1, 128, 1) == (17408 + 256, 128 * 4 + 64)
    assert fp6_k_raw_buffer_sizes(2, 129, 3) == (
        2 * 3 * 2 * 17408 + 256,
        2 * 129 * 3 * 4 + 64,
    )
    assert fp4_v_padded_sequence(128) == 128
    assert fp4_v_padded_sequence(129) == 256
    assert fp4_v_raw_buffer_size(2, 129, 3) == 2 * 256 * 3 * 64 + 64


@pytest.mark.parametrize(
    "batch,sequence,heads", [(1, 128, 5), (1, 129, 2), (2, 257, 3)]
)
def test_mha_v4_mxfp4_v_backing_storage_covers_logical_view(batch, sequence, heads):
    padded_sequence = fp4_v_padded_sequence(sequence)
    payload_size = batch * heads * padded_sequence * 64
    raw_size = fp4_v_raw_buffer_size(batch, sequence, heads)
    max_logical_offset = (
        (batch - 1) * heads * padded_sequence * 64
        + (sequence - 1) * 64
        + (heads - 1) * padded_sequence * 64
        + 127
    )

    assert raw_size == payload_size + 64
    assert max_logical_offset < raw_size


@pytest.mark.skipif(get_gfx() != "gfx950", reason="gfx950 MXFP4 V validation")
@pytest.mark.parametrize("sequence", [1, 127, 128, 129, 257])
def test_mha_v4_mxfp4_v_pack_matches_reference(sequence):
    torch.manual_seed(sequence)
    value = torch.randn((2, sequence, 3, 128), device="cuda", dtype=torch.bfloat16)
    raw, scale = quantize_v_mxfp4(value)
    raw_again, scale_again = quantize_v_mxfp4(value)
    expected_raw, expected_scale = _reference_mxfp4_v(value)

    assert raw.shape == (fp4_v_raw_buffer_size(2, sequence, 3),)
    assert scale.shape == (2, 3, ((sequence + 127) // 128) * 512)
    assert raw.dtype == scale.dtype == torch.uint8
    assert torch.equal(raw, expected_raw)
    assert torch.equal(scale, expected_scale)
    assert torch.equal(raw, raw_again)
    assert torch.equal(scale, scale_again)
    assert torch.count_nonzero(raw[-64:]) == 0
    logical = mxfp4_v_view(raw, scale, sequence)
    assert logical.shape == value.shape
    assert logical.stride() == (
        3 * fp4_v_padded_sequence(sequence) * 64,
        64,
        fp4_v_padded_sequence(sequence) * 64,
        1,
    )


@pytest.mark.skipif(get_gfx() != "gfx950", reason="gfx950 MXFP6 K validation")
@pytest.mark.parametrize("sequence", [128, 129, 257])
def test_mha_v4_mxfp6_k_raw_views(sequence):
    torch.manual_seed(sequence)
    value = torch.randn((2, sequence, 3, 128), device="cuda", dtype=torch.bfloat16)
    dense = torch.empty((2, sequence, 3, 96), device="cuda", dtype=torch.uint8)
    dense_scale = torch.empty((2, sequence, 3, 4), device="cuda", dtype=torch.uint8)
    rotate_activation_mxfp6_quant(dense, dense_scale, value, 1.0)
    expected_raw, expected_scale_raw = reorder_fp6_k_lds_order_triton(
        dense, dense_scale, return_raw=True
    )
    raw, scale_raw = quantize_mxfp6_k(value)
    packed, scale = mxfp6_k_view(raw, scale_raw, 2, sequence, 3)

    assert packed.shape == (2, sequence, 3, 96)
    assert scale.shape == (2, sequence, 3, 4)
    assert packed.untyped_storage().data_ptr() == raw.untyped_storage().data_ptr()
    assert scale.untyped_storage().data_ptr() == scale_raw.untyped_storage().data_ptr()
    tiles = (sequence + 127) // 128
    for batch_head in range(2 * 3):
        for tile in range(tiles):
            base = batch_head * tiles * 17408 + tile * 17408
            assert torch.equal(
                raw[base : base + 12288], expected_raw[base : base + 12288]
            )
            assert torch.equal(
                raw[base + 16384 : base + 17408],
                expected_raw[base + 16384 : base + 17408],
            )
    assert torch.equal(
        scale_raw[: dense_scale.numel()], expected_scale_raw[: dense_scale.numel()]
    )


@pytest.mark.skipif(get_gfx() != "gfx950", reason="gfx950 MXFP4 K validation")
@pytest.mark.parametrize("sequence", [1, 127, 128, 129, 257])
def test_mha_v4_mxfp4_k_coalesced_layout(sequence):
    torch.manual_seed(sequence)
    value = torch.randn((2, sequence, 3, 128), device="cuda", dtype=torch.bfloat16)
    dense, dense_scale = quantize_mxfp4_q(value, 1.0)
    raw, scale = quantize_mxfp4_k(value)
    coalesced = mxfp4_k_view(raw, scale)

    tiles = (sequence + 127) // 128
    token = torch.arange(sequence, device="cuda")
    chunk = torch.arange(4, device="cuda")
    byte = torch.arange(16, device="cuda")
    raw_offset = (
        torch.arange(2, device="cuda")[:, None, None, None, None] * (3 * tiles * 8192)
        + torch.arange(3, device="cuda")[None, None, :, None, None] * (tiles * 8192)
        + (token // 128)[None, :, None, None, None] * 8192
        + chunk[None, None, None, :, None] * 2048
        + (token % 128)[None, :, None, None, None] * 16
        + byte[None, None, None, None, :]
    )
    expected = dense.unflatten(-1, (4, 16))
    assert torch.equal(raw[raw_offset], expected)

    assert torch.equal(scale, dense_scale)
    assert coalesced.stride() == (3 * tiles * 8192, 64, tiles * 8192, 1)


def test_mha_v4_rejects_unsupported_contracts():
    q = torch.empty((1, 128, 2, 128), device="cuda", dtype=torch.bfloat16)
    with pytest.raises(NotImplementedError, match="do not produce LSE"):
        mha_v4(
            q,
            q,
            q,
            AttentionFormat.FP8,
            AttentionFormat.FP8,
            AttentionFormat.FP8,
            return_lse=True,
        )
    with pytest.raises(ValueError, match="matching Q and K formats"):
        mha_v4(
            q,
            q,
            q,
            AttentionFormat.FP8,
            AttentionFormat.INT8,
            AttentionFormat.FP8,
        )


@pytest.mark.parametrize(
    "q_format",
    [
        AttentionFormat.FP16,
        AttentionFormat.FP8_E5M2,
        AttentionFormat.FP8_E5M2_FNUZ,
        AttentionFormat.UINT8,
        AttentionFormat.INT4,
        AttentionFormat.UINT4,
    ],
)
def test_mha_v4_rejects_reserved_raw_formats(q_format):
    q = torch.empty((1, 128, 2, 128), device="cuda", dtype=torch.bfloat16)
    with pytest.raises((ValueError, NotImplementedError)):
        mha_v4(
            q,
            q,
            q,
            q_format,
            q_format,
            AttentionFormat.FP8,
        )


@pytest.mark.skipif(get_gfx() != "gfx950", reason="gfx950 MHA v4 validation")
def test_mha_v4_packed_rejects_wrong_scale_recipe():
    q = torch.zeros((1, 128, 2, 128), device="cuda", dtype=torch.int8)
    v = torch.zeros((1, 128, 2, 128), device="cuda", dtype=torch.float8_e4m3fn)
    scale = torch.ones(1, device="cuda", dtype=torch.float32)
    with pytest.raises(ValueError, match="unsupported scale recipe"):
        mha_v4_packed(
            q,
            q,
            v,
            scale,
            scale,
            scale,
            AttentionFormat.INT8,
            AttentionFormat.INT8,
            AttentionFormat.FP8,
            AttentionScaleMode.E8M0_PER_1X32,
            AttentionScaleMode.E8M0_PER_1X32,
            AttentionScaleMode.F32_PER_CHANNEL,
        )


@pytest.mark.skipif(get_gfx() != "gfx950", reason="gfx950 MXFP8 validation")
def test_mha_v4_packed_accepts_mxfp8_scale_recipe():
    q = torch.zeros((1, 128, 2, 128), device="cuda", dtype=torch.float8_e4m3fn)
    qk_scale = torch.ones((1, 128, 2, 4), device="cuda", dtype=torch.uint8)
    v_scale = torch.ones(1, device="cuda", dtype=torch.float32)
    mha_v4_packed(
        q,
        q,
        q,
        qk_scale,
        qk_scale,
        v_scale,
        AttentionFormat.FP8,
        AttentionFormat.FP8,
        AttentionFormat.FP8,
        AttentionScaleMode.E8M0_PER_1X32,
        AttentionScaleMode.E8M0_PER_1X32,
        AttentionScaleMode.F32_PER_TENSOR,
    )


@pytest.mark.skipif(get_gfx() != "gfx950", reason="gfx950 MHA v4 validation")
def test_mha_v4_packed_rejects_wrong_fp8_encoding():
    q = torch.zeros((1, 128, 2, 128), device="cuda", dtype=torch.float8_e4m3fn)
    scale = torch.ones(1, device="cuda", dtype=torch.float32)
    with pytest.raises(RuntimeError, match="must be FP8 E4M3 FNUZ"):
        mha_v4_packed(
            q,
            q,
            q,
            scale,
            scale,
            scale,
            AttentionFormat.FP8_E4M3_FNUZ,
            AttentionFormat.FP8_E4M3_FNUZ,
            AttentionFormat.FP8_E4M3_FNUZ,
            AttentionScaleMode.F32_PER_TENSOR,
            AttentionScaleMode.F32_PER_TENSOR,
            AttentionScaleMode.F32_PER_TENSOR,
        )


@pytest.mark.skipif(get_gfx() != "gfx950", reason="gfx950 MXFP4 K validation")
def test_mha_v4_packed_rejects_wrong_mxfp4_k_layout():
    q = torch.zeros((1, 128, 2, 64), device="cuda", dtype=torch.uint8)
    scale = torch.ones((1, 128, 2, 4), device="cuda", dtype=torch.uint8)
    v_fp8 = torch.zeros((1, 128, 2, 128), device="cuda", dtype=torch.float8_e4m3fn)
    v_scale = torch.ones((1, 2, 128), device="cuda", dtype=torch.float32)

    for v_format, value, value_scale, v_scale_mode in (
        (AttentionFormat.FP8, v_fp8, v_scale, AttentionScaleMode.F32_PER_CHANNEL),
        (
            AttentionFormat.MXFP4,
            q.new_zeros((1, 128, 2, 128)),
            q.new_zeros((1, 2, 512)),
            AttentionScaleMode.E8M0_PER_1X32,
        ),
    ):
        with pytest.raises(ValueError, match="coalesced MHA v4 tile layout"):
            mha_v4_packed(
                q,
                q,
                value,
                scale,
                scale,
                value_scale,
                AttentionFormat.MXFP4,
                AttentionFormat.MXFP4,
                v_format,
                AttentionScaleMode.E8M0_PER_1X32,
                AttentionScaleMode.E8M0_PER_1X32,
                v_scale_mode,
            )

    raw, k_scale = quantize_mxfp4_k(
        torch.zeros((1, 128, 2, 128), device="cuda", dtype=torch.bfloat16)
    )
    coalesced_k = mxfp4_k_view(raw, k_scale)
    assert coalesced_k.stride() == (16384, 64, 8192, 1)


@pytest.mark.skipif(get_gfx() != "gfx950", reason="gfx950 MHA v4 validation")
@pytest.mark.parametrize(
    ("q_format", "v_format"),
    [
        (AttentionFormat.BF16, AttentionFormat.BF16),
        (AttentionFormat.INT8, AttentionFormat.FP8),
        (AttentionFormat.FP8, AttentionFormat.FP8),
    ],
)
def test_mha_v4_zero_inputs_are_finite(q_format, v_format):
    q = torch.zeros((1, 128, 2, 128), device="cuda", dtype=torch.bfloat16)
    out = mha_v4(q, q, q, q_format, q_format, v_format)
    torch.cuda.synchronize()
    assert torch.count_nonzero(out) == 0
    assert torch.isfinite(out).all()


@pytest.mark.skipif(get_gfx() != "gfx950", reason="gfx950 GQA validation")
def test_mha_v4_mxfp4_gqa_matches_repeated_kv():
    torch.manual_seed(41)
    q = torch.randn((2, 129, 64, 128), device="cuda", dtype=torch.bfloat16)
    k = torch.randn((2, 257, 4, 128), device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)

    gqa = mha_v4(
        q,
        k,
        v,
        AttentionFormat.MXFP4,
        AttentionFormat.MXFP4,
        AttentionFormat.FP8,
    )
    mha = mha_v4(
        q,
        k.repeat_interleave(16, dim=2),
        v.repeat_interleave(16, dim=2),
        AttentionFormat.MXFP4,
        AttentionFormat.MXFP4,
        AttentionFormat.FP8,
    )
    torch.cuda.synchronize()

    assert torch.equal(gqa, mha)


@pytest.mark.skipif(
    get_gfx() not in ("gfx942", "gfx950"), reason="gfx942/gfx950 I8FP8 validation"
)
def test_mha_v4_packed_i8fp8_compile_parity():
    torch.manual_seed(17)
    q = torch.randint(-32, 33, (1, 512, 5, 128), device="cuda", dtype=torch.int8)
    k = torch.randint(-32, 33, (1, 512, 5, 128), device="cuda", dtype=torch.int8)
    v = torch.randn((1, 512, 5, 128), device="cuda").to(dtypes.fp8)
    q_descale = torch.tensor([0.02], device="cuda")
    k_descale = torch.tensor([0.03], device="cuda")
    v_descale = torch.tensor([0.04], device="cuda")
    scale = 128**-0.5
    fp8_format = native_fp8_format()

    eager = mha_v4_packed(
        q,
        k,
        v,
        q_descale,
        k_descale,
        v_descale,
        AttentionFormat.INT8,
        AttentionFormat.INT8,
        fp8_format,
        AttentionScaleMode.F32_PER_TENSOR,
        AttentionScaleMode.F32_PER_TENSOR,
        AttentionScaleMode.F32_PER_TENSOR,
        softmax_scale=scale,
    )
    compiled = torch.compile(mha_v4_packed, fullgraph=True)(
        q,
        k,
        v,
        q_descale,
        k_descale,
        v_descale,
        AttentionFormat.INT8,
        AttentionFormat.INT8,
        fp8_format,
        AttentionScaleMode.F32_PER_TENSOR,
        AttentionScaleMode.F32_PER_TENSOR,
        AttentionScaleMode.F32_PER_TENSOR,
        softmax_scale=scale,
    )
    torch.cuda.synchronize()
    assert torch.equal(eager, compiled)


@pytest.mark.skipif(
    get_gfx() not in ("gfx942", "gfx950"), reason="gfx942/gfx950 FP8 validation"
)
def test_mha_v4_packed_fp8_compile_parity():
    torch.manual_seed(23)
    q = torch.randn((1, 512, 5, 128), device="cuda").to(dtypes.fp8)
    k = torch.randn((1, 512, 5, 128), device="cuda").to(dtypes.fp8)
    v = torch.randn((1, 512, 5, 128), device="cuda").to(dtypes.fp8)
    q_descale = torch.tensor([0.02], device="cuda")
    k_descale = torch.tensor([0.03], device="cuda")
    v_descale = torch.tensor([0.04], device="cuda")
    fp8_format = native_fp8_format()
    scale_modes = scale_modes_for_formats(fp8_format, fp8_format, fp8_format)
    scale = 128**-0.5

    eager = mha_v4_packed(
        q,
        k,
        v,
        q_descale,
        k_descale,
        v_descale,
        fp8_format,
        fp8_format,
        fp8_format,
        *scale_modes,
        softmax_scale=scale,
    )
    compiled = torch.compile(mha_v4_packed, fullgraph=True)(
        q,
        k,
        v,
        q_descale,
        k_descale,
        v_descale,
        fp8_format,
        fp8_format,
        fp8_format,
        *scale_modes,
        softmax_scale=scale,
    )
    torch.cuda.synchronize()
    assert torch.equal(eager, compiled)


@pytest.mark.skipif(get_gfx() != "gfx950", reason="gfx950 MHA v4 validation")
def test_mha_v4_native_schema_mutates_only_out():
    q = torch.zeros((1, 128, 2, 128), device="cuda", dtype=torch.float8_e4m3fn)
    scale = torch.ones(1, device="cuda", dtype=torch.float32)
    mha_v4_packed(
        q,
        q,
        q,
        scale,
        scale,
        scale,
        AttentionFormat.FP8,
        AttentionFormat.FP8,
        AttentionFormat.FP8,
        AttentionScaleMode.F32_PER_TENSOR,
        AttentionScaleMode.F32_PER_TENSOR,
        AttentionScaleMode.F32_PER_TENSOR,
    )

    schema = str(torch.ops.aiter.mha_v4_fwd_launch.default._schema)
    assert "Tensor q" in schema
    assert "Tensor k" in schema
    assert "Tensor v" in schema
    assert "Tensor(a6!) out" in schema
    assert schema.endswith("-> ()")


@pytest.mark.skipif(get_gfx() != "gfx950", reason="gfx950 MHA v4 validation")
@pytest.mark.parametrize(
    ("q_format", "v_format"),
    [
        (AttentionFormat.BF16, AttentionFormat.BF16),
        (AttentionFormat.INT8, AttentionFormat.FP8),
        (AttentionFormat.FP8, AttentionFormat.FP8),
        (AttentionFormat.FP8, AttentionFormat.MXFP6),
        (AttentionFormat.MXFP4, AttentionFormat.FP8),
        (AttentionFormat.MXFP4, AttentionFormat.MXFP4),
        (AttentionFormat.MXFP6_E2M3, AttentionFormat.FP8),
        (AttentionFormat.MXFP6_E2M3, AttentionFormat.MXFP4),
    ],
)
def test_mha_v4_raw_compile_parity(q_format, v_format):
    torch._dynamo.reset()
    torch.manual_seed(31)
    q = torch.randn((1, 512, 5, 128), device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    eager_out = torch.empty_like(q)
    compiled_out = torch.empty_like(q)

    eager = mha_v4(q, k, v, q_format, q_format, v_format, out=eager_out)
    compiled = torch.compile(mha_v4, fullgraph=True)(
        q, k, v, q_format, q_format, v_format, out=compiled_out
    )
    churn = torch.empty((16 * 1024 * 1024,), device="cuda", dtype=torch.uint8)
    consumed = compiled.contiguous()
    torch.cuda.synchronize()

    assert eager.data_ptr() == eager_out.data_ptr()
    assert compiled.data_ptr() == compiled_out.data_ptr()
    assert torch.equal(eager, compiled)
    assert torch.isfinite(consumed).all()
    assert churn.numel() == 16 * 1024 * 1024


@pytest.mark.skipif(get_gfx() != "gfx950", reason="gfx950 MXFP8 validation")
def test_mha_v4_mxfp8_raw_compile_parity():
    torch.manual_seed(41)
    q = torch.randn((1, 257, 5, 128), device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    eager_out = torch.empty_like(q)
    compiled_out = torch.empty_like(q)

    eager = mha_v4_mxfp8(q, k, v, out=eager_out)
    compiled = torch.compile(mha_v4_mxfp8, fullgraph=True)(q, k, v, out=compiled_out)
    torch.cuda.synchronize()

    assert eager.data_ptr() == eager_out.data_ptr()
    assert compiled.data_ptr() == compiled_out.data_ptr()
    assert torch.equal(eager, compiled)
    assert torch.isfinite(compiled).all()


@pytest.mark.skipif(get_gfx() != "gfx950", reason="gfx950 MXFP4 V validation")
@pytest.mark.parametrize("q_format", [AttentionFormat.MXFP4, AttentionFormat.MXFP6])
def test_mha_v4_raw_mxfp4_v_supports_unaligned_sequence(q_format):
    torch.manual_seed(37)
    q = torch.randn((1, 129, 2, 128), device="cuda", dtype=torch.bfloat16)
    k = torch.randn((1, 257, 2, 128), device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)

    eager = mha_v4(q, k, v, q_format, q_format, AttentionFormat.MXFP4)
    compiled = torch.compile(mha_v4, fullgraph=True)(
        q, k, v, q_format, q_format, AttentionFormat.MXFP4
    )
    torch.cuda.synchronize()

    assert torch.equal(eager, compiled)
    assert torch.isfinite(compiled).all()


def _mha_v4_sparse_co_available() -> bool:
    gfx = get_gfx()
    asm_dir = os.environ.get("AITER_ASM_DIR", os.path.join(AITER_ROOT_DIR, "hsa"))
    if gfx == "gfx942":
        return os.path.isfile(
            os.path.join(
                asm_dir, "gfx942", "fmha_v4_fwd", "MI300", "fwd_hd128_fp8_sparse.co"
            )
        )
    return os.path.isfile(
        os.path.join(asm_dir, "gfx950", "fmha_v4_fwd", "fwd_hd128_fp8_sparse.co")
    )


_MHA_V4_SPARSE_ARCH = get_gfx() in ("gfx942", "gfx950")


def test_mha_v4_packed_rejects_partial_lut():
    dummy = torch.empty(0)
    with pytest.raises(ValueError, match="all be set or all omitted"):
        mha_v4_packed(
            dummy,
            dummy,
            dummy,
            dummy,
            dummy,
            dummy,
            AttentionFormat.INT8,
            AttentionFormat.INT8,
            AttentionFormat.FP8,
            AttentionScaleMode.F32_PER_TENSOR,
            AttentionScaleMode.F32_PER_TENSOR,
            AttentionScaleMode.F32_PER_TENSOR,
            kv_block_indices=dummy,
        )


def test_mha_v4_rejects_wrong_block_mask_shape():
    q = torch.zeros((1, 256, 2, 128), dtype=torch.bfloat16)
    mask = torch.ones((1, 2, 1, 1), dtype=torch.bool)
    with pytest.raises(ValueError, match="block_mask must have shape"):
        mha_v4(
            q,
            q,
            q,
            AttentionFormat.FP8,
            AttentionFormat.FP8,
            AttentionFormat.FP8,
            block_mask=mask,
        )


@pytest.mark.skipif(not _MHA_V4_SPARSE_ARCH, reason="gfx942/gfx950 sparse schema")
def test_mha_v4_sparse_schema_mutates_only_out():
    dense = str(torch.ops.aiter.mha_v4_fwd_launch.default._schema)
    assert "Tensor kv_block_indices" not in dense
    assert "Tensor(a6!) out" in dense

    sparse = str(torch.ops.aiter.mha_v4_fwd_sparse_launch.default._schema)
    assert "Tensor kv_block_indices" in sparse
    assert "Tensor lut_start" in sparse
    assert "Tensor lut_count" in sparse
    assert "Tensor(a6!) out" in sparse
    assert sparse.endswith("-> ()")


def _work_table_counts(total, pattern):
    """LUT lengths covering the tie structures the ordering has to get right."""
    if pattern == "uniform":
        return torch.full((total,), 7, device="cuda", dtype=torch.int32)
    if pattern == "zeros":
        return torch.zeros((total,), device="cuda", dtype=torch.int32)
    if pattern == "random":
        return torch.randint(0, 64, (total,), device="cuda", dtype=torch.int32)
    if pattern == "wide_random":
        return torch.randint(0, 8192, (total,), device="cuda", dtype=torch.int32)
    if pattern == "two_values":
        alternating = torch.arange(total, device="cuda") % 3 == 0
        return torch.where(alternating, 9, 4).to(torch.int32)
    if pattern == "descending":
        return torch.arange(total, 0, -1, device="cuda", dtype=torch.int32)
    return torch.arange(1, total + 1, device="cuda", dtype=torch.int32)


def _unpack_work_table(table, nhead, q_tiles):
    q_idx = (table & 0xFFFF).long()
    h_idx = ((table >> 16) & 0xFF).long()
    b_idx = ((table >> 24) & 0xFF).long()
    return (b_idx * nhead + h_idx) * q_tiles + q_idx


# The table has one entry per (batch, head, query tile). 8192 is the point where the builder hands
# the sort to ATen, so straddle it, and include sizes that are not multiples of a wave or workgroup.
@pytest.mark.skipif(not _MHA_V4_SPARSE_ARCH, reason="gfx942/gfx950 sparse validation")
@pytest.mark.parametrize(
    ("batch", "nhead", "q_tiles"),
    [
        (1, 1, 1),
        (1, 8, 3),
        (2, 5, 7),
        (1, 16, 32),
        (1, 32, 32),
        (1, 5, 296),
        (4, 16, 64),
        (8, 16, 64),
        (8, 32, 64),
    ],
)
@pytest.mark.parametrize(
    "pattern",
    [
        "uniform",
        "zeros",
        "random",
        "wide_random",
        "two_values",
        "descending",
        "ascending",
    ],
)
def test_mha_v4_sparse_work_table_is_longest_lut_first(batch, nhead, q_tiles, pattern):
    torch.manual_seed(7)
    total = batch * nhead * q_tiles
    counts = _work_table_counts(total, pattern)

    table = mha_v4_sparse_work_table(counts, batch, nhead, q_tiles)
    visited = _unpack_work_table(table, nhead, q_tiles)

    # Every tile exactly once. This is the part a wrong table would turn into a wrong result.
    assert torch.equal(visited.sort().values, torch.arange(total, device="cuda"))

    # Longest LUT first, so no heavy tile straggles behind the rest.
    ordered = counts[visited]
    assert bool((ordered[:-1] >= ordered[1:]).all())

    # Ties keep raster order, which is what leaves uniform counts spatially coherent. A stable
    # reference sort pins the whole permutation, not just the two properties above.
    expected = torch.argsort(counts, descending=True, stable=True)
    assert torch.equal(visited, expected)


@pytest.mark.skipif(not _MHA_V4_SPARSE_ARCH, reason="gfx942/gfx950 sparse validation")
@pytest.mark.parametrize(
    "batch,nhead,q_tiles", [(1, 16, 32), (1, 5, 296), (8, 16, 64), (8, 32, 64)]
)
def test_mha_v4_sparse_work_table_leaves_uniform_counts_in_raster_order(
    batch, nhead, q_tiles
):
    """Top-k sparsity gives every tile the same LUT length, and that case must not be shuffled."""
    total = batch * nhead * q_tiles
    counts = torch.full((total,), 5, device="cuda", dtype=torch.int32)

    table = mha_v4_sparse_work_table(counts, batch, nhead, q_tiles)

    visited = _unpack_work_table(table, nhead, q_tiles)
    assert torch.equal(visited, torch.arange(total, device="cuda"))


@pytest.mark.skipif(not _MHA_V4_SPARSE_ARCH, reason="gfx942/gfx950 sparse validation")
@pytest.mark.skipif(
    not _mha_v4_sparse_co_available(),
    reason="sorted-sparse MHA v4 code object is not deployed",
)
@pytest.mark.parametrize(
    ("q_format", "v_format"),
    [
        pytest.param(
            native_fp8_format(),
            native_fp8_format(),
            id="fp8",
        ),
        pytest.param(
            AttentionFormat.FP8,
            AttentionFormat.MXFP6,
            marks=pytest.mark.skipif(
                get_gfx() != "gfx950", reason="gfx950 MXFP6 sparse"
            ),
            id="f8f6",
        ),
        pytest.param(
            AttentionFormat.INT8,
            native_fp8_format(),
            id="i8fp8",
        ),
    ],
)
def test_mha_v4_sparse_all_true_mask_matches_dense(q_format, v_format):
    torch.manual_seed(41)
    q = torch.randn((1, 256, 2, 128), device="cuda", dtype=torch.bfloat16)
    k = torch.randn((1, 256, 2, 128), device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    kv_tiles = 256 // mha_v4_kv_tile()
    mask = torch.ones((1, 2, 1, kv_tiles), device="cuda", dtype=torch.bool)
    dense = mha_v4(q, k, v, q_format, q_format, v_format)
    sparse = mha_v4(
        q,
        k,
        v,
        q_format,
        q_format,
        v_format,
        block_mask=mask,
    )
    torch.cuda.synchronize()
    assert torch.equal(dense, sparse)
    assert torch.isfinite(sparse).all()


@pytest.mark.skipif(not _MHA_V4_SPARSE_ARCH, reason="gfx942/gfx950 sparse validation")
@pytest.mark.skipif(
    not _mha_v4_sparse_co_available(),
    reason="sorted-sparse MHA v4 code object is not deployed",
)
def test_mha_v4_sparse_block_mask_compiles_without_graph_breaks():
    """The mask path derives its geometry from host state, which Dynamo cannot trace.

    mha_v4_kv_tile() reads the manifest and get_gfx() shells out to rocminfo, so both sit behind
    torch_compile_guard. Without that the sparse mask path costs graph breaks per trace and fails
    under fullgraph, which no other test in this file would notice.
    """
    torch.manual_seed(41)
    kv_tile = mha_v4_kv_tile()
    q = torch.randn((1, 256, 2, 128), device="cuda", dtype=torch.bfloat16)
    k = torch.randn((1, 4 * kv_tile, 2, 128), device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    mask = torch.ones((1, 2, 1, 4), device="cuda", dtype=torch.bool)
    fp8_format = native_fp8_format()

    def call():
        return mha_v4(q, k, v, fp8_format, fp8_format, fp8_format, block_mask=mask)

    explained = torch._dynamo.explain(call)()
    assert explained.break_reasons == [], [
        str(reason.reason) for reason in explained.break_reasons
    ]

    eager = call()
    compiled = torch.compile(call, fullgraph=True)()
    torch.cuda.synchronize()

    assert torch.equal(eager, compiled)


@pytest.mark.skipif(not _MHA_V4_SPARSE_ARCH, reason="gfx942/gfx950 sparse validation")
@pytest.mark.skipif(
    not _mha_v4_sparse_co_available(),
    reason="sorted-sparse MHA v4 code object is not deployed",
)
@pytest.mark.parametrize(
    ("q_format", "v_format"),
    [
        pytest.param(
            native_fp8_format(),
            native_fp8_format(),
            id="fp8",
        ),
        pytest.param(
            AttentionFormat.MXFP4,
            native_fp8_format(),
            marks=pytest.mark.skipif(
                get_gfx() != "gfx950", reason="gfx950 MXFP4 sparse"
            ),
            id="mxfp4",
        ),
    ],
)
def test_mha_v4_sparse_gqa_all_true_mask_matches_repeated_kv(q_format, v_format):
    torch.manual_seed(41)
    query_heads = 8
    kv_heads = 2
    gqa_ratio = query_heads // kv_heads
    q = torch.randn((1, 256, query_heads, 128), device="cuda", dtype=torch.bfloat16)
    k = torch.randn((1, 256, kv_heads, 128), device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    kv_tiles = 256 // mha_v4_kv_tile()
    mask = torch.ones((1, query_heads, 1, kv_tiles), device="cuda", dtype=torch.bool)
    k_repeated = k.repeat_interleave(gqa_ratio, dim=2)
    v_repeated = v.repeat_interleave(gqa_ratio, dim=2)

    gqa_dense = mha_v4(q, k, v, q_format, q_format, v_format)
    gqa_sparse = mha_v4(q, k, v, q_format, q_format, v_format, block_mask=mask)
    mha_dense = mha_v4(q, k_repeated, v_repeated, q_format, q_format, v_format)
    mha_sparse = mha_v4(
        q,
        k_repeated,
        v_repeated,
        q_format,
        q_format,
        v_format,
        block_mask=mask,
    )
    torch.cuda.synchronize()

    assert torch.equal(gqa_dense, mha_dense)
    assert torch.equal(gqa_sparse, gqa_dense)
    assert torch.equal(gqa_sparse, mha_sparse)
    assert torch.isfinite(gqa_sparse).all()


class _Operand(NamedTuple):
    """A quantized MHA v4 operand and the descale it was produced with."""

    quantized: torch.Tensor
    descale: torch.Tensor


def _sparse_fp8_operands(sequence_k, heads=2, sequence_q=256, batch=1, seed=0):
    """Quantize once so sparse and reference runs share descales exactly.

    Re-quantizing a KV slice would pick a different per-tensor amax, which shifts every
    value and hides whether the kernel read the KV blocks the LUT named.
    """
    torch.manual_seed(seed)
    q = torch.randn(
        (batch, sequence_q, heads, 128), device="cuda", dtype=torch.bfloat16
    )
    k = torch.randn(
        (batch, sequence_k, heads, 128), device="cuda", dtype=torch.bfloat16
    )
    v = torch.randn_like(k)
    return (
        _Operand(*quantize_fp8_rotated(q)),
        _Operand(*quantize_fp8_rotated(k)),
        _Operand(*quantize_fp8(v)),
    )


def _sparse_fp8_launch(q, k, v, block_mask=None):
    lut = {}
    if block_mask is not None:
        indices, start, count = block_attn_mask_to_ragged_lut(
            block_mask,
            num_heads=block_mask.shape[1],
            return_none_if_dense=False,
        )
        lut = {
            "kv_block_indices": indices,
            "lut_start": start,
            "lut_count": count,
        }
    fp8_format = native_fp8_format()
    return mha_v4_packed(
        q.quantized,
        k.quantized,
        v.quantized,
        q.descale,
        k.descale,
        v.descale,
        fp8_format,
        fp8_format,
        fp8_format,
        AttentionScaleMode.F32_PER_TENSOR,
        AttentionScaleMode.F32_PER_TENSOR,
        AttentionScaleMode.F32_PER_TENSOR,
        **lut,
    )


def _gather_kv_tiles(operand, tiles):
    """Concatenate the named KV tiles, leaving the quantized bytes and descale untouched."""
    kv_tile = mha_v4_kv_tile()
    gathered = torch.cat(
        [operand.quantized[:, tile * kv_tile : (tile + 1) * kv_tile] for tile in tiles],
        dim=1,
    )
    return _Operand(gathered.contiguous(), operand.descale)


def _tile_mask(heads, kv_tiles, tiles, q_tiles=1, batch=1):
    mask = torch.zeros(
        (batch, heads, q_tiles, kv_tiles), device="cuda", dtype=torch.bool
    )
    for tile in tiles:
        mask[:, :, :, tile] = True
    return mask


@pytest.mark.skipif(not _MHA_V4_SPARSE_ARCH, reason="gfx942/gfx950 sparse validation")
@pytest.mark.skipif(
    not _mha_v4_sparse_co_available(),
    reason="sorted-sparse MHA v4 code object is not deployed",
)
@pytest.mark.parametrize("tiles", [(0,), (1,), (3,), (0, 2), (1, 2, 3)])
def test_mha_v4_sparse_reads_only_the_kv_tiles_the_lut_names(tiles):
    """A kernel that ignored kv_block_indices would pass every all-True test."""
    heads = 2
    kv_tile = mha_v4_kv_tile()
    kv_tiles = 4
    q, k, v = _sparse_fp8_operands(sequence_k=kv_tiles * kv_tile, heads=heads)

    mask = _tile_mask(heads, kv_tiles, tiles)
    sparse = _sparse_fp8_launch(q, k, v, block_mask=mask)
    # Dense over exactly the selected tiles: same quantized bytes, same descales, so the
    # only difference is which KV blocks take part.
    dense = _sparse_fp8_launch(
        q, _gather_kv_tiles(k, tiles), _gather_kv_tiles(v, tiles)
    )
    torch.cuda.synchronize()

    assert torch.equal(sparse, dense)
    assert torch.isfinite(sparse).all()


@pytest.mark.skipif(not _MHA_V4_SPARSE_ARCH, reason="gfx942/gfx950 sparse validation")
@pytest.mark.skipif(
    not _mha_v4_sparse_co_available(),
    reason="sorted-sparse MHA v4 code object is not deployed",
)
def test_mha_v4_sparse_distinct_kv_tiles_give_distinct_results():
    """Guards the reference itself: selecting different tiles must change the output."""
    heads = 2
    kv_tile = mha_v4_kv_tile()
    kv_tiles = 4
    q, k, v = _sparse_fp8_operands(sequence_k=kv_tiles * kv_tile, heads=heads)

    outputs = [
        _sparse_fp8_launch(q, k, v, block_mask=_tile_mask(heads, kv_tiles, (tile,)))
        for tile in range(kv_tiles)
    ]
    torch.cuda.synchronize()

    for tile in range(1, kv_tiles):
        assert not torch.equal(outputs[0], outputs[tile]), (
            f"kv tile 0 and kv tile {tile} produced identical output, so the kernel is "
            "not reading kv_block_indices"
        )


@pytest.mark.skipif(not _MHA_V4_SPARSE_ARCH, reason="gfx942/gfx950 sparse validation")
@pytest.mark.skipif(
    not _mha_v4_sparse_co_available(),
    reason="sorted-sparse MHA v4 code object is not deployed",
)
def test_mha_v4_sparse_gives_each_head_its_own_kv_tiles():
    """4-D masks may give heads different KV lists; each head must follow its own row."""
    heads = 3
    kv_tile = mha_v4_kv_tile()
    kv_tiles = 4
    per_head = ((0,), (3,), (1, 2))
    q, k, v = _sparse_fp8_operands(sequence_k=kv_tiles * kv_tile, heads=heads)

    mask = torch.zeros((1, heads, 1, kv_tiles), device="cuda", dtype=torch.bool)
    for head, tiles in enumerate(per_head):
        for tile in tiles:
            mask[:, head, :, tile] = True
    sparse = _sparse_fp8_launch(q, k, v, block_mask=mask)
    torch.cuda.synchronize()

    for head, tiles in enumerate(per_head):
        dense = _sparse_fp8_launch(
            q, _gather_kv_tiles(k, tiles), _gather_kv_tiles(v, tiles)
        )
        torch.cuda.synchronize()
        assert torch.equal(
            sparse[:, :, head], dense[:, :, head]
        ), f"head {head} did not attend to tiles {tiles}"


@pytest.mark.skipif(not _MHA_V4_SPARSE_ARCH, reason="gfx942/gfx950 sparse validation")
@pytest.mark.skipif(
    not _mha_v4_sparse_co_available(),
    reason="sorted-sparse MHA v4 code object is not deployed",
)
def test_mha_v4_sparse_follows_the_lut_across_query_tiles():
    """Multiple query tiles exercise the work table on a real launch, not just its ordering."""
    heads = 2
    kv_tile = mha_v4_kv_tile()
    kv_tiles = 4
    q_tiles = 2
    q, k, v = _sparse_fp8_operands(
        sequence_k=kv_tiles * kv_tile, heads=heads, sequence_q=256 * q_tiles
    )

    mask = torch.zeros((1, heads, q_tiles, kv_tiles), device="cuda", dtype=torch.bool)
    mask[:, :, 0, 0] = True
    mask[:, :, 1, 3] = True
    sparse = _sparse_fp8_launch(q, k, v, block_mask=mask)
    torch.cuda.synchronize()

    for q_tile, tiles in ((0, (0,)), (1, (3,))):
        dense = _sparse_fp8_launch(
            q, _gather_kv_tiles(k, tiles), _gather_kv_tiles(v, tiles)
        )
        torch.cuda.synchronize()
        rows = slice(q_tile * 256, (q_tile + 1) * 256)
        assert torch.equal(
            sparse[:, rows], dense[:, rows]
        ), f"query tile {q_tile} did not attend to tiles {tiles}"


@pytest.mark.skipif(not _MHA_V4_SPARSE_ARCH, reason="gfx942/gfx950 sparse validation")
@pytest.mark.skipif(
    not _mha_v4_sparse_co_available(),
    reason="sorted-sparse MHA v4 code object is not deployed",
)
@pytest.mark.parametrize("tail_rows", [64, 128, 200])
def test_mha_v4_sparse_partial_query_tile_follows_the_lut(tail_rows):
    """A trailing partial query tile has to select and zero the same way a full one does.

    Production sequence lengths are not multiples of the 256-row query tile, and the empty-row
    no-op is built on the same tail masking the partial tile uses, so the two belong in one
    case: the short tile must still read the KV blocks its row names, and an all-False row on
    that tile must come back zero rather than reading the masked-off remainder.
    """
    heads = 2
    kv_tile = mha_v4_kv_tile()
    kv_tiles = 4
    q_tiles = 3
    per_tile = ((0, (0,)), (1, (1, 2)), (2, (3,)))
    q, k, v = _sparse_fp8_operands(
        sequence_k=kv_tiles * kv_tile,
        heads=heads,
        sequence_q=256 * (q_tiles - 1) + tail_rows,
    )

    mask = torch.zeros((1, heads, q_tiles, kv_tiles), device="cuda", dtype=torch.bool)
    for q_tile, tiles in per_tile:
        for tile in tiles:
            mask[:, :, q_tile, tile] = True
    mask[:, 1, q_tiles - 1, :] = False  # head 1's partial-tile row selects nothing
    sparse = _sparse_fp8_launch(q, k, v, block_mask=mask)
    torch.cuda.synchronize()

    for q_tile, tiles in per_tile:
        dense = _sparse_fp8_launch(
            q, _gather_kv_tiles(k, tiles), _gather_kv_tiles(v, tiles)
        )
        torch.cuda.synchronize()
        rows = slice(q_tile * 256, min((q_tile + 1) * 256, sparse.shape[1]))
        live_heads = 1 if q_tile == q_tiles - 1 else heads
        assert torch.equal(
            sparse[:, rows, :live_heads], dense[:, rows, :live_heads]
        ), f"query tile {q_tile} did not attend to tiles {tiles}"

    tail = slice((q_tiles - 1) * 256, sparse.shape[1])
    assert torch.equal(
        sparse[:, tail, 1], torch.zeros_like(sparse[:, tail, 1])
    ), "empty row on the partial query tile is not zero"
    # Without this the case would also pass on a kernel that skipped the short tile entirely,
    # since both sides of the comparison above would then be zero.
    assert (
        sparse[:, tail, 0].abs().max() > 0
    ), "live partial-tile row came back degenerate"
    assert torch.isfinite(sparse).all(), "partial query tile leaked NaN or infinity"


def _gfx950_only(launch, label):
    return pytest.param(
        launch,
        id=label,
        marks=pytest.mark.skipif(get_gfx() != "gfx950", reason="gfx950 MX sparse"),
    )


# An FP8 Q always canonicalizes to per-tensor scales, so MXFP8 is unreachable through raw mha_v4
# and goes through its own entry point instead.
_EMPTY_ROW_LAUNCHES = [
    pytest.param(
        lambda q, k, v, m: mha_v4(
            q,
            k,
            v,
            native_fp8_format(),
            native_fp8_format(),
            native_fp8_format(),
            block_mask=m,
        ),
        id="fp8",
    ),
    pytest.param(
        lambda q, k, v, m: mha_v4(
            q,
            k,
            v,
            AttentionFormat.INT8,
            AttentionFormat.INT8,
            native_fp8_format(),
            block_mask=m,
        ),
        id="i8fp8",
    ),
    _gfx950_only(lambda q, k, v, m: mha_v4_mxfp8(q, k, v, block_mask=m), "mxfp8"),
    _gfx950_only(
        lambda q, k, v, m: mha_v4(
            q,
            k,
            v,
            native_fp8_format(),
            native_fp8_format(),
            AttentionFormat.MXFP6,
            block_mask=m,
        ),
        "f8f6",
    ),
    _gfx950_only(
        lambda q, k, v, m: mha_v4(
            q,
            k,
            v,
            AttentionFormat.MXFP6,
            AttentionFormat.MXFP6,
            native_fp8_format(),
            block_mask=m,
        ),
        "mxfp6",
    ),
    _gfx950_only(
        lambda q, k, v, m: mha_v4(
            q,
            k,
            v,
            AttentionFormat.MXFP6,
            AttentionFormat.MXFP6,
            AttentionFormat.MXFP4,
            block_mask=m,
        ),
        "f6f4",
    ),
    _gfx950_only(
        lambda q, k, v, m: mha_v4(
            q,
            k,
            v,
            AttentionFormat.MXFP4,
            AttentionFormat.MXFP4,
            native_fp8_format(),
            block_mask=m,
        ),
        "mxfp4",
    ),
    _gfx950_only(
        lambda q, k, v, m: mha_v4(
            q,
            k,
            v,
            AttentionFormat.MXFP4,
            AttentionFormat.MXFP4,
            AttentionFormat.MXFP4,
            block_mask=m,
        ),
        "f4f4",
    ),
]


@pytest.mark.skipif(not _MHA_V4_SPARSE_ARCH, reason="gfx942/gfx950 sparse validation")
@pytest.mark.skipif(
    not _mha_v4_sparse_co_available(),
    reason="sorted-sparse MHA v4 code object is not deployed",
)
@pytest.mark.parametrize("launch", _EMPTY_ROW_LAUNCHES)
def test_mha_v4_sparse_empty_row_writes_zeros(launch):
    """An all-False row selects no KV block, so its output tile must be zero, not garbage.

    Its lut_start also sits one past the last kv_block_indices entry, which is what used to walk
    the prologue's unguarded reads into a multi-gigabyte scalar offset and fault the kernel.
    """
    heads = 2
    kv_tiles = 4
    selected = (0, 1)
    torch.manual_seed(0)
    q = torch.randn((1, 256, heads, 128), device="cuda", dtype=torch.bfloat16)
    k = torch.randn(
        (1, kv_tiles * mha_v4_kv_tile(), heads, 128),
        device="cuda",
        dtype=torch.bfloat16,
    )
    v = torch.randn_like(k)

    mask = torch.zeros((1, heads, 1, kv_tiles), device="cuda", dtype=torch.bool)
    for tile in selected:
        mask[:, 0, :, tile] = True  # head 0 selects two tiles; head 1 stays all-False
    out = launch(q, k, v, mask)
    torch.cuda.synchronize()

    assert torch.equal(
        out[:, :, 1], torch.zeros_like(out[:, :, 1])
    ), "empty row is not zero"
    assert torch.isfinite(out).all(), "empty row leaked NaN or infinity"
    assert out[:, :, 0].abs().max() > 0, "live row came back degenerate"

    # The empty row must not perturb the row that does select tiles: give head 1 a tile and head 0's
    # output has to stay bit-identical, since each workgroup owns one (batch, head, query tile).
    mask[:, 1, :, 0] = True
    populated = launch(q, k, v, mask)
    torch.cuda.synchronize()
    assert torch.equal(
        out[:, :, 0], populated[:, :, 0]
    ), "empty row disturbed the live row"


def test_mha_v4_rejects_non_bool_block_mask():
    """Counts come from a sum but the fill uses truthiness, so non-bool masks disagree."""
    q = torch.zeros((1, 256, 2, 128), dtype=torch.bfloat16)
    mask = torch.ones((1, 2, 1, 256 // mha_v4_kv_tile()), dtype=torch.int32)
    with pytest.raises(ValueError, match="block_mask must be a bool tensor"):
        mha_v4(
            q,
            q,
            q,
            AttentionFormat.FP8,
            AttentionFormat.FP8,
            AttentionFormat.FP8,
            block_mask=mask,
        )


@pytest.mark.skipif(
    torch.cuda.device_count() < 2, reason="needs two GPUs to mismatch devices"
)
def test_mha_v4_rejects_block_mask_on_another_device():
    q = torch.zeros((1, 256, 2, 128), dtype=torch.bfloat16, device="cuda:0")
    mask = torch.ones(
        (1, 2, 1, 256 // mha_v4_kv_tile()), dtype=torch.bool, device="cuda:1"
    )
    with pytest.raises(ValueError, match="block_mask must be on the same device"):
        mha_v4(
            q,
            q,
            q,
            AttentionFormat.FP8,
            AttentionFormat.FP8,
            AttentionFormat.FP8,
            block_mask=mask,
        )


@pytest.mark.skipif(not _MHA_V4_SPARSE_ARCH, reason="gfx942/gfx950 sparse validation")
@pytest.mark.skipif(
    not _mha_v4_sparse_co_available(),
    reason="sorted-sparse MHA v4 code object is not deployed",
)
def test_mha_v4_sparse_rejects_empty_kv_block_indices():
    """Rows may be empty, but the ASM still dereferences the row base, so the buffer cannot be."""
    heads = 2
    kv_tile = mha_v4_kv_tile()
    kv_tiles = 4
    q, k, v = _sparse_fp8_operands(sequence_k=kv_tiles * kv_tile, heads=heads)
    rows = heads  # batch 1, one query tile
    device = q.quantized.device
    fp8_format = native_fp8_format()
    with pytest.raises(RuntimeError, match="must be non-empty"):
        mha_v4_packed(
            q.quantized,
            k.quantized,
            v.quantized,
            q.descale,
            k.descale,
            v.descale,
            fp8_format,
            fp8_format,
            fp8_format,
            AttentionScaleMode.F32_PER_TENSOR,
            AttentionScaleMode.F32_PER_TENSOR,
            AttentionScaleMode.F32_PER_TENSOR,
            kv_block_indices=torch.zeros(0, dtype=torch.int32, device=device),
            lut_start=torch.zeros(rows, dtype=torch.int32, device=device),
            lut_count=torch.ones(rows, dtype=torch.int32, device=device),
        )


@pytest.mark.skipif(not _MHA_V4_SPARSE_ARCH, reason="gfx942/gfx950 sparse validation")
@pytest.mark.skipif(
    not _mha_v4_sparse_co_available(),
    reason="sorted-sparse MHA v4 code object is not deployed",
)
@pytest.mark.skipif(
    os.environ.get("AITER_MHA_V4_VALIDATE_LUT", "0") in ("0", ""),
    reason="opt-in LUT validation is disabled",
)
@pytest.mark.parametrize(
    "mutate,message",
    [
        pytest.param(
            lambda indices, start, count: indices.fill_(9999),
            "outside",
            id="index_out_of_range",
        ),
        pytest.param(
            lambda indices, start, count: start.fill_(-1),
            "negative",
            id="negative_start",
        ),
    ],
)
def test_mha_v4_sparse_validation_rejects_malformed_lut(mutate, message):
    """Only reachable with AITER_MHA_V4_VALIDATE_LUT=1; otherwise these fault in the ASM."""
    heads = 2
    kv_tile = mha_v4_kv_tile()
    kv_tiles = 4
    q, k, v = _sparse_fp8_operands(sequence_k=kv_tiles * kv_tile, heads=heads)
    mask = _tile_mask(heads, kv_tiles, (0, 1))
    indices, start, count = block_attn_mask_to_ragged_lut(
        mask, num_heads=heads, return_none_if_dense=False
    )
    mutate(indices, start, count)
    fp8_format = native_fp8_format()
    with pytest.raises(RuntimeError, match=message):
        mha_v4_packed(
            q.quantized,
            k.quantized,
            v.quantized,
            q.descale,
            k.descale,
            v.descale,
            fp8_format,
            fp8_format,
            fp8_format,
            AttentionScaleMode.F32_PER_TENSOR,
            AttentionScaleMode.F32_PER_TENSOR,
            AttentionScaleMode.F32_PER_TENSOR,
            kv_block_indices=indices,
            lut_start=start,
            lut_count=count,
        )
