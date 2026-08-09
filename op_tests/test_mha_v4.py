# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch

from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.mha_v4 import (
    AttentionFormat,
    AttentionScaleMode,
    _quantize_mxfp4,
    _quantize_v_mxfp4_raw,
    mha_v4,
    mha_v4_packed,
    mxfp4_k_view,
    quantize_mxfp4_k,
)
from aiter.ops.triton.quant.mxfp6_fmha_pack import fp6_k_raw_buffer_sizes
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


def _reference_mxfp4_v(value):
    batch, sequence, heads, _ = value.shape
    padded_sequence = fp4_v_padded_sequence(sequence)
    tiles = padded_sequence // 128
    padded = torch.nn.functional.pad(value.float(), (0, 0, 0, 0, 0, padded_sequence - sequence))
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
    scale = torch.empty((batch, heads, tiles * 512), dtype=torch.uint8, device=value.device)
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
                    exponent = torch.ceil(torch.log2(torch.clamp_min(amax, 1e-12) / 6.0))
                    exponents.append(exponent)
                    normalized[:, :, columns] = block[:, :, columns] / torch.exp2(exponent[:, :, None])

                code = _e2m1_code_ties_low(normalized)
                packed = code[..., 0::2] | (code[..., 1::2] << 4)
                payload[:, :, tile * 8192 + unit * 1024 : tile * 8192 + (unit + 1) * 1024] = packed.flatten(2)

                scale_base = tile * 512 + token_half * 256
                for token_block, exponent in enumerate(exponents):
                    encoded = (exponent + 127).clamp(0, 255).to(torch.uint8)
                    for pair in range(16):
                        offset = scale_base + token_block * 128 + 8 * pair + channel_block
                        scale[:, :, offset] = encoded[:, :, 2 * pair]
                        scale[:, :, offset + 4] = encoded[:, :, 2 * pair + 1]
    return raw, scale


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
    raw, scale = _quantize_v_mxfp4_raw(value)
    raw_again, scale_again = _quantize_v_mxfp4_raw(value)
    expected_raw, expected_scale = _reference_mxfp4_v(value)

    assert raw.shape == (fp4_v_raw_buffer_size(2, sequence, 3),)
    assert scale.shape == (2, 3, ((sequence + 127) // 128) * 512)
    assert raw.dtype == scale.dtype == torch.uint8
    assert torch.equal(raw, expected_raw)
    assert torch.equal(scale, expected_scale)
    assert torch.equal(raw, raw_again)
    assert torch.equal(scale, scale_again)
    assert torch.count_nonzero(raw[-64:]) == 0


@pytest.mark.skipif(get_gfx() != "gfx950", reason="gfx950 MXFP4 K validation")
@pytest.mark.parametrize("sequence", [1, 127, 128, 129, 257])
def test_mha_v4_mxfp4_k_coalesced_layout(sequence):
    torch.manual_seed(sequence)
    value = torch.randn(
        (2, sequence, 3, 128), device="cuda", dtype=torch.bfloat16
    )
    dense, dense_scale = _quantize_mxfp4(value, 1.0)
    raw, scale = quantize_mxfp4_k(value)
    coalesced = mxfp4_k_view(raw, scale)

    tiles = (sequence + 127) // 128
    token = torch.arange(sequence, device="cuda")
    chunk = torch.arange(4, device="cuda")
    byte = torch.arange(16, device="cuda")
    raw_offset = (
        torch.arange(2, device="cuda")[:, None, None, None, None]
        * (3 * tiles * 8192)
        + torch.arange(3, device="cuda")[None, None, :, None, None]
        * (tiles * 8192)
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


@pytest.mark.skipif(get_gfx() != "gfx950", reason="gfx950 six-format validation")
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


@pytest.mark.skipif(get_gfx() != "gfx950", reason="gfx950 six-format validation")
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


@pytest.mark.skipif(get_gfx() != "gfx950", reason="gfx950 six-format validation")
@pytest.mark.parametrize("q_format", [AttentionFormat.INT8, AttentionFormat.FP8])
def test_mha_v4_zero_inputs_are_finite(q_format):
    q = torch.zeros((1, 128, 2, 128), device="cuda", dtype=torch.bfloat16)
    out = mha_v4(q, q, q, q_format, q_format, AttentionFormat.FP8)
    torch.cuda.synchronize()
    assert torch.count_nonzero(out) == 0
    assert torch.isfinite(out).all()


@pytest.mark.skipif(get_gfx() != "gfx950", reason="gfx950 six-format validation")
def test_mha_v4_packed_i8fp8_compile_parity():
    torch.manual_seed(17)
    q = torch.randint(-32, 33, (1, 512, 5, 128), device="cuda", dtype=torch.int8)
    k = torch.randint(-32, 33, (1, 512, 5, 128), device="cuda", dtype=torch.int8)
    v = torch.randn((1, 512, 5, 128), device="cuda").to(torch.float8_e4m3fn)
    q_descale = torch.tensor([0.02], device="cuda")
    k_descale = torch.tensor([0.03], device="cuda")
    v_descale = torch.tensor([0.04], device="cuda")
    scale = 128**-0.5

    eager = mha_v4_packed(
        q,
        k,
        v,
        q_descale,
        k_descale,
        v_descale,
        AttentionFormat.INT8,
        AttentionFormat.INT8,
        AttentionFormat.FP8,
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
        AttentionFormat.FP8,
        AttentionScaleMode.F32_PER_TENSOR,
        AttentionScaleMode.F32_PER_TENSOR,
        AttentionScaleMode.F32_PER_TENSOR,
        softmax_scale=scale,
    )
    torch.cuda.synchronize()
    assert torch.equal(eager, compiled)


@pytest.mark.skipif(get_gfx() != "gfx950", reason="gfx950 six-format validation")
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


@pytest.mark.skipif(get_gfx() != "gfx950", reason="gfx950 six-format validation")
@pytest.mark.parametrize(
    ("q_format", "v_format"),
    [
        (AttentionFormat.INT8, AttentionFormat.FP8),
        (AttentionFormat.FP8, AttentionFormat.FP8),
        (AttentionFormat.MXFP4, AttentionFormat.FP8),
        (AttentionFormat.MXFP4, AttentionFormat.MXFP4),
        (AttentionFormat.MXFP6_E2M3, AttentionFormat.FP8),
        (AttentionFormat.MXFP6_E2M3, AttentionFormat.MXFP4),
    ],
)
def test_mha_v4_raw_compile_parity(q_format, v_format):
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
