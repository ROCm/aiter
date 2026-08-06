# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import numpy as np
import pytest
import torch

import aiter
from aiter.ops.triton.moe.quant_moe import downcast_to_mxfp
from aiter.ops.triton.quant import (
    quantize_f4f4_solo_k,
    quantize_f4f4_solo_v,
    sage_quant_f4f4_solo,
)
from aiter.ops.triton.quant.sage_attention_quant_wrappers import (
    create_hadamard_matrix,
    rotation_smooth_qk,
)
from aiter.ops.triton.utils._triton import arch_info

TILE = 128
TILE_BYTES = 8192
SCALE_TILE_BYTES = 512
E2M1 = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], np.float32)
E2M1_MIDPOINTS = np.array([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], np.float32)


def _require_fp4():
    if not torch.cuda.is_available() or not arch_info.is_fp4_avail():
        pytest.skip("MXFP4 GPU support is required")


def _mx_quantize_lastdim(x):
    """Independent NumPy reference for downcast_to_mxfp ROUND_UP arithmetic."""
    x = np.asarray(x, np.float32)
    assert x.shape[-1] % 32 == 0
    blocks = x.reshape(*x.shape[:-1], x.shape[-1] // 32, 32)
    amax = np.max(np.abs(blocks), axis=-1)
    ratio = (amax / np.float32(6.0)).astype(np.float32)
    ratio_bits = ratio.view(np.uint32)
    scale_bits = (ratio_bits.astype(np.uint64) + np.uint64(0x007FFFFF)).astype(
        np.uint32
    ) & np.uint32(0x7F800000)
    scales = scale_bits.view(np.float32)
    inv_scale = np.zeros_like(scales)
    np.divide(
        np.float32(1.0),
        scales,
        out=inv_scale,
        where=scales != 0,
    )
    normalized = (blocks * inv_scale[..., None]).astype(np.float32)
    magnitude_code = np.searchsorted(
        E2M1_MIDPOINTS, np.abs(normalized), side="right"
    ).astype(np.uint8)
    codes = magnitude_code | (np.signbit(normalized).astype(np.uint8) << 3)
    e8m0 = (scale_bits >> np.uint32(23)).astype(np.uint8)
    return codes.reshape(x.shape), e8m0


def _f4f4_v_quantize_lastdim(x):
    """Independent NumPy reference for the latest f4f4 V MX arithmetic."""
    x = np.asarray(x, np.float32)
    assert x.shape[-1] % 32 == 0
    blocks = x.reshape(*x.shape[:-1], x.shape[-1] // 32, 32)
    amax = np.max(np.abs(blocks), axis=-1)
    exponent = np.ceil(
        np.log2(np.maximum(amax, np.float32(1.0e-12)) / np.float32(6.0))
    ).astype(np.int32)
    normalized = (blocks / np.exp2(exponent).astype(np.float32)[..., None]).astype(
        np.float32
    )
    magnitude_code = np.searchsorted(
        E2M1_MIDPOINTS, np.abs(normalized), side="left"
    ).astype(np.uint8)
    codes = magnitude_code | ((normalized < 0).astype(np.uint8) << 3)
    e8m0 = np.clip(exponent + 127, 0, 255).astype(np.uint8)
    return codes.reshape(x.shape), e8m0


def _k_c0_byte(token, channel):
    h_sel = token // 64
    token64 = token % 64
    n = token64 // 32
    token32 = token64 % 32
    half = channel // 64
    parity = (channel % 64) // 32
    channel32 = channel % 32
    lane = parity * 32 + token32
    return (
        h_sel * 4096
        + (n * 2 + half) * 1024
        + lane * 16
        + (channel32 // 8) * 4
        + (channel32 % 8) // 2
    )


def _reference_k(k):
    b, sk, h, _ = k.shape
    n_tiles = (sk + TILE - 1) // TILE
    codes, scales = _mx_quantize_lastdim(k)
    channel = np.arange(128)
    local = channel & 31
    permutation = (channel & ~31) | ((local >> 1) | ((local & 1) << 4))
    permuted = codes[..., permutation]
    sk_pad = n_tiles * TILE
    if sk_pad != sk:
        permuted = np.concatenate(
            [permuted, np.repeat(permuted[:, -1:], sk_pad - sk, axis=1)],
            axis=1,
        )
    packed = permuted[..., 0::2] | (permuted[..., 1::2] << 4)
    packed = packed.reshape(b, n_tiles, TILE, h, 64).transpose(0, 3, 1, 2, 4)
    token = np.arange(TILE)[:, None]
    even_channel = np.arange(0, 128, 2)[None, :]
    offsets = np.broadcast_to(_k_c0_byte(token, even_channel), (TILE, 64))
    out = np.zeros((b, h, n_tiles, TILE_BYTES), np.uint8)
    out[..., offsets] = packed
    return out, scales, codes


def _v_gather_index():
    nibble_offset = np.arange(16384)
    byte_offset = nibble_offset // 2
    nibble = nibble_offset % 2
    u = byte_offset // 1024
    lane = (byte_offset % 1024) // 16
    byte_in_lane = byte_offset % 16
    elem = (byte_in_lane // 4) * 8 + (byte_in_lane % 4) * 2 + nibble
    token = (u & 1) * 64 + (elem // 4) * 8 + (lane // 32) * 4 + elem % 4
    bit2 = (token >> 2) & 1
    bit5 = (token >> 5) & 1
    token = (token & 0b1011011) | (bit5 << 2) | (bit2 << 5)
    channel = (u // 2) * 32 + lane % 32
    return token * 128 + channel


V_GATHER_INDEX = _v_gather_index()


def _reference_v(v):
    b, sk, h, _ = v.shape
    n_tiles = (sk + TILE - 1) // TILE
    out = np.empty((b, h, n_tiles, TILE_BYTES), np.uint8)
    scale_image = np.empty((b, h, n_tiles, SCALE_TILE_BYTES), np.uint8)
    valid_codes = np.empty_like(v, dtype=np.uint8)
    valid_scales = np.empty_like(v, dtype=np.uint8)

    for batch in range(b):
        for head in range(h):
            for tile in range(n_tiles):
                lo = tile * TILE
                hi = min(lo + TILE, sk)
                values = np.empty((TILE, 128), np.float32)
                values[: hi - lo] = v[batch, lo:hi, head]
                values[hi - lo :] = v[batch, sk - 1, head]
                codes_dmajor, scales = _f4f4_v_quantize_lastdim(values.T)
                codes = codes_dmajor.T
                valid_codes[batch, lo:hi, head] = codes[: hi - lo]
                token_blocks = np.arange(hi - lo) // 32
                valid_scales[batch, lo:hi, head] = scales[:, token_blocks].T
                nibbles = codes.reshape(-1)[V_GATHER_INDEX]
                out[batch, head, tile] = nibbles[0::2] | (nibbles[1::2] << 4)
                image = scale_image[batch, head, tile]
                for kv_block in range(4):
                    for channel in range(128):
                        n = channel // 32
                        c = channel % 32
                        offset = (
                            (kv_block // 2) * 256 + (kv_block % 2) * 128 + c * 4 + n
                        )
                        image[offset] = scales[channel, kv_block]
    return (
        out,
        scale_image.reshape(b, h, n_tiles * SCALE_TILE_BYTES),
        valid_codes,
        valid_scales,
    )


def _tile_storage(view):
    b, sk, h, _ = view.shape
    n_tiles = (sk + TILE - 1) // TILE
    return torch.as_strided(
        view,
        (b, h, n_tiles, TILE_BYTES),
        (view.stride(0), view.stride(2), TILE_BYTES, 1),
    )


def _dequantize(codes, scales):
    scale = np.exp2(scales.astype(np.int16) - 127).astype(np.float32)
    magnitude = E2M1[codes & 7]
    signed = np.where((codes & 8) != 0, -magnitude, magnitude)
    shape = codes.shape
    return (
        signed.reshape(*shape[:-1], shape[-1] // 32, 32) * scale[..., None]
    ).reshape(shape)


def _make_signed_input(shape, seed):
    torch.manual_seed(seed)
    x = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    signs = torch.ones(128, device="cuda", dtype=torch.bfloat16)
    signs[::2] = -1
    x = x.abs() * signs
    return x


@pytest.mark.parametrize("b,sk,h", [(1, 129, 1), (1, 257, 2), (2, 129, 2)])
def test_f4f4_solo_k_matches_numpy(b, sk, h):
    _require_fp4()
    k = _make_signed_input((b, sk, h, 128), 1000 + sk + h)
    k[:, :32, 0, :32] = 0
    k_np = k.float().cpu().numpy()

    actual, actual_scale = quantize_f4f4_solo_k(k)
    expected, expected_scale, expected_codes = _reference_k(k_np)
    actual_bytes = _tile_storage(actual).cpu().numpy()

    np.testing.assert_array_equal(actual_bytes, expected)
    np.testing.assert_array_equal(actual_scale.cpu().numpy(), expected_scale)
    n_tiles = (sk + TILE - 1) // TILE
    assert actual.shape == (b, sk, h, 64)
    assert actual.dtype == torch.uint8
    assert actual.stride() == (h * n_tiles * TILE_BYTES, 64, n_tiles * TILE_BYTES, 1)
    assert actual_scale.shape == (b, sk, h, 4)
    assert actual_scale.dtype == torch.uint8
    assert actual.untyped_storage().nbytes() == b * h * n_tiles * TILE_BYTES
    assert actual_scale.untyped_storage().nbytes() >= actual_scale.numel() + 64

    dequantized = _dequantize(expected_codes, expected_scale)
    assert np.all(expected_scale[:, :32, 0, 0] == 0)
    assert np.all(dequantized[:, :32, 0, :32] == 0)
    nonzero = dequantized != 0
    assert np.array_equal(np.signbit(dequantized[nonzero]), np.signbit(k_np[nonzero]))


@pytest.mark.parametrize("b,sk,h", [(1, 129, 1), (1, 257, 2), (2, 129, 2)])
def test_f4f4_solo_v_matches_numpy(b, sk, h):
    _require_fp4()
    v = _make_signed_input((b, sk, h, 128), 2000 + sk + h)
    v[:, :32, 0, :4] = 0
    v_np = v.float().cpu().numpy()

    actual, actual_scale = quantize_f4f4_solo_v(v)
    expected, expected_scale, expected_codes, value_scales = _reference_v(v_np)
    actual_bytes = _tile_storage(actual).cpu().numpy()

    np.testing.assert_array_equal(actual_bytes, expected)
    np.testing.assert_array_equal(actual_scale.cpu().numpy(), expected_scale)
    n_tiles = (sk + TILE - 1) // TILE
    data_bytes = b * h * n_tiles * TILE_BYTES
    assert actual.shape == (b, sk, h, 128)
    assert actual.dtype == torch.uint8
    assert actual.stride() == (h * n_tiles * TILE_BYTES, 64, n_tiles * TILE_BYTES, 1)
    assert actual_scale.shape == (b, h, n_tiles * SCALE_TILE_BYTES)
    assert actual_scale.dtype == torch.uint8
    assert actual.untyped_storage().nbytes() >= data_bytes + TILE_BYTES
    assert (
        actual_scale.untyped_storage().nbytes()
        >= actual_scale.numel() + SCALE_TILE_BYTES
    )
    raw_data = torch.as_strided(actual, (data_bytes + TILE_BYTES,), (1,))
    raw_scale = torch.as_strided(
        actual_scale, (actual_scale.numel() + SCALE_TILE_BYTES,), (1,)
    )
    assert torch.count_nonzero(raw_data[data_bytes:]).item() == 0
    assert torch.count_nonzero(raw_scale[actual_scale.numel() :]).item() == 0

    magnitude = E2M1[expected_codes & 7]
    signed = np.where((expected_codes & 8) != 0, -magnitude, magnitude)
    dequantized = signed * np.exp2(value_scales.astype(np.int16) - 127).astype(
        np.float32
    )
    assert np.all(value_scales[:, :32, 0, :4] == 85)
    assert np.all(dequantized[:, :32, 0, :4] == 0)
    nonzero = dequantized != 0
    assert np.array_equal(np.signbit(dequantized[nonzero]), np.signbit(v_np[nonzero]))


@pytest.mark.parametrize("q_smoothing", [False, True])
def test_sage_quant_f4f4_solo_reuses_q_path_and_supports_gqa(q_smoothing):
    _require_fp4()
    b, sq, sk, hq, h_kv = 1, 65, 129, 4, 2
    q = _make_signed_input((b, sq, hq, 128), 3001)
    k = _make_signed_input((b, sk, h_kv, 128), 3002)
    v = _make_signed_input((b, sk, h_kv, 128), 3003)
    rotation = create_hadamard_matrix(32, device="cuda", dtype=torch.bfloat16) / (
        32**0.5
    )
    kwargs = {
        "FP8_TYPE": aiter.dtypes.fp8,
        "FP8_MAX": torch.finfo(aiter.dtypes.fp8).max,
        "BLKQ": 128,
        "BLKK": 64,
        "q_smoothing": q_smoothing,
        "layout": "bshd",
        "R": rotation,
        "BLOCK_R": 32,
    }

    q_out, q_scale, k_out, k_scale, v_out, v_scale, delta = sage_quant_f4f4_solo(
        q, k, v, **kwargs
    )
    q_rot, _, expected_delta = rotation_smooth_qk(
        q,
        k,
        128,
        R=rotation,
        BLOCK_R=32,
        q_smoothing=q_smoothing,
        layout="bshd",
        sm_scale=(128**-0.5 * 1.4426950408889634),
    )
    expected_q, expected_q_scale = downcast_to_mxfp(q_rot, torch.uint8, axis=-1)

    assert torch.equal(q_out, expected_q)
    assert torch.equal(q_scale, expected_q_scale)
    assert (q_out.shape, q_scale.shape) == (
        (b, sq, hq, 64),
        (b, sq, hq, 4),
    )
    assert k_out.shape == (b, sk, h_kv, 64)
    assert k_scale.shape == (b, sk, h_kv, 4)
    assert v_out.shape == (b, sk, h_kv, 128)
    assert v_scale.shape == (b, h_kv, 2 * SCALE_TILE_BYTES)
    if q_smoothing:
        torch.testing.assert_close(delta, expected_delta)
    else:
        assert delta is None and expected_delta is None
