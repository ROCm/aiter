# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import triton
import triton.language as tl

from aiter.ops.triton._triton_kernels.quant.sage_attention_quant import (
    _e2m1_code,
)


@triton.jit
def pack_f4f4_solo_k_lds_kernel(
    k_ptr,  # canonical packed MXFP4 K [b, sk, h_kv, 64]
    out_ptr,  # uint8 [b, h_kv, nT, 8192]
    stride_kb,
    stride_ks,
    stride_kh,
    stride_kd,
    h_kv,
    nT,
    S,
):
    """Repack canonical MXFP4 K into the solo C0-only LDS-order tile image."""
    pid = tl.program_id(0).to(tl.int64)
    row_chunk = pid % 4
    t = (pid // 4) % nT
    bh = pid // (4 * nT)
    bb = bh // h_kv
    hh = bh % h_kv

    token_in_tile = row_chunk * 32 + tl.arange(0, 32)
    source_token = tl.minimum(t * 128 + token_in_tile, S - 1)
    out_byte = tl.arange(0, 64)

    # The solo K srcA path observes a five-bit rotate within each 32-channel
    # block. Its inverse turns each output byte into channels (j, 16+j).
    d_block = out_byte // 16
    pair = out_byte % 16
    channel_lo = d_block * 32 + pair
    channel_hi = channel_lo + 16
    source_byte_lo = channel_lo // 2
    source_byte_hi = channel_hi // 2
    source_shift_lo = (channel_lo & 1) * 4
    source_shift_hi = (channel_hi & 1) * 4

    source_base = k_ptr + bb * stride_kb + hh * stride_kh
    packed_lo = tl.load(
        source_base
        + source_token[:, None] * stride_ks
        + source_byte_lo[None, :] * stride_kd
    ).to(tl.int32)
    packed_hi = tl.load(
        source_base
        + source_token[:, None] * stride_ks
        + source_byte_hi[None, :] * stride_kd
    ).to(tl.int32)
    code_lo = (packed_lo >> source_shift_lo[None, :]) & 0xF
    code_hi = (packed_hi >> source_shift_hi[None, :]) & 0xF
    packed = (code_lo | (code_hi << 4)).to(tl.uint8)

    # _fp4_k_c0_byte(token_in_tile, 2*out_byte), expressed elementwise.
    channel_even = 2 * out_byte
    h_sel = token_in_tile // 64
    token64 = token_in_tile % 64
    n = token64 // 32
    token32 = token64 % 32
    half = channel_even // 64
    parity = (channel_even % 64) // 32
    channel32 = channel_even % 32
    lane = parity[None, :] * 32 + token32[:, None]
    tile_offset = (
        h_sel[:, None] * 4096
        + (n[:, None] * 2 + half[None, :]) * 1024
        + lane * 16
        + (channel32[None, :] // 8) * 4
        + (channel32[None, :] % 8) // 2
    )
    out_base = out_ptr + (bb * h_kv + hh) * nT * 8192 + t * 8192
    tl.store(out_base + tile_offset, packed)


@triton.jit
def quantize_f4f4_solo_v_lds_kernel(
    v_ptr,  # V [b, sk, h_kv, 128], arbitrary strides
    out_ptr,  # uint8 [b, h_kv, nT, 8192]
    scale_ptr,  # uint8 [b, h_kv, nT, 512]
    byte_map_ptr,  # int32 [4, 16]: output-byte selector per 32-token MX block
    stride_vb,
    stride_vs,
    stride_vh,
    stride_vd,
    h_kv,
    nT,
    S,
):
    """Fused MXFP4 quantization and solo pre-transposed LDS-order V packing."""
    pid = tl.program_id(0).to(tl.int64)
    kv_block = pid % 4
    d_block = (pid // 4) % 4
    t = (pid // 16) % nT
    bh = pid // (16 * nT)
    bb = bh // h_kv
    hh = bh % h_kv

    # Each selected output byte contributes its low/high nibble to this one
    # (output-channel, 32-token) MX block.
    field = tl.arange(0, 32)
    byte_slot = field // 2
    nibble = field & 1
    selector = tl.load(byte_map_ptr + kv_block * 16 + byte_slot)
    k_half = selector // 32
    lane_half = (selector // 16) % 2
    byte_in_lane = selector % 16
    elem = (byte_in_lane // 4) * 8 + (byte_in_lane % 4) * 2 + nibble
    token_in_tile = k_half * 64 + (elem // 4) * 8 + lane_half * 4 + (elem % 4)
    # Undo the srcA token permutation observed by the solo FP4 V path.
    bit2 = (token_in_tile >> 2) & 1
    bit5 = (token_in_tile >> 5) & 1
    token_in_tile = (token_in_tile & 91) | (bit5 << 2) | (bit2 << 5)
    source_token = tl.minimum(t * 128 + token_in_tile, S - 1)

    channel = tl.arange(0, 32)
    source_channel = d_block * 32 + channel
    offsets = (
        bb * stride_vb
        + source_token[None, :] * stride_vs
        + hh * stride_vh
        + source_channel[:, None] * stride_vd
    )
    values = tl.load(v_ptr + offsets).to(tl.float32)
    amax = tl.max(tl.abs(values), axis=1)
    exponent = tl.ceil(tl.log2(tl.maximum(amax, 1.0e-12) / 6.0))
    codes = _e2m1_code(values / tl.exp2(exponent[:, None]))
    code_pairs = tl.reshape(codes, [32, 16, 2])
    code_lo, code_hi = tl.split(code_pairs)
    packed = (code_lo | (code_hi << 4)).to(tl.uint8)

    out_slot = tl.arange(0, 16)
    out_selector = tl.load(byte_map_ptr + kv_block * 16 + out_slot)
    out_k_half = out_selector // 32
    out_lane_half = (out_selector // 16) % 2
    out_byte_in_lane = out_selector % 16
    u = 2 * d_block + out_k_half
    lane = out_lane_half[None, :] * 32 + channel[:, None]
    tile_offset = u[None, :] * 1024 + lane * 16 + out_byte_in_lane[None, :]
    out_base = out_ptr + (bb * h_kv + hh) * nT * 8192 + t * 8192
    tl.store(out_base + tile_offset, packed)

    # uint8 V-scale image ABI:
    #   byte(k, L, n) = E8M0[d=n*32+L%32, kvblk=k*2+L//32]
    #   offset = k*256 + L*4 + n.
    e8m0 = tl.minimum(tl.maximum(exponent + 127.0, 0.0), 255.0).to(tl.uint8)
    scale_offset = (kv_block // 2) * 256 + (kv_block % 2) * 128 + channel * 4 + d_block
    scale_base = scale_ptr + (bb * h_kv + hh) * nT * 512 + t * 512
    tl.store(scale_base + scale_offset, e8m0)
