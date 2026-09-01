# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL building blocks for the paged FP8 MQA-logits kernel (gfx950 / CDNA4).

logits[row, n] = sum_h ReLU(<Q[row, h, :], K[n, :]> * kv_scale[n]) * weights[row, h]
"""

from functools import lru_cache

import flydsl.expr as fx
import torch
from flydsl.expr import arith, range_constexpr, rocdl
from flydsl.expr.typing import T

from ..tensor_shim import GTensor, _to_raw

Vec = fx.Vector

# 32x32x64 scaled fp8 MFMA (gfx950 / CDNA4). A/B are vector<8xi32> per lane
# (32 fp8 bytes), C/D are 16 f32 D-regs. lane_div_N = lane//32, lane_mod_N = lane%32.
MFMA_M = 32
MFMA_N = 32
MFMA_K = 64


def _mfma_scale_f32_32x32x64_fp8(result_type, operands):
    """operands: [a, b, c] or [a, b, c, cbsz, blgp, opsel_a, scale_a, opsel_b, scale_b]."""
    # Neutral e8m0 scale: 4 packed bytes of 127 => 2^(127-127) = 1.0 each.
    neutral = arith.constant(0x7F7F7F7F, type=T.i32)
    ops = list(operands)
    if len(ops) < 9:
        while len(ops) < 6:
            ops.append(0)
        ops.extend([neutral, 0, neutral])
    return rocdl.mfma_scale_f32_32x32x64_f8f6f4(result_type, ops)


MFMA_FN = _mfma_scale_f32_32x32x64_fp8

DEFAULT_COMPILE_HINTS = {
    "waves_per_eu": 2,
    "fast_fp_math": True,
}


# Via Uint32 because `//` on a signed Int32 lowers to floordivsi, which expands
# to several instructions for negative-operand rounding. Everything divided here
# is non-negative, so this keeps the single-instruction divui/remui.
def udiv(a, b):
    return fx.Int32(fx.Uint32(a) // fx.Uint32(b))


def umod(a, b):
    return fx.Int32(fx.Uint32(a) % fx.Uint32(b))


@lru_cache(maxsize=8)
def device_cu_count(device_index: int) -> int:
    """Compute-unit count for a CUDA/HIP device (cached); 256 if unavailable."""
    try:
        return torch.cuda.get_device_properties(device_index).multi_processor_count
    except Exception:  # noqa: BLE001
        return 256


def load_pack_v8i32(i32_view, byte_off_i32, lane8):
    """Load 32 fp8 bytes as the vector<8xi32> A/B operand of the 32x32x64 MFMA.

    Each lane holds 4 K-groups of 8 bytes at ``lane8 + kk*16``. Reads go out as
    2-dword loads because a ``v8i8`` buffer_load fails to lower.
    """

    def _load_i64(off):
        v2 = i32_view.vec_load((udiv(byte_off_i32 + lane8 + off, 4),), vec_size=2)
        return Vec(v2).bitcast(fx.Int64)[0].ir_value()

    i64_0 = _load_i64(0)  # K-group 0: k = lane_div_N*8 + 0..7
    i64_1 = _load_i64(16)  # K-group 1: k = lane_div_N*8 + 16..23
    i64_2 = _load_i64(32)  # K-group 2: k = lane_div_N*8 + 32..39
    i64_3 = _load_i64(48)  # K-group 3: k = lane_div_N*8 + 48..55
    return Vec.from_elements([i64_0, i64_1, i64_2, i64_3], fx.Int64).bitcast(fx.Int32)


def make_kv_key_view(kv_cache, head_size, kv_block_size, index_dim, preshuffle):
    """Dword view of the co-packed KV keys over the gather's own axes:
    ``(block, token//16, token%16, K-step, sub-group, lane half)``.

    The two data layouts differ only in this stride vector -- block-flat puts
    token c at ``c*D``, shuffle_weight(16,16) at ``(c//16)*16D + (c%16)*16``
    with hidden groups every 256B -- so Preshuffle is a layout, not a second
    gather path. index_dim is a multiple of 4, so the block stride divides.
    """
    D = head_size
    blk = udiv(fx.Int32(kv_block_size) * index_dim, 4)
    stride = (
        (blk, D * 4, 4, 256, 64, 2) if preshuffle else (blk, D * 4, D // 4, 16, 4, 2)
    )
    return GTensor(kv_cache, dtype=T.i32, shape=(-1,) * 6, stride=stride)


def make_kv_scale_view(kv_cache, head_size, kv_block_size, index_dim):
    """f32 ``(block, token)`` view of the co-packed scales. The tail follows the
    KVB*D key bytes and is token-ordered in both layouts, so it folds into base.
    """
    return GTensor(
        kv_cache,
        dtype=T.f32,
        shape=(-1, kv_block_size),
        stride=(udiv(fx.Int32(kv_block_size) * index_dim, 4), 1),
        base_offset=kv_block_size * head_size // 4,
    )


def load_pack_kv(key_view, physical, tok_in_block, kk_step, lane_div_N):
    """vector<8xi32> B operand for one KV token and K-step. Layout-agnostic:
    the view's strides carry the block-flat/preshuffle difference.
    """
    c_hi, c_lo = udiv(tok_in_block, 16), umod(tok_in_block, 16)
    # Slice everything but `sub` first: that fixes the token's base offset once
    # instead of re-deriving it inside each of the four loads.
    sub_view = key_view[physical, c_hi, c_lo, kk_step, None, lane_div_N]

    def _load_i64(sub):
        v2 = sub_view.vec_load((sub,), vec_size=2)
        return Vec(v2).bitcast(fx.Int64)[0].ir_value()

    return Vec.from_elements(
        [_load_i64(sub) for sub in range_constexpr(4)], fx.Int64
    ).bitcast(fx.Int32)


def make_out_row_view(logits, stride_out, row_i32):
    """1-D output GTensor for ``row_i32``, with the row byte base folded into
    the pointer in i64 so the column offset stays i32. A 2-D (row, col) view
    would compute ``row*stride + col`` in i32 and silently overflow past 2^31.
    """
    stride_i64 = arith.extui(T.i64, _to_raw(stride_out))
    _ri64 = arith.extui(T.i64, _to_raw(row_i32))
    _byte = arith.muli(arith.muli(_ri64, stride_i64), arith.constant(4, type=T.i64))
    _idx = arith.index_cast(T.index, _byte)
    return GTensor(logits, dtype=T.f32, shape=(-1,), static_bytes_offset_i64=_idx)


def mfma_head_reduce(
    a_row,
    b_col,
    w_row,
    kv_scale,
    *,
    m_tiles,
    k_steps,
    dreg_count=16,
    mfma_fn=MFMA_FN,
):
    """One column's logit: fp8 MFMA over heads, ReLU * weight sum, kv-scale,
    in-wave head reduce.

    ``kv_scale`` (>= 0) is applied once to the column sum rather than per head:
    ReLU is positively homogeneous, so ReLU(s*x) = s*ReLU(x) and hoisting it out
    of the head sum is exact.
    """
    res_ty = Vec.make_type(dreg_count, fx.Float32)
    f32_0 = fx.Float32(0.0)
    col_sum = f32_0
    for mi in range_constexpr(m_tiles):
        acc = Vec.filled(dreg_count, 0.0, fx.Float32)
        for kk in range_constexpr(k_steps):
            acc = mfma_fn(res_ty, [a_row[mi][kk], b_col[kk], acc, 0, 0, 0])
        for ii in range_constexpr(dreg_count):
            score = fx.Float32(Vec(acc)[ii])
            relu = score.maximumf(f32_0)
            col_sum = col_sum + relu * w_row[mi][ii]
    col_sum = col_sum * fx.Float32(kv_scale)

    # For 32x32 tile: lane_div_N = lane // 32 (0 or 1); shuffle_xor(32) sums them.
    shuffles = (32,) if dreg_count == 16 else (16, 32)
    for sh in shuffles:
        peer = col_sum.shuffle_xor(sh, 64)
        col_sum = col_sum + peer
    return col_sum
