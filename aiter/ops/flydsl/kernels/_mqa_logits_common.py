# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Shared FlyDSL building blocks for the FP8 MQA-logits kernels.

Both the dense (``fp8_mqa_logits.py``) and paged (``fp8_paged_mqa_logits.py``)
kernels compute the same core per query row / KV column::

    logits[row, n] = sum_h ReLU(<Q[row, h, :], K[n, :]> * kv_scale[n]) * weights[row, h]

They differ only in how K / kv_scale are addressed (contiguous vs paged gather)
and in the windowing / output layout. Everything that is identical -- the fp8
16x16x32 MFMA compute, the ReLU * weight head-sum + kv-scale hoist, the in-wave
``shuffle_xor`` head reduce, the fp8 dword-pack load, the FN->FNUZ byte patch,
and the i64 byte-base per-row output view -- lives here so neither kernel
duplicates it.
"""

# No `from __future__ import annotations`: FlyDSL arg typing needs real
# annotation objects, not PEP 563 strings.

from functools import lru_cache

import torch

import flydsl.expr as fx
from flydsl.expr import arith, range_constexpr, rocdl
from flydsl.expr.numeric import ArithValue
from flydsl.expr.typing import T

from .tensor_shim import GTensor, _to_raw

Vec = fx.Vector

# MFMA tile dims: 32x32x64 scaled fp8 (CDNA4, gfx950).
# K=64 per step → K_STEPS = head_size // 64 (2 for D=128).
# A/B operands are vector<8xi32> (4 packed i64s = 32 fp8 bytes per lane).
# 16 f32 D-regs/lane, same output tile shape as 32x32x16 (same D-reg layout).
# lane_div_N = lane // 32 (0 or 1), lane_mod_N = lane % 32 (KV col index).
# Scale operands use neutral e8m0 (0x7F7F7F7F = 1.0 per byte, 4 packed).
MFMA_M = 32
MFMA_N = 32
MFMA_K = 64


# The scaled 32x32x64 MFMA: A/B are vector<8xi32>, same D-reg layout as 32x32x16.
# operands: [a128, b128, c, cbsz=0, blgp=0, opsel_a=0, scale_a, opsel_b=0, scale_b]
# where a128/b128 are Vec(8xi32) (4 i64s packed), scale_a/scale_b are i32 values
# with 4 packed e8m0 bytes. Neutral: 0x7F7F7F7F (all 1.0).
def _mfma_scale_f32_32x32x64_fp8(result_type, operands):
    from flydsl.expr.rocdl import _unwrap_mfma_operand

    a = _unwrap_mfma_operand(operands[0])  # vector<8xi32>
    b = _unwrap_mfma_operand(operands[1])  # vector<8xi32>
    c = _unwrap_mfma_operand(operands[2])  # vector<16xf32>
    # Neutral scale: 0x7F7F7F7F = 4 packed e8m0 bytes, each 127 → 2^(127-127)=1.0
    neutral = arith.constant(0x7F7F7F7F, type=T.i32)
    return rocdl.mfma_scale_f32_32x32x64_f8f6f4_(
        result_type, a, b, c, 0, 0, 0, neutral, 0, neutral
    )


MFMA_FN = _mfma_scale_f32_32x32x64_fp8

DEFAULT_COMPILE_HINTS = {
    "waves_per_eu": 2,
    "fast_fp_math": True,
}


def i32_add(a, b):
    """i32 add (result stays Int32, not index type)."""
    return fx.Int32(arith.addi(_to_raw(a), _to_raw(b)))


@lru_cache(maxsize=8)
def device_cu_count(device_index: int) -> int:
    """Compute-unit count for a CUDA/HIP device (cached); 304 if unavailable."""
    try:
        return torch.cuda.get_device_properties(device_index).multi_processor_count
    except Exception:
        return 304


def load_pack_i64(i32_view, byte_off_i32):
    """Load 8 fp8 bytes as 2 i32 dwords and bitcast to i64.

    fp8 operands are read 8 bytes at a time as 2 i32 dwords (a ``v8i8``
    buffer_load fails to lower on gfx942), then bitcast to i64 for the MFMA.
    ``i32_view`` is a ``GTensor(..., dtype=T.i32)`` over the fp8 bytes and
    ``byte_off_i32`` is the byte offset of the first of the 8 bytes.
    """
    dword_off = fx.Int32(arith.divui(_to_raw(byte_off_i32), _to_raw(fx.Int32(4))))
    v2 = i32_view.vec_load((dword_off,), vec_size=2)
    return Vec(v2).bitcast(fx.Int64)[0].ir_value()


def load_pack_v8i32(i32_view, byte_off_i32, lane8):
    """Load 32 fp8 bytes as vector<8xi32> for the 32x32x64 scaled MFMA.

    The 32x32x64 MFMA (K=64) A/B operands are vector<8xi32> per lane.
    Each lane holds 32 fp8 bytes covering 4 consecutive K-groups of 16.
    Same layout as 4 consecutive 32x32x16 K-steps: the per-lane K-bytes are
    at positions [lane_div_N*8 + kk*16 + 0..7] for kk in 0..3.

    ``byte_off_i32`` is the base byte offset (tok_byte or q_row_base + h_a*stride)
    for the first K-group. ``lane8 = lane_div_N * 8`` is the within-K-group offset.
    Returns a vector<8xi32> value (MLIR Value).
    """

    def _load_i64(off):
        dw = fx.Int32(
            arith.divui(_to_raw(byte_off_i32 + lane8 + off), _to_raw(fx.Int32(4)))
        )
        v2 = i32_view.vec_load((dw,), vec_size=2)
        return Vec(v2).bitcast(fx.Int64)[0].ir_value()

    i64_0 = _load_i64(0)  # K-group 0: k = lane_div_N*8 + 0..7
    i64_1 = _load_i64(16)  # K-group 1: k = lane_div_N*8 + 16..23
    i64_2 = _load_i64(32)  # K-group 2: k = lane_div_N*8 + 32..39
    i64_3 = _load_i64(48)  # K-group 3: k = lane_div_N*8 + 48..55
    return Vec.from_elements([i64_0, i64_1, i64_2, i64_3], fx.Int64).bitcast(fx.Int32)


def fn_to_fnuz_i64(raw_i64):
    """Map FN byte 0x80 (neg-zero) -> 0x00 in 8 packed fp8 bytes.

    An e4m3 FN (OCP, bias 7) operand encodes exactly 2x the value the gfx942
    FNUZ (bias 8) MFMA reads for the same byte, except FN -0 = 0x80 which is
    FNUZ NaN; patch that byte to +0 and let the caller undo the 2x via the
    kv_scale (ReLU positive-homogeneity keeps the result exact).
    """
    lo_i32 = arith.TruncIOp(T.i32, raw_i64).result
    hi_i64 = arith.ShRUIOp(raw_i64, arith.constant(32, type=T.i64)).result
    hi_i32 = arith.TruncIOp(T.i32, hi_i64).result

    def _fix_i32(src):
        result = arith.constant(0, type=T.i32)
        for byte_idx in range_constexpr(4):
            shift = arith.constant(byte_idx * 8, type=T.i32)
            byte_val = arith.andi(
                arith.shrui(src, shift),
                arith.constant(0xFF, type=T.i32),
            )
            is_0x80 = arith.cmpi(
                arith.CmpIPredicate.eq,
                byte_val,
                arith.constant(0x80, type=T.i32),
            )
            cleaned = arith.select(
                is_0x80,
                arith.constant(0, type=T.i32),
                byte_val,
            )
            result = arith.ori(result, arith.shli(cleaned, shift))
        return result

    lo_fix = _fix_i32(lo_i32)
    hi_fix = _fix_i32(hi_i32)
    lo_64 = arith.ExtUIOp(T.i64, lo_fix).result
    hi_64 = arith.ShLIOp(
        arith.ExtUIOp(T.i64, hi_fix).result, arith.constant(32, type=T.i64)
    ).result
    return arith.OrIOp(lo_64, hi_64).result


def make_out_row_view(logits, stride_i64, row_i32):
    """1-D output GTensor for ``row_i32``; byte base computed in i64.

    The row's i64 byte offset (``row * stride * 4`` bytes for f32) goes into the
    base pointer so the remaining column offset stays in i32. A 2-D (row, col)
    view would compute ``row * stride + col`` in i32 and overflow past 2^31
    (~46k-square dense outputs / large paged ``max_model_len``), silently
    mis-writing.
    """
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
    res_ty,
    f32_0,
    fm_fast,
    mfma_fn=MFMA_FN,
    dreg_count=16,
):
    """One column's logit: fp8 MFMA over heads, ReLU * weight sum, kv-scale, head reduce.

    ``a_row``  : ``a_row[mi][kk]`` Q fragments (i64 or vector<8xi32>) for this row.
    ``b_col``  : ``b_col[kk]`` K fragments (same type as a_row) for this KV tile.
    ``w_row``  : ``w_row[mi][ii]`` f32 head weights for this query row.
    ``kv_scale``: f32 per-KV-token dequant scale (>= 0), hoisted out of the
                  head sum (ReLU is positive-homogeneous, so ReLU(s*x)=s*ReLU(x)
                  and the whole column sum is scaled once instead of per head).
    ``dreg_count``: number of f32 D-regs per lane (16 for 32x32 tiles).

    Returns the reduced f32 ``col_sum`` (raw ir value). The in-wave (width 64)
    head reduce uses shuffle_xor(32) for the 32x32 MFMA (lane_div_N in {0,1}).
    """
    col_sum = _to_raw(f32_0)
    for mi in range_constexpr(m_tiles):
        acc = Vec.filled(dreg_count, 0.0, fx.Float32)
        for kk in range_constexpr(k_steps):
            acc = mfma_fn(res_ty, [a_row[mi][kk], b_col[kk], acc, 0, 0, 0])
        for ii in range_constexpr(dreg_count):
            score = Vec(acc)[ii].ir_value()
            relu = arith.maximumf(score, _to_raw(f32_0))
            wsc = arith.MulFOp(relu, w_row[mi][ii], fastmath=fm_fast).result
            col_sum = arith.AddFOp(col_sum, wsc, fastmath=fm_fast).result
    col_sum = arith.MulFOp(col_sum, kv_scale, fastmath=fm_fast).result

    # For 32x32 tile: lane_div_N = lane // 32 (0 or 1); shuffle_xor(32) sums them.
    # For 16x16 tile: lane_div_N = lane // 16 (0..3); shuffle_xor(16) + xor(32).
    shuffles = (32,) if dreg_count == 16 else (16, 32)
    for sh in shuffles:
        peer = _to_raw(ArithValue(col_sum).shuffle_xor(sh, 64))
        col_sum = arith.AddFOp(col_sum, peer, fastmath=fm_fast).result
    return col_sum
