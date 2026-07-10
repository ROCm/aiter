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

# MFMA tile dims of the fp8 16x16x32 atom: MFMA_M x MFMA_N output tile,
# MFMA_K fp8 elements reduced per MFMA step.
MFMA_M = 16
MFMA_N = 16
MFMA_K = 32

# The fp8 16x16x32 MFMA intrinsic. On gfx942 it reads operands as e4m3 FNUZ
# (bias 8); on gfx950 it reads the native e4m3 FN (OCP). See the FN->FNUZ patch
# below and the per-kernel ``convert_*_fn`` gates.
MFMA_FN = rocdl.mfma_f32_16x16x32_fp8_fp8

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
    _byte = arith.muli(
        arith.muli(_ri64, stride_i64), arith.constant(4, type=T.i64)
    )
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
):
    """One column's logit: fp8 MFMA over heads, ReLU * weight sum, kv-scale, head reduce.

    ``a_row``  : ``a_row[mi][kk]`` packed-i64 Q fragments for this query row.
    ``b_col``  : ``b_col[kk]`` packed-i64 K fragments for this KV column tile.
    ``w_row``  : ``w_row[mi][ii]`` f32 head weights for this query row.
    ``kv_scale``: f32 per-KV-token dequant scale (>= 0), hoisted out of the
                  head sum (ReLU is positive-homogeneous, so ReLU(s*x)=s*ReLU(x)
                  and the whole column sum is scaled once instead of per head).

    Returns the reduced f32 ``col_sum`` (raw ir value). The in-wave (width 64)
    head reduce uses ``shuffle_xor`` offsets 16 and 32.
    """
    col_sum = _to_raw(f32_0)
    for mi in range_constexpr(m_tiles):
        acc = Vec.filled(4, 0.0, fx.Float32)
        for kk in range_constexpr(k_steps):
            acc = mfma_fn(res_ty, [a_row[mi][kk], b_col[kk], acc, 0, 0, 0])
        for ii in range_constexpr(4):
            score = Vec(acc)[ii].ir_value()
            relu = arith.maximumf(score, _to_raw(f32_0))
            wsc = arith.MulFOp(relu, w_row[mi][ii], fastmath=fm_fast).result
            col_sum = arith.AddFOp(col_sum, wsc, fastmath=fm_fast).result
    col_sum = arith.MulFOp(col_sum, kv_scale, fastmath=fm_fast).result

    for sh in (16, 32):
        peer = _to_raw(ArithValue(col_sum).shuffle_xor(sh, 64))
        col_sum = arith.AddFOp(col_sum, peer, fastmath=fm_fast).result
    return col_sum
