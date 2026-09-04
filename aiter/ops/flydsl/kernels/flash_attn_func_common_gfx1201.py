# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Shared representation-independent helpers for gfx1201 flash attention."""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl.expr import arith, const_expr, range_constexpr, rocdl
from flydsl.expr import math as fmath
from flydsl.expr.typing import T
from flydsl.expr.utils.arith import _to_raw as _raw


def _llvm_value(value):
    if hasattr(value, "ir_value") and not isinstance(value, ir.Value):
        return value.ir_value()
    return value


def pointer_to_llvm_ptr(ptr):
    ptr_i64 = fx.Int64(fx.ptrtoint(ptr)).ir_value()
    return llvm.IntToPtrOp(ir.Type.parse("!llvm.ptr"), ptr_i64).result


def pointer_load(result_type, ptr):
    return llvm.LoadOp(result_type, _llvm_value(ptr)).result


def pointer_store(value, ptr):
    return llvm.StoreOp(_llvm_value(value), _llvm_value(ptr))


def fast_add(a, b):
    return arith.addf(_raw(a), _raw(b), fastmath=arith.FastMathFlags.fast)


def fast_sub(a, b):
    return arith.subf(_raw(a), _raw(b), fastmath=arith.FastMathFlags.fast)


def fast_mul(a, b):
    return arith.mulf(_raw(a), _raw(b), fastmath=arith.FastMathFlags.fast)


def fast_max(a, b):
    return arith.MaxNumFOp(_raw(a), _raw(b), fastmath=arith.FastMathFlags.fast).result


def wave32_peer(value):
    return fx.Float32(value).shuffle_xor(16, 32)


def kv_load_schedule(block_size, head_dim, block_n, vec_width):
    """Return cooperative-load geometry derived only from static tile sizes."""
    threads_per_row = head_dim // vec_width
    rows_per_batch = block_size // threads_per_row
    if rows_per_batch >= block_n:
        return threads_per_row, rows_per_batch, 1, rows_per_batch > block_n
    return threads_per_row, rows_per_batch, block_n // rows_per_batch, False


def flatten_and_mask_scores(
    s_accs,
    kv_block_start,
    klane,
    q_row_i32,
    seq_len_real,
    c_neg_inf,
    *,
    num_s_accs,
    causal,
    tail_mask,
):
    """Flatten WMMA score fragments and apply causal or tail masking."""
    s_raw = []
    for st in range_constexpr(num_s_accs):
        for r in range_constexpr(8):
            s_raw.append(fx.Vector(s_accs[st])[r])

    if const_expr(causal or tail_mask):
        kv_start_i32 = fx.Int32(kv_block_start)
        klane_off_i32 = fx.Int32(klane) * 8
        masked = []
        for acc in range_constexpr(num_s_accs):
            for r in range_constexpr(8):
                idx = acc * 8 + r
                col_i32 = kv_start_i32 + acc * 16 + r + klane_off_i32
                pred = (
                    col_i32 > q_row_i32
                    if const_expr(causal)
                    else col_i32 >= seq_len_real
                )
                masked.append(pred.select(c_neg_inf, s_raw[idx]))
        return masked
    return s_raw


def update_online_softmax(
    s_raw,
    m_running,
    l_running,
    o_accs,
    sm_scale_log2e,
    c_zero_f,
    *,
    num_s_vals,
    d_chunks,
):
    """Update online-softmax state and return probabilities for GEMM2."""
    local_max = s_raw[0]
    for r in range_constexpr(num_s_vals - 1):
        local_max = fast_max(local_max, s_raw[r + 1])
    row_max = fast_max(local_max, wave32_peer(local_max))
    m_new = fast_max(m_running, row_max)

    diff_m_scaled = fast_mul(fast_sub(m_running, m_new), sm_scale_log2e)
    corr = rocdl.exp2(ir.F32Type.get(), _raw(diff_m_scaled))
    neg_scaled_max = fast_sub(c_zero_f, fast_mul(sm_scale_log2e, m_new))

    p_vals = []
    local_sum = _raw(c_zero_f)
    for r in range_constexpr(num_s_vals):
        diff = fmath.fma(s_raw[r], _raw(sm_scale_log2e), neg_scaled_max)
        p = rocdl.exp2(ir.F32Type.get(), _raw(diff))
        p_vals.append(p)
        local_sum = fast_add(local_sum, p)

    tile_sum = fast_add(local_sum, wave32_peer(local_sum))
    l_new = fast_add(fast_mul(corr, l_running), tile_sum)
    corr_vec = fx.Vector.from_elements([corr], fx.Float32).broadcast_to(8).ir_value()
    for dc in range_constexpr(d_chunks):
        o_accs[dc] = fast_mul(o_accs[dc], corr_vec)
    return p_vals, m_new, l_new, o_accs


def next_kv_tile_start(kv_block_start, kv_upper, block_n, zero):
    """Return the next tile start, wrapping the final speculative load to zero."""
    next_start = kv_block_start + block_n
    return (next_start < kv_upper).select(next_start, zero)


def configure_gpu_module(ctx, waves_per_eu, flat_work_group_size, daz):
    """Apply launch attributes shared by both attention kernels."""
    if const_expr(waves_per_eu is not None):
        value = int(waves_per_eu)
        if const_expr(value >= 1):
            for op in ctx.gpu_module_body.operations:
                if const_expr(getattr(op, "OPERATION_NAME", None) == "gpu.func"):
                    op.attributes["rocdl.waves_per_eu"] = ir.IntegerAttr.get(
                        T.i32, value
                    )
    if const_expr(flat_work_group_size is not None):
        value = int(flat_work_group_size)
        if const_expr(value >= 1):
            flat_wg_attr = ir.StringAttr.get(f"{value},{value}")
            for op in ctx.gpu_module_body.operations:
                if const_expr(getattr(op, "OPERATION_NAME", None) == "gpu.func"):
                    op.attributes["rocdl.flat_work_group_size"] = flat_wg_attr

    passthrough_entries = []
    if const_expr(daz):
        for name, value in (
            ("denormal-fp-math-f32", "preserve-sign,preserve-sign"),
            ("no-nans-fp-math", "true"),
            ("unsafe-fp-math", "true"),
        ):
            passthrough_entries.append(
                ir.ArrayAttr.get([ir.StringAttr.get(name), ir.StringAttr.get(value)])
            )
    for op in ctx.gpu_module_body.operations:
        if const_expr(getattr(op, "OPERATION_NAME", None) == "gpu.func"):
            op.attributes["passthrough"] = ir.ArrayAttr.get(passthrough_entries)


def pointer_arg(value):
    """Convert tensor-like launch arguments to raw FlyDSL pointers."""
    if not hasattr(value, "data_ptr"):
        return value
    type_name = type(value).__name__
    module_name = type(value).__module__
    ptr = (
        0
        if type_name == "FakeTensor" or "fake_tensor" in module_name
        else value.data_ptr()
    )
    return flyc.from_c_void_p(fx.Uint8, ptr)


def wrap_pointer_args(args, kwargs, positional_indices, keyword_names):
    """Convert selected positional and keyword launch arguments to pointers."""
    args = list(args)
    for idx in positional_indices:
        if idx < len(args):
            args[idx] = pointer_arg(args[idx])
    for name in keyword_names:
        if name in kwargs:
            kwargs[name] = pointer_arg(kwargs[name])
    return tuple(args), kwargs
