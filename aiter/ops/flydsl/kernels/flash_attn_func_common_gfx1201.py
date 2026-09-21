# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Shared representation-independent helpers for gfx1201 flash attention."""

import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr.typing import T

from .tensor_shim import ptr_arg as _ptr_arg


def kv_load_schedule(block_size, head_dim, block_n, vec_width):
    """Return cooperative-load geometry derived only from static tile sizes."""
    threads_per_row = head_dim // vec_width
    total_chunks = block_n * threads_per_row
    num_batches = (total_chunks + block_size - 1) // block_size
    needs_guard = num_batches * block_size != total_chunks
    return threads_per_row, num_batches, needs_guard


def flatten_scores(s_accs, *, num_s_accs):
    """Flatten WMMA score fragments in their logical column order."""
    scores = []
    for acc in range_constexpr(num_s_accs):
        for row in range_constexpr(8):
            scores.append(fx.Vector(s_accs[acc])[row])
    return scores


def mask_scores(
    scores,
    kv_block_start,
    klane,
    q_row_i32,
    seq_len_real,
    c_neg_inf,
    *,
    num_s_accs,
    causal,
):
    """Apply causal or real-sequence tail masking to flattened scores."""
    kv_start_i32 = fx.Int32(kv_block_start)
    klane_off_i32 = fx.Int32(klane) * 8
    masked = []
    for acc in range_constexpr(num_s_accs):
        for row in range_constexpr(8):
            idx = acc * 8 + row
            # Each accumulator covers eight columns in a 16-column WMMA half.
            col_i32 = kv_start_i32 + acc * 16 + row + klane_off_i32
            pred = (
                col_i32 > q_row_i32 if const_expr(causal) else col_i32 >= seq_len_real
            )
            masked.append(pred.select(c_neg_inf, scores[idx]))
    return masked


def configure_gpu_module(ctx, waves_per_eu, flat_work_group_size, daz):
    """Apply raw GPU/LLVM attributes without public FlyDSL launch wrappers."""
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
    return _ptr_arg(value)


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
