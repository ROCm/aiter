# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Top-k reduce + ReduceScatter as one kernel, for reduce-epilogue GEMM2 rows."""

from __future__ import annotations

import functools
import os

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl.expr import gpu

from ..moe_reduce import (
    FP8_VEC,
    _moe_reduction_body,
    _pick_reduce_block,
    ceildiv,
    fp8out_row_bytes,
    fp8out_scale_blk,
)
from ..tensor_shim import _run_compiled as run_compiled
from ..tensor_shim import ptr_arg
from .reduce_scatter import MAX_SERVICE_BLOCKS, RS_UNIT_ELEMS
from .rs_tail import emit_rs_tail, read_epoch, rs_tail_slots

__all__ = ["compile_reduce_rs", "run_reduce_rs"]

_SERVICE_BLOCKS = min(
    MAX_SERVICE_BLOCKS, int(os.environ.get("AITER_TP_REDUCE_RS_SERVICE", "128"))
)


@functools.lru_cache(maxsize=256)
def compile_reduce_rs(
    *,
    tp_size: int,
    topk: int,
    model_dim: int,
    dtype_str: str,
    out_dtype_str: str,
    use_weight: bool,
    scale_blk: int | None = None,
    pitch_align: int | None = None,
    service_blocks: int = _SERVICE_BLOCKS,
):
    """Build the fused reduction + ReduceScatter launcher for one shape."""
    if not 1 <= tp_size <= 8:
        raise ValueError(f"tp_size must be in [1, 8], got {tp_size}")
    if model_dim % RS_UNIT_ELEMS:
        raise ValueError(
            f"model_dim must be a multiple of {RS_UNIT_ELEMS}, got {model_dim}"
        )

    V = FP8_VEC if dtype_str == "fp8" else 128 // (32 if dtype_str == "f32" else 16)
    block = _pick_reduce_block(model_dim, V)
    gy = ceildiv(model_dim, block * V)
    if dtype_str == "fp8":
        scale_blk = fp8out_scale_blk(model_dim) if scale_blk is None else int(scale_blk)
        fp8_row_stride = fp8out_row_bytes(
            model_dim, scale_blk=scale_blk, pitch_align=pitch_align
        )
    else:
        scale_blk, fp8_row_stride = FP8_VEC, model_dim

    tail_slots = rs_tail_slots(tp_size)
    name = (
        f"mega_moe_tp_reduce_rs_{dtype_str}_{out_dtype_str}_t{topk}_n{model_dim}"
        f"_w{int(use_weight)}_s{scale_blk}_r{fp8_row_stride}_b{block}"
        f"_tp{tp_size}_sv{service_blocks}"
    )

    @flyc.kernel(name=name, known_block_size=[block, 1, 1])
    def reduce_rs_kernel(
        X: fx.Pointer,
        Y: fx.Pointer,
        expert_mask: fx.Pointer,
        topk_ids: fx.Pointer,
        topk_weights: fx.Pointer,
        i32_m_tokens: fx.Int32,
        arg_desc: fx.Int64,
        i32_rank: fx.Int32,
        i32_rows: fx.Int32,
    ):
        epoch_addr, epoch = read_epoch(arg_desc, tail_slots)
        _moe_reduction_body(
            X,
            Y,
            expert_mask,
            topk_ids,
            topk_weights,
            i32_m_tokens,
            topk,
            model_dim,
            dtype_str,
            False,
            0,
            out_dtype_str,
            use_weight,
            scale_blk,
            fp8_row_stride,
            block,
        )
        gx = fx.Int32(gpu.grid_dim.x)
        cta = fx.Int32(gpu.block_id("y")) * gx + fx.Int32(gpu.block_id("x"))
        emit_rs_tail(
            arg_desc,
            i32_rank,
            i32_rows,
            epoch_addr,
            epoch,
            cta,
            gx * fx.Int32(gy),
            fx.Int32(gpu.thread_id("x")),
            tp_size=tp_size,
            model_dim=model_dim,
            block=block,
            service_blocks=service_blocks,
        )

    @flyc.jit
    def launch(
        X: fx.Pointer,
        Y: fx.Pointer,
        expert_mask: fx.Pointer,
        topk_ids: fx.Pointer,
        topk_weights: fx.Pointer,
        i32_m_tokens: fx.Int32,
        arg_desc: fx.Int64,
        i32_rank: fx.Int32,
        i32_rows: fx.Int32,
        stream: fx.Stream,
    ):
        reduce_rs_kernel(
            X,
            Y,
            expert_mask,
            topk_ids,
            topk_weights,
            i32_m_tokens,
            arg_desc,
            i32_rank,
            i32_rows,
        ).launch(
            grid=(fx.Int64(i32_m_tokens), gy, 1), block=(block, 1, 1), stream=stream
        )

    launch.block = block
    return launch


def run_reduce_rs(
    *,
    target,
    partial,
    output,
    token_num,
    topk,
    model_dim,
    tp_size,
    rank,
    local_rows,
    desc_ptr,
    is_fp8=False,
    topk_weights=None,
    fp8_scale_blk=None,
    fp8_pitch_align=None,
    stream=None,
):
    """Host side: reduce the staged routes into the arena, then ReduceScatter."""
    out_dtype_str = "bf16" if partial.dtype == torch.bfloat16 else "f16"
    if is_fp8:
        from ..mxfp4_gemm_common import FP8OUT_PITCH_ALIGN

        dtype_str = "fp8"
        X = target
        fp8_scale_blk = (
            fp8out_scale_blk(model_dim)
            if fp8_scale_blk is None
            else int(fp8_scale_blk)
        )
        fp8_pitch_align = (
            FP8OUT_PITCH_ALIGN if fp8_pitch_align is None else int(fp8_pitch_align)
        )
    else:
        dtype_str = out_dtype_str
        X = target.view(token_num, topk, model_dim)
        fp8_scale_blk = fp8_pitch_align = None

    use_weight = topk_weights is not None
    tw = (
        topk_weights.to(torch.float32).contiguous()
        if use_weight
        else torch.empty(0, device=partial.device, dtype=torch.float32)
    )
    empty_i32 = torch.empty(0, device=partial.device, dtype=torch.int32)
    launch = compile_reduce_rs(
        tp_size=int(tp_size),
        topk=int(topk),
        model_dim=int(model_dim),
        dtype_str=dtype_str,
        out_dtype_str=out_dtype_str,
        use_weight=use_weight,
        scale_blk=fp8_scale_blk,
        pitch_align=fp8_pitch_align,
    )
    run_compiled(
        launch,
        ptr_arg(X),
        ptr_arg(partial),
        ptr_arg(empty_i32),
        ptr_arg(empty_i32),
        ptr_arg(tw),
        int(token_num),
        int(desc_ptr),
        int(rank),
        int(local_rows),
        stream if stream is not None else torch.cuda.current_stream(),
    )
    return output[:local_rows]
