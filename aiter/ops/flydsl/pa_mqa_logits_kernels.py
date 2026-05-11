# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""High-level FlyDSL paged MQA-logits wrapper (gfx950)."""

from __future__ import annotations

import functools
from typing import Optional, Tuple

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir as _ir
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import arith
from flydsl.expr.typing import T

from aiter.ops.flydsl.kernels.pa_mqa_logits_fp4 import (
    DEFAULT_BLOCK_THREADS,
    build_pa_mqa_logits_fp4_module,
    flydsl_pa_mqa_logits_fp4_schedule,
)

__all__ = [
    "flydsl_pa_mqa_logits_fp4",
    "flydsl_pa_mqa_logits_fp4_schedule",
]


@functools.cache
def _get_compiled_pa_mqa_logits_fp4(
    block_k: int,
    kv_block_size: int,
    max_blocks_per_seq: int,
    max_chunks_per_cta: int,
    num_warps: int,
    next_n: int,
    heads: int,
    head_dim: int,
):
    """Build kernel + JIT launcher for one shape config; cached by signature."""
    kfn, alloc = build_pa_mqa_logits_fp4_module(
        block_k=block_k,
        kv_block_size=kv_block_size,
        max_blocks_per_seq=max_blocks_per_seq,
        max_chunks_per_cta=max_chunks_per_cta,
        num_warps=num_warps,
        next_n=next_n,
        heads=heads,
        head_dim=head_dim,
    )
    block_threads = getattr(alloc, "block_threads", DEFAULT_BLOCK_THREADS)

    @flyc.jit
    def launch_kernel(
        out,
        q,
        qs,
        kv,
        kvs,
        bt,
        w,
        cta_info_,
        stride_out: fx.Int32,
        gx: fx.Int32,
        stream: fx.Stream,
    ):
        alloc.finalized = False
        cctx = CompilationContext.get_current()
        with _ir.InsertionPoint(cctx.gpu_module_body):
            alloc.finalize()
        gxi = arith.index_cast(T.index, gx.ir_value())
        kfn(out, q, qs, kv, kvs, bt, w, cta_info_, stride_out).launch(
            grid=(gxi,), block=(block_threads, 1, 1), stream=stream
        )

    return launch_kernel


def flydsl_pa_mqa_logits_fp4(
    q_packed: torch.Tensor,
    q_scale: torch.Tensor,
    kv_cache: torch.Tensor,
    kv_scale: torch.Tensor,
    block_tables: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    out_logits: torch.Tensor,
    *,
    block_k: int = 256,
    num_warps: int = 4,
    parallel_unit_num: int = 512,
    schedule: Optional[Tuple[int, torch.Tensor, int]] = None,
    stream: Optional[torch.cuda.Stream] = None,
) -> torch.Tensor:
    """Compute MQA logits (FP4 Q/KV, gfx950). Writes out_logits in-place and returns it.

    Pass `schedule` (tuple from `flydsl_pa_mqa_logits_fp4_schedule`) to skip the per-call
    schedule recompute — useful in benchmark loops or when reusing across calls
    with the same context_lens.
    """
    batch, next_n, heads, head_dim_packed = q_packed.shape
    head_dim = head_dim_packed * 2
    kv_block_size = kv_cache.shape[3]
    max_blocks_per_seq = block_tables.shape[1]

    if schedule is None:
        schedule = flydsl_pa_mqa_logits_fp4_schedule(
            context_lens, block_k, parallel_unit_num, next_n=next_n
        )
    safe, cta_info, total_ctas = schedule

    launch = _get_compiled_pa_mqa_logits_fp4(
        block_k=block_k,
        kv_block_size=kv_block_size,
        max_blocks_per_seq=max_blocks_per_seq,
        max_chunks_per_cta=safe,
        num_warps=num_warps,
        next_n=next_n,
        heads=heads,
        head_dim=head_dim,
    )

    if stream is None:
        stream = torch.cuda.current_stream()

    t_max = out_logits.shape[1]
    launch(
        out_logits,
        q_packed,
        q_scale,
        kv_cache,
        kv_scale,
        block_tables,
        weights,
        cta_info,
        t_max,
        total_ctas,
        stream,
    )
    return out_logits
