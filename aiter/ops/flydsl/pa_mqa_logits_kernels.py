# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""High-level FlyDSL paged MQA-logits ops (gfx950).

Wraps :mod:`aiter.ops.flydsl.kernels.pa_mqa_logits_fp4` (Q FP4, KV FP4) into
a single ``flydsl_pa_mqa_logits_fp4`` op that takes torch tensors, hides the
kernel build / persistent-grid scheduling / JIT launcher boilerplate, and
caches compiled artifacts per shape config via :func:`functools.cache`.
"""

from __future__ import annotations

import functools
from typing import Optional

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
    compute_varctx_schedule,
)

__all__ = [
    "flydsl_pa_mqa_logits_fp4",
    "compute_varctx_schedule",
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
    """Build kernel + JIT launcher for one shape config; cached by signature.

    Cache key includes ``max_chunks_per_cta`` because it controls compile-time
    pipeline unrolling (kernel re-builds when host-side ``safe_chunks_per_cta``
    grows beyond the previously compiled bound).
    """
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
    stream: Optional[torch.cuda.Stream] = None,
) -> torch.Tensor:
    """Compute MQA logits ``L[b,n,t] = sum_h(relu(Q[b,n,h,:] · K[b,t,:]) * W[b,n,h])``.

    Both Q and KV are FP4 e2m1 with UE8M0 block scales (block_size=32). The
    MFMA runs natively on FP4 operands (cbsz=4, blgp=4); no in-kernel dequant.
    gfx950 only.

    Tensor layouts (kernel ABI — caller must preshuffle):

    * ``q_packed``:    ``[B, NEXT_N, H, D//2]`` uint8
        Natural per-head FP4 layout (low nibble = element 0).
    * ``q_scale``:     ``[B, NEXT_N, K_TILES, 4, 16, QS_PAD]`` uint8
        Preshuffled e8m0 scales. ``K_TILES = D // 128``,
        ``QS_PAD = ceil(H/16/4) * 4``. See
        :func:`aiter.ops.flydsl.kernels.pa_mqa_logits_fp4.build_pa_mqa_logits_fp4_module`
        docstring for the layout.
    * ``kv_cache``:    ``[num_blocks, K_TILES, 4, kv_block_size, 16]`` uint8
        Paged + preshuffled FP4 KV.
    * ``kv_scale``:    ``[num_blocks, K_TILES, 4, kv_block_size]`` uint8
        Paged + preshuffled e8m0 KV scales.
    * ``block_tables``: ``[B, max_blocks_per_seq]`` int32
    * ``weights``:     ``[B*NEXT_N, H]`` fp32
    * ``context_lens``: ``[B]`` int32 — per-batch context length.
    * ``out_logits``:  ``[B*NEXT_N, T_max]`` fp32 — written in-place.
        **Caller must pre-fill with -inf**; the kernel only writes valid
        logit positions and relies on pre-init for OOB / past-context tokens.

    Args:
        block_k: tokens per chunk (multiple of MFMA_N=16, divisible by num_warps).
        num_warps: warps per CTA (BLOCK = num_warps * 64).
        parallel_unit_num: target persistent CTA count (typically
            ``cu_count * waves_per_eu``).
        stream: CUDA stream to launch on; defaults to current.

    Returns:
        ``out_logits`` (the same tensor, written in-place).
    """
    batch, next_n, heads, head_dim_packed = q_packed.shape
    head_dim = head_dim_packed * 2
    kv_block_size = kv_cache.shape[3]
    max_blocks_per_seq = block_tables.shape[1]

    safe, cta_info, total_ctas = compute_varctx_schedule(
        context_lens, block_k, parallel_unit_num, next_n=next_n
    )

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
