# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Hand-written HIP prefill-phase FP8 MQA indexer logits + per-row top-K for gfx950.

The dense half of the DeepSeek-V3.2 / GLM-5 sparse-attention lightning indexer and
the per-row top-K that follows it:

    logits[m, n] = sum_h relu(Q[m,h,:] . K[n,:]) * w[m,h] * kv_scale[n]

ranked without writing the fp32 ``[M, N]`` logits: a 1/32 sample of each row's keys
gives it a threshold, the logits scan keeps only the keys above it, and a select
ranks those candidates (rows it cannot settle are recomputed exactly). Same inputs
as ``aiter.ops.fp8_mqa_logits``; replaces ``fp8_mqa_logits`` + ``top_k_per_row_prefill``.
Kernel design: see csrc/kernels/fp8_mqa_topk.cu.

Fixed at n_heads=32, head_dim=128 -- the GLM-5.x indexer shape.
``is_supported()`` gates on that, so a caller that also has to serve other shapes
can route them to the unfused path rather than trip a TORCH_CHECK.
"""

from torch import Tensor

from ..jit.core import compile_ops
from ..jit.utils.chip_info import get_gfx

MD_NAME = "module_fp8_mqa_topk"

SUPPORTED_GFX = ("gfx950",)
NUM_HEADS = 32
HEAD_DIM = 128
MAX_TOPK = 512


def is_supported(num_heads: int, head_dim: int) -> bool:
    """True when this kernel can run this shape on this device."""
    return (
        get_gfx() in SUPPORTED_GFX and num_heads == NUM_HEADS and head_dim == HEAD_DIM
    )


@compile_ops(MD_NAME, fc_name="fp8_mqa_topk")
def fp8_mqa_topk(
    q_fp8: Tensor,
    k_fp8: Tensor,
    kv_scale: Tensor,
    weights: Tensor,
    cu_seqlen_ks: Tensor,
    cu_seqlen_ke: Tensor,
    topk_indices: Tensor,
    TotalCuCount: int = 256,
    fallback_rows: Tensor | None = None,
    force_fallback: bool = False,
) -> Tensor:
    """Indexer logits + a per-row top-K in one call that never writes fp32 logits.

    q_fp8       [M, 32, 128] fp8   k_fp8    [N, 128] fp8
    kv_scale    [N] f32            weights  [M, 32] f32
    cu_seqlen_ks/ke [M] i32 -- row m is valid on [ks[m], ke[m]); either bound may sit
    outside [0, N), the row is then empty over the part that does.

    topk_indices [M, K] i32, K <= 512, written in place (and returned): row m gets the
    indices, relative to its first valid key max(ks[m], 0), of its K largest logits in
    [ks[m], ke[m]) in no particular order, or 0..len-1 then -1 when the row has len <= K
    (``top_k_per_row_prefill`` writes the same sets as absolute columns). The
    logits are bit-identical to ``fp8_mqa_logits``', so the selection is a top-K of that
    kernel's output (exact ties at the K-th value broken arbitrarily).

    TotalCuCount  CU count of the scan's grid model (256 on MI355X).
    fallback_rows optional [1] i32, incremented by the number of rows that took the
                  exact fallback (diagnostics).
    force_fallback test hook: no candidates, so every row of more than K keys takes the
                  exact fallback.

    Workspace (allocated per call): about M * (N / 8 + 32 KiB * SplitN) + 512 * N bytes.
    """
