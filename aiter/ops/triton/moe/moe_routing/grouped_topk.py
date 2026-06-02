from __future__ import annotations

import torch
import triton
import triton.language as tl

from aiter.ops.triton.moe.moe_routing.bitmatrix import Bitmatrix
from aiter.ops.triton._triton_kernels.moe.moe_routing.grouped_topk import _grouped_topk

# SPDX-License-Identifier: MIT
"""Single-fused Triton grouped-top-k routing kernel.

Drop-in replacement for the ``topk(...)`` call inside
``aiter/ops/triton/moe/moe_routing/routing.py::routing_a8w4`` (lines 338-347).
Same return contract — ``(y_vals, y_indx, Bitmatrix)`` — so downstream
``sort_tokens`` / ``sort_tokens_fused`` consume the output unchanged.

Algorithm (single kernel launch, mirrors the structure of aiter's ``_topk``
and ``_hash_routing`` in ``_triton_kernels/moe/moe_routing/topk.py``):

  1. Memset bitmatrix scratchpad / partials (same lane-borrowing trick as
     ``_topk``: the first ``s_blocks + sp_blocks`` programs do nothing but
     zero-fill).
  2. Load the row of router logits.
  3. Apply ``score_mode`` per element ('softmax' / 'sigmoid' / 'sqrtsoftplus' /
     'none').
  4. Per-group score reduction over an *arbitrary* expert→group mapping
     (``ExpertGroup`` int32 table):
       - HAS_BIAS → top-2 sum on bias-augmented scores (DeepSeek-V3 rule;
         mirrors ``biased_grouped_topk_torch``).
       - else    → per-group max (DeepSeek-V2 rule; mirrors
         ``grouped_topk_torch``).
  5. Pick top ``TOPK_GROUP`` groups via repeated argmax (NUM_EXPERT_GROUP is
     small, so the unrolled loop is tiny).
  6. Mask experts in non-selected groups to ``-inf`` on the bias-augmented
     scores, then do per-expert top-``N_EXPTS_ACT`` via repeated argmax.
  7. Gather *unbiased* weights at the selected indices (matches the
     ``noaux_tc`` semantics — bias used for selection only, weights from the
     untouched score).
  8. Optional renorm + ``routed_scaling_factor`` scale.
  9. Pack selected indices into the (n_cols_words, n_rows_pad32).T uint32
     bitmatrix layout the kernel emits, identical to ``_topk``.

Constraints (DeepSeek-class envelope):
  - n_expts_tot ≤ 256 (single ``BLOCK_N`` pass; no streaming loop).
  - num_expert_group ≤ 16.
  - topk_group ≤ num_expert_group.
  - n_expts_act (top_k) ≤ 16.
  - BLOCK_M = 1 (the per-group 3-D intermediate is
    ``[BLOCK_M, BLOCK_N, NUM_EXPERT_GROUP]`` fp32 — at BLOCK_M=1 that's
    ≤ 256 * 16 * 4 = 16 KiB, fits in registers / LDS comfortably).
"""

def grouped_topk(
    x: torch.Tensor,
    k: int,
    num_expert_group: int,
    topk_group: int,
    *,
    expert_group: torch.Tensor | None = None,
    apply_softmax: bool = False,        # accepted for parity with topk(); ignored
    HIST_BLOCK_M: int = 32,
    score_mode: str = "softmax",
    bias: torch.Tensor | None = None,
    renorm: bool = False,
    routed_scaling_factor: float = 1.0,
    num_fused_shared_experts: int = 0,
    shared_experts_score: float = 1.0,
):
    """Triton grouped top-k expert selection. See module docstring.

    Returns ``(y_vals, y_indx, bitmatrix)`` matching the contract of
    ``aiter.ops.triton.moe.moe_routing.topk.topk``:

      - y_vals: ``(n_rows, k + num_fused_shared_experts)`` in ``x.dtype``.
      - y_indx: ``(n_rows, k + num_fused_shared_experts)`` ``int16``.

    When ``num_fused_shared_experts > 0`` the routed top-k selection occupies
    the first ``k`` columns and the always-on shared expert(s) occupy the next
    ``num_fused_shared_experts`` columns — expert id ``n_cols + i``, weight
    ``shared_experts_score`` (appended after the routed renorm, mirroring
    ``init_aiter_topK_meta_data`` / ``rocm_aiter_grouped_topk``). The bitmatrix
    is widened to ``n_cols + num_fused_shared_experts`` columns so ``sort_tokens``
    counts the shared bucket.

      - bitmatrix: real :class:`Bitmatrix`; same uint32
        ``(n_cols_words, n_rows_pad32).T`` storage / scratchpad layout the
        ``_topk`` kernel emits, so ``sort_tokens`` and ``sort_tokens_fused``
        consume it unchanged.
    """
    assert x.dim() == 2
    n_rows, n_cols = x.shape
    assert n_cols <= 256, (
        f"DeepSeek-class envelope: n_expts_tot ({n_cols}) must be <= 256"
    )
    # Fused shared experts are appended (always-on) AFTER the routed selection;
    # they occupy expert ids [n_cols, n_cols + num_fused_shared_experts).
    n_shared = num_fused_shared_experts
    assert n_shared >= 0
    n_total = n_cols + n_shared          # experts incl. shared (bitmatrix width)
    k_out = k + n_shared                 # output width (routed top-k + shared)
    assert num_expert_group > 1
    assert num_expert_group <= 16, (
        f"NUM_EXPERT_GROUP ({num_expert_group}) > 16 not supported"
    )
    assert 0 < topk_group <= num_expert_group
    assert 0 < k <= 16
    assert score_mode in ("softmax", "sigmoid", "sqrtsoftplus", "none"), (
        f"unknown score_mode {score_mode!r}"
    )
    has_bias = bias is not None
    if has_bias:
        assert bias.dim() == 1 and bias.shape[0] == n_cols
        assert bias.dtype == torch.float32
        assert score_mode in ("sqrtsoftplus", "sigmoid"), (
            "bias only supported with sqrtsoftplus / sigmoid"
        )

    dev = x.device

    # Default expert→group mapping = contiguous DeepSeek layout.
    if expert_group is None:
        assert n_cols % num_expert_group == 0, (
            f"n_expts_tot ({n_cols}) not divisible by num_expert_group "
            f"({num_expert_group}); pass an explicit expert_group table."
        )
        g_size = n_cols // num_expert_group
        expert_group = (
            torch.arange(n_cols, device=dev, dtype=torch.int32) // g_size
        ).to(torch.int32)
    else:
        assert expert_group.dim() == 1 and expert_group.shape[0] == n_cols
        assert expert_group.dtype == torch.int32

    # Block sizes — single BLOCK_N pass for DeepSeek envelope. BLOCK_N must
    # cover the shared-expert columns too so their bits fit in the bitmatrix.
    BLOCK_M = 1
    BLOCK_N = max(32, triton.next_power_of_2(n_total))
    N_EXPTS_PAD = BLOCK_N
    # Mirror topk(): pad to ≥ 2 to dodge tl.argmax/topk(k=1) compile quirks.
    N_EXPTS_ACT_PAD = max(2, triton.next_power_of_2(k_out))
    BLOCK_S = 128
    BLOCK_SP = 128
    TILE_SIZE = 8

    # Outputs (same shapes / dtypes as topk(...)), widened by the shared slots.
    y_vals = torch.empty((n_rows, k_out), dtype=x.dtype, device=dev)
    y_indx = torch.empty((n_rows, k_out), dtype=torch.int16, device=dev)

    # Bitmatrix in transposed-uint32 storage layout (identical to topk()).
    n_cols_pad = triton.cdiv(n_total, BLOCK_N) * BLOCK_N
    n_cols_words = n_cols_pad // 32
    bitmatrix_data = torch.empty(
        (n_cols_words, triton.cdiv(n_rows, 32) * 32),
        dtype=torch.uint32,
        device=dev,
    )
    bitmatrix_data = torch.transpose(bitmatrix_data, 0, 1)[:n_rows]

    # Scratchpads. The per-column sum buffer consumed by Bitmatrix.sum() /
    # sort_tokens must cover the full padded column count (n_cols_pad), which
    # widens with the shared experts; sizing by n_total alone can under-allocate
    # (e.g. n_total=257 -> n_cols_pad=512 but cdiv(257,128)*128=384).
    s_blocks = triton.cdiv(n_cols_pad, BLOCK_S)
    s_cols = s_blocks * BLOCK_S
    scratchpad = torch.empty((s_cols,), dtype=torch.int32, device=dev)
    BLOCK_MM = HIST_BLOCK_M * TILE_SIZE
    pids_x = triton.cdiv(n_rows, BLOCK_MM)
    scratchpad_partials = torch.empty(
        (n_cols_pad, pids_x * TILE_SIZE), dtype=torch.int32, device=dev
    )
    scratchpad_partials = torch.transpose(scratchpad_partials, 0, 1)
    sp_size = scratchpad_partials.numel()
    sp_blocks = triton.cdiv(sp_size, BLOCK_SP)

    pids = max(triton.cdiv(n_rows, BLOCK_M), s_blocks + sp_blocks)

    _grouped_topk[(pids,)](
        x,
        x.stride(0),
        expert_group,
        y_vals,
        y_indx,
        y_vals.stride(0),
        bitmatrix_data,
        bitmatrix_data.stride(0),
        bitmatrix_data.stride(1),
        n_rows,
        n_cols,
        scratchpad,
        BLOCK_S,
        s_blocks,
        scratchpad_partials,
        BLOCK_SP,
        sp_blocks,
        sp_size,
        BLOCK_M=BLOCK_M,
        N_EXPTS_PAD=N_EXPTS_PAD,
        BLOCK_N=BLOCK_N,
        N_EXPTS_ACT=k,
        N_EXPTS_ACT_PAD=N_EXPTS_ACT_PAD,
        NUM_EXPERT_GROUP=num_expert_group,
        TOPK_GROUP=topk_group,
        Bias=bias,
        SCORE_MODE=score_mode,
        HAS_BIAS=has_bias,
        APPLY_RENORM=renorm,
        ROUTED_SCALING=routed_scaling_factor,
        N_SHARED=n_shared,
        SHARED_SCORE=shared_experts_score,
        num_warps=4,
    )

    bitmatrix = Bitmatrix(
        bitmatrix_data,
        shape=[n_rows, n_cols_words * 32],
        scratchpad=scratchpad,
        scratchpad_partials=scratchpad_partials,
    )
    return y_vals, y_indx, bitmatrix
