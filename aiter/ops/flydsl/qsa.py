# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL QSA public surface (SILOTIGER-1047).

``qsa_k1_block_ids`` writes indexer ``block_ids [M, 512]`` from paged
compressed K. Rows up to 512 blocks use the fused emit kernel; longer rows
use tiled BLOCK_N=32 BF16 MFMA scoring plus the stable decode radix below
32768 columns and streaming radix at or above that width. Single-request
prefill scores 16 query rows per workgroup. The indexer head count is 4 or
8, each a separate compile.

``qsa_k2`` writes sparse GQA ``o [M, Hq, D]`` from paged K/V at the selected
token ids. Decode runs a BLOCK_N=16 two-wave tile with split-K and an LSE
merge; prefill runs BLOCK_N=32 two waves and writes output directly once the
grid alone fills the machine. Softmax is online in log2 space and the split
partials are FP32. Expand+tail and the sigmoid gate stay unfused.

``qsa_oracle`` is the fp32 reference. The concrete shapes all of these are
validated against are test fixtures in ``op_tests/qsa_shapes.py``.

``qsa_layer`` is the ``qwen4_exp`` opt-in. ``backend`` is ``auto``,
``flydsl``, or ``triton``. The default is ``triton``: live AMD paged MQA,
HIP top-k, expand+tail, and sparse GQA. Calling the wrapper without a
backend leaves that path as it is. ``flydsl`` runs K1, the same vendored
expand+tail, and K2. ``auto`` launches FlyDSL only for the one query shape
whose end-to-end layer was measured to beat live AMD; every other shape
stays on Triton. Sigmoid and partial RoPE stay outside the layer.
"""

import torch

from .kernels.qsa import (
    QsaGqaSpec,
    QsaIndexerSpec,
    QsaOracleResult,
    gather_paged_cache,
    gather_qsa_caches,
    pack_paged_cache,
    qsa_expand_tail,
    qsa_indexer_scores,
    qsa_oracle,
    qsa_sparse_gqa,
    qsa_topk_blocks,
    qsa_visible_blocks,
)
from .kernels.qsa.k1 import qsa_k1_block_ids, qsa_k1_serves
from .kernels.qsa.k2 import qsa_k2, qsa_k2_serves

# The Flash-Next / qwen4_exp GQA query, as ``(n_q_heads, head_dim)``. K2
# serves any structurally valid shape but its launch policy is fitted to
# this one, so auto pins it here; K1's ``heads=(4,)`` gate pins the indexer.
_MEASURED_GQA_QUERY = (24, 256)
_BACKENDS = ("auto", "flydsl", "triton")

# GPU 6 / gfx950, cold ``--rotate 0``, page_size 16. Every swept row beat
# live AMD with err=0: M in {1, 8, 512} and L in {512, 2048, 8192, 32768},
# both the short emit rows and the long score rows. M strictly between 8
# and 512 was not in that sweep, so auto leaves it on Triton. n_columns is
# not a filter: every measured width won.

__all__ = [
    "QsaGqaSpec",
    "QsaIndexerSpec",
    "QsaOracleResult",
    "gather_paged_cache",
    "gather_qsa_caches",
    "normalize_qsa_backend",
    "pack_paged_cache",
    "qsa_auto_uses_flydsl",
    "qsa_expand_tail",
    "qsa_indexer_scores",
    "qsa_k1_block_ids",
    "qsa_k2",
    "qsa_layer",
    "qsa_oracle",
    "qsa_sparse_gqa",
    "qsa_topk_blocks",
    "qsa_visible_blocks",
]


def normalize_qsa_backend(backend: str | None) -> str:
    """Map a ``qsa_layer`` backend to ``auto``, ``flydsl``, or ``triton``.

    ``None`` is ``triton``, matching the live AMD path.
    """
    if backend is None:
        return "triton"
    normalized = str(backend).lower()
    if normalized not in _BACKENDS:
        raise ValueError(
            f"backend must be one of: {', '.join(_BACKENDS)}, got {backend!r}"
        )
    return normalized


def _measured_queries(q_indexer: torch.Tensor, q_gqa: torch.Tensor) -> bool:
    """Whether both query tensors are the shape auto was measured on.

    Only the GQA query is pinned here. The ``heads=(4,)`` K1 gate below
    already restricts the indexer to bfloat16 ``[M, 4, 128]``, and the K2
    gate re-checks the GQA dtype, but nothing downstream pins the GQA
    shape or makes the two halves agree on ``M``.
    """
    return (
        q_indexer.dim() == 3
        and tuple(q_gqa.shape[1:]) == _MEASURED_GQA_QUERY
        and q_indexer.shape[0] == q_gqa.shape[0]
    )


def _measured_rows(rows: int, n_columns: int) -> bool:
    """Whether this row count is inside the recorded end-to-end win.

    ``n_columns`` is accepted so the predicate stays host-side and explicit.
    The sweep won at every measured width, so it does not exclude a row.
    """
    del n_columns
    return rows <= 8 or rows >= 512


def qsa_auto_uses_flydsl(
    q_indexer: torch.Tensor,
    index_k_cache: torch.Tensor,
    index_page_table: torch.Tensor,
    q_gqa: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    kv_page_table: torch.Tensor,
    indices: torch.Tensor,
) -> bool:
    """Whether ``auto`` may launch FlyDSL. Host shapes only; no device sync.

    Any query shape but the measured one stays on Triton, as does the
    measured one when ``M`` falls between the swept decode band
    (``M <= 8``) and prefill band (``M >= 512``). A shape the kernels
    cannot serve stays on Triton too, so ``auto`` does not turn a dispatch
    miss into an exception.
    """
    if not _measured_queries(q_indexer, q_gqa):
        return False
    if index_k_cache.dim() != 4 or index_page_table.dim() != 2:
        return False
    n_columns = int(index_page_table.shape[1]) * int(index_k_cache.shape[1])
    if not _measured_rows(int(q_indexer.shape[0]), n_columns):
        return False
    if (
        qsa_k1_serves(q_indexer, index_k_cache, index_page_table, heads=(4,))
        is not None
    ):
        return False
    return qsa_k2_serves(q_gqa, k_cache, v_cache, indices, kv_page_table) is None


def _expand_block_ids(
    block_ids: torch.Tensor,
    query_positions: torch.Tensor,
    context_lens: torch.Tensor,
    token_to_req: torch.Tensor,
    compress_ratio: int,
    token_topk: int,
    indices: torch.Tensor | None,
) -> torch.Tensor:
    # Phase 5 owns fusing expand into K1 or K2. Until then this is the same
    # Triton kernel the live AMD path launches.
    from aiter.ops.triton.attention.qsa_vllm_amd import expand_qsa_block_indices_cuda

    return expand_qsa_block_indices_cuda(
        block_ids,
        query_positions,
        context_lens,
        token_to_req,
        compress_ratio,
        token_topk,
        out=indices,
    )


def _qsa_layer_flydsl(
    q_indexer: torch.Tensor,
    index_k_cache: torch.Tensor,
    index_page_table: torch.Tensor,
    q_gqa: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    kv_page_table: torch.Tensor,
    token_to_req: torch.Tensor,
    query_positions: torch.Tensor,
    context_lens: torch.Tensor,
    indices: torch.Tensor | None,
    block_ids: torch.Tensor | None,
    out: torch.Tensor | None,
    token_topk: int,
    compress_ratio: int,
    score_scale: float | None,
    softmax_scale: float | None,
) -> torch.Tensor:
    if score_scale is None:
        score_scale = float(q_indexer.shape[2]) ** -0.5
    block_ids = qsa_k1_block_ids(
        q_indexer,
        index_k_cache,
        index_page_table,
        token_to_req,
        query_positions,
        context_lens,
        out=block_ids,
        score_scale=score_scale,
        heads=(int(q_indexer.shape[1]),),
    )
    indices = _expand_block_ids(
        block_ids,
        query_positions,
        context_lens,
        token_to_req,
        compress_ratio,
        token_topk,
        indices,
    )
    return qsa_k2(
        q_gqa,
        k_cache,
        v_cache,
        indices,
        kv_page_table,
        token_to_req,
        out=out,
        softmax_scale=softmax_scale,
    )


def _qsa_layer_triton(
    q_indexer: torch.Tensor,
    index_k_cache: torch.Tensor,
    index_page_table: torch.Tensor,
    q_gqa: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    kv_page_table: torch.Tensor,
    token_to_req: torch.Tensor,
    query_positions: torch.Tensor,
    context_lens: torch.Tensor,
    indices: torch.Tensor | None,
    out: torch.Tensor | None,
    token_topk: int,
    compress_ratio: int,
) -> torch.Tensor:
    from aiter.ops.triton.attention.qsa_vllm_amd import (
        qsa_select_paged_tokens,
        qsa_sparse_paged_attention,
    )

    indices, _block_ids = qsa_select_paged_tokens(
        q_indexer,
        index_k_cache,
        index_page_table,
        token_to_req,
        query_positions,
        context_lens,
        token_topk,
        compress_ratio,
        out=indices,
    )
    return qsa_sparse_paged_attention(
        q_gqa,
        k_cache,
        v_cache,
        indices,
        kv_page_table,
        token_to_req,
        out=out,
    )


def qsa_layer(
    q_indexer: torch.Tensor,
    index_k_cache: torch.Tensor,
    index_page_table: torch.Tensor,
    q_gqa: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    kv_page_table: torch.Tensor,
    token_to_req: torch.Tensor,
    query_positions: torch.Tensor,
    context_lens: torch.Tensor,
    indices: torch.Tensor | None = None,
    block_ids: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
    token_topk: int = 2048,
    compress_ratio: int = 4,
    score_scale: float | None = None,
    softmax_scale: float | None = None,
    backend: str | None = None,
) -> torch.Tensor:
    """Run one QSA layer: indexer select, expand+tail, sparse GQA.

    ``backend="triton"`` (the default) is the live AMD path.
    ``backend="flydsl"`` is K1 + vendored expand + K2.
    ``backend="auto"`` uses FlyDSL only when ``qsa_auto_uses_flydsl`` is set.
    """
    selected = normalize_qsa_backend(backend)
    if selected == "auto":
        width = token_topk + compress_ratio - 1
        rows = q_indexer.shape[0]
        probe = indices
        if probe is None:
            probe = torch.empty(rows, width, dtype=torch.int32, device=q_indexer.device)
        selected = (
            "flydsl"
            if qsa_auto_uses_flydsl(
                q_indexer,
                index_k_cache,
                index_page_table,
                q_gqa,
                k_cache,
                v_cache,
                kv_page_table,
                probe,
            )
            else "triton"
        )
    if selected == "flydsl":
        return _qsa_layer_flydsl(
            q_indexer,
            index_k_cache,
            index_page_table,
            q_gqa,
            k_cache,
            v_cache,
            kv_page_table,
            token_to_req,
            query_positions,
            context_lens,
            indices,
            block_ids,
            out,
            token_topk,
            compress_ratio,
            score_scale,
            softmax_scale,
        )
    return _qsa_layer_triton(
        q_indexer,
        index_k_cache,
        index_page_table,
        q_gqa,
        k_cache,
        v_cache,
        kv_page_table,
        token_to_req,
        query_positions,
        context_lens,
        indices,
        out,
        token_topk,
        compress_ratio,
    )
