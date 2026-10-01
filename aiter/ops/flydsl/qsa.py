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
``flydsl``, or ``triton``. The default is ``auto``: FlyDSL for the query
shapes whose end-to-end layer was measured to beat live AMD, and Triton
for every other shape. A GQA query missing from that table logs once.
``triton`` is live AMD paged MQA, HIP top-k, expand+tail, and sparse GQA.
``flydsl`` runs K1, the same vendored expand+tail, and K2. Sigmoid and
partial RoPE stay outside the layer.
"""

import functools

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
from .kernels.qsa.arch import qsa_arch_is_supported
from .kernels.qsa.k1 import qsa_k1_block_ids, qsa_k1_serves
from .kernels.qsa.k2 import qsa_k2, qsa_k2_serves

# GQA query ``(n_q_heads, head_dim)`` -> the indexer head counts swept with
# it. K2 serves any structurally valid shape, so this table is the only
# thing keeping auto off an untuned one. The measured thing is the pair.
_MEASURED_QUERIES = {
    (24, 256): (4, 8),  # Flash-Next / qwen4_exp, TP1
    (12, 256): (4, 8),  # Flash-Next TP2: 12 query heads, 1 KV head
    (6, 256): (4, 8),  # Flash-Next TP4: 6 query heads, 1 KV head
    (3, 256): (4, 8),  # Flash-Next TP8: 3 query heads, 1 KV head
    (10, 128): (4, 8),
}
_BACKENDS = ("auto", "flydsl", "triton")

# GPU 6 / gfx950, cold ``--rotate 0``, page_size 16, M in {1, 2, 3, 4, 6, 8,
# 12, 16, 32, 64, 128, 256, 512} crossed with L in {512, 2048, 8192, 32768}.
# All 104 rows beat live AMD with err=0: 24x256 by 1.03x to 1.67x, 10x128 by
# 1.17x to 2.66x, and 10x128 also won at both indexer widths. The same grid
# with an 8-head indexer on 24x256 also won all 52 rows, err=0, by 1.08x to
# 1.70x. Indexer width does not change the K2 ladder. That reaches
# every launch config the K2 policy can pick, and M past 512 reuses M=512's
# BN32 single-split config with a larger grid, so neither M nor the
# selection width filters this gate. The ``_launch_config`` decode bands are
# now fitted at both head widths. A per-rank shard with a different KV-head
# count is still its own grid, because the K2 launch is
# ``(rows, kv_heads, splits)``. GPU 7 / gfx942, cold ``rotate=0``, page size
# 16, the same M x L grid: q ``[M, 12, 256]`` over 1 KV head and the
# replicated 4-head indexer. All 52 rows beat live AMD with err=0, by 1.73x
# to 6.06x. The same grid with an 8-head indexer also won all 52 rows,
# FlyDSL err=0, by 1.46x to 4.86x. GPU 6 / gfx950, the same cold grid:
# q ``[M, 6, 256]`` over 1 KV head (Flash-Next TP4, group 6). The 4-head
# indexer won all 52 rows with err=0, by 1.28x to 3.29x. The 8-head indexer
# also won all 52, FlyDSL err=0, by 1.38x to 3.28x. The same gfx950 grid
# with q ``[M, 3, 256]`` over 1 KV head (Flash-Next TP8, group 3) won all
# 52 rows at both indexer widths, FlyDSL err=0: 4-head by 1.21x to 3.29x,
# 8-head by 1.27x to 3.34x.

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

    ``None`` is ``auto``.
    """
    if backend is None:
        return "auto"
    normalized = str(backend).lower()
    if normalized not in _BACKENDS:
        raise ValueError(
            f"backend must be one of: {', '.join(_BACKENDS)}, got {backend!r}"
        )
    return normalized


def _measured_heads(
    q_indexer: torch.Tensor, q_gqa: torch.Tensor
) -> tuple[int, ...] | None:
    """Indexer head counts swept with this GQA query, or None if untuned.

    Narrowing the head count is left to ``qsa_k1_serves`` below, which
    checks it along with the indexer dtype and D. What only this level can
    check is the GQA query shape, since K2 serves any structurally valid
    one, and that the two halves of the layer agree on ``M``.
    """
    if q_indexer.dim() != 3 or q_indexer.shape[0] != q_gqa.shape[0]:
        return None
    return _MEASURED_QUERIES.get(tuple(q_gqa.shape[1:]))


def _gqa_table_key(q_gqa: torch.Tensor) -> tuple[int, ...] | None:
    """Per-rank GQA query ``(n_q_heads, head_dim)``, or None if not a query."""
    if q_gqa.dim() != 3:
        return None
    return tuple(int(s) for s in q_gqa.shape[1:])


@functools.cache
def _log_unmeasured_gqa_query(shape: tuple[int, ...]) -> None:
    """Log one table miss. The same shape does not log again."""
    from aiter import logger

    logger.warning(
        "QSA auto: GQA query shape %s is not in the measured table; using Triton",
        shape,
    )


def _gqa_device_arch(q_gqa: torch.Tensor) -> str | None:
    """``gcnArchName`` of ``q_gqa``'s GPU, or None when it is not on a GPU.

    This is a host query of the device name. It does not sync.
    """
    if not q_gqa.is_cuda:
        return None
    return torch.cuda.get_device_properties(q_gqa.device).gcnArchName


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
    """Whether ``auto`` may launch FlyDSL.

    Any query pair but a measured one stays on Triton. So does a shape the
    kernels cannot serve, and a GPU outside gfx942/gfx950. Those keep
    ``auto`` from turning a dispatch miss into an exception. An explicit
    ``flydsl`` backend does not consult this predicate, so it still raises
    on an unsupported arch. Host tensors have no arch and are judged on
    shape alone. ``M`` and the selection width are not filters: the sweep
    above won at every one it measured.
    """
    arch = _gqa_device_arch(q_gqa)
    if arch is not None and not qsa_arch_is_supported(arch):
        return False
    heads = _measured_heads(q_indexer, q_gqa)
    if heads is None:
        return False
    if qsa_k1_serves(q_indexer, index_k_cache, index_page_table, heads) is not None:
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

    ``backend="auto"`` (the default) uses FlyDSL only when
    ``qsa_auto_uses_flydsl`` is set, and Triton otherwise. A GQA query
    that is not in the measured table logs once on this path. A GPU
    outside gfx942/gfx950 also stays on Triton. ``backend="flydsl"`` is
    K1 + vendored expand + K2, and that override still raises on an
    unsupported arch.
    ``backend="triton"`` is the live AMD path.
    """
    selected = normalize_qsa_backend(backend)
    if selected == "auto":
        key = _gqa_table_key(q_gqa)
        if key is None or key not in _MEASURED_QUERIES:
            missed = key if key is not None else tuple(int(s) for s in q_gqa.shape)
            _log_unmeasured_gqa_query(missed)
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
