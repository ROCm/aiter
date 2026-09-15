# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL kernels for Qwen sparse attention (QSA) decode on gfx950.

  ``sparse_decode_attn.py``    one CTA per (row, kv head, KV split), reading the
                               selected KV straight from the token pool
  ``sparse_decode_reduce.py``  merges the KV-split partials

The public entry point is ``aiter.ops.flydsl.qsa_sparse_decode``.
"""
from .oracle import (
    QsaOracleResult,
    qsa_expand_tail,
    qsa_indexer_scores,
    qsa_oracle,
    qsa_sparse_gqa,
    qsa_topk_blocks,
    qsa_visible_blocks,
)
from .paged import (
    gather_paged_cache,
    gather_qsa_family_a_caches,
    pack_paged_cache,
)
from .shapes import (
    FAMILY_A_GQA,
    FAMILY_A_INDEXER,
    FAMILY_A_SCORE_SCALE,
    FAMILY_B_GQA,
    FAMILY_B_INDEXER,
    FAMILY_B_INDEXER_H8,
    QsaGqaSpec,
    QsaIndexerSpec,
)

__all__ = [
    "FAMILY_A_GQA",
    "FAMILY_A_INDEXER",
    "FAMILY_A_SCORE_SCALE",
    "FAMILY_B_GQA",
    "FAMILY_B_INDEXER",
    "FAMILY_B_INDEXER_H8",
    "QsaGqaSpec",
    "QsaIndexerSpec",
    "QsaOracleResult",
    "gather_paged_cache",
    "gather_qsa_family_a_caches",
    "pack_paged_cache",
    "qsa_expand_tail",
    "qsa_indexer_scores",
    "qsa_oracle",
    "qsa_sparse_gqa",
    "qsa_topk_blocks",
    "qsa_visible_blocks",
]
