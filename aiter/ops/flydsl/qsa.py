# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL QSA public surface (SILOTIGER-1047).

Phase 0: fp32 oracle + shared family A/B shapes.
Phase 2a: family A FlyDSL K1 writes ``block_ids [M, 512]`` from paged
compressed K (no score matrix). Expand+tail and K2 are still separate.

Shapes (flattened tokens ``M``; activations BF16 unless noted):

Family A -- Flash-Next production
    Indexer: ``q [M, 4, 128]``, paged compressed ``k`` (1 KV head, D=128),
    ``r=4``, top-512 blocks, expand+tail width 2051.
    GQA: ``q [M, 24, 256]``, paged ``k/v [..., 2, 256]``, group 12, partial
    RoPE 64, sigmoid gate (unfused in the oracle).

Family B -- #4882 Gluon-parity
    Indexer: ``H`` 4 or 8, ``D=128``.
    GQA: ``q [M, 10, 128]``, 2 KV heads (group 5), selection width 2051.
"""

from .kernels.qsa import (
    FAMILY_A_GQA,
    FAMILY_A_INDEXER,
    FAMILY_A_SCORE_SCALE,
    FAMILY_B_GQA,
    FAMILY_B_INDEXER,
    FAMILY_B_INDEXER_H8,
    QsaGqaSpec,
    QsaIndexerSpec,
    QsaOracleResult,
    gather_paged_cache,
    gather_qsa_family_a_caches,
    pack_paged_cache,
    qsa_expand_tail,
    qsa_indexer_scores,
    qsa_oracle,
    qsa_sparse_gqa,
    qsa_topk_blocks,
    qsa_visible_blocks,
)
from .kernels.qsa.k1_family_a import qsa_k1_family_a_block_ids

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
    "qsa_k1_family_a_block_ids",
    "qsa_oracle",
    "qsa_sparse_gqa",
    "qsa_topk_blocks",
    "qsa_visible_blocks",
]
