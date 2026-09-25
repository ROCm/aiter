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
"""

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
from .kernels.qsa.k1 import qsa_k1_block_ids
from .kernels.qsa.k2 import qsa_k2

__all__ = [
    "QsaGqaSpec",
    "QsaIndexerSpec",
    "QsaOracleResult",
    "gather_paged_cache",
    "gather_qsa_caches",
    "pack_paged_cache",
    "qsa_expand_tail",
    "qsa_indexer_scores",
    "qsa_k1_block_ids",
    "qsa_k2",
    "qsa_oracle",
    "qsa_sparse_gqa",
    "qsa_topk_blocks",
    "qsa_visible_blocks",
]
