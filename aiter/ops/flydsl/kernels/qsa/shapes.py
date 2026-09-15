# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""QSA family A / B tensor shapes shared by the oracle, tests, and kernels.

Family A is Flash-Next production (must win). Family B is #4882 Gluon-parity
only -- Gluon sparse GQA does not auto-dispatch on family A (group 12, D=256).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class QsaIndexerSpec:
    """Paged mean-pooled indexer (ReLU-sum over heads, no per-head ``w_h``)."""

    n_heads: int
    kv_heads: int
    head_dim: int
    compress_ratio: int
    block_budget: int
    dtype: torch.dtype = torch.bfloat16

    @property
    def token_budget(self) -> int:
        return self.block_budget * self.compress_ratio

    @property
    def index_width(self) -> int:
        # 512 blocks * 4 tokens + incomplete tail of 0..3.
        return self.token_budget + self.compress_ratio - 1


@dataclass(frozen=True)
class QsaGqaSpec:
    """Sparse GQA over uncompressed paged K/V at the expanded token ids."""

    n_heads: int
    kv_heads: int
    head_dim: int
    rope_dim: int
    dtype: torch.dtype = torch.bfloat16
    # Sigmoid output gate is fused in K2 later; the oracle attends, then
    # callers may multiply ``o * sigmoid(gate)`` outside.
    sigmoid_gate: bool = True

    @property
    def group_size(self) -> int:
        return self.n_heads // self.kv_heads


# Flash-Next / Qwen3.8-Flash-Next / qwen4_exp (SILOTIGER-1047 family A).
# Indexer in:  q [M, 4, 128] BF16, paged compressed k (1 KV head, D=128).
# Indexer out: block_ids [M, 512], token indices [M, <=2051].
# GQA in:      q [M, 24, 256] BF16, paged k/v [..., 2, 256], indices.
# GQA out:     o [M, 24, 256] BF16 (pre-o_proj).
FAMILY_A_INDEXER = QsaIndexerSpec(
    n_heads=4,
    kv_heads=1,
    head_dim=128,
    compress_ratio=4,
    block_budget=512,
)
FAMILY_A_GQA = QsaGqaSpec(
    n_heads=24,
    kv_heads=2,
    head_dim=256,
    rope_dim=64,
)

# #4882 Gluon-validated shapes (family B). Indexer H is 4 or 8, D=128.
# Sparse GQA: D=128, group size 5, selection_width=2051 (Hq=10, Hk=2).
FAMILY_B_INDEXER = QsaIndexerSpec(
    n_heads=4,
    kv_heads=1,
    head_dim=128,
    compress_ratio=4,
    block_budget=512,
)
FAMILY_B_INDEXER_H8 = QsaIndexerSpec(
    n_heads=8,
    kv_heads=1,
    head_dim=128,
    compress_ratio=4,
    block_budget=512,
)
FAMILY_B_GQA = QsaGqaSpec(
    n_heads=10,
    kv_heads=2,
    head_dim=128,
    rope_dim=0,
    sigmoid_gate=False,
)

# Optional serving scale applied after the ReLU-sum. A positive constant
# cannot change top-k argmax (ties stay ties).
FAMILY_A_SCORE_SCALE = FAMILY_A_INDEXER.head_dim**-0.5  # 1/sqrt(128)
