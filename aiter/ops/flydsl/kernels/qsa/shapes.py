# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""QSA tensor shape descriptors shared by the oracle, tests, and kernels.

Only the descriptors live here. The concrete shapes the kernels are
validated against are test fixtures and live in ``op_tests/qsa_shapes.py``;
the kernels carry their own structural constants rather than importing one
model's numbers.
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
