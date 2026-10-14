# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Flash-Next and Gluon-parity QSA shapes the FlyDSL kernels are validated against.

Flash-Next is the production shape. Gluon-parity is the #4882 shape.
Gluon sparse GQA does not auto-dispatch on Flash-Next (group 12, D=256).

These are test fixtures. The kernels declare their own structural constants
so they do not depend on a model registry.
"""

from __future__ import annotations

from op_tests.qsa_ref import QsaGqaSpec, QsaIndexerSpec

# Flash-Next / Qwen3.8-Flash-Next / qwen4_exp (SILOTIGER-1047).
# Indexer in:  q [M, 4, 128] BF16, paged compressed k (1 KV head, D=128).
# Indexer out: block_ids [M, 512], token indices [M, <=2051].
# GQA in:      q [M, 24, 256] BF16, paged k/v [..., 2, 256], indices.
# GQA out:     o [M, 24, 256] BF16 (pre-o_proj).
FLASH_NEXT_INDEXER = QsaIndexerSpec(
    n_heads=4,
    kv_heads=1,
    head_dim=128,
    compress_ratio=4,
    block_budget=512,
)
FLASH_NEXT_GQA = QsaGqaSpec(
    n_heads=24,
    kv_heads=2,
    head_dim=256,
    rope_dim=64,
)

# #4882 Gluon-parity shapes. Indexer H is 4 or 8, D=128.
# Sparse GQA: D=128, group size 5, selection_width=2051 (Hq=10, Hk=2).
GLUON_PARITY_INDEXER = QsaIndexerSpec(
    n_heads=4,
    kv_heads=1,
    head_dim=128,
    compress_ratio=4,
    block_budget=512,
)
GLUON_PARITY_INDEXER_H8 = QsaIndexerSpec(
    n_heads=8,
    kv_heads=1,
    head_dim=128,
    compress_ratio=4,
    block_budget=512,
)
GLUON_PARITY_GQA = QsaGqaSpec(
    n_heads=10,
    kv_heads=2,
    head_dim=128,
    rope_dim=0,
    sigmoid_gate=False,
)

# Optional serving scale applied after the ReLU-sum. A positive constant
# cannot change top-k argmax (ties stay ties).
FLASH_NEXT_SCORE_SCALE = FLASH_NEXT_INDEXER.head_dim**-0.5  # 1/sqrt(128)
