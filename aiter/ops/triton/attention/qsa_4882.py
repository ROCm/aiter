# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Public import for pinned AITER #4882 QSA (Triton + gfx950 Gluon)."""

from aiter.ops.triton._triton_kernels.attention.qsa_4882 import (
    AITER_4882_QSA_PIN,
    gluon_qsa_available,
    qsa_expand_block_indices,
    qsa_paged_mqa_logits,
    qsa_select_paged_tokens,
    qsa_sparse_paged_gqa,
)

__all__ = [
    "AITER_4882_QSA_PIN",
    "gluon_qsa_available",
    "qsa_expand_block_indices",
    "qsa_paged_mqa_logits",
    "qsa_select_paged_tokens",
    "qsa_sparse_paged_gqa",
]
