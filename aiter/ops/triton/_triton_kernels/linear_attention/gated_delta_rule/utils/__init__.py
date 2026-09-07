# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
# Adapted from flash-linear-attention: Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

from .cumsum import (
    chunk_local_cumsum,
    chunk_local_cumsum_scalar,
    chunk_local_cumsum_vector,
)
from .index import (
    prepare_chunk_indices,
    prepare_chunk_offsets,
    prepare_num_chunks,
    prepare_rebased_cu_seqlens,
)
from .l2norm import l2norm_bwd, l2norm_fwd
from .prefill_metadata import (
    GatedDeltaRulePrefillMetadata,
    build_gated_delta_rule_prefill_metadata,
)
from .solve_tril import solve_tril
from .wy_representation import chunk_scaled_dot_kkt_fwd, recompute_w_u_fwd

__all__ = [
    "GatedDeltaRulePrefillMetadata",
    "build_gated_delta_rule_prefill_metadata",
    "chunk_local_cumsum",
    "chunk_local_cumsum_scalar",
    "chunk_local_cumsum_vector",
    "chunk_scaled_dot_kkt_fwd",
    "l2norm_bwd",
    "l2norm_fwd",
    "prepare_chunk_indices",
    "prepare_chunk_offsets",
    "prepare_num_chunks",
    "prepare_rebased_cu_seqlens",
    "recompute_w_u_fwd",
    "solve_tril",
]
