# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

from torch import Tensor

from ..jit.core import compile_ops


@compile_ops(
    "module_minimax_m3_index_score",
    fc_name="minimax_m3_decode_index_score",
    develop=True,
)
def minimax_m3_decode_index_score(
    idx_q: Tensor,
    index_kv_cache: Tensor,
    score: Tensor,
    block_table: Tensor,
    seq_lens: Tensor,
    total_q: int,
    head_dim: int,
    init_blocks: int,
    local_blocks: int,
    sm_scale: float,
    max_query_len: int,
    block_size: int,
    num_kv_chunks: int,
) -> None:
    pass
