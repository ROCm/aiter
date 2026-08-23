# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Sparse block selection for index-head attention."""

import torch

from csrc.cpp_itfs.torch_utils import direct_register_custom_op

from .msa_block_select import (
    pa_sparse_block_score_decode as pa_sparse_block_score_decode_core,
)
from .msa_block_select import (
    pa_sparse_block_score_prefill as pa_sparse_block_score_prefill_core,
)
from .msa_block_select import pa_sparse_block_topk as pa_sparse_block_topk_core


def pa_sparse_block_score_decode(
    q_idx: torch.Tensor,
    key_cache_idx: torch.Tensor,
    score: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    init_blocks: int = 0,
    local_blocks: int = 0,
    query_len: int = 1,
    max_seq_len: int = 0,
) -> None:
    pa_sparse_block_score_decode_core(
        q_idx,
        key_cache_idx,
        score,
        block_table,
        seq_lens,
        init_blocks,
        local_blocks,
        query_len,
        max_seq_len,
    )


direct_register_custom_op(
    "pa_sparse_block_score_decode",
    pa_sparse_block_score_decode,
    ["score"],
)


def pa_sparse_block_score_prefill(
    q_idx: torch.Tensor,
    key_cache_idx: torch.Tensor,
    score: torch.Tensor,
    block_table: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    seq_lens: torch.Tensor,
    init_blocks: int = 0,
    local_blocks: int = 0,
    max_query_len: int = 0,
    max_seq_len: int = 0,
) -> None:
    pa_sparse_block_score_prefill_core(
        q_idx,
        key_cache_idx,
        score,
        block_table,
        cu_seqlens_q,
        seq_lens,
        init_blocks,
        local_blocks,
        max_query_len,
        max_seq_len,
    )


direct_register_custom_op(
    "pa_sparse_block_score_prefill",
    pa_sparse_block_score_prefill,
    ["score"],
)


def pa_sparse_block_topk(
    score: torch.Tensor,
    topk_idx: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    sparse_bt: torch.Tensor,
    sparse_ctx: torch.Tensor,
    max_seq_len: int = 0,
    block_size: int = 0,
    query_len: int = 1,
    num_waves: int = 0,
    num_valid_pages: torch.Tensor | None = None,
    row_req_id: torch.Tensor | None = None,
    kv_lens: torch.Tensor | None = None,
    num_kv_heads: int = 1,
    pages_per_block: int = 8,
) -> None:
    pa_sparse_block_topk_core(
        score,
        topk_idx,
        block_table,
        seq_lens,
        max_seq_len,
        block_size,
        query_len,
        num_waves,
        num_valid_pages,
        sparse_bt,
        sparse_ctx,
        row_req_id,
        kv_lens,
        num_kv_heads,
        pages_per_block,
    )


direct_register_custom_op(
    "pa_sparse_block_topk",
    pa_sparse_block_topk,
    ["topk_idx", "sparse_bt", "sparse_ctx"],
)
