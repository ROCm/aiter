# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Torch custom-op registration for the top-k-index decode block-scoring pass.

Mirrors msa_attention.py's pattern for the incumbent, including its mutation
list: "score" is the written-in-place output, and a registration that omitted it
would let the compiler reorder or elide the write (cudagraph invariant G3).

Registered in a module of its own so msa_attention.py is not modified: the
incumbent's three registrations keep their exact form.
"""

import torch

from csrc.cpp_itfs.torch_utils import direct_register_custom_op

from .topk_index_score import (
    topk_index_score_decode as _topk_index_score_decode_core,
)

# Sentinel for 'take the tuning table' -- torch schema inference needs a
# concrete annotated type, and 0 is a MEANINGFUL aux_k (the control leg), so it
# cannot double as 'unset'.
OPUS_AUX_K_FROM_TABLE = -1


def topk_index_score_decode(
    q_idx: torch.Tensor,
    key_cache_idx: torch.Tensor,
    score: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    sm_scale: float,
    query_len: int = 1,
    max_seq_len: int = 0,
    aux_k: int = OPUS_AUX_K_FROM_TABLE,
) -> None:
    _topk_index_score_decode_core(
        q_idx,
        key_cache_idx,
        score,
        block_table,
        seq_lens,
        sm_scale,
        query_len,
        max_seq_len,
        None if aux_k == OPUS_AUX_K_FROM_TABLE else aux_k,
    )


direct_register_custom_op(
    "topk_index_score_decode",
    topk_index_score_decode,
    ["score"],
)
