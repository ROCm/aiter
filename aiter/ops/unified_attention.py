# SPDX-License-Identifier: MIT
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

from typing import Optional

import torch

from ..jit.core import compile_ops


def gen_unified_attention_fwd_fake(
    output: torch.Tensor,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    query_start_len: torch.Tensor,
    mask_type: int,
    scale_s: float,
    scale: float,
    scale_k: float,
    scale_v: float,
    scale_out: float,
    window_size_left: int = -1,
    window_size_right: int = -1,
) -> None:
    return None


@compile_ops("module_unified_attention", gen_fake=gen_unified_attention_fwd_fake)
def unified_attention_fwd(
    output: torch.Tensor,          # [num_tokens, num_heads_q, head_size]
    query: torch.Tensor,           # [num_tokens, num_heads_q, head_size]
    key_cache: torch.Tensor,       # [num_blks, blk_size, num_kv_heads, head_size]
    value_cache: torch.Tensor,     # [num_blks, blk_size, num_kv_heads, head_size]
    block_tables: torch.Tensor,    # [num_seqs, max_num_blocks_per_seq]
    seq_lens: torch.Tensor,        # [num_seqs]
    query_start_len: torch.Tensor, # [num_seqs + 1]
    mask_type: int,                # 0: no mask, 1: causal top-left, 2: causal bottom-right
    scale_s: float,
    scale: float,
    scale_k: float,
    scale_v: float,
    scale_out: float,
    # Sliding-window-attention (FA convention): negative = unbounded on that side.
    # Default (-1, -1) means "no SWA" and reproduces the historical behaviour
    # (routes through IsLocal=false instances).
    window_size_left: int = -1,
    window_size_right: int = -1,
) -> None: ...
