# SPDX-License-Identifier: MIT
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

from typing import Optional

import torch

from ..jit.core import compile_ops


# -----------------------------------------------------------------------------
# JIT-compiled C++ kernel entry point. This is the raw 1:1 binding to the
# pybind symbol `unified_attention_fwd` exposed by `module_unified_attention`.
# All caller-facing code should go through `unified_attention_fwd` below, which
# adds the transparent split-KV path.
# -----------------------------------------------------------------------------
def _gen_unified_attention_fwd_kernel_fake(
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
    cache_ptr_int32_overflow_possible: bool = False,
    num_splits: int = 1,
    o_acc_workspace: Optional[torch.Tensor] = None,
    lse_acc_workspace: Optional[torch.Tensor] = None,
) -> None:
    return None


@compile_ops(
    "module_unified_attention",
    fc_name="unified_attention_fwd",
    gen_fake=_gen_unified_attention_fwd_kernel_fake,
)
def _unified_attention_fwd_kernel(
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
    cache_ptr_int32_overflow_possible: bool = False,
    num_splits: int = 1,
    o_acc_workspace: Optional[torch.Tensor] = None,
    lse_acc_workspace: Optional[torch.Tensor] = None,
) -> None: ...


# -----------------------------------------------------------------------------
# Public entry point. Today this is a 1:1 pass-through to the JIT kernel.
# A follow-up commit will add transparent split-KV (heuristic + workspace
# auto-allocation + LSE combine) behind an `allow_splitkv` flag.
# -----------------------------------------------------------------------------
def unified_attention_fwd(
    output: torch.Tensor,           # [num_tokens, num_heads_q, head_size]
    query: torch.Tensor,            # [num_tokens, num_heads_q, head_size]
    key_cache: torch.Tensor,        # [num_blks, blk_size, num_kv_heads, head_size]
    value_cache: torch.Tensor,      # [num_blks, blk_size, num_kv_heads, head_size]
    block_tables: torch.Tensor,     # [num_seqs, max_num_blocks_per_seq]
    seq_lens: torch.Tensor,         # [num_seqs]
    query_start_len: torch.Tensor,  # [num_seqs + 1]
    mask_type: int,                 # 0: no mask, 2: causal
    scale_s: float,
    scale: float,
    scale_k: float,
    scale_v: float,
    scale_out: float,
    cache_ptr_int32_overflow_possible: bool = False,
    # KV-segment parallelism (FlashDecoding-style split-KV). When num_splits
    # > 1 the kernel launches a 3D grid with z-dim == num_splits and writes
    # each CTA's partial (o_acc, lse) into the workspaces; the caller then
    # reduces across the split axis to produce the final output. With
    # num_splits == 1 the workspaces are ignored and `output` is written in
    # the usual way.
    num_splits: int = 1,
    # o_acc_workspace  : float32 [num_q_heads, num_splits, num_tokens, head_size]
    # lse_acc_workspace: float32 [num_q_heads, num_splits, num_tokens]
    o_acc_workspace: Optional[torch.Tensor] = None,
    lse_acc_workspace: Optional[torch.Tensor] = None,
) -> None:
    _unified_attention_fwd_kernel(
        output,
        query,
        key_cache,
        value_cache,
        block_tables,
        seq_lens,
        query_start_len,
        mask_type,
        scale_s,
        scale,
        scale_k,
        scale_v,
        scale_out,
        cache_ptr_int32_overflow_possible,
        num_splits,
        o_acc_workspace,
        lse_acc_workspace,
    )
