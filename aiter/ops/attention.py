# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import torch
from typing import Tuple, Optional
from ..jit.core import (
    compile_ops,
)
from csrc.cpp_itfs.pa.pa import paged_attention_rocm as paged_attention_rocm_core
from csrc.cpp_itfs.pa.pa_v1 import paged_attention_v1 as paged_attention_v1_core
from csrc.cpp_itfs.pa.pa_ragged import (
    paged_attention_ragged as paged_attention_ragged_core,
)
from csrc.cpp_itfs.torch_utils import direct_register_custom_op

MD_NAME = "module_attention"


@compile_ops("module_attention")
def pa_fwd_naive(
    # [num_seqs, num_heads, head_size]
    query: torch.Tensor,
    # [num_blocks, num_kv_heads, head_size/x, block_size, x]
    key_cache: torch.Tensor,
    # [num_blocks, num_kv_heads, head_size, block_size]
    value_cache: torch.Tensor,
    # [num_seqs, max_num_blocks_per_seq]
    block_tables: torch.Tensor,
    # [num_seqs]
    context_lens: torch.Tensor,
    k_dequant_scales: torch.Tensor,
    v_dequant_scales: torch.Tensor,
    max_seq_len: int,
    num_kv_heads: int,
    scale_s: float,
    scale_k: float,
    scale_v: float,
    block_size: int,
    quant_algo: int,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor: ...


@compile_ops("module_attention_asm")
def pa_fwd_asm(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    max_num_blocks: int,
    max_qlen: int = 1,
    K_QScale: Optional[torch.Tensor] = None,
    V_QScale: Optional[torch.Tensor] = None,
    out_: Optional[torch.Tensor] = None,
    qo_indptr: Optional[torch.Tensor] = None,
    high_precision: Optional[
        int
    ] = 1,  # [0, 1, 2] 2 is the highest precision, this is only for fp8 kvcache
    kernelName: str = "",
) -> torch.Tensor: ...


def paged_attention_rocm(
    out: torch.Tensor,
    exp_sums: torch.Tensor,
    max_logits: torch.Tensor,
    tmp_out: torch.Tensor,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    num_kv_heads: int,
    scale: float,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    block_size: int,
    max_context_len: int,
    alibi_slopes: Optional[torch.Tensor],
    kv_cache_dtype: str,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    fp8_out_scale: Optional[torch.Tensor] = None,
    partition_size: int = 256,
    mtp: int = 1,
) -> torch.Tensor:
    paged_attention_rocm_core(
        out,
        exp_sums,
        max_logits,
        tmp_out,
        query,
        key_cache,
        value_cache,
        num_kv_heads,
        scale,
        block_tables,
        context_lens,
        block_size,
        max_context_len,
        alibi_slopes,
        kv_cache_dtype,
        k_scale,
        v_scale,
        fp8_out_scale,
        partition_size,
        mtp,
    )
    return out


direct_register_custom_op(
    "paged_attention_rocm",
    paged_attention_rocm,
    ["out", "exp_sums", "max_logits", "tmp_out"],
)


def paged_attention_v1(
    out: torch.Tensor,
    workspace_buffer: torch.Tensor,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    scale: float,
    block_tables: torch.Tensor,
    cu_query_lens: Optional[torch.Tensor],
    context_lens: torch.Tensor,
    max_context_len: int,
    alibi_slopes: Optional[torch.Tensor],
    kv_cache_dtype: str,
    kv_cache_layout: str,
    logits_soft_cap: float,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    fp8_out_scale: Optional[torch.Tensor] = None,
    partition_size: int = 256,
    mtp: int = 1,
) -> torch.Tensor:
    paged_attention_v1_core(
        out,
        workspace_buffer,
        query,
        key_cache,
        value_cache,
        scale,
        block_tables,
        cu_query_lens,
        context_lens,
        max_context_len,
        alibi_slopes,
        kv_cache_dtype,
        kv_cache_layout,
        logits_soft_cap,
        k_scale,
        v_scale,
        fp8_out_scale,
        partition_size,
        mtp,
    )
    return out


direct_register_custom_op(
    "paged_attention_v1",
    paged_attention_v1,
    ["out", "workspace_buffer"],
)


def paged_attention_ragged(
    out: torch.Tensor,
    workspace_buffer: torch.Tensor,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    scale: float,
    kv_indptr: torch.Tensor,
    kv_page_indices: torch.Tensor,
    kv_last_page_lens: torch.Tensor,
    block_size: int,
    max_num_partitions: int,
    alibi_slopes: Optional[torch.Tensor],
    kv_cache_dtype: str,
    kv_cache_layout: str,
    logits_soft_cap: float,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    fp8_out_scale: Optional[torch.Tensor] = None,
    partition_size: int = 256,
    mtp: int = 1,
) -> torch.Tensor:
    paged_attention_ragged_core(
        out,
        workspace_buffer,
        query,
        key_cache,
        value_cache,
        scale,
        kv_indptr,
        kv_page_indices,
        kv_last_page_lens,
        block_size,
        max_num_partitions,
        alibi_slopes,
        kv_cache_dtype,
        kv_cache_layout,
        logits_soft_cap,
        k_scale,
        v_scale,
        fp8_out_scale,
        partition_size,
        mtp,
    )
    return out


direct_register_custom_op(
    "paged_attention_ragged",
    paged_attention_ragged,
    ["out", "workspace_buffer"],
)


MD_NAME = "module_mla_asm"


@compile_ops(MD_NAME)
def mla_decode_stage1_asm_fwd(
    # [num_seqs, num_heads, head_size]
    Q: torch.Tensor,
    # [num_page, page_size, num_kv_heads, kv_lora_rank + qk_rope_head_dim]
    KV: torch.Tensor,
    # [batch_size+1]
    qo_indptr: torch.Tensor,
    # [batch_size+1]
    kv_indptr: torch.Tensor,
    # [num_page_used]
    kv_page_indices: torch.Tensor,
    # [batch_size]
    kv_last_page_lens: torch.Tensor,
    num_kv_splits_indptr: torch.Tensor,
    max_seqlen_q: int,
    softmax_scale: float,
    # [batch_size, num_kv_splits, num_heads, v_head_dim]
    splitData: torch.Tensor,
    # [batch_size, num_kv_splits, num_heads,  1]
    splitLse: torch.Tensor,
): ...


@compile_ops(MD_NAME)
def mla_prefill_asm_fwd(
    # [num_seqs, num_heads, head_size]
    Q: torch.Tensor,
    # [num_page, page_size, num_kv_heads, kv_lora_rank + qk_rope_head_dim]
    KV: torch.Tensor,
    # [batch_size+1]
    qo_indptr: torch.Tensor,
    # [batch_size+1]
    kv_indptr: torch.Tensor,
    # [num_page_used]
    kv_page_indices: torch.Tensor,
    # [batch_size]
    kv_last_page_lens: torch.Tensor,
    max_seqlen_q: int,
    softmax_scale: float,
    # [batch_size, num_kv_splits, num_heads, v_head_dim]
    splitData: torch.Tensor,
    # [batch_size, num_kv_splits, num_heads,  1]
    splitLse: torch.Tensor,
): ...


@compile_ops("module_mla_metadata")
def get_mla_metadata_v0(
    seqlens: torch.Tensor,
    num_heads_per_head_k: int,
    num_heads_k: int,
) -> Tuple[torch.Tensor, int]:
    """
    Arguments:
        cumulated seqlens: (batch_size + 1), dtype torch.int32.
        num_heads_per_head_k: Equals to num_heads_q // num_heads_k.
        num_heads_k: num_heads_k.
    Returns:
        cumulated num_kv_splits: (batch_size + 1), dtype torch.int32.
        max_num_splits: (1), dtype torch.int32.
    """
    ...


@compile_ops("module_mla_metadata")
def get_mla_metadata_v1(
    seqlens_qo_indptr: torch.Tensor,
    seqlens_kv_indptr: torch.Tensor,
    num_heads_per_head_k: int,
    num_heads_k: int,
    is_causal: bool,
    work_metadata_ptrs: torch.Tensor,
    work_indptr: torch.Tensor,
    work_info: torch.Tensor,
    reduce_indptr: torch.Tensor,
    reduce_final_map: torch.Tensor,
    reduce_partial_map: torch.Tensor,
):
    """
    Inputs:
        cumulated seqlens of q/o: (batch_size + 1), dtype torch.int32.
        cumulated seqlens of k/v: (batch_size + 1), dtype torch.int32.
        num_heads_per_head_k: Equals to num_heads_q // num_heads_k.
        num_heads_k: num_heads_k.
        is_causal: whether causal mask is enabled.
    Outputs:
        [0] work_metadata_ptrs  (2)                 Two 64-bits pointers point to the 1st element of work_indptr and
                                                    work_info.
        [1] work_indptr:        (#cu_part + 1),     The IDs of work handled by each cu_part.
        [2] work_info           (#work, 8)
        [2.0] bs_index:         (#work),            The index of batch handled by each work.
        [2.1] partial_index:    (#work),            The index of tile in output buffer when splits. -1 means no split.
        [2.2] q_start:          (#work),            The global index in seq where q/o starts. Use global index here can
                                                    reduce memory access count in kernel.
        [2.3] q_end:            (#work),            The global index in seq where q/o ends (not included).
        [2.4] kv_start:         (#work),            The global index in seq where k/v starts.
        [2.5] kv_end:           (#work),            The global index in seq where k/v ends (not included).
        [2.6] pad               (#work, 2),         Pad to 8 DWs.
        [3] reduce_indptr:      (sum(qo_seqlen_blk_count) + 1),
                                                    The IDs in reduce_partial_map indicates the tiles should be merged
                                                    together.
        [4] reduce_final_map:   (sum(qo_seqlen_blk_count)),
                                                    The final output location of each group of tiles.
        [5] reduce_partial_map: (#partial_tiles),   The locations in partial buffer of partial tiles waiting for being
                                                    reduced.
    """
    ...


@compile_ops("module_mla_metadata")
def get_mla_metadata_v1_no_redundant(
    seqlens_qo_indptr: torch.Tensor,
    seqlens_kv_indptr: torch.Tensor,
    num_heads_per_head_k: int,
    num_heads_k: int,
    is_causal: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Arguments:
        cumulated seqlens of q/o: (batch_size + 1), dtype torch.int32.
        cumulated seqlens of k/v: (batch_size + 1), dtype torch.int32.
        num_heads_per_head_k: Equals to num_heads_q // num_heads_k.
        num_heads_k: num_heads_k.
        is_causal: whether causal mask is enabled.
    Returns:
        [0] work_metadata_ptrs  (2)                  Two 64-bits pointers point to the 1st element of work_indptr and
                                                     work_info.
        [1] work_indptr:        (#work_cu + 1),      The IDs of work handled by each cu_part.
        [2] work_info           (#work, 8)
        [2.0] bs_index:         (#work),             The index of batch handled by each work.
        [2.1] partial_index:    (#work),             The index of tile in output buffer when splits. -1 means no split.
        [2.2] q_start:          (#work),             The global index in seq where q/o starts. Use global index here can
                                                     reduce memory access count in kernel.
        [2.3] q_end:            (#work),             The global index in seq where q/o ends (not included).
        [2.4] kv_start:         (#work),             The global index in seq where k/v starts.
        [2.5] kv_end:           (#work),             The global index in seq where k/v ends (not included).
        [2.6] pad               (#work, 2),          Pad to 8 DWs.
        [3] reduce_indptr:      (#reduce_tiles + 1), The IDs in reduce_partial_map indicates the tiles should be merged
                                                     together.
        [4] reduce_final_map:   (#reduce_tiles),     The final output location of each group of tiles.
        [5] reduce_partial_map: (#partial_tiles),    The locations in partial buffer of partial tiles waiting for being
                                                     reduced.
    """
    ...


@compile_ops("module_mla_reduce")
def mla_reduce_v1(
    partial_output: torch.Tensor,
    partial_lse: torch.Tensor,
    reduce_indptr: torch.Tensor,
    reduce_final_map: torch.Tensor,
    reduce_partial_map: torch.Tensor,
    final_output: torch.Tensor,
    final_lse: Optional[torch.Tensor] = None,
): ...
