# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import math
from typing import Optional

import triton
import torch
from aiter.ops.triton._triton_kernels.pa_decode import (
    _paged_attn_decode_v1_wo_dot_kernel,
    _paged_attn_decode_v1_w_dot_kernel,
    _paged_attn_decode_v2_wo_dot_kernel,
    _paged_attn_decode_v2_wo_dot_reduce_kernel,
    _paged_attn_decode_v2_w_dot_kernel,
    _paged_attn_decode_v2_w_dot_reduce_kernel,
    _paged_attn_decode_v1_wo_dot_kernel_per_token_quant,
    _paged_attn_decode_v1_w_dot_kernel_per_token_quant,
    _paged_attn_decode_v2_wo_dot_kernel_per_token_quant,
    _paged_attn_decode_v2_wo_dot_reduce_kernel_per_token_quant,
    _paged_attn_decode_v2_w_dot_kernel_per_token_quant,
    _paged_attn_decode_v2_w_dot_reduce_kernel_per_token_quant,
)

from aiter.ops.triton.utils.logger import AiterTritonLogger

_LOGGER = AiterTritonLogger()

# This code is derived from sglang and FLASHNN projects
# https://github.com/AlibabaPAI/FLASHNN/blob/main/flashnn/triton_kernels/paged_attn.py

_SEQ_PARTITION_SIZE = 1024  # HIP


def paged_attention_decode(
    output: torch.Tensor,  # [num_seqs, num_kv_heads*query_grp_sz, head_sz]
    query: torch.Tensor,  # [num_seqs, num_kv_heads*query_grp_sz, head_sz]
    key_cache: torch.Tensor,  # [num_blks, num_kv_heads, kv_blk_sz, head_sz]
    value_cache: torch.Tensor,  # [num_blks, num_kv_heads, kv_blk_sz, head_sz]
    seq_lens: torch.Tensor,  # [num_seqs]
    block_tables: torch.Tensor,  # [num_seqs, max_num_blks_per_seq]
    attn_scale: float,
    max_seq_len: int,
    compute_type,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    num_seq_partitions: int = 0,  # TODO use this below
    alibi_slopes: torch.Tensor = None,
) -> None:
    """
    Paged attention decode with automatic V1/V2 dispatch and quantization support.
    V1 for short sequences (≤8192), V2 with sequence partitioning for longer sequences.

    Args:
        output (torch.Tensor): Pre-allocated output with shape (num_seqs, num_q_heads, head_dim).
        query (torch.Tensor): Query tensor with shape (num_seqs, num_q_heads, head_dim).
        key_cache (torch.Tensor): Paged key cache with shape (num_blocks, num_kv_heads, block_size, head_dim).
        value_cache (torch.Tensor): Paged value cache with shape (num_blocks, num_kv_heads, block_size, head_dim).
        seq_lens (torch.Tensor): Sequence lengths with shape (num_seqs,).
        block_tables (torch.Tensor): Block table mapping with shape (num_seqs, max_blocks_per_seq).
        attn_scale (float): Attention scale, typically 1/sqrt(head_dim).
        max_seq_len (int): Maximum sequence length in batch.
        compute_type: Compute precision type.
        k_scale (torch.Tensor): Key quantization scale. Scalar for per-tensor,
            shape (num_blocks, num_kv_heads, block_size) for per-token.
        v_scale (torch.Tensor): Value quantization scale with same shape as k_scale.
        num_seq_partitions (int): Number of sequence partitions (not currently used).
        alibi_slopes (Optional[torch.Tensor]): ALiBi position bias slopes.

    Returns:
        None. Results written in-place to output.
    """

    _LOGGER.info(
        f"PA_DECODE: q={tuple(query.shape)} key_cache={tuple(key_cache.shape)} value_cache={tuple(value_cache.shape)}"
    )
    # get num_seqs, num_kv_heads, kv_blk_sz, head_sz and query_grp_sz
    num_seqs = query.shape[0]
    num_q_heads = query.shape[1]
    num_kv_heads = key_cache.shape[1]

    max_num_partitions = (max_seq_len + _SEQ_PARTITION_SIZE - 1) // _SEQ_PARTITION_SIZE

    use_v1 = max_seq_len <= 8192 and (
        max_num_partitions == 1 or num_seqs * num_q_heads > 512
    )
    if k_scale.numel() > 1:
        if use_v1:
            paged_attn_decode_v1_per_token_quant(
                output,
                query,
                key_cache,
                value_cache,
                block_tables,
                seq_lens,
                max_seq_len,
                compute_type,
                num_kv_heads,
                attn_scale,
                alibi_slopes,
                k_scale,
                v_scale,
            )
        else:
            paged_attn_decode_v2_per_token_quant(
                output,
                query,
                key_cache,
                value_cache,
                block_tables,
                seq_lens,
                max_seq_len,
                compute_type,
                num_kv_heads,
                attn_scale,
                alibi_slopes,
                k_scale,
                v_scale,
                max_num_partitions,
            )
    else:
        if use_v1:
            paged_attn_decode_v1(
                output,
                query,
                key_cache,
                value_cache,
                block_tables,
                seq_lens,
                max_seq_len,
                compute_type,
                num_kv_heads,
                attn_scale,
                alibi_slopes,
                k_scale.item(),
                v_scale.item(),
            )
        else:
            paged_attn_decode_v2(
                output,
                query,
                key_cache,
                value_cache,
                block_tables,
                seq_lens,
                max_seq_len,
                compute_type,
                num_kv_heads,
                attn_scale,
                alibi_slopes,
                k_scale.item(),
                v_scale.item(),
                max_num_partitions,
            )


def paged_attn_decode_v1(
    output: torch.Tensor,  # [num_seqs, num_kv_heads*query_grp_sz, head_sz]
    query: torch.Tensor,  # [num_seqs, num_kv_heads*query_grp_sz, head_sz]
    key_cache: torch.Tensor,  # [num_blks, num_kv_heads, kv_blk_sz, head_sz]
    value_cache: torch.Tensor,  # [num_blks, num_kv_heads, kv_blk_sz, head_sz]
    block_tables: torch.Tensor,  # [num_seqs, max_num_blks_per_seq]
    seq_lens: torch.Tensor,  # [num_seqs]
    max_seq_len: int,
    compute_type,
    num_kv_heads: int,
    scale: float,
    alibi_slopes: Optional[torch.Tensor],
    k_scale: float,
    v_scale: float,
    tp_rank: int = 0,
    blocksparse_local_blocks: int = 0,
    blocksparse_vert_stride: int = 0,
    blocksparse_block_size: int = 64,
    blocksparse_head_sliding_step: int = 0,
):
    """
    #TODO: Add Doc
    """

    num_seqs = query.shape[0]
    num_q_heads = query.shape[1]
    kv_blk_sz = key_cache.shape[2]
    head_sz = key_cache.shape[3]
    query_grp_sz = query.shape[1] // num_kv_heads
    query_grp_sz_pow2 = triton.next_power_of_2(query_grp_sz)
    kv_blk_sz_pow2 = triton.next_power_of_2(kv_blk_sz)
    head_sz_pow2 = triton.next_power_of_2(head_sz)

    # MHA- Multi-Head Attention
    if query_grp_sz == 1:
        grid = (num_q_heads, num_seqs, 1)
        _paged_attn_decode_v1_wo_dot_kernel[grid](
            output,
            query,
            key_cache,
            value_cache,
            block_tables,
            seq_lens,
            alibi_slopes,
            scale,
            k_scale,
            v_scale,
            query.stride(0),
            query.stride(1),
            output.stride(0),
            output.stride(1),
            key_cache.stride(0),
            key_cache.stride(1),
            key_cache.stride(2),
            block_tables.stride(0),
            compute_type=compute_type,
            KV_BLK_SZ=kv_blk_sz,
            KV_BLK_SZ_POW2=kv_blk_sz_pow2,
            HEAD_SZ=head_sz,
            HEAD_SZ_POW2=head_sz_pow2,
            QUERY_GRP_SZ=query_grp_sz,
        )
    # GQA - Grouped Query Attention
    else:
        grid = (num_seqs, num_kv_heads, 1)
        if query_grp_sz <= 16:
            query_grp_sz_pow2 = 16
        else:
            query_grp_sz_pow2 = triton.next_power_of_2(query_grp_sz)
        _paged_attn_decode_v1_w_dot_kernel[grid](
            output,
            query,
            key_cache,
            value_cache,
            block_tables,
            seq_lens,
            alibi_slopes,
            scale,
            k_scale,
            v_scale,
            output.stride(0),
            output.stride(1),
            query.stride(0),
            query.stride(1),
            query.stride(2),
            key_cache.stride(0),
            key_cache.stride(1),
            key_cache.stride(2),
            key_cache.stride(3),
            block_tables.stride(0),
            compute_type=compute_type,
            HEAD_SZ=head_sz,
            HEAD_SZ_POW2=head_sz_pow2,
            QUERY_GRP_SZ=query_grp_sz,
            QUERY_GRP_SZ_POW2=query_grp_sz_pow2,
            KV_BLK_SZ=kv_blk_sz,
            KV_BLK_SZ_POW2=kv_blk_sz,
        )


def paged_attn_decode_v2(
    output: torch.Tensor,  # [num_seqs, num_kv_heads*query_grp_sz, head_sz],
    query: torch.Tensor,  # [num_seqs, num_kv_heads*query_grp_sz, head_sz],
    key_cache: torch.Tensor,  # [num_blks, num_kv_heads, kv_blk_sz, head_sz] ,
    value_cache: torch.Tensor,  # [num_blks, num_kv_heads, kv_blk_sz, head_sz] ,
    block_tables: torch.Tensor,  # [num_seqs, max_num_blks_per_seq],
    seq_lens: torch.Tensor,  # [num_seqs],
    max_seq_len: int,
    compute_type,
    num_kv_heads: int,
    scale: float,
    alibi_slopes: Optional[torch.Tensor],
    k_scale: float,
    v_scale: float,
    max_num_partitions: int,
    tp_rank: int = 0,
    blocksparse_local_blocks: int = 0,
    blocksparse_vert_stride: int = 0,
    blocksparse_block_size: int = 64,
    blocksparse_head_sliding_step: int = 0,
):
    """
    #TODO: Add Doc
    """

    num_seqs = query.shape[0]
    num_q_heads = query.shape[1]
    kv_blk_sz = key_cache.shape[2]
    head_sz = key_cache.shape[3]
    query_grp_sz = num_q_heads // num_kv_heads
    query_grp_sz_pow2 = triton.next_power_of_2(query_grp_sz)

    # Note: There is a bug in triton.next_power_of_2 function which causes it
    # to update the passed in arg, so that's why we have a workaround here
    # max_num_partitions_pow2 = triton.next_power_of_2(max_num_partitions)
    if max_num_partitions == 0:
        max_num_partitions_pow2 = 1
    else:
        max_num_partitions_pow2 = 2 ** math.ceil(math.log2(max_num_partitions))

    kv_blk_sz_pow2 = triton.next_power_of_2(kv_blk_sz)
    head_sz_pow2 = triton.next_power_of_2(head_sz)

    # MHA
    if query_grp_sz == 1:
        grid = (num_q_heads, num_seqs, max_num_partitions)
        shape_info = (num_seqs, num_q_heads, max_num_partitions)
        exp_sums = torch.empty(
            size=shape_info, dtype=torch.float32, device=output.device
        )
        max_logits = torch.empty(
            size=shape_info, dtype=torch.float32, device=output.device
        )
        tmp_output = torch.empty(
            (*shape_info, head_sz), dtype=output.dtype, device=output.device
        )
        _paged_attn_decode_v2_wo_dot_kernel[grid](
            exp_sums,
            max_logits,
            tmp_output,
            query,
            key_cache,
            value_cache,
            block_tables,
            seq_lens,
            scale,
            k_scale,
            v_scale,
            alibi_slopes,
            exp_sums.stride(0),
            exp_sums.stride(1),
            tmp_output.stride(0),
            tmp_output.stride(1),
            tmp_output.stride(2),
            query.stride(0),
            query.stride(1),
            key_cache.stride(0),
            key_cache.stride(1),
            key_cache.stride(2),
            block_tables.stride(0),
            block_tables.stride(1),
            compute_type=compute_type,
            KV_BLK_SZ=kv_blk_sz,
            KV_BLK_SZ_POW2=kv_blk_sz_pow2,
            HEAD_SZ=head_sz,
            HEAD_SZ_POW2=head_sz_pow2,
            QUERY_GRP_SZ=query_grp_sz,
            SEQ_PARTITION_SZ=_SEQ_PARTITION_SIZE,
        )
        grid = (num_q_heads, num_seqs, 1)
        _paged_attn_decode_v2_wo_dot_reduce_kernel[grid](
            output,
            exp_sums,
            max_logits,
            tmp_output,
            seq_lens,
            output.stride(0),
            output.stride(1),
            exp_sums.stride(0),
            exp_sums.stride(1),
            tmp_output.stride(0),
            tmp_output.stride(1),
            tmp_output.stride(2),
            HEAD_SZ=head_sz,
            HEAD_SZ_POW2=head_sz_pow2,
            SEQ_PARTITION_SZ=_SEQ_PARTITION_SIZE,
            MAX_NUM_SEQ_PARTITIONS_POW2=int(max_num_partitions_pow2),
        )
    # GQA
    else:
        grid = (num_seqs, num_kv_heads, max_num_partitions)
        shape_info = (num_seqs, num_kv_heads, max_num_partitions, query_grp_sz)
        max_logits = torch.empty(shape_info, dtype=torch.float32, device=output.device)
        exp_sums = torch.empty(shape_info, dtype=torch.float32, device=output.device)
        tmp_output = torch.empty(
            *shape_info, head_sz, dtype=output.dtype, device=output.device
        )
        if query_grp_sz <= 16:
            query_grp_sz_pow2 = 16
        else:
            query_grp_sz_pow2 = triton.next_power_of_2(query_grp_sz)
        _paged_attn_decode_v2_w_dot_kernel[grid](
            exp_sums,
            max_logits,
            tmp_output,
            query,
            key_cache,
            value_cache,
            block_tables,
            seq_lens,
            scale,
            k_scale,
            v_scale,
            alibi_slopes,
            exp_sums.stride(0),
            exp_sums.stride(1),
            exp_sums.stride(2),
            tmp_output.stride(0),
            tmp_output.stride(1),
            tmp_output.stride(2),
            tmp_output.stride(3),
            query.stride(0),
            query.stride(1),
            key_cache.stride(0),
            key_cache.stride(1),
            key_cache.stride(2),
            block_tables.stride(0),
            compute_type=compute_type,
            HEAD_SZ=head_sz,
            HEAD_SZ_POW2=head_sz_pow2,
            QUERY_GRP_SZ=query_grp_sz,
            QUERY_GRP_SZ_POW2=query_grp_sz_pow2,
            KV_BLK_SZ=kv_blk_sz,
            KV_BLK_SZ_POW2=kv_blk_sz_pow2,
            SEQ_PARTITION_SZ=_SEQ_PARTITION_SIZE,
        )
        grid = (num_seqs, num_kv_heads, 1)
        _paged_attn_decode_v2_w_dot_reduce_kernel[grid](
            output,
            exp_sums,
            max_logits,
            tmp_output,
            seq_lens,
            output.stride(0),
            output.stride(1),
            exp_sums.stride(0),
            exp_sums.stride(1),
            exp_sums.stride(2),
            tmp_output.stride(0),
            tmp_output.stride(1),
            tmp_output.stride(2),
            tmp_output.stride(3),
            HEAD_SZ=head_sz,
            HEAD_SZ_POW2=head_sz_pow2,
            QUERY_GRP_SZ=query_grp_sz,
            QUERY_GRP_SZ_POW2=query_grp_sz_pow2,
            SEQ_PARTITION_SZ=_SEQ_PARTITION_SIZE,
            MAX_NUM_SEQ_PARTITIONS_POW2=int(triton.next_power_of_2(max_num_partitions)),
        )


def paged_attn_decode_v1_per_token_quant(
    output: torch.Tensor,  # [num_seqs, num_kv_heads*query_grp_sz, head_sz]
    query: torch.Tensor,  # [num_seqs, num_kv_heads*query_grp_sz, head_sz]
    key_cache: torch.Tensor,  # [num_blks, num_kv_heads, kv_blk_sz, head_sz]
    value_cache: torch.Tensor,  # [num_blks, num_kv_heads, kv_blk_sz, head_sz]
    block_tables: torch.Tensor,  # [num_seqs, max_num_blks_per_seq]
    seq_lens: torch.Tensor,  # [num_seqs]
    max_seq_len: int,
    compute_type,
    num_kv_heads: int,
    scale: float,
    alibi_slopes: Optional[torch.Tensor],
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    tp_rank: int = 0,
    blocksparse_local_blocks: int = 0,
    blocksparse_vert_stride: int = 0,
    blocksparse_block_size: int = 64,
    blocksparse_head_sliding_step: int = 0,
):
    """
    #TODO: Add Doc
    """

    num_seqs = query.shape[0]
    num_q_heads = query.shape[1]
    kv_blk_sz = key_cache.shape[2]
    head_sz = key_cache.shape[3]
    query_grp_sz = query.shape[1] // num_kv_heads
    query_grp_sz_pow2 = triton.next_power_of_2(query_grp_sz)
    kv_blk_sz_pow2 = triton.next_power_of_2(kv_blk_sz)
    head_sz_pow2 = triton.next_power_of_2(head_sz)

    # MHA- Multi-Head Attention
    if query_grp_sz == 1:
        grid = (num_q_heads, num_seqs, 1)
        _paged_attn_decode_v1_wo_dot_kernel_per_token_quant[grid](
            output,
            query,
            key_cache,
            value_cache,
            block_tables,
            seq_lens,
            alibi_slopes,
            scale,
            k_scale,
            v_scale,
            query.stride(0),
            query.stride(1),
            output.stride(0),
            output.stride(1),
            key_cache.stride(0),
            key_cache.stride(1),
            key_cache.stride(2),
            block_tables.stride(0),
            k_scale.stride(0),
            k_scale.stride(1),
            k_scale.stride(2),
            compute_type=compute_type,
            KV_BLK_SZ=kv_blk_sz,
            KV_BLK_SZ_POW2=kv_blk_sz_pow2,
            HEAD_SZ=head_sz,
            HEAD_SZ_POW2=head_sz_pow2,
            QUERY_GRP_SZ=query_grp_sz,
        )
    # GQA - Grouped Query Attention
    else:
        grid = (num_seqs, num_kv_heads, 1)
        if query_grp_sz <= 16:
            query_grp_sz_pow2 = 16
        else:
            query_grp_sz_pow2 = triton.next_power_of_2(query_grp_sz)
        _paged_attn_decode_v1_w_dot_kernel_per_token_quant[grid](
            output,
            query,
            key_cache,
            value_cache,
            block_tables,
            seq_lens,
            alibi_slopes,
            scale,
            k_scale,
            v_scale,
            output.stride(0),
            output.stride(1),
            query.stride(0),
            query.stride(1),
            query.stride(2),
            key_cache.stride(0),
            key_cache.stride(1),
            key_cache.stride(2),
            key_cache.stride(3),
            block_tables.stride(0),
            k_scale.stride(0),
            k_scale.stride(1),
            k_scale.stride(2),
            compute_type=compute_type,
            HEAD_SZ=head_sz,
            HEAD_SZ_POW2=head_sz_pow2,
            QUERY_GRP_SZ=query_grp_sz,
            QUERY_GRP_SZ_POW2=query_grp_sz_pow2,
            KV_BLK_SZ=kv_blk_sz,
            KV_BLK_SZ_POW2=kv_blk_sz,
        )


def paged_attn_decode_v2_per_token_quant(
    output: torch.Tensor,  # [num_seqs, num_kv_heads*query_grp_sz, head_sz],
    query: torch.Tensor,  # [num_seqs, num_kv_heads*query_grp_sz, head_sz],
    key_cache: torch.Tensor,  # [num_blks, num_kv_heads, kv_blk_sz, head_sz] ,
    value_cache: torch.Tensor,  # [num_blks, num_kv_heads, kv_blk_sz, head_sz] ,
    block_tables: torch.Tensor,  # [num_seqs, max_num_blks_per_seq],
    seq_lens: torch.Tensor,  # [num_seqs],
    max_seq_len: int,
    compute_type,
    num_kv_heads: int,
    scale: float,
    alibi_slopes: Optional[torch.Tensor],
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    max_num_partitions: int,
    tp_rank: int = 0,
    blocksparse_local_blocks: int = 0,
    blocksparse_vert_stride: int = 0,
    blocksparse_block_size: int = 64,
    blocksparse_head_sliding_step: int = 0,
):
    """
    #TODO: Add Doc
    """

    num_seqs = query.shape[0]
    num_q_heads = query.shape[1]
    kv_blk_sz = key_cache.shape[2]
    head_sz = key_cache.shape[3]
    query_grp_sz = num_q_heads // num_kv_heads
    query_grp_sz_pow2 = triton.next_power_of_2(query_grp_sz)

    # Note: There is a bug in triton.next_power_of_2 function which causes it
    # to update the passed in arg, so that's why we have a workaround here
    # max_num_partitions_pow2 = triton.next_power_of_2(max_num_partitions)
    if max_num_partitions == 0:
        max_num_partitions_pow2 = 1
    else:
        max_num_partitions_pow2 = 2 ** math.ceil(math.log2(max_num_partitions))

    kv_blk_sz_pow2 = triton.next_power_of_2(kv_blk_sz)
    head_sz_pow2 = triton.next_power_of_2(head_sz)

    # MHA
    if query_grp_sz == 1:
        grid = (num_q_heads, num_seqs, max_num_partitions)
        shape_info = (num_seqs, num_q_heads, max_num_partitions)
        exp_sums = torch.empty(
            size=shape_info, dtype=torch.float32, device=output.device
        )
        max_logits = torch.empty(
            size=shape_info, dtype=torch.float32, device=output.device
        )
        tmp_output = torch.empty(
            (*shape_info, head_sz), dtype=output.dtype, device=output.device
        )
        _paged_attn_decode_v2_wo_dot_kernel_per_token_quant[grid](
            exp_sums,
            max_logits,
            tmp_output,
            query,
            key_cache,
            value_cache,
            block_tables,
            seq_lens,
            scale,
            k_scale,
            v_scale,
            alibi_slopes,
            exp_sums.stride(0),
            exp_sums.stride(1),
            tmp_output.stride(0),
            tmp_output.stride(1),
            tmp_output.stride(2),
            query.stride(0),
            query.stride(1),
            key_cache.stride(0),
            key_cache.stride(1),
            key_cache.stride(2),
            block_tables.stride(0),
            block_tables.stride(1),
            k_scale.stride(0),
            k_scale.stride(1),
            k_scale.stride(2),
            compute_type=compute_type,
            KV_BLK_SZ=kv_blk_sz,
            KV_BLK_SZ_POW2=kv_blk_sz_pow2,
            HEAD_SZ=head_sz,
            HEAD_SZ_POW2=head_sz_pow2,
            QUERY_GRP_SZ=query_grp_sz,
            SEQ_PARTITION_SZ=_SEQ_PARTITION_SIZE,
        )
        grid = (num_q_heads, num_seqs, 1)
        _paged_attn_decode_v2_wo_dot_reduce_kernel_per_token_quant[grid](
            output,
            exp_sums,
            max_logits,
            tmp_output,
            seq_lens,
            output.stride(0),
            output.stride(1),
            exp_sums.stride(0),
            exp_sums.stride(1),
            tmp_output.stride(0),
            tmp_output.stride(1),
            tmp_output.stride(2),
            HEAD_SZ=head_sz,
            HEAD_SZ_POW2=head_sz_pow2,
            SEQ_PARTITION_SZ=_SEQ_PARTITION_SIZE,
            MAX_NUM_SEQ_PARTITIONS_POW2=int(max_num_partitions_pow2),
        )
    # GQA
    else:
        grid = (num_seqs, num_kv_heads, max_num_partitions)
        shape_info = (num_seqs, num_kv_heads, max_num_partitions, query_grp_sz)
        max_logits = torch.empty(shape_info, dtype=torch.float32, device=output.device)
        exp_sums = torch.empty(shape_info, dtype=torch.float32, device=output.device)
        tmp_output = torch.empty(
            *shape_info, head_sz, dtype=output.dtype, device=output.device
        )
        if query_grp_sz <= 16:
            query_grp_sz_pow2 = 16
        else:
            query_grp_sz_pow2 = triton.next_power_of_2(query_grp_sz)
        _paged_attn_decode_v2_w_dot_kernel_per_token_quant[grid](
            exp_sums,
            max_logits,
            tmp_output,
            query,
            key_cache,
            value_cache,
            block_tables,
            seq_lens,
            scale,
            k_scale,
            v_scale,
            alibi_slopes,
            exp_sums.stride(0),
            exp_sums.stride(1),
            exp_sums.stride(2),
            tmp_output.stride(0),
            tmp_output.stride(1),
            tmp_output.stride(2),
            tmp_output.stride(3),
            query.stride(0),
            query.stride(1),
            key_cache.stride(0),
            key_cache.stride(1),
            key_cache.stride(2),
            block_tables.stride(0),
            k_scale.stride(0),
            k_scale.stride(1),
            k_scale.stride(2),
            compute_type=compute_type,
            HEAD_SZ=head_sz,
            HEAD_SZ_POW2=head_sz_pow2,
            QUERY_GRP_SZ=query_grp_sz,
            QUERY_GRP_SZ_POW2=query_grp_sz_pow2,
            KV_BLK_SZ=kv_blk_sz,
            KV_BLK_SZ_POW2=kv_blk_sz_pow2,
            SEQ_PARTITION_SZ=_SEQ_PARTITION_SIZE,
        )
        grid = (num_seqs, num_kv_heads, 1)
        _paged_attn_decode_v2_w_dot_reduce_kernel_per_token_quant[grid](
            output,
            exp_sums,
            max_logits,
            tmp_output,
            seq_lens,
            output.stride(0),
            output.stride(1),
            exp_sums.stride(0),
            exp_sums.stride(1),
            exp_sums.stride(2),
            tmp_output.stride(0),
            tmp_output.stride(1),
            tmp_output.stride(2),
            tmp_output.stride(3),
            HEAD_SZ=head_sz,
            HEAD_SZ_POW2=head_sz_pow2,
            QUERY_GRP_SZ=query_grp_sz,
            QUERY_GRP_SZ_POW2=query_grp_sz_pow2,
            SEQ_PARTITION_SZ=_SEQ_PARTITION_SIZE,
            MAX_NUM_SEQ_PARTITIONS_POW2=int(triton.next_power_of_2(max_num_partitions)),
        )
