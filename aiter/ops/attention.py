# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import math
from typing import Optional, Tuple

import torch
from csrc.cpp_itfs.pa.pa import paged_attention_rocm as paged_attention_rocm_core
from csrc.cpp_itfs.pa.pa_ragged import (
    paged_attention_ragged as paged_attention_ragged_core,
)
from csrc.cpp_itfs.pa.pa_v1 import paged_attention_v1 as paged_attention_v1_core
from csrc.cpp_itfs.torch_utils import direct_register_custom_op
from aiter.ops.triton.gluon.pa_decode_gluon import pa_decode_gluon

from aiter import dtypes

from ..jit.core import compile_ops

MD_NAME = "module_attention"

direct_register_custom_op(
    "pa_decode_gluon",
    pa_decode_gluon,
    ["output", "exp_sums", "max_logits", "temporary_output"],
)


def gen_pa_fwd_native_fake(
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
) -> torch.Tensor:
    if out is not None:
        return out
    else:
        return torch.empty_like(query)


def gen_pa_fwd_asm(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    block_tables_stride0: int,
    max_qlen: int = 1,
    K_QScale: Optional[torch.Tensor] = None,
    V_QScale: Optional[torch.Tensor] = None,
    out_: Optional[torch.Tensor] = None,
    qo_indptr: Optional[torch.Tensor] = None,
    high_precision: Optional[
        int
    ] = 1,  # [0, 1, 2] 2 is the highest precision, this is only for fp8 kvcache
    kernelName: Optional[str] = None,
):
    if out_ is not None:
        return out_
    else:
        return torch.empty_like(Q)


@compile_ops("module_attention", gen_fake=gen_pa_fwd_native_fake)
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


@compile_ops("module_attention_asm", gen_fake=gen_pa_fwd_asm)
def pa_fwd_asm(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    block_tables_stride0: int,
    max_qlen: int = 1,
    K_QScale: Optional[torch.Tensor] = None,
    V_QScale: Optional[torch.Tensor] = None,
    out_: Optional[torch.Tensor] = None,
    qo_indptr: Optional[torch.Tensor] = None,
    high_precision: Optional[
        int
    ] = 1,  # [0, 1, 2] 2 is the highest precision, this is only for fp8 kvcache
    kernelName: Optional[str] = None,
) -> torch.Tensor: ...


def _should_use_asm_kernel(
    num_seqs: int,
    num_heads: int,
    kv_cache_tensor_dtype: torch.dtype,
) -> bool:

    if kv_cache_tensor_dtype == torch.int8:
        return True

    # Get GPU compute units (CUs)
    gpu = torch.cuda.current_device()
    device_properties = torch.cuda.get_device_properties(gpu)
    cu_num = device_properties.multi_processor_count
    # ASM kernel becomes relevant, once the total_heads is sufficiently large compared to CUs
    total_heads = num_seqs * num_heads
    return total_heads > 2 * cu_num


def paged_attention_common(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    exp_sums: torch.Tensor,
    max_logits: torch.Tensor,
    tmp_out: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    block_tables_stride0: int,
    scale: float,
    max_qlen: int = 1,
    max_seq_len: int = 1,
    K_QScale_hip: Optional[torch.Tensor] = None,  # [num_seqs, num_heads]
    V_QScale_hip: Optional[torch.Tensor] = None,
    K_QScale_asm: Optional[
        torch.Tensor
    ] = None,  # [num_blocks, num_kv_heads, block_size]
    V_QScale_asm: Optional[torch.Tensor] = None,
    out_: Optional[torch.Tensor] = None,
    qo_indptr: Optional[torch.Tensor] = None,
    high_precision: Optional[
        int
    ] = 1,  # [0, 1, 2] 2 is the highest precision, this is only for fp8 kvcache
    kernelName: Optional[str] = None,
    kv_cache_dtype: str = "auto",
    kv_cache_tensor_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """
    Paged attention forward pass with automatic kernel selection.
    ASM is favored for int8 kv caches, for short ctx_len, or when the workload exceeds
    the heuristic thresholds for larger ctx_len values.
    PA is normally using per tensor quant and this is what has been tested, however,
    per head quant can be supported as well in principle, but not tested.
    """
    kv_cache_tensor_dtype = (
        kv_cache_tensor_dtype if kv_cache_tensor_dtype is not None else K.dtype
    )
    num_seqs, num_heads, head_size = Q.shape

    use_asm_kernel = (
        _should_use_asm_kernel(num_seqs, num_heads, kv_cache_tensor_dtype)
        or high_precision == 2
    )

    if use_asm_kernel:
        output = pa_fwd_asm(
            Q,
            K,
            V,
            block_tables,
            context_lens,
            block_tables_stride0,
            max_qlen,
            K_QScale_asm,
            V_QScale_asm,
            out_,
            qo_indptr,
            high_precision,
            kernelName,
        )
        return output

    # Use ROCm paged attention kernel for smaller workloads / common path.
    output = out_ if out_ is not None else torch.empty_like(Q)

    paged_attention_rocm(
        out=output,
        exp_sums=exp_sums,
        max_logits=max_logits,
        tmp_out=tmp_out,
        query=Q,
        key_cache=K,
        value_cache=V,
        num_kv_heads=int(K.size(1)),
        scale=scale,
        block_tables=block_tables,
        context_lens=context_lens,
        block_size=int(K.size(3)),
        max_context_len=max_seq_len,
        alibi_slopes=None,
        kv_cache_dtype=kv_cache_dtype,
        k_scale=K_QScale_hip,
        v_scale=V_QScale_hip,
        fp8_out_scale=None,
        partition_size=256,
        mtp=1,
        q_scale=None,
    )
    return output


def gen_pa_ps_fwd_asm(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_page_indices: torch.Tensor,
    context_lens: torch.Tensor,
    softmax_scale: float,  # better have ?
    max_qlen: int = 1,
    K_QScale: Optional[torch.Tensor] = None,
    V_QScale: Optional[torch.Tensor] = None,
    out_: Optional[torch.Tensor] = None,
    qo_indptr: Optional[torch.Tensor] = None,
    # work_meta_data: Optional[torch.Tensor] = None,
    work_indptr: Optional[torch.Tensor] = None,
    work_info: Optional[torch.Tensor] = None,
    splitData: Optional[torch.Tensor] = None,
    splitLse: Optional[torch.Tensor] = None,
    high_precision: Optional[
        int
    ] = 1,  # [0, 1, 2] 2 is the highest precision, this is only for fp8 kvcache
    kernelName: Optional[str] = None,
) -> torch.Tensor:
    if out_ is not None:
        return out_
    else:
        return torch.empty_like(Q)


@compile_ops("module_attention_asm", gen_fake=gen_pa_fwd_asm)
def pa_ps_fwd_asm(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_page_indices: torch.Tensor,
    context_lens: torch.Tensor,
    softmax_scale: float,  # better have ?
    max_qlen: int = 1,
    K_QScale: Optional[torch.Tensor] = None,
    V_QScale: Optional[torch.Tensor] = None,
    out_: Optional[torch.Tensor] = None,
    qo_indptr: Optional[torch.Tensor] = None,
    # work_meta_data: Optional[torch.Tensor] = None,
    work_indptr: Optional[torch.Tensor] = None,
    work_info: Optional[torch.Tensor] = None,
    splitData: Optional[torch.Tensor] = None,
    splitLse: Optional[torch.Tensor] = None,
    mask: int = 0,
    high_precision: Optional[
        int
    ] = 1,  # [0, 1, 2] 2 is the highest precision, this is only for fp8 kvcache
    kernelName: Optional[str] = None,
) -> torch.Tensor: ...


def pa_reduce_v1(
    partial_output: torch.Tensor,
    partial_lse: torch.Tensor,
    reduce_indptr: torch.Tensor,
    reduce_final_map: Optional[torch.Tensor],
    reduce_partial_map: torch.Tensor,
    max_seqlen_q: int,
    final_output: torch.Tensor,
    final_lse: Optional[torch.Tensor] = None,
) -> None:
    mla_reduce_v1(
        partial_output,
        partial_lse,
        reduce_indptr,
        reduce_final_map,
        reduce_partial_map,
        max_seqlen_q,
        final_output,
        final_lse,
    )


def pa_persistent_fwd(
    Q: torch.Tensor,  # [sum_qlen, kv_heads * gqa + kv_heads * 2, head_dim]
    K: torch.Tensor,  # [num_blocks, kv_heads, head_dim / x, block_size, x]
    V: torch.Tensor,  # [num_blocks, kv_heads, block_size / x, head_dim, x]
    output: torch.Tensor,
    max_qlen: int,  # default = 1
    qo_indptr: torch.Tensor,  # [batch+1], qolen prefix sum
    kv_indptr: torch.Tensor,  # [batch+1], kv_used_pages prefix sum
    kv_indices: torch.Tensor,  # [sum_kv_used_pages], packed kv ids
    context_lens: torch.Tensor,  # [batch]
    # work_meta_data: torch.Tensor,
    work_indptr: torch.Tensor,
    work_info: torch.Tensor,
    reduce_indptr: torch.Tensor,
    reduce_final_map: torch.Tensor,
    reduce_partial_map: torch.Tensor,
    K_QScale: Optional[torch.Tensor] = None,  # [num_blocks, kv_heads, block_size]
    V_QScale: Optional[torch.Tensor] = None,  # [num_blocks, kv_heads, block_size]
    softmax_scale: Optional[float] = None,
    mask: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    device = Q.device
    total_s, nhead, v_head_dim = output.shape
    if softmax_scale is None:
        softmax_scale = 1.0 / (v_head_dim**0.5)
    logits = torch.empty(
        (reduce_partial_map.size(0) * max_qlen, 1, nhead, v_head_dim),
        dtype=dtypes.fp32,
        device=device,
    )
    splitLse = torch.empty(
        (reduce_partial_map.size(0) * max_qlen, 1, nhead, 1),
        dtype=dtypes.fp32,
        device=device,
    )
    final_lse = torch.empty((total_s, nhead), dtype=dtypes.fp32, device=device)

    pa_ps_fwd_asm(
        Q,
        K,
        V,
        kv_indptr,
        kv_indices,
        context_lens,
        softmax_scale,
        max_qlen,
        K_QScale,
        V_QScale,
        output,
        qo_indptr,
        work_indptr,
        work_info,
        logits,
        splitLse,
        mask,
    )
    pa_reduce_v1(
        logits,
        splitLse,
        reduce_indptr,
        reduce_final_map,
        reduce_partial_map,
        max_qlen,
        output,
        final_lse,
    )

    return logits, final_lse


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
    q_scale: Optional[torch.Tensor] = None,
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
        q_scale,
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
    sliding_window: int = 0,
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
        sliding_window=sliding_window,
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
    num_kv_splits_indptr: Optional[torch.Tensor],
    work_meta_data: Optional[torch.Tensor],
    work_indptr: Optional[torch.Tensor],
    work_info_set: Optional[torch.Tensor],
    max_seqlen_q: int,
    page_size: int,
    nhead_kv: int,
    softmax_scale: float,
    # [batch_size, num_kv_splits, num_heads, v_head_dim]
    splitData: torch.Tensor,
    # [batch_size, num_kv_splits, num_heads,  1]
    splitLse: torch.Tensor,
    output: torch.Tensor,
    # [batch_size, num_heads, v_head_dim]
    q_scale: Optional[torch.Tensor] = None,
    kv_scale: Optional[torch.Tensor] = None,
    # [1] pertensor
) -> None: ...


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
) -> None: ...


def get_pa_metadata_info_v1(
    batch_size: int,
    num_head_k: int = 1,
):
    """
    Returns:
        1. Shape of work_metadata_ptrs followed by its scalar type.
        2. Shape of work_indptr followed by its scalar type.
        3. Shape of work_info_set followed by its scalar type.
        4. Shape of reduce_indptr followed by its scalar type.
        5. Shape of reduce_final_map followed by its scalar type.
        6. Shape of reduce_partial_map followed by its scalar type.
    """

    gpu = torch.cuda.current_device()
    device_properties = torch.cuda.get_device_properties(gpu)
    cu_num = device_properties.multi_processor_count

    tile_cnt = batch_size
    max_work = (tile_cnt + cu_num - 1) * num_head_k
    max_split_tiles = min(batch_size + cu_num - 1, (cu_num - 1) * 2)

    return (
        ((2), torch.uint64),  # work_metadata_ptrs
        ((cu_num + 1), torch.int32),  # work_indptr
        ((max_work, 8), torch.int32),  # work_info_set
        ((tile_cnt + 1), torch.int32),  # reduce_indptr
        ((tile_cnt, 2), torch.int32),  # reduce_final_map
        (max_split_tiles, torch.int32),  # reduce_partial_map
    )


@compile_ops("module_pa_metadata")
def get_pa_metadata_v1(
    seqlens_qo_indptr: torch.Tensor,
    pages_kv_indptr: torch.Tensor,
    context_lens: torch.Tensor,
    num_heads_per_head_k: int,
    num_heads_k: int,
    is_causal: bool,
    work_metadata_ptrs: torch.Tensor,
    work_indptr: torch.Tensor,
    work_info: torch.Tensor,
    reduce_indptr: torch.Tensor,
    reduce_final_map: torch.Tensor,
    reduce_partial_map: torch.Tensor,
    kv_granularity: int = 16,
    block_size: int = 16,
    max_seqlen_qo: int = -1,
    uni_seqlen_qo: int = -1,
    fast_mode: bool = True,
    topk: int = -1,
    max_split_per_batch: int = -1,
) -> None:
    """
    Inputs:
        cumulated seqlens of q/o: (batch_size + 1), dtype torch.int32.
        cumulated used pages of k/v: (batch_size + 1), dtype torch.int32.
        context_lens: seqlens of k/v, dtype torch.int32.
        num_heads_per_head_k: Equals to num_heads_q // num_heads_k.
        num_heads_k: num_heads_k.
        is_causal: Whether causal mask is enabled.
        Options: Detailed settings for spliting. All of them are optional.
            kv_granularity: default=16. The granularity on kv sequence length when cutting batch.
            max_seqlen_qo: default=-1. Used to check lds usage and save time. value less than 1 means unknown.
            uni_seqlen_qo: default=-1. Sequence length of qo is uniform across batches. value less than 1 means the
                           length is not fixed.
            fast_mode: default=True. Whether user wants metadata become as fast as possible. Note that fast
                       mode may lead to bad overall performance.
            topk: default=-1. Top-k tokens selected for sparse attention. -1 means non-sparse attention.
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
        [2.4] kv_start:         (#work),            The global index in kv_indices where k/v starts.
        [2.5] kv_end:           (#work),            The global index in kv_indices where k/v ends (not included). Note
                                                    that this value indicates the end of last qo sequence if there are
                                                    multiple qo sequences included in the current work and causal mask
                                                    is enabled.
        [2.6] kv_offset:        (#work),            Not used.
        [2.7] pad               (#work, 1),         The start index(low 16bits) and end index(high 16bits) of q heads.
        [3] reduce_indptr:      (sum(qo_seqlen_blk_count) + 1),
                                                    The IDs in reduce_partial_map indicates the tiles should be merged
                                                    together.
        [4] reduce_final_map:   (sum(qo_seqlen_blk_count)),
                                                    The final output location of each group of tiles.
        [5] reduce_partial_map: (#partial_tiles),   The locations in partial buffer of partial tiles waiting for being
                                                    reduced.
    """
    ...


def get_ps_metadata_info_v1(
    batch_size: int,
    num_head_k: int,
    max_qlen: int,
    qlen_granularity: int = 256,
):
    """
    Returns:
        1. Shape of work_metadata_ptrs followed by its scalar type.
        2. Shape of work_indptr followed by its scalar type.
        3. Shape of work_info followed by its scalar type.
        4. Shape of reduce_indptr followed by its scalar type.
        5. Shape of reduce_final_map followed by its scalar type.
        6. Shape of reduce_partial_map followed by its scalar type.
    """

    device = torch.cuda.current_device()
    device_properties = torch.cuda.get_device_properties(device)
    cu_num = device_properties.multi_processor_count

    num_clusters = math.gcd(num_head_k, cu_num)
    cus_per_cluster = cu_num // num_clusters

    max_qo_split_per_batch = math.ceil(max_qlen / qlen_granularity)

    qo_tile_cnt = batch_size * max_qo_split_per_batch
    # TODO: consider split q to reduce max_works & max_partials
    max_works = (batch_size + cus_per_cluster - 1) * max_qo_split_per_batch * num_head_k
    max_partials = (
        min(batch_size + cus_per_cluster - 1, (cus_per_cluster - 1) * 2)
        * max_qo_split_per_batch
    )

    return (
        (2, torch.uint64),  # work_metadata_ptrs
        (cu_num + 1, torch.int32),  # work_indptr
        ((max_works, 8), torch.int32),  # work_info
        (qo_tile_cnt + 1, torch.int32),  # reduce_indptr
        ((qo_tile_cnt, 2), torch.int32),  # reduce_final_map
        (max_partials, torch.int32),  # reduce_partial_map
    )


@compile_ops("module_ps_metadata")
def get_ps_metadata_v1(
    seqlens_qo_indptr: torch.Tensor,
    pages_kv_indptr: torch.Tensor,
    context_lens: torch.Tensor,
    gqa_ratio: int,
    num_heads_k: int,
    work_metadata_ptrs: torch.Tensor,
    work_indptr: torch.Tensor,
    work_info: torch.Tensor,
    reduce_indptr: torch.Tensor,
    reduce_final_map: torch.Tensor,
    reduce_partial_map: torch.Tensor,
    qhead_granularity: int = 1,
    qlen_granularity: int = 256,
    kvlen_granularity: int = 16,
    block_size: int = 16,
    is_causal: bool = True,
) -> None: ...


@compile_ops(MD_NAME)
def mla_prefill_ps_asm_fwd(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    qo_indptr: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_page_indices: torch.Tensor,
    work_indptr: Optional[torch.Tensor],
    work_info_set: Optional[torch.Tensor],
    max_seqlen_q: int,
    softmax_scale: float,
    is_causal: bool,
    splitData: torch.Tensor,
    splitLse: torch.Tensor,
    output: torch.Tensor,
    q_scale: Optional[torch.Tensor] = None,
    k_scale: Optional[torch.Tensor] = None,
    v_scale: Optional[torch.Tensor] = None,
) -> None: ...


def get_mla_metadata_info_v1(
    batch_size: int,
    max_seqlen_qo: int,
    num_head_qo: int,
    q_dtype: torch.dtype,
    kv_dtype: torch.dtype,
    is_sparse: bool,
    fast_mode: bool = True,
    num_kv_splits: int = 32,
    intra_batch_mode: bool = False,
):
    """
    Returns:
        1. Shape of work_metadata_ptrs followed by its scalar type.
        2. Shape of work_indptr followed by its scalar type.
        3. Shape of work_info_set followed by its scalar type.
        4. Shape of reduce_indptr followed by its scalar type.
        5. Shape of reduce_final_map followed by its scalar type.
        6. Shape of reduce_partial_map followed by its scalar type.
    """

    assert num_head_qo % 16 == 0
    gpu = torch.cuda.current_device()
    device_properties = torch.cuda.get_device_properties(gpu)
    cu_num = device_properties.multi_processor_count

    max_qo_tiles_per_batch = (
        int(math.ceil(max_seqlen_qo * num_head_qo / 128))
        if num_head_qo == 16
        or (num_head_qo == 128 and kv_dtype == dtypes.fp8 and q_dtype == dtypes.fp8)
        else int(math.ceil(max_seqlen_qo * num_head_qo / 16))
    )
    batch_size = batch_size * max_seqlen_qo if is_sparse else batch_size
    tile_cnt = batch_size * max_qo_tiles_per_batch

    if fast_mode:
        max_work = (batch_size + cu_num - 1) * max_qo_tiles_per_batch
        max_split_tiles = (
            min(batch_size + cu_num - 1, (cu_num - 1) * 2) * max_qo_tiles_per_batch
        )
    else:
        max_work = tile_cnt * cu_num
        max_split_tiles = tile_cnt * cu_num

    if not intra_batch_mode:
        return (
            ((2), torch.uint64),  # work_metadata_ptrs
            ((cu_num + 1), torch.int32),  # work_indptr
            ((max_work, 8), torch.int32),  # work_info_set
            ((tile_cnt + 1), torch.int32),  # reduce_indptr
            ((tile_cnt, 2), torch.int32),  # reduce_final_map
            (max_split_tiles, torch.int32),  # reduce_partial_map
        )
    else:
        return (
            ((2), torch.uint64),  # work_metadata_ptrs
            (cu_num + 1, torch.int32),  # work_indptr
            ((tile_cnt * num_kv_splits, 8), torch.int32),  # work_info_set
            ((tile_cnt + 1), torch.int32),  # reduce_indptr
            ((tile_cnt, 2), torch.int32),  # reduce_final_map
            (tile_cnt * num_kv_splits, torch.int32),  # reduce_partial_map
        )


@compile_ops("module_mla_metadata")
def get_mla_metadata_v1(
    seqlens_qo_indptr: torch.Tensor,
    seqlens_kv_indptr: torch.Tensor,
    kv_last_page_lens: torch.Tensor,
    num_heads_per_head_k: int,
    num_heads_k: int,
    is_causal: bool,
    work_metadata_ptrs: torch.Tensor,
    work_info_set: torch.Tensor,
    work_indptr: torch.Tensor,
    reduce_indptr: torch.Tensor,
    reduce_final_map: torch.Tensor,
    reduce_partial_map: torch.Tensor,
    page_size: int = 1,
    kv_granularity: int = 16,
    max_seqlen_qo: int = -1,
    uni_seqlen_qo: int = -1,
    fast_mode: bool = True,
    topk: int = -1,
    max_split_per_batch: int = -1,
    intra_batch_mode: bool = False,
    dtype_q: Optional[torch.dtype] = None,
    dtype_kv: Optional[torch.dtype] = None,
) -> None:
    """
    Inputs:
        cumulated seqlens of q/o: (batch_size + 1), dtype torch.int32.
        cumulated page indices of k/v: (batch_size + 1), dtype torch.int32.
        Length of last page of k/v: (batch_size), dtype torch.int32.
        num_heads_per_head_k: Equals to num_heads_q // num_heads_k.
        num_heads_k: num_heads_k.
        is_causal: Whether causal mask is enabled.
        Options: Detailed settings for spliting. All of them are optional.
            page_size: default=1. The size of a page.
            kv_granularity: default=16. The granularity on kv page nums when cutting batch.
            max_seqlen_qo: default=-1. Used to check lds usage and save time. value less than 1 means unknown.
            uni_seqlen_qo: default=-1. Sequence length of qo is uniform across batches. value less than 1 means the
                           length is not fixed.
            fast_mode: default=True. Whether user wants metadata become as fast as possible. Note that fast
                       mode may lead to bad overall performance.
            intra_batch_mode: default=False. Fake non persistent mode. Same splits for each batch.
            topk: default=-1. Top-k tokens selected for sparse attention. -1 means non-sparse attention.
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
        [2.4] kv_start:         (#work),            The global index in page where k/v starts.
        [2.5] kv_end:           (#work),            The global index in page where k/v ends (not included). Note that
                                                    this value indicates the end of last qo sequence if there are
                                                    multiple qo sequences included in the current work and causal mask
                                                    is enabled when page_size is 1.
        [2.6] kv_offset:        (#work),            Remaining length in seq from kv_end to the end of current batch.
        [2.7] pad               (#work, 1),         Pad to 8 DWs.
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
    kv_granularity: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Arguments:
        cumulated seqlens of q/o: (batch_size + 1), dtype torch.int32.
        cumulated seqlens of k/v: (batch_size + 1), dtype torch.int32.
        num_heads_per_head_k: Equals to num_heads_q // num_heads_k.
        num_heads_k: num_heads_k.
        is_causal: whether causal mask is enabled.
        kv_granularity: the granularity on kv sequence length when cutting batch.
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
    reduce_final_map: Optional[torch.Tensor],
    reduce_partial_map: torch.Tensor,
    max_seqlen_q: int,
    final_output: torch.Tensor,
    final_lse: Optional[torch.Tensor] = None,
) -> None: ...
