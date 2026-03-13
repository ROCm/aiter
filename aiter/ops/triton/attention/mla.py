# The kernels in this file are adapted from vLLM:
# https://github.com/vllm-project/vllm/blob/main/vllm/attention/ops/triton_unified_attention.py
import triton
import torch
from aiter.ops.triton.utils.device_info import get_num_sms
import math
from aiter.ops.triton._triton_kernels.attention.mla import (
    _mla_2d_kernel,
    _mla_3d_kernel,
    _mla_3d_reduce_kernel,
)


def select_2d_config(
    block_size,
    head_size,
    max_seqlen_k,
    num_queries_per_kv,
    num_2d_prgms,
):
    TILE_SIZE = block_size
    num_stages_2d = 1
    num_warps = 4

    return {
        "TILE_SIZE": TILE_SIZE,
        "num_warps": num_warps,
        "num_stages": num_stages_2d,
        "waves_per_eu": 2,
    }


def select_3d_config(block_size, max_seqlen_k, target_num_prgms, num_2d_prgms):
    reduce_num_warps = 2
    attn_warps = 2
    TILE_SIZE = block_size
    MAX_SEGMENTS = min(128, math.ceil(max_seqlen_k / TILE_SIZE))
    num_segments = math.ceil(target_num_prgms / num_2d_prgms)
    num_segments = min(num_segments, MAX_SEGMENTS)
    num_segments = triton.next_power_of_2(num_segments)
    num_segments = min(num_segments, 128)
    MIN_SEGMENTS = 16 if TILE_SIZE <= 16 else 8
    num_segments = max(num_segments, MIN_SEGMENTS)
    if num_segments == MIN_SEGMENTS:
        reduce_num_warps = 1
    attn_config = {
        "TILE_SIZE": TILE_SIZE,
        "NUM_SEGMENTS_PER_SEQ": num_segments,
        "num_warps": attn_warps,
        "num_stages": 2,
        "waves_per_eu": 2,
    }
    reduce_config = {
        "TILE_SIZE": TILE_SIZE,
        "NUM_SEGMENTS_PER_SEQ": num_segments,
        "num_warps": reduce_num_warps,
        "num_stages": 1,
        "waves_per_eu": 2,
    }
    return attn_config, reduce_config


def use_2d_kernel(
    head_size,
    sliding_window,
    all_decode,
    max_seqlen_q,
    max_seqlen_k,
    target_num_prgms,
    num_2d_prgms,
):
    return (
        (sliding_window > 0)
        or (max_seqlen_k <= 512)
        or (num_2d_prgms > target_num_prgms)
    )


def mla_prefill_fwd(
    q,  # [num_tokens_per_seq * num_seqs, num_query_heads, qk_lora_rank + qk_rope_head_dim]
    kv_buffer,  # [num_blocks, block_size, num_kv_heads, qk_lora_rank + qk_rope_head_dim]
    out,
    cu_seqlens_q,  # [num_seqs + 1]
    seqused_k,  # [num_seqs]
    max_seqlen_kv: int,
    block_tables,  # [batch_size, max_num_blocks_per_seq]
    softmax_scale: float,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    causal: bool,
    q_descale=None,
    kv_descale=None,
    out_scale=None,
):
    assert causal, "Only causal attention is supported"

    total_num_tokens, num_query_heads, qk_head_dim = q.shape
    num_blocks, block_size, num_kv_heads, _ = kv_buffer.shape
    num_seqs = len(seqused_k)
    num_queries_per_kv = num_query_heads // num_kv_heads

    assert (
        kv_lora_rank + qk_rope_head_dim == qk_head_dim
    ), "qk_head_dim must be equal to kv_lora_rank + qk_rope_head_dim"

    BLOCK_M = 128
    BLOCK_Q = BLOCK_M // num_queries_per_kv
    assert BLOCK_Q >= 1
    # Ideally we would launch with kernel with:
    # \sum_i[ceil(query_len[i] / BLOCK_Q)] blocks.
    # However, it is slow to realize the query_lens on cpu.
    # Instead we use upper-bound:
    # \sum_i[ceil(query_len[i] / BLOCK_Q)]
    #   <= \sum_i[floor(query_len[i] / BLOCK_Q) + 1]
    #    = \sum_i[floor(query_len[i] / BLOCK_Q)] + num_seqs
    #   <= floor(\sum_i(query_len[i]) / BLOCK_Q) + num_seqs
    #    = floor(q.shape[0] / BLOCK_Q) + num_seqs
    cu_count = get_num_sms()
    total_num_q_blocks = q.shape[0] // BLOCK_Q + num_seqs
    num_2d_prgms = total_num_q_blocks * num_kv_heads
    # if batch contains a prefill
    attn_config = select_2d_config(
        block_size,
        kv_lora_rank,
        max_seqlen_kv,
        num_queries_per_kv,
        num_2d_prgms,
    )

    _mla_2d_kernel[(num_kv_heads, total_num_q_blocks)](
        output_ptr=out,
        query_ptr=q,
        kv_buffer_ptr=kv_buffer,
        block_tables_ptr=block_tables,
        seq_lens_ptr=seqused_k,
        scale=softmax_scale,
        q_scale_ptr=q_descale,
        kv_scale_ptr=kv_descale,
        out_scale_ptr=out_scale,
        num_query_heads=num_query_heads,
        num_queries_per_kv=num_queries_per_kv,
        block_tables_stride=block_tables.stride(0),
        query_stride_0=q.stride(0),
        query_stride_1=q.stride(1),
        output_stride_0=out.stride(0),
        output_stride_1=out.stride(1),
        KV_LORA_RANK=kv_lora_rank,
        QK_ROPE_HEAD_DIM=qk_rope_head_dim,
        stride_kv_buffer_0=kv_buffer.stride(0),
        stride_kv_buffer_1=kv_buffer.stride(1),
        stride_kv_buffer_2=kv_buffer.stride(2),
        stride_kv_buffer_3=kv_buffer.stride(3),
        query_start_len_ptr=cu_seqlens_q,
        num_seqs=num_seqs,
        BLOCK_Q=BLOCK_Q,
        BLOCK_M=BLOCK_M,
        **attn_config,
    )


def mla_decode_fwd(
    q,  # [num_tokens_per_seq * num_seqs, num_query_heads, qk_lora_rank + qk_rope_head_dim]
    kv_buffer,  # [num_blocks, block_size, num_kv_heads, qk_lora_rank + qk_rope_head_dim]
    out,
    cu_seqlens_q,  # [num_seqs + 1]
    seqused_k,  # [num_seqs]
    max_seqlen_kv: int,
    block_tables,  # [batch_size, max_num_blocks_per_seq]
    softmax_scale: float,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    causal: bool,
    q_descale=None,
    kv_descale=None,
    out_scale=None,
):
    assert causal, "Only causal attention is supported"

    total_num_tokens, num_query_heads, qk_head_dim = q.shape
    num_blocks, block_size, num_kv_heads, _ = kv_buffer.shape
    num_seqs = len(seqused_k)
    num_tokens_per_seq = total_num_tokens // num_seqs
    num_queries_per_kv = num_query_heads // num_kv_heads

    assert (
        kv_lora_rank + qk_rope_head_dim == qk_head_dim
    ), "qk_head_dim must be equal to kv_lora_rank + qk_rope_head_dim"

    BLOCK_M = 16
    BLOCK_Q = BLOCK_M // num_queries_per_kv
    assert BLOCK_Q >= 1
    # Ideally we would launch with kernel with:
    # \sum_i[ceil(query_len[i] / BLOCK_Q)] blocks.
    # However, it is slow to realize the query_lens on cpu.
    # Instead we use upper-bound:
    # \sum_i[ceil(query_len[i] / BLOCK_Q)]
    #   <= \sum_i[floor(query_len[i] / BLOCK_Q) + 1]
    #    = \sum_i[floor(query_len[i] / BLOCK_Q)] + num_seqs
    #   <= floor(\sum_i(query_len[i]) / BLOCK_Q) + num_seqs
    #    = floor(q.shape[0] / BLOCK_Q) + num_seqs
    cu_count = get_num_sms()
    ALL_DECODE = num_tokens_per_seq == 1
    if ALL_DECODE:
        total_num_q_blocks = num_seqs
    else:
        total_num_q_blocks = ((num_tokens_per_seq + BLOCK_Q - 1) // BLOCK_Q) * num_seqs
    target_num_prgms = cu_count * 4
    num_2d_prgms = total_num_q_blocks * num_kv_heads
    # if batch contains a prefill

    attn_config, reduce_config = select_3d_config(
        block_size,
        max_seqlen_kv,
        target_num_prgms,
        num_2d_prgms,
    )
    NUM_SEGMENTS = attn_config["NUM_SEGMENTS_PER_SEQ"]
    segm_output = torch.empty(
        total_num_tokens,
        num_query_heads,
        NUM_SEGMENTS,
        triton.next_power_of_2(kv_lora_rank),
        dtype=torch.float32,
        device=q.device,
    )
    segm_max = torch.empty(
        total_num_tokens,
        num_query_heads,
        NUM_SEGMENTS,
        dtype=torch.float32,
        device=q.device,
    )
    segm_expsum = torch.empty(
        total_num_tokens,
        num_query_heads,
        NUM_SEGMENTS,
        dtype=torch.float32,
        device=q.device,
    )

    _mla_3d_kernel[(total_num_q_blocks, num_kv_heads, NUM_SEGMENTS)](
        segm_output_ptr=segm_output,
        segm_max_ptr=segm_max,
        segm_expsum_ptr=segm_expsum,
        query_ptr=q,
        kv_buffer_ptr=kv_buffer,
        block_tables_ptr=block_tables,
        seq_lens_ptr=seqused_k,
        scale=softmax_scale,
        q_scale_ptr=q_descale,
        kv_scale_ptr=kv_descale,
        num_query_heads=num_query_heads,
        num_queries_per_kv=num_queries_per_kv,
        block_tables_stride=block_tables.stride(0),
        query_stride_0=q.stride(0),
        query_stride_1=q.stride(1),
        KV_LORA_RANK=kv_lora_rank,
        QK_ROPE_HEAD_DIM=qk_rope_head_dim,
        stride_kv_buffer_0=kv_buffer.stride(0),
        stride_kv_buffer_1=kv_buffer.stride(1),
        stride_kv_buffer_2=kv_buffer.stride(2),
        stride_kv_buffer_3=kv_buffer.stride(3),
        query_start_len_ptr=cu_seqlens_q,
        num_tokens_per_seq=num_tokens_per_seq,
        BLOCK_Q=BLOCK_Q,
        BLOCK_M=BLOCK_M,
        ALL_DECODE=ALL_DECODE,
        **attn_config,
    )
    _mla_3d_reduce_kernel[(total_num_tokens, num_query_heads)](
        output_ptr=out,
        segm_output_ptr=segm_output,
        segm_max_ptr=segm_max,
        segm_expsum_ptr=segm_expsum,
        seq_lens_ptr=seqused_k,
        out_scale_ptr=out_scale,
        num_seqs=num_seqs,
        num_query_heads=num_query_heads,
        output_stride_0=out.stride(0),
        output_stride_1=out.stride(1),
        block_tables_stride=block_tables.stride(0),
        num_tokens_per_seq=num_tokens_per_seq,
        KV_LORA_RANK=kv_lora_rank,
        query_start_len_ptr=cu_seqlens_q,
        BLOCK_Q=BLOCK_Q,
        **reduce_config,
    )
