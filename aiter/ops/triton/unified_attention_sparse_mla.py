from aiter.ops.triton._triton_kernels.unified_attention_sparse_mla import (
    _kernel_unified_attention_sparse_mla_2d,
)


def unified_attention_sparse_mla(
    q,
    kv,
    out,
    cu_seqlens_q,
    max_seqlen_q,
    seqused_k,
    max_seqlen_k,
    softmax_scale,
    topk_indices,
    block_table,
    kv_lora_rank,
):
    """
    This function computes the sparse attention.

    Note: topk_indices index the KV cache, not block_table.

    Q:             [seq_len, NUM_HEADS, kv_lora_rank + rope_rank], dtype bfloat16
    KV:            [seq_len_kv, 1, kv_lora_rank + rope_rank], dtype bfloat16
    cu_seqlens_q:  [BATCH + 1], dtype int32
    max_seqlen_q:  scalar, dtype int32
    max_seqlen_k:  scalar, dtype int32
    softmax_scale: scalar, dtype float32
    topk_indices:  [seq_len, TOP_K], dtype int32
    block_table:   [BATCH, MAX_NUM_BLOCKS_PER_BATCH], dtype int32
    kv_lora_rank:  scalar, dtype int32

    Returns:
    out (in-place):  [seq_len, NUM_HEADS, kv_lora_rank], dtype bfloat16
    """

    # TODO: This kernel is not optimized and simplified for initial development.

    block_size = kv.shape[1]
    num_seqs = len(seqused_k)
    num_query_heads = q.shape[1]
    num_kv_heads = 1
    num_queries_per_kv = num_query_heads // num_kv_heads
    head_size = q.shape[2]
    topk_count = topk_indices.shape[1]
    k = kv
    v = kv[..., :kv_lora_rank]

    BLOCK_M = 16

    total_num_q_blocks = q.shape[0] * (num_query_heads // BLOCK_M)
    ALL_DECODE = max_seqlen_q == 1

    ROPE_RANK = head_size - kv_lora_rank
    KV_LORA_RANK = kv_lora_rank
    TILE_SIZE = block_size
    num_stages_2d = 1
    num_warps = 4
    _kernel_unified_attention_sparse_mla_2d[(total_num_q_blocks,)](
        output_ptr=out,
        query_ptr=q,
        key_cache_ptr=k,
        value_cache_ptr=v,
        block_tables_ptr=block_table,
        topk_indices_ptr=topk_indices,
        seq_lens_ptr=seqused_k,
        scale=softmax_scale,
        num_query_heads=num_query_heads,
        num_queries_per_kv=num_queries_per_kv,
        block_table_stride=block_table.stride(0),
        query_stride_0=q.stride(0),
        query_stride_1=q.stride(1),
        output_stride_0=out.stride(0),
        output_stride_1=out.stride(1),
        BLOCK_SIZE=block_size,
        stride_k_cache_0=k.stride(0),
        stride_k_cache_1=k.stride(1),
        stride_k_cache_2=k.stride(2),
        stride_k_cache_3=k.stride(3),
        stride_v_cache_0=v.stride(0),
        stride_v_cache_1=v.stride(1),
        stride_v_cache_2=v.stride(2),
        stride_v_cache_3=v.stride(3),
        topk_count=topk_count,
        query_start_len_ptr=cu_seqlens_q,
        num_seqs=num_seqs,
        BLOCK_M=BLOCK_M,
        ROPE_RANK=ROPE_RANK,
        KV_LORA_RANK=KV_LORA_RANK,
        TILE_SIZE=TILE_SIZE,
        ALL_DECODE=ALL_DECODE,
        num_warps=num_warps,
        num_stages=num_stages_2d,
    )
