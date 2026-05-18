import torch

from aiter.ops.triton._triton_kernels.attention.unified_attention_sparse_mla_fp8 import (
    _kernel_unified_attention_sparse_mla_2d,
    _kernel_unified_attention_sparse_mla_csr_2d,
)


def _coerce_scale(s, device):
    """Coerce Python scalars into 0-d float32 tensors on the right device.

    SGLang sometimes passes Python ``int``/``float`` for scale args; the
    kernel expects a tensor pointer. ``None`` is preserved (compile-time
    no-op via the ``Q_SCALE/K_SCALE/V_SCALE`` constexpr flags).
    """
    if s is None:
        return None
    if isinstance(s, (int, float)):
        return torch.tensor(s, dtype=torch.float32, device=device)
    return s


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
    kv_indptr=None,
    kv_indices=None,
    max_sparse_len=None,
    q_scale=None,
    k_scale=None,
    v_scale=None,
):
    """
    This function computes the sparse attention.

    Note: topk_indices index the KV cache, not block_table.

    Q:             [seq_len, NUM_HEADS, kv_lora_rank + rope_rank], dtype bfloat16
    KV:            [seq_len_kv, 1, kv_lora_rank + rope_rank], dtype bfloat16 or fp8_e4m3
    cu_seqlens_q:  [BATCH + 1], dtype int32
    max_seqlen_q:  scalar, dtype int32
    max_seqlen_k:  scalar, dtype int32
    softmax_scale: scalar, dtype float32
    topk_indices:  [seq_len, TOP_K], dtype int32
    kv_indptr:     Optional [seq_len + 1], dtype int32
    kv_indices:    Optional [nnz], dtype int32
    block_table:   [BATCH, MAX_NUM_BLOCKS_PER_BATCH], dtype int32
    kv_lora_rank:  scalar, dtype int32
    q_scale:       Optional scalar tensor (or python float) for per-tensor FP8 Q scale
    k_scale:       Optional scalar tensor (or python float) for per-tensor FP8 K scale
    v_scale:       Optional scalar tensor (or python float) for per-tensor FP8 V scale

    Returns:
    out (in-place):  [seq_len, NUM_HEADS, kv_lora_rank], dtype bfloat16
    """

    # TODO: This kernel is not optimized and simplified for initial development.
    use_csr = kv_indptr is not None or kv_indices is not None
    if use_csr:
        assert kv_indptr is not None and kv_indices is not None
        # SGLang allocates kv_indptr with max_bs+1 entries; only the first
        # q.shape[0]+1 are meaningful. Accept >= to support both shapes.
        assert kv_indptr.shape[0] >= q.shape[0] + 1
        if max_sparse_len is None:
            row_lens = kv_indptr[1 : q.shape[0] + 1] - kv_indptr[: q.shape[0]]
            max_sparse_len = int(row_lens.max().item())
    else:
        assert topk_indices is not None

    # Coerce python scalars (SGLang sometimes passes float) into device tensors.
    q_scale_t = _coerce_scale(q_scale, q.device)
    k_scale_t = _coerce_scale(k_scale, q.device)
    v_scale_t = _coerce_scale(v_scale, q.device)
    Q_SCALE = q_scale_t is not None
    K_SCALE = k_scale_t is not None
    V_SCALE = v_scale_t is not None

    block_size = kv.shape[1]
    num_seqs = len(seqused_k)
    num_query_heads = q.shape[1]
    num_kv_heads = 1
    num_queries_per_kv = num_query_heads // num_kv_heads
    head_size = q.shape[2]
    topk_count = topk_indices.shape[1] if topk_indices is not None else 0
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
    if use_csr:
        _kernel_unified_attention_sparse_mla_csr_2d[(total_num_q_blocks,)](
            output_ptr=out,
            query_ptr=q,
            key_cache_ptr=k,
            value_cache_ptr=v,
            kv_indptr_ptr=kv_indptr,
            kv_indices_ptr=kv_indices,
            seq_lens_ptr=seqused_k,
            scale=softmax_scale,
            q_scale=q_scale_t,
            k_scale=k_scale_t,
            v_scale=v_scale_t,
            num_query_heads=num_query_heads,
            num_queries_per_kv=num_queries_per_kv,
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
            max_sparse_len=max_sparse_len,
            query_start_len_ptr=cu_seqlens_q,
            num_seqs=num_seqs,
            BLOCK_M=BLOCK_M,
            ROPE_RANK=ROPE_RANK,
            KV_LORA_RANK=KV_LORA_RANK,
            TILE_SIZE=TILE_SIZE,
            ALL_DECODE=ALL_DECODE,
            Q_SCALE=Q_SCALE,
            K_SCALE=K_SCALE,
            V_SCALE=V_SCALE,
            num_warps=num_warps,
            num_stages=num_stages_2d,
        )
        return

    _kernel_unified_attention_sparse_mla_2d[(total_num_q_blocks,)](
        output_ptr=out,
        query_ptr=q,
        key_cache_ptr=k,
        value_cache_ptr=v,
        block_tables_ptr=block_table,
        topk_indices_ptr=topk_indices,
        seq_lens_ptr=seqused_k,
        scale=softmax_scale,
        q_scale=q_scale_t,
        k_scale=k_scale_t,
        v_scale=v_scale_t,
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
        Q_SCALE=Q_SCALE,
        K_SCALE=K_SCALE,
        V_SCALE=V_SCALE,
        num_warps=num_warps,
        num_stages=num_stages_2d,
    )
