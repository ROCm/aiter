from aiter.ops.triton._triton_kernels.attention.unified_attention_sparse_mla import (
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
    attn_sink=None,
    return_lse=False,
    lse_out=None,  # if provided, writes lse into this tensor (else allocates one when return_lse=True)
    kv_scales=None,  # optional [num_blocks, block_size, num_tiles] uint8 (e8m0 byte view).
                     # When provided, kv must be FP8 storage and the kernel dequants in-place.
    fp8_tile_size=64,
    # V4 split storage: optional trailing BF16 channels (e.g. RoPE) stored separately.
    # When provided alongside kv_scales:
    #   - kv covers the leading FP8 segment of size `fp8_segment_dim` channels
    #     within the rounded-up `kv_lora_rank` channel space
    #   - (k_bf16, v_bf16) cover the trailing BF16 segment of size `bf16_head_dim`
    #     channels, occupying [fp8_segment_dim, fp8_segment_dim + bf16_head_dim)
    #     in the output / Q channel space
    # Q is contiguous BF16 spanning [0, kv_lora_rank). For V4: kv_lora_rank=512,
    # fp8_segment_dim=448, bf16_head_dim=64.
    k_bf16=None,
    v_bf16=None,
    bf16_head_dim=0,
    fp8_segment_dim=0,
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

    ALL_DECODE = max_seqlen_q == 1
    # Tuned on gfx950 / DSV4 shapes (tune_prefill_v4.py + tune_sparse_mla_v4.py):
    #   decode (ALL_DECODE): BLOCK_M=16 fastest across batch=1..32, topk=256/512/1024
    #   prefill (varlen)   : BLOCK_M=64 gives 1.54x-1.85x over BLOCK_M=16 across
    #                        SWA(256)/HCA(768)/CSA(1152) at q=128/512/2048.
    # Must satisfy num_query_heads % BLOCK_M == 0 (else (heads // BLOCK_M) == 0 and
    # no programs launch). Fall back to BLOCK_M=16 for small head counts.
    if ALL_DECODE or num_query_heads % 64 != 0:
        BLOCK_M = 16
    else:
        BLOCK_M = 64

    total_num_q_blocks = q.shape[0] * (num_query_heads // BLOCK_M)

    # lse output: [num_tokens, num_query_heads] FP32 (sink-FREE; sink is folded
    # back in by the caller for two-pool merge if needed).
    if return_lse and lse_out is None:
        import torch
        lse_out = torch.empty(
            (q.shape[0], num_query_heads), dtype=torch.float32, device=q.device
        )
    if lse_out is not None:
        lse_stride_0 = lse_out.stride(0)
        lse_stride_1 = lse_out.stride(1)
    else:
        lse_stride_0 = 0
        lse_stride_1 = 0

    ROPE_RANK = head_size - kv_lora_rank
    KV_LORA_RANK = kv_lora_rank
    TILE_SIZE = block_size
    num_stages_2d = 1
    num_warps = 4

    KV_FP8 = kv_scales is not None
    if KV_FP8:
        # Caller's view: kv is FP8 covering the FP8 segment (size KV_LORA_RANK)
        # of each cache row. kv_scales is uint8 (e8m0 byte view), shape
        # [num_blocks, block_size, KV_LORA_RANK // fp8_tile_size]. Strides are
        # passed raw — Triton applies them in fp8/uint8 element units (both 1
        # byte, so byte-stride math works out for V4's interleaved layout).
        kv_scales_stride_0 = kv_scales.stride(0)
        kv_scales_stride_1 = kv_scales.stride(1)
        kv_scales_stride_2 = kv_scales.stride(2)
    else:
        kv_scales_stride_0 = 0
        kv_scales_stride_1 = 0
        kv_scales_stride_2 = 0

    # V4 split storage: when bf16_head_dim > 0, the kernel does an additional
    # QK and PV partial dot from the trailing BF16 cache segment. Q's last
    # bf16_head_dim channels live contiguously in q after the FP8 lora channels.
    if bf16_head_dim > 0:
        assert k_bf16 is not None and v_bf16 is not None, (
            "k_bf16 and v_bf16 must be provided when bf16_head_dim > 0"
        )
        stride_k_bf16_0 = k_bf16.stride(0)
        stride_k_bf16_1 = k_bf16.stride(1)
        stride_k_bf16_2 = k_bf16.stride(2)
        stride_k_bf16_3 = k_bf16.stride(3)
        stride_v_bf16_0 = v_bf16.stride(0)
        stride_v_bf16_1 = v_bf16.stride(1)
        stride_v_bf16_2 = v_bf16.stride(2)
        stride_v_bf16_3 = v_bf16.stride(3)
    else:
        stride_k_bf16_0 = 0
        stride_k_bf16_1 = 0
        stride_k_bf16_2 = 0
        stride_k_bf16_3 = 1
        stride_v_bf16_0 = 0
        stride_v_bf16_1 = 0
        stride_v_bf16_2 = 0
        stride_v_bf16_3 = 1

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
        attn_sink_ptr=attn_sink,
        HAS_ATTN_SINK=attn_sink is not None,
        lse_ptr=lse_out,
        lse_stride_0=lse_stride_0,
        lse_stride_1=lse_stride_1,
        RETURN_LSE=lse_out is not None,
        kv_scales_cache_ptr=kv_scales,
        stride_kv_scales_0=kv_scales_stride_0,
        stride_kv_scales_1=kv_scales_stride_1,
        stride_kv_scales_2=kv_scales_stride_2,
        KV_FP8=KV_FP8,
        FP8_TILE=fp8_tile_size,
        key_bf16_cache_ptr=k_bf16,
        value_bf16_cache_ptr=v_bf16,
        stride_k_bf16_0=stride_k_bf16_0,
        stride_k_bf16_1=stride_k_bf16_1,
        stride_k_bf16_2=stride_k_bf16_2,
        stride_k_bf16_3=stride_k_bf16_3,
        stride_v_bf16_0=stride_v_bf16_0,
        stride_v_bf16_1=stride_v_bf16_1,
        stride_v_bf16_2=stride_v_bf16_2,
        stride_v_bf16_3=stride_v_bf16_3,
        BF16_HEAD_DIM=bf16_head_dim,
        FP8_SEGMENT_DIM=fp8_segment_dim,
        num_warps=num_warps,
        num_stages=num_stages_2d,
    )
    return lse_out
