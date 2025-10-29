import triton
import triton.language as tl


@triton.jit
def cdiv_fn(x, y):
    return (x + y - 1) // y


@triton.jit
def apply_softcap(S, x):
    Sdiv = S / x
    p1 = tl.exp(Sdiv)
    p2 = tl.exp(-Sdiv)
    return x * (p1 - p2) / (p1 + p2)


@triton.jit
def find_seq_idx(
    query_start_len_ptr,
    target_idx,
    num_seqs,
    BLOCK_Q: tl.constexpr,
    use_q_block_mode: tl.constexpr,
):
    left: tl.int32 = 0
    right = num_seqs
    while left < right:
        mid = (left + right) // 2
        val = tl.load(query_start_len_ptr + mid)
        mid_val = val // BLOCK_Q + mid if use_q_block_mode else val

        if mid_val <= target_idx:
            left = mid + 1
        else:
            right = mid

    return left - 1


@triton.jit
def kernel_unified_attention_sparse_mla_2d(
    output_ptr,  # [num_tokens, num_query_heads, KV_LORA_RANK]
    query_ptr,  # [num_tokens, num_query_heads, KV_LORA_RANK]
    key_cache_ptr,  # [num_blks, blk_size, 1, KV_LORA_RANK + ROPE_RANK]
    value_cache_ptr,  # [num_blks, blk_size, 1, KV_LORA_RANK]
    block_tables_ptr,  # [num_seqs, max_num_blocks_per_seq]
    topk_indices_ptr,  # [num_tokens, topk]
    seq_lens_ptr,  # [num_seqs]
    scale,  # float32
    num_query_heads: tl.constexpr,  # int
    num_queries_per_kv: tl.constexpr,  # int
    block_table_stride: tl.int64,  # int
    query_stride_0: tl.int64,  # int
    query_stride_1: tl.int64,  # int
    output_stride_0: tl.int64,  # int
    output_stride_1: tl.int64,  # int
    BLOCK_SIZE: tl.constexpr,  # int
    stride_k_cache_0: tl.int64,  # int
    stride_k_cache_1: tl.int64,  # int
    stride_k_cache_2: tl.int64,  # int
    stride_k_cache_3: tl.constexpr,  # int
    stride_v_cache_0: tl.int64,  # int
    stride_v_cache_1: tl.int64,  # int
    stride_v_cache_2: tl.int64,  # int
    stride_v_cache_3: tl.constexpr,  # int
    topk_count: tl.constexpr,
    query_start_len_ptr,  # [num_seqs+1]
    num_seqs: tl.int32,
    BLOCK_M: tl.constexpr,  # int
    ROPE_RANK: tl.constexpr,
    KV_LORA_RANK: tl.constexpr,
    TILE_SIZE: tl.constexpr,
    ALL_DECODE: tl.constexpr = False,
):
    """
    TODO(cagri):
    -- Cleanup needed for variables related to num_queries_per_kv
    -- Masking can be simplified
    -- Tests fail when all topk indices are all -1, not likely to be the case in practice probably?
    """
    # only one query per program
    # these can be removed but keeps the kernel similar to the MHA way
    BLOCK_Q: tl.constexpr = 1
    kv_head_idx = 0  # assume there is single kv head

    q_block_global_idx = tl.program_id(0)
    q_ind = q_block_global_idx // (num_query_heads // BLOCK_M)
    head_ind = q_block_global_idx % (num_query_heads // BLOCK_M)
    seq_idx = find_seq_idx(query_start_len_ptr, q_ind, num_seqs, BLOCK_Q, False)
    q_block_start_idx = tl.load(query_start_len_ptr + seq_idx)

    q_block_local_idx = q_ind - q_block_start_idx
    cur_batch_in_all_start_index = tl.load(query_start_len_ptr + seq_idx)
    cur_batch_in_all_stop_index = tl.load(query_start_len_ptr + seq_idx + 1)
    cur_batch_query_len = cur_batch_in_all_stop_index - cur_batch_in_all_start_index

    if q_block_local_idx * BLOCK_Q >= cur_batch_query_len:
        return

    offs_m = tl.arange(0, BLOCK_M) + head_ind * BLOCK_M

    # load Q in two parts with different dim offsets
    offs_lora = tl.arange(0, KV_LORA_RANK)
    offs_rope = tl.arange(KV_LORA_RANK, KV_LORA_RANK + ROPE_RANK)

    query_pos = q_block_local_idx * BLOCK_Q + offs_m // num_queries_per_kv

    query_offset_0 = cur_batch_in_all_start_index + query_pos
    query_offset_1 = kv_head_idx * num_queries_per_kv + offs_m % num_queries_per_kv

    query_mask_0 = query_pos < cur_batch_query_len
    query_mask_1 = query_offset_1 < num_query_heads

    if ALL_DECODE or BLOCK_M >= num_query_heads:
        Q_cache_modifier: tl.constexpr = ".cg"
    else:
        Q_cache_modifier: tl.constexpr = ""

    # load Q in two parts
    # q_pe: (BLOCK_M, ROPE_RANK)
    q_rope_offset = (
        query_offset_0[:, None] * query_stride_0
        + query_offset_1[:, None] * query_stride_1
        + offs_rope[None, :]
    )
    Q_rope = tl.load(
        query_ptr + q_rope_offset,
        mask=query_mask_0[:, None] & query_mask_1[:, None],
        other=0.0,
        cache_modifier=Q_cache_modifier,
    )

    # q_lora: (BLOCK_M, KV_LORA_RANK)
    q_lora_offset = (
        query_offset_0[:, None] * query_stride_0
        + query_offset_1[:, None] * query_stride_1
        + offs_lora[None, :]
    )
    Q_lora = tl.load(
        query_ptr + q_lora_offset,
        mask=query_mask_0[:, None] & query_mask_1[:, None],
        other=0.0,
        cache_modifier=Q_cache_modifier,
    )

    M = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    L = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, KV_LORA_RANK], dtype=tl.float32)

    block_table_offset = seq_idx * block_table_stride

    # iterate topk indices in tiles of TILE_SIZE
    num_tiles = (topk_count + TILE_SIZE - 1) // TILE_SIZE
    KV_cache_modifier: tl.constexpr = ".cg" if ALL_DECODE else ""
    for t in range(0, num_tiles):
        tile_start = t * TILE_SIZE
        offs_t = tl.arange(0, TILE_SIZE)
        valid_t = (tile_start + offs_t) < topk_count

        # load top-k token positions for this query
        # TODO(cagri): use proper stride instead of topk_count
        topk_row_ptr = topk_indices_ptr + q_ind * topk_count
        topk_pos = tl.load(topk_row_ptr + tile_start + offs_t, mask=valid_t, other=0)
        # ignore -1, means not valid
        valid_t = valid_t & (topk_pos != -1)

        # map positions to block id and in-block offset
        physical_block_idx = topk_pos // BLOCK_SIZE
        slot = topk_pos % BLOCK_SIZE
        # Compute S = scale * (q_rope k_rope + q_lora k_lora)
        # q_rope: (BLOCK_M, ROPE_RANK) k_rope: (ROPE_RANK, TILE_SIZE)
        # q_lora: (BLOCK_M, KV_LORA_RANK) k_lora: (KV_LORA_RANK, TILE_SIZE)
        S = tl.zeros([BLOCK_M, TILE_SIZE], dtype=tl.float32)
        # load k in two parts
        # K_rope: (ROPE_RANK, TILE_SIZE)
        k_rope_ptrs = (
            key_cache_ptr
            + physical_block_idx[None, :] * stride_k_cache_0
            + kv_head_idx * stride_k_cache_2
            + offs_rope[:, None] * stride_k_cache_3
            + slot[None, :] * stride_k_cache_1
        )
        K_rope = tl.load(
            k_rope_ptrs,
            mask=valid_t[None, :],
            other=0.0,
            cache_modifier=KV_cache_modifier,
        )
        S += scale * tl.dot(Q_rope, K_rope)
        # K_lora: (KV_LORA_RANK, TILE_SIZE)
        k_lora_ptrs = (
            key_cache_ptr
            + physical_block_idx[None, :] * stride_k_cache_0
            + kv_head_idx * stride_k_cache_2
            + offs_lora[:, None] * stride_k_cache_3
            + slot[None, :] * stride_k_cache_1
        )
        K_lora = tl.load(
            k_lora_ptrs,
            mask=valid_t[None, :],
            other=0.0,
            cache_modifier=KV_cache_modifier,
        )

        S += scale * tl.dot(Q_lora, K_lora)

        S = tl.where(
            query_mask_1[:, None] & query_mask_0[:, None] & valid_t[None, :],
            S,
            float("-inf"),
        )

        m_j = tl.maximum(M, tl.max(S, axis=1))
        m_j = tl.where(m_j > float("-inf"), m_j, 0.0)
        P = tl.exp(S - m_j[:, None])
        l_j = tl.sum(P, axis=1)
        alpha = tl.exp(M - m_j)

        acc = acc * alpha[:, None]
        L = L * alpha + l_j
        M = m_j

        # load V with shape (TILE_SIZE, KV_LORA_RANK)
        v_lora_ptrs = (
            value_cache_ptr
            + physical_block_idx[:, None] * stride_v_cache_0
            + kv_head_idx * stride_v_cache_2
            + slot[:, None] * stride_v_cache_1
            + offs_lora[None, :] * stride_v_cache_3
        )
        V_lora = tl.load(
            v_lora_ptrs,
            mask=valid_t[:, None],
            other=0.0,
            cache_modifier=KV_cache_modifier,
        )

        acc += tl.dot(P.to(V_lora.dtype), V_lora)

    # epilogue
    one_over_L = 1.0 / L[:, None]
    acc = acc * one_over_L

    output_offs_lora = (
        query_offset_0[:, None] * output_stride_0
        + query_offset_1[:, None] * output_stride_1
        + offs_lora[None, :]
    )
    tl.store(
        output_ptr + output_offs_lora,
        acc,
        mask=query_mask_0[:, None] & query_mask_1[:, None],
    )
