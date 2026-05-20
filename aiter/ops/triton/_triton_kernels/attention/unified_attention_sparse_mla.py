import os

import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Autotune configuration
# ---------------------------------------------------------------------------
UA_SPARSE_MLA_AUTOTUNE = os.environ.get(
    "UNIFIED_ATTENTION_SPARSE_MLA_AUTOTUNE", "0"
).lower() in ("1", "true", "yes", "on")


def _get_2d_autotune_configs():
    configs = []
    for ts in [64, 128, 256]:
        for pv in [False, True]:
            for nw in [4, 8]:
                for ns in [1, 2, 3]:
                    for wpe in [1, 2]:
                        configs.append(
                            triton.Config(
                                {
                                    "TILE_SIZE": ts,
                                    "PRELOAD_V": pv,
                                    "waves_per_eu": wpe,
                                },
                                num_warps=nw,
                                num_stages=ns,
                            )
                        )
    return configs


def _get_3d_autotune_configs():
    """3D autotune does NOT include NUM_SEGMENTS_PER_SEQ — that controls the
    launch grid and must be chosen by the wrapper heuristic."""
    configs = []
    for ts in [32, 64, 128]:
        for pv in [False, True]:
            for nw in [4, 8]:
                for ns in [1, 2]:
                    for wpe in [1, 2]:
                        configs.append(
                            triton.Config(
                                {
                                    "TILE_SIZE": ts,
                                    "PRELOAD_V": pv,
                                    "waves_per_eu": wpe,
                                },
                                num_warps=nw,
                                num_stages=ns,
                            )
                        )
    return configs


def _get_reduce_autotune_configs():
    configs = []
    for nw in [1, 2, 4]:
        for ns in [1, 2]:
            for wpe in [1, 2, 4]:
                configs.append(
                    triton.Config(
                        {"waves_per_eu": wpe},
                        num_warps=nw,
                        num_stages=ns,
                    )
                )
    return configs


@triton.jit
def fast_exp2(x):
    return tl.math.exp2(x)


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


# ---------------------------------------------------------------------------
# 2D kernel — handles both dense top-k and CSR (kv_indptr/kv_indices) inputs.
# USE_CSR selects which set of index pointers is live; the other set may be
# passed any non-null placeholder (the wrapper reuses the live index pointer).
# ---------------------------------------------------------------------------
@triton.jit
def _kernel_unified_attention_sparse_mla_2d(
    output_ptr,  # [num_tokens, num_query_heads, KV_LORA_RANK]
    query_ptr,  # [num_tokens, num_query_heads, KV_LORA_RANK]
    key_cache_ptr,  # [num_blks, blk_size, 1, KV_LORA_RANK + ROPE_RANK]
    value_cache_ptr,  # [num_blks, blk_size, 1, KV_LORA_RANK]
    topk_indices_ptr,  # dense path: [num_tokens, topk]; CSR path: unused
    kv_indptr_ptr,     # CSR path: [num_tokens + 1]; dense path: unused
    kv_indices_ptr,    # CSR path: [nnz]; dense path: unused
    seq_lens_ptr,  # [num_seqs]
    scale,  # float32
    q_scale,  # None or scalar (per-tensor fp8)
    k_scale,  # None or scalar (per-tensor fp8)
    v_scale,  # None or scalar (per-tensor fp8)
    num_query_heads: tl.constexpr,  # int
    num_queries_per_kv: tl.constexpr,  # int
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
    topk_count: tl.constexpr,       # dense path loop bound (0 in CSR mode)
    max_sparse_len: tl.constexpr,   # CSR path loop bound (0 in dense mode)
    query_start_len_ptr,  # [num_seqs+1]
    num_seqs: tl.int32,
    BLOCK_M: tl.constexpr,  # int
    ROPE_RANK: tl.constexpr,
    KV_LORA_RANK: tl.constexpr,
    TILE_SIZE: tl.constexpr,
    USE_CSR: tl.constexpr,
    ALL_DECODE: tl.constexpr = False,
    PRELOAD_V: tl.constexpr = False,
    Q_SCALE: tl.constexpr = False,
    K_SCALE: tl.constexpr = False,
    V_SCALE: tl.constexpr = False,
):
    # only one query per program
    BLOCK_Q: tl.constexpr = 1
    kv_head_idx = 0  # assume there is single kv head

    # exp2 conversion constant — fold into scale once.
    RCP_LN2: tl.constexpr = 1.4426950408889634
    qk_scale = scale * RCP_LN2
    if Q_SCALE:
        qk_scale = qk_scale * tl.load(q_scale)
    if K_SCALE:
        qk_scale = qk_scale * tl.load(k_scale)

    NUM_HEAD_BLOCKS: tl.constexpr = (num_query_heads + BLOCK_M - 1) // BLOCK_M
    q_block_global_idx = tl.program_id(0)
    q_ind = q_block_global_idx // NUM_HEAD_BLOCKS
    head_ind = q_block_global_idx % NUM_HEAD_BLOCKS
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
    query_mask_1 = (query_offset_1 < num_query_heads) & (offs_m < num_query_heads)

    if ALL_DECODE or BLOCK_M >= num_query_heads:
        Q_cache_modifier: tl.constexpr = ".cg"
    else:
        Q_cache_modifier: tl.constexpr = ""

    # load Q in two parts
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

    if USE_CSR:
        row_start = tl.load(kv_indptr_ptr + q_ind)
        row_end = tl.load(kv_indptr_ptr + q_ind + 1)
        row_len = row_end - row_start
        num_tiles = (max_sparse_len + TILE_SIZE - 1) // TILE_SIZE
    else:
        num_tiles = (topk_count + TILE_SIZE - 1) // TILE_SIZE

    KV_cache_modifier: tl.constexpr = ".cg" if ALL_DECODE else ""
    for t in range(0, num_tiles):
        tile_start = t * TILE_SIZE
        offs_t = tl.arange(0, TILE_SIZE)
        if USE_CSR:
            valid_t = (tile_start + offs_t) < row_len
            pos = tl.load(
                kv_indices_ptr + row_start + tile_start + offs_t,
                mask=valid_t,
                other=-1,
            )
        else:
            valid_t = (tile_start + offs_t) < topk_count
            pos = tl.load(
                topk_indices_ptr + q_ind * topk_count + tile_start + offs_t,
                mask=valid_t,
                other=-1,
            )
        valid_t = valid_t & (pos != -1)

        physical_block_idx = pos // BLOCK_SIZE
        slot = pos % BLOCK_SIZE

        # K_rope
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
        K_rope = K_rope.to(Q_rope.dtype)

        S = tl.dot(Q_rope, K_rope)

        # K_lora
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
        K_lora = K_lora.to(Q_lora.dtype)

        S = tl.dot(Q_lora, K_lora, acc=S)
        S = S * qk_scale

        if PRELOAD_V:
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
            V_lora = V_lora.to(Q_lora.dtype)

        S = tl.where(
            query_mask_1[:, None] & query_mask_0[:, None] & valid_t[None, :],
            S,
            float("-inf"),
        )

        m_j = tl.maximum(M, tl.max(S, axis=1))
        m_j = tl.where(m_j > float("-inf"), m_j, 0.0)
        P = tl.math.exp2(S - m_j[:, None])
        l_j = tl.sum(P, axis=1)
        alpha = tl.math.exp2(M - m_j)

        acc = acc * alpha[:, None]
        L = L * alpha + l_j
        M = m_j

        if not PRELOAD_V:
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
            V_lora = V_lora.to(Q_lora.dtype)

        acc += tl.dot(P.to(V_lora.dtype), V_lora)

    one_over_L = tl.where(L[:, None] == 0.0, 0.0, 1.0 / L[:, None])
    acc = acc * one_over_L
    if V_SCALE:
        acc = acc * tl.load(v_scale)

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


# ---------------------------------------------------------------------------
# 3D split-K CSR kernel (split along the KV dimension for parallelism)
# ---------------------------------------------------------------------------
@triton.jit
def _kernel_unified_attention_sparse_mla_csr_3d(
    segm_output_ptr,  # [num_tokens, num_query_heads, NUM_SEGMENTS_PER_SEQ, KV_LORA_RANK]
    segm_max_ptr,  # [num_tokens, num_query_heads, NUM_SEGMENTS_PER_SEQ]
    segm_expsum_ptr,  # [num_tokens, num_query_heads, NUM_SEGMENTS_PER_SEQ]
    query_ptr,
    key_cache_ptr,
    value_cache_ptr,
    kv_indptr_ptr,
    kv_indices_ptr,
    scale,
    q_scale,
    k_scale,
    v_scale,
    num_query_heads: tl.constexpr,
    num_queries_per_kv: tl.constexpr,
    query_stride_0: tl.int64,
    query_stride_1: tl.int64,
    BLOCK_SIZE: tl.constexpr,
    stride_k_cache_0: tl.int64,
    stride_k_cache_1: tl.int64,
    stride_k_cache_2: tl.int64,
    stride_k_cache_3: tl.constexpr,
    stride_v_cache_0: tl.int64,
    stride_v_cache_1: tl.int64,
    stride_v_cache_2: tl.int64,
    stride_v_cache_3: tl.constexpr,
    max_sparse_len: tl.constexpr,
    query_start_len_ptr,
    num_seqs: tl.int32,
    BLOCK_M: tl.constexpr,
    ROPE_RANK: tl.constexpr,
    KV_LORA_RANK: tl.constexpr,
    TILE_SIZE: tl.constexpr,
    NUM_SEGMENTS_PER_SEQ: tl.constexpr,
    segm_out_stride_tok: tl.int64,
    segm_out_stride_head: tl.int64,
    segm_stat_stride_tok: tl.int64,
    segm_stat_stride_head: tl.int64,
    ALL_DECODE: tl.constexpr = False,
    PRELOAD_V: tl.constexpr = False,
    Q_SCALE: tl.constexpr = False,
    K_SCALE: tl.constexpr = False,
    V_SCALE: tl.constexpr = False,
):
    """
    Split-K decode kernel: each program processes a segment of KV indices
    (range of CSR positions) for one (query token, head-block) pair. Per-segment
    partial outputs (acc, M, expsum) are written out, then reduced by a
    separate kernel.
    """
    BLOCK_Q: tl.constexpr = 1
    kv_head_idx = 0

    RCP_LN2: tl.constexpr = 1.4426950408889634
    qk_scale = scale * RCP_LN2
    if Q_SCALE:
        qk_scale = qk_scale * tl.load(q_scale)
    if K_SCALE:
        qk_scale = qk_scale * tl.load(k_scale)

    NUM_HEAD_BLOCKS: tl.constexpr = (num_query_heads + BLOCK_M - 1) // BLOCK_M
    q_block_global_idx = tl.program_id(0)
    segm_idx = tl.program_id(1)

    q_ind = q_block_global_idx // NUM_HEAD_BLOCKS
    head_ind = q_block_global_idx % NUM_HEAD_BLOCKS
    seq_idx = find_seq_idx(query_start_len_ptr, q_ind, num_seqs, BLOCK_Q, False)
    q_block_start_idx = tl.load(query_start_len_ptr + seq_idx)
    q_block_local_idx = q_ind - q_block_start_idx
    cur_batch_in_all_start_index = tl.load(query_start_len_ptr + seq_idx)
    cur_batch_in_all_stop_index = tl.load(query_start_len_ptr + seq_idx + 1)
    cur_batch_query_len = cur_batch_in_all_stop_index - cur_batch_in_all_start_index

    if q_block_local_idx * BLOCK_Q >= cur_batch_query_len:
        return

    row_start = tl.load(kv_indptr_ptr + q_ind)
    row_end = tl.load(kv_indptr_ptr + q_ind + 1)
    row_len = row_end - row_start

    # Segment slice — each segment covers contiguous CSR positions
    segm_len = cdiv_fn(max_sparse_len, NUM_SEGMENTS_PER_SEQ)
    segm_start = segm_idx * segm_len
    segm_end = tl.minimum(segm_start + segm_len, row_len)

    offs_m = tl.arange(0, BLOCK_M) + head_ind * BLOCK_M
    offs_lora = tl.arange(0, KV_LORA_RANK)
    offs_rope = tl.arange(KV_LORA_RANK, KV_LORA_RANK + ROPE_RANK)

    query_pos = q_block_local_idx * BLOCK_Q + offs_m // num_queries_per_kv
    query_offset_0 = cur_batch_in_all_start_index + query_pos
    query_offset_1 = kv_head_idx * num_queries_per_kv + offs_m % num_queries_per_kv

    query_mask_0 = query_pos < cur_batch_query_len
    query_mask_1 = (query_offset_1 < num_query_heads) & (offs_m < num_query_heads)

    if ALL_DECODE or BLOCK_M >= num_query_heads:
        Q_cache_modifier: tl.constexpr = ".cg"
    else:
        Q_cache_modifier: tl.constexpr = ""

    # If this segment has no work, still need to write sentinel stats so reduce
    # can ignore it.
    if segm_start >= row_len:
        # write -inf / 0 stats
        stat_offset = (
            query_offset_0 * segm_stat_stride_tok
            + query_offset_1 * segm_stat_stride_head
            + segm_idx
        )
        tl.store(
            segm_max_ptr + stat_offset,
            tl.full([BLOCK_M], float("-inf"), dtype=tl.float32),
            mask=query_mask_0 & query_mask_1,
        )
        tl.store(
            segm_expsum_ptr + stat_offset,
            tl.zeros([BLOCK_M], dtype=tl.float32),
            mask=query_mask_0 & query_mask_1,
        )
        return

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
    L = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, KV_LORA_RANK], dtype=tl.float32)

    num_tiles = cdiv_fn(segm_len, TILE_SIZE)
    KV_cache_modifier: tl.constexpr = ".cg" if ALL_DECODE else ""
    for t in range(0, num_tiles):
        tile_start = segm_start + t * TILE_SIZE
        offs_t = tl.arange(0, TILE_SIZE)
        valid_t = (tile_start + offs_t) < segm_end

        kv_pos = tl.load(
            kv_indices_ptr + row_start + tile_start + offs_t,
            mask=valid_t,
            other=-1,
        )
        valid_t = valid_t & (kv_pos != -1)

        physical_block_idx = kv_pos // BLOCK_SIZE
        slot = kv_pos % BLOCK_SIZE

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
        K_rope = K_rope.to(Q_rope.dtype)

        S = tl.dot(Q_rope, K_rope)

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
        K_lora = K_lora.to(Q_lora.dtype)

        S = tl.dot(Q_lora, K_lora, acc=S)
        S = S * qk_scale

        if PRELOAD_V:
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
            V_lora = V_lora.to(Q_lora.dtype)

        S = tl.where(
            query_mask_1[:, None] & query_mask_0[:, None] & valid_t[None, :],
            S,
            float("-inf"),
        )

        m_j = tl.maximum(M, tl.max(S, axis=1))
        m_j = tl.where(m_j > float("-inf"), m_j, 0.0)
        P = tl.math.exp2(S - m_j[:, None])
        l_j = tl.sum(P, axis=1)
        alpha = tl.math.exp2(M - m_j)

        acc = acc * alpha[:, None]
        L = L * alpha + l_j
        M = m_j

        if not PRELOAD_V:
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
            V_lora = V_lora.to(Q_lora.dtype)

        acc += tl.dot(P.to(V_lora.dtype), V_lora)

    if V_SCALE:
        acc = acc * tl.load(v_scale)

    # Write segment partial (acc is unnormalized, stats are M and L)
    out_offset = (
        query_offset_0[:, None] * segm_out_stride_tok
        + query_offset_1[:, None] * segm_out_stride_head
        + segm_idx * KV_LORA_RANK
        + offs_lora[None, :]
    )
    tl.store(
        segm_output_ptr + out_offset,
        acc,
        mask=query_mask_0[:, None] & query_mask_1[:, None],
    )

    stat_offset = (
        query_offset_0 * segm_stat_stride_tok
        + query_offset_1 * segm_stat_stride_head
        + segm_idx
    )
    tl.store(
        segm_max_ptr + stat_offset,
        M,
        mask=query_mask_0 & query_mask_1,
    )
    tl.store(
        segm_expsum_ptr + stat_offset,
        L,
        mask=query_mask_0 & query_mask_1,
    )


@triton.jit
def _kernel_unified_attention_sparse_mla_csr_reduce(
    output_ptr,  # [num_tokens, num_query_heads, KV_LORA_RANK]
    segm_output_ptr,  # [num_tokens, num_query_heads, NUM_SEGMENTS_PER_SEQ, KV_LORA_RANK]
    segm_max_ptr,
    segm_expsum_ptr,
    output_stride_0: tl.int64,
    output_stride_1: tl.int64,
    segm_out_stride_tok: tl.int64,
    segm_out_stride_head: tl.int64,
    segm_stat_stride_tok: tl.int64,
    segm_stat_stride_head: tl.int64,
    KV_LORA_RANK: tl.constexpr,
    NUM_SEGMENTS_PER_SEQ: tl.constexpr,
):
    """
    Reduce per-segment partials into a single output via log-sum-exp combine.
    Each program: one (token, head) pair.
    """
    tok_idx = tl.program_id(0)
    head_idx = tl.program_id(1)

    offs_seg = tl.arange(0, NUM_SEGMENTS_PER_SEQ)

    stat_off = (
        tok_idx * segm_stat_stride_tok + head_idx * segm_stat_stride_head + offs_seg
    )
    seg_max = tl.load(segm_max_ptr + stat_off)
    seg_sum = tl.load(segm_expsum_ptr + stat_off)

    # Filter empty segments
    valid = seg_sum > 0.0
    seg_max = tl.where(valid, seg_max, float("-inf"))

    g_max = tl.max(seg_max, axis=0)
    g_max_safe = tl.where(g_max > float("-inf"), g_max, 0.0)

    alpha = tl.math.exp2(seg_max - g_max_safe)
    alpha = tl.where(valid, alpha, 0.0)

    new_sum = tl.sum(alpha * seg_sum, axis=0)
    inv = tl.where(new_sum > 0.0, 1.0 / new_sum, 0.0)

    offs_d = tl.arange(0, KV_LORA_RANK)
    acc = tl.zeros([KV_LORA_RANK], dtype=tl.float32)
    for s in range(0, NUM_SEGMENTS_PER_SEQ):
        seg_alpha = tl.load(segm_max_ptr + tok_idx * segm_stat_stride_tok + head_idx * segm_stat_stride_head + s)
        seg_l = tl.load(segm_expsum_ptr + tok_idx * segm_stat_stride_tok + head_idx * segm_stat_stride_head + s)
        is_valid = seg_l > 0.0
        a = tl.math.exp2(seg_alpha - g_max_safe)
        a = tl.where(is_valid, a, 0.0)
        seg_acc = tl.load(
            segm_output_ptr
            + tok_idx * segm_out_stride_tok
            + head_idx * segm_out_stride_head
            + s * KV_LORA_RANK
            + offs_d
        )
        acc += seg_acc * a

    acc = acc * inv

    out_off = tok_idx * output_stride_0 + head_idx * output_stride_1 + offs_d
    tl.store(output_ptr + out_off, acc)


# ---------------------------------------------------------------------------
# Autotuner wrappers (optional)
# ---------------------------------------------------------------------------
if UA_SPARSE_MLA_AUTOTUNE:
    _2d_topk_autotuner = triton.autotune(
        configs=_get_2d_autotune_configs(),
        key=[
            "num_query_heads",
            "KV_LORA_RANK",
            "BLOCK_SIZE",
            "BLOCK_M",
            "topk_count",
            "Q_SCALE",
            "K_SCALE",
            "V_SCALE",
        ],
    )(_kernel_unified_attention_sparse_mla_2d)

    _2d_csr_autotuner = triton.autotune(
        configs=_get_2d_autotune_configs(),
        key=[
            "num_query_heads",
            "KV_LORA_RANK",
            "BLOCK_SIZE",
            "BLOCK_M",
            "max_sparse_len",
            "Q_SCALE",
            "K_SCALE",
            "V_SCALE",
        ],
    )(_kernel_unified_attention_sparse_mla_2d)

    _3d_csr_autotuner = triton.autotune(
        configs=_get_3d_autotune_configs(),
        key=[
            "num_query_heads",
            "KV_LORA_RANK",
            "BLOCK_SIZE",
            "BLOCK_M",
            "max_sparse_len",
            "NUM_SEGMENTS_PER_SEQ",
            "Q_SCALE",
            "K_SCALE",
            "V_SCALE",
        ],
    )(_kernel_unified_attention_sparse_mla_csr_3d)
else:
    _2d_topk_autotuner = None
    _2d_csr_autotuner = None
    _3d_csr_autotuner = None
