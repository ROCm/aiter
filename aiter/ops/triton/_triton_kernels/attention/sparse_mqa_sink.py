import triton
import triton.language as tl


@triton.jit
def _find_seq_idx(cu_seqlens_q_ptr, token_idx, num_seqs):
    left: tl.int32 = 0
    right = num_seqs
    while left < right:
        mid = (left + right) // 2
        val = tl.load(cu_seqlens_q_ptr + mid)
        if val <= token_idx:
            left = mid + 1
        else:
            right = mid
    return left - 1


@triton.jit
def _sparse_mqa_sink_kernel(
    out_ptr,  # [num_tokens, num_heads, head_dim]
    q_ptr,  # [num_tokens, num_heads, head_dim]
    kv_ptr,  # [num_blocks, block_size, head_dim]
    topk_ptr,  # [num_tokens, topk]
    attn_sink_ptr,  # [num_heads]
    block_table_ptr,  # [num_seqs, max_blocks_per_seq]
    cu_seqlens_q_ptr,  # [num_seqs + 1]
    seqused_k_ptr,  # [num_seqs]
    scale,
    q_stride_t: tl.int64,
    q_stride_h: tl.int64,
    q_stride_d: tl.int64,
    out_stride_t: tl.int64,
    out_stride_h: tl.int64,
    out_stride_d: tl.int64,
    kv_stride_b: tl.int64,
    kv_stride_s: tl.int64,
    kv_stride_d: tl.int64,
    topk_stride_t: tl.int64,
    topk_stride_k: tl.int64,
    block_table_stride_b: tl.int64,
    block_table_stride_blk: tl.int64,
    num_heads: tl.constexpr,
    head_dim: tl.constexpr,
    topk_count: tl.constexpr,
    block_size: tl.constexpr,
    num_seqs: tl.int32,
    BLOCK_H: tl.constexpr,
    BLOCK_D: tl.constexpr,
    TILE_K: tl.constexpr,
):
    """Sparse MQA with DSv4's attention-sink denominator.

    One program handles one query token and BLOCK_H query heads. KV is MQA:
    all query heads share the same [topk, head_dim] key/value rows.
    """
    head_blocks: tl.constexpr = (num_heads + BLOCK_H - 1) // BLOCK_H
    program_id = tl.program_id(0)
    token_id = program_id // head_blocks
    head_block = program_id % head_blocks

    seq_idx = _find_seq_idx(cu_seqlens_q_ptr, token_id, num_seqs)
    seq_start = tl.load(cu_seqlens_q_ptr + seq_idx)
    seq_end = tl.load(cu_seqlens_q_ptr + seq_idx + 1)
    if token_id >= seq_end:
        return
    local_token = token_id - seq_start
    kv_len = tl.load(seqused_k_ptr + seq_idx)

    offs_h = head_block * BLOCK_H + tl.arange(0, BLOCK_H)
    offs_d = tl.arange(0, BLOCK_D)
    h_mask = offs_h < num_heads
    d_mask = offs_d < head_dim

    q = tl.load(
        q_ptr
        + token_id * q_stride_t
        + offs_h[:, None] * q_stride_h
        + offs_d[None, :] * q_stride_d,
        mask=h_mask[:, None] & d_mask[None, :],
        other=0.0,
        cache_modifier=".cg",
    )

    sink = tl.load(attn_sink_ptr + offs_h, mask=h_mask, other=float("-inf")).to(
        tl.float32
    )
    has_sink = sink > -3.0e38
    m_i = tl.where(has_sink, sink, float("-inf"))
    l_i = tl.where(has_sink, 1.0, 0.0)
    acc = tl.zeros((BLOCK_H, BLOCK_D), dtype=tl.float32)

    for tile_start in range(0, topk_count, TILE_K):
        offs_k = tile_start + tl.arange(0, TILE_K)
        topk_pos = tl.load(
            topk_ptr + token_id * topk_stride_t + offs_k * topk_stride_k,
            mask=offs_k < topk_count,
            other=-1,
        )
        valid_k = (offs_k < topk_count) & (topk_pos >= 0) & (topk_pos < kv_len)

        logical_block = topk_pos // block_size
        slot = topk_pos - logical_block * block_size
        physical_block = tl.load(
            block_table_ptr
            + seq_idx * block_table_stride_b
            + logical_block * block_table_stride_blk,
            mask=valid_k,
            other=0,
        )

        k = tl.load(
            kv_ptr
            + physical_block[None, :] * kv_stride_b
            + slot[None, :] * kv_stride_s
            + offs_d[:, None] * kv_stride_d,
            mask=d_mask[:, None] & valid_k[None, :],
            other=0.0,
            cache_modifier=".cg",
        )
        scores = tl.dot(q, k) * scale
        scores = tl.where(h_mask[:, None] & valid_k[None, :], scores, float("-inf"))

        m_new = tl.maximum(m_i, tl.max(scores, axis=1))
        m_new = tl.where(m_new > float("-inf"), m_new, 0.0)
        p = tl.exp(scores - m_new[:, None])
        alpha = tl.exp(m_i - m_new)
        l_new = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]

        v = tl.load(
            kv_ptr
            + physical_block[:, None] * kv_stride_b
            + slot[:, None] * kv_stride_s
            + offs_d[None, :] * kv_stride_d,
            mask=valid_k[:, None] & d_mask[None, :],
            other=0.0,
            cache_modifier=".cg",
        )
        acc += tl.dot(p.to(v.dtype), v)
        m_i = m_new
        l_i = l_new

    acc = acc * tl.where(l_i[:, None] > 0.0, 1.0 / l_i[:, None], 0.0)
    tl.store(
        out_ptr
        + token_id * out_stride_t
        + offs_h[:, None] * out_stride_h
        + offs_d[None, :] * out_stride_d,
        acc,
        mask=h_mask[:, None] & d_mask[None, :],
    )
