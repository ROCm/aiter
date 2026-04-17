# The kernels in this file are adapted from vLLM:
# https://github.com/vllm-project/vllm/blob/main/vllm/attention/ops/triton_unified_attention.py
import triton
import triton.language as tl
import torch
from aiter.ops.triton.utils.types import e4m3_dtype
from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr

float8_info = torch.finfo(e4m3_dtype)


@triton.jit
def fast_exp(x):
    RCP_LN2: tl.constexpr = 1.4426950408889634
    return tl.math.exp2(x * RCP_LN2)


@triton.jit
def cdiv_fn(x, y):
    return (x + y - 1) // y


@triton.jit
def apply_softcap(S, x):
    Sdiv = S / x
    p1 = tl.math.exp2(Sdiv)
    p2 = tl.math.exp2(-Sdiv)
    return x * (p1 - p2) / (p1 + p2)


@triton.jit
def _find_seq_idx(
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


_mla_prefill_fwd_kernel_repr = make_kernel_repr(
    "_mla_prefill_fwd_kernel",
    [
        "num_query_heads",
        "num_kv_heads",
        "TILE_SIZE",
        "KV_LORA_RANK",
        "QK_ROPE_HEAD_DIM",
        "BLOCK_Q",
        "BLOCK_M",
        "NUM_SEGMENTS_PER_SEQ",
        "num_warps",
        "num_stages",
    ],
)


@triton.jit(repr=_mla_prefill_fwd_kernel_repr)
def _mla_prefill_fwd_kernel(
    output_ptr,  # [num_tokens, num_query_heads, head_size]
    query_ptr,  # [num_tokens, num_query_heads, head_size]
    kv_buffer_ptr,  # [num_blks, blk_size, num_kv_heads, head_size]
    block_tables_ptr,  # [num_seqs, max_num_blocks_per_seq]
    seq_lens_ptr,  # [num_seqs]
    scale: tl.constexpr,  # float32
    q_scale_ptr,  # float32
    kv_scale_ptr,  # float32
    out_scale_ptr,  # float32
    num_query_heads: tl.constexpr,  # int
    num_kv_heads: tl.constexpr,  # int
    block_tables_stride: tl.int64,  # int
    query_stride_0: tl.int64,  # int
    query_stride_1: tl.int64,  # int, should be equal to head_size
    output_stride_0: tl.int64,  # int
    output_stride_1: tl.int64,  # int, should be equal to head_size
    KV_LORA_RANK: tl.constexpr,  # int
    QK_ROPE_HEAD_DIM: tl.constexpr,  # int
    stride_kv_buffer_0: tl.int64,  # int
    stride_kv_buffer_1: tl.int64,  # int
    stride_kv_buffer_2: tl.int64,  # int
    stride_kv_buffer_3: tl.constexpr,  # int
    query_start_len_ptr,  # [num_seqs+1]
    num_seqs: tl.int32,
    TILE_SIZE: tl.constexpr,  # int
    BLOCK_Q: tl.constexpr,  # int
    BLOCK_M: tl.constexpr,  # int
    num_warps: tl.constexpr,  # int
    num_stages: tl.constexpr,  # int
    FP8_MIN: tl.constexpr = float8_info.min,
    FP8_MAX: tl.constexpr = float8_info.max,
):
    kv_head_idx = tl.program_id(0)
    q_block_global_idx = tl.program_id(1)

    # needed to use exp2 (exp2 -> exp conversion)
    RCP_LN2 = 1.4426950408889634
    qk_scale = scale * RCP_LN2

    seq_idx = _find_seq_idx(
        query_start_len_ptr, q_block_global_idx, num_seqs, BLOCK_Q, True
    )

    q_block_start_idx = tl.load(query_start_len_ptr + seq_idx) // BLOCK_Q + seq_idx

    q_block_local_idx = q_block_global_idx - q_block_start_idx

    cur_batch_in_all_start_index = tl.load(query_start_len_ptr + seq_idx)
    cur_batch_in_all_stop_index = tl.load(query_start_len_ptr + seq_idx + 1)

    cur_batch_query_len = cur_batch_in_all_stop_index - cur_batch_in_all_start_index

    # tl.device_print("cur_batch_query_len", cur_batch_query_len)
    if q_block_local_idx * BLOCK_Q >= cur_batch_query_len:
        return

    qk_factor: tl.float32 = qk_scale
    if q_scale_ptr is not None:
        q_scale = tl.load(q_scale_ptr)
        qk_factor = qk_factor * q_scale
    else:
        q_scale = None

    if kv_scale_ptr is not None:
        kv_scale = tl.load(kv_scale_ptr)
        qk_factor = qk_factor * kv_scale
    else:
        kv_scale = None
    out_scale = None
    if out_scale_ptr is not None:
        out_scale = 1 / tl.load(out_scale_ptr)

    offs_m = tl.arange(0, BLOCK_M)
    offs_lora_rank = tl.arange(0, KV_LORA_RANK)
    offs_rope_head_dim = tl.arange(0, QK_ROPE_HEAD_DIM)
    offs_t = tl.arange(0, TILE_SIZE)

    num_queries_per_kv: tl.constexpr = num_query_heads // num_kv_heads
    query_pos = q_block_local_idx * BLOCK_Q + offs_m // num_queries_per_kv

    query_offset_0 = cur_batch_in_all_start_index + query_pos
    query_offset_1 = kv_head_idx * num_queries_per_kv + offs_m % num_queries_per_kv
    query_offset = (
        query_offset_0[:, None] * query_stride_0
        + query_offset_1[:, None] * query_stride_1
    )

    query_mask_0 = query_pos < cur_batch_query_len
    query_mask_1 = query_offset_1 < num_query_heads

    # Q_lora : (BLOCK_M, KV_LORA_RANK)
    # Q_rope : (BLOCK_M, QK_ROPE_HEAD_DIM)
    Q_lora = tl.load(
        query_ptr + query_offset + offs_lora_rank[None, :],
        mask=query_mask_0[:, None] & query_mask_1[:, None],
        other=0.0,
    )
    Q_rope = tl.load(
        query_ptr + query_offset + (KV_LORA_RANK + offs_rope_head_dim)[None, :],
        mask=query_mask_0[:, None] & query_mask_1[:, None],
        other=0.0,
    )

    block_tables_ptr_shifted = block_tables_ptr + seq_idx * block_tables_stride

    M = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    L = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, KV_LORA_RANK], dtype=tl.float32)

    # sequence len for this particular sequence
    seq_len = tl.load(seq_lens_ptr + seq_idx)

    # context length for this particular sequences
    context_len = seq_len - cur_batch_query_len

    # compute the length of the longest sequence prefix spanned by any
    # query token in the current q_block (q_block_local_idx)
    max_seq_prefix_len = (
        context_len
        + q_block_local_idx * BLOCK_Q
        + (BLOCK_M - 1) // num_queries_per_kv
        + 1
    )

    # adjust for potential padding in the last q_block by considering the
    # actual sequence length
    max_seq_prefix_len = tl.minimum(max_seq_prefix_len, seq_len)

    # calculate the number of tiles that need to be processed to
    # cover the longest sequence prefix (due to causal masking, tiles beyond
    # this prefix can be skipped)
    num_tiles = cdiv_fn(max_seq_prefix_len, TILE_SIZE)

    # ---- Sliding-window tile pruning --------------------
    # Default: keep previous global behavior
    tile_start = 0
    tile_end = num_tiles
    seq_offset = offs_t

    # iterate through tiles (now limited to the sliding window range)
    for j in range(tile_start, tile_end):
        physical_block_idx = tl.load(block_tables_ptr_shifted + j).to(tl.int64)

        kv_offset = (
            physical_block_idx * stride_kv_buffer_0 + kv_head_idx * stride_kv_buffer_2
        )

        kv_lora_offset = (
            kv_offset
            + offs_t[:, None] * stride_kv_buffer_1
            + offs_lora_rank[None, :] * stride_kv_buffer_3
        )
        # KV_lora : (BLOCK_M, KV_LORA_RANK)
        KV_lora = tl.load(
            kv_buffer_ptr + kv_lora_offset,
        )

        k_rope_offset = (
            kv_offset
            + offs_t[None, :] * stride_kv_buffer_1
            + (KV_LORA_RANK + offs_rope_head_dim)[:, None] * stride_kv_buffer_3
        )
        # K_rope : (BLOCK_M, QK_ROPE_HEAD_DIM)
        K_rope = tl.load(
            kv_buffer_ptr + k_rope_offset,
        )

        seq_mask = seq_offset[None, :] < context_len + query_pos[:, None] + 1

        S_lora = tl.dot(Q_lora, KV_lora.trans(1, 0).to(Q_lora.dtype))
        S_rope = tl.dot(Q_rope, K_rope.to(Q_lora.dtype))
        S = qk_factor * (S_lora + S_rope)

        S = tl.where(
            query_mask_1[:, None] & query_mask_0[:, None] & seq_mask, S, float("-inf")
        )

        # compute running maximum
        # m_j : (BLOCK_M,)
        m_j = tl.maximum(M, tl.max(S, axis=1))

        # For sliding window there's a chance the max is -inf due to masking of
        # the entire row. In this case we need to set m_j 0 to avoid NaN
        m_j = tl.where(m_j > float("-inf"), m_j, 0.0)

        # P : (BLOCK_M, TILE_SIZE,)
        P = tl.math.exp2(S - m_j[:, None])

        # l_j : (BLOCK_M,)
        l_j = tl.sum(P, axis=1)

        # alpha : (BLOCK_M, )
        alpha = tl.math.exp2(M - m_j)

        # acc : (BLOCK_M, KV_LORA_RANK)
        acc = acc * alpha[:, None]

        # update constants
        L = L * alpha + l_j
        M = m_j

        # acc : (BLOCK_M, KV_LORA_RANK)
        acc += tl.dot(P.to(KV_lora.dtype), KV_lora)
        seq_offset += TILE_SIZE

    # epilogue
    # This helps the compiler do Newton Raphson on l_i vs on acc which is much larger.
    if kv_scale_ptr is not None:
        one_over_L = kv_scale / L[:, None]
    else:
        one_over_L = 1.0 / L[:, None]
    acc = acc * one_over_L

    if out_scale_ptr is not None:
        acc = acc * out_scale
        acc = tl.clamp(acc, FP8_MIN, FP8_MAX)

    output_offset = (
        query_offset_0[:, None] * output_stride_0
        + query_offset_1[:, None] * output_stride_1
        + offs_lora_rank[None, :]
    )

    tl.store(
        output_ptr + output_offset,
        acc,
        mask=query_mask_0[:, None] & query_mask_1[:, None],
    )


_mla_decode_fwd_kernel_repr = make_kernel_repr(
    "_mla_decode_fwd_kernel",
    [
        "num_query_heads",
        "num_kv_heads",
        "num_tokens_per_seq",
        "TILE_SIZE",
        "KV_LORA_RANK",
        "QK_ROPE_HEAD_DIM",
        "BLOCK_Q",
        "BLOCK_M",
        "NUM_SEGMENTS_PER_SEQ",
        "num_warps",
        "num_stages",
    ],
)


@triton.jit(repr=_mla_decode_fwd_kernel_repr)
def _mla_decode_fwd_kernel(
    segm_output_ptr,  # [total_num_tokens, num_query_heads, KV_LORA_RANK + qk_rope_head_dim]
    segm_max_ptr,  # [total_num_tokens, num_query_heads, num_segments]
    segm_expsum_ptr,  # [total_num_tokens, num_query_heads, num_segments]
    query_ptr,  # [total_num_tokens, num_query_heads, head_size]
    kv_buffer_ptr,  # [num_blks, blk_size, num_kv_heads, head_size]
    block_tables_ptr,  # [num_seqs, max_num_blocks_per_seq]
    seq_lens_ptr,  # [num_seqs]
    scale,  # float32
    q_scale_ptr,  # float32
    kv_scale_ptr,  # float32
    num_query_heads: tl.constexpr,  # int
    num_kv_heads: tl.constexpr,  # int
    block_tables_stride: tl.int64,  # int
    query_stride_0: tl.int64,  # int
    query_stride_1: tl.int64,  # int, should be equal to head_size
    KV_LORA_RANK: tl.constexpr,  # int
    QK_ROPE_HEAD_DIM: tl.constexpr,  # int
    stride_kv_buffer_0: tl.int64,  # int
    stride_kv_buffer_1: tl.int64,  # int
    stride_kv_buffer_2: tl.int64,  # int
    stride_kv_buffer_3: tl.constexpr,  # int
    query_start_len_ptr,  # [num_seqs+1]
    num_tokens_per_seq: tl.int32,
    TILE_SIZE: tl.constexpr,  # int
    BLOCK_Q: tl.constexpr,  # int
    BLOCK_M: tl.constexpr,  # int
    NUM_SEGMENTS_PER_SEQ: tl.constexpr,  # int
    num_warps: tl.constexpr,  # int
    num_stages: tl.constexpr,  # int
    ALL_DECODE: tl.constexpr = False,  # bool
):
    q_block_global_idx = tl.program_id(0)
    kv_head_idx = tl.program_id(1)
    segm_idx = tl.program_id(2)

    # needed to use exp2 (exp2 -> exp conversion)
    RCP_LN2 = 1.4426950408889634
    qk_scale = scale * RCP_LN2
    num_q_blocks_per_seq = cdiv_fn(num_tokens_per_seq, BLOCK_Q)

    if ALL_DECODE:
        seq_idx = q_block_global_idx
    else:
        seq_idx = q_block_global_idx // num_q_blocks_per_seq

    q_start_idx = tl.load(query_start_len_ptr + seq_idx)
    q_block_local_idx = q_block_global_idx - seq_idx * num_q_blocks_per_seq

    # sequence len for this particular sequence
    seq_len = tl.load(seq_lens_ptr + seq_idx)

    # number of segments for this particular sequence
    num_segments = NUM_SEGMENTS_PER_SEQ
    tiles_per_segment = cdiv_fn(seq_len, num_segments * TILE_SIZE)

    if segm_idx * tiles_per_segment * TILE_SIZE >= seq_len:
        return

    qk_factor: tl.float32 = qk_scale
    if q_scale_ptr is not None:
        q_scale = tl.load(q_scale_ptr)
        qk_factor = qk_factor * q_scale
    else:
        q_scale = None

    if kv_scale_ptr is not None:
        kv_scale = tl.load(kv_scale_ptr)
        qk_factor = qk_factor * kv_scale
    else:
        kv_scale = None

    offs_m = tl.arange(0, BLOCK_M)
    offs_lora_rank = tl.arange(0, KV_LORA_RANK)
    offs_rope_head_dim = tl.arange(0, QK_ROPE_HEAD_DIM)
    offs_t = tl.arange(0, TILE_SIZE)

    num_queries_per_kv: tl.constexpr = num_query_heads // num_kv_heads
    query_pos = q_block_local_idx * BLOCK_Q + offs_m // num_queries_per_kv

    query_offset_0 = q_start_idx + query_pos
    query_offset_1 = kv_head_idx * num_queries_per_kv + offs_m % num_queries_per_kv
    query_offset = (
        query_offset_0[:, None] * query_stride_0
        + query_offset_1[:, None] * query_stride_1
    )

    query_mask_0 = query_pos < num_tokens_per_seq
    query_mask_1 = query_offset_1 < num_query_heads

    # Q_lora : (BLOCK_M, KV_LORA_RANK)
    # Q_rope : (BLOCK_M, QK_ROPE_HEAD_DIM)
    Q_lora = tl.load(
        query_ptr + query_offset + offs_lora_rank[None, :],
        mask=query_mask_0[:, None] & query_mask_1[:, None],
        other=0.0,
    )
    Q_rope = tl.load(
        query_ptr + query_offset + (KV_LORA_RANK + offs_rope_head_dim)[None, :],
        mask=query_mask_0[:, None] & query_mask_1[:, None],
        other=0.0,
    )

    block_tables_ptr_shifted = block_tables_ptr + seq_idx * block_tables_stride

    M = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    L = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, KV_LORA_RANK], dtype=tl.float32)

    # context length for this particular sequences
    context_len = seq_len - num_tokens_per_seq

    # compute the length of the longest sequence prefix spanned by any
    # query token in the current q_block (q_block_local_idx)
    max_seq_prefix_len = (
        context_len
        + q_block_local_idx * BLOCK_Q
        + (BLOCK_M - 1) // num_queries_per_kv
        + 1
    )

    # adjust for potential padding in the last q_block by considering the
    # actual sequence length
    max_seq_prefix_len = tl.minimum(max_seq_prefix_len, seq_len)

    # calculate the number of tiles that need to be processed to
    # cover the longest sequence prefix (due to causal masking, tiles beyond
    # this prefix can be skipped)
    num_tiles = cdiv_fn(max_seq_prefix_len, TILE_SIZE)

    KV_cache_modifier: tl.constexpr = ".cg" if ALL_DECODE else ""
    seq_offset = segm_idx * tiles_per_segment * TILE_SIZE + offs_t

    # iterate through tiles within current segment
    for j in range(
        segm_idx * tiles_per_segment,
        min((segm_idx + 1) * tiles_per_segment, num_tiles),
    ):
        physical_block_idx = tl.load(block_tables_ptr_shifted + j).to(tl.int64)

        kv_offset = (
            physical_block_idx * stride_kv_buffer_0 + kv_head_idx * stride_kv_buffer_2
        )

        kv_lora_offset = (
            kv_offset
            + offs_t[:, None] * stride_kv_buffer_1
            + offs_lora_rank[None, :] * stride_kv_buffer_3
        )
        # KV_lora : (BLOCK_M, KV_LORA_RANK)
        KV_lora = tl.load(
            kv_buffer_ptr + kv_lora_offset,
            cache_modifier=KV_cache_modifier,
        )

        k_rope_offset = (
            kv_offset
            + offs_t[None, :] * stride_kv_buffer_1
            + (KV_LORA_RANK + offs_rope_head_dim)[:, None] * stride_kv_buffer_3
        )
        # K_rope : (BLOCK_M, QK_ROPE_HEAD_DIM)
        K_rope = tl.load(
            kv_buffer_ptr + k_rope_offset,
            cache_modifier=KV_cache_modifier,
        )

        seq_mask = seq_offset[None, :] < context_len + query_pos[:, None] + 1

        S_lora = tl.dot(Q_lora, KV_lora.trans(1, 0).to(Q_lora.dtype))
        S_rope = tl.dot(Q_rope, K_rope.to(Q_lora.dtype))
        S = qk_factor * (S_lora + S_rope)

        S = tl.where(
            query_mask_1[:, None] & query_mask_0[:, None] & seq_mask, S, float("-inf")
        )

        # compute running maximum
        # m_j : (BLOCK_M,)
        m_j = tl.maximum(M, tl.max(S, axis=1))

        # For sliding window there's a chance the max is -inf due to masking of
        # the entire row. In this case we need to set m_j 0 to avoid NaN
        m_j = tl.where(m_j > float("-inf"), m_j, 0.0)

        # P : (BLOCK_M, TILE_SIZE,)
        P = tl.math.exp2(S - m_j[:, None])

        # l_j : (BLOCK_M,)
        l_j = tl.sum(P, axis=1)

        # alpha : (BLOCK_M, )
        alpha = tl.math.exp2(M - m_j)

        # acc : (BLOCK_M, KV_LORA_RANK)
        acc = acc * alpha[:, None]

        # update constants
        L = L * alpha + l_j
        M = m_j

        # acc : (BLOCK_M, KV_LORA_RANK)
        acc += tl.dot(P.to(KV_lora.dtype), KV_lora)
        seq_offset += TILE_SIZE

    if kv_scale_ptr is not None:
        acc = acc * kv_scale

    segm_output_offset = (
        query_offset_0[:, None].to(tl.int64)
        * (num_query_heads * NUM_SEGMENTS_PER_SEQ * KV_LORA_RANK)
        + query_offset_1[:, None] * (NUM_SEGMENTS_PER_SEQ * KV_LORA_RANK)
        + segm_idx * KV_LORA_RANK
        + tl.arange(0, KV_LORA_RANK)[None, :]
    )
    tl.store(
        segm_output_ptr + segm_output_offset,
        acc,
        mask=query_mask_0[:, None] & query_mask_1[:, None],
    )
    segm_offset = (
        query_offset_0.to(tl.int64) * (num_query_heads * NUM_SEGMENTS_PER_SEQ)
        + query_offset_1 * NUM_SEGMENTS_PER_SEQ
        + segm_idx
    )
    tl.store(segm_max_ptr + segm_offset, M, mask=query_mask_0 & query_mask_1)
    tl.store(segm_expsum_ptr + segm_offset, L, mask=query_mask_0 & query_mask_1)


@triton.jit
def _mla_decode_fwd_reduce_kernel(
    output_ptr,  # [num_tokens, num_query_heads, head_size]
    segm_output_ptr,
    # [num_tokens, num_query_heads, max_num_segments, head_size]
    segm_max_ptr,  # [num_tokens, num_query_heads, max_num_segments]
    segm_expsum_ptr,  # [num_tokens, num_query_heads, max_num_segments]
    seq_lens_ptr,  # [num_seqs]
    out_scale_ptr,  # float32
    num_seqs,  # int
    num_query_heads: tl.constexpr,  # int
    output_stride_0: tl.int64,  # int
    output_stride_1: tl.int64,  # int, should be equal to head_size
    block_tables_stride: tl.int64,  # int
    num_tokens_per_seq: tl.int32,
    TILE_SIZE: tl.constexpr,  # int
    KV_LORA_RANK: tl.constexpr,  # int
    query_start_len_ptr,  # [num_seqs+1]
    BLOCK_Q: tl.constexpr,  # int
    NUM_SEGMENTS_PER_SEQ: tl.constexpr,  # int
    ALL_DECODE: tl.constexpr = False,  # int
):
    query_token_idx = tl.program_id(0)
    query_head_idx = tl.program_id(1)

    out_scale = None
    if out_scale_ptr is not None:
        out_scale = 1 / tl.load(out_scale_ptr)

    if ALL_DECODE:
        seq_idx = query_token_idx
    else:
        seq_idx = query_token_idx // num_tokens_per_seq

    # sequence len for this particular sequence
    seq_len = tl.load(seq_lens_ptr + seq_idx)

    # number of segments for this particular sequence
    num_segments = NUM_SEGMENTS_PER_SEQ
    tiles_per_segment = cdiv_fn(seq_len, num_segments * TILE_SIZE)

    # create masks for subsequent loads
    act_num_segments = cdiv_fn(seq_len, tiles_per_segment * TILE_SIZE)
    segm_mask = tl.arange(0, NUM_SEGMENTS_PER_SEQ) < tl.full(
        [NUM_SEGMENTS_PER_SEQ], act_num_segments, dtype=tl.int32
    )

    # load segment maxima
    segm_offset = (
        query_token_idx.to(tl.int64) * (num_query_heads * NUM_SEGMENTS_PER_SEQ)
        + query_head_idx * NUM_SEGMENTS_PER_SEQ
        + tl.arange(0, NUM_SEGMENTS_PER_SEQ)
    )
    segm_max = tl.load(segm_max_ptr + segm_offset, mask=segm_mask, other=float("-inf"))
    overall_max = tl.max(segm_max)

    # load and rescale segment exp sums
    segm_expsum = tl.load(segm_expsum_ptr + segm_offset, mask=segm_mask, other=0.0)
    segm_expsum = segm_expsum * tl.math.exp2(segm_max - overall_max)
    overall_expsum = tl.sum(segm_expsum)

    # load, rescale, and add segment attention outputs
    segm_output_offset = (
        query_token_idx.to(tl.int64)
        * (num_query_heads * NUM_SEGMENTS_PER_SEQ * KV_LORA_RANK)
        + query_head_idx * (NUM_SEGMENTS_PER_SEQ * KV_LORA_RANK)
        + tl.arange(0, NUM_SEGMENTS_PER_SEQ)[:, None] * KV_LORA_RANK
        + tl.arange(0, KV_LORA_RANK)[None, :]
    )
    segm_output = tl.load(
        segm_output_ptr + segm_output_offset,
        mask=segm_mask[:, None],
        other=0.0,
    )
    segm_output *= tl.math.exp2(segm_max - overall_max)[:, None]
    acc_sum = tl.sum(segm_output, axis=0)
    # safely divide by overall_expsum, returning 0.0 if overall_expsum is 0
    acc = tl.where(overall_expsum == 0.0, 0.0, acc_sum / overall_expsum)

    if out_scale_ptr is not None:
        acc = acc * out_scale

    # write result
    output_offset = (
        query_token_idx * output_stride_0
        + query_head_idx * output_stride_1
        + tl.arange(0, KV_LORA_RANK)
    )
    tl.store(output_ptr + output_offset, acc)
