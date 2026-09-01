import torch
import triton
import triton.language as tl

from aiter.ops.triton.utils.types import e4m3_dtype
from aiter.ops.triton._triton_kernels.attention.unified_attention import (
    cdiv_fn,
    find_seq_idx,
)

float8_info = torch.finfo(e4m3_dtype)


# ---- KR tailsplit ----
# Split-KV variant of kernel_unified_attention_2d used to fill the trailing,
# partially-occupied dispatch round of the 2-D launch.  Same math (online
# softmax), but the KV range of each q-block is split NUM_SEGMENTS ways and the
# per-segment (acc, max, expsum) are reduced by reduce_segments_tail.


@triton.jit
def kernel_unified_attention_2d_split(
    segm_output_ptr,  # [nrows, NUM_SEGMENTS, HEAD_SIZE_PADDED] fp32
    segm_max_ptr,  # [nrows, NUM_SEGMENTS] fp32
    segm_expsum_ptr,  # [nrows, NUM_SEGMENTS] fp32
    query_ptr,
    key_cache_ptr,
    value_cache_ptr,
    sink_ptr,
    block_tables_ptr,
    seq_lens_ptr,
    scale: tl.constexpr,
    q_descale_ptr,
    k_descale_ptr,
    v_descale_ptr,
    num_query_heads: tl.constexpr,
    num_queries_per_kv: tl.constexpr,
    num_kv_heads: tl.constexpr,
    block_table_stride: tl.int64,
    query_stride_0: tl.int64,
    query_stride_1: tl.int64,
    BLOCK_SIZE: tl.constexpr,
    TILE_SIZE: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    HEAD_SIZE_PADDED: tl.constexpr,
    USE_SINKS: tl.constexpr,
    stride_k_cache_0: tl.int64,
    stride_k_cache_1: tl.int64,
    stride_k_cache_2: tl.int64,
    stride_k_cache_3: tl.constexpr,
    stride_v_cache_0: tl.int64,
    stride_v_cache_1: tl.int64,
    stride_v_cache_2: tl.int64,
    stride_v_cache_3: tl.constexpr,
    query_start_len_ptr,
    BLOCK_Q: tl.constexpr,
    num_seqs: tl.int32,
    BLOCK_M: tl.constexpr,
    q_block_offset: tl.int32,
    NUM_SEGMENTS: tl.constexpr,
):
    kv_head_idx = tl.program_id(0)
    tail_blk = tl.program_id(1)
    segm_idx = tl.program_id(2)

    RCP_LN2 = 1.4426950408889634
    qk_scale = scale * RCP_LN2

    q_block_global_idx = q_block_offset + tail_blk

    seq_idx = find_seq_idx(
        query_start_len_ptr, q_block_global_idx, num_seqs, BLOCK_Q, True
    )
    q_block_start_idx = tl.load(query_start_len_ptr + seq_idx) // BLOCK_Q + seq_idx
    q_block_local_idx = q_block_global_idx - q_block_start_idx
    cur_batch_in_all_start_index = tl.load(query_start_len_ptr + seq_idx)
    cur_batch_in_all_stop_index = tl.load(query_start_len_ptr + seq_idx + 1)
    cur_batch_query_len = cur_batch_in_all_stop_index - cur_batch_in_all_start_index

    if q_block_local_idx * BLOCK_Q >= cur_batch_query_len:
        return

    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_SIZE_PADDED)
    offs_t = tl.arange(0, TILE_SIZE)
    query_pos = q_block_local_idx * BLOCK_Q + offs_m // num_queries_per_kv

    query_offset_0 = cur_batch_in_all_start_index + query_pos
    query_offset_1 = kv_head_idx * num_queries_per_kv + offs_m % num_queries_per_kv
    query_offset = (
        query_offset_0[:, None] * query_stride_0
        + query_offset_1[:, None] * query_stride_1
        + offs_d[None, :]
    )

    if HEAD_SIZE_PADDED != HEAD_SIZE:
        dim_mask = offs_d < HEAD_SIZE
    else:
        dim_mask = tl.full((1,), 1, dtype=tl.int1)
    query_mask_0 = query_pos < cur_batch_query_len
    query_mask_1 = query_offset_1 < num_query_heads

    Q = tl.load(
        query_ptr + query_offset,
        mask=dim_mask[None, :] & query_mask_0[:, None] & query_mask_1[:, None],
        other=0.0,
    )

    block_table_offset = seq_idx * block_table_stride

    if USE_SINKS and segm_idx == 0:
        M = (
            tl.load(
                sink_ptr + query_offset_1, mask=query_mask_1, other=float("-inf")
            ).to(dtype=tl.float32)
            * RCP_LN2
        )
    else:
        M = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)

    L = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_SIZE_PADDED], dtype=tl.float32)

    seq_len = tl.load(seq_lens_ptr + seq_idx)
    context_len = seq_len - cur_batch_query_len

    max_seq_prefix_len = (
        context_len
        + q_block_local_idx * BLOCK_Q
        + (BLOCK_M - 1) // num_queries_per_kv
        + 1
    )
    max_seq_prefix_len = tl.minimum(max_seq_prefix_len, seq_len)
    num_tiles = cdiv_fn(max_seq_prefix_len, TILE_SIZE)

    tiles_per_segment = cdiv_fn(num_tiles, NUM_SEGMENTS)
    tile_start = segm_idx * tiles_per_segment
    tile_end = tl.minimum(tile_start + tiles_per_segment, num_tiles)

    if q_descale_ptr is not None:
        qk_scale = qk_scale * tl.load(q_descale_ptr)
    if k_descale_ptr is not None and v_descale_ptr is not None:
        k_descale = tl.load(k_descale_ptr)
        v_descale = tl.load(v_descale_ptr)
        qk_scale = qk_scale * k_descale
    else:
        v_descale = None

    for j in range(tile_start, tile_end):
        seq_offset = j * TILE_SIZE + offs_t
        if TILE_SIZE == BLOCK_SIZE:
            tile_mask = tl.full((1,), 1, dtype=tl.int1)
        else:
            tile_mask = seq_offset < max_seq_prefix_len

        physical_block_idx = tl.load(
            block_tables_ptr + block_table_offset + seq_offset // BLOCK_SIZE
        ).to(tl.int64)

        v_offset = (
            physical_block_idx[:, None] * stride_v_cache_0
            + kv_head_idx * stride_v_cache_2
            + offs_d[None, :] * stride_v_cache_3
            + (seq_offset % BLOCK_SIZE)[:, None] * stride_v_cache_1
        )
        v_mask = dim_mask[None, :] & tile_mask[:, None]

        k_offset = (
            physical_block_idx[None, :] * stride_k_cache_0
            + kv_head_idx * stride_k_cache_2
            + offs_d[:, None] * stride_k_cache_3
            + (seq_offset % BLOCK_SIZE)[None, :] * stride_k_cache_1
        )
        k_mask = dim_mask[:, None] & tile_mask[None, :]

        K = tl.load(key_cache_ptr + k_offset, mask=k_mask, other=0.0).to(Q.dtype)
        V = tl.load(value_cache_ptr + v_offset, mask=v_mask, other=0.0).to(Q.dtype)

        S = qk_scale * tl.dot(Q, K)

        seq_mask = seq_offset[None, :] < context_len + query_pos[:, None] + 1
        S = tl.where(
            query_mask_1[:, None] & query_mask_0[:, None] & seq_mask, S, float("-inf")
        )

        m_j = tl.maximum(M, tl.max(S, axis=1))
        m_j = tl.where(m_j > float("-inf"), m_j, 0.0)
        P = tl.math.exp2(S - m_j[:, None])
        l_j = tl.sum(P, axis=1)
        alpha = tl.math.exp2(M - m_j)
        acc = acc * alpha[:, None]
        L = L * alpha + l_j
        M = m_j
        acc = tl.dot(P.to(V.dtype), V, acc=acc)

    if v_descale is not None:
        acc = acc * v_descale

    row = (tail_blk * num_kv_heads + kv_head_idx).to(tl.int64) * BLOCK_M + offs_m.to(
        tl.int64
    )
    segm_output_offset = (
        row[:, None] * (NUM_SEGMENTS * HEAD_SIZE_PADDED)
        + segm_idx * HEAD_SIZE_PADDED
        + offs_d[None, :]
    )
    tl.store(
        segm_output_ptr + segm_output_offset,
        acc,
        mask=dim_mask[None, :] & query_mask_0[:, None] & query_mask_1[:, None],
    )
    segm_offset = row * NUM_SEGMENTS + segm_idx
    tl.store(segm_max_ptr + segm_offset, M, mask=query_mask_0 & query_mask_1)
    tl.store(segm_expsum_ptr + segm_offset, L, mask=query_mask_0 & query_mask_1)


@triton.jit
def reduce_segments_tail(
    output_ptr,
    segm_output_ptr,
    segm_max_ptr,
    segm_expsum_ptr,
    query_start_len_ptr,
    out_scale_ptr,
    num_seqs: tl.int32,
    num_query_heads: tl.constexpr,
    num_queries_per_kv: tl.constexpr,
    num_kv_heads: tl.constexpr,
    output_stride_0: tl.int64,
    output_stride_1: tl.int64,
    HEAD_SIZE: tl.constexpr,
    HEAD_SIZE_PADDED: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_M: tl.constexpr,
    NUM_SEGMENTS: tl.constexpr,
    q_block_offset: tl.int32,
    FP8_MIN: tl.constexpr = float8_info.min,
    FP8_MAX: tl.constexpr = float8_info.max,
):
    row = tl.program_id(0)
    m = row % BLOCK_M
    t = row // BLOCK_M
    kv_head_idx = t % num_kv_heads
    tail_blk = t // num_kv_heads

    q_block_global_idx = q_block_offset + tail_blk
    seq_idx = find_seq_idx(
        query_start_len_ptr, q_block_global_idx, num_seqs, BLOCK_Q, True
    )
    q_block_start_idx = tl.load(query_start_len_ptr + seq_idx) // BLOCK_Q + seq_idx
    q_block_local_idx = q_block_global_idx - q_block_start_idx
    cur_batch_in_all_start_index = tl.load(query_start_len_ptr + seq_idx)
    cur_batch_in_all_stop_index = tl.load(query_start_len_ptr + seq_idx + 1)
    cur_batch_query_len = cur_batch_in_all_stop_index - cur_batch_in_all_start_index

    query_pos = q_block_local_idx * BLOCK_Q + m // num_queries_per_kv
    head_idx = kv_head_idx * num_queries_per_kv + m % num_queries_per_kv
    if query_pos >= cur_batch_query_len:
        return
    if head_idx >= num_query_heads:
        return

    offs_s = tl.arange(0, NUM_SEGMENTS)
    offs_d = tl.arange(0, HEAD_SIZE_PADDED)
    base = row.to(tl.int64) * NUM_SEGMENTS

    segm_max = tl.load(segm_max_ptr + base + offs_s)
    overall_max = tl.max(segm_max)
    rescale = tl.math.exp2(segm_max - overall_max)
    segm_expsum = tl.load(segm_expsum_ptr + base + offs_s)
    overall_expsum = tl.sum(segm_expsum * rescale)

    segm_output = tl.load(
        segm_output_ptr
        + base * HEAD_SIZE_PADDED
        + offs_s[:, None] * HEAD_SIZE_PADDED
        + offs_d[None, :],
        mask=(offs_d < HEAD_SIZE)[None, :],
        other=0.0,
    )
    acc_sum = tl.sum(segm_output * rescale[:, None], axis=0)
    acc = tl.where(overall_expsum == 0.0, 0.0, acc_sum / overall_expsum)
    if out_scale_ptr is not None:
        acc = acc / tl.load(out_scale_ptr)
    if output_ptr.type.element_ty.is_fp8():
        acc = tl.clamp(acc, FP8_MIN, FP8_MAX)

    out_off = (
        (cur_batch_in_all_start_index + query_pos).to(tl.int64) * output_stride_0
        + head_idx * output_stride_1
        + offs_d
    )
    tl.store(
        output_ptr + out_off,
        acc.to(output_ptr.type.element_ty),
        mask=offs_d < HEAD_SIZE,
    )
