import triton
import triton.language as tl
import torch
import math


@triton.jit
def fp8_mqa_logits_kernel(
    Q_ptr,  # fp8e4m3 [seq_len, H, D]
    KV_ptr,  # fp8e4m3 [seq_len_kv, D]
    kv_scales_ptr,  # fp32 [seq_len_kv]
    weights_ptr,  # fp32 [seq_len, H]
    cu_start_ptr,  # int32 [seq_len]
    cu_end_ptr,  # int32 [seq_len]
    logits_ptr,  # fp32 [seq_len, seq_len_kv]
    seq_len,
    seq_len_kv,
    NUM_HEADS: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    # strides
    stride_q_s,
    stride_q_h: tl.constexpr,
    stride_q_d: tl.constexpr,
    stride_kv_s,
    stride_kv_d: tl.constexpr,
    stride_w_s,
    stride_w_h: tl.constexpr,
    stride_logits_s,
    stride_logits_k,
    # block sizes
    BLOCK_Q: tl.constexpr,
    BLOCK_KV: tl.constexpr,
):
    pid_q = tl.program_id(0)

    row_offsets = pid_q * BLOCK_Q + tl.arange(0, BLOCK_Q)
    q_mask = row_offsets < seq_len

    h_inds = tl.arange(0, NUM_HEADS)
    d_inds = tl.arange(0, HEAD_SIZE)

    # load Q[BLOCK_Q, NUM_HEADS, HEAD_SIZE]
    q_ptrs = (
        Q_ptr
        + row_offsets[:, None, None] * stride_q_s
        + h_inds[None, :, None] * stride_q_h
        + d_inds[None, None, :] * stride_q_d
    )

    q_block = tl.load(
        q_ptrs, mask=q_mask[:, None, None], other=0.0, cache_modifier=".cg"
    )
    q_block = tl.reshape(q_block, (BLOCK_Q * NUM_HEADS, HEAD_SIZE))

    w_ptrs = (
        weights_ptr + row_offsets[:, None] * stride_w_s + h_inds[None, :] * stride_w_h
    )
    w_block = tl.load(w_ptrs, mask=q_mask[:, None], other=0.0, cache_modifier=".cg").to(
        tl.float32
    )

    # Load start/end for each row in this block
    start_inds = tl.load(cu_start_ptr + row_offsets, mask=q_mask, other=seq_len_kv)
    end_inds = tl.load(cu_end_ptr + row_offsets, mask=q_mask, other=0)

    # Compute kv tile range, both ends are inclusive
    block_min_start = tl.min(start_inds)
    block_max_end = tl.max(end_inds)

    block_min_start = tl.maximum(block_min_start, 0)
    block_max_end = tl.minimum(block_max_end, seq_len_kv)

    kv_tile_start = block_min_start // BLOCK_KV
    kv_tile_end = (block_max_end + BLOCK_KV - 1) // BLOCK_KV

    # Loop over KV tiles
    for kv_tile_ind in tl.range(kv_tile_start, kv_tile_end):
        kv_col_offsets = kv_tile_ind * BLOCK_KV + tl.arange(0, BLOCK_KV)
        kv_col_mask = kv_col_offsets < seq_len_kv

        # Load KV tile [HEAD_SIZE, BLOCK_KV]
        kv_ptrs = (
            KV_ptr
            + kv_col_offsets[None, :] * stride_kv_s
            + d_inds[:, None] * stride_kv_d
        )
        kv_block = tl.load(kv_ptrs, mask=kv_col_mask[None, :], other=0.0)
        # [BLOCK_Q*NUM_HEADS, BLOCK_KV] = [BLOCK_Q*NUM_HEADS, HEAD_SIZE] x [HEAD_SIZE, BLOCK_KV]
        scores = tl.dot(q_block, kv_block, input_precision="ieee").to(tl.float32)

        # Multiply by kv_scales (broadcast along rows)
        kv_scales = tl.load(
            kv_scales_ptr + kv_col_offsets, mask=kv_col_mask, other=0.0
        ).to(tl.float32)
        scores = scores * kv_scales[None, :]
        scores = tl.reshape(scores, (BLOCK_Q, NUM_HEADS, BLOCK_KV))

        # ReLU
        scores = tl.maximum(scores, 0.0)

        # Apply per-row head weights and sum over heads
        scores = scores * w_block[:, :, None]
        # [BLOCK_Q, BK]
        scores = tl.sum(scores, axis=1)
        kv_cols = kv_col_offsets[None, :]
        # [BQ, BK]
        in_window = (kv_cols >= start_inds[:, None]) & (kv_cols < end_inds[:, None])
        store_mask = (
            (row_offsets[:, None] < seq_len)
            & (kv_col_offsets[None, :] < seq_len_kv)
            & in_window
        )
        scores = tl.where(store_mask, scores, float("-inf"))

        # Store to logits [BLOCK_Q, BK]
        logits_ptrs = (
            logits_ptr
            + row_offsets[:, None] * stride_logits_s
            + kv_col_offsets[None, :] * stride_logits_k
        )
        tl.store(logits_ptrs, scores, mask=store_mask)


def fp8_mqa_logits(
    Q,
    KV,
    kv_scales,
    weights,
    cu_starts,
    cu_ends,
):
    """
    Q:           [seq_len, NUM_HEADS, HEAD_SIZE], dtype float8
    KV:          [seq_len_kv, HEAD_SIZE], dtype float8
    kv_scales:   [seq_len_kv], dtype float32
    weights:     [seq_len, NUM_HEADS], dtype float32
    cu_starts:   [seq_len], dtype int32
    cu_ends:     [seq_len], dtype int32

    Returns:
    logits:      [seq_len, seq_len_kv], dtype float32 (must be initialized to -inf, because of causal masking)
    """
    # TODO (cagri): double check what value to put for causally masked logits, 0 or -inf?
    # TODO (cagri): Tune/optimize
    BLOCK_Q = 1
    BLOCK_KV = 128
    seq_len, num_heads, head_size = Q.shape
    seq_len_kv = KV.shape[0]
    # TODO (cagri): Currently assuming num_heads and head_size is power of 2.
    assert (num_heads & (num_heads - 1) == 0), "num q. heads should be power of 2."
    assert (head_size & (head_size - 1) == 0), "head size should be power of 2."
    # Initialize with -inf because of causal masking
    logits = torch.full((seq_len, seq_len_kv), 
                        fill_value=-float('inf'), 
                        dtype=torch.float32, device=Q.device)

    stride_q_s, stride_q_h, stride_q_d = Q.stride()
    stride_kv_s, stride_kv_d = KV.stride()
    stride_w_s, stride_w_h = weights.stride()
    stride_logits_s, stride_logits_k = logits.stride()
    fp8_mqa_logits_kernel[(triton.cdiv(seq_len, BLOCK_Q),)](
        Q_ptr=Q,
        KV_ptr=KV,
        kv_scales_ptr=kv_scales,
        weights_ptr=weights,
        cu_start_ptr=cu_starts,
        cu_end_ptr=cu_ends,
        logits_ptr=logits,
        seq_len=seq_len,
        seq_len_kv=seq_len_kv,
        NUM_HEADS=num_heads,
        HEAD_SIZE=head_size,
        stride_q_s=stride_q_s,
        stride_q_h=stride_q_h,
        stride_q_d=stride_q_d,
        stride_kv_s=stride_kv_s,
        stride_kv_d=stride_kv_d,
        stride_w_s=stride_w_s,
        stride_w_h=stride_w_h,
        stride_logits_s=stride_logits_s,
        stride_logits_k=stride_logits_k,
        BLOCK_Q=BLOCK_Q,
        BLOCK_KV=BLOCK_KV,
        num_warps=4,
        num_stages=2,
        waves_per_eu=2,
    )

    return logits
