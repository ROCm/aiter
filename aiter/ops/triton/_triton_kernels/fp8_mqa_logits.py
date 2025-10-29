import triton
import triton.language as tl


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
    stride_q_s: tl.int64,
    stride_q_h: tl.constexpr,
    stride_q_d: tl.constexpr,
    stride_kv_s: tl.int64,
    stride_kv_d: tl.constexpr,
    stride_w_s: tl.int64,
    stride_w_h: tl.constexpr,
    stride_logits_s: tl.int64,
    stride_logits_k: tl.int64,
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

    logits_row_ptrs = logits_ptr + row_offsets[:, None] * stride_logits_s

    # Loop over KV tiles
    for kv_tile_ind in tl.range(kv_tile_start, kv_tile_end):
        kv_col_offsets = (kv_tile_ind * BLOCK_KV + tl.arange(0, BLOCK_KV))[None, :]
        kv_col_mask = kv_col_offsets < seq_len_kv

        # Load KV tile [HEAD_SIZE, BLOCK_KV]
        kv_ptrs = KV_ptr + kv_col_offsets * stride_kv_s + d_inds[:, None] * stride_kv_d
        kv_block = tl.load(kv_ptrs, kv_col_mask, other=0.0)
        kv_scales = tl.load(kv_scales_ptr + kv_col_offsets, kv_col_mask, other=0.0).to(
            tl.float32
        )
        # [BLOCK_Q*NUM_HEADS, BLOCK_KV] = [BLOCK_Q*NUM_HEADS, HEAD_SIZE] x [HEAD_SIZE, BLOCK_KV]
        scores = tl.dot(q_block, kv_block)

        # Multiply by kv_scales (broadcast along rows)
        scores = scores * kv_scales
        scores = tl.reshape(scores, (BLOCK_Q, NUM_HEADS, BLOCK_KV))

        # ReLU
        scores = tl.maximum(scores, 0.0)

        # Apply per-row head weights and sum over heads
        scores = scores * w_block[:, :, None]
        # [BLOCK_Q, BK]
        scores = tl.sum(scores, axis=1)

        # [BQ, BK]
        in_window = (kv_col_offsets >= start_inds[:, None]) & (
            kv_col_offsets < end_inds[:, None]
        )
        store_mask = (
            (row_offsets[:, None] < seq_len) & (kv_col_offsets < seq_len_kv) & in_window
        )

        # Store to logits [BLOCK_Q, BK]
        logits_ptrs = logits_row_ptrs + kv_col_offsets * stride_logits_k
        tl.store(logits_ptrs, scores, mask=store_mask)