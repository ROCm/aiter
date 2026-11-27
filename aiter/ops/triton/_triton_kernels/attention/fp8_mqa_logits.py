import triton
import triton.language as tl

from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr

_fp8_mqa_logits_kernel_repr = make_kernel_repr(
    "_fp8_mqa_logits_kernel",
    [
        "NUM_HEADS",
        "HEAD_SIZE",
        "BLOCK_KV",
        "PACKED",
    ],
)


@triton.jit(repr=_fp8_mqa_logits_kernel_repr)
def _fp8_mqa_logits_kernel(
    Q_ptr,  # fp8e4m3 [seq_len, H, D]
    KV_ptr,  # fp8e4m3 [seq_len_kv, D]
    kv_scales_ptr,  # fp32 [seq_len_kv]
    weights_ptr,  # fp32 [seq_len, H]
    cu_start_ptr,  # int32 [seq_len]
    cu_end_ptr,  # int32 [seq_len]
    logits_ptr,  # fp32 [seq_len, seq_len_kv]; flat when PACKED
    out_offsets_ptr,  # int32/int64 [seq_len]: each row's start when PACKED
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
    BLOCK_KV: tl.constexpr,
    PACKED: tl.constexpr = False,
):
    row_id = tl.program_id(0)
    # go from larger to smaller in terms of work
    # to reduce the tail effect
    row_id = tl.num_programs(0) - row_id - 1
    tl.assume(row_id >= 0)
    tl.assume(stride_q_s > 0)
    tl.assume(stride_q_h > 0)
    tl.assume(stride_q_d > 0)
    tl.assume(stride_kv_s > 0)
    tl.assume(stride_kv_d > 0)
    tl.assume(stride_w_s > 0)
    tl.assume(stride_w_h > 0)

    h_inds = tl.arange(0, NUM_HEADS)[:, None]
    d_inds = tl.arange(0, HEAD_SIZE)

    # load Q[BLOCK_Q, NUM_HEADS, HEAD_SIZE]
    q_ptrs = (
        Q_ptr + row_id * stride_q_s + h_inds * stride_q_h + d_inds[None, :] * stride_q_d
    )

    q_block = tl.load(q_ptrs, cache_modifier=".cg")
    w_ptrs = weights_ptr + row_id * stride_w_s + h_inds * stride_w_h
    w_block = tl.load(w_ptrs, cache_modifier=".cg").to(tl.float32)

    # Load start/end for each row in this block
    start_ind = tl.load(cu_start_ptr + row_id)
    end_ind = tl.load(cu_end_ptr + row_id)

    start_ind = tl.maximum(start_ind, 0)
    end_ind = tl.minimum(end_ind, seq_len_kv)
    if PACKED:
        row_base = tl.load(out_offsets_ptr + row_id).to(tl.int64) - start_ind
    else:
        row_base = row_id * stride_logits_s
    logits_row_ptrs = logits_ptr + row_base
    shifted_end = end_ind - start_ind
    shifted_unmasked_end = shifted_end // BLOCK_KV * BLOCK_KV

    kv_col_offsets = tl.arange(0, BLOCK_KV) + start_ind
    kv_ptrs = (
        KV_ptr + kv_col_offsets[None, :] * stride_kv_s + d_inds[:, None] * stride_kv_d
    )

    kv_scales_ptrs = kv_scales_ptr + kv_col_offsets

    logits_ptrs = logits_row_ptrs + kv_col_offsets * stride_logits_k

    # Loop over KV tiles
    for _ in tl.range(0, shifted_unmasked_end, BLOCK_KV):
        kv_block = tl.load(kv_ptrs)
        kv_scales = tl.load(kv_scales_ptrs)

        # [NUM_HEADS, BLOCK_KV] = [NUM_HEADS, HEAD_SIZE] x [HEAD_SIZE, BLOCK_KV]
        scores = tl.dot(q_block, kv_block)
        # Multiply by kv_scales (broadcast along rows)
        scores = scores * kv_scales[None, :]
        # ReLU
        scores = tl.maximum(scores, 0.0)
        scores = scores * w_block
        # [NUM_HEADS, BLOCK_KV] -> [BLOCK_KV, ]
        scores = tl.sum(scores, axis=0)
        tl.store(logits_ptrs, scores)

        kv_ptrs += BLOCK_KV * stride_kv_s
        kv_scales_ptrs += BLOCK_KV
        logits_ptrs += BLOCK_KV * stride_logits_k
        kv_col_offsets += BLOCK_KV

    # masked load
    kv_col_mask = kv_col_offsets < end_ind
    kv_block = tl.load(kv_ptrs, mask=kv_col_mask[None, :], other=0.0)
    kv_scales = tl.load(kv_scales_ptrs, mask=kv_col_mask, other=0.0)

    # [NUM_HEADS, BLOCK_KV] = [NUM_HEADS, HEAD_SIZE] x [HEAD_SIZE, BLOCK_KV]
    scores = tl.dot(q_block, kv_block)
    # Multiply by kv_scales (broadcast along rows)
    scores = scores * kv_scales[None, :]
    # ReLU
    scores = tl.maximum(scores, 0.0)
    scores = scores * w_block
    # [NUM_HEADS, BLOCK_KV] -> [BLOCK_KV, ]
    scores = tl.sum(scores, axis=0)
    # masked store
    in_window = (kv_col_offsets >= start_ind) & (kv_col_offsets < end_ind)
    tl.store(logits_ptrs, scores, mask=in_window)


@triton.jit
def _fp8_paged_mqa_logits_rows_kernel(
    q_ptr,  # fp8 [B, NEXT_N, 32, 128], contiguous per request
    kv_val_ptr,  # fp8 view of the block-flat cache [num_blocks, PAGE * 132]
    kv_scale_ptr,  # fp32 view of the same cache
    weights_ptr,  # fp32 [B * NEXT_N, 32]
    ctx_ptr,  # int32 [B], or [B * NEXT_N] with CTX_PER_ROW
    block_tables_ptr,  # int32 [B, max_blocks]
    logits_ptr,  # fp32 [B * NEXT_N, >= max_model_len]
    stride_q_b,
    stride_blk,  # bytes per cache page
    stride_bt,
    stride_logits,
    max_blocks,
    num_blocks,
    max_model_len,
    NEXT_N: tl.constexpr,
    CTX_PER_ROW: tl.constexpr,
    PAGE: tl.constexpr,
    N_SPLITS: tl.constexpr,
    NUM_STAGES: tl.constexpr = 1,
):
    """Short-context paged decode logits: one query row, pages s, s + N_SPLITS, ..."""
    row = tl.program_id(0)
    split = tl.program_id(1)
    b = row // NEXT_N
    n = row % NEXT_N
    if CTX_PER_ROW:
        ctx = tl.load(ctx_ptr + row)
    else:
        ctx = tl.load(ctx_ptr + b) - NEXT_N + n + 1
    ctx = tl.minimum(tl.maximum(ctx, 0), max_model_len)
    h = tl.arange(0, 32)
    d = tl.arange(0, 128)
    q = tl.load(
        q_ptr + b.to(tl.int64) * stride_q_b + (n * 32 + h[:, None]) * 128 + d[None, :]
    )
    w = tl.load(weights_ptr + row * 32 + h)
    kc = tl.arange(0, PAGE)
    bt_row = block_tables_ptr + b.to(tl.int64) * stride_bt
    o_row = logits_ptr + row.to(tl.int64) * stride_logits
    for p in tl.range(split, tl.cdiv(ctx, PAGE), N_SPLITS, num_stages=NUM_STAGES):
        page = tl.load(bt_row + tl.minimum(p, max_blocks - 1))
        page = tl.minimum(tl.maximum(page, 0), num_blocks - 1).to(tl.int64)
        pos = p * PAGE + kc
        valid = pos < ctx
        base = page * stride_blk
        k = tl.load(
            kv_val_ptr + base + kc[:, None] * 128 + d[None, :],
            mask=valid[:, None],
            other=0.0,
        )
        sc = tl.load(kv_scale_ptr + base // 4 + PAGE * 32 + kc, mask=valid, other=0.0)
        s = tl.dot(k, tl.trans(q))  # [PAGE, 32]
        logit = tl.sum(tl.maximum(s, 0.0) * w[None, :], axis=1) * sc
        tl.store(o_row + pos, logit, mask=valid)
