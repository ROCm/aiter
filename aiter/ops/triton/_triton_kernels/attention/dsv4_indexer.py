import triton
import triton.language as tl


@triton.jit
def _dsv4_indexer_dense_kernel(
    out_ptr,  # [num_tokens, topk]
    positions_ptr,  # [num_tokens]
    out_stride_t: tl.int64,
    out_stride_k: tl.int64,
    n_committed: tl.constexpr,
    offset: tl.int32,
    ratio: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    token_id = tl.program_id(0)
    offs_k = tl.arange(0, BLOCK_K)
    pos = tl.load(positions_ptr + token_id).to(tl.int32)
    causal_limit = (pos + 1) // ratio
    valid = (offs_k < n_committed) & (offs_k < causal_limit)
    out = tl.where(valid, offs_k + offset, -1)
    tl.store(
        out_ptr + token_id * out_stride_t + offs_k * out_stride_k,
        out,
        mask=offs_k < n_committed,
    )


@triton.jit
def _dsv4_indexer_score_kernel(
    score_ptr,  # [num_tokens, kv_len], fp32
    q_ptr,  # [num_tokens, num_heads, head_dim]
    kv_ptr,  # [kv_len, head_dim]
    weights_ptr,  # [num_tokens, num_heads]
    positions_ptr,  # [num_tokens]
    q_stride_t: tl.int64,
    q_stride_h: tl.int64,
    q_stride_d: tl.int64,
    kv_stride_t: tl.int64,
    kv_stride_d: tl.int64,
    weights_stride_t: tl.int64,
    weights_stride_h: tl.int64,
    score_stride_t: tl.int64,
    score_stride_k: tl.int64,
    num_heads: tl.constexpr,
    head_dim: tl.constexpr,
    kv_len: tl.constexpr,
    ratio: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    token_id = tl.program_id(0)
    tile_id = tl.program_id(1)

    offs_t = tile_id * BLOCK_T + tl.arange(0, BLOCK_T)
    offs_d = tl.arange(0, BLOCK_D)
    d_mask = offs_d < head_dim
    acc = tl.zeros((BLOCK_T,), dtype=tl.float32)

    for h_start in range(0, num_heads, BLOCK_H):
        offs_h = h_start + tl.arange(0, BLOCK_H)
        h_mask = offs_h < num_heads

        q = tl.load(
            q_ptr
            + token_id * q_stride_t
            + offs_h[:, None] * q_stride_h
            + offs_d[None, :] * q_stride_d,
            mask=h_mask[:, None] & d_mask[None, :],
            other=0.0,
            cache_modifier=".cg",
        )
        kv = tl.load(
            kv_ptr + offs_t[None, :] * kv_stride_t + offs_d[:, None] * kv_stride_d,
            mask=(offs_t[None, :] < kv_len) & d_mask[:, None],
            other=0.0,
            cache_modifier=".cg",
        )
        dots = tl.dot(q, kv)
        dots = tl.maximum(dots, 0.0)
        weights = tl.load(
            weights_ptr + token_id * weights_stride_t + offs_h * weights_stride_h,
            mask=h_mask,
            other=0.0,
            cache_modifier=".cg",
        ).to(tl.float32)
        acc += tl.sum(dots * weights[:, None], axis=0)

    pos = tl.load(positions_ptr + token_id).to(tl.int32)
    causal_limit = (pos + 1) // ratio
    valid = (offs_t < kv_len) & (offs_t < causal_limit)
    acc = tl.where(valid, acc, float("-inf"))
    tl.store(
        score_ptr + token_id * score_stride_t + offs_t * score_stride_k,
        acc,
        mask=offs_t < kv_len,
    )


@triton.jit
def _dsv4_indexer_finalize_kernel(
    out_ptr,  # [num_tokens, topk], int32
    values_ptr,  # [num_tokens, topk], fp32
    indices_ptr,  # [num_tokens, topk], int64 from aiter topk
    out_stride_t: tl.int64,
    out_stride_k: tl.int64,
    values_stride_t: tl.int64,
    values_stride_k: tl.int64,
    indices_stride_t: tl.int64,
    indices_stride_k: tl.int64,
    offset: tl.int32,
    topk: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    token_id = tl.program_id(0)
    offs_k = tl.arange(0, BLOCK_K)
    values = tl.load(
        values_ptr + token_id * values_stride_t + offs_k * values_stride_k,
        mask=offs_k < topk,
        other=float("-inf"),
    )
    indices = tl.load(
        indices_ptr + token_id * indices_stride_t + offs_k * indices_stride_k,
        mask=offs_k < topk,
        other=-1,
    ).to(tl.int32)
    out = tl.where(values > -3.0e38, indices + offset, -1)
    tl.store(
        out_ptr + token_id * out_stride_t + offs_k * out_stride_k,
        out,
        mask=offs_k < topk,
    )
