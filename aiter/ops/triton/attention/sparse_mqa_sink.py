import torch
import triton

from aiter.ops.triton._triton_kernels.attention.sparse_mqa_sink import (
    _sparse_mqa_sink_kernel,
)


def sparse_mqa_sink(
    q: torch.Tensor,
    kv: torch.Tensor,
    out: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    seqused_k: torch.Tensor,
    softmax_scale: float,
    topk_indices: torch.Tensor,
    block_table: torch.Tensor,
    attn_sink: torch.Tensor,
    *,
    tile_k: int = 64,
    block_h: int = 4,
    block_d: int = 128,
    score_d: int = 64,
) -> torch.Tensor:
    """Sparse MQA with DSv4 attention-sink semantics.

    Args:
        q: [num_tokens, num_heads, head_dim], BF16/FP16.
        kv: [num_blocks, block_size, head_dim], BF16/FP16.
        out: [num_tokens, num_heads, head_dim], same dtype as q.
        cu_seqlens_q: [num_seqs + 1], int32 token offsets.
        seqused_k: [num_seqs], int32 logical KV lengths before padding.
        softmax_scale: scalar multiplier for q @ k.
        topk_indices: [num_tokens, topk], int32 logical KV positions. -1 is invalid.
        block_table: [num_seqs, max_blocks_per_seq], int32 logical->physical block IDs.
        attn_sink: [num_heads], FP32 sink logits included in the denominator only.
    """
    assert q.dim() == 3, f"q must be [T, H, D], got {q.shape}"
    assert kv.dim() == 3, f"kv must be [num_blocks, block_size, D], got {kv.shape}"
    assert out.shape == q.shape, f"out shape {out.shape} must match q {q.shape}"
    assert topk_indices.dim() == 2 and topk_indices.shape[0] == q.shape[0]
    assert cu_seqlens_q.dim() == 1
    assert seqused_k.dim() == 1
    assert block_table.dim() == 2
    assert attn_sink.shape == (q.shape[1],)
    assert kv.shape[2] == q.shape[2]

    num_tokens, num_heads, head_dim = q.shape
    block_size = kv.shape[1]
    topk_count = topk_indices.shape[1]
    num_seqs = seqused_k.shape[0]
    assert (
        cu_seqlens_q.shape[0] == num_seqs + 1
    ), (
        "cu_seqlens_q must have length num_seqs + 1, "
        f"got {cu_seqlens_q.shape[0]} vs {num_seqs + 1}"
    )
    assert (
        block_table.shape[0] == num_seqs
    ), f"block_table rows {block_table.shape[0]} must match num_seqs {num_seqs}"

    if q.numel() == 0:
        return out

    assert num_seqs > 0, "non-empty q requires at least one sequence"
    assert cu_seqlens_q[0] == 0, "cu_seqlens_q must start with 0"
    assert (
        cu_seqlens_q[-1] == num_tokens
    ), f"cu_seqlens_q[-1] {cu_seqlens_q[-1]} must equal num_tokens {num_tokens}"

    q = q.contiguous()
    kv = kv.contiguous()
    topk_indices = topk_indices.contiguous()
    cu_seqlens_q = cu_seqlens_q.contiguous()
    seqused_k = seqused_k.contiguous()
    block_table = block_table.contiguous()
    attn_sink = attn_sink.contiguous()

    # Keep the accumulator footprint comparable to the original 8x64 tile
    # while halving output-D tiles. That cuts repeated QK score work for
    # DSv4's 512-wide value vector from 8x to 4x.
    block_h = min(block_h, triton.next_power_of_2(num_heads))
    block_d = min(block_d, triton.next_power_of_2(head_dim))
    score_d = min(score_d, triton.next_power_of_2(head_dim))
    tile_k = min(tile_k, triton.next_power_of_2(max(topk_count, 1)))
    head_blocks = triton.cdiv(num_heads, block_h)
    dim_blocks = triton.cdiv(head_dim, block_d)
    grid = (num_tokens, head_blocks, dim_blocks)

    _sparse_mqa_sink_kernel[grid](
        out,
        q,
        kv,
        topk_indices,
        attn_sink,
        block_table,
        cu_seqlens_q,
        seqused_k,
        float(softmax_scale),
        q.stride(0),
        q.stride(1),
        q.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        kv.stride(0),
        kv.stride(1),
        kv.stride(2),
        topk_indices.stride(0),
        topk_indices.stride(1),
        block_table.stride(0),
        block_table.stride(1),
        num_heads,
        head_dim,
        topk_count,
        block_size,
        num_seqs,
        BLOCK_H=block_h,
        BLOCK_D=block_d,
        SCORE_D=score_d,
        TILE_K=tile_k,
        num_warps=4,
        num_stages=1,
    )
    return out
