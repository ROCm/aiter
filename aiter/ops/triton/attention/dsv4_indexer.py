import torch
import triton

from aiter.ops.triton._triton_kernels.attention.dsv4_indexer import (
    _dsv4_indexer_dense_kernel,
    _dsv4_indexer_finalize_kernel,
    _dsv4_indexer_score_kernel,
)
from aiter.ops.triton.topk import topk as _aiter_topk


def dsv4_indexer_topk(
    q: torch.Tensor,
    kv: torch.Tensor,
    weights: torch.Tensor,
    positions: torch.Tensor,
    index_topk: int,
    offset: int,
    *,
    ratio: int = 4,
    block_t: int = 64,
    block_h: int = 8,
) -> torch.Tensor:
    """DeepSeek-V4 Indexer scorer + causal top-k.

    Computes the Indexer's learned sparse compressed-KV selection without
    materializing the Torch fallback's [tokens, heads, committed] score tensor:

        score[t, k] = sum_h relu(q[t, h] @ kv[k]) * weights[t, h]

    Args:
        q: [num_tokens, 64, 128], BF16/FP16/FP8-like storage accepted by Triton.
        kv: [num_committed, 128], compressed Indexer KV.
        weights: [num_tokens, 64], FP32/BF16, already includes model scaling.
        positions: [num_tokens], absolute token positions.
        index_topk: model top-k cap, 512 for V4-Flash or 1024 for V4-Pro.
        offset: index offset into the sparse-attention [window || compressed] KV.
        ratio: compression ratio. DSv4 CSA Indexer uses 4.

    Returns:
        [num_tokens, min(index_topk, num_committed)] int32. Future entries are -1.
    """
    assert q.dim() == 3, f"q must be [T, H, D], got {q.shape}"
    assert kv.dim() == 2, f"kv must be [N, D], got {kv.shape}"
    assert weights.dim() == 2, f"weights must be [T, H], got {weights.shape}"
    assert positions.dim() == 1, f"positions must be [T], got {positions.shape}"
    assert q.shape[0] == weights.shape[0] == positions.shape[0]
    assert q.shape[1] == weights.shape[1]
    assert q.shape[2] == kv.shape[1]
    assert ratio > 0

    num_tokens, num_heads, head_dim = q.shape
    n_committed = kv.shape[0]
    actual_topk = min(int(index_topk), n_committed)
    if actual_topk <= 0:
        return torch.empty((num_tokens, 0), device=q.device, dtype=torch.int32)

    q = q.contiguous()
    kv = kv.contiguous()
    weights = weights.contiguous()
    positions = positions.contiguous()
    out = torch.empty((num_tokens, actual_topk), device=q.device, dtype=torch.int32)

    # If top-k covers every committed compressed entry, the order does not
    # affect downstream sparse attention. Emit dense causal indices and skip the
    # expensive learned scorer entirely. This is the common 1k1k DSv4 case
    # where n_committed=256 and index_topk is 512/1024.
    if actual_topk == n_committed:
        block_k = triton.next_power_of_2(max(actual_topk, 1))
        _dsv4_indexer_dense_kernel[(num_tokens,)](
            out,
            positions,
            out.stride(0),
            out.stride(1),
            n_committed,
            int(offset),
            int(ratio),
            BLOCK_K=block_k,
            num_warps=4,
            num_stages=1,
        )
        return out

    score = torch.empty((num_tokens, n_committed), device=q.device, dtype=torch.float32)
    block_t = min(block_t, triton.next_power_of_2(max(n_committed, 1)))
    block_h = min(block_h, triton.next_power_of_2(num_heads))
    block_d = triton.next_power_of_2(head_dim)
    grid = (num_tokens, triton.cdiv(n_committed, block_t))
    _dsv4_indexer_score_kernel[grid](
        score,
        q,
        kv,
        weights,
        positions,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        kv.stride(0),
        kv.stride(1),
        weights.stride(0),
        weights.stride(1),
        score.stride(0),
        score.stride(1),
        num_heads,
        head_dim,
        n_committed,
        int(ratio),
        BLOCK_T=block_t,
        BLOCK_H=block_h,
        BLOCK_D=block_d,
        num_warps=4,
        num_stages=1,
    )
    values, indices = _aiter_topk(score, actual_topk, dim=-1)
    block_k = triton.next_power_of_2(max(actual_topk, 1))
    _dsv4_indexer_finalize_kernel[(num_tokens,)](
        out,
        values,
        indices,
        out.stride(0),
        out.stride(1),
        values.stride(0),
        values.stride(1),
        indices.stride(0),
        indices.stride(1),
        int(offset),
        actual_topk,
        BLOCK_K=block_k,
        num_warps=4,
        num_stages=1,
    )
    return out
