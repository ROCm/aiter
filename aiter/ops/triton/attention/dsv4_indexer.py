import torch
import triton

from aiter.ops.triton._triton_kernels.attention.dsv4_indexer import (
    _dsv4_indexer_dense_batched_kernel,
    _dsv4_indexer_dense_kernel,
    _dsv4_indexer_finalize_kernel,
    _dsv4_indexer_score_batched_kernel,
    _dsv4_indexer_score_kernel,
)
from aiter.ops.triton.topk import topk as _aiter_topk

_DEQUANT_DTYPES = (torch.float16, torch.bfloat16)


def dsv4_indexer_topk(
    q: torch.Tensor,
    kv: torch.Tensor,
    weights: torch.Tensor,
    positions: torch.Tensor,
    index_topk: int,
    offset: int,
    *,
    seq_ids: torch.Tensor | None = None,
    kv_lens: torch.Tensor | None = None,
    ratio: int = 4,
    block_t: int = 64,
    block_h: int = 8,
) -> torch.Tensor:
    """DeepSeek-V4 Indexer scorer + causal top-k.

    Computes the Indexer's learned sparse compressed-KV selection without
    materializing the Torch fallback's [tokens, heads, committed] score tensor:

        score[t, k] = sum_h relu(q[t, h] @ kv[k]) * weights[t, h]

    Args:
        q: [num_tokens, 64, 128], dequantized BF16/FP16.
        kv: [num_committed, 128] or [num_seqs, max_committed, 128],
            dequantized BF16/FP16 compressed Indexer KV.
        weights: [num_tokens, 64], FP32/BF16, already includes model scaling.
        positions: [num_tokens], absolute token positions.
        index_topk: model top-k cap, 512 for V4-Flash or 1024 for V4-Pro.
        offset: index offset into the sparse-attention [window || compressed] KV.
        seq_ids: optional [num_tokens] int32/int64 sequence IDs. Required when
            kv is batched.
        kv_lens: optional [num_seqs] int32/int64 committed KV length per sequence.
            Required when kv is batched and shorter than max_committed.
        ratio: compression ratio. DSv4 CSA Indexer uses 4.

    Returns:
        [num_tokens, min(index_topk, max_committed)] int32. Future entries are -1.

    This op does not unpack native DSv4 FP4/FP8 cache layouts or apply their
    scale tensors. Callers must pass dequantized BF16/FP16 Q/KV tensors.
    """
    assert q.dim() == 3, f"q must be [T, H, D], got {q.shape}"
    assert kv.dim() in (2, 3), f"kv must be [N, D] or [B, N, D], got {kv.shape}"
    assert weights.dim() == 2, f"weights must be [T, H], got {weights.shape}"
    assert positions.dim() == 1, f"positions must be [T], got {positions.shape}"
    assert positions.dtype in (torch.int32, torch.int64)
    assert q.shape[0] == weights.shape[0] == positions.shape[0]
    assert q.shape[1] == weights.shape[1]
    assert q.is_cuda and kv.is_cuda, "q and kv must be CUDA tensors"
    assert (
        weights.device == q.device
        and positions.device == q.device
        and kv.device == q.device
    ), "q, kv, weights, and positions must be on the same device"
    assert q.dtype in _DEQUANT_DTYPES, f"q must be dequantized BF16/FP16, got {q.dtype}"
    assert (
        kv.dtype in _DEQUANT_DTYPES
    ), f"kv must be dequantized BF16/FP16, got {kv.dtype}"
    assert weights.dtype in (
        torch.float16,
        torch.bfloat16,
        torch.float32,
    ), f"weights must be FP16/BF16/FP32, got {weights.dtype}"
    assert q.shape[2] == kv.shape[-1]
    assert index_topk >= 0
    assert ratio > 0

    num_tokens, num_heads, head_dim = q.shape
    is_batched = kv.dim() == 3
    n_committed = kv.shape[1] if is_batched else kv.shape[0]
    if is_batched:
        assert seq_ids is not None, "seq_ids is required when kv is batched"
        assert seq_ids.dim() == 1 and seq_ids.shape[0] == num_tokens
        assert seq_ids.device == q.device, "seq_ids must be on the same device as q"
        assert seq_ids.dtype in (torch.int32, torch.int64)
        if kv_lens is None:
            kv_lens = torch.full(
                (kv.shape[0],), n_committed, device=kv.device, dtype=torch.int32
            )
        assert kv_lens.dim() == 1 and kv_lens.shape[0] == kv.shape[0]
        assert kv_lens.device == q.device, "kv_lens must be on the same device as q"
        assert kv_lens.dtype in (torch.int32, torch.int64)
        if hasattr(torch, "_assert_async"):
            torch._assert_async(((seq_ids >= 0) & (seq_ids < kv.shape[0])).all())
            torch._assert_async(((kv_lens >= 0) & (kv_lens <= n_committed)).all())
        else:
            assert bool(
                ((seq_ids >= 0) & (seq_ids < kv.shape[0])).all()
            ), "seq_ids must be in range"
            assert bool(
                ((kv_lens >= 0) & (kv_lens <= n_committed)).all()
            ), "kv_lens must be in range"
    else:
        assert seq_ids is None, "seq_ids requires batched kv"
        assert kv_lens is None, "kv_lens requires batched kv"
    actual_topk = min(int(index_topk), n_committed)
    if actual_topk <= 0:
        return torch.empty((num_tokens, 0), device=q.device, dtype=torch.int32)
    if num_tokens == 0:
        return torch.empty((0, actual_topk), device=q.device, dtype=torch.int32)

    q = q.contiguous()
    kv = kv.contiguous()
    weights = weights.contiguous()
    positions = positions.contiguous()
    if seq_ids is not None:
        seq_ids = seq_ids.contiguous()
    if kv_lens is not None:
        kv_lens = kv_lens.contiguous()
    out = torch.empty((num_tokens, actual_topk), device=q.device, dtype=torch.int32)

    # If top-k covers every committed compressed entry, the order does not
    # affect downstream sparse attention. Emit dense causal indices and skip the
    # expensive learned scorer entirely. This is the common 1k1k DSv4 case
    # where n_committed=256 and index_topk is 512/1024.
    if actual_topk == n_committed:
        block_k = triton.next_power_of_2(max(actual_topk, 1))
        if is_batched:
            _dsv4_indexer_dense_batched_kernel[(num_tokens,)](
                out,
                positions,
                seq_ids,
                kv_lens,
                out.stride(0),
                out.stride(1),
                n_committed,
                int(offset),
                int(ratio),
                BLOCK_K=block_k,
                num_warps=4,
                num_stages=1,
            )
        else:
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
    if is_batched:
        _dsv4_indexer_score_batched_kernel[grid](
            score,
            q,
            kv,
            weights,
            positions,
            seq_ids,
            kv_lens,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            kv.stride(0),
            kv.stride(1),
            kv.stride(2),
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
    else:
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
