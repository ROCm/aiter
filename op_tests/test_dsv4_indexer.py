import pytest
import torch

from aiter.ops.triton.attention.dsv4_indexer import dsv4_indexer_topk


def _reference(q, kv, weights, positions, index_topk, offset, ratio=4):
    qf = q.float()
    kvf = kv.float()
    wf = weights.float()
    scores = torch.einsum("thd,nd->thn", qf, kvf)
    scores = (scores.relu_() * wf.unsqueeze(-1)).sum(dim=1)
    valid = torch.arange(kv.shape[0], device=q.device).unsqueeze(0) < (
        (positions.to(torch.long) + 1) // ratio
    ).unsqueeze(1)
    scores = scores.masked_fill(~valid, float("-inf"))
    k = min(index_topk, kv.shape[0])
    if k == 0:
        return torch.empty((q.shape[0], 0), device=q.device, dtype=torch.int32)
    values, indices = scores.topk(k, dim=-1)
    return torch.where(values > -3.0e38, indices.to(torch.int32) + offset, -1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
def test_dsv4_indexer_dense_causal_indices():
    torch.manual_seed(0)
    tokens, heads, dim, committed = 9, 64, 128, 16
    q = torch.randn(tokens, heads, dim, device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(committed, dim, device="cuda", dtype=torch.bfloat16)
    weights = torch.randn(tokens, heads, device="cuda", dtype=torch.float32)
    positions = torch.arange(tokens, device="cuda", dtype=torch.int64) + 3

    out = dsv4_indexer_topk(q, kv, weights, positions, 64, 128)
    expected = torch.arange(committed, device="cuda", dtype=torch.int32).expand(tokens, -1) + 128
    valid = torch.arange(committed, device="cuda").unsqueeze(0) < (
        (positions + 1) // 4
    ).unsqueeze(1)
    expected = torch.where(valid, expected, -1)
    torch.testing.assert_close(out, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
def test_dsv4_indexer_scored_topk_matches_torch():
    torch.manual_seed(1)
    tokens, heads, dim, committed, k = 7, 64, 128, 80, 12
    q = torch.randn(tokens, heads, dim, device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(committed, dim, device="cuda", dtype=torch.bfloat16)
    weights = torch.randn(tokens, heads, device="cuda", dtype=torch.float32)
    positions = torch.arange(tokens, device="cuda", dtype=torch.int64) + committed * 4

    out = dsv4_indexer_topk(q, kv, weights, positions, k, 128)
    ref = _reference(q, kv, weights, positions, k, 128)
    torch.testing.assert_close(out, ref)
