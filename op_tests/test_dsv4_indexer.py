import pytest
import torch

from aiter.ops.triton.attention.dsv4_indexer import dsv4_indexer_topk


def _reference(
    q,
    kv,
    weights,
    positions,
    index_topk,
    offset,
    ratio=4,
    seq_ids=None,
    kv_lens=None,
):
    qf = q.float()
    kvf = kv.float()
    wf = weights.float()
    if kv.dim() == 3:
        assert seq_ids is not None
        kvf = kvf[seq_ids.long()]
        max_committed = kv.shape[1]
    else:
        max_committed = kv.shape[0]
    if kv.dim() == 3:
        scores = torch.einsum("thd,tnd->thn", qf, kvf)
    else:
        scores = torch.einsum("thd,nd->thn", qf, kvf)
    scores = (scores.relu_() * wf.unsqueeze(-1)).sum(dim=1)
    valid_limit = (positions.to(torch.long) + 1) // ratio
    if kv_lens is not None:
        valid_limit = torch.minimum(
            valid_limit, kv_lens[seq_ids.long()].to(torch.long)
        )
    valid = torch.arange(max_committed, device=q.device).unsqueeze(
        0
    ) < valid_limit.unsqueeze(1)
    scores = scores.masked_fill(~valid, float("-inf"))
    k = min(index_topk, max_committed)
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
    expected = (
        torch.arange(committed, device="cuda", dtype=torch.int32).expand(tokens, -1)
        + 128
    )
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
def test_dsv4_indexer_zero_committed_returns_empty():
    q = torch.empty(3, 64, 128, device="cuda", dtype=torch.bfloat16)
    kv = torch.empty(0, 128, device="cuda", dtype=torch.bfloat16)
    weights = torch.empty(3, 64, device="cuda", dtype=torch.float32)
    positions = torch.arange(3, device="cuda", dtype=torch.int64)

    out = dsv4_indexer_topk(q, kv, weights, positions, 512, 128)
    assert out.shape == (3, 0)
    assert out.dtype == torch.int32


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
def test_dsv4_indexer_batched_dense_causal_indices():
    torch.manual_seed(2)
    tokens, heads, dim, committed = 4, 64, 128, 16
    q = torch.randn(tokens, heads, dim, device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(2, committed, dim, device="cuda", dtype=torch.bfloat16)
    weights = torch.randn(tokens, heads, device="cuda", dtype=torch.float32)
    positions = torch.tensor([3, 7, 63, 63], device="cuda", dtype=torch.int64)
    seq_ids = torch.tensor([0, 1, 0, 1], device="cuda", dtype=torch.int32)
    kv_lens = torch.tensor([5, 9], device="cuda", dtype=torch.int32)

    out = dsv4_indexer_topk(
        q, kv, weights, positions, 64, 128, seq_ids=seq_ids, kv_lens=kv_lens
    )
    expected = (
        torch.arange(committed, device="cuda", dtype=torch.int32).expand(tokens, -1)
        + 128
    )
    valid_limit = torch.minimum((positions + 1) // 4, kv_lens[seq_ids.long()])
    valid = torch.arange(committed, device="cuda").unsqueeze(0) < valid_limit.unsqueeze(
        1
    )
    expected = torch.where(valid, expected, -1)
    torch.testing.assert_close(out, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
def test_dsv4_indexer_batched_scored_topk_no_cross_sequence_leakage():
    tokens, heads, dim, committed, k = 4, 64, 128, 32, 4
    q = torch.zeros(tokens, heads, dim, device="cuda", dtype=torch.bfloat16)
    q[:, 0, 0] = 1
    kv = torch.zeros(2, committed, dim, device="cuda", dtype=torch.bfloat16)
    kv[0, :, 0] = torch.arange(committed, device="cuda", dtype=torch.float32)
    kv[1, :, 0] = torch.arange(committed, 0, -1, device="cuda", dtype=torch.float32)
    weights = torch.zeros(tokens, heads, device="cuda", dtype=torch.float32)
    weights[:, 0] = 1
    positions = torch.full((tokens,), committed * 4, device="cuda", dtype=torch.int64)
    seq_ids = torch.tensor([0, 1, 0, 1], device="cuda", dtype=torch.int32)
    kv_lens = torch.full((2,), committed, device="cuda", dtype=torch.int32)

    out = dsv4_indexer_topk(
        q, kv, weights, positions, k, 128, seq_ids=seq_ids, kv_lens=kv_lens
    )
    ref = _reference(
        q, kv, weights, positions, k, 128, seq_ids=seq_ids, kv_lens=kv_lens
    )
    torch.testing.assert_close(out, ref)
    assert int(out[0, 0]) == 128 + committed - 1
    assert int(out[1, 0]) == 128


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.parametrize(("committed", "k"), [(2048, 512), (4096, 1024)])
def test_dsv4_indexer_large_row_topk_matches_torch(committed, k):
    torch.manual_seed(3)
    tokens, heads, dim = 1, 64, 128
    q = torch.randn(tokens, heads, dim, device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(committed, dim, device="cuda", dtype=torch.bfloat16)
    weights = torch.randn(tokens, heads, device="cuda", dtype=torch.float32)
    positions = torch.full((tokens,), committed * 4, device="cuda", dtype=torch.int64)

    out = dsv4_indexer_topk(q, kv, weights, positions, k, 128)
    ref = _reference(q, kv, weights, positions, k, 128)
    torch.testing.assert_close(out, ref)
