import pytest
import torch

from aiter.ops.triton.attention.sparse_mqa_sink import sparse_mqa_sink


def _reference(q, kv, topk, attn_sink, scale):
    t, h, d = q.shape
    out = torch.empty_like(q)
    qf = q.float()
    kvf = kv.float()
    for i in range(t):
        valid = topk[i] >= 0
        if not bool(valid.any()):
            out[i].zero_()
            continue
        idx = topk[i, valid].long()
        k = kvf[idx]
        scores = torch.matmul(qf[i], k.t()) * scale
        combined = torch.cat([scores, attn_sink.float().view(h, 1)], dim=-1)
        weights = torch.softmax(combined, dim=-1)[..., :-1]
        out[i] = torch.matmul(weights, k).to(out.dtype)
    return out


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.parametrize("topk_count", [16, 48])
def test_sparse_mqa_sink_matches_torch(topk_count):
    torch.manual_seed(0)
    tokens, heads, dim = 5, 16, 64
    kv_len, block_size = 73, 32
    num_blocks = (kv_len + block_size - 1) // block_size
    padded = num_blocks * block_size

    q = torch.randn(tokens, heads, dim, device="cuda", dtype=torch.bfloat16)
    kv_flat = torch.randn(padded, dim, device="cuda", dtype=torch.bfloat16)
    kv_flat[kv_len:].zero_()
    kv_blocks = kv_flat.view(num_blocks, block_size, dim)
    topk = torch.randint(0, kv_len, (tokens, topk_count), device="cuda", dtype=torch.int32)
    topk[0, -3:] = -1
    attn_sink = torch.randn(heads, device="cuda", dtype=torch.float32)
    cu = torch.tensor([0, tokens], device="cuda", dtype=torch.int32)
    seqused = torch.tensor([kv_len], device="cuda", dtype=torch.int32)
    block_table = torch.arange(num_blocks, device="cuda", dtype=torch.int32).view(1, -1)
    out = torch.empty_like(q)

    sparse_mqa_sink(q, kv_blocks, out, cu, seqused, dim**-0.5, topk, block_table, attn_sink)
    ref = _reference(q, kv_flat[:kv_len], topk, attn_sink, dim**-0.5)
    torch.testing.assert_close(out, ref, rtol=2e-2, atol=2e-2)
