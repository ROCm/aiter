import pytest
import torch

from aiter.ops.triton.attention.sparse_mqa_sink import sparse_mqa_sink


def _reference(
    q, kv_blocks, topk, attn_sink, scale, cu_seqlens_q, seqused_k, block_table
):
    t, h, d = q.shape
    out = torch.empty_like(q)
    qf = q.float()
    kvf = kv_blocks.float()
    cu_cpu = cu_seqlens_q.cpu().tolist()
    for i in range(t):
        seq_idx = next(seq for seq in range(len(cu_cpu) - 1) if cu_cpu[seq + 1] > i)
        kv_len = int(seqused_k[seq_idx].item())
        valid = (topk[i] >= 0) & (topk[i] < kv_len)
        if not bool(valid.any()):
            out[i].zero_()
            continue
        idx = topk[i, valid].long()
        logical_block = idx // kv_blocks.shape[1]
        slot = idx % kv_blocks.shape[1]
        physical_block = block_table[seq_idx, logical_block].long()
        k = kvf[physical_block, slot]
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
    topk = torch.randint(
        0, kv_len, (tokens, topk_count), device="cuda", dtype=torch.int32
    )
    topk[0, -3:] = -1
    attn_sink = torch.randn(heads, device="cuda", dtype=torch.float32)
    cu = torch.tensor([0, tokens], device="cuda", dtype=torch.int32)
    seqused = torch.tensor([kv_len], device="cuda", dtype=torch.int32)
    block_table = torch.arange(num_blocks, device="cuda", dtype=torch.int32).view(1, -1)
    out = torch.empty_like(q)

    sparse_mqa_sink(
        q, kv_blocks, out, cu, seqused, dim**-0.5, topk, block_table, attn_sink
    )
    ref = _reference(q, kv_blocks, topk, attn_sink, dim**-0.5, cu, seqused, block_table)
    torch.testing.assert_close(out, ref, rtol=2e-2, atol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.parametrize(
    ("heads", "topk_count", "kv_len", "tokens"),
    [
        (64, 160, 256, 2),  # HCA 4K-style top-k
        (64, 640, 768, 2),  # V4-Flash CSA
        (128, 1152, 1280, 1),  # V4-Pro CSA
        (64, 2048, 2304, 1),  # HCA long-context smoke
    ],
)
def test_sparse_mqa_sink_dsv4_shapes_match_torch(heads, topk_count, kv_len, tokens):
    torch.manual_seed(1)
    dim, block_size = 512, 256
    num_blocks = (kv_len + block_size - 1) // block_size
    padded = num_blocks * block_size

    q = torch.randn(tokens, heads, dim, device="cuda", dtype=torch.bfloat16)
    kv_flat = torch.randn(padded, dim, device="cuda", dtype=torch.bfloat16)
    kv_flat[kv_len:].zero_()
    kv_blocks = kv_flat.view(num_blocks, block_size, dim)
    topk = torch.randint(
        0, kv_len, (tokens, topk_count), device="cuda", dtype=torch.int32
    )
    topk[0, -min(17, topk_count) :] = -1
    attn_sink = torch.linspace(-8, 8, heads, device="cuda", dtype=torch.float32)
    cu = torch.tensor([0, tokens], device="cuda", dtype=torch.int32)
    seqused = torch.tensor([kv_len], device="cuda", dtype=torch.int32)
    block_table = torch.arange(num_blocks, device="cuda", dtype=torch.int32).view(1, -1)
    out = torch.empty_like(q)

    sparse_mqa_sink(
        q, kv_blocks, out, cu, seqused, dim**-0.5, topk, block_table, attn_sink
    )
    ref = _reference(q, kv_blocks, topk, attn_sink, dim**-0.5, cu, seqused, block_table)
    torch.testing.assert_close(out, ref, rtol=3e-2, atol=3e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
def test_sparse_mqa_sink_multi_sequence_block_table_matches_torch():
    torch.manual_seed(2)
    tokens, heads, dim = 5, 64, 512
    topk_count, block_size = 160, 64
    kv_lens = [130, 177]
    max_blocks = max((length + block_size - 1) // block_size for length in kv_lens)
    total_blocks = 8

    q = torch.randn(tokens, heads, dim, device="cuda", dtype=torch.bfloat16)
    kv_blocks = torch.randn(
        total_blocks, block_size, dim, device="cuda", dtype=torch.bfloat16
    )
    kv_blocks[4:7].add_(8.0)  # make cross-sequence leakage obvious
    block_table = torch.tensor(
        [[2, 0, 1], [6, 4, 5]], device="cuda", dtype=torch.int32
    )[:, :max_blocks]
    cu = torch.tensor([0, 2, tokens], device="cuda", dtype=torch.int32)
    seqused = torch.tensor(kv_lens, device="cuda", dtype=torch.int32)
    topk = torch.empty(tokens, topk_count, device="cuda", dtype=torch.int32)
    for i, kv_len in enumerate([kv_lens[0]] * 2 + [kv_lens[1]] * 3):
        topk[i] = torch.randint(
            0, kv_len, (topk_count,), device="cuda", dtype=torch.int32
        )
    topk[1, -5:] = -1
    topk[3, -7:] = -1
    attn_sink = torch.randn(heads, device="cuda", dtype=torch.float32)
    out = torch.empty_like(q)

    sparse_mqa_sink(
        q, kv_blocks, out, cu, seqused, dim**-0.5, topk, block_table, attn_sink
    )
    ref = _reference(q, kv_blocks, topk, attn_sink, dim**-0.5, cu, seqused, block_table)
    torch.testing.assert_close(out, ref, rtol=3e-2, atol=3e-2)
