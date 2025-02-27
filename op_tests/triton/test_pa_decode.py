import triton
import triton.language as tl
import torch
import pytest
import random
from aiter.ops.triton.pa_decode import paged_attention_decode


def paged_attention_decode_ref(
    output,  # [num_seqs, num_q_heads, head_sz]
    query,  # [num_seqs, num_kv_heads, head_sz],
    k_cache,  # [num_seqs, num_kv_heads, head_sz/x, blk_sz, x]
    v_cache,  # [num_seqs, num_kv_heads, head_sz, blk_sz]
    blk_tables,  # [num_seq, max_num_blks_per_seq]
    ctx_lens,  # [num_seqs]
) -> None:
    num_q_heads = query.shape[1]
    num_kv_heads = v_cache.shape[1]
    q_grp_sz = num_q_heads // num_kv_heads
    head_sz = v_cache.shape[2]
    kv_blk_sz = v_cache.shape[3]

    num_seqs = query.shape[0]
    for s in range(num_seqs):
        q = query[s].unsqueeze(0)
        blk_tbl = blk_tables[s]
        ctx_len = ctx_lens[s]

        keys = []
        values = []
        for j in range(ctx_len):
            blk_number = int(blk_tbl[j // kv_blk_sz])
            blk_offset = j % kv_blk_sz

            k = k_cache[blk_number, :, :, blk_offset, :]
            k = k.reshape(num_kv_heads, head_sz)
            if q_grp_sz != 1:
                k = k.repeat_interleave(q_grp_sz, 0)
            keys.append(k)

            v = v_cache[blk_number, :, :, blk_offset]
            if q_grp_sz != 1:
                v = v.repeat_interleave(q_grp_sz, 0)
            values.append(v)

        keys = torch.stack(keys, dim=0)
        values = torch.stack(values, dim=0)

        scale = 1.0 / (head_sz**0.5)
        q = q * scale
        attn = torch.einsum("qhd,khd->hqk", q, keys)
        attn = torch.softmax(attn, dim=-1)
        out = torch.einsum("hqk, khd->qhd", attn, values)
        out = out.view(num_q_heads, head_sz)
        output[s].copy_(out, non_blocking=True)


tl_to_torch_dtype = {tl.bfloat16: torch.bfloat16, tl.float16: torch.float16}


@pytest.mark.parametrize(
    "B, H_Q, H_KV, D, KV_BLK_SZ, SEQ_LEN",
    [
        # basic
        (1, 1, 1, 1, 1, 1),
        (1, 1, 1, 1, 1, 10),
        (2, 1, 1, 1, 1, 10),
        (4, 1, 1, 1, 1, 10),
        (8, 1, 1, 1, 1, 10),
        (16, 1, 1, 1, 1, 10),
        (64, 1, 1, 1, 1, 10),
        # H_Q and H_KV > 1
        (1, 4, 4, 1, 1, 1),
        (1, 4, 4, 1, 1, 10),
        # Head_dim > 1
        (1, 1, 1, 8, 1, 1),
        (1, 1, 1, 8, 1, 10),
        # H_Q and H_KV > 1 and Head_dim > 1
        (1, 4, 4, 8, 1, 1),
        (1, 4, 4, 8, 1, 10),
        (4, 4, 4, 8, 1, 10),
        (16, 4, 4, 8, 1, 10),
        (32, 4, 4, 8, 1, 10),
        # H_Q and H_KV > 1 and Head_dim > 1 and KV_BLK_SZ > 1
        (1, 1, 1, 1, 1, 1),
        (1, 1, 1, 1, 4, 4),
        (1, 1, 1, 1, 4, 8),
        (1, 1, 1, 1, 4, 10),
        (1, 1, 1, 1, 16, 1),
        (1, 1, 1, 1, 16, 8),
        (1, 1, 1, 1, 16, 16),
        (1, 1, 1, 1, 16, 32),
        (1, 1, 1, 1, 16, 30),
        (1, 1, 1, 1, 16, 56),
        (1, 1, 1, 1, 16, 128),
        (4, 1, 1, 1, 16, 128),
        (16, 1, 1, 1, 16, 128),
        (32, 1, 1, 1, 16, 128),
        (64, 1, 1, 1, 16, 221),
        # GQA Basic
        (1, 2, 1, 16, 16, 1),
        (1, 2, 1, 16, 16, 10),
        # GQA Basic
        (1, 4, 2, 16, 16, 1),
        (1, 4, 2, 16, 16, 10),
        (1, 4, 2, 128, 16, 1),
        (1, 4, 2, 128, 16, 10),
        (1, 6, 2, 128, 16, 1),
        (1, 6, 2, 128, 16, 10),
        (1, 6, 2, 128, 16, 16),
        (1, 6, 2, 128, 16, 30),
        (1, 6, 2, 128, 16, 32),
        (1, 6, 2, 128, 16, 48),
        (1, 6, 2, 128, 16, 56),
        (1, 6, 2, 128, 16, 64),
        (1, 6, 2, 128, 16, 128),
        (1, 8, 2, 128, 16, 128),
        (4, 8, 2, 128, 16, 128),
        (16, 8, 2, 128, 16, 128),
        (32, 8, 2, 128, 16, 128),
        (64, 8, 2, 128, 16, 200),
    ],
)
# q_dtype, kv_dtype, compute_type, output_type
# INT8xINT8 -> BF16-> BF16
# FP8xFP8 -> BF16-> FP8
# BF16xINT8 -> BF16-> BF16
# BF16xFP8 -> BF16-> BF16
# BF16xBF16->BF16->BF16
# FP16xFP16->FP16->FP16
@pytest.mark.parametrize(
    "dtype, kv_cache_dtype, compute_type, output_type",
    [
        (torch.float16, torch.float16, tl.float16, torch.float16),
        (torch.bfloat16, torch.bfloat16, tl.bfloat16, torch.bfloat16),
        (torch.bfloat16, torch.float8_e4m3fnuz, tl.bfloat16, torch.bfloat16),
        (torch.bfloat16, torch.int8, tl.bfloat16, torch.bfloat16),
        (torch.float8_e4m3fnuz, torch.float8_e4m3fnuz, tl.bfloat16, torch.bfloat16),
        (torch.int8, torch.int8, tl.bfloat16, torch.bfloat16),
    ],
)
def test_paged_attn(
    B,
    H_Q,
    H_KV,
    D,
    KV_BLK_SZ,
    SEQ_LEN,
    dtype,
    kv_cache_dtype,
    compute_type,
    output_type,
):
    torch.set_printoptions(threshold=100000)
    num_blocks = 4

    if dtype not in (torch.bfloat16, torch.float16, torch.float32):
        query = torch.randn(
            B, H_Q, D, dtype=torch.float16, device="cuda"
        )  # assumption dtype is 8bits or lower
        query = query.to(dtype=dtype, device="cuda")
    else:
        query = torch.randn(B, H_Q, D, dtype=dtype, device="cuda")

    if kv_cache_dtype not in (torch.bfloat16, torch.float16, torch.float32):
        x = min(D, 16 // torch.tensor([], dtype=torch.float16).element_size())
        key_cache = torch.randn(
            num_blocks, H_KV, D // x, KV_BLK_SZ, x, dtype=torch.float16, device="cuda"
        )
        value_cache = torch.randn(
            num_blocks, H_KV, D, KV_BLK_SZ, dtype=torch.float16, device="cuda"
        )
        # torch doesn't have randn for fp8 data type, so we convert here
        key_cache = key_cache.to(dtype=kv_cache_dtype)
        value_cache = value_cache.to(dtype=kv_cache_dtype)
    else:
        x = min(D, 16 // torch.tensor([], dtype=kv_cache_dtype).element_size())
        key_cache = torch.randn(
            num_blocks, H_KV, D // x, KV_BLK_SZ, x, dtype=kv_cache_dtype, device="cuda"
        )
        value_cache = torch.randn(
            num_blocks, H_KV, D, KV_BLK_SZ, dtype=kv_cache_dtype, device="cuda"
        )

    key_cache_tri = key_cache.permute(0, 1, 3, 2, 4).flatten(3, 4).contiguous().cuda()
    value_cache_tri = value_cache.permute(0, 1, 3, 2).contiguous().cuda()

    context_lens = torch.full((B,), SEQ_LEN, device="cuda")
    max_context_len = max(context_lens)
    max_num_blks_per_seq = (max_context_len + KV_BLK_SZ - 1) // KV_BLK_SZ

    block_tables = []
    for i in range(B):
        block_table = [
            random.randint(0, num_blocks - 1) for _ in range(max_num_blks_per_seq)
        ]
        block_tables.append(block_table)
    block_tables = torch.tensor(block_tables, dtype=torch.int32, device="cuda")
    attn_scale = 1.0 / (D**0.5)

    triton_output = torch.zeros(B, H_Q, D, dtype=output_type, device="cuda")
    paged_attention_decode(
        triton_output,
        query,
        key_cache_tri,
        value_cache_tri,
        context_lens,
        block_tables,
        attn_scale,
        max_context_len,
        compute_type,
    )

    # torch doesn't have support for fp8 data type, so we convert here
    if dtype not in (torch.bfloat16, torch.float16, torch.float32):
        query = query.to(tl_to_torch_dtype[compute_type])

    if kv_cache_dtype not in (torch.bfloat16, torch.float16):
        key_cache = key_cache.to(dtype=tl_to_torch_dtype[compute_type])
        value_cache = value_cache.to(dtype=tl_to_torch_dtype[compute_type])
    torch_output = torch.zeros(B, H_Q, D, dtype=output_type, device="cuda")
    paged_attention_decode_ref(
        torch_output, query, key_cache, value_cache, block_tables, context_lens
    )

    triton.testing.assert_close(triton_output, torch_output, rtol=1e-02, atol=1e-02)
