# SPDX-License-Identifier: MIT
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

import torch
from aiter.ops.unified_attention import unified_attention_fwd


def test_unified_attention_ck(
    num_seqs=4,
    max_seq_len=256,
    num_heads_q=32,
    num_kv_heads=32,
    head_size=128,
    page_blk_size=64,
    dtype=torch.float16,
    mask_type=0,
    device="cuda",
):
    num_pages = (max_seq_len + page_blk_size - 1) // page_blk_size
    total_num_pages = num_seqs * num_pages

    seq_lens = torch.randint(1, max_seq_len + 1, (num_seqs,), dtype=torch.int32, device=device)
    query_start_len = torch.zeros(num_seqs + 1, dtype=torch.int32, device=device)
    query_start_len[1:] = torch.cumsum(seq_lens, dim=0)
    num_tokens = int(query_start_len[-1].item())

    query = torch.randn(num_tokens, num_heads_q, head_size, dtype=dtype, device=device)
    output = torch.empty_like(query)

    key_cache = torch.randn(total_num_pages, page_blk_size, num_kv_heads, head_size, dtype=dtype, device=device)
    value_cache = torch.randn(total_num_pages, page_blk_size, num_kv_heads, head_size, dtype=dtype, device=device)

    block_tables = torch.arange(total_num_pages, dtype=torch.int32, device=device).reshape(num_seqs, num_pages)

    scale_s = 1.0 / (head_size ** 0.5)

    unified_attention_fwd(
        output, query, key_cache, value_cache,
        block_tables, seq_lens, query_start_len,
        mask_type=mask_type,
        scale_s=scale_s,
        scale=1.0, scale_k=1.0, scale_v=1.0, scale_out=1.0,
    )
    torch.cuda.synchronize()

    assert not torch.isnan(output).any(), "Output contains NaN"
    assert not torch.isinf(output).any(), "Output contains Inf"
    print(f"  PASS: num_seqs={num_seqs}, num_tokens={num_tokens}, "
          f"heads_q={num_heads_q}, heads_kv={num_kv_heads}, hdim={head_size}, "
          f"dtype={dtype}, mask_type={mask_type}, "
          f"output_abs_mean={output.abs().mean().item():.6f}")


if __name__ == "__main__":
    print("=== CK Unified Attention Tests ===")

    print("\n[fp16, no mask]")
    test_unified_attention_ck(dtype=torch.float16, mask_type=0)

    print("[fp16, causal mask]")
    test_unified_attention_ck(dtype=torch.float16, mask_type=2)

    print("[bf16, no mask]")
    test_unified_attention_ck(dtype=torch.bfloat16, mask_type=0)

    print("[bf16, causal mask]")
    test_unified_attention_ck(dtype=torch.bfloat16, mask_type=2)

    print("[fp16, single seq]")
    test_unified_attention_ck(num_seqs=1, max_seq_len=512, dtype=torch.float16, mask_type=2)

    print("[fp16, many short seqs]")
    test_unified_attention_ck(num_seqs=32, max_seq_len=64, dtype=torch.float16, mask_type=0)

    print("\n=== All tests passed ===")
