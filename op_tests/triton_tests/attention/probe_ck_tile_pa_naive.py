#!/usr/bin/env python3
# SPDX-License-Identifier: MIT

import time
import torch
import aiter


def main() -> int:
    if not torch.cuda.is_available():
        print("CUDA/HIP device not available.")
        return 2

    device = "cuda"
    torch.manual_seed(0)

    # Small decode-style probe that should hit CK-backed pa_fwd_naive.
    num_seqs = 1
    num_q_heads = 64
    num_kv_heads = 8
    head_size = 64
    block_size = 32
    max_seq_len = 1001

    num_blocks = 4096
    max_num_blocks_per_seq = (max_seq_len + block_size - 1) // block_size

    # Q: [num_seqs, num_heads, head_size]
    q = torch.randn(num_seqs, num_q_heads, head_size, device=device, dtype=torch.bfloat16)

    # K cache: [num_blocks, num_kv_heads, head_size/x, block_size, x]
    # V cache: [num_blocks, num_kv_heads, head_size, block_size]
    x = 16 // torch.tensor([], dtype=torch.bfloat16, device=device).element_size()
    k_cache = torch.randn(
        num_blocks,
        num_kv_heads,
        head_size // x,
        block_size,
        x,
        device=device,
        dtype=torch.bfloat16,
    )
    v_cache = torch.randn(
        num_blocks,
        num_kv_heads,
        head_size,
        block_size,
        device=device,
        dtype=torch.bfloat16,
    )

    block_tables = torch.randint(
        0,
        num_blocks,
        (num_seqs, max_num_blocks_per_seq),
        device=device,
        dtype=torch.int32,
    )
    context_lens = torch.tensor([max_seq_len], device=device, dtype=torch.int32)

    # quant_algo=0 ("NO") => scales can be empty tensors.
    k_dequant_scales = torch.empty((0,), device=device, dtype=torch.float32)
    v_dequant_scales = torch.empty((0,), device=device, dtype=torch.float32)

    scale_s = 1.0 / (head_size**0.5)
    scale_k = 1.0
    scale_v = 1.0
    quant_algo = 0

    # Warmup
    for _ in range(3):
        out = aiter.pa_fwd_naive(
            q,
            k_cache,
            v_cache,
            block_tables,
            context_lens,
            k_dequant_scales,
            v_dequant_scales,
            max_seq_len,
            num_kv_heads,
            scale_s,
            scale_k,
            scale_v,
            block_size,
            quant_algo,
        )
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    iters = 20
    for _ in range(iters):
        out = aiter.pa_fwd_naive(
            q,
            k_cache,
            v_cache,
            block_tables,
            context_lens,
            k_dequant_scales,
            v_dequant_scales,
            max_seq_len,
            num_kv_heads,
            scale_s,
            scale_k,
            scale_v,
            block_size,
            quant_algo,
        )
    torch.cuda.synchronize()
    t1 = time.perf_counter()

    print("CK probe: SUCCESS")
    print(f"output.shape={tuple(out.shape)} dtype={out.dtype}")
    print(f"avg_ms={(t1 - t0) * 1e3 / iters:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
