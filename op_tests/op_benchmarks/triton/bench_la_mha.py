# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

import torch
import triton
from bisect import bisect_right

# Import both attention functions
from aiter.ops.triton.lean_atten import _persistent_lean_attention
from aiter.ops.triton.mha import flash_attn_func

# --- Helper functions copied from bench_la.py ---

def get_lean_attention_params(
    causal, batch_size, max_seqlen_q, max_seqlen_k, num_heads, BLOCK_M, BLOCK_N, num_SMs
):
    """
    Mirrors the get_num_splits_and_buffer_sizes logic from the host code.
    """
    num_m_blocks = (max_seqlen_q + BLOCK_M - 1) // BLOCK_M
    if max_seqlen_q == 1:
        causal = False
    tiles_per_head = 0
    if causal:
        for i in range(num_m_blocks):
            tiles_per_head += (i + 1) * (BLOCK_M // BLOCK_N)
        tiles_per_head *= batch_size
    else:
        num_n_blocks_total = (max_seqlen_k + BLOCK_N - 1) // BLOCK_N
        tiles_per_head = num_m_blocks * num_n_blocks_total
    total_tiles = tiles_per_head * num_heads
    lean_griddimz = num_SMs
    if lean_griddimz == 0:
        return 0, 0, 0, 0, 0
    max_tiles_per_wg = (total_tiles + lean_griddimz - 1) // lean_griddimz
    high_load_wgs = total_tiles % lean_griddimz
    if high_load_wgs == 0 and total_tiles > 0:
        high_load_wgs = lean_griddimz
    return (
        tiles_per_head,
        num_m_blocks,
        lean_griddimz,
        high_load_wgs,
        max_tiles_per_wg,
    )

def calculate_max_output_tiles_analytically(
    causal: bool,
    batch_size: int,
    max_seqlen_q: int,
    max_seqlen_k: int,
    num_heads: int,
    num_SMs: int,
    BLOCK_M: int,
    BLOCK_N: int,
):
    """
    Calculates the maximum number of output tiles any single workgroup will process.
    """
    MASKED_BLOCKS = BLOCK_M // BLOCK_N
    if causal and BLOCK_M % BLOCK_N != 0:
        raise ValueError("For causal, BLOCK_M must be a multiple of BLOCK_N")
    (
        tiles_per_head,
        num_m_blocks,
        num_wgs,
        high_load_wgs,
        max_tiles_per_wg,
    ) = get_lean_attention_params(
        causal, batch_size, max_seqlen_q, max_seqlen_k, num_heads, BLOCK_M, BLOCK_N, num_SMs
    )
    if num_wgs == 0:
        return 0
    m_block_boundaries = []
    if causal:
        total_blocks = 0
        for i in range(num_m_blocks):
            pair_idx = i // 2
            q_block_idx = pair_idx if (i % 2) == 0 else num_m_blocks - 1 - pair_idx
            task_size = (q_block_idx + 1) * MASKED_BLOCKS
            total_blocks += task_size
            m_block_boundaries.append(total_blocks)
    max_total_output_tiles = 0
    for wg_id in range(num_wgs):
        total_output_tiles_for_wg = 0
        if wg_id < high_load_wgs:
            start_iter = max_tiles_per_wg * wg_id
            end_iter = start_iter + max_tiles_per_wg
        else:
            start_iter = (max_tiles_per_wg - 1) * (wg_id - high_load_wgs) + high_load_wgs * max_tiles_per_wg
            end_iter = start_iter + (max_tiles_per_wg - 1)
        start_head = start_iter // tiles_per_head
        end_head = (end_iter - 1) // tiles_per_head
        for head_idx in range(start_head, end_head + 1):
            head_start_iter = head_idx * tiles_per_head
            wg_start_in_head = max(start_iter, head_start_iter)
            wg_end_in_head = min(end_iter, head_start_iter + tiles_per_head)
            if not causal:
                total_output_tiles_for_wg += 1
                continue
            relative_start = wg_start_in_head - head_start_iter
            relative_end = wg_end_in_head - head_start_iter
            start_m_idx = bisect_right(m_block_boundaries, relative_start)
            end_m_idx = bisect_right(m_block_boundaries, relative_end - 1)
            tiles_in_this_head = (end_m_idx - start_m_idx) + 1
            total_output_tiles_for_wg += tiles_in_this_head
        max_total_output_tiles = max(max_total_output_tiles, total_output_tiles_for_wg)
    return max_total_output_tiles

# --- Unified Benchmark Configuration ---

configs = []
# We define a common set of parameters for an apples-to-apples comparison.
# The `provider` will switch between 'lean' and 'mha'.
configs.append(
    triton.testing.Benchmark(
        x_names=["CAUSAL", "BATCH", "NUM_HEADS", "SEQLEN_Q", "SEQLEN_K", "HEAD_SZ"],
        x_vals=[
            (True, 1, 64, 8192, 8192, 128),
            (True, 2, 64, 8192, 8192, 128),
            (True, 1, 64, 16384, 16384, 128),
            (True, 1, 96, 4096, 4096, 128),
            (True, 1, 128, 4096, 4096, 128),
            # Add other configurations here for comparison
            # (True, 1, 64, 4096, 4096, 128),
            # (True, 4, 32, 2048, 2048, 64),
        ],
        line_arg="provider",
        line_vals=["lean", "mha"],
        line_names=["Lean Attn (ms)", "MHA (ms)"],
        ylabel="ms",
        plot_name="attention-comparison",
        args={},
    )
)

@triton.testing.perf_report(configs)
def bench_attention(
    CAUSAL,
    BATCH,
    NUM_HEADS,
    SEQLEN_Q,
    SEQLEN_K,
    HEAD_SZ,
    provider,
    device="cuda",
):
    warmup = 25
    rep = 100
    fn = None
    init_dtype = torch.float16
    sm_scale = 0.5

    if provider == 'lean':
        # --- LEAN ATTENTION SETUP ---
        # Use reasonable defaults for LA-specific tuning parameters
        total_programs = 912  # Typically num_SMs
        BLOCK_M = 128
        BLOCK_N = 64
        waves_per_eu = 2
        num_warps = 4

        # LA uses ragged tensors, so we create a list of sequence lengths
        n_ctx = [SEQLEN_K] * BATCH
        sum_n_ctx = sum(n_ctx)

        max_output_tile_cnt = calculate_max_output_tiles_analytically(
            causal=CAUSAL,
            batch_size=BATCH,
            max_seqlen_q=SEQLEN_Q,
            max_seqlen_k=sum_n_ctx,
            num_heads=NUM_HEADS,
            num_SMs=total_programs,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
        )
        max_output_tile_cnt += 4

        # Create the cumulative sequence length tensor for LA
        list_num_block_n = [(s + BLOCK_N - 1) // BLOCK_N for s in n_ctx]
        len_sum = 0
        list_sum_block_n = []
        for i in range(BATCH):
            len_sum += list_num_block_n[i]
            list_sum_block_n.append(len_sum)
        batch_num_block_n = torch.tensor(list_sum_block_n, device=device, dtype=torch.int32)

        # Allocate LA tensors (different shapes than MHA)
        q = torch.randn((SEQLEN_Q * BATCH, NUM_HEADS, HEAD_SZ), dtype=init_dtype, device=device)
        k = torch.randn((sum_n_ctx, NUM_HEADS, HEAD_SZ), dtype=init_dtype, device=device)
        v = torch.randn((sum_n_ctx, NUM_HEADS, HEAD_SZ), dtype=init_dtype, device=device)
        Mp = torch.empty((total_programs, SEQLEN_Q), device=device, dtype=torch.float32)
        Lp = torch.empty((total_programs, SEQLEN_Q), device=device, dtype=torch.float32)
        Op = torch.empty((total_programs, SEQLEN_Q, HEAD_SZ), device=device, dtype=torch.float32)
        locks = torch.zeros((total_programs,), device=device, dtype=torch.int32)
        
        XCD_REMAP =True 
        NUM_XCDS = 8
        fn = lambda: _persistent_lean_attention(
            q, k, v, Mp, Lp, Op, locks, batch_num_block_n,
            total_programs, BLOCK_M, BLOCK_N, XCD_REMAP, NUM_XCDS, CAUSAL, BATCH, sm_scale,
            num_warps, waves_per_eu
        )

    elif provider == 'mha':
        # --- MHA SETUP ---
        # MHA uses standard uniform tensors
        q = torch.randn((BATCH, SEQLEN_Q, NUM_HEADS, HEAD_SZ), dtype=init_dtype, device=device)
        k = torch.randn((BATCH, SEQLEN_K, NUM_HEADS, HEAD_SZ), dtype=init_dtype, device=device)
        v = torch.randn((BATCH, SEQLEN_K, NUM_HEADS, HEAD_SZ), dtype=init_dtype, device=device)

        # MHA function has a simpler interface
        fn = lambda: flash_attn_func(q, k, v, causal=CAUSAL)

    # Run the benchmark for the selected provider
    if fn is not None:
        # --- FIX: Assign the single float return value from do_bench to ms ---
        ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
        return ms
    else:
        return None


def main():
    bench_attention.run(save_path=".", print_data=True)

if __name__ == "__main__":
    sys.exit(main())
