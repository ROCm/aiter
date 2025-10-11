# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import sys, os
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

import torch
import triton
from bisect import bisect_right

# Import both attention functions
from aiter.ops.triton.lean_atten import (
    persistent_lean_attention,
    _persistent_lean_attention,
    get_num_splits_and_buffer_sizes,
)
from aiter.ops.triton.mha import flash_attn_func
# from aiter.ops.triton.utils import arch_info

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
        causal,
        batch_size,
        max_seqlen_q,
        max_seqlen_k,
        num_heads,
        BLOCK_M,
        BLOCK_N,
        num_SMs,
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
            start_iter = (max_tiles_per_wg - 1) * (
                wg_id - high_load_wgs
            ) + high_load_wgs * max_tiles_per_wg
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


# --- Unified Benchmark Configuration with GQA (H_Q, H_K) ---

csv_configs = [
    # (True, 1, 16, 16, 1024, 1024, 128),
    # (True, 1, 16, 16, 2048, 2048, 128),
    # (True, 1, 16, 16, 4096, 4096, 128),
    # (True, 1, 16, 16, 8192, 8192, 128),
    # (True, 1, 16, 16, 16384, 16384, 128),
    # (True, 1, 48, 48, 1024, 1024, 128),
    # (True, 1, 48, 48, 2048, 2048, 128),
    # (True, 1, 48, 48, 4096, 4096, 128),
    # (True, 1, 48, 48, 8192, 8192, 128),
    # (True, 1, 48, 48, 16384, 16384, 128),
    # (True, 1, 64, 64, 1024, 1024, 128),
    # (True, 1, 64, 64, 2048, 2048, 128),
    # (True, 1, 64, 64, 4096, 4096, 128),
    # (True, 1, 64, 64, 8192, 8192, 128),
    # (True, 1, 64, 64, 16384, 16384, 128),
    # (True, 4, 16, 16, 1024, 1024, 128),
    # (True, 4, 16, 16, 2048, 2048, 128),
    # (True, 4, 16, 16, 4096, 4096, 128),
    # (True, 4, 16, 16, 8192, 8192, 128),
    # (True, 4, 16, 16, 16384, 16384, 128),
    # (True, 4, 48, 48, 1024, 1024, 128),
    # (True, 4, 48, 48, 2048, 2048, 128),
    # (True, 4, 48, 48, 4096, 4096, 128),
    # (True, 4, 48, 48, 8192, 8192, 128),
    # --- Decode (SEQLEN_Q=1, BLOCK_M=1) cases ---
    (True, 1, 64, 64, 1, 8192, 128),
    (True, 2, 64, 64, 1, 8192, 128),

    # (True, 4, 48, 48, 16384, 16384, 128),
#     (True, 4, 64, 64, 1024, 1024, 128),
#     (True, 4, 64, 64, 2048, 2048, 128),
#     (True, 4, 64, 64, 4096, 4096, 128),
#     (True, 4, 64, 64, 8192, 8192, 128),
#     (True, 4, 64, 64, 16384, 16384, 128),
#     (True, 8, 16, 16, 1024, 1024, 128),
#     (True, 8, 16, 16, 2048, 2048, 128),
#     (True, 8, 16, 16, 4096, 4096, 128),
#     (True, 8, 16, 16, 8192, 8192, 128),
#     (True, 8, 16, 16, 16384, 16384, 128),
#     (True, 8, 48, 48, 1024, 1024, 128),
#     (True, 8, 48, 48, 2048, 2048, 128),
#     (True, 8, 48, 48, 4096, 4096, 128),
#     # (True, 8, 48, 48, 8192, 8192, 128),
#     # (True, 8, 48, 48, 16384, 16384, 128),
#     (True, 8, 64, 64, 1024, 1024, 128),
#     (True, 8, 64, 64, 2048, 2048, 128),
#     (True, 8, 64, 64, 4096, 4096, 128),
#     (True, 8, 64, 64, 8192, 8192, 128),
]
configs = []
# Common comparison set. We include both MHA (only when H_Q == H_K) and Lean (supports GQA).
configs.append(
    triton.testing.Benchmark(
        x_names=["CAUSAL", "BATCH", "H_Q", "H_K", "SEQLEN_Q", "SEQLEN_K", "HEAD_SZ"],
        x_vals=csv_configs,
        # x_vals=[
        #     # (True, 1, 32, 8, 8192, 8192, 128),
        #     # (True, 1, 64, 8, 8192, 8192, 128),
        #     (True, 1, 16, 16, 1024, 1024, 128),
        #     (True, 1, 16, 16, 2048, 2048, 128),
        #     (True, 1, 48, 48, 8192, 8192, 128),
        #     (True, 1, 64, 64, 16384, 16384, 128),
        #     # (True, 1, 128, 8, 8192, 8192, 128),
        #     # (True, 1, 128, 128, 8192, 8192, 56),
        #     (False, 1, 64, 64, 128, 16384, 128),
        #     (False, 1, 96, 96, 128, 32768, 128),
        #     (True, 1, 64, 64, 8192, 8192, 128),
        #     (True, 2, 64, 64, 8192, 8192, 128),
        #     (True, 1, 64, 64, 16384, 16384, 128),
        #     # # GQA variants for Lean path
        #     # (False, 1, 64, 16, 8192, 8192, 128),  # group size 4
        #     # (False, 2, 64, 32, 4096, 4096, 128),  # group size 2
        #     # (False, 2, 64, 16, 4096, 2048, 128),  # ragged decode
        # ],
        line_arg="provider",
        line_vals=["lean", "mha"],
        line_names=["Lean Attn (ms)", "MHA (ms)"],
        ylabel="ms",
        plot_name="attention-comparison-gqa",
        args={},
    )
)


@triton.testing.perf_report(configs)
def bench_attention(
    CAUSAL,
    BATCH,
    H_Q,
    H_K,
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

    if provider == "lean":
        # --- LEAN ATTENTION SETUP (mirrors test_la.py) ---
        # Ragged K/V list
        n_ctx = [SEQLEN_K] * BATCH
        sum_n_ctx = sum(n_ctx)

        # Tile sizes (use lean_atten defaults: 128x64 for prefill; force BLOCK_M=1 for decode)
        XCD_REMAP = False
        BLOCK_M = 1 if SEQLEN_Q == 1 else 128
        BLOCK_N = 64

        # Build batch_num_block_n (cumulative BLOCK_N counts per request)
        list_num_block_n = [(s + BLOCK_N - 1) // BLOCK_N for s in n_ctx]
        len_sum = 0
        list_sum_block_n = []
        for i in range(BATCH):
            len_sum += list_num_block_n[i]
            list_sum_block_n.append(len_sum)
        batch_num_block_n = torch.tensor(
            list_sum_block_n, device=device, dtype=torch.int32
        )

        # Determine total_programs using scheduling helper
        num_SMs = 304
        (
            _num_m_blocks,
            _num_n_blocks,
            _high_load_tbs,
            _max_tiles_per_tb,
            _tiles_per_head,
            total_programs,
            _num_splits,
            _even_split,
        ) = get_num_splits_and_buffer_sizes(
            CAUSAL,
            BATCH,
            SEQLEN_Q,
            sum_n_ctx,
            H_Q,
            H_K,
            BLOCK_M,
            BLOCK_N,
            num_SMs,
            XCD_REMAP,
            8,
        )

        # Allocate tensors
        q = torch.randn(
            (SEQLEN_Q * BATCH, H_Q, HEAD_SZ), dtype=init_dtype, device=device
        )
        k = torch.randn((sum_n_ctx, H_K, HEAD_SZ), dtype=init_dtype, device=device)
        v = torch.randn((sum_n_ctx, H_K, HEAD_SZ), dtype=init_dtype, device=device)

        # Temp/result buffers sized like inner API
        head_dim_padded = max(16, 1 << (int(HEAD_SZ - 1).bit_length()))
        Mp = torch.empty((total_programs, BLOCK_M), device=device, dtype=torch.float32)
        Lp = torch.empty((total_programs, BLOCK_M), device=device, dtype=torch.float32)
        Op = torch.empty(
            (total_programs, SEQLEN_Q, head_dim_padded),
            device=device,
            dtype=torch.float32,
        )
        locks = torch.zeros((total_programs,), device=device, dtype=torch.int32)
        # Kernel launch parameters (use common defaults)
        num_warps = 8
        waves_per_eu = 2

        fn = lambda: _persistent_lean_attention(
            q,
            k,
            v,
            Mp,
            Lp,
            Op,
            locks,
            batch_num_block_n,
            total_programs,
            BLOCK_M,
            BLOCK_N,
            XCD_REMAP,
            CAUSAL,
            BATCH,
            sm_scale,
            num_warps,
            waves_per_eu,
            {},
        )
    elif provider == "mha":
        # --- MHA BASELINE (requires H_Q == H_K) ---
        if H_Q != H_K:
            return 0.0
        q = torch.randn(
            (BATCH, SEQLEN_Q, H_Q, HEAD_SZ), dtype=init_dtype, device=device
        )
        k = torch.randn(
            (BATCH, SEQLEN_K, H_Q, HEAD_SZ), dtype=init_dtype, device=device
        )
        v = torch.randn(
            (BATCH, SEQLEN_K, H_Q, HEAD_SZ), dtype=init_dtype, device=device
        )
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

    # --- Post-process and print extended metrics ---
    csv_path = "attention-comparison-gqa.csv"
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        rows = []
        for _, r in df.iterrows():
            CAUSAL = (
                bool(r["CAUSAL"])
                if not isinstance(r["CAUSAL"], str)
                else (r["CAUSAL"].lower() == "true")
            )
            BATCH = int(r["BATCH"]) if not pd.isna(r["BATCH"]) else 1
            H_Q = int(r["H_Q"]) if not pd.isna(r["H_Q"]) else 1
            H_K = int(r["H_K"]) if not pd.isna(r["H_K"]) else H_Q
            SEQLEN_Q = int(r["SEQLEN_Q"]) if not pd.isna(r["SEQLEN_Q"]) else 1
            SEQLEN_K = int(r["SEQLEN_K"]) if not pd.isna(r["SEQLEN_K"]) else 1
            HEAD_SZ = int(r["HEAD_SZ"]) if not pd.isna(r["HEAD_SZ"]) else 128

            la_ms = float(r.get("Lean Attn (ms)", float("nan")))
            mha_ms = float(r.get("MHA (ms)", float("nan")))

            XCD_REMAP = False
            BLOCK_M = 1 if SEQLEN_Q == 1 else 128
            BLOCK_N = 64

            sum_n_ctx = BATCH * SEQLEN_K
            (
                _num_m_blocks,
                _num_n_blocks,
                _high_load,
                max_tiles_per_wg,
                _tiles_per_head,
                total_programs,
                _num_splits,
                _even_split,
            ) = get_num_splits_and_buffer_sizes(
                CAUSAL,
                BATCH,
                SEQLEN_Q,
                sum_n_ctx,
                H_Q,
                H_K,
                BLOCK_M,
                BLOCK_N,
                304,
                XCD_REMAP,
                8,
            )

            flops = 4.0 * BATCH * H_Q * SEQLEN_Q * SEQLEN_K * HEAD_SZ
            la_tflops = (flops / 1e12) / (la_ms / 1000.0) if la_ms > 0 else float("nan")
            mha_tflops = (
                (flops / 1e12) / (mha_ms / 1000.0) if mha_ms > 0 else float("nan")
            )
            speedup = (
                ((mha_ms - la_ms) / (mha_ms + 1e-9)) * 100.0
                if mha_ms == mha_ms and la_ms == la_ms
                else float("nan")
            )

            rows.append(
                {
                    "causal": CAUSAL,
                    "B": BATCH,
                    "H_Q": H_Q,
                    "H_K": H_K,
                    "NK": SEQLEN_K,
                    "NQ": SEQLEN_Q,
                    "D": HEAD_SZ,
                    "total_programs": int(total_programs),
                    # "max_tiles_per_wg": int(max_tiles_per_wg),
                    "BLOCK_M": int(BLOCK_M),
                    "BLOCK_N": int(BLOCK_N),
                    "la ms": la_ms,
                    "mha ms": mha_ms,
                    "mha->la speedup (%)": speedup,
                    "la tflops": la_tflops,
                    "mha tflops": mha_tflops,
                }
            )

        if rows:
            df_ext = pd.DataFrame(rows)
            print("\nExtended results:")
            print(df_ext.to_string(index=False))
            df_ext.to_csv("attention-comparison-gqa-extended.csv", index=False)


if __name__ == "__main__":
    sys.exit(main())
