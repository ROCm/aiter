#!/usr/bin/env python3
"""
Benchmark CK vs Triton unified attention for multiple shapes from CSV.
Uses CUDA graph mode for production-accurate timings.
"""

import argparse
import csv
import sys
import time
import torch
import math


def parse_window(window_str):
    """Parse window string like '-1,-1' or '127,0'."""
    if not window_str or window_str == "-1,-1":
        return (-1, -1)
    left, right = window_str.split(',')
    return (int(left), int(right))


def make_tensors(batch, seqlen_q, seqlen_k, num_q_heads, num_kv_heads, head_size,
                 block_size=32, dtype=torch.bfloat16, device="cuda"):
    """Create test tensors for given shape."""
    total_q = batch * seqlen_q
    max_blocks_per_seq = (seqlen_k + block_size - 1) // block_size
    num_blocks = max(1024, 2 * max_blocks_per_seq)
    scale = 1.0 / math.sqrt(head_size)

    q = torch.randn(total_q, num_q_heads, head_size, dtype=dtype, device=device)
    k = torch.randn(num_blocks, block_size, num_kv_heads, head_size, dtype=dtype, device=device)
    v = torch.randn_like(k)

    cu_seqlens_q = torch.arange(0, batch + 1, dtype=torch.int32, device=device) * seqlen_q
    seq_lens_k = torch.full((batch,), seqlen_k, dtype=torch.int32, device=device)
    block_table = torch.randint(0, num_blocks, (batch, max_blocks_per_seq),
                                dtype=torch.int32, device=device)

    return {
        'q': q, 'k': k, 'v': v,
        'cu_seqlens_q': cu_seqlens_q,
        'seq_lens_k': seq_lens_k,
        'block_table': block_table,
        'scale': scale,
    }


def run_ck(out, tensors):
    """Run CK unified attention."""
    from aiter.ops.unified_attention import unified_attention_fwd

    unified_attention_fwd(
        out, tensors['q'], tensors['k'], tensors['v'],
        tensors['block_table'], tensors['seq_lens_k'], tensors['cu_seqlens_q'],
        mask_type=2, scale_s=tensors['scale'],
        scale=1.0, scale_k=1.0, scale_v=1.0, scale_out=1.0
    )


def run_triton(out, tensors, window, softcap):
    """Run Triton unified attention."""
    from aiter.ops.triton.attention import unified_attention as ua_mod

    ua_mod.unified_attention(
        out, tensors['q'], tensors['k'], tensors['v'],
        tensors['cu_seqlens_q'], None,
        tensors['seq_lens_k'], None,
        tensors['block_table'],
        causal=True,
        window_size=window,
        softcap=softcap,
        softmax_scale=tensors['scale']
    )


def time_kernel_graph(fn, warmup=10, iters=50):
    """Time a kernel function using CUDA graph mode."""
    # Warmup
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    # Capture graph
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    torch.cuda.synchronize()

    # Benchmark
    t0 = time.perf_counter()
    for _ in range(iters):
        graph.replay()
    torch.cuda.synchronize()

    return (time.perf_counter() - t0) * 1e3 / iters


def benchmark_shape(row, warmup=10, iters=50):
    """Benchmark a single shape from CSV row."""
    batch = int(row['num_seqs'])
    seqlen_q = int(row['max_seqlen_q'])
    seqlen_k = int(row['max_seqlen_k'])
    num_q_heads = int(row['num_q_heads'])
    num_kv_heads = int(row['num_kv_heads'])
    head_size = int(row['head_size'])
    block_size = int(row['block_size'])
    window = parse_window(row['window_size'])
    softcap = float(row['softcap'])

    dtype = torch.bfloat16 if row['q_dtype'] == 'torch.bfloat16' else torch.float16
    phase = row['phase']

    try:
        # Create tensors
        tensors = make_tensors(batch, seqlen_q, seqlen_k, num_q_heads, num_kv_heads,
                               head_size, block_size, dtype)

        out_ck = torch.zeros_like(tensors['q'])
        out_triton = torch.zeros_like(tensors['q'])

        # Test CK
        try:
            run_ck(out_ck, tensors)
            torch.cuda.synchronize()
            ck_ok = True
        except Exception as e:
            # CK may not support all shapes (e.g., head_dim=128, window, softcap)
            ck_ok = False
            ck_ms = float('nan')

        # Test Triton
        try:
            run_triton(out_triton, tensors, window, softcap)
            torch.cuda.synchronize()
        except Exception as e:
            return None  # Skip if Triton fails

        # Benchmark
        if ck_ok:
            ck_ms = time_kernel_graph(
                lambda: run_ck(out_ck, tensors),
                warmup, iters
            )

        triton_ms = time_kernel_graph(
            lambda: run_triton(out_triton, tensors, window, softcap),
            warmup, iters
        )

        if ck_ok:
            speedup = triton_ms / ck_ms
            winner = "CK" if speedup >= 1.0 else "Triton"
        else:
            speedup = float('nan')
            winner = "Triton"

        return {
            'phase': phase,
            'batch': batch,
            'seqlen_q': seqlen_q,
            'seqlen_k': seqlen_k,
            'ck_ms': ck_ms,
            'triton_ms': triton_ms,
            'speedup': speedup,
            'winner': winner,
            'ck_ok': ck_ok
        }

    except Exception as e:
        print(f"Error benchmarking shape: {e}", file=sys.stderr)
        return None


def main():
    parser = argparse.ArgumentParser(description="Benchmark shapes from CSV")
    parser.add_argument("csv_file", help="CSV file with test shapes")
    parser.add_argument("--warmup", type=int, default=10,
                       help="Warmup iterations (default: 10)")
    parser.add_argument("--iters", type=int, default=50,
                       help="Benchmark iterations (default: 50)")
    parser.add_argument("--limit", type=int, default=None,
                       help="Limit number of shapes to test")
    args = parser.parse_args()

    print("=" * 100)
    print("CK vs Triton Unified Attention - Batch Benchmark (CUDA Graph Mode)")
    print("=" * 100)
    print(f"CSV file: {args.csv_file}")
    print(f"Warmup: {args.warmup}, Iterations: {args.iters}")
    print()

    # Read CSV
    with open(args.csv_file) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if args.limit:
        rows = rows[:args.limit]

    print(f"Testing {len(rows)} shapes...")
    print()

    # Print header
    print(f"{'#':<4} {'Phase':<8} {'Batch':<6} {'SQ':<6} {'SK':<8} {'CK (ms)':<10} {'Triton (ms)':<12} {'Speedup':<10} {'Winner':<8}")
    print("-" * 100)

    results = []
    for i, row in enumerate(rows, 1):
        result = benchmark_shape(row, args.warmup, args.iters)

        if result is None:
            continue

        results.append(result)

        # Print result
        if result['ck_ok']:
            print(f"{i:<4} {result['phase']:<8} {result['batch']:<6} {result['seqlen_q']:<6} "
                  f"{result['seqlen_k']:<8} {result['ck_ms']:<10.4f} {result['triton_ms']:<12.4f} "
                  f"{result['speedup']:<10.2f}x {result['winner']:<8}")
        else:
            print(f"{i:<4} {result['phase']:<8} {result['batch']:<6} {result['seqlen_q']:<6} "
                  f"{result['seqlen_k']:<8} {'CK_FAIL':<10} {result['triton_ms']:<12.4f} "
                  f"{'-':<10} {result['winner']:<8}")

    print("-" * 100)

    # Summary statistics
    valid_results = [r for r in results if r['ck_ok']]

    if valid_results:
        decode_results = [r for r in valid_results if r['phase'] == 'decode']
        prefill_results = [r for r in valid_results if r['phase'] == 'prefill']

        print()
        print("=" * 100)
        print("SUMMARY")
        print("=" * 100)

        def print_stats(results, label):
            if not results:
                return
            speedups = [r['speedup'] for r in results]
            ck_wins = len([r for r in results if r['winner'] == 'CK'])

            print(f"\n{label}:")
            print(f"  Total shapes:    {len(results)}")
            print(f"  CK wins:         {ck_wins}/{len(results)} ({100*ck_wins/len(results):.1f}%)")
            print(f"  Triton wins:     {len(results)-ck_wins}/{len(results)} ({100*(len(results)-ck_wins)/len(results):.1f}%)")
            print(f"  Median speedup:  {sorted(speedups)[len(speedups)//2]:.2f}x")
            print(f"  Min speedup:     {min(speedups):.2f}x")
            print(f"  Max speedup:     {max(speedups):.2f}x")

        print_stats(valid_results, "Overall")
        print_stats(decode_results, "Decode")
        print_stats(prefill_results, "Prefill")

        print()
        print("=" * 100)


if __name__ == "__main__":
    main()
