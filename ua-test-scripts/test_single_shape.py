#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
Single-shape test and benchmark for CK vs Triton unified attention.

Usage:
  # Test mode - verify correctness
  python test_single_shape.py -b 128 -sq 1 -sk 4096 -hq 64 -hk 8 -d 64 --test

  # Bench mode - performance measurement
  python test_single_shape.py -b 128 -sq 1 -sk 4096 -hq 64 -hk 8 -d 64 --warmup 20 --iters 100

  # Both modes
  python test_single_shape.py -b 128 -sq 1 -sk 4096 -hq 64 -hk 8 -d 64 --test --warmup 20 --iters 100
"""

from __future__ import annotations

import argparse
import math
import sys
import time

import torch


def parse_args():
    p = argparse.ArgumentParser(description="Single-shape CK vs Triton UA test/bench")

    # Shape parameters
    p.add_argument("-b", "--batch", type=int, required=True,
                   help="Number of sequences (batch size)")
    p.add_argument("-sq", "--seqlen-q", type=int, required=True,
                   help="Query sequence length (use 1 for decode)")
    p.add_argument("-sk", "--seqlen-k", type=int, required=True,
                   help="KV sequence length (context length)")
    p.add_argument("-hq", "--num-q-heads", type=int, required=True,
                   help="Number of query heads")
    p.add_argument("-hk", "--num-kv-heads", type=int, required=True,
                   help="Number of KV heads")
    p.add_argument("-d", "--head-size", type=int, required=True,
                   help="Head dimension")

    # Optional parameters
    p.add_argument("--block-size", type=int, default=32,
                   help="KV cache block size (default: 32)")
    p.add_argument("--num-blocks", type=int, default=None,
                   help="Total number of KV cache blocks (default: auto)")
    p.add_argument("--dtype", choices=["bf16", "fp16", "fp8"], default="bf16",
                   help="Data type for Q/K/V (default: bf16). "
                        "`fp8` quantises Q/K/V to e4m3 (e4m3fn on gfx950, "
                        "e4m3fnuz on gfx942), passes per-tensor scalar "
                        "descales to both kernels, and forces the output "
                        "dtype to bf16 to match the CK fp8 problem traits "
                        "and the Triton FP8 reference.")
    p.add_argument("--causal", action="store_true", default=True,
                   help="Use causal masking (default: True)")
    p.add_argument("--no-causal", dest="causal", action="store_false",
                   help="Disable causal masking (override --causal default)")
    p.add_argument("--window-left", type=int, default=-1,
                   help="Sliding window left (default: -1 = no window)")
    p.add_argument("--window-right", type=int, default=-1,
                   help="Sliding window right (default: -1 = no window)")
    p.add_argument("--softcap", type=float, default=0.0,
                   help="Softcap value (default: 0.0 = disabled)")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed (default: 42)")

    # Test/bench modes
    p.add_argument("--test", action="store_true",
                   help="Run correctness test with torch.testing.assert_allclose")
    p.add_argument("--warmup", type=int, default=10,
                   help="Warmup iterations for benchmarking (default: 10)")
    p.add_argument("--iters", type=int, default=50,
                   help="Benchmark iterations (default: 50)")
    p.add_argument("--use-graph", action="store_true", default=True,
                   help="Use CUDA graph for benchmarking (default: True)")
    p.add_argument("--no-graph", dest="use_graph", action="store_false",
                   help="Disable CUDA graph (use eager mode instead)")
    p.add_argument("--show-output", action="store_true",
                   help="Always show first 20 elements (even without --test)")
    p.add_argument("--no-splitkv", action="store_true",
                   help="Force single-launch CK path (disable the transparent "
                        "split-KV wrapper). Useful for isolating whether large "
                        "FP8 mismatches come from the split-KV combine step.")
    p.add_argument("--force-overflow-path", action="store_true",
                   help="Force cache_ptr_int32_overflow_possible=True even when "
                        "the cache size doesn't require it. Useful for stress-"
                        "testing the 64-bit-base global_load_lds path.")

    # Backend selection
    p.add_argument("--only-ck", action="store_true",
                   help="Run only CK kernel (skip Triton)")
    p.add_argument("--only-triton", action="store_true",
                   help="Run only Triton kernel (skip CK)")

    # Tolerance for testing. The defaults are tight for bf16/fp16, where
    # the two backends do the same MFMA sequence in fp32 and the only
    # source of disagreement is reduction ordering. For FP8 we relax the
    # defaults — both backends share the *same* per-tensor descales and
    # the *same* quantised tensors, so the disagreement is still purely
    # MFMA ordering, but the post-quantisation signal-to-noise is lower
    # so atol/rtol need to be ~10× larger to avoid false positives on
    # very small output magnitudes.
    p.add_argument("--atol", type=float, default=None,
                   help="Absolute tolerance for allclose "
                        "(default: 1.5e-2 for bf16/fp16, 1.5e-1 for fp8)")
    p.add_argument("--rtol", type=float, default=None,
                   help="Relative tolerance for allclose "
                        "(default: 1e-2 for bf16/fp16, 5e-2 for fp8)")

    args = p.parse_args()
    if args.atol is None:
        args.atol = 1.5e-1 if args.dtype == "fp8" else 1.5e-2
    if args.rtol is None:
        args.rtol = 5e-2 if args.dtype == "fp8" else 1e-2
    return args


def _pick_fp8_dtype():
    """Return the e4m3 dtype matching the current GPU arch.

    gfx950 (MI355) uses OCP `e4m3fn` (s.4.3, bias 7, NaN-only specials).
    gfx942 (MI300/MI300X/MI325) uses `e4m3fnuz` (s.4.3, bias 8, no specials).
    The CK FP8 path resolves the same selector at compile time via
    `CK_TILE_USE_OCP_FP8`, so we mirror it here to keep host-side
    quantisation in lock-step with what the kernel consumes.
    """
    try:
        arch = torch.cuda.get_device_properties(0).gcnArchName
    except Exception:
        arch = ""
    if "gfx950" in arch:
        return torch.float8_e4m3fn
    return torch.float8_e4m3fnuz


def _quantize_tensor_per_tensor_fp8(x: torch.Tensor, fp8_dtype):
    """Quantise a tensor to FP8 with a single per-tensor scalar descale.

    Returns `(x_fp8, descale)` where `descale` is a Python float such
    that `descale * x_fp8.to(fp32) ≈ x`. The scale is picked so the
    largest absolute value fills the FP8 range:
        descale = amax(|x|) / FP8_MAX
        x_fp8 = (x / descale).clamp(-FP8_MAX, FP8_MAX).to(fp8_dtype)
    Matches the standard "per-tensor symmetric FP8" recipe used by
    vLLM, SGLang, and the Triton FP8 unified_attention test.

    Memory note: the naive `x.float() * scale` materialises a full
    FP32 working tensor (4 B/elem) alongside the source (2 B/elem) and
    the FP8 result (1 B/elem) -- for KV-cache-sized inputs that 4×
    intermediate blows past available VRAM (e.g. num_blocks=1.2M ×
    block_size=32 × hk=8 × d=128 is 73 GiB bf16 / 146 GiB fp32). We
    process in fixed-size chunks instead, so the working buffer is
    bounded by `chunk * sizeof(x.dtype)` regardless of input size.
    """
    finfo = torch.finfo(fp8_dtype)
    fp8_max = float(finfo.max)
    amax = x.detach().abs().amax().clamp(min=1e-12)
    descale = (amax / fp8_max).item()
    # 0-d scalar in the input dtype -- multiplying by this keeps the
    # working buffer at sizeof(x.dtype) per element instead of fp32.
    scale = (1.0 / descale)
    q = torch.empty_like(x, dtype=fp8_dtype)
    flat_x = x.reshape(-1)
    flat_q = q.reshape(-1)
    # 16 Mi elements per chunk ≈ 32 MiB working buffer in bf16. Tuned
    # to be small relative to L2 + leave headroom for other allocs but
    # large enough that the per-chunk launch overhead stays negligible.
    chunk = 1 << 24
    for off in range(0, flat_x.numel(), chunk):
        end = min(off + chunk, flat_x.numel())
        flat_q[off:end] = (
            (flat_x[off:end] * scale).clamp_(-fp8_max, fp8_max).to(fp8_dtype)
        )
    return q, descale


def make_tensors(args, device="cuda"):
    """Create test tensors matching the specified shape.

    For dtype=fp8 we first sample bf16 random tensors (so both backends
    see the same "real" values), per-tensor-quantise them to FP8 with
    scalar descales, and run the kernels on the FP8 tensors. The
    descales are returned alongside the tensors so `run_ck`/`run_triton`
    can plumb them into both backends in their native ABIs.
    """
    is_fp8 = args.dtype == "fp8"
    if args.dtype == "bf16":
        dtype = torch.bfloat16
    elif args.dtype == "fp16":
        dtype = torch.float16
    else:  # fp8 — source samples in bf16, then quantise
        dtype = torch.bfloat16

    b = args.batch
    sq = args.seqlen_q
    sk = args.seqlen_k
    hq = args.num_q_heads
    hk = args.num_kv_heads
    d = args.head_size
    blk = args.block_size

    total_q = b * sq
    max_blocks_per_seq = (sk + blk - 1) // blk
    if args.num_blocks is not None:
        num_blocks = args.num_blocks
    else:
        num_blocks = max(1024, 2 * max_blocks_per_seq)

    scale = 1.0 / math.sqrt(d)

    # Tensors. For FP8 we quantise Q/K/V one at a time and free each
    # BF16 source as soon as its FP8 image is built, so peak VRAM is
    # (bf16 source + fp8 result) for one tensor rather than (bf16 +
    # bf16 + bf16 + fp32 intermediate + fp8 + fp8 + fp8) -- the latter
    # OOMs on KV-cache-sized inputs (e.g. num_blocks ~1.2M).
    q = torch.randn(total_q, hq, d, dtype=dtype, device=device)

    q_descale = k_descale = v_descale = 1.0
    q_descale_t = k_descale_t = v_descale_t = None

    if is_fp8:
        fp8_dtype = _pick_fp8_dtype()

        q_fp8, q_descale = _quantize_tensor_per_tensor_fp8(q, fp8_dtype)
        del q
        q = q_fp8

        k_src = torch.randn(num_blocks, blk, hk, d, dtype=dtype, device=device)
        k, k_descale = _quantize_tensor_per_tensor_fp8(k_src, fp8_dtype)
        del k_src

        v_src = torch.randn(num_blocks, blk, hk, d, dtype=dtype, device=device)
        v, v_descale = _quantize_tensor_per_tensor_fp8(v_src, fp8_dtype)
        del v_src

        # Return BF16 blocks to the OS so subsequent allocations (output
        # buffers, descale tensors) don't trip the high-water mark.
        torch.cuda.empty_cache()

        # 1-element float32 tensors for the Triton kernel's `tl.load`.
        # Allocated once and cached so the benchmark loop is allocation-
        # free and the CUDA graph capture sees a stable address.
        q_descale_t = torch.tensor([q_descale], dtype=torch.float32, device=device)
        k_descale_t = torch.tensor([k_descale], dtype=torch.float32, device=device)
        v_descale_t = torch.tensor([v_descale], dtype=torch.float32, device=device)
    else:
        k = torch.randn(num_blocks, blk, hk, d, dtype=dtype, device=device)
        v = torch.randn_like(k)

    # Metadata
    cu_seqlens_q = torch.arange(0, b + 1, dtype=torch.int32, device=device) * sq
    seq_lens_k = torch.full((b,), sk, dtype=torch.int32, device=device)
    block_table = torch.randint(0, num_blocks, (b, max_blocks_per_seq),
                                dtype=torch.int32, device=device)

    # for 10 first samples add the num blocks as the first value in the row, in order to test possible overflow if the
    # kv cache size is larger than the int32 max.
    block_table[:10, 0] = num_blocks - 1

    return {
        'q': q, 'k': k, 'v': v,
        'cu_seqlens_q': cu_seqlens_q,
        'seq_lens_k': seq_lens_k,
        'block_table': block_table,
        'scale': scale,
        'total_q': total_q,
        'num_blocks': num_blocks,
        'q_descale': q_descale,
        'k_descale': k_descale,
        'v_descale': v_descale,
        'q_descale_t': q_descale_t,
        'k_descale_t': k_descale_t,
        'v_descale_t': v_descale_t,
        'is_fp8': is_fp8,
    }


def run_ck(out, tensors, args):
    """Run CK unified attention."""
    from aiter.ops.unified_attention import unified_attention_fwd

    # Calculate if int32 overflow is possible
    # Overflow occurs when: block_idx * stride_k_cache_0 > INT32_MAX
    # where stride_k_cache_0 = block_size * num_kv_heads * head_size
    num_blocks = tensors['k'].shape[0]
    block_size = tensors['k'].shape[1]
    num_kv_heads = tensors['k'].shape[2]
    head_size = tensors['k'].shape[3]

    stride_k_cache_0 = block_size * num_kv_heads * head_size
    INT32_MAX = 2**31 - 1

    # Maximum addressable cache size with int32 indexing
    max_cache_size = INT32_MAX
    # Actual cache size required for largest block index
    cache_size = num_blocks * stride_k_cache_0

    cache_ptr_int32_overflow_possible = cache_size > max_cache_size
    if getattr(args, 'force_overflow_path', False):
        cache_ptr_int32_overflow_possible = True

    unified_attention_fwd(
        out,
        tensors['q'],
        tensors['k'],
        tensors['v'],
        tensors['block_table'],
        tensors['seq_lens_k'],
        tensors['cu_seqlens_q'],
        mask_type=2 if args.causal else 0,
        scale_s=tensors['scale'],
        scale=1.0,
        scale_k=1.0,
        scale_v=1.0,
        scale_out=1.0,
        cache_ptr_int32_overflow_possible=cache_ptr_int32_overflow_possible,
        # Skip the transparent split-KV wrapper when the user passes
        # --no-splitkv. The flag is purely diagnostic; the default behaviour
        # is unchanged.
        allow_splitkv=not getattr(args, 'no_splitkv', False),
        # CK kernel folds q_descale*k_descale into the softmax scale and
        # applies v_descale outside the K/V loop. Pass scalars; non-FP8
        # paths use 1.0f which the kernel optimises away.
        q_descale=float(tensors['q_descale']),
        k_descale=float(tensors['k_descale']),
        v_descale=float(tensors['v_descale']),
    )


def run_triton(out, tensors, args):
    """Run Triton unified attention.

    Triton's `kernel_unified_attention_{2d,3d}` reads each FP8 descale via
    `tl.load(q_scale)` — a 1-element float32 device tensor — and folds
    `q_scale * k_scale` into the softmax `qk_scale`, then multiplies the
    accumulated o_acc by `v_scale` after the loop. We allocate the
    descale tensors once at tensor-creation time (see `make_tensors`)
    and cache them on the tensors dict so this hot path stays
    allocation-free.
    """
    from aiter.ops.triton.attention import unified_attention as ua_mod
    window = (args.window_left, args.window_right)

    q_descale_t = tensors.get('q_descale_t')
    k_descale_t = tensors.get('k_descale_t')
    v_descale_t = tensors.get('v_descale_t')

    ua_mod.unified_attention(
        q=tensors['q'],
        k=tensors['k'],
        v=tensors['v'],
        out=out,
        cu_seqlens_q=tensors['cu_seqlens_q'],
        max_seqlen_q=args.seqlen_q,
        seqused_k=tensors['seq_lens_k'],
        max_seqlen_k=args.seqlen_k,
        softmax_scale=tensors['scale'],
        causal=args.causal,
        window_size=window,
        block_table=tensors['block_table'],
        softcap=args.softcap,
        q_descale=q_descale_t,
        k_descale=k_descale_t,
        v_descale=v_descale_t,
        alibi_slopes=None,
        output_scale=None,
        qq_bias=None,
        sinks=None,
    )

    


def compute_flops_and_bandwidth(args):
    """Compute theoretical FLOPs and memory bandwidth for attention."""
    total_q = args.batch * args.seqlen_q

    # FLOPs calculation for attention: Q@K^T, softmax, attn@V
    # Q@K^T: [total_q, num_q_heads, head_dim] @ [seqlen_k, num_kv_heads, head_dim]^T
    # For GQA, each query head group shares KV heads
    # FLOPs ≈ 2 * total_q * seqlen_k * head_dim * num_q_heads (QK^T matmul)
    #       + 2 * total_q * seqlen_k * head_dim * num_q_heads (attn@V matmul)
    flops = 4 * total_q * args.seqlen_k * args.head_size * args.num_q_heads

    # Memory calculation (bytes transferred). FP8 halves the per-element
    # cost on the dominant K/V cache reads (Q is small in decode) so the
    # bandwidth number is meaningful — but the output is still bf16 in
    # both backends, so we keep it separate. Q/K/V are uniformly 1 B for
    # FP8 and 2 B for bf16/fp16.
    bytes_per_elem = 1 if args.dtype == 'fp8' else 2
    bytes_per_out = 2  # bf16 always

    # Read Q: total_q * num_q_heads * head_dim
    mem_q = total_q * args.num_q_heads * args.head_size * bytes_per_elem

    # Read K cache: batch * seqlen_k * num_kv_heads * head_dim
    mem_k = args.batch * args.seqlen_k * args.num_kv_heads * args.head_size * bytes_per_elem

    # Read V cache: batch * seqlen_k * num_kv_heads * head_dim
    mem_v = args.batch * args.seqlen_k * args.num_kv_heads * args.head_size * bytes_per_elem

    # Write O: total_q * num_q_heads * head_dim (always 2 B / bf16/fp16)
    mem_o = total_q * args.num_q_heads * args.head_size * bytes_per_out

    total_mem_bytes = mem_q + mem_k + mem_v + mem_o

    return {
        'flops': flops,
        'mem_bytes': total_mem_bytes,
        'mem_gb': total_mem_bytes / 1e9
    }


def time_kernel(fn, warmup, iters, use_graph=False):
    """Time a kernel function."""
    # Warmup
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    if use_graph:
        # CUDA graph mode
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            fn()
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(iters):
            graph.replay()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) * 1e3 / iters
    else:
        # Eager mode with CUDA events
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]

        for i in range(iters):
            starts[i].record()
            fn()
            ends[i].record()

        torch.cuda.synchronize()
        times = [s.elapsed_time(e) for s, e in zip(starts, ends)]
        times.sort()
        return times[iters // 2]  # median


def print_separator(char='=', width=80):
    print(char * width)


def main():
    args = parse_args()

    # Set seed
    torch.manual_seed(args.seed)

    # Print configuration
    print_separator()
    print("CK vs Triton Unified Attention - Single Shape Test")
    print_separator()
    print(f"Shape Configuration:")
    print(f"  Batch size:        {args.batch}")
    print(f"  Query seqlen:      {args.seqlen_q}")
    print(f"  KV seqlen:         {args.seqlen_k}")
    print(f"  Q heads:           {args.num_q_heads}")
    print(f"  KV heads:          {args.num_kv_heads}")
    print(f"  Head size:         {args.head_size}")
    print(f"  Block size:        {args.block_size}")
    print(f"  Data type:         {args.dtype}")
    print(f"  Phase:             {'decode' if args.seqlen_q == 1 else 'prefill'}")
    print(f"  Total Q tokens:    {args.batch * args.seqlen_q}")
    print(f"")
    print(f"Attention Parameters:")
    print(f"  Causal:            {args.causal}")
    print(f"  Window:            ({args.window_left}, {args.window_right})")
    print(f"  Softcap:           {args.softcap}")
    print(f"")

    # GPU info
    try:
        gpu = torch.cuda.get_device_properties(0)
        print(f"GPU: {gpu.gcnArchName}  CUs={gpu.multi_processor_count}  "
              f"HBM={gpu.total_memory>>20}MB")
    except Exception:
        pass
    print_separator()
    print()

    # Create tensors
    print("Creating tensors...")
    tensors = make_tensors(args)

    # Allocate output buffers. For FP8 inputs the kernels write a higher-
    # precision output (bf16 in both CK and Triton FP8 paths) so we can't
    # reuse the Q dtype here; `torch.zeros_like(q)` would give an FP8
    # buffer with no defined arithmetic.
    out_dtype = (
        torch.bfloat16 if tensors.get('is_fp8') else tensors['q'].dtype
    )
    out_ck = torch.zeros(
        tensors['q'].shape, dtype=out_dtype, device=tensors['q'].device,
    )
    out_triton = torch.zeros(
        tensors['q'].shape, dtype=out_dtype, device=tensors['q'].device,
    )

    # Determine which kernels to run
    run_ck_kernel = not args.only_triton
    run_triton_kernel = not args.only_ck

    # Run kernels
    ck_ok = False
    if run_ck_kernel:
        print("Running CK unified attention...")
        try:
            run_ck(out_ck, tensors, args)
            torch.cuda.synchronize()
            ck_ok = (not torch.isnan(out_ck).any().item()
                     and not (out_ck == 0).all().item())
            if not ck_ok:
                print("  WARNING: CK output contains all zeros or NaNs!")
        except Exception as e:
            print(f"  ERROR: {type(e).__name__}: {e}")
            ck_ok = False

    if run_triton_kernel:
        print("Running Triton unified attention...")
        try:
            run_triton(out_triton, tensors, args)
            torch.cuda.synchronize()
        except Exception as e:
            print(f"  ERROR: {type(e).__name__}: {e}")
            sys.exit(1)

    print()

    # Test mode - correctness check
    if args.test or args.show_output:
        print_separator('-')
        print("CORRECTNESS CHECK" if (run_ck_kernel and run_triton_kernel) else "OUTPUT CHECK")
        print_separator('-')

        if run_ck_kernel and run_triton_kernel and ck_ok:
            # Show first 20 elements
            print("\nFirst 20 elements comparison:")
            print(f"  CK:     {out_ck.flatten()[:20].tolist()}")
            print(f"  Triton: {out_triton.flatten()[:20].tolist()}")

            # Compute diff stats
            ck_f = out_ck.float()
            tr_f = out_triton.float()
            diff = (ck_f - tr_f).abs()
            max_diff = diff.max().item()
            mean_diff = diff.mean().item()

            print(f"\nDifference statistics:")
            print(f"  Max abs diff:  {max_diff:.6e}")
            print(f"  Mean abs diff: {mean_diff:.6e}")

            # For FP8 we expect a small population of "bad rows" where the
            # row-wide MFMA-ordering noise compounds. Report what fraction
            # of elements is above various thresholds so the test report
            # captures both the bulk distribution and the tail.
            num_el = diff.numel()
            for thr in (0.01, 0.05, 0.1, 0.2, 0.5):
                bad = (diff > thr).sum().item()
                if bad > 0:
                    print(f"  Mismatch > {thr:>4.2f}: {bad} / {num_el} "
                          f"({100.0 * bad / num_el:.4f}%)")
            # Per-(token, head) row analysis: count rows where the mean diff
            # across the head_dim exceeds a threshold. Catches "whole row
            # corrupted" FP8 failure modes that get masked by per-element
            # tail percentiles.
            row_mean = diff.mean(dim=-1)  # [T, H]
            n_bad_rows = (row_mean > 0.05).sum().item()
            n_rows = row_mean.numel()
            print(f"  Row-mean > 0.05: {n_bad_rows} / {n_rows} rows "
                  f"({100.0 * n_bad_rows / n_rows:.4f}%)")

            # For FP8 we routinely see a handful of pixels with large
            # absolute diff (compounded fp8 quantisation noise in the
            # online softmax). Show the worst N to confirm both backends
            # are in the same magnitude regime.
            topk = min(8, diff.numel())
            top_vals, top_idx = diff.flatten().topk(topk)
            ck_top = ck_f.flatten()[top_idx]
            tr_top = tr_f.flatten()[top_idx]
            T, H, D = out_ck.shape
            print(f"  Top-{topk} mismatches (linear idx, diff, CK, Triton):")
            for i in range(topk):
                idx = int(top_idx[i].item())
                t = idx // (H * D)
                h = (idx // D) % H
                d = idx % D
                print(f"    idx={idx:>10d} (t={t},h={h},d={d}) "
                      f"diff={top_vals[i].item():+.4e} "
                      f"ck={ck_top[i].item():+.4e} "
                      f"tr={tr_top[i].item():+.4e}")
        elif run_ck_kernel and ck_ok:
            print("\nFirst 20 elements (CK only):")
            print(f"  CK:     {out_ck.flatten()[:20].tolist()}")
        elif run_triton_kernel:
            print("\nFirst 20 elements (Triton only):")
            print(f"  Triton: {out_triton.flatten()[:20].tolist()}")

        # Allclose test (only if both kernels ran)
        if args.test and run_ck_kernel and run_triton_kernel:
            if ck_ok:
                try:
                    torch.testing.assert_close(
                        out_ck, out_triton,
                        atol=args.atol, rtol=args.rtol
                    )
                    print(f"\n✓ PASS: Outputs match within atol={args.atol}, rtol={args.rtol}")
                except AssertionError as e:
                    print(f"\n✗ FAIL: Outputs do NOT match!")
                    print(f"  {e}")
                    sys.exit(1)
            else:
                print("\n✗ SKIP: CK kernel failed, cannot compare outputs")
                sys.exit(1)

        print()

    # Benchmark mode
    if args.warmup > 0 and args.iters > 0:
        print_separator('-')
        print("BENCHMARK")
        print_separator('-')
        print(f"  Warmup iterations: {args.warmup}")
        print(f"  Bench iterations:  {args.iters}")
        print(f"  Timing mode:       {'CUDA graph' if args.use_graph else 'eager'}")
        print()

        # Benchmark CK
        ck_ms = float('nan')
        if run_ck_kernel:
            if ck_ok:
                print("Benchmarking CK...")
                # Add marker for rocprof filtering
                print("BENCH_START_CK")
                ck_ms = time_kernel(
                    lambda: run_ck(out_ck, tensors, args),
                    args.warmup, args.iters, use_graph=args.use_graph
                )
                print("BENCH_END_CK")
                print(f"  CK time: {ck_ms:.4f} ms")
            else:
                print("  CK: SKIPPED (kernel failed)")

        # Benchmark Triton
        triton_ms = float('nan')
        if run_triton_kernel:
            print("Benchmarking Triton...")
            print("BENCH_START_TRITON")
            triton_ms = time_kernel(
                lambda: run_triton(out_triton, tensors, args),
                args.warmup, args.iters, use_graph=args.use_graph
            )
            print("BENCH_END_TRITON")
            print(f"  Triton time: {triton_ms:.4f} ms")

        # Summary
        print()
        print_separator('=')
        print("SUMMARY")
        print_separator('=')

        if run_ck_kernel and run_triton_kernel:
            if ck_ok and not math.isnan(ck_ms):
                speedup = triton_ms / ck_ms
                winner = "CK" if speedup >= 1.0 else "Triton"
                print(f"  CK:      {ck_ms:8.4f} ms")
                print(f"  Triton:  {triton_ms:8.4f} ms")
                print(f"  Speedup: {speedup:8.4f}x  ({winner} wins)")
            else:
                print(f"  Triton:  {triton_ms:8.4f} ms")
                print(f"  CK:      FAILED")
        elif run_ck_kernel:
            if not math.isnan(ck_ms):
                print(f"  CK:      {ck_ms:8.4f} ms")
            else:
                print(f"  CK:      FAILED")
        elif run_triton_kernel:
            print(f"  Triton:  {triton_ms:8.4f} ms")

        # Compute TFLOPs and bandwidth
        compute_info = compute_flops_and_bandwidth(args)
        print()
        print("Performance Metrics:")

        if run_ck_kernel and ck_ok and not math.isnan(ck_ms):
            ck_tflops = (compute_info['flops'] / 1e12) / (ck_ms / 1e3)
            ck_bw_gbs = compute_info['mem_gb'] / (ck_ms / 1e3)
            print(f"  CK TFLOPs:       {ck_tflops:8.2f}")
            print(f"  CK Bandwidth:    {ck_bw_gbs:8.2f} GB/s")

        if run_triton_kernel and not math.isnan(triton_ms):
            triton_tflops = (compute_info['flops'] / 1e12) / (triton_ms / 1e3)
            triton_bw_gbs = compute_info['mem_gb'] / (triton_ms / 1e3)
            print(f"  Triton TFLOPs:   {triton_tflops:8.2f}")
            print(f"  Triton Bandwidth:{triton_bw_gbs:8.2f} GB/s")

        if args.test and ck_ok and run_ck_kernel and run_triton_kernel:
            print()
            print(f"  Correctness: ✓ PASS")

        print_separator('=')


if __name__ == "__main__":
    main()
