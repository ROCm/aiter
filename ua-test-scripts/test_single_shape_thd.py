#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
Single-shape test and benchmark for CK unified-attention (contiguous /
THD K/V) vs Triton's fav3-sage attention.

The CK kernel runs in `kv_contiguous=True` mode — `block_tables` is
ignored and K/V are read directly from the contiguous tensor as
`offset = token * row_stride`. The Triton reference is
`fav3_sage_wrapper_func` (non-causal SageAttention), which internally
quantises BF16 inputs to int8/bf16 for fp8-class throughput. Both
backends therefore operate in an fp8-class regime, and we compare
their outputs with an fp8 tolerance (atol=3e-1, rtol=2.5e-1 with a
small mismatch percentage allowance — same recipe as
`test_fav3_sage.py`).

Defaults match the user's workload:
  - dtype     = fp8       (CK fp8 path, sage internal quantisation)
  - causal    = False     (sage is intended for non-causal workloads)
  - batch     = 1
  - layout    = THD       (contiguous K/V via kv_contiguous=True)

Examples:
  # Correctness check at sq=sk=2048, GQA-8, d=128:
  python test_single_shape_thd.py -b 1 -sq 2048 -sk 2048 -hq 64 -hk 8 \
      -d 128 --test

  # Bench at sq=sk=75600, hq=hk=5:
  python test_single_shape_thd.py -b 1 -sq 75600 -sk 75600 -hq 5 -hk 5 \
      -d 128 --warmup 5 --iters 30
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time

# The Triton fav3-sage kernel autotuner currently picks configs that
# crash the upstream `tritonamdgpu-block-pingpong` pass on a number of
# (seqlen, num_heads) combinations (e.g. sq=sk=1024, hq=hk=16 or
# sq=sk=2048, hq=hk=8). Disable the pingpong pass globally so sage
# compiles deterministically across shapes — this only affects the
# scheduling of Triton's pipeline (a small perf hit on some configs)
# and not numerical correctness. Set TRITON_HIP_USE_BLOCK_PINGPONG=1
# in the environment if you want to force-enable it.
os.environ.setdefault("TRITON_HIP_USE_BLOCK_PINGPONG", "0")

import torch


def parse_args():
    p = argparse.ArgumentParser(
        description="CK (contiguous THD) vs Triton fav3-sage single-shape test/bench",
    )

    p.add_argument("-b", "--batch",         type=int, default=1,
                   help="Number of sequences (default: 1)")
    p.add_argument("-sq", "--seqlen-q",     type=int, required=True)
    p.add_argument("-sk", "--seqlen-k",     type=int, required=True)
    p.add_argument("-hq", "--num-q-heads",  type=int, required=True)
    p.add_argument("-hk", "--num-kv-heads", type=int, required=True)
    p.add_argument("-d",  "--head-size",    type=int, default=128)
    p.add_argument("--dtype", choices=["bf16", "fp16", "fp8"], default="fp8",
                   help="Data type (default: fp8 — sage's native regime)")
    p.add_argument("--causal", action="store_true",
                   help="Use causal mask (default: off; sage is non-causal)")
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--test", action="store_true",
                   help="Run correctness check between CK contig and sage")
    p.add_argument("--show-output", action="store_true")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters",  type=int, default=50)
    p.add_argument("--use-graph", action="store_true",
                   help="Time with CUDA graph (eager mode by default)")

    p.add_argument("--only-ck",     action="store_true")
    p.add_argument("--only-triton", action="store_true")

    # fp8 tolerance — same defaults as op_tests/triton_tests/attention/test_fav3_sage.py
    p.add_argument("--atol", type=float, default=3.0e-1)
    p.add_argument("--rtol", type=float, default=2.5e-1)
    p.add_argument("--max-diff-pct", type=float, default=0.5,
                   help="Max %% of elements allowed to exceed atol/rtol "
                        "(fp8 paths have a small tail of high-magnitude "
                        "deviations from compounded quantisation noise)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# FP8 helpers (same recipe as test_single_shape.py).
# ---------------------------------------------------------------------------
def _pick_fp8_dtype():
    try:
        arch = torch.cuda.get_device_properties(0).gcnArchName
    except Exception:
        arch = ""
    return torch.float8_e4m3fn if "gfx950" in arch else torch.float8_e4m3fnuz


def _quantize_per_tensor_fp8(x: torch.Tensor, fp8_dtype):
    finfo = torch.finfo(fp8_dtype)
    fp8_max = float(finfo.max)
    amax = x.detach().abs().amax().clamp(min=1e-12)
    descale = (amax / fp8_max).item()
    scale = 1.0 / descale
    q = (x * scale).clamp_(-fp8_max, fp8_max).to(fp8_dtype)
    return q, descale


def make_tensors(args, device="cuda"):
    """Create BF16 reference inputs and (for fp8) their quantised CK images."""
    b   = args.batch
    sq  = args.seqlen_q
    sk  = args.seqlen_k
    hq  = args.num_q_heads
    hk  = args.num_kv_heads
    d   = args.head_size
    assert b == 1, "test_single_shape_thd currently only supports batch=1 (single sequence)"

    is_fp8 = args.dtype == "fp8"
    src_dtype = (torch.float16 if args.dtype == "fp16" else torch.bfloat16)

    # BF16/FP16 reference inputs. THD layout for CK: [total_q, hq, d].
    q_ref = torch.randn(sq, hq, d, dtype=src_dtype, device=device)
    k_ref = torch.randn(sk, hk, d, dtype=src_dtype, device=device)
    v_ref = torch.randn(sk, hk, d, dtype=src_dtype, device=device)

    # CK inputs (FP8 if requested, else passthrough). Per-tensor descales.
    q_descale = k_descale = v_descale = 1.0
    if is_fp8:
        fp8_dtype = _pick_fp8_dtype()
        q_ck, q_descale = _quantize_per_tensor_fp8(q_ref, fp8_dtype)
        k_ck, k_descale = _quantize_per_tensor_fp8(k_ref, fp8_dtype)
        v_ck, v_descale = _quantize_per_tensor_fp8(v_ref, fp8_dtype)
    else:
        q_ck, k_ck, v_ck = q_ref, k_ref, v_ref

    # CK signature reshape: K/V want a 4-D "paged" view. With kv_contiguous=True
    # the paged-axis dimensions are ignored except for the row-stride read in
    # `stride_k_cache_1`. We use [1, sk, hk, d] (num_blks=1, page_size=sk) so
    # the row stride is hk*d which is what the contiguous code path expects.
    k_ck_4d = k_ck.unsqueeze(0).contiguous()    # [1, sk, hk, d]
    v_ck_4d = v_ck.unsqueeze(0).contiguous()

    # Triton fav3-sage wants BSHD with batch dim: [1, sq/sk, h, d].
    # It always takes high-precision (bf16/fp16) inputs and quantises
    # internally — so we feed the *unquantised* reference tensors.
    q_tri = q_ref.unsqueeze(0).contiguous()
    k_tri = k_ref.unsqueeze(0).contiguous()
    v_tri = v_ref.unsqueeze(0).contiguous()

    # Block tables: ignored by the CK contig path, but the pybind reads
    # `block_tables.data_ptr<int32_t>()`, so it must still be a valid
    # int32 tensor with at least one element.
    block_tables = torch.zeros((1, 1), dtype=torch.int32, device=device)
    seq_lens     = torch.tensor([sk], dtype=torch.int32, device=device)
    cu_seqlens_q = torch.tensor([0, sq], dtype=torch.int32, device=device)
    scale        = 1.0 / math.sqrt(d)

    return {
        "q_ck": q_ck, "k_ck_4d": k_ck_4d, "v_ck_4d": v_ck_4d,
        "q_tri": q_tri, "k_tri": k_tri, "v_tri": v_tri,
        "block_tables": block_tables,
        "seq_lens": seq_lens,
        "cu_seqlens_q": cu_seqlens_q,
        "scale": scale,
        "q_descale": q_descale, "k_descale": k_descale, "v_descale": v_descale,
        "is_fp8": is_fp8,
    }


# ---------------------------------------------------------------------------
# CK contig
# ---------------------------------------------------------------------------
def run_ck(out, tensors, args):
    from aiter.ops.unified_attention import unified_attention_fwd
    unified_attention_fwd(
        out, tensors["q_ck"], tensors["k_ck_4d"], tensors["v_ck_4d"],
        tensors["block_tables"], tensors["seq_lens"], tensors["cu_seqlens_q"],
        mask_type=2 if args.causal else 0,
        scale_s=tensors["scale"],
        scale=1.0, scale_k=1.0, scale_v=1.0, scale_out=1.0,
        allow_splitkv=False,
        q_descale=tensors["q_descale"],
        k_descale=tensors["k_descale"],
        v_descale=tensors["v_descale"],
        kv_contiguous=True,
    )


# ---------------------------------------------------------------------------
# Triton fav3-sage
# ---------------------------------------------------------------------------
def run_triton(out, tensors, args):
    from aiter.ops.triton.attention.fav3_sage import fav3_sage_wrapper_func
    triton_out = fav3_sage_wrapper_func(
        tensors["q_tri"], tensors["k_tri"], tensors["v_tri"],
        softmax_scale=tensors["scale"],
        causal=args.causal,
        inference_mode=True,
        layout="bshd",
    )
    # fav3-sage returns the output as FP32 BSHD; squeeze the batch dim
    # and cast to the CK output dtype before copying.
    out.copy_(triton_out.squeeze(0).to(out.dtype))


# ---------------------------------------------------------------------------
# FLOPs / mem accounting
# ---------------------------------------------------------------------------
def compute_flops_and_bandwidth(args):
    total_q = args.batch * args.seqlen_q
    flops_full = 4 * total_q * args.seqlen_k * args.head_size * args.num_q_heads
    flops = flops_full * (0.5 if args.causal else 1.0)

    bytes_per_elem = 1 if args.dtype == "fp8" else 2
    bytes_per_out  = 2
    mem_q = total_q * args.num_q_heads * args.head_size * bytes_per_elem
    mem_k = args.seqlen_k * args.num_kv_heads * args.head_size * bytes_per_elem
    mem_v = mem_k
    mem_o = total_q * args.num_q_heads * args.head_size * bytes_per_out
    return {
        "flops":     flops,
        "mem_bytes": mem_q + mem_k + mem_v + mem_o,
        "mem_gb":   (mem_q + mem_k + mem_v + mem_o) / 1e9,
    }


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------
def time_kernel(fn, warmup, iters, use_graph=False):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    if use_graph:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            fn()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            graph.replay()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) * 1e3 / iters

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends   = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        starts[i].record(); fn(); ends[i].record()
    torch.cuda.synchronize()
    times = sorted(s.elapsed_time(e) for s, e in zip(starts, ends))
    return times[iters // 2]  # median


# ---------------------------------------------------------------------------
# FP8 correctness check (same recipe as test_fav3_sage.fp8_assert_close).
# ---------------------------------------------------------------------------
def fp8_check(out_ck, out_tri, atol, rtol, max_diff_pct):
    abs_diff = (out_ck.float() - out_tri.float()).abs()
    rel_diff = abs_diff / out_tri.float().abs().clamp(min=1e-6)
    failed = (abs_diff > atol) & (rel_diff > rtol)
    pct = failed.sum().item() / failed.numel() * 100
    return pct, abs_diff, rel_diff


def print_separator(ch="=", w=80):
    print(ch * w)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    print_separator()
    print("CK (contiguous THD) vs Triton fav3-sage — single-shape test/bench")
    print_separator()
    print(f"Shape Configuration:")
    print(f"  Batch size:        {args.batch}")
    print(f"  Query seqlen:      {args.seqlen_q}")
    print(f"  KV seqlen:         {args.seqlen_k}")
    print(f"  Q heads:           {args.num_q_heads}")
    print(f"  KV heads:          {args.num_kv_heads}")
    print(f"  Head size:         {args.head_size}")
    print(f"  Data type:         {args.dtype}")
    print(f"  Causal:            {args.causal}")
    print(f"  Phase:             {'decode' if args.seqlen_q == 1 else 'prefill'}")
    try:
        gpu = torch.cuda.get_device_properties(0)
        print(f"  GPU:               {gpu.gcnArchName}  CUs={gpu.multi_processor_count}")
    except Exception:
        pass
    print_separator()
    print()

    print("Creating tensors...")
    tensors = make_tensors(args)
    out_dtype = torch.bfloat16 if tensors["is_fp8"] else (
        torch.float16 if args.dtype == "fp16" else torch.bfloat16
    )
    out_ck     = torch.zeros(args.seqlen_q, args.num_q_heads,
                             args.head_size, dtype=out_dtype, device="cuda")
    out_triton = torch.zeros_like(out_ck)

    run_ck_kernel     = not args.only_triton
    run_triton_kernel = not args.only_ck

    ck_ok = False
    if run_ck_kernel:
        print("Running CK contiguous-THD attention...")
        try:
            run_ck(out_ck, tensors, args)
            torch.cuda.synchronize()
            ck_ok = (not torch.isnan(out_ck).any().item()
                     and not (out_ck == 0).all().item())
            if not ck_ok:
                print("  WARNING: CK output is NaN or all-zero")
        except Exception as e:
            print(f"  ERROR: {type(e).__name__}: {e}")
            ck_ok = False

    if run_triton_kernel:
        print("Running Triton fav3-sage attention...")
        try:
            run_triton(out_triton, tensors, args)
            torch.cuda.synchronize()
        except Exception as e:
            print(f"  ERROR: {type(e).__name__}: {e}")
            sys.exit(1)
    print()

    # ---- Correctness ----
    if args.test or args.show_output:
        print_separator("-")
        print("CORRECTNESS CHECK"
              if (run_ck_kernel and run_triton_kernel) else "OUTPUT CHECK")
        print_separator("-")

        if run_ck_kernel and run_triton_kernel and ck_ok:
            ck_f = out_ck.float(); tr_f = out_triton.float()
            diff = (ck_f - tr_f).abs()
            n = diff.numel()
            print(f"  CK out shape:     {tuple(out_ck.shape)}  dtype={out_ck.dtype}")
            print(f"  Triton out shape: {tuple(out_triton.shape)}  dtype={out_triton.dtype}")
            print(f"  CK     stats: min={ck_f.min().item():+.4f} max={ck_f.max().item():+.4f} "
                  f"mean={ck_f.mean().item():+.4f} std={ck_f.std().item():.4f}")
            print(f"  Triton stats: min={tr_f.min().item():+.4f} max={tr_f.max().item():+.4f} "
                  f"mean={tr_f.mean().item():+.4f} std={tr_f.std().item():.4f}")

            max_diff  = diff.max().item()
            mean_diff = diff.mean().item()
            cos = torch.nn.functional.cosine_similarity(
                ck_f.reshape(1, -1), tr_f.reshape(1, -1)).item()
            print(f"  max  abs diff : {max_diff:.4e}")
            print(f"  mean abs diff : {mean_diff:.4e}")
            print(f"  cos similarity: {cos:.6f}")

            for thr in (0.01, 0.05, 0.1, 0.2, 0.5):
                bad = (diff > thr).sum().item()
                if bad > 0:
                    print(f"  Elements > {thr:>4.2f}: {bad}/{n} ({100*bad/n:.4f}%)")

            # Top mismatches
            topk = min(8, n)
            tv, ti = diff.flatten().topk(topk)
            T, H, D = out_ck.shape
            print(f"  Top-{topk} mismatches (linear idx, diff, CK, Triton):")
            for i in range(topk):
                idx = int(ti[i].item())
                t, h, dd = idx // (H * D), (idx // D) % H, idx % D
                print(f"    (t={t},h={h},d={dd})  diff={tv[i].item():+.4e}  "
                      f"ck={ck_f.flatten()[idx].item():+.4e}  "
                      f"tr={tr_f.flatten()[idx].item():+.4e}")

            if args.test:
                if tensors["is_fp8"]:
                    pct, _, _ = fp8_check(out_ck, out_triton,
                                          args.atol, args.rtol,
                                          args.max_diff_pct)
                    fail_desc = (f"|abs|>{args.atol} AND |rel|>{args.rtol}")
                else:
                    # Non-fp8: simpler abs-only threshold.
                    failed = (out_ck.float() - out_triton.float()).abs() > args.atol
                    pct = failed.sum().item() / failed.numel() * 100
                    fail_desc = f"|abs|>{args.atol}"
                print()
                print(f"  Elements failing ({fail_desc}): "
                      f"{pct:.4f}%   (allowed: {args.max_diff_pct}%)")
                if pct <= args.max_diff_pct:
                    print(f"  ✓ PASS")
                else:
                    print(f"  ✗ FAIL")
                    sys.exit(1)
        else:
            print("  Skipped (one or both kernels did not produce output)")
        print()

    # ---- Benchmark ----
    if args.warmup > 0 and args.iters > 0:
        print_separator("-")
        print("BENCHMARK")
        print_separator("-")
        ck_ms = float("nan"); tri_ms = float("nan")
        if run_ck_kernel and ck_ok:
            print("Benchmarking CK contig...")
            ck_ms = time_kernel(lambda: run_ck(out_ck, tensors, args),
                                args.warmup, args.iters, use_graph=args.use_graph)
            print(f"  CK time:     {ck_ms:.4f} ms")
        if run_triton_kernel:
            print("Benchmarking Triton fav3-sage...")
            tri_ms = time_kernel(lambda: run_triton(out_triton, tensors, args),
                                 args.warmup, args.iters, use_graph=args.use_graph)
            print(f"  Triton time: {tri_ms:.4f} ms")

        print()
        print_separator("=")
        print("SUMMARY")
        print_separator("=")

        info = compute_flops_and_bandwidth(args)
        if run_ck_kernel and ck_ok and not math.isnan(ck_ms):
            ck_tflops = (info["flops"] / 1e12) / (ck_ms / 1e3)
            ck_bw     = info["mem_gb"] / (ck_ms / 1e3)
            print(f"  CK:     {ck_ms:8.4f} ms   {ck_tflops:7.2f} TFLOPs/s   "
                  f"{ck_bw:7.2f} GB/s")
        if run_triton_kernel and not math.isnan(tri_ms):
            tr_tflops = (info["flops"] / 1e12) / (tri_ms / 1e3)
            tr_bw     = info["mem_gb"] / (tri_ms / 1e3)
            print(f"  Triton: {tri_ms:8.4f} ms   {tr_tflops:7.2f} TFLOPs/s   "
                  f"{tr_bw:7.2f} GB/s")

        if (run_ck_kernel and ck_ok and not math.isnan(ck_ms)
                and run_triton_kernel and not math.isnan(tri_ms)):
            speedup = tri_ms / ck_ms
            winner = "CK" if speedup >= 1.0 else "Triton"
            print(f"  Speedup: {speedup:.3f}x  ({winner} wins)")

        print_separator("=")


if __name__ == "__main__":
    main()
