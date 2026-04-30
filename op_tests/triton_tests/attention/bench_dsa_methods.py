"""
Benchmark and correctness test for all 3 backward methods in deepseek_sparse_attention.py.

Usage:
  python bench_dsa_methods.py                   # correctness + benchmark (all 3 methods)
  python bench_dsa_methods.py --bench-only      # benchmark only
  python bench_dsa_methods.py --test-only       # correctness only

Three backward strategies:
  1. "fused"              — single fused kernel (baseline)
  2. "recompute"          — split dQ+dKV, full S/P/dS recomputation (1.18x, 0 extra memory)
  3. "split_intermediate" — split dQ+dKV, stores dS/P intermediates (1.68x, 2 GiB extra)
"""
import sys
import os
import argparse
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

from aiter.ops.triton._triton_kernels.attention.deepseek_sparse_attention import (
    sparse_mla_fwd,
    sparse_mla_bwd,
    sparse_mla_train,
)

METHODS = ["fused", "recompute", "split_intermediate"]


# =====================================================================
# Correctness: compare all 3 methods against each other
# =====================================================================
def test_methods_agree(
    batch=1, seq_len=256, num_heads=32,
    kv_lora_rank=512, rope_rank=64, topk=128,
    device="cuda",
):
    """Verify all 3 backward methods produce the same dQ and dKV."""
    d_qk = kv_lora_rank + rope_rank
    total_tokens = batch * seq_len
    scale = 1.0 / (d_qk ** 0.5)

    torch.manual_seed(42)
    q = torch.randn(total_tokens, num_heads, d_qk, dtype=torch.bfloat16, device=device)
    kv = torch.randn(total_tokens, 1, d_qk, dtype=torch.bfloat16, device=device)
    topk_indices = torch.randint(0, total_tokens, (total_tokens, topk),
                                 dtype=torch.int32, device=device)

    o, lse = sparse_mla_fwd(q, kv, topk_indices, kv_lora_rank, scale)
    do = torch.randn_like(o)

    results = {}
    for method in METHODS:
        dq, dkv = sparse_mla_bwd(q, kv, o, do, topk_indices, lse,
                                 kv_lora_rank, scale, method=method)
        results[method] = (dq.float(), dkv.float())

    dq_ref, dkv_ref = results["fused"]
    dq_ref_norm = dq_ref.abs().max().item()
    dkv_ref_norm = dkv_ref.abs().max().item()

    all_ok = True
    print(f"\n  Config: B={batch} S={seq_len} H={num_heads} D={d_qk} TOPK={topk}")
    print(f"  {'Method':<22s} {'dQ max_rel':>12s} {'dKV max_rel':>12s}  {'Status':>6s}")
    print(f"  {'-'*58}")

    for method in METHODS:
        dq_m, dkv_m = results[method]
        dq_rel = (dq_m - dq_ref).abs().max().item() / (dq_ref_norm + 1e-8)
        dkv_rel = (dkv_m - dkv_ref).abs().max().item() / (dkv_ref_norm + 1e-8)

        ok = dq_rel < 0.01 and dkv_rel < 0.01
        if method == "fused":
            ok = True  # reference
        all_ok &= ok
        status = "PASS" if ok else "FAIL"
        if method == "fused":
            status = "REF"
        print(f"  {method:<22s} {dq_rel:12.2e} {dkv_rel:12.2e}  {status:>6s}")

    return all_ok


def test_autograd_methods(
    batch=1, seq_len=128, num_heads=16,
    kv_lora_rank=256, rope_rank=64, topk=64,
    device="cuda",
):
    """Verify autograd works with all 3 backward methods."""
    d_qk = kv_lora_rank + rope_rank
    total_tokens = batch * seq_len
    scale = 1.0 / (d_qk ** 0.5)

    topk_indices = torch.randint(0, total_tokens, (total_tokens, topk),
                                 dtype=torch.int32, device=device)

    all_ok = True
    print(f"\n  Autograd test: B={batch} S={seq_len} H={num_heads} D={d_qk} TOPK={topk}")

    for method in METHODS:
        torch.manual_seed(42)
        q = torch.randn(total_tokens, num_heads, d_qk, dtype=torch.bfloat16,
                        device=device, requires_grad=True)
        kv = torch.randn(total_tokens, 1, d_qk, dtype=torch.bfloat16,
                         device=device, requires_grad=True)

        o, lse = sparse_mla_train(q, kv, topk_indices, kv_lora_rank, scale,
                                  bwd_method=method)
        do = torch.randn_like(o)
        o.backward(do)

        has_dq = q.grad is not None and q.grad.abs().max().item() > 0
        has_dkv = kv.grad is not None and kv.grad.abs().max().item() > 0
        ok = has_dq and has_dkv
        all_ok &= ok
        status = "PASS" if ok else "FAIL"
        print(f"    {method:<22s} dQ={'OK' if has_dq else 'MISS'}  "
              f"dKV={'OK' if has_dkv else 'MISS'}  [{status}]")

    return all_ok


# =====================================================================
# Benchmark: all 3 methods
# =====================================================================
def benchmark_kernel(run_fn, reps=50):
    for _ in range(5):
        run_fn()
    torch.cuda.synchronize()

    ev0 = torch.cuda.Event(enable_timing=True)
    ev1 = torch.cuda.Event(enable_timing=True)
    ev0.record()
    for _ in range(reps):
        run_fn()
    ev1.record()
    torch.cuda.synchronize()
    return ev0.elapsed_time(ev1) / reps


def benchmark_methods(
    batch=1, seq_len=4096, num_heads=128,
    kv_lora_rank=512, rope_rank=64, topk=1024,
    device="cuda", reps=50,
):
    """Benchmark all 3 backward methods side-by-side."""
    d_qk = kv_lora_rank + rope_rank
    total_tokens = batch * seq_len
    scale = 1.0 / (d_qk ** 0.5)

    torch.manual_seed(42)
    q = torch.randn(total_tokens, num_heads, d_qk, dtype=torch.bfloat16, device=device)
    kv = torch.randn(total_tokens, 1, d_qk, dtype=torch.bfloat16, device=device)
    topk_indices = torch.randint(0, total_tokens, (total_tokens, topk),
                                 dtype=torch.int32, device=device)

    o, lse = sparse_mla_fwd(q, kv, topk_indices, kv_lora_rank, scale)
    do = torch.randn_like(o)

    # FLOPs for backward (approximate)
    flops_bwd = total_tokens * num_heads * topk * (
        2 * d_qk           # S recompute
        + 2 * kv_lora_rank  # dP + P@V for dKV
        + 2 * d_qk          # dQ
        + 2 * d_qk          # dKV
    )

    label = f"B{batch}_S{seq_len}_H{num_heads}_topk{topk}"
    print(f"\n  {label}")
    print(f"  {'Method':<22s} {'Time (ms)':>10s} {'TFLOPS':>8s} {'Speedup':>8s}  {'Extra mem':>10s}")
    print(f"  {'-'*64}")

    baseline_ms = None
    for method in METHODS:
        def run(m=method):
            sparse_mla_bwd(q, kv, o, do, topk_indices, lse,
                          kv_lora_rank, scale, method=m)

        try:
            ms = benchmark_kernel(run, reps=reps)
            tflops = flops_bwd / (ms * 1e-3) / 1e12

            if baseline_ms is None:
                baseline_ms = ms
            speedup = baseline_ms / ms

            if method == "split_intermediate":
                extra = f"{total_tokens * num_heads * topk * 2 * 2 / 1024**3:.1f} GiB"
            else:
                extra = "0"

            print(f"  {method:<22s} {ms:10.2f} {tflops:8.1f} {speedup:8.2f}x {extra:>10s}")
        except Exception as e:
            print(f"  {method:<22s}   FAILED: {e}")

    return baseline_ms


def benchmark_fwd(
    batch=1, seq_len=4096, num_heads=128,
    kv_lora_rank=512, rope_rank=64, topk=1024,
    device="cuda", reps=50,
):
    """Benchmark forward pass."""
    d_qk = kv_lora_rank + rope_rank
    total_tokens = batch * seq_len
    scale = 1.0 / (d_qk ** 0.5)

    torch.manual_seed(42)
    q = torch.randn(total_tokens, num_heads, d_qk, dtype=torch.bfloat16, device=device)
    kv = torch.randn(total_tokens, 1, d_qk, dtype=torch.bfloat16, device=device)
    topk_indices = torch.randint(0, total_tokens, (total_tokens, topk),
                                 dtype=torch.int32, device=device)

    def run():
        sparse_mla_fwd(q, kv, topk_indices, kv_lora_rank, scale)

    ms = benchmark_kernel(run, reps=reps)

    flops_fwd = total_tokens * num_heads * topk * (2 * d_qk + 2 * kv_lora_rank)
    tflops = flops_fwd / (ms * 1e-3) / 1e12

    label = f"B{batch}_S{seq_len}_H{num_heads}_topk{topk}"
    print(f"  {label:<30s} {ms:8.2f} ms  {tflops:8.1f} TFLOPS")
    return ms


# =====================================================================
# Main
# =====================================================================
def main():
    parser = argparse.ArgumentParser(description="Benchmark DSA backward methods")
    parser.add_argument("--test-only", action="store_true")
    parser.add_argument("--bench-only", action="store_true")
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    os.environ["HIP_VISIBLE_DEVICES"] = str(args.gpu)
    device = "cuda"

    print(f"GPU: {torch.cuda.get_device_name(0)}")

    if not args.bench_only:
        print("\n" + "=" * 64)
        print("  CORRECTNESS: All 3 backward methods agree")
        print("=" * 64)

        all_ok = True
        all_ok &= test_methods_agree(1, 128, 16, 256, 64, 64, device)
        all_ok &= test_methods_agree(1, 256, 32, 512, 64, 128, device)
        all_ok &= test_methods_agree(1, 256, 128, 512, 64, 128, device)

        print("\n" + "=" * 64)
        print("  CORRECTNESS: Autograd with all 3 methods")
        print("=" * 64)
        all_ok &= test_autograd_methods(device=device)

        print()
        if all_ok:
            print("All correctness tests PASSED!")
        else:
            print("Some tests FAILED!")
            if args.test_only:
                sys.exit(1)

    if not args.test_only:
        print("\n" + "=" * 64)
        print("  BENCHMARK: Forward")
        print("=" * 64)
        fwd_configs = [
            (1, 4096, 128, 512, 64, 1024),
            (1, 4096, 128, 512, 64, 2048),
            (1, 8192, 128, 512, 64, 1024),
        ]
        for cfg in fwd_configs:
            try:
                benchmark_fwd(*cfg, device=device)
            except Exception as e:
                print(f"  SKIPPED {cfg}: {e}")

        print("\n" + "=" * 64)
        print("  BENCHMARK: Backward (3 methods)")
        print("=" * 64)
        bwd_configs = [
            (1, 4096, 128, 512, 64, 1024),
            (1, 4096, 128, 512, 64, 2048),
            (1, 8192, 128, 512, 64, 1024),
        ]
        for cfg in bwd_configs:
            try:
                benchmark_methods(*cfg, device=device)
            except Exception as e:
                print(f"  SKIPPED {cfg}: {e}")


if __name__ == "__main__":
    main()
