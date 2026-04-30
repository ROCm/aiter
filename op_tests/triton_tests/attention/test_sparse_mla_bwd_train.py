"""
Test correctness of sparse MLA backward training kernels.

Compares dQ and dKV against a PyTorch autograd reference.
"""

import sys
import os
import argparse
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

from aiter.ops.triton._triton_kernels.attention.deepseek_sparse_attention import (
    sparse_mla_bwd,
    sparse_mla_fwd,
    sparse_mla_train,
)


# ============================================================
# Reference: differentiable forward in PyTorch
# ============================================================
def reference_sparse_mla_fwd_differentiable(q, kv, topk_indices, kv_lora_rank=512, scale=None):
    """
    Pure PyTorch differentiable sparse MLA forward.
    Supports autograd for computing reference dQ and dKV.

    Args:
        q:             [total_tokens, num_heads, d_qk] float32 (requires_grad)
        kv:            [total_tokens, 1, d_qk] float32 (requires_grad)
        topk_indices:  [total_tokens, topk] int32
        kv_lora_rank:  int
        scale:         float

    Returns:
        o: [total_tokens, num_heads, kv_lora_rank] float32
    """
    total_tokens, num_heads, d_qk = q.shape
    topk = topk_indices.shape[1]

    if scale is None:
        scale = 1.0 / (d_qk ** 0.5)

    kv_sq = kv.squeeze(1)  # [total_tokens, d_qk]

    outputs = []
    for i in range(total_tokens):
        qi = q[i]  # [num_heads, d_qk]
        idx = topk_indices[i]  # [topk]
        valid = idx != -1

        if not valid.any():
            outputs.append(torch.zeros(num_heads, kv_lora_rank, dtype=q.dtype, device=q.device))
            continue

        valid_idx = idx[valid].long()
        kv_sel = kv_sq[valid_idx]  # [n_valid, d_qk]

        # Score: q @ k^T
        scores = scale * (qi @ kv_sel.T)  # [num_heads, n_valid]

        # Softmax
        probs = torch.softmax(scores, dim=-1)

        # Output: P @ V (V = first kv_lora_rank dims)
        v_sel = kv_sel[:, :kv_lora_rank]  # [n_valid, kv_lora_rank]
        o_i = probs @ v_sel  # [num_heads, kv_lora_rank]
        outputs.append(o_i)

    return torch.stack(outputs, dim=0)


def compute_reference_grads(q_bf16, kv_bf16, topk_indices, do_bf16, kv_lora_rank=512, scale=None):
    """
    Compute reference dQ and dKV using PyTorch autograd.

    Returns:
        dq_ref:  [total_tokens, num_heads, d_qk] float32
        dkv_ref: [total_tokens, 1, d_qk] float32
    """
    q_f = q_bf16.float().detach().requires_grad_(True)
    kv_f = kv_bf16.float().detach().requires_grad_(True)
    do_f = do_bf16.float()

    o_ref = reference_sparse_mla_fwd_differentiable(q_f, kv_f, topk_indices, kv_lora_rank, scale)
    o_ref.backward(do_f)

    return q_f.grad, kv_f.grad


# ============================================================
# Test: preprocess kernel (Delta)
# ============================================================
def test_preprocess(total_tokens=256, num_heads=32, kv_lora_rank=512, device="cuda"):
    """Verify Delta = rowsum(O * dO)."""
    from aiter.ops.triton._triton_kernels.attention.deepseek_sparse_attention import (
        _sparse_mla_bwd_preprocess,
    )

    torch.manual_seed(42)
    o = torch.randn(total_tokens, num_heads, kv_lora_rank, dtype=torch.bfloat16, device=device)
    do = torch.randn(total_tokens, num_heads, kv_lora_rank, dtype=torch.bfloat16, device=device)

    # Reference
    delta_ref = (o.float() * do.float()).sum(dim=-1)  # [total_tokens, num_heads]

    # Kernel
    import triton
    delta = torch.empty(total_tokens, num_heads, dtype=torch.float32, device=device)
    BLOCK_H = min(64, triton.next_power_of_2(num_heads))
    grid = (total_tokens, triton.cdiv(num_heads, BLOCK_H))
    _sparse_mla_bwd_preprocess[grid](
        O_ptr=o, dO_ptr=do, Delta_ptr=delta,
        stride_o_t=o.stride(0), stride_o_h=o.stride(1),
        num_heads=num_heads, D_V=kv_lora_rank, BLOCK_H=BLOCK_H,
    )

    max_diff = (delta - delta_ref).abs().max().item()
    passed = max_diff < 1e-1
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] Preprocess: T={total_tokens}, H={num_heads}, D_V={kv_lora_rank}, max_diff={max_diff:.6f}")
    return passed


# ============================================================
# Test: full backward (dQ + dKV)
# ============================================================
def test_backward(
    batch=1,
    seq_len=256,
    num_heads=32,
    kv_lora_rank=512,
    rope_rank=64,
    topk=128,
    device="cuda",
):
    d_qk = kv_lora_rank + rope_rank
    total_tokens = batch * seq_len
    scale = 1.0 / (d_qk ** 0.5)

    torch.manual_seed(42)
    q = torch.randn(total_tokens, num_heads, d_qk, dtype=torch.bfloat16, device=device)
    kv = torch.randn(total_tokens, 1, d_qk, dtype=torch.bfloat16, device=device)
    do = torch.randn(total_tokens, num_heads, kv_lora_rank, dtype=torch.bfloat16, device=device)

    # Generate topk indices
    topk_indices = torch.full((total_tokens, topk), -1, dtype=torch.int32, device=device)
    for b in range(batch):
        for t in range(seq_len):
            global_t = b * seq_len + t
            n_valid = min(topk, seq_len)
            perm = torch.randperm(seq_len, device=device)[:n_valid]
            abs_pos = perm + b * seq_len
            topk_indices[global_t, :n_valid] = abs_pos.int()

    # Run forward to get O and LSE
    o, lse = sparse_mla_fwd(q, kv, topk_indices, kv_lora_rank, scale)

    # Run our backward
    dq, dkv = sparse_mla_bwd(q, kv, o, do, topk_indices, lse, kv_lora_rank, scale)

    # Reference backward
    dq_ref, dkv_ref = compute_reference_grads(q, kv, topk_indices, do, kv_lora_rank, scale)

    # Compare dQ
    dq_f = dq.float()
    max_diff_dq = (dq_f - dq_ref).abs().max().item()
    mean_diff_dq = (dq_f - dq_ref).abs().mean().item()
    # Relative error for dQ
    dq_ref_norm = dq_ref.abs().mean().item()
    rel_err_dq = mean_diff_dq / (dq_ref_norm + 1e-8)

    # Compare dKV
    dkv_f = dkv.float()
    max_diff_dkv = (dkv_f - dkv_ref).abs().max().item()
    mean_diff_dkv = (dkv_f - dkv_ref).abs().mean().item()
    dkv_ref_norm = dkv_ref.abs().mean().item()
    rel_err_dkv = mean_diff_dkv / (dkv_ref_norm + 1e-8)

    # Tolerances: bf16 accumulation + atomics add noise
    passed_dq = rel_err_dq < 0.1  # 10% relative error
    passed_dkv = rel_err_dkv < 0.15  # 15% relative error (atomics add noise)
    passed = passed_dq and passed_dkv
    status = "PASS" if passed else "FAIL"

    print(f"  [{status}] B={batch}, S={seq_len}, H={num_heads}, "
          f"d_v={kv_lora_rank}, d_rope={rope_rank}, topk={topk}")
    print(f"    dQ:  max_diff={max_diff_dq:.6f}, mean_diff={mean_diff_dq:.6f}, "
          f"rel_err={rel_err_dq:.4f}")
    print(f"    dKV: max_diff={max_diff_dkv:.6f}, mean_diff={mean_diff_dkv:.6f}, "
          f"rel_err={rel_err_dkv:.4f}")

    return passed


# ============================================================
# Test: end-to-end autograd (SparseMlaFunc)
# ============================================================
def test_autograd_e2e(
    batch=1,
    seq_len=128,
    num_heads=16,
    kv_lora_rank=256,
    rope_rank=64,
    topk=64,
    device="cuda",
):
    d_qk = kv_lora_rank + rope_rank
    total_tokens = batch * seq_len
    scale = 1.0 / (d_qk ** 0.5)

    torch.manual_seed(42)
    q = torch.randn(total_tokens, num_heads, d_qk, dtype=torch.bfloat16, device=device, requires_grad=True)
    kv = torch.randn(total_tokens, 1, d_qk, dtype=torch.bfloat16, device=device, requires_grad=True)

    topk_indices = torch.full((total_tokens, topk), -1, dtype=torch.int32, device=device)
    for b in range(batch):
        for t in range(seq_len):
            global_t = b * seq_len + t
            n_valid = min(topk, seq_len)
            perm = torch.randperm(seq_len, device=device)[:n_valid]
            abs_pos = perm + b * seq_len
            topk_indices[global_t, :n_valid] = abs_pos.int()

    # Forward + backward through autograd
    o, lse = sparse_mla_train(q, kv, topk_indices, kv_lora_rank, scale)

    # Create upstream gradient
    do = torch.randn_like(o)
    dlse = torch.zeros_like(lse)

    # Backward
    o.backward(do, retain_graph=True)
    lse.backward(dlse)

    # Check gradients exist
    has_dq = q.grad is not None and q.grad.abs().max().item() > 0
    has_dkv = kv.grad is not None and kv.grad.abs().max().item() > 0

    passed = has_dq and has_dkv
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] Autograd E2E: B={batch}, S={seq_len}, H={num_heads}, "
          f"d_v={kv_lora_rank}, topk={topk}")
    if has_dq:
        print(f"    dQ  max={q.grad.abs().max().item():.6f}")
    else:
        print(f"    dQ  MISSING!")
    if has_dkv:
        print(f"    dKV max={kv.grad.abs().max().item():.6f}")
    else:
        print(f"    dKV MISSING!")

    return passed


# ============================================================
# Test: edge case — all-invalid indices
# ============================================================
def test_all_invalid(
    total_tokens=64,
    num_heads=16,
    kv_lora_rank=256,
    rope_rank=64,
    topk=32,
    device="cuda",
):
    d_qk = kv_lora_rank + rope_rank
    scale = 1.0 / (d_qk ** 0.5)

    torch.manual_seed(42)
    q = torch.randn(total_tokens, num_heads, d_qk, dtype=torch.bfloat16, device=device)
    kv = torch.randn(total_tokens, 1, d_qk, dtype=torch.bfloat16, device=device)
    do = torch.randn(total_tokens, num_heads, kv_lora_rank, dtype=torch.bfloat16, device=device)

    # All indices invalid
    topk_indices = torch.full((total_tokens, topk), -1, dtype=torch.int32, device=device)

    o, lse = sparse_mla_fwd(q, kv, topk_indices, kv_lora_rank, scale)
    dq, dkv = sparse_mla_bwd(q, kv, o, do, topk_indices, lse, kv_lora_rank, scale)

    # With all-invalid indices, gradients should be zero
    dq_zero = dq.abs().max().item() < 1e-6
    dkv_zero = dkv.abs().max().item() < 1e-6
    passed = dq_zero and dkv_zero
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] All-invalid indices: dQ_max={dq.abs().max().item():.8f}, "
          f"dKV_max={dkv.abs().max().item():.8f}")
    return passed


# ============================================================
# Benchmark
# ============================================================
def benchmark_backward(
    batch=1,
    seq_len=4096,
    num_heads=128,
    kv_lora_rank=512,
    rope_rank=64,
    topk=1024,
    device="cuda",
):
    d_qk = kv_lora_rank + rope_rank
    total_tokens = batch * seq_len
    scale = 1.0 / (d_qk ** 0.5)

    torch.manual_seed(42)
    q = torch.randn(total_tokens, num_heads, d_qk, dtype=torch.bfloat16, device=device)
    kv = torch.randn(total_tokens, 1, d_qk, dtype=torch.bfloat16, device=device)

    topk_indices = torch.randint(0, total_tokens, (total_tokens, topk), dtype=torch.int32, device=device)

    # Forward
    o, lse = sparse_mla_fwd(q, kv, topk_indices, kv_lora_rank, scale)
    do = torch.randn_like(o)

    # Warmup + autotune
    for _ in range(5):
        sparse_mla_bwd(q, kv, o, do, topk_indices, lse, kv_lora_rank, scale)
    torch.cuda.synchronize()

    # Benchmark
    ev0 = torch.cuda.Event(enable_timing=True)
    ev1 = torch.cuda.Event(enable_timing=True)
    ev0.record()
    reps = 100
    for _ in range(reps):
        sparse_mla_bwd(q, kv, o, do, topk_indices, lse, kv_lora_rank, scale)
    ev1.record()
    torch.cuda.synchronize()
    ms = ev0.elapsed_time(ev1) / reps

    # BWD flops: ~5x the dot products of forward
    # S recompute (2x) + dP (1x) + dQ (2x) + dKV (2x) = ~7 matmuls vs fwd's 3
    flops_bwd = total_tokens * num_heads * topk * (
        2 * d_qk       # recompute S: Q @ K^T (lora + rope)
        + 2 * kv_lora_rank  # dP: dO @ K^T, and P @ V for dKV
        + 2 * d_qk       # dQ: dS @ K (lora + rope)
        + 2 * d_qk       # dKV: dS^T @ Q (lora + rope)
    )
    tflops = flops_bwd / (ms * 1e-3) / 1e12

    from aiter.ops.triton._triton_kernels.attention.deepseek_sparse_attention import (
        _sparse_mla_bwd_kernel,
    )
    best = _sparse_mla_bwd_kernel.best_config
    cfg = f"BH={best.kwargs['BLOCK_H']} TK={best.kwargs['TILE_K']} w={best.num_warps} s={best.num_stages}"

    lbl = f"B{batch}_S{seq_len}_H{num_heads}_topk{topk}"
    print(f"  {lbl:<30s} {ms:8.3f} ms  {tflops:8.1f} TFLOPS  [{cfg}]")

    return ms, tflops


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-only", action="store_true")
    parser.add_argument("--bench-only", action="store_true")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device index")
    args = parser.parse_args()

    os.environ["HIP_VISIBLE_DEVICES"] = str(args.gpu)
    device = "cuda"

    if not args.bench_only:
        print("=" * 60)
        print("Preprocess Tests")
        print("=" * 60)
        all_passed = True
        all_passed &= test_preprocess(256, 32, 512, device)
        all_passed &= test_preprocess(128, 128, 512, device)
        all_passed &= test_preprocess(64, 16, 256, device)

        print()
        print("=" * 60)
        print("Backward Correctness Tests")
        print("=" * 60)

        test_configs = [
            # (batch, seq_len, num_heads, kv_lora_rank, rope_rank, topk)
            (1, 128, 16, 256, 64, 64),
            (1, 256, 32, 512, 64, 128),
            (1, 256, 128, 512, 64, 128),
            (1, 64, 16, 256, 64, 64),
            (1, 512, 128, 512, 64, 256),
        ]

        for cfg in test_configs:
            all_passed &= test_backward(*cfg, device=device)

        print()
        print("=" * 60)
        print("Edge Case Tests")
        print("=" * 60)
        all_passed &= test_all_invalid(device=device)

        print()
        print("=" * 60)
        print("End-to-End Autograd Tests")
        print("=" * 60)
        all_passed &= test_autograd_e2e(device=device)

        print()
        if all_passed:
            print("All tests PASSED!")
        else:
            print("Some tests FAILED!")
            if args.test_only:
                sys.exit(1)

    if not args.test_only:
        print()
        print("=" * 60)
        print("Backward Benchmarks")
        print("=" * 60)

        bench_configs = [
            # --- Synthetic configs (from TileLang paper / AITER) ---
            # (batch, seq_len, num_heads, kv_lora_rank, rope_rank, topk)
            (1, 4096, 128, 512, 64, 1024),
            (1, 4096, 128, 512, 64, 2048),
            (1, 8192, 128, 512, 64, 1024),
            (1, 8192, 128, 512, 64, 2048),
            (2, 4096, 128, 512, 64, 1024),
            (1, 4096, 32, 256, 64, 1024),
            (1, 4096, 16, 512, 64, 1024),
            # --- Real-world configs from DeepSeek-V3.2 DSA training ---
            # Source: https://arxiv.org/pdf/2512.02556
            # DeepSeek-V3.2 trains DSA at 128K seq_len, topk=2048, H=128,
            # kv_lora_rank=512, rope_rank=64.
            # Warmup: 1K steps, 16 seqs × 128K / 2048 GPUs ≈ 1M tokens/GPU
            # Main:  15K steps, 480 seqs × 128K / 2048 GPUs ≈ 30K tokens/GPU
            (1, 32768, 128, 512, 64, 2048),    # 32K tokens, ~30K/GPU from main DSA training
            (1, 65536, 128, 512, 64, 2048),    # 64K tokens
            (1, 131072, 128, 512, 64, 2048),   # 128K tokens, single full DSA training sequence
        ]

        for cfg in bench_configs:
            try:
                benchmark_backward(*cfg, device=device)
            except Exception as e:
                print(f"  SKIPPED {cfg}: {e}")


if __name__ == "__main__":
    main()
