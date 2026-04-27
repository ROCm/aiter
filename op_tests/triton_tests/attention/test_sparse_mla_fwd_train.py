"""
Test and benchmark for sparse MLA forward training kernel.

Compares against a PyTorch reference implementation for correctness,
and benchmarks against the original AITER inference kernel.
"""

import sys
import os
import time
import argparse
import torch
import triton

# Ensure aiter is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

from aiter.ops.triton._triton_kernels.attention.deepseek_sparse_attention import (
    sparse_mla_fwd,
)
from aiter.ops.triton.attention.unified_attention_sparse_mla import (
    unified_attention_sparse_mla,
)


# ============================================================
# Reference implementation (PyTorch)
# ============================================================
def reference_sparse_mla_fwd(q, kv, topk_indices, kv_lora_rank=512, scale=None):
    """
    Pure PyTorch reference for sparse MLA forward.

    Args:
        q:             [total_tokens, num_heads, d_qk] bfloat16
        kv:            [total_tokens, 1, d_qk] bfloat16
        topk_indices:  [total_tokens, topk] int32 (absolute positions, -1 for invalid)
        kv_lora_rank:  int
        scale:         float

    Returns:
        o:   [total_tokens, num_heads, kv_lora_rank] float32
        lse: [total_tokens, num_heads] float32
    """
    total_tokens, num_heads, d_qk = q.shape
    topk = topk_indices.shape[1]

    if scale is None:
        scale = 1.0 / (d_qk ** 0.5)

    if kv.dim() == 2:
        kv = kv.unsqueeze(1)

    q_f = q.float()
    kv_f = kv.float().squeeze(1)  # [total_tokens, d_qk]

    o = torch.zeros(total_tokens, num_heads, kv_lora_rank, dtype=torch.float32, device=q.device)
    lse = torch.full((total_tokens, num_heads), float("-inf"), dtype=torch.float32, device=q.device)

    for i in range(total_tokens):
        qi = q_f[i]  # [num_heads, d_qk]
        idx = topk_indices[i]  # [topk]
        valid = idx != -1

        if not valid.any():
            continue

        valid_idx = idx[valid].long()
        kv_sel = kv_f[valid_idx]  # [n_valid, d_qk]

        # Score: q @ k^T
        k_sel = kv_sel  # [n_valid, d_qk]
        scores = scale * (qi @ k_sel.T)  # [num_heads, n_valid]

        # Softmax
        m = scores.max(dim=-1).values  # [num_heads]
        exp_scores = torch.exp(scores - m.unsqueeze(-1))
        l = exp_scores.sum(dim=-1)  # [num_heads]

        # Output: P @ V
        v_sel = kv_sel[:, :kv_lora_rank]  # [n_valid, kv_lora_rank]
        o[i] = (exp_scores @ v_sel) / l.unsqueeze(-1)
        lse[i] = m + torch.log(l)

    return o, lse


# ============================================================
# Prepare data for AITER inference kernel comparison
# ============================================================
def prepare_aiter_inputs(q, kv, topk_indices, batch_size, seq_len, kv_lora_rank):
    """
    Convert contiguous training tensors to AITER's paged format.
    Uses identity block_table (block i -> physical block i).
    """
    total_tokens, num_heads, d_qk = q.shape
    block_size = 64  # AITER's block size

    num_blocks_per_seq = (seq_len + block_size - 1) // block_size
    total_blocks = batch_size * num_blocks_per_seq

    # Create blocked KV: [total_blocks, block_size, 1, d_qk]
    kv_sq = kv.squeeze(1)  # [total_tokens, d_qk]
    blocked_kv = torch.zeros(total_blocks, block_size, 1, d_qk, dtype=kv.dtype, device=kv.device)
    for b in range(batch_size):
        for blk in range(num_blocks_per_seq):
            src_start = b * seq_len + blk * block_size
            src_end = min(src_start + block_size, b * seq_len + seq_len)
            n = src_end - src_start
            blocked_kv[b * num_blocks_per_seq + blk, :n, 0, :] = kv_sq[src_start:src_end]

    # Identity block table
    block_table = torch.arange(total_blocks, dtype=torch.int32, device=kv.device).view(batch_size, num_blocks_per_seq)

    # Convert topk_indices (absolute positions) to positions within blocked KV
    # Since block_table is identity, the position in blocked KV is:
    # block_idx * block_size + offset = same as absolute position within seq
    # We need to convert from global absolute to per-sequence position mapped through block_table
    indices_in_kvcache = topk_indices.clone()
    for b in range(batch_size):
        mask = (topk_indices[b * seq_len:(b + 1) * seq_len] != -1)
        # Convert absolute to per-seq position
        per_seq = topk_indices[b * seq_len:(b + 1) * seq_len] - b * seq_len
        # Map through block_table
        blk_idx = per_seq // block_size
        offset = per_seq % block_size
        mapped = torch.where(
            mask,
            block_table[b][blk_idx.clamp(0).long()] * block_size + offset,
            torch.tensor(-1, dtype=torch.int32, device=kv.device),
        )
        indices_in_kvcache[b * seq_len:(b + 1) * seq_len] = mapped

    # cu_seqlens_q
    cu_seqlens_q = torch.arange(0, batch_size + 1, dtype=torch.int32, device=q.device) * seq_len
    seqused_k = torch.full((batch_size,), seq_len, dtype=torch.int32, device=q.device)

    return blocked_kv, block_table, indices_in_kvcache, cu_seqlens_q, seqused_k


# ============================================================
# Test correctness
# ============================================================
def test_correctness(
    batch=1,
    seq_len=256,
    num_heads=128,
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

    # Generate random topk indices (absolute positions within each sequence)
    topk_indices = torch.full((total_tokens, topk), -1, dtype=torch.int32, device=device)
    for b in range(batch):
        for t in range(seq_len):
            global_t = b * seq_len + t
            n_valid = min(topk, seq_len)
            # Random subset of positions in this sequence
            perm = torch.randperm(seq_len, device=device)[:n_valid]
            abs_pos = perm + b * seq_len  # convert to absolute position
            topk_indices[global_t, :n_valid] = abs_pos.int()

    # Run our training kernel
    o_train, lse_train = sparse_mla_fwd(q, kv, topk_indices, kv_lora_rank, scale)

    # Run reference
    o_ref, lse_ref = reference_sparse_mla_fwd(q, kv, topk_indices, kv_lora_rank, scale)

    # Compare O
    o_train_f = o_train.float()
    max_diff_o = (o_train_f - o_ref).abs().max().item()
    mean_diff_o = (o_train_f - o_ref).abs().mean().item()

    # Compare LSE
    max_diff_lse = (lse_train - lse_ref).abs().max().item()
    mean_diff_lse = (lse_train - lse_ref).abs().mean().item()

    passed = max_diff_o < 2e-2 and max_diff_lse < 1e-1
    status = "PASS" if passed else "FAIL"

    print(f"  [{status}] B={batch}, S={seq_len}, H={num_heads}, "
          f"d_v={kv_lora_rank}, d_rope={rope_rank}, topk={topk}")
    print(f"    O:   max_diff={max_diff_o:.6f}, mean_diff={mean_diff_o:.6f}")
    print(f"    LSE: max_diff={max_diff_lse:.6f}, mean_diff={mean_diff_lse:.6f}")

    return passed


# ============================================================
# Benchmark
# ============================================================
def benchmark_kernel(fn, warmup=50, rep=200):
    """Simple benchmark using torch.cuda.Event."""
    # Warmup
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(rep)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(rep)]

    for i in range(rep):
        start_events[i].record()
        fn()
        end_events[i].record()

    torch.cuda.synchronize()
    times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    times.sort()
    # Use median
    median_ms = times[len(times) // 2]
    mean_ms = sum(times) / len(times)
    return median_ms, mean_ms


def naive_masked_dense_attn(q_batched, kv_batched, topk_indices_batched, kv_lora_rank, scale):
    """
    Naive masked dense attention baseline (ZhengKai91 approach).
    Creates a boolean mask from topk_indices and does full dense attention with masking.

    Args:
        q_batched:           [batch, seq_len, num_heads, d_qk] bfloat16
        kv_batched:          [batch, seq_len, 1, d_qk] bfloat16
        topk_indices_batched:[batch, seq_len, topk] int32 (per-seq relative positions)
        kv_lora_rank:        int
        scale:               float
    """
    B, S, H, d_qk = q_batched.shape
    topk = topk_indices_batched.shape[2]

    # Build boolean mask [batch, seq_len, seq_len] from topk_indices
    mask = torch.zeros(B, S, S, dtype=torch.bool, device=q_batched.device)
    # Scatter True at topk positions
    idx = topk_indices_batched.long().clamp(min=0)  # [B, S, topk]
    valid = topk_indices_batched != -1
    # Expand and scatter
    mask.scatter_(2, idx, valid)

    # Dense attention with mask
    # Q: [B, S, H, d_qk], K: [B, S, 1, d_qk] -> broadcast to all heads
    q_f = q_batched.float()
    k_f = kv_batched.float().expand(-1, -1, H, -1)  # [B, S, H, d_qk]
    v_f = kv_batched[..., :kv_lora_rank].float().expand(-1, -1, H, -1)  # [B, S, H, d_v]

    # Transpose to [B, H, S, d] for bmm
    q_t = q_f.permute(0, 2, 1, 3)  # [B, H, S, d_qk]
    k_t = k_f.permute(0, 2, 1, 3)  # [B, H, S, d_qk]
    v_t = v_f.permute(0, 2, 1, 3)  # [B, H, S, d_v]

    scores = scale * torch.matmul(q_t, k_t.transpose(-2, -1))  # [B, H, S, S]

    # Apply mask: [B, 1, S, S] broadcast across heads
    attn_mask = mask.unsqueeze(1)  # [B, 1, S, S]
    scores = scores.masked_fill(~attn_mask, float("-inf"))

    probs = torch.softmax(scores, dim=-1)
    o = torch.matmul(probs.to(v_t.dtype), v_t)  # [B, H, S, d_v]
    o = o.permute(0, 2, 1, 3)  # [B, S, H, d_v]
    return o.to(torch.bfloat16)


def run_benchmark(
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

    # Generate topk indices (per-sequence relative positions for masked baseline)
    topk_indices = torch.full((total_tokens, topk), -1, dtype=torch.int32, device=device)
    topk_indices_rel = torch.full((batch, seq_len, topk), -1, dtype=torch.int32, device=device)
    for b in range(batch):
        for t in range(seq_len):
            global_t = b * seq_len + t
            n_valid = min(topk, seq_len)
            perm = torch.randperm(seq_len, device=device)[:n_valid]
            topk_indices[global_t, :n_valid] = (perm + b * seq_len).int()
            topk_indices_rel[b, t, :n_valid] = perm.int()

    # ---- Benchmark training kernel ----
    def run_train():
        return sparse_mla_fwd(q, kv, topk_indices, kv_lora_rank, scale)

    median_train, mean_train = benchmark_kernel(run_train)

    # ---- Benchmark AITER inference kernel ----
    blocked_kv, block_table, indices_in_kvcache, cu_seqlens_q, seqused_k = (
        prepare_aiter_inputs(q, kv, topk_indices, batch, seq_len, kv_lora_rank)
    )
    o_aiter = torch.empty(total_tokens, num_heads, kv_lora_rank, dtype=q.dtype, device=device)

    def run_aiter():
        unified_attention_sparse_mla(
            q, blocked_kv, o_aiter,
            cu_seqlens_q, seq_len, seqused_k, seq_len,
            scale, indices_in_kvcache, block_table, kv_lora_rank,
        )

    median_aiter, mean_aiter = benchmark_kernel(run_aiter)

    # ---- Benchmark naive masked dense attention ----
    q_batched = q.view(batch, seq_len, num_heads, d_qk)
    kv_batched = kv.view(batch, seq_len, 1, d_qk)

    median_masked = None
    try:
        def run_masked():
            return naive_masked_dense_attn(
                q_batched, kv_batched, topk_indices_rel, kv_lora_rank, scale,
            )
        median_masked, _ = benchmark_kernel(run_masked, warmup=10, rep=50)
    except torch.cuda.OutOfMemoryError:
        median_masked = float("inf")
        print("    Masked baseline: OOM")
    except Exception as e:
        median_masked = float("inf")
        print(f"    Masked baseline: {e}")

    # Compute TFLOPS (for sparse kernels: only topk FLOPs)
    flops = total_tokens * num_heads * topk * (2 * d_qk + 2 * kv_lora_rank)
    tflops_train = flops / (median_train * 1e-3) / 1e12
    tflops_aiter = flops / (median_aiter * 1e-3) / 1e12

    # For masked: FLOPs are O(N^2) — full dense attention
    flops_dense = total_tokens * num_heads * seq_len * (2 * d_qk + 2 * kv_lora_rank)
    tflops_masked = flops_dense / (median_masked * 1e-3) / 1e12 if median_masked != float("inf") else 0

    speedup_vs_aiter = median_aiter / median_train
    speedup_vs_masked = median_masked / median_train if median_masked != float("inf") else float("inf")

    print(f"  B={batch}, S={seq_len}, H={num_heads}, d_v={kv_lora_rank}, "
          f"d_rope={rope_rank}, topk={topk}")
    print(f"    Train kernel:  median={median_train:.3f} ms, "
          f"TFLOPS={tflops_train:.1f}")
    print(f"    AITER kernel:  median={median_aiter:.3f} ms, "
          f"TFLOPS={tflops_aiter:.1f}")
    if median_masked != float("inf"):
        print(f"    Masked dense:  median={median_masked:.3f} ms, "
              f"TFLOPS={tflops_masked:.1f} (dense FLOPs)")
    else:
        print(f"    Masked dense:  OOM or error")
    print(f"    vs AITER: {speedup_vs_aiter:.2f}x, vs Masked: {speedup_vs_masked:.1f}x")
    print()

    return {
        "batch": batch, "seq_len": seq_len, "num_heads": num_heads,
        "kv_lora_rank": kv_lora_rank, "rope_rank": rope_rank, "topk": topk,
        "train_ms": median_train, "aiter_ms": median_aiter,
        "masked_ms": median_masked if median_masked != float("inf") else None,
        "speedup_aiter": speedup_vs_aiter,
        "speedup_masked": speedup_vs_masked if median_masked != float("inf") else None,
        "tflops_train": tflops_train, "tflops_aiter": tflops_aiter,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-only", action="store_true")
    parser.add_argument("--bench-only", action="store_true")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device index")
    args = parser.parse_args()

    os.environ["HIP_VISIBLE_DEVICES"] = str(args.gpu)
    device = "cuda"

    # ============================================================
    # Correctness tests
    # ============================================================
    if not args.bench_only:
        print("=" * 60)
        print("Correctness Tests")
        print("=" * 60)

        test_configs = [
            # (batch, seq_len, num_heads, kv_lora_rank, rope_rank, topk)
            # Small configs for quick validation
            (1, 128, 16, 256, 64, 64),
            (1, 256, 32, 512, 64, 128),
            (1, 256, 128, 512, 64, 128),
            # AITER test configs
            (1, 64, 16, 256, 64, 64),
            (1, 177, 32, 512, 64, 78),
            (8, 64, 32, 512, 64, 64),
            # DeepSeek V3 config
            (1, 512, 128, 512, 64, 256),
            (2, 256, 128, 512, 64, 128),
        ]

        all_passed = True
        for cfg in test_configs:
            passed = test_correctness(*cfg, device=device)
            all_passed = all_passed and passed

        print()
        if all_passed:
            print("All correctness tests PASSED!")
        else:
            print("Some tests FAILED!")
            if args.test_only:
                sys.exit(1)

    # ============================================================
    # Benchmarks
    # ============================================================
    if not args.test_only:
        print()
        print("=" * 60)
        print("Benchmarks")
        print("=" * 60)

        bench_configs = [
            # --- Synthetic configs (from TileLang paper / AITER) ---
            # (batch, seq_len, num_heads, kv_lora_rank, rope_rank, topk)
            (1, 4096, 128, 512, 64, 1024),   # TileLang default
            (1, 4096, 128, 512, 64, 2048),   # TileLang default with larger topk
            (1, 8192, 128, 512, 64, 1024),   # Longer sequence
            (1, 8192, 128, 512, 64, 2048),   # Longer sequence, larger topk
            (2, 4096, 128, 512, 64, 1024),   # Multi-batch
            # Smaller head dim configs (from AITER tests)
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

        results = []
        for cfg in bench_configs:
            try:
                r = run_benchmark(*cfg, device=device)
                results.append(r)
            except Exception as e:
                print(f"  SKIPPED B={cfg[0]}, S={cfg[1]}: {e}")
                print()

        # Summary table
        if results:
            print("=" * 80)
            print("Summary")
            print("=" * 80)
            print(f"{'Config':<35} {'Train(ms)':>10} {'AITER(ms)':>10} {'Masked(ms)':>11} {'vs AITER':>9} {'vs Masked':>10}")
            print("-" * 87)
            for r in results:
                cfg = f"B{r['batch']}_S{r['seq_len']}_H{r['num_heads']}_topk{r['topk']}"
                masked_str = f"{r['masked_ms']:>11.1f}" if r.get('masked_ms') else "       OOM"
                speedup_m = f"{r['speedup_masked']:>9.1f}x" if r.get('speedup_masked') else "      N/A"
                print(f"{cfg:<35} {r['train_ms']:>10.3f} {r['aiter_ms']:>10.3f} {masked_str} {r['speedup_aiter']:>8.2f}x {speedup_m}")


if __name__ == "__main__":
    main()
