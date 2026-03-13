# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""
Accuracy and performance tests for Triton MLA Decode Attention (gfx1201).

Tests Stage1, Stage2, and the combined pipeline against a PyTorch reference
with paged KV cache (page table lookup).

Usage:
    # Accuracy (default)
    python -m op_tests.triton_tests.attention.test_mla_decode
    python -m op_tests.triton_tests.attention.test_mla_decode --mode accuracy

    # Performance
    python -m op_tests.triton_tests.attention.test_mla_decode --mode perf

    # Custom config
    python -m op_tests.triton_tests.attention.test_mla_decode -b 32 -s 1024 --splits 4

    # Pytest
    pytest op_tests/triton_tests/attention/test_mla_decode.py -v
"""

import argparse
import random

import pytest
import torch
import torch.nn.functional as F
import triton

from aiter.ops.triton.gluon.mla_decode import (
    mla_decode,
    mla_decode_stage1,
    mla_decode_stage2,
)

try:
    from aiter.test_common import checkAllclose, perftest
except ImportError:
    checkAllclose = None
    perftest = None


# ============================================================
# Helpers
# ============================================================
def cdiv(a: int, b: int) -> int:
    return -(a // -b)


def bench_sync(fn, warmup=50, rep=100):
    """Benchmark with cuda sync only — no L2 cache flush between iterations."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for _ in range(rep):
        fn()
    end_event.record()
    torch.cuda.synchronize()
    return start_event.elapsed_time(end_event) / rep


# ============================================================
# PyTorch reference: paged MLA decode attention
# ============================================================
def torch_paged_mla_stage1(q, k_buffer, v_buffer, req_to_page, seq_lens,
                           num_kv_splits, sm_scale, page_size):
    B, H_Q, D_QK = q.shape
    D_V = v_buffer.shape[-1]
    H_KV = k_buffer.shape[2]
    kv_group_size = H_Q // H_KV

    attn_logits = torch.zeros(B, H_Q, num_kv_splits, D_V + 1,
                              dtype=torch.float32, device=q.device)

    for b in range(B):
        sl = seq_lens[b].item()
        page_ids = req_to_page[b, :cdiv(sl, page_size)].long()
        k_flat = k_buffer[page_ids].reshape(-1, H_KV, D_QK)[:sl]
        v_flat = v_buffer[page_ids].reshape(-1, H_KV, D_V)[:sl]
        kv_len_per_split = cdiv(sl, num_kv_splits)

        for s in range(num_kv_splits):
            ss, se = kv_len_per_split * s, min(kv_len_per_split * s + kv_len_per_split, sl)
            if se <= ss:
                continue
            for h in range(H_Q):
                kv_h = h // kv_group_size
                scores = (q[b, h, :].float() @ k_flat[ss:se, kv_h, :].float().T) * sm_scale
                e_max = scores.max()
                exp_s = torch.exp(scores - e_max)
                e_sum = exp_s.sum()
                attn_logits[b, h, s, :D_V] = (exp_s @ v_flat[ss:se, kv_h, :].float()) / e_sum
                attn_logits[b, h, s, D_V] = e_max + torch.log(e_sum)
    return attn_logits


def torch_paged_mla_combined(q, k_buffer, v_buffer, req_to_page, seq_lens,
                             sm_scale, page_size):
    B, H_Q, D_QK = q.shape
    D_V = v_buffer.shape[-1]
    H_KV = k_buffer.shape[2]
    kv_group_size = H_Q // H_KV

    o = torch.zeros(B, H_Q, D_V, dtype=q.dtype, device=q.device)
    for b in range(B):
        sl = seq_lens[b].item()
        page_ids = req_to_page[b, :cdiv(sl, page_size)].long()
        k_flat = k_buffer[page_ids].reshape(-1, H_KV, D_QK)[:sl]
        v_flat = v_buffer[page_ids].reshape(-1, H_KV, D_V)[:sl]
        for h in range(H_Q):
            kv_h = h // kv_group_size
            scores = (q[b, h, :].float() @ k_flat[:, kv_h, :].float().T) * sm_scale
            w = F.softmax(scores, dim=0)
            o[b, h, :] = (w @ v_flat[:, kv_h, :].float()).to(q.dtype)
    return o


# ============================================================
# Data generation
# ============================================================
def make_test_data(batch_size, seq_len, num_q_heads, num_kv_heads, head_dim,
                   kv_lora_rank, page_size, num_kv_splits, dtype, device,
                   seed=42, varlen=False, num_pages_override=None):
    """
    Generate test data.

    Args:
        num_pages_override: If set, use this as total_pages (for perf testing
            to match mla_default_gluon.py's 20363-page KV cache).
            If None, allocate just enough pages for accuracy testing.
    """
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    num_pages_per_seq = cdiv(seq_len, page_size)
    if num_pages_override is not None:
        total_pages = num_pages_override
    else:
        total_pages = batch_size * num_pages_per_seq + 128

    cache_seqlens = torch.full((batch_size,), seq_len, dtype=torch.int32, device=device)
    if varlen:
        alignment = page_size * num_kv_splits
        for i in range(batch_size):
            v = max(int(random.normalvariate(seq_len, seq_len / 4)), alignment)
            v = min(v, seq_len)
            v = (v // alignment) * alignment
            cache_seqlens[i] = max(v, alignment)

    req_to_page = torch.randint(0, total_pages, (batch_size, num_pages_per_seq),
                                dtype=torch.int32, device=device)

    q = torch.randn(batch_size, num_q_heads, head_dim, dtype=dtype, device=device)
    k_buffer = torch.randn(total_pages, page_size, num_kv_heads, head_dim,
                            dtype=dtype, device=device)
    # Non-contiguous V (slice of K, same as mla_default_gluon.py) for realistic perf
    v_buffer = k_buffer[:, :, :, :kv_lora_rank]

    return q, k_buffer, v_buffer, req_to_page, cache_seqlens


# ============================================================
# Accuracy tests
# ============================================================
MLA_CONFIGS = {
    "num_q_heads": 16,
    "num_kv_heads": 1,
    "head_dim": 576,
    "kv_lora_rank": 512,
    "page_size": 16,
    "num_pages": 20363,  # match mla_default_gluon.py for fair perf comparison
}


def run_accuracy_test(batch_size, seq_len, num_kv_splits, dtype, varlen=False):
    cfg = MLA_CONFIGS
    sm_scale = 1.0 / (cfg["head_dim"] ** 0.5)
    device = "cuda"

    q, k_buf, v_buf, req_to_page, seqlens = make_test_data(
        batch_size, seq_len, cfg["num_q_heads"], cfg["num_kv_heads"],
        cfg["head_dim"], cfg["kv_lora_rank"], cfg["page_size"],
        num_kv_splits, dtype, device, varlen=varlen,
    )

    attn_logits = torch.zeros(
        batch_size, cfg["num_q_heads"], num_kv_splits, cfg["kv_lora_rank"] + 2,
        dtype=torch.float32, device=device,
    )
    o_kernel = torch.empty(
        batch_size, cfg["num_q_heads"], cfg["kv_lora_rank"],
        dtype=dtype, device=device,
    )

    results = {}
    tag = f"B={batch_size} S={seq_len} splits={num_kv_splits} {'varlen' if varlen else ''} {dtype}"
    print(f"\n--- {tag} ---")

    # Stage1
    mla_decode_stage1(
        q, k_buf, v_buf, attn_logits, req_to_page, seqlens,
        num_kv_splits, sm_scale, cfg["page_size"],
    )
    torch.cuda.synchronize()
    ref_logits = torch_paged_mla_stage1(
        q, k_buf, v_buf, req_to_page, seqlens,
        num_kv_splits, sm_scale, cfg["page_size"],
    )
    kernel_v = attn_logits[:, :, :, :cfg["kv_lora_rank"]]
    ref_v = ref_logits[:, :, :, :cfg["kv_lora_rank"]]
    diff_s1 = (kernel_v.float() - ref_v.float()).abs().max().item()
    cos_s1 = F.cosine_similarity(kernel_v.float().reshape(1, -1), ref_v.float().reshape(1, -1)).item()
    s1_pass = diff_s1 < 2e-2 and cos_s1 > 0.999
    results["stage1"] = (s1_pass, diff_s1, cos_s1)
    print(f"  Stage1: {'PASS' if s1_pass else 'FAIL'}  max_diff={diff_s1:.4e}  cos={cos_s1:.6f}")

    # Stage2 (feed PyTorch stage1 data)
    attn_logits_ref_padded = torch.zeros_like(attn_logits)
    attn_logits_ref_padded[:, :, :, :cfg["kv_lora_rank"] + 1] = ref_logits
    o_s2 = torch.empty_like(o_kernel)
    mla_decode_stage2(attn_logits_ref_padded, q, o_s2, v_buf, seqlens, num_kv_splits)
    torch.cuda.synchronize()
    o_ref = torch_paged_mla_combined(q, k_buf, v_buf, req_to_page, seqlens, sm_scale, cfg["page_size"])
    diff_s2 = (o_s2.float() - o_ref.float()).abs().max().item()
    cos_s2 = F.cosine_similarity(o_s2.float().reshape(1, -1), o_ref.float().reshape(1, -1)).item()
    s2_pass = diff_s2 < 2e-2 and cos_s2 > 0.999
    results["stage2"] = (s2_pass, diff_s2, cos_s2)
    print(f"  Stage2: {'PASS' if s2_pass else 'FAIL'}  max_diff={diff_s2:.4e}  cos={cos_s2:.6f}")

    # Combined
    attn_logits.zero_()
    mla_decode(
        q, k_buf, v_buf, o_kernel, req_to_page, seqlens,
        attn_logits, num_kv_splits, sm_scale, cfg["page_size"],
    )
    torch.cuda.synchronize()
    diff_c = (o_kernel.float() - o_ref.float()).abs().max().item()
    cos_c = F.cosine_similarity(o_kernel.float().reshape(1, -1), o_ref.float().reshape(1, -1)).item()
    c_pass = diff_c < 2e-2 and cos_c > 0.999
    results["combined"] = (c_pass, diff_c, cos_c)
    print(f"  Combined: {'PASS' if c_pass else 'FAIL'}  max_diff={diff_c:.4e}  cos={cos_c:.6f}")

    all_pass = all(p for p, _, _ in results.values())
    return all_pass, results


# ============================================================
# Performance tests
# ============================================================
def run_perf_test(batch_size, seq_len, num_kv_splits, dtype, bench_mode="sync"):
    """
    Args:
        bench_mode: "sync"  — warmup + cuda sync timing, no L2 cache flush (default).
                    "bench" — triton.testing.do_bench with L2 cache flush.
    """
    cfg = MLA_CONFIGS
    sm_scale = 0.13523378
    device = "cuda"

    q, k_buf, v_buf, req_to_page, seqlens = make_test_data(
        batch_size, seq_len, cfg["num_q_heads"], cfg["num_kv_heads"],
        cfg["head_dim"], cfg["kv_lora_rank"], cfg["page_size"],
        num_kv_splits, dtype, device,
        num_pages_override=cfg["num_pages"],
    )

    attn_logits = torch.zeros(
        batch_size, cfg["num_q_heads"], num_kv_splits, cfg["kv_lora_rank"] + 2,
        dtype=torch.float32, device=device,
    )
    o_kernel = torch.empty(
        batch_size, cfg["num_q_heads"], cfg["kv_lora_rank"],
        dtype=dtype, device=device,
    )

    kernel_fn = lambda: mla_decode(
        q, k_buf, v_buf, o_kernel, req_to_page, seqlens,
        attn_logits, num_kv_splits, sm_scale, cfg["page_size"],
    )

    if bench_mode == "bench":
        ms = triton.testing.do_bench(kernel_fn, warmup=50, rep=100, return_mode="mean")
    else:
        ms = bench_sync(kernel_fn, warmup=50, rep=100)

    total_tokens = seqlens.sum().item()
    qk_flops = total_tokens * cfg["num_q_heads"] * cfg["head_dim"] * 2
    av_flops = total_tokens * cfg["num_q_heads"] * cfg["kv_lora_rank"] * 2
    tflops = (qk_flops + av_flops) / 1e12 / (ms * 1e-3)

    elem_bytes = q.element_size()
    k_bytes = total_tokens * cfg["num_kv_heads"] * cfg["head_dim"] * elem_bytes
    q_bytes = batch_size * cfg["num_q_heads"] * cfg["head_dim"] * elem_bytes
    o_bytes = batch_size * cfg["num_q_heads"] * cfg["kv_lora_rank"] * elem_bytes
    total_bytes = k_bytes + q_bytes + o_bytes
    bw_tb_s = total_bytes / 1e12 / (ms * 1e-3)

    print(f"  [{bench_mode:5s}] B={batch_size:<4d} S={seq_len:<6d} splits={num_kv_splits}  "
          f"time={ms:.3f}ms  "
          f"{tflops:.2f} TFLOP/s  "
          f"{bw_tb_s * 1e3:.2f} GB/s")

    return {
        "batch": batch_size, "seq_len": seq_len, "splits": num_kv_splits,
        "bench_mode": bench_mode, "ms": ms, "tflops": tflops, "bw_gb_s": bw_tb_s * 1e3,
    }


# ============================================================
# Pytest parametrized tests
# ============================================================
ACCURACY_CASES = [
    (32, 1024, 2, torch.bfloat16, False),
    (32, 1024, 2, torch.bfloat16, True),
    (32, 1024, 4, torch.bfloat16, False),
    (32, 1024, 2, torch.float16, False),
    (8, 256, 2, torch.bfloat16, False),
    (64, 2048, 4, torch.bfloat16, False),
]


@pytest.mark.parametrize("batch,seq_len,splits,dtype,varlen", ACCURACY_CASES)
def test_mla_decode_accuracy(batch, seq_len, splits, dtype, varlen):
    passed, _ = run_accuracy_test(batch, seq_len, splits, dtype, varlen)
    assert passed, f"Accuracy test failed for B={batch} S={seq_len} splits={splits}"


# ============================================================
# CLI
# ============================================================
def parse_args():
    p = argparse.ArgumentParser(description="MLA Decode Attention — Accuracy & Performance")
    p.add_argument("--mode", choices=["accuracy", "perf", "all"], default="accuracy")
    p.add_argument("-b", "--batch", type=int, nargs="+", default=[32])
    p.add_argument("-s", "--seq_len", type=int, nargs="+", default=[1024])
    p.add_argument("--splits", type=int, nargs="+", default=[2])
    p.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    p.add_argument("--bench", choices=["sync", "bench"], default="sync",
                   help="sync: cuda sync only (no L2 flush, default); bench: triton do_bench (L2 flush)")
    p.add_argument("--varlen", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16

    if args.mode in ("accuracy", "all"):
        print("\n" + "=" * 70)
        print("ACCURACY TESTS")
        print("=" * 70)
        all_pass = True
        for b in args.batch:
            for s in args.seq_len:
                for sp in args.splits:
                    passed, _ = run_accuracy_test(b, s, sp, dtype, args.varlen)
                    all_pass = all_pass and passed
        print(f"\n{'ALL PASSED' if all_pass else 'SOME FAILED'}")

    if args.mode in ("perf", "all"):
        print("\n" + "=" * 70)
        print(f"PERFORMANCE TESTS  (bench_mode={args.bench})")
        print("=" * 70)
        for b in args.batch:
            for s in args.seq_len:
                for sp in args.splits:
                    run_perf_test(b, s, sp, dtype, bench_mode=args.bench)


if __name__ == "__main__":
    main()
