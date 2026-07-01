"""Benchmark FlyDSL backward vs Triton backward on the canonical shape.

Shape: B=1, HQ=HK=5, Sq=Sk=75600, D=128, bf16, non-causal

Run inside docker:
  HIP_VISIBLE_DEVICES=0 python op_tests/flydsl_tests/bench_flydsl_bwd.py
"""

import math
import torch
from aiter.ops.triton.attention.mha import (
    flash_attn_func,
    mha_set_use_flydsl_bwd_kernel,
)

# ── shape ───────────────────────────────────────────────────────────────────
B, HQ, HK, SQ, SK, D = 1, 5, 5, 75600, 75600, 128
DTYPE = torch.bfloat16
CAUSAL = False
WARMUP = 5
ITERS = 20

# Attention backward FLOPs (standard approximation):
#   4 matrix multiplies of shape [B, H, S, S] @ [B, H, S, D]:
#   4 × 2 × B × H × Sq × Sk × D
FLOPS = 4 * 2 * B * HQ * SQ * SK * D


def benchmark(label, use_flydsl: bool):
    mha_set_use_flydsl_bwd_kernel(use_flydsl)
    torch.manual_seed(42)
    q = torch.randn(B, SQ, HQ, D, dtype=DTYPE, device="cuda", requires_grad=True)
    k = torch.randn(B, SK, HK, D, dtype=DTYPE, device="cuda")
    v = torch.randn(B, SK, HK, D, dtype=DTYPE, device="cuda")

    # Forward (no FlyDSL fwd, just Triton to get lse for the backward)
    out, lse = flash_attn_func(
        q, k, v, softmax_scale=D**-0.5, causal=CAUSAL, return_lse=True
    )
    dout = torch.randn_like(out)

    # Warmup
    for _ in range(WARMUP):
        if q.grad is not None:
            q.grad = None
        out.backward(dout, retain_graph=True)
    torch.cuda.synchronize()

    # Timed iterations
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(ITERS):
        if q.grad is not None:
            q.grad = None
        out.backward(dout, retain_graph=True)
    end.record()
    torch.cuda.synchronize()

    ms = start.elapsed_time(end) / ITERS
    tflops = FLOPS / (ms * 1e-3) / 1e12
    print(f"{label:30s}  {ms:.3f} ms   {tflops:.1f} TFLOPS")
    return tflops


print(f"Shape: B={B} HQ={HQ} HK={HK} Sq={SQ} Sk={SK} D={D} {DTYPE} causal={CAUSAL}")
print(f"{'':30s}  {'ms':>8}   {'TFLOPS':>10}")
print("-" * 55)

t_triton = benchmark("Triton backward (baseline)", use_flydsl=False)
t_flydsl = benchmark("FlyDSL MFMA backward", use_flydsl=True)

print("-" * 55)
print(f"FlyDSL / Triton speedup: {t_flydsl / t_triton:.2f}×")
print(f"CK target: ~836 TFLOPS  (FlyDSL is {t_flydsl / 836:.0%} of CK)")
