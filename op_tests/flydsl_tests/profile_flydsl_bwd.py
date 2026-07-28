"""Per-kernel profiler for the FlyDSL backward pipeline.

Times each of the 3 passes (preprocess / dK-dV / dQ) in isolation with CUDA
events on the canonical shape, so we know where the time actually goes before
optimizing. Mirrors the launch sequence in aiter/ops/flydsl/fmha_bwd_kernels.py.

Run inside docker:
  HIP_VISIBLE_DEVICES=0 python op_tests/flydsl_tests/profile_flydsl_bwd.py
"""

import math
import os
import torch
from aiter.ops.triton.attention.mha import flash_attn_func
from aiter.ops.flydsl.kernels.fmha_bwd_preprocess import (
    build_fmha_bwd_preprocess_module,
)
from aiter.ops.flydsl.kernels.fmha_bwd_kernel import build_fmha_bwd_kernel_module

# ── shape ───────────────────────────────────────────────────────────────────
# Env overrides let rocprof run a small, fast, counter-attributable pass:
#   PROF_SQ=8192 PROF_ITERS=2 PROF_WARMUP=1 rocprofv3 ... -- python <this>
B, HQ, HK, D = 1, 5, 5, 128
SQ = SK = int(
    os.environ.get("PROF_SQ", 75648)
)  # canonical shape rounded to a mult of 64
DTYPE = torch.bfloat16
SM_SCALE = 1.0 / math.sqrt(D)
WARMUP = int(os.environ.get("PROF_WARMUP", 5))
ITERS = int(os.environ.get("PROF_ITERS", 20))

# 4 of the 5 GEMMs are [S,S]@[S,D]; the full backward is ~5. Report per-kernel
# ms plus the whole-pipeline TFLOPS on the standard 5-GEMM count for context.
FLOPS_5 = 5 * 2 * B * HQ * SQ * SK * D


def _bhs(t):  # BSHD -> (batch, head, seq) strides
    return int(t.stride(0)), int(t.stride(2)), int(t.stride(1))


def _bhsd(t):  # BSHD -> (batch, head, seq, dim) strides (preprocess wants 4)
    return int(t.stride(0)), int(t.stride(2)), int(t.stride(1)), int(t.stride(3))


def time_kernel(name, fn, flops=None):
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(ITERS):
        fn()
    end.record()
    torch.cuda.synchronize()
    ms = start.elapsed_time(end) / ITERS
    extra = f"   {flops / (ms * 1e-3) / 1e12:8.1f} TFLOPS" if flops else ""
    print(f"  {name:26s} {ms:8.3f} ms{extra}")
    return ms


def main():
    torch.manual_seed(42)
    q = torch.randn(B, SQ, HQ, D, dtype=DTYPE, device="cuda", requires_grad=True)
    k = torch.randn(B, SK, HK, D, dtype=DTYPE, device="cuda")
    v = torch.randn(B, SK, HK, D, dtype=DTYPE, device="cuda")
    out, lse = flash_attn_func(
        q, k, v, softmax_scale=SM_SCALE, causal=False, return_lse=True
    )
    do = torch.randn_like(out)

    q_ = q.detach().contiguous()
    k_ = k.contiguous()
    v_ = v.contiguous()
    out_ = out.detach().contiguous()
    do_ = do.contiguous()
    lse_ = lse.contiguous()

    dq_f32 = torch.zeros(B, SQ, HQ, D, dtype=torch.float32, device="cuda")
    dk_f32 = torch.zeros(B, SK, HK, D, dtype=torch.float32, device="cuda")
    dv_f32 = torch.zeros(B, SK, HK, D, dtype=torch.float32, device="cuda")
    delta = torch.zeros(B, HQ, SQ, dtype=torch.float32, device="cuda")

    prep = build_fmha_bwd_preprocess_module(head_dim=D, dtype="bf16")
    dkdv = build_fmha_bwd_kernel_module(head_dim=D, block_m=16, dtype="bf16")

    def run_prep():
        prep(
            out_,
            do_,
            delta,
            *_bhsd(out_),
            *_bhsd(do_),
            int(delta.stride(0)),
            int(delta.stride(1)),
            int(delta.stride(2)),
            SQ,
            B,
            HQ,
        )

    def run_dkdv():
        dkdv(
            q_,
            k_,
            v_,
            do_,
            dq_f32,
            dk_f32,
            dv_f32,
            lse_,
            delta,
            SM_SCALE,
            *_bhs(q_),
            *_bhs(k_),
            *_bhs(v_),
            *_bhs(do_),
            *_bhs(dq_f32),
            *_bhs(dk_f32),
            *_bhs(dv_f32),
            int(lse_.stride(0)),
            int(lse_.stride(1)),
            int(lse_.stride(2)),
            int(delta.stride(0)),
            int(delta.stride(1)),
            int(delta.stride(2)),
            SQ,
            SK,
            HQ,
            B,
        )

    print(f"Shape: B={B} HQ={HQ} Sq={SQ} D={D} {DTYPE}")
    print(f"{'':26s} {'ms':>10}")
    print("-" * 52)
    t_prep = time_kernel("1. preprocess (delta)", run_prep)
    t_dkdv = time_kernel("2. dQ/dK/dV (fused)", run_dkdv)
    total = t_prep + t_dkdv
    print("-" * 52)
    print(
        f"  {'sum of 2 kernels':26s} {total:8.3f} ms   "
        f"{FLOPS_5 / (total * 1e-3) / 1e12:8.1f} TFLOPS"
    )
    print(f"  breakdown: prep {t_prep/total:.0%}  dQ/dK/dV {t_dkdv/total:.0%}")
    print("  CK target ~836 TFLOPS")


if __name__ == "__main__":
    main()
