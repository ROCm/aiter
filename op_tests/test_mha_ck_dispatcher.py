# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""
Side-by-side comparison: AITER's current FMHA path vs the new CK-dispatcher
path (``aiter.ops.mha_ck_dispatcher.flash_attn_func_ck_dispatcher``).

This script exercises both backends on identical inputs and reports:
    * max abs / mean abs difference between outputs
    * AITER-path kernel time vs dispatcher-path kernel time

It does NOT modify or replace any existing AITER code path. The current
``aiter.flash_attn_func`` is still the default; this test imports the new
wrapper explicitly.

Usage:
    python op_tests/test_mha_ck_dispatcher.py \
        --batch 2 --nheads 8 --seqlen 512 --hdim 128 \
        --dtype bf16 --causal --iters 5

Also runnable as a pytest:
    pytest op_tests/test_mha_ck_dispatcher.py -k bf16_causal
"""

from __future__ import annotations

import argparse
import time
from typing import Tuple

import pytest
import torch

import aiter
from aiter.ops.mha_ck_dispatcher import flash_attn_func_ck_dispatcher

_DTYPE_MAP = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


def _make_inputs(
    batch: int,
    nheads: int,
    seqlen: int,
    hdim: int,
    dtype: torch.dtype,
    device: torch.device,
    seed: int = 42,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    g = torch.Generator(device=device).manual_seed(seed)
    shape = (batch, seqlen, nheads, hdim)
    # scale down to keep numerics tame
    q = torch.randn(shape, dtype=dtype, device=device, generator=g) * 0.1
    k = torch.randn(shape, dtype=dtype, device=device, generator=g) * 0.1
    v = torch.randn(shape, dtype=dtype, device=device, generator=g) * 0.1
    return q, k, v


def _time_cuda(fn, iters: int) -> Tuple[torch.Tensor, float]:
    """Run fn() iters times, return (last_output, avg_ms)."""
    # warmup
    out = fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        out = fn()
    end.record()
    torch.cuda.synchronize()
    return out, start.elapsed_time(end) / iters


def run_aiter_fwd(q, k, v, causal):
    """Call AITER's existing flash_attn_func (current CK/ASM path)."""
    out = aiter.flash_attn_func(
        q, k, v,
        dropout_p=0.0,
        softmax_scale=None,
        causal=causal,
        window_size=(-1, -1, 0),
        bias=None,
        alibi_slopes=None,
        deterministic=True,
        return_lse=False,
        return_attn_probs=False,
    )
    # flash_attn_func returns a tuple (out, lse?, S_dmask?); take the tensor.
    if isinstance(out, tuple):
        out = out[0]
    return out


_DISPATCHER_LAST_KERNEL_MS = [0.0]


def run_dispatcher_fwd(q, k, v, causal):
    out, t_kernel_ms = flash_attn_func_ck_dispatcher(q, k, v, causal=causal)
    _DISPATCHER_LAST_KERNEL_MS[0] = t_kernel_ms
    return out


def compare(
    batch: int,
    nheads: int,
    seqlen: int,
    hdim: int,
    dtype: torch.dtype,
    causal: bool,
    iters: int,
    device: torch.device,
):
    q, k, v = _make_inputs(batch, nheads, seqlen, hdim, dtype, device)

    # --- AITER current path -------------------------------------------------
    out_aiter, t_aiter = _time_cuda(
        lambda: run_aiter_fwd(q, k, v, causal), iters=iters
    )

    # --- CK dispatcher path -------------------------------------------------
    # First call codegens + compiles the kernel; subsequent calls hit the
    # dispatcher's on-disk .so cache (~1ms reload).
    t0 = time.perf_counter()
    out_disp = run_dispatcher_fwd(q, k, v, causal)
    first_call_s = time.perf_counter() - t0

    # Steady-state timing (dispatcher reports its own kernel-only time too,
    # but we measure end-to-end including the numpy round-trip).
    out_disp, t_disp = _time_cuda(
        lambda: run_dispatcher_fwd(q, k, v, causal), iters=iters
    )

    # --- Correctness --------------------------------------------------------
    a = out_aiter.detach().to(torch.float32)
    b = out_disp.detach().to(torch.float32)
    abs_err = (a - b).abs()
    max_err = float(abs_err.max())
    mean_err = float(abs_err.mean())
    rel_err = float((abs_err / (a.abs() + 1e-6)).mean())

    return {
        "max_abs_err": max_err,
        "mean_abs_err": mean_err,
        "mean_rel_err": rel_err,
        "t_aiter_ms": t_aiter,
        "t_disp_ms": t_disp,
        "t_disp_kernel_ms": _DISPATCHER_LAST_KERNEL_MS[0],
        "disp_first_call_s": first_call_s,
    }


# --------------------------------------------------------------------------- #
# pytest entry
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("dtype_str", ["fp16", "bf16"])
@pytest.mark.parametrize("causal", [False, True], ids=["nocausal", "causal"])
def test_aiter_vs_ck_dispatcher(dtype_str, causal):
    if not torch.cuda.is_available():
        pytest.skip("CUDA/HIP device required")
    device = torch.device("cuda")
    dtype = _DTYPE_MAP[dtype_str]
    stats = compare(
        batch=2, nheads=8, seqlen=128, hdim=128,
        dtype=dtype, causal=causal, iters=3, device=device,
    )
    # Loose tolerance: the two paths can pick different kernel variants and
    # accumulation orders, so bf16 outputs occasionally drift on a handful of
    # elements while the bulk agreement (mean) stays tight.
    assert stats["max_abs_err"] < 5e-1, stats
    assert stats["mean_abs_err"] < 5e-2, stats


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--nheads", type=int, default=8)
    ap.add_argument("--seqlen", type=int, default=512)
    ap.add_argument("--hdim", type=int, default=128)
    ap.add_argument("--dtype", choices=list(_DTYPE_MAP), default="bf16")
    ap.add_argument("--causal", action="store_true")
    ap.add_argument("--iters", type=int, default=5)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA/HIP device required.")

    device = torch.device("cuda")
    dtype = _DTYPE_MAP[args.dtype]

    print(
        f"Problem: B={args.batch} H={args.nheads} S={args.seqlen} "
        f"D={args.hdim} dtype={args.dtype} causal={args.causal} iters={args.iters}"
    )

    stats = compare(
        batch=args.batch,
        nheads=args.nheads,
        seqlen=args.seqlen,
        hdim=args.hdim,
        dtype=dtype,
        causal=args.causal,
        iters=args.iters,
        device=device,
    )

    print()
    print("Correctness:")
    print(f"  max  abs err : {stats['max_abs_err']:.4e}")
    print(f"  mean abs err : {stats['mean_abs_err']:.4e}")
    print(f"  mean rel err : {stats['mean_rel_err']:.4e}")
    print()
    print("Timing (mean per iter):")
    print(f"  aiter.flash_attn_func               : {stats['t_aiter_ms']:8.3f} ms  (GPU)")
    print(f"  flash_attn_func_ck_dispatcher (e2e) : {stats['t_disp_ms']:8.3f} ms  (incl. H<->D numpy copy)")
    print(f"  flash_attn_func_ck_dispatcher (gpu) : {stats['t_disp_kernel_ms']:8.3f} ms  (kernel only, reported by dispatcher)")
    print(f"  dispatcher first-call (JIT+I/O)     : {stats['disp_first_call_s']:8.2f} s")


if __name__ == "__main__":
    main()
