# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Accuracy + perf test for the submission-ready flydsl FP8 einsum kernel,
exercising the LOW-LEVEL compile factories directly (explicit tiles / split_k),
unlike op_tests/test_fp8_einsum.py which drives the auto-dispatched entry point.

Covers the four modes of `aiter.ops.flydsl.kernels.fp8_einsum_ready`:
  - bf16 output, non-split-K        (compile_fp8_einsum_clean_ue8m0)
  - bf16 output, split-K            (compile_fp8_einsum_clean_ue8m0, split_k>1)
  - fp8 D-128 group-quant output    (compile_fp8_einsum_clean_ue8m0_qz)
  - fp8 group-quant, split-K        (compile_fp8_einsum_clean_ue8m0_qz_splitk)

Accuracy uses the DeepGEMM-style cosine error `calc_diff` against the fp32
dequantized reference einsum (the kernel's own fp8 input quant is part of its
contract). qz paths add a second fp8 cast at the output -> 2e-3 tolerance vs
1e-3 for bf16. Every check is NaN-aware. Split-K configs additionally report
run-to-run stability (non-split-K is bit-exact; split-K reduces via a
non-associative global atomic add, so it is stable to a tiny cosine tolerance,
not bit-exact).

Follows the style of op_tests/test_fp8_einsum.py:
  - uses aiter.test_common.benchmark / run_perftest
  - argparse-driven sweep, pandas markdown summary at the end
  - runs as a script: `python op_tests/test_fp8_einsum_ready.py [--mode ...]`
"""

import argparse

import pandas as pd
import torch

import aiter
from aiter.test_common import benchmark, run_perftest

from aiter.ops.shuffle import shuffle_weight
from aiter.ops.flydsl.kernels.fp8_einsum_ready import (
    compile_fp8_einsum_clean_ue8m0,
    compile_fp8_einsum_clean_ue8m0_qz,
    compile_fp8_einsum_clean_ue8m0_qz_splitk,
)

torch.set_default_device("cuda")

# Reference shape used throughout the sweep (V4-Pro wo_a grouped LoRA tp2).
H, D, R = 8, 1024, 4096


# ─────────────────────────────────────────────────────────────────────────────
# Packed-UE8M0 input construction (self-contained — no perf-dir dependency).
# ─────────────────────────────────────────────────────────────────────────────
def _per_128k_amax(t, k_dim):
    shape = list(t.shape)
    nk = shape[k_dim] // 128
    new_shape = shape[:k_dim] + [nk, 128] + shape[k_dim + 1 :]
    return t.float().reshape(new_shape).abs().amax(dim=k_dim + 1)


def _fp32_to_ue8m0_byte(s):
    s = s.clamp_min(2.0**-126).float()
    m, e = torch.frexp(s)
    e = e.to(torch.int32)
    byte = torch.where(m == 0.5, e - 1 + 127, e + 127)
    return byte.clamp(0, 254).to(torch.int32)


def _pack_ue8m0(b):
    *lead, nk = b.shape
    b = b.to(torch.int32).reshape(*lead, nk // 4, 4)
    return (
        (
            (b[..., 0] & 0xFF)
            | ((b[..., 1] & 0xFF) << 8)
            | ((b[..., 2] & 0xFF) << 16)
            | ((b[..., 3] & 0xFF) << 24)
        )
        .contiguous()
        .to(torch.int32)
    )


def build_inputs(H, D, R, B, dev, seed=0):
    """Return kernel-ABI inputs + the fp32 dequantized reference einsum.

    The reference dequantizes the *fp8-rounded* tensors (so it matches what the
    kernel actually consumes), then computes the einsum in fp32.
    """
    torch.manual_seed(seed)
    x_bf16 = torch.randn(B, H, R, dtype=torch.bfloat16, device=dev)
    y_bf16 = torch.randn(H, D, R, dtype=torch.bfloat16, device=dev)

    sx_amax = _per_128k_amax(x_bf16, 2)
    sx_byte = _fp32_to_ue8m0_byte((sx_amax / 448.0).clamp_min(2.0**-126))
    sx_pow2 = torch.ldexp(torch.ones_like(sx_amax), sx_byte - 127)
    x_f32 = x_bf16.float().view(B, H, R // 128, 128)
    x_fp8 = (
        (x_f32 / sx_pow2.unsqueeze(-1))
        .clamp(-448, 448)
        .to(torch.float8_e4m3fn)
        .view(B, H, R)
    )
    sx_i32 = _pack_ue8m0(sx_byte)
    x_ref = (x_fp8.float().view(B, H, R // 128, 128) * sx_pow2.unsqueeze(-1)).view(
        B, H, R
    )

    sy_amax = y_bf16.float().view(H, D // 128, 128, R // 128, 128).abs().amax((2, 4))
    sy_byte = _fp32_to_ue8m0_byte((sy_amax / 448.0).clamp_min(2.0**-126))
    sy_pow2 = torch.ldexp(torch.ones_like(sy_amax), sy_byte - 127)
    y_blk = y_bf16.float().view(H, D // 128, 128, R // 128, 128)
    y_fp8 = (
        (y_blk / sy_pow2.unsqueeze(2).unsqueeze(-1))
        .clamp(-448, 448)
        .to(torch.float8_e4m3fn)
        .view(H, D, R)
    )
    y_pre = shuffle_weight(y_fp8, layout=(16, 32)).contiguous()
    sy_i32 = _pack_ue8m0(sy_byte)
    y_ref = (
        y_fp8.float().view(H, D // 128, 128, R // 128, 128)
        * sy_pow2.unsqueeze(2).unsqueeze(-1)
    ).view(H, D, R)

    ref = torch.einsum("bhr,hdr->bhd", x_ref, y_ref)
    return x_fp8, y_pre, sx_i32, sy_i32, ref


def calc_diff(x, y):
    """DeepGEMM-style cosine error: 1 - 2<x,y>/(<x,x>+<y,y>)."""
    x64, y64 = x.double(), y.double()
    denom = (x64 * x64 + y64 * y64).sum()
    return 0.0 if denom == 0 else float(1.0 - 2.0 * (x64 * y64).sum() / denom)


def _nan_count(z):
    return int(torch.isnan(z.float()).sum())


def _stream():
    return torch.cuda.current_stream().cuda_stream


def _stable(outs, tol=1e-3):
    """Run-to-run stability. Non-split-K is bit-exact; split-K reduces via a
    non-associative global atomic add (atomicAdds complete in
    hardware-nondeterministic order) so the bf16-rounded result can differ in
    the last bit between runs. Accept bit-identity OR all runs within a tiny
    cosine tolerance (>= the measured bf16-ULP noise <=~1e-4, << a real race)."""
    if all(torch.equal(outs[0], o) for o in outs[1:]):
        return True
    a0 = outs[0].float()
    return all(calc_diff(o.float(), a0) < tol for o in outs[1:])


def _tf(us, H, D, R, B):
    return (2 * B * H * D * R) / (us * 1e-6) / 1e12 if us > 0 else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Per-config tests (low-level factories, explicit tiles / split_k)
# ─────────────────────────────────────────────────────────────────────────────
@benchmark()
def test_bf16_nonsplitk(H, D, R, B, tm, tn, tk, bsw):
    """bf16 output, non-split-K — explicit tile."""
    x, y, sx, sy, ref = build_inputs(H, D, R, B, "cuda")
    launch = compile_fp8_einsum_clean_ue8m0(
        H=H, D=D, R=R, tile_m=tm, tile_n=tn, tile_k=tk, block_swizzle_n=bsw
    )
    z = torch.empty((B, H, D), dtype=torch.bfloat16, device="cuda")

    def run():
        launch(z, x, y, sx, sy, B, _stream())
        return z

    run()  # prime the flydsl JIT cache outside the profiled region
    z_kernel, kernel_us = run_perftest(run)

    cos_err = calc_diff(z_kernel.float(), ref)
    nan = _nan_count(z_kernel)
    ok = nan == 0 and cos_err < 1e-3
    aiter.logger.info(
        f"  bf16 t{tm}x{tn}x{tk}_bsw{bsw}: cos_err={cos_err:.3e} nan={nan} "
        f"{'PASS' if ok else 'FAIL'}"
    )
    return {
        "cfg": f"{tm}x{tn}x{tk}_bsw{bsw}",
        "cos_err": cos_err,
        "nan": nan,
        "pass": ok,
        "kernel_us": kernel_us,
        "TFLOPS": _tf(kernel_us, H, D, R, B),
    }


@benchmark()
def test_bf16_splitk(H, D, R, B, sk):
    """bf16 output, split-K — correctness + run-to-run stability."""
    x, y, sx, sy, ref = build_inputs(H, D, R, B, "cuda")
    launch = compile_fp8_einsum_clean_ue8m0(
        H=H, D=D, R=R, tile_m=128, tile_n=128, tile_k=128, split_k=sk
    )

    def run():
        z = torch.empty((B, H, D), dtype=torch.bfloat16, device="cuda")
        launch(z, x, y, sx, sy, B, _stream())
        return z

    run()
    outs = [run() for _ in range(4)]
    torch.cuda.synchronize()
    _, kernel_us = run_perftest(run)

    cos_err = calc_diff(outs[0].float(), ref)
    nan = _nan_count(outs[0])
    det = _stable(outs)
    ok = nan == 0 and cos_err < 1e-3 and det
    aiter.logger.info(
        f"  bf16-splitk sk{sk}: cos_err={cos_err:.3e} nan={nan} det={det} "
        f"{'PASS' if ok else 'FAIL'}"
    )
    return {
        "cfg": f"sk{sk}",
        "cos_err": cos_err,
        "nan": nan,
        "det": det,
        "pass": ok,
        "kernel_us": kernel_us,
        "TFLOPS": _tf(kernel_us, H, D, R, B),
    }


@benchmark()
def test_qz(H, D, R, B, tn):
    """fp8 D-128 group-quant output, non-split-K — explicit tile_n."""
    x, y, sx, sy, ref = build_inputs(H, D, R, B, "cuda")
    launch = compile_fp8_einsum_clean_ue8m0_qz(
        H=H, D=D, R=R, tile_m=128, tile_n=tn, tile_k=128
    )
    z = torch.empty((B, H, D), dtype=torch.float8_e4m3fn, device="cuda")
    sz = torch.empty((B, H, D // 128), dtype=torch.float32, device="cuda")

    def run():
        launch(z, sz, x, y, sx, sy, B, _stream())
        return z

    run()
    _, kernel_us = run_perftest(run)

    deq = z.float() * sz.unsqueeze(-1).repeat_interleave(128, -1).view(z.shape).float()
    cos_err = calc_diff(deq, ref)
    nan = _nan_count(z)
    sz_pos = bool((sz > 0).all().item())
    ok = nan == 0 and cos_err < 2e-3 and sz_pos
    aiter.logger.info(
        f"  qz tn{tn}: cos_err={cos_err:.3e} nan={nan} sz_pos={sz_pos} "
        f"{'PASS' if ok else 'FAIL'}"
    )
    return {
        "cfg": f"tn{tn}",
        "cos_err": cos_err,
        "nan": nan,
        "sz_pos": sz_pos,
        "pass": ok,
        "kernel_us": kernel_us,
        "TFLOPS": _tf(kernel_us, H, D, R, B),
    }


@benchmark()
def test_qz_splitk(H, D, R, B, sk):
    """fp8 group-quant output, split-K (2-pass) — correctness + stability."""
    x, y, sx, sy, ref = build_inputs(H, D, R, B, "cuda")
    launch = compile_fp8_einsum_clean_ue8m0_qz_splitk(
        H=H, D=D, R=R, tile_m=128, tile_n=128, tile_k=128, split_k=sk
    )

    def run():
        return launch(x, y, sx, sy, B, _stream())

    run()
    fp8_outs, sz0, deq0 = [], None, None
    for i in range(3):
        z_fp8, sz = run()
        torch.cuda.synchronize()
        if i == 0:
            sz0 = sz
            deq0 = (
                z_fp8.float()
                * sz.unsqueeze(-1).repeat_interleave(128, -1).view(z_fp8.shape).float()
            )
        fp8_outs.append(z_fp8.clone())
    _, kernel_us = run_perftest(lambda: run()[0])

    cos_err = calc_diff(deq0, ref)
    nan = _nan_count(fp8_outs[0])
    det = _stable(fp8_outs)
    sz_pos = bool((sz0 > 0).all().item())
    ok = nan == 0 and cos_err < 2e-3 and det and sz_pos
    aiter.logger.info(
        f"  qz-splitk sk{sk}: cos_err={cos_err:.3e} nan={nan} det={det} "
        f"sz_pos={sz_pos} {'PASS' if ok else 'FAIL'}"
    )
    return {
        "cfg": f"sk{sk}",
        "cos_err": cos_err,
        "nan": nan,
        "det": det,
        "sz_pos": sz_pos,
        "pass": ok,
        "kernel_us": kernel_us,
        "TFLOPS": _tf(kernel_us, H, D, R, B),
    }


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
# bf16 non-split-K tile configs to sweep (the kernel's reg-B MFMA layout).
_BF16_TILES = [
    (128, 128, 128, 0),
    (128, 128, 128, 4),
    (128, 256, 128, 0),
    (256, 256, 128, 4),
    (64, 128, 256, 0),
    (128, 128, 256, 0),
    (256, 128, 128, 0),
]

parser = argparse.ArgumentParser(
    formatter_class=argparse.RawTextHelpFormatter,
    description="Accuracy + perf for the low-level flydsl fp8 einsum factories.",
)
parser.add_argument(
    "--mode",
    type=str,
    choices=["bf16", "bf16_splitk", "qz", "qz_splitk", "all"],
    default="all",
    help="Which kernel mode to test.",
)
args = parser.parse_args()


def _summary(title, rows):
    df = pd.DataFrame(rows)
    aiter.logger.info("%s (markdown):\n%s", title, df.to_markdown(index=False))
    return int((~df["pass"]).sum())


n_fail = 0

# ── bf16 non-split-K (tile sweep) ──
if args.mode in ("bf16", "all"):
    rows = []
    for B in (4096, 16384):
        for tm, tn, tk, bsw in _BF16_TILES:
            rows.append(
                {
                    "B": B,
                    **test_bf16_nonsplitk(
                        H=H, D=D, R=R, B=B, tm=tm, tn=tn, tk=tk, bsw=bsw
                    ),
                }
            )
    n_fail += _summary("fp8_einsum_ready bf16 non-split-K", rows)

# ── bf16 split-K (determinism) ──
if args.mode in ("bf16_splitk", "all"):
    rows = []
    for B in (512, 4096, 16384):
        for sk in (2, 4, 8):
            rows.append({"B": B, **test_bf16_splitk(H=H, D=D, R=R, B=B, sk=sk)})
    n_fail += _summary("fp8_einsum_ready bf16 split-K", rows)

# ── qz non-split-K ──
if args.mode in ("qz", "all"):
    rows = []
    for B in (4096,):
        for tn in (128, 256):
            rows.append({"B": B, **test_qz(H=H, D=D, R=R, B=B, tn=tn)})
    n_fail += _summary("fp8_einsum_ready qz", rows)

# ── qz split-K (determinism) ──
if args.mode in ("qz_splitk", "all"):
    rows = []
    for B in (512, 4096):
        for sk in (2, 4):
            rows.append({"B": B, **test_qz_splitk(H=H, D=D, R=R, B=B, sk=sk)})
    n_fail += _summary("fp8_einsum_ready qz split-K", rows)

aiter.logger.info("fp8_einsum_ready: %d failing config(s)", n_fail)
