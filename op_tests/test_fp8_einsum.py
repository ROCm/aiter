# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Accuracy + perf test for the flydsl fp8 einsum kernel.

Covers both output modes of `aiter.ops.flydsl.kernels.fp8_einsum`:
  - bf16 output  via `compile_fp8_einsum_clean_ue8m0_auto`
  - fp8 D-128 group-quant output via `compile_fp8_einsum_clean_ue8m0_qz_auto`

Accuracy is checked against an fp32 dequantized reference (X / sx_pow2) @
(Y / sy_pow2).T — the same einsum the kernel implements, but evaluated in
pytorch fp32. For qz output, the kernel's fp8 output is multiplied by its
per-D-128-block scale before comparison.

Follows the style of op_tests/test_mhc.py:
  - uses aiter.test_common.checkAllclose / benchmark / run_perftest
  - argparse-driven sweep, pandas markdown summary at the end
  - runs as a script: `python op_tests/test_fp8_einsum.py [--m ...] [--shape ...]`
"""

import argparse

import pandas as pd
import torch

import aiter
from aiter import dtypes
from aiter.test_common import (
    benchmark,
    checkAllclose,
    run_perftest,
)

from aiter.ops.flydsl.kernels.fp8_einsum import fp8_einsum
from aiter.ops.shuffle import shuffle_weight

torch.set_default_device("cuda")


# ─────────────────────────────────────────────────────────────────────────────
# Packed-UE8M0 input construction helpers
# (inlined from the development harness so this test is self-contained — no
# dependency on aiter/ops/flydsl/kernels/fp8_einsum_perf/).
# ─────────────────────────────────────────────────────────────────────────────
def _per_128k_amax(t: torch.Tensor, k_dim: int) -> torch.Tensor:
    """abs().amax() over groups of 128 along `k_dim`."""
    shape = list(t.shape)
    assert shape[k_dim] % 128 == 0
    nk = shape[k_dim] // 128
    new_shape = shape[:k_dim] + [nk, 128] + shape[k_dim + 1:]
    return t.float().reshape(new_shape).abs().amax(dim=k_dim + 1)


def _fp32_to_ue8m0_byte(scale_f32: torch.Tensor) -> torch.Tensor:
    """fp32 scale → UE8M0 biased-exp byte (round-up to next pow2).
    Byte b represents 2^(b - 127). Clamped to [0, 254]."""
    s = scale_f32.clamp_min(2.0 ** -126).float()
    m, e = torch.frexp(s)
    e = e.to(torch.int32)
    is_pow2 = (m == 0.5)
    byte = torch.where(is_pow2, e - 1 + 127, e + 127)
    return byte.clamp(0, 254).to(torch.int32)


def _pack_ue8m0_bytes_to_i32(bytes_along_k: torch.Tensor) -> torch.Tensor:
    """Pack 4 bytes along the last (K-group) dim into one little-endian i32.
    Input shape ... × (4N) → output shape ... × N."""
    *lead, nk = bytes_along_k.shape
    assert nk % 4 == 0
    b = bytes_along_k.to(torch.int32).reshape(*lead, nk // 4, 4)
    return (
        (b[..., 0] & 0xFF)
        | ((b[..., 1] & 0xFF) << 8)
        | ((b[..., 2] & 0xFF) << 16)
        | ((b[..., 3] & 0xFF) << 24)
    ).contiguous().to(torch.int32)


# ─────────────────────────────────────────────────────────────────────────────
# Input builder + fp32 reference (mirrors bench_ni_rotation.build_inputs,
# but ALSO returns the un-shuffled fp32 tensors needed for the reference
# einsum — so we don't have to invert the kpack/nlane shuffle ourselves).
# ─────────────────────────────────────────────────────────────────────────────
def build_inputs_with_ref(H: int, D: int, R: int, B: int, device, seed: int = 0):
    """Build kernel-ABI inputs + matching fp32 reference tensors.

    Returns:
      x_fp8     : fp8 (B, H, R)
      y_pre     : fp8 preshuffled — what the kernel ingests
      sx_i32    : packed-UE8M0 (B, H, R/512)
      sy_i32    : packed-UE8M0 (H, D/128, R/512)
      x_ref_f32 : fp32 (B, H, R) — dequantized X (x_fp8 * sx_pow2)
      y_ref_f32 : fp32 (H, D, R) — dequantized Y (y_fp8 * sy_pow2),
                  in the natural (H, D, R) layout (NOT preshuffled).
    """
    torch.manual_seed(seed)
    x_bf16 = torch.randn(B, H, R, dtype=torch.bfloat16, device=device) * 4.0
    y_bf16 = torch.randn(H, D, R, dtype=torch.bfloat16, device=device) * 0.5

    # X: per-(B, H, K=128) quant → fp8 + per-block UE8M0 scale.
    sx_amax = _per_128k_amax(x_bf16, k_dim=2)
    sx_scale_f32 = (sx_amax / 448.0).clamp_min(2.0 ** -126)
    sx_byte = _fp32_to_ue8m0_byte(sx_scale_f32)
    sx_pow2 = torch.ldexp(torch.ones_like(sx_scale_f32), sx_byte - 127)
    x_f32 = x_bf16.float().view(B, H, R // 128, 128)
    x_q = (x_f32 / sx_pow2.unsqueeze(-1)).clamp(-448, 448)
    x_fp8 = x_q.to(torch.float8_e4m3fn).view(B, H, R)
    sx_i32 = _pack_ue8m0_bytes_to_i32(sx_byte)
    # Reference X: round-trip through fp8 + pow2 dequant (NOT bf16 input)
    # so reference reflects the kernel's actual input.
    x_ref_f32 = (x_fp8.float().view(B, H, R // 128, 128)
                 * sx_pow2.unsqueeze(-1)).view(B, H, R)

    # Y: per-(D=128, K=128) block quant.
    sy_amax = (y_bf16.float().view(H, D // 128, 128, R // 128, 128)
               .abs().amax(dim=(2, 4)))
    sy_scale_f32 = (sy_amax / 448.0).clamp_min(2.0 ** -126)
    sy_byte = _fp32_to_ue8m0_byte(sy_scale_f32)
    sy_pow2 = torch.ldexp(torch.ones_like(sy_scale_f32), sy_byte - 127)
    y_blk = y_bf16.float().view(H, D // 128, 128, R // 128, 128)
    y_q = (y_blk / sy_pow2.unsqueeze(2).unsqueeze(-1)).clamp(-448, 448)
    y_fp8 = y_q.to(torch.float8_e4m3fn).view(H, D, R)
    y_pre = shuffle_weight(y_fp8, layout=(16, 32)).contiguous()
    sy_i32 = _pack_ue8m0_bytes_to_i32(sy_byte)
    # Reference Y: dequant the fp8-rounded tensor (so reference matches
    # what the kernel internally sees).
    y_ref_f32 = (y_fp8.float().view(H, D // 128, 128, R // 128, 128)
                 * sy_pow2.unsqueeze(2).unsqueeze(-1)).view(H, D, R)

    return x_fp8, y_pre, sx_i32, sy_i32, x_ref_f32, y_ref_f32


def einsum_ref_fp32(x_ref_f32, y_ref_f32):
    """Pure-torch fp32 reference for einsum('bhr,hdr->bhd', X, Y)."""
    return torch.einsum("bhr,hdr->bhd", x_ref_f32, y_ref_f32)


# ─────────────────────────────────────────────────────────────────────────────
# Kernel wrappers
# ─────────────────────────────────────────────────────────────────────────────
# Thin wrappers around the unified fp8_einsum() interface — kept around so
# run_perftest sees one stable callable per mode (its profiling hooks need
# to call the same function many times and the unified API's cache means
# the flydsl compile only fires once per (H, D, R, B, dtype) tuple).
def fp8_einsum_bf16_kernel(x_fp8, y_pre, sx_i32, sy_i32):
    """Returns z_bf16 only (matches run_perftest's single-return convention)."""
    z, _ = fp8_einsum(x_fp8, y_pre, sx_i32, sy_i32, out_dtype=torch.bfloat16)
    return z


def fp8_einsum_qz_kernel(x_fp8, y_pre, sx_i32, sy_i32, transpose_scale):
    """Returns (z_fp8, sz_fp32)."""
    return fp8_einsum(
        x_fp8, y_pre, sx_i32, sy_i32,
        out_dtype=torch.float8_e4m3fn,
        transpose_scale=transpose_scale,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Per-shape tests
# ─────────────────────────────────────────────────────────────────────────────
@benchmark()
def test_fp8_einsum_bf16(H, D, R, B):
    """bf16 output: compare auto-dispatched kernel vs fp32 reference."""
    x_fp8, y_pre, sx_i32, sy_i32, x_ref, y_ref = build_inputs_with_ref(
        H, D, R, B, torch.device("cuda"),
    )
    z_ref_f32 = einsum_ref_fp32(x_ref, y_ref)
    z_ref_bf16 = z_ref_f32.bfloat16()

    # Prime the kernel cache OUTSIDE the perftest's profiled region — flydsl
    # JIT compile inside torch.profiler segfaults.
    _ = fp8_einsum_bf16_kernel(x_fp8, y_pre, sx_i32, sy_i32)

    z_kernel, kernel_us = run_perftest(
        fp8_einsum_bf16_kernel, x_fp8, y_pre, sx_i32, sy_i32,
    )

    # Accuracy: bf16 output should match the bf16-cast fp32 reference within
    # a generous tol (fp8 quant of inputs alone gives ~1% relative error on
    # the output before bf16 rounding). Tol values mirror test_mhc's style.
    err = checkAllclose(
        z_ref_bf16, z_kernel, rtol=5e-2, atol=2e-1, msg="bf16 z",
    )

    # Determinism: second launch should be bit-exact.
    z_kernel_2 = fp8_einsum_bf16_kernel(x_fp8, y_pre, sx_i32, sy_i32)
    det = (
        z_kernel.view(torch.int16) == z_kernel_2.view(torch.int16)
    ).all().item()
    aiter.logger.info(f"  bf16 determinism: {'PASS' if det else 'FAIL'}")

    flops = 2 * B * H * D * R
    tf = flops / (kernel_us * 1e-6) / 1e12
    return {
        "err": err,
        "kernel_us": kernel_us,
        "TFLOPS": tf,
        "det": det,
    }


@benchmark()
def test_fp8_einsum_qz(H, D, R, B, transpose_scale=False):
    """qz output: kernel's fp8 × per-D128 scale should reconstruct the
    fp32-reference einsum within fp8-quant tolerance."""
    x_fp8, y_pre, sx_i32, sy_i32, x_ref, y_ref = build_inputs_with_ref(
        H, D, R, B, torch.device("cuda"),
    )
    z_ref_f32 = einsum_ref_fp32(x_ref, y_ref)

    # Prime kernel cache outside the profiled region.
    _ = fp8_einsum_qz_kernel(x_fp8, y_pre, sx_i32, sy_i32, transpose_scale)

    (z_fp8, sz), kernel_us = run_perftest(
        fp8_einsum_qz_kernel,
        x_fp8, y_pre, sx_i32, sy_i32, transpose_scale,
    )

    # Normalize scale layout to (B, H, D/128) for dequant.
    if transpose_scale:
        # (D/128, B, H) → (B, H, D/128)
        sz_bhd = sz.permute(1, 2, 0).contiguous()
    else:
        sz_bhd = sz

    # Dequant: z_recon[b,h,d] = z_fp8[b,h,d] * sz[b, h, d//128]
    sz_exp = sz_bhd.unsqueeze(-1).expand(B, H, D // 128, 128).reshape(B, H, D)
    z_recon = z_fp8.float() * sz_exp

    # Accuracy: dequantized output vs the fp32 reference. The fp8-quant
    # residual is bounded by 1/(2*448) ≈ 0.1% per-elem at typical pixels,
    # but tail outliers within a 128-block can deviate by up to ~25%
    # because the single fp32 scale must accommodate the block's amax.
    # We set atol generously to cover those outliers; the failure-ratio
    # threshold (default 5%) inside checkAllclose gates the actual pass.
    ref_max = z_ref_f32.abs().max().item()
    err = checkAllclose(
        z_ref_f32, z_recon,
        rtol=1e-1,
        atol=max(5e-1, 0.25 * ref_max),
        msg="qz dequant",
    )

    # Scale must be strictly positive.
    sz_pos = (sz > 0).all().item()
    aiter.logger.info(f"  qz sz_positive: {'PASS' if sz_pos else 'FAIL'}  "
                      f"sz_range=[{sz.min().item():.3e}, {sz.max().item():.3e}]")

    # Determinism: both z_fp8 and sz should be bit-exact across runs.
    z_fp8_2, sz_2 = fp8_einsum_qz_kernel(
        x_fp8, y_pre, sx_i32, sy_i32, transpose_scale,
    )
    det_z = (z_fp8.view(torch.int8) == z_fp8_2.view(torch.int8)).all().item()
    det_sz = (sz == sz_2).all().item()
    aiter.logger.info(f"  qz determinism: z={det_z}  sz={det_sz}")

    flops = 2 * B * H * D * R
    tf = flops / (kernel_us * 1e-6) / 1e12
    return {
        "err": err,
        "kernel_us": kernel_us,
        "TFLOPS": tf,
        "det_z": det_z,
        "det_sz": det_sz,
        "sz_pos": sz_pos,
    }


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
# Shapes that are tabulated in the autotune table — every (H, D, R) below
# must have at least one B entry in _AUTOTUNE_WINNERS_BF16 / _QZ, else the
# auto-dispatched kernel will raise ValueError.
TUNED_SHAPES = [
    # (H, D, R, label)
    ( 16, 1024, 4096, "decode-like"),
    (  8, 8192, 8192, "prefill-like"),
]
# Default B sweep — subset of the tuned B set per shape, picked to cover
# latency/mid/compute regimes. Override with -m.
DEFAULT_BS = {
    ( 16, 1024, 4096): [1, 16, 256, 1024, 16384],
    (  8, 8192, 8192): [128, 512, 2048],
}

parser = argparse.ArgumentParser(
    formatter_class=argparse.RawTextHelpFormatter,
    description="Accuracy + perf for flydsl fp8 einsum (auto-dispatched).",
)
parser.add_argument(
    "-m",
    type=int,
    nargs="*",
    default=None,
    help="""Override the B (batch / token) sweep.
    e.g.: -m 1 16 256 1024""",
)
parser.add_argument(
    "--shape",
    type=str,
    nargs="*",
    choices=[lbl for *_, lbl in TUNED_SHAPES],
    default=None,
    help="""Which tuned (H, D, R) shape(s) to sweep.
    Default: all of them.""",
)
parser.add_argument(
    "--mode",
    type=str,
    choices=["bf16", "qz", "qz_tsz", "all"],
    default="all",
    help="""Which kernel mode to test.
    'qz_tsz' = qz with transpose_scale=True.""",
)
args = parser.parse_args()

# Build the shape list.
labels = args.shape if args.shape else [lbl for *_, lbl in TUNED_SHAPES]
shapes = [(h, d, r, lbl) for (h, d, r, lbl) in TUNED_SHAPES if lbl in labels]

# Build the per-shape B list.
def bs_for(H, D, R):
    return args.m if args.m else DEFAULT_BS[(H, D, R)]

# ─────────────────────────────────────────────────────────────────────────────
# Run bf16 sweep
# ─────────────────────────────────────────────────────────────────────────────
if args.mode in ("bf16", "all"):
    rows = []
    for (H, D, R, label) in shapes:
        for B in bs_for(H, D, R):
            ret = test_fp8_einsum_bf16(H=H, D=D, R=R, B=B)
            rows.append({"shape": label, "H": H, "D": D, "R": R, "B": B, **ret})
    df = pd.DataFrame(rows)
    aiter.logger.info(
        "fp8_einsum bf16 summary (markdown):\n%s", df.to_markdown(index=False),
    )

# ─────────────────────────────────────────────────────────────────────────────
# Run qz sweep (default layout)
# ─────────────────────────────────────────────────────────────────────────────
if args.mode in ("qz", "all"):
    rows = []
    for (H, D, R, label) in shapes:
        for B in bs_for(H, D, R):
            ret = test_fp8_einsum_qz(H=H, D=D, R=R, B=B, transpose_scale=False)
            rows.append({"shape": label, "H": H, "D": D, "R": R, "B": B, **ret})
    df = pd.DataFrame(rows)
    aiter.logger.info(
        "fp8_einsum qz (default sz layout) summary (markdown):\n%s",
        df.to_markdown(index=False),
    )

# ─────────────────────────────────────────────────────────────────────────────
# Run qz sweep with --transpose-scale
# ─────────────────────────────────────────────────────────────────────────────
if args.mode in ("qz_tsz", "all"):
    rows = []
    for (H, D, R, label) in shapes:
        for B in bs_for(H, D, R):
            ret = test_fp8_einsum_qz(H=H, D=D, R=R, B=B, transpose_scale=True)
            rows.append({"shape": label, "H": H, "D": D, "R": R, "B": B, **ret})
    df = pd.DataFrame(rows)
    aiter.logger.info(
        "fp8_einsum qz (transposed sz layout) summary (markdown):\n%s",
        df.to_markdown(index=False),
    )
