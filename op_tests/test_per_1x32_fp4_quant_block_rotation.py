# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""Precision + performance comparison for per-1x32 MXFP4 quant with a
per-32-element block rotation, across 6 implementations:

  dsl_withrotation        FlyDSL MFMA-fused rotation + 1x32 fp4 quant
                          (rotation applied internally in f32).
  dsl_withrotation_trans  Same kernel, with R fed pre-transposed
                          (kernel runs in ``rot_transposed=True`` mode).
  dsl_withrotation_scalar FlyDSL scalar (non-MFMA) fused rotation +
                          1x32 fp4 quant (rotation done with vector
                          mul-acc per token; no 16x16x16 MFMA tiles).
                          Useful when M < 16, where MFMA tiles waste
                          lanes.
  dsl_withoutrotation     FlyDSL plain per-1x32 fp4 quant, fed
                          torch-rotated (bf16/fp16) input.
  hip                     HIP per-1x32 fp4 quant, fed torch-rotated input.
  triton                  Triton per-1x32 fp4 quant, fed torch-rotated input.

For each (m, n, dtype) row two views are reported:

  * QUANT-ONLY: time of the quant op alone (dsl_withrotation* includes
    their fused rotation; the other three got pre-rotated input so they
    are quant-only).
  * FULL PIPELINE: torch ``einsum`` rotation done INSIDE the timing loop
    for hip/triton/dsl_withoutrotation, so they are apples-to-apples
    with the fused dsl_withrotation paths.

Precision is reported vs an f32 reference (rotation done in f32, ideal
RNE-rounded E8M0 scale).

Usage::

    python op_tests/test_per_1x32_fp4_quant_block_rotation.py
    python op_tests/test_per_1x32_fp4_quant_block_rotation.py -m 1024 -n 4096 -d bf16
"""
import argparse
import itertools

import pandas as pd
import torch

import aiter
from aiter import dtypes, get_hip_quant, get_triton_quant
from aiter.ops.flydsl import (
    flydsl_per_1x32_fp4_quant,
    flydsl_per_1x32_fp4_quant_block_rotation,
    flydsl_per_1x32_fp4_quant_block_rotation_mfma,
)
from aiter.test_common import benchmark, run_perftest

torch.set_default_device("cuda")


# ---------- f32 reference helpers --------------------------------------------
def _apply_block_rotation_f32(x: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    """y[m, b*g + h] = sum_g x[m, b*g + g] * R[b, h, g], done in f32."""
    g = R.shape[-1]
    m, n = x.shape
    B = n // g
    y = torch.einsum("mbg,bhg->mbh", x.float().reshape(m, B, g), R.float())
    return y.reshape(m, n)


def _ref_e8m0_scale_from_amax(amax_f32: torch.Tensor) -> torch.Tensor:
    """Bit-exact replica of the kernel's RNE-pow2 -> E8M0 byte calculation."""
    u = amax_f32.view(torch.int32)
    exp = (u >> 23) & 0xFF
    bit22 = (u >> 22) & 1
    bit21 = (u >> 21) & 1
    lo21 = u & 0x1FFFFF
    round_up = (bit22 != 0) & ((bit21 != 0) | (lo21 != 0) | (exp != 0))
    exp_rounded = exp + round_up.to(torch.int32)
    is_inf_nan = exp == 0xFF
    exp_final = torch.where(
        is_inf_nan,
        torch.tensor(0xFF, dtype=torch.int32, device=exp.device),
        exp_rounded,
    )
    next_pow2 = (exp_final << 23).view(torch.float32)
    inv_scale = next_pow2 * 0.25
    return ((inv_scale.view(torch.int32) >> 23) & 0xFF).to(torch.uint8)


_FP4_LUT = torch.tensor(
    [
        0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
        -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
    ],
    dtype=torch.float32,
)


def _dequant_mxfp4(y_packed: torch.Tensor, scale_e8m0: torch.Tensor):
    lut = _FP4_LUT.to(y_packed.device)
    m, half = y_packed.shape
    n = half * 2
    g, B = 32, n // 32
    yb = y_packed.to(torch.int32)
    lo = yb & 0x0F
    hi = (yb >> 4) & 0x0F
    vals = torch.stack([lo, hi], dim=-1).reshape(m, n).long()
    deq = lut[vals]
    scale_f = torch.pow(
        torch.tensor(2.0, device=scale_e8m0.device),
        (scale_e8m0.to(torch.int32) - 127).float(),
    )
    return deq * scale_f.unsqueeze(-1).expand(-1, -1, g).reshape(m, n)


# ---------- impl wrappers ----------------------------------------------------
def _build_impls(m: int, n: int, dtype: torch.dtype, R: torch.Tensor,
                 R_T: torch.Tensor, x_rot_bf16: torch.Tensor):
    """Return ``(quant_only_impls, full_pipeline_impls)`` dicts.

    The quant-only dict has the 5 impls timed against ``x`` directly:
    ``dsl_withrotation*`` consume raw ``x`` and apply the fused rotation;
    the other three are HARD-WIRED to use the pre-rotated input
    ``x_rot_bf16`` (so their ``x_in`` argument is ignored on purpose).

    The full-pipeline dict additionally does the torch einsum rotation
    INSIDE the timed function for the three quant-only impls, so they
    can be compared head-to-head with the fused dsl paths.
    """
    g, B = 32, n // 32

    def _dsl_withrotation(x_in, quant_dtype=dtypes.fp4x2, shuffle=False):
        y = torch.empty(m, n // 2, dtype=torch.uint8, device=x_in.device)
        s = torch.empty(m, B, dtype=torch.uint8, device=x_in.device)
        flydsl_per_1x32_fp4_quant_block_rotation_mfma(y, x_in, R, s)
        return y.view(dtypes.fp4x2), s.view(dtypes.fp8_e8m0)

    def _dsl_withrotation_trans(x_in, quant_dtype=dtypes.fp4x2, shuffle=False):
        y = torch.empty(m, n // 2, dtype=torch.uint8, device=x_in.device)
        s = torch.empty(m, B, dtype=torch.uint8, device=x_in.device)
        flydsl_per_1x32_fp4_quant_block_rotation_mfma(
            y, x_in, R_T, s, rot_transposed=True
        )
        return y.view(dtypes.fp4x2), s.view(dtypes.fp8_e8m0)

    def _dsl_withrotation_scalar(x_in, quant_dtype=dtypes.fp4x2, shuffle=False):
        y = torch.empty(m, n // 2, dtype=torch.uint8, device=x_in.device)
        s = torch.empty(m, B, dtype=torch.uint8, device=x_in.device)
        flydsl_per_1x32_fp4_quant_block_rotation(y, x_in, R, s)
        return y.view(dtypes.fp4x2), s.view(dtypes.fp8_e8m0)

    def _dsl_withoutrotation(x_in, quant_dtype=dtypes.fp4x2, shuffle=False):
        # x_in is ignored on purpose; we always feed the pre-rotated input
        # so this represents "quant-only cost on rotated data".
        y = torch.empty(m, n // 2, dtype=torch.uint8, device=x_rot_bf16.device)
        s = torch.empty(m, B, dtype=torch.uint8, device=x_rot_bf16.device)
        flydsl_per_1x32_fp4_quant(y, x_rot_bf16, s)
        return y.view(dtypes.fp4x2), s.view(dtypes.fp8_e8m0)

    def _hip(x_in, quant_dtype=dtypes.fp4x2, shuffle=False):
        return get_hip_quant(aiter.QuantType.per_1x32)(
            x_rot_bf16, quant_dtype=quant_dtype, shuffle=shuffle
        )

    def _triton(x_in, quant_dtype=dtypes.fp4x2, shuffle=False):
        return get_triton_quant(aiter.QuantType.per_1x32)(
            x_rot_bf16, quant_dtype=quant_dtype, shuffle=shuffle
        )

    quant_only = {
        "dsl_withrotation":        _dsl_withrotation,
        "dsl_withrotation_trans":  _dsl_withrotation_trans,
        "dsl_withrotation_scalar": _dsl_withrotation_scalar,
        "dsl_withoutrotation":     _dsl_withoutrotation,
        "hip":                     _hip,
        "triton":                  _triton,
    }

    # Full pipeline: torch rotation included for the 3 quant-only impls.
    def _rotate_in_loop(x_in):
        return (
            torch.einsum(
                "mbg,bhg->mbh",
                x_in.reshape(m, B, g).float(),
                R.float(),
            )
            .reshape(m, n)
            .to(dtype)
            .contiguous()
        )

    def _full_dsl_withoutrotation(x_in, quant_dtype=dtypes.fp4x2, shuffle=False):
        x_rot = _rotate_in_loop(x_in)
        y = torch.empty(m, n // 2, dtype=torch.uint8, device=x_rot.device)
        s = torch.empty(m, B, dtype=torch.uint8, device=x_rot.device)
        flydsl_per_1x32_fp4_quant(y, x_rot, s)
        return y.view(dtypes.fp4x2), s.view(dtypes.fp8_e8m0)

    def _full_hip(x_in, quant_dtype=dtypes.fp4x2, shuffle=False):
        x_rot = _rotate_in_loop(x_in)
        return get_hip_quant(aiter.QuantType.per_1x32)(
            x_rot, quant_dtype=quant_dtype, shuffle=shuffle
        )

    def _full_triton(x_in, quant_dtype=dtypes.fp4x2, shuffle=False):
        x_rot = _rotate_in_loop(x_in)
        return get_triton_quant(aiter.QuantType.per_1x32)(
            x_rot, quant_dtype=quant_dtype, shuffle=shuffle
        )

    full_pipeline = {
        "dsl_withrotation":        _dsl_withrotation,
        "dsl_withrotation_trans":  _dsl_withrotation_trans,
        "dsl_withrotation_scalar": _dsl_withrotation_scalar,
        "dsl_withoutrotation":     _full_dsl_withoutrotation,
        "hip":                     _full_hip,
        "triton":                  _full_triton,
    }
    return quant_only, full_pipeline


# ---------- precision metrics ------------------------------------------------
def _precision(name, fp4, scale, ref_f32, s_ref):
    m, n = ref_f32.shape
    g = 32
    s_u8 = scale.view(torch.uint8)
    y_u8 = fp4.view(torch.uint8)
    s_diff = (s_ref.to(torch.int32) - s_u8.to(torch.int32)).abs()
    deq = _dequant_mxfp4(y_u8, s_u8)
    bucket = torch.pow(
        torch.tensor(2.0, device=s_u8.device),
        (s_u8.to(torch.int32) - 127).float(),
    ).unsqueeze(-1).expand(-1, -1, g).reshape(m, n)
    resid = (deq - ref_f32).abs()
    ref_amax = max(float(ref_f32.abs().max().item()), 1e-12)
    return {
        f"{name} s_mism%": float((s_diff != 0).float().mean().item() * 100),
        f"{name} s_ulp":   int(s_diff.max().item()),
        f"{name} resid":   float(resid.max().item() / ref_amax),
    }


# ---------- benchmark entry point --------------------------------------------
@benchmark()
def test_per_1x32_fp4_quant_block_rotation(m, n, dtype):
    torch.manual_seed(0)
    g = 32
    B = n // g
    x = torch.randn((m, n), dtype=dtype) * 5.0
    R = torch.randn(B, g, g, dtype=dtype)
    Q, _ = torch.linalg.qr(R.float())
    R = Q.to(dtype).contiguous()
    R_T = R.transpose(-1, -2).contiguous()

    # f32 reference + pre-rotated bf16 input (the inputs hip/triton expect).
    ref_f32 = _apply_block_rotation_f32(x, R)
    s_ref = _ref_e8m0_scale_from_amax(
        ref_f32.abs().reshape(m, B, g).amax(dim=-1)
    )
    x_rot_bf16 = ref_f32.to(dtype).contiguous()

    quant_only, full_pipeline = _build_impls(m, n, dtype, R, R_T, x_rot_bf16)

    ret = {}
    # Precision via quant-only callers (they produce the same fp4 output
    # regardless of whether the rotation is done in the loop or not).
    for name, fn in quant_only.items():
        fp4, scale = fn(x)
        ret.update(_precision(name, fp4, scale, ref_f32, s_ref))

    # Quant-only us
    for name, fn in quant_only.items():
        (_, _), us = run_perftest(
            fn, x, num_iters=200, num_warmup=20,
            quant_dtype=dtypes.fp4x2, shuffle=False,
        )
        ret[f"{name} us"] = us

    # Full-pipeline us (rotation done inside the timed loop).
    for name, fn in full_pipeline.items():
        (_, _), us = run_perftest(
            fn, x, num_iters=200, num_warmup=20,
            quant_dtype=dtypes.fp4x2, shuffle=False,
        )
        ret[f"{name} full_us"] = us
    return ret


# ---------- CLI --------------------------------------------------------------
parser = argparse.ArgumentParser(
    formatter_class=argparse.RawTextHelpFormatter,
    description=(
        "Precision + performance comparison of per-1x32 MXFP4 quant with "
        "per-32-element block rotation across DSL/HIP/Triton implementations."
    ),
)
parser.add_argument(
    "-d", "--dtype",
    type=dtypes.str2Dtype,
    nargs="*",
    default=[dtypes.d_dtypes["bf16"], dtypes.d_dtypes["fp16"]],
    help="Input dtype(s). e.g.: -d bf16",
)
parser.add_argument(
    "-n", "--n",
    type=int,
    nargs="*",
    default=[4096, 7168],
    help="N (hidden / cols). Must be a multiple of 32.",
)
parser.add_argument(
    "-m", "--m",
    type=int,
    nargs="*",
    default=[1, 4, 8, 16, 32, 64, 128, 1024, 16384],
    help="M (tokens). e.g.: -m 1024",
)


def main():
    args = parser.parse_args()
    for dtype in args.dtype:
        df = []
        for n in args.n:
            if n % 32:
                raise ValueError(f"n={n} is not a multiple of 32")
            for m in args.m:
                df.append(
                    test_per_1x32_fp4_quant_block_rotation(m, n, dtype)
                )
        df = pd.DataFrame(df)
        try:
            table = df.to_markdown(index=False)
        except ImportError:
            # ``tabulate`` not installed -> fall back to plain string format.
            table = df.to_string(index=False)
        aiter.logger.info(
            "per_1x32_fp4_quant_block_rotation summary (dtype=%s):\n%s",
            dtype, table,
        )


if __name__ == "__main__":
    main()
