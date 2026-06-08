# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Does ``per_1x32_f4_quant_hip`` (HIP/asm path, quant.py:555) reproduce the same
MXFP4 result as the torch reference (``per_1x32_f4_quant``, quant.py:89) and the
Triton path (``per_1x32_f4_quant_triton``, quant.py:616)?

For each shape/dtype we run all three and compare two ways:

  1. Bit-pattern: packed fp4 payload (uint8) and e8m0 block scale (uint8).
     These are NOT expected to be bit-identical -- the three impls round the
     block scale slightly differently (torch rounds at the 0.5 mantissa bit,
     others differ). The stable contract is that the e8m0 exponent never differs
     by more than ONE step.

  2. Dequantized value: decode fp4*scale back to f32 and compare. This is the
     "same result" metric that actually matters for downstream GEMM accuracy --
     we report max|abs diff| between impls and, against the original x, the
     mean-squared error of each impl (so we can see HIP is no worse than triton).

Run on the GPU box:
    python op_tests/check_per_1x32_f4_quant_hip_equiv.py
    python op_tests/check_per_1x32_f4_quant_hip_equiv.py -m 256 -n 7168
"""

import argparse
import os
import sys

os.environ.setdefault("AITER_USE_SYSTEM_TRITON", "1")

import torch
from aiter import dtypes
from aiter.ops.quant import (
    per_1x32_f4_quant,
    per_1x32_f4_quant_triton,
    per_1x32_f4_quant_hip,
)
from aiter.utility.fp4_utils import mxfp4_to_f32, e8m0_to_f32

BLOCK = 32


def _u8(t):
    return t.view(torch.uint8).contiguous()


def _dequant(y, scale, N):
    """Decode (packed fp4 payload, e8m0 scale) -> f32 of shape (M, N)."""
    vals = mxfp4_to_f32(_u8(y))  # (M, N)
    sc = e8m0_to_f32(_u8(scale))  # (M, N/32)
    sc = sc.repeat_interleave(BLOCK, dim=-1)  # (M, N)
    return vals[..., :N].float() * sc[..., :N].float()


def _scale_exp_delta(sa, sb):
    a, b = _u8(sa).int(), _u8(sb).int()
    if a.shape != b.shape:
        return None
    return int((a - b).abs().max().item())


def compare(M, N, dtype, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn((M, N), generator=g, device="cuda", dtype=dtype)

    # shuffle=False so the scale layout is directly comparable across impls.
    y_ref, s_ref = per_1x32_f4_quant(x, quant_dtype=dtypes.fp4x2, shuffle=False)
    y_tri, s_tri = per_1x32_f4_quant_triton(x, quant_dtype=dtypes.fp4x2, shuffle=False)
    y_hip, s_hip = per_1x32_f4_quant_hip(x, quant_dtype=dtypes.fp4x2, shuffle=False)

    # ---- bit-pattern agreement (HIP vs the other two) ----
    y_hip_u8 = _u8(y_hip)
    y_eq_ref = y_hip_u8.shape == _u8(y_ref).shape and torch.equal(y_hip_u8, _u8(y_ref))
    y_eq_tri = y_hip_u8.shape == _u8(y_tri).shape and torch.equal(y_hip_u8, _u8(y_tri))
    de_ref = _scale_exp_delta(s_hip, s_ref)
    de_tri = _scale_exp_delta(s_hip, s_tri)

    # ---- dequantized agreement + accuracy vs original x ----
    xf = x.float()
    d_ref = _dequant(y_ref, s_ref, N)
    d_tri = _dequant(y_tri, s_tri, N)
    d_hip = _dequant(y_hip, s_hip, N)

    max_abs_hip_ref = (d_hip - d_ref).abs().max().item()
    max_abs_hip_tri = (d_hip - d_tri).abs().max().item()

    def _mse(d):
        return (d - xf).pow(2).mean().item()

    mse_ref, mse_tri, mse_hip = _mse(d_ref), _mse(d_tri), _mse(d_hip)

    # Contract: HIP's e8m0 scale is within 1 exponent step of both references,
    # and its quantization error is no worse than the torch reference's (allow a
    # tiny 5% slack for the different rounding mode).
    scale_bounded = (de_ref is not None and de_ref <= 1) and (
        de_tri is not None and de_tri <= 1
    )
    acc_ok = mse_hip <= mse_ref * 1.05 + 1e-12
    ok = scale_bounded and acc_ok

    bitnote = []
    bitnote.append("y==ref" if y_eq_ref else "y!=ref")
    bitnote.append("y==tri" if y_eq_tri else "y!=tri")

    print(
        f"[{'PASS' if ok else 'FAIL'}] {str(dtype):14} {M:>5}x{N:<6} | "
        f"{' '.join(bitnote)} | exp|d| ref={de_ref} tri={de_tri} | "
        f"deq max|d| ref={max_abs_hip_ref:.3e} tri={max_abs_hip_tri:.3e} | "
        f"mse hip={mse_hip:.3e} ref={mse_ref:.3e} tri={mse_tri:.3e}"
    )
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--rows", type=int, action="append", default=None)
    ap.add_argument("-n", "--cols", type=int, action="append", default=None)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("needs CUDA/ROCm GPU")

    # The HIP fp4x2 kernel (csrc/kernels/quant_kernels.cu) is guarded by
    # `#if defined(__Float4_e2m1fn_x2)` -- the native fp4 type only exists on
    # gfx950/gfx12. On other archs the branch is compiled out and the kernel
    # AITER_CHECK(false)s (process abort), so guard before calling it.
    arch = torch.cuda.get_device_properties(0).gcnArchName
    if not any(arch.startswith(a) for a in ("gfx95", "gfx12")):
        print(
            f"[SKIP] device arch {arch!r} has no HIP fp4x2 kernel "
            f"(needs gfx950/gfx12; the fp4 branch is #if'd out elsewhere).\n"
            f"per_1x32_f4_quant_hip cannot reproduce the result on this box; "
            f"run on a gfx950/gfx12 GPU to compare."
        )
        sys.exit(0)

    if args.rows and args.cols:
        shapes = [(m, n) for m in args.rows for n in args.cols]
    else:
        shapes = [(64, 4096), (160, 2880), (1, 4096), (256, 512), (4096, 7168)]

    print(
        "per_1x32_f4_quant_hip vs torch ref (per_1x32_f4_quant) and triton\n"
        "(per_1x32_f4_quant_triton). PASS = e8m0 scale within 1 exp step of both\n"
        "AND dequant MSE no worse than torch ref.\n"
    )
    all_ok = True
    for dtype in (torch.bfloat16, torch.float16):
        for i, (M, N) in enumerate(shapes):
            all_ok = compare(M, N, dtype, seed=i) and all_ok
    print()
    print("ALL PASS" if all_ok else "SOME FAILED")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
