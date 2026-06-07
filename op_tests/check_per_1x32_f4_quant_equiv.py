# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Check whether ``per_1x32_f4_quant`` (torch reference, quant.py:75) and
``per_1x32_f4_quant_triton`` (Triton kernel, quant.py:562) produce *identical*
output, so we can decide if the fused_moe.py:951 call can be swapped to the
Triton variant without changing numerics.

Compares both returned tensors:
  - y      : packed fp4 payload (viewed as uint8)
  - scale  : per-1x32 e8m0 block scale (viewed as uint8)

for several shapes / dtypes. Exits non-zero if any shape differs.

Run on the GPU box:
    python op_tests/check_per_1x32_f4_quant_equiv.py
"""

import os
import sys

os.environ.setdefault("AITER_USE_SYSTEM_TRITON", "1")

import torch
from aiter import dtypes
from aiter.ops.quant import per_1x32_f4_quant, per_1x32_f4_quant_triton


def _u8(t):
    return t.view(torch.uint8).contiguous()


def compare(M, N, dtype, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn((M, N), generator=g, device="cuda", dtype=dtype)

    y_ref, s_ref = per_1x32_f4_quant(x, quant_dtype=dtypes.fp4x2, shuffle=False)
    y_tri, s_tri = per_1x32_f4_quant_triton(x, quant_dtype=dtypes.fp4x2, shuffle=False)

    y_ref_u8, y_tri_u8 = _u8(y_ref), _u8(y_tri)
    s_ref_u8, s_tri_u8 = _u8(s_ref), _u8(s_tri)

    shape_ok = (y_ref_u8.shape == y_tri_u8.shape) and (s_ref_u8.shape == s_tri_u8.shape)
    if not shape_ok:
        print(
            f"[{dtype} {M}x{N}] SHAPE MISMATCH "
            f"y ref{tuple(y_ref_u8.shape)} tri{tuple(y_tri_u8.shape)} | "
            f"scale ref{tuple(s_ref_u8.shape)} tri{tuple(s_tri_u8.shape)}"
        )
        return False

    y_eq = torch.equal(y_ref_u8, y_tri_u8)
    s_eq = torch.equal(s_ref_u8, s_tri_u8)

    # Per-nibble payload diff (each byte packs two fp4 values).
    y_diff = (y_ref_u8 != y_tri_u8)
    s_diff = (s_ref_u8 != s_tri_u8)
    y_bad = int(y_diff.sum().item())
    s_bad = int(s_diff.sum().item())
    y_tot = y_ref_u8.numel()
    s_tot = s_ref_u8.numel()

    status = "IDENTICAL" if (y_eq and s_eq) else "DIFFERS"
    print(
        f"[{dtype} {M}x{N}] {status}: "
        f"payload {y_bad}/{y_tot} bytes differ, "
        f"scale {s_bad}/{s_tot} bytes differ"
    )
    if not (y_eq and s_eq):
        # Show the magnitude of scale disagreement (e8m0 exponent units).
        if s_bad:
            ds = (s_ref_u8.int() - s_tri_u8.int())
            print(
                f"    scale exp delta: min={int(ds.min())} max={int(ds.max())} "
                f"(0 = same exponent)"
            )
        if y_bad:
            # Decode low/high nibbles to compare the actual fp4 codes.
            r_lo, r_hi = y_ref_u8 & 0xF, (y_ref_u8 >> 4) & 0xF
            t_lo, t_hi = y_tri_u8 & 0xF, (y_tri_u8 >> 4) & 0xF
            nib_bad = int(((r_lo != t_lo).sum() + (r_hi != t_hi).sum()).item())
            print(f"    fp4 nibbles differing: {nib_bad}/{2 * y_tot}")
    return y_eq and s_eq


def main():
    if not torch.cuda.is_available():
        raise SystemExit("needs CUDA/ROCm GPU")
    all_ok = True
    for dtype in (torch.bfloat16, torch.float16):
        for M, N in [(64, 4096), (160, 2880), (1, 4096), (256, 512)]:
            ok = compare(M, N, dtype)
            all_ok = all_ok and ok
    print()
    print("RESULT:", "ALL IDENTICAL" if all_ok else "NOT identical (see above)")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
