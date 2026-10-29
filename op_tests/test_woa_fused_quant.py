# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Bit-exactness for the fused MXFP8-quant epilogue of the batched mxfp8_32 GEMM.

The `wo_a` GEMM writes bf16 `y`, which ATOM then `quantize_fp8`s (group-32,
amax floored 1e-10, ceil_pow2(amax/448)) before `wo_b`. `y_scale_out=` folds
that quant into the GEMM epilogue. This checks the fused (y8, y_scale) are
byte-identical to quantize_fp8 of the unfused bf16 output.

Run (gfx1250):
    python op_tests/test_woa_fused_quant.py
    python op_tests/test_woa_fused_quant.py -s 48,128,256 -b 2 -n 1024 -k 4096
"""

import argparse

import torch

from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl.batched_gemm_a8w8_gfx1250 import run_bmm_a8w8_mxfp8_32_gfx1250
from aiter.ops.shuffle import shuffle_weight

# reuse the sibling test's quant/preshuffle oracles
from op_tests.test_flydsl_batched_gemm import (
    blockwise_randn,
    expand_32x32_to_n32k4,
    quant_act_e8m0,
    quant_weight_e8m0,
)

torch.set_default_device("cuda")
SEED = 0
FP8_MAX = 448.0


def atom_quantize_fp8(y_bf16: torch.Tensor):
    """Torch mirror of ATOM `quantize_fp8` (blockscale_kernels/quantization.py):
    group-32 over the flattened row, amax floored 1e-10, code=ceil_pow2(amax/448),
    scale=2**(code-127), q=clamp(y/scale, +-448)->e4m3. `y_bf16` is [rows, C]."""
    rows, c = y_bf16.shape
    yv = y_bf16.to(dtypes.fp32).view(rows, c // 32, 32)
    amax = yv.abs().amax(dim=-1).clamp(min=1e-10)
    raw = (amax * (1.0 / FP8_MAX)).view(torch.int32)
    code = (raw >> 23) + ((raw & 0x7FFFFF) != 0).to(torch.int32)
    code = code.clamp(0, 255).to(torch.int32)
    scale = (code << 23).view(torch.float32)
    q = (yv / scale[..., None]).clamp(-FP8_MAX, FP8_MAX).to(dtypes.fp8)
    return q.view(rows, c).contiguous(), code.to(torch.uint8).contiguous()


def run_case(b, m, n, k):
    torch.manual_seed(SEED)
    o_mbn = blockwise_randn(m, b, k, dtype=dtypes.bf16)
    w_bnk = blockwise_randn(b, n, k, dtype=dtypes.bf16, rows=32)

    x8, xs8 = quant_act_e8m0(o_mbn, 32)
    w8, ws8 = quant_weight_e8m0(w_bnk, 32, 32)
    w_shuf = shuffle_weight(w8)
    w_scale_n32k4 = expand_32x32_to_n32k4(ws8)

    # 1) unfused bf16 output, then ATOM quantize_fp8 of it -> reference.
    y_bf16 = torch.empty(m, b, n, dtype=dtypes.bf16)
    run_bmm_a8w8_mxfp8_32_gfx1250(x8, w_shuf, xs8, w_scale_n32k4, y_bf16)
    y8_ref, yscale_ref = atom_quantize_fp8(y_bf16.view(m, b * n))

    # 2) fused epilogue: e4m3 out + row-major e8m0 scale in one launch.
    y8 = torch.empty(m, b, n, dtype=dtypes.fp8)
    yscale = torch.empty(m * (b * n // 32), dtype=torch.uint8)
    run_bmm_a8w8_mxfp8_32_gfx1250(
        x8, w_shuf, xs8, w_scale_n32k4, y8, y_scale_out=yscale
    )

    scale_mismatch = (yscale.view(m, b * n // 32) != yscale_ref).sum().item()
    a = y8.view(m, b * n).view(torch.uint8).to(torch.int32)
    bb = y8_ref.view(torch.uint8).to(torch.int32)
    fp8_mismatch = (a != bb).sum().item()
    tot = m * b * n
    print(
        f"B={b} M={m} N={n} K={k}: "
        f"scale mismatch {scale_mismatch}/{m * b * n // 32}, "
        f"fp8 mismatch {fp8_mismatch}/{tot} ({100.0 * fp8_mismatch / tot:.4f}%)"
    )
    return scale_mismatch, fp8_mismatch, tot


def main():
    if get_gfx() != "gfx1250":
        print(f"fused-quant epilogue is gfx1250-only; skipping on {get_gfx()}")
        return
    p = argparse.ArgumentParser()
    p.add_argument("-s", type=str, default="16,48,128,256")
    p.add_argument("-b", type=int, default=2)
    p.add_argument("-n", type=int, default=1024)
    p.add_argument("-k", type=int, default=4096)
    a = p.parse_args()
    bad = 0
    for m in (int(x) for x in a.s.split(",")):
        sc, fp, tot = run_case(a.b, m, a.n, a.k)
        if sc != 0 or fp > 0:
            bad += 1
    print("ALL BIT-EXACT" if bad == 0 else f"{bad} shape(s) mismatched")


if __name__ == "__main__":
    main()
