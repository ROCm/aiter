# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Correctness + perf for the FlyDSL GDN gated RMSNorm + out_proj decode kernel.

The model path is the end of a Gated DeltaNet layer (Qwen3.5, Qwen3-Next), where
vLLM's ``_output_projection`` runs ``RMSNormGated`` and then the bf16 out_proj GEMM:

    x    : [m, h, d]  contiguous  (GDN core output, h = value heads on this rank)
    z    : [m, h, d]  contiguous  (output gate)
    nw   : [d]                    (RMSNormGated weight, shared by all heads)
    w    : [n, h * d] contiguous  (out_proj weight, nn.Linear layout)
    out  : [m, n]     preallocated
    y    = bf16(x * rsqrt(mean(x^2, -1) + eps) * nw * silu(z))
    out  = y.flatten(-2) @ w.T

The shapes are given as (m, n, k) with k = h * d and d = 128.

Candidates:
    flydsl       the fused kernel, norm and GEMM in one launch.
    gemm_a16w16  aiter's bf16 GEMM on a y that is computed before timing. It is an
                 out_proj GEMM without the norm kernel that an unfused path also runs.

Run:
    python op_tests/test_flydsl_gdn_gated_rmsnorm_out_proj.py
    python op_tests/test_flydsl_gdn_gated_rmsnorm_out_proj.py -s 1,8192,2048
"""

import argparse

import pandas as pd
import torch
import torch.nn.functional as F

import aiter
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl import (
    flydsl_gdn_gated_rmsnorm_out_proj,
    flydsl_gdn_gated_rmsnorm_out_proj_supported,
)
from aiter.test_common import (
    benchmark,
    checkAllclose,
    run_perftest,
)
from aiter.tuned_gemm import gemm_a16w16

torch.set_default_device("cuda")

# The public wrapper supports gfx950 only. Positive allow-list: an unknown new card
# must not silently run an unbuilt kernel.
SUPPORTED_GFX = ["gfx950"]
SEED = 0
HEAD_DIM = 128
EPS = 1e-6

# (n, k) pairs: n is the hidden size, k = local value heads * 128.
#   (8192, 2048): amd/Qwen3.8-2.4T-A95B-Quark-MXFP4 at TP8, 16 of 128 value heads.
#   (8192, 4096): the same model at TP4.
#   (4096, 1024): Qwen/Qwen3.5-397B-A17B at TP8, 8 of 64 value heads.
#   (2048, 1024): Qwen/Qwen3-Next-80B-A3B-Instruct at TP4, 8 of 32 value heads.
# The wrapper runs k from 1024 to 4096, up to 8 tokens at k = 1024 and up to 5 tokens
# above that.
_NK = [(8192, 2048), (8192, 4096), (4096, 1024), (2048, 1024)]
_DEFAULT_MNK = [
    (m, n, k) for (n, k) in _NK for m in (range(1, 9) if k <= 1024 else range(1, 6))
]
# n = 4104 is a multiple of 8 but not of 16, so the wrapper falls back to the tile
# with 2 columns per wave and 4 waves per workgroup.
_DEFAULT_MNK.append((2, 4104, 1024))


def run_torch(x, z, norm_weight, weight, eps):
    # Reference only: fp32 math, cast back. Not timed, not in the table.
    # y is rounded to bf16 before the GEMM, as the unfused model path does.
    xf = x.to(dtypes.fp32)
    rrms = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    y = xf * rrms * norm_weight.to(dtypes.fp32) * F.silu(z.to(dtypes.fp32))
    y = y.to(x.dtype).flatten(-2)
    return y, torch.mm(y.to(dtypes.fp32), weight.t().to(dtypes.fp32)).to(x.dtype)


@benchmark()
def test_gdn_gated_rmsnorm_out_proj(m, n, k, dtype):
    torch.manual_seed(SEED)
    h = k // HEAD_DIM
    # Every head gets a different scale. A norm per head removes it, and a norm over
    # any other group of values would leave a large error.
    head_scale = torch.linspace(0.25, 4.0, h).view(1, h, 1)
    x = (torch.randn((m, h, HEAD_DIM)) * head_scale).to(dtype)
    z = torch.randn((m, h, HEAD_DIM), dtype=dtype)
    norm_weight = (1.0 + 0.1 * torch.randn(HEAD_DIM)).to(dtype)
    weight = (0.02 * torch.randn((n, k))).to(dtype)
    # A dirty preallocated output shows that the kernel writes every element.
    out = torch.randn((m, n), dtype=dtype)
    y, ref = run_torch(x, z, norm_weight, weight, EPS)

    # vLLM passes no out=, so the wrapper allocates the result. Check that path once.
    err_alloc = checkAllclose(
        ref.to(dtypes.fp32),
        flydsl_gdn_gated_rmsnorm_out_proj(x, z, norm_weight, weight, eps=EPS).to(
            dtypes.fp32
        ),
        rtol=1e-2,
        atol=1e-2,
        msg="flydsl without out=: gdn gated rmsnorm + out_proj",
    )
    assert (
        err_alloc == 0
    ), f"flydsl without out= differs from the reference: {err_alloc}"

    # Every tensor is passed as an argument, so run_perftest rotates copies of them
    # and each timed call reads a weight that is not in the cache, as in a forward
    # pass through many layers.
    def run_flydsl(x, z, norm_weight, weight, out):
        return flydsl_gdn_gated_rmsnorm_out_proj(
            x, z, norm_weight, weight, eps=EPS, out=out
        )

    def run_gemm_a16w16(y, weight):
        return gemm_a16w16(y, weight)

    # Bytes each candidate moves: the fused kernel reads x, z, the norm weight, and w,
    # and the GEMM alone reads y and w. Both write out.
    elem = x.element_size()
    candidates = {
        "flydsl": (
            run_flydsl,
            (x, z, norm_weight, weight, out),
            (2 * m * k + HEAD_DIM + n * k + m * n) * elem,
        ),
        "gemm_a16w16": (run_gemm_a16w16, (y, weight), (m * k + n * k + m * n) * elem),
    }

    flops = 2 * m * n * k
    ret = {"gfx": get_gfx()}
    for name, (fn, args, nbytes) in candidates.items():
        result, us = run_perftest(fn, *args)
        err = checkAllclose(
            ref.to(dtypes.fp32),
            result.to(dtypes.fp32),
            rtol=1e-2,
            atol=1e-2,
            msg=f"{name}: gdn gated rmsnorm + out_proj",
        )
        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = flops / us / 1e6
        ret[f"{name} TB/s"] = nbytes / us / 1e6
        ret[f"{name} err"] = err
    assert ret["flydsl err"] == 0, f"flydsl result differs from the reference: {ret}"
    return ret


def summarize(title, rows):
    df = pd.DataFrame(rows)
    if df.empty:
        return
    aiter.logger.info("%s summary (markdown):\n%s", title, df.to_markdown(index=False))


def main():
    if get_gfx() not in SUPPORTED_GFX:
        aiter.logger.warning(
            "flydsl gdn gated rmsnorm + out_proj unsupported on %s, skipping",
            get_gfx(),
        )
        return

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="config input of test",
    )
    parser.add_argument(
        "-d",
        "--dtype",
        type=dtypes.str2Dtype,
        choices=[dtypes.d_dtypes["bf16"]],
        nargs="*",
        default="bf16,",
        metavar="{bf16}",
        help="""Data type.
        e.g.: -d bf16""",
    )
    parser.add_argument(
        "-s",
        "--mnk",
        type=dtypes.str2tuple,
        nargs="*",
        default=_DEFAULT_MNK,
        help="""Shape (m, n, k): m tokens, n = hidden size, k = heads * 128.
        e.g.:   -s 1,8192,2048
                --mnk 4,4096,1024""",
    )
    args = parser.parse_args()

    for dtype in args.dtype:
        rows = []
        for m, n, k in args.mnk:
            if k % HEAD_DIM or not flydsl_gdn_gated_rmsnorm_out_proj_supported(
                m, k // HEAD_DIM, HEAD_DIM, n, dtype
            ):
                aiter.logger.warning(
                    "shape m=%s n=%s k=%s is not supported by the kernel, skipping",
                    m,
                    n,
                    k,
                )
                continue
            rows.append(test_gdn_gated_rmsnorm_out_proj(m, n, k, dtype))
        summarize(f"flydsl_gdn_gated_rmsnorm_out_proj {dtype}", rows)


if __name__ == "__main__":
    main()
