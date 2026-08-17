# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Correctness + perf for flydsl strided-batched a8w4 GEMM
(``aiter/ops/flydsl/kernels/mxfp4_preshuffle_batched_gemm_gfx1250_tdm.py``).

DeepSeek-V4 grouped-output LoRA (``wo_a``) in ATOM is BF16 x BF16:

    o   : [M(tokens), B(groups), K]  bf16   (mbn physical)
    wo_a: [B, N(o_lora_rank), K]     bf16
    y   : [M, B, N] = einsum("mbk,bnk->mbn")

This test pins the a8w4 path the model will run:
  1. ``quant_act_mxfp8_mbn``  — fused MXFP8 quant + n32k4 scale layout,
  2. ``preshuffle_a8w4_weight_mbn`` — offline MXFP4 weight + n32k4 scale,
  3. ``flydsl_batched_gemm_a8w4_v2`` (layout='mbn', preallocated ``out=``).

Shape mapping (V4-Pro, ``config.json``):
  b = n_local_groups = o_groups // tp  (swept via ``-b``)
  m = num_tokens                     (swept via ``-s`` first dim)
  n = o_lora_rank = 1024
  k = n_heads * head_dim // o_groups = 4096

Run:
    python op_tests/test_flydsl_batched_gemm_a8w4_wo_a.py
    python op_tests/test_flydsl_batched_gemm_a8w4_wo_a.py -s 128,1024,4096 -b 2
"""

import argparse
import itertools

import pandas as pd
import torch

import aiter
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl.batched_gemm_mxfp4 import (
    flydsl_batched_gemm_a8w4_v2,
    preshuffle_a8w4_weight_mbn,
    quant_act_mxfp8_mbn,
)
from aiter.test_common import benchmark, checkAllclose, run_perftest

torch.set_default_device("cuda")

SUPPORTED_GFX = ["gfx1250"]
SEED = 0


def _ref_quant_n32k4_mbn(o_mbn: torch.Tensor):
    """Torch oracle for fused ``dynamic_mxfp8_quant_n32k4_mbn`` (bit-exact scales)."""
    from aiter.ops.shuffle import shuffle_scale_n32k4
    from aiter.ops.triton.quant import dynamic_mxfp8_quant

    M, B, K = o_mbn.shape
    a_fp8, a_scale = dynamic_mxfp8_quant(o_mbn.reshape(M * B, K))
    a_fp8 = a_fp8.view(M, B, K).contiguous()
    a_scale = a_scale.view(M, B, K // 32)
    M32 = ((M + 31) // 32) * 32
    if M32 != M:
        pad = torch.zeros(
            (M32 - M, B, K // 32), dtype=a_scale.dtype, device=a_scale.device
        )
        a_scale = torch.cat([a_scale, pad], dim=0)
    a_sh = shuffle_scale_n32k4(a_scale.transpose(0, 1).contiguous(), experts_cnt=B)
    a_sh = a_sh.transpose(0, 1).contiguous()  # [M32//32, B, (K//32)*32]
    return a_fp8, a_sh


def run_torch_wo_a(o_mbn: torch.Tensor, w_bnk: torch.Tensor, dtype=dtypes.bf16):
    """BF16 reference: y[m,b,n] = sum_k o[m,b,k] * w[b,n,k]. Not timed."""
    return torch.einsum(
        "mbk,bnk->mbn", o_mbn.to(dtypes.fp32), w_bnk.to(dtypes.fp32)
    ).to(dtype)


@benchmark()
def test_fused_quant_n32k4_mbn(b, m, k, dtype):
    """Fused MXFP8 quant + n32k4 scale must match the torch reshuffle path."""
    from aiter.ops.triton.quant import dynamic_mxfp8_quant_n32k4_mbn

    torch.manual_seed(SEED)
    o = torch.randn(m, b, k, dtype=dtype) * 0.1

    ref_fp8, ref_scale = _ref_quant_n32k4_mbn(o)

    candidates = {
        "fused": lambda: dynamic_mxfp8_quant_n32k4_mbn(o),
    }

    # Traffic: read bf16 activation, write fp8 payload + e8m0 scale buffer.
    nbytes = (
        m * b * k * o.element_size()
        + m * b * k  # fp8 payload
        + ((m + 31) // 32) * b * (k // 32) * 32  # n32k4 scale
    )

    ret = {"gfx": get_gfx()}
    for name, fn in candidates.items():
        (got_fp8, got_scale), us = run_perftest(fn)
        err_fp8 = checkAllclose(
            ref_fp8.view(torch.uint8).to(dtypes.fp32),
            got_fp8.view(torch.uint8).to(dtypes.fp32),
            rtol=0,
            atol=0,
            tol_err_ratio=0.0,
            msg=f"{name}: fp8 payload",
        )
        err_scale = checkAllclose(
            ref_scale.to(dtypes.fp32),
            got_scale.to(dtypes.fp32),
            rtol=0,
            atol=0,
            tol_err_ratio=0.0,
            msg=f"{name}: e8m0 scale",
        )
        ret[f"{name} us"] = us
        ret[f"{name} TB/s"] = nbytes / us / 1e6
        ret[f"{name} fp8 err"] = err_fp8
        ret[f"{name} scale err"] = err_scale
    return ret


@benchmark()
def test_wo_a_a8w4_gemm(b, m, n, k, dtype, layout):
    """Strided-batched a8w4 GEMM vs BF16 einsum (lossy quant tolerances)."""
    torch.manual_seed(SEED)
    # Model path: o is physically [m, b, k] (mbn); kernel out is a transposed
    # view of preallocated [m, b, n] — same as test_batched_gemm_bf16 mbn case.
    o_mbn = torch.randn(m, b, k, dtype=dtype) * 0.1
    w_bnk = torch.randn(b, n, k, dtype=dtype) * 0.1

    if layout == "mbn":
        y_phys = torch.empty(m, b, n, dtype=dtype)
        y_out = y_phys.transpose(0, 1)  # [b, m, n] view passed to kernel
    else:
        y_phys = torch.empty(b, m, n, dtype=dtype)
        y_out = y_phys

    # Offline weight prep (not in the hot path); act quant mirrors every forward.
    w_codes, w_scales = preshuffle_a8w4_weight_mbn(w_bnk)
    a_fp8, a_scales = quant_act_mxfp8_mbn(o_mbn)

    ref = run_torch_wo_a(o_mbn, w_bnk, dtype)

    def _launch():
        flydsl_batched_gemm_a8w4_v2(
            a_fp8,
            w_codes,
            a_scales,
            w_scales,
            N=n,
            dtype=dtype,
            layout=layout,
            out=y_phys,
        )
        return y_out

    candidates = {"flydsl_v2": _launch}

    flops = 2 * b * m * n * k
    # mxfp8 A + mxfp4 B + scale traffic + bf16 C
    nbytes = (
        m * b * k  # fp8 A
        + b * n * (k // 2)  # mxfp4 B codes
        + ((m + 31) // 32) * b * (k // 32) * 32  # A scale
        + b * (n // 32) * (k // 32) * 32  # B scale
        + b * m * n * y_phys.element_size()  # bf16 out
    )

    ret = {"gfx": get_gfx()}
    for name, fn in candidates.items():
        out, us = run_perftest(fn)
        if layout == "mbn":
            out_mbn = out.transpose(0, 1).contiguous()
        else:
            out_mbn = out
        ref_f = ref.to(dtypes.fp32).flatten()
        out_f = out_mbn.to(dtypes.fp32).flatten()
        cos = torch.nn.functional.cosine_similarity(ref_f, out_f, dim=0).item()
        denom = ref_f.abs().mean().clamp_min(1e-6)
        mre = (ref_f - out_f).abs().mean().item() / denom.item()
        # MXFP8 x MXFP4 vs BF16: cosine is the primary gate (see original wo_a test).
        err = 0.0 if cos >= 0.99 else (1.0 - cos)
        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = flops / us / 1e6
        ret[f"{name} TB/s"] = nbytes / us / 1e6
        ret[f"{name} cos"] = cos
        ret[f"{name} mre"] = mre
        ret[f"{name} err"] = err
    return ret


def summarize(title: str, rows: list) -> None:
    df = pd.DataFrame(rows)
    aiter.logger.info("%s (markdown):\n%s", title, df.to_markdown(index=False))


def main():
    if get_gfx() not in SUPPORTED_GFX:
        aiter.logger.warning(
            "flydsl batched a8w4 unsupported on %s; skipping", get_gfx()
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
        help="Data type.\n        e.g.: -d bf16",
    )
    parser.add_argument(
        "-b",
        "--batch",
        type=int,
        nargs="*",
        # n_local_groups = o_groups // tp for V4 deployments (see test_batched_gemm_bf16).
        default=[16, 8, 4, 2],
        help="Batch size (n_local_groups).\n        e.g.: -b 2",
    )
    parser.add_argument(
        "-s",
        "--mnk",
        type=dtypes.str2tuple,
        nargs="*",
        default=[
            (1, 1024, 4096),
            (17, 1024, 4096),  # M not a multiple of 32 -> super padding/OOB
            (32, 1024, 4096),
            (128, 1024, 4096),
            (1024, 1024, 4096),
            (4096, 1024, 4096),
        ],
        help="Shape m,n,k.\n        e.g.: -s 128,1024,4096",
    )
    parser.add_argument(
        "-l",
        "--layout",
        type=str,
        choices=["mbn"],
        nargs="*",
        default=["mbn"],
        help="Output layout (wo_a uses mbn only).\n        e.g.: -l mbn",
    )
    args = parser.parse_args()

    for dtype in args.dtype:
        quant_rows = []
        for b, (m, n, k) in itertools.product(args.batch, args.mnk):
            quant_rows.append(test_fused_quant_n32k4_mbn(b, m, k, dtype))
        summarize("fused_quant_n32k4_mbn", quant_rows)

        gemm_rows = []
        for layout, b, (m, n, k) in itertools.product(
            args.layout, args.batch, args.mnk
        ):
            gemm_rows.append(test_wo_a_a8w4_gemm(b, m, n, k, dtype, layout))
        summarize("wo_a_a8w4_gemm", gemm_rows)


if __name__ == "__main__":
    main()
