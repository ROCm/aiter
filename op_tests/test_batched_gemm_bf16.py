# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import argparse
import itertools

import aiter
import pandas as pd
import torch
import torch.nn.functional as F
from aiter import dtypes
from aiter.ops.triton.gemm.batched.batched_gemm_bf16 import (
    batched_gemm_bf16 as batched_gemm_bf16_triton,
)
from aiter.test_common import (
    benchmark,
    checkAllclose,
    run_perftest,
)
from aiter.jit.utils.chip_info import get_gfx

torch.set_default_device("cuda")

# triton + torch_einsum run on every arch; only the CK path is arch-limited.
CK_SUPPORTED_GFX = ["gfx942", "gfx950"]


def run_torch(x, weight, dtype=dtypes.bf16):
    B, M = x.size(0), x.size(1)
    N = weight.size(1)
    out = torch.empty(B, M, N, dtype=dtypes.bf16, device="cuda")
    for b in range(B):
        out[b, :, :] = F.linear(
            x[b, :, :].to(dtypes.fp32), weight[b, :, :].to(dtypes.fp32)
        )
    return out.to(dtype)


@benchmark()
def test_gemm(b, m, n, k, dtype, layout):
    weight = torch.randint(-20, 20, (b, n, k), dtype=dtypes.bf16)
    # Input (x) and output (y) layout, both logically [b, m, k] / [b, m, n]:
    #   mbn: transposed views of contiguous [m, b, *] tensors (physically
    #        [m, b, k] / [m, b, n]). Matches atom/models/deepseek_v4.py grouped
    #        output LoRA, where o is contiguous [s, g, d] and the gemm gets
    #        o.transpose(0, 1) / writes empty(m, b, n).transpose(0, 1).
    #   bmn: plain contiguous [b, m, k] / [b, m, n].
    if layout == "mbn":
        x = torch.randint(-20, 20, (m, b, k), dtype=dtypes.bf16).transpose(0, 1)
        y = torch.empty(m, b, n, dtype=dtypes.bf16).transpose(0, 1)
    else:
        x = torch.randint(-20, 20, (b, m, k), dtype=dtypes.bf16)
        y = torch.empty(b, m, n, dtype=dtypes.bf16)

    ref = run_torch(x, weight, dtype)

    # torch.einsum runs on the model's natural [s, g, d] / [g, r, d] layout
    # (no transpose around the call), producing [s, g, r].
    o_sgd = x.transpose(0, 1).contiguous()

    gemm_funcs = {
        # triton path mirrors the model call (preallocated transposed YQ)
        "triton": lambda: batched_gemm_bf16_triton(x, weight, YQ=y),
        "torch_einsum": lambda: torch.einsum(
            "sgd,grd->sgr" if layout == "mbn" else "sgd,grd->gsr", o_sgd, weight
        ),
    }
    # CK is arch-limited (gfx942/gfx950) and only supports a contiguous input; on
    # the mbn (transposed mbk) input it returns wrong results. Skip it otherwise.
    if layout == "bmn" and get_gfx() in CK_SUPPORTED_GFX:
        gemm_funcs["ck"] = lambda: aiter.batched_gemm_bf16_CK(x, weight)
    # batched GEMM b x ([m,k] @ [n,k]^T -> [m,n]):
    #   FLOPs   = 2 * b * m * n * k  (multiply-add)
    #   bytes   = (x + weight + out) elements * dtype size
    # TFLOPS = FLOPs / us / 1e6;  TB/s = bytes / us / 1e6.
    flops = 2 * b * m * n * k
    nbytes = (b * m * k + b * n * k + b * m * n) * x.element_size()

    ret = {"gfx": get_gfx()}
    for name, gemm_func in gemm_funcs.items():
        out, us = run_perftest(gemm_func)
        if name == "torch_einsum" and layout == "mbn":
            out = out.transpose(0, 1)  # [s, g, r] -> [g, s, r] == [b, m, n]
        err = checkAllclose(
            ref.to(dtypes.fp32),
            out.to(dtypes.fp32),
            rtol=1e-2,
            atol=1e-2,
            msg=f"{name}: batched gemm bf16",
        )
        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = flops / us / 1e6
        ret[f"{name} TB/s"] = nbytes / us / 1e6
        ret[f"{name} err"] = err

    return ret


def main():
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
        "-b",
        "--batch",
        type=int,
        nargs="*",
        # n_local_groups = o_groups // tp for the real V4 deployments:
        #   16: V4-Pro dp-attn (o_groups=16, tp1);  8: V4-Pro tp2;
        #    4: V4-Flash tp2 (o_groups=8);          2: V4-Pro tp8.
        default=[16, 8, 4, 2],
        help="""Batch size.
        e.g.: -b 16""",
    )
    parser.add_argument(
        "-s",
        "--mnk",
        type=dtypes.str2tuple,
        nargs="*",
        default=[
            # deepseek-v4 grouped output LoRA (attn.wo_a) — the only shape this
            # op runs in the real model:
            #   n = o_lora_rank (1024), k = n_heads * head_dim // o_groups (4096).
            # V4-Pro and V4-Flash share (n, k); they differ only by batch
            # (n_local_groups), swept via -b above.
            (1, 1024, 4096),
            (32, 1024, 4096),
            (64, 1024, 4096),
            (128, 1024, 4096),
            (192, 1024, 4096),
            (256, 1024, 4096),
            (320, 1024, 4096),
            (512, 1024, 4096),
            (1024, 1024, 4096),
            (2048, 1024, 4096),
            (4096, 1024, 4096),
            (8192, 1024, 4096),
        ],
        help="""Shape of mnk.
        e.g.:   -s 1,1280,8192
                --mnk 1,1280,8192""",
    )
    parser.add_argument(
        "-l",
        "--layout",
        type=str,
        choices=["bmn", "mbn"],
        nargs="*",
        default=["bmn", "mbn"],
        help="""Output (YQ) layout for the triton kernel (both logically [b, m, n]):
        mbn = transposed view (physically [m, b, n], the deepseek-v4 model path),
        bmn = plain contiguous [b, m, n].
        e.g.: -l mbn""",
    )
    args = parser.parse_args()

    for dtype in args.dtype:
        df = []
        for layout, b, (m, n, k) in itertools.product(
            args.layout, args.batch, args.mnk
        ):
            ret = test_gemm(b, m, n, k, dtype, layout)
            df.append(ret)
        df = pd.DataFrame(df)
        aiter.logger.info(
            "batched_gemm_bf16 summary (markdown):\n%s", df.to_markdown(index=False)
        )


if __name__ == "__main__":
    main()
