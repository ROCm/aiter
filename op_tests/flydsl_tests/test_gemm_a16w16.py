# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import argparse
import itertools
from functools import partial

import pandas as pd
import torch
import torch.nn.functional as F

import aiter
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl.gemm_kernels import flydsl_hgemm
from aiter.ops.triton.gemm.basic.gemm_a16w16 import gemm_a16w16 as triton_gemm_a16w16
from aiter.test_common import benchmark, checkAllclose, run_perftest

torch.set_default_device("cuda")

SUPPORTED_GFX = ["gfx950", "gfx1250"]
LAYOUTS = ["TN", "TT", "NN", "NT"]
ACTIVATIONS = {
    "gelu": F.gelu,
    "gelu_tanh": partial(F.gelu, approximate="tanh"),
    "silu": F.silu,
    "silu_exp2": F.silu,
    "relu": F.relu,
}
SENTINEL = 4096.0

DEFAULT_MNK = {
    "gfx1250": [
        (64, 64, 64),
        (256, 256, 256),
        (32, 256, 128),
        (256, 32, 128),
        (128, 128, 1024),
        (1024, 128, 128),
        (100, 190, 256),
        (129, 257, 512),
        (128, 192, 48),
        (65, 190, 1000),
        (64, 5120, 2880),
        (64, 2880, 4096),
        (64, 128, 2880),
    ],
    "gfx950": [
        (8, 4096, 4096),
        (64, 4096, 4096),
        (256, 4096, 4096),
        (1024, 1024, 1024),
        (32, 384, 7168),
        (8, 7168, 2048),
    ],
}
DEFAULT_CONFIGS = {
    "gfx1250": [
        (64, 5120, 2880, 64, 64, 256, 4, 2, 4, 2, 1),
        (64, 2880, 4096, 64, 64, 128, 4, 2, 6, 4, 1),
        (64, 128, 2880, 16, 16, 256, 1, 1, 3, 6, 1),
        (128, 128, 512, 64, 64, 128, 2, 2, 2, 2, 0),
        (128, 128, 512, 64, 64, 128, 2, 2, 2, 4, 0),
        (64, 64, 768, 64, 64, 128, 2, 2, 2, 3, 0),
        (64, 64, 768, 64, 64, 128, 2, 2, 2, 6, 0),
        (128, 256, 1024, 64, 64, 128, 2, 2, 2, 4, 0),
        (256, 128, 1024, 32, 32, 128, 2, 2, 2, 8, 0),
        (128, 128, 1024, 64, 64, 128, 2, 2, 3, 4, 0),
        (64, 64, 576, 64, 64, 64, 2, 2, 2, 2, 0),
        (64, 100, 512, 64, 64, 64, 2, 2, 2, 2, 0),
        (100, 100, 256, 128, 128, 32, 2, 4, 3, 1, 0),
        (129, 257, 512, 128, 128, 32, 2, 4, 3, 1, 0),
        (65, 190, 256, 128, 128, 32, 2, 4, 3, 1, 0),
        (250, 120, 512, 64, 64, 128, 2, 2, 2, 1, 0),
        (33, 65, 256, 32, 32, 128, 2, 2, 3, 1, 0),
        (64, 64, 1024, 32, 32, 128, 2, 2, 3, 1, 0),
        (128, 128, 1024, 128, 128, 32, 2, 4, 3, 1, 1),
    ],
    "gfx950": [
        (64, 4096, 4096, 32, 32, 128, 2, 2, 6, 1, 0),
        (128, 4096, 4096, 32, 64, 128, 1, 4, 4, 1, 0),
        (256, 4096, 4096, 64, 64, 128, 4, 2, 4, 1, 0),
        (512, 4096, 4096, 64, 128, 64, 2, 4, 5, 1, 0),
        (1024, 4096, 4096, 128, 128, 64, 2, 4, 4, 1, 0),
        (1024, 1024, 1024, 64, 64, 64, 1, 4, 4, 1, 0),
        (2048, 2048, 2048, 128, 128, 64, 2, 4, 4, 1, 0),
        (4096, 4096, 4096, 256, 256, 64, 2, 4, 2, 1, 1),
        (8, 7168, 2048, 16, 16, 64, 1, 1, 8, 1, 0),
        (32, 14336, 4096, 32, 64, 128, 2, 2, 5, 1, 0),
    ],
}


def generate_inputs(m, n, k, dtype, layout="TN", bias=False):
    if layout[0] == "T":
        x = torch.randn(m, k, dtype=dtype)
    else:
        x = torch.randn(k, m, dtype=dtype).T
    if layout[1] == "T":
        w = torch.randn(k, n, dtype=dtype).T
    else:
        w = torch.randn(n, k, dtype=dtype)
    b = torch.randn(n, dtype=dtype) if bias else None
    return x, w, b


def run_torch(x, w, bias=None, activation=None, dtype=dtypes.bf16):
    out = F.linear(
        x.to(dtypes.fp32),
        w.to(dtypes.fp32),
        None if bias is None else bias.to(dtypes.fp32),
    )
    if activation is not None:
        out = ACTIVATIONS[activation](out)
    return out.to(dtype)


def run_candidates(candidates, ref, m, n, k, in_bytes, out_bytes, msg):
    flops = 2 * m * n * k
    nbytes = (m * k + n * k) * in_bytes + m * n * out_bytes
    ret = {"gfx": get_gfx()}
    for name, fn in candidates.items():
        out, us = run_perftest(fn)
        err = checkAllclose(
            ref.to(dtypes.fp32),
            out.to(dtypes.fp32),
            rtol=1e-2,
            atol=1e-2,
            msg=f"{name}: {msg}",
        )
        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = flops / us / 1e6
        ret[f"{name} TB/s"] = nbytes / us / 1e6
        ret[f"{name} err"] = err
    return ret


@benchmark()
def test_gemm_a16w16(m, n, k, dtype, layout):
    x, w, _ = generate_inputs(m, n, k, dtype, layout)
    ref = run_torch(x, w, dtype=dtype)
    candidates = {
        "flydsl": lambda: flydsl_hgemm(x, w, out_dtype=dtype),
        "triton": lambda: triton_gemm_a16w16(x, w, dtype=dtype),
    }
    return run_candidates(
        candidates, ref, m, n, k, x.element_size(), ref.element_size(), "gemm a16w16"
    )


@benchmark()
def test_gemm_a16w16_fused(m, n, k, dtype, activation, bias):
    from aiter.ops.flydsl.gemm_a16w16_gfx1250 import gemm_a16w16

    x, w, b = generate_inputs(m, n, k, dtype, bias=bias)
    ref = run_torch(x, w, b, activation, dtype)
    candidates = {
        "flydsl": lambda: gemm_a16w16(x, w, bias=b, dtype=dtype, activation=activation),
        "triton": lambda: triton_gemm_a16w16(
            x, w, bias=b, dtype=dtype, activation=activation
        ),
    }
    return run_candidates(
        candidates,
        ref,
        m,
        n,
        k,
        x.element_size(),
        ref.element_size(),
        f"gemm a16w16 + {activation}",
    )


@benchmark()
def test_gemm_a16w16_config(
    m,
    n,
    k,
    dtype,
    otype,
    block_m,
    block_n,
    block_k,
    m_waves,
    n_waves,
    stages,
    split_k,
    unroll,
):
    x, w, _ = generate_inputs(m, n, k, dtype)
    ref = run_torch(x, w, dtype=otype)
    parent = torch.full((m + block_m, n + block_n), SENTINEL, dtype=otype)
    y = parent[:m, :n]
    cfg = {
        "out": y,
        "out_dtype": otype,
        "block_m": block_m,
        "block_n": block_n,
        "block_k": block_k,
        "stages": stages,
        "split_k": split_k,
        "m_waves": m_waves,
        "n_waves": n_waves,
        "k_waves": 1,
        "group_m": 0,
        "policy": "ht" if unroll else "ft",
    }
    assert flydsl_hgemm(x, w, **cfg) is y
    candidates = {
        "flydsl": lambda: flydsl_hgemm(x, w, **cfg),
        "triton": lambda: triton_gemm_a16w16(x, w, dtype=otype),
    }
    ret = run_candidates(
        candidates,
        ref,
        m,
        n,
        k,
        x.element_size(),
        ref.element_size(),
        "gemm a16w16 config",
    )
    assert torch.all(parent[m:] == SENTINEL) and torch.all(
        parent[:, n:] == SENTINEL
    ), "flydsl wrote outside the [m, n] output view"
    return ret


def summarize(name, rows):
    aiter.logger.info(
        "%s summary (markdown):\n%s", name, pd.DataFrame(rows).to_markdown(index=False)
    )


def main():
    gfx = get_gfx()
    if gfx not in SUPPORTED_GFX:
        aiter.logger.warning("flydsl a16w16 gemm unsupported on %s; skipping", gfx)
        return

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="config input of test",
    )
    parser.add_argument(
        "-d",
        "--dtype",
        type=dtypes.str2Dtype,
        choices=[dtypes.d_dtypes["bf16"], dtypes.d_dtypes["fp16"]],
        nargs="*",
        default="bf16,",
        metavar="{bf16,fp16}",
        help="""Input dtype of x and w.
        e.g.: -d bf16""",
    )
    parser.add_argument(
        "-o",
        "--otype",
        type=dtypes.str2Dtype,
        choices=[dtypes.d_dtypes[t] for t in ("bf16", "fp16", "fp32")],
        nargs="*",
        default="bf16,fp32,",
        metavar="{bf16,fp16,fp32}",
        help="""Output dtype for the kernel-config sweep (-c): fp32 keeps split-K
        lossless, bf16/fp16 take the f32-accumulate-then-cast path.
        e.g.: -o fp32""",
    )
    parser.add_argument(
        "-s",
        "--mnk",
        type=dtypes.str2tuple,
        nargs="*",
        default=DEFAULT_MNK[gfx],
        help="""Shape of mnk.
        e.g.: -s 64,5120,2880""",
    )
    parser.add_argument(
        "-l",
        "--layout",
        type=str,
        choices=LAYOUTS,
        nargs="*",
        default=LAYOUTS,
        help="""(x, w) memory layout. x: T = row-major [m,k], N = transposed view;
        w: N = row-major [n,k], T = transposed view. TN is the nn.Linear call.
        e.g.: -l TN NT""",
    )
    parser.add_argument(
        "-a",
        "--activation",
        type=str,
        choices=list(ACTIVATIONS),
        nargs="*",
        default=list(ACTIVATIONS),
        help="""Fused epilogue activation (gfx1250 kernel only).
        e.g.: -a gelu silu""",
    )
    parser.add_argument(
        "--bias",
        type=dtypes.str2bool,
        nargs="*",
        default=[False, True],
        help="""Fuse a bias add into the epilogue.
        e.g.: --bias 1""",
    )
    parser.add_argument(
        "-c",
        "--config",
        type=dtypes.str2tuple,
        nargs="*",
        default=DEFAULT_CONFIGS[gfx],
        help="""Explicit flydsl_hgemm config as
        m,n,k,block_m,block_n,block_k,m_waves,n_waves,stages,split_k,unroll
        (unroll=1 -> policy ht; the fields of a tuned flydsl_hgemm_* csv name).
        e.g.: -c 64,5120,2880,64,64,256,4,2,4,2,1""",
    )
    args = parser.parse_args()

    for dtype in args.dtype:
        summarize(
            "gemm_a16w16",
            [
                test_gemm_a16w16(m, n, k, dtype, layout)
                for layout, (m, n, k) in itertools.product(args.layout, args.mnk)
            ],
        )
        if gfx == "gfx1250":
            summarize(
                "gemm_a16w16 fused epilogue",
                [
                    test_gemm_a16w16_fused(m, n, k, dtype, act, bias)
                    for act, bias, (m, n, k) in itertools.product(
                        args.activation, args.bias, args.mnk
                    )
                ],
            )
        else:
            aiter.logger.warning(
                "%s: skipping fused-epilogue table (activation fusion is gfx1250-only)",
                gfx,
            )
        summarize(
            "gemm_a16w16 kernel config",
            [
                test_gemm_a16w16_config(m, n, k, dtype, otype, *tile)
                for otype, (m, n, k, *tile) in itertools.product(
                    args.otype, args.config
                )
            ],
        )


if __name__ == "__main__":
    main()
