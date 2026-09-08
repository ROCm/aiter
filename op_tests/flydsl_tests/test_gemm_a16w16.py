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
from aiter.ops.flydsl.gemm_a16w16_gfx1250 import gemm_a16w16 as flydsl_gemm_a16w16
from aiter.ops.flydsl.utils import is_flydsl_available
from aiter.ops.triton.gemm.basic.gemm_a16w16 import gemm_a16w16 as triton_gemm_a16w16
from aiter.test_common import benchmark, checkAllclose, run_perftest

torch.set_default_device("cuda")

SUPPORTED_GFX = ["gfx1250"]
LAYOUTS = ["TN", "TT", "NN", "NT"]
ACTIVATIONS = {
    "gelu": F.gelu,
    "gelu_tanh": partial(F.gelu, approximate="tanh"),
    "silu": F.silu,
    "silu_exp2": F.silu,
    "relu": F.relu,
}
SENTINEL = 4096.0


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
    # [m,k] @ [n,k]^T: FLOPs = 2*m*n*k; bytes = x + w read, y written.
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
        "flydsl": lambda: flydsl_gemm_a16w16(x, w, dtype=dtype),
        "flydsl_cb": lambda: flydsl_gemm_a16w16(
            x, w, dtype=dtype, variant="compute_bound"
        ),
        "triton": lambda: triton_gemm_a16w16(x, w, dtype=dtype),
    }
    return run_candidates(
        candidates, ref, m, n, k, x.element_size(), ref.element_size(), "gemm a16w16"
    )


@benchmark()
def test_gemm_a16w16_fused(m, n, k, dtype, activation, bias):
    x, w, b = generate_inputs(m, n, k, dtype, bias=bias)
    ref = run_torch(x, w, b, activation, dtype)
    candidates = {
        "flydsl": lambda: flydsl_gemm_a16w16(
            x, w, bias=b, dtype=dtype, activation=activation
        ),
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
    tile_m,
    tile_n,
    tile_k,
    m_warp,
    n_warp,
    num_buffers,
    split_k,
    unroll,
):
    x, w, _ = generate_inputs(m, n, k, dtype)
    ref = run_torch(x, w, dtype=otype)
    parent = torch.full((m + tile_m, n + tile_n), SENTINEL, dtype=otype)
    y = parent[:m, :n]
    cfg = {
        "dtype": otype,
        "y": y,
        "tile_m": tile_m,
        "tile_n": tile_n,
        "tile_k": tile_k,
        "m_warp": m_warp,
        "n_warp": n_warp,
        "num_buffers": num_buffers,
        "split_k": split_k,
        "main_loop_unroll": bool(unroll),
    }
    assert flydsl_gemm_a16w16(x, w, **cfg) is y
    candidates = {
        "flydsl": lambda: flydsl_gemm_a16w16(x, w, **cfg),
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
    if get_gfx() not in SUPPORTED_GFX or not is_flydsl_available():
        aiter.logger.warning(
            "flydsl gemm_a16w16 needs gfx1250 + flydsl; skipping on %s", get_gfx()
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
        default=[
            (64, 64, 64),  # single tile
            (256, 256, 256),  # aligned multi-tile
            (32, 256, 128),  # m < tile_m
            (256, 32, 128),  # n < tile_n
            (128, 128, 1024),  # long main loop + drain
            (1024, 128, 128),
            (100, 190, 256),  # ragged m/n
            (129, 257, 512),  # one past a tile boundary
            (128, 192, 48),  # ragged k, K < 2 * tile_k
            (65, 190, 1000),  # ragged m/n/k
            (64, 5120, 2880),
            (64, 2880, 4096),
            (64, 128, 2880),
        ],
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
        help="""Fused epilogue activation.
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
        default=[
            (64, 5120, 2880, 64, 64, 256, 4, 2, 4, 2, 1),
            (64, 2880, 4096, 64, 64, 128, 4, 2, 6, 4, 1),
            (64, 128, 2880, 16, 16, 256, 1, 1, 3, 6, 1),
            # split-K: k-tiles per split from 1 up to 4, num_buffers 2 and 3
            (128, 128, 512, 64, 64, 128, 2, 2, 2, 2, 0),
            (128, 128, 512, 64, 64, 128, 2, 2, 2, 4, 0),
            (64, 64, 768, 64, 64, 128, 2, 2, 2, 3, 0),
            (64, 64, 768, 64, 64, 128, 2, 2, 2, 6, 0),
            (128, 256, 1024, 64, 64, 128, 2, 2, 2, 4, 0),
            (256, 128, 1024, 32, 32, 128, 2, 2, 2, 8, 0),
            (128, 128, 1024, 64, 64, 128, 2, 2, 3, 4, 0),
            (64, 64, 576, 64, 64, 64, 2, 2, 2, 2, 0),  # ragged k per split
            (64, 100, 512, 64, 64, 64, 2, 2, 2, 2, 0),  # ragged n + split-K
            # ragged m/n edge tiles at each tile size
            (100, 100, 256, 128, 128, 32, 2, 4, 3, 1, 0),
            (129, 257, 512, 128, 128, 32, 2, 4, 3, 1, 0),
            (65, 190, 256, 128, 128, 32, 2, 4, 3, 1, 0),
            (250, 120, 512, 64, 64, 128, 2, 2, 2, 1, 0),
            (33, 65, 256, 32, 32, 128, 2, 2, 3, 1, 0),
            (64, 64, 1024, 32, 32, 128, 2, 2, 3, 1, 0),
            # default tile with main-loop unroll
            (128, 128, 1024, 128, 128, 32, 2, 4, 3, 1, 1),
        ],
        help="""Explicit kernel config as
        m,n,k,tile_m,tile_n,tile_k,m_warp,n_warp,num_buffers,split_k,unroll
        (the fields a tuned flydsl_a16w16_gfx1250_* kernel name encodes).
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
        summarize(
            "gemm_a16w16 fused epilogue",
            [
                test_gemm_a16w16_fused(m, n, k, dtype, act, bias)
                for act, bias, (m, n, k) in itertools.product(
                    args.activation, args.bias, args.mnk
                )
            ],
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
