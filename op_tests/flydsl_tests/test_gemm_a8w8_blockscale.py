# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import argparse
import itertools

import pandas as pd
import torch

import aiter
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl.gemm_a8w8_blockscale_f32_gfx1250 import gemm_a8w8_blockscale
from aiter.ops.shuffle import shuffle_weight, shuffle_weight_gfx1250
from aiter.ops.triton.gemm.basic.gemm_a8w8_blockscale import (
    gemm_a8w8_blockscale_preshuffle,
)
from aiter.test_common import benchmark, checkAllclose, run_perftest

torch.set_default_device("cuda")

SUPPORTED_GFX = ["gfx1250"]
SCALE_BLOCK_N = 128
SCALE_BLOCK_K = 128
LAYOUTS = ["dense", "padded"]
SENTINEL = 4096.0

DEFAULT_MNK = [
    (128, 128, 128),
    (128, 256, 96),
    (200, 256, 1056),
    (7, 256, 512),
    (129, 144, 512),
    (200, 272, 512),
    (255, 16, 1024),
    (1, 8192, 1024),
    (32, 32768, 1024),
    (64, 4096, 4096),
    (128, 1536, 7168),
    (128, 7168, 1536),
    (512, 2048, 2048),
    (128, 256, 12288),
]
DEFAULT_CONFIGS = [
    (128, 256, 256, 128, 128, 128, 2, 4, 2, 0, 1),
    (128, 256, 256, 128, 128, 128, 2, 4, 3, 0, 1),
    (256, 512, 512, 128, 128, 128, 2, 4, 4, 0, 1),
    (128, 128, 1024, 128, 128, 128, 2, 4, 3, 1, 1),
    (128, 256, 1024, 128, 128, 128, 2, 4, 3, 1, 2),
    (256, 512, 2048, 128, 128, 128, 2, 4, 3, 1, 4),
    (128, 128, 4096, 128, 128, 128, 2, 4, 3, 1, 8),
    (1024, 1024, 1024, 128, 128, 256, 2, 4, 3, 0, 1),
    (200, 256, 1152, 128, 128, 256, 2, 4, 2, 0, 1),
    (129, 144, 512, 128, 128, 128, 2, 4, 3, 1, 1),
    (200, 272, 1024, 128, 128, 128, 2, 4, 3, 1, 2),
    (129, 16, 1024, 128, 128, 128, 2, 4, 3, 1, 2),
    (128, 256, 12288, 128, 128, 128, 2, 4, 3, 1, 2),
    (256, 512, 16384, 128, 128, 128, 2, 4, 3, 0, 1),
]


def pad_rows(t):
    wide = torch.empty((t.shape[0], 2 * t.shape[1]), dtype=t.dtype)
    wide[:, : t.shape[1]] = t
    return wide[:, : t.shape[1]]


def generate_inputs(m, n, k, layout="dense"):
    x = (torch.rand((m, k)) / 10).to(dtypes.fp8)
    w = (torch.rand((n, k)) / 10).to(dtypes.fp8)
    x_scale = torch.rand((m, -(-k // SCALE_BLOCK_K)))
    w_scale = torch.rand((-(-n // SCALE_BLOCK_N), -(-k // SCALE_BLOCK_K)))
    if layout == "padded":
        x, x_scale = pad_rows(x), pad_rows(x_scale)
    return x, w, x_scale, w_scale


def run_torch(x, w, x_scale, w_scale, dtype=dtypes.bf16):
    m, k = x.shape
    n = w.shape[0]
    xs = x_scale.repeat_interleave(SCALE_BLOCK_K, dim=1)[:m, :k]
    ws = w_scale.repeat_interleave(SCALE_BLOCK_N, dim=0).repeat_interleave(
        SCALE_BLOCK_K, dim=1
    )[:n, :k]
    out = (x.to(dtypes.fp32) * xs) @ (w.to(dtypes.fp32) * ws).T
    return out.to(dtype)


def triton_supported(k, dtype):
    # The triton preshuffle kernel emits bf16/fp16 only and needs whole scale blocks
    # along K; ragged K and fp32 output are flydsl-only rows.
    return dtype != dtypes.fp32 and k % SCALE_BLOCK_K == 0


def run_triton(x, w, x_scale, w_scale, dtype):
    n, k = w.shape
    w_tri = shuffle_weight(w, layout=(16, 16)).reshape(n // 16, k * 16)
    xs_t = x_scale.transpose(0, 1).contiguous().view(*x_scale.shape)
    return lambda: gemm_a8w8_blockscale_preshuffle(x, w_tri, xs_t, w_scale, dtype)


def run_candidates(candidates, ref, x, w, x_scale, w_scale, msg):
    m, k = x.shape
    n = w.shape[0]
    flops = 2 * m * n * k
    nbytes = (
        (m * k + n * k) * x.element_size()
        + (x_scale.numel() + w_scale.numel()) * x_scale.element_size()
        + m * n * ref.element_size()
    )
    tol = 1e-3 if ref.dtype == dtypes.fp32 else 1e-2
    ret = {"gfx": get_gfx()}
    for name, fn in candidates.items():
        out, us = run_perftest(fn)
        err = checkAllclose(
            ref.to(dtypes.fp32),
            out.to(dtypes.fp32),
            rtol=tol,
            atol=tol,
            msg=f"{name}: {msg}",
        )
        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = flops / us / 1e6
        ret[f"{name} TB/s"] = nbytes / us / 1e6
        ret[f"{name} err"] = err
    return ret


@benchmark()
def test_gemm_a8w8_blockscale(m, n, k, dtype, layout):
    x, w, x_scale, w_scale = generate_inputs(m, n, k, layout)
    ref = run_torch(x, w, x_scale, w_scale, dtype)
    w_fly = shuffle_weight_gfx1250(w)
    candidates = {
        "flydsl": lambda: gemm_a8w8_blockscale(x, w_fly, x_scale, w_scale, dtype=dtype),
    }
    if triton_supported(k, dtype):
        candidates["triton"] = run_triton(x, w, x_scale, w_scale, dtype)
    return run_candidates(
        candidates, ref, x, w, x_scale, w_scale, "gemm a8w8 blockscale"
    )


@benchmark()
def test_gemm_a8w8_blockscale_config(
    m,
    n,
    k,
    dtype,
    tile_m,
    tile_n,
    tile_k,
    m_warp,
    n_warp,
    num_buffers,
    memory_bound,
    split_k,
):
    x, w, x_scale, w_scale = generate_inputs(m, n, k)
    ref = run_torch(x, w, x_scale, w_scale, dtype)
    w_fly = shuffle_weight_gfx1250(w)
    parent = torch.full((m + tile_m, n + tile_n), SENTINEL, dtype=dtype)
    y = parent[:m, :n]
    cfg = {
        "y": y,
        "dtype": dtype,
        "tile_m": tile_m,
        "tile_n": tile_n,
        "tile_k": tile_k,
        "m_warp": m_warp,
        "n_warp": n_warp,
        "num_buffers": num_buffers,
        "variant": "memory_bound" if memory_bound else "compute_bound",
        "split_k": split_k,
    }
    assert gemm_a8w8_blockscale(x, w_fly, x_scale, w_scale, **cfg) is y
    candidates = {
        "flydsl": lambda: gemm_a8w8_blockscale(x, w_fly, x_scale, w_scale, **cfg),
    }
    if triton_supported(k, dtype):
        candidates["triton"] = run_triton(x, w, x_scale, w_scale, dtype)
    ret = run_candidates(
        candidates, ref, x, w, x_scale, w_scale, "gemm a8w8 blockscale config"
    )
    assert torch.all(parent[m:] == SENTINEL) and torch.all(
        parent[:, n:] == SENTINEL
    ), "flydsl wrote outside the [m, n] output view"
    return ret


def test_gemm_a8w8_blockscale_guards():
    x, w, x_scale, w_scale = generate_inputs(128, 256, 1024)
    w_fly = shuffle_weight_gfx1250(w)
    for bad_args, exc in (
        ((x.t().contiguous().t(), w_fly, x_scale, w_scale), AssertionError),
        ((x, w_fly, x_scale, w_scale, None, dtypes.bf16), ValueError),
    ):
        try:
            gemm_a8w8_blockscale(*bad_args, variant="compute_bound", split_k=2)
        except exc:
            continue
        raise AssertionError(f"expected {exc.__name__} from gemm_a8w8_blockscale")


def summarize(name, rows):
    aiter.logger.info(
        "%s summary (markdown):\n%s", name, pd.DataFrame(rows).to_markdown(index=False)
    )


def main():
    gfx = get_gfx()
    if gfx not in SUPPORTED_GFX:
        aiter.logger.warning(
            "flydsl a8w8 blockscale gemm unsupported on %s; skipping", gfx
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
        choices=[dtypes.d_dtypes[t] for t in ("bf16", "fp16", "fp32")],
        nargs="*",
        default="bf16,",
        metavar="{bf16,fp16,fp32}",
        help="""Output dtype (inputs are always fp8 e4m3 with f32 block scales).
        e.g.: -d bf16""",
    )
    parser.add_argument(
        "-s",
        "--mnk",
        type=dtypes.str2tuple,
        nargs="*",
        default=DEFAULT_MNK,
        help="""Shape of mnk.
        e.g.: -s 128,1536,7168""",
    )
    parser.add_argument(
        "-l",
        "--layout",
        type=str,
        choices=LAYOUTS,
        nargs="*",
        default=LAYOUTS,
        help="""x / x_scale row layout: dense = contiguous, padded = sliced out of a
        2x wider buffer (runtime lda / lds).
        e.g.: -l dense""",
    )
    parser.add_argument(
        "-c",
        "--config",
        type=dtypes.str2tuple,
        nargs="*",
        default=DEFAULT_CONFIGS,
        help="""Explicit kernel config as
        m,n,k,tile_m,tile_n,tile_k,m_warp,n_warp,num_buffers,memory_bound,split_k
        (memory_bound=1 -> variant memory_bound, else compute_bound).
        e.g.: -c 128,256,1024,128,128,128,2,4,3,1,2""",
    )
    args = parser.parse_args()

    test_gemm_a8w8_blockscale_guards()
    for dtype in args.dtype:
        summarize(
            "gemm_a8w8_blockscale",
            [
                test_gemm_a8w8_blockscale(m, n, k, dtype, layout)
                for layout, (m, n, k) in itertools.product(args.layout, args.mnk)
            ],
        )
        summarize(
            "gemm_a8w8_blockscale kernel config",
            [
                test_gemm_a8w8_blockscale_config(m, n, k, dtype, *tile)
                for (m, n, k, *tile) in args.config
            ],
        )


if __name__ == "__main__":
    main()
