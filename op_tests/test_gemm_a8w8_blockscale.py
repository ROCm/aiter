# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import argparse
import os
import sys
from functools import partial

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import aiter
import pandas as pd
import torch
import torch.nn.functional as F
from aiter import dtypes
from aiter.ops.shuffle import shuffle_weight
from aiter.test_common import checkAllclose
from einops import rearrange

# Import CK / CKTile kernel instance lists from csrc
aiter_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(aiter_root, "csrc", "ck_gemm_a8w8_blockscale"))

from gemm_a8w8_blockscale_instance import candidate_kernels_dict as ck_kernels
from gemm_a8w8_blockscale_cktile_instance import (
    candidate_kernels_cktile_dict as cktile_kernels,
)

block_shape = (128, 128)

# ASM kernel names for gfx950 (from hsa/gfx950/fp8gemm_blockscale/fp8gemm_bf16_blockscale.csv)
# These are all b-preshuffle kernels with tile_n=128 and varying tile_m.
ASM_KERNELS = {
    "asm_128x128": "_ZN5aiter43fp8gemm_bf16_blockscale_BpreShuffle_128x128E",
    "asm_32x128": "_ZN5aiter42fp8gemm_bf16_blockscale_BpreShuffle_32x128E",
    "asm_48x128": "_ZN5aiter42fp8gemm_bf16_blockscale_BpreShuffle_48x128E",
    "asm_64x128": "_ZN5aiter42fp8gemm_bf16_blockscale_BpreShuffle_64x128E",
    "asm_80x128": "_ZN5aiter42fp8gemm_bf16_blockscale_BpreShuffle_80x128E",
    "asm_96x128": "_ZN5aiter42fp8gemm_bf16_blockscale_BpreShuffle_96x128E",
}


def time_kernel_us(fn, num_warmup=3, num_iters=20):
    """Time a callable using CUDA events. Returns average time in microseconds."""
    for _ in range(num_warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(num_iters):
        fn()
    end.record()
    torch.cuda.synchronize()

    return start.elapsed_time(end) * 1000.0 / num_iters


def run_torch_ref(x, weight, x_scale, w_scale, dtype=dtypes.bf16):
    """Reference implementation using PyTorch for correctness verification."""
    block_shape_n, block_shape_k = block_shape
    m, k = x.shape
    n = weight.shape[0]
    scale_n = (n + block_shape_n - 1) // block_shape_n
    scale_k = (k + block_shape_k - 1) // block_shape_k
    x_ = x.to(x_scale.dtype).view(m, k // block_shape[1], block_shape[1]) * x_scale.unsqueeze(-1)
    x_ = x_.view(m, k)

    w_scale_ = rearrange(
        w_scale.view(-1, 1)
        .repeat(1, block_shape_n * block_shape_k)
        .view(scale_n, scale_k, block_shape_n, block_shape_k),
        "num_blk_n num_blk_k blk_n blk_k -> (num_blk_n blk_n) (num_blk_k blk_k)",
    )
    w_scale_ = w_scale_[:n, :k]
    weight_ = weight.to(w_scale_.dtype) * w_scale_

    out = F.linear(x_.to(dtypes.fp32), weight_.to(dtypes.fp32))
    return out.to(dtype)


def sweep_kernels(kernel_dict, run_fn, make_args_fn, ref, category, num_warmup, num_iters, verbose):
    """Generic sweep over a dict of kernel instances/names.

    Args:
        kernel_dict: {id: kernel_or_name} to iterate
        run_fn: callable(out, kid) -> None  that runs the kernel writing into out
        make_args_fn: callable() -> out  that creates a fresh output tensor
        ref: reference output for correctness check
        category: string label for verbose printing
        num_warmup/num_iters: timing parameters
        verbose: print per-instance results

    Returns:
        (all_results, best_result)  where each result is a dict with
        keys: id, name, us, err
    """
    all_results = []
    best = None

    for kid, kernel_or_name in kernel_dict.items():
        name = kernel_or_name.name if hasattr(kernel_or_name, "name") else kernel_or_name
        out = make_args_fn()
        try:
            fn = partial(run_fn, out, kid)
            avg_us = time_kernel_us(fn, num_warmup, num_iters)
            err = checkAllclose(ref, out, printLog=False)
            result = {"id": kid, "name": name, "us": avg_us, "err": err}
            all_results.append(result)
            if err <= 0.05 and (best is None or avg_us < best["us"]):
                best = result
        except Exception as e:
            result = {"id": kid, "name": name, "us": float("inf"), "err": -1}
            all_results.append(result)
            if verbose:
                print(f"    {category}[{kid}] EXCEPTION: {e}")
            continue

    if verbose and all_results:
        from tabulate import tabulate as tabfmt

        rows = []
        for r in all_results:
            is_best = best is not None and r["id"] == best["id"]
            marker = " *" if is_best else ""
            rows.append([
                f"{r['id']}{marker}",
                r["name"],
                f"{r['us']:.1f}" if r["us"] != float("inf") else "FAIL",
                f"{r['err']:.4f}" if r["err"] >= 0 else "ERR",
            ])
        print(f"  [{category}] {len(all_results)} instances:")
        print(tabfmt(rows, headers=["id", "kernel", "us", "err"], tablefmt="simple", stralign="right"))
        print()

    if best is None:
        best = {"id": -1, "name": "NONE", "us": float("inf"), "err": -1}
    return all_results, best


def test_gemm(dtype, m, n, k, num_warmup, num_iters, verbose):
    """Run all kernel categories for a single (M, N, K) shape.
    Returns a list of row-dicts (one per category)."""
    block_shape_n, block_shape_k = block_shape
    scale_n = (n + block_shape_n - 1) // block_shape_n
    scale_k = (k + block_shape_k - 1) // block_shape_k

    x = (torch.rand((m, k), dtype=dtypes.fp32, device="cuda") / 10).to(dtypes.fp8)
    weight = (torch.rand((n, k), dtype=dtypes.fp32, device="cuda") / 10).to(dtypes.fp8)
    x_scale = torch.rand([m, scale_k], dtype=dtypes.fp32, device="cuda")
    w_scale = torch.rand([scale_n, scale_k], dtype=dtypes.fp32, device="cuda")

    weight_shuf = shuffle_weight(weight, layout=(16, 16))
    x_scale_t = x_scale.transpose(0, 1).contiguous().view(*x_scale.shape)

    ref = run_torch_ref(x, weight, x_scale, w_scale, dtype)
    flops = m * n * k * 2

    def make_out():
        return torch.empty(m, n, dtype=dtype, device="cuda")

    def make_row(category, best):
        return {
            "M": m,
            "N": n,
            "K": k,
            "category": category,
            "us": round(best["us"], 2),
            "TFLOPS": round(flops / best["us"] / 1e6, 2) if best["us"] != float("inf") else 0,
            "err": round(best["err"], 4) if best["err"] >= 0 else -1,
            "kernel": best["name"],
        }

    rows = []

    # --- 1. CK (no preshuffle) ---
    _, best_ck = sweep_kernels(
        ck_kernels,
        lambda out, kid: aiter.gemm_a8w8_blockscale_tune(x, weight, x_scale, w_scale, out, kid, 0),
        make_out, ref, "CK", num_warmup, num_iters, verbose,
    )
    rows.append(make_row("CK", best_ck))

    # --- 2. CKTile (no preshuffle) ---
    _, best_cktile = sweep_kernels(
        cktile_kernels,
        lambda out, kid: aiter.gemm_a8w8_blockscale_cktile_tune(x, weight, x_scale, w_scale, out, kid, 0),
        make_out, ref, "CKTile", num_warmup, num_iters, verbose,
    )
    rows.append(make_row("CKTile", best_cktile))

    # --- 3. CKTile (b-preshuffle) ---
    _, best_cktile_p = sweep_kernels(
        cktile_kernels,
        lambda out, kid: aiter.gemm_a8w8_blockscale_bpreshuffle_cktile_tune(
            x, weight_shuf, x_scale_t, w_scale, out, kid, 0,
        ),
        make_out, ref, "CKTile_P", num_warmup, num_iters, verbose,
    )
    rows.append(make_row("CKTile_P", best_cktile_p))

    # --- 4. ASM (b-preshuffle) ---
    asm_dict = {label: label for label in ASM_KERNELS}
    _, best_asm = sweep_kernels(
        asm_dict,
        lambda out, label: aiter.gemm_a8w8_blockscale_bpreshuffle_asm(
            x, weight_shuf, out, x_scale_t, w_scale, kernelName=ASM_KERNELS[label],
        ),
        make_out, ref, "ASM", num_warmup, num_iters, verbose,
    )
    rows.append(make_row("ASM", best_asm))

    return rows


parser = argparse.ArgumentParser(
    formatter_class=argparse.RawTextHelpFormatter,
    description="Comprehensive FP8 blockscale GEMM comparison: CK vs CKTile vs CKTile(preshuffle) vs ASM",
)
parser.add_argument(
    "-d",
    "--dtype",
    type=dtypes.str2Dtype,
    choices=[dtypes.d_dtypes["bf16"]],
    nargs="*",
    default=[dtypes.d_dtypes["bf16"]],
    metavar="{bf16}",
    help="Output data type (only bf16 supported).\n  e.g.: -d bf16",
)
parser.add_argument(
    "-m",
    type=int,
    nargs="*",
    default=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192],
    help="M dimensions to test.\n  e.g.: -m 32 64 128",
)
parser.add_argument(
    "-nk",
    type=dtypes.str2tuple,
    nargs="*",
    default=[
        (24576, 1536),
    ],
    help="(N, K) pairs.\n  e.g.: -nk 24576,1536 7168,16384",
)
parser.add_argument(
    "--num_warmup",
    type=int,
    default=3,
    help="Warmup iterations per kernel instance (default: 3)",
)
parser.add_argument(
    "--num_iters",
    type=int,
    default=20,
    help="Timing iterations per kernel instance (default: 20)",
)
parser.add_argument(
    "--verbose",
    action="store_true",
    help="Print per-instance results during sweep",
)

args = parser.parse_args()

print(f"CK instances: {len(ck_kernels)}, CKTile instances: {len(cktile_kernels)}, ASM kernels: {len(ASM_KERNELS)}")
print(f"Timing: {args.num_warmup} warmup + {args.num_iters} iters per kernel instance")
print()

all_rows = []
for dtype in args.dtype:
    for m in args.m:
        for n, k in args.nk:
            print(f"--- M={m}, N={n}, K={k} ---")
            shape_rows = test_gemm(dtype, m, n, k, args.num_warmup, args.num_iters, args.verbose)
            all_rows.extend(shape_rows)

df = pd.DataFrame(all_rows, columns=["M", "N", "K", "category", "us", "TFLOPS", "err", "kernel"])

print()
print(df.to_markdown(index=False))
print()
aiter.logger.info("gemm_a8w8_blockscale comprehensive summary (markdown):\n%s", df.to_markdown(index=False))
