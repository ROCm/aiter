# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Correctness + perf for the FlyDSL 8-wave kernels (gfx950): the a8w8 ptpc
bpreshuffle GEMM and the fused gather + kv_b_proj built on its pipeline.

The model path (``gemm_a8w8_bpreshuffle`` -> tuned ``flydsl_bpreshuffle_8w_*`` row):

    xq  : [M, K]  fp8, per-token fp32 scale [M, 1]
    wq  : [N, K]  fp8 in shuffle_weight (16, 16) layout, per-channel scale [N, 1]
    out : [M, N]  bf16, preallocated

The gather path (``gather_kv_b_proj_flydsl``, MLA prefix expansion):

    k_buffer : [num_blocks, 1, 576] fp8 cache, gathered by kv_indices [num_tokens]
    wq       : [heads * (nope + v), 512] fp8 in shuffle_weight (16, 16) layout
    k_prefix : [num_tokens, heads, nope + 64] bf16, preallocated
    v_prefix : [num_tokens, heads, v] bf16, preallocated

A cross-wave LDS race on the last K tile used to corrupt 16-64 row (or column)
bands of a GEMM tile on some calls only; the gather kernel shares that pipeline
tail. So every kernel is launched ``--repeat`` times on the same inputs into a
NaN-filled output and each result is checked.

Run:
    python op_tests/test_flydsl_gemm_a8w8_8wave.py
    python op_tests/test_flydsl_gemm_a8w8_8wave.py -s 12288,6144,7168 -r 50
    python op_tests/test_flydsl_gemm_a8w8_8wave.py -g 16384,16,128,128 -r 100
"""

import argparse
import itertools

import pandas as pd
import torch

import aiter
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl.gather_kv_b_proj import gather_kv_b_proj_flydsl
from aiter.ops.gemm_op_a8w8 import gemm_a8w8_bpreshuffle_flydsl
from aiter.ops.shuffle import shuffle_weight
from aiter.test_common import (
    benchmark,
    checkAllclose,
    run_perftest,
)

torch.set_default_device("cuda")

# The 8-wave kernel uses the CDNA4 MFMA_Scale atom: gfx950 only.
SUPPORTED_GFX = ["gfx950"]
SEED = 0

_KERNELS = [
    "flydsl_bpreshuffle_8w_256x256x128_F8_F8_B16_2x4",
    "flydsl_bpreshuffle_8w_256x256x128_F8_F8_B16_2x0",
    "flydsl_bpreshuffle_8w_128x256x128_F8_F8_B16_1x8",
]

# Tuned kimik3 shapes; before the fix (32768, 6400, 7168) failed on ~50% of calls.
_MNK = [
    (32768, 6400, 7168),
    (12288, 6144, 7168),
    (1000, 7168, 3584),
]

KV_C_DIM = 512
KV_PE_DIM = 64

# (num_tokens, heads, qk_nope_head_dim, v_head_dim): DeepSeek MLA 128+128 at 16
# heads per rank, on both the 128-row (2048) and 256-row (16384) M tile, and the
# GLM-5.2 192+256 head, which splits each head over two workgroups.
_GATHER = [
    (2048, 16, 128, 128),
    (16384, 16, 128, 128),
    (8192, 16, 192, 256),
]


def run_torch(xq, wq, x_scale, w_scale, dtype=dtypes.bf16):
    # Reference only: fp32 dequant matmul, cast back. Not timed, not in the table.
    out = (xq.to(dtypes.fp32) @ wq.to(dtypes.fp32).t()) * x_scale * w_scale.t()
    return out.to(dtype)


def run_torch_gather(k_buffer, kv_indices, wq, w_scale, heads, nope, v_dim):
    # Reference only: fp32 gather + dequant matmul, k_scale = 1. Not timed.
    latent = k_buffer[kv_indices.long()].reshape(-1, KV_C_DIM + KV_PE_DIM).float()
    kv_c, k_pe = latent.split([KV_C_DIM, KV_PE_DIM], dim=-1)
    proj = (kv_c @ (wq.to(dtypes.fp32) * w_scale).t()).view(-1, heads, nope + v_dim)
    rope = k_pe.unsqueeze(1).expand(-1, heads, -1)
    return torch.cat([proj[..., :nope], rope], dim=-1), proj[..., nope:]


@benchmark()
def test_gemm_8wave(m, n, k, kernel, repeat):
    torch.manual_seed(SEED)
    x = torch.randn((m, k), dtype=dtypes.bf16)
    w = torch.randn((n, k), dtype=dtypes.bf16)
    xq, x_scale = aiter.pertoken_quant(x, quant_dtype=dtypes.fp8)
    wq, w_scale = aiter.pertoken_quant(w, quant_dtype=dtypes.fp8)
    wq_shuffled = shuffle_weight(wq, layout=(16, 16))
    out = torch.empty((m, n), dtype=dtypes.bf16)
    ref = run_torch(xq, wq, x_scale, w_scale).to(dtypes.fp32)

    def _run():
        return gemm_a8w8_bpreshuffle_flydsl(
            xq, wq_shuffled, x_scale, w_scale, out, {"kernelName": kernel}
        )

    # The race is intermittent: check every launch, not just the last one.
    bad_calls = 0
    for _ in range(repeat):
        out.fill_(float("nan"))
        y = _run().to(dtypes.fp32)
        if (
            checkAllclose(
                ref, y, rtol=2e-2, atol=5e-2, tol_err_ratio=0.0, printLog=False
            )
            > 0
        ):
            bad_calls += 1

    y, us = run_perftest(_run)
    err = checkAllclose(
        ref, y.to(dtypes.fp32), rtol=2e-2, atol=5e-2, msg=f"{kernel}: 8wave gemm"
    )
    flops = 2 * m * n * k
    nbytes = m * k + n * k + m * n * out.element_size()
    return {
        "gfx": get_gfx(),
        "8wave us": us,
        "8wave TFLOPS": flops / us / 1e6,
        "8wave TB/s": nbytes / us / 1e6,
        "8wave err": err,
        "8wave bad calls": bad_calls,
    }


@benchmark()
def test_gather_kv_b_proj_8wave(num_tokens, heads, nope, v_dim, repeat):
    torch.manual_seed(SEED)
    num_blocks = 2 * num_tokens
    k_buffer = torch.randn((num_blocks, 1, KV_C_DIM + KV_PE_DIM)).to(dtypes.fp8)
    kv_indices = torch.randperm(num_blocks)[:num_tokens].to(torch.int32)
    kv_indptr = torch.tensor([0, num_tokens], dtype=torch.int32)
    k_scale = torch.ones(1, dtype=dtypes.fp32)
    weight_n = heads * (nope + v_dim)
    wq = torch.randn((weight_n, KV_C_DIM)).to(dtypes.fp8)
    w_scale = torch.rand((weight_n, 1), dtype=dtypes.fp32) + 0.5
    wq_shuffled = shuffle_weight(wq, layout=(16, 16))
    k_prefix = torch.empty((num_tokens, heads, nope + KV_PE_DIM), dtype=dtypes.bf16)
    v_prefix = torch.empty((num_tokens, heads, v_dim), dtype=dtypes.bf16)
    k_ref, v_ref = run_torch_gather(
        k_buffer, kv_indices, wq, w_scale, heads, nope, v_dim
    )

    def _run():
        gather_kv_b_proj_flydsl(
            k_buffer,
            k_scale,
            kv_indptr,
            kv_indices,
            kv_indptr,
            wq_shuffled,
            w_scale,
            k_prefix,
            v_prefix,
        )
        return k_prefix, v_prefix

    def _check(out, printLog=False):
        return max(
            checkAllclose(
                ref,
                y.to(dtypes.fp32),
                rtol=2e-2,
                atol=5e-2,
                tol_err_ratio=0.0,
                printLog=printLog,
                msg=f"gather_kv_b_proj 8wave {name}",
            )
            for name, ref, y in zip(("k_prefix", "v_prefix"), (k_ref, v_ref), out)
        )

    # The race is intermittent: check every launch, not just the last one.
    bad_calls = 0
    for _ in range(repeat):
        k_prefix.fill_(float("nan"))
        v_prefix.fill_(float("nan"))
        if _check(_run()) > 0:
            bad_calls += 1

    out, us = run_perftest(_run)
    err = _check(out, printLog=True)
    flops = 2 * num_tokens * weight_n * KV_C_DIM
    nbytes = (
        num_tokens * (KV_C_DIM + KV_PE_DIM)
        + weight_n * KV_C_DIM
        + (k_prefix.numel() + v_prefix.numel()) * k_prefix.element_size()
    )
    return {
        "gfx": get_gfx(),
        "gather us": us,
        "gather TFLOPS": flops / us / 1e6,
        "gather TB/s": nbytes / us / 1e6,
        "gather err": err,
        "gather bad calls": bad_calls,
    }


def main():
    if get_gfx() not in SUPPORTED_GFX:
        aiter.logger.warning("flydsl 8wave gemm unsupported on %s; skipping", get_gfx())
        return

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="config input of test",
    )
    parser.add_argument(
        "-s",
        "--mnk",
        type=dtypes.str2tuple,
        nargs="*",
        default=_MNK,
        help="""Shape of mnk.
        e.g.: -s 32768,6400,7168""",
    )
    parser.add_argument(
        "-k",
        "--kernel",
        type=str,
        nargs="*",
        default=_KERNELS,
        help="""8-wave kernelName(s).
        e.g.: -k flydsl_bpreshuffle_8w_256x256x128_F8_F8_B16_2x4""",
    )
    parser.add_argument(
        "-g",
        "--gather",
        type=dtypes.str2tuple,
        nargs="*",
        default=_GATHER,
        help="""Gather shape: num_tokens,heads,qk_nope_head_dim,v_head_dim.
        e.g.: -g 16384,16,128,128""",
    )
    parser.add_argument(
        "-r",
        "--repeat",
        type=int,
        nargs="*",
        default=[20],
        help="""Launches checked per shape and kernel.
        e.g.: -r 50""",
    )
    args = parser.parse_args()

    gemm_rows = [
        test_gemm_8wave(m, n, k, kernel, repeat)
        for (m, n, k), kernel, repeat in itertools.product(
            args.mnk, args.kernel, args.repeat
        )
    ]
    gather_rows = [
        test_gather_kv_b_proj_8wave(*shape, repeat)
        for shape, repeat in itertools.product(args.gather, args.repeat)
    ]

    failed = []
    for name, rows, prefix in (
        ("gemm", gemm_rows, "8wave"),
        ("gather_kv_b_proj", gather_rows, "gather"),
    ):
        if not rows:
            continue
        df = pd.DataFrame(rows)
        aiter.logger.info(
            "flydsl 8wave %s summary (markdown):\n%s", name, df.to_markdown(index=False)
        )
        bad = df[(df[f"{prefix} bad calls"] > 0) | (df[f"{prefix} err"] > 0)]
        if not bad.empty:
            failed.append(f"{name}:\n{bad.to_string()}")
    if failed:
        raise AssertionError(
            "flydsl 8wave kernels mismatched the reference:\n" + "\n".join(failed)
        )


if __name__ == "__main__":
    main()
