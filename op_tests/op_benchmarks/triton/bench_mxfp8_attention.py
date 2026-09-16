# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Benchmark for the MXFP8 Flash Attention v2 kernel (attn_fwd_mxfp8).

Usage:
    python bench_mxfp8_attention.py
    python bench_mxfp8_attention.py --causal
    python bench_mxfp8_attention.py -metric bandwidth
"""

import argparse
import math

import torch
import triton

from aiter.ops.triton.attention.mxfp8_attention import mxfp8_attention_forward
from aiter.ops.triton.quant.quant_mxfp8 import convert_to_mxfp8
from aiter.ops.triton.utils._triton.arch_info import get_arch
from op_tests.op_benchmarks.triton.utils.benchmark_utils import get_caller_name_no_ext

_QUANT_BLOCK = 32
_FP8 = torch.float8_e4m3fn

# Representative shapes: (B, H, S, D)
_SHAPES = [
    (1, 32, 512, 128),
    (1, 32, 1024, 128),
    (1, 32, 2048, 128),
    (1, 32, 4096, 128),
    (4, 32, 1024, 128),
    (4, 32, 2048, 128),
]


def _quantize_bshd(x, quant_block_size=_QUANT_BLOCK):
    """2D-block MXFP8 quantize (B, S, H, D) -> fp8 + e8m0 (B, S/qbs, H, D/qbs)."""
    b, s, h, d = x.shape
    x2 = x.permute(0, 2, 1, 3).contiguous().reshape(b * h * s, d)
    y, scale = convert_to_mxfp8(
        x2,
        _FP8,
        quant_block_size=quant_block_size,
        is_2d_block=True,
    )
    y = y.reshape(b, h, s, d).permute(0, 2, 1, 3).contiguous()
    scale = (
        scale.reshape(b, h, s // quant_block_size, d // quant_block_size)
        .permute(0, 2, 1, 3)
        .contiguous()
    )
    return y, scale


def _make_inputs(B, H, S, D, dtype, quant_block_size=_QUANT_BLOCK, device="cuda"):
    q_hp = torch.randn(B, S, H, D, dtype=dtype, device=device) * 0.1
    k_hp = torch.randn(B, S, H, D, dtype=dtype, device=device) * 0.1
    v_hp = torch.randn(B, S, H, D, dtype=dtype, device=device) * 0.1
    q, q_scale = _quantize_bshd(q_hp, quant_block_size)
    k, k_scale = _quantize_bshd(k_hp, quant_block_size)
    v, v_scale = _quantize_bshd(v_hp, quant_block_size)
    return q, k, v, q_scale, k_scale, v_scale


def benchmark(args):
    arch = get_arch()
    if arch != "gfx950":
        print(f"Skipping: MXFP8 attention requires gfx950 (CDNA4), got {arch}")
        return

    causal = args.causal
    unit = "ms" if args.metric == "time" else "GB/s"
    dtype = torch.bfloat16
    quant_block_size = 32

    x_vals = [(B, H, S, D) for B, H, S, D in _SHAPES]

    config = triton.testing.Benchmark(
        x_names=["B", "H", "S", "D"],
        x_vals=x_vals,
        line_arg="provider",
        line_vals=["attn_fwd_mxfp8"],
        line_names=[
            f"attn_fwd_mxfp8 ({'causal' if causal else 'non-causal'}) ({unit})"
        ],
        styles=[("green", "-")],
        ylabel=unit,
        plot_name=get_caller_name_no_ext(),
        args={},
    )

    @triton.testing.perf_report([config])
    def _run(B, H, S, D, provider):
        q, k, v, q_scale, k_scale, v_scale = _make_inputs(
            B, H, S, D, dtype, quant_block_size
        )
        sm_scale = 1.0 / math.sqrt(D)
        fn = lambda: mxfp8_attention_forward(
            q,
            k,
            v,
            q_scale=q_scale,
            k_scale=k_scale,
            v_scale=v_scale,
            sm_scale=sm_scale,
            causal=causal,
            use_mxfp8=True,
            block_m=64,
            block_n=64,
            quant_block_size=quant_block_size,
            layout="bshd",
        )
        # reads: Q/K/V (fp8) + e8m0 scales; writes: Out (bf16)
        n_elem = B * H * S * D
        n_scale = B * H * (S // quant_block_size) * (D // quant_block_size)
        mem = n_elem * 3 * q.element_size() + n_scale * 3 + n_elem * 2

        ms = triton.testing.do_bench(fn, warmup=25, rep=100)
        if args.metric == "time":
            return ms
        return mem * 1e-9 / (ms * 1e-3)

    _run.run(save_path="." if args.o else None, print_data=True, show_plots=False)


def parse_args():
    parser = argparse.ArgumentParser(
        prog="Benchmark MXFP8 Flash Attention v2 kernel", allow_abbrev=False
    )
    parser.add_argument("--causal", action="store_true", default=False)
    parser.add_argument(
        "-metric",
        nargs="?",
        const="time",
        choices=["time", "bandwidth"],
        default="time",
    )
    parser.add_argument("-o", action="store_true", default=False)
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(0)
    benchmark(args)


if __name__ == "__main__":
    main()
