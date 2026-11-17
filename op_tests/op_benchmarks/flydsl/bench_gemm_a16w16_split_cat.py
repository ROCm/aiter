# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""Benchmark gfx1250 BF16 GEMM + MLA split/cat against its fused epilogue.

The default is the DeepSeek-R1 prefill trace point:
    python op_tests/op_benchmarks/flydsl/bench_gemm_a16w16_split_cat.py

Exercise all requested M tails:
    python op_tests/op_benchmarks/flydsl/bench_gemm_a16w16_split_cat.py --all-m
"""

import argparse

import pandas as pd
import torch

import aiter
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl.gemm_a16w16_gfx1250 import gemm_a16w16
from aiter.test_common import run_perftest

HEADS = 128
K_DIM = 512
K_NOPE = 128
V_DIM = 128
ROPE_DIM = 64
N_DIM = HEADS * (K_NOPE + V_DIM)
M_TAILS = (1, 31, 32, 128, 1024, 9424, 16384)

CONFIG = {
    "tile_m": 128,
    "tile_n": 256,
    "tile_k": 64,
    "m_warp": 2,
    "n_warp": 4,
    "num_buffers": 2,
}


def _pitched_output(m, width, row_pad):
    return torch.empty_strided(
        (m, HEADS, width),
        (HEADS * width + row_pad, width, 1),
        dtype=torch.bfloat16,
        device="cuda",
    )


def bench_case(m, row_pad):
    torch.manual_seed(0)
    x = torch.randn((m, K_DIM), dtype=torch.bfloat16, device="cuda")
    w = torch.randn((N_DIM, K_DIM), dtype=torch.bfloat16, device="cuda")
    rope = torch.randn((m, 1, ROPE_DIM), dtype=torch.bfloat16, device="cuda")
    k_out = _pitched_output(m, K_NOPE + ROPE_DIM, row_pad)
    v_out = _pitched_output(m, V_DIM, row_pad)

    def baseline():
        projected = gemm_a16w16(x, w, dtype=torch.bfloat16, **CONFIG).view(
            m, HEADS, K_NOPE + V_DIM
        )
        k = torch.cat((projected[..., :K_NOPE], rope.expand(-1, HEADS, -1)), dim=-1)
        return k, projected[..., K_NOPE:]

    def fused():
        return gemm_a16w16(
            x,
            w,
            dtype=torch.bfloat16,
            k_out=k_out,
            v_out=v_out,
            k_rope=rope,
            split_sizes=(K_NOPE, V_DIM),
            **CONFIG,
        )

    ref_k, ref_v = baseline()
    actual_k, actual_v = fused()
    if not torch.equal(actual_k, ref_k) or not torch.equal(actual_v, ref_v):
        raise AssertionError(f"M={m}: fused output is not bitwise equal to baseline")

    _, baseline_us = run_perftest(baseline)
    _, fused_us = run_perftest(fused)
    return {
        "M": m,
        "baseline us": round(baseline_us, 2),
        "fused us": round(fused_us, 2),
        "saved us": round(baseline_us - fused_us, 2),
        "speedup": round(baseline_us / fused_us, 3),
        "row pad": row_pad,
    }


def main():
    if get_gfx() != "gfx1250":
        aiter.logger.warning("gfx1250 required; found %s", get_gfx())
        return
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--m", type=int, nargs="+", default=[9424])
    parser.add_argument(
        "--all-m", action="store_true", help="benchmark 1,31,32,128,1024,9424,16384"
    )
    parser.add_argument(
        "--row-pad",
        type=int,
        default=0,
        help="extra BF16 elements between output rows",
    )
    args = parser.parse_args()
    cases = M_TAILS if args.all_m else args.m
    rows = [bench_case(m, args.row_pad) for m in cases]
    aiter.logger.info(
        "gfx1250 BF16 GEMM split-cat:\n%s",
        pd.DataFrame(rows).to_markdown(index=False),
    )


if __name__ == "__main__":
    main()
