# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# Minimal rocprofv3/ATT driver for DSv4 wo_a fp8/e8m0 mxscale flatmm kernels.
# Runs exactly one selected no-split kernel path repeatedly so profiler output is
# not mixed with the broader op_tests/test_uniform_scale_wo_a candidate sweep.

import argparse

import torch
from aiter import dtypes
from aiter.ops.opus.bmm_op import _opus_bmm_a8w8_mxscale_flatmm_splitk_mmajor_raw

from op_tests.test_uniform_scale_wo_a import (
    _quant_block_e8m0,
    _quant_per_token_e8m0,
)

torch.set_default_device("cuda")


KIDS = {
    "m64n32k256": 320,
    "m32n64k256": 640,
    "m32n64k256_selfload": 646,
    "m64n64k128": 650,
    "m64n64k128_scale_prefetch": 653,
    "flatmm64": 64,
    "m128n128k128_wg1": 128,
    "m128n128k128_scale_prefetch": 137,
    "m64n128k256_wg1": 138,
    "m128n64k256_wg1": 139,
    "m64n256k128_nphase": 129,
    "m128n128k128_persistent_mouter_wg1": 131,
    "m128n128k128_persistent_mouter_wg1_skip_scale_wait": 144,
    "m128n256k128_wave8n2": 132,
    "m128n256k128_wave4n2_selfload": 133,
    "m128n256k128_wave4n2_selfload_issue_next": 140,
    "m128n256k128_wave4n2_selfload_skip_scale_wait": 141,
    "m128n256k128_wave4n2_selfload_single_lds": 145,
    "m128n256k128_wave4n2_selfload_issue_after_mma": 146,
    "m128n256k128_wave4n2_selfload_on_demand_scale_pack": 147,
    "m256n128k128_wave4m2_selfload": 134,
    "m256n128k128_wave4m2_selfload_skip_scale_wait": 142,
    "m256n128k128_wave4m2_selfload_on_demand_scale_pack": 148,
    "m512n256k256_scale_pipeline": 149,
    "m256n256k128_scale_pipeline": 150,
    "m256n256k1024_scale_pipeline": 151,
    "m256n256k1024_scale_pipeline_lb1": 152,
    "m64n32k256_wg1": 322,
    "m32n64k256_wg1": 642,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=512)
    parser.add_argument("--n", type=int, default=1024)
    parser.add_argument("--k", type=int, default=4096)
    parser.add_argument("--g", type=int, default=2)
    parser.add_argument("--variant", choices=KIDS.keys(), default="m32n64k256")
    parser.add_argument("--dtype", choices=["fp32", "bf16"], default="bf16")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    args = parser.parse_args()

    ydt = dtypes.fp32 if args.dtype == "fp32" else dtypes.bf16
    O_bf16 = (torch.rand((args.g, args.m, args.k), dtype=dtypes.fp32) / 10).to(
        dtypes.bf16
    )
    W_bf16 = (torch.rand((args.g, args.n, args.k), dtype=dtypes.fp32) / 10).to(
        dtypes.bf16
    )
    O_mx, xs_mx, _ = _quant_per_token_e8m0(O_bf16)
    W_mx, ws_mx, _ = _quant_block_e8m0(W_bf16)

    O_mx_in = O_mx.transpose(0, 1)
    xs_mx_in = xs_mx.transpose(0, 1)
    Y = torch.empty((args.m, args.g, args.n), dtype=ydt)
    kid = KIDS[args.variant]

    def run_once():
        _opus_bmm_a8w8_mxscale_flatmm_splitk_mmajor_raw(
            O_mx_in, W_mx, Y, xs_mx_in, ws_mx, 1, kid
        )

    for _ in range(args.warmup):
        run_once()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(args.iters):
        run_once()
    end.record()
    torch.cuda.synchronize()

    avg_us = start.elapsed_time(end) * 1000.0 / args.iters
    tflops = (2.0 * args.m * args.n * args.k * args.g) / (avg_us * 1.0e6)
    print(
        f"[profile_mx_flatmm] variant={args.variant} kid={kid} splitK=1 "
        f"shape=({args.m},{args.n},{args.k}) g={args.g} dtype={args.dtype} "
        f"iters={args.iters} avg_us={avg_us:.3f} tflops={tflops:.2f}"
    )


if __name__ == "__main__":
    main()
