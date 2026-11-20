# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Standalone six-plane reduce + RMSNorm, not a full GEMM/block benchmark.

Comparator: existing AITER Triton split-K reducer (its native 32x32 tile), then
existing fused_qk_rmsnorm. Same FP32 partials, BF16 projection and both normalized
outputs, preallocated for both arms. The comparator uses a tree sum; the HIP op
uses a left-to-right sum. Both must pass an independent FP32 reference check.

The HIP reducer accepts arbitrary M; its future Gluon producer's M128/M256
restriction does not apply here. M4..1024 covers C1..256 with four verification
rows per request. M1/M2 are additional operator sizes; M0 is correctness-only.

Clean timing: triton.testing.do_bench median with its cache flush, three
interleaved repetitions. Profiles are separate, never used for timing credit.
"""

import argparse
import json
from pathlib import Path

import torch
import triton
from triton.testing import do_bench

from aiter import splitk_reduce_qk_rmsnorm
from aiter.ops.enum import QuantType
from aiter.ops.fused_qk_rmsnorm_group_quant import fused_qk_rmsnorm
from aiter.ops.triton._triton_kernels.common.splitk_reduce import (
    _gemm_splitk_reduce_kernel,
)


def make_case(m):
    generator = torch.Generator(device="cuda").manual_seed(2624 + m)
    partial = torch.randn((6, m, 2624), device="cuda", generator=generator)
    qw = (torch.rand(2048, device="cuda", generator=generator) + 0.5).bfloat16()
    kw = (torch.rand(512, device="cuda", generator=generator) + 0.5).bfloat16()
    outputs = {
        arm: tuple(
            torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
            for n in (2624, 2048, 512)
        )
        for arm in ("separate", "fused")
    }

    def separate():
        out, q, k = outputs["separate"]
        _gemm_splitk_reduce_kernel[(triton.cdiv(m, 32), 82)](
            partial,
            out,
            None,
            m,
            2624,
            *partial.stride(),
            *out.stride(),
            32,
            32,
            6,
            8,
            ADD_BIAS=False,
            activation="",
            use_activation=False,
        )
        fused_qk_rmsnorm(
            q_out_quantized=q,
            k_out=k,
            q=out[:, :2048],
            q_weight=qw,
            q_epsilon=1e-5,
            k=out[:, 2048:2560],
            k_weight=kw,
            k_epsilon=1e-5,
            quant_type=QuantType.No,
        )

    def fused():
        out, q, k = outputs["fused"]
        splitk_reduce_qk_rmsnorm(partial, qw, 1e-5, kw, 1e-5, out=out, q_out=q, k_out=k)

    ref = partial.sum(0)
    refs = [ref]
    # Both paths normalize their once-BF16-rounded projection.
    rounded = ref.bfloat16().float()
    for x, w in ((rounded[:, :2048], qw), (rounded[:, 2048:2560], kw)):
        refs.append(
            x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-5) * w.float()
        )
    metrics = {}
    for arm, fn in (("separate", separate), ("fused", fused)):
        fn()
        metrics[arm] = []
        for actual, expected in zip(outputs[arm], refs):
            torch.testing.assert_close(actual.float(), expected, atol=0.02, rtol=0.02)
            error = (actual.float() - expected).square().mean().sqrt()
            nrmse = (error / expected.square().mean().sqrt()).item()
            assert nrmse <= 0.01, (arm, nrmse)
            metrics[arm].append(nrmse)
    return {"separate": separate, "fused": fused}, metrics


def clean_median_ms(fn):
    result = do_bench(fn, warmup=25, rep=100, quantiles=[0.5])
    # Triton versions return either a scalar for one quantile or a sequence.
    return float(result[0] if isinstance(result, (list, tuple)) else result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-M",
        nargs="+",
        type=int,
        default=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024],
        choices=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024],
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile-dir", type=Path)
    args = parser.parse_args()
    report = {
        "scope": "six FP32 partial planes to BF16 projection and two RMSNorm outputs",
        "timer": "triton.testing.do_bench quantiles=[0.5], warmup=25ms, rep=100ms",
        "cache_condition": "do_bench cache flush per timing iteration",
        "comparator": "AITER _gemm_splitk_reduce_kernel 32x32 six splits padded to eight + fused_qk_rmsnorm",
        "torch": torch.__version__,
        "triton": triton.__version__,
        "device": torch.cuda.get_device_name(),
        "cases": [],
    }
    for m in args.M:
        arms, metrics = make_case(m)
        rows = []
        for repetition in range(3):
            for arm, fn in arms.items():
                median_ms = clean_median_ms(fn)
                rows.append(
                    {
                        "repetition": repetition,
                        "arm": arm,
                        "median_us": median_ms * 1000,
                    }
                )
        if args.profile_dir:
            args.profile_dir.mkdir(parents=True, exist_ok=True)
            for arm, fn in arms.items():
                # All JIT and correctness work happened before profiling.
                with torch.profiler.profile(
                    activities=[
                        torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA,
                    ]
                ) as prof:
                    fn()
                    torch.cuda.synchronize()
                prof.export_chrome_trace(
                    str(args.profile_dir / f"m{m}-{arm}.trace.json")
                )
        report["cases"].append({"m": m, "nrmse_projection_q_k": metrics, "clean": rows})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
