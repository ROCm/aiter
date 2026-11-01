# SPDX-License-Identifier: MIT

import argparse
import importlib.metadata
import json
import statistics
from pathlib import Path

import torch

from aiter.ops.flydsl.gemm_a4w4 import (
    flydsl_gemm_a4w4,
    prepare_gemm_a4w4_weight,
)
from aiter.ops.quant import per_1x32_f4_quant, quant_mxfp4_hip
from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import gemm_afp4wfp4
from aiter.test_common import run_perftest
from aiter.utility import fp4_utils
from aiter.utility.mx_types import MxScaleRoundModeInt


def _decode(packed, scales):
    return fp4_utils.mxfp4_to_f32(packed) * fp4_utils.e8m0_to_f32(
        scales
    ).repeat_interleave(32, -1)


def _relative_l2(actual, expected):
    if not torch.isfinite(actual).all():
        raise ValueError("Nonfinite benchmark output")
    return (
        (actual.float() - expected).norm() / expected.norm().clamp_min(1e-30)
    ).item()


def _paired_timings(functions, *, mode, calls, pairs):
    graphs = {}
    if mode == "graph":
        for name, function in functions.items():
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for _ in range(calls):
                    function()
            graph.replay()
            torch.cuda.synchronize()
            graphs[name] = graph
    samples = []
    for pair in range(pairs):
        order = list(functions)
        if pair % 2:
            order.reverse()
        sample = {"pair": pair, "order": order, "us": {}}
        for name in order:
            function = graphs[name].replay if mode == "graph" else functions[name]
            _, elapsed = run_perftest(
                function,
                num_warmup=1,
                num_iters=1 if mode == "graph" else calls,
                num_rotate_args=1,
                use_cuda_event=True,
            )
            sample["us"][name] = elapsed / calls if mode == "graph" else elapsed
        samples.append(sample)
    return samples


def main() -> None:
    parser = argparse.ArgumentParser(description="Dense FlyDSL inline A4W4 benchmark")
    parser.add_argument("--shape", default="16,1536,768")
    parser.add_argument("--bm", type=int, choices=(16, 64), default=None)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--rep", type=int, default=50)
    parser.add_argument("--pairs", type=int, default=7)
    parser.add_argument("--mode", choices=("graph", "event"), default="graph")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    m, n, k = map(int, args.shape.split(","))
    if min(m, n, k, args.rep, args.pairs) <= 0 or args.warmup < 1:
        parser.error("shape, repetitions, pairs and warmup must be positive")

    from aiter.ops.triton.quant import dynamic_mxfp4_quant

    torch.manual_seed(4782)
    a = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
    w, w_scale = per_1x32_f4_quant(
        torch.randn((n, k), device="cuda", dtype=torch.bfloat16),
        round_mode=MxScaleRoundModeInt.RoundUp,
    )
    prepared = prepare_gemm_a4w4_weight(w, w_scale)
    outputs = {
        name: torch.empty((m, n), device=a.device, dtype=torch.bfloat16)
        for name in ("flydsl_roundup", "hip_roundup_triton", "triton_even")
    }

    def direct():
        return flydsl_gemm_a4w4(a, prepared, _bm=args.bm, out=outputs["flydsl_roundup"])

    def split_roundup():
        quantized, scales = quant_mxfp4_hip(a, round_mode=MxScaleRoundModeInt.RoundUp)
        return gemm_afp4wfp4(
            quantized.view(torch.uint8),
            w.view(torch.uint8),
            scales.view(torch.uint8),
            w_scale.view(torch.uint8),
            dtype=torch.bfloat16,
            y=outputs["hip_roundup_triton"],
            backend="triton",
        )

    def split_even():
        quantized, scales = dynamic_mxfp4_quant(a, backend="triton")
        return gemm_afp4wfp4(
            quantized,
            w.view(torch.uint8),
            scales,
            w_scale.view(torch.uint8),
            dtype=torch.bfloat16,
            y=outputs["triton_even"],
            backend="triton",
        )

    functions = {
        "flydsl_roundup": direct,
        "hip_roundup_triton": split_roundup,
        "triton_even": split_even,
    }
    decoded_weight = _decode(w, w_scale)
    reference = {}
    for name, rounding in (
        ("roundup", MxScaleRoundModeInt.RoundUp),
        ("even", MxScaleRoundModeInt.Even),
    ):
        quantized, scales = per_1x32_f4_quant(a, round_mode=rounding)
        reference[name] = _decode(quantized, scales) @ decoded_weight.T
    bf16_reference = a.float() @ decoded_weight.T
    errors = {}
    for name, function in functions.items():
        actual = function()
        expected = reference["even" if name == "triton_even" else "roundup"]
        errors[name] = {
            "matched_reference": _relative_l2(actual, expected),
            "bf16_reference": _relative_l2(actual, bf16_reference),
        }
        if errors[name]["matched_reference"] >= 0.01:
            raise ValueError(
                f"{name} exceeds 1% matched-contract relative-L2: {errors[name]}"
            )

    for _ in range(args.warmup):
        for function in functions.values():
            function()
    torch.cuda.synchronize()
    samples = _paired_timings(
        functions, mode=args.mode, calls=args.rep, pairs=args.pairs
    )
    for name, actual in outputs.items():
        expected = reference["even" if name == "triton_even" else "roundup"]
        if _relative_l2(actual, expected) >= 0.01:
            raise ValueError(f"{name} failed post-timing numerical validation")
    medians = {
        name: statistics.median(sample["us"][name] for sample in samples)
        for name in functions
    }
    result = {
        "shape": [m, n, k],
        "bm_override": args.bm,
        "mode": args.mode,
        "calls": args.rep,
        "pairs": args.pairs,
        "samples": samples,
        "median_us": medians,
        "relative_l2": errors,
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "flydsl": importlib.metadata.version("flydsl"),
        "gpu": torch.cuda.get_device_name(),
        "scope": "Prepared weights and outputs; JIT excluded. Split APIs allocate quant intermediates. "
        "Even is a different activation rounding contract, not an equivalent-precision baseline. "
        "Event timing includes wrapper launch gaps; graph timing does not.",
    }
    if args.output:
        with args.output.open("x") as stream:
            json.dump(result, stream, indent=2)
    print(json.dumps(result, indent=2))
    print("| Candidate | us | TFLOPS | minimum tensor traffic TB/s | matched rel-L2 |")
    print("|---|---:|---:|---:|---:|")
    nbytes = a.numel() * a.element_size() + w.numel() + w_scale.numel() + m * n * 2
    for name, elapsed in medians.items():
        print(
            f"| {name} | {elapsed:.3f} | {2*m*n*k/elapsed/1e6:.2f} | "
            f"{nbytes/elapsed/1e6:.3f} | {errors[name]['matched_reference']:.6f} |"
        )
    print(
        "Traffic is a lower bound excluding split intermediates; graph and event samples are not pooled."
    )


if __name__ == "__main__":
    main()
