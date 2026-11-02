# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Compare actual dense dispatch with its original CK/ASM fallback on gfx950."""

import argparse
import gc
import hashlib
import json
import math
import statistics
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch

import torch

import aiter
from aiter.jit.utils.chip_info import get_cu_num, get_gfx_runtime
from aiter.ops import gemm_op_a4w4 as dispatch
from aiter.ops.shuffle import shuffle_weight
from aiter.utility import dtypes, fp4_utils


def byte_equal(a, b):
    return torch.equal(
        a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)
    )


def check_output(value, reference, baseline):
    error = (value.float() - reference).norm()
    norm = reference.norm()
    l2 = float(error / norm) if norm else float(error)
    result = {
        "finite": bool(value.isfinite().all()),
        "relative_l2": l2,
        "bit_exact": byte_equal(value, baseline),
    }
    assert (
        result["finite"] and (l2 < 0.01 if norm else l2 == 0) and result["bit_exact"]
    ), result
    return result


def kernel_names(fn):
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA]
    ) as prof:
        fn()
        torch.cuda.synchronize()
    return sorted(
        {
            event.name
            for event in prof.events()
            if event.device_type == torch.autograd.DeviceType.CUDA
        }
    )


def measure_shape(shape, seed, samples, eviction):
    m, n, k = shape
    torch.manual_seed(seed)
    x = torch.randn((m, k), dtype=torch.bfloat16, device="cuda")
    w = torch.randn((n, k), dtype=torch.bfloat16, device="cuda")
    quant = aiter.get_hip_quant(aiter.QuantType.per_1x32)
    aq, a_scale = quant(x, quant_dtype=dtypes.fp4x2, shuffle=True)
    wq, b_scale = quant(w, quant_dtype=dtypes.fp4x2, shuffle=True)
    b = shuffle_weight(wq, layout=(16, 16))
    plain_a, plain_as = quant(x, quant_dtype=dtypes.fp4x2, shuffle=False)
    plain_b, plain_bs = quant(w, quant_dtype=dtypes.fp4x2, shuffle=False)
    assert byte_equal(aq, plain_a) and byte_equal(wq, plain_b)
    a_f32 = fp4_utils.mxfp4_to_f32(plain_a)[:m, :k] * fp4_utils.e8m0_to_f32(plain_as)[
        :m, : k // 32
    ].repeat_interleave(32, dim=1)
    b_f32 = fp4_utils.mxfp4_to_f32(plain_b)[:n, :k] * fp4_utils.e8m0_to_f32(plain_bs)[
        :n, : k // 32
    ].repeat_interleave(32, dim=1)
    reference = a_f32 @ b_f32.T

    def call(weight=b, dynamic=False):
        a, sa = (
            quant(x, quant_dtype=dtypes.fp4x2, shuffle=True)
            if dynamic
            else (aq, a_scale)
        )
        return dispatch.gemm_a4w4(a, weight, sa, b_scale, dtype=torch.bfloat16)

    with patch.object(dispatch, "_gemm_a4w4_triton_preshuffle", return_value=None):
        baseline = call().clone()
        base_kernels = kernel_names(call)
    candidate = call()
    checks = check_output(candidate, reference, baseline)
    check_output(call(dynamic=True), reference, baseline)
    candidate_kernels = kernel_names(call)
    row = {
        "M": m,
        "N": n,
        "K": k,
        "seed": seed,
        "gpu": get_gfx_runtime(),
        "cu_num": get_cu_num(),
        "baseline_config": dispatch.get_GEMM_config(m, n, k),
        "baseline_kernels": base_kernels,
        "candidate_kernels": candidate_kernels,
        "changed_route": base_kernels != candidate_kernels,
        "checks": checks,
        "modes": {},
    }
    # Evict before each cold sample; each replay visits distinct weight addresses
    # once. Hot samples deliberately reuse a single weight. Eviction is untimed.
    calls = min(64, max(1, math.ceil(512 * 1024**2 / b.nbytes)))
    weights = [b] + [b.clone() for _ in range(calls - 1)]
    for mode in ("gemm_cold", "quant_gemm_cold", "gemm_hot"):
        arms = {}
        for name in ("baseline", "candidate"):
            context = (
                patch.object(
                    dispatch, "_gemm_a4w4_triton_preshuffle", return_value=None
                )
                if name == "baseline"
                else nullcontext()
            )
            with context:
                for _ in range(40):
                    call(dynamic=mode == "quant_gemm_cold")
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    for weight in weights:
                        out = call(
                            weights[0] if mode == "gemm_hot" else weight,
                            mode == "quant_gemm_cold",
                        )
                for _ in range(10):
                    graph.replay()
                arms[name] = (graph, out)
        timings = {
            name: [] for name in ("baseline_before", "candidate", "baseline_after")
        }
        for _ in range(samples):
            for name, values in timings.items():
                graph, out = arms["candidate" if name == "candidate" else "baseline"]
                if mode != "gemm_hot":
                    eviction.add_(1)
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
                    enable_timing=True
                )
                start.record()
                graph.replay()
                end.record()
                end.synchronize()
                values.append(start.elapsed_time(end) * 1000 / calls)
        for graph, out in arms.values():
            graph.replay()
            torch.cuda.synchronize()
            check_output(out, reference, baseline)
        medians = {name: statistics.median(values) for name, values in timings.items()}
        row["modes"][mode] = {
            "samples_us": timings,
            "median_us": medians,
            "speedup": min(medians["baseline_before"], medians["baseline_after"])
            / medians["candidate"],
            "baseline_drift": max(medians["baseline_before"], medians["baseline_after"])
            / min(medians["baseline_before"], medians["baseline_after"])
            - 1,
            "graph_checks_passed": True,
            "calls_per_graph": calls,
        }
        del arms, graph, out
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shape", action="append", help="M,N,K; repeat for multiple shapes"
    )
    parser.add_argument(
        "--full-range",
        action="store_true",
        help="All integer M=32..64, N=256/512/768, K=256/512/768/1024",
    )
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20261102)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.rounds < 2 or args.samples < 10:
        parser.error("Use at least two independent rounds and ten samples per arm/mode")
    assert get_gfx_runtime() == "gfx950" and get_cu_num() == 256
    shapes = {tuple(map(int, value.split(","))) for value in (args.shape or [])}
    if args.full_range:
        shapes.update(
            (m, n, k)
            for m in range(32, 65)
            for n in (256, 512, 768)
            for k in (256, 512, 768, 1024)
        )
    shapes.update((m, 512, 7168) for m in (32, 64, 128, 256))
    shapes.update(
        {
            (31, 512, 1024),
            (65, 512, 1024),
            (32, 1024, 1024),
            (32, 512, 1280),
            (128, 4608, 7168),
            (512, 512, 7168),
        }
    )
    if not args.shape and not args.full_range:
        shapes.update({(32, 256, 256), (33, 512, 768), (48, 768, 512), (64, 768, 1024)})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    source_file = Path(dispatch.__file__).resolve()
    config_file = Path(
        dispatch.AITER_CONFIGS.AITER_CONFIG_GEMM_A4W4_TRITON_PRESHUFFLE_FILE
    )
    source_hash = hashlib.sha256(source_file.read_bytes()).hexdigest()
    config_hash = hashlib.sha256(config_file.read_bytes()).hexdigest()
    eviction = torch.zeros(512 * 1024**2, dtype=torch.uint8, device="cuda")
    with args.output.open("x") as stream, torch.inference_mode():
        for round_id in range(args.rounds):
            for shape in sorted(shapes):
                row = measure_shape(shape, args.seed + round_id, args.samples, eviction)
                row["round"] = round_id + 1
                row["aiter_file"] = str(Path(dispatch.__file__).resolve())
                row["dispatch_sha256"] = source_hash
                row["config_sha256"] = config_hash
                row["torch_version"] = torch.__version__
                row["hip_version"] = torch.version.hip
                stream.write(json.dumps(row, default=str) + "\n")
                stream.flush()
                print(
                    json.dumps(
                        {
                            key: row[key]
                            for key in ("round", "M", "N", "K", "changed_route")
                        }
                    ),
                    flush=True,
                )
                gc.collect()
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
