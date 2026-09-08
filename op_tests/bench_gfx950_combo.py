# SPDX-License-Identifier: MIT
"""MI355X uBench: FlyDSL row softmax and BF16 A @ B.T.

Reuses uBench's input generators, profiler timer, JSON tables and SMI monitor.
Graph-event samples additionally expose launch amortization and run-to-run noise.
FlyDSL kernel sources are supplied explicitly via --flydsl-root.
"""

from __future__ import annotations

import argparse
import ast
import importlib
import json
import math
import os
import statistics
import sys
from pathlib import Path

SOFTMAX_SHAPES = [
    (1823, 781),
    (1, 1),
    (128, 1),
    (1, 128),
    (8192, 8192),
    (4096, 8192),
    (359, 1),
    (1, 359),
    (1, 131072),
    (1, 89999),
    (32768, 8192),
    (128, 1024),
    (4096, 1024),
    (4096, 4096),
]


def gemm_cases(source):
    tree = ast.parse(Path(source).read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "_TUNED" for t in node.targets
        ):
            rows = ast.literal_eval(node.value)
            keys = (
                "block_m",
                "block_n",
                "block_k",
                "stages",
                "split_k",
                "m_waves",
                "n_waves",
                "k_waves",
                "group_m",
                "hti",
            )
            cases = []
            for row in rows:
                cfg = dict(zip(keys, row[3:]))
                cfg["policy"] = "hti" if cfg.pop("hti") else "ft"
                cases.append((tuple(row[:3]), cfg))
            return cases
    raise ValueError("No _TUNED cases found in the HGEMM unit test")


def graph_samples(fn, *, iters, rounds, output=None):
    import torch

    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(iters):
            fn()
    # Poison output outside capture. An empty/wrong-stream graph must not pass
    # by leaving the result of the eager correctness call in the buffer.
    if output is not None:
        output.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    samples = []
    for _ in range(rounds):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000 / iters)
    if any(not math.isfinite(x) or x <= 0 for x in samples):
        raise RuntimeError(f"Invalid GPU timing: {samples}")
    return samples


def correctness(out, ref, op, dtype):
    import torch

    yf = out.float()
    rf = ref.float()
    if not torch.isfinite(yf).all():
        raise AssertionError("nonfinite output")
    delta = (yf - rf).abs()
    if op == "softmax":
        scale = rf.abs() + rf.amax(-1, keepdim=True)
        scaled = (delta / scale.clamp_min(1e-30)).max().item()
        bound = {"fp32": 1e-5, "fp16": 5e-3, "bf16": 2e-2}[dtype]
        row_error = (yf.sum(-1) - 1).abs().max().item()
        rounding_row_error = (rf.to(out.dtype).float().sum(-1) - 1).abs().max().item()
        if (
            scaled > bound
            or row_error
            > rounding_row_error + {"fp32": 1e-5, "fp16": 2e-3, "bf16": 1e-2}[dtype]
        ):
            raise AssertionError(
                f"softmax scaled_error={scaled}, row_sum_error={row_error}"
            )
        return {
            "max_abs_error": delta.max().item(),
            "scaled_error": scaled,
            "row_sum_error": row_error,
            "reference_rounding_row_sum_error": rounding_row_error,
        }
    # BF16 output rounding alone is ~0.4%; use an all-element, reference-scale
    # floor for cancellation, plus a normalized RMS gate. No % pass allowance.
    rms = rf.square().mean().sqrt().item()
    nrmse = delta.square().mean().sqrt().item() / max(rms, 1e-20)
    tol = 0.02 * rf.abs() + 0.01 * max(rms, 1e-6)
    violations = (delta > tol).sum().item()
    if violations or nrmse > 0.01:
        raise AssertionError(f"GEMM violations={violations}, normalized_RMSE={nrmse}")
    return {
        "max_abs_error": delta.max().item(),
        "normalized_rmse": nrmse,
        "violations": violations,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--perf", action="store_true", help="compatibility with combo CLI")
    p.add_argument(
        "--ops", nargs="+", choices=["softmax", "a16w16"], default=["softmax", "a16w16"]
    )
    p.add_argument("--flydsl-root", type=Path, default=None)
    p.add_argument(
        "--data-init",
        nargs="+",
        choices=["zero", "constant", "uniform", "norm"],
        default=["norm"],
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--dtype", nargs="+", choices=["bf16", "fp16", "fp32"], default=["bf16"]
    )
    p.add_argument("--case-index", type=int, nargs="+", default=None)
    p.add_argument("--iters", type=int, default=64)
    p.add_argument("--rounds", type=int, default=5)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--configs", type=Path, help="JSON mapping case key to candidate config"
    )
    p.add_argument("--variant", default="baseline")
    p.add_argument("--softmax-module", default="kernels.norm.softmax_kernel")
    p.add_argument(
        "--gemm-module",
        help="optional FlyDSL GEMM implementation module for A/B experiments",
    )
    p.add_argument(
        "--ubench-profiler",
        action="store_true",
        help="cross-check via upstream run_perftest",
    )
    p.add_argument("--smi-monitor", action="store_true")
    p.add_argument("--smi-device", type=int, default=0)
    p.add_argument("--smi-interval", type=float, default=0.05)
    p.add_argument("--smi-duration", type=float, default=0.5)
    args = p.parse_args()
    if args.iters < 2 or args.rounds < 1:
        p.error("iters >=2 and rounds >=1 required")
    if args.smi_interval <= 0 or args.smi_duration <= 0:
        p.error("SMI durations must be positive")
    if args.flydsl_root:
        sys.path.insert(0, str(args.flydsl_root.resolve()))
    import torch
    from ubench_common import fill, make_generator, print_json_table

    import aiter
    from aiter.jit.utils.chip_info import get_gfx
    from aiter.test_common import run_perftest

    if get_gfx() != "gfx950":
        p.error(f"gfx950 required; detected {get_gfx()}")
    if args.smi_device != torch.cuda.current_device():
        p.error("SMI device must match the active benchmark GPU")
    torch.backends.cuda.matmul.allow_tf32 = False
    overrides = json.loads(args.configs.read_text()) if args.configs else {}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    failures = []
    for op in args.ops:
        if op == "softmax":
            module = importlib.import_module(args.softmax_module)
            cases = [(s, {}) for s in SOFTMAX_SHAPES]
            dtypes = args.dtype
        else:
            from aiter.ops.flydsl import flydsl_hgemm

            if args.gemm_module:
                from aiter.ops.flydsl import gemm_kernels

                gemm_kernels.gemm_a16w16 = importlib.import_module(
                    args.gemm_module
                ).gemm_a16w16
            source = (
                Path(aiter.__file__).resolve().parents[1]
                / "op_tests/test_flydsl_hgemm.py"
            )
            cases = gemm_cases(source)
            dtypes = ["bf16"]
        for ci, (shape, base_cfg) in enumerate(cases):
            if args.case_index is not None and ci not in args.case_index:
                continue
            for dtype in dtypes:
                for dist in args.data_init:
                    key = f"{op}:{ci}:{dtype}"
                    cfg = overrides.get(key, base_cfg)
                    row = {
                        "arch": "gfx950",
                        "op": op,
                        "case_index": ci,
                        "shape": list(shape),
                        "dtype": dtype,
                        "data_init": dist,
                        "seed": args.seed,
                        "variant": args.variant,
                        "config": cfg,
                    }
                    print("CASE " + json.dumps(row), flush=True)
                    try:
                        gen = make_generator(args.seed)
                        dt = {
                            "bf16": torch.bfloat16,
                            "fp16": torch.float16,
                            "fp32": torch.float32,
                        }[dtype]
                        if op == "softmax":
                            m, n = shape
                            a = fill((m, n), dist, gen, dtype=dt)
                            out = torch.empty_like(a)
                            ref = torch.softmax(a.float(), dim=-1)
                            launch = module.build_softmax_module(
                                m,
                                n,
                                {"bf16": "bf16", "fp16": "f16", "fp32": "f32"}[dtype],
                                **cfg,
                            )

                            def fn(launch=launch, a=a, out=out, m=m):
                                launch(a, out, m, stream=torch.cuda.current_stream())
                                return out
                        else:
                            m, n, k = shape
                            a = fill((m, k), dist, gen, dtype=dt)
                            b = fill((n, k), dist, gen, dtype=dt)
                            out = torch.empty((m, n), device="cuda", dtype=dt)
                            ref = a.float() @ b.float().T

                            def fn(a=a, b=b, out=out, cfg=cfg):
                                return flydsl_hgemm(a, b, out=out, **cfg)

                        fn()
                        torch.cuda.synchronize()
                        row.update(correctness(out, ref, op, dtype))
                        samples = graph_samples(
                            fn, iters=args.iters, rounds=args.rounds, output=out
                        )
                        row.update(correctness(out, ref, op, dtype))
                        row.update(
                            {
                                "samples_us": samples,
                                "median_us": statistics.median(samples),
                                "min_us": min(samples),
                                "max_us": max(samples),
                                "timing": "hipgraph_events",
                            }
                        )
                        if args.ubench_profiler:
                            _, us = run_perftest(
                                fn,
                                num_iters=33,
                                num_warmup=5,
                                num_rotate_args=1,
                                testGraph=True,
                            )
                            if not math.isfinite(us) or us <= 0:
                                raise RuntimeError(
                                    f"invalid uBench profiler latency {us}"
                                )
                            row["ubench_profiler_us"] = float(us)
                        if op == "softmax":
                            row["effective_gbs"] = (
                                2 * m * n * a.element_size() / row["median_us"] / 1e3
                            )
                        else:
                            row["tflops"] = 2 * m * n * k / row["median_us"] / 1e6
                        if args.smi_monitor:
                            from smi_monitor import replay_with_smi

                            os.environ.update(
                                AITER_SMI_MONITOR="1",
                                AITER_SMI_DEVICE=str(args.smi_device),
                                AITER_SMI_INTERVAL=str(args.smi_interval),
                                AITER_SMI_DURATION=str(args.smi_duration),
                            )
                            try:
                                row["smi"] = replay_with_smi(
                                    fn,
                                    label=key,
                                    synchronize=torch.cuda.synchronize,
                                    estimated_us=row["median_us"],
                                )
                            finally:
                                os.environ.pop("AITER_SMI_MONITOR", None)
                        row["status"] = "passed"
                    except Exception as exc:  # noqa: BLE001 - retain per-case failure and exit nonzero
                        import traceback

                        traceback.print_exc()
                        row.update(
                            status="failed", error=f"{type(exc).__name__}: {exc}"
                        )
                        failures.append(key)
                    rows.append(row)
                    print_json_table(op, [row])
                    args.output.write_text(json.dumps(rows, indent=2) + "\n")
                    torch.cuda.empty_cache()
    if not rows:
        p.error("no test cases selected")
    if failures:
        raise SystemExit(f"{len(failures)} cases failed: {failures}")


if __name__ == "__main__":
    main()
