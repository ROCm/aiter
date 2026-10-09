"""Benchmark the packaged native K128 kernel with stock AITER inputs."""

import argparse
import json

import torch

from aiter.ops.triton._gluon_kernels.gfx1250.gemm.basic.warp_pipeline import (
    CONFIGS,
    _adapter,
    prepare_gemm_warp_pipeline,
)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--shape", nargs=3, type=int, required=True, metavar=("M", "N", "K"))
    p.add_argument("--repeats", type=int, default=100)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument(
        "--scale-layout", choices=["mcontiguous", "b64"], default="mcontiguous"
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--check", action="store_true")
    p.add_argument("--verify-only", action="store_true")
    args = p.parse_args()
    if args.repeats < 1 or args.warmup < 0:
        p.error("repeats must be positive and warmup nonnegative")
    m, n, k = args.shape
    key = f"{m},{n},{k}"
    if key not in CONFIGS:
        p.error(
            "No selected configuration for this shape; use the Python API for explicit configurations"
        )
    from op_tests.op_benchmarks.triton.bench_gemm_afp8wfp8 import generate_inputs

    torch.manual_seed(args.seed)
    a, w, _, sa, _, sw = generate_inputs(m, n, k, False, 128, (128, 128), False)
    data = (a, w, sa, sw)
    warm = prepare_gemm_warp_pipeline(*data, scale_layout=args.scale_layout)
    compiled = warm.launch()
    torch.cuda.synchronize()
    assert [t.numel() for t in warm.inputs[3:]] == [sa.numel(), sw.numel()]
    if args.check or args.verify_only:
        cfg = dict(CONFIGS[key])
        split = cfg.pop("register_split_output")
        padded = cfg.pop("output_padding")
        for x in ["scale_mode", "num_warps"]:
            cfg.pop(x, None)
        cfg["native_scale128"] = False
        reference = _adapter(split, padded, "b64").prepare_gemm_warp_pipeline(
            *data, **cfg
        )
        reference()
        torch.cuda.synchronize()
        assert torch.equal(warm.output, reference.output), (
            "Mismatch against expanded reference"
        )
        print("Verification passed", flush=True)
    result = {
        "shape": args.shape,
        "config": dict(CONFIGS[key], scale_mode=args.scale_layout),
        "vgprs": compiled.n_regs,
        "spills": compiled.n_spills,
    }
    if not args.verify_only:
        for _ in range(args.warmup):
            warm()
        runs = [
            prepare_gemm_warp_pipeline(
                *(t.clone() for t in data), scale_layout=args.scale_layout
            )
            for _ in range(args.repeats)
        ]
        assert len({t.data_ptr() for fn in runs for t in fn.inputs}) == 5 * args.repeats
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        stream = torch.cuda.Stream()
        with torch.cuda.graph(graph, stream=stream):
            for fn in runs:
                fn()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        us = start.elapsed_time(end) * 1000 / args.repeats
        if args.check:
            assert all(torch.equal(fn.output, warm.output) for fn in runs)
        result.update(
            us=us,
            tflops=2 * m * n * k / (us * 1e6),
            buffer_sets=args.repeats,
            timed_launches=args.repeats,
            graph_replays=1,
        )
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
