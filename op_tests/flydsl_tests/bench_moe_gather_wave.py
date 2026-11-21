# SPDX-License-Identifier: MIT
"""Compare LDS and wave route caches on the same gfx1250 device.

Run from the AITER root:
  python op_tests/flydsl_tests/bench_moe_gather_wave.py --tokens 4 16 64 128 256

This times the gather epilogue only, not the whole MoE or serving workload.
Two independent LDS graphs provide an A/A control. Source buffers rotate;
the fixed-shape synthetic inputs do not model a full decoder's working set.
"""

import argparse
import json
import random
import statistics

import torch

from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.ops.flydsl.grouped_moe_gfx1250 import _choose_gather_reduce_vec
from aiter.ops.flydsl.kernels.moe_gather_reduce import build_moe_gather_reduce_module
from aiter.ops.flydsl.kernels.tensor_shim import (
    _run_compiled,
    current_stream_arg,
    ptr_arg,
)


def benchmark(tokens, seed, samples, calls):
    torch.manual_seed(seed + tokens)
    rng = random.Random(seed + tokens)
    d, topk = 6144, 5
    rows = tokens * topk + 1
    sources = [
        (torch.randn(rows, d, device="cuda") * 0.25).bfloat16() for _ in range(4)
    ]
    ids = torch.randint(1, rows, (tokens, topk), device="cuda", dtype=torch.int32)
    ids[:, -1] = torch.arange(tokens, device="cuda", dtype=torch.int32) + 1
    ids[::3, 0] = -1
    weights = torch.rand(tokens, topk, device="cuda")
    weights[:, -1] = 1.0
    valid = torch.tensor([tokens], device="cuda", dtype=torch.int32)
    vec = _choose_gather_reduce_vec(tokens, d)
    graphs, outputs = [], []
    for use_wave in (False, True, False):
        launch = build_moe_gather_reduce_module(
            d, topk, "bf16", 1, vec, "f32", wave_routes=use_wave
        )
        dst = [torch.empty_like(s[:tokens]) for s in sources]

        def run(slot, launch=launch, dst=dst):
            _run_compiled(
                launch,
                ptr_arg(sources[slot]),
                ptr_arg(ids),
                ptr_arg(weights),
                ptr_arg(dst[slot]),
                tokens,
                rows * (d // 2),
                ptr_arg(valid),
                current_stream_arg(),
            )

        for slot in range(4):
            run(slot)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for iteration in range(calls):
                run(iteration % 4)
        graphs.append(graph)
        outputs.append(dst)

    try:
        for slot in range(4):
            torch.testing.assert_close(
                outputs[0][slot], outputs[1][slot], atol=0, rtol=0
            )
        for graph in graphs:
            for _ in range(5):
                graph.replay()
        times = [[], [], []]
        for _ in range(samples):
            order = list(range(3))
            rng.shuffle(order)
            for index in order:
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                graphs[index].replay()
                end.record()
                end.synchronize()
                times[index].append(start.elapsed_time(end) * 1000 / calls)
        medians = [statistics.median(values) for values in times]
        baseline = (medians[0] + medians[2]) / 2
        return {
            "tokens": tokens,
            "model_dim": d,
            "topk": topk,
            "vec_dwords": vec,
            "seed": seed,
            "source_buffers": len(sources),
            "source_bytes": sum(s.nbytes for s in sources),
            "calls_per_graph": calls,
            "samples": samples,
            "arms": ["lds_A", "wave", "lds_A2"],
            "median_us": medians,
            "speedup": baseline / medians[1],
            "aa_delta_us": abs(medians[0] - medians[2]),
            "samples_us": times,
            "scope": "gather epilogue only, not whole-MoE or end-to-end latency",
        }
    finally:
        for graph in graphs:
            graph.reset()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tokens", type=int, nargs="+", default=[4, 12, 16, 40, 64, 128, 256]
    )
    parser.add_argument("--seed", type=int, default=10201)
    parser.add_argument("--samples", type=int, default=101)
    parser.add_argument("--calls", type=int, default=16)
    args = parser.parse_args()
    if get_gfx_runtime() != "gfx1250":
        parser.error("this benchmark requires gfx1250")
    if args.samples < 3 or args.calls < 4 or any(m < 1 for m in args.tokens):
        parser.error("samples>=3, calls>=4 and positive token counts required")
    torch.set_num_threads(1)
    for tokens in args.tokens:
        print(
            json.dumps(benchmark(tokens, args.seed, args.samples, args.calls)),
            flush=True,
        )


if __name__ == "__main__":
    main()
