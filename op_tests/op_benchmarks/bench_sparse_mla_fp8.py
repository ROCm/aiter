# SPDX-License-Identifier: MIT
"""BF16 Q -> BF16 O, FP8 KV: HIP and Gluon on identical random slot IDs."""

import argparse
import json
import math
from pathlib import Path
import statistics
import sys

import torch

from aiter import sparse_mla_fp8_fwd
from aiter.ops.triton.attention.sparse_mla import sparse_mla_fwd as gluon

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from test_sparse_mla_fp8 import assert_reference


class RecordLaunch:
    def __init__(self, kernel):
        self.kernel = kernel
        self.block_m = None

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            self.block_m = kwargs["BLOCK_M"]
            return self.kernel[grid](*args, **kwargs)

        return launch


def graph_for(fn, repetitions):
    fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(repetitions):
            out = fn()
    for _ in range(4):
        graph.replay()
    torch.cuda.synchronize()
    return graph, out


def measure(calls, repetitions, count):
    graphs = {name: graph_for(fn, repetitions) for name, fn in calls.items()}
    samples = {name: [] for name in calls}
    for repeat in range(count):
        for name in list(graphs)[:: 1 if repeat % 2 == 0 else -1]:
            begin = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            begin.record()
            graphs[name][0].replay()
            end.record()
            end.synchronize()
            samples[name].append(begin.elapsed_time(end) * 1000 / repetitions)
    return samples


def check_gluon(out, q, kv, ids, block_m):
    # Source-order model of Gluon's dynamic quantization. This does not assert
    # that Torch division and GPU reciprocal instructions are bitwise equal.
    blocks = q.float().reshape(q.shape[0], 64 // block_m, block_m, 512)
    amax = blocks.abs().amax(dim=(-2, -1), keepdim=True).clamp_min(1e-30)
    encoded = (blocks * (448.0 / amax)).to(torch.float8_e4m3fn).reshape(q.shape)
    scales = (
        (amax / 448.0)
        .expand(q.shape[0], 64 // block_m, block_m, 1)
        .reshape(q.shape[0], 64, 1)
    )
    return assert_reference(out, None, encoded, kv, ids, q_scale=scales)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--queries", type=int, nargs="+", default=[2**i for i in range(12)]
    )
    parser.add_argument(
        "--all-small", action="store_true", help="Q1-64 plus 65/128/.../2048"
    )
    parser.add_argument("--pool-bytes", type=int, default=5_608_046_592)
    parser.add_argument("--valid-k", type=int, default=2051)
    parser.add_argument("--capacity", type=int, default=2112)
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--graph-repetitions", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.all_small:
        args.queries = list(range(1, 65)) + [65, 128, 256, 512, 1024, 2048]
    assert 0 < args.valid_k <= args.capacity <= 2176
    assert args.pool_bytes > 0 and args.pool_bytes % 512 == 0
    assert args.samples > 0 and args.graph_repetitions > 0
    assert all(1 <= n <= 2048 for n in args.queries)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "config.json").write_text(
        json.dumps(vars(args), default=str, indent=2)
    )
    slots = args.pool_bytes // 512
    kv = torch.empty((slots, 512), device="cuda", dtype=torch.float8_e4m3fn)
    # Bound temporary storage while initializing the complete random pool.
    torch.manual_seed(2041103)
    for start in range(0, slots, 65536):
        kv[start : start + 65536].copy_(
            torch.randn(
                kv[start : start + 65536].shape, device="cuda", dtype=torch.bfloat16
            ).to(kv.dtype)
        )
    one = torch.ones((1,), device="cuda", dtype=torch.float32)
    medians = {n: {"HIP": [], "Gluon": []} for n in args.queries}
    with (args.output / "phases.jsonl").open("w") as log:
        for phase in range(4):
            order = args.queries[:: 1 if phase % 2 == 0 else -1]
            for n in order:
                torch.manual_seed(2121037 + (phase // 2) * 10007 + n)
                q = torch.randn((n, 64, 512), device="cuda", dtype=torch.bfloat16)
                torch.manual_seed(2041129 + n)
                ids = torch.randint(
                    slots, (n, args.capacity), device="cuda", dtype=torch.int32
                )
                ids[:, args.valid_k :] = -1
                indptr = (
                    torch.arange(n + 1, device="cuda", dtype=torch.int32)
                    * args.capacity
                )

                def run_gluon():
                    return gluon(
                        q,
                        kv,
                        indptr,
                        ids.reshape(-1),
                        1 / 16,
                        kv_scale=one,
                        kv_lora_rank=512,
                        qk_rope_head_dim=0,
                        has_invalid=True,
                        dot_precision="fp8",
                    )[0]

                calls = {
                    "HIP": lambda: sparse_mla_fp8_fwd(q, kv, ids, 1 / 16)[0],
                    "Gluon": run_gluon,
                }
                hip_out = calls["HIP"]()
                namespace = gluon.__globals__
                original = namespace["_sparse_mla_gfx950"]
                tap = RecordLaunch(original)
                namespace["_sparse_mla_gfx950"] = tap
                try:
                    gluon_out = run_gluon()
                finally:
                    namespace["_sparse_mla_gfx950"] = original
                errors = {
                    "HIP": assert_reference(hip_out, None, q, kv, ids),
                    "Gluon": check_gluon(gluon_out, q, kv, ids, tap.block_m),
                }
                if phase % 2:
                    calls = dict(reversed(list(calls.items())))
                samples = measure(calls, args.graph_repetitions, args.samples)
                row = dict(
                    phase=phase,
                    q=n,
                    rrms=errors,
                    gluon_block_m=tap.block_m,
                    samples_us=samples,
                    medians_us={k: statistics.median(v) for k, v in samples.items()},
                )
                log.write(json.dumps(row) + "\n")
                log.flush()
                for k, v in row["medians_us"].items():
                    medians[n][k].append(v)
                print(
                    json.dumps({k: v for k, v in row.items() if k != "samples_us"}),
                    flush=True,
                )
    rows = []
    for n in args.queries:
        hip, g = (statistics.median(medians[n][k]) for k in ["HIP", "Gluon"])
        rows.append(
            dict(
                q=n,
                hip_us=hip,
                gluon_us=g,
                speedup_pct=100 * (g / hip - 1),
                logical_kv_tbps=n * args.valid_k * 512 / hip / 1e6,
            )
        )
    result = dict(
        rows=rows,
        minimum_speedup_pct=min(r["speedup_pct"] for r in rows),
        geometric_mean_speedup_pct=100
        * (
            math.exp(
                statistics.mean(math.log(r["gluon_us"] / r["hip_us"]) for r in rows)
            )
            - 1
        ),
        numerical_checks_passed=True,
        endpoint="BF16 Q to BF16 O",
        kv_dtype="E4M3FN",
        logical_bandwidth_is_hbm_traffic=False,
    )
    (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
