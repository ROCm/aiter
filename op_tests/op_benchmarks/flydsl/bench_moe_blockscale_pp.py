# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Block-FP8 MoE forward on gfx950 (D 4096 / I 1024, GLM-5.3-Flash): FlyDSL vs main vs HipKittens.

The whole forward is timed with PR 6742's protocol (bench_hk_fp8_moe.py). Each
candidate is captured into a warm HIP graph of 10 calls, the graphs are
replayed alternately for 21 rounds, and the median per call is reported.

  flydsl  fused_moe on a one-row tuned table naming impl__flydsl_bspp__default
  main    fused_moe on the GLM-5.3-Flash tuned tables at --main-ref
  hk      aiter.ops.hk_fp8_moe (PR 6742), when importable

Every output is checked against the FP32 reference (relative L2 < 0.06), and
the kernels each candidate launched are recorded by name. For stable numbers,
run one case per process on an idle GPU:

  python op_tests/op_benchmarks/flydsl/bench_moe_blockscale_pp.py -t 8192 -k 8
"""

import argparse
import csv
import io
import json
import statistics
import subprocess
import sys
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import aiter  # noqa: E402
from aiter import dtypes  # noqa: E402
from op_tests.test_moe_blockscale_pp import (  # noqa: E402
    ROUTED_EXPERTS,
    TOL,
    bspp_row,
    dequant,
    make_routing,
    rel_l2,
    run_torch,
    tuned_rows,
    weights,
)

GLM_CSV = {
    8: "aiter/configs/model_configs/a8w8_blockscale_tuned_fmoe_glm5_3_flash_gfx950.csv",
    9: "aiter/configs/model_configs/a8w8_blockscale_tuned_fmoe_glm5_3_flash_shared_gfx950.csv",
}


def main_rows(ref, topk):
    text = subprocess.check_output(
        ["git", "-C", str(ROOT), "show", f"{ref}:{GLM_CSV[topk]}"], text=True
    )
    return list(csv.DictReader(io.StringIO(text)))


def capture(call, reps):
    for _ in range(5):
        call()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(reps):
            out = call()
    for _ in range(5):
        graph.replay()
    torch.cuda.synchronize()
    return graph, out


def kernels(call):
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as p:
        call()
        torch.cuda.synchronize()
    return sorted(
        {
            ev.name.split("(")[0]
            for ev in p.events()
            if ev.device_type == torch.autograd.DeviceType.CUDA
        }
    )


def bench_case(tokens, topk, main_ref, reps, rounds):
    from aiter.fused_moe import fused_moe
    from aiter.ops.shuffle import shuffle_weight

    E = ROUTED_EXPERTS + int(topk == 9)
    w13, s13, w2, s2 = weights(E, 1024, 4096)
    w13s, w2s = shuffle_weight(w13, (16, 16)), shuffle_weight(w2, (16, 16))
    w13s.is_shuffled = w2s.is_shuffled = True
    x, ids, tw = make_routing(tokens, E, topk, seed=1)
    rows = torch.linspace(0, tokens - 1, min(tokens, 256)).long()
    ref = run_torch(
        x[rows], dequant(w13, s13), dequant(w2, s2), tw[rows], ids[rows], 0.0
    )

    def aiter_call(out):
        return lambda: fused_moe(
            x,
            w13s,
            w2s,
            tw,
            ids,
            activation=aiter.ActivationType.Silu,
            quant_type=aiter.QuantType.per_1x128,
            w1_scale=s13,
            w2_scale=s2,
            dtype=dtypes.bf16,
            output=out,
        )

    tables = {"flydsl": [bspp_row(tokens, E, topk)], "main": main_rows(main_ref, topk)}
    graphs, info = {}, {}
    for name, table in tables.items():
        call = aiter_call(torch.empty_like(x))
        with tuned_rows(table):
            graphs[name], out = capture(call, reps)
            info[name] = {"kernels": kernels(call)}
        info[name]["err"] = rel_l2(out[rows], ref)
    assert any(k.startswith("flydsl_moe_bs") for k in info["flydsl"]["kernels"])
    try:
        from aiter.ops.hk_fp8_moe import hk_fp8_moe
    except ImportError:
        hk_fp8_moe = None  # PR 6742 is not part of this checkout
    if hk_fp8_moe is not None:
        hk_out = torch.empty_like(x)
        call = lambda: hk_fp8_moe(x, ids, tw, w13s, s13, w2s, s2, out=hk_out)  # noqa
        graphs["hk"], out = capture(call, reps)
        info["hk"] = {"kernels": kernels(call), "err": rel_l2(out[rows], ref)}

    samples = {name: [] for name in graphs}
    start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    for it in range(rounds):
        for name in list(graphs) if it % 2 == 0 else list(graphs)[::-1]:
            start.record()
            graphs[name].replay()
            end.record()
            end.synchronize()
            samples[name].append(start.elapsed_time(end) * 1000 / reps)

    routed = int((ids >= 0).logical_and(ids < E).sum())
    flops = routed * 2 * (4096 * 2 * 1024 + 1024 * 4096)
    result = []
    for name, v in samples.items():
        err = info[name]["err"]
        assert err < TOL, f"T={tokens} top-k {topk} {name}: rel L2 {err:.4f}"
        result.append(
            {
                "tokens": tokens,
                "topk": topk,
                "candidate": name,
                "us": statistics.median(v),
                "min us": min(v),
                "TFLOPS": flops / statistics.median(v) / 1e6,
                "err": err,
                "kernels": " ".join(info[name]["kernels"]),
                "samples": v,
            }
        )
    return result


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter, description=__doc__
    )
    parser.add_argument(
        "-t", "--tokens", type=int, nargs="*", default=[4096, 8192, 16384]
    )
    parser.add_argument("-k", "--topk", type=int, nargs="*", default=[8, 9])
    parser.add_argument(
        "--main-ref",
        default="origin/main",
        help="git ref whose GLM-5.3-Flash tuned tables define the main candidate",
    )
    parser.add_argument("--reps", type=int, default=10, help="calls per graph")
    parser.add_argument("--rounds", type=int, default=21, help="alternating replays")
    parser.add_argument("--json", type=Path, help="write every sample here")
    args = parser.parse_args()
    torch.set_grad_enabled(False)
    torch.set_default_device("cuda")

    result = []
    for topk in args.topk:
        for tokens in args.tokens:
            result += bench_case(tokens, topk, args.main_ref, args.reps, args.rounds)
    if args.json:
        args.json.write_text(json.dumps(result, indent=1) + "\n")
    df = pd.DataFrame(result).drop(columns=["samples", "kernels"])
    aiter.logger.info(
        "moe_blockscale_pp whole forward (markdown):\n%s",
        df.to_markdown(index=False, floatfmt=".4g"),
    )


if __name__ == "__main__":
    main()
