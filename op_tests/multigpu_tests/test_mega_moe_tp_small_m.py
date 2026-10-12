# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Compare small-tail variants against the original path on identical inputs.

HIP_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 \
    op_tests/multigpu_tests/test_mega_moe_tp_small_m.py --tokens 8 16 32 64 256

One engine supports all sizes, so selecting the small path does not depend on
max_local_tokens. Eager and alternating graph replays vary inputs and routing,
including an empty route list, and check y/residual/FP8/scale/BF16 bit for bit.
"""

import argparse
import os
import sys
from dataclasses import replace
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from op_tests.op_benchmarks.flydsl import bench_mega_moe_TP as bench


def snapshot(outputs, residual):
    return [t.clone() for t in (*outputs, residual)]


def check_equal(expected, actual, ctx):
    names = (
        ("y", "q", "scale", "bf16", "residual")
        if len(actual) == 5
        else ("y", "q", "scale", "residual")
    )
    failures = [
        name
        for name, ref, got in zip(names, expected, actual)
        if not torch.equal(ref, got)
    ]
    if not bench.all_ranks_ok(not failures, ctx.device):
        raise AssertionError(f"rank {ctx.rank}: small tail differs: {failures}")


def make_runner(eng, cfg, m, x, weights, ids, residual, source_res, nw, bf16):
    def run(mask):
        eng._cfgs[m] = replace(cfg, small_m_opt=mask if m * eng.tp <= 32 else 0)
        residual.copy_(source_res)
        return eng.forward(x, weights, ids, tail=(residual, residual, nw), bf16=bf16)

    return run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, nargs="+", default=[8, 16, 32, 64, 256])
    parser.add_argument("--mt", type=int, default=1)
    parser.add_argument("--comm-dtype", choices=["fp8", "bf16"], default="fp8")
    parser.add_argument("--variants", type=int, nargs="+", default=[1, 2, 3, 4, 7])
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--model", choices=list(bench.MODELS), default="m3")
    args = parser.parse_args()
    os.environ["AITER_MEGAMOE_TP_LB_MIN"] = "1"
    os.environ["AITER_MEGAMOE_TP_LB_MT_SMALL"] = str(args.mt)
    os.environ["AITER_MEGAMOE_TP_LB_NPP"] = "2"
    os.environ["AITER_MEGAMOE_TP_LB_Q"] = "3"
    os.environ["AITER_MEGAMOE_TP_SMALL_M_OPT"] = "7"
    if bench.get_gfx() not in bench.SUPPORTED_GFX:
        print("Small Mega tail requires gfx950; skipped.")
        return
    ctx = bench.setup_dist()
    try:
        tokens = sorted(set(args.tokens))
        shape = bench.MODELS[args.model]
        wt = bench.build_sharded_weights(shape, ctx, ctx.world, 123)
        layer = bench.MegaMoeTP(
            wt, ctx, max(tokens) // ctx.world, comm_dtype=args.comm_dtype
        )
        eng = layer.engine.engine
        for M in tokens:
            m = M // ctx.world
            cfg = eng.config(m)
            assert cfg.small_m_opt == (7 if M <= 32 else 0)
            for gemma, bf16 in ((True, False), (False, True)):
                eng.tn_gemma = gemma
                inp = bench.make_inputs(shape, ctx, ctx.world, M, 123, "balanced")
                x = inp.x_local
                ids, weights = inp.topk_ids_local, inp.topk_weights_local
                source_res = torch.randn_like(x)
                residual = torch.empty_like(x)
                nw = torch.empty(
                    shape.model_dim, dtype=torch.bfloat16, device=ctx.device
                )
                nw.fill_(0.5)

                run = make_runner(
                    eng, cfg, m, x, weights, ids, residual, source_res, nw, bf16
                )

                masks = [0] + (args.variants if M <= 32 else [7])
                graphs = {}
                outputs = {}
                for mask in masks:
                    eng._cfgs[m] = replace(cfg, small_m_opt=mask if M <= 32 else 0)
                    eng.prepare([m], tail=not bf16, tail_bf16=bf16)
                    run(mask)
                    bench.barrier()
                    error = eng.poll_errors()
                    if not bench.all_ranks_ok(error == 0, ctx.device):
                        raise AssertionError(
                            f"rank={ctx.rank} warm M={M} mask={mask} watchdog={error}"
                        )
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        outputs[mask] = run(mask)
                    graphs[mask] = graph
                    bench.barrier()

                for iteration in range(4):
                    # Same storage, different values across epochs and graph replays.
                    gen = torch.Generator(device=ctx.device).manual_seed(
                        100 + iteration + ctx.rank
                    )
                    x.normal_(generator=gen)
                    source_res.normal_(generator=gen)
                    if iteration == 3:
                        ids.fill_(shape.experts)  # exercise lb_none with no GEMMs
                    else:
                        ids.random_(0, shape.experts, generator=gen)
                        weights.uniform_(0.0, 1.0, generator=gen)
                    for graph_mode in (False, True):
                        expected = None
                        for mask in masks:
                            if graph_mode:
                                graphs[mask].replay()
                                out = outputs[mask]
                            else:
                                out = run(mask)
                            bench.barrier()
                            actual = snapshot(out, residual)
                            if expected is None:
                                expected = actual
                            else:
                                check_equal(expected, actual, ctx)
                            error = eng.poll_errors()
                            if not bench.all_ranks_ok(error == 0, ctx.device):
                                raise AssertionError(
                                    f"rank={ctx.rank} M={M} mask={mask} iteration={iteration} "
                                    f"graph={graph_mode} watchdog={error}"
                                )
                if args.benchmark and gemma:
                    ids.random_(0, shape.experts)
                    for mask in masks:
                        us = bench.time_us(
                            graphs[mask].replay,
                            iters=100,
                            warmup=10,
                            device=ctx.device,
                            rounds=5,
                        )
                        if ctx.is_main:
                            print(
                                f"SMALL_TAIL_PERF M={M} mask={mask} us={us:.3f}",
                                flush=True,
                            )
                graphs.clear()
                if ctx.is_main:
                    print(
                        f"SMALL_TAIL_OK M={M} mt={args.mt} comm={args.comm_dtype} gemma={gemma} bf16={bf16}",
                        flush=True,
                    )
    finally:
        bench.cleanup_dist()


if __name__ == "__main__":
    main()
