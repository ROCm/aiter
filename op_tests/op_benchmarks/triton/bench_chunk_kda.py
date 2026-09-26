# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import argparse
import sys

import torch
import triton

from aiter.ops.triton.attention.chunk_kda import (
    CHUNK_SIZE,
    chunk_kda,
    chunk_kda_prepare,
    chunk_kda_walk,
)
from aiter.ops.triton.utils._triton import arch_info
from op_tests.op_benchmarks.triton.utils.benchmark_utils import get_caller_name_no_ext

try:  # the Triton chunk path vLLM runs on gfx1250 today
    from vllm.models.kimi_k3.amd.ops.third_party.kda import chunk_kda_with_fused_gate

    HAS_VLLM = True
except ImportError:
    chunk_kda_with_fused_gate = None
    HAS_VLLM = False

K3_HEAD_DIM = 128
K3_NUM_HEADS = [24, 12]  # 96 heads sharded tp4 / tp8
K3_LOWER_BOUND = -5.0


def make_inputs(B, T, H, device):
    """B sequences of T tokens, raw projections; q/k/v are bands of one fused projection."""
    D, total = K3_HEAD_DIM, B * T
    mixed = torch.randn(1, total, 3 * H * D, dtype=torch.bfloat16, device=device)
    q, k, v = (
        mixed[..., i * H * D : (i + 1) * H * D].unflatten(-1, (H, D)) for i in range(3)
    )
    return dict(
        q=q,
        k=k,
        v=v,
        g=torch.randn(1, total, H, D, dtype=torch.bfloat16, device=device),
        beta=torch.randn(1, total, H, dtype=torch.bfloat16, device=device),
        A_log=torch.log(torch.empty(H, device=device).uniform_(1, 16)),
        dt_bias=torch.randn(H * D, device=device),
        cu_seqlens=torch.arange(0, total + 1, T, dtype=torch.int32, device=device),
    )


def traffic_bytes(B, T, H, stage):
    D = K3_HEAD_DIM
    tok = B * T * H * D * 2  # one [T, H, 128] bf16 tensor
    nt = B * triton.cdiv(T, CHUNK_SIZE) * H
    ws = (
        3 * tok + tok // 2 + nt * D * (CHUNK_SIZE * 2 + 4)
    )  # qg, w, u, aqk, kg_t, decay
    state = B * H * D * D * 4
    prepare = 4 * tok + B * T * H * 2 + ws
    walk = ws + tok + 2 * state
    return {"prepare": prepare, "walk": walk, "total": prepare + walk}[stage]


def _time(fn, args):
    """One measurement in ms; cudagraph keeps host launch cost out of the span."""
    if args.timing == "cudagraph":
        return _bench_graph(fn, args.graph_ms, args.n_replays)
    return triton.testing.do_bench(
        fn, warmup=args.warmup, rep=args.rep, quantiles=[0.5, 0.2, 0.8]
    )


def _bench_graph(fn, graph_ms, n_replays):
    """Replay a graph of ~graph_ms of launches; returns (median, p20, p80) ms."""
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    ev0, ev1 = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    ev0.record()
    for _ in range(20):
        fn()
    ev1.record()
    torch.cuda.synchronize()
    est_ms = ev0.elapsed_time(ev1) / 20
    n_per_graph = max(1, int(graph_ms / est_ms)) if est_ms > 0 else 1000

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            fn()
        side.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=side):
            for _ in range(n_per_graph):
                fn()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()

    per_iter = []
    for _ in range(n_replays):
        s, e = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        s.record()
        graph.replay()
        e.record()
        torch.cuda.synchronize()
        per_iter.append(s.elapsed_time(e) / n_per_graph)
    per_iter.sort()
    lo = per_iter[int(0.2 * (len(per_iter) - 1))]
    hi = per_iter[int(0.8 * (len(per_iter) - 1))]
    return per_iter[len(per_iter) // 2], lo, hi


def benchmark(args):
    lines = ["gluon", "gluon_prepare"] + [f"gluon_walk:{c}" for c in args.walk_configs]
    if HAS_VLLM and "vllm" in args.backends:
        lines.append("vllm")
    configs = [
        triton.testing.Benchmark(
            x_names=["H", "B", "T"],
            x_vals=[
                (h, b, t)
                for h in args.num_heads
                for b in args.batch_sizes
                for t in args.seq_lens
            ],
            line_arg="provider",
            line_vals=lines,
            line_names=lines,
            plot_name=get_caller_name_no_ext(),
            styles=[],
            ylabel="us" if args.metric == "time" else "TB/s",
            args={},
        )
    ]

    @triton.testing.perf_report(configs)
    def bench_chunk_kda(H, B, T, provider):
        torch.manual_seed(0)
        inp = make_inputs(B, T, H, args.device)
        state = torch.randn(B, H, K3_HEAD_DIM, K3_HEAD_DIM, device=args.device)
        idx = torch.arange(B, dtype=torch.int32, device=args.device)
        has_init = torch.ones(B, dtype=torch.bool, device=args.device)
        paged = dict(state_cache=state, state_indices=idx, has_initial_state=has_init)
        stage = "total"
        if provider == "gluon":
            out = torch.empty_like(inp["v"])

            def fn():
                chunk_kda(**inp, lower_bound=K3_LOWER_BOUND, out=out, **paged)

        elif provider == "gluon_prepare":
            stage = "prepare"

            def fn():
                chunk_kda_prepare(**inp, lower_bound=K3_LOWER_BOUND)

        elif provider.startswith("gluon_walk:"):
            stage = "walk"
            bv, nw = (int(x) for x in provider.split(":")[1].split(","))
            ws = chunk_kda_prepare(**inp, lower_bound=K3_LOWER_BOUND)
            out = torch.empty_like(ws["u"])

            def fn():
                chunk_kda_walk(
                    **ws,
                    cu_seqlens=inp["cu_seqlens"],
                    out=out,
                    config={"BV": bv, "num_warps": nw},
                    **paged,
                )

        else:

            def fn():
                chunk_kda_with_fused_gate(
                    q=inp["q"],
                    k=inp["k"],
                    v=inp["v"],
                    raw_g=inp["g"],
                    raw_beta=inp["beta"],
                    A_log=inp["A_log"],
                    g_bias=inp["dt_bias"],
                    initial_state=state,
                    output_final_state=True,
                    lower_bound=K3_LOWER_BOUND,
                    use_qk_l2norm_in_kernel=True,
                    cu_seqlens=inp["cu_seqlens"],
                )

        ms, lo, hi = _time(fn, args)
        if args.metric == "time":
            return ms * 1e3, lo * 1e3, hi * 1e3
        mem = traffic_bytes(B, T, H, stage)
        return mem / ms * 1e-9, mem / hi * 1e-9, mem / lo * 1e-9

    bench_chunk_kda.run(
        save_path="." if args.o else None, print_data=True, show_plots=False
    )


def parse_args():
    parser = argparse.ArgumentParser(
        prog="Benchmark chunked KDA prefill", allow_abbrev=False
    )
    parser.add_argument("--batch_sizes", type=int, nargs="+", default=[1, 3, 8])
    parser.add_argument("--seq_lens", type=int, nargs="+", default=[1024, 4096])
    parser.add_argument("--num_heads", type=int, nargs="+", default=K3_NUM_HEADS)
    parser.add_argument(
        "--walk_configs",
        nargs="+",
        default=["32,2", "64,4", "128,4"],
        help="walk BV,num_warps points to time on their own",
    )
    parser.add_argument("--backends", nargs="+", default=["gluon", "vllm"])
    parser.add_argument("--metric", choices=["time", "throughput"], default="time")
    parser.add_argument(
        "--timing", choices=["cudagraph", "do_bench"], default="do_bench"
    )
    parser.add_argument("--graph_ms", type=float, default=20.0)
    parser.add_argument("--n_replays", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--rep", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "-o", action="store_true", default=False, help="Write CSV results"
    )
    return parser.parse_args()


def main():
    if arch_info.get_arch() != "gfx1250":
        sys.exit(f"chunk KDA gluon needs gfx1250, got {arch_info.get_arch()}")
    if not HAS_VLLM:
        print("vllm not importable -- dropping the Triton chunk baseline.")
    benchmark(parse_args())


if __name__ == "__main__":
    main()
