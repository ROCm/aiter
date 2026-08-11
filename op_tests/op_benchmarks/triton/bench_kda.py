# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import argparse
import sys

import torch
import torch.nn.functional as F
import triton

from aiter.ops.triton._triton_kernels.kda import fused_recurrent_kda_triton
from aiter.ops.triton.attention.kda import fused_recurrent_kda
from aiter.ops.triton.utils._triton import arch_info
from op_tests.op_benchmarks.triton.utils.benchmark_utils import get_caller_name_no_ext

try:
    from fla.ops.kda.fused_recurrent import fused_recurrent_kda_fwd as fla_kda

    HAS_FLA = True
except ImportError:
    fla_kda = None
    HAS_FLA = False

BACKENDS = {
    "gluon": "aiter gluon",
    "triton": "aiter triton",
    "fla": "fla triton",
}

K3_HEAD_DIM = 128
K3_NUM_HEADS = [24, 12]  # 96 heads sharded tp4 / tp8
K3_LOWER_BOUND = -5.0


def make_inputs(B, T, H, D, dtype, device, paged, gate, num_accepted=0):
    """K3 serves `full_k3`: raw q/k/g/beta, gate chain and l2norm fused in."""
    total_T = B * T
    q = torch.rand(1, total_T, H, D, dtype=dtype, device=device)
    k = torch.rand(1, total_T, H, D, dtype=dtype, device=device)
    v = torch.rand(1, total_T, H, D, dtype=dtype, device=device)
    beta = torch.rand(1, total_T, H, dtype=torch.float32, device=device)
    cu_seqlens = torch.arange(0, total_T + 1, step=T, device=device).long()

    if gate:
        g = torch.randn(1, total_T, H, D, dtype=torch.float32, device=device)
        A_log = torch.log(
            torch.empty(H, dtype=torch.float32, device=device).uniform_(1, 16)
        )
        dt_bias = torch.randn(H * D, dtype=torch.float32, device=device)
    else:
        q, k = F.normalize(q, p=2, dim=-1), F.normalize(k, p=2, dim=-1)
        g = F.logsigmoid(
            torch.randn(1, total_T, H, D, dtype=torch.float32, device=device)
        )
        beta = beta.sigmoid()
        A_log = dt_bias = None

    if paged:
        state = torch.randn(total_T, H, D, D, dtype=torch.float32, device=device)
        indices = torch.arange(total_T, device=device, dtype=torch.int32).view(B, T)
    else:
        state = torch.randn(B, H, D, D, dtype=torch.float32, device=device)
        indices = None

    if paged and num_accepted:
        accepted = torch.full(
            (B,), min(num_accepted, T), device=device, dtype=torch.int32
        )
    else:
        accepted = None

    return {
        "q": q,
        "k": k,
        "v": v,
        "g": g,
        "beta": beta,
        "A_log": A_log,
        "dt_bias": dt_bias,
        "lower_bound": K3_LOWER_BOUND if gate else None,
        "use_gate_in_kernel": gate,
        "use_qk_l2norm_in_kernel": gate,
        "use_beta_sigmoid_in_kernel": gate,
        "initial_state": state,
        "cu_seqlens": cu_seqlens,
        "ssm_state_indices": indices,
        "num_accepted_tokens": accepted,
    }


def traffic_bytes(B, T, H, D, dtype, paged):
    slab = H * D * D * 4
    state = B * slab + (B * T * slab if paged else B * slab)
    e = torch.tensor([], dtype=dtype).element_size()
    per_tok = 3 * H * D * e + H * D * 4 + H * 4 + H * D * e
    return state + B * T * per_tok


def _time(fn, args):
    """One timing measurement, in ms.

    do_bench brackets each iteration with a stream-ordered event pair.  Those are
    markers on the GPU timeline, not CPU timestamps, so when the host cannot enqueue
    faster than the GPU drains (this wrapper costs ~41 us/call against 12-26 us
    kernels at small batch) the GPU idles *between* the start event and the launch,
    and that stall lands inside the measured span.  Small-batch rows then report host
    cost rather than kernel cost.

    cudagraph mode captures rep/estimate unrolled calls and times one replay, so no
    host work happens inside the measurement at all.  The trade is that it drops
    do_bench's 256 MB L2 flush between iterations (upstream notes this), so a working
    set that fits in cache is measured warm -- for KDA that is the state pool, which
    at 24 heads is ~6 MB at B=4 but ~400 MB at B=256.  Small-batch cudagraph numbers
    are therefore optimistic relative to a real decode step, where 68 other layers
    run between two touches of the same state.
    """
    if args.timing == "cudagraph":
        return _bench_graph(fn, args.graph_ms, args.n_replays)
    return triton.testing.do_bench(
        fn, warmup=args.warmup, rep=args.rep, quantiles=[0.5, 0.2, 0.8]
    )


def _bench_graph(fn, graph_ms, n_replays):
    """HIP-graph replay, same shape as the a8w8 blockscale preshuffle bench: probe
    with events, capture `graph_ms` worth of launches, replay `n_replays` times.

    Each replay is timed separately rather than lumped into one wall-clock span, so
    the spread across replays is visible instead of being averaged away.  Returns
    (median, p20, p80) of the per-iteration time in ms.

    Caveat worth knowing before trusting a small delta: all replays run one captured
    graph in one process, so they sample the kernel but not JIT state, clocks, or
    allocation.  Measured process-to-process spread at B<=16 reached 71% while
    within-run spread was ~2%, so raising this alone will not make small shapes
    reproducible -- that needs repeated invocations of the whole benchmark.
    """
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


def gluon_lines(args):
    """One line per (BV, NUM_WARPS, SK) x NUM_BUFFERS point of the tuning space."""
    lines = []
    for spec in args.gluon_configs:
        bv, nw, sk = (int(x) for x in spec.split(","))
        if (bv * sk) % (32 * nw):
            sys.exit(f"illegal gluon config {spec}: BV*SK must be a multiple of 32*NW")
        for nb in args.num_buffers:
            for ts in args.tdm_store:
                for tl in args.tdm_load:
                    for csu in args.cache_state_updates:
                        for tf in args.tdm_fused_load:
                            lines.append(
                                f"gluon:{bv},{nw},{sk},{nb},{ts},{tl},{csu},{tf}"
                            )
    return lines


def benchmark(args):
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    D = args.head_dim

    line_vals = []
    for b in args.backends:
        if b == "gluon":
            line_vals += gluon_lines(args)
        elif b != "fla" or HAS_FLA:
            line_vals.append(b)
    line_names = [
        f"gluon BV{v.split(':')[1]}" if v.startswith("gluon:") else BACKENDS[v]
        for v in line_vals
    ]

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
            line_vals=line_vals,
            line_names=line_names,
            plot_name=get_caller_name_no_ext(),
            styles=[],
            ylabel="us" if args.metric == "time" else "TB/s",
            args={},
        )
    ]

    spread = {}

    @triton.testing.perf_report(configs)
    def bench_kda(H, B, T, provider):
        # Reseed per measurement so every backend sees byte-identical inputs, and
        # hand in `out` -- otherwise each timed iteration allocates and zeroes it.
        torch.manual_seed(0)
        inputs = make_inputs(
            B, T, H, D, dtype, args.device, args.paged, args.gate, args.num_accepted
        )
        mem = traffic_bytes(B, T, H, D, dtype, args.paged)
        shared = dict(
            inputs,
            out=torch.empty_like(inputs["v"]),
            output_final_state=True,
            inplace_final_state=True,
            state_v_first=not args.state_k_first,
        )

        if provider.startswith("gluon:"):
            bv, nw, sk, nb, ts, tl, csu, tf = (
                int(x) for x in provider.split(":")[1].split(",")
            )

            def fn():
                fused_recurrent_kda(
                    **shared,
                    BV=bv,
                    SK=sk,
                    num_warps=nw,
                    num_buffers=nb,
                    use_tdm_store=bool(ts),
                    use_tdm_load=bool(tl),
                    cache_state_updates=bool(csu),
                    use_tdm_fused_load=bool(tf),
                )

        elif provider == "triton":

            def fn():
                fused_recurrent_kda_triton(**shared)

        else:

            def fn():
                fla_kda(**shared)

        runs = [_time(fn, args) for _ in range(args.runs)]
        med = sorted(r[0] for r in runs)[len(runs) // 2]
        lo, hi = min(r[1] for r in runs), max(r[2] for r in runs)
        best, worst = min(r[0] for r in runs), max(r[0] for r in runs)
        spread[(H, B, T, provider)] = (best, med, worst, lo, hi)

        ms = best if args.reduce == "best" else med
        if args.metric == "time":
            return ms * 1e3, lo * 1e3, hi * 1e3
        return mem / ms * 1e-9, mem / hi * 1e-9, mem / lo * 1e-9

    bench_kda.run(save_path="." if args.o else None, print_data=True, show_plots=False)
    report_spread(spread, line_vals, line_names, args)


def report_spread(spread, line_vals, line_names, args):
    """perf_report drops the quantile columns before printing, so surface them here:
    without them a 5% delta is unreadable against a noise floor we never measured."""
    name = dict(zip(line_vals, line_names))
    flush = "no L2 flush, cache-warm" if args.timing == "cudagraph" else "L2 flushed"
    print(
        f"\nvariability (us) -- timing={args.timing} ({flush}), "
        + (
            f"{args.runs} run(s) x {args.n_replays} replays "
            f"x {args.graph_ms}ms/graph"
            if args.timing == "cudagraph"
            else f"{args.runs} run(s) x rep={args.rep}ms each"
        )
    )
    hdr = f"{'H':>4}{'B':>5}{'T':>3}  {'backend':<28}{'best':>9}{'med':>9}{'worst':>9}"
    hdr += f"{'run-run':>9}{'p20-p80':>9}"
    print(hdr)
    print("-" * len(hdr))
    for (H, B, T, prov), (best, med, worst, lo, hi) in spread.items():
        rr = worst / best if best else float("nan")
        iq = hi / lo if lo else float("nan")
        flag = "  <-- noisy" if rr > 1.10 or iq > 1.10 else ""
        print(
            f"{H:>4}{B:>5}{T:>3}  {name[prov]:<28}{best * 1e3:9.1f}{med * 1e3:9.1f}"
            f"{worst * 1e3:9.1f}{rr:8.2f}x{iq:8.2f}x{flag}"
        )


def parse_args():
    parser = argparse.ArgumentParser(prog="Benchmark KDA decode", allow_abbrev=False)
    parser.add_argument("--batch_sizes", type=int, nargs="+", default=[4, 16, 64, 256])
    parser.add_argument("--seq_lens", type=int, nargs="+", default=[1, 4, 8])
    parser.add_argument("--num_heads", type=int, nargs="+", default=K3_NUM_HEADS)
    parser.add_argument("--head_dim", type=int, default=K3_HEAD_DIM)
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp32"])
    parser.add_argument(
        "--gluon_configs",
        type=str,
        nargs="+",
        default=["32,4,32"],
        help="BV,NUM_WARPS,SK triples, one benchmark line each",
    )
    parser.add_argument(
        "--num_buffers",
        type=int,
        nargs="+",
        default=[2],
        choices=[1, 2],
        help="1: sync loads; 2: register prefetch of the next token",
    )
    parser.add_argument(
        "--tdm_store",
        type=int,
        nargs="+",
        default=[0],
        choices=[0, 1],
        help="1 routes the state stores through LDS + TDM async_store",
    )
    parser.add_argument(
        "--tdm_load",
        type=int,
        nargs="+",
        default=[0],
        choices=[0, 1],
        help="1 loads the initial state via TDM async_load + LDS",
    )
    parser.add_argument(
        "--cache_state_updates",
        type=int,
        nargs="+",
        default=[0],
        choices=[0, 1],
        help="1 stores each token's (a, k, err) state update instead of a full "
        "state snapshot (paged + inplace + v-first only)",
    )
    parser.add_argument(
        "--tdm_fused_load",
        type=int,
        nargs="+",
        default=[0],
        choices=[0, 1],
        help="1 stages the per-token q/k/g/v operands through an LDS ring "
        "filled by one fused TDM instruction per token",
    )
    parser.add_argument(
        "--num_accepted",
        type=int,
        default=0,
        help="tokens the verifier accepted per sequence (clamped to T); paged only. "
        "0 leaves num_accepted_tokens unset, which skips the cached-update replay",
    )
    parser.add_argument("--gate", action="store_true", default=True)
    parser.add_argument("--no_gate", dest="gate", action="store_false")
    parser.add_argument("--state_k_first", action="store_true", default=False)
    parser.add_argument("--paged", action="store_true", default=False)
    parser.add_argument(
        "--backends",
        type=str,
        nargs="+",
        default=["gluon", "triton"],
        choices=list(BACKENDS),
    )
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--rep", type=int, default=100)
    parser.add_argument(
        "--timing",
        choices=["cudagraph", "do_bench"],
        default="cudagraph",
        help="cudagraph: capture graph_ms worth of launches and replay n_replays "
        "times, so no host work sits inside the measurement (small batch stops "
        "reporting the ~41 us/call wrapper cost) -- but no L2 flush between "
        "iterations. do_bench: per-iteration event pairs with a 256 MB flush",
    )
    parser.add_argument(
        "--n_replays",
        type=int,
        default=5,
        help="cudagraph only: graph replays per measurement, each timed separately",
    )
    parser.add_argument(
        "--graph_ms",
        type=float,
        default=100.0,
        help="cudagraph only: target ms of work captured into one graph",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="repeat the whole do_bench this many times to expose run-to-run drift",
    )
    parser.add_argument(
        "--reduce",
        choices=["best", "median"],
        default="median",
        help="which of the --runs measurements lands in the main table",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "-metric",
        nargs="?",
        const="bandwidth",
        choices=["time", "bandwidth"],
        default="bandwidth",
        help="Metrics for the kernel benchmark.",
    )
    parser.add_argument(
        "-o",
        action="store_true",
        default=False,
        help="Write performance results to CSV file",
    )
    return parser.parse_args()


def run_bench(args):
    if arch_info.get_arch() != "gfx1250":
        sys.exit(f"KDA gluon decode is gfx1250 only, got {arch_info.get_arch()}")
    if "fla" in args.backends and not HAS_FLA:
        print("fla not importable -- dropping the upstream backend.")
        print("  PYTHONPATH=/path/to/flash-linear-attention to enable it")
    torch.manual_seed(0)
    benchmark(args)


def main():
    run_bench(parse_args())


if __name__ == "__main__":
    main()
