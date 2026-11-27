# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import sys

import torch
import torch.nn.functional as F
import triton

from aiter.ops.triton.attention.kda import fused_recurrent_kda
from aiter.ops.triton.attention.kda_gating import kda_gating_decode
from aiter.ops.triton.utils._triton import arch_info
from op_tests.op_benchmarks.triton.utils.argparse import get_parser
from op_tests.op_benchmarks.triton.utils.benchmark_utils import get_caller_name_no_ext
from op_tests.op_benchmarks.triton.utils.rotating_buffer_bench import (
    RotatingBufferBench,
)

try:
    from fla.ops.kda.fused_recurrent import fused_recurrent_kda_fwd as fla_kda

    HAS_FLA = True
except ImportError:
    fla_kda = None
    HAS_FLA = False

BACKENDS = {
    "gluon": "aiter gluon",
    "fla": "fla triton",
    "gating_same": "kda_gating same",
    "gating_split": "kda_gating split",
    "gating_fork": "kda_gating fork",
    "gating_spec": "kda_gating spec verify",
    "recurrent_same": "fused_recurrent_kda same",
    "recurrent_spec": "fused_recurrent_kda spec verify",
}
# kda_gating_decode on gfx1250 in Kimi-K3's layouts, for each H, and
# fused_recurrent_kda on the same inputs where it has one slot table. Only the
# spec backends have more than one token per sequence.
CASE_BACKENDS = (
    "gating_same",
    "gating_split",
    "gating_fork",
    "gating_spec",
    "recurrent_same",
    "recurrent_spec",
)

K3_HEAD_DIM = 128
K3_NUM_HEADS = [24, 12]  # 96 heads sharded tp4 / tp8
K3_LOWER_BOUND = -5.0


def make_inputs(
    B,
    T,
    H,
    D,
    dtype,
    device,
    paged,
    gate,
    num_accepted=0,
    fused=False,
    W=4,
    state_dtype=torch.float32,
):
    """K3 serves `full_k3`: raw q/k/g/beta, gate chain and l2norm fused in."""
    total_T = B * T
    q = torch.rand(1, total_T, H, D, dtype=dtype, device=device)
    k = torch.rand(1, total_T, H, D, dtype=dtype, device=device)
    v = torch.rand(1, total_T, H, D, dtype=dtype, device=device)
    beta = torch.rand(
        1, total_T, H, dtype=dtype if fused else torch.float32, device=device
    )
    cu_seqlens = torch.arange(0, total_T + 1, step=T, device=device).long()

    if gate:
        g_dtype = dtype if fused else torch.float32
        g = torch.randn(1, total_T, H, D, dtype=g_dtype, device=device)
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
        state = torch.randn(total_T, H, D, D, dtype=state_dtype, device=device)
        indices = torch.arange(total_T, device=device, dtype=torch.int32).view(B, T)
    else:
        state = torch.randn(B, H, D, D, dtype=state_dtype, device=device)
        indices = None

    if paged and num_accepted:
        accepted = torch.full(
            (B,), min(num_accepted, T), device=device, dtype=torch.int32
        )
    else:
        accepted = None

    extra = {}
    if fused:
        # spec decoding keeps vLLM's W-2+T tap window, read from num_accepted-1
        taps = W - 1 if accepted is None else W - 2 + T
        extra = {
            "conv_state": torch.randn(
                state.shape[0], 3 * H * D, taps, dtype=dtype, device=device
            ),
            "conv_weight": torch.randn(3, W, H * D, dtype=torch.float32, device=device)
            * 0.1,
            "out_gate": torch.randn_like(v),
            "norm_weight": torch.rand(D, dtype=torch.float32, device=device) + 0.5,
        }

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
        **extra,
    }


def traffic_bytes(
    B,
    T,
    H,
    D,
    dtype,
    device,
    paged,
    csu=False,
    BV=32,
    fused=False,
    W=4,
    state_dtype=torch.float32,
):
    slab = H * D * D * torch.tensor([], dtype=state_dtype, device=device).element_size()
    if csu:
        record = H * (D // BV) * (2 * D + BV) * 4
        writes = B * (slab + (T - 1) * record)
    else:
        writes = B * T * slab if paged else B * slab
    e = torch.tensor([], dtype=dtype, device=device).element_size()
    per_tok = 3 * H * D * e + H * D * 4 + H * 4 + H * D * e
    if fused:
        per_tok += 3 * H * D * (W - 1) * e * 2 + H * D * e
    return B * slab + writes + B * T * per_tok


def make_gating_case(B, T, H, provider, device):
    """Arguments of ``kda_gating_decode`` in Kimi-K3's layouts.

    beta is a view of the fused in_proj row ``[q | k | v | g | beta | f_a]``.
    ``same`` passes the write table as the read table, ``split`` a copy of
    it, and ``fork`` reads each state from a slot that no sequence writes.
    ``spec`` has ``T`` tokens per sequence, a 2D write table and 1 to ``T``
    accepted tokens.
    """
    mode = provider.split("_")[1]
    D = K3_HEAD_DIM
    tokens = B * T
    act = {"dtype": torch.bfloat16, "device": device}
    fp32 = {"dtype": torch.float32, "device": device}
    q, k, v, a = (torch.randn((1, tokens, H, D), **act) for _ in range(4))
    in_proj = torch.randn((tokens, 4 * H * D + H + D), **act)
    # Only the slots that the call reads or writes, so that a rotating pool
    # spans the memory that it measures.
    slots = {"fork": 2 * B, "spec": tokens}.get(mode, B)
    perm = torch.randperm(slots, device=device).to(torch.int32)
    write = perm[:B]
    read = None
    accepted = None
    if mode == "same":
        read = write
    elif mode == "split":
        read = write.clone()
    elif mode == "fork":
        read = perm[B : 2 * B]
    else:
        write = perm[:tokens].view(B, T)
        accepted = torch.randint(1, T + 1, (B,), dtype=torch.int32, device=device)
    return {
        "A_log": torch.randn(H, **fp32) * 0.5,
        "a": a,
        "b": in_proj[:, 4 * H * D : 4 * H * D + H].unsqueeze(0),
        "dt_bias": torch.randn(H * D, **fp32) * 2.0,
        "q": q,
        "k": k,
        "v": v,
        "o": torch.empty((tokens, H, D), **act),
        "initial_state": torch.randn((slots, H, D, D), **fp32),
        "cu_seqlens": torch.arange(0, tokens + 1, T, dtype=torch.int32, device=device),
        "ssm_state_indices": write,
        "ssm_state_indices_in": read,
        "num_accepted_tokens": accepted,
    }


def run_gating_case(case):
    result = kda_gating_decode(
        **case, use_qk_l2norm_in_kernel=True, lower_bound=K3_LOWER_BOUND
    )
    # None means nothing ran, which would time an empty graph.
    if result is None:
        raise RuntimeError("kda_gating_kernel does not cover this case")
    return result


def run_recurrent_case(case):
    """``fused_recurrent_kda`` on a gating case. It reads and writes the
    write-slot table only, so it runs the cases without a fork."""
    return fused_recurrent_kda(
        case["q"],
        case["k"],
        case["v"],
        case["a"],
        case["b"],
        A_log=case["A_log"],
        dt_bias=case["dt_bias"],
        initial_state=case["initial_state"],
        cu_seqlens=case["cu_seqlens"],
        ssm_state_indices=case["ssm_state_indices"],
        num_accepted_tokens=case["num_accepted_tokens"],
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        use_beta_sigmoid_in_kernel=True,
        lower_bound=K3_LOWER_BOUND,
        out=case["o"].unsqueeze(0),
    )


def gating_bytes(case):
    """The activations of each token, one state read for each sequence and one
    state write for each token."""
    _, tokens, h, d = case["q"].shape
    sequences = case["cu_seqlens"].numel() - 1
    act = case["q"].element_size()
    state = case["initial_state"].element_size()
    return tokens * (5 * h * d + h) * act + (sequences + tokens) * h * d * d * state


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


def gluon_lines(args):
    """One line per (BV, NUM_WARPS, SK) x NUM_BUFFERS point of the tuning space."""
    lines = []
    for spec in args.gluon_configs:
        bv, nw, sk = (int(x) for x in spec.split(","))
        if (bv * sk) % (32 * nw):
            sys.exit(f"illegal gluon config {spec}: BV*SK must be a multiple of 32*NW")
        if args.fused and bv != args.head_dim:
            sys.exit(f"--fused needs BV == head_dim ({args.head_dim}), got {spec}")
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
    state_dtype = torch.bfloat16 if args.state_dtype == "bf16" else torch.float32
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
        torch.manual_seed(0)
        if provider in CASE_BACKENDS:
            if T != 1 and not provider.endswith("_spec"):
                # A 1D slot table gives each sequence one token.
                return float("nan"), float("nan"), float("nan")

            def make():
                return make_gating_case(B, T, H, provider, args.device)

            if provider.startswith("recurrent_"):
                run = run_recurrent_case
            else:
                run = run_gating_case
            return _record_case(H, B, T, provider, make, run, gating_bytes)

        inputs = make_inputs(
            B,
            T,
            H,
            D,
            dtype,
            args.device,
            args.paged,
            args.gate,
            args.num_accepted,
            fused=args.fused,
            state_dtype=state_dtype,
        )
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
            mem = traffic_bytes(
                B,
                T,
                H,
                D,
                dtype,
                args.device,
                args.paged,
                csu=bool(csu),
                BV=bv,
                fused=args.fused,
                state_dtype=state_dtype,
            )

            def fn():
                fused_recurrent_kda(
                    **shared,
                    cache_state_updates=bool(csu),
                    config={
                        "BV": bv,
                        "SK": sk,
                        "num_warps": nw,
                        "num_buffers": nb,
                        "use_tdm_store": bool(ts),
                        "use_tdm_load": bool(tl),
                        "use_tdm_fused_load": bool(tf),
                    },
                )

        else:
            mem = traffic_bytes(
                B, T, H, D, dtype, args.device, args.paged, state_dtype=state_dtype
            )

            def fn():
                fla_kda(**shared)

        return _record(H, B, T, provider, lambda: _time(fn, args), mem)

    def _record_case(H, B, T, provider, make, run, bytes_of):
        """Time ``run`` on ``make()``; on a pool of cases for --timing rotating."""
        case = make()
        if args.timing == "rotating":
            bencher = RotatingBufferBench.from_dict(
                {
                    "min_pool_len": 1,
                    "kernels_per_graph": 100,
                    "num_replays": args.n_replays,
                }
            )
            bencher.build_pool([lambda: tuple(make().values())], [tuple(case)])

            def time_once():
                ms = bencher.bench(lambda **entry: run(entry))
                return ms, ms, ms

        else:

            def time_once():
                return _time(lambda: run(case), args)

        return _record(H, B, T, provider, time_once, bytes_of(case))

    def _record(H, B, T, provider, time_once, mem):
        runs = [time_once() for _ in range(args.runs)]
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
    name = dict(zip(line_vals, line_names))
    flush, detail = {
        "cudagraph": (
            "no L2 flush, cache-warm",
            f"{args.n_replays} replays x {args.graph_ms}ms/graph",
        ),
        "rotating": ("a pool of input sets, cold state", f"{args.n_replays} replays"),
        "do_bench": ("L2 flushed", f"rep={args.rep}ms each"),
    }[args.timing]
    print(
        f"\nvariability (us) -- timing={args.timing} ({flush}), "
        f"{args.runs} run(s) x {detail}"
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
    parser = get_parser(kernel_name="KDA decode")
    parser.add_argument("--batch_sizes", type=int, nargs="+", default=[4, 16, 64, 256])
    parser.add_argument("--seq_lens", type=int, nargs="+", default=[1, 4, 8])
    parser.add_argument("--num_heads", type=int, nargs="+", default=K3_NUM_HEADS)
    parser.add_argument("--head_dim", type=int, default=K3_HEAD_DIM)
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp32"])
    parser.add_argument("--state_dtype", default="fp32", choices=["fp32", "bf16"])
    parser.add_argument(
        "--gluon_configs",
        type=str,
        nargs="+",
        default=None,
        help="BV,NUM_WARPS,SK triples, one benchmark line each (default per arch)",
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
    parser.add_argument("--fused", action="store_true", default=False)
    parser.add_argument(
        "--backends",
        type=str,
        nargs="+",
        default=["gluon"],
        choices=list(BACKENDS),
    )
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--rep", type=int, default=100)
    parser.add_argument(
        "--timing",
        choices=["cudagraph", "rotating", "do_bench"],
        default="cudagraph",
        help="cudagraph: capture graph_ms worth of launches and replay n_replays "
        "times, so no host work sits inside the measurement (small batch stops "
        "reporting the ~41 us/call wrapper cost) -- but no L2 flush between "
        "iterations. rotating: a CUDA graph where each call takes the next input "
        "set of a pool larger than the caches (RotatingBufferBench), so each "
        "call reads a cold state, as in a decode step; gating and recurrent "
        "backends only. "
        "do_bench: per-iteration event pairs with a 256 MB flush",
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


def _check_case_shapes(args):
    """The case backends build Kimi-K3's head size, dtypes and state layout."""
    problems = []
    if arch_info.get_arch() not in ("gfx1250", "gfx1250-strict"):
        problems.append(f"gfx1250 (got {arch_info.get_arch()})")
    if args.head_dim != K3_HEAD_DIM:
        problems.append(f"--head_dim {K3_HEAD_DIM}")
    if args.dtype != "bf16" or args.state_dtype != "fp32":
        problems.append("--dtype bf16 --state_dtype fp32")
    if not args.gate or args.fused or args.num_accepted or args.state_k_first:
        problems.append("gate on, and no --fused, --num_accepted, or --state_k_first")
    if problems:
        sys.exit("gating and recurrent backends need " + "; ".join(problems))


def run_bench(args):
    if any(b in CASE_BACKENDS for b in args.backends):
        _check_case_shapes(args)
    if args.timing == "rotating" and any(b not in CASE_BACKENDS for b in args.backends):
        sys.exit("--timing rotating supports the gating and recurrent backends only")
    if arch_info.get_arch() not in ("gfx950", "gfx1250", "gfx1250-strict"):
        sys.exit(f"KDA gluon decode needs gfx950/gfx1250, got {arch_info.get_arch()}")
    if args.gluon_configs is None:
        # The gfx950 kernel holds the whole [V, K] state: BV == V.
        gfx950 = arch_info.get_arch() == "gfx950"
        args.gluon_configs = [f"{args.head_dim},4,16"] if gfx950 else ["32,4,32"]
    if "fla" in args.backends and not HAS_FLA:
        print("fla not importable -- dropping the upstream backend.")
        print("  PYTHONPATH=/path/to/flash-linear-attention to enable it")
    torch.manual_seed(0)
    # Launches, streams, events and graphs use the current device.
    with torch.cuda.device(args.device):
        benchmark(args)


def main():
    run_bench(parse_args())


if __name__ == "__main__":
    main()
