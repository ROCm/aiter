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


def make_inputs(B, T, H, D, dtype, device, paged, gate):
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
    }


def traffic_bytes(B, T, H, D, dtype, paged):
    slab = H * D * D * 4
    state = B * slab + (B * T * slab if paged else B * slab)
    e = torch.tensor([], dtype=dtype).element_size()
    per_tok = 3 * H * D * e + H * D * 4 + H * 4 + H * D * e
    return state + B * T * per_tok


def gluon_lines(args):
    """One line per (BV, NUM_WARPS, SK) x NUM_BUFFERS point of the tuning space."""
    lines = []
    for spec in args.gluon_configs:
        bv, nw, sk = (int(x) for x in spec.split(","))
        if (bv * sk) % (32 * nw):
            sys.exit(f"illegal gluon config {spec}: BV*SK must be a multiple of 32*NW")
        for nb in args.num_buffers:
            for nt in args.nt_stream:
                for ts in args.tdm_store:
                    lines.append(f"gluon:{bv},{nw},{sk},{nb},{nt},{ts}")
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
        inputs = make_inputs(B, T, H, D, dtype, args.device, args.paged, args.gate)
        mem = traffic_bytes(B, T, H, D, dtype, args.paged)
        shared = dict(
            inputs,
            out=torch.empty_like(inputs["v"]),
            output_final_state=True,
            inplace_final_state=True,
            state_v_first=not args.state_k_first,
        )

        if provider.startswith("gluon:"):
            bv, nw, sk, nb, nt, ts = (int(x) for x in provider.split(":")[1].split(","))

            def fn():
                fused_recurrent_kda(
                    **shared,
                    BV=bv,
                    SK=sk,
                    num_warps=nw,
                    num_buffers=nb,
                    nt_stream=bool(nt),
                    use_tdm_store=bool(ts),
                )

        elif provider == "triton":

            def fn():
                fused_recurrent_kda_triton(**shared)

        else:

            def fn():
                fla_kda(**shared)

        runs = [
            triton.testing.do_bench(
                fn, warmup=args.warmup, rep=args.rep, quantiles=[0.5, 0.2, 0.8]
            )
            for _ in range(args.runs)
        ]
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
    print(f"\nvariability (us) -- {args.runs} run(s) x rep={args.rep}ms each")
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
        choices=[1, 2, 3],
        help="1: sync loads; 2: register prefetch of the next token; "
        "3: also fires a prologue TDM prefetch of the token burst into L2",
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
        "--nt_stream",
        type=int,
        nargs="+",
        default=[0],
        choices=[0, 1],
        help="1 marks the per-token operands non-temporal (th:TH_LOAD_NT)",
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
