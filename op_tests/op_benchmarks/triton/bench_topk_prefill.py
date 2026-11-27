import triton

from aiter.ops.triton.topk_prefill import top_k_per_row_prefill
from op_tests.op_benchmarks.triton.utils.argparse import (
    get_do_bench_graph_kwargs,
    get_parser,
    get_rot_bufs_kwargs,
    override_parser_action,
)
from op_tests.op_benchmarks.triton.utils.benchmark_utils import (
    get_caller_name_no_ext,
)
from op_tests.op_benchmarks.triton.utils.rotating_buffer_bench import (
    RotatingBufferBench,
)
from op_tests.triton_tests.topk.test_topk_prefill_gluon import (
    generate_topk_prefill_inputs,
)

# K3.1 prefill: 8192-token chunks; the chunk ending at ctx has rows of lengths ctx - 8191 .. ctx
KIMI_K31_SHAPES = [
    ("causal", 8192, 8192),
    ("causal", 8192, 16384),
    ("causal", 8192, 32768),
    ("causal", 8192, 65536),
    ("causal", 8192, 131072),
    ("causal", 8192, 262143),
    ("causal", 5461, 24576),
    ("packed", 8192, 2048),
]


def run_benchmark(args):
    if args.rows is not None:
        shapes = [(args.layout, args.rows, args.ctx)]
    else:
        shapes = KIMI_K31_SHAPES
    x_vals_list = [
        [layout, rows, ctx, args.dist, args.topk] for layout, rows, ctx in shapes
    ]
    if args.metric == "time":
        ylabel = "Time (ms)"
    elif args.metric == "bandwidth":
        ylabel = "Bandwidth (GB/s)"
    else:
        raise NotImplementedError(f"{args.metric} is not supported")

    benchmark = triton.testing.Benchmark(
        x_names=["layout", "rows", "ctx", "dist", "topk"],
        x_vals=x_vals_list,
        line_arg="unit",
        line_vals=[ylabel],
        line_names=[ylabel],
        styles=[("green", "-")],
        ylabel=ylabel,
        plot_name=get_caller_name_no_ext(),
        args={"metric": args.metric},
    )

    @triton.testing.perf_report([benchmark])
    def bench_topk_prefill(layout, rows, ctx, dist, topk, metric, **kwargs):
        bencher = RotatingBufferBench.from_dict(
            get_rot_bufs_kwargs(args) | get_do_bench_graph_kwargs(args)
        )
        pool = bencher.build_pool(
            [lambda: generate_topk_prefill_inputs(dist, layout, rows, ctx, k=topk)],
            [["logits", "row_starts", "row_ends", "indices"]],
        )
        time_ms = bencher.bench(lambda **kw: top_k_per_row_prefill(**kw, k=topk))
        if metric == "time":
            return time_ms
        lengths = (pool[0]["row_ends"] - pool[0]["row_starts"]).clamp(min=0)
        read_bytes = 4 * int(lengths.sum())
        write_bytes = 4 * rows * topk
        return (read_bytes + write_bytes) / (time_ms * 1e-3) / 1e9

    bench_topk_prefill.run(save_path="." if args.o else None, print_data=True)


def main():
    parser = get_parser(kernel_name="DSA prefill top-k")
    parser.add_argument(
        "--layout",
        choices=["causal", "packed"],
        default="causal",
        help="causal: the last rows tokens of one ctx-token prompt; "
        "packed: rows tokens of ctx-token prompts",
    )
    parser.add_argument(
        "--rows", type=int, default=None, help="Rows (default: Kimi-K3.1 sweep)"
    )
    parser.add_argument("--ctx", type=int, default=32768, help="Context length")
    parser.add_argument(
        "--dist",
        choices=[
            "random",
            "relu_sparse",
            "few_values",
            "all_equal",
            "nan_rows",
            "zeros50",
            "bf16",
            "signed_zeros",
            "inf",
            "ascending",
            "recency",
        ],
        default="random",
        help="Logit value distribution",
    )
    parser.add_argument("--topk", type=int, default=2048, help="k per row")
    parser.add_argument(
        "-o", action="store_true", help="Write performance results to CSV file"
    )
    override_parser_action(
        parser, action="--metric", choices=["time", "bandwidth"], default="time"
    )
    args = parser.parse_args()
    run_benchmark(args)


if __name__ == "__main__":
    main()
