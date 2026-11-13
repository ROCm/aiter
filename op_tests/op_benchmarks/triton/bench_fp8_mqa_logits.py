import torch
import triton

from aiter.ops.triton.attention.fp8_mqa_logits import fp8_mqa_logits
from aiter.ops.triton.utils.types import e4m3_dtype
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
from op_tests.triton_tests.attention.test_fp8_mqa_logits import (
    per_custom_dims_cast_to_fp8,
)


def calculate_tflops(start_inds, end_inds, num_heads_q, head_dim, time_ms):
    time_s = time_ms * 1e-3
    start_inds = start_inds.to("cpu").numpy()
    end_inds = end_inds.to("cpu").numpy()
    total_flops = 0.0
    for i in range(len(start_inds)):
        start = start_inds[i]
        end = end_inds[i]
        total_flops += 2.0 * num_heads_q * head_dim * (end - start)
    # TFLOPs = total FLOPs / (time in seconds * 1e12)
    tflops = total_flops / (time_s * 1e12)

    return tflops


def run_benchmark(args):
    if not 0 < args.seq_q_l <= args.seq_kv_l:
        raise ValueError("query length must be positive and no larger than key length")
    x_names = [
        "batch_size",
        "seq_q_l",
        "seq_kv_l",
        "num_heads_q",
        "head_dim",
        "clean_logits",
        "layout",
    ]
    x_vals_list = [
        [
            args.batch_size,
            args.seq_q_l,
            args.seq_kv_l,
            args.num_heads_q,
            args.head_dim,
            args.clean_logits,
            args.layout,
        ]
    ]
    if args.metric == "time":
        ylabel = "Time (ms)"
    elif args.metric == "throughput":
        ylabel = "TFLOPs"
    else:
        raise NotImplementedError(f"{args.metric} is not supported")

    line_names = [ylabel]
    line_vals = [ylabel]
    benchmark = triton.testing.Benchmark(
        x_names=x_names,
        x_vals=x_vals_list,
        line_arg="unit",
        line_vals=line_vals,
        line_names=line_names,
        styles=[("green", "-")],
        ylabel=ylabel,
        plot_name=get_caller_name_no_ext(),
        args={"metric": args.metric},
    )

    @triton.testing.perf_report([benchmark])
    def bench_fp8_mqa_logits(
        batch_size,
        seq_q_l,
        seq_kv_l,
        num_heads_q,
        head_dim,
        clean_logits,
        layout,
        metric,
        **kwargs,
    ):
        s_q = batch_size * seq_q_l
        s_k = batch_size * seq_kv_l

        def inputs():
            q = torch.randn(
                s_q, num_heads_q, head_dim, device="cuda", dtype=torch.bfloat16
            ).to(e4m3_dtype)
            kv, scales = per_custom_dims_cast_to_fp8(
                torch.randn(s_k, head_dim, device="cuda", dtype=torch.bfloat16),
                (0,),
                False,
            )
            weights = torch.randn(s_q, num_heads_q, device="cuda", dtype=torch.float32)
            row = torch.arange(s_q, device="cuda", dtype=torch.int32)
            starts = row // seq_q_l * seq_kv_l
            visible = seq_kv_l - seq_q_l + row % seq_q_l + 1
            ends = starts + visible
            logits, offsets = None, None
            if layout == "packed":
                spans = (visible + 63) // 64 * 64
                offsets = spans.cumsum(0, dtype=torch.int32) - spans
                capacity = batch_size * sum(
                    (length + 63) // 64 * 64
                    for length in range(seq_kv_l - seq_q_l + 1, seq_kv_l + 1)
                )
                logits = torch.empty(capacity, device="cuda", dtype=torch.float32)
            elif layout == "dense":
                logits = torch.empty(
                    s_q, triton.cdiv(s_k, 256) * 256, device="cuda", dtype=torch.float32
                )[:, :s_k]
            return q, kv, scales, weights, starts, ends, logits, offsets

        bencher = RotatingBufferBench.from_dict(
            get_rot_bufs_kwargs(args) | get_do_bench_graph_kwargs(args)
        )
        pool = bencher.build_pool(
            [inputs],
            [
                [
                    "Q",
                    "KV",
                    "kv_scales",
                    "weights",
                    "cu_starts",
                    "cu_ends",
                    "logits",
                    "out_offsets",
                ]
            ],
        )
        time_ms = bencher.bench(
            lambda **kwargs: fp8_mqa_logits(**kwargs, clean_logits=clean_logits)
        )
        tflops = calculate_tflops(
            pool[0]["cu_starts"], pool[0]["cu_ends"], num_heads_q, head_dim, time_ms
        )

        # Return exactly one scalar depending on which metric is active
        if metric == "time":
            return time_ms
        elif metric == "throughput":
            return tflops
        else:
            raise ValueError("Unknown metric: " + metric)

    bench_fp8_mqa_logits.run(save_path="." if args.o else None, print_data=True)


def main():
    parser = get_parser(kernel_name="FP8 MQA Logits")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size")
    parser.add_argument("--num_heads_q", type=int, default=64, help="num. q heads")
    parser.add_argument("--head_dim", type=int, default=128, help="head dim size")
    parser.add_argument(
        "--layout",
        choices=["allocated", "dense", "packed"],
        default="allocated",
        help="Logits allocation and row layout",
    )
    parser.add_argument(
        "--seq_q_l", type=int, default=4096, help="Input sequence length"
    )
    parser.add_argument(
        "--seq_kv_l", type=int, default=4096, help="Output sequence length"
    )
    parser.add_argument(
        "--clean_logits", type=int, default=1, help="Clean the untouched logits"
    )
    parser.add_argument(
        "-o", action="store_true", help="Write performance results to CSV file"
    )
    override_parser_action(
        parser, action="--metric", choices=["time", "throughput"], default="throughput"
    )
    args = parser.parse_args()
    run_benchmark(args)


if __name__ == "__main__":
    main()
