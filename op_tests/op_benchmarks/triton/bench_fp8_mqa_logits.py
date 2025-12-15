import torch
import triton
import argparse
from aiter.ops.triton.fp8_mqa_logits import fp8_mqa_logits
from aiter.ops.triton.utils.types import e4m3_dtype
from op_tests.triton_tests.attention.test_fp8_mqa_logits import (
    per_custom_dims_cast_to_fp8,
    generate_cp_test_data,
)
from op_tests.op_benchmarks.triton.utils.benchmark_utils import (
    print_vgpr,
    get_caller_name_no_ext,
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
    x_names = ["seq_q_l", "seq_kv_l", "num_heads_q", "head_dim"]
    x_vals_list = [[args.seq_q_l, args.seq_kv_l, args.num_heads_q, args.head_dim]]
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
        seq_q_l, seq_kv_l, num_heads_q, head_dim, metric, **kwargs
    ):
        q = torch.randn(
            seq_q_l, num_heads_q, head_dim, device="cuda", dtype=torch.bfloat16
        )
        kv = torch.randn(seq_kv_l, head_dim, device="cuda", dtype=torch.bfloat16)
        weights = torch.randn(seq_q_l, num_heads_q, device="cuda", dtype=torch.float32)

        ks = torch.zeros(seq_q_l, dtype=torch.int, device="cuda")
        ke = torch.arange(seq_q_l, dtype=torch.int, device="cuda") + (
            seq_kv_l - seq_q_l
        )

        q_fp8 = q.to(e4m3_dtype)
        kv_fp8, scales = per_custom_dims_cast_to_fp8(kv, (0,), False)

        func = lambda: fp8_mqa_logits(q_fp8, kv_fp8, scales, weights, ks, ke)

        time_ms = triton.testing.do_bench(func, warmup=25, rep=100)
        tflops = calculate_tflops(ks, ke, num_heads_q, head_dim, time_ms)

        # Return exactly one scalar depending on which metric is active
        if metric == "time":
            return time_ms
        elif metric == "throughput":
            return tflops
        else:
            raise ValueError("Unknown metric: " + metric)

    bench_fp8_mqa_logits.run(save_path="." if args.o else None, print_data=True)


def main():
    parser = argparse.ArgumentParser(
        description="FP8 MQA Logits Benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--num_heads_q", type=int, default=64, help="num. q heads")
    parser.add_argument("--head_dim", type=int, default=128, help="head dim size")
    parser.add_argument(
        "--seq_q_l", type=int, default=4096, help="Input sequence length"
    )
    parser.add_argument(
        "--seq_kv_l", type=int, default=4096, help="Output sequence length"
    )
    parser.add_argument(
        "-o", action="store_true", help="Write performance results to CSV file"
    )
    parser.add_argument(
        "--metric",
        type=str,
        choices=["time", "throughput"],
        default="throughput",
        help="metric to plot",
    )
    args = parser.parse_args()
    run_benchmark(args)


if __name__ == "__main__":
    main()
