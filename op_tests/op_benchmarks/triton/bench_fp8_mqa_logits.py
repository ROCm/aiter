import argparse

import torch
import triton

from aiter.ops.triton.attention.fp8_mqa_logits import fp8_mqa_logits
from aiter.ops.triton.utils.types import e4m3_dtype
from op_tests.op_benchmarks.triton.utils.benchmark_utils import (
    get_caller_name_no_ext,
)
from op_tests.op_benchmarks.triton.utils.profiler_bench import do_bench_profiler
from op_tests.triton_tests.attention.test_fp8_mqa_logits import (
    per_custom_dims_cast_to_fp8,
)

# Matches both the triton (_fp8_mqa_logits_kernel) and the gluon
# (_gluon_fp8_mqa_logits_kernel) variant of the op.
KERNEL_NAME = "fp8_mqa_logits_kernel"
# With clean_logits the op pre-fills its fp32 output with -inf via torch.full,
# which is a real part of its cost.
FILL_KERNEL_NAME = "FillFunctor<float>"


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
    x_names = [
        "batch_size",
        "seq_q_l",
        "seq_kv_l",
        "num_heads_q",
        "head_dim",
        "clean_logits",
    ]
    x_vals_list = [
        [
            args.batch_size,
            args.seq_q_l,
            args.seq_kv_l,
            args.num_heads_q,
            args.head_dim,
            args.clean_logits,
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
        args={"metric": args.metric, "flush_l2": args.flush_l2},
    )

    @triton.testing.perf_report([benchmark])
    def bench_fp8_mqa_logits(
        batch_size,
        seq_q_l,
        seq_kv_l,
        num_heads_q,
        head_dim,
        clean_logits,
        metric,
        flush_l2,
        **kwargs,
    ):
        s_q = batch_size * seq_q_l
        s_k = batch_size * seq_kv_l

        q = torch.randn(s_q, num_heads_q, head_dim, device="cuda", dtype=torch.bfloat16)
        kv = torch.randn(s_k, head_dim, device="cuda", dtype=torch.bfloat16)
        kv_fp8, scales = per_custom_dims_cast_to_fp8(kv, (0,), False)
        kv = (kv_fp8.to(torch.float32) * scales[:, None]).to(torch.bfloat16)
        weights = torch.randn(s_q, num_heads_q, device="cuda", dtype=torch.float32)

        ks = torch.zeros(s_q, dtype=torch.int, device="cuda")
        ke = torch.zeros(s_q, dtype=torch.int, device="cuda")
        arange_q = torch.arange(seq_q_l, dtype=torch.int, device="cuda")
        for b in range(batch_size):
            qs = b * seq_q_l
            kvs = b * seq_kv_l
            ks[qs : qs + seq_q_l] = kvs
            ke[qs : qs + seq_q_l] = kvs + (seq_kv_l - seq_q_l) + arange_q + 1

        q_fp8 = q.to(e4m3_dtype)

        def func():
            return fp8_mqa_logits(q_fp8, kv_fp8, scales, weights, ks, ke, clean_logits)

        kernel_names = [KERNEL_NAME]
        if clean_logits:
            kernel_names.append(FILL_KERNEL_NAME)

        time_ms = do_bench_profiler(
            func,
            kernel_names,
            num_warmup=25,
            num_iters=100,
            flush_l2=flush_l2,
            return_mode="median",
        )
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
    parser.add_argument("--batch_size", type=int, default=1, help="batch size")
    parser.add_argument("--num_heads_q", type=int, default=64, help="num. q heads")
    parser.add_argument("--head_dim", type=int, default=128, help="head dim size")
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
        "--flush_l2",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Flush the L2 cache before every benchmark iteration",
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
