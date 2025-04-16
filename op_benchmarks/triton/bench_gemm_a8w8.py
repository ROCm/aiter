import argparse
import sys
import torch
import triton
import triton.language as tl
from aiter.ops.triton.gemm_a8w8 import gemm_a8w8
from op_tests.triton.test_gemm_a8w8 import generate_gemm_a8w8_inputs


def get_x_vals():
    x_vals = [
        # qkv_proj
        (1, 1280, 8192),
        (32, 1280, 8192),
        (64, 1280, 8192),
        (128, 1280, 8192),
        (192, 1280, 8192),
        (256, 1280, 8192),
        (320, 1280, 8192),
        (512, 1280, 8192),
        (1024, 1280, 8192),
        (2048, 1280, 8192),
        (4096, 1280, 8192),
        (8192, 1280, 8192),
        (16384, 1280, 8192),
    ]
    return x_vals


def run_benchmark(args):
    M = args.M
    N = args.N
    K = args.K

    x_vals_list = get_x_vals()
    x_names = ['M', 'N', 'K']

    if args.metric == 'time':
        ylabel = 'Time (ms)'
    elif args.metric == 'throughput':
        ylabel = 'Throughput (TFLOPS)'
    elif args.metric == 'bandwidth':
        ylabel = 'Bandwidth (GB/s)'
    else:
        raise NotImplementedError(f"{args.metric} is not supported")
    line_names = ["Triton"]
    line_vals = ['triton']

    benchmark = triton.testing.Benchmark(
        x_names=x_names, x_vals=x_vals_list,
        line_arg='provider', line_vals=line_vals, line_names=line_names,
        styles=[('green', '-')],
        ylabel=ylabel, plot_name=f'GEMM A8W8 Benchmark', args={"metric": args.metric})

    @triton.testing.perf_report([benchmark])
    def bench_gemm_a8w8(M, N, K, metric, provider):
        # NOTE: Assume bias and output has the same dtype
        c_dtype = torch.bfloat16
        x, weight, x_scale, w_scale, bias = \
            generate_gemm_a8w8_inputs(M, N, K, c_dtype)
        # flops
        flops = 2.0 * M * N * K
        # memory transfer
        mem_read = (M * K) * x.element_size() + (N * K) * weight.element_size()
        mem_write = (M * N) * bias.element_size()
        mem = mem_read + mem_write

        ms = triton.testing.do_bench(
            lambda: gemm_a8w8(x, weight, x_scale, w_scale, bias, c_dtype),
            warmup=25, rep=100)

        # Return exactly one scalar depending on which metric is active
        if metric == 'time':
            return ms
        elif metric == 'throughput':
            tflops = flops / ms * 1e-9
            return tflops
        elif metric == 'bandwidth':
            bandwidth = mem / (ms * 1e-3) * 1e-9  # GB/s
            return bandwidth
        else:
            raise ValueError("Unknown metric: " + metric)

    bench_gemm_a8w8.run(save_path=".", print_data=True)


def parse_args():
    parser = argparse.ArgumentParser(
        prog="Benchmark A8W8 GEMM",
        allow_abbrev=False,
    )
    parser.add_argument("-M", type=int, default=0, help="M dimension")
    parser.add_argument("-N", type=int, default=0, help="N dimension")
    parser.add_argument("-K", type=int, default=0, help="K dimension")
    parser.add_argument("--metric", type=str, choices=["time", "throughput", "bandwidth"], default="throughput", help="metric to plot")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    run_benchmark(args)


if __name__ == '__main__':
    sys.exit(main())
