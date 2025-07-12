from functools import partial

import torch
import triton

from aiter.ops.triton.routing import (
    _routing_sigmoid_top1_kernel,
    routing_sigmoid_top1,
    torch_routing_sigmoid_top1,
)
from op_tests.op_benchmarks.triton.utils.argparse import (
    get_parser,
)
from op_tests.op_benchmarks.triton.utils.benchmark_utils import get_model_configs


def _get_compiled(fn):
    compiled_fn = torch.compile(
        fn, backend="inductor", fullgraph=True, options={"max_autotune": True}
    )
    return compiled_fn


def run_benchmark(args, x_vals_list):
    """
    Runs benchmark given the --model argument.
    """
    x_names = ["M", "N", "K"]

    if args.metric == "time":
        ylabel = "Time (ms)"
    elif args.metric == "throughput":
        ylabel = "Throughput (TFLOPS)"
    elif args.metric == "bandwidth":
        ylabel = "Bandwidth (GB/s)"
    else:
        raise NotImplementedError(f"{args.metric} is not supported")
    line_names = ["Triton"]
    line_vals = line_names
    benchmark = triton.testing.Benchmark(
        x_names=x_names,
        x_vals=x_vals_list,
        line_arg="provider",
        line_vals=line_vals,
        line_names=line_names,
        styles=[("green", "-")],  # match line names to colors
        ylabel=ylabel,
        plot_name="Benchmark Routing Layer",
        args={"metric": args.metric},
    )

    @triton.testing.perf_report([benchmark])
    def bench_routing_layer(M, N, K, **kwargs):
        dtype = torch.bfloat16
        device = "cuda"
        x = torch.randn((M, K), dtype=dtype, device=device)
        w = torch.randn((K, N), dtype=dtype, device=device) * 0.1
        TOPK = 1
        flops = 2.0 * M * N * K
        # memory transfer
        mem_read = (M * K) * x.element_size() + (N * K) * w.element_size()
        mem_write = (M * N) * x.element_size()
        mem = mem_read + mem_write

        ms = triton.testing.do_bench(
            lambda: routing_sigmoid_top1(x, w, TOPK), warmup=25, rep=100
        )
        if args.metric == "time":
            return ms
        elif args.metric == "throughput":
            tflops = flops / ms * 1e-9
            return tflops
        elif args.metric == "bandwidth":
            bandwidth = mem / (ms * 1e-3) * 1e-9  # GB/s
            return bandwidth
        else:
            raise ValueError("Unknown metric: " + args.metric)

    bench_routing_layer.run(save_path=".", print_data=True)


def benchmark(M, N, K):
    TOPK = 1

    torch.manual_seed(7)

    dtype = torch.bfloat16
    device = "cuda"

    x = torch.randn((M, K), dtype=dtype, device=device)
    w = torch.randn((K, N), dtype=dtype, device=device) * 0.1

    dummy_ids = torch.ones((M, 1), dtype=torch.int32, device="cuda") * N
    dummy_weights = torch.ones((M, 1), dtype=torch.float32, device="cuda")
    _eager = partial(
        torch_routing_sigmoid_top1, dummy_ids=dummy_ids, dummy_weights=dummy_weights
    )
    _compiled = _get_compiled(_eager)

    def eager_fn():
        return _eager(x, w, TOPK)

    def triton_fn():
        return routing_sigmoid_top1(x, w, TOPK)

    def compile_fn():
        return _compiled(x, w, TOPK)

    # warmup
    for _ in range(5):
        eager_fn()
        triton_fn()
        compile_fn()

    with torch.cuda.stream(torch.cuda.Stream()):
        ms_eager_time = triton.testing.do_bench_cudagraph(eager_fn)

    with torch.cuda.stream(torch.cuda.Stream()):
        ms_triton_time = triton.testing.do_bench_cudagraph(triton_fn)

    with torch.cuda.stream(torch.cuda.Stream()):
        ms_compile_time = triton.testing.do_bench_cudagraph(compile_fn)

    print(
        f"{M=} {K=} {N=} {TOPK=}, "
        f"{ms_eager_time=:.3f}, {ms_triton_time=:.3f}, {ms_compile_time=:.3f}, "
        f"speedup_vs_eager: {ms_eager_time / ms_triton_time:.3f}, "
        f"speedup_vs_compile: {ms_compile_time / ms_triton_time:.3f}\n"
        f"best triton config: {getattr(_routing_sigmoid_top1_kernel, 'best_config', '')}"
    )


def benchmark_prefill():
    print("=== PREFILL SHAPEs ===")
    benchmark(M=1024, K=5120, N=128)
    benchmark(M=1024, K=5120, N=16)
    benchmark(M=2048, K=5120, N=128)
    benchmark(M=2048, K=5120, N=16)
    benchmark(M=4096, K=5120, N=128)
    benchmark(M=4096, K=5120, N=16)
    benchmark(M=8192, K=5120, N=128)
    benchmark(M=8192, K=5120, N=16)


def benchmark_decode():
    print("=== DECODE SHAPEs ===")
    benchmark(M=64, K=5120, N=128)
    benchmark(M=64, K=5120, N=16)
    benchmark(M=128, K=5120, N=128)
    benchmark(M=128, K=5120, N=16)
    benchmark(M=256, K=5120, N=128)
    benchmark(M=256, K=5120, N=16)


def parse_args():
    parser = get_parser(kernel_name="Routing Sigmoid Top1")
    parser.add_argument("-M", type=int, default=1024)
    parser.add_argument("-K", type=int, default=5120)
    parser.add_argument("-N", type=int, default=128)
    parser.add_argument("-benchmark_prefill", action="store_true")
    parser.add_argument("-benchmark_decode", action="store_true")
    parser.add_argument(
        "--shape",
        type=int,
        nargs=3,
        metavar=("M", "N", "K"),
        help="user-defined shape to benchmark.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    assert (
        args.model is None or args.shape is None
    ), "Specify one of --model or --shape."
    if args.model is not None:
        config_file = args.model_configs
        configs = get_model_configs(config_path=config_file, models=args.model)
        x_vals = []
        for _, config in configs.items():
            # layer takes (M, K) as input and produces (M, N) -> N is the number of experts
            assert (
                "n_routed_experts" in config
            ), "Missing 'n_routed_experts' in config. Is this model using MoE?"
            N = config["n_routed_experts"]
            K = config["hidden_size"]
            for M in [2**i for i in range(15)]:
                x_vals.append((M, N, K))
        run_benchmark(args, x_vals)
    elif args.shape is not None:
        M, N, K = args.shape
        run_benchmark(args, [(M, N, K)])
    else:
        # Default
        M, N, K = 1024, 128, 5120
        run_benchmark(args, [(M, N, K)])

    if args.benchmark_prefill:
        benchmark_prefill()
    if args.benchmark_decode:
        # no gain for decode shape
        benchmark_decode()


if __name__ == "__main__":
    main()
