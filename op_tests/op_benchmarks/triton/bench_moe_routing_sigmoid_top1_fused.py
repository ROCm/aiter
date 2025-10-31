import argparse
from functools import partial

import torch
import triton

from aiter.ops.triton.moe_routing_sigmoid_top1_fused import (
    _routing_sigmoid_top1_kernel,
    routing_sigmoid_top1,
)
from op_tests.op_benchmarks.triton.utils.benchmark_utils import (
    get_available_models,
    get_model_configs,
    get_caller_name_no_ext,
    get_evaluation_label,
    get_evaluation_unit,
)
from op_tests.triton_tests.test_moe_routing_sigmoid_top1_fused import (
    torch_routing_sigmoid_top1,
)


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

    line_names = [
        get_evaluation_label("time"),
        get_evaluation_label("throughput", only_unit=True),
        get_evaluation_label("bandwidth"),
    ]
    line_vals = ["time", "throughput", "bandwidth"]

    benchmark = triton.testing.Benchmark(
        x_names=x_names,
        x_vals=x_vals_list,
        line_arg="metric",
        line_vals=line_vals,
        line_names=line_names,
        styles=[("red", "-"), ("blue", "-"), ("yellow", "-")],
        ylabel=f"{get_evaluation_label("time", space=True)} / "
        + f"{get_evaluation_label("throughput", space=True)} / "
        + f"{get_evaluation_label("bandwidth", space=True)}",
        plot_name=get_caller_name_no_ext(),
        args={},
    )

    @triton.testing.perf_report([benchmark])
    def bench_routing_layer(M, N, K, metric, **kwargs):
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
        if metric == "time":
            return ms
        elif metric == "throughput":
            tflops = flops / ms * 1e-9
            return tflops
        elif metric == "bandwidth":
            bandwidth = mem / (ms * 1e-3) * 1e-9  # GB/s
            return bandwidth
        else:
            raise ValueError("Unknown metric: " + metric)

    bench_routing_layer.run(save_path="." if args.o else None, print_data=True)


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
    parser = argparse.ArgumentParser(
        prog="Benchmark Routing Sigmoid Top1",
        allow_abbrev=False,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    available_models = get_available_models()
    model_help = (
        "Model name to benchmark. Select from: ["
        + ", ".join(available_models)
        + "]. Use 'all' to benchmark all models or leave blank for the default benchmark script."
    )
    parser.add_argument(
        "--model-configs",
        type=str,
        default="utils/model_configs.json",
        help="Model config json file.",
    )
    parser.add_argument("--model", type=str, help=model_help)
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
    parser.add_argument(
        "-o", action="store_true", help="Write performance results to CSV file"
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
