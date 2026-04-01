import sys
import argparse
import torch
import triton
from aiter.ops.triton.utils.types import torch_to_triton_dtype, str_to_torch_dtype
from aiter.ops.triton.moe.moe_op import fused_moe as triton_moe
from aiter.ops.triton.moe.moe_op_silu_fused import fused_moe_silu as triton_moe_silu
from aiter.ops.triton.fusions.fused_silu_mul import fused_silu_mul_last_dim
from op_tests.triton_tests.moe.test_moe import input_helper, input_helper_int4_w4a16
from op_tests.op_benchmarks.triton.utils.benchmark_utils import (
    get_model_configs,
    get_available_models,
    print_vgpr,
    get_caller_name_no_ext,
)


def model_benchmark_configs(args):
    no_bench_stage2 = args.no_bench_stage2
    config_file = args.model_configs
    configs = get_model_configs(
        config_path=config_file, models="mixtral" if args.model is None else args.model
    )
    moe_configs = []
    M = args.M if args.M else 4096  # check size
    # M, K, N, E, top_k

    for model_name, config in configs.items():
        N1 = config["intermediate_size"]
        K1 = config["hidden_size"]
        if no_bench_stage2:
            N2 = config["hidden_size"]
            K2 = config["intermediate_size"] // 2

        E = config["num_expert"]
        top_k = config["top_k"]

        moe_configs.append((model_name, M, N1, K1, E, top_k))
        if no_bench_stage2:
            moe_configs.append((model_name, M, N2, K2, E, top_k))

    return moe_configs


def silu_mul_benchmark_configs(args):
    """Like ``model_benchmark_configs`` but supports multiple M via ``-silu_mul_M_list``."""
    configs = get_model_configs(
        config_path=args.model_configs,
        models="mixtral" if args.model is None else args.model,
    )
    if not configs:
        return []
    no_bench_stage2 = args.no_bench_stage2
    if args.silu_mul_M_list:
        default_ms = [
            int(x.strip())
            for x in args.silu_mul_M_list.split(",")
            if x.strip()
        ]
    elif args.M:
        default_ms = [args.M]
    else:
        default_ms = [4096]

    moe_configs = []
    for model_name, config in configs.items():
        ms_model = config.get("silu_mul_benchmark_M", default_ms)
        if not isinstance(ms_model, list):
            ms_model = [int(ms_model)]
        else:
            ms_model = [int(m) for m in ms_model]
        for M in ms_model:
            N1 = config["intermediate_size"]
            K1 = config["hidden_size"]
            E = config["num_expert"]
            top_k = config["top_k"]
            moe_configs.append((model_name, M, N1, K1, E, top_k))
            if no_bench_stage2:
                N2 = config["hidden_size"]
                K2 = config["intermediate_size"] // 2
                moe_configs.append((model_name, M, N2, K2, E, top_k))
    return moe_configs


def fused_moe(
    M,
    N,
    K,
    top_k,
    E,
    routed_weight=False,
    dtype=torch.float16,
    int4_w4a16=False,
    fp8_w8a8=False,
    int8_w8a16=False,
    group_size=128,
    has_zp=True,
    silu_fused=False,
):
    moe_fn = triton_moe_silu if silu_fused else triton_moe

    if int4_w4a16:
        (
            a,
            b,
            triton_out,
            triton_out_silu,
            b_zp,
            b_scale,
            topk_weights,
            topk_ids,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            config,
        ) = input_helper_int4_w4a16(
            M,
            N,
            K,
            top_k,
            E,
            routed_weight=routed_weight,
            dtype=dtype,
            group_size=group_size,
            has_zp=has_zp,
        )

        return lambda: moe_fn(  # noqa: E731
            a,
            b,
            triton_out_silu if silu_fused else triton_out,
            None,
            b_scale,
            b_zp,
            topk_weights,
            topk_ids,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            routed_weight,
            top_k,
            torch_to_triton_dtype[dtype],
            use_fp8_w8a8=False,
            use_int8_w8a16=False,
            use_int4_w4a16=True,
            block_shape=(0, group_size),
            config=config,
        )
    else:
        (
            a,
            b,
            triton_out,
            triton_out_silu,
            b_zp,
            a_scale,
            b_scale,
            topk_weights,
            topk_ids,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            config,
        ) = input_helper(
            M,
            N,
            K,
            top_k,
            E,
            routed_weight=routed_weight,
            dtype=dtype,
            fp8_w8a8=fp8_w8a8,
            int8_w8a16=int8_w8a16,
        )

        return lambda: moe_fn(
            a,
            b,
            triton_out_silu if silu_fused else triton_out,
            a_scale,
            b_scale,
            b_zp,
            topk_weights,
            topk_ids,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            routed_weight,
            top_k,
            torch_to_triton_dtype[dtype],
            fp8_w8a8,
            int8_w8a16,
            use_int4_w4a16=False,
            config=config,
        )


def run_benchmark(args):
    routed_weight = args.routed_weight
    int8_w8a16 = args.int8_w8a16
    fp8_w8a8 = args.fp8_w8a8
    int4_w4a16 = args.int4_w4a16
    group_size = args.group_size
    has_zp = args.has_zp
    print_time = args.print_time
    silu_fused = args.silu_fused
    dtype = str_to_torch_dtype[args.dtype]
    fp8_type = str_to_torch_dtype[args.fp8_type]

    if silu_fused:
        args.no_bench_stage2 = True

    if int4_w4a16:
        assert group_size is not None, "set group_size with -group_size"

    kernel_name = "_fused_moe_kernel"
    if (int8_w8a16 or int4_w4a16) and (group_size is not None) and group_size > 0:
        kernel_name = "_fused_moe_kernel_gptq_awq"

    x_vals_list = model_benchmark_configs(args)
    x_names = ["model", "M", "N", "K", "E", "top_k"]

    if print_time:
        line_names = ["Time_(ms)"]
        line_vals = ["time"]
    else:
        line_names = ["Time_(ms)", "TFLOPS", "Bandwidth_(GB/s)"]
        line_vals = ["time", "tflops", "bandwidth"]

    benchmark = triton.testing.Benchmark(
        x_names=x_names,
        x_vals=x_vals_list,
        line_arg="metric",
        line_vals=line_vals,
        line_names=line_names,
        styles=[("red", "-"), ("blue", "-"), ("yellow", "-")],
        ylabel="ms / TFLOPS / GB/s",
        plot_name=get_caller_name_no_ext(),
        args={},
    )

    @triton.testing.perf_report([benchmark])
    def bench_moe_gemm(M, N, K, E, top_k, metric, model=None):

        # (M, K) * (top_k, N, K) -> (M, top_k, N). 2 for multiplication and accumulation
        flops = 2.0 * M * top_k * K * N
        # The weight is applied on the gemm product which has the shape of (M, top_k, N)
        if routed_weight:
            flops += M * top_k * N

        if fp8_w8a8:
            a_bytes = b_bytes = torch.tensor([], dtype=fp8_type).element_size()
            c_bytes = torch.tensor([], dtype=dtype).element_size()
        elif int8_w8a16:
            b_bytes = torch.tensor([], dtype=torch.int8).element_size()
            a_bytes = c_bytes = torch.tensor([], dtype=dtype).element_size()
        else:
            a_bytes = b_bytes = c_bytes = torch.tensor([], dtype=dtype).element_size()
        # TODO add the int4 case

        max_expert_loaded = min(E, top_k * M)
        # (M, K) memory load for A (E,  N,  K) for B not (top_k,  N,  K) because we are in total bringing in all expert matrices into the chip from memory. It's just that not all multiply the same A.
        mem_read = (M * K) * a_bytes + (max_expert_loaded * N * K) * b_bytes

        mem_write = (M * top_k * N) * c_bytes
        if silu_fused:
            mem = mem_read + (mem_write // 2)
            flops += M * top_k * N
        else:
            mem = mem_read + mem_write

        fn = fused_moe(
            M,
            N,
            K,
            top_k,
            E,
            routed_weight=routed_weight,
            dtype=torch.float16,
            int4_w4a16=int4_w4a16,
            fp8_w8a8=fp8_w8a8,
            int8_w8a16=int8_w8a16,
            group_size=group_size,
            has_zp=has_zp,
            silu_fused=silu_fused,
        )

        ms = triton.testing.do_bench(fn, warmup=25, rep=100)

        bandwidth = mem / (ms * 1e-3) * 1e-9  # GB/s
        tflops = flops / ms * 1e-9

        # Return exactly one scalar depending on which metric is active
        if metric == "time":
            return ms
        elif metric == "tflops":
            return tflops
        elif metric == "bandwidth":
            return bandwidth
        else:
            raise ValueError("Unknown metric: " + metric)

    bench_moe_gemm.run(save_path="." if args.o else None, print_data=True)


def run_silu_mul_benchmark(args):
    """Benchmark last-dim fused SiLU-and-mul (same activation as silu-fused MoE)."""
    print_time = args.print_time
    dtype = str_to_torch_dtype[args.dtype]

    if print_time:
        line_names = ["Time_(ms)"]
        line_vals = ["time"]
    else:
        line_names = ["Time_(ms)", "GFLOPS", "Bandwidth_(GB/s)"]
        line_vals = ["time", "gflops", "bandwidth"]

    x_vals_list = silu_mul_benchmark_configs(args)
    x_names = ["model", "M", "N", "K", "E", "top_k"]

    benchmark = triton.testing.Benchmark(
        x_names=x_names,
        x_vals=x_vals_list,
        line_arg="metric",
        line_vals=line_vals,
        line_names=line_names,
        styles=[("red", "-"), ("blue", "-"), ("yellow", "-")],
        ylabel="ms / GFLOPS / GB/s",
        plot_name=get_caller_name_no_ext() + "_silu_mul",
        args={},
    )

    @triton.testing.perf_report([benchmark])
    def bench_silu_mul(M, N, K, E, top_k, metric, model=None):
        # Match MoE post-GEMM layout: (M * top_k, N); N must be even for gate/up pairs.
        n_even = N if N % 2 == 0 else N - 1
        if n_even < 2:
            return 0.0
        n_rows = M * top_k
        d = n_even // 2
        x = torch.randn(n_rows, n_even, device="cuda", dtype=dtype)
        out = torch.empty(n_rows, d, device="cuda", dtype=dtype)

        elem = torch.tensor([], dtype=dtype).element_size()
        mem_read = n_rows * n_even * elem
        mem_write = n_rows * d * elem
        # Rough op count: SiLU + mul per output element
        flops = float(n_rows * d * 8)

        def fn():
            return fused_silu_mul_last_dim(x, out)

        ms = triton.testing.do_bench(fn, warmup=25, rep=100)
        bandwidth = (mem_read + mem_write) / (ms * 1e-3) * 1e-9
        gflops = flops / ms * 1e-6

        if metric == "time":
            return ms
        elif metric == "gflops":
            return gflops
        elif metric == "bandwidth":
            return bandwidth
        else:
            raise ValueError("Unknown metric: " + metric)

    bench_silu_mul.run(save_path="." if args.o else None, print_data=True)


def parse_args():
    parser = argparse.ArgumentParser(
        prog="Benchmark MoE GEMM",
        allow_abbrev=False,
    )
    parser.add_argument(
        "-model_configs",
        type=str,
        default="utils/model_configs.json",
        help="Model config json file.",
    )
    available_models = get_available_models()  # Dynamically load model names
    model_help = (
        "Model name to benchmark. Select from: ["
        + ", ".join(available_models)
        + "]. Use 'all' to benchmark all models or leave blank for the default benchmark script."
    )
    parser.add_argument("--model", type=str, default=None, help=model_help)
    parser.add_argument("-M", type=int, default=0, help="M dimension")
    parser.add_argument(
        "-group_size", type=int, default=None, help="group_size for in4"
    )
    parser.add_argument("-routed_weight", action="store_true", default=False)
    parser.add_argument("-int8_w8a16", action="store_true", default=False)
    parser.add_argument("-fp8_w8a8", action="store_true", default=False)
    parser.add_argument("-int4_w4a16", action="store_true", default=False)
    parser.add_argument("-has_zp", action="store_true", default=False)
    parser.add_argument("-print_time", action="store_true", default=False)
    parser.add_argument(
        "-print_vgpr",
        action="store_true",
        default=False,
        help="Print VGPR usage for Triton kernels.",
    )
    parser.add_argument("-no_bench_stage2", action="store_false", default=True)
    parser.add_argument("-dtype", default="fp16")
    parser.add_argument("-fp8_type", default="e5m2fnuz")
    parser.add_argument("-silu_fused", action="store_true", default=False)
    parser.add_argument(
        "-bench_silu_mul",
        action="store_true",
        default=False,
        help="Benchmark fused last-dim SiLU-and-mul only (uses model M, N, top_k).",
    )
    parser.add_argument(
        "-silu_mul_M_list",
        type=str,
        default=None,
        help="Comma-separated token counts M for silu_mul bench (e.g. 4,8193,7238). "
        "Row count is M * top_k. Implies multiple table rows when set.",
    )
    parser.add_argument(
        "-o", action="store_true", help="Write performance results to CSV file"
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    if args.bench_silu_mul:
        if args.print_vgpr:

            def fun():
                return run_silu_mul_benchmark(args)

            print_vgpr(fun, get_caller_name_no_ext() + "_silu_mul")
            return 0
        run_silu_mul_benchmark(args)
        return 0

    if args.print_vgpr:
        print("Retrieving VGPR usage for Triton kernels...")

        def fun():
            return run_benchmark(args)

        print_vgpr(fun, get_caller_name_no_ext())
        return 0
    run_benchmark(args)


if __name__ == "__main__":
    sys.exit(main())
