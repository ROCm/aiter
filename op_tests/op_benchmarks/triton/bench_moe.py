import sys
import argparse
import torch
import triton
import triton.language as tl
from aiter.ops.triton.utils.types import (
        torch_to_triton_dtype,
        str_to_torch_dtype)
from aiter.ops.triton.moe_op import fused_moe as triton_moe
from op_tests.triton_tests.test_moe import input_helper, input_helper_int4_w4a16
from utils.benchmark_utils import (
    get_model_configs,
    get_available_models,
)


def model_benchmark_configs(args):
    config_file = args.model_configs
    configs = get_model_configs(
        config_path=config_file, models="mistral" if args.model is None else args.model
    )
    moe_configs = []
    M = args.M if args.M else 4096  # check size
    # M, K, N, E, top_k

    for model_name, config in configs.items():
        N1 = config["intermediate_size"]
        K1 = config["hidden_size"]

        N2 = config["hidden_size"]
        K2 = config["intermediate_size"] // 2

        E = 8
        top_k = 2

        moe_configs.append((model_name, M, N1, K1, E, top_k))
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
):
    if int4_w4a16:
        (
            a,
            b,
            triton_out,
            _,
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

        return lambda: triton_moe(  # noqa: E731
            a,
            b,
            triton_out,
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
            _,
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

        return lambda: triton_moe(
            a,
            b,
            triton_out,
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
    dtype = str_to_torch_dtype[args.dtype]
    fp8_type = str_to_torch_dtype[args.fp8_type]

    if int4_w4a16:
        assert group_size != None, "set group_size with -group_size"

    kernel_name = "_fused_moe_kernel"
    if (int8_w8a16 or int4_w4a16) and (group_size is not None) and group_size > 0:
        kernel_name = "_fused_moe_kernel_gptq_awq"

    x_vals_list = model_benchmark_configs(args)
    x_names = ["model", "M", "N", "K", "E", "top_k"]

    line_names = ["Time (ms)", "TFLOPS", "Bandwidth (GB/s)"]
    line_vals = ["time", "tflops", "bandwidth"]

    benchmark = triton.testing.Benchmark(
        x_names=x_names,
        x_vals=x_vals_list,
        line_arg="metric",
        line_vals=line_vals,
        line_names=line_names,
        styles=[("red", "-"), ("blue", "-"), ("yellow", "-")],
        ylabel="ms / TFLOPS / GB/s",
        plot_name=f"{kernel_name}-benchmark",
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

        # (M, K) memory load for A (E,  N,  K) for B not (top_k,  N,  K) because we are in total bringing in all expert matrices into the chip from memory. It's just that not all multiply the same A.
        mem_read = (M * K) * a_bytes + (E * N * K) * b_bytes

        mem_write = (M * top_k * N) * c_bytes
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

    bench_moe_gemm.run(save_path=".", print_data=True)


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
    parser.add_argument("-model", type=str, default=None, help=model_help)
    parser.add_argument("-M", type=int, default=0, help="M dimension")
    parser.add_argument(
        "-group_size", type=int, default=None, help="group_size for in4"
    )
    parser.add_argument("-routed_weight", action="store_true", default=False)
    parser.add_argument("-int8_w8a16", action="store_true", default=False)
    parser.add_argument("-fp8_w8a8", action="store_true", default=False)
    parser.add_argument("-int4_w4a16", action="store_true", default=False)
    parser.add_argument("-has_zp", action="store_true", default=False)
    parser.add_argument("-dtype", default="fp16")
    parser.add_argument("-fp8_type", default="e5m2fnuz")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    run_benchmark(args)


if __name__ == "__main__":
    sys.exit(main())
