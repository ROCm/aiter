import sys
import argparse
import triton
import triton.language as tl
from aiter.ops.triton.moe_op import fused_moe as triton_moe
from utils.moe import generate_moe_alignment, generate_moe_input, get_optimal_moe_config
from utils.benchmark_utils import (
    get_model_configs,
    get_available_models,
)
from utils.common import str_to_torch_dtype, torch_to_tl_dtype


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


def run_benchmark(args):
    routed_weight = args.routed_weight
    input_dtype = args.input_dtype
    aux_dtype = args.aux_dtype
    group_size = args.group_size
    has_zp = args.has_zp

    assert (
        input_dtype != "int4_w4a16" or group_size is not None
    ), "User has to set group_size explicitly for int4 weight"

    kernel_name = "fused_moe_kernel"
    if input_dtype in ["int8_w8a16", "int4_w4a16"] and group_size > 0:
        kernel_name = "fused_moe_kernel_gptq_awq"
    input_dtype_str = "" if input_dtype is None else input_dtype
    kernel_name += input_dtype_str + "_" + aux_dtype

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
        args={"aux_dtype": aux_dtype, "input_dtype": input_dtype},
    )

    @triton.testing.perf_report([benchmark])
    def bench_moe_gemm(M, N, K, E, top_k, metric, input_dtype, aux_dtype, model=None):

        a, b, triton_out, _, b_zp, a_scale, b_scale = generate_moe_input(
            M, N, K, top_k, E, group_size, has_zp, input_dtype, aux_dtype
        )
        config = get_optimal_moe_config(input_dtype, aux_dtype, M)
        topk_weights, topk_ids, sorted_token_ids, expert_ids, num_tokens_post_padded = (
            generate_moe_alignment(M, E, top_k, config["BLOCK_SIZE_M"])
        )
        aux_dtype = str_to_torch_dtype[aux_dtype]
        # (M, K) * (top_k, N, K) -> (M, top_k, N). 2 for multiplication and accumulation
        flops = 2.0 * M * top_k * K * N
        # The weight is applied on the gemm product which has the shape of (M, top_k, N)
        if routed_weight:
            flops += M * top_k * N

        # Variables to compute bandwidth
        mem_read = a.numel() * a.element_size() + b.numel() * b.element_size()
        mem_write = triton_out.numel() * triton_out.element_size()
        mem = mem_read + mem_write

        fp8_w8a8 = input_dtype == "fp8_w8a8"
        int8_w8a16 = input_dtype == "int8_w8a16"
        fn = lambda: triton_moe(  # noqa: E731
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
            config,
            torch_to_tl_dtype[aux_dtype],
            fp8_w8a8,
            int8_w8a16,
            use_int4_w4a16=False,
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
    parser.add_argument("--model", type=str, default=None, help=model_help)
    parser.add_argument("-M", type=int, default=0, help="M dimension")
    parser.add_argument(
        "--group-size", type=int, default=None, help="group_size for int4"
    )
    parser.add_argument("--routed-weight", action="store_true")
    parser.add_argument(
        "--input-dtype",
        type=str,
        choices=["int8_w8a16", "fp8_w8a8", "int4_w4a16", None],
        default=None,
    )
    parser.add_argument("--aux-dtype", default="fp16")
    parser.add_argument("--has-zp", action="store_true")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    run_benchmark(args)


if __name__ == "__main__":
    sys.exit(main())
