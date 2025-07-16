import argparse
import sys
import torch
import triton
from aiter.ops.triton.utils.types import torch_to_triton_dtype, str_to_torch_dtype
import aiter.ops.triton.utils.arch_info as arch_info
from aiter.ops.triton.moe_op_mxfp4 import fused_moe_mxfp4
from op_tests.triton_tests.test_moe import torch_moe_align_block_size_ref
from op_tests.triton_tests.test_moe_mx import (
    alloc_rand,
    torch_dynamic_mxfp4_quant,
)
from op_tests.op_benchmarks.triton.utils.benchmark_utils import (
    get_available_models,
    get_model_configs,
)


def model_benchmark_configs(args):
    config_file = args.model_configs
    configs = get_model_configs(
        config_path=config_file, models="mistral" if args.model is None else args.model
    )
    moe_configs = []
    M = args.M if args.M else 128  # check size
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
    a_dtype_str = args.a_dtype
    b_dtype_str = "mxfp4_e2m1"
    swizzle_mx = args.swizzle_mx

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
        plot_name=f"MoE Benchmark {a_dtype_str} x {b_dtype_str}",
        args={"a_dtype": a_dtype_str, "swizzle_mx": swizzle_mx},
    )

    @triton.testing.perf_report([benchmark])
    def bench_moe_gemm(M, N, K, E, top_k, metric, a_dtype, swizzle_mx, model=None):
        is_a_mixed_input = a_dtype_str.startswith("mx")
        is_b_mixed_input = b_dtype_str.startswith("mx")
        a_dtype = str_to_torch_dtype[a_dtype_str]
        c_dtype = torch.bfloat16 if is_a_mixed_input else a_dtype
        fp16_dtype = torch.float16 if a_dtype_str == "fp16" else torch.bfloat16
        a_tri = alloc_rand((M, K), dtype=fp16_dtype, device="cuda", requires_grad=False)
        b_tri = alloc_rand(
            (E, N, K), dtype=fp16_dtype, device="cuda", requires_grad=False
        )
        c_tri = torch.zeros(
            (M, top_k, N), dtype=c_dtype, device="cuda", requires_grad=False
        )
        a_scale = torch.tensor([1.00], dtype=torch.float32, device="cuda")
        b_scale = torch.tensor([1.00] * E, dtype=torch.float32, device="cuda")

        config = {
            "BLOCK_SIZE_M": 32,
            "BLOCK_SIZE_N": 256,
            "BLOCK_SIZE_K": 128,
            "GROUP_SIZE_M": 4,
            "num_warps": 8,
            "num_stages": 2,
            "waves_per_eu": 0,
            "matrix_instr_nonkdim": 16,
            "kpack": 1,
        }

        values = torch.randn(M, E, dtype=torch.float16, device="cuda")
        softmax_vals = torch.softmax(values, dim=1)
        topk_weights, topk_ids = torch.topk(softmax_vals, k=top_k, dim=1)

        sorted_token_ids, expert_ids, num_tokens_post_padded = (
            torch_moe_align_block_size_ref(topk_ids, config["BLOCK_SIZE_M"], E)
        )

        if is_a_mixed_input:
            a_tri, a_mx_scales = torch_dynamic_mxfp4_quant(a_tri)
        else:
            a_mx_scales = None

        if is_b_mixed_input:
            b_tri, b_mx_scales = torch_dynamic_mxfp4_quant(b_tri)
        # (M, K) * (top_k, N, K) -> (M, top_k, N). 2 for multiplication and accumulation
        flops = 2.0 * M * top_k * K * N
        # The weight is applied on the gemm product which has the shape of (M, top_k, N)
        if routed_weight:
            flops += M * top_k * N

        # Variables to compute bandwidth
        mem_read = (
            a_tri.numel() * a_tri.element_size() + b_tri.numel() * b_tri.element_size()
        )
        mem_write = c_tri.numel() * c_tri.element_size()
        mem = mem_read + mem_write

        fn = lambda: fused_moe_mxfp4(  # noqa: E731
            a_tri,
            b_tri,
            c_tri,
            a_scale,
            b_scale,
            a_mx_scales,
            b_mx_scales,
            topk_weights,
            topk_ids,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            routed_weight,
            top_k,
            swizzle_mx,
            swizzle_mx,
            config,
            torch_to_triton_dtype[c_tri.dtype],
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


def parse_args():
    parser = argparse.ArgumentParser(
        prog="Benchmark MoE with micro scaled format",
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
    parser.add_argument("-M", type=int, help="M dimension")
    parser.add_argument("--routed-weight", action="store_true")
    parser.add_argument("--swizzle-mx", action="store_true")
    parser.add_argument(
        "-A",
        "--a-dtype",
        type=str,
        choices=["bf16", "fp16", "fp8_e5m2", "mxfp4_e2m1"],
        default="mxfp4_e2m1",
    )
    parser.add_argument(    
        "-o",
        action="store_true",
        help="Write performance results to CSV file"
    )
    args = parser.parse_args()
    return args


def main():
    if not (arch_info.is_fp4_avail()):
        print(f"MXFP4 not supported on this architecture")
        sys.exit(0)

    args = parse_args()
    run_benchmark(args)


if __name__ == "__main__":
    sys.exit(main())
