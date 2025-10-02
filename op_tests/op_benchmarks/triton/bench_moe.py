import sys
import argparse
import torch
import triton
from aiter.ops.triton.utils.types import torch_to_triton_dtype, str_to_torch_dtype
from aiter.ops.triton.moe_op import fused_moe as triton_moe
from aiter.ops.triton.moe_op_e2e import e2e_moe as triton_e2e_moe
from aiter.ops.triton.moe_op_silu_fused import fused_moe_silu as triton_moe_silu
from op_tests.triton_tests.test_moe import input_helper, input_helper_int4_w4a16
from op_tests.triton_tests.test_moe import input_helper_e2e
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


def fused_moe(
    M,
    N,
    K,
    top_k,
    E,
    routed_weight=False,
    dtype=torch.bfloat16,
    int4_w4a16=False,
    fp8_w8a8=False,
    int8_w8a16=False,
    block_shape=None,
    has_zp=True,
    silu_fused=False,
    e2e_fused=False
):
    moe_fn = triton_moe_silu if silu_fused else triton_moe
    block_shape_k = 128 if (block_shape == None or not block_shape[1]) else block_shape[1]

    if e2e_fused:
        (
            a,
            w1,
            w2,
            triton_out,
            a_scale,
            w1_scale,
            w2_scale,
            topk_weights,
            topk_ids,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            config,
        ) = input_helper_e2e(
            M,
            N,
            K,
            top_k,
            E,
            routed_weight=routed_weight,
            dtype=dtype,
            fp8_w8a8=fp8_w8a8,
            blockshape=block_shape,
            int8_w8a16=int8_w8a16,
            persistent=False,
        )

        # intermediate is none for persistent mode
        return lambda: triton_e2e_moe(
            a,
            w1,
            w2,
            None,
            triton_out,
            a_scale,
            w1_scale,
            w2_scale,
            topk_weights,
            sorted_token_ids,
            topk_ids,
            expert_ids,
            num_tokens_post_padded,
            routed_weight,
            top_k,
            fp8_w8a8,
            int8_w8a16,
            block_shape=block_shape,
            config=config,
        )

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
            group_size=block_shape_k,
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
            block_shape=block_shape,
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
            blockshape=block_shape,
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
            block_shape=block_shape,
            config=config,
        )


def run_benchmark(args):
    routed_weight = args.routed_weight
    int8_w8a16 = args.int8_w8a16
    fp8_w8a8 = args.fp8_w8a8
    int4_w4a16 = args.int4_w4a16
    block_shape = args.block_shape
    # group_size = args.group_size
    has_zp = args.has_zp
    print_time = args.print_time
    silu_fused = args.silu_fused
    e2e_fused = args.e2e_fused
    dtype = str_to_torch_dtype[args.dtype]
    fp8_type = str_to_torch_dtype[args.fp8_type]

    assert not (e2e_fused and silu_fused)

    if silu_fused:
        args.no_bench_stage2 = True

    if e2e_fused:
        args.no_bench_stage2 = False

    if block_shape:
        block_shape_n, block_shape_k = block_shape[0], block_shape[1]
    else:
        block_shape_n, block_shape_k = None, None

    if int4_w4a16:
        assert block_shape != None, "set group_size with -group_size"

    kernel_name = "_fused_moe_kernel"
    if (int8_w8a16 or int4_w4a16) and (block_shape_k is not None) and block_shape_k > 0:
        kernel_name = "_fused_moe_kernel_gptq_awq"

    # python3 op_tests/test_moe_blockscale.py -d bf16 -m 32 -dim 7168 -idim 512 -e 512 -k 8
    if args.M and args.N and args.K and args.TopK and args.E:
        x_vals_list = [("custom shape", args.M, args.N, args.K, args.E, args.TopK)]
    else:
        x_vals_list = model_benchmark_configs(args)
    x_names = ["model", "M", "N", "K", "E", "top_k"]

    if print_time:
        line_names = ["Time_(ms)"]
        line_vals = ["time"]
    else:
        line_names = ["Time_(ms)", "TFLOPS", "Bandwidth_(GB/s)", "Arithmetic_Intensity_(Flops/Byte)"]
        line_vals = ["time", "tflops", "bandwidth", "ai"]

    benchmark = triton.testing.Benchmark(
        x_names=x_names,
        x_vals=x_vals_list,
        line_arg="metric",
        line_vals=line_vals,
        line_names=line_names,
        styles=[("red", "-"), ("blue", "-"), ("yellow", "-"), ("green", "-")],
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
        elif e2e_fused:
            flops += 2.0 * M * top_k * K * (N // 2)
            flops += M * top_k * N
            # The weight is applied on the gemm product which has the shape of (M, top_k, N)
            if routed_weight:
                flops += M * top_k * (N // 2)
            mem_read_expert2 = (max_expert_loaded * (N // 2) * K) * b_bytes
            mem = mem_read + mem_read_expert2 + (M * top_k * K) * c_bytes
        else:
            mem = mem_read + mem_write

        fn = fused_moe(
            M,
            N,
            K,
            top_k,
            E,
            routed_weight=routed_weight,
            dtype=torch.bfloat16,
            int4_w4a16=int4_w4a16,
            fp8_w8a8=fp8_w8a8,
            int8_w8a16=int8_w8a16,
            block_shape=block_shape,
            has_zp=has_zp,
            silu_fused=silu_fused,
            e2e_fused=e2e_fused
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
        elif metric == "ai":
            return flops / mem
        else:
            raise ValueError("Unknown metric: " + metric)

    bench_moe_gemm.run(save_path="." if args.o else None, print_data=True)


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
    parser.add_argument("-M", type=int, default=0, help="num tokens")
    parser.add_argument("-N", type=int, default=0, help="intermediate dimension")
    parser.add_argument("-K", type=int, default=0, help="hidden dimension (input/output dimension of moe)")
    parser.add_argument("-TopK", type=int, default=0, help="topk experts chosen per token")
    parser.add_argument("-E", type=int, default=0, help="number of experts")

    parser.add_argument('-block_shape', nargs=2, type=int, default=None, help='block shape n and k')

    parser.add_argument("-routed_weight", action="store_true", default=False)
    parser.add_argument("-int8_w8a16", action="store_true", default=False)
    parser.add_argument("-fp8_w8a8", action="store_true", default=False)
    parser.add_argument("-int4_w4a16", action="store_true", default=False)
    parser.add_argument("-has_zp", action="store_true", default=False)
    parser.add_argument("-print_time", action="store_true", default=False)
    parser.add_argument("-e2e_fused", action="store_true", default=False)
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
        "-o", action="store_true", help="Write performance results to CSV file"
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    if args.print_vgpr:
        print("Retrieving VGPR usage for Triton kernels...")

        def fun():
            return run_benchmark(args)

        print_vgpr(fun, get_caller_name_no_ext())
        return 0
    run_benchmark(args)


if __name__ == "__main__":
    sys.exit(main())
