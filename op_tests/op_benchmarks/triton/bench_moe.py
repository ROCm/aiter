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
    M = args.M if args.M else 32  # check size
    # M, K, N, E, top_k
    for model_name, config in configs.items():
        if not all(
            key in config
            for key in ["moe_intermediate_size", "hidden_size", "num_expert", "top_k"]
        ):
            print(
                f"Missing MoE keys ('moe_intermediate_size', 'hidden_size', 'num_expert', 'top_k') in model configuration for {model_name}: skipping this model."
            )
        else:
            N1 = config["moe_intermediate_size"] # // args.tp
            K1 = config["hidden_size"]
            if not no_bench_stage2:
                N2 = config["hidden_size"]
                K2 = config["moe_intermediate_size"] // 2 # // args.tp

            E = config["num_expert"]
            top_k = config["top_k"]

            moe_configs.append((model_name, M, N1, K1, E, top_k, "both" if args.e2e_fused else "stage1"))
            if not no_bench_stage2:
                moe_configs.append((model_name, M, N2, K2, E, top_k, "stage2"))

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
    e2e_fused=False,
    tp=1,
    stage="stage1",
):
    moe_fn = triton_moe_silu if silu_fused else triton_moe
    block_shape_k = (
        128 if (block_shape is None or not block_shape[1]) else block_shape[1]
    )

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
            dtype=dtype,
            fp8_w8a8=fp8_w8a8,
            blockshape=block_shape,
            tp=tp,
        )
        # tensor parallel slicing
        if tp > 1:
            w1 = w1[:, :N // tp, :].contiguous()
            w2 = w2[:, :, :(N//2//tp)].contiguous()
            if fp8_w8a8:
                num_w1_scales_per_gpu = triton.cdiv(N // tp, block_shape[0])
                w1_scale = w1_scale[:, :num_w1_scales_per_gpu].contiguous()
                num_w2_scales_per_gpu = triton.cdiv((N // 2) // tp, block_shape[0])
                w2_scale = w2_scale[:, :, :num_w2_scales_per_gpu].contiguous()

        fn = lambda: triton_e2e_moe(
            a,
            w1,
            w2,
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
            block_shape=block_shape,
            config=config,
        )

    elif int4_w4a16:
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

        fn = lambda: moe_fn(  # noqa: E731
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

        # tensor parallel slicing
        if tp > 1:
            if stage == "stage1":
                b = b[:, :N // tp, :].contiguous()
                if fp8_w8a8:
                    num_b_scales_per_gpu = triton.cdiv(N // tp, block_shape[0])
                    b_scale = b_scale[:, :num_b_scales_per_gpu].contiguous()
            else:
                b = b[:, :, :(K//tp)].contiguous()
                if fp8_w8a8:
                    num_b_scales_per_gpu = triton.cdiv(K // tp, block_shape[1])
                    b_scale = b_scale[:, :, :num_b_scales_per_gpu].contiguous()

        fn = lambda: moe_fn(
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

    return fn, len(
        torch.unique(expert_ids)
    )  # second value for memory bandwidth calculation


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

    if silu_fused or e2e_fused:
        args.no_bench_stage2 = True


    if int4_w4a16:
        assert block_shape != None, "set group_size with -group_size"

    # python3 op_tests/test_moe_blockscale.py -d bf16 -m 32 -dim 7168 -idim 512 -e 512 -k 8
    if args.model is not None:
        assert (args.N and args.K and args.E and args.TopK) == 0, (
            "When -model is set, do not set -N, -K, -E or -TopK, as they are model specific."
        )
    
    if args.N and args.K and args.TopK and args.E:
        x_vals_list = [("custom shape", args.M if args.M else 32, args.N, args.K, args.E, args.TopK, "custom")]
    else:
        x_vals_list = model_benchmark_configs(args)
    x_names = ["model", "M", "N", "K", "E", "top_k", "stage"]

    if print_time:
        line_names = ["Time_(ms)"]
        line_vals = ["time"]
    else:
        line_names = [
            "Time_(ms)",
            "TFLOPS",
            "Bandwidth_(GB/s)",
            "Arithmetic_Intensity_(Flops/Byte)",
        ]
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
    def bench_moe_gemm(M, N, K, E, stage, top_k, metric, model=None):
        fn, num_expert_loaded = fused_moe(
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
            e2e_fused=e2e_fused,
            tp=args.tp,
            stage=stage,
        )
        # num_expert_loaded: len(torch.unique(expert_ids))
        # max_expert_loaded = min(E, top_k * M)
        # print("num_expert_loaded:", num_expert_loaded, "max_expert_loaded:", max_expert_loaded)

        if args.tp > 1: # take tensor parallelism into account when calculating performance metrics
            if stage == "stage1":
                N = N // args.tp
            else:
                K = K // args.tp
        
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

        mem_read = (M * K) * a_bytes + (num_expert_loaded * N * K) * b_bytes

        mem_write = (M * top_k * N) * c_bytes
        if silu_fused:
            mem = mem_read + (mem_write // 2)
            flops += M * top_k * N
        elif e2e_fused:
            # second gemm
            flops += 2.0 * M * top_k * K * (N // 2)
            # silu
            flops += M * top_k * N
            # The weight is applied on the gemm product which has the shape of (M, top_k, N)
            if routed_weight:
                flops += M * top_k * (N // 2)
            mem_read_expert2 = (num_expert_loaded * (N // 2) * K) * b_bytes
            mem = mem_read + mem_read_expert2 + (M * top_k * K) * c_bytes
        else:
            mem = mem_read + mem_write

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
    parser.add_argument( "-tp", type=int, default=1, help="tensor paraller degree")
    parser.add_argument(
        "-K",
        type=int,
        default=0,
        help="hidden dimension (input/output dimension of moe)",
    )
    parser.add_argument(
        "-TopK", type=int, default=0, help="topk experts chosen per token"
    )
    parser.add_argument("-E", type=int, default=0, help="number of experts")

    parser.add_argument(
        "-block_shape", nargs=2, type=int, default=None, help="block shape n and k"
    )

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
    parser.add_argument("-no_bench_stage2", action="store_true", default=False)
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
