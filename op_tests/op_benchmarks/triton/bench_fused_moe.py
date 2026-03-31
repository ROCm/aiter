#!/usr/bin/env python3
from __future__ import annotations

"""Benchmark fused MoE SiLU (fused_moe_silu) using model shapes from utils/model_configs.json.

Same workflow as bench_moe.py, but the kernel is always ``fused_moe_silu`` (no non-fused MoE path).

Example:
  python bench_fused_moe.py --model llama3-8B --dtype fp16
  python bench_fused_moe.py --model mixtral-7B -dtype bf16 -M 4096 -o
  python bench_fused_moe.py --model mixtral-7B -dtype fp16 -M 1,4,16,64,256,1024,4096 -o
"""
import argparse
import sys

import torch
import triton
from aiter.ops.triton.utils.types import torch_to_triton_dtype, str_to_torch_dtype
from aiter.ops.triton.moe.moe_op_silu_fused import fused_moe_silu as triton_moe_silu
from op_tests.triton_tests.moe.test_moe import input_helper, input_helper_int4_w4a16
from op_tests.op_benchmarks.triton.utils.benchmark_utils import (
    get_model_configs,
    get_available_models,
    print_vgpr,
    get_caller_name_no_ext,
)

DEFAULT_M_SIZES = (1, 4, 16, 64, 256, 1024, 4096)


def parse_m_sizes(s: str | None) -> list[int]:
    """Parse -M: comma-separated ints; empty/None -> default sweep."""
    if s is None or not str(s).strip():
        return list(DEFAULT_M_SIZES)
    parts = [p.strip() for p in str(s).split(",") if p.strip()]
    if not parts:
        return list(DEFAULT_M_SIZES)
    return [int(x) for x in parts]



def model_benchmark_configs(args):
    """Build (model_name, M, N, K, E, top_k) tuples from model_configs.json (MoE models)."""
    no_bench_stage2 = args.no_bench_stage2
    config_file = args.model_configs
    configs = get_model_configs(
        config_path=config_file, models="mixtral" if args.model is None else args.model
    )
    moe_configs = []
    m_list = getattr(args, "M_list", None) or list(DEFAULT_M_SIZES)

    for M in m_list:
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


def fused_moe_silu_bench_fn(
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
    """Return a zero-arg callable that runs fused_moe_silu (same tensor setup as bench_moe silu_fused)."""
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

        return lambda: triton_moe_silu(
            a,
            b,
            triton_out_silu,
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

    return lambda: triton_moe_silu(
        a,
        b,
        triton_out_silu,
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
    dtype = str_to_torch_dtype[args.dtype]
    fp8_type = str_to_torch_dtype[args.fp8_type]

    # Match bench_moe.py behavior when -silu_fused: include stage-2 shape in the sweep.
    args.no_bench_stage2 = True

    if int4_w4a16:
        assert group_size is not None, "set group_size with -group_size"

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
    def bench_fused_moe_gemm(M, N, K, E, top_k, metric, model=None):
        flops = 2.0 * M * top_k * K * N
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

        max_expert_loaded = min(E, top_k * M)
        mem_read = (M * K) * a_bytes + (max_expert_loaded * N * K) * b_bytes
        mem_write = (M * top_k * N) * c_bytes
        mem = mem_read + (mem_write // 2)
        flops += M * top_k * N

        fn = fused_moe_silu_bench_fn(
            M,
            N,
            K,
            top_k,
            E,
            routed_weight=routed_weight,
            dtype=dtype,
            int4_w4a16=int4_w4a16,
            fp8_w8a8=fp8_w8a8,
            int8_w8a16=int8_w8a16,
            group_size=group_size,
            has_zp=has_zp,
        )

        ms = triton.testing.do_bench(fn, warmup=25, rep=100)

        bandwidth = mem / (ms * 1e-3) * 1e-9
        tflops = flops / ms * 1e-9

        if metric == "time":
            return ms
        if metric == "tflops":
            return tflops
        if metric == "bandwidth":
            return bandwidth
        raise ValueError("Unknown metric: " + metric)

    bench_fused_moe_gemm.run(save_path="." if args.o else None, print_data=True)


def parse_args():
    parser = argparse.ArgumentParser(
        prog="Benchmark fused MoE SiLU (fused_moe_silu)",
        allow_abbrev=False,
    )
    parser.add_argument(
        "-model_configs",
        "--model-configs",
        type=str,
        default="utils/model_configs.json",
        dest="model_configs",
        help="Model config json file.",
    )
    available_models = get_available_models()
    model_help = (
        "Model name to benchmark. Select from: ["
        + ", ".join(available_models)
        + "]. Use 'all' to benchmark all models or leave blank for the default (mixtral)."
    )
    parser.add_argument(
        "--model",
        "-model",
        type=str,
        default=None,
        help=model_help,
    )
    parser.add_argument(
        "-M",
        type=str,
        default=",".join(str(x) for x in DEFAULT_M_SIZES),
        help=(
            "Token batch size(s), comma-separated. Default: "
            + ",".join(str(x) for x in DEFAULT_M_SIZES)
            + ". Example: -M 4096 or -M 1,64,4096"
        ),
    )
    parser.add_argument("-group_size", type=int, default=None, help="group_size for int4")
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
    parser.add_argument(
        "--dtype",
        "-dtype",
        default="fp16",
        help="Torch dtype key for str_to_torch_dtype (e.g. fp16, bf16, float16).",
    )
    parser.add_argument("-fp8_type", default="e5m2fnuz")
    parser.add_argument(
        "-o",
        action="store_true",
        help="Write performance results to CSV file",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    args.M_list = parse_m_sizes(args.M)

    if args.print_vgpr:
        print("Retrieving VGPR usage for Triton kernels...")

        def fun():
            return run_benchmark(args)

        print_vgpr(fun, get_caller_name_no_ext())
        return 0
    run_benchmark(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
