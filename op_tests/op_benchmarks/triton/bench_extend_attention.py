from aiter.ops.triton import extend_attention, prefill_attention

import triton

from op_tests.op_benchmarks.triton.utils.benchmark_utils import (
    get_model_configs,
    get_available_models,
    print_vgpr,
    get_caller_name_no_ext,
)
from op_tests.triton_tests.test_extend_attention import input_helper

import torch
import argparse


def is_hip():
    return triton.runtime.driver.active.get_current_target().backend == "hip"


is_hip_ = is_hip()


def extend_forward(
    q_extend,
    k_extend,
    v_extend,
    k_buffer,
    v_buffer,
    kv_indptr,
    kv_indices,
    qo_indptr,
    custom_mask,
    mask_indptr,
    max_len_extend,
    causal,
    sm_scale=1.0,
    logit_cap=0.0,
):
    out = torch.empty(
        (*q_extend.shape[:-1], v_extend.shape[-1]),
        dtype=q_extend.dtype,
        device=q_extend.device,
    )
    extend_attention.extend_attention_fwd(
        q_extend,
        k_extend,
        v_extend,
        out,
        k_buffer,
        v_buffer,
        qo_indptr,
        kv_indptr,
        kv_indices,
        custom_mask,
        causal,
        mask_indptr,
        max_len_extend,
        sm_scale=sm_scale,
        logit_cap=logit_cap,
    )
    return out


def prefill_forward(
    q_extend, k_extend, v_extend, b_start_loc, b_seq_len, max_len_extend, causal
):
    out = torch.empty(
        (*q_extend.shape[:-1], v_extend.shape[-1]),
        dtype=q_extend.dtype,
        device=q_extend.device,
    )
    prefill_attention.context_attention_fwd(
        q_extend,
        k_extend,
        v_extend,
        out,
        b_start_loc,
        b_seq_len,
        max_len_extend,
        causal=causal,
    )
    return out


def get_extend_benchmark_configs():
    x_names = [
        "B",
        "H",
        "prefix",
        "extend",
        "kv_lora_rank",
        "qk_rope_head_dim",
        "v_head_dim",
        "attn_impl",
    ]
    x_vals_list = [
        (2, 16, 1024, 1024, 256, 0, 128, "non-absorb"),
        (2, 16, 4096, 4096, 512, 64, 128, "non-absorb"),
        (2, 16, 8192, 4096, 512, 64, 128, "non-absorb"),
        (2, 16, 8192, 4096, 512, 64, 128, "absorb"),
        (2, 16, 16324, 8192, 512, 64, 128, "absorb"),
    ]
    return x_names, x_vals_list


def get_prefill_benchmark_configs():
    x_names = [
        "B",
        "H",
        "prefix",
        "extend",
        "kv_lora_rank",
        "qk_rope_head_dim",
        "v_head_dim",
        "attn_impl",
    ]
    x_vals_list = [
        (2, 16, 0, 1024, 256, 0, 128, "non-absorb"),
        (2, 16, 0, 4096, 512, 64, 128, "non-absorb"),
        (2, 16, 0, 4096, 512, 64, 128, "non-absorb"),
        (2, 16, 0, 4096, 512, 64, 128, "absorb"),
        (2, 16, 0, 8192, 512, 64, 128, "absorb"),
    ]
    return x_names, x_vals_list


def model_benchmark_configs(args):
    config_file = args.model_configs
    # Only deepseek models are supported for this benchmark.
    if args.model == "all":
        configs = get_model_configs(config_path=config_file, models="deepseek")
    else:
        assert (
            "deepseek" in args.model
        ), "Only deepseek models are supported for this benchmark."
        configs = get_model_configs(config_path=config_file, models=args.model)

    batch_size = args.b if args.b else 1

    x_names = [
        "model",
        "B",
        "H",
        "prefix",
        "extend",
        "kv_lora_rank",
        "qk_rope_head_dim",
        "v_head_dim",
        "attn_impl",
    ]

    x_vals_list = []

    for model_name, config in configs.items():
        HQ = config["num_attention_heads"] // 8  # tp8 mode
        prefix = args.prefix if args.prefix else 16324
        extend = args.extend if args.extend else 8192
        attn_impl = args.attn_impl if args.attn_impl else "non-absorb"
        x_vals_list.append(
            (model_name, batch_size, HQ, prefix, extend, 512, 64, 128, attn_impl)
        )

    return x_names, x_vals_list


def benchmark(args):
    dtype = arg_to_torch_dtype[args.dtype]
    torch.set_default_dtype(dtype)

    configs = []

    if args.model:
        x_names, x_vals_list = model_benchmark_configs(args)
    else:
        if args.mode == "extend":
            x_names, x_vals_list = get_extend_benchmark_configs()
        elif args.mode == "prefill":
            x_names, x_vals_list = get_prefill_benchmark_configs()

    line_vals = ["fwd_Time_(ms)"]

    configs.append(
        triton.testing.Benchmark(
            x_names=x_names,
            x_vals=x_vals_list,
            line_arg="metric",
            line_vals=line_vals,
            line_names=line_vals,
            styles=[("red", "-"), ("green", "-")],
            ylabel="ms",
            plot_name=get_caller_name_no_ext(),
            args={
                "sm_scale": 1.0,
                "logit_cap": 0.0,
                "device": args.device,
                "provider": "extend_attention_fwd",
            },
        )
    )

    @triton.testing.perf_report(configs)
    def bench_MLA(
        B,
        H,
        prefix,
        extend,
        kv_lora_rank,
        qk_rope_head_dim,
        v_head_dim,
        attn_impl,
        sm_scale,
        logit_cap,
        device,
        provider=None,
        model=None,
        **kwargs,
    ):
        warmup = 25
        rep = 100
        (
            q_extend,
            k_extend,
            v_extend,
            k_buffer,
            v_buffer,
            kv_indptr,
            kv_indices,
            qo_indptr,
            custom_mask,
            mask_indptr,
            max_len_extend,
            B_Start_Loc,
            B_Loc,
            B_Seqlen,
        ) = input_helper(
            B,
            H,
            prefix,
            extend,
            kv_lora_rank,
            qk_rope_head_dim,
            v_head_dim,
            dtype,
            device,
        )

        if provider == "extend_attention_fwd":

            def extend_attention():
                return extend_forward(
                    q_extend,
                    k_extend,
                    v_extend,
                    k_buffer,
                    v_buffer,
                    kv_indptr,
                    kv_indices,
                    qo_indptr,
                    custom_mask,
                    mask_indptr,
                    max_len_extend,
                    args.causal,
                    sm_scale,
                    logit_cap,
                )

            def context_attention():
                return extend_forward(
                    q_extend,
                    k_extend,
                    v_extend,
                    B_Start_Loc,
                    B_Seqlen,
                    max_len_extend,
                    args.causal,
                )

            if provider == "extend_attention_fwd":
                fn = extend_attention
            elif provider == "context_attention_fwd":
                assert (
                    prefix == 0
                ), "Prefix length must be 0 for context attention. Try setting -mode prefill."
                fn = context_attention
            else:
                raise ValueError(f"Unknown provider: {provider}")

        ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
        return ms

    bench_MLA.run(save_path="." if args.o else None, print_data=True, show_plots=False)
    return x_vals_list, x_names, line_vals


arg_to_torch_dtype = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
}


def parse_args():
    parser = argparse.ArgumentParser(
        prog="Benchmark MLA Prefill",
        allow_abbrev=False,
    )
    parser.add_argument(
        "--model_configs",
        type=str,
        default="utils/model_configs.json",
        help="Model config json file.",
    )
    available_models = get_available_models(
        filter="deepseek"
    )  # Dynamically load model names
    model_help = (
        "Model name to benchmark. Select from: ["
        + ", ".join(available_models)
        + "]. Use 'all' to benchmark all models. Provide model family (the part before -) to benchmark all models in that family. One can provide multiple as --model \"llama3,mistral_7B\""
    )
    parser.add_argument("--model", type=str, default="", help=model_help)
    parser.add_argument("-b", type=int, default=0, help="Batch size")
    parser.add_argument("--prefix", type=int, default=0, help="Prefix length")
    parser.add_argument("--extend", type=int, default=0, help="Extend length")
    parser.add_argument(
        "--attn_impl",
        type=str,
        default="non-absorb",
        help="Whether to use absorbed or non-absorbed attention. Options: absorb, non-absorb",
    )
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "-print_vgpr",
        action="store_true",
        default=False,
        help="Prints the VGPR usage of the compiled triton kernel.",
    )
    parser.add_argument("-causal", action="store_true", default=False)
    parser.add_argument(
        "-equal_seqlens",
        action="store_true",
        default=False,
        help="Equal sequence lengths, i.e. total (prefix|extend) tokens = B * (prefix|extend). Otherwise we have randint(1, (prefix|extend), (B,)) as sequence lengths.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="extend",
        help="Mode of the benchmark. Options: extend, prefill",
    )
    parser.add_argument(
        "-o",
        action="store_true",
        default=False,
        help="Write performance results to CSV file",
    )
    return parser.parse_args()


arg_to_torch_dtype = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
}


def run_bench(args):
    torch.manual_seed(0)
    torch.set_default_device(args.device)
    benchmark(args)


def main():
    args = parse_args()
    if args.print_vgpr:  # print the vgpr usage of the kernel
        print_vgpr(lambda: run_bench(args), table_start=get_caller_name_no_ext())
        return 0
    run_bench(args)


if __name__ == "__main__":
    main()
