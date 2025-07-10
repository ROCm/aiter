from aiter.ops.triton import extend_attention, prefill_attention

import triton

from utils.benchmark_utils import get_model_configs, get_available_models, print_vgpr
import torch
import argparse


def is_hip():
    return triton.runtime.driver.active.get_current_target().backend == "hip"


is_hip_ = is_hip()


def input_helper(
    B,
    H,
    prefix_length,
    extend_length,
    kv_lora_rank,
    qk_rope_head_dim,
    v_head_dim,
    dtype,
    device,
    attn_impl="absorb",
    equal_seqlens=False,
    requires_grad=False,
):
    torch.manual_seed(0)

    if not equal_seqlens:
        max_extend_length = extend_length
        max_prefix_length = prefix_length

        seqlens_extend = torch.randint(
            1, max_extend_length + 1, (B,), dtype=torch.int32
        )
        if prefix_length == 0:
            seqlens_prefix = torch.full((B,), prefix_length)
        else:
            seqlens_prefix = torch.randint(
                1, max_prefix_length + 1, (B,), dtype=torch.int32
            )

    else:
        seqlens_extend = torch.full((B,), extend_length)
        seqlens_prefix = torch.full((B,), prefix_length)

    B_Seqlen = seqlens_extend + seqlens_prefix
    B_Seqlen = B_Seqlen.to(device="cuda")

    cu_seqlens_extend = torch.cat(
        [
            torch.tensor([0], dtype=torch.int32),
            seqlens_extend.cumsum(dim=0, dtype=torch.int32),
        ]
    )
    cu_seqlens_prefix = torch.cat(
        [
            torch.tensor([0], dtype=torch.int32),
            seqlens_prefix.cumsum(dim=0, dtype=torch.int32),
        ]
    )

    cu_seqlens_extend = cu_seqlens_extend.to(device="cuda")
    cu_seqlens_prefix = cu_seqlens_prefix.to(device="cuda")

    B_Start_Loc = cu_seqlens_extend

    total_extend = cu_seqlens_extend[-1].item()
    total_prefix = cu_seqlens_prefix[-1].item()

    if attn_impl == "absorb":
        Lq = kv_lora_rank + qk_rope_head_dim
        Lk = kv_lora_rank + qk_rope_head_dim
        Lv = kv_lora_rank
    else:
        Lq = v_head_dim + qk_rope_head_dim
        Lk = v_head_dim + qk_rope_head_dim
        Lv = v_head_dim

    q_extend = torch.randn(
        total_extend, H, Lq, dtype=dtype, device=device
    ).requires_grad_(requires_grad)

    # extend parts
    k_extend = torch.randn(
        total_extend, 1, Lk, dtype=dtype, device=device
    ).requires_grad_(requires_grad)
    v_extend = k_extend[..., :Lv]

    # extend indexing
    qo_indptr = cu_seqlens_extend  # torch.arange(B + 1, device=device) * (extend_length) # 0, extend_length, extend_length*2

    # prefix parts
    k_buffer = torch.randn(
        total_prefix, 1, Lk, dtype=dtype, device=device
    ).requires_grad_(requires_grad)
    v_buffer = k_buffer[..., :Lv]

    if attn_impl != "absorb":
        # simulate v = kv_latent * w_vc which changes the values compared to k
        v_extend = torch.randn_like(v_extend)
        v_buffer = torch.randn_like(v_buffer)

    # prefix indexing
    kv_indptr = cu_seqlens_prefix  # torch.arange(B + 1, device=device) * prefix_length # 0, prefix_length, prefix_length*2
    kv_indices = torch.arange(total_prefix, device=device)

    max_prefix = seqlens_prefix.max().item()
    B_Loc = torch.full((B, max_prefix), -1, dtype=torch.int32, device=device)
    for b in range(B):
        start = cu_seqlens_prefix[b].item()
        end = cu_seqlens_prefix[b + 1].item()
        B_Loc[b, : seqlens_prefix[b]] = torch.arange(start, end, device=device)
    B_Loc = B_Loc.unsqueeze(-1)  # [B, max_prefix, 1]

    custom_mask = None
    mask_indptr = None
    max_len_extend = extend_length

    return (
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
    )


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

    line_vals = ["extend_attention_fwd"]

    plot_name = (
        args.plot_name + f"-causal-{args.causal}-equal_seqlens-{args.equal_seqlens}"
    )

    configs.append(
        triton.testing.Benchmark(
            x_names=x_names,
            x_vals=x_vals_list,
            line_arg="provider",
            line_vals=line_vals,
            line_names=line_vals,
            styles=[("red", "-"), ("green", "-")],
            ylabel="ms",
            plot_name=plot_name,
            args={"sm_scale": 1.0, "logit_cap": 0.0, "device": args.device},
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

    bench_MLA.run(save_path=None, print_data=True, show_plots=False)
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
        "-model_configs",
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
        + "]. Use 'all' to benchmark all models. Provide model family (the part before -) to benchmark all models in that family. One can provide multiple as -model \"llama3,mistral_7B\""
    )
    parser.add_argument(
        "-plot_name",
        type=str,
        default="MLA-prefill",
        help="Name for the results plot|table",
    )
    parser.add_argument("-model", type=str, default="", help=model_help)
    parser.add_argument("-b", type=int, default=0, help="Batch size")
    parser.add_argument("-prefix", type=int, default=0, help="Prefix length")
    parser.add_argument("-extend", type=int, default=0, help="Extend length")
    parser.add_argument(
        "-attn_impl",
        type=str,
        default="non-absorb",
        help="Whether to use absorbed or non-absorbed attention. Options: absorb, non-absorb",
    )
    parser.add_argument("-dtype", default="bf16")
    parser.add_argument("-device", default="cuda")
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
        "-mode",
        type=str,
        default="extend",
        help="Mode of the benchmark. Options: extend, prefill",
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
        print_vgpr(lambda: run_bench(args), table_start=args.plot_name)
        return 0
    run_bench(args)


if __name__ == "__main__":
    main()
