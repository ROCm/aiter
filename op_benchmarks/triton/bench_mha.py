import triton
import triton.language as tl
from utils.benchmark_utils import get_model_configs, get_available_models, print_vgpr
import torch

import os
import sys
import warnings
import argparse
from aiter.ops.triton.mha import (
    flash_attn_varlen_func,
    flash_attn_func,
    flash_attn_varlen_fp8_func,
    flash_attn_fp8_func,
)

# Add two parent directories to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(parent_dir)


def mha_varlen_input_helper(
    Z, HQ, HK, N_CTX_Q, N_CTX_K, D_HEAD, dtype, equal_seqlens=False, requires_grad=True
):
    torch.manual_seed(20)

    # Random sequence lengths. Using N_CTX * Z as kind of maximum possible sum of individual seqs
    if not equal_seqlens:
        max_seqlens_q = N_CTX_Q
        max_seqlens_k = N_CTX_K
        if N_CTX_Q == N_CTX_K:
            seqlens_q = torch.randint(1, max_seqlens_q + 1, (Z,), dtype=torch.int32)
            seqlens_k = seqlens_q
        else:
            seqlens_q = torch.randint(1, max_seqlens_q + 1, (Z,), dtype=torch.int32)
            seqlens_k = torch.randint(1, max_seqlens_k + 1, (Z,), dtype=torch.int32)
    else:
        seqlens_q = torch.full((Z,), N_CTX_Q)
        seqlens_k = torch.full((Z,), N_CTX_K)

    # Calculate cumulative sequence lengths
    cu_seqlens_q = torch.cat(
        [
            torch.tensor([0], dtype=torch.int32),
            seqlens_q.cumsum(dim=0, dtype=torch.int32),
        ]
    )
    cu_seqlens_k = torch.cat(
        [
            torch.tensor([0], dtype=torch.int32),
            seqlens_k.cumsum(dim=0, dtype=torch.int32),
        ]
    )

    cu_seqlens_q = cu_seqlens_q.to(device="cuda")
    cu_seqlens_k = cu_seqlens_k.to(device="cuda")
    # Initialize q, k, v with variable lengths
    total_q = cu_seqlens_q[-1].item()
    total_k = cu_seqlens_k[-1].item()

    q = (
        torch.randn((total_q, HQ, D_HEAD), dtype=dtype, device="cuda")
        .normal_(mean=0.0, std=0.5)
        .requires_grad_(requires_grad)
    )
    k = (
        torch.randn((total_k, HK, D_HEAD), dtype=dtype, device="cuda")
        .normal_(mean=0.0, std=0.5)
        .requires_grad_(requires_grad)
    )
    v = (
        torch.randn((total_k, HK, D_HEAD), dtype=dtype, device="cuda")
        .normal_(mean=0.0, std=0.5)
        .requires_grad_(requires_grad)
    )
    sm_scale = D_HEAD**-0.5
    return q, k, v, cu_seqlens_q, cu_seqlens_k, sm_scale


def mha_input_helper(Z, HQ, HK, N_CTX_Q, N_CTX_K, D_HEAD, dtype, requires_grad=True):
    torch.manual_seed(20)

    # Initialize q, k, v
    # bshd layout supported
    q_tensor_shape = (Z, N_CTX_Q, HQ, D_HEAD)
    k_tensor_shape = (Z, N_CTX_K, HK, D_HEAD)

    q = torch.randn(
        q_tensor_shape, dtype=dtype, device="cuda", requires_grad=requires_grad
    )
    k = torch.randn(
        k_tensor_shape, dtype=dtype, device="cuda", requires_grad=requires_grad
    )
    v = torch.randn(
        k_tensor_shape, dtype=dtype, device="cuda", requires_grad=requires_grad
    )

    sm_scale = D_HEAD**-0.5
    # max_seqlens_q = N_CTX_Q
    # max_seqlens_k = N_CTX_K

    return q, k, v, sm_scale


def nonvarlen_benchmark_configs():
    configs = [
        (16, 16, 16, 1024, 1024),
        (8, 16, 16, 2048, 2048),
        (4, 16, 16, 4096, 4096),
        (2, 16, 16, 8192, 8192),
        (1, 16, 16, 16384, 16384),
        (2, 48, 48, 1024, 1024),
        (2, 48, 48, 2048, 1024),
        (2, 48, 48, 4096, 8192),
        (2, 48, 48, 8192, 4096),
        (2, 48, 48, 16384, 8192),
        (8, 16, 16, 1989, 15344),
        (4, 16, 16, 4097, 163),
        (2, 16, 16, 8122, 2159),
        (1, 16, 16, 16281, 7),
        (2, 48, 48, 1021, 1020),
        (2, 48, 48, 2001, 2048),
        (2, 48, 48, 3996, 9639),
        (2, 48, 48, 8181, 1021),
    ]
    return configs


def varlen_benchmark_configs():
    configs = [
        (2, 16, 4, 1024, 1024),
        (8, 16, 2, 2048, 2048),
        (4, 16, 8, 4096, 4096),
        (2, 16, 4, 8192, 8192),
        (2, 16, 8, 16384, 16384),
        (2, 48, 12, 1024, 1024),
        (2, 48, 24, 2048, 2048),
        (2, 48, 8, 4096, 4096),
        (2, 48, 4, 8192, 8192),
        (2, 48, 2, 16384, 16384),
        (2, 64, 32, 1024, 1024),
        (4, 64, 16, 2048, 2048),
        (4, 64, 8, 4096, 4096),
        (4, 64, 32, 8192, 8192),
        (4, 128, 16, 16384, 16384),
    ]
    return configs


def model_benchmark_configs(args):
    config_file = args.model_configs
    configs = get_model_configs(config_path=config_file, models=args.model)
    fa_configs = []
    batch_size = args.b if args.b else 1

    for model_name, config in configs.items():
        HQ = config["num_attention_heads"]
        HK = (
            HQ
            if config["num_key_value_heads"] is None
            else config["num_key_value_heads"]
        )
        N_CTX_Q = args.sq if args.sq else 8192
        N_CTX_K = args.sk if args.sk else N_CTX_Q
        HEAD_DIM = config["hidden_size"] // HQ
        fa_configs.append((model_name, batch_size, HQ, HK, N_CTX_Q, N_CTX_K, HEAD_DIM))

    return fa_configs


def test_correctness(custom, args):
    dtype = arg_to_torch_dtype[args.dtype]
    hk = args.hq if not args.hk else args.hk
    sk = args.sq if not args.sk else args.sk
    head_size = 128 if not args.d else args.d
    mode = "fwd"
    x_names = ["BATCH", "HQ", "HK", "N_CTX_Q", "N_CTX_K"]
    causal = args.causal
    varlen = args.layout == "thd"

    if custom:
        x_vals_list = [(args.b, args.hq, hk, args.sq, sk)]
    else:
        if varlen:
            x_vals_list = varlen_benchmark_configs()
        else:
            x_vals_list = nonvarlen_benchmark_configs()

        if args.model:
            x_vals_list = model_benchmark_configs(args)
            x_names = ["model", "BATCH", "HQ", "HK", "N_CTX_Q", "N_CTX_K", "D_HEAD"]

    def bench_flash_attention(
        BATCH,
        HQ,
        HK,
        N_CTX_Q,
        N_CTX_K,
        D_HEAD,
        dtype,
        causal,
        mode,
        provider,
        device="cuda",
        model=None,
    ):
        assert mode in ["fwd", "bwd"]
        requires_grad = False
        # Bwd pass only supports causal=True right now
        if mode == "bwd":
            causal = True
            requires_grad = True

        if varlen:
            q, k, v, cu_seqlens_q, cu_seqlens_k, sm_scale = mha_varlen_input_helper(
                BATCH,
                HQ,
                HK,
                N_CTX_Q,
                N_CTX_K,
                D_HEAD,
                dtype,
                equal_seqlens=args.equal_seqlens,
                requires_grad=requires_grad,
            )
        else:
            q, k, v, sm_scale = mha_input_helper(
                BATCH,
                HQ,
                HK,
                N_CTX_Q,
                N_CTX_K,
                D_HEAD,
                dtype,
                requires_grad=requires_grad,
            )

        if "Torch" in provider:
            assert (
                not varlen or args.equal_seqlens
            ), "Torch sdpa does not support variable sequence lengths."
            q = q.view(BATCH, N_CTX_Q, HQ, D_HEAD).transpose(1, 2)
            k = k.view(BATCH, N_CTX_K, HK, D_HEAD).transpose(1, 2)
            v = v.view(BATCH, N_CTX_K, HK, D_HEAD).transpose(1, 2)
            if (
                HQ != HK
            ):  # TODO: sdpa(..., enable_gqa=True) works but gives very bad perf
                k = k.repeat_interleave(q.size(-3) // k.size(-3), -3)
                v = v.repeat_interleave(q.size(-3) // v.size(-3), -3)
            fn = lambda: torch.nn.functional.scaled_dot_product_attention(  # noqa: E731
                q, k, v, attn_mask=None, dropout_p=0.0, is_causal=causal, scale=sm_scale
            )
        else:
            o = torch.empty_like(q)
            if varlen:
                if args.fp8:
                    fn = lambda: flash_attn_varlen_fp8_func(  # noqa: E731
                        q,
                        k,
                        v,
                        cu_seqlens_q,
                        cu_seqlens_k,
                        N_CTX_Q,
                        N_CTX_K,
                        dropout_p=0.0,
                        softmax_scale=sm_scale,
                        causal=causal,
                    )
                else:
                    fn = lambda: flash_attn_varlen_func(  # noqa: E731
                        q,
                        k,
                        v,
                        cu_seqlens_q,
                        cu_seqlens_k,
                        N_CTX_Q,
                        N_CTX_K,
                        dropout_p=0.0,
                        softmax_scale=sm_scale,
                        causal=causal,
                    )
            else:
                if args.fp8:
                    fn = lambda: flash_attn_fp8_func(  # noqa: E731
                        q, k, v, dropout_p=0.0, softmax_scale=sm_scale, causal=causal
                    )
                else:
                    fn = lambda: flash_attn_func(  # noqa: E731
                        q, k, v, dropout_p=0.0, softmax_scale=sm_scale, causal=causal
                    )

            if mode == "bwd":
                o, _ = fn()
                do = torch.randn_like(o)
                fn = lambda: o.backward(do, retain_graph=True)  # noqa: E731

        return fn()

    # Test correctness of the triton kernel by comparing the output to the torch sdpa output
    for config in x_vals_list:
        # Build a dictionary from x_names and config values, and add D_HEAD
        cfg = {name: value for name, value in zip(x_names, config)}
        cfg["D_HEAD"] = head_size  # head size computed above

        # Run benchmark with Triton provider
        triton_result = bench_flash_attention(
            **cfg, dtype=dtype, causal=causal, mode=mode, provider="Triton"
        )
        triton_result = triton_result[0]

        # Run benchmark with Torch provider
        torch_result = bench_flash_attention(
            **cfg, dtype=dtype, causal=causal, mode=mode, provider="Torch"
        )

        torch_result = torch_result.transpose(1, 2)
        if varlen:
            torch_result = torch_result.flatten(
                0, 1
            )  # Triton kernel flattens batch and sequence length dims

        # Check that the results are close
        torch.testing.assert_close(triton_result, torch_result, rtol=2e-2, atol=2e-2)
        print(f"Results are close for config: {cfg} for triton kernel and torch.sdpa!")


def run_benchmark(custom, args):
    dtype = arg_to_torch_dtype[args.dtype]
    hk = args.hq if not args.hk else args.hk
    sk = args.sq if not args.sk else args.sk
    head_size = 128 if not args.d else args.d
    mode = "fwd"
    x_names = ["BATCH", "HQ", "HK", "N_CTX_Q", "N_CTX_K"]
    causal = args.causal
    varlen = args.layout == "thd"

    configs = []
    plot_name = f"fused-attention-{mode}-D_HEAD-{head_size}-layout-{args.layout}-fp8-{args.fp8}-causal-{causal}"
    extra_args = {"D_HEAD": head_size, "dtype": dtype, "causal": causal, "mode": mode}
    if custom:
        x_vals_list = [(args.b, args.hq, hk, args.sq, sk)]
    else:
        if varlen:
            x_vals_list = varlen_benchmark_configs()
        else:
            x_vals_list = nonvarlen_benchmark_configs()

        if args.model:
            x_vals_list = model_benchmark_configs(args)
            x_names = ["model", "BATCH", "HQ", "HK", "N_CTX_Q", "N_CTX_K", "D_HEAD"]
            plot_name = f"fused-attention-{mode}-layout-{args.layout}-fp8-{args.fp8}-causal-{causal}"
            extra_args = {"dtype": dtype, "causal": causal, "mode": mode}

    unit = "TFLOPS"
    if args.return_time:
        unit = "ms"
    if args.return_bandwidth:
        unit = "GB/s"

    if args.bench_torch:
        if args.return_all:
            line_vals = [
                f"{provider} ({unit})"
                for provider in ["Triton", "Torch"]
                for unit in ["ms", "TFLOPS", "TB/s"]
            ]
        else:
            line_vals = [f"Triton ({unit})", f"Torch ({unit})"]
    else:
        line_vals = [unit]
        if args.return_all:
            line_vals = ["ms", "TFLOPS", "GB/s"]

    configs.append(
        triton.testing.Benchmark(
            x_names=x_names,
            x_vals=x_vals_list,
            line_arg="provider",
            line_vals=line_vals,
            line_names=line_vals,
            styles=[("red", "-"), ("green", "-"), ("blue", "-")] * 2,
            ylabel=unit,
            plot_name=plot_name,
            args=extra_args,
        )
    )

    @triton.testing.perf_report(configs)
    def bench_flash_attention(
        BATCH,
        HQ,
        HK,
        N_CTX_Q,
        N_CTX_K,
        D_HEAD,
        dtype,
        causal,
        mode,
        provider,
        device="cuda",
        model=None,
    ):
        assert mode in ["fwd", "bwd"]

        warmup = 25
        rep = 100

        requires_grad = False
        # Bwd pass only supports causal=True right now
        if mode == "bwd":
            causal = True
            requires_grad = True

        flops_per_matmul = 0
        if varlen:
            q, k, v, cu_seqlens_q, cu_seqlens_k, sm_scale = mha_varlen_input_helper(
                BATCH,
                HQ,
                HK,
                N_CTX_Q,
                N_CTX_K,
                D_HEAD,
                dtype,
                equal_seqlens=args.equal_seqlens,
                requires_grad=requires_grad,
            )
            num_contexts = len(cu_seqlens_q) - 1
            for i in range(0, num_contexts):
                seqlen_q = (cu_seqlens_q[i + 1] - cu_seqlens_q[i]).item()
                seqlen_k = (cu_seqlens_k[i + 1] - cu_seqlens_k[i]).item()
                # x2 in both cases for 2 GEMMs
                if causal:
                    valid_out_elements = (
                        ((seqlen_k**2 + seqlen_k) / 2)
                        if seqlen_q > seqlen_k
                        else (seqlen_q * seqlen_k - ((seqlen_q**2 - seqlen_q) / 2))
                    )
                    flops_per_matmul += valid_out_elements * HQ * D_HEAD * 2
                else:
                    flops_per_matmul += seqlen_q * seqlen_k * HQ * D_HEAD * 2
        else:
            q, k, v, sm_scale = mha_input_helper(
                BATCH,
                HQ,
                HK,
                N_CTX_Q,
                N_CTX_K,
                D_HEAD,
                dtype,
                requires_grad=requires_grad,
            )
            if causal:
                # Same calculation as if varlen/if causal above
                valid_out_elements = (
                    ((N_CTX_K**2 + N_CTX_K) / 2)
                    if N_CTX_Q > N_CTX_K
                    else (N_CTX_Q * N_CTX_K - ((N_CTX_Q**2 - N_CTX_Q) / 2))
                )
                flops_per_matmul = 2.0 * BATCH * HQ * valid_out_elements * D_HEAD
            else:
                flops_per_matmul = 2.0 * BATCH * HQ * N_CTX_Q * N_CTX_K * D_HEAD

        if "Torch" in provider:
            assert (
                not varlen or args.equal_seqlens
            ), "Torch sdpa does not support variable sequence lengths. Hint: if you are using -layout thd, set -equal_seqlens aswell."
            # torch.sdpa assumes bhsd layout
            q = q.view(BATCH, N_CTX_Q, HQ, D_HEAD).transpose(1, 2)
            k = k.view(BATCH, N_CTX_K, HK, D_HEAD).transpose(1, 2)
            v = v.view(BATCH, N_CTX_K, HK, D_HEAD).transpose(1, 2)
            if (
                HQ != HK
            ):  # TODO: sdpa(..., enable_gqa=True) works but gives very bad perf
                k = k.repeat_interleave(q.size(-3) // k.size(-3), -3)
                v = v.repeat_interleave(q.size(-3) // v.size(-3), -3)
            fn = lambda: torch.nn.functional.scaled_dot_product_attention(  # noqa: E731
                q, k, v, attn_mask=None, dropout_p=0.0, is_causal=causal, scale=sm_scale
            )
        else:
            o = torch.empty_like(q)
            if varlen:
                if args.fp8:
                    fn = lambda: flash_attn_varlen_fp8_func(  # noqa: E731
                        q,
                        k,
                        v,
                        cu_seqlens_q,
                        cu_seqlens_k,
                        N_CTX_Q,
                        N_CTX_K,
                        dropout_p=0.0,
                        softmax_scale=sm_scale,
                        causal=causal,
                    )
                else:
                    fn = lambda: flash_attn_varlen_func(  # noqa: E731
                        q,
                        k,
                        v,
                        cu_seqlens_q,
                        cu_seqlens_k,
                        N_CTX_Q,
                        N_CTX_K,
                        dropout_p=0.0,
                        softmax_scale=sm_scale,
                        causal=causal,
                    )
            else:
                if args.fp8:
                    fn = lambda: flash_attn_fp8_func(  # noqa: E731
                        q, k, v, dropout_p=0.0, softmax_scale=sm_scale, causal=causal
                    )
                else:
                    fn = lambda: flash_attn_func(  # noqa: E731
                        q, k, v, dropout_p=0.0, softmax_scale=sm_scale, causal=causal
                    )

            if mode == "bwd":
                o, _ = fn()
                do = torch.randn_like(o)
                fn = lambda: o.backward(do, retain_graph=True)  # noqa: E731

        ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
        total_flops = 2 * flops_per_matmul
        if mode == "bwd":
            total_flops *= 2.5  # 2.0(bwd) + 0.5(recompute)

        input_bytes = (
            1 if args.fp8 else q.element_size()
        )  # size of element in q,k,v in bytes
        output_bytes = q.element_size()
        if varlen:
            total_num_tokens_q = cu_seqlens_q[-1].item()
            total_num_tokens_k = cu_seqlens_k[-1].item()
            mem = (
                total_num_tokens_q * HQ * D_HEAD * input_bytes
                + 2 * total_num_tokens_k * HK * D_HEAD * input_bytes
                + total_num_tokens_q * HQ * D_HEAD * output_bytes
            )
        else:
            total_num_tokens_q = BATCH * N_CTX_Q
            total_num_tokens_k = BATCH * N_CTX_K
            mem = (
                total_num_tokens_q * HQ * D_HEAD * input_bytes
                + 2 * total_num_tokens_k * HK * D_HEAD * input_bytes
                + total_num_tokens_q * HQ * D_HEAD * output_bytes
            )

        if "ms" in provider:
            return ms
        elif "TFLOPS" in provider:
            return total_flops / ms * 1e-9
        else:  # bandwidth GB/s
            return mem / ms * 1e-3

    bench_flash_attention.run(save_path=".", print_data=True, show_plots=True)


def supported_layouts():
    layouts = (
        "bshd: Q, K, V are individual tensors of [batch, seqlen_q/k, num_heads, head_size]. "
        "thd: Q, K, V are individual tensors of [total_q/k, num_heads, head_size]. "
    )
    return layouts


# argparse lacks support for boolean argument type (sigh...)
def str2bool(v):
    if isinstance(v, bool) or v is None:
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def parse_args():

    parser = argparse.ArgumentParser(
        prog="Benchmark FlashAttention",
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
        + "]. Use 'all' to benchmark all models. Provide model family (the part before -) to benchmark all models in that family. One can provide multiple as -model \"llama3,mistral_7B\""
    )
    parser.add_argument("-model", type=str, default="", help=model_help)
    parser.add_argument("-b", type=int, default=0)
    parser.add_argument("-hq", type=int, default=0)
    parser.add_argument("-hk", type=int, default=0)
    parser.add_argument("-sq", type=int, default=0)
    parser.add_argument("-sk", type=int, default=0)
    parser.add_argument(
        "-equal_seqlens",
        action="store_true",
        default=False,
        help="If specified, uses equal sequence lengths with thd layout, i.e t = b * sq",
    )
    parser.add_argument("-d", type=int, default=0)
    parser.add_argument("-causal", type=str2bool, default=None)
    parser.add_argument("-fp8", action="store_true", default=False)
    parser.add_argument("-quantize_p", action="store_true", default=False)
    parser.add_argument("-dtype", default="fp16")
    parser.add_argument("-bench_torch", action="store_true", default=False)
    parser.add_argument("-print_vgpr", action="store_true", default=False)
    parser.add_argument(
        "-return_all",
        action="store_true",
        default=False,
        help="Prints TFLOPS, walltime, bandwidth.",
    )
    # prints TFLOPS without setting the following
    parser.add_argument(
        "-return_time", action="store_true", default=False, help="Prints only walltime."
    )
    parser.add_argument(
        "-return_bandwidth",
        action="store_true",
        default=False,
        help="Prints only memory bandwidth.",
    )
    parser.add_argument(
        "-test_correctness",
        action="store_true",
        default=False,
        help="Tests correctness of the Triton provider comparing the output to the Torch sdpa.",
    )
    parser.add_argument("-layout", type=str, default=None, help=supported_layouts())
    parser.add_argument(
        "-persistent",
        nargs="?",
        const="fixed",
        choices=["fixed", "dynamic"],
        default=None,
        help="Enable persistent kernels. Use '-persistent dynamic' for dynamic scheduling of the tiles.",
    )
    return parser.parse_args()


arg_to_torch_dtype = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
}


def main():
    args = parse_args()

    if args.model:
        if args.causal is None:  # User didn’t specify -causal
            args.causal = True
        if args.layout is None:  # User didn’t specify -layout
            args.layout = "thd"
        print(
            f"Note: using -model config defaults: causal={args.causal}, layout={args.layout}. This is the most common real life scenario, but can be overridden with -causal and -layout flags."
        )
    else:
        # the defaults for causal and varlen when not using the -model
        if args.causal is None:  # User didn’t specify -causal
            args.causal = False
        if args.layout is None:  # User didn’t specify -layout
            args.layout = "bshd"

    custom_config = False
    assert (
        not args.test_correctness or (not args.layout == "thd") or args.equal_seqlens
    ), "Varlen not supported for -test_correctness, so use -equal_seqlens if using thd layout."

    assert (
        args.layout == "thd" or not args.equal_seqlens or args.model
    ), "Equal sequence lengths arg must be used with the thd layout or a model config."
    if args.hq or args.hk or args.d:
        custom_config = True
        assert (
            args.b and args.hq and args.sq and args.d
        ), "If custom config is specified, please provide \
                all of batch, number of Q heads, Q sequence length \
                and head size."

    if args.model:
        assert not (
            args.hq or args.hk or args.d
        ), "Specifying model fixes hq, hk and d already. Do not provide them!"

    assert (
        args.dtype in arg_to_torch_dtype
    ), "Only fp16, bf16 and f32 types currently supported."

    assert (
        args.layout in supported_layouts()
    ), f"{args.layout} is not in supported layouts: {supported_layouts()}."

    if args.test_correctness:
        test_correctness(custom_config, args)

    if args.layout == "thd" and args.equal_seqlens:
        warnings.warn(
            "Using 'thd' layout with equal_seqlen=True incurs an extra sequence length lookup cost "
            "compared to 'bshd' layout. Consider using 'bshd' for better performance.",
            category=RuntimeWarning,
        )

    if args.print_vgpr:
        assert not args.bench_torch, "Do not use -bench_torch with -print_vgpr."
        print("Retrieving VGPR usage for Triton kernels...")
        fun = lambda: run_benchmark(custom_config, args)  # noqa: E731
        print_vgpr(fun, "fused-attention")
        return 0

    run_benchmark(custom_config, args)


if __name__ == "__main__":
    import sys

    sys.exit(main())
