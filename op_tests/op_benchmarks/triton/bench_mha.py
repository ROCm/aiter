import torch
import sys
import warnings
import argparse
import itertools
import triton
from aiter.ops.triton.mha import (
    flash_attn_func,
    flash_attn_fp8_func,
    flash_attn_varlen_func,
    flash_attn_varlen_fp8_func,
    mha_set_use_fused_bwd_kernel,
)
from aiter.test_mha_common import (
    generate_random_padding_mask,
    generate_qkv,
)
from op_tests.op_benchmarks.triton.utils.argparse import get_parser
from op_tests.op_benchmarks.triton.utils.benchmark_utils import (
    get_model_configs,
    print_vgpr,
    get_caller_name_no_ext,
)

def nonvarlen_benchmark_configs():
    batch_sizes = [1, 4, 16]
    N_HEADS = [16, 48]
    seq_len_q = [1, 1024, 4096]
    seq_len_k = [163, 8192]
    configs = list(itertools.product(batch_sizes, N_HEADS, seq_len_q, seq_len_k))
    configs = [
        (batch_size, N_HEAD, N_HEAD, seq_len_q, seq_len_k)
        for batch_size, N_HEAD, seq_len_q, seq_len_k in configs
    ]
    return configs


def varlen_benchmark_configs():
    batch_sizes = [1, 4, 8]
    N_HEADS = [16, 48]
    seq_len_q = [1, 1024, 4096]
    seq_len_k = [163, 8192]
    configs = list(itertools.product(batch_sizes, N_HEADS, seq_len_q, seq_len_k))
    configs = [
        (batch_size, N_HEAD, N_HEAD, seq_len_q, seq_len_k)
        for batch_size, N_HEAD, seq_len_q, seq_len_k in configs
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
        N_CTX_Q = args.sq if args.sq else [2**i for i in range(1, 14)]
        N_CTX_K = args.sk if args.sk else N_CTX_Q
        HEAD_DIM = config["hidden_size"] // HQ
        if isinstance(N_CTX_Q, list):
            for seq_len in N_CTX_Q:
                fa_configs.append(
                    (model_name, batch_size, HQ, HK, seq_len, seq_len, HEAD_DIM)
                )
        else:
            fa_configs.append(
                (model_name, batch_size, HQ, HK, N_CTX_Q, N_CTX_K, HEAD_DIM)
            )

    return fa_configs


def pad_rearrange_dropout_mask(
    S_dmask,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    seqlen_q,
    seqlen_k,
    num_q_heads,
):
    batch_size = cu_seqlens_q.numel() - 1

    padded_dropout_mask = torch.ones(
        (batch_size, num_q_heads, seqlen_q, seqlen_k), device="cuda"
    )
    for b in range(batch_size):
        start_q = cu_seqlens_q[b].item()
        end_q = cu_seqlens_q[b + 1].item()
        start_k = cu_seqlens_k[b].item()
        end_k = cu_seqlens_k[b + 1].item()

        seqlen_q = end_q - start_q
        seqlen_k = end_k - start_k
        for h in range(S_dmask.shape[1]):
            padded_dropout_mask[b, h, :max_seqlen_q, :max_seqlen_k] = S_dmask[
                b, h, :, :
            ]

    return padded_dropout_mask


def create_benchmark_configs(custom, args):
    dtype = arg_to_torch_dtype[args.dtype]
    hk = args.hq if not args.hk else args.hk
    sk = args.sq if not args.sk else args.sk
    head_size = 128 if not args.d else args.d
    mode = args.mode
    x_names = ["BATCH", "HQ", "HK", "N_CTX_Q", "N_CTX_K"]
    causal = args.causal
    varlen = args.layout == "thd"

    configs = []
    plot_name = get_caller_name_no_ext()
    extra_args = {"D_HEAD": head_size, "dtype": dtype, "causal": causal, "mode": mode}

    if custom:
        x_vals_list = [(args.b, args.hq, hk, args.sq, sk)]
    else:
        if varlen:
            x_vals_list = varlen_benchmark_configs()  # Assume this exists
        else:
            x_vals_list = nonvarlen_benchmark_configs()  # Assume this exists

        if args.model:
            x_vals_list = model_benchmark_configs(args)
            x_names = ["model", "BATCH", "HQ", "HK", "N_CTX_Q", "N_CTX_K", "D_HEAD"]
            plot_name = f"fused-attention-{mode}-layout-{args.layout}-fp8-{args.fp8}-causal-{causal}"
            extra_args = {"dtype": dtype, "causal": causal, "mode": mode}

    if args.metric == "time":
        unit = "ms"
    elif args.metric == "throughput":
        unit = "TFLOPS"
    elif args.metric == "bandwidth":
        unit = "GB/s"
    else:
        raise ValueError("Unknown metric: " + args.metric)

    if mode == "bwd":
        if args.fused_bwd:
            line_vals = [f"fused-bwd({unit})"]
        else:
            line_vals = [f"fused-bwd({unit})", f"bwd({unit})"]
    else:
        line_vals = [f"fwd({unit})"]

    if args.bench_torch:
        line_vals = [f"Triton({unit})", f"Torch({unit})"]

    if args.test_mode:
        if args.fused_bwd:
            line_vals = [f"fused-bwd({unit})"]
        else:
            line_vals = [f"bwd({unit})"]

    configs.append(
        triton.testing.Benchmark(
            x_names=x_names,
            x_vals=x_vals_list,
            line_arg="provider",
            line_vals=line_vals,
            line_names=line_vals,
            styles=[("red", "-"), ("green", "-"), ("yellow", "-")],
            ylabel=unit,
            plot_name=plot_name,
            args=extra_args,
        )
    )
    return configs


def run_benchmark(custom, args):
    torch.manual_seed(20)

    @triton.testing.perf_report(create_benchmark_configs(custom, args))
    def bench_mha(
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
        dropout=0.0,
        model=None,
        sm_scale=None,
        device="cuda",
    ):
        """
        Benchmark or test function for multi-head attention backward pass.
        In test_mode, verifies output matching with non-varlen inputs.
        """
        assert dropout <= 0.0, "Dropout not supported in this benchmark."
        requires_grad = mode == "bwd" or args.test_mode
        return_lse = True
        return_attn_probs = False
        varlen = args.layout == "thd"

        global _USE_FUSED_BWD

        fused_backward = "fused-bwd" in provider

        mha_set_use_fused_bwd_kernel(fused_backward)

        # Default softmax scale to match standard attention
        if sm_scale is None:
            sm_scale = 1.0 / (D_HEAD**0.5)

        # Test mode: run tests from op_tests with specified shapes
        if args.test_mode:
            import op_tests.triton_tests.test_mha as test_mha

            print(
                f"Testing kernel implementation <{provider}> against Torch with shape:"
            )
            print(
                f"BATCH={BATCH}, HQ={HQ}, HK={HK}, N_CTX_Q={N_CTX_Q}, N_CTX_K={N_CTX_K}, D_HEAD={D_HEAD}"
            )
            if varlen:
                test_mha.test_mha(
                    BATCH,
                    N_CTX_Q,
                    N_CTX_K,
                    HQ,
                    HK,
                    D_HEAD,
                    dropout,
                    True,
                    True,
                    causal,
                    args.fp8,
                    dtype,
                )
                print("Forward test passed!")
                test_mha.test_mha_backward_varlen(
                    BATCH,
                    N_CTX_Q,
                    N_CTX_K,
                    HQ,
                    HK,
                    D_HEAD,
                    dropout,
                    causal,
                    args.fp8,
                    dtype,
                )
                print("Backward test passed!")
            else:
                test_mha.test_mha_varlen(
                    BATCH,
                    N_CTX_Q,
                    N_CTX_K,
                    HQ,
                    HK,
                    D_HEAD,
                    dropout,
                    True,
                    True,
                    causal,
                    args.fp8,
                    dtype,
                )
                print("Forward test passed!")
                test_mha.test_mha_backward(
                    BATCH,
                    N_CTX_Q,
                    N_CTX_K,
                    HQ,
                    HK,
                    D_HEAD,
                    dropout,
                    causal,
                    args.fp8,
                    dtype,
                )
                print("Backward test passed!")

            return 0

        # Generate base inputs
        q = torch.randn((BATCH, N_CTX_Q, HQ, D_HEAD), device=device, dtype=dtype)
        k = torch.randn((BATCH, N_CTX_K, HK, D_HEAD), device=device, dtype=dtype)
        v = torch.randn((BATCH, N_CTX_K, HK, D_HEAD), device=device, dtype=dtype)
        q.requires_grad = requires_grad
        k.requires_grad = requires_grad
        v.requires_grad = requires_grad

        # FLOPS calculation variables
        flops_per_matmul = 0

        # Input preparation
        if varlen:
            query_padding_mask = generate_random_padding_mask(
                N_CTX_Q, BATCH, device, mode="full" if args.equal_seqlens else "random"
            )
            key_padding_mask = generate_random_padding_mask(
                N_CTX_K, BATCH, device, mode="full" if args.equal_seqlens else "random"
            )
            (
                q_unpad,
                k_unpad,
                v_unpad,
                cu_seqlens_q,
                cu_seqlens_k,
                max_seqlen_q,
                max_seqlen_k,
                q,
                k,
                v,
                _,
                _,
                _,
            ) = generate_qkv(
                q, k, v, query_padding_mask, key_padding_mask, kvpacked=False
            )
            q_unpad.requires_grad = True
            k_unpad.requires_grad = True
            v_unpad.requires_grad = True

            q_input, k_input, v_input = q_unpad, k_unpad, v_unpad

            num_contexts = len(cu_seqlens_q) - 1
            for i in range(num_contexts):
                seqlen_q = (cu_seqlens_q[i + 1] - cu_seqlens_q[i]).item()
                seqlen_k = (cu_seqlens_k[i + 1] - cu_seqlens_k[i]).item()
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
            q_input, k_input, v_input = q, k, v

            if causal:
                valid_out_elements = (
                    ((N_CTX_K**2 + N_CTX_K) / 2)
                    if N_CTX_Q > N_CTX_K
                    else (N_CTX_Q * N_CTX_K - ((N_CTX_Q**2 - N_CTX_Q) / 2))
                )
                flops_per_matmul = 2.0 * BATCH * HQ * valid_out_elements * D_HEAD
            else:
                flops_per_matmul = 2.0 * BATCH * HQ * N_CTX_Q * N_CTX_K * D_HEAD

        # Benchmark mode
        if varlen:
            if args.fp8:

                def fn():
                    return flash_attn_varlen_fp8_func(
                        q_input,
                        k_input,
                        v_input,
                        cu_seqlens_q,
                        cu_seqlens_k,
                        max_seqlen_q,
                        max_seqlen_k,
                        dropout_p=dropout,
                        softmax_scale=sm_scale,
                        causal=causal,
                        return_lse=return_lse,
                        return_attn_probs=return_attn_probs,
                    )

            else:

                def fn():
                    return flash_attn_varlen_func(
                        q_input,
                        k_input,
                        v_input,
                        cu_seqlens_q,
                        cu_seqlens_k,
                        max_seqlen_q,
                        max_seqlen_k,
                        dropout_p=dropout,
                        softmax_scale=sm_scale,
                        causal=causal,
                        return_lse=return_lse,
                        return_attn_probs=return_attn_probs,
                    )

        else:
            if args.fp8:

                def fn():
                    return flash_attn_fp8_func(
                        q_input,
                        k_input,
                        v_input,
                        dropout_p=dropout,
                        softmax_scale=sm_scale,
                        causal=causal,
                        return_lse=return_lse,
                        return_attn_probs=return_attn_probs,
                    )

            else:

                def fn():
                    return flash_attn_func(
                        q_input,
                        k_input,
                        v_input,
                        dropout_p=dropout,
                        softmax_scale=sm_scale,
                        causal=causal,
                        return_lse=return_lse,
                        return_attn_probs=return_attn_probs,
                    )

        if mode == "bwd":
            with torch.enable_grad():
                triton_out = fn()[0]
                d_out = torch.randn_like(triton_out)

                def fn():
                    grads = torch.autograd.grad(
                        triton_out,
                        (q_input, k_input, v_input),
                        d_out,
                        retain_graph=True,
                    )
                    return grads

        ms = triton.testing.do_bench(fn)

        total_flops = 2 * flops_per_matmul
        if mode == "bwd":
            total_flops *= 2.5  # 2.0(bwd) + 0.5(recompute)

        input_bytes = q.element_size()
        output_bytes = q.element_size()
        if varlen:
            total_num_tokens_q = cu_seqlens_q[-1].item()
            total_num_tokens_k = cu_seqlens_k[-1].item()
        else:
            total_num_tokens_q = BATCH * N_CTX_Q
            total_num_tokens_k = BATCH * N_CTX_K
        mem = (
            total_num_tokens_q * HQ * D_HEAD * input_bytes
            + 2 * total_num_tokens_k * HK * D_HEAD * input_bytes
            + total_num_tokens_q * HQ * D_HEAD * output_bytes
        )
        # return ms
        if "ms" in provider:
            return ms
        elif "TFLOPS" in provider:
            return total_flops / ms * 1e-9
        else:  # GB/s
            return mem / ms * 1e-6

    bench_mha.run(save_path="." if args.o else None, print_data=True)


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
    parser = get_parser(kernel_name="FlashAttention")
    parser.add_argument(
        "-mode", type=str, default="fwd", help="fwd:forward kernel, bwd:backward kernel"
    )
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
    parser.add_argument("--dtype", default="fp16")
    parser.add_argument("-bench_torch", action="store_true", default=False)
    parser.add_argument("-fused_bwd", action="store_true", default=False)
    parser.add_argument("-print_vgpr", action="store_true", default=False)
    parser.add_argument(
        "-test_mode",
        action="store_true",
        default=False,
        help="Tests correctness of the Triton provider comparing the output to the Torch sdpa.",
    )

    parser.add_argument("--layout", type=str, default=None, help=supported_layouts())
    parser.add_argument(
        "-metric",
        nargs="?",
        const="throughput",
        choices=["time", "throughput", "bandwidth"],
        default=None,
        help="Metrics for the kernel benchmark.",
    )
    parser.add_argument(
        "-persistent",
        nargs="?",
        const="fixed",
        choices=["fixed", "dynamic"],
        default=None,
        help="Enable persistent kernels. Use '-persistent dynamic' for dynamic scheduling of the tiles.",
    )
    parser.add_argument(
        "-o", action="store_true", help="Write performance results to CSV file"
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
        if args.causal is None:  # User didn't specify -causal
            args.causal = True
        if args.layout is None:  # User didn't specify -layout
            args.layout = "thd"
        print(
            f"Note: using -model config defaults: causal={True}, layout={'thd'}. This is the most common real life scenario, but can be overridden with -causal and -layout flags."
        )
    else:
        # the defaults for causal and varlen when not using the -model
        if args.causal is None:  # User didn't specify -causal
            args.causal = False
        if args.layout is None:  # User didn't specify -layout
            args.layout = "bshd"

    custom_config = False

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

    if args.layout == "thd" and args.equal_seqlens:
        warnings.warn(
            "Using 'thd' layout with equal_seqlen=True incurs an extra sequence length lookup cost "
            "compared to 'bshd' layout. Consider using 'bshd' for better performance.",
            category=RuntimeWarning,
        )

    if args.print_vgpr:
        assert not args.bench_torch, "Do not use -bench_torch with -print_vgpr."
        print("Retrieving VGPR usage for Triton kernels...")

        def fun():
            return run_benchmark(custom_config, args)

        print_vgpr(fun, get_caller_name_no_ext())
        return 0

    run_benchmark(custom_config, args)


if __name__ == "__main__":
    import sys

    sys.exit(main())
