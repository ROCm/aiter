import torch
import warnings
import argparse
import itertools
from dataclasses import dataclass
from typing import Callable
import triton
from aiter.ops.triton.attention.mha import (
    flash_attn_func,
    flash_attn_varlen_func,
    mha_set_use_fused_bwd_kernel,
)
from aiter.ops.triton.attention.mha_v3 import (
    flash_attn_fp8_func,
    flash_attn_varlen_fp8_func,
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


@dataclass(frozen=True)
class Provider:
    label: str
    make_fn: Callable[..., Callable]


# Registry populated by create_benchmark_configs, looked up by bench_mha
_PROVIDERS: dict[str, Provider] = {}


def _make_bf16_fn(q, k, v, **kw):
    mha_set_use_fused_bwd_kernel(False)
    return lambda: flash_attn_func(
        q,
        k,
        v,
        dropout_p=kw["dropout"],
        softmax_scale=kw["sm_scale"],
        causal=kw["causal"],
        return_lse=kw["return_lse"],
        return_attn_probs=kw["return_attn_probs"],
        sink=kw["sink"],
    )


def _make_bf16_varlen_fn(q, k, v, **kw):
    mha_set_use_fused_bwd_kernel(False)
    return lambda: flash_attn_varlen_func(
        q,
        k,
        v,
        kw["cu_seqlens_q"],
        kw["cu_seqlens_k"],
        kw["max_seqlen_q"],
        kw["max_seqlen_k"],
        dropout_p=kw["dropout"],
        softmax_scale=kw["sm_scale"],
        causal=kw["causal"],
        return_lse=kw["return_lse"],
        return_attn_probs=kw["return_attn_probs"],
        sink=kw["sink"],
    )


def _make_bf16_fused_fn(q, k, v, **kw):
    if kw.get("has_pe") or kw.get("has_sink"):
        warnings.warn("Skipping: PE or sink not supported for this provider.")
        return None
    mha_set_use_fused_bwd_kernel(True)
    return lambda: flash_attn_func(
        q,
        k,
        v,
        dropout_p=kw["dropout"],
        softmax_scale=kw["sm_scale"],
        causal=kw["causal"],
        return_lse=kw["return_lse"],
        return_attn_probs=kw["return_attn_probs"],
        sink=kw["sink"],
    )


def _make_bf16_fused_varlen_fn(q, k, v, **kw):
    if kw.get("has_pe") or kw.get("has_sink"):
        warnings.warn("Skipping: PE or sink not supported for this provider.")
        return None
    mha_set_use_fused_bwd_kernel(True)
    return lambda: flash_attn_varlen_func(
        q,
        k,
        v,
        kw["cu_seqlens_q"],
        kw["cu_seqlens_k"],
        kw["max_seqlen_q"],
        kw["max_seqlen_k"],
        dropout_p=kw["dropout"],
        softmax_scale=kw["sm_scale"],
        causal=kw["causal"],
        return_lse=kw["return_lse"],
        return_attn_probs=kw["return_attn_probs"],
        sink=kw["sink"],
    )


def _make_fp8_fn(q, k, v, **kw):
    if kw.get("has_pe") or kw.get("has_sink"):
        warnings.warn("Skipping: PE or sink not supported for this provider.")
        return None
    mha_set_use_fused_bwd_kernel(False)
    return lambda: flash_attn_fp8_func(
        q,
        k,
        v,
        softmax_scale=kw["sm_scale"],
        causal=kw["causal"],
    )


def _make_fp8_varlen_fn(q, k, v, **kw):
    if kw.get("has_pe") or kw.get("has_sink"):
        warnings.warn("Skipping: PE or sink not supported for this provider.")
        return None
    mha_set_use_fused_bwd_kernel(False)
    return lambda: flash_attn_varlen_fp8_func(
        q,
        k,
        v,
        kw["cu_seqlens_q"],
        kw["cu_seqlens_k"],
        kw["max_seqlen_q"],
        kw["max_seqlen_k"],
        softmax_scale=kw["sm_scale"],
        causal=kw["causal"],
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
                    (
                        model_name,
                        batch_size,
                        HQ,
                        HK,
                        seq_len,
                        seq_len,
                        HEAD_DIM,
                        HEAD_DIM,
                    )
                )
        else:
            fa_configs.append(
                (model_name, batch_size, HQ, HK, N_CTX_Q, N_CTX_K, HEAD_DIM, HEAD_DIM)
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
    dtype = arg_to_torch_dtype[args.tensor_dtype]
    hk = args.hq if not args.hk else args.hk
    sk = args.sq if not args.sk else args.sk
    head_size = 128 if not args.d else args.d
    head_size_v = head_size if not args.dv else args.dv
    mode = args.mode
    x_names = ["BATCH", "HQ", "HK", "N_CTX_Q", "N_CTX_K"]
    causal = args.causal
    varlen = args.layout == "thd"

    configs = []
    plot_name = get_caller_name_no_ext()
    extra_args = {
        "D_HEAD": head_size,
        "D_HEAD_V": head_size_v,
        "dtype": dtype,
        "causal": causal,
        "mode": mode,
    }

    if custom:
        x_vals_list = [(args.b, args.hq, hk, args.sq, sk)]
    else:
        if varlen:
            x_vals_list = varlen_benchmark_configs()  # Assume this exists
        else:
            x_vals_list = nonvarlen_benchmark_configs()  # Assume this exists

        if args.model:
            x_vals_list = model_benchmark_configs(args)
            x_names = [
                "model",
                "BATCH",
                "HQ",
                "HK",
                "N_CTX_Q",
                "N_CTX_K",
                "D_HEAD",
                "D_HEAD_V",
            ]
            plot_name = f"fused-attention-{mode}-layout-{args.layout}-dtype-{args.dtype}-causal-{causal}"
            extra_args = {"dtype": dtype, "causal": causal, "mode": mode}

    if args.metric == "time":
        unit = "ms"
    elif args.metric == "throughput":
        unit = "TFLOPS"
    elif args.metric == "bandwidth":
        unit = "GB/s"
    else:
        raise ValueError("Unknown metric: " + args.metric)

    dtype_to_fn = {
        "bf16": _make_bf16_varlen_fn if varlen else _make_bf16_fn,
        "fp16": _make_bf16_varlen_fn if varlen else _make_bf16_fn,
        "fp32": _make_bf16_varlen_fn if varlen else _make_bf16_fn,
        "fp8": _make_fp8_varlen_fn if varlen else _make_fp8_fn,
    }
    bf16_fused_fn = _make_bf16_fused_varlen_fn if varlen else _make_bf16_fused_fn

    providers = []
    for d in args.dtypes:
        label = d.upper()
        if mode == "bwd":
            if args.fused_bwd and d != "fp8":
                providers.append(Provider(f"{label}-fused-bwd({unit})", bf16_fused_fn))
            else:
                providers.append(Provider(f"{label}-bwd({unit})", dtype_to_fn[d]))
        else:
            providers.append(Provider(f"{label}-fwd({unit})", dtype_to_fn[d]))

    if args.bench_torch:
        bf16_fn = dtype_to_fn["bf16"]
        providers = [
            Provider(f"Triton({unit})", bf16_fn),
            Provider(f"Torch({unit})", bf16_fn),
        ]

    _PROVIDERS.clear()
    for p in providers:
        _PROVIDERS[p.label] = p
    line_vals = [p.label for p in providers]

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
        D_HEAD_V,
        dtype,
        causal,
        mode,
        provider,
        dropout=0.0,
        model=None,
        sm_scale=None,
        device="cuda",
    ):
        assert dropout <= 0.0, "Dropout not supported in this benchmark."
        requires_grad = mode == "bwd"
        return_lse = True
        return_attn_probs = False
        varlen = args.layout == "thd"
        has_pe = D_HEAD > D_HEAD_V
        provider_obj = _PROVIDERS[provider]

        # Default softmax scale to match standard attention
        if sm_scale is None:
            sm_scale = 1.0 / (D_HEAD**0.5)

        # Generate base inputs
        q = torch.randn(
            (BATCH, N_CTX_Q, HQ, D_HEAD),
            device=device,
            dtype=dtype,
            requires_grad=requires_grad,
        )
        k = torch.randn(
            (BATCH, N_CTX_K, HK, D_HEAD),
            device=device,
            dtype=dtype,
            requires_grad=requires_grad,
        )
        v = torch.randn(
            (BATCH, N_CTX_K, HK, D_HEAD_V),
            device=device,
            dtype=dtype,
            requires_grad=requires_grad,
        )
        sink = (
            torch.randn((HQ,), device=device, dtype=dtype, requires_grad=requires_grad)
            if args.sink
            else None
        )

        # FLOPS calculation variables
        total_flops = 0.0

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
            q_unpad.requires_grad = requires_grad
            k_unpad.requires_grad = requires_grad
            v_unpad.requires_grad = requires_grad

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
                    total_flops += valid_out_elements * HQ * (D_HEAD + D_HEAD_V) * 2.0
                else:
                    total_flops += seqlen_q * seqlen_k * HQ * (D_HEAD + D_HEAD_V) * 2.0
        else:
            q_input, k_input, v_input = q, k, v

            if causal:
                valid_out_elements = (
                    ((N_CTX_K**2 + N_CTX_K) / 2)
                    if N_CTX_Q > N_CTX_K
                    else (N_CTX_Q * N_CTX_K - ((N_CTX_Q**2 - N_CTX_Q) / 2))
                )
                total_flops += (
                    2.0 * BATCH * HQ * valid_out_elements * (D_HEAD + D_HEAD_V)
                )
            else:
                total_flops += (
                    2.0 * BATCH * HQ * N_CTX_Q * N_CTX_K * (D_HEAD + D_HEAD_V)
                )

        # Build fn from provider
        fn_kwargs = dict(
            sm_scale=sm_scale,
            causal=causal,
            dropout=dropout,
            return_lse=return_lse,
            return_attn_probs=return_attn_probs,
            sink=sink,
            has_pe=has_pe,
            has_sink=args.sink,
        )
        if varlen:
            fn_kwargs.update(
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
            )
        fn = provider_obj.make_fn(q_input, k_input, v_input, **fn_kwargs)
        if fn is None:
            return 0

        if mode == "bwd":
            with torch.enable_grad():
                triton_out = fn()[0]
                d_out = torch.randn_like(triton_out)

                grad_inputs = (q_input, k_input, v_input)
                if sink is not None:
                    grad_inputs += (sink,)

                def fn():
                    grads = torch.autograd.grad(
                        triton_out,
                        grad_inputs,
                        d_out,
                        retain_graph=True,
                    )
                    return grads

        if args.profile is not None:
            import os

            # Warmup
            for _ in range(3):
                fn()
                torch.cuda.synchronize()

            prof = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                record_shapes=True,
                with_stack=True,
            )
            prof.start()
            for _ in range(5):
                fn()
                torch.cuda.synchronize()
            prof.stop()

            shape_str = f"B{BATCH}_HQ{HQ}_HK{HK}_SQ{N_CTX_Q}_SK{N_CTX_K}_D{D_HEAD}"
            print(f"\n--- Profile: {mode} {shape_str} ---")
            print(
                prof.key_averages().table(
                    sort_by="self_cuda_time_total",
                    row_limit=30,
                )
            )
            trace_dir = os.path.join(args.profile, f"{mode}_{shape_str}")
            os.makedirs(trace_dir, exist_ok=True)
            prof.export_chrome_trace(os.path.join(trace_dir, "trace.json"))
            return 0

        ms = triton.testing.do_bench(fn)

        if mode == "bwd":
            total_flops *= 2.5  # 2.0(bwd) + 0.5(recompute)

        if varlen:
            total_num_tokens_q = cu_seqlens_q[-1].item()
            total_num_tokens_k = cu_seqlens_k[-1].item()
        else:
            total_num_tokens_q = BATCH * N_CTX_Q
            total_num_tokens_k = BATCH * N_CTX_K
        q_size = total_num_tokens_q * HQ * D_HEAD * q.element_size()
        k_size = total_num_tokens_k * HK * D_HEAD * k.element_size()
        v_size = total_num_tokens_k * HK * D_HEAD_V * v.element_size()
        o_size = total_num_tokens_q * HQ * D_HEAD_V * q.element_size()
        if mode == "fwd":
            # read q, k, v
            mem_read = q_size + k_size + v_size
            # write o
            mem_write = o_size
        else:
            # read q, k, v, do
            mem_read = q_size + k_size + v_size + o_size
            # write dq, dk, dv
            mem_write = q_size + k_size + v_size
        mem = mem_read + mem_write

        # return ms
        if "ms" in provider:
            return ms
        elif "TFLOPS" in provider:
            return total_flops / ms * 1e-9
        else:  # GB/s
            return mem / ms * 1e-6

    try:
        bench_mha.run(save_path=args.o, print_data=True)
    except Exception as e:
        print(f"\n[WARN] {args.mode} benchmark failed: {e}", flush=True)


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


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
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
    parser.add_argument(
        "-d",
        type=int,
        default=0,
        help="Q and K head size, if -dv is absent then -d specifies V head size too",
    )
    parser.add_argument("-dv", type=int, default=0, help="optional V head size")
    parser.add_argument("-causal", type=str2bool, default=None)
    parser.add_argument("-quantize_p", action="store_true", default=False)
    parser.add_argument(
        "--dtype",
        default="bf16",
        help="Comma-separated compute types to benchmark: bf16, fp16, fp32, fp8 (e.g. --dtype bf16,fp8)",
    )
    parser.add_argument("-bench_torch", action="store_true", default=False)
    parser.add_argument("-fused_bwd", action="store_true", default=False)
    parser.add_argument("-print_vgpr", action="store_true", default=False)
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
        "-o",
        type=str,
        default=None,
        metavar="DIR",
        help="Write performance results to CSV in DIR",
    )
    parser.add_argument(
        "--profile",
        type=str,
        default=None,
        metavar="DIR",
        help="Enable torch.profiler and write chrome traces to DIR.",
    )
    parser.add_argument(
        "-sink", action="store_true", default=False, help="use attention sink"
    )
    return parser.parse_args(args=args)


VALID_DTYPES = {"fp16", "bf16", "fp32", "fp8"}

arg_to_torch_dtype = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
}


def post_process_args(args: argparse.Namespace) -> tuple[argparse.Namespace, bool]:
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
    if args.hq or args.hk or args.d or args.dv:
        custom_config = True
        if not args.dv:
            args.dv = args.d
        assert (
            args.b and args.hq and args.sq and args.d and args.dv
        ), "If custom config is specified, please provide \
                all of batch, number of Q heads, Q sequence length \
                and head size."

    if args.model:
        assert not (
            args.hq or args.hk or args.d or args.dv
        ), "Specifying model fixes hq, hk and d already. Do not provide them!"

    args.dtypes = [d.strip() for d in args.dtype.split(",")]
    for d in args.dtypes:
        assert (
            d in VALID_DTYPES
        ), f"Unknown dtype '{d}'. Supported: {sorted(VALID_DTYPES)}"
    # Tensor dtype is the first non-fp8 dtype, or bf16 if only fp8
    args.tensor_dtype = next((d for d in args.dtypes if d != "fp8"), "bf16")

    assert (
        args.layout in supported_layouts()
    ), f"{args.layout} is not in supported layouts: {supported_layouts()}."

    if args.layout == "thd" and args.equal_seqlens:
        warnings.warn(
            "Using 'thd' layout with equal_seqlen=True incurs an extra sequence length lookup cost "
            "compared to 'bshd' layout. Consider using 'bshd' for better performance.",
            category=RuntimeWarning,
        )

    return args, custom_config


def main(args: list[str] | None = None) -> None:
    parsed_args = parse_args(args=args)
    parsed_args, custom_config = post_process_args(parsed_args)

    if parsed_args.print_vgpr:
        assert not parsed_args.bench_torch, "Do not use -bench_torch with -print_vgpr."
        print("Retrieving VGPR usage for Triton kernels...")

        def fun():
            return run_benchmark(custom_config, parsed_args)

        print_vgpr(fun, get_caller_name_no_ext())
        return

    run_benchmark(custom_config, parsed_args)


if __name__ == "__main__":
    main()
