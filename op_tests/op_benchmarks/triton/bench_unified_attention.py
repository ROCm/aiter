from sys import exit

import argparse
import itertools

import torch
import triton

from aiter.ops.triton.attention.unified_attention import unified_attention
from aiter.ops.triton.quant.sage_attention_quant_wrappers import sage_quant_v2
from op_tests.op_benchmarks.triton.utils.argparse import get_parser
from op_tests.op_benchmarks.triton.utils.benchmark_utils import (
    get_model_configs,
    print_vgpr,
    get_caller_name_no_ext,
)
from op_tests.triton_tests.attention.test_unified_attention import (
    ref_attn,
    make_unified_attn_inputs,
)
from aiter.ops.triton.utils.types import e4m3_dtype


def nonvarlen_benchmark_configs():
    batch_sizes = [1, 4, 16]
    N_HEADS = [16, 48]
    seq_len_q = [1, 1024, 4096]
    seq_len_k = [8192]
    HEAD_DIM = 128
    V_HEAD_DIM = HEAD_DIM
    configs = list(itertools.product(batch_sizes, N_HEADS, seq_len_q, seq_len_k))
    configs = [
        (batch_size, N_HEAD, N_HEAD, seq_len_q, seq_len_k, HEAD_DIM, V_HEAD_DIM)
        for batch_size, N_HEAD, seq_len_q, seq_len_k in configs
    ]
    return configs


def varlen_benchmark_configs():
    batch_sizes = [1, 4, 8]
    N_HEADS = [16, 48]
    seq_len_q = [1, 1024, 4096]
    seq_len_k = [8192]
    HEAD_DIM = 128
    V_HEAD_DIM = HEAD_DIM
    configs = list(itertools.product(batch_sizes, N_HEADS, seq_len_q, seq_len_k))
    configs = [
        (batch_size, N_HEAD, N_HEAD, seq_len_q, seq_len_k, HEAD_DIM, V_HEAD_DIM)
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
        V_HEAD_DIM = HEAD_DIM
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
                        V_HEAD_DIM,
                    )
                )
        else:
            fa_configs.append(
                (model_name, batch_size, HQ, HK, N_CTX_Q, N_CTX_K, HEAD_DIM, V_HEAD_DIM)
            )

    return fa_configs


def create_benchmark_configs(custom, args):
    dtype = arg_to_torch_dtype[args.dtype]
    hk = args.hq if not args.hk else args.hk
    sk = args.sq if not args.sk else args.sk
    head_size = 128 if not args.d else args.d
    head_size_v = head_size if not args.dv else args.dv
    decode_p = args.decode
    x_names = [
        "BATCH",
        "HQ",
        "HK",
        "N_CTX_Q",
        "N_CTX_K",
        "D_HEAD",
        "D_HEAD_V",
        "DECODE_P",
    ]
    causal = args.causal
    varlen = args.layout == "thd"

    configs = []
    plot_name = get_caller_name_no_ext()
    extra_args = {
        "dtype": dtype,
        "causal": causal,
    }

    if custom:
        x_vals_list = [(args.b, args.hq, hk, args.sq, sk, head_size, head_size_v)]
    else:
        if varlen:
            x_vals_list = varlen_benchmark_configs()
        else:
            x_vals_list = nonvarlen_benchmark_configs()

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
                "DECODE_P",
            ]
            plot_name = f"fused-attention-layout-{args.layout}-sagev2-{args.sagev2}-causal-{causal}"
            extra_args = {"dtype": dtype, "causal": causal}

    for i in range(len(x_vals_list)):
        x_vals_list[i] = (*x_vals_list[i], decode_p)

    if args.metric == "time":
        unit = "ms"
    elif args.metric == "throughput":
        unit = "TFLOPS"
    elif args.metric == "bandwidth":
        unit = "GB/s"
    else:
        raise ValueError("Unknown metric: " + args.metric)

    line_vals = [f"fwd({unit})"]

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
        DECODE_P,
        dtype,
        causal,
        provider,
        model=None,
    ):
        varlen = not args.equal_seqlens
        if not varlen:
            seqlens_q = torch.tensor(
                [N_CTX_Q for _ in range(BATCH)], dtype=torch.int32, device="cuda"
            )
            seqlens_k = torch.tensor(
                [N_CTX_K for _ in range(BATCH)], dtype=torch.int32, device="cuda"
            )
        else:
            seqlens_q = torch.randint(
                1, N_CTX_Q + 1, (BATCH,), dtype=torch.int32, device="cuda"
            )
            seqlens_k = torch.randint(
                N_CTX_Q, N_CTX_K + 1, (BATCH,), dtype=torch.int32, device="cuda"
            )

        # turn DECODE_P of the samples to decode samples (seqlen_q == 1)
        if DECODE_P > 0.0:
            num_decode = int(round(DECODE_P * BATCH))
            if num_decode > 0:
                # choose which samples become decode samples
                decode_idx = torch.randperm(BATCH, device=seqlens_q.device)[:num_decode]
                seqlens_q[decode_idx] = 1

        head_size_qk = D_HEAD
        head_size_v = D_HEAD_V

        soft_cap = args.softcap if args.softcap is not None else 0.0
        block_size = args.block_size if args.block_size else 512

        max_num_blocks_per_seq = (seqlens_k.max().item() + block_size - 1) // block_size
        min_required_blocks = BATCH * max_num_blocks_per_seq
        num_blocks = (
            args.num_blocks if args.num_blocks else max(min_required_blocks * 4, 2048)
        )

        (
            query,
            key_cache,
            value_cache,
            cu_query_lens,
            cu_key_lens,
            query_lens,
            kv_lens,
            block_tables,
            output,
            max_query_len,
            max_kv_len,
            scale,
        ) = make_unified_attn_inputs(
            seq_lens=list(zip(seqlens_q.tolist(), seqlens_k.tolist())),
            num_heads=(HQ, HK),
            head_size_qk=head_size_qk,
            head_size_v=head_size_v,
            dtype=dtype,
            block_size=block_size,
            num_blocks=num_blocks,
            kv_layout=args.kv_layout,
        )

        num_query_heads = HQ
        window_size = (
            (args.sliding_window - 1, 0)
            if args.sliding_window is not None
            else (-1, -1)
        )
        sinks = (
            torch.randn(num_query_heads, dtype=torch.bfloat16, device="cuda")
            if args.use_sinks
            else None
        )

        q_input, k_input, v_input = query, key_cache, value_cache
        q_descale = k_descale = v_descale = None

        if args.fp8_full:
            FP8_TYPE = e4m3_dtype
            fp8_max = torch.finfo(FP8_TYPE).max

            q_abs_max = query.abs().amax().clamp(min=1e-9)
            q_descale = (q_abs_max / fp8_max).to(torch.float32).unsqueeze(0).cuda()
            q_fp8 = (query * (fp8_max / q_abs_max)).to(FP8_TYPE)

            k_abs_max = key_cache.abs().amax().clamp(min=1e-9)
            k_descale = (k_abs_max / fp8_max).to(torch.float32).unsqueeze(0).cuda()
            k_fp8 = (key_cache * (fp8_max / k_abs_max)).to(FP8_TYPE)

            v_abs_max = value_cache.abs().amax().clamp(min=1e-9)
            v_descale = (v_abs_max / fp8_max).to(torch.float32).unsqueeze(0).cuda()
            v_fp8 = (value_cache * (fp8_max / v_abs_max)).to(FP8_TYPE)

            q_input, k_input, v_input = q_fp8, k_fp8, v_fp8
        elif args.sagev2:
            q_input, q_descale, k_input, k_descale, v_input, v_descale = sage_quant_v2(
                query,
                key_cache,
                value_cache,
                hadamard_rotation=True,
                R=None,
                BLOCK_R=args.BLOCK_R if args.BLOCK_R else D_HEAD,
                layout_k=args.kv_layout,
                v_descale=None,
            )

        fn = lambda: unified_attention(
            q=q_input,
            k=k_input,
            v=v_input,
            out=output,
            cu_seqlens_q=cu_query_lens,
            seqused_k=kv_lens,
            max_seqlen_q=max_query_len,
            max_seqlen_k=max_kv_len,
            softmax_scale=scale,
            causal=causal,
            window_size=window_size,
            block_table=block_tables,
            softcap=soft_cap,
            q_descale=q_descale,
            k_descale=k_descale,
            v_descale=v_descale,
            sinks=sinks,
            sage_mxfp4=args.sagev2,
            kv_layout=args.kv_layout,
            cu_seqlens_k=cu_key_lens,
        )

        ref_output = ref_attn(
            query,
            key_cache,
            value_cache,
            query_lens,
            kv_lens,
            block_tables,
            scale,
            args.sliding_window,
            soft_cap,
            sinks,
            causal=causal,
            kv_layout=args.kv_layout,
        )

        if args.fp8_full:
            atol, rtol = 1.5e-1, 1.5e-1
        elif args.sagev2:
            atol, rtol = 3.5e-1, 2.5e-1
            # mae = (output - ref_output).abs().mean().item()
            # assert mae < 0.1, f"MXFP4 mean absolute error too high: {mae}"
        else:
            atol, rtol = 1.5e-2, 1e-2
        torch.testing.assert_close(output, ref_output, atol=atol, rtol=rtol)

        ms = triton.testing.do_bench(fn)

        run_correctness = args.test
        if run_correctness:
            fn()
            ref_output = ref_attn(
                query,
                key_cache,
                value_cache,
                query_lens,
                kv_lens,
                block_tables,
                scale,
                args.sliding_window,
                soft_cap,
                sinks,
                causal=causal,
                kv_layout=args.kv_layout,
            )
            if args.fp8_full:
                atol, rtol = 1.5e-1, 1.5e-1
            elif args.sagev2:
                atol, rtol = 3.5e-1, 2.5e-1
                # mae = (output - ref_output).abs().mean().item()
                # assert mae < 0.1, f"MXFP4 mean absolute error too high: {mae}"
            else:
                atol, rtol = 1.5e-2, 1e-2
            torch.testing.assert_close(output, ref_output, atol=atol, rtol=rtol)

        # calculate perf metrics
        total_flops = 0
        num_contexts = len(cu_query_lens) - 1
        for i in range(num_contexts):
            seqlen_q = (cu_query_lens[i + 1] - cu_query_lens[i]).item()
            seqlen_k = (cu_key_lens[i + 1] - cu_key_lens[i]).item()
            if causal:
                valid_out_elements = (
                    ((seqlen_k**2 + seqlen_k) / 2)
                    if seqlen_q > seqlen_k
                    else (seqlen_q * seqlen_k - ((seqlen_q**2 - seqlen_q) / 2))
                )
                total_flops += valid_out_elements * HQ * (D_HEAD + D_HEAD_V) * 2.0
            else:
                total_flops += seqlen_q * seqlen_k * HQ * (D_HEAD + D_HEAD_V) * 2.0

        total_num_tokens_q = cu_query_lens[-1].item()
        total_num_tokens_k = cu_key_lens[-1].item()

        q_size = total_num_tokens_q * HQ * D_HEAD * query.element_size()
        k_size = total_num_tokens_k * HK * D_HEAD * key_cache.element_size()
        v_size = total_num_tokens_k * HK * D_HEAD_V * value_cache.element_size()
        o_size = total_num_tokens_q * HQ * D_HEAD_V * query.element_size()

        # read q, k, v
        mem_read = q_size + k_size + v_size
        # write o
        mem_write = o_size
        # total mem
        mem = mem_read + mem_write

        # return ms
        if "ms" in provider:
            return ms
        elif "TFLOPS" in provider:
            return total_flops / ms * 1e-9
        else:  # GB/s
            return mem / ms * 1e-6

    bench_mha.run(None, print_data=True)


def supported_layouts():
    layouts = (
        "thd: Q, K, V are individual tensors of [total_q/k, num_heads, head_size]. "
    )
    return layouts


# argparse lacks support for boolean argument type
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
    parser = get_parser(kernel_name="UnifiedAttention")

    parser.add_argument("-b", type=int, default=0)
    parser.add_argument("-hq", type=int, default=0)
    parser.add_argument("-hk", type=int, default=0)

    def parse_int_or_list(value):
        if "," in value:
            return [int(x) for x in value.split(",")]
        else:
            return int(value)

    parser.add_argument(
        "-sq",
        type=parse_int_or_list,
        default=0,
        help="Query sequence length - can be a single number or comma-separated list. -b is overwritten as the list length.",
    )
    parser.add_argument(
        "-sk",
        type=parse_int_or_list,
        default=0,
        help="Key sequence length - can be a single number or comma-separated list. Defaults to the same as sq if 0",
    )

    parser.add_argument(
        "-kv_layout",
        type=str,
        choices=["cache", "thd"],
        default="cache",
        help="Layout for key-value tensors: 'cache' for paged cache layout or 'thd' for THD layout",
    )

    parser.add_argument(
        "-num_blocks",
        type=parse_int_or_list,
        default=0,
        help="number of blocks in kv cache (0=auto)",
    )

    parser.add_argument(
        "-block_size",
        type=parse_int_or_list,
        default=544,
        help="block size in kv cache",
    )

    parser.add_argument(
        "-test",
        action="store_true",
        default=False,
        help="test the correctness of each benchmark shape",
    )

    parser.add_argument(
        "-causal",
        action="store_true",
        default=False,
        help="apply causal masking",
    )

    parser.add_argument(
        "-equal_seqlens",
        action="store_true",
        default=False,
        help="If specified, uses equal sequence lengths with thd layout, i.e t = b * sq",
    )

    parser.add_argument(
        "-use_sinks",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "-d",
        type=int,
        default=0,
        help="Q and K head size, if -dv is absent then -d specifies V head size too",
    )
    parser.add_argument("-unified_attention", type=int, default=1)
    parser.add_argument(
        "-sliding_window",
        type=int,
        default=None,
        help="optional sliding window size, default None = not active.",
    )
    parser.add_argument("-softcap", type=float, default=0.0)
    parser.add_argument("-dv", type=int, default=0, help="optional V head size")
    parser.add_argument(
        "-rope_size",
        type=int,
        default=0,
        help="RoPE size for MLA workloads (0=disabled). When >0, V head dim = d - rope_size",
    )
    parser.add_argument(
        "-decode",
        nargs="?",  # 0 or 1 values
        const=1.0,  # value if just `-decode`
        default=0.0,  # value if `-decode` not given at all
        type=float,
        metavar="P",  # shown as -decode P in help
        help="portion of decode samples in batch (omit P for all=1.0)",
    )
    parser.add_argument("-fp8", action="store_true", default=False)
    parser.add_argument(
        "-fp8_full",
        action="store_true",
        default=False,
        help="Full FP8 path: quantize Q, K, V to FP8 with q_descale, k_descale, v_descale",
    )
    parser.add_argument("-sagev2", action="store_true", default=False)
    parser.add_argument("-dtype", default="fp16")
    parser.add_argument("-print_vgpr", action="store_true", default=False)

    parser.add_argument(
        "-metric",
        nargs="?",
        const="throughput",
        choices=["time", "throughput", "bandwidth"],
        default=None,
        help="Metrics for the kernel benchmark.",
    )

    return parser.parse_args()


arg_to_torch_dtype = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
}


def main():
    args = parse_args()
    args.layout = "thd"
    if args.model:
        if args.causal is None:  # User didn't specify -causal
            args.causal = True
        print(
            f"Note: using -model config defaults: causal={True}. This is the most common real life scenario, but can be overridden with -causal and -layout flags."
        )
    else:
        # the defaults for causal and varlen when not using the -model
        if args.causal is None:  # User didn't specify -causal
            args.causal = False

    custom_config = False

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

    assert (
        args.dtype in arg_to_torch_dtype
    ), "Only fp16, bf16 and f32 types currently supported."

    assert (
        args.layout in supported_layouts()
    ), f"{args.layout} is not in supported layouts: {supported_layouts()}."

    if args.print_vgpr:
        print("Retrieving VGPR usage for Triton kernels...")

        def fun():
            return run_benchmark(custom_config, args)

        print_vgpr(fun, get_caller_name_no_ext())
        return 0

    run_benchmark(custom_config, args)


if __name__ == "__main__":
    exit(main())
