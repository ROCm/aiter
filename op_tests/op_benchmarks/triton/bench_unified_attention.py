import torch
import sys
import warnings
import argparse
import itertools
import triton
import aiter
from op_tests.op_benchmarks.triton.utils.argparse import get_parser
from op_tests.op_benchmarks.triton.utils.benchmark_utils import (
    get_model_configs,
    print_vgpr,
    get_caller_name_no_ext,
)
from aiter.ops.triton.attention.unified_attention import unified_attention
from op_tests.triton_tests.attention.test_unified_attention import ref_paged_attn


def nonvarlen_benchmark_configs():
    batch_sizes = [1, 4, 16]
    N_HEADS = [16, 48]
    seq_len_q = [1, 1024, 4096]
    seq_len_k = [163, 8192]
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
    seq_len_k = [163, 8192]
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
    x_names = ["BATCH", "HQ", "HK", "N_CTX_Q", "N_CTX_K", "D_HEAD", "D_HEAD_V", "DECODE_P"]
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
            plot_name = f"fused-attention-layout-{args.layout}-fp8-{args.fp8}-causal-{causal}"
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
        assert args.layout == "thd"
        varlen = not args.equal_seqlens

        if not varlen:
            seqlens_q = torch.tensor([N_CTX_Q for _ in range(BATCH)], dtype=torch.int32, device="cuda")
            seqlens_k = torch.tensor([N_CTX_K for _ in range(BATCH)], dtype=torch.int32, device="cuda")
        else:
            seqlens_q = torch.randint(1,N_CTX_Q + 1, (BATCH,), dtype=torch.int32, device="cuda")
            seqlens_k = torch.randint(1,N_CTX_K + 1, (BATCH,), dtype=torch.int32, device="cuda")

        # turn DECODE_P of the samples to decode samples (seqlen_q == 1)
        if DECODE_P > 0.0:
            num_decode = int(round(DECODE_P * BATCH))
            if num_decode > 0:
                # choose which samples become decode samples
                decode_idx = torch.randperm(BATCH, device=seqlens_q.device)[:num_decode]
                seqlens_q[decode_idx] = 1
        
        num_seqs = BATCH
        num_query_heads = HQ
        num_kv_heads = HK
        head_size = D_HEAD
        assert num_query_heads % num_kv_heads == 0
        max_query_len = max(seqlens_q).item()
        max_kv_len = max(seqlens_k).item()
        soft_cap = args.softcap
        block_size = args.block_size if args.block_size else 512
        num_blocks = args.num_blocks if args.num_blocks else (max_kv_len * BATCH // block_size + 1)

        window_size = (args.sliding_window - 1, 0) if args.sliding_window is not None else (-1, -1)
        scale = D_HEAD**-0.5

        query = torch.randn(
            sum(seqlens_q), num_query_heads, head_size, dtype=dtype, device="cuda"
        )
        key_cache = torch.randn(
            num_blocks, block_size, num_kv_heads, head_size, dtype=dtype, device="cuda"
        )
        value_cache = torch.randn_like(key_cache)
        cu_seqlens_q = torch.zeros(len(seqlens_q) + 1, dtype=torch.int32, device="cuda")
        cu_seqlens_q[1:] = seqlens_q.cumsum(dim=0, dtype=torch.int32)
        cu_seqlens_k = torch.zeros(len(seqlens_k) + 1, dtype=torch.int32, device="cuda")
        cu_seqlens_k[1:] = seqlens_k.cumsum(dim=0, dtype=torch.int32)

        max_num_blocks_per_seq = (max_kv_len + block_size - 1) // block_size
        block_tables = torch.randint(
            0,
            num_blocks,
            (num_seqs, max_num_blocks_per_seq),
            dtype=torch.int32,
            device="cuda",
        )
        if args.use_sinks:
            sinks = torch.randn(num_query_heads, dtype=torch.bfloat16, device="cuda")
        else:
            sinks = None
        
        if args.test:
            ref_output = ref_paged_attn(
                query=query,
                key_cache=key_cache,
                value_cache=value_cache,
                query_lens=seqlens_q,
                kv_lens=seqlens_k,
                block_tables=block_tables,
                scale=scale,
                sliding_window=args.sliding_window,
                soft_cap=soft_cap,
                sinks=sinks,
            )
        
        if args.fp8:
            FP8_TYPE = aiter.dtypes.fp8
            FP8_MAX = torch.finfo(FP8_TYPE).max
            # TODO: providing q descale is not supported????
            # q_descale = query.max().to(torch.float32) / FP8_MAX 
            # query = query / q_descale
            query = query.to(FP8_TYPE)
            q_descale = None
            k_descale = key_cache.max().to(torch.float32) / FP8_MAX 
            key_cache = (key_cache / k_descale).to(query.dtype)
            v_descale = value_cache.max().to(torch.float32) / FP8_MAX 
            value_cache = (value_cache / v_descale).to(query.dtype)
        else:
            q_descale, k_descale, v_descale = None, None, None

        output = torch.empty_like(query)
        fn =  lambda: unified_attention(
            q=query,
            k=key_cache,
            v=value_cache,
            out=output,
            cu_seqlens_q=cu_seqlens_q,
            seqused_k=seqlens_k,
            max_seqlen_q=max_query_len,
            max_seqlen_k=max_kv_len,
            softmax_scale=scale,
            causal=causal,
            window_size=window_size,
            block_table=block_tables,
            softcap=soft_cap if soft_cap is not None else 0,
            q_descale=q_descale,
            k_descale=k_descale,
            v_descale=v_descale,
            sinks=sinks,
        )
        
        if args.test:
            atol, rtol = 1.5e-2, 1e-2
            if args.fp8:
                atol, rtol = 1.5e-1, 1.5e-1
            fn() # evaluate function
            torch.testing.assert_close(
                output, ref_output, atol=atol, rtol=rtol
            ), f"{torch.max(torch.abs(output - ref_output))}"
        
        ms = triton.testing.do_bench(fn)
        
        # calculate perf metrics
        total_flops = 0
        if varlen:
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

        if varlen:
            total_num_tokens_q = cu_seqlens_q[-1].item()
            total_num_tokens_k = cu_seqlens_k[-1].item()
        else:
            total_num_tokens_q = BATCH * N_CTX_Q
            total_num_tokens_k = BATCH * N_CTX_K
        
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

    parser.add_argument("-b", type=int, default=0)
    parser.add_argument("-hq", type=int, default=0)
    parser.add_argument("-hk", type=int, default=0)

    def parse_int_or_list(value):
        if "," in value:
            return (
                value.strip()
            )  # if list, return stripped string and parse when creating tensor
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
        "-num_blocks",
        type=parse_int_or_list,
        default=0,
        help="number of blocks in kv cache",
    )

    parser.add_argument(
        "-block_size",
        type=parse_int_or_list,
        default=0,
        help="block size in kv cache",
    )

    parser.add_argument(
        "-test",
        action="store_true",
        default=False,
        help="test the correctness of each benchmark shape",
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
    parser.add_argument("-sliding_window", type=int, default=None, help="optional sliding window size, default None = not active.")
    parser.add_argument("-softcap", type=float, default=0.0)
    parser.add_argument("-dv", type=int, default=0, help="optional V head size")
    parser.add_argument(
        "-decode",
        nargs="?",          # 0 or 1 values
        const=1.0,          # value if just `-decode`
        default=0.0,        # value if `-decode` not given at all
        type=float,
        metavar="P",        # shown as -decode P in help
        help="portion of decode samples in batch (omit P for all=1.0)",
    )
    parser.add_argument("-fp8", action="store_true", default=False)
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
    args.causal = True
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