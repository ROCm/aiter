import triton
from op_tests.op_benchmarks.triton.utils.benchmark_utils import (
    get_model_configs,
    print_vgpr,
)
import torch
import sys
import argparse
import itertools

from aiter.ops.triton.mla_decode_rope import (
    decode_attention_fwd_grouped_rope,
)
from op_tests.op_benchmarks.triton.utils.argparse import get_parser
from op_tests.triton_tests.test_mla_decode_rope import input_helper, ref_preprocess

arg_to_torch_dtype = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
}


def nonvarlen_benchmark_configs(args: argparse.Namespace):
    batch_sizes = [1, 4, 16] if args.b == 0 else [args.b]
    N_HEADS = [16, 48] if args.hq == 0 else [args.hq]
    seq_len_k = [163, 8192] if args.sk == 0 else [args.sk]

    kv_lora_rank = 512
    qk_rope_head_dim = 64

    if args.use_neox_style_rope:
        rotary_dim = qk_rope_head_dim // 2
    else:
        rotary_dim = qk_rope_head_dim

    configs = list(itertools.product(batch_sizes, N_HEADS, seq_len_k))
    configs = [
        (batch_size, N_HEAD, seq_len_k, kv_lora_rank, qk_rope_head_dim, rotary_dim)
        for batch_size, N_HEAD, seq_len_k in configs
    ]
    return configs


def model_benchmark_configs(args: argparse.Namespace):
    config_file = args.model_configs
    configs = get_model_configs(config_path=config_file, models=args.model)
    fa_configs = []
    batch_size = args.b if args.b else 4

    for model_name, config in configs.items():
        num_q_heads = config["num_attention_heads"]
        num_kv_heads = config["num_key_value_heads"]
        assert (
            num_q_heads == num_kv_heads
        ), """Grouped Query Attention benchmarking not yet supported - try using a model
            with the same number of query and key/value heads (e.g Deepseek-V3)"""
        qk_rope_head_dim = config.get("qk_rope_head_dim", 64)
        kv_lora_rank = config.get("kv_lora_rank", 512)
        rotary_dim = (
            qk_rope_head_dim // 2 if args.use_neox_style_rope else qk_rope_head_dim
        )
        N_CTX_K = args.sk if args.sk else [2**i for i in range(1, 14)]
        if isinstance(N_CTX_K, list):
            for seq_len in N_CTX_K:
                fa_configs.append(
                    (
                        batch_size,
                        num_q_heads,
                        seq_len,
                        kv_lora_rank,
                        qk_rope_head_dim,
                        rotary_dim,
                    )
                )
        else:
            fa_configs.append(
                (
                    batch_size,
                    num_q_heads,
                    N_CTX_K,
                    kv_lora_rank,
                    qk_rope_head_dim,
                    rotary_dim,
                )
            )

    return fa_configs


def create_benchmark_configs(args: argparse.Namespace):
    dtype = arg_to_torch_dtype[args.dtype]
    x_names = ["BATCH", "H", "S", "kv_lora_rank", "qk_rope_head_dim", "rotary_dim"]

    configs = []
    plot_name = "MLA-decode-RoPE"
    extra_args = {
        "dtype": dtype,
        "use_rope": not args.no_rope,
        "is_neox_style": args.use_neox_style_rope,
    }

    if args.model:
        x_vals_list = model_benchmark_configs(args)
        plot_name += f"-{args.model}"
    else:
        x_vals_list = nonvarlen_benchmark_configs(args)

    if args.metric == "time":
        unit = "ms"
    elif args.metric == "throughput":
        unit = "TFLOPS"
    elif args.metric == "bandwidth":
        unit = "GB/s"
    else:
        raise ValueError("Unknown metric: " + args.metric)

    line_vals = [f"{unit}"]
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


def run_benchmark(args: argparse.Namespace):
    torch.manual_seed(20)

    @triton.testing.perf_report(create_benchmark_configs(args))
    def bench_mla(
        BATCH: int,
        H: int,  # number of query heads, equal to the number of k/v heads
        S: int,
        kv_lora_rank: int,
        qk_rope_head_dim: int,
        rotary_dim: int,
        dtype: torch.dtype,
        use_rope: bool,
        is_neox_style: bool = False,
        num_kv_splits: int = 2,
        sm_scale: float = 1.0,
        logit_cap: float = 0.0,
        device="cuda",
        metric: str = "throughput",
        **kwargs,
    ):
        """
        Benchmarks our multi-head latent attention decode kernel.

        Todo:
        - Support variable length sequences (by writing new data generation fns that generate the
        appropriate paged kv cache).
        - Support GQA benchmarking (e.g generate inputs where q_heads = kv_heads * N.
        Right now q_heads == kv_heads).
        """

        kv_indptr, kv_indices, q, kv_cache, attn_logits, rotary_emb, positions = (
            input_helper(
                BATCH,
                H,
                S,
                kv_lora_rank,
                rotary_dim,
                qk_rope_head_dim,
                num_kv_splits,
                dtype,
                device,
            )
        )

        k_input, v_input = ref_preprocess(kv_cache, kv_lora_rank)

        tri_o = torch.empty(BATCH, H, kv_lora_rank, dtype=kv_cache.dtype, device=device)
        # we need to return the rope'd k_pe_tokens to be saved in cache
        k_pe_tokens = torch.empty(
            BATCH, qk_rope_head_dim, dtype=kv_cache.dtype, device=device
        )

        # FLOPS calculation
        num_q_heads = H
        num_kv_heads = k_input.shape[1]
        assert num_q_heads >= num_kv_heads

        # Note: As far as I understand it, rotary_dim <= qk_rope_head_dim,
        # with the latter being the portion of the query/key dim reserved for RoPE
        # and the former being the number of dims that RoPE is actually applied to.
        # Please correct the following calculations if the above is incorrect.

        # per batch:
        attn_nope_flops = num_q_heads * kv_lora_rank * S * 2
        attn_rope_flops = num_q_heads * qk_rope_head_dim * S * 2
        av_flops = (
            num_q_heads * kv_lora_rank * S * 2
        )  # multiplying attention map with v

        # flops to apply RoPE (per batch)
        rope_q_flops = 2 * num_q_heads * rotary_dim
        rope_k_flops = (
            2 * num_kv_heads * rotary_dim
        )  # only one token, since prev. rotated ks are cached

        total_flops = BATCH * (
            attn_nope_flops + attn_rope_flops + av_flops + rope_q_flops + rope_k_flops
        )

        # Memory transfer calculations (per batch)
        # bytes read:
        q_elems_read = num_q_heads * (kv_lora_rank + qk_rope_head_dim)
        k_rope_elems_read = num_kv_heads * qk_rope_head_dim * S
        kv_nope_elems_read = num_kv_heads * kv_lora_rank * S
        cos_sine_cache_read = (num_q_heads + num_kv_heads) * rotary_dim

        # total indices read (across the full batch)
        kv_indptrs_read = BATCH + 1
        kv_indices_read = BATCH * S

        bytes_read = (
            BATCH * q_elems_read * q.element_size()
            + BATCH * k_rope_elems_read * k_input.element_size()
            + BATCH * kv_nope_elems_read * k_input.element_size()
            + BATCH * cos_sine_cache_read * rotary_emb.cos_sin_cache.element_size()
            + kv_indptrs_read
            + kv_indices_read
        )

        # bytes written:
        out_elems = num_q_heads * kv_lora_rank
        new_k_pe_elems = qk_rope_head_dim  # to add to kv cache

        bytes_written = (
            BATCH * out_elems * tri_o.element_size()
            + BATCH * new_k_pe_elems * k_pe_tokens.element_size()
        )

        mem = bytes_read + bytes_written

        ms = triton.testing.do_bench(
            lambda: decode_attention_fwd_grouped_rope(
                q,
                k_input,
                v_input,
                tri_o,
                kv_indptr,
                kv_indices,
                k_pe_tokens if use_rope else None,
                kv_lora_rank,
                rotary_dim if use_rope else None,
                rotary_emb.cos_sin_cache if use_rope else None,
                positions if use_rope else None,
                attn_logits,
                num_kv_splits,
                sm_scale,
                logit_cap,
                use_rope,
                is_neox_style,
            ),
            warmup=25,
            rep=100,
        )

        # Return exactly one scalar depending on which metric is active
        if metric == "time":
            return ms
        elif metric == "throughput":
            tflops = total_flops / ms * 1e-9
            return tflops
        elif metric == "bandwidth":
            bandwidth = mem / (ms * 1e-3) * 1e-9  # GB/s
            return bandwidth
        else:
            raise ValueError("Unknown metric: " + metric)

    bench_mla.run(save_path=".", print_data=True, show_plots=False)


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
    parser = get_parser(kernel_name="MLA Decode with RoPE")
    parser.add_argument("-b", type=int, default=0)
    parser.add_argument(
        "-hq",
        type=int,
        default=0,
        help="Number of query heads (equal to number of key/value heads)",
    )
    parser.add_argument(
        "-sk",
        type=int,
        default=0,
        help="Sequence length (since this is decode, this is the length of the key/value sequence)",
    )
    parser.add_argument(
        "--no-rope",
        action="store_true",
        default=False,
        help="Disable rotary positional embeddings.",
    )
    parser.add_argument(
        "--use-neox-style-rope",
        action="store_true",
        default=False,
        help="Use Neox style rotary positional embeddings over vanilla RoPE. This is incompatible with the --no-rope flag.",
    )
    parser.add_argument("--dtype", default="fp16")
    parser.add_argument("--print_vgpr", action="store_true", default=False)

    return parser.parse_args()


def main():
    args = parse_args()

    # if args.b or args.hq or args.hk or args.sk:
    # assert (
    #     args.b and args.hq and args.sk and args.d
    # ), "If custom config is specified, please provide \
    #         all of batch, number of Q heads, sequence length, and head size."

    if args.model:
        assert not (args.hq), "Specifying model fixes hq already. Do not provide it!"

    assert (
        args.dtype in arg_to_torch_dtype
    ), "Only fp16, bf16 and f32 types currently supported."

    if args.print_vgpr:
        print("Retrieving VGPR usage for Triton kernels...")

        def fun():
            return run_benchmark(args)

        plot_name = "MLA-decode-RoPE"
        if args.model:
            plot_name += f"-{args.model}"
        print_vgpr(fun, plot_name)
        return 0

    print(args)

    run_benchmark(args)


if __name__ == "__main__":
    import sys

    sys.exit(main())
