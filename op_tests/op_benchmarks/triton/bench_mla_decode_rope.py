import triton
from op_tests.op_benchmarks.triton.utils.benchmark_utils import (
    get_model_configs,
    print_vgpr,
)
import torch
import sys
import warnings
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

def nonvarlen_benchmark_configs():
    batch_sizes = [1, 4, 16]
    N_HEADS = [16, 48]
    seq_len_k = [163, 8192]
    configs = list(itertools.product(batch_sizes, N_HEADS, seq_len_k))
    configs = [
        (batch_size, N_HEAD, N_HEAD, seq_len_k)
        for batch_size, N_HEAD, seq_len_k in configs
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


def create_benchmark_configs(custom: bool, args: argparse.Namespace):
    dtype = arg_to_torch_dtype[args.dtype]
    hk = args.hq if not args.hk else args.hk
    sk = args.sq if not args.sk else args.sk
    head_size = 128 if not args.d else args.d
    x_names = ["BATCH", "HQ", "HK", "N_CTX_K"]

    configs = []
    plot_name = f"MLA-decode-RoPE-Latent-Dim-{head_size}"
    extra_args = {
        "D_HEAD": head_size,
        "dtype": dtype,
    }

    if custom:
        x_vals_list = [(args.b, args.hq, hk, args.sq, sk)]
    else:
        if args.model:
            x_vals_list = model_benchmark_configs(args)
            x_names = ["BATCH", "HQ", "HK", "N_CTX_K", "D_HEAD"]
            plot_name += f"-{args.model}"
            extra_args = {"dtype": dtype}
        else:
            x_vals_list = nonvarlen_benchmark_configs()

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


def run_benchmark(custom: bool, args: argparse.Namespace):
    torch.manual_seed(20)

    @triton.testing.perf_report(create_benchmark_configs(custom, args))
    def bench_mla(
        BATCH: int,
        H: int,
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
    parser.add_argument("-hq", type=int, default=0)
    parser.add_argument("-hk", type=int, default=0)
    parser.add_argument("-sk", type=int, default=0)
    parser.add_argument("-use-rope", action="store_true", default=False,
                        help="Enable rotary positional embeddings.")
    parser.add_argument("-d", type=int, default=0)
    parser.add_argument("--dtype", default="fp16")
    parser.add_argument("-print_vgpr", action="store_true", default=False)
    parser.add_argument(
        "-return_all",
        action="store_true",
        default=False,
        help="Prints TFLOPS, walltime, bandwidth.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    custom_config = False

    if args.hq or args.hk or args.sk or args.d:
        custom_config = True
        assert (
            args.b and args.hq and args.sk and args.d
        ), "If custom config is specified, please provide \
                all of batch, number of Q heads, sequence length, and head size."

    if args.model:
        assert not (
            args.hq or args.hk or args.d
        ), "Specifying model fixes hq, hk and d already. Do not provide them!"

    assert (
        args.dtype in arg_to_torch_dtype
    ), "Only fp16, bf16 and f32 types currently supported."

    if args.print_vgpr:
        assert not args.bench_torch, "Do not use -bench_torch with -print_vgpr."
        print("Retrieving VGPR usage for Triton kernels...")

        def fun():
            return run_benchmark(custom_config, args)

        print_vgpr(fun, "fused-attention")
        return 0
    
    print(args)

    run_benchmark(custom_config, args)


if __name__ == "__main__":
    import sys

    sys.exit(main())
