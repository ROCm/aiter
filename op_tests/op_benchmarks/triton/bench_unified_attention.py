import sys
import random
import torch
import argparse
import triton
from typing import Optional
from aiter.ops.triton.unified_attention import unified_attention
from aiter.ops.triton.utils.types import e4m3_dtype
from op_tests.op_benchmarks.triton.utils.benchmark_utils import (
    get_model_configs,
    get_available_models,
    get_dtype_bytes,
    get_caller_name_no_ext,
    print_vgpr,
)


def input_helper(
    num_seqs,
    query_lens,
    kv_lens,
    num_query_heads,
    num_kv_heads,
    head_size,
    block_size,
    num_blocks,
    dtype,
    q_dtype=None,
):
    """Helper function to generate input tensors for unified attention testing."""
    max_query_len = max(query_lens)
    max_kv_len = max(kv_lens)

    # Generate query tensor
    query = torch.randn(
        sum(query_lens), num_query_heads, head_size, dtype=dtype, device="cuda"
    )

    # Generate key and value caches
    key_cache = torch.randn(
        num_blocks, block_size, num_kv_heads, head_size, dtype=dtype, device="cuda"
    )
    value_cache = torch.randn_like(key_cache)

    # Optionally quantize to fp8
    maybe_quantized_query = query
    maybe_quantized_key_cache = key_cache
    maybe_quantized_value_cache = value_cache
    q_descale = None
    k_descale = None
    v_descale = None

    if q_dtype is not None:
        maybe_quantized_query = query.to(q_dtype)
        maybe_quantized_key_cache = key_cache.to(q_dtype)
        maybe_quantized_value_cache = value_cache.to(q_dtype)

        scale_shape = (num_seqs, num_kv_heads)
        k_descale = torch.rand(scale_shape, dtype=torch.float32, device="cuda")
        v_descale = torch.rand(scale_shape, dtype=torch.float32, device="cuda")

    # Generate cumulative query lengths
    cu_query_lens = torch.tensor(
        [0] + query_lens, dtype=torch.int32, device="cuda"
    ).cumsum(dim=0, dtype=torch.int32)

    # Generate kv lengths tensor
    kv_lens_tensor = torch.tensor(kv_lens, dtype=torch.int32, device="cuda")

    # Generate block tables
    max_num_blocks_per_seq = (max_kv_len + block_size - 1) // block_size
    block_tables = torch.randint(
        0,
        num_blocks,
        (num_seqs, max_num_blocks_per_seq),
        dtype=torch.int32,
        device="cuda",
    )

    # Generate sinks
    sinks = torch.randn(num_query_heads, dtype=torch.bfloat16, device="cuda")

    # Output tensor
    output = torch.empty_like(query)

    return (
        maybe_quantized_query,
        maybe_quantized_key_cache,
        maybe_quantized_value_cache,
        output,
        cu_query_lens,
        kv_lens_tensor,
        max_query_len,
        max_kv_len,
        block_tables,
        sinks,
        q_descale,
        k_descale,
        v_descale,
    )


def model_benchmark_configs(args):
    """Generate benchmark configs based on model configurations."""
    config_file = args.model_configs
    configs = get_model_configs(
        config_path=config_file,
        models="llama3,deepseek" if args.model is None else args.model,
    )
    ua_configs = []

    for model_name, config in configs.items():
        num_query_heads = config["num_attention_heads"]
        num_kv_heads = (
            num_query_heads
            if config["num_key_value_heads"] is None
            else config["num_key_value_heads"]
        )
        head_size = config["hidden_size"] // num_query_heads

        # Use provided seq lens or default values
        if args.seq_pattern == "decode":
            # Decode: batch of single-token queries with varying context lengths
            num_seqs = args.b if args.b else 16
            query_lens = [1] * num_seqs
            if args.sk:
                kv_lens = [args.sk] * num_seqs
            else:
                # Vary context lengths for more realistic benchmarking
                kv_lens = [random.randint(512, 2048) for _ in range(num_seqs)]
        elif args.seq_pattern == "prefill":
            # Prefill: single sequence with longer query length
            num_seqs = args.b if args.b else 1
            query_lens = [args.sq if args.sq else 1024] * num_seqs
            kv_lens = query_lens.copy()
        else:  # mixed
            # Mixed: some decode, some prefill
            num_seqs = args.b if args.b else 8
            decode_seqs = num_seqs // 2
            prefill_seqs = num_seqs - decode_seqs
            query_lens = [1] * decode_seqs + [args.sq if args.sq else 256] * prefill_seqs
            kv_lens = (
                [random.randint(512, 2048) for _ in range(decode_seqs)]
                + query_lens[decode_seqs:]
            )

        ua_configs.append(
            (model_name, num_seqs, tuple(query_lens), tuple(kv_lens),
             num_query_heads, num_kv_heads, head_size)
        )

    return ua_configs


def custom_benchmark_configs(args):
    """Generate custom benchmark configs from command-line arguments."""
    num_query_heads = args.hq
    num_kv_heads = args.hk if args.hk else num_query_heads
    head_size = args.d if args.d else 128

    if args.seq_pattern == "decode":
        num_seqs = args.b if args.b else 16
        query_lens = [1] * num_seqs
        kv_lens = [args.sk if args.sk else 1024] * num_seqs
    elif args.seq_pattern == "prefill":
        num_seqs = args.b if args.b else 1
        query_lens = [args.sq if args.sq else 1024] * num_seqs
        kv_lens = query_lens.copy()
    else:  # mixed
        num_seqs = args.b if args.b else 8
        decode_seqs = num_seqs // 2
        prefill_seqs = num_seqs - decode_seqs
        query_lens = [1] * decode_seqs + [args.sq if args.sq else 256] * prefill_seqs
        kv_lens = [args.sk if args.sk else 1024] * decode_seqs + query_lens[decode_seqs:]

    return [
        ("custom", num_seqs, tuple(query_lens), tuple(kv_lens),
         num_query_heads, num_kv_heads, head_size)
    ]


def run_benchmark(args):
    dtype = arg_to_torch_dtype[args.dtype]
    q_dtype = e4m3_dtype if args.fp8 else None

    # Determine benchmark configurations
    if args.model:
        x_vals_list = model_benchmark_configs(args)
        x_names = ["model", "num_seqs", "query_lens", "kv_lens", "HQ", "HK", "HEAD_DIM"]
    else:
        x_vals_list = custom_benchmark_configs(args)
        x_names = ["config", "num_seqs", "query_lens", "kv_lens", "HQ", "HK", "HEAD_DIM"]

    line_names = ["Time_(ms)", "TFLOPS", "Bandwidth_(GB/s)"]
    line_vals = ["time", "tflops", "bandwidth"]

    benchmark = triton.testing.Benchmark(
        x_names=x_names,
        x_vals=x_vals_list,
        line_arg="metric",
        line_vals=line_vals,
        line_names=line_names,
        styles=[("red", "-"), ("blue", "-"), ("green", "-")],
        ylabel="ms / TFLOPS / GB/s",
        plot_name=get_caller_name_no_ext(),
        args={},
    )

    @triton.testing.perf_report([benchmark])
    def bench_unified_attention(
        num_seqs, query_lens, kv_lens, HQ, HK, HEAD_DIM, metric, model=None, config=None
    ):
        block_size = args.block_size if args.block_size else 16
        num_blocks = args.num_blocks if args.num_blocks else 8192
        sliding_window = args.sliding_window
        soft_cap = args.soft_cap

        # Convert tuples back to lists
        query_lens = list(query_lens)
        kv_lens = list(kv_lens)

        # Generate inputs
        (
            query,
            key_cache,
            value_cache,
            output,
            cu_query_lens,
            kv_lens_tensor,
            max_query_len,
            max_kv_len,
            block_tables,
            sinks,
            q_descale,
            k_descale,
            v_descale,
        ) = input_helper(
            num_seqs,
            query_lens,
            kv_lens,
            HQ,
            HK,
            HEAD_DIM,
            block_size,
            num_blocks,
            dtype,
            q_dtype,
        )

        # Softmax scale
        scale = HEAD_DIM**-0.5
        window_size = (sliding_window - 1, 0) if sliding_window is not None else (-1, -1)

        # Define the kernel function
        def fn():
            unified_attention(
                q=query,
                k=key_cache,
                v=value_cache,
                out=output,
                cu_seqlens_q=cu_query_lens,
                seqused_k=kv_lens_tensor,
                max_seqlen_q=max_query_len,
                max_seqlen_k=max_kv_len,
                softmax_scale=scale,
                causal=True,
                window_size=window_size,
                block_table=block_tables,
                softcap=soft_cap if soft_cap is not None else 0,
                q_descale=q_descale,
                k_descale=k_descale,
                v_descale=v_descale,
                sinks=sinks if args.use_sinks else None,
            )

        # Benchmark the kernel
        ms = triton.testing.do_bench(fn, warmup=25, rep=100)

        # Calculate FLOPs
        # For each sequence: 2 * query_len * kv_len * num_query_heads * (2 * head_size)
        # The factor of 2 comes from: QK^T matmul + softmax(QK^T)V matmul
        # Each matmul is roughly 2 * M * N * K FLOPs
        # For causal attention, only lower triangular elements are computed (~half)
        total_flops = 0.0
        for q_len, kv_len in zip(query_lens, kv_lens):
            # For causal attention, calculate valid elements in the causal mask
            if q_len > kv_len:
                # All kv_len tokens are visible to all queries
                valid_out_elements = (kv_len**2 + kv_len) / 2
            else:
                # Triangular mask: q_len * kv_len - upper_triangular_zeros
                valid_out_elements = q_len * kv_len - ((q_len**2 - q_len) / 2)
            
            # QK^T and softmax(QK^T)V: 2 operations, each with 2*valid_elements*head_size FLOPs per head
            total_flops += valid_out_elements * HQ * HEAD_DIM * 2.0 * 2

        # Calculate memory traffic
        q_size = sum(query_lens) * HQ * HEAD_DIM * get_dtype_bytes(dtype)
        k_size = num_blocks * block_size * HK * HEAD_DIM * get_dtype_bytes(dtype)
        v_size = num_blocks * block_size * HK * HEAD_DIM * get_dtype_bytes(dtype)
        o_size = sum(query_lens) * HQ * HEAD_DIM * get_dtype_bytes(dtype)
        block_table_size = num_seqs * ((max_kv_len + block_size - 1) // block_size) * 4

        # Read: q, k_cache, v_cache, block_tables
        mem_read = q_size + k_size + v_size + block_table_size
        # Write: output
        mem_write = o_size
        mem = mem_read + mem_write

        # Calculate metrics
        bandwidth = mem / ms * 1e-6  # GB/s
        tflops = total_flops / ms * 1e-9  # TFLOPS

        # Return the requested metric
        if metric == "time":
            return ms
        elif metric == "tflops":
            return tflops
        elif metric == "bandwidth":
            return bandwidth
        else:
            raise ValueError("Unknown metric: " + metric)

    bench_unified_attention.run(save_path="." if args.o else None, print_data=True)


def parse_args():
    parser = argparse.ArgumentParser(
        prog="Benchmark Unified Attention (Flash Attention with Paged KV Cache)",
        allow_abbrev=False,
    )
    parser.add_argument(
        "-model_configs",
        type=str,
        default="utils/model_configs.json",
        help="Model config json file.",
    )
    available_models = get_available_models()
    model_help = (
        "Model name to benchmark. Select from: ["
        + ", ".join(available_models)
        + "]. Use 'all' to benchmark all models or leave blank for custom config."
    )
    parser.add_argument("--model", type=str, default=None, help=model_help)
    parser.add_argument("-b", type=int, default=0, help="Number of sequences (batch size)")
    parser.add_argument("-hq", type=int, default=0, help="Number of query heads")
    parser.add_argument("-hk", type=int, default=0, help="Number of key/value heads")
    parser.add_argument("-sq", type=int, default=0, help="Query sequence length")
    parser.add_argument("-sk", type=int, default=0, help="Key/value sequence length")
    parser.add_argument("-d", type=int, default=0, help="Head dimension")
    parser.add_argument("-dtype", default="fp16", help="Data type (fp16, bf16, fp32)")
    parser.add_argument("-fp8", action="store_true", default=False, help="Use FP8 quantization")
    parser.add_argument(
        "-block_size", type=int, default=16, help="Block size for paged attention"
    )
    parser.add_argument(
        "-num_blocks", type=int, default=8192, help="Total number of blocks"
    )
    parser.add_argument(
        "-sliding_window", type=int, default=None, help="Sliding window size"
    )
    parser.add_argument(
        "-soft_cap", type=float, default=None, help="Soft capping value"
    )
    parser.add_argument(
        "-seq_pattern",
        type=str,
        default="decode",
        choices=["decode", "prefill", "mixed"],
        help="Sequence pattern: decode (batch of single tokens), prefill (long sequences), or mixed",
    )
    parser.add_argument(
        "-use_sinks", action="store_true", default=False, help="Use sink tokens"
    )
    parser.add_argument(
        "-o", action="store_true", help="Write performance results to CSV file"
    )
    parser.add_argument(
        "-print_vgpr",
        action="store_true",
        default=False,
        help="Print VGPR usage for Triton kernels.",
    )
    args = parser.parse_args()
    return args


arg_to_torch_dtype = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
}


def main():
    args = parse_args()

    # Validate arguments
    if not args.model and args.hq == 0:
        print("Error: Must specify either --model or provide custom config with -hq")
        return 1

    if args.print_vgpr:
        print("Retrieving VGPR usage for Triton kernels...")
        fun = lambda: run_benchmark(args)
        print_vgpr(fun, get_caller_name_no_ext())
        return 0

    run_benchmark(args)


if __name__ == "__main__":
    sys.exit(main())
