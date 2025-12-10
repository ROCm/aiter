"""Batch Prefill with Paged KV Cache Benchmark.

This benchmark measures the performance of the batch prefill operation with paged KV cache,
which is used in LLM inference for efficient memory management.

Usage:

    Custom Configuration:
    Run with custom parameters:

    python bench_batch_prefill.py -b 4 -hq 32 -hk 8 -sq 1024 -sk 1024 -d 128 -dtype fp16

    Model-Based Configuration:
    Run using predefined model configurations:

    python bench_batch_prefill.py --model llama3-8B -b 1 -dtype fp16 -causal

    Available models: llama3-8B, llama3-70B, llama3-405B, mixtral-7B, mixtral-22B, deepseek-V3

Options:
    -b: Batch size (number of sequences)
    -hq: Number of query heads
    -hk: Number of key/value heads (for GQA/MQA)
    -sq: Query/output sequence length
    -sk: Key/value sequence length
    -d: Head dimension
    -dtype: Data type (fp16 or bf16 - fp32 not supported)
    -page_size: Page size for paged attention (default: 1)
    -causal: Enable causal attention mask
    -logits_soft_cap: Enable logits soft capping with specified value
    -equal_seqlens: Use equal sequence lengths for all sequences
    -o: Save results to CSV file

Examples:

    Basic benchmark with causal attention:
    python bench_batch_prefill.py -b 2 -hq 16 -hk 4 -sq 512 -sk 512 -d 128 -dtype fp16 -causal

    Benchmark with logits soft cap:
    python bench_batch_prefill.py -b 4 -hq 32 -hk 8 -sq 1024 -sk 1024 -d 128 -dtype bf16 -logits_soft_cap 30.0

    Save results to CSV:
    python bench_batch_prefill.py --model llama3-8B -dtype fp16 -causal -o

Metrics:
    The benchmark reports three metrics for each configuration:
    1. Time (ms): Execution time in milliseconds
    2. TFLOPS: Throughput in teraFLOPS
    3. Bandwidth (GB/s): Memory bandwidth utilization

Notes:
    - This benchmark uses the CK (Composable Kernel) backend, not Triton kernels
    - Currently optimized for page_size=1 which is the most tested configuration
    - Supports both MHA (Multi-Head Attention), GQA (Grouped Query Attention), and MQA (Multi-Query Attention)
    - Causal attention is commonly used in autoregressive language model inference
    - Only fp16 and bf16 data types are supported (fp32 is not supported by the kernel)
    - VGPR analysis is not applicable since this uses CK kernels, not Triton kernels
"""

import sys
import os
import argparse
import itertools

import torch
import aiter
from op_tests.op_benchmarks.triton.utils.benchmark_utils import (
    get_model_configs,
    get_available_models,
    get_dtype_bytes,
    get_caller_name_no_ext,
)

import triton

# Suppress verbose aiter logging
os.environ.setdefault("AITER_LOG_LEVEL", "ERROR")


def model_benchmark_configs(args):
    """Generate benchmark configs based on model configurations."""
    config_file = args.model_configs
    configs = get_model_configs(
        config_path=config_file,
        models="llama3,deepseek" if args.model is None else args.model,
    )
    batch_prefill_configs = []
    batch_size = args.b if args.b else 1

    for model_name, config in configs.items():
        num_qo_heads = config["num_attention_heads"]
        num_kv_heads = (
            num_qo_heads
            if config["num_key_value_heads"] is None
            else config["num_key_value_heads"]
        )
        head_dim = config["hidden_size"] // num_qo_heads

        # Use provided seq lens or default values
        qo_len = args.sq if args.sq else 2048
        kv_len = args.sk if args.sk else qo_len

        batch_prefill_configs.append(
            (
                model_name,
                batch_size,
                qo_len,
                kv_len,
                num_qo_heads,
                num_kv_heads,
                head_dim,
            )
        )

    return batch_prefill_configs


def custom_benchmark_configs(args):
    """Generate custom benchmark configs from command-line arguments."""
    batch_sizes = [1, 4, 8] if not args.b else [args.b]
    qo_lens = [1024, 2048, 4096, 8192] if not args.sq else [args.sq]
    kv_lens = qo_lens if not args.sk else [args.sk]
    num_qo_heads = args.hq if args.hq else 32
    num_kv_heads = args.hk if args.hk else (num_qo_heads // 4)
    head_dim = args.d if args.d else 128

    configs = []
    for batch_size, qo_len, kv_len in itertools.product(batch_sizes, qo_lens, kv_lens):
        configs.append(
            ("custom", batch_size, qo_len, kv_len, num_qo_heads, num_kv_heads, head_dim)
        )
    return configs


def convert_lens_to_indptr(lens):
    """Convert sequence lengths to indptr format."""
    return torch.cumsum(torch.cat((torch.tensor([0]), lens)), dim=0).int()


def run_benchmark(args):
    dtype = arg_to_torch_dtype[args.dtype]
    # Suppress repetitive warnings from aiter framework
    import logging

    logging.getLogger("aiter").setLevel(logging.ERROR)

    # Determine benchmark configurations
    if args.model:
        x_vals_list = model_benchmark_configs(args)
        x_names = ["model", "BATCH", "QO_LEN", "KV_LEN", "HQ", "HK", "HEAD_DIM"]
    else:
        x_vals_list = custom_benchmark_configs(args)
        x_names = ["config", "BATCH", "QO_LEN", "KV_LEN", "HQ", "HK", "HEAD_DIM"]

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
    def bench_batch_prefill(BATCH, QO_LEN, KV_LEN, HQ, HK, HEAD_DIM, metric, **kwargs):
        """
        Benchmark function for batch prefill with paged KV cache.
        """
        causal = args.causal
        logits_soft_cap = args.logits_soft_cap if args.logits_soft_cap else 0.0
        page_size = args.page_size if args.page_size else 1

        # Generate sequence lengths first
        if args.equal_seqlens:
            qo_lens = torch.full((BATCH,), QO_LEN, dtype=torch.int32)
            kv_lens = torch.full((BATCH,), KV_LEN, dtype=torch.int32)
            actual_qo_len = QO_LEN
            actual_kv_len = KV_LEN
        else:
            # Randomize lengths for more realistic benchmark
            qo_lens = torch.randint(
                max(1, QO_LEN // 2), QO_LEN + 1, (BATCH,), dtype=torch.int32
            )
            kv_lens = torch.randint(
                max(1, KV_LEN // 2), KV_LEN + 1, (BATCH,), dtype=torch.int32
            )
            # Store average lengths for reporting
            actual_qo_len = int(qo_lens.float().mean().item())
            actual_kv_len = int(kv_lens.float().mean().item())

        # Create q tensor with correct total length
        total_q_tokens = qo_lens.sum().item()
        q = torch.randn(total_q_tokens, HQ, HEAD_DIM, device="cuda", dtype=dtype)

        # Calculate number of pages needed based on actual max lengths
        max_num_pages_per_seq = (kv_lens.max().item() + page_size - 1) // page_size
        total_num_pages = max_num_pages_per_seq * BATCH

        # Create paged KV cache
        # Note: page_size must be 1 for current implementation
        # Shape: [total_num_pages, 2, num_kv_heads, page_size, head_dim]
        kv_data = torch.randn(
            total_num_pages, 2, HK, page_size, HEAD_DIM, device="cuda", dtype=dtype
        )
        # Split into k_cache and v_cache
        # Expected final shape: [num_blocks, num_heads_k, head_size] when page_size=1
        chunks = torch.chunk(kv_data, 2, dim=1)
        # Use squeeze without args to remove all size-1 dimensions, then reshape as needed
        k_cache = chunks[0].squeeze()
        v_cache = chunks[1].squeeze()
        # Ensure correct shape for the kernel: [num_blocks, num_heads_k, head_size]
        if page_size == 1:
            if k_cache.dim() == 2:  # [num_blocks, head_size] when HK=1
                k_cache = k_cache.unsqueeze(1)  # [num_blocks, 1, head_size]
                v_cache = v_cache.unsqueeze(1)
        else:
            # For page_size > 1, reshape to flatten page dimension into the block dimension
            # [total_num_pages, HK, page_size, HEAD_DIM] -> [total_num_pages * page_size, HK, HEAD_DIM]
            k_cache = k_cache.reshape(total_num_pages * page_size, HK, HEAD_DIM)
            v_cache = v_cache.reshape(total_num_pages * page_size, HK, HEAD_DIM)

        q_indptr = convert_lens_to_indptr(qo_lens).to("cuda")
        kv_num_used_pages = (kv_lens + page_size - 1) // page_size
        kv_indptr = convert_lens_to_indptr(kv_num_used_pages).to("cuda")

        # Generate random page indices
        kv_page_indices = torch.randperm(total_num_pages, dtype=torch.int32)[
            :total_num_pages
        ].to("cuda")

        # Softmax scale
        softmax_scale = HEAD_DIM**-0.5

        # Define the kernel function
        def fn():
            return aiter.mha_batch_prefill_func(
                q,
                k_cache,
                v_cache,
                q_indptr,
                kv_indptr,
                kv_page_indices,
                torch.max(qo_lens).item(),
                torch.max(kv_lens).item(),
                dropout_p=0.0,
                softmax_scale=softmax_scale,
                logits_soft_cap=logits_soft_cap,
                causal=causal,
            )

        # Benchmark the kernel
        ms = triton.testing.do_bench(fn, warmup=25, rep=100)

        # Calculate FLOPs
        # For each batch: 2 * qo_len * kv_len * num_qo_heads * (2 * head_dim)
        # The factor of 2 comes from: QK^T matmul + softmax(QK^T)V matmul
        total_flops = 0.0
        for i in range(BATCH):
            q_len = qo_lens[i].item()
            k_len = kv_lens[i].item()

            if causal:
                # For causal attention, calculate valid elements in the causal mask
                if q_len > k_len:
                    valid_out_elements = (k_len**2 + k_len) / 2
                else:
                    valid_out_elements = q_len * k_len - ((q_len**2 - q_len) / 2)
            else:
                valid_out_elements = q_len * k_len

            # QK^T and softmax(QK^T)V: 2 operations, each with 2*valid_elements*head_dim FLOPs per head
            total_flops += valid_out_elements * HQ * HEAD_DIM * 2.0 * 2

        # Calculate memory traffic using actual lengths
        total_q_tokens = qo_lens.sum().item()
        total_kv_tokens = kv_lens.sum().item()

        q_size = total_q_tokens * HQ * HEAD_DIM * get_dtype_bytes(dtype)
        k_size = total_kv_tokens * HK * HEAD_DIM * get_dtype_bytes(dtype)
        v_size = total_kv_tokens * HK * HEAD_DIM * get_dtype_bytes(dtype)
        o_size = total_q_tokens * HQ * HEAD_DIM * get_dtype_bytes(dtype)
        indptr_size = (BATCH + 1) * 4 * 2  # q_indptr and kv_indptr
        page_indices_size = total_num_pages * 4

        # Read: q, k_cache, v_cache, indptrs, page_indices
        mem_read = q_size + k_size + v_size + indptr_size + page_indices_size
        # Write: output
        mem_write = o_size
        mem = mem_read + mem_write

        # Calculate metrics
        bandwidth = mem / ms * 1e-6  # GB/s
        tflops = total_flops / ms * 1e-9  # TFLOPS

        # Print actual lengths when using randomized lengths
        if not args.equal_seqlens and metric == "time":
            print(
                f"  [Note: QO_LEN={QO_LEN}, KV_LEN={KV_LEN} are max constraints. "
                f"Actual randomized - Avg QO: {actual_qo_len}, Avg KV: {actual_kv_len}, "
                f"Max used: {qo_lens.max().item()}/{kv_lens.max().item()}]"
            )

        # Return the requested metric
        if metric == "time":
            return ms
        elif metric == "tflops":
            return tflops
        elif metric == "bandwidth":
            return bandwidth
        else:
            raise ValueError("Unknown metric: " + metric)

    # Print configuration information
    if not args.equal_seqlens:
        print("=" * 70)
        print("RANDOMIZED MODE: Sequence lengths will vary between [max/2, max]")
        print("Table shows max constraints; actual values printed below.")
        print("=" * 70)

    bench_batch_prefill.run(save_path="." if args.o else None, print_data=True)


def parse_args():
    parser = argparse.ArgumentParser(
        prog="Benchmark Batch Prefill with Paged KV Cache",
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
    parser.add_argument("-b", type=int, default=0, help="Batch size")
    parser.add_argument("-hq", type=int, default=0, help="Number of query heads")
    parser.add_argument("-hk", type=int, default=0, help="Number of key/value heads")
    parser.add_argument("-sq", type=int, default=0, help="Query/output sequence length")
    parser.add_argument("-sk", type=int, default=0, help="Key/value sequence length")
    parser.add_argument("-d", type=int, default=0, help="Head dimension")
    parser.add_argument(
        "-dtype",
        default="fp16",
        choices=["fp16", "bf16"],
        help="Data type (fp16 or bf16)",
    )
    parser.add_argument(
        "-page_size", type=int, default=1, help="Page size for paged attention"
    )
    parser.add_argument(
        "-causal",
        action="store_true",
        default=False,
        help="Use causal attention mask",
    )
    parser.add_argument(
        "-logits_soft_cap",
        type=float,
        default=0.0,
        help="Logits soft cap value (0.0 for no cap)",
    )
    parser.add_argument(
        "--no-equal_seqlens",
        dest="equal_seqlens",
        action="store_false",
        default=True,
        help="Use randomized sequence lengths (default uses equal lengths for all sequences)",
    )
    parser.add_argument(
        "-o", action="store_true", help="Write performance results to CSV file"
    )
    args = parser.parse_args()
    return args


arg_to_torch_dtype = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


def main():
    args = parse_args()

    # Validate arguments
    if not args.model and args.hq == 0:
        print("Error: Must specify either --model or provide custom config with -hq")
        return 1

    # Note: VGPR printing not supported - this benchmark uses CK (Composable Kernel) backend,
    # not Triton. VGPR analysis only applies to Triton kernels.

    run_benchmark(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
