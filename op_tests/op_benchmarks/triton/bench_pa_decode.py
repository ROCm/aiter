import sys
import random
import torch
import argparse
import triton
from aiter.ops.triton.pa_decode import paged_attention_decode
from op_tests.op_benchmarks.triton.utils.benchmark_utils import (
    get_model_configs,
    get_available_models,
    get_dtype_bytes,
    get_caller_name_no_ext,
    print_vgpr,
)
from aiter.ops.triton.utils.types import torch_to_triton_dtype


def input_helper(
    B,
    H_Q,
    H_KV,
    D,
    KV_BLK_SZ,
    SEQ_LEN,
    dtype,
    kv_cache_dtype,
    output_type,
    num_blocks=4,
):
    """Helper function to generate input tensors for paged attention testing."""
    # Query tensor generation
    if dtype not in (torch.bfloat16, torch.float16, torch.float32):
        query = torch.randn(
            B, H_Q, D, dtype=torch.float16, device="cuda"
        )  # assumption dtype is 8bits or lower
        query = query.to(dtype=dtype, device="cuda")
    else:
        query = torch.randn(B, H_Q, D, dtype=dtype, device="cuda")

    if kv_cache_dtype not in (torch.bfloat16, torch.float16, torch.float32):
        x = min(D, 16 // torch.tensor([], dtype=torch.float16).element_size())
        key_cache = torch.randn(
            num_blocks, H_KV, D // x, KV_BLK_SZ, x, dtype=torch.float16, device="cuda"
        )
        value_cache = torch.randn(
            num_blocks, H_KV, D, KV_BLK_SZ, dtype=torch.float16, device="cuda"
        )
        key_cache = torch.clamp(
            key_cache, min=1e-3
        )  # For FP8 case, this is needed to prevent NANs
        value_cache = torch.clamp(
            value_cache, min=1e-3
        )  # For FP8 case, this is needed to prevent NANs

        # torch doesn't have randn for fp8 data type, so we convert here
        key_cache = key_cache.to(dtype=kv_cache_dtype)
        value_cache = value_cache.to(dtype=kv_cache_dtype)
    else:
        x = min(D, 16 // torch.tensor([], dtype=kv_cache_dtype).element_size())
        key_cache = torch.randn(
            num_blocks, H_KV, D // x, KV_BLK_SZ, x, dtype=kv_cache_dtype, device="cuda"
        )
        value_cache = torch.randn(
            num_blocks, H_KV, D, KV_BLK_SZ, dtype=kv_cache_dtype, device="cuda"
        )
        key_cache = torch.clamp(
            key_cache, min=1e-3
        )  # For FP8 case, this is needed to prevent NANs
        value_cache = torch.clamp(
            value_cache, min=1e-3
        )  # For FP8 case, this is needed to prevent NANs

    key_cache_tri = key_cache.permute(0, 1, 3, 2, 4).flatten(3, 4).contiguous().cuda()
    value_cache_tri = value_cache.permute(0, 1, 3, 2).contiguous().cuda()

    context_lens = torch.full((B,), SEQ_LEN, device="cuda")
    max_context_len = max(context_lens)
    max_num_blks_per_seq = (max_context_len + KV_BLK_SZ - 1) // KV_BLK_SZ

    block_tables = []
    for i in range(B):
        block_table = [
            random.randint(0, num_blocks - 1) for _ in range(max_num_blks_per_seq)
        ]
        block_tables.append(block_table)
    block_tables = torch.tensor(block_tables, dtype=torch.int32, device="cuda")

    output = torch.zeros(B, H_Q, D, dtype=output_type, device="cuda")

    return (
        query,
        output,
        key_cache,
        value_cache,
        key_cache_tri,
        value_cache_tri,
        context_lens,
        block_tables,
        max_context_len,
    )


def model_benchmark_configs(args):
    config_file = args.model_configs
    configs = get_model_configs(
        config_path=config_file,
        models="llama3,deepseek" if args.model is None else args.model,
    )
    fa_configs = []
    BS = args.b if args.b else 1024

    for model_name, config in configs.items():
        HQ = config["num_attention_heads"]
        HK = (
            HQ
            if config["num_key_value_heads"] is None
            else config["num_key_value_heads"]
        )
        SEQ_LEN = args.sq if args.sq else 8192
        HEAD_DIM = config["hidden_size"] // HQ
        fa_configs.append((model_name, BS, HQ, HK, SEQ_LEN, HEAD_DIM))

    return fa_configs


def paged_attn_decode(
    BS,
    H_Q,
    H_KV,
    D,
    KV_BLK_SZ,
    SEQ_LEN,
    num_blocks,
    dtype,
    kv_cache_dtype,
    compute_type,
    output_type,
):
    (
        query,
        triton_output,
        _,
        _,
        key_cache_tri,
        value_cache_tri,
        context_lens,
        block_tables,
        max_context_len,
    ) = input_helper(
        BS,
        H_Q,
        H_KV,
        D,
        KV_BLK_SZ,
        SEQ_LEN,
        dtype,
        kv_cache_dtype,
        output_type,
        num_blocks,
    )
    attn_scale = 1.0 / (D**0.5)
    k_scale = torch.tensor([1.0])
    v_scale = torch.tensor([1.0])

    return lambda: paged_attention_decode(  # noqa: E731
        output=triton_output,
        query=query,
        key_cache=key_cache_tri,
        value_cache=value_cache_tri,
        seq_lens=context_lens,
        block_tables=block_tables,
        attn_scale=attn_scale,
        max_seq_len=max_context_len,
        compute_type=compute_type,
        k_scale=k_scale,
        v_scale=v_scale,
    )


def run_benchmark(args):
    dtype = arg_to_torch_dtype[args.dtype]
    kv_cache_dtype = arg_to_torch_dtype[args.kv_cache_dtype]
    compute_type = torch_to_triton_dtype[arg_to_torch_dtype[args.compute_type]]
    output_type = arg_to_torch_dtype[args.output_type]

    x_vals_list = model_benchmark_configs(args)
    x_names = ["model", "BS", "HQ", "HK", "SEQ_LEN", "HEAD_DIM"]

    model_name = "paged-attn-decode"

    line_names = ["Time_(ms)", "TFLOPS", "Bandwidth_(GB/s)"]
    line_vals = ["time", "tflops", "bandwidth"]

    benchmark = triton.testing.Benchmark(
        x_names=x_names,
        x_vals=x_vals_list,
        line_arg="metric",
        line_vals=line_vals,
        line_names=line_names,
        styles=[("red", "-"), ("blue", "-"), ("yellow", "-")],
        ylabel="ms / TFLOPS / GB/s",
        plot_name=get_caller_name_no_ext(),
        args={},
    )

    @triton.testing.perf_report([benchmark])
    def bench_paged_attn_decode(BS, HQ, HK, SEQ_LEN, HEAD_DIM, metric, model=None):
        # TODO tune this
        KV_BLK_SZ = 128
        num_blocks = 4
        fn = paged_attn_decode(
            BS,
            HQ,
            HK,
            HEAD_DIM,
            KV_BLK_SZ,
            SEQ_LEN,
            num_blocks,
            dtype,
            kv_cache_dtype,
            compute_type,
            output_type,
        )

        ms = triton.testing.do_bench(fn, warmup=25, rep=100)

        # query and output
        mem = (BS * HQ * HEAD_DIM) * (
            get_dtype_bytes(dtype) + get_dtype_bytes(output_type)
        )
        # kv_cache
        mem += (
            num_blocks * HK * KV_BLK_SZ * HEAD_DIM * get_dtype_bytes(kv_cache_dtype) * 2
        )
        # block_tables int32
        mem += BS * ((SEQ_LEN + KV_BLK_SZ - 1) // KV_BLK_SZ) * 4
        # context_lens fp32
        mem += BS * 4

        # bhd bhsd => bhs bhsd => bhs, 2 for multiplication and accumulation. and there are 2 gemms
        flops = (2.0 * BS * HQ * SEQ_LEN * HEAD_DIM) * 2

        bandwidth = mem / (ms * 1e-3) * 1e-9  # GB/s
        tflops = flops / ms * 1e-9

        # Return exactly one scalar depending on which metric is active
        if metric == "time":
            return ms
        elif metric == "tflops":
            return tflops
        elif metric == "bandwidth":
            return bandwidth
        else:
            raise ValueError("Unknown metric: " + metric)

    bench_paged_attn_decode.run(save_path="." if args.o else None, print_data=True)


def parse_args():
    parser = argparse.ArgumentParser(
        prog="Benchmark Paged Attention decode",
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
        + "]. Use 'all' to benchmark all models or leave blank for the default benchmark script."
    )
    parser.add_argument("--model", type=str, default=None, help=model_help)
    parser.add_argument("-b", type=int, default=0)
    parser.add_argument("-hq", type=int, default=0)
    parser.add_argument("-hk", type=int, default=0)
    parser.add_argument("-sq", type=int, default=0)
    parser.add_argument("-dtype", default="fp16")
    parser.add_argument("-kv_cache_dtype", default="fp16")
    parser.add_argument("-compute_type", default="fp16")
    parser.add_argument("-output_type", default="fp16")
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
    "e5m2fnuz": torch.float8_e5m2fnuz,
    "e4m3fnuz": torch.float8_e4m3fnuz,
}


def main():
    args = parse_args()
    if args.print_vgpr:
        print("Retrieving VGPR usage for Triton kernels...")
        fun = lambda: run_benchmark(args)  # noqa: E731
        print_vgpr(fun, get_caller_name_no_ext())
        return 0
    run_benchmark(args)
    
if __name__ == "__main__":
    sys.exit(main())
