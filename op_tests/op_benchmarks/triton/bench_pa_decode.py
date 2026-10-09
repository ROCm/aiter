import argparse
import sys

import torch
import triton

from aiter.ops.triton.attention.pa_decode import paged_attention_decode
from aiter.ops.triton.utils.types import torch_to_triton_dtype
from op_tests.op_benchmarks.triton.utils.benchmark_utils import (
    get_available_models,
    get_caller_name_no_ext,
    get_dtype_bytes,
    get_model_configs,
    print_vgpr,
)


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
    *,
    backend,
    value_head_size=None,
    query_length=1,
    value_transposed=False,
):
    """Generate distinct full-context pages in the selected backend's layout."""
    if backend not in ("triton", "gluon"):
        raise ValueError(f"Unsupported backend: {backend}")
    max_num_blks_per_seq = triton.cdiv(SEQ_LEN, KV_BLK_SZ)
    num_blocks = B * max_num_blks_per_seq
    D_V = D if value_head_size is None else value_head_size
    # Query tensor generation
    if dtype not in (torch.bfloat16, torch.float16, torch.float32):
        query = torch.randn(
            B * query_length, H_Q, D, dtype=torch.float16, device="cuda"
        )  # assumption dtype is 8bits or lower
        query = query.to(dtype=dtype, device="cuda")
    else:
        query = torch.randn(B * query_length, H_Q, D, dtype=dtype, device="cuda")

    def make_cache(shape):
        source_dtype = (
            kv_cache_dtype
            if kv_cache_dtype in (torch.bfloat16, torch.float16, torch.float32)
            else torch.float16
        )
        cache = torch.randn(shape, dtype=source_dtype, device="cuda")
        # Preserve the existing FP8 input convention without another full-size copy.
        cache.clamp_(min=1e-3)
        return cache.to(kv_cache_dtype)

    if backend == "triton":
        key_cache = make_cache((num_blocks, H_KV, KV_BLK_SZ, D))
        value_cache = make_cache((num_blocks, H_KV, KV_BLK_SZ, D_V))
    else:
        x = min(D, 16 // kv_cache_dtype.itemsize)
        key_cache = make_cache((num_blocks, H_KV, D // x, KV_BLK_SZ, x))
        if value_transposed:
            value_cache = make_cache((num_blocks, H_KV, KV_BLK_SZ // x, D_V, x))
        else:
            value_cache = make_cache((num_blocks, H_KV, D_V, KV_BLK_SZ))

    context_lens = torch.full((B,), SEQ_LEN, device="cuda")
    # The Triton kernels multiply block IDs by cache strides before pointer addition.
    cache_elements = num_blocks * H_KV * KV_BLK_SZ * max(D, D_V)
    block_table_dtype = torch.int64 if cache_elements > 2**31 - 1 else torch.int32
    block_tables = torch.arange(
        num_blocks, dtype=block_table_dtype, device="cuda"
    ).view(B, max_num_blks_per_seq)

    output = torch.zeros(B * query_length, H_Q, D_V, dtype=output_type, device="cuda")

    return (
        query,
        output,
        key_cache,
        value_cache,
        context_lens,
        block_tables,
    )


def model_benchmark_configs(args):
    configs = get_model_configs(
        config_path=args.model_configs,
        models="llama3,deepseek" if args.model is None else args.model,
    )
    fa_configs = []
    BS = args.b if args.b else 1024
    SEQ_LEN = args.sq if args.sq else 8192
    if args.hq:
        HEAD_DIM = args.head_dim if args.head_dim else 128
        return [
            (
                "custom",
                BS,
                args.hq,
                args.hk or args.hq,
                SEQ_LEN,
                HEAD_DIM,
                args.value_head_dim or HEAD_DIM,
                args.query_length,
            )
        ]

    for model_name, config in configs.items():
        HQ = config["num_attention_heads"]
        HK = (
            HQ
            if config["num_key_value_heads"] is None
            else config["num_key_value_heads"]
        )
        HEAD_DIM = config["hidden_size"] // HQ
        fa_configs.append(
            (
                model_name,
                BS,
                HQ,
                HK,
                SEQ_LEN,
                HEAD_DIM,
                args.value_head_dim or HEAD_DIM,
                args.query_length,
            )
        )

    return fa_configs


def paged_attn_decode(
    BS,
    H_Q,
    H_KV,
    D,
    KV_BLK_SZ,
    SEQ_LEN,
    dtype,
    kv_cache_dtype,
    compute_type,
    output_type,
    backend="triton",
    value_head_size=None,
    query_length=1,
):
    if value_head_size is None:
        value_head_size = D
    if backend == "triton" and (value_head_size != D or query_length != 1):
        raise ValueError(
            "Independent V dimensions and query_length > 1 require --backend gluon"
        )
    (
        query,
        output,
        key_cache,
        value_cache,
        context_lens,
        block_tables,
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
        backend=backend,
        value_head_size=value_head_size,
        query_length=query_length,
        value_transposed=backend == "gluon" and KV_BLK_SZ == 64,
    )
    attn_scale = 1.0 / (D**0.5)

    if backend == "gluon":
        from aiter.ops.triton.gluon.pa_decode_gluon import (
            get_recommended_splits,
            pa_decode_gluon,
        )

        context_lens = context_lens.to(torch.int32)
        scale = torch.ones(1, dtype=torch.float32, device="cuda")
        return lambda: pa_decode_gluon(
            output=output,
            query=query,
            key_cache=key_cache,
            value_cache=value_cache,
            context_lengths=context_lens,
            block_tables=block_tables,
            softmax_scale=attn_scale,
            query_length=query_length,
            max_context_partition_num=get_recommended_splits(BS, H_KV),
            compute_type=compute_type,
            key_scale=scale,
            value_scale=scale,
        )

    k_scale = torch.tensor([1.0])
    v_scale = torch.tensor([1.0])

    return lambda: paged_attention_decode(
        output=output,
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        seq_lens=context_lens,
        block_tables=block_tables,
        attn_scale=attn_scale,
        max_seq_len=SEQ_LEN,
        compute_type=compute_type,
        k_scale=k_scale,
        v_scale=v_scale,
    )


def run_benchmark(args):
    dtype = arg_to_torch_dtype[args.dtype]
    kv_cache_dtype = arg_to_torch_dtype[args.kv_cache_dtype]
    compute_type = arg_to_torch_dtype[args.compute_type]
    if args.backend == "triton":
        compute_type = torch_to_triton_dtype[compute_type]
    output_type = arg_to_torch_dtype[args.output_type]

    x_vals_list = model_benchmark_configs(args)
    x_names = [
        "model",
        "BS",
        "HQ",
        "HK",
        "SEQ_LEN",
        "HEAD_DIM",
        "VALUE_HEAD_DIM",
        "QUERY_LEN",
    ]

    plot_name = get_caller_name_no_ext()
    benchmark = triton.testing.Benchmark(
        x_names=x_names,
        x_vals=x_vals_list,
        line_arg="metric",
        line_vals=["time", "tflops", "bandwidth"],
        line_names=["Time_(ms)", "TFLOPS", "Bandwidth_(GB/s)"],
        styles=[("red", "-"), ("blue", "-"), ("yellow", "-")],
        ylabel="ms / TFLOPS / GB/s",
        plot_name=plot_name,
        args={},
    )

    timings = {}

    @triton.testing.perf_report([benchmark])
    def bench_paged_attn_decode(
        BS,
        HQ,
        HK,
        SEQ_LEN,
        HEAD_DIM,
        VALUE_HEAD_DIM,
        QUERY_LEN,
        metric,
        model=None,
    ):
        PAGE_SIZE = args.page_size
        num_blocks = BS * triton.cdiv(SEQ_LEN, PAGE_SIZE)
        shape = (BS, HQ, HK, SEQ_LEN, HEAD_DIM, VALUE_HEAD_DIM, QUERY_LEN, PAGE_SIZE)
        if shape not in timings:
            fn = paged_attn_decode(
                BS,
                HQ,
                HK,
                HEAD_DIM,
                PAGE_SIZE,
                SEQ_LEN,
                dtype,
                kv_cache_dtype,
                compute_type,
                output_type,
                args.backend,
                value_head_size=VALUE_HEAD_DIM,
                query_length=QUERY_LEN,
            )
            timings[shape] = triton.testing.do_bench(fn, warmup=25, rep=100)
        ms = timings[shape]

        # query and output
        mem = (BS * QUERY_LEN * HQ) * (
            HEAD_DIM * get_dtype_bytes(dtype)
            + VALUE_HEAD_DIM * get_dtype_bytes(output_type)
        )
        # Effective KV traffic: count each sequence's logical tokens once.
        mem += (
            BS
            * HK
            * SEQ_LEN
            * (HEAD_DIM + VALUE_HEAD_DIM)
            * get_dtype_bytes(kv_cache_dtype)
        )
        # Large cache offsets require 64-bit block IDs.
        cache_elements = num_blocks * HK * PAGE_SIZE * max(HEAD_DIM, VALUE_HEAD_DIM)
        mem += num_blocks * (8 if cache_elements > 2**31 - 1 else 4)
        # Gluon converts context lengths to int32; Triton keeps int64.
        mem += BS * (4 if args.backend == "gluon" else 8)

        # QK and PV GEMMs; successive query tokens have a causal frontier.
        attended_tokens = QUERY_LEN * SEQ_LEN - QUERY_LEN * (QUERY_LEN - 1) // 2
        flops = 2.0 * BS * HQ * attended_tokens * (HEAD_DIM + VALUE_HEAD_DIM)

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

    frames = bench_paged_attn_decode.run(
        save_path="." if args.o else None,
        print_data=False,
        return_df=True,
    )
    result = frames[0]
    unit_suffix = f" ({benchmark.ylabel})"
    result.rename(
        columns={f"{name}{unit_suffix}": name for name in benchmark.line_names},
        inplace=True,
    )
    print(f"{plot_name}:")
    print(result.to_string())
    if args.o:
        result.to_csv(f"{plot_name}.csv", index=False, float_format="%.6f")


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
    parser.add_argument("-b", type=int, default=0, help="Batch size; default: 1024")
    parser.add_argument("-hq", type=int, default=0)
    parser.add_argument("-hk", type=int, default=0)
    parser.add_argument(
        "-sq", type=int, default=0, help="Context length; default: 8192"
    )
    parser.add_argument(
        "--head-dim",
        type=int,
        help="Q/K head width; default: selected model width or 128 for custom heads",
    )
    parser.add_argument(
        "--page-size",
        type=int,
        help="KV tokens per page; default: 16 for Gluon, 128 for Triton",
    )
    parser.add_argument(
        "--value-head-dim",
        type=int,
        help="V/output width; default: selected model width or Q/K width",
    )
    parser.add_argument(
        "--query-length",
        type=int,
        choices=[1, 2, 3, 4],
        default=1,
        help="Query length; default: 1",
    )
    parser.add_argument("-dtype", default="fp16")
    parser.add_argument("-kv_cache_dtype", default="fp16")
    parser.add_argument("-compute_type", default="fp16")
    parser.add_argument("-output_type", default="fp16")
    parser.add_argument(
        "--backend",
        choices=["triton", "gluon"],
        default="triton",
        help="triton: paged_attention_decode; gluon: pa_decode_gluon (PS mode).",
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
    if args.page_size is None:
        args.page_size = 16 if args.backend == "gluon" else 128
    if args.page_size <= 0:
        parser.error("--page-size must be positive")
    return args


arg_to_torch_dtype = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
    "e5m2fnuz": torch.float8_e5m2fnuz,
    "e4m3fnuz": torch.float8_e4m3fnuz,
    "e4m3fn": torch.float8_e4m3fn,
}


def main():
    args = parse_args()
    if args.print_vgpr:
        print("Retrieving VGPR usage for Triton kernels...")
        fun = lambda: run_benchmark(args)
        print_vgpr(fun, get_caller_name_no_ext())
        return 0
    run_benchmark(args)


if __name__ == "__main__":
    sys.exit(main())
