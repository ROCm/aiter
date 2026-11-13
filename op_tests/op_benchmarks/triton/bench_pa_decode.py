import sys

import torch
import triton
from triton.testing import runtime

from aiter.ops.triton.attention.pa_decode import paged_attention_decode
from aiter.ops.triton.utils.types import torch_to_triton_dtype
from op_tests.op_benchmarks.triton.utils.argparse import (
    get_parser,
    override_parser_action,
)
from op_tests.op_benchmarks.triton.utils.benchmark_utils import (
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
    num_blocks=4,
):
    """Helper function to generate input tensors for paged attention testing."""
    # Query tensor generation
    if dtype not in (torch.bfloat16, torch.float16, torch.float32):
        query = torch.randn(
            B, H_Q, D, dtype=torch.float16, device="cuda"
        )  # assumption dtype is 8bits or lower
        query = query.to(dtype=dtype)
    else:
        query = torch.randn(B, H_Q, D, dtype=dtype, device="cuda")

    if kv_cache_dtype not in (torch.bfloat16, torch.float16, torch.float32):
        x = min(
            D,
            16 // torch.tensor([], dtype=kv_cache_dtype, device="cuda").element_size(),
        )
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
        x = min(
            D,
            16 // torch.tensor([], dtype=kv_cache_dtype, device="cuda").element_size(),
        )
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

    key_cache_tri = key_cache.permute(0, 1, 3, 2, 4).flatten(3, 4).contiguous()
    value_cache_tri = value_cache.permute(0, 1, 3, 2).contiguous()

    context_lens = torch.full((B,), SEQ_LEN, device="cuda")
    max_context_len = int(context_lens.max())
    max_num_blks_per_seq = (max_context_len + KV_BLK_SZ - 1) // KV_BLK_SZ

    if num_blocks >= B * max_num_blks_per_seq:
        # Distinct pages, so the KV cache is read from HBM and not re-read from cache.
        block_tables = torch.randperm(num_blocks, dtype=torch.int32, device="cuda")[
            : B * max_num_blks_per_seq
        ].view(B, max_num_blks_per_seq)
    else:
        # Built on the GPU: a host-to-device copy of this table faults on gfx1250.
        block_tables = torch.randint(
            0, num_blocks, (B, max_num_blks_per_seq), dtype=torch.int32, device="cuda"
        )

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
    SEQ_LEN = args.sq if args.sq else 8192
    if args.hq:
        return [("custom", BS, args.hq, args.hk if args.hk else args.hq, SEQ_LEN, 128)]

    for model_name, config in configs.items():
        HQ = config["num_attention_heads"]
        HK = (
            HQ
            if config["num_key_value_heads"] is None
            else config["num_key_value_heads"]
        )
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
    backend="triton",
):
    (
        query,
        triton_output,
        key_cache,
        value_cache,
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

    if backend == "gluon":
        from aiter.ops.triton.gluon.pa_decode_gluon import (
            get_recommended_splits,
            pa_decode_gluon,
        )

        context_lens = context_lens.to(torch.int32)
        scale = torch.ones(1, dtype=torch.float32, device="cuda")
        num_splits = get_recommended_splits(
            BS, H_KV, max_context_length=max_context_len
        )
        return lambda: pa_decode_gluon(
            output=triton_output,
            query=query,
            key_cache=key_cache,
            value_cache=value_cache,
            context_lengths=context_lens,
            block_tables=block_tables,
            softmax_scale=attn_scale,
            query_length=1,
            max_context_partition_num=num_splits,
            compute_type=compute_type,
            key_scale=scale,
            value_scale=scale,
        )

    # Keep on the CPU: the wrapper calls .item() on the scales in CUDA graph capture.
    k_scale = torch.tensor([1.0], dtype=torch.float32, device="cpu")
    v_scale = torch.tensor([1.0], dtype=torch.float32, device="cpu")

    return lambda: paged_attention_decode(
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


def triton_max_blocks(HK, KV_BLK_SZ, HEAD_DIM):
    """Most pages a cache can hold with Triton's signed int32 cache offsets."""
    return (1 << 31) // (HK * KV_BLK_SZ * HEAD_DIM)


def kv_pool_pages(HK, KV_BLK_SZ, HEAD_DIM, kv_cache_dtype, free_bytes):
    """Pages for the KV pool: at most 4 GB per cache, and input_helper's peak in half the free memory."""
    page_elems = HK * KV_BLK_SZ * HEAD_DIM
    kv = get_dtype_bytes(kv_cache_dtype)
    # 8-bit caches are built from fp16 tensors.
    src = kv if kv_cache_dtype in (torch.bfloat16, torch.float16, torch.float32) else 2
    # Live bytes per element at the clamp, at the conversion and with the Triton copies.
    peak = max(3 * src, 2 * src + kv, src + 2 * kv, 4 * kv)
    return min((4 << 30) // (page_elems * kv), free_bytes // 2 // (page_elems * peak))


def bench_ms(fn, flush_cache=True):
    """Time of one fn call in ms, with the caches flushed before each call by default."""
    if not flush_cache:
        return triton.testing.do_bench_cudagraph(fn, rep=100)
    cache = runtime.driver.active.get_empty_cache_for_benchmark()
    flush = cache.zero_
    # Graph replays run back to back, so time flush + fn and subtract the flush.
    with_flush = triton.testing.do_bench_cudagraph(lambda: (flush(), fn()), rep=100)
    return with_flush - triton.testing.do_bench_cudagraph(flush, rep=100)


def run_benchmark(args):
    dtype = arg_to_torch_dtype[args.dtype]
    kv_cache_dtype = arg_to_torch_dtype[args.kv_cache_dtype]
    compute_type = arg_to_torch_dtype[args.compute_type]
    if args.backend == "triton":
        compute_type = torch_to_triton_dtype[compute_type]
    output_type = arg_to_torch_dtype[args.output_type]

    x_vals_list = model_benchmark_configs(args)
    x_names = ["model", "BS", "HQ", "HK", "SEQ_LEN", "HEAD_DIM"]
    # TODO tune this
    KV_BLK_SZ = 16 if args.backend == "gluon" else 128

    # Check every row before the sweep, so a bad -num_blocks cannot lose earlier rows.
    if args.backend == "triton" and args.num_blocks:
        for model, _, _, HK, _, HEAD_DIM in x_vals_list:
            max_blocks = triton_max_blocks(HK, KV_BLK_SZ, HEAD_DIM)
            if args.num_blocks > max_blocks:
                raise ValueError(
                    f"-num_blocks must be <= {max_blocks} for {model} "
                    "to keep Triton cache offsets within signed int32."
                )

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

    pools = {}  # One pool size per row, so every metric of a row reads the same cache.

    @triton.testing.perf_report([benchmark])
    def bench_paged_attn_decode(BS, HQ, HK, SEQ_LEN, HEAD_DIM, metric, model=None):
        kv_bytes = get_dtype_bytes(kv_cache_dtype)
        pages_needed = BS * triton.cdiv(SEQ_LEN, KV_BLK_SZ)
        row = (BS, HQ, HK, SEQ_LEN, HEAD_DIM)
        if row not in pools:
            if args.num_blocks:
                num_blocks = args.num_blocks
            else:
                # One page per (sequence, block slot), as far as memory allows.
                torch.cuda.empty_cache()  # Cached blocks of earlier rows are not free.
                free_bytes, _ = torch.cuda.mem_get_info()
                num_blocks = min(
                    pages_needed,
                    kv_pool_pages(HK, KV_BLK_SZ, HEAD_DIM, kv_cache_dtype, free_bytes),
                )
            if args.backend == "triton":
                num_blocks = min(num_blocks, triton_max_blocks(HK, KV_BLK_SZ, HEAD_DIM))
            if num_blocks < 1:
                raise ValueError(
                    f"{model}: not enough free GPU memory for one KV page. "
                    "Free memory or reduce the batch size or sequence length."
                )
            pools[row] = num_blocks
        num_blocks = pools[row]
        if num_blocks < pages_needed and metric == "time":  # once per row
            print(
                f"{model}: KV pool is {num_blocks} pages for {pages_needed} needed, "
                f"so pages are reused {pages_needed / num_blocks:.1f}x"
            )
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
            args.backend,
        )

        fn()
        torch.cuda.synchronize()
        ms = bench_ms(fn, flush_cache=not args.no_flush_cache)

        # query and output
        mem = (BS * HQ * HEAD_DIM) * (
            get_dtype_bytes(dtype) + get_dtype_bytes(output_type)
        )
        # K and V of every attended token
        mem += BS * SEQ_LEN * HK * HEAD_DIM * kv_bytes * 2
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
    parser = get_parser(kernel_name="Paged Attention decode")
    parser.add_argument(
        "-model_configs",
        type=str,
        default="utils/model_configs.json",
        help="Model config json file.",
    )
    parser.add_argument("-b", type=int, default=0)
    parser.add_argument("-hq", type=int, default=0)
    parser.add_argument("-hk", type=int, default=0)
    parser.add_argument("-sq", type=int, default=0)
    parser.add_argument(
        "-num_blocks",
        type=int,
        default=0,
        help="Physical KV pages. 0 = one page per sequence block slot (capped at 4 GB per cache tensor and by half of the free memory, counting the peak buffers of input creation).",
    )
    parser.add_argument("-dtype", default="fp16")
    parser.add_argument("-kv_cache_dtype", default="fp16")
    parser.add_argument("-compute_type", default="fp16")
    parser.add_argument("-output_type", default="fp16")
    override_parser_action(parser, "--backend", default="triton")

    parser.add_argument(
        "-no_flush_cache",
        action="store_true",
        help="Do not flush the caches before each call (flushing is the default).",
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
