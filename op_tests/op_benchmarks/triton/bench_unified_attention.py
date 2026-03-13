import math
import torch
import itertools
import argparse
import triton
from aiter.ops.triton.attention.unified_attention import (
    unified_attention,
    use_2d_kernel,
)
from aiter.ops.triton.utils.device_info import get_num_sms
from aiter.ops.triton.utils.types import get_fp8_dtypes
from op_tests.op_benchmarks.triton.utils.benchmark_utils import (
    print_vgpr,
    get_caller_name_no_ext,
)

e5m2_dtype, e4m3_dtype = get_fp8_dtypes()


def calculate_num_seq_decode_split(
    num_tokens, num_seqs, num_heads_q, num_heads_k, block_size
):
    """num_tokens = total query tokens (sum of query lengths); num_seqs = batch size."""
    BLOCK_M = 16
    num_queries_per_kv = num_heads_q // num_heads_k
    BLOCK_Q = BLOCK_M // num_queries_per_kv
    # TODO (cagri): what happens when BLOCK_M < num_queries_per_kv?
    # should we consider that case?
    if BLOCK_Q == 0:
        BLOCK_M = triton.next_power_of_2(num_queries_per_kv)
        BLOCK_Q = BLOCK_M // num_queries_per_kv
    total_num_q_blocks = num_tokens // BLOCK_Q + num_seqs
    num_2d_prgms = total_num_q_blocks * num_heads_k
    target_num_prgms = get_num_sms() * 4  # match unified_attention.select_3d_config
    TILE_SIZE = block_size
    num_segments = math.ceil(target_num_prgms / num_2d_prgms)
    num_segments = triton.next_power_of_2(num_segments)
    num_segments = min(num_segments, 128)
    MIN_SEGMENTS = 16 if TILE_SIZE <= 16 else 8
    num_segments = max(num_segments, MIN_SEGMENTS)
    return num_segments


def calculate_mem_bw(
    batch_size,
    seq_q_l,
    seq_kv_l,
    num_heads_q,
    num_heads_k,
    head_size,
    block_size,
    window_size,
    time_us,
    use_3d,
    num_tokens=None,
    num_seqs=None,
):
    """Memory bandwidth (GB/s). For 3D path, counts attn + reduce traffic: Q,K,V read;
    segment buffers written then read; final output written."""
    if num_tokens is None:
        num_tokens = batch_size * seq_q_l
    if num_seqs is None:
        num_seqs = batch_size
    if window_size > 0:
        seq_kv_l = window_size
    bytes_per_q_token = num_heads_q * head_size * 2
    Q_total = num_tokens * bytes_per_q_token
    K = V = seq_kv_l * num_heads_k * head_size * 2
    if use_3d is False:
        # 2D: one kernel reads Q,K,V and writes out (same shape as Q)
        mem = Q_total + num_seqs * (K + V) + Q_total
    else:
        num_splits = calculate_num_seq_decode_split(
            num_tokens, num_seqs, num_heads_q, num_heads_k, block_size
        )
        main_piece = num_heads_q * num_splits
        head_size_padded = triton.next_power_of_2(head_size)
        segment_buffers_per_token = (
            main_piece * head_size_padded + 2 * main_piece
        ) * 4  # float32
        segment_total = num_tokens * segment_buffers_per_token
        mem = (
            Q_total
            + num_seqs * (K + V)  # attention reads
            + 2 * segment_total  # write + read segment buffers
            + Q_total  # final output write
        )
    return (mem / 1e9) / (time_us * 1e-6)


def calculate_tflops(
    batch_size,
    seq_q_l,
    seq_kv_l,
    num_heads_q,
    num_heads_k,
    head_size,
    block_size,
    window_size,
    time_us,
    use_3d,
    num_tokens=None,
    total_qk=None,
    num_seqs=None,
):
    """Throughput in TFLOPS. For 3D path, adds reduce kernel FLOPs."""
    if num_tokens is None:
        num_tokens = batch_size * seq_q_l
    if total_qk is None:
        total_qk = batch_size * seq_q_l * seq_kv_l
    if num_seqs is None:
        num_seqs = batch_size
    if window_size > 0:
        seq_kv_l = window_size
    # FLOPs for QK^T (multiply + add)
    flops_qk = (2.0 * total_qk * num_heads_q * head_size) // 2
    # FLOPs for A x V (multiply + add)
    flops_av = (2.0 * total_qk * num_heads_q * head_size) // 2
    # FLOPs for softmax
    flops_softmax = (5.0 * num_heads_q * total_qk) // 2
    total_flops = flops_qk + flops_av + flops_softmax
    if use_3d:
        num_segments = calculate_num_seq_decode_split(
            num_tokens, num_seqs, num_heads_q, num_heads_k, block_size
        )
        # Reduce kernel: rescale by exp2, sum over segments, normalize (per token per head)
        head_size_padded = triton.next_power_of_2(head_size)
        flops_reduce = (
            num_tokens * num_heads_q * num_segments * (2 * head_size_padded + 6)
        )
        total_flops = total_flops + flops_reduce
    time_s = time_us * 1e-6
    tflops = total_flops / (time_s * 1e12)
    return tflops


def generate_data(
    seq_lens,
    num_blocks=32768,
    block_size=32,
    head_size=64,
    num_heads=(16, 2),
    sliding_window=None,
    q_dtype=e4m3_dtype,
    dtype=torch.bfloat16,
):
    """Generate inputs for unified_attention"""
    torch.cuda.manual_seed(0)
    num_seqs = len(seq_lens)
    query_lens = [x[0] for x in seq_lens]
    kv_lens = [x[1] for x in seq_lens]
    num_query_heads = num_heads[0]
    num_kv_heads = num_heads[1]
    assert num_query_heads % num_kv_heads == 0
    max_query_len = max(query_lens)
    max_kv_len = max(kv_lens)
    if sliding_window is not None and sliding_window > 0:
        window_size = (sliding_window - 1, 0)
    else:
        window_size = (-1, -1)
    scale = head_size**-0.5

    query = torch.randn(
        sum(query_lens),
        num_query_heads,
        head_size,
        dtype=dtype,
        device="cuda",
    )
    key_cache = torch.randn(
        num_blocks,
        block_size,
        num_kv_heads,
        head_size,
        dtype=dtype,
        device="cuda",
    )
    value_cache = torch.randn_like(key_cache)
    cu_query_lens = torch.tensor(
        [0] + query_lens,
        dtype=torch.int32,
        device="cuda",
    ).cumsum(dim=0, dtype=torch.int32)
    kv_lens = torch.tensor(kv_lens, dtype=torch.int32, device="cuda")

    max_num_blocks_per_seq = (max_kv_len + block_size - 1) // block_size
    block_tables = torch.randint(
        0,
        num_blocks,
        (num_seqs, max_num_blocks_per_seq),
        dtype=torch.int32,
        device="cuda",
    )
    values = torch.arange(0, num_blocks, dtype=torch.int32)
    values = values[torch.randperm(num_blocks)]
    block_tables = (
        values[: num_seqs * max_num_blocks_per_seq]
        .view(num_seqs, max_num_blocks_per_seq)
        .contiguous()
        .to("cuda")
    )

    sinks = torch.randn(num_query_heads, dtype=dtype, device="cuda")
    output = torch.empty_like(query)

    maybe_quantized_query = query
    maybe_quantized_key_cache = key_cache
    maybe_quantized_value_cache = value_cache
    q_descale = None
    k_descale = None
    v_descale = None
    if q_dtype is not None:
        # QKV are drawn from N(0, 1): no need for a fp8 scaling factor
        maybe_quantized_query = query.to(q_dtype)
        maybe_quantized_key_cache = key_cache.to(q_dtype)
        maybe_quantized_value_cache = value_cache.to(q_dtype)
        scale_shape = (num_seqs, num_kv_heads)
        q_descale = None  # Not yet supported
        k_descale = torch.rand(scale_shape, dtype=torch.float32, device="cuda")
        v_descale = torch.rand(scale_shape, dtype=torch.float32, device="cuda")

    return (
        maybe_quantized_query,
        maybe_quantized_key_cache,
        maybe_quantized_value_cache,
        sinks,
        output,
        cu_query_lens,
        kv_lens,
        max_query_len,
        max_kv_len,
        scale,
        window_size,
        block_tables,
        q_descale,
        k_descale,
        v_descale,
    )


def default_benchmark_configs():
    """Default grid (bs, num_heads_q, num_heads_k, head_size, seq_q_l, seq_kv_l, block_size, window_size)."""
    batch_sizes = [1, 4, 16]
    num_heads_q = [16, 48]
    num_heads_k = [2, 8]
    head_size = [64]
    seq_q_l = [1, 1024, 4096]
    seq_kv_l = [1024, 8192]
    block_size = [16, 64]
    window_size = [0, 128]
    configs = list(
        itertools.product(
            batch_sizes,
            num_heads_q,
            num_heads_k,
            head_size,
            seq_q_l,
            seq_kv_l,
            block_size,
            window_size,
        )
    )
    configs = [c for c in configs if c[1] % c[2] == 0 and c[4] <= c[5]]
    return configs


def create_benchmark_configs(custom, args):
    """Build triton.testing.Benchmark list."""
    dtype = arg_to_torch_dtype[args.dtype]
    x_names = [
        "bs",
        "num_heads_q",
        "num_heads_k",
        "head_size",
        "seq_q_l",
        "seq_kv_l",
        "block_size",
        "window_size",
    ]
    plot_name = get_caller_name_no_ext()
    extra_args = {"dtype": dtype, "device": "cuda"}

    if custom:
        window_size = args.window_size if args.window_size is not None else 0
        x_vals_list = [
            (
                args.bs,
                args.num_heads_q,
                args.num_heads_k,
                args.head_size,
                args.seq_q_l,
                args.seq_kv_l,
                args.block_size,
                window_size,
            )
        ]
    else:
        x_vals_list = default_benchmark_configs()

    if args.metric == "time":
        unit = "ms"
    elif args.metric == "throughput":
        unit = "TFLOPS"
    elif args.metric == "bandwidth":
        unit = "GB/s"
    else:
        raise ValueError(f"Unknown metric: {args.metric}")

    line_vals = [f"fwd({unit})"]
    configs = [
        triton.testing.Benchmark(
            x_names=x_names,
            x_vals=x_vals_list,
            line_arg="provider",
            line_vals=line_vals,
            line_names=line_vals,
            styles=[("red", "-")],
            ylabel=unit,
            plot_name=plot_name,
            args=extra_args,
        )
    ]
    return configs


def run_benchmark(custom, args):
    torch.manual_seed(20)

    @triton.testing.perf_report(create_benchmark_configs(custom, args))
    def bench_unified_attention(
        bs,
        num_heads_q,
        num_heads_k,
        head_size,
        seq_q_l,
        seq_kv_l,
        block_size,
        window_size,
        dtype,
        provider,
        device="cuda",
    ):
        """Benchmark unified attention forward."""
        # Build seq_lens: default all same (seq_q_l, seq_kv_l) per batch; or mixed if prefill_cnt/decode_cnt set
        prefill_cnt = args.prefill_cnt
        decode_cnt = args.decode_cnt
        if prefill_cnt == -1 and decode_cnt == -1:
            seq_lens = [(seq_q_l, seq_kv_l)] * bs
        else:
            import random

            seq_lens = []
            for _ in range(decode_cnt):
                seq_lens.append((1, seq_kv_l))
            for _ in range(prefill_cnt):
                seq_lens.append((seq_q_l, seq_kv_l))
            random.shuffle(seq_lens)

        (
            query,
            key_cache,
            value_cache,
            sinks,
            output,
            cu_query_lens,
            seqused_k,
            max_query_len,
            max_kv_len,
            scale,
            window_size_t,
            block_tables,
            q_descale,
            k_descale,
            v_descale,
        ) = generate_data(
            seq_lens,
            num_blocks=32768,
            block_size=block_size,
            head_size=head_size,
            num_heads=(num_heads_q, num_heads_k),
            sliding_window=window_size if window_size > 0 else None,
            q_dtype=torch.bfloat16,
            dtype=dtype,
        )

        def fn():
            return unified_attention(
                q=query,
                k=key_cache,
                v=value_cache,
                out=output,
                cu_seqlens_q=cu_query_lens,
                seqused_k=seqused_k,
                max_seqlen_q=max_query_len,
                max_seqlen_k=max_kv_len,
                softmax_scale=scale,
                causal=True,
                window_size=window_size_t,
                block_table=block_tables,
                softcap=0,
                q_descale=q_descale,
                k_descale=k_descale,
                v_descale=v_descale,
                sinks=sinks,
            )

        ms = triton.testing.do_bench(fn)
        time_us = ms * 1e3

        num_seqs = len(seq_lens)
        num_tokens = sum(q for q, _ in seq_lens)

        num_queries_per_kv = num_heads_q // num_heads_k
        BLOCK_M = (
            16
            if num_queries_per_kv <= 16
            else triton.next_power_of_2(num_queries_per_kv)
        )
        BLOCK_Q = BLOCK_M // num_queries_per_kv
        assert BLOCK_Q >= 1
        total_num_q_blocks = num_tokens // BLOCK_Q + num_seqs
        target_num_prgms = get_num_sms() * 4
        num_2d_prgms = total_num_q_blocks * num_heads_k
        sliding_window_val = window_size if window_size > 0 else 0
        all_decode = seq_q_l == 1
        use_3d = not use_2d_kernel(
            head_size,
            sliding_window_val,
            all_decode,
            seq_q_l,
            seq_kv_l,
            target_num_prgms,
            num_2d_prgms,
        )

        total_qk = sum(q * k for q, k in seq_lens)
        tflops = calculate_tflops(
            bs,
            seq_q_l,
            seq_kv_l,
            num_heads_q,
            num_heads_k,
            head_size,
            block_size,
            window_size,
            time_us,
            use_3d,
            num_tokens=num_tokens,
            total_qk=total_qk,
            num_seqs=num_seqs,
        )
        bw_gbs = calculate_mem_bw(
            bs,
            seq_q_l,
            seq_kv_l,
            num_heads_q,
            num_heads_k,
            head_size,
            block_size,
            window_size,
            time_us,
            use_3d,
            num_tokens=num_tokens,
            num_seqs=num_seqs,
        )

        if "ms" in provider:
            return ms
        elif "TFLOPS" in provider:
            return tflops
        else:
            return bw_gbs

    bench_unified_attention.run(save_path="." if args.o else None, print_data=True)


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="Benchmark Unified Attention",
        allow_abbrev=False,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--bs", type=int, default=0, help="Batch size (num sequences)")
    parser.add_argument(
        "--num_heads_q", type=int, default=0, help="Number of query heads"
    )
    parser.add_argument(
        "--num_heads_k", type=int, default=0, help="Number of K/V heads"
    )
    parser.add_argument("--head_size", type=int, default=0, help="Head dimension")
    parser.add_argument("--seq_q_l", type=int, default=0, help="Query sequence length")
    parser.add_argument(
        "--seq_kv_l", type=int, default=0, help="Key/Value sequence length"
    )
    parser.add_argument("--block_size", type=int, default=0, help="KV cache block size")
    parser.add_argument(
        "--window_size",
        type=int,
        default=0,
        help="Sliding window size (0 = no window)",
    )
    parser.add_argument(
        "--prefill_cnt",
        type=int,
        default=-1,
        help="Number of prefill requests (mixed batch)",
    )
    parser.add_argument(
        "--decode_cnt",
        type=int,
        default=-1,
        help="Number of decode requests (mixed batch)",
    )
    parser.add_argument("--dtype", default="bf16", choices=["fp32", "fp16", "bf16"])
    parser.add_argument(
        "--metric",
        type=str,
        choices=["time", "throughput", "bandwidth"],
        default="throughput",
        help="Metric: time (ms), throughput (TFLOPS), bandwidth (GB/s).",
    )
    parser.add_argument("--print_vgpr", action="store_true", help="Print VGPR usage")
    parser.add_argument("-o", action="store_true", help="Write results to CSV")
    return parser.parse_args(args=args)


arg_to_torch_dtype = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
}


def main(args: list[str] | None = None) -> None:
    args = parse_args(args=args)

    custom_config = False
    if any(
        [
            args.bs,
            args.num_heads_q,
            args.num_heads_k,
            args.head_size,
            args.seq_q_l,
            args.seq_kv_l,
            args.block_size,
        ]
    ):
        custom_config = True
        assert (
            args.bs
            and args.num_heads_q
            and args.seq_q_l
            and args.seq_kv_l
            and args.head_size
        ), "Custom config requires --bs, --num_heads_q, --head_size, --seq_q_l, --seq_kv_l."
        if not args.num_heads_k:
            args.num_heads_k = args.num_heads_q
        if args.num_heads_q % args.num_heads_k != 0:
            raise ValueError("num_heads_q must be divisible by num_heads_k")
        if not args.block_size:
            args.block_size = 16

    assert args.dtype in arg_to_torch_dtype, "Only fp16, bf16, fp32 supported."

    if args.print_vgpr:

        def fun():
            return run_benchmark(custom_config, args)

        print_vgpr(fun, get_caller_name_no_ext())
        return

    run_benchmark(custom_config, args)


if __name__ == "__main__":
    main()
