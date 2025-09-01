import torch
import sys
import math
import random
import triton

from aiter.ops.triton.pa_prefill import context_attention_fwd
from op_tests.op_benchmarks.triton.utils.benchmark_utils import (
    get_model_configs,
    get_dtype_bytes,
    get_caller_name_no_ext,
    print_vgpr,
)
from op_tests.op_benchmarks.triton.utils.argparse import get_parser
from op_tests.triton_tests.test_pa_prefill import (
    seed_everything,
    STR_DTYPE_TO_TORCH_DTYPE,
)


def _get_alibi_slopes(total_num_heads: int) -> torch.Tensor:
    closest_power_of_2 = 2 ** math.floor(math.log2(total_num_heads))
    base = torch.tensor(
        2 ** (-(2 ** -(math.log2(closest_power_of_2) - 3))),
        dtype=torch.float32,
    )
    powers = torch.arange(1, 1 + closest_power_of_2, dtype=torch.int32)
    slopes = torch.pow(base, powers)

    if closest_power_of_2 != total_num_heads:
        extra_base = torch.tensor(
            2 ** (-(2 ** -(math.log2(2 * closest_power_of_2) - 3))),
            dtype=torch.float32,
        )
        num_remaining_heads = min(
            closest_power_of_2, total_num_heads - closest_power_of_2
        )
        extra_powers = torch.arange(
            start=1, end=1 + 2 * num_remaining_heads, step=2, dtype=torch.int32
        )
        slopes = torch.cat([slopes, torch.pow(extra_base, extra_powers)], dim=0)
    return slopes


def input_helper(
    BS,
    MAX_SEQ_LEN,
    MAX_CTX_LEN,
    cache_size,
    block_size,
    max_block_per_request,
    num_heads: int,
    head_size: int,
    num_queries_per_kv: int,
    dtype: torch.dtype,
    kv_cache_dtype: str,
    device: str,
    use_alibi_slope: bool,
):
    seed_everything(0)
    torch.set_default_device(device)

    # Need this, otherwise when we capture the graph the process
    # for GPU 1 would run on both GPU0 and GPU1 and things would hang
    #
    # see also similar issue: https://github.com/Dao-AILab/flash-attention/issues/523
    torch.cuda.set_device(device)

    if use_alibi_slope:
        alibi_slopes = _get_alibi_slopes(num_heads).to(device)

    query_lens = [random.randint(16, MAX_SEQ_LEN) for _ in range(BS)]
    ctx_lens = [random.randint(16, MAX_CTX_LEN) for _ in range(BS)]
    seq_lens = [a + b for a, b in zip(query_lens, ctx_lens)]
    num_kv_heads = num_heads // num_queries_per_kv

    num_tokens = sum(query_lens)
    query = torch.empty(num_tokens, num_heads, head_size, dtype=dtype)
    query.uniform_(-1e-3, 1e-3)
    output = torch.empty(num_tokens, num_heads, head_size, dtype=dtype)

    kv = torch.empty(sum(seq_lens), 2, num_kv_heads, head_size, dtype=dtype)
    kv.uniform_(-1e-3, 1e-3)
    key, value = kv.unbind(dim=1)

    if kv_cache_dtype == "auto":
        cache_dtype = dtype
    else:
        cache_dtype = STR_DTYPE_TO_TORCH_DTYPE[kv_cache_dtype]
    k_cache = torch.zeros(
        cache_size, block_size, num_kv_heads, head_size, dtype=cache_dtype
    )
    v_cache = torch.zeros(
        cache_size, block_size, num_kv_heads, head_size, dtype=cache_dtype
    )
    k = torch.zeros(sum(query_lens), num_kv_heads, head_size, dtype=dtype)
    v = torch.zeros(sum(query_lens), num_kv_heads, head_size, dtype=dtype)
    values = torch.arange(0, cache_size, dtype=torch.long)
    values = values[torch.randperm(cache_size)]
    block_table = values[: BS * max_block_per_request].view(BS, max_block_per_request)
    b_seq_len = torch.tensor(seq_lens, dtype=torch.long)
    b_ctx_len = torch.tensor(ctx_lens, dtype=torch.long)
    b_start_loc = torch.cumsum(torch.tensor([0] + query_lens, dtype=torch.long), dim=0)
    max_input_len = MAX_SEQ_LEN
    # copy kv to cache
    b_seq_start_loc = torch.cumsum(
        torch.tensor([0] + seq_lens[:-1], dtype=torch.long), dim=0
    )
    for i in range(BS):
        for j in range(query_lens[i]):
            k[b_start_loc[i] + j].copy_(key[b_seq_start_loc[i] + b_ctx_len[i] + j])
            v[b_start_loc[i] + j].copy_(value[b_seq_start_loc[i] + b_ctx_len[i] + j])
        cur_ctx = 0
        block_id = 0
        while cur_ctx < b_ctx_len[i]:
            start_loc = b_seq_start_loc[i] + cur_ctx
            if cur_ctx + block_size > b_ctx_len[i]:
                end_loc = b_seq_start_loc[i] + b_ctx_len[i]
            else:
                end_loc = start_loc + block_size
            start_slot = block_table[i, block_id] * block_size
            end_slot = start_slot + end_loc - start_loc
            k_cache.view(-1, num_kv_heads, head_size)[start_slot:end_slot].copy_(
                key[start_loc:end_loc]
            )
            v_cache.view(-1, num_kv_heads, head_size)[start_slot:end_slot].copy_(
                value[start_loc:end_loc]
            )
            cur_ctx += block_size
            block_id += 1
    # transpose K_cache[num_blocks, block_size, num_kv_heads, head_size]
    # to K_cache[num_blocks, num_kv_heads, head_size/8, block_size, 8]
    k_cache = (
        k_cache.view(-1, block_size, num_kv_heads, head_size // 8, 8)
        .permute(0, 2, 3, 1, 4)
        .contiguous()
    )
    # transpose V_cache[num_blocks, block_size, num_kv_heads, head_size]
    # to V_cache[num_blocks, num_kv_heads, head_size, block_size]
    v_cache = (
        v_cache.view(-1, block_size, num_kv_heads, head_size)
        .permute(0, 2, 3, 1)
        .contiguous()
    )
    k_scale = v_scale = torch.tensor(1.0, dtype=torch.float32, device=device)

    if use_alibi_slope:
        return (
            query,
            k,
            v,
            output,
            k_cache,
            v_cache,
            block_table,
            b_start_loc,
            b_seq_len,
            max_input_len,
            k_scale,
            v_scale,
            alibi_slopes,
        )
    else:
        return (
            query,
            k,
            v,
            output,
            k_cache,
            v_cache,
            block_table,
            b_start_loc,
            b_seq_len,
            max_input_len,
            k_scale,
            v_scale,
            None,
        )


def model_benchmark_configs(args):
    config_file = args.model_configs
    configs = get_model_configs(
        config_path=config_file,
        models="llama3,deepseek" if args.model is None else args.model,
    )
    fa_configs = []
    BS = args.b if args.b else 16

    for model_name, config in configs.items():
        HQ = config["num_attention_heads"]
        HK = (
            HQ
            if config["num_key_value_heads"] is None
            else config["num_key_value_heads"]
        )
        SEQ_LEN = args.sq if args.sq else 1024
        HEAD_DIM = config["hidden_size"] // HQ
        fa_configs.append((model_name, BS, HQ, HK, SEQ_LEN, HEAD_DIM))

    return fa_configs


def run_benchmark(args):
    dtype = arg_to_torch_dtype[args.dtype]
    kv_cache_dtype = args.kv_cache_dtype
    use_alibi_slope = args.use_alibi_slope

    x_vals_list = model_benchmark_configs(args)
    x_names = ["model", "BS", "HQ", "HK", "MAX_SEQ_LEN", "HEAD_DIM"]

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
    def bench_paged_attn_decode(BS, HQ, HK, MAX_SEQ_LEN, HEAD_DIM, metric, model=None):
        # TODO tune this
        MAX_CTX_LEN = MAX_SEQ_LEN
        max_block_per_request = 1024

        block_size = MAX_SEQ_LEN // max_block_per_request

        cache_size = max_block_per_request * BS

        if kv_cache_dtype == "auto":
            torch_kv_cache_dtype = dtype
        else:
            torch_kv_cache_dtype = STR_DTYPE_TO_TORCH_DTYPE[kv_cache_dtype]

        num_queries_per_kv = HQ // HK

        (
            query,
            k,
            v,
            output,
            k_cache,
            v_cache,
            block_table,
            b_start_loc,
            b_seq_len,
            max_input_len,
            k_scale,
            v_scale,
            alibi_slopes,
        ) = input_helper(
            BS=BS,
            MAX_SEQ_LEN=MAX_SEQ_LEN,
            MAX_CTX_LEN=MAX_CTX_LEN,
            cache_size=cache_size,
            block_size=block_size,
            max_block_per_request=max_block_per_request,
            num_heads=HQ,
            head_size=HEAD_DIM,
            num_queries_per_kv=num_queries_per_kv,
            dtype=dtype,
            kv_cache_dtype=kv_cache_dtype,
            device=[
                f"cuda:{i}" for i in range(1 if torch.cuda.device_count() == 1 else 2)
            ][0],
            use_alibi_slope=use_alibi_slope,
        )

        num_tokens = query.shape[0]
        fn = lambda: context_attention_fwd(  # noqa: E731
            query,
            k,
            v,
            output,
            kv_cache_dtype,
            k_cache,
            v_cache,
            block_table,
            b_start_loc,
            b_seq_len,
            max_input_len,
            k_scale,
            v_scale,
            alibi_slopes=alibi_slopes,
        )
        ms = triton.testing.do_bench(fn, warmup=25, rep=100)

        # query and output
        mem = (num_tokens * HQ * HEAD_DIM) * get_dtype_bytes(dtype) * 2
        # kv_cache
        mem += (
            cache_size
            * block_size
            * HK
            * HEAD_DIM
            * get_dtype_bytes(torch_kv_cache_dtype)
            * 2
        )
        # k, v
        mem += num_tokens * HK * HEAD_DIM * get_dtype_bytes(dtype) * 2
        # block_tables int32
        mem += BS * max_block_per_request * 4
        # b_seq_len int32
        mem += BS * 4
        # b_start_loc int32
        mem += BS * 4

        # cache
        flops = (2.0 * BS * HQ * (num_tokens // BS) * (num_tokens // BS) * HEAD_DIM) * 2
        # casual
        flops += (
            (2.0 * BS * HQ * max_block_per_request * (num_tokens // BS) * HEAD_DIM)
            * 2
            // 2
        )

        bandwidth = mem / (ms * 1e-3) * 1e-9  # GB/s
        # bandwidth = mem / (ms * 1e-3) * 1e-9  # GB/s
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
    parser = get_parser(kernel_name="Paged Attention Decode")

    parser.add_argument("-b", type=int, default=0)
    parser.add_argument("-hq", type=int, default=0)
    parser.add_argument("-hk", type=int, default=0)
    parser.add_argument("-sq", type=int, default=0)
    parser.add_argument("-use_alibi_slope", action="store_true", default=False)
    parser.add_argument("--dtype", default="fp16")
    parser.add_argument("--kv_cache_dtype", default="auto")
    parser.add_argument("--compute_type", default="fp16")
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
