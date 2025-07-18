import triton
import triton.language as tl
from utils.benchmark_utils import (
    get_model_configs,
    get_available_models,
    get_dtype_bytes,
)
import torch
import argparse
from aiter.ops.triton.pa_decode import paged_attention_decode
from aiter.ops.triton.lean_atten_paged import persistent_lean_attention_paged
import sys
import random

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
        block_table = random.sample(range(0, num_blocks), num_blocks) + [
            random.randint(0, num_blocks - 1)
            for _ in range(max_num_blks_per_seq - num_blocks)
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


def input_la_helper(
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
    total_programs = 304
    BLOCK_M = 16
    BLOCK_N = KV_BLK_SZ

    n_ctx = [SEQ_LEN for _ in range(0, B)]
    N_CTX_Q = 16

    try:
        sum_n_ctx = sum(int(n) for n in n_ctx)
    except ValueError:
        print(f"N_CTX contains non-numeric values: {n_ctx}")

    # N_CTX is a list of context lengthes for all the req in a batch
    # First, calculate #BLOCK_N for each context length "list_num_block_n"
    # Second, Convert it to a list of assumulative lengthes "list_sum_block_n"
    # Third, convert list to a tensor "batch_num_block_n"
    for s in n_ctx:
        # print(f"s={s}")
        list_num_block_n = [
            (int(str(s).strip()) + BLOCK_N - 1) // BLOCK_N for s in n_ctx
        ]
    len_sum = 0
    list_sum_block_n = []
    for i in range(B):
        len_sum += list_num_block_n[i]
        list_sum_block_n.append(len_sum)
    batch_num_block_n = torch.tensor(list_sum_block_n, device="cuda", dtype=torch.int32)

    sm_scale = 0.5

    # Allocate Tensors
    q = torch.empty((H_Q, N_CTX_Q * B, D), dtype=dtype, device="cuda").normal_(
        mean=0.0, std=0.5
    )
    k = torch.empty((H_Q, sum_n_ctx, D), dtype=dtype, device="cuda").normal_(
        mean=0.0, std=0.5
    )
    v = torch.empty((H_Q, sum_n_ctx, D), dtype=dtype, device="cuda").normal_(
        mean=0.0, std=0.5
    )

    num_kv_blocks = sum_n_ctx // BLOCK_N + (1 if (sum_n_ctx % BLOCK_N != 0) else 0)

    block_tables = []
    for head in range(H_Q):
        b = random.sample(range(0, num_kv_blocks), num_kv_blocks)
        block_tables.append(b)
    kv_block_tables = torch.tensor(block_tables, dtype=torch.int32, device="cuda")

    # LeanAttention Specific Parameters
    Mp = torch.empty((total_programs, N_CTX_Q), device=q.device, dtype=torch.float32)
    Lp = torch.empty((total_programs, N_CTX_Q), device=q.device, dtype=torch.float32)
    Op = torch.empty((total_programs, N_CTX_Q, D), device=q.device, dtype=torch.float32)

    locks = torch.zeros((total_programs,), device=q.device, dtype=torch.int32)

    return (
        q,
        k,
        v,
        kv_block_tables,
        Mp,
        Lp,
        Op,
        total_programs,
        locks,
        batch_num_block_n,
        BLOCK_M,
        BLOCK_N,
    )


def model_benchmark_configs(args):
    config_file = args.model_configs
    configs = get_model_configs(
        config_path=config_file,
        models="llama3,deepseek" if args.model == None else args.model,
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
    OP,
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
    (
        q,
        k,
        v,
        kv_block_tables,
        Mp,
        Lp,
        Op,
        total_programs,
        locks,
        batch_num_block_n,
        BLOCK_M,
        BLOCK_N,
    ) = input_la_helper(
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

    fn = None
    if OP == "PagedAttention":
        fn = lambda: paged_attention_decode(
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
    elif OP == "LeanAttentionPaged":
        fn = lambda: persistent_lean_attention_paged(
            q=q,
            k=k,
            v=v,
            kv_block_tables=kv_block_tables,
            Mp=Mp,
            Lp=Lp,
            Op=Op,
            locks=locks,
            batch_num_block_n=batch_num_block_n,
            total_programs=total_programs,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            # d: int,
            batch_size=BS,
            sm_scale=0.5,
            num_warps=4,
            waves_per_eu=2,
        )
    else:
        print(f"Unknown op {OP}")

    return fn


def run_benchmark(args):
    dtype = arg_to_torch_dtype[args.dtype]
    kv_cache_dtype = arg_to_torch_dtype[args.kv_cache_dtype]
    compute_type = torch_to_triton_dtype[arg_to_torch_dtype[args.compute_type]]
    output_type = arg_to_torch_dtype[args.output_type]

    BS = args.b if args.b else 1

    x_vals_list = []
    for op in ["PagedAttention", "LeanAttentionPaged"]:
        for HQ in [32]:
            HK = HQ
            for SEQ_LEN in [
                512,
                1024,
                2 * 1024,
                4 * 1024,
                8 * 1024,
                16 * 1024,
                32 * 1024,
            ]:
                for HEAD_DIM in [128]:
                    x_vals_list.append((op, BS, HQ, HK, SEQ_LEN, HEAD_DIM))

    x_names = ["OP", "BS", "HQ", "HK", "SEQ_LEN", "HEAD_DIM"]

    model_name = "paged-attn-decode"

    line_names = ["Time (ms)"]
    line_vals = ["time"]

    benchmark = triton.testing.Benchmark(
        x_names=x_names,
        x_vals=x_vals_list,
        line_arg="metric",
        line_vals=line_vals,
        line_names=line_names,
        styles=[("red", "-")],
        ylabel="ms",
        plot_name=f"{model_name}-benchmark",
        args={},
    )

    @triton.testing.perf_report([benchmark])
    def bench_paged_attn_decode(OP, BS, HQ, HK, SEQ_LEN, HEAD_DIM, metric):
        KV_BLK_SZ = 16
        num_blocks = 64
        fn = paged_attn_decode(
            OP,
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

        # Return exactly one scalar depending on which metric is active
        if metric == "time":
            return ms
        else:
            raise ValueError("Unknown metric: " + metric)

    bench_paged_attn_decode.run(save_path=".", print_data=True)


def parse_args():
    parser = argparse.ArgumentParser(
        prog="Benchmark Paged Attention decode",
        allow_abbrev=False,
    )
    parser.add_argument("-b", type=int, default=0)
    parser.add_argument("-hq", type=int, default=0)
    parser.add_argument("-hk", type=int, default=0)
    parser.add_argument("-sq", type=int, default=0)
    parser.add_argument("-dtype", default="fp16")
    parser.add_argument("-kv_cache_dtype", default="fp16")
    parser.add_argument("-compute_type", default="fp16")
    parser.add_argument("-output_type", default="fp16")
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
    run_benchmark(args)


if __name__ == "__main__":
    sys.exit(main())
