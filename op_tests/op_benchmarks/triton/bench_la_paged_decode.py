import functools
import random
import sys

import torch
import triton
from utils.benchmark_utils import get_model_configs

from aiter.ops.triton.attention.lean_atten_paged import persistent_lean_attention_paged
from aiter.ops.triton.attention.pa_decode import paged_attention_decode
from aiter.ops.triton.utils.types import torch_to_triton_dtype
from op_tests.op_benchmarks.triton.utils.argparse import (
    get_do_bench_graph_kwargs,
    get_parser,
    get_rot_bufs_kwargs,
)
from op_tests.op_benchmarks.triton.utils.benchmark_utils import (
    do_bench_aiter_triton,
    generate_rotating_buffers_pool,
    get_caller_name_no_ext,
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
        block_table = random.sample(range(num_blocks), num_blocks) + [
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

    n_ctx = [SEQ_LEN for _ in range(B)]
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
        b = random.sample(range(num_kv_blocks), num_kv_blocks)
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


def _generate_pa_decode_pool_inputs(
    B,
    H_Q,
    H_KV,
    D,
    KV_BLK_SZ,
    SEQ_LEN,
    dtype,
    kv_cache_dtype,
    output_type,
    num_blocks,
):
    (
        query,
        output,
        _key_cache,
        _value_cache,
        key_cache_tri,
        value_cache_tri,
        context_lens,
        block_tables,
        _max_context_len,
    ) = input_helper(
        B,
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
    # max_context_len is a GPU scalar and is always == SEQ_LEN here (context_lens
    # is uniformly filled); callers use the plain SEQ_LEN int instead so nothing
    # forces a device sync inside the captured region.
    return (
        query,
        output,
        key_cache_tri,
        value_cache_tri,
        context_lens,
        block_tables,
    )


def _generate_la_paged_pool_inputs(
    B,
    H_Q,
    H_KV,
    D,
    KV_BLK_SZ,
    SEQ_LEN,
    dtype,
    kv_cache_dtype,
    output_type,
    num_blocks,
):
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
        B,
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


def run_benchmark(args):
    dtype = arg_to_torch_dtype[args.dtype]
    kv_cache_dtype = arg_to_torch_dtype[args.kv_cache_dtype]
    compute_type = torch_to_triton_dtype[arg_to_torch_dtype[args.compute_type]]
    output_type = arg_to_torch_dtype[args.output_type]
    kwargs_do_bench_graph = get_do_bench_graph_kwargs(args)
    kwargs_rot_bufs = get_rot_bufs_kwargs(args)

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

    line_names = ["Time_(ms)"]
    line_vals = ["time"]

    benchmark = triton.testing.Benchmark(
        x_names=x_names,
        x_vals=x_vals_list,
        line_arg="metric",
        line_vals=line_vals,
        line_names=line_names,
        styles=[("red", "-")],
        ylabel="ms",
        plot_name=get_caller_name_no_ext(),
        args={},
    )

    @triton.testing.perf_report([benchmark])
    def bench_paged_attn_decode(OP, BS, HQ, HK, SEQ_LEN, HEAD_DIM, metric):
        KV_BLK_SZ = 16
        num_blocks = 64
        attn_scale = 1.0 / (HEAD_DIM**0.5)

        if OP == "PagedAttention":
            _p_gen = functools.partial(
                _generate_pa_decode_pool_inputs,
                BS,
                HQ,
                HK,
                HEAD_DIM,
                KV_BLK_SZ,
                SEQ_LEN,
                dtype,
                kv_cache_dtype,
                output_type,
                num_blocks,
            )
            _out_val_keys = [
                "query",
                "output",
                "key_cache_tri",
                "value_cache_tri",
                "context_lens",
                "block_tables",
            ]
            rot_bufs = generate_rotating_buffers_pool(
                [_p_gen], [_out_val_keys], **kwargs_rot_bufs
            )

            k_scale = torch.tensor([1.0])
            v_scale = torch.tensor([1.0])

            def _run(
                query,
                output,
                key_cache_tri,
                value_cache_tri,
                context_lens,
                block_tables,
            ):
                paged_attention_decode(
                    output=output,
                    query=query,
                    key_cache=key_cache_tri,
                    value_cache=value_cache_tri,
                    seq_lens=context_lens,
                    block_tables=block_tables,
                    attn_scale=attn_scale,
                    max_seq_len=SEQ_LEN,
                    compute_type=compute_type,
                    k_scale=k_scale,
                    v_scale=v_scale,
                )

        elif OP == "LeanAttentionPaged":
            _p_gen = functools.partial(
                _generate_la_paged_pool_inputs,
                BS,
                HQ,
                HK,
                HEAD_DIM,
                KV_BLK_SZ,
                SEQ_LEN,
                dtype,
                kv_cache_dtype,
                output_type,
                num_blocks,
            )
            _out_val_keys = [
                "q",
                "k",
                "v",
                "kv_block_tables",
                "Mp",
                "Lp",
                "Op",
                "total_programs",
                "locks",
                "batch_num_block_n",
                "BLOCK_M",
                "BLOCK_N",
            ]
            rot_bufs = generate_rotating_buffers_pool(
                [_p_gen], [_out_val_keys], **kwargs_rot_bufs
            )

            def _run(
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
            ):
                persistent_lean_attention_paged(
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
                    batch_size=BS,
                    sm_scale=0.5,
                    num_warps=4,
                    waves_per_eu=2,
                )

        else:
            raise ValueError(f"Unknown op {OP}")

        ms = do_bench_aiter_triton(_run, rot_bufs, **kwargs_do_bench_graph)

        # Return exactly one scalar depending on which metric is active
        if metric == "time":
            return ms
        else:
            raise ValueError("Unknown metric: " + metric)

    bench_paged_attn_decode.run(save_path="." if args.o else None, print_data=True)


def parse_args():
    parser = get_parser(kernel_name="Paged Attention decode")
    parser.add_argument("-b", type=int, default=0)
    parser.add_argument("-hq", type=int, default=0)
    parser.add_argument("-hk", type=int, default=0)
    parser.add_argument("-sq", type=int, default=0)
    parser.add_argument("-dtype", default="fp16")
    parser.add_argument("-kv_cache_dtype", default="fp16")
    parser.add_argument("-compute_type", default="fp16")
    parser.add_argument("-output_type", default="fp16")
    parser.add_argument(
        "-print_vgpr",
        action="store_true",
        default=False,
        help="Prints the VGPR usage of the compiled triton kernel.",
    )
    parser.add_argument(
        "-o", action="store_true", help="Write performance results to CSV file"
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
        fun = lambda: run_benchmark(args)
        print_vgpr(fun, get_caller_name_no_ext())
        return 0
    run_benchmark(args)


if __name__ == "__main__":
    sys.exit(main())
