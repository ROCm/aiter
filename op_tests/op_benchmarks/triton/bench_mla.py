# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# import hip
# hip.hip.hipInit(0)

import functools

import torch
import triton

from aiter.ops.triton.attention.mla import mla_decode_fwd, mla_prefill_fwd
from aiter.ops.triton.utils._triton import arch_info
from aiter.ops.triton.utils.types import e4m3_dtype
from op_tests.op_benchmarks.triton.utils.argparse import (
    get_do_bench_graph_kwargs,
    get_parser,
    get_rot_bufs_kwargs,
    override_parser_action,
)
from op_tests.op_benchmarks.triton.utils.benchmark_utils import (
    do_bench_aiter_triton,
    generate_rotating_buffers_pool,
    get_caller_name_no_ext,
)
from op_tests.triton_tests.attention.test_mla import input_helper, shuffle_kv_buffer

DEVICE_ARCH = arch_info.get_arch()
IS_DEVICE_ARCH_GFX12 = DEVICE_ARCH in ("gfx1250",)


def _generate_inputs(
    batch_size: int,
    decode_qlen: int,
    ctx_lens: int,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    block_size: int,
    num_query_heads: int,
    num_kv_heads: int,
    varlen: bool,
    q_dtype: torch.dtype,
    kv_dtype: torch.dtype,
    out_dtype: torch.dtype,
    use_out_scale: bool,
    shuffled_kv_cache: bool,
    device="cuda",
):
    (
        cu_seqlens_q,
        seq_lens_kv,
        max_seqlen_kv,
        block_tables,
        kv_buffer,
        query,
        sm_scale,
        _,
    ) = input_helper(
        batch_size,
        ctx_lens,
        decode_qlen,
        num_query_heads,
        num_kv_heads,
        kv_lora_rank,
        qk_rope_head_dim,
        block_size,
        None,  # tight, unique block allocation for realistic benchmark footprint
        varlen,
        device,
    )
    total_num_query_tokens = cu_seqlens_q[-1].item()
    kv_buffer = kv_buffer.to(kv_dtype)
    query = query.to(q_dtype)

    q_descale = None
    kv_descale = None
    if q_dtype != torch.bfloat16:
        q_descale = torch.rand((1,), dtype=torch.float32, device=device)

    if kv_dtype != torch.bfloat16:
        kv_descale = torch.rand((1,), dtype=torch.float32, device=device)

    out_scale = None
    if use_out_scale:
        out_scale = 1 / torch.rand(1, dtype=torch.float32, device=device)

    output = torch.empty(
        (total_num_query_tokens, num_query_heads, kv_lora_rank),
        dtype=out_dtype,
        device=device,
    )

    maybe_shuffled_kv_buffer = (
        shuffle_kv_buffer(kv_buffer, kv_lora_rank) if shuffled_kv_cache else kv_buffer
    )

    return (
        cu_seqlens_q,
        seq_lens_kv,
        max_seqlen_kv,
        block_tables,
        query,
        maybe_shuffled_kv_buffer,
        output,
        sm_scale,
        q_descale,
        kv_descale,
        out_scale,
    )


def benchmark(args):
    batch_size = args.batch_size
    decode_qlen = args.decode_qlen
    ctx_lens = args.ctx_lens
    kv_lora_rank = args.kv_lora_rank
    qk_rope_head_dim = args.qk_rope_head_dim
    BLOCK_SIZE = args.block_size
    num_query_heads = args.num_query_heads
    num_kv_heads = args.num_kv_heads
    varlen = args.varlen
    # Prefill only supports the unshuffled KV layout.
    shuffled_kv_cache = args.shuffled_kv_cache if decode_qlen > 0 else False
    q_dtype = e4m3_dtype if args.q_dtype == "fp8" else torch.bfloat16
    kv_dtype = e4m3_dtype if args.kv_dtype == "fp8" else torch.bfloat16
    out_dtype = e4m3_dtype if args.out_dtype == "fp8" else torch.bfloat16
    use_out_scale = out_dtype != torch.bfloat16
    backend = args.backend
    skip_reduce = args.skip_reduce
    kwargs_do_bench_graph = get_do_bench_graph_kwargs(args)
    kwargs_rot_bufs = get_rot_bufs_kwargs(args)

    # Prefill, adjust graph capture args. Otherwise, it takes too long to run
    if decode_qlen == 0 and kwargs_do_bench_graph["kernels_per_graph"] > 20:
        print("Prefill: setting kernels_per_graph=20 to reduce benchmark run time")
        kwargs_do_bench_graph["kernels_per_graph"] = 20

    if shuffled_kv_cache:
        assert BLOCK_SIZE >= 16, "Block size must be at least 16 for shuffled KV cache"

    configs = []

    x_names = [
        "batch_size",
        "decode_qlen",
        "ctx_lens",
        "num_query_heads",
        "num_kv_heads",
        "kv_lora_rank",
        "qk_rope_head_dim",
        "block_size",
        "varlen",
        "q_dtype",
        "kv_dtype",
        "out_dtype",
        "use_out_scale",
        "backend",
        "shuffled_kv_cache",
    ]

    x_vals_list = [
        (
            batch_size,
            decode_qlen,
            ctx_lens,
            num_query_heads,
            num_kv_heads,
            kv_lora_rank,
            qk_rope_head_dim,
            BLOCK_SIZE,
            varlen,
            q_dtype,
            kv_dtype,
            out_dtype,
            use_out_scale,
            backend,
            shuffled_kv_cache,
        )
    ]

    if args.metric == "time":
        unit = "ms"
    elif args.metric == "bandwidth":
        unit = "TB/s"
    else:
        raise ValueError("Unknown metric: " + args.metric)

    line_vals = [args.metric]
    line_names = [("Gluon " if backend == "gluon" else "Triton ") + args.metric]

    configs.append(
        triton.testing.Benchmark(
            x_names=x_names,
            x_vals=x_vals_list,
            line_arg="provider",
            line_vals=line_vals,
            line_names=line_names,
            plot_name=get_caller_name_no_ext(),
            styles=[("red", "-"), ("green", "-")],
            ylabel=unit,
            args={},
        )
    )

    @triton.testing.perf_report(configs)
    def bench_mla(
        batch_size: int,
        decode_qlen: int,
        ctx_lens: int,
        num_query_heads: int,
        num_kv_heads: int,
        kv_lora_rank: int,
        qk_rope_head_dim: int,
        block_size: int,
        varlen: bool,
        q_dtype: torch.dtype,
        kv_dtype: torch.dtype,
        out_dtype: torch.dtype,
        use_out_scale: bool,
        backend: str,
        shuffled_kv_cache: bool,
        provider,
    ):
        # TODO account for varlen cases

        # assert not varlen, "varlen=True not supported yet with rotating buffers"

        _p_gen = functools.partial(
            _generate_inputs,
            batch_size,
            decode_qlen,
            ctx_lens,
            kv_lora_rank,
            qk_rope_head_dim,
            block_size,
            num_query_heads,
            num_kv_heads,
            varlen,
            q_dtype,
            kv_dtype,
            out_dtype,
            use_out_scale,
            shuffled_kv_cache,
            device=args.device,
        )
        _out_val_keys = [
            "cu_seqlens_q",
            "seqused_k",
            "max_seqlen_kv",
            "block_tables",
            "q",
            "kv_buffer",
            "out",
            "softmax_scale",
            "q_descale",
            "kv_descale",
            "out_scale",
        ]
        rot_bufs = generate_rotating_buffers_pool(
            [_p_gen], [_out_val_keys], **kwargs_rot_bufs
        )

        # Calculate Bytes accessed
        # If varlen is set
        #  1) seqused_kv is varied across batch.
        #  2) With prefill, query length is also varied across batch. For decode, query length is fixed always
        #  we average the bytes accessed over the rotating buffers pool
        if decode_qlen > 0:
            _op_fn = functools.partial(
                mla_decode_fwd,
                kv_lora_rank=kv_lora_rank,
                qk_rope_head_dim=qk_rope_head_dim,
                causal=True,
                shuffled_kv_cache=shuffled_kv_cache,
                skip_reduce=skip_reduce,
            )
        else:
            _op_fn = functools.partial(
                mla_prefill_fwd,
                kv_lora_rank=kv_lora_rank,
                qk_rope_head_dim=qk_rope_head_dim,
                causal=True,
                shuffled_kv_cache=shuffled_kv_cache,
            )
        mem_in = []
        mem_out = []
        for sample in rot_bufs:
            out = _op_fn(**sample)
            mem_in.append(
                sample["q"].numel() * sample["q"].itemsize
                + sample["seqused_k"].sum().item()
                * num_kv_heads
                * (kv_lora_rank + qk_rope_head_dim)
                * 2
                * kv_dtype.itemsize
            )
            if decode_qlen > 0 and skip_reduce and isinstance(out, tuple):
                assert (
                    isinstance(out, tuple) and len(out) == 3
                ), "Output should be a tuple of 3 tensors for skip_reduce and decode_qlen > 0 1"
                segm_output, segm_max, segm_expsum = out
                mem_out.append(
                    segm_output.numel() * segm_output.itemsize
                    + segm_max.numel() * segm_max.itemsize
                    + segm_expsum.numel() * segm_expsum.itemsize
                )
            else:
                # A single segment returns the final tensor even with skip_reduce.
                mem_out.append(out.numel() * out.itemsize)
        mem = (sum(mem_in) + sum(mem_out)) / len(rot_bufs) * 1e-12

        # Run benchmark
        ms = do_bench_aiter_triton(_op_fn, rot_bufs, **kwargs_do_bench_graph)

        # Compute metric
        if provider == "time":
            return ms
        else:  # BW TB/s
            return mem / ms * 1e3

    bench_mla.run(save_path="." if args.o else None, print_data=True, show_plots=False)


def parse_args():
    parser = get_parser(kernel_name="MLA Decode/Prefill")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument(
        "--decode_qlen",
        type=int,
        default=1,
        help="Set to 0 to benchmark mla_prefill_fwd",
    )
    parser.add_argument("--ctx_lens", type=int, default=8192)
    parser.add_argument("--kv_lora_rank", type=int, default=512)
    parser.add_argument("--qk_rope_head_dim", type=int, default=64)
    parser.add_argument("--block_size", type=int, default=64)
    parser.add_argument("--shuffled_kv_cache", type=bool, default=True)
    parser.add_argument("--num_query_heads", type=int, default=16)
    parser.add_argument("--num_kv_heads", type=int, default=1)
    parser.add_argument("--varlen", type=bool, default=True)
    parser.add_argument("--q_dtype", type=str, default="bf16")
    parser.add_argument("--kv_dtype", type=str, default="bf16")
    parser.add_argument("--out_dtype", type=str, default="bf16")
    override_parser_action(
        parser, action="--metric", choices=["time", "bandwidth"], default="bandwidth"
    )
    parser.add_argument("--skip_reduce", type=bool, default=True)
    parser.add_argument("--device", default="cuda")

    parser.add_argument(
        "-o",
        action="store_true",
        default=False,
        help="Write performance results to CSV file",
    )

    return parser.parse_args()


def run_bench(args):
    torch.manual_seed(0)
    benchmark(args)


def main():
    args = parse_args()
    run_bench(args)


if __name__ == "__main__":
    main()
