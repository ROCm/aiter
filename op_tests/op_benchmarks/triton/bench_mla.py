# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# import hip
# hip.hip.hipInit(0)

import random
import triton
import torch

from aiter.ops.triton.attention.mla import mla_decode_fwd
from aiter.ops.triton.attention.mla import mla_prefill_fwd
from op_tests.triton_tests.attention.test_mla import shuffle_kv_buffer
from aiter.ops.triton.utils.types import e4m3_dtype
import aiter.ops.triton.utils._triton.arch_info as arch_info
import argparse
from op_tests.op_benchmarks.triton.utils.benchmark_utils import (
    get_caller_name_no_ext,
)

DEVICE_ARCH = arch_info.get_arch()
IS_DEVICE_ARCH_GFX12 = DEVICE_ARCH in ("gfx1250",)


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
    shuffled_kv_cache = args.shuffled_kv_cache
    q_dtype = e4m3_dtype if args.q_dtype == "fp8" else torch.bfloat16
    kv_dtype = e4m3_dtype if args.kv_dtype == "fp8" else torch.bfloat16
    out_dtype = e4m3_dtype if args.out_dtype == "fp8" else torch.bfloat16
    use_out_scale = out_dtype != torch.bfloat16
    backend = args.backend
    skip_reduce = args.skip_reduce

    if shuffled_kv_cache:
        assert IS_DEVICE_ARCH_GFX12, "Gluon Unified Attention only supports gfx1250"
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
        warmup = 25
        rep = 100

        cu_seqlens_q = torch.zeros(batch_size + 1, dtype=torch.int, device="cuda")
        seq_lens_qo = torch.empty(batch_size, dtype=torch.int, device="cuda")
        seq_lens_kv = torch.empty(batch_size, dtype=torch.int, device="cuda")
        if varlen:
            for i in range(batch_size):
                seq_lens_kv[i] = max(
                    random.normalvariate(ctx_lens, ctx_lens / 2), ctx_lens
                )
        else:
            seq_lens_kv.fill_(ctx_lens)
        if decode_qlen > 0:
            seq_lens_qo.fill_(decode_qlen)
        else:
            if varlen:
                for i in range(batch_size):
                    seq_lens_qo[i] = max(
                        min(random.normalvariate(ctx_lens, ctx_lens / 2), ctx_lens), 1
                    )
            else:
                seq_lens_qo.fill_(ctx_lens)

        cu_seqlens_q[1 : batch_size + 1] = torch.cumsum(seq_lens_qo, dim=0)
        total_num_query_tokens = cu_seqlens_q[-1].item()

        max_seqlen_kv = seq_lens_kv.max().item()
        max_num_blocks_per_seq = (max_seqlen_kv + block_size - 1) // block_size
        num_blocks = batch_size * max_num_blocks_per_seq
        block_tables = torch.randperm(
            num_blocks, dtype=torch.int32, device="cuda"
        ).reshape(batch_size, max_num_blocks_per_seq)
        qk_head_dim = kv_lora_rank + qk_rope_head_dim
        kv_buffer = torch.randn(
            (num_blocks, block_size, num_kv_heads, qk_head_dim),
            dtype=torch.bfloat16,
            device="cuda",
        ).to(kv_dtype)
        query = torch.randn(
            (total_num_query_tokens, num_query_heads, qk_head_dim),
            dtype=torch.bfloat16,
            device="cuda",
        ).to(q_dtype)
        sm_scale = 1.0 / (qk_head_dim**0.5)

        q_descale = None
        kv_descale = None
        if q_dtype != torch.bfloat16:
            q_descale = torch.rand((1,), dtype=torch.float32, device="cuda")

        if kv_dtype != torch.bfloat16:
            kv_descale = torch.rand((1,), dtype=torch.float32, device="cuda")

        out_scale = None
        if use_out_scale:
            out_scale = 1 / torch.rand(1, dtype=torch.float32, device="cuda")

        output = torch.empty(
            (total_num_query_tokens, num_query_heads, kv_lora_rank), dtype=out_dtype
        )

        maybe_shuffled_kv_buffer = (
            shuffle_kv_buffer(kv_buffer, kv_lora_rank)
            if shuffled_kv_cache
            else kv_buffer
        )

        if decode_qlen > 0:
            out = mla_decode_fwd(
                query,
                maybe_shuffled_kv_buffer,
                output,
                cu_seqlens_q=cu_seqlens_q,
                seqused_k=seq_lens_kv,
                max_seqlen_kv=max_seqlen_kv,
                block_tables=block_tables,
                softmax_scale=sm_scale,
                kv_lora_rank=kv_lora_rank,
                qk_rope_head_dim=qk_rope_head_dim,
                causal=True,
                q_descale=q_descale,
                kv_descale=kv_descale,
                out_scale=out_scale,
                use_gluon=(backend == "gluon"),
                shuffled_kv_cache=shuffled_kv_cache,
            )
        else:
            out = mla_prefill_fwd(
                query,
                kv_buffer,
                out,
                cu_seqlens_q=cu_seqlens_q,
                seqused_k=seq_lens_kv,
                max_seqlen_kv=max_seqlen_kv,
                block_tables=block_tables,
                softmax_scale=sm_scale,
                kv_lora_rank=kv_lora_rank,
                qk_rope_head_dim=qk_rope_head_dim,
                causal=True,
                q_descale=q_descale,
                kv_descale=kv_descale,
                out_scale=out_scale,
                use_gluon=(backend == "gluon"),
                shuffled_kv_cache=shuffled_kv_cache,
            )

        mem_in = (
            query.numel() * query.itemsize
            + seq_lens_kv.sum().item()
            * num_kv_heads
            * (kv_lora_rank + qk_rope_head_dim)
            * 2
            * kv_dtype.itemsize
        )
        if decode_qlen > 0 and skip_reduce:
            assert (
                isinstance(out, tuple) and len(out) == 3
            ), "Output should be a tuple of 3 tensors for skip_reduce and decode_qlen > 0 1"
            segm_output, segm_max, segm_expsum = out
            mem_out = (
                segm_output.numel() * segm_output.itemsize
                + segm_max.numel() * segm_max.itemsize
                + segm_expsum.numel() * segm_expsum.itemsize
            )
        else:
            mem_out = out.numel() * query.itemsize
        mem = (mem_in + mem_out) * 1e-12

        def fn():
            if decode_qlen > 0:
                out = mla_decode_fwd(
                    query,
                    maybe_shuffled_kv_buffer,
                    output,
                    cu_seqlens_q=cu_seqlens_q,
                    seqused_k=seq_lens_kv,
                    max_seqlen_kv=max_seqlen_kv,
                    block_tables=block_tables,
                    softmax_scale=sm_scale,
                    kv_lora_rank=kv_lora_rank,
                    qk_rope_head_dim=qk_rope_head_dim,
                    causal=True,
                    q_descale=q_descale,
                    kv_descale=kv_descale,
                    out_scale=out_scale,
                    use_gluon=(backend == "gluon"),
                    shuffled_kv_cache=shuffled_kv_cache,
                )
            else:
                out = mla_prefill_fwd(
                    query,
                    kv_buffer,
                    out,
                    cu_seqlens_q=cu_seqlens_q,
                    seqused_k=seq_lens_kv,
                    max_seqlen_kv=max_seqlen_kv,
                    block_tables=block_tables,
                    softmax_scale=sm_scale,
                    kv_lora_rank=kv_lora_rank,
                    qk_rope_head_dim=qk_rope_head_dim,
                    causal=True,
                    q_descale=q_descale,
                    kv_descale=kv_descale,
                    out_scale=out_scale,
                    use_gluon=(backend == "gluon"),
                    shuffled_kv_cache=shuffled_kv_cache,
                )

        ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
        if "ms" in provider:
            return ms
        else:  # BW TB/s
            return mem / ms * 1e3

    bench_mla.run(save_path="." if args.o else None, print_data=True, show_plots=False)
    # return x_vals_list, x_names, line_vals


def parse_args():
    parser = argparse.ArgumentParser(
        prog="Benchmark MLA Decode/Prefill",
        allow_abbrev=False,
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument(
        "--decode_qlen", type=int, default=1
    )  # set to 0 to bench prefill
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
    parser.add_argument("--backend", type=str, default="triton")
    parser.add_argument("--skip_reduce", type=bool, default=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "-metric",
        nargs="?",
        const="bandwidth",
        choices=["time", "bandwidth"],
        default="bandwidth",
        help="Metrics for the kernel benchmark.",
    )
    parser.add_argument(
        "-o",
        action="store_true",
        default=False,
        help="Write performance results to CSV file",
    )

    return parser.parse_args()


def run_bench(args):
    torch.manual_seed(0)
    torch.set_default_device(args.device)
    benchmark(args)


def main():
    args = parse_args()
    run_bench(args)


if __name__ == "__main__":
    main()
