# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# import hip
# hip.hip.hipInit(0)
from re import T
from typing import Optional

import triton
import torch

from aiter.ops.triton.attention.unified_attention import (
    unified_attention as triton_unified_attention,
)
from aiter.ops.triton.gluon.unified_attention_3d import (
    unified_attention as gluon_unified_attention,
)
from op_tests.triton_tests.attention.test_unified_attention import shuffle_kv_cache
from aiter.ops.triton.utils.types import e4m3_dtype
import aiter.ops.triton.utils._triton.arch_info as arch_info
import argparse

DEVICE_ARCH = arch_info.get_arch()
IS_DEVICE_ARCH_GFX12 = DEVICE_ARCH in ("gfx1250",)


def benchmark(args):
    num_seqs = args.num_seqs
    seq_lens = args.seq_lens
    HEAD_SIZE = args.head_size
    BLOCK_SIZE = args.block_size
    num_query_heads = args.num_query_heads
    num_kv_heads = args.num_kv_heads
    shuffled_kv_cache = args.shuffled_kv_cache
    q_dtype = e4m3_dtype if args.q_dtype == "fp8" else torch.bfloat16
    kv_dtype = e4m3_dtype if args.kv_dtype == "fp8" else torch.bfloat16
    backend = args.backend

    if backend == "gluon":
        assert IS_DEVICE_ARCH_GFX12, "Gluon Unified Attention only supports gfx1250"
    if shuffled_kv_cache:
        assert BLOCK_SIZE >= 64, "Block size must be at least 64 for shuffled KV cache"

    num_tokens = num_seqs * seq_lens
    num_blocks = (num_tokens + BLOCK_SIZE - 1) // BLOCK_SIZE

    configs = []

    x_names = [
        "seq_lens",
        "num_heads",
        "head_size",
        "sliding_window",
        "block_size",
        "soft_cap",
        "num_blocks",
        "q_dtype",
        "kv_dtype",
        "o_dtype",
        "shuffled_kv_cache",
        "backend",
    ]

    x_vals_list = [
        (
            [(1, int(seq_lens))] * int(num_seqs),
            (num_query_heads, num_kv_heads),
            HEAD_SIZE,
            None,
            BLOCK_SIZE,
            None,
            num_blocks,
            q_dtype,
            kv_dtype,
            torch.bfloat16,
            shuffled_kv_cache,
            backend,
        )
    ]

    configs.append(
        triton.testing.Benchmark(
            x_names=x_names,
            x_vals=x_vals_list,
            # line_arg="provider",
            # line_vals=line_vals,
            # line_names=line_vals,
            styles=[("red", "-"), ("green", "-")],
            ylabel="ms",
            plot_name="unified_attention",
            # args={"sm_scale": 1.0, "logit_cap": 0.0, "device": args.device},
        )
    )

    @triton.testing.perf_report(configs)
    def bench_unified_attention(
        seq_lens: list[tuple[int, int]],
        num_heads: tuple[int, int],
        head_size: int,
        sliding_window: Optional[int],
        block_size: int,
        soft_cap: Optional[float],
        num_blocks: int,
        q_dtype: torch.dtype,
        kv_dtype: torch.dtype,
        o_dtype: torch.dtype,
        shuffled_kv_cache: bool,
        backend: str,
    ):
        warmup = 25
        rep = 100

        num_seqs = len(seq_lens)
        query_lens = [x[0] for x in seq_lens]
        kv_lens = [x[1] for x in seq_lens]
        num_query_heads = num_heads[0]
        num_kv_heads = num_heads[1]
        assert num_query_heads % num_kv_heads == 0
        max_query_len = max(query_lens)
        max_kv_len = max(kv_lens)
        window_size = (
            (sliding_window - 1, 0) if sliding_window is not None else (-1, -1)
        )
        scale = head_size**-0.5

        query = torch.randn(
            sum(query_lens),
            num_query_heads,
            head_size,
            dtype=torch.bfloat16,
            device="cuda",
        ).to(q_dtype)
        key_cache = torch.randn(
            num_blocks,
            block_size,
            num_kv_heads,
            head_size,
            dtype=torch.bfloat16,
            device="cuda",
        ).to(kv_dtype)
        value_cache = torch.randn(
            num_blocks,
            block_size,
            num_kv_heads,
            head_size,
            dtype=torch.bfloat16,
            device="cuda",
        ).to(kv_dtype)
        cu_query_lens = torch.tensor(
            [0] + query_lens, dtype=torch.int32, device="cuda"
        ).cumsum(dim=0, dtype=torch.int32)
        kv_lens = torch.tensor(kv_lens, dtype=torch.int32, device="cuda")

        max_num_blocks_per_seq = (max_kv_len + block_size - 1) // block_size
        # block_tables = torch.randint(
        #     0,
        #     num_blocks,
        #     (num_seqs, max_num_blocks_per_seq),
        #     dtype=torch.int32,
        #     device="cuda",
        # )
        assert num_seqs * max_num_blocks_per_seq == num_blocks
        block_tables = torch.randperm(
            num_blocks, dtype=torch.int32, device="cuda"
        ).reshape(num_seqs, max_num_blocks_per_seq)
        sinks = torch.randn(num_query_heads, dtype=torch.bfloat16, device="cuda")
        output = torch.randn(
            sum(query_lens), num_query_heads, head_size, dtype=o_dtype, device="cuda"
        )

        q_descale = None
        k_descale = None
        v_descale = None
        if q_dtype != torch.bfloat16:
            q_descale = torch.rand((1,), dtype=torch.float32, device="cuda")

        if kv_dtype != torch.bfloat16:
            k_descale = torch.rand((1,), dtype=torch.float32, device="cuda")
            v_descale = torch.rand((1,), dtype=torch.float32, device="cuda")

        if shuffled_kv_cache:
            maybe_shuffled_key_cache, maybe_shuffled_value_cache = shuffle_kv_cache(
                key_cache, value_cache
            )
        else:
            maybe_shuffled_key_cache = key_cache
            maybe_shuffled_value_cache = value_cache

        def fn():
            if backend == "gluon":
                impl = gluon_unified_attention
                kwargs = {
                    "use_tdm": True,
                    "num_tdm_gather": 1,
                    "use_async": False,
                    "shuffled_kv_cache": shuffled_kv_cache,
                }
            else:
                impl = triton_unified_attention
                kwargs = {
                    "shuffled_kv_cache": shuffled_kv_cache,
                }
            impl(
                q=query,
                k=maybe_shuffled_key_cache,
                v=maybe_shuffled_value_cache,
                out=output,
                cu_seqlens_q=cu_query_lens,
                seqused_k=kv_lens,
                max_seqlen_q=max_query_len,
                max_seqlen_k=max_kv_len,
                softmax_scale=scale,
                causal=True,
                window_size=window_size,
                block_table=block_tables,
                softcap=soft_cap if soft_cap is not None else 0,
                q_descale=q_descale,
                k_descale=k_descale,
                v_descale=v_descale,
                sinks=sinks,
                **kwargs,
            )

        ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
        return ms

    bench_unified_attention.run(
        save_path="." if args.o else None, print_data=True, show_plots=False
    )
    return x_vals_list, x_names, line_vals


def parse_args():
    parser = argparse.ArgumentParser(
        prog="Benchmark Unified Attention",
        allow_abbrev=False,
    )
    parser.add_argument("--num_seqs", type=int, default=32)
    parser.add_argument("--seq_lens", type=int, default=8192)
    parser.add_argument("--head_size", type=int, default=64)
    parser.add_argument("--block_size", type=int, default=64)
    parser.add_argument("--shuffled_kv_cache", type=bool, default=True)
    parser.add_argument("--num_query_heads", type=int, default=64)
    parser.add_argument("--num_kv_heads", type=int, default=8)
    parser.add_argument("--q_dtype", type=str, default="fp8")
    parser.add_argument("--kv_dtype", type=str, default="fp8")
    parser.add_argument("--backend", type=str, default="gluon")
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
    torch.set_default_device(args.device)
    benchmark(args)


def main():
    args = parse_args()
    run_bench(args)


if __name__ == "__main__":
    main()
