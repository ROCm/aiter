<<<<<<< HEAD
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# import hip
# hip.hip.hipInit(0)
from re import T
from typing import Optional

import triton
import torch

from aiter.ops.triton.attention.unified_attention import (
    unified_attention,
)
from op_tests.triton_tests.attention.test_unified_attention import shuffle_kv_cache
from aiter.ops.triton.utils.types import e4m3_dtype
import aiter.ops.triton.utils._triton.arch_info as arch_info
import argparse
from op_tests.op_benchmarks.triton.utils.benchmark_utils import (
    get_caller_name_no_ext,
)

DEVICE_ARCH = arch_info.get_arch()
IS_DEVICE_ARCH_GFX12 = DEVICE_ARCH in ("gfx1250",)


def benchmark(args):
    num_seqs = args.num_seqs
    q_len = args.q_len
    seq_lens = args.seq_lens
    HEAD_SIZE = args.head_size
    BLOCK_SIZE = args.block_size
    num_query_heads = args.num_query_heads
    num_kv_heads = args.num_kv_heads
    shuffled_kv_cache = args.shuffled_kv_cache
    q_dtype = e4m3_dtype if args.q_dtype == "fp8" else torch.bfloat16
    kv_dtype = e4m3_dtype if args.kv_dtype == "fp8" else torch.bfloat16
    skip_reduce = args.skip_reduce

    if shuffled_kv_cache:
        assert IS_DEVICE_ARCH_GFX12, "Gluon Unified Attention only supports gfx1250"
        assert BLOCK_SIZE >= 64, "Block size must be at least 64 for shuffled KV cache"

    num_tokens = num_seqs * seq_lens
    num_blocks = (num_tokens + BLOCK_SIZE - 1) // BLOCK_SIZE

    configs = []

    x_names = [
        "num_seqs",
        "q_len",
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
        "skip_reduce",
    ]

    x_vals_list = [
        (
            num_seqs,
            q_len,
            seq_lens,
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
            skip_reduce,
        )
    ]

    if args.metric == "time":
        unit = "ms"
    elif args.metric == "bandwidth":
        unit = "TB/s"
    else:
        raise ValueError("Unknown metric: " + args.metric)

    line_vals = [args.metric]
    line_names = [
        (
            "Gluon-TDM-shuffled "
            if shuffled_kv_cache
            else "Triton-nonpipelined-unshuffled "
        )
        + args.metric
    ]

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
    def bench_unified_attention(
        num_seqs: int,
        q_len: int,
        seq_lens: int,
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
        skip_reduce: bool,
        provider,
    ):
        warmup = 25
        rep = 100

        seq_lens_list = [(q_len, seq_lens)] * num_seqs

        num_seqs = len(seq_lens_list)
        query_lens = [x[0] for x in seq_lens_list]
        kv_lens = [x[1] for x in seq_lens_list]
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

        out = unified_attention(
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
            shuffled_kv_cache=shuffled_kv_cache,
            skip_reduce=skip_reduce,
        )
        mem_in = (
            query.numel() * query.itemsize
            + kv_lens.sum().item() * num_kv_heads * head_size * 2 * kv_dtype.itemsize
        )
        if q_len == 1 and skip_reduce:
            assert (
                isinstance(out, tuple) and len(out) == 3
            ), "Output should be a tuple of 3 tensors for skip_reduce and q_len == 1"
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
            unified_attention(
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
                shuffled_kv_cache=shuffled_kv_cache,
                skip_reduce=skip_reduce,
            )

        ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
        if "ms" in provider:
            return ms
        else:  # BW TB/s
            return mem / ms * 1e3

    bench_unified_attention.run(
        save_path="." if args.o else None, print_data=True, show_plots=False
    )
    # return x_vals_list, x_names, line_vals


def parse_args():
    parser = argparse.ArgumentParser(
        prog="Benchmark Unified Attention",
        allow_abbrev=False,
    )
    parser.add_argument("--num_seqs", type=int, default=32)
    parser.add_argument("--q_len", type=int, default=1)
    parser.add_argument("--seq_lens", type=int, default=8192)
    parser.add_argument("--head_size", type=int, default=64)
    parser.add_argument("--block_size", type=int, default=64)
    parser.add_argument("--shuffled_kv_cache", type=bool, default=True)
    parser.add_argument("--num_query_heads", type=int, default=64)
    parser.add_argument("--num_kv_heads", type=int, default=8)
    parser.add_argument("--q_dtype", type=str, default="bf16")
    parser.add_argument("--kv_dtype", type=str, default="bf16")
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
=======
import itertools
import sys

import torch
import triton

from aiter.ops.triton.attention.unified_attention import unified_attention
from aiter.ops.triton.utils.types import e4m3_dtype
from op_tests.op_benchmarks.triton.utils.argparse import get_parser
from op_tests.triton_tests.attention.test_unified_attention import ref_paged_attn

FP8_TYPE = e4m3_dtype
FP8_MAX = torch.finfo(FP8_TYPE).max


def default_benchmark_configs():
    batch_sizes = [1, 4, 8]
    n_heads = [16, 48]
    seq_len_q = [1, 1024, 4096]
    seq_len_k = [8192]
    head_dim = 128
    v_head_dim = head_dim
    configs = list(itertools.product(batch_sizes, n_heads, seq_len_q, seq_len_k))
    return [(bs, nh, nh, sq, sk, head_dim, v_head_dim) for bs, nh, sq, sk in configs]


def quantize_to_fp8(tensor):
    """Per-tensor symmetric FP8 quantization. Returns (quantized, descale)."""
    abs_max = tensor.abs().amax().clamp(min=1e-9)
    descale = (abs_max / FP8_MAX).to(torch.float32).unsqueeze(0).cuda()
    quantized = (tensor * (FP8_MAX / abs_max)).to(FP8_TYPE)
    return quantized, descale


def make_inputs(
    seq_lens,
    num_heads,
    head_size_qk,
    head_size_v,
    block_size,
    num_blocks,
    fp8_q,
    fp8_kv,
    fp8_output,
    out_scale_value,
):
    torch.cuda.empty_cache()
    torch.manual_seed(0)

    num_seqs = len(seq_lens)
    query_lens = [x[0] for x in seq_lens]
    kv_lens_list = [x[1] for x in seq_lens]
    num_query_heads, num_kv_heads = num_heads
    assert num_query_heads % num_kv_heads == 0

    max_query_len = max(query_lens)
    max_kv_len = max(kv_lens_list)
    scale = head_size_qk**-0.5

    query = torch.randn(
        sum(query_lens),
        num_query_heads,
        head_size_qk,
        dtype=torch.bfloat16,
        device="cuda",
    )
    key_cache = torch.randn(
        num_blocks,
        block_size,
        num_kv_heads,
        head_size_qk,
        dtype=torch.bfloat16,
        device="cuda",
    )
    value_cache = torch.randn(
        num_blocks,
        block_size,
        num_kv_heads,
        head_size_v,
        dtype=torch.bfloat16,
        device="cuda",
    )

    max_num_blocks_per_seq = (max_kv_len + block_size - 1) // block_size
    block_tables = torch.randint(
        0,
        num_blocks,
        (num_seqs, max_num_blocks_per_seq),
        dtype=torch.int32,
        device="cuda",
    )
    cu_query_lens = torch.tensor(
        [0] + query_lens,
        dtype=torch.int32,
        device="cuda",
    ).cumsum(dim=0, dtype=torch.int32)
    kv_lens = torch.tensor(kv_lens_list, dtype=torch.int32, device="cuda")
    query_lens_t = torch.tensor(query_lens, dtype=torch.int32, device="cuda")

    q_fp8, q_descale = quantize_to_fp8(query) if fp8_q else (None, None)
    k_fp8, k_descale = quantize_to_fp8(key_cache) if fp8_kv else (None, None)
    v_fp8, v_descale = quantize_to_fp8(value_cache) if fp8_kv else (None, None)

    out_scale = None
    out_dtype = torch.bfloat16
    if fp8_output:
        out_scale = torch.tensor([out_scale_value], dtype=torch.float32, device="cuda")
        out_dtype = FP8_TYPE

    output = torch.empty(
        cu_query_lens[-1].item(),
        num_query_heads,
        head_size_v,
        dtype=out_dtype,
        device="cuda",
    )

    return dict(
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        q_fp8=q_fp8,
        k_fp8=k_fp8,
        v_fp8=v_fp8,
        q_descale=q_descale,
        k_descale=k_descale,
        v_descale=v_descale,
        out_scale=out_scale,
        output=output,
        cu_query_lens=cu_query_lens,
        kv_lens=kv_lens,
        query_lens=query_lens_t,
        block_tables=block_tables,
        max_query_len=max_query_len,
        max_kv_len=max_kv_len,
        scale=scale,
    )


def _mode_label(args):
    parts = []
    if args.fp8:
        parts.append("fp8")
    elif args.fp8_kv:
        parts.append("fp8_kv")
    else:
        parts.append("bf16")
    if args.fp8_output:
        parts.append("fp8out")
    return "_".join(parts) + "_fwd"


def run_benchmark(custom, args):
    torch.manual_seed(20)

    any_fp8 = args.fp8 or args.fp8_kv or args.fp8_output
    label = _mode_label(args)

    def create_configs():
        hk = args.hq if not args.hk else args.hk
        sk = args.sq if not args.sk else args.sk
        head_size = 128 if not args.d else args.d
        head_size_v = head_size if not args.dv else args.dv
        decode_p = args.decode

        x_names = [
            "BATCH",
            "HQ",
            "HK",
            "N_CTX_Q",
            "N_CTX_K",
            "D_HEAD",
            "D_HEAD_V",
            "DECODE_P",
        ]

        if isinstance(args.sq, list):
            batch_size = len(args.sq)
        elif isinstance(args.sk, list):
            batch_size = len(args.sk)
        else:
            batch_size = args.b if args.b else 1

        if custom:
            x_vals_list = [
                (batch_size, args.hq, hk, args.sq, sk, head_size, head_size_v)
            ]
        else:
            x_vals_list = default_benchmark_configs()

        x_vals_list = [(*v, decode_p) for v in x_vals_list]

        unit = {"time": "ms", "throughput": "TFLOPS", "bandwidth": "GB/s"}[args.metric]

        return [
            triton.testing.Benchmark(
                x_names=x_names,
                x_vals=x_vals_list,
                line_arg="provider",
                line_vals=[label],
                line_names=[label],
                styles=[("red", "-")],
                ylabel=unit,
                plot_name=f"bench_unified_attention_{label}",
                args={},
            )
        ]

    @triton.testing.perf_report(create_configs())
    def bench_fn(
        BATCH,
        HQ,
        HK,
        N_CTX_Q,
        N_CTX_K,
        D_HEAD,
        D_HEAD_V,
        DECODE_P,
        provider,
    ):
        varlen = not args.equal_seqlens

        if isinstance(N_CTX_Q, list):
            seqlens_q = torch.tensor(N_CTX_Q, dtype=torch.int32, device="cuda")
        elif varlen:
            seqlens_q = torch.randint(
                1, N_CTX_Q + 1, (BATCH,), dtype=torch.int32, device="cuda"
            )
        else:
            seqlens_q = torch.full((BATCH,), N_CTX_Q, dtype=torch.int32, device="cuda")

        if isinstance(N_CTX_K, list):
            seqlens_k = torch.tensor(N_CTX_K, dtype=torch.int32, device="cuda")
        elif varlen:
            seqlens_k = torch.randint(
                1, N_CTX_K + 1, (BATCH,), dtype=torch.int32, device="cuda"
            )
        else:
            seqlens_k = torch.full((BATCH,), N_CTX_K, dtype=torch.int32, device="cuda")

        seqlens_k = torch.maximum(seqlens_k, seqlens_q)

        if DECODE_P > 0.0:
            num_decode = int(round(DECODE_P * BATCH))
            if num_decode > 0:
                decode_idx = torch.randperm(BATCH, device=seqlens_q.device)[:num_decode]
                seqlens_q[decode_idx] = 1

        block_size = args.block_size if args.block_size else 512
        max_num_blocks_per_seq = (seqlens_k.max().item() + block_size - 1) // block_size
        min_required_blocks = BATCH * max_num_blocks_per_seq
        num_blocks = (
            args.num_blocks if args.num_blocks else max(min_required_blocks * 4, 2048)
        )

        inputs = make_inputs(
            seq_lens=list(zip(seqlens_q.tolist(), seqlens_k.tolist())),
            num_heads=(HQ, HK),
            head_size_qk=D_HEAD,
            head_size_v=D_HEAD_V,
            block_size=block_size,
            num_blocks=num_blocks,
            fp8_q=args.fp8,
            fp8_kv=args.fp8 or args.fp8_kv,
            fp8_output=args.fp8_output,
            out_scale_value=args.out_scale,
        )

        q_tensor = inputs["q_fp8"] if args.fp8 else inputs["query"]
        k_tensor = inputs["k_fp8"] if (args.fp8 or args.fp8_kv) else inputs["key_cache"]
        v_tensor = (
            inputs["v_fp8"] if (args.fp8 or args.fp8_kv) else inputs["value_cache"]
        )

        window_size = (
            (args.sliding_window - 1, 0)
            if args.sliding_window is not None
            else (-1, -1)
        )

        def fn():
            return unified_attention(
                q=q_tensor,
                k=k_tensor,
                v=v_tensor,
                out=inputs["output"],
                cu_seqlens_q=inputs["cu_query_lens"],
                seqused_k=inputs["kv_lens"],
                max_seqlen_q=inputs["max_query_len"],
                max_seqlen_k=inputs["max_kv_len"],
                softmax_scale=inputs["scale"],
                causal=True,
                window_size=window_size,
                block_table=inputs["block_tables"],
                softcap=0,
                q_descale=inputs["q_descale"],
                k_descale=inputs["k_descale"],
                v_descale=inputs["v_descale"],
                output_scale=inputs["out_scale"],
            )

        ms = triton.testing.do_bench_cudagraph(fn)

        if args.test:
            fn()
            ref_output = ref_paged_attn(
                query=inputs["query"],
                key_cache=inputs["key_cache"],
                value_cache=inputs["value_cache"],
                query_lens=inputs["query_lens"],
                kv_lens=inputs["kv_lens"],
                block_tables=inputs["block_tables"],
                scale=inputs["scale"],
                sliding_window=args.sliding_window,
                soft_cap=None,
                q_descale=inputs["q_descale"],
                k_descale=inputs["k_descale"],
                v_descale=inputs["v_descale"],
                output_scale=inputs["out_scale"],
                out_dtype=inputs["output"].dtype,
            )
            if any_fp8:
                atol, rtol = 1.5e-1, 1.5e-1
            else:
                atol, rtol = 1.5e-2, 1e-2
            out_f32 = inputs["output"].to(torch.float32)
            ref_f32 = ref_output.to(torch.float32)
            max_diff = torch.max(torch.abs(out_f32 - ref_f32)).item()
            shape_str = f"(B={BATCH}, HQ={HQ}, HK={HK}, SQ={N_CTX_Q}, SK={N_CTX_K}, D={D_HEAD}, DV={D_HEAD_V})"
            try:
                torch.testing.assert_close(out_f32, ref_f32, atol=atol, rtol=rtol)
                print(f"  PASS {shape_str}  max_diff={max_diff:.6f}")
            except AssertionError as e:
                print(f"  FAIL {shape_str}  max_diff={max_diff:.6f}")
                print(f"    {e}")

        cu_query_lens = inputs["cu_query_lens"]
        num_contexts = len(cu_query_lens) - 1
        total_flops = 0.0
        for i in range(num_contexts):
            sq = (cu_query_lens[i + 1] - cu_query_lens[i]).item()
            sk = seqlens_k[i].item()
            valid = sq * sk - ((sq**2 - sq) / 2)
            total_flops += valid * HQ * (D_HEAD + D_HEAD_V) * 2.0

        total_q = cu_query_lens[-1].item()
        total_k = seqlens_k.sum().item()
        q_bytes = total_q * HQ * D_HEAD * q_tensor.element_size()
        k_bytes = total_k * HK * D_HEAD * k_tensor.element_size()
        v_bytes = total_k * HK * D_HEAD_V * v_tensor.element_size()
        o_bytes = total_q * HQ * D_HEAD_V * inputs["output"].element_size()
        mem = q_bytes + k_bytes + v_bytes + o_bytes

        if args.metric == "time":
            return ms
        elif args.metric == "throughput":
            return total_flops / ms * 1e-9
        elif args.metric == "bandwidth":
            return mem / ms * 1e-6
        else:
            raise ValueError(f"Unknown metric: {args.metric}")

    bench_fn.run(None, print_data=True)


def parse_int_or_list(value):
    if "," in value:
        return [int(x) for x in value.split(",")]
    return int(value)


def parse_args():
    parser = get_parser(kernel_name="Unified Attention")

    parser.add_argument("-b", type=int, default=0)
    parser.add_argument("-hq", type=int, default=0)
    parser.add_argument("-hk", type=int, default=0)
    parser.add_argument(
        "-sq",
        type=parse_int_or_list,
        default=0,
        help="Query sequence length (single int or comma-separated list)",
    )
    parser.add_argument(
        "-sk",
        type=parse_int_or_list,
        default=0,
        help="Key sequence length (single int or comma-separated list, defaults to sq if 0)",
    )
    parser.add_argument("-d", type=int, default=0, help="Q/K head size")
    parser.add_argument("-dv", type=int, default=0, help="V head size (defaults to -d)")
    parser.add_argument(
        "-num_blocks", type=int, default=0, help="KV cache blocks (0=auto)"
    )
    parser.add_argument("-block_size", type=int, default=0, help="KV cache block size")
    parser.add_argument(
        "-test",
        action="store_true",
        default=False,
        help="Verify correctness against reference implementation for each shape",
    )
    parser.add_argument(
        "-equal_seqlens",
        action="store_true",
        default=False,
        help="Use equal sequence lengths (no varlen); default is random varlen",
    )
    parser.add_argument(
        "-fp8",
        action="store_true",
        default=False,
        help="Quantize Q, K, V to FP8 e4m3 with per-tensor descales",
    )
    parser.add_argument(
        "-fp8_kv",
        action="store_true",
        default=False,
        help="Quantize only K, V to FP8 e4m3 (Q stays bf16)",
    )
    parser.add_argument(
        "-fp8_output",
        action="store_true",
        default=False,
        help="Output tensor in FP8 with output_scale",
    )
    parser.add_argument(
        "-out_scale",
        type=float,
        default=1.0,
        help="Output scale factor when -fp8_output is set (default: 1.0)",
    )
    parser.add_argument(
        "-decode",
        nargs="?",
        const=1.0,
        default=0.0,
        type=float,
        metavar="P",
        help="Portion of decode samples (seqlen_q=1) in batch; omit P for all=1.0",
    )
    parser.add_argument(
        "-sliding_window",
        type=int,
        default=None,
        help="Sliding window size (default: disabled)",
>>>>>>> origin/main
    )

    return parser.parse_args()


<<<<<<< HEAD
def run_bench(args):
    torch.manual_seed(0)
    torch.set_default_device(args.device)
    benchmark(args)


def main():
    args = parse_args()
    run_bench(args)


if __name__ == "__main__":
    main()
=======
def main():
    args = parse_args()

    if args.fp8 and args.fp8_kv:
        raise ValueError(
            "-fp8 already quantizes K/V; -fp8_kv is redundant. Use one or the other."
        )

    custom_config = False

    if args.hq or args.hk or args.d or args.dv:
        custom_config = True
        if not args.dv:
            args.dv = args.d
        assert (
            args.b and args.hq and args.sq and args.d and args.dv
        ), "Custom config requires -b, -hq, -sq, -d (and optionally -dv)"

    run_benchmark(custom_config, args)


if __name__ == "__main__":
    sys.exit(main())
>>>>>>> origin/main
