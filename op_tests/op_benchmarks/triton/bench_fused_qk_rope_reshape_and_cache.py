# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""
Benchmark for the triton fused_qk_rope_reshape_and_cache kernel.

Triton only: untested on gluon/gfx12

Default shapes mirror the shapes obtained ATOM makes testing gpt-oss-120b
using prefill at TP1 with a bf16 KV cache and concurrency 16, for simplicity:

    q=[M, 64, 64]  k=[M, 8, 64]  v=[M, 8, 64]   views into one packed QKV buffer
    key_cache=[169788, 8, 8, 16, 8]             non-flash, x=8
    value_cache=[169788, 8, 2, 64, 8]           shuffled value layout
    cos/sin=[131072, 1, 1, 32]                  neox, front-half freqs reused
    flash_layout=False, apply_scale=False, output_zeros=False

Metrics: time (default), throughput, bandwidth.

Usage Example:
    python bench_fused_qk_rope_reshape_and_cache.py --metric bandwidth -M 1024,2048,4096,8172
"""

import argparse

import torch
import triton

from aiter.ops.triton.fusions.fused_kv_cache import fused_qk_rope_reshape_and_cache
from aiter.ops.triton.utils import arch_info
from aiter.ops.triton.utils.types import e4m3_dtype
from op_tests.op_benchmarks.triton.utils.argparse import get_parser
from op_tests.op_benchmarks.triton.utils.benchmark_utils import (
    get_caller_name_no_ext,
)

DEVICE_ARCH = arch_info.get_arch()

CACHE_DTYPES = {
    "bf16": torch.bfloat16,
    "fp8": e4m3_dtype,
}

CACHE_LAYOUTS = ("flash", "nonflash", "nonflash_v_shuffle")

SLOT_PATTERNS = ("blocked", "random")

DEFAULT_M = [15360]  # test prefill
DEFAULT_QH = 64
DEFAULT_KH = 8  # 64 / TP8
DEFAULT_D = 64
# gpt-oss-120b, TP1, bf16 cache, block 16.
DEFAULT_NUM_BLOCKS = 169788
DEFAULT_BLOCK_SIZE = 16

X_NAMES = ["M", "QH", "KH", "D", "block_size"]


def get_x_vals(args):
    m_vals = args.M if isinstance(args.M, list) else [args.M]
    return [(M, args.QH, args.KH, args.D, args.block_size) for M in m_vals]


def bench_qk_rope_fn(
    M,
    QH,
    KH,
    D,
    block_size,
    metric,
    args,
    **kwargs,
):
    # cache str -> torch type
    cache_dtype = CACHE_DTYPES[args.cache_dtype]
    num_blocks = args.num_blocks

    # paged cache chunk width
    x_size = 16 // cache_dtype.itemsize

    # max position ~= max(ISL) + OSL.
    max_pos = args.isl + args.osl

    assert (
        args.isl > 0 and args.osl > 0
    ), f"ISL > 0 and OSL > 0, got: {args.isl=} {args.osl=}"
    # powers of 2 for triton
    assert D == triton.next_power_of_2(D), f"D must be a power of 2, got {D=}"
    assert block_size == triton.next_power_of_2(
        block_size
    ), f"block_size must be a power of 2, got {block_size=}"
    # check enough room in cache
    assert (
        M <= num_blocks * block_size
    ), f"Not enough cache slots for {M} tokens: {num_blocks=} {block_size=}"

    assert DEVICE_ARCH != "gfx1250", "Only triton kernel tested/supported"

    # ensure non-flash is divisible by chunk size
    if args.cache_layout != "flash":
        assert (
            D % x_size == 0
        ), f"Non-flash key cache layout needs D % {x_size} == 0, got {D=}"
    if args.cache_layout == "nonflash_v_shuffle":
        assert (
            block_size % x_size == 0
        ), f"Shuffled value cache layout needs block_size % {x_size} == 0, got {block_size=}"

    torch.cuda.empty_cache()

    dtype = torch.bfloat16  # qkv dtype

    # assumed contiguous, split into 3
    qkv = torch.randn((M, (QH + 2 * KH) * D), dtype=dtype, device="cuda")
    q_flat, k_flat, v_flat = torch.split(qkv, [QH * D, KH * D, KH * D], dim=-1)
    q = q_flat.view(M, QH, D)
    k = k_flat.view(M, KH, D)
    v = v_flat.view(M, KH, D)

    # mock rotation tables
    d_freq = D // 2 if args.reuse_freqs_front_part else D
    freqs = torch.randn((max_pos, 1, 1, d_freq), dtype=dtype, device="cuda")
    cos = torch.cos(freqs)
    sin = torch.sin(freqs)

    positions = torch.randint(0, max_pos, (M,), device="cuda")

    # create mock caches
    if args.cache_layout == "flash":
        key_cache = torch.zeros(
            (num_blocks, block_size, KH, D), dtype=cache_dtype, device="cuda"
        )
        value_cache = torch.zeros_like(key_cache)
    else:
        key_cache = torch.zeros(
            (num_blocks, KH, D // x_size, block_size, x_size),
            dtype=cache_dtype,
            device="cuda",
        )
        if args.cache_layout == "nonflash_v_shuffle":
            value_cache = torch.zeros(
                (num_blocks, KH, block_size // x_size, D, x_size),
                dtype=cache_dtype,
                device="cuda",
            )
        else:
            value_cache = torch.zeros(
                (num_blocks, KH, D, block_size), dtype=cache_dtype, device="cuda"
            )

    # map slots, blocked (prefill) or random (decode)
    if args.slot_pattern == "blocked":
        n_blocks = triton.cdiv(M, block_size)
        blocks = torch.randperm(num_blocks, device="cuda")[:n_blocks]
        within = torch.arange(block_size, device="cuda")
        slot_mapping = (blocks[:, None] * block_size + within[None, :]).reshape(-1)[:M]
    else:
        slot_mapping = torch.randperm(num_blocks * block_size, device="cuda")[:M]
    slot_mapping = slot_mapping.contiguous()

    # scale only applied with fp8
    apply_scale = cache_dtype != torch.bfloat16
    if apply_scale:
        k_scale = torch.ones((), dtype=torch.float32, device="cuda")
        v_scale = torch.ones((), dtype=torch.float32, device="cuda")
    else:
        k_scale = None
        v_scale = None

    # HAVE_ZEROS in caller
    zeros_out = torch.empty_like(q) if args.output_zeros else None

    ms = triton.testing.do_bench_cudagraph(
        lambda: fused_qk_rope_reshape_and_cache(
            q=q,
            k=k,
            v=v,
            key_cache=key_cache,
            value_cache=value_cache,
            slot_mapping=slot_mapping,
            pos=positions,
            cos=cos,
            sin=sin,
            k_scale=k_scale,
            v_scale=v_scale,
            is_neox=args.rotate_style == "neox",
            flash_layout=args.cache_layout == "flash",
            apply_scale=apply_scale,
            # RoPE is written back in place, as the model call site does.
            q_out=q,
            k_out=k,
            output_zeros=args.output_zeros,
            zeros_out=zeros_out,
            upcast_operand=args.upcast_operand,
        ),
        return_mode="median",
    )

    flops = M * QH * D * 3 + M * KH * D * 3  # rope

    elem = q.element_size()

    # q + k + v
    mem_read = M * QH * D * elem + 2 * M * KH * D * elem

    cache_bytes_per_token = KH * D * key_cache.element_size()

    # q + k + kv cache
    mem_write = M * QH * D * elem + M * KH * D * elem + 2 * M * cache_bytes_per_token

    mem = mem_read + mem_write

    if metric == "time":
        return ms
    elif metric == "throughput":
        return flops / ms * 1e-9  # TFLOPS
    elif metric == "bandwidth":
        return mem / ms * 1e-6  # GB/s
    else:
        raise ValueError("Unknown metric: " + metric)


def run_benchmark(args):
    x_vals_list = get_x_vals(args)

    metric_to_unit = {
        "time": "Time_(ms)",
        "throughput": "TFLOPS",
        "bandwidth": "Bandwidth_(GB/s)",
    }
    metric_to_ylabel = {
        "time": "Time (ms)",
        "throughput": "Throughput (TFLOPS)",
        "bandwidth": "Bandwidth (GB/s)",
    }
    if args.metric not in metric_to_unit:
        raise NotImplementedError(f"{args.metric} is not supported")
    unit = metric_to_unit[args.metric]

    benchmark = triton.testing.Benchmark(
        x_names=X_NAMES,
        x_vals=x_vals_list,
        line_arg="provider",
        line_vals=[unit],
        line_names=[unit],
        styles=[("green", "-")],
        ylabel=metric_to_ylabel[args.metric],
        plot_name=get_caller_name_no_ext(),
        args={"metric": args.metric, "args": args},
    )

    triton.testing.perf_report([benchmark])(bench_qk_rope_fn).run(print_data=True)


def parse_int_or_list(value):
    if "," in value:
        return [int(x) for x in value.split(",")]
    return int(value)


def parse_args(args: list[str] | None = None):
    parser = get_parser(kernel_name="fused_qk_rope_reshape_and_cache")
    parser.set_defaults(metric="time")
    parser.add_argument(
        "-M",
        type=parse_int_or_list,
        default=DEFAULT_M,
        help="Number of tokens (single int or comma-separated list for multiple)",
    )
    parser.add_argument("-QH", type=int, default=DEFAULT_QH, help="Number of Q heads")
    parser.add_argument("-KH", type=int, default=DEFAULT_KH, help="Number of KV heads")
    parser.add_argument("-D", type=int, default=DEFAULT_D, help="Head dimension")
    parser.add_argument(
        "--num_blocks",
        type=int,
        default=DEFAULT_NUM_BLOCKS,
        help="Number of KV cache blocks",
    )
    parser.add_argument(
        "--block_size",
        type=int,
        default=DEFAULT_BLOCK_SIZE,
        help="KV cache block size",
    )
    parser.add_argument(
        "--isl",
        type=int,
        default=1024,
        help="Input sequence length. With --osl this bounds the positions, and "
        "so the number of rows in the cos/sin table",
    )
    parser.add_argument(
        "--osl",
        type=int,
        default=1024,
        help="Output sequence length",
    )
    parser.add_argument(
        "--cache_dtype",
        type=str,
        choices=list(CACHE_DTYPES),
        default="bf16",
        help="KV cache dtype",
    )
    parser.add_argument(
        "--cache_layout",
        type=str,
        choices=CACHE_LAYOUTS,
        default="nonflash_v_shuffle",
        help="KV cache layout",
    )
    parser.add_argument(
        "--slot_pattern",
        type=str,
        choices=SLOT_PATTERNS,
        default="blocked",
        help="How slot_mapping scatters tokens over the cache (blocked = prefill)",
    )
    parser.add_argument(
        "--rotate_style",
        type=str,
        choices=["gptj", "neox"],
        default="neox",
        help="RoPE rotate style, gptj/neox",
    )
    parser.add_argument(
        "--reuse_freqs_front_part",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="cos/sin hold only the front half of the frequencies (d_freq == D // 2)",
    )
    parser.add_argument(
        "--output_zeros",
        action="store_true",
        default=False,
        help="Also write the zeros_out tensor",
    )
    parser.add_argument(
        "--upcast_operand",
        action="store_true",
        default=False,
        help="Upcast the RoPE operands to fp32 inside the kernel",
    )
    return parser.parse_args(args=args)


def main(args: list[str] | None = None) -> None:
    parsed_args = parse_args(args=args)
    torch.manual_seed(0)
    run_benchmark(parsed_args)


if __name__ == "__main__":
    main()
