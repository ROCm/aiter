# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""
Benchmark for fused_qk_rope_cat_and_cache_mla with q_nope_prestored=True.

With q_nope_prestored, a q_nope producer (e.g. a bf16-absorb k-up GEMM) has
already written q_nope into q_out[..., :D_lora], so the op only ropes q_pe into
q_out[..., D_lora:] and writes the KV cache.

Default shapes mirror DeepSeek-R1 MLA decode with the ATOM fp8 shuffled KV
cache (128 heads):

    q_nope=[M, QH, 512]   (only used for shapes when prestored)
    q_pe=[M, QH, 64]
    k_nope=[M_slot, 1, 512], k_pe=[M_slot, 1, 64]
    kv_cache=[num_blocks, 1, 64, 576] fp8, shuffled, block_size=64
    q_out=[M, QH, 576] fp8, k_scale applied, GPT-J rope, front-half freqs

M_slot = M + k_only: --k_only adds tokens that only carry k (mixed batch).

Providers:
    full            q_nope_prestored=False (reads q_nope, writes all of q_out)
    prestored       q_nope_prestored=True, default dispatch (gfx1250 gluon
                    head-tiled kernel on pure decode, else Triton)
    prestored_triton  q_nope_prestored=True, Triton kernel forced

Metrics: time (default), bandwidth.

Usage Example:
    python bench_fused_qpe_rope_and_cache_mla.py -M 64,256,1536,4096
    python bench_fused_qpe_rope_and_cache_mla.py --k_only 333 -M 1024
"""

import torch
import triton

import aiter.ops.triton.fusions.fused_kv_cache as fused_kv_cache_mod
from aiter.ops.triton.fusions.fused_kv_cache import fused_qk_rope_cat_and_cache_mla
from aiter.ops.triton.utils.types import e4m3_dtype
from op_tests.op_benchmarks.triton.utils.argparse import get_parser
from op_tests.op_benchmarks.triton.utils.benchmark_utils import (
    get_caller_name_no_ext,
)

CACHE_DTYPES = {
    "bf16": torch.bfloat16,
    "fp8": e4m3_dtype,
}

PROVIDERS = ("full", "prestored", "prestored_triton")

DEFAULT_M = [1, 64, 256, 1024, 1536, 4096]
DEFAULT_QH = 128  # DeepSeek-R1
DEFAULT_D_LORA = 512
DEFAULT_D_PE = 64
DEFAULT_NUM_KV_CACHE_TOKENS = 262144

X_NAMES = ["M", "QH", "D_lora", "D_pe"]


def get_x_vals(args):
    m_vals = args.M if isinstance(args.M, list) else [args.M]
    return [(M, args.QH, args.D_lora, args.D_pe) for M in m_vals]


def bench_qpe_rope_and_cache_mla_fn(
    M,
    QH,
    D_lora,
    D_pe,
    provider,
    metric,
    args,
    **kwargs,
):
    cache_dtype = CACHE_DTYPES[args.cache_dtype]
    shuffled = not args.unshuffled
    block_size = 64 if shuffled else 1
    num_kv_cache_tokens = args.num_kv_cache_tokens
    m_slot = M + args.k_only
    assert (
        m_slot <= num_kv_cache_tokens
    ), f"Not enough cache slots for {m_slot} tokens: {num_kv_cache_tokens=}"

    torch.cuda.empty_cache()
    dtype = torch.bfloat16

    q_nope = torch.randn((M, QH, D_lora), dtype=dtype, device="cuda")
    q_pe = torch.randn((M, QH, D_pe), dtype=dtype, device="cuda")
    k_nope = torch.randn((m_slot, 1, D_lora), dtype=dtype, device="cuda")
    k_pe = torch.randn((m_slot, 1, D_pe), dtype=dtype, device="cuda")
    if shuffled:
        kv_cache = torch.zeros(
            (num_kv_cache_tokens // block_size, 1, block_size, D_lora + D_pe),
            dtype=cache_dtype,
            device="cuda",
        )
    else:
        kv_cache = torch.zeros(
            (num_kv_cache_tokens, 1, D_lora + D_pe), dtype=cache_dtype, device="cuda"
        )
    slot_mapping = torch.randperm(num_kv_cache_tokens, device="cuda")[:m_slot]
    freqs = torch.randn((args.max_pos, D_pe // 2), dtype=dtype, device="cuda")
    cos = torch.cos(freqs)
    sin = torch.sin(freqs)
    positions = torch.randint(0, args.max_pos, (m_slot,), device="cuda")
    k_scale = torch.ones([1], dtype=torch.float32, device="cuda")[0]
    q_out = torch.empty((M, QH, D_lora + D_pe), dtype=cache_dtype, device="cuda")
    prestored = provider != "full"
    if prestored:
        q_out[..., :D_lora] = q_nope.to(cache_dtype)

    fused_kv_cache_mod._QPE_ROPE_IMPL = (
        "triton" if provider == "prestored_triton" else "gluon"
    )
    fn = lambda: fused_qk_rope_cat_and_cache_mla(
        q_nope,
        q_pe,
        k_nope,
        k_pe,
        kv_cache,
        slot_mapping,
        positions,
        cos,
        sin,
        k_scale,
        False,
        apply_scale=(cache_dtype != dtype),
        q_out=q_out,
        shuffled_kv_cache=shuffled,
        q_nope_prestored=prestored,
    )

    ms = triton.testing.do_bench_cudagraph(fn, return_mode="median")

    elem = q_pe.element_size()
    cache_elem = kv_cache.element_size()
    q_out_elem = q_out.element_size()
    q_d = D_pe if prestored else D_lora + D_pe
    # q (q_pe, + q_nope when not prestored) + k_nope + k_pe
    mem_read = M * QH * q_d * elem + m_slot * (D_lora + D_pe) * elem
    # q_out (pe part only when prestored) + k_pe_out + kv cache
    mem_write = (
        M * QH * q_d * q_out_elem
        + m_slot * D_pe * elem
        + m_slot * (D_lora + D_pe) * cache_elem
    )
    mem = mem_read + mem_write

    if metric == "time":
        return ms
    elif metric == "bandwidth":
        return mem / ms * 1e-6  # GB/s
    else:
        raise ValueError("Unknown metric: " + metric)


def run_benchmark(args):
    x_vals_list = get_x_vals(args)

    metric_to_ylabel = {
        "time": "Time (ms)",
        "bandwidth": "Bandwidth (GB/s)",
    }
    if args.metric not in metric_to_ylabel:
        raise NotImplementedError(
            f"{args.metric} is not supported: the kernel does no arithmetic"
        )

    benchmark = triton.testing.Benchmark(
        x_names=X_NAMES,
        x_vals=x_vals_list,
        line_arg="provider",
        line_vals=list(PROVIDERS),
        line_names=list(PROVIDERS),
        styles=[("green", "-"), ("blue", "-"), ("red", "-")],
        ylabel=metric_to_ylabel[args.metric],
        plot_name=get_caller_name_no_ext(),
        args={"metric": args.metric, "args": args},
    )

    triton.testing.perf_report([benchmark])(bench_qpe_rope_and_cache_mla_fn).run(
        print_data=True
    )


def parse_int_or_list(value):
    if "," in value:
        return [int(x) for x in value.split(",")]
    return int(value)


def parse_args(args: list[str] | None = None):
    parser = get_parser(kernel_name="fused_qpe_rope_and_cache_mla")
    parser.set_defaults(metric="time")
    parser.add_argument(
        "-M",
        type=parse_int_or_list,
        default=DEFAULT_M,
        help="Number of q tokens (single int or comma-separated list for multiple)",
    )
    parser.add_argument("-QH", type=int, default=DEFAULT_QH, help="Number of Q heads")
    parser.add_argument(
        "-D_lora", type=int, default=DEFAULT_D_LORA, help="kv_lora_rank"
    )
    parser.add_argument(
        "-D_pe", type=int, default=DEFAULT_D_PE, help="qk_rope_head_dim"
    )
    parser.add_argument(
        "--k_only",
        type=int,
        default=0,
        help="Extra tokens that only carry k (mixed prefill + decode batch)",
    )
    parser.add_argument(
        "--num_kv_cache_tokens",
        type=int,
        default=DEFAULT_NUM_KV_CACHE_TOKENS,
        help="Number of KV cache slots",
    )
    parser.add_argument(
        "--cache_dtype",
        type=str,
        choices=list(CACHE_DTYPES),
        default="fp8",
        help="KV cache and q_out dtype",
    )
    parser.add_argument(
        "--unshuffled",
        action="store_true",
        help="Use a flat (block_size=1) KV cache instead of the shuffled layout",
    )
    parser.add_argument(
        "--max_pos",
        type=int,
        default=8192,
        help="Rows in the cos/sin table",
    )
    return parser.parse_args(args=args)


def main(args: list[str] | None = None) -> None:
    parsed_args = parse_args(args=args)
    torch.manual_seed(0)
    run_benchmark(parsed_args)


if __name__ == "__main__":
    main()
