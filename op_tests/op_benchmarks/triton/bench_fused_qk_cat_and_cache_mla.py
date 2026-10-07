# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""
Benchmark for the triton fused_qk_cat_and_cache_mla kernel (the NoPE form of
fused_qk_rope_cat_and_cache_mla).

Triton only: the gfx1250 gluon kernel has no NoPE form.

Default shapes mirror sglang's MLA absorb decode path for Kimi-K3 (96 heads,
QH=12 at TP8) with the flat fp8 KV pool:

    q_nope=[M, QH, 512]   transposed view of the absorbed BMM output [QH, M, 512]
    q_pe=[M, QH, 64]      tail slice of the q_b_proj split [M, QH, 128 + 64]
    k_nope=[M, 1, 512]    contiguous
    k_pe=[M, 1, 64]       tail slice of the kv_a latent [M, 512 + 64]
    kv_cache=[262144, 1, 576] fp8, block_size=1, unshuffled
    q_out dtype = bf16 at QH=12, else the cache dtype (as sglang picks it)
    k_scale applied

The "rope" line runs fused_qk_rope_cat_and_cache_mla on the same shapes (GPT-J,
front-half freqs), so the NoPE specialization can be compared against it.

Metrics: time (default), bandwidth.

Usage Example:
    python bench_fused_qk_cat_and_cache_mla.py --metric bandwidth -M 1,32,256
"""

import torch
import triton

from aiter.ops.triton.fusions.fused_kv_cache import (
    fused_qk_cat_and_cache_mla,
    fused_qk_rope_cat_and_cache_mla,
)
from aiter.ops.triton.utils._triton import arch_info
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

Q_OUT_DTYPES = ("auto", *CACHE_DTYPES)

PROVIDERS = ("nope", "rope")

DEFAULT_M = [1, 8, 32, 64, 128, 256]  # decode batch sizes
DEFAULT_QH = 12  # Kimi-K3: 96 heads / TP8
DEFAULT_D_LORA = 512
DEFAULT_D_PE = 64
DEFAULT_D_QK_NOPE = 128
DEFAULT_NUM_KV_CACHE_TOKENS = 262144

X_NAMES = ["M", "QH", "D_lora", "D_pe"]


def get_x_vals(args):
    m_vals = args.M if isinstance(args.M, list) else [args.M]
    return [(M, args.QH, args.D_lora, args.D_pe) for M in m_vals]


def bench_cat_and_cache_mla_fn(
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
    num_kv_cache_tokens = args.num_kv_cache_tokens
    KH = 1  # MLA: a single latent KV head

    assert DEVICE_ARCH != "gfx1250", "Only triton kernel tested/supported"
    assert (
        M <= num_kv_cache_tokens
    ), f"Not enough cache slots for {M} tokens: {num_kv_cache_tokens=}"

    torch.cuda.empty_cache()

    dtype = torch.bfloat16

    # Layouts as sglang hands them over; see the module docstring.
    q_nope = torch.randn((QH, M, D_lora), dtype=dtype, device="cuda").transpose(0, 1)
    q_pe = torch.randn((M, QH, DEFAULT_D_QK_NOPE + D_pe), dtype=dtype, device="cuda")[
        ..., DEFAULT_D_QK_NOPE:
    ]
    latent = torch.randn((M, D_lora + D_pe), dtype=dtype, device="cuda")
    k_nope = latent[:, :D_lora].contiguous().unsqueeze(1)
    k_pe = latent[:, D_lora:].unsqueeze(1)

    kv_cache = torch.zeros(
        (num_kv_cache_tokens, KH, D_lora + D_pe), dtype=cache_dtype, device="cuda"
    )
    slot_mapping = torch.randperm(num_kv_cache_tokens, device="cuda")[:M]

    k_scale = torch.ones([1], dtype=torch.float32, device="cuda")[0]
    apply_scale = cache_dtype != dtype

    # sglang's rule: the aiter gluon MLA decode needs bf16 q with 12 heads.
    if args.q_out_dtype == "auto":
        q_out_dtype = dtype if QH == 12 else cache_dtype
    else:
        q_out_dtype = CACHE_DTYPES[args.q_out_dtype]

    if provider == "nope":
        fn = lambda: fused_qk_cat_and_cache_mla(  # noqa: E731
            q_nope,
            q_pe,
            k_nope,
            k_pe,
            kv_cache,
            slot_mapping,
            k_scale,
            apply_scale=apply_scale,
            q_out_dtype=q_out_dtype,
        )
    else:
        max_pos = args.max_pos
        freqs = torch.randn((max_pos, D_pe // 2), dtype=dtype, device="cuda")
        cos = torch.cos(freqs)
        sin = torch.sin(freqs)
        positions = torch.randint(0, max_pos, (M,), device="cuda")
        fn = lambda: fused_qk_rope_cat_and_cache_mla(  # noqa: E731
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
            apply_scale=apply_scale,
            q_out_dtype=q_out_dtype,
        )

    ms = triton.testing.do_bench_cudagraph(fn, return_mode="median")

    elem = q_nope.element_size()
    cache_elem = kv_cache.element_size()

    # q_nope + q_pe + k_nope + k_pe
    mem_read = M * QH * (D_lora + D_pe) * elem + M * KH * (D_lora + D_pe) * elem
    q_out_elem = q_out_dtype.itemsize
    # q_out + k_pe_out + kv cache
    mem_write = (
        M * QH * (D_lora + D_pe) * q_out_elem
        + M * KH * D_pe * elem
        + M * KH * (D_lora + D_pe) * cache_elem
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
        styles=[("green", "-"), ("blue", "-")],
        ylabel=metric_to_ylabel[args.metric],
        plot_name=get_caller_name_no_ext(),
        args={"metric": args.metric, "args": args},
    )

    triton.testing.perf_report([benchmark])(bench_cat_and_cache_mla_fn).run(
        print_data=True
    )


def parse_int_or_list(value):
    if "," in value:
        return [int(x) for x in value.split(",")]
    return int(value)


def parse_args(args: list[str] | None = None):
    parser = get_parser(kernel_name="fused_qk_cat_and_cache_mla")
    parser.set_defaults(metric="time")
    parser.add_argument(
        "-M",
        type=parse_int_or_list,
        default=DEFAULT_M,
        help="Number of decode tokens (single int or comma-separated list for multiple)",
    )
    parser.add_argument("-QH", type=int, default=DEFAULT_QH, help="Number of Q heads")
    parser.add_argument(
        "-D_lora", type=int, default=DEFAULT_D_LORA, help="kv_lora_rank"
    )
    parser.add_argument(
        "-D_pe", type=int, default=DEFAULT_D_PE, help="qk_rope_head_dim"
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
        help="KV cache dtype",
    )
    parser.add_argument(
        "--q_out_dtype",
        type=str,
        choices=Q_OUT_DTYPES,
        default="auto",
        help="q_out dtype; auto picks bf16 at QH=12 and the cache dtype otherwise",
    )
    parser.add_argument(
        "--max_pos",
        type=int,
        default=8192,
        help="Rows in the cos/sin table for the rope line",
    )
    return parser.parse_args(args=args)


def main(args: list[str] | None = None) -> None:
    parsed_args = parse_args(args=args)
    torch.manual_seed(0)
    run_benchmark(parsed_args)


if __name__ == "__main__":
    main()
