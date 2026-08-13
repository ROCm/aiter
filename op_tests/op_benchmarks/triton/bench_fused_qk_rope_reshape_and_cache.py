# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""
Benchmark for fused_qk_rope_reshape_and_cache.

The defaults reproduce the call the kernel actually gets in ATOM serving
openai/gpt-oss-120b at TP1 with a bf16 KV cache and block_size 16 (the config
run_offline_benchmark.sh drives: BS=16, ISL=1024, OSL=1024), as observed at the
attention_mha.py call site:

    q=[T, 64, 64]  k=[T, 8, 64]  v=[T, 8, 64]     (views into a packed QKV buffer)
    k_cache=[169788, 8, 8, 16, 8]                 (nonflash, x=8)
    v_cache=[169788, 8, 2, 64, 8]                 (shuffled value layout)
    cos/sin=[131072, 1, 1, 32]                    (neox, front-half freqs reused)
    flash_layout=False, apply_scale=False, offs=None, output_zeros=False,
    q_out=q, k_out=k                              (RoPE written back in place)

with T = 16 for decode (batch size) and T in {1024, 15360} for the chunked
prefill of a 16 x 1024-token batch.
"""

import argparse

import torch
import triton

from aiter.ops.triton.fusions.fused_kv_cache import fused_qk_rope_reshape_and_cache
from aiter.ops.triton.utils._triton import arch_info
from aiter.ops.triton.utils.types import e4m3_dtype
from op_tests.op_benchmarks.triton.utils.argparse import get_parser
from op_tests.op_benchmarks.triton.utils.benchmark_utils import (
    get_caller_name_no_ext,
    get_model_configs,
    print_vgpr,
)
from op_tests.triton_tests.rope.test_rope import generate_rope_inputs

DEVICE_ARCH = arch_info.get_arch()

# non-flash key_cache: (num_blocks, KH, D // x, block_size, x)
# in ATOM x = 16 // element_size(), defaulting to bf16
X_SIZE = 8

CACHE_DTYPES = {
    "bf16": torch.bfloat16,
    "fp8": e4m3_dtype,
    "fp4": torch.uint8,
}

CACHE_LAYOUTS = ("flash", "nonflash", "nonflash_v_shuffle")

SLOT_PATTERNS = ("blocked", "random")

QKV_LAYOUTS = ("packed", "split")

MAX_OFFSET = 256

DEFAULT_M = [15360] # test prefill
DEFAULT_QH = 64
DEFAULT_KH = 8 # 64 / TP8
DEFAULT_D = 64
# gpt-oss-120b, TP1, bf16 cache, block 16.
DEFAULT_NUM_BLOCKS = 169788
DEFAULT_BLOCK_SIZE = 16
DEFAULT_MAX_EMBD_POS = 131072
# Positions only ever reach ISL + OSL, sequence ends
DEFAULT_MAX_POS = 2048

X_NAMES = ["M", "QH", "KH", "D", "num_blocks", "block_size", "max_embd_pos"]


def check_config(
    M,
    QH,
    KH,
    D,
    num_blocks,
    block_size,
    max_embd_pos,
    cache_dtype,
    cache_layout,
    offs,
    max_pos,
):
    assert (
        0 < max_pos <= max_embd_pos
    ), f"--max_pos must be in (0, max_embd_pos], got {max_pos=} {max_embd_pos=}"
    assert QH % KH == 0, f"QH must be a multiple of KH, got {QH=} {KH=}"
    assert D == triton.next_power_of_2(D), f"D must be a power of 2, got {D=}"
    assert block_size == triton.next_power_of_2(
        block_size
    ), f"block_size must be a power of 2, got {block_size=}"
    assert (
        M <= num_blocks * block_size
    ), f"Not enough cache slots for {M} tokens: {num_blocks=} {block_size=}"

    if cache_dtype == torch.uint8:
        assert DEVICE_ARCH in (
            "gfx1250",
        ), f"NVFP4 KV cache is only supported on gfx1250, got {DEVICE_ARCH}"
        assert D % 16 == 0, f"NVFP4 KV cache needs D % 16 == 0, got {D=}"
        assert (
            block_size % 128 == 0
        ), f"NVFP4 KV cache needs block_size % 128 == 0, got {block_size=}"
    elif cache_dtype == e4m3_dtype:
        assert DEVICE_ARCH in (
            "gfx1250",
            "gfx950",
        ), f"FP8 KV cache is only supported on gfx1250 and gfx950, got {DEVICE_ARCH}"

    if cache_layout != "flash":
        assert (
            D % X_SIZE == 0
        ), f"Non-flash key cache layout needs D % {X_SIZE} == 0, got {D=}"
    if cache_layout == "nonflash_v_shuffle":
        assert (
            block_size % X_SIZE == 0
        ), f"Shuffled value cache layout needs block_size % {X_SIZE} == 0, got {block_size=}"

    if offs:
        assert max_pos > 2 * MAX_OFFSET, (
            f"--offs needs a position range wider than {2 * MAX_OFFSET} to keep "
            f"pos + offs in range, got {max_pos=}"
        )


def get_x_vals(args):
    m_vals = args.M if isinstance(args.M, list) else [args.M]

    if args.model:
        configs = get_model_configs(config_path=args.model_configs, models=args.model)
        head_dims = []
        for config in configs.values():
            qh = config["num_attention_heads"]
            kh = config["num_key_value_heads"]
            d = config.get("head_dim", config["hidden_size"] // qh)
            head_dims.append((qh, kh, d))
    else:
        head_dims = [(args.QH, args.KH, args.D)]

    return [
        (M, QH, KH, D, args.num_blocks, args.block_size, args.max_embd_pos)
        for QH, KH, D in head_dims
        for M in m_vals
    ]


def make_inputs(
    M,
    QH,
    KH,
    D,
    num_blocks,
    block_size,
    max_embd_pos,
    cache_dtype,
    cache_layout,
    rotate_style,
    reuse_freqs_front_part,
    offs,
    max_pos,
    slot_pattern,
    qkv_layout,
    output_zeros,
    dtype,
):
    torch.cuda.empty_cache()

    flash_layout = cache_layout == "flash"
    value_shuffle_layout = cache_layout == "nonflash_v_shuffle"
    
    if cache_dtype == torch.uint8:
        flash_layout = False

    q, k, _, _, _, _, _, _, _ = generate_rope_inputs(
        1,
        M,
        KH,
        QH // KH,
        D,
        cached=True,
        reuse_freqs_front_part=reuse_freqs_front_part,
        nope=False,
        pos=True,
        offs=False,
        two_inputs=True,
        layout="thd",
        dtype=dtype,
    )
    v = torch.randn_like(k)

    d_freq = D // 2 if reuse_freqs_front_part else D
    freqs = torch.randn((max_embd_pos, 1, 1, d_freq), dtype=dtype, device="cuda")
    cos = torch.cos(freqs)
    sin = torch.sin(freqs)
    
    if offs:
        positions = torch.randint(MAX_OFFSET, max_pos - MAX_OFFSET, (M,), device="cuda")
        offsets = torch.randint(-MAX_OFFSET, MAX_OFFSET + 1, (M,), device="cuda")
    else:
        positions = torch.randint(0, max_pos, (M,), device="cuda")
        offsets = None

    if qkv_layout == "packed":
        qkv = torch.empty((M, (QH + 2 * KH) * D), dtype=dtype, device="cuda")
        q_flat, k_flat, v_flat = torch.split(
            qkv, [QH * D, KH * D, KH * D], dim=-1
        )
        q_flat.copy_(q.reshape(M, -1))
        k_flat.copy_(k.reshape(M, -1))
        v_flat.copy_(v.reshape(M, -1))
        q = q_flat.view(M, QH, D)
        k = k_flat.view(M, KH, D)
        v = v_flat.view(M, KH, D)

    if cache_dtype == torch.uint8:
        d_cache = D // 2 + D // 16
        key_cache = torch.zeros(
            (num_blocks, KH, block_size, d_cache), dtype=torch.uint8, device="cuda"
        )
        value_cache = torch.zeros_like(key_cache)
    elif flash_layout:
        key_cache = torch.zeros(
            (num_blocks, block_size, KH, D), dtype=cache_dtype, device="cuda"
        )
        value_cache = torch.zeros_like(key_cache)
    else:
        key_cache = torch.zeros(
            (num_blocks, KH, D // X_SIZE, block_size, X_SIZE),
            dtype=cache_dtype,
            device="cuda",
        )
        if value_shuffle_layout:
            value_cache = torch.zeros(
                (num_blocks, KH, block_size // X_SIZE, D, X_SIZE),
                dtype=cache_dtype,
                device="cuda",
            )
        else:
            value_cache = torch.zeros(
                (num_blocks, KH, D, block_size), dtype=cache_dtype, device="cuda"
            )

    if slot_pattern == "blocked":
        n_blocks = triton.cdiv(M, block_size)
        blocks = torch.randperm(num_blocks, device="cuda")[:n_blocks]
        within = torch.arange(block_size, device="cuda")
        slot_mapping = (blocks[:, None] * block_size + within[None, :]).reshape(-1)[:M]
    else:
        slot_mapping = torch.randperm(num_blocks * block_size, device="cuda")[:M]
    slot_mapping = slot_mapping.contiguous()

    k_scale = torch.ones((), dtype=torch.float32, device="cuda")
    v_scale = torch.ones((), dtype=torch.float32, device="cuda")

    q_out = q
    k_out = k
    zeros_out = torch.empty_like(q) if output_zeros else None

    return {
        "q": q,
        "k": k,
        "v": v,
        "key_cache": key_cache,
        "value_cache": value_cache,
        "slot_mapping": slot_mapping,
        "positions": positions,
        "cos": cos,
        "sin": sin,
        "k_scale": k_scale,
        "v_scale": v_scale,
        "is_neox": rotate_style == "neox",
        "flash_layout": flash_layout,
        "apply_scale": cache_dtype != torch.bfloat16,
        "offs": offsets,
        "q_out": q_out,
        "k_out": k_out,
        "output_zeros": output_zeros,
        "zeros_out": zeros_out,
    }


def bench_fn(
    M,
    QH,
    KH,
    D,
    num_blocks,
    block_size,
    max_embd_pos,
    metric,
    args,
    **kwargs,
):
    cache_dtype = CACHE_DTYPES[args.cache_dtype]
    check_config(
        M,
        QH,
        KH,
        D,
        num_blocks,
        block_size,
        max_embd_pos,
        cache_dtype,
        args.cache_layout,
        args.offs,
        args.max_pos,
    )
    inputs = make_inputs(
        M,
        QH,
        KH,
        D,
        num_blocks,
        block_size,
        max_embd_pos,
        cache_dtype,
        args.cache_layout,
        args.rotate_style,
        args.reuse_freqs_front_part,
        args.offs,
        args.max_pos,
        args.slot_pattern,
        args.qkv_layout,
        args.output_zeros,
        torch.bfloat16,
    )

    def fn():
        return fused_qk_rope_reshape_and_cache(
            inputs["q"],
            inputs["k"],
            inputs["v"],
            inputs["key_cache"],
            inputs["value_cache"],
            inputs["slot_mapping"],
            inputs["positions"],
            inputs["cos"],
            inputs["sin"],
            inputs["k_scale"],
            inputs["v_scale"],
            inputs["is_neox"],
            flash_layout=inputs["flash_layout"],
            apply_scale=inputs["apply_scale"],
            offs=inputs["offs"],
            q_out=inputs["q_out"],
            k_out=inputs["k_out"],
            output_zeros=inputs["output_zeros"],
            zeros_out=inputs["zeros_out"],
            upcast_operand=args.upcast_operand,
        )

    ms = triton.testing.do_bench_cudagraph(fn, rep=args.rep, return_mode="median")

    flops = M * QH * D * 3 + M * KH * D * 3 # rope

    q = inputs["q"]
    key_cache = inputs["key_cache"]
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

    triton.testing.perf_report([benchmark])(bench_fn).run(
        save_path="." if args.o else None, print_data=True
    )


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
        "--max_embd_pos",
        type=int,
        default=DEFAULT_MAX_EMBD_POS,
        help="Number of rows in the cos/sin table that positions gather from",
    )
    parser.add_argument(
        "--max_pos",
        type=int,
        default=DEFAULT_MAX_POS,
        help="Max positions",
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
        "--qkv_layout",
        type=str,
        choices=QKV_LAYOUTS,
        default="packed",
        help="qkv layout: packed/split",
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
        "--offs",
        action="store_true",
        default=False,
        help="Add a per-token offset tensor to the positions",
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
    parser.add_argument(
        "--rep",
        type=int,
        default=20,
        help="Target measurement window in ms passed to do_bench_cudagraph",
    )
    parser.add_argument(
        "-print_vgpr",
        action="store_true",
        default=False,
        help="Print VGPR usage for Triton kernels",
    )
    parser.add_argument(
        "-o", action="store_true", help="Write performance results to CSV file"
    )
    return parser.parse_args(args=args)


def main(args: list[str] | None = None) -> None:
    parsed_args = parse_args(args=args)
    torch.manual_seed(0)
    if parsed_args.print_vgpr:
        print("Retrieving VGPR usage for Triton kernels...")
        print_vgpr(lambda: run_benchmark(parsed_args), get_caller_name_no_ext())
        return
    run_benchmark(parsed_args)


if __name__ == "__main__":
    main()
