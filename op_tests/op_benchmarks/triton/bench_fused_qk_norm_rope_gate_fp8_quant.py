# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Benchmark fused Q/K norm, RoPE, gate extraction, and FP8 Q/K/V quantization.

The defaults model Qwen3 Next full-attention shapes on gfx950. The benchmark
uses preallocated outputs, matching the graph-captured vLLM integration.

Usage:
    python bench_fused_qk_norm_rope_gate_fp8_quant.py \
        -M 128,1024,4096,8192 --metric time
"""

import functools

import torch
import triton

from aiter.ops.triton.rope.fused_qk_norm_rope_gate_fp8_quant import (
    FP8_DTYPE,
    MAX_SEQUENCES,
    fused_qk_norm_rope_gate_fp8_quant,
)
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


def parse_int_or_list(value):
    if "," in value:
        return [int(item) for item in value.split(",")]
    return int(value)


def get_x_vals(args):
    token_counts = args.M if isinstance(args.M, list) else [args.M]
    return [(tokens,) for tokens in token_counts]


def balanced_cu_seqlens(total_tokens, requested_sequences, device):
    num_sequences = min(total_tokens, requested_sequences)
    base, remainder = divmod(total_tokens, num_sequences)
    lengths = torch.full((num_sequences,), base, dtype=torch.int32, device=device)
    lengths[:remainder] += 1
    cu_seqlens = torch.zeros(num_sequences + 1, dtype=torch.int32, device=device)
    torch.cumsum(lengths, dim=0, out=cu_seqlens[1:])
    return cu_seqlens


def _generate_inputs(
    tokens, num_query_heads, num_kv_heads, head_dim, rotary_dim, num_sequences
):
    device = torch.device("cuda")
    q_width = num_query_heads * head_dim
    kv_width = num_kv_heads * head_dim
    q_gate = torch.randn(
        tokens,
        2 * q_width,
        dtype=torch.bfloat16,
        device=device,
    )
    key = torch.randn(tokens, kv_width, dtype=torch.bfloat16, device=device)
    value = torch.randn_like(key)
    query_norm_weight = torch.randn(head_dim, dtype=torch.bfloat16, device=device)
    key_norm_weight = torch.randn_like(query_norm_weight)
    positions = torch.arange(tokens, dtype=torch.int64, device=device)
    cos_sin_cache = torch.randn(
        tokens,
        rotary_dim,
        dtype=torch.bfloat16,
        device=device,
    )
    cu_seqlens = balanced_cu_seqlens(tokens, num_sequences, device)

    # Allocate this pool entry's own output buffers up front (one setup call,
    # not timed) so the timed call below can write into them in place, exactly
    # mirroring the graph-captured vLLM integration.
    output = fused_qk_norm_rope_gate_fp8_quant(
        q_gate,
        key,
        value,
        query_norm_weight,
        key_norm_weight,
        cos_sin_cache,
        positions,
        cu_seqlens,
        num_actual_tokens=tokens,
        num_query_heads=num_query_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        rotary_dim=rotary_dim,
    )
    torch.cuda.synchronize()

    return (
        q_gate,
        key,
        value,
        query_norm_weight,
        key_norm_weight,
        positions,
        cos_sin_cache,
        cu_seqlens,
        output.query,
        output.key,
        output.gate,
        output.query_fp8,
        output.key_fp8,
        output.value_fp8,
        output.query_descale,
        output.key_descale,
        output.value_descale,
    )


def benchmark_fused_qkv(tokens, metric, args, **_):
    kwargs_do_bench_graph = get_do_bench_graph_kwargs(args)
    kwargs_rot_bufs = get_rot_bufs_kwargs(args)
    _p_gen = functools.partial(
        _generate_inputs,
        tokens,
        args.num_query_heads,
        args.num_kv_heads,
        args.head_dim,
        args.rotary_dim,
        args.num_sequences,
    )
    _out_val_keys = [
        "q_gate",
        "key",
        "value",
        "query_norm_weight",
        "key_norm_weight",
        "positions",
        "cos_sin_cache",
        "cu_seqlens",
        "query_out",
        "key_out",
        "gate_out",
        "query_fp8_out",
        "key_fp8_out",
        "value_fp8_out",
        "query_descale_out",
        "key_descale_out",
        "value_descale_out",
    ]
    rot_bufs = generate_rotating_buffers_pool(
        [_p_gen], [_out_val_keys], **kwargs_rot_bufs
    )

    # Shapes/dtypes are identical across the pool, so any entry works for sizing.
    sample = rot_bufs[0]

    def _run(
        q_gate,
        key,
        value,
        query_norm_weight,
        key_norm_weight,
        positions,
        cos_sin_cache,
        cu_seqlens,
        query_out,
        key_out,
        gate_out,
        query_fp8_out,
        key_fp8_out,
        value_fp8_out,
        query_descale_out,
        key_descale_out,
        value_descale_out,
    ):
        fused_qk_norm_rope_gate_fp8_quant(
            q_gate,
            key,
            value,
            query_norm_weight,
            key_norm_weight,
            cos_sin_cache,
            positions,
            cu_seqlens,
            num_actual_tokens=tokens,
            num_query_heads=args.num_query_heads,
            num_kv_heads=args.num_kv_heads,
            head_dim=args.head_dim,
            rotary_dim=args.rotary_dim,
            query_out=query_out,
            key_out=key_out,
            gate_out=gate_out,
            query_fp8_out=query_fp8_out,
            key_fp8_out=key_fp8_out,
            value_fp8_out=value_fp8_out,
            query_descale_out=query_descale_out,
            key_descale_out=key_descale_out,
            value_descale_out=value_descale_out,
        )

    time_ms = do_bench_aiter_triton(_run, rot_bufs, **kwargs_do_bench_graph)
    if metric == "time":
        return time_ms
    if metric == "bandwidth":
        tensors = (
            sample["q_gate"],
            sample["key"],
            sample["value"],
            sample["query_norm_weight"],
            sample["key_norm_weight"],
            sample["cos_sin_cache"],
            sample["positions"],
            sample["cu_seqlens"],
            sample["query_out"],
            sample["key_out"],
            sample["gate_out"],
            sample["query_fp8_out"],
            sample["key_fp8_out"],
            sample["value_fp8_out"],
            sample["query_descale_out"],
            sample["key_descale_out"],
            sample["value_descale_out"],
        )
        total_bytes = sum(tensor.numel() * tensor.element_size() for tensor in tensors)
        return total_bytes / time_ms * 1.0e-6
    raise ValueError(f"Unsupported metric: {metric}")


def run_benchmark(args):
    ylabel = {
        "time": "Time (ms)",
        "bandwidth": "Bandwidth (GB/s)",
    }[args.metric]
    benchmark = triton.testing.Benchmark(
        x_names=["tokens"],
        x_vals=get_x_vals(args),
        line_arg="provider",
        line_vals=[args.metric],
        line_names=[ylabel],
        styles=[("green", "-")],
        ylabel=ylabel,
        plot_name=get_caller_name_no_ext(),
        args={"metric": args.metric, "args": args},
    )
    triton.testing.perf_report([benchmark])(benchmark_fused_qkv).run(
        save_path="." if args.output else None,
        print_data=True,
    )


def main():
    parser = get_parser(kernel_name="fused_qk_norm_rope_gate_fp8_quant")
    parser.add_argument(
        "-M",
        type=parse_int_or_list,
        default=[128, 1024, 4096, 8192],
        help="token count or comma-separated token counts",
    )
    parser.add_argument("--num-sequences", type=int, default=16)
    parser.add_argument("--num-query-heads", type=int, default=8)
    parser.add_argument("--num-kv-heads", type=int, default=1)
    parser.add_argument("--head-dim", type=int, default=256)
    parser.add_argument("--rotary-dim", type=int, default=64)
    override_parser_action(
        parser, action="--metric", choices=["time", "bandwidth"], default="time"
    )
    parser.add_argument(
        "-o",
        "--output",
        action="store_true",
        help="write benchmark CSV output",
    )
    args = parser.parse_args()
    if args.num_query_heads % args.num_kv_heads != 0:
        parser.error("--num-query-heads must be divisible by --num-kv-heads")
    if args.num_sequences > MAX_SEQUENCES:
        parser.error(f"--num-sequences must be <= {MAX_SEQUENCES}")
    if args.rotary_dim > args.head_dim or args.rotary_dim % 2:
        parser.error("--rotary-dim must be even and <= --head-dim")
    if FP8_DTYPE != torch.float8_e4m3fn:
        parser.error("benchmark requires gfx950 E4M3-FN FP8")
    run_benchmark(args)


if __name__ == "__main__":
    main()
