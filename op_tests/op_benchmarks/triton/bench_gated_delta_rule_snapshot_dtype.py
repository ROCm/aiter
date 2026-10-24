# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Benchmark BF16 and FP32 GDN chunk snapshots independently of SSM state.

This benchmark times the two kernels affected by ``snapshot_dtype``:

* ``chunk_gated_delta_rule_fwd_h_hip_kernel`` (K5 snapshot producer)
* ``chunk_fwd_kernel_o_opt_vk`` (K6 snapshot consumer)

The persistent initial/final state remains FP32 in both runs so the reported
difference isolates the temporary chunk-snapshot policy.
"""

import functools

import torch
import triton

from aiter.ops.chunk_gated_delta_rule_fwd_h import (
    chunk_gated_delta_rule_fwd_h_hip,
)
from aiter.ops.triton._triton_kernels.gated_delta_net.prefill import (
    chunk_fwd_o_opt_vk,
)
from op_tests.op_benchmarks.triton.utils.argparse import (
    get_do_bench_graph_kwargs,
    get_parser,
    get_rot_bufs_kwargs,
)
from op_tests.op_benchmarks.triton.utils.benchmark_utils import (
    do_bench_aiter_triton,
    generate_rotating_buffers_pool,
)


def _dtype_name(dtype: torch.dtype) -> str:
    return "bf16" if dtype == torch.bfloat16 else "fp32"


def _generate_pool_entry(batch_size, sequence_length, num_heads, snapshot_dtype):
    """Build one full K5/K6 input set, with K6's h/v_new primed by one K5 call.

    Pure setup, not measured. Each call is one rotating-buffer pool entry:
    independent random tensors, so replaying through the pool during timing
    defeats cache residency the same way a repeat-and-average loop would.
    """
    device = torch.device("cuda")
    chunk_size = 64
    head_dim = 128
    num_chunks = triton.cdiv(sequence_length, chunk_size)

    k = torch.randn(
        batch_size,
        sequence_length,
        num_heads,
        head_dim,
        device=device,
        dtype=torch.bfloat16,
    )
    q = torch.randn_like(k)
    w = torch.randn(
        batch_size,
        num_heads,
        sequence_length,
        head_dim,
        device=device,
        dtype=torch.bfloat16,
    )
    u = torch.randn_like(w)
    g = torch.randn(
        batch_size,
        num_heads,
        sequence_length,
        device=device,
        dtype=torch.float32,
    )

    # Keep persistent state FP32 while varying only the chunk snapshot dtype.
    initial_state = torch.randn(
        batch_size,
        num_heads,
        head_dim,
        head_dim,
        device=device,
        dtype=torch.float32,
    )
    final_state = torch.empty_like(initial_state)
    h = torch.empty(
        batch_size,
        num_chunks,
        num_heads,
        head_dim,
        head_dim,
        device=device,
        dtype=snapshot_dtype,
    )
    v_new = torch.empty_like(u)
    output = torch.empty_like(k)

    empty_fp32 = torch.empty(0, device=device, dtype=torch.float32)
    empty_i32 = torch.empty(0, device=device, dtype=torch.int32)
    chunk_offsets = torch.arange(
        0,
        (batch_size + 1) * num_chunks,
        num_chunks,
        device=device,
        dtype=torch.int32,
    )

    # Initialize K6's h/v_new through one untimed K5 call, same as the
    # reference this benchmark is modeled on.
    chunk_gated_delta_rule_fwd_h_hip(
        k,
        w,
        u,
        g,
        empty_fp32,
        initial_state,
        empty_i32,
        empty_i32,
        chunk_offsets,
        h,
        v_new,
        final_state,
        64,
        True,
        True,
        True,
        True,
        True,
    )

    return (
        q,
        k,
        w,
        u,
        g,
        initial_state,
        final_state,
        h,
        v_new,
        output,
        empty_fp32,
        empty_i32,
        chunk_offsets,
    )


_POOL_FIELDS = [
    "q",
    "k",
    "w",
    "u",
    "g",
    "initial_state",
    "final_state",
    "h",
    "v_new",
    "output",
    "empty_fp32",
    "empty_i32",
    "chunk_offsets",
]


def _make_kernel_fns(chunk_size=64):
    def kernel_k5(
        k,
        w,
        u,
        g,
        empty_fp32,
        initial_state,
        empty_i32,
        chunk_offsets,
        h,
        v_new,
        final_state,
        **_,
    ):
        chunk_gated_delta_rule_fwd_h_hip(
            k,
            w,
            u,
            g,
            empty_fp32,
            initial_state,
            empty_i32,
            empty_i32,
            chunk_offsets,
            h,
            v_new,
            final_state,
            64,
            True,
            True,
            True,
            True,
            True,
        )

    def kernel_k6(q, k, v_new, output, h, g, **_):
        chunk_fwd_o_opt_vk(
            q=q,
            k=k,
            v=v_new,
            o=output,
            h=h,
            g=g,
            chunk_size=chunk_size,
            use_exp2=True,
        )

    return kernel_k5, kernel_k6


def benchmark_snapshot_dtype(
    snapshot_dtype: torch.dtype,
    batch_size: int,
    sequence_length: int,
    num_heads: int,
    kwargs_rot_bufs: dict,
    kwargs_do_bench_graph: dict,
) -> tuple[float, float, float]:
    pool_gen = functools.partial(
        _generate_pool_entry, batch_size, sequence_length, num_heads, snapshot_dtype
    )
    rot_bufs = generate_rotating_buffers_pool(
        [pool_gen], [_POOL_FIELDS], **kwargs_rot_bufs
    )
    kernel_k5, kernel_k6 = _make_kernel_fns()

    k5_ms = do_bench_aiter_triton(kernel_k5, rot_bufs, **kwargs_do_bench_graph)
    k6_ms = do_bench_aiter_triton(kernel_k6, rot_bufs, **kwargs_do_bench_graph)
    snapshot_mib = rot_bufs[0]["h"].numel() * rot_bufs[0]["h"].element_size() / (1024**2)
    return k5_ms * 1000, k6_ms * 1000, snapshot_mib


def main() -> None:
    parser = get_parser(kernel_name="Gated Delta Rule Snapshot Dtype")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--sequence-length", type=int, default=8192)
    parser.add_argument("--num-heads", type=int, default=16)
    args = parser.parse_args()

    kwargs_do_bench_graph = get_do_bench_graph_kwargs(args)
    kwargs_rot_bufs = get_rot_bufs_kwargs(args)

    print("snapshot_dtype,k5_us,k6_us,snapshot_mib")
    for snapshot_dtype in (torch.bfloat16, torch.float32):
        k5_us, k6_us, snapshot_mib = benchmark_snapshot_dtype(
            snapshot_dtype=snapshot_dtype,
            batch_size=args.batch_size,
            sequence_length=args.sequence_length,
            num_heads=args.num_heads,
            kwargs_rot_bufs=kwargs_rot_bufs,
            kwargs_do_bench_graph=kwargs_do_bench_graph,
        )
        print(
            f"{_dtype_name(snapshot_dtype)},{k5_us:.3f},"
            f"{k6_us:.3f},{snapshot_mib:.3f}"
        )


if __name__ == "__main__":
    main()
