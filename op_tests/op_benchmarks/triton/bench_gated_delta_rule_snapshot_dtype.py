# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Benchmark BF16 and FP32 GDN chunk snapshots independently of SSM state.

This benchmark times the two kernels affected by ``snapshot_dtype``:

* ``chunk_gated_delta_rule_fwd_h_hip_kernel`` (K5 snapshot producer)
* ``chunk_fwd_kernel_o_opt_vk`` (K6 snapshot consumer)

The persistent initial/final state remains FP32 in both runs so the reported
difference isolates the temporary chunk-snapshot policy.
"""

import argparse

import torch
import triton

from aiter.ops.chunk_gated_delta_rule_fwd_h import (
    chunk_gated_delta_rule_fwd_h_hip,
)
from aiter.ops.triton._triton_kernels.gated_delta_rule.prefill import (
    chunk_fwd_o_opt_vk,
)


def _dtype_name(dtype: torch.dtype) -> str:
    return "bf16" if dtype == torch.bfloat16 else "fp32"


def benchmark_snapshot_dtype(
    snapshot_dtype: torch.dtype,
    batch_size: int,
    sequence_length: int,
    num_heads: int,
    warmup_ms: int,
    rep_ms: int,
) -> tuple[float, float, float]:
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

    def run_k5() -> None:
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

    def run_k6() -> None:
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

    # Compile/autotune before timing and initialize K6 inputs through K5.
    run_k5()
    run_k6()
    torch.cuda.synchronize()

    k5_ms = triton.testing.do_bench(run_k5, warmup=warmup_ms, rep=rep_ms)
    k6_ms = triton.testing.do_bench(run_k6, warmup=warmup_ms, rep=rep_ms)
    snapshot_mib = h.numel() * h.element_size() / (1024**2)
    return k5_ms * 1000, k6_ms * 1000, snapshot_mib


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark BF16/FP32 GDN chunk snapshot producer and consumer."
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--sequence-length", type=int, default=8192)
    parser.add_argument("--num-heads", type=int, default=16)
    parser.add_argument("--warmup-ms", type=int, default=25)
    parser.add_argument("--rep-ms", type=int, default=100)
    args = parser.parse_args()

    print("snapshot_dtype,k5_us,k6_us,snapshot_mib")
    for snapshot_dtype in (torch.bfloat16, torch.float32):
        k5_us, k6_us, snapshot_mib = benchmark_snapshot_dtype(
            snapshot_dtype=snapshot_dtype,
            batch_size=args.batch_size,
            sequence_length=args.sequence_length,
            num_heads=args.num_heads,
            warmup_ms=args.warmup_ms,
            rep_ms=args.rep_ms,
        )
        print(
            f"{_dtype_name(snapshot_dtype)},{k5_us:.3f},"
            f"{k6_us:.3f},{snapshot_mib:.3f}"
        )


if __name__ == "__main__":
    main()
