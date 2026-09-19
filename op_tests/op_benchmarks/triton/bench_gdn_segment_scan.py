# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Benchmark the segmented affine-scan GDN K5 against the stock FlyDSL K5.

Both arms run the same indexed-varlen K5 contract (chunk snapshots, v_new and
an in-place final state) so the reported delta is the hidden-state pass only.
The segmented path is gated on packed batch size, so the sweep walks the shapes
that decide the route: N=1 long prefills win, N>=4 packs stay on FlyDSL.
"""

import argparse
import contextlib

import torch
import triton

from aiter.ops.flydsl import linear_attention_prefill_kernels as flydsl_mod
from aiter.ops.flydsl.linear_attention_prefill_kernels import (
    chunk_gated_delta_rule_fwd_h_flydsl_opt,
    gdn_prepare_fwd_flydsl,
)
from aiter.ops.prefill_batch_metadata import (
    build_gated_delta_rule_prefill_metadata,
)
from aiter.ops.triton.gated_delta_net.gdn_segment_scan import gdn_segment_scan_fwd

_CHUNK = 64
_HEAD_DIM = 128


_GATE = "_GDN_K5_SEGMENT_MIN_TOTAL_CHUNKS"


@contextlib.contextmanager
def _segment_scan_disabled():
    """Force the FlyDSL wrapper onto stock K5 so the control arm is the old path.

    The segmented route has no runtime switch, so the control arm raises the
    chunk gate out of reach for the duration of the measurement.
    """
    previous = getattr(flydsl_mod, _GATE)
    setattr(flydsl_mod, _GATE, 1 << 30)
    try:
        yield
    finally:
        setattr(flydsl_mod, _GATE, previous)


def _inputs(seq_lens: tuple[int, ...], num_heads: int, num_kv_heads: int):
    device = torch.device("cuda")
    tokens = sum(seq_lens)
    k = torch.randn(
        1, tokens, num_kv_heads, _HEAD_DIM, device=device, dtype=torch.bfloat16
    )
    v = torch.randn(
        1, tokens, num_heads, _HEAD_DIM, device=device, dtype=torch.bfloat16
    )
    k = torch.nn.functional.normalize(k.float(), dim=-1).to(torch.bfloat16)
    v = torch.nn.functional.normalize(v.float(), dim=-1).to(torch.bfloat16)
    g_raw = torch.full((1, tokens, num_heads), -0.05, device=device)
    beta = torch.full((1, tokens, num_heads), 0.7, device=device)
    w, u, g = gdn_prepare_fwd_flydsl(k=k, v=v, g=g_raw, beta=beta, use_exp2=True)

    offsets = (0, *tuple(sum(seq_lens[: i + 1]) for i in range(len(seq_lens))))
    cu_seqlens = torch.tensor(offsets, dtype=torch.int32, device=device)
    metadata = build_gated_delta_rule_prefill_metadata(
        seq_lens, cu_seqlens=cu_seqlens, chunk_size=_CHUNK
    )
    pool = torch.randn(
        len(seq_lens),
        num_heads,
        _HEAD_DIM,
        _HEAD_DIM,
        device=device,
        dtype=torch.bfloat16,
    )
    state_indices = torch.arange(len(seq_lens), dtype=torch.int32, device=device)
    return k, w, u, g, cu_seqlens, metadata, pool, state_indices


def benchmark_shape(
    seq_lens: tuple[int, ...],
    num_heads: int,
    num_kv_heads: int,
    warmup_ms: int,
    rep_ms: int,
) -> tuple[float, float]:
    k, w, u, g, cu_seqlens, metadata, pool, state_indices = _inputs(
        seq_lens, num_heads, num_kv_heads
    )

    def run_stock() -> None:
        chunk_gated_delta_rule_fwd_h_flydsl_opt(
            k=k,
            w=w,
            u=u,
            g=g,
            initial_state=pool,
            output_final_state=True,
            use_exp2=True,
            g_head_major=True,
            cu_seqlens=cu_seqlens,
            prefill_metadata=metadata,
            initial_state_indices=state_indices,
        )

    def run_segmented() -> None:
        gdn_segment_scan_fwd(
            k=k,
            w=w,
            u=u,
            g=g,
            initial_state=pool,
            output_final_state=True,
            seq_lens=seq_lens,
            state_indices=state_indices,
            # Not in place: repeated timing iterations would otherwise feed each
            # rep the previous rep's state and drift the inputs.
            inplace_final_state=False,
            snapshot_dtype=torch.bfloat16,
            state_dtype=torch.bfloat16,
        )

    # Compile and autotune outside the timed region.
    with _segment_scan_disabled():
        run_stock()
        torch.cuda.synchronize()
        stock_ms = triton.testing.do_bench(run_stock, warmup=warmup_ms, rep=rep_ms)

    run_segmented()
    torch.cuda.synchronize()
    segmented_ms = triton.testing.do_bench(run_segmented, warmup=warmup_ms, rep=rep_ms)
    return stock_ms * 1000, segmented_ms * 1000


def _default_shapes() -> tuple[tuple[int, ...], ...]:
    return (
        (8192,),
        (16384,),
        (32768,),
        (16384, 16384),
        (8192, 8192, 8192, 8192),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark segmented vs stock FlyDSL GDN K5."
    )
    parser.add_argument("--num-heads", type=int, default=16)
    parser.add_argument("--num-kv-heads", type=int, default=8)
    parser.add_argument("--warmup-ms", type=int, default=25)
    parser.add_argument("--rep-ms", type=int, default=100)
    parser.add_argument(
        "--seq-lens",
        action="append",
        default=None,
        help="Comma-separated packed sequence lengths; repeat for more shapes.",
    )
    args = parser.parse_args()

    if args.seq_lens:
        shapes = tuple(
            tuple(int(token) for token in spec.split(",")) for spec in args.seq_lens
        )
    else:
        shapes = _default_shapes()

    print("seq_lens,packed_batch,total_chunks,stock_us,segmented_us,delta_pct")
    for seq_lens in shapes:
        stock_us, segmented_us = benchmark_shape(
            seq_lens=seq_lens,
            num_heads=args.num_heads,
            num_kv_heads=args.num_kv_heads,
            warmup_ms=args.warmup_ms,
            rep_ms=args.rep_ms,
        )
        total_chunks = sum(triton.cdiv(length, _CHUNK) for length in seq_lens)
        delta_pct = (segmented_us / stock_us - 1.0) * 100.0
        print(
            f"{'+'.join(str(length) for length in seq_lens)},"
            f"{len(seq_lens)},{total_chunks},"
            f"{stock_us:.3f},{segmented_us:.3f},{delta_pct:+.1f}"
        )


if __name__ == "__main__":
    main()
