# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Benchmark: does the cross-layer 5D stride pattern slow down the prefill kernel?

This script answers exactly one question: when ``mha_batch_prefill_func`` is
fed a non-contiguous ``[N, H, B, D]`` cross-layer view instead of a contiguous
``[N, B, H, D]`` packed buffer, does the prefill kernel itself run faster,
slower, or the same?

It runs ``aiter.mha_batch_prefill_func`` on identical numerical K/V data
through two layouts:

  1. ``packed_contig``   -- legacy LINEAR_LAYOUT, contiguous ``[N, B, H, D]``.
                            Per-layer buffer allocated standalone.
  2. ``xlayer_strided``  -- LINEAR_HEADS_FIRST_LAYOUT, non-contiguous
                            ``[N, H, B, D]`` sliced from a single 6D
                            ``[N, H, L, 2, B, D]`` cross-layer physical buffer.

For each shape we report:

  * ``packed_us``         kernel time on the contiguous packed layout
                          (median CUDA event time, microseconds).
  * ``xlayer_us``         kernel time on the non-contiguous cross-layer view
                          (median CUDA event time, microseconds).
  * ``xlayer/packed``     ratio of xlayer to packed kernel time. > 1.0 means
                          the stride change costs us kernel performance.
                          < 1.0 means the stride change is faster.
                          ~1.0 means it makes no measurable difference.

Before timing, packed and xlayer outputs are checked numerically equal so the
two paths are doing the same work. Per-call overhead from the Python wrapper
is included in both numbers (apples-to-apples), but no per-step memcpy or
allocation of the packed buffer is included -- this is strictly the
kernel-launch cost.

Usage:
    PYTHONPATH=$PWD python op_tests/bench_cross_layer_5d.py
    PYTHONPATH=$PWD python op_tests/bench_cross_layer_5d.py --preset hy3_16k_b1
    PYTHONPATH=$PWD python op_tests/bench_cross_layer_5d.py \\
        --batch_size 1 --prefill_len 8192 --num_layers 80 \\
        --num_kv_heads 1 --num_qo_heads 32 --head_dim 128
"""

import argparse
import dataclasses
import sys
from typing import List, Tuple

import torch

import aiter


# ---------------------------------------------------------------------------
# Shape presets (intended as realistic prefill workloads, not exhaustive sweeps).
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Shape:
    name: str
    batch_size: int
    prefill_len: int
    num_kv_heads: int
    num_qo_heads: int
    head_dim: int
    num_layers: int
    page_size: int = 16
    causal: bool = True
    dtype: torch.dtype = torch.bfloat16


PRESETS: List[Shape] = [
    # HY3-like: GQA H_kv=1 -> H_q=32, head_dim=128, page_size=16, 80 layers,
    # bf16. Three context lengths; only the kernel-perf delta matters here,
    # not absolute time.
    Shape("hy3_2k_b1", 1, 2048, 1, 32, 128, 80),
    Shape("hy3_8k_b1", 1, 8192, 1, 32, 128, 80),
    Shape("hy3_16k_b1", 1, 16384, 1, 32, 128, 80),
    # GQA H_kv=2 to expose the head-stride difference between the two
    # layouts. Per-block stride magnitude is the same; intra-block access
    # pattern differs (packed: heads-within-token contiguous; xlayer:
    # tokens-within-head contiguous).
    Shape("gqa2_4k_b1", 1, 4096, 2, 16, 128, 80),
    # Smaller layer count to confirm the cross-layer factor in strides
    # doesn't scale the kernel cost.
    Shape("hy3_4k_l4", 1, 4096, 1, 32, 128, 4),
]


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------


def _median(values: List[float]) -> float:
    s = sorted(values)
    n = len(s)
    return s[n // 2] if n % 2 == 1 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def time_cuda(fn, warmup: int = 3, iters: int = 20) -> Tuple[float, float, float]:
    """Run ``fn()`` and return ``(median_ms, min_ms, mean_ms)`` of CUDA-event times."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize()
    times = [s.elapsed_time(e) for s, e in zip(starts, ends)]  # ms
    return _median(times), min(times), sum(times) / len(times)


# ---------------------------------------------------------------------------
# Build the input tensors for one shape
# ---------------------------------------------------------------------------


def build_inputs(shape: Shape, device: str = "cuda"):
    """Build all tensors needed to run ``mha_batch_prefill_func`` on ``shape``.

    Returns a dict with:
      q, cu_seqlens_q, kv_indptr, kv_page_indices, kv_last_page_lens,
      max_seqlen_q, max_seqlen_k, num_blocks, page_size,
      kv_packed -- 5D ``[2, N, B, H, D]`` contiguous K/V (legacy layout),
      kv_xlayer -- 5D ``[2, N, H, B, D]`` non-contiguous view of the 6D
                   physical buffer, layer slot filled with the same
                   numerical data as ``kv_packed`` (so kernel outputs can
                   also be sanity-checked equal).
    """
    B = shape.page_size
    D = shape.head_dim
    H = shape.num_kv_heads
    L = shape.num_layers
    bs = shape.batch_size
    plen = shape.prefill_len
    layer_idx = L // 2  # arbitrary non-zero layer

    # Round prefill_len up to multiple of page_size so the last page is full.
    pages_per_seq = (plen + B - 1) // B
    full_kv_len = pages_per_seq * B
    last_page_len = plen - (pages_per_seq - 1) * B
    if last_page_len <= 0:
        last_page_len = B

    num_blocks = pages_per_seq * bs

    torch.manual_seed(0xC8055CA5)

    # 1. Cross-layer 6D physical buffer + per-layer 5D non-contiguous view.
    physical = torch.zeros(
        num_blocks, H, L, 2, B, D, dtype=shape.dtype, device=device,
    )
    # (L, 2, N, H, B, D) -- non-contiguous per-layer view family.
    logical6d = physical.permute(2, 3, 0, 1, 4, 5)
    # Random K/V data in the layer-of-interest slot.
    layer_data = (
        torch.randn(2, num_blocks, H, B, D, dtype=shape.dtype, device=device) * 0.1
    )
    logical6d[layer_idx].copy_(layer_data)
    kv_xlayer = logical6d[layer_idx]  # (2, N, H, B, D), non-contiguous
    assert kv_xlayer.stride() == (B * D, L * 2 * H * B * D, L * 2 * B * D, D, 1)
    assert not kv_xlayer.is_contiguous()

    # 2. Packed [2, N, B, H, D] contiguous baseline. Same numerical data as the
    #    cross-layer slot but laid out the legacy way. Built via permute
    #    (cheap; not part of the timed region).
    kv_packed = torch.empty(2, num_blocks, B, H, D, dtype=shape.dtype, device=device)
    # kv_xlayer[i] is (N, H, B, D); permute to (N, B, H, D) and copy.
    kv_packed[0].copy_(kv_xlayer[0].permute(0, 2, 1, 3))
    kv_packed[1].copy_(kv_xlayer[1].permute(0, 2, 1, 3))
    assert kv_packed.is_contiguous()

    # 3. Page tables and queries.
    cu_seqlens_q = torch.tensor(
        [i * plen for i in range(bs + 1)], dtype=torch.int32, device=device,
    )
    kv_indptr = torch.tensor(
        [i * pages_per_seq for i in range(bs + 1)], dtype=torch.int32, device=device,
    )
    kv_page_indices = torch.arange(num_blocks, dtype=torch.int32, device=device)
    kv_last_page_lens = torch.full(
        (bs,), last_page_len, dtype=torch.int32, device=device,
    )

    q = torch.randn(
        bs * plen, shape.num_qo_heads, D, dtype=shape.dtype, device=device,
    ) * 0.1

    return {
        "q": q,
        "cu_seqlens_q": cu_seqlens_q,
        "kv_indptr": kv_indptr,
        "kv_page_indices": kv_page_indices,
        "kv_last_page_lens": kv_last_page_lens,
        "max_seqlen_q": plen,
        "max_seqlen_k": full_kv_len,
        "kv_packed": kv_packed,
        "kv_xlayer": kv_xlayer,
        "causal": shape.causal,
    }


# ---------------------------------------------------------------------------
# The two call paths under test
# ---------------------------------------------------------------------------


def call_packed_contig(args):
    """Run prefill on the contiguous packed [N, B, H, D] K and V tensors."""
    return aiter.mha_batch_prefill_func(
        args["q"],
        args["kv_packed"][0],
        args["kv_packed"][1],
        args["cu_seqlens_q"],
        args["kv_indptr"],
        args["kv_page_indices"],
        args["max_seqlen_q"],
        args["max_seqlen_k"],
        causal=args["causal"],
        kv_last_page_lens=args["kv_last_page_lens"],
    )


def call_xlayer_strided(args):
    """Run prefill on the non-contiguous [N, H, B, D] cross-layer 5D view."""
    return aiter.mha_batch_prefill_func(
        args["q"],
        cu_seqlens_q=args["cu_seqlens_q"],
        kv_indptr=args["kv_indptr"],
        kv_page_indices=args["kv_page_indices"],
        max_seqlen_q=args["max_seqlen_q"],
        max_seqlen_k=args["max_seqlen_k"],
        causal=args["causal"],
        kv_cache=args["kv_xlayer"],
        kv_last_page_lens=args["kv_last_page_lens"],
    )


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def bench_one(shape: Shape, *, warmup: int, iters: int, sanity: bool) -> dict:
    args = build_inputs(shape)

    if sanity:
        out_pk = call_packed_contig(args)
        out_xl = call_xlayer_strided(args)
        out_pk_t = out_pk[0] if isinstance(out_pk, (list, tuple)) else out_pk
        out_xl_t = out_xl[0] if isinstance(out_xl, (list, tuple)) else out_xl
        # bf16 tolerance per the existing cross-layer matches_contiguous test.
        torch.testing.assert_close(out_xl_t, out_pk_t, rtol=2e-2, atol=1e-2)

    pk_med, pk_min, _ = time_cuda(
        lambda: call_packed_contig(args), warmup=warmup, iters=iters,
    )
    xl_med, xl_min, _ = time_cuda(
        lambda: call_xlayer_strided(args), warmup=warmup, iters=iters,
    )

    return {
        "shape": shape.name,
        "params": (
            f"bs={shape.batch_size} plen={shape.prefill_len} "
            f"H_kv={shape.num_kv_heads} H_q={shape.num_qo_heads} "
            f"D={shape.head_dim} L={shape.num_layers} causal={shape.causal}"
        ),
        "packed_us": pk_med * 1000.0,
        "packed_min_us": pk_min * 1000.0,
        "xlayer_us": xl_med * 1000.0,
        "xlayer_min_us": xl_min * 1000.0,
        "ratio_xlayer_over_packed": (xl_med / pk_med) if pk_med > 0 else float("nan"),
    }


def print_table(rows: List[dict]) -> None:
    headers = [
        "shape",
        "packed_us",
        "xlayer_us",
        "xlayer/packed",
    ]
    widths = [18, 12, 12, 14]
    sep = "  "

    print(sep.join(h.ljust(w) for h, w in zip(headers, widths)))
    print(sep.join("-" * w for w in widths))
    for r in rows:
        cells = [
            r["shape"],
            f"{r['packed_us']:.1f}",
            f"{r['xlayer_us']:.1f}",
            f"{r['ratio_xlayer_over_packed']:.3f}",
        ]
        print(sep.join(c.ljust(w) for c, w in zip(cells, widths)))

    # Per-row params printed underneath in a smaller block, for readability.
    print()
    for r in rows:
        print(f"  {r['shape']:<18}  {r['params']}")
    print()
    print("  xlayer/packed > 1.0 -> non-contiguous cross-layer strides cost kernel perf")
    print("  xlayer/packed < 1.0 -> non-contiguous cross-layer strides help kernel perf")
    print("  xlayer/packed ~ 1.0 -> stride change has no measurable effect on kernel")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument(
        "--preset",
        choices=[s.name for s in PRESETS] + ["all"],
        default="all",
        help="Run a single preset shape, or 'all' (default).",
    )
    p.add_argument("--batch_size", type=int)
    p.add_argument("--prefill_len", type=int)
    p.add_argument("--num_kv_heads", type=int)
    p.add_argument("--num_qo_heads", type=int)
    p.add_argument("--head_dim", type=int, default=128)
    p.add_argument("--num_layers", type=int, default=80)
    p.add_argument("--page_size", type=int, default=16)
    p.add_argument("--causal", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument(
        "--no-sanity",
        dest="sanity",
        action="store_false",
        default=True,
        help="Skip the packed-vs-xlayer numerical equality check before timing.",
    )
    return p.parse_args()


def main():
    args = parse_args()

    if not torch.cuda.is_available():
        print("CUDA/HIP device not available", file=sys.stderr)
        sys.exit(1)

    # Custom shape via CLI (overrides preset).
    if all(
        getattr(args, f) is not None
        for f in ("batch_size", "prefill_len", "num_kv_heads", "num_qo_heads")
    ):
        shapes = [
            Shape(
                name="cli",
                batch_size=args.batch_size,
                prefill_len=args.prefill_len,
                num_kv_heads=args.num_kv_heads,
                num_qo_heads=args.num_qo_heads,
                head_dim=args.head_dim,
                num_layers=args.num_layers,
                page_size=args.page_size,
                causal=args.causal,
            )
        ]
    elif args.preset == "all":
        shapes = list(PRESETS)
    else:
        shapes = [s for s in PRESETS if s.name == args.preset]

    print(f"# device : {torch.cuda.get_device_name(0)}")
    print(f"# torch  : {torch.__version__} (HIP {torch.version.hip})")
    print(f"# warmup={args.warmup}, iters={args.iters}, sanity={args.sanity}")
    print()

    rows = []
    for s in shapes:
        try:
            row = bench_one(s, warmup=args.warmup, iters=args.iters, sanity=args.sanity)
        except Exception as e:
            print(f"  {s.name}  FAILED: {type(e).__name__}: {e}", file=sys.stderr)
            continue
        rows.append(row)
        # Streaming output: print each row as soon as it's measured (so a long
        # bench is observable while it's still running).
        print(
            f"  {row['shape']:<18}  "
            f"packed={row['packed_us']:>9.1f}us  "
            f"xlayer={row['xlayer_us']:>9.1f}us  "
            f"  xlayer/packed={row['ratio_xlayer_over_packed']:>5.3f}"
        )

    if not rows:
        sys.exit(2)

    print()
    print_table(rows)


if __name__ == "__main__":
    main()
