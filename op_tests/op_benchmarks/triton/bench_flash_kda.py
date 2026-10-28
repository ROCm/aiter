# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""
Performance benchmark for the FlashKDA two-kernel forward pass.

Sweeps sequence length and head count by default; supports custom shapes via
--shape and the Kimi-K3 TP8 prefill forwards via --k3-tp8. FlashKDA is timed
by default; --impl chunk_kda times the Gluon chunk_kda on the same inputs. Run
bench_chunk_delta_attn.py for the default pipeline and compare the Time(ms)
column. TFLOPS uses the same approximation as that script but at C=32, the
chunk size FlashKDA requires, so the two TFLOPS columns count different amounts
of intra-chunk work.

Usage examples
--------------
# Sweep default shapes, report time / TFLOPS / BW
python bench_flash_kda.py

# Single shape: B=1 T=16384 H=12 K=128 V=128
python bench_flash_kda.py --shape 1 16384 12 128 128

# Kimi-K3 at H=96, with q, k and v as bands of one fused projection
python bench_flash_kda.py --qkv-bands --shape 1 4096 96 128 128 --shape 1 16384 96 128 128

# The 26 prefill forwards of Kimi-K3 TP8+DCP8 rank 0, then chunk_kda on them
python bench_flash_kda.py --k3-tp8
python bench_flash_kda.py --k3-tp8 --impl chunk_kda

# Pin segment lengths instead of letting the heuristic choose
python bench_flash_kda.py --seg-sweep

# Save CSV
python bench_flash_kda.py -o
"""

import functools
import itertools
import math
import os
import sys

# Skip CK/HIP native .so loading – Triton kernels only
os.environ.setdefault("AITER_TRITON_ONLY", "1")


# Ensure repo root is on the path when running this script directly
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch
import triton

from aiter.ops.triton._triton_kernels.kimi_delta_attn.flash_kda import (
    FLASH_KDA_CHUNK,
    flash_kda_fwd,
)
from aiter.ops.triton.attention.chunk_kda import chunk_kda, prepare_chunk_kda_metadata
from op_tests.op_benchmarks.triton.utils.argparse import get_parser
from op_tests.op_benchmarks.triton.utils.benchmark_utils import get_caller_name_no_ext

# (B, T, H, K, V) – H=12 is the Kimi K3 shape this path was built for; the rest
# bracket it so a regression in the segmentation heuristic shows up as well as a
# kernel one. K and V are fixed at 128, the only width FlashKDA supports.
DEFAULT_SHAPES = [
    (1, 256, 12, 128, 128),
    (1, 512, 12, 128, 128),
    (1, 1664, 12, 128, 128),
    (1, 2048, 12, 128, 128),
    (1, 8192, 12, 128, 128),
    (1, 16384, 12, 128, 128),
    (1, 16384, 64, 128, 128),
    (1, 16384, 96, 128, 128),
    (1, 8192, 32, 128, 128),
    (1, 8192, 64, 128, 128),
    (2, 2664, 12, 128, 128),
    (4, 4096, 16, 128, 128),
    (8, 2048, 16, 128, 128),
]
# Segment lengths in chunks; 0 means one segment, i.e. segmentation off.
SEG_SWEEP = [0, 16, 32, 64, 128]
# The prefill forwards of Kimi-K3 TP8+DCP8 rank 0 in order: the new and the
# cached tokens of each sequence in the forward. A sequence with cached tokens
# continues its prefix, so it reads its initial state from the state cache.
K3_TP8_HEADS = 12
K3_TP8_STEPS = [
    ((12934,), (0,)),
    ((13295, 3072), (0, 0)),
    ((9170, 7168), (3072, 0)),
    ((6740, 9600), (7168, 0)),
    ((2370, 11585, 2368), (9600, 0, 0)),
    ((9315, 7040), (2368, 0)),
    ((5872, 10496), (7040, 0)),
    ((1019, 13074, 2240), (10496, 0, 0)),
    ((11321, 5056), (2240, 0)),
    ((6419, 9920), (5056, 0)),
    ((2431, 13801, 128), (9920, 0, 0)),
    ((13958, 2368), (128, 0)),
    ((10840, 5504), (2368, 0)),
    ((8749, 7616), (5504, 0)),
    ((5665, 10688), (7616, 0)),
    ((2872, 13331, 128), (10688, 0, 0)),
    ((12899, 3456), (128, 0)),
    ((10694, 5632), (3456, 0)),
    ((6631, 9728), (5632, 0)),
    ((4080, 12288), (9728, 0)),
    ((1104, 13662, 1600), (12288, 0, 0)),
    ((9964, 6400), (1600, 0)),
    ((6658, 9664), (6400, 0)),
    ((4263, 12096), (9664, 0)),
    ((502, 11971, 3904), (12096, 0, 0)),
    ((9992,), (3904,)),
]

K_DIM = 128
LOWER_BOUND = -5.0
DTYPE = torch.bfloat16
DEVICE = "cuda"


def _make_inputs(B, T, H, seed=42, qkv_bands=False):
    """Create raw benchmark inputs (gate, l2norm and beta sigmoid are in-kernel).

    With qkv_bands, q, k and v are views of one fused [B, T, 3 * H * 128]
    projection, as the model passes them.
    """
    torch.manual_seed(seed)
    K = K_DIM
    if qkv_bands:
        q, k, v = torch.randn(B, T, 3, H, K, device=DEVICE, dtype=DTYPE).unbind(2)
    else:
        q, k, v = (
            torch.randn(B, T, H, K, device=DEVICE, dtype=DTYPE) for _ in range(3)
        )
    return {
        "q": q,
        "k": k,
        "v": v,
        "g": torch.randn(B, T, H, K, device=DEVICE, dtype=DTYPE) * 0.1,
        "beta": torch.randn(B, T, H, device=DEVICE, dtype=torch.float32),
        "A_log": torch.randn(H, device=DEVICE, dtype=torch.float32).abs() * 0.5,
        "dt_bias": torch.randn(H * K, device=DEVICE, dtype=torch.float32) * 0.1,
        "scale": 1.0 / math.sqrt(K),
    }


def _make_tp8_inputs(new, cached, qkv_bands=False):
    """One TP8 forward: its sequences in one varlen call, their states in a paged cache."""
    H, N, T = K3_TP8_HEADS, len(new), sum(new)
    t = _make_inputs(1, T, H, qkv_bands=qkv_bands)
    t.update(
        cu_seqlens=torch.tensor(
            [0, *itertools.accumulate(new)], dtype=torch.int32, device=DEVICE
        ),
        out=torch.empty(1, T, H, K_DIM, device=DEVICE, dtype=DTYPE),
        state_v_first=True,
        state_cache=torch.randn(N, H, K_DIM, K_DIM, device=DEVICE),
        state_indices=torch.arange(N, dtype=torch.int32, device=DEVICE),
        has_initial_state=torch.tensor([c > 0 for c in cached], device=DEVICE),
    )
    return t


def _chunk_kda_fn(t):
    """chunk_kda on the inputs of one FlashKDA call; a batch becomes one varlen call."""
    B, T = t["q"].shape[:2]
    tokens = {key: t[key].flatten(0, 1)[None] for key in ("q", "k", "v", "g", "beta")}
    cu_seqlens = t.get("cu_seqlens")
    if cu_seqlens is None:
        cu_seqlens = torch.arange(0, B * T + 1, T, dtype=torch.int32, device=DEVICE)
    # Built once: the metadata syncs the host.
    chunk_indices, chunk_offsets = prepare_chunk_kda_metadata(cu_seqlens)
    buffers = {
        key: t[key]
        for key in ("out", "state_cache", "state_indices", "has_initial_state")
        if key in t
    }

    def fn():
        return chunk_kda(
            **tokens,
            A_log=t["A_log"],
            dt_bias=t["dt_bias"],
            lower_bound=LOWER_BOUND,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            chunk_offsets=chunk_offsets,
            scale=t["scale"],
            **buffers,
        )

    return fn


def _flops(B, T, H, K, V, chunk_size):
    """
    Approximate FLOPs for the chunk_delta_attn forward pass.

    Dominant terms:
      - Intra-chunk QK:       B * T * H * chunk_size * K * 2
      - Intra-chunk AV:       B * T * H * chunk_size * V * 2
      - Inter-chunk KV state: B * T * H * K * V * 2
      - Output QH:            B * T * H * K * V * 2
    """
    return B * T * H * 2 * (chunk_size * K + chunk_size * V + 2 * K * V)


def _bytes(B, T, H, K, V, HV, elem_bytes=2):
    """Approximate memory traffic (inputs read + output written)."""
    read = (
        B * T * H * K * elem_bytes  # q
        + B * T * H * K * elem_bytes  # k
        + B * T * HV * V * elem_bytes  # v
        + B * T * HV * K * elem_bytes  # g
        + B * T * HV * elem_bytes  # beta
    )
    write = B * T * HV * V * elem_bytes  # o
    return read + write


def _seg_label(chunks_per_seg):
    if chunks_per_seg is None:
        return "auto"
    return "1seg" if chunks_per_seg <= 0 else str(chunks_per_seg)


def run_benchmark(args):
    if args.k3_tp8:
        x_vals_list = [
            (1, sum(new), K3_TP8_HEADS, K_DIM, K_DIM) for new, _ in K3_TP8_STEPS
        ]
    elif args.shape is not None:
        x_vals_list = [tuple(shape) for shape in args.shape]
        for _, _, _, K, V in x_vals_list:
            if K != K_DIM or V != K_DIM:
                raise SystemExit(
                    f"flash_kda only supports K=V={K_DIM}, got K={K} V={V}"
                )
    else:
        x_vals_list = DEFAULT_SHAPES

    header = f"{'B':>4} {'T':>6} {'H':>4} {'K':>4} {'V':>4}  {'Time(ms)':>10}  {'TFLOPS':>8}  {'BW(GB/s)':>10}"
    if args.seg_sweep:
        header += f"  {'seg':>6}"
    if args.k3_tp8:
        header += "  seqlens"
    print(header)
    print("-" * len(header))

    rows = []
    for i, (B, T, H, K, V) in enumerate(x_vals_list):
        HV = H
        if args.k3_tp8:
            t = _make_tp8_inputs(*K3_TP8_STEPS[i], qkv_bands=args.qkv_bands)
        else:
            t = _make_inputs(B, T, H, qkv_bands=args.qkv_bands)

        if args.seg_sweep:
            # A segment longer than the sequence would leave the grid empty.
            segs = [None] + [m for m in SEG_SWEEP if not m or m * FLASH_KDA_CHUNK <= T]
        else:
            segs = [None]

        for m in segs:
            if args.impl == "chunk_kda":
                fn = _chunk_kda_fn(t)
            else:
                fn = functools.partial(
                    flash_kda_fwd, **t, lower_bound=LOWER_BOUND, chunks_per_seg=m
                )

            # Time-based warmup: run until WARMUP_MS elapsed so JIT/autotune completes.
            _elapsed = 0.0
            while _elapsed < args.warmup_ms:
                _t0 = torch.cuda.Event(enable_timing=True)
                _t1 = torch.cuda.Event(enable_timing=True)
                _t0.record()
                fn()
                _t1.record()
                torch.cuda.synchronize()
                _elapsed += _t0.elapsed_time(_t1)

            ms = triton.testing.do_bench(fn, warmup=0, rep=args.rep_ms)
            tflops = _flops(B, T, H, K, V, FLASH_KDA_CHUNK) / ms * 1e-9
            bw = _bytes(B, T, H, K, V, HV) / (ms * 1e-3) * 1e-9

            line = f"{B:>4} {T:>6} {H:>4} {K:>4} {V:>4}  {ms:>10.4f}  {tflops:>8.2f}  {bw:>10.1f}"
            row = [B, T, H, K, V, ms, tflops, bw]
            if args.seg_sweep:
                line += f"  {_seg_label(m):>6}"
                row.append(_seg_label(m))
            if args.k3_tp8:
                seqlens = "+".join(map(str, K3_TP8_STEPS[i][0]))
                line += f"  {seqlens}"
                row.append(seqlens)
            print(line)
            rows.append(tuple(row))

    if args.o:
        import csv

        fname = f"{get_caller_name_no_ext()}.csv"
        cols = ["B", "T", "H", "K", "V", "Time_ms", "TFLOPS", "BW_GBs"]
        if args.seg_sweep:
            cols.append("seg")
        if args.k3_tp8:
            cols.append("seqlens")
        with open(fname, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(cols)
            w.writerows(rows)
        print(f"\nSaved to {fname}")


def parse_args(args=None):
    parser = get_parser(kernel_name="FlashKDA forward")
    cases = parser.add_mutually_exclusive_group()
    cases.add_argument(
        "--shape",
        type=int,
        nargs=5,
        action="append",
        metavar=("B", "T", "H", "K", "V"),
        help="Shape to benchmark instead of the default sweep; repeat for more "
        f"shapes. K and V must be {K_DIM}.",
    )
    cases.add_argument(
        "--k3-tp8",
        action="store_true",
        help=f"Time the {len(K3_TP8_STEPS)} prefill forwards of Kimi-K3 TP8+DCP8 "
        f"rank 0 (H={K3_TP8_HEADS}) instead of the default sweep. Each forward is "
        "one varlen call. The sequences that continue a cached prefix read their "
        "initial state from a paged state cache.",
    )
    parser.add_argument(
        "--qkv-bands",
        action="store_true",
        help="Pass q, k and v as views of one fused projection, as the model "
        "does, instead of as three packed tensors.",
    )
    parser.add_argument(
        "--impl",
        choices=["flash_kda", "chunk_kda"],
        default="flash_kda",
        help="Kernel to time. chunk_kda is the Gluon chunked KDA prefill; it gets "
        "the same inputs as FlashKDA.",
    )
    parser.add_argument(
        "--seg-sweep",
        action="store_true",
        help="Also time pinned segment lengths, which is how the defaults in "
        "_choose_chunks_per_seg were calibrated. Rerun after changing anything "
        "that moves the crossover: segment cost, the recurrence kernel's block "
        "count, or the hardware.",
    )
    parser.add_argument(
        "-o",
        action="store_true",
        help="Write results to a CSV file in the current directory.",
    )
    parser.add_argument(
        "--warmup-ms",
        type=float,
        default=300.0,
        help="Warmup budget in ms (time-based, ensures JIT/autotune completes).",
    )
    parser.add_argument(
        "--rep-ms",
        type=float,
        default=500.0,
        help="Measurement budget in ms passed to triton.testing.do_bench.",
    )
    args = parser.parse_args(args=args)
    if args.seg_sweep and args.impl != "flash_kda":
        parser.error("--seg-sweep pins FlashKDA segment lengths; use --impl flash_kda")
    return args


def main(args=None):
    run_benchmark(parse_args(args=args))


if __name__ == "__main__":
    main()
