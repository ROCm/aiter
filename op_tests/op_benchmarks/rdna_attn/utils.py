# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Roofline, knobs and --dtype shared by matrix.py, tune.py and floor.py."""

import argparse
import os

# Measured on each GPU: sustained DRAM GiB/s, WMMA f16/bf16 TFLOPS, SIMDs,
# per-kernel dispatch overhead under a HIP graph (us), DRAM round trip (us)
# and cycles of one 16x16x16 f16 WMMA; gfx1201's on an RX 9070 XT. The GPU is
# the current device's (RDNA_ROOFLINE_ARCH overrides it, e.g. to analyse
# results elsewhere); any other GPU gets gfx1151's.
PEAKS = {
    "gfx1151": (230.0, 43.0, 80, 1.48, 0.545, 32),
    "gfx1150": (81.0, 17.2, 32, 1.48, 0.545, 32),
    "gfx1201": (596.0, 128.0, 128, 2.15, 0.30, 16),
}


def _arch() -> str:
    arch = os.environ.get("RDNA_ROOFLINE_ARCH")
    if arch:
        return arch
    try:
        from aiter.jit.utils.chip_info import get_gfx_runtime

        return get_gfx_runtime()
    except Exception:  # noqa: BLE001 - no GPU here, or aiter is not built
        return "gfx1151"


PEAK_GIBS, PEAK_TFLOPS, SIMDS, DISPATCH_US, DRAM_LATENCY_US, WMMA_CYCLES = PEAKS.get(
    _arch(), PEAKS["gfx1151"]
)

# Prefill model (roofline(realistic=True)): whole 16x16 score tiles, plus
# SOFTMAX_CYCLES of VALU work per tile that does not overlap the WMMAs.
WMMA_TILE = 16
WMMA_GHZ = PEAK_TFLOPS * 1e12 * WMMA_CYCLES / (SIMDS * 2 * WMMA_TILE**3) / 1e9
VALU_OPS_PER_SCORE = 4.5
VALU_LANES = 32
SOFTMAX_CYCLES = VALU_OPS_PER_SCORE * WMMA_TILE**2 / VALU_LANES
# Plus three dependent DRAM_LATENCY_US round trips: page table, KV, output.

# Arithmetic intensity (FLOP/byte) where the roofs meet.
RIDGE = PEAK_TFLOPS * 1e12 / (PEAK_GIBS * 1024**3)

# Decode launch knobs tune.py searches and matrix.py can force.
KNOBS = (
    "max_segments",
    "row_groups",
    "min_segment_blocks",
    "waves",
    "head_dim_split",
    "prefetch",
    "v_in_lds",
    "dot_product",
)

# Prefill launch knobs, likewise.
PREFILL_KNOBS = (
    "waves",
    "key_tile",
    "head_dim_split",
    "v_t",
    "double_buffer",
    "key_waves",
    "max_segments",
    "head_group",
    "row_tiles",
)


def _visible_keys(m: int, s: int, window: int) -> int:
    """Keys seen by the m queries of a sequence of length s, summed."""
    prefix = s - m
    if not window:
        return m * prefix + m * (m + 1) // 2
    return sum(min(prefix + t + 1, window) for t in range(m))


def score_tiles(hq: int, hkv: int, m: int, s: int, window: int = 0) -> int:
    """Fewest 16x16 score tiles a WMMA kernel computes."""
    t, g, prefix = WMMA_TILE, hq // hkv, s - m
    tiles = 0
    for first_row in range(0, m * g, t):
        first, last = first_row // g, min(m - 1, (first_row + t - 1) // g)
        begin = max(0, prefix + first - (window - 1)) // t * t if window else 0
        end = -(-(prefix + last + 1) // t) * t
        tiles += (end - begin) // t
    return tiles * hkv


def attention_flops(hq: int, d: int, m: int, s: int, window: int = 0) -> float:
    """FLOPs of Q@K and P@V over the keys each query sees."""
    return 4.0 * hq * d * _visible_keys(m, s, window)


def attention_bytes(
    hq: int,
    hkv: int,
    d: int,
    m: int,
    s: int,
    itemsize: int = 2,
    block_size: int = 16,
    window: int = 0,
) -> float:
    """Bytes moved: Q, the visible KV, the output and the block table."""
    keys = s if not window else min(s, window + m - 1)
    query = m * hq * d * itemsize
    kv = keys * hkv * 2 * d * itemsize
    out = m * hq * d * itemsize
    table = -(-keys // block_size) * 4
    return query + kv + out + table


def attention_compute_us(
    hq: int, hkv: int, d: int, m: int, s: int, window: int = 0, realistic: bool = False
) -> float:
    """Compute time of `roofline`, in microseconds."""
    if not realistic:
        return attention_flops(hq, d, m, s, window) / (PEAK_TFLOPS * 1e6)
    tiles = score_tiles(hq, hkv, m, s, window)
    return tiles * 4 * d * WMMA_TILE**2 / (PEAK_TFLOPS * 1e6) + (
        tiles * SOFTMAX_CYCLES / (SIMDS * WMMA_GHZ * 1e3)
    )


def roofline(
    hq: int,
    hkv: int,
    d: int,
    m: int,
    s: int,
    itemsize: int = 2,
    block_size: int = 16,
    window: int = 0,
    realistic: bool = False,
) -> tuple[float, float, str]:
    """The time floor of one attention call of one sequence: the longer of
    its bytes at peak bandwidth and its FLOPs at peak throughput, plus one
    dispatch.

    Args:
        hq: Query heads.
        hkv: KV heads.
        d: Head dimension.
        m: Query tokens.
        s: Sequence length, the m tokens included.
        itemsize: Bytes per element of Q, K, V and the output.
        block_size: KV cache page size, for the block table.
        window: Sliding window in keys, 0 for full attention.
        realistic: Use the WMMA prefill model.

    Returns:
        (microseconds, FLOP/byte, "memory" or "compute").

    """
    flops = attention_flops(hq, d, m, s, window)
    moved = attention_bytes(hq, hkv, d, m, s, itemsize, block_size, window)
    memory_us = moved / (PEAK_GIBS * 1024**3) * 1e6
    compute_us = attention_compute_us(hq, hkv, d, m, s, window, realistic)
    bound = "compute" if compute_us > memory_us else "memory"
    fixed = DISPATCH_US + (3 * DRAM_LATENCY_US if realistic else 0)
    return fixed + max(memory_us, compute_us), flops / moved, bound


def roofline_us(
    hq: int,
    hkv: int,
    d: int,
    m: int,
    s: int,
    itemsize: int = 2,
    block_size: int = 16,
    window: int = 0,
) -> float:
    """`roofline`'s floor alone, in microseconds."""
    return roofline(hq, hkv, d, m, s, itemsize, block_size, window)[0]


def roofline_noncausal(
    hq: int, hkv: int, d: int, segs: list[int], itemsize: int = 2
) -> tuple[float, str]:
    """The floor of one non-causal call over sequences of `segs` tokens, each
    query seeing every key of its sequence, with the WMMA prefill model: Q, K,
    V and the output at peak bandwidth against the 16x16 score tiles at peak
    throughput, plus a dispatch and three DRAM round trips.

    Returns:
        (microseconds, "memory" or "compute").

    """
    t, g = WMMA_TILE, hq // hkv
    n = sum(segs)
    moved = (2 * n * hq + 2 * n * hkv) * d * itemsize
    memory_us = moved / (PEAK_GIBS * 1024**3) * 1e6
    tiles = sum(-(-s * g // t) * -(-s // t) for s in segs) * hkv
    compute_us = tiles * 4 * d * t**2 / (PEAK_TFLOPS * 1e6) + (
        tiles * SOFTMAX_CYCLES / (SIMDS * WMMA_GHZ * 1e3)
    )
    bound = "compute" if compute_us > memory_us else "memory"
    return DISPATCH_US + 3 * DRAM_LATENCY_US + max(memory_us, compute_us), bound


def spec_roofline(
    spec: str,
    hq: int,
    hkv: int,
    d: int,
    itemsize: int = 2,
    block_size: int = 16,
    window: int = 0,
    realistic: bool = False,
) -> tuple[float, float, str]:
    """`roofline` of a batch spec (batch_spec.py): (us, FLOP/byte, bound)."""
    from batch_spec import parse_batch_spec

    reqs = parse_batch_spec(spec)
    args = (itemsize, block_size, window)
    flops = sum(attention_flops(hq, d, r.q_len, r.kv_len, window) for r in reqs)
    moved = sum(attention_bytes(hq, hkv, d, r.q_len, r.kv_len, *args) for r in reqs)
    memory_us = moved / (PEAK_GIBS * 1024**3) * 1e6
    compute_us = sum(
        attention_compute_us(hq, hkv, d, r.q_len, r.kv_len, window, realistic)
        for r in reqs
    )
    bound = "compute" if compute_us > memory_us else "memory"
    fixed = DISPATCH_US + (3 * DRAM_LATENCY_US if realistic else 0)
    return fixed + max(memory_us, compute_us), flops / moved, bound


def add_dtype_argument(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--dtype",
        choices=("fp16", "bf16"),
        default="fp16",
        help="element type of Q, the KV cache and the output",
    )


def torch_dtype(name: str):
    """The torch dtype for a `--dtype` value."""
    import torch

    return {"fp16": torch.float16, "bf16": torch.bfloat16}[name]
