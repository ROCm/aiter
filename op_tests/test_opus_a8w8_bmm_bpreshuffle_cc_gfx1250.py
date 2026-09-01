# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Correctness gate for the gfx1250 a8w8 mxscale bpreshuffle BMM, CLUSTER-LAUNCH
fused split-K variant (opus_bmm_a8w8_mxscale_bpreshuffle_clusterclaunch).

The sibling file test_opus_a8w8_bmm_bpreshuffle_gfx1250.py covers the fragment
maps and the scale path at splitK=1. This one covers what the cluster variant
ADDS: the balanced K split, the DataWs workspace round-trip, the -3 cluster
barrier, the fused last-split reduce, and the B multicast across M-peers.

TWO QUESTIONS, KEPT APART. The fp8 fragment maps are shared with the sibling
kernel, so a map bug would fail both and tell you nothing about split-K:

  Q1  splitK > 1 vs the SAME kernel at splitK = 1  -- split-K plumbing alone.
      A map bug cancels out of this comparison.
  Q2  the kernel vs an fp64 reference               -- everything, end to end.

Inputs are small integers with power-of-two e8m0 scales, so every product and
partial sum is exactly representable in fp32 and the fp32 path must match the
fp64 reference BIT-EXACTLY at every splitK. That exactness is the whole point:
it leaves no tolerance for a split-K bug to hide in. The bf16 path is allowed
the single rounding of its C store (the suite's rtol=0.02 / atol=0.5).

WHY THE WORKSPACE IS FP32 EVEN FOR A BF16 C. Measured, not assumed. With bf16
partials the max absolute error barely moves (15 -> 17 on a tile whose absmax is
4924, under one bf16 ULP) but 0.05%-1.3% of cells fall outside atol=0.5: a cell
whose FINAL value is near zero but whose PARTIALS are large has each partial
rounded at the partial's magnitude, and the cancellation leaves an absolute
error the final magnitude cannot absorb. With fp32 partials the max error is
IDENTICAL at splitK 1/2/4/8 -- split-K adds nothing at all. If you ever trade
back to bf16 partials for bandwidth, this file is what will catch it.

Usage (on gfx1250 hardware, inside the dev container):
    ROCM_HOME=/opt/rocm GPU_ARCHS=gfx1250 PYTHONPATH=. \
        python3 op_tests/test_opus_a8w8_bmm_bpreshuffle_cc_gfx1250.py

See the sibling file's docstring for why ROCM_HOME is not optional here.
"""
from __future__ import annotations

import sys

import torch

from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.ops.opus.bmm_op import (
    _opus_bmm_a8w8_mxscale_bpreshuffle_clusterclaunch_raw as _cc_raw,
    bmm_a8w8_mxscale_bpreshuffle_cc_ws,
)
from aiter.ops.shuffle import shuffle_weight

torch.set_default_device("cuda")

GROUP_K = 128
E8M0_ONE = 127  # 0x7F -> 2^0
E8M0_SPREAD = 2

# fp32 must be bit-exact (see docstring); bf16 gets its one store rounding.
FP32_RTOL, FP32_ATOL = 1e-6, 1e-3
BF16_RTOL, BF16_ATOL = 0.02, 0.5

SPLITS = (1, 2, 4, 8)


def _fp8_ints(shape, lo, hi, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return torch.randint(lo, hi + 1, shape, generator=g, device="cuda").to(dtypes.fp8)


def _e8m0(shape, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return torch.randint(
        E8M0_ONE - E8M0_SPREAD,
        E8M0_ONE + E8M0_SPREAD + 1,
        shape,
        generator=g,
        device="cuda",
        dtype=torch.uint8,
    ).view(dtypes.fp8_e8m0)


def _ref_bmm(O, W, xs, ws):
    """fp64 reference. Scales are per (row/col, K-group), applied before the sum."""
    a = O.to(torch.float64)
    b = W.to(torch.float64)
    sa = torch.exp2(xs.view(torch.uint8).to(torch.float64) - E8M0_ONE)
    sb = torch.exp2(ws.view(torch.uint8).to(torch.float64) - E8M0_ONE)
    g, m, k = a.shape
    n = b.shape[1]
    a = a.view(g, m, k // GROUP_K, GROUP_K) * sa.unsqueeze(-1)
    b = b.view(g, n, k // GROUP_K, GROUP_K) * sb.unsqueeze(-1)
    return torch.einsum("gmcd,gncd->gmn", a, b)


def _build(m, n, k, g, seed=0):
    O = _fp8_ints((g, m, k), 1, 4, seed)
    W = _fp8_ints((g, n, k), -2, 2, seed + 1)
    xs = _e8m0((g, m, k // GROUP_K), seed + 2)
    ws = _e8m0((g, n, k // GROUP_K), seed + 3)
    # Axis order is fixed at (M, batch, *) for A / Y / x_scale; wo_a and w_scale
    # stay batch-major. The transposes are what make the launcher's *_batch
    # strides meaningful and must not be "simplified" away.
    return (
        O.transpose(0, 1).contiguous(),
        shuffle_weight(W, layout=(16, 16)).contiguous(),
        xs.transpose(0, 1).contiguous(),
        ws,
        _ref_bmm(O, W, xs, ws).transpose(0, 1),
    )


def _run(O_in, W_shuf, xs_in, wsc, m, n, g, kid, splitK, mcwg, out_dtype, bias=None):
    Y = torch.zeros(m, g, n, dtype=out_dtype)
    # The workspace is FP32 for both C dtypes -- see the module docstring.
    ws = bmm_a8w8_mxscale_bpreshuffle_cc_ws(m, n, g, splitK, kid)
    empty = torch.empty(0, dtype=out_dtype)
    _cc_raw(
        O_in, W_shuf, Y, xs_in, wsc, ws,
        empty if bias is None else bias,
        splitK, mcwg, kid,
    )
    torch.cuda.synchronize()
    return Y


# name,                m,   n,    k,   g, kid, mcwg
CASES = [
    ("prefill kid0",   256, 256, 2048, 1, 0,  1),
    ("prefill mcwg2",  256, 256, 2048, 1, 0,  2),   # B multicast across M-peers
    ("prefill batch2", 256, 256, 2048, 2, 0,  1),   # the *_batch strides
    ("prefill kid13",  256, 256, 2048, 1, 13, 1),   # A's scale panel in LDS
    ("decode kid1",     16, 256, 4096, 1, 1,  1),
    ("decode kid4",     16, 256, 4096, 1, 4,  1),   # 6-wave tile
    ("decode batch4",   16, 128, 4096, 4, 1,  1),
    ("K tail 2176",    256, 256, 2176, 1, 0,  1),   # partial B_K tile inside a split
    ("masked M/N",      48,  48, 2048, 2, 0,  1),   # tile edges past m / n
]


def main() -> int:
    gfx = get_gfx_runtime()
    print(f"runtime gfx = {gfx}")
    if gfx != "gfx1250":
        print("SKIP: cluster-launch bpreshuffle BMM is gfx1250-only")
        return 0

    failures = 0
    for out_dtype, rtol, atol in (
        (dtypes.fp32, FP32_RTOL, FP32_ATOL),
        (dtypes.bf16, BF16_RTOL, BF16_ATOL),
    ):
        label = str(out_dtype).split(".")[-1]
        print(f"\n=== out={label}  rtol={rtol} atol={atol} ===")
        for name, m, n, k, g, kid, mcwg in CASES:
            O_in, W_shuf, xs_in, wsc, ref = _build(m, n, k, g)
            base = _run(O_in, W_shuf, xs_in, wsc, m, n, g, kid, 1, mcwg, out_dtype)
            cells = []
            ok = True
            for sk in SPLITS:
                # Every split must own >= 1 B_K tile; the launcher enforces it.
                if sk > (k + 255) // 256 or sk * mcwg > 16:
                    cells.append(f"sk{sk}:n/a")
                    continue
                y = _run(O_in, W_shuf, xs_in, wsc, m, n, g, kid, sk, mcwg, out_dtype)
                yd = y.to(torch.float64)
                # Q2: against the fp64 reference.
                bad_ref = int(((yd - ref).abs() > atol + rtol * ref.abs()).sum())
                # Q1: against splitK=1. fp32 must be IDENTICAL; bf16 may differ
                # only by what its own store rounding already allows.
                bad_b = int(
                    (
                        (yd - base.to(torch.float64)).abs()
                        > atol + rtol * base.to(torch.float64).abs()
                    ).sum()
                )
                cells.append(f"sk{sk}:{'.' if not (bad_ref or bad_b) else f'R{bad_ref}/B{bad_b}'}")
                ok &= not (bad_ref or bad_b)
            failures += not ok
            print(
                f"{'PASS' if ok else 'FAIL'}  {name:<15} m={m:<4} n={n:<4} k={k:<5} "
                f"g={g} kid={kid:<2} mcwg={mcwg} | {' '.join(cells)}"
            )

    # -- guards: the launcher must REFUSE these, not silently cope -----------
    O_in, W_shuf, xs_in, wsc, _ = _build(256, 256, 2048, 1)
    guards = 0
    for why, fn in (
        (
            "undersized workspace",
            lambda: _cc_raw(
                O_in, W_shuf, torch.zeros(256, 1, 256, dtype=dtypes.fp32), xs_in, wsc,
                torch.empty(8, dtype=dtypes.fp32), torch.empty(0, dtype=dtypes.fp32),
                4, 1, 0,
            ),
        ),
        (
            # k=1024 at B_K=256 is FOUR K-tiles, so splitK=8 would leave four
            # splits owning nothing. k=2048 would NOT test this: it is exactly
            # eight tiles, so splitK=8 is legal there.
            "splitK past the K-tile count",
            lambda: _run(*_build(256, 256, 1024, 1)[:4], 256, 256, 1, 0, 8, 1,
                         dtypes.fp32),
        ),
        (
            "mClusterWg=2 with a single M-tile",
            lambda: _run(
                *_build(16, 256, 4096, 1)[:4], 16, 256, 1, 1, 2, dtypes.fp32
            ),
        ),
    ):
        try:
            fn()
            torch.cuda.synchronize()
            print(f"FAIL  guard: {why} was ACCEPTED")
            guards += 1
        except Exception:
            print(f"PASS  guard: {why} rejected as expected")

    total = failures + guards
    print(f"\n{'ALL PASSED' if total == 0 else f'{total} FAILURES'}")
    return total


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
