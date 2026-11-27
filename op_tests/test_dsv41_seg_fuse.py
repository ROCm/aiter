# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""silu_mul_mxfp8 must equal silu_and_mul followed by a group-32 MXFP8 quant
of the BF16 result, bit for bit (codes and E8M0 scales)."""

import argparse

import pytest
import torch

import aiter
from aiter import dtypes
from aiter.ops.dsv41_seg_fuse import silu_mul_mxfp8

ROWS = [
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    12,
    16,
    24,
    31,
    32,
    33,
    48,
    64,
    96,
    128,
    150,
    186,
    192,
    200,
    256,
    300,
    384,
    512,
    1024,
    2048,
    4096,
    4100,
    8192,
    12288,
    16384,
]
DIMS = [2304, 32, 512, 1536, 4096]
LIMITS = [10.0, 0.0]


def _gfx950() -> bool:
    if not torch.cuda.is_available():
        return False
    return "gfx950" in torch.cuda.get_device_properties(0).gcnArchName


pytestmark = pytest.mark.skipif(not _gfx950(), reason="silu_mul_mxfp8 is gfx950 only")


def ref_mxfp8(mid: torch.Tensor):
    """Group-32 MXFP8 quant of BF16 values (ATOM quantize_fp8 / Mega-MHC epilogue)."""
    rows, d = mid.shape
    v = mid.float().view(rows, d // 32, 32)
    amax = v.abs().amax(-1).clamp_min(1e-10)
    inv448 = torch.tensor(1.0 / 448.0, dtype=torch.float32, device=mid.device)
    bits = (amax * inv448).view(torch.int32)
    e = ((bits + 0x7FFFFF) >> 23).clamp(1, 254)
    scale = (e << 23).view(torch.float32)
    q = (v / scale.unsqueeze(-1)).to(dtypes.fp8).view(rows, d)
    return q, e.to(torch.uint8)


def reference(c: torch.Tensor, limit: float):
    mid = torch.empty((c.shape[0], c.shape[1] // 2), dtype=c.dtype, device=c.device)
    aiter.silu_and_mul(mid, c, limit)
    return ref_mxfp8(mid)


def gen(rows: int, d: int, seed: int, limit: float) -> torch.Tensor:
    g = torch.Generator(device="cuda").manual_seed(seed)
    row_scale = torch.exp(torch.randn(rows, 1, device="cuda", generator=g) * 1.5)
    c = torch.randn(rows, 2 * d, device="cuda", generator=g) * row_scale
    if rows > 4:
        lim = limit if limit > 0 else 10.0
        edge = torch.tensor([lim, -lim, lim * 1.5, -lim * 3], device="cuda")
        c[0].zero_()  # all-zero row: amax floor
        c[1, : min(64, 2 * d)] = edge.repeat(16)[: min(64, 2 * d)]  # at / beyond limit
        c[2] *= 1e-30  # tiny magnitudes
        c[3, :d] = -c[3, :d].abs() * 1e3  # silu ~ 0 on the gate half
    return c.to(torch.bfloat16)


def check(rows: int, d: int, seed: int, limit: float):
    c = gen(rows, d, seed, limit)
    rq, rs = reference(c, limit)
    q, s = silu_mul_mxfp8(c, limit)
    assert q.dtype == dtypes.fp8 and s.dtype == torch.float8_e8m0fnu
    assert q.shape == (rows, d) and s.shape == (rows, d // 32)
    nq = int((q.view(torch.uint8) != rq.view(torch.uint8)).sum())
    ns = int((s.view(torch.uint8) != rs).sum())
    return nq, ns


@pytest.mark.parametrize("limit", LIMITS)
@pytest.mark.parametrize("d", DIMS)
@pytest.mark.parametrize("rows", ROWS)
def test_silu_mul_mxfp8_bitwise(rows, d, limit):
    if d != 2304 and rows > 1024:
        pytest.skip("large row counts only at the V4.1 shared-expert width")
    for seed in (1, 2, 3):
        nq, ns = check(rows, d, seed * 1000 + rows, limit)
        assert (
            nq == 0 and ns == 0
        ), f"rows={rows} d={d} limit={limit} seed={seed}: {nq} codes, {ns} scales differ"


def test_silu_mul_mxfp8_leading_dims_and_out():
    c = gen(24, 2304, 7, 10.0)
    rq, rs = reference(c, 10.0)
    q = torch.empty((2, 12, 2304), dtype=dtypes.fp8, device="cuda")
    s = torch.empty((2, 12, 72), dtype=torch.float8_e8m0fnu, device="cuda")
    q2, s2 = silu_mul_mxfp8(c.view(2, 12, 4608), 10.0, out=q, scale=s)
    assert q2.data_ptr() == q.data_ptr() and s2.data_ptr() == s.data_ptr()
    assert torch.equal(q.view(24, 2304).view(torch.uint8), rq.view(torch.uint8))
    assert torch.equal(s.view(24, 72).view(torch.uint8), rs)


def test_silu_mul_mxfp8_rejects_bad_width():
    with pytest.raises(ValueError):
        silu_mul_mxfp8(torch.zeros(4, 96, dtype=torch.bfloat16, device="cuda"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-d", type=int, nargs="*", default=DIMS)
    ap.add_argument("-m", type=int, nargs="*", default=ROWS)
    args = ap.parse_args()
    bad = total = 0
    for limit in LIMITS:
        for d in args.d:
            for rows in args.m:
                if d != 2304 and rows > 1024:
                    continue
                for seed in (1, 2, 3):
                    nq, ns = check(rows, d, seed * 1000 + rows, limit)
                    total += 1
                    if nq or ns:
                        bad += 1
                        print(
                            f"MISMATCH rows={rows} d={d} limit={limit} seed={seed}: q {nq} s {ns}"
                        )
    print(f"silu_mul_mxfp8: {total} cases, {bad} mismatches")
