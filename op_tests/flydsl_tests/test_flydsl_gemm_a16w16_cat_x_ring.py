# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Correctness and perf of the gfx950 concat-then-GEMM ring (aiter.ops.flydsl.gemm_a16w16_cat_x_ring).

pytest op_tests/flydsl_tests/test_flydsl_gemm_a16w16_cat_x_ring.py
python op_tests/flydsl_tests/test_flydsl_gemm_a16w16_cat_x_ring.py
"""

import argparse

import pandas as pd
import pytest
import torch

pytest.importorskip("flydsl")

import aiter
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.ops.flydsl.gemm_a16w16_cat_x_ring import (
    cat_x_ring_supported,
    gemm_a16w16_cat_x_ring,
    warmup_cat_x_ring,
)
from aiter.ops.flydsl.gemm_a16w16_ring import shuffle_weight_ring
from aiter.test_common import benchmark, checkAllclose, run_perftest
from aiter.tuned_gemm import tgemm

N, SEG_K, NSEG = 1536, 6144, 5
_ROWS = (1, 7, 8, 16, 17, 32, 48, 64, 96, 128)
_SKIP = pytest.mark.skipif(
    get_gfx_runtime() != "gfx950", reason="gfx950 and FlyDSL are required"
)


def run_torch(xs, w):
    return (torch.cat(xs, dim=-1).float() @ w.float().T).to(dtypes.bf16)


def _inputs(m, row_pad=0, seed=0):
    g = torch.Generator(device="cuda")
    g.manual_seed(seed + 7919 * m)
    buf = torch.randn(NSEG, m, SEG_K + row_pad, device="cuda", generator=g).to(
        dtypes.bf16
    )
    xs = [buf[i, :, :SEG_K] for i in range(NSEG)]
    w = (torch.randn(N, NSEG * SEG_K, device="cuda", generator=g) * 0.02).to(
        dtypes.bf16
    )
    return xs, w


@_SKIP
@pytest.mark.parametrize("row_pad", [0, 64])
@pytest.mark.parametrize("m", _ROWS)
def test_cat_x_ring_matches_torch(m, row_pad):
    xs, w = _inputs(m, row_pad)
    ws = shuffle_weight_ring(w)
    out = gemm_a16w16_cat_x_ring(xs, ws)
    # fp32 accumulate in a different K order than torch: bf16 rounding of O(10) outputs.
    torch.testing.assert_close(
        out.float(), run_torch(xs, w).float(), atol=0.05, rtol=1e-2
    )
    assert torch.equal(out, gemm_a16w16_cat_x_ring(xs, ws))  # deterministic


@_SKIP
@pytest.mark.parametrize("ks", [5, 10, 20])
def test_cat_x_ring_split_choice(ks):
    xs, w = _inputs(32)
    ref = run_torch(xs, w).float()
    out = gemm_a16w16_cat_x_ring(xs, shuffle_weight_ring(w), ks=ks)
    torch.testing.assert_close(out.float(), ref, atol=0.05, rtol=1e-2)


@_SKIP
def test_cat_x_ring_graph_replay():
    xs, w = _inputs(16)
    ws = shuffle_weight_ring(w)
    eager = gemm_a16w16_cat_x_ring(xs, ws)
    out = torch.empty_like(eager)
    ws_buf = torch.empty(10, 16, N, dtype=dtypes.fp32, device="cuda")
    warmup_cat_x_ring(16, N, SEG_K, ws)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        gemm_a16w16_cat_x_ring(xs, ws, out=out, workspace=ws_buf)
    out.zero_()
    g.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, eager)


@_SKIP
def test_cat_x_ring_rejects_unsupported():
    xs, w = _inputs(8)
    ws = shuffle_weight_ring(w)
    with pytest.raises(ValueError):
        gemm_a16w16_cat_x_ring(xs[:4], ws)
    assert not cat_x_ring_supported(129, N, SEG_K)


@benchmark()
def bench_cat_x(m):
    xs, w = _inputs(m)
    ws = shuffle_weight_ring(w)
    ref = run_torch(xs, w)
    traffic = (NSEG * m * SEG_K + N * NSEG * SEG_K + m * N) * 2
    flops = 2.0 * m * N * NSEG * SEG_K
    cands = {
        "cat_x_ring": lambda: gemm_a16w16_cat_x_ring(xs, ws),
        "cat+tgemm": lambda: tgemm.mm(torch.cat(xs, dim=-1), w),
    }
    ret = {}
    for name, fn in cands.items():
        out, us = run_perftest(fn, num_iters=50, num_warmup=10)
        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = flops / us / 1e6
        ret[f"{name} TB/s"] = traffic / us / 1e6
        ret[f"{name} err"] = checkAllclose(
            ref.float(), out.float(), atol=0.05, rtol=1e-2, msg=name
        )
    return ret


def main():
    parser = argparse.ArgumentParser(description="concat-then-GEMM ring")
    parser.add_argument(
        "-m", "--rows", type=int, nargs="*", default=[8, 16, 32, 64, 128]
    )
    args = parser.parse_args()
    if get_gfx_runtime() != "gfx950":
        aiter.logger.info("needs gfx950; skipped")
        return
    df = pd.DataFrame(
        [bench_cat_x(m) for m in args.rows if cat_x_ring_supported(m, N, SEG_K)]
    )
    aiter.logger.info(
        "gemm_a16w16_cat_x_ring summary:\n%s", df.to_markdown(index=False)
    )


if __name__ == "__main__":
    main()
