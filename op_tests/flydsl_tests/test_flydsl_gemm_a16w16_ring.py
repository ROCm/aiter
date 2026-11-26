# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Correctness and perf of the gfx950 small-M BF16 ring GEMM (aiter.ops.flydsl.gemm_a16w16_ring).

pytest op_tests/flydsl_tests/test_flydsl_gemm_a16w16_ring.py
python op_tests/flydsl_tests/test_flydsl_gemm_a16w16_ring.py
"""

import argparse
import itertools

import pandas as pd
import pytest
import torch

pytest.importorskip("flydsl")

import aiter
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.ops.flydsl.gemm_a16w16_ring import (
    gemm_a16w16_ring,
    ring_supported,
    shuffle_weight_ring,
    unshuffle_weight_ring,
)
from aiter.test_common import benchmark, checkAllclose, run_perftest
from aiter.tuned_gemm import tgemm

# (N, K): a 200064-vocab lm_head shard at TP4 (the production shape), and two other legal shapes.
_SHAPES = ((50016, 6144), (32000, 4096), (24576, 7168))
_ROWS = (1, 7, 8, 16, 17, 32, 48, 64, 65, 96, 97, 128)

_SKIP = pytest.mark.skipif(
    get_gfx_runtime() != "gfx950", reason="gfx950 and FlyDSL are required"
)


def run_torch(x, w):
    return (x.float() @ w.float().T).to(dtypes.bf16)


def _inputs(m, n, k, seed=0):
    g = torch.Generator(device="cuda")
    g.manual_seed(seed + 7919 * m)
    x = (torch.randn(m, k, device="cuda", generator=g) * 0.05).to(dtypes.bf16)
    w = (torch.randn(n, k, device="cuda", generator=g) * 0.02).to(dtypes.bf16)
    return x, w


@_SKIP
@pytest.mark.parametrize("n,k", _SHAPES)
@pytest.mark.parametrize("m", _ROWS)
def test_ring_matches_torch(m, n, k):
    x, w = _inputs(m, n, k)
    ref = run_torch(x, w)
    ws = shuffle_weight_ring(w)
    out = gemm_a16w16_ring(x, ws)
    # fp32 accumulation in a different K order than torch: allow a bf16 ulp of the output.
    torch.testing.assert_close(out, ref, atol=2e-3, rtol=2e-2)
    # The layout and the output buffer do not change a bit of the result.
    out_rm = torch.empty_like(out)
    gemm_a16w16_ring(x, w, out=out_rm, shuffled=False)
    assert torch.equal(out, out_rm)


@_SKIP
def test_ring_rows_tier_invariant():
    """A row's result does not depend on how many rows ride with it (one M half vs two, 1..128 rows)."""
    n, k = _SHAPES[0]
    x, w = _inputs(128, n, k)
    ws = shuffle_weight_ring(w)
    full = gemm_a16w16_ring(x, ws)
    for m in (1, 7, 16, 33, 64, 65, 100):
        assert torch.equal(gemm_a16w16_ring(x[:m].contiguous(), ws), full[:m])


@_SKIP
def test_ring_graph_replay():
    n, k = _SHAPES[0]
    x, w = _inputs(32, n, k)
    ws = shuffle_weight_ring(w)
    eager = gemm_a16w16_ring(x, ws)
    out = torch.empty_like(eager)
    gemm_a16w16_ring(x, ws, out=out)  # compile outside the capture
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        gemm_a16w16_ring(x, ws, out=out)
    out.zero_()
    g.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, eager)


def test_ring_rejects_weights_past_32bit_offsets():
    """N=K=65536 passes the LDS/ring checks but its 8 GiB weight overflows the 32-bit DMA offset."""
    from aiter.ops.flydsl.kernels.gemm_a16w16_ring_gfx950 import fits

    assert fits(50016, 6144, 1, 1, 4)
    assert not fits(65536, 65536, 1, 1, 4)


@_SKIP
def test_ring_rejects_non_2d_weight():
    """A weight with an extra dimension (e.g. a trailing size-0 axis) must not reach the raw-pointer launch:
    w.shape[1] == k alone does not catch it."""
    x, w = _inputs(8, *_SHAPES[0])
    with pytest.raises(ValueError, match="contiguous 16-byte-aligned"):
        gemm_a16w16_ring(x, w.unsqueeze(-1))


@_SKIP
def test_ring_rejects_cpu_and_foreign_device_tensors():
    x, w = _inputs(8, *_SHAPES[0])
    with pytest.raises(ValueError):
        gemm_a16w16_ring(x.cpu(), w)
    with pytest.raises(ValueError):
        gemm_a16w16_ring(x, w.cpu())
    with pytest.raises(ValueError):
        gemm_a16w16_ring(x, w, out=torch.empty(8, _SHAPES[0][0], dtype=dtypes.bf16))


@_SKIP
@pytest.mark.parametrize("m", (7, 64, 65, 128))
def test_ring_reads_aiter_preshuffled_weight_in_place(m):
    """AITER's shuffle_weight(w, (16, 16)) layout needs no second copy: bit-identical to the row-major result."""
    from aiter.ops.shuffle import shuffle_weight

    n, k = _SHAPES[0]
    x, w = _inputs(m, n, k)
    ws = shuffle_weight(w, layout=(16, 16))
    out = gemm_a16w16_ring(x, ws, shuffled="aiter")
    assert torch.equal(out, gemm_a16w16_ring(x, w, shuffled=False))
    assert torch.equal(out, gemm_a16w16_ring(x, shuffle_weight_ring(w)))


@_SKIP
def test_ring_tilings_are_bit_identical_per_mt():
    from aiter.ops.flydsl.gemm_a16w16_ring import ring_tilings

    n, k = _SHAPES[0]
    x, w = _inputs(48, n, k)
    ref = gemm_a16w16_ring(x, w, shuffled=False)
    tilings = [t for t in ring_tilings(48) if ring_supported(48, n, k, tiling=t)]
    assert len(tilings) > 3
    for t in tilings:
        assert torch.equal(gemm_a16w16_ring(x, w, shuffled=False, tiling=t), ref), t
    with pytest.raises(ValueError):
        gemm_a16w16_ring(x, w, shuffled=False, tiling=(1, 1, 4))  # 16 rows < M=48


@_SKIP
@pytest.mark.parametrize("preshuffle", (False, True))
@pytest.mark.parametrize("m", (8, 64, 65, 128))
def test_flydsl_hgemm_ring_policy_and_tuned_dispatch(
    m, preshuffle, tmp_path, monkeypatch
):
    """policy='rg' through flydsl_hgemm, and a tuned CSV row naming the kernel through tgemm.mm."""
    import aiter.tuned_gemm as tuned
    from aiter.jit.utils.chip_info import get_cu_num
    from aiter.ops.flydsl.gemm_a16w16_ring import ring_config, ring_tilings
    from aiter.ops.flydsl.gemm_kernels import flydsl_hgemm, flydsl_hgemm_kernel_name
    from aiter.ops.shuffle import shuffle_weight

    n, k = _SHAPES[0]
    x, w = _inputs(m, n, k)
    ref = gemm_a16w16_ring(x, w, shuffled=False)
    tiling = next(t for t in ring_tilings(m) if ring_supported(m, n, k, tiling=t))
    config = ring_config(*tiling)
    config.pop("use_half_tile_interleaved")
    wt = shuffle_weight(w, layout=(16, 16)) if preshuffle else w
    out = flydsl_hgemm(x, wt, bpreshuffle=preshuffle, **config)
    assert torch.equal(out, ref)
    with pytest.raises(ValueError):
        flydsl_hgemm(
            x, wt, bias=torch.zeros(n, dtype=dtypes.bf16, device="cuda"), **config
        )

    name = flydsl_hgemm_kernel_name(
        dtype=dtypes.bf16,
        out_dtype=dtypes.bf16,
        config={**config, "use_half_tile_interleaved": False},
        has_bias=False,
    )
    key = (
        "gfx950",
        get_cu_num(),
        m,
        n,
        k,
        False,
        str(dtypes.bf16),
        str(dtypes.bf16),
        False,
        preshuffle,
    )
    row = {"libtype": "flydsl", "solidx": 0, "splitK": 1, "kernelName": name}
    monkeypatch.setattr(tuned, "get_GEMM_A16W16_config_", lambda: {key: row})
    tuned.get_GEMM_A16W16_config.cache_clear()
    calls = []
    orig = tuned.solMap["flydsl"]
    monkeypatch.setitem(
        tuned.solMap, "flydsl", lambda *a, **kw: calls.append(1) or orig(*a, **kw)
    )
    try:
        got = tgemm.mm(x, wt)
    finally:
        tuned.get_GEMM_A16W16_config.cache_clear()
    assert calls, "tgemm.mm did not reach the tuned FlyDSL ring row"
    assert torch.equal(got, ref)


def test_ring_shuffle_roundtrip():
    w = torch.randn(512 * 16, 128, dtype=dtypes.bf16)
    assert torch.equal(unshuffle_weight_ring(shuffle_weight_ring(w)), w)


@benchmark()
def bench_ring(m, n, k):
    x, w = _inputs(m, n, k)
    ws = shuffle_weight_ring(w)
    ref = run_torch(x, w)
    out_ring, us_ring = run_perftest(
        gemm_a16w16_ring, x, ws, num_iters=50, num_warmup=10
    )
    out_tg, us_tg = run_perftest(tgemm.mm, x, w, None, num_iters=50, num_warmup=10)
    flops = 2.0 * m * n * k
    traffic = (m * k + n * k + m * n) * 2
    ret = {}
    for name, us, o in (("ring", us_ring, out_ring), ("tgemm", us_tg, out_tg)):
        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = flops / us / 1e6
        ret[f"{name} TB/s"] = traffic / us / 1e6
        ret[f"{name} err"] = checkAllclose(
            ref.float(), o.float(), atol=2e-3, rtol=2e-2, msg=name
        )
        # checkAllclose reports the mismatch fraction, it does not raise: a wrong candidate must fail the run
        assert ret[f"{name} err"] == 0, f"{name} mismatches the reference at M={m}"
    return ret


def main():
    parser = argparse.ArgumentParser(description="BF16 small-M ring GEMM")
    parser.add_argument(
        "-m", "--rows", type=int, nargs="*", default=[8, 16, 32, 64, 96, 128]
    )
    parser.add_argument(
        "-s",
        "--nk",
        type=int,
        nargs=2,
        action="append",
        metavar=("N", "K"),
        default=None,
    )
    args = parser.parse_args()
    if get_gfx_runtime() != "gfx950":
        aiter.logger.info("ring GEMM needs gfx950; skipped")
        return
    shapes = [tuple(s) for s in args.nk] if args.nk else [_SHAPES[0]]
    rows = [
        bench_ring(m, n, k)
        for (n, k), m in itertools.product(shapes, args.rows)
        if ring_supported(m, n, k)
    ]
    if not rows:
        aiter.logger.info("no requested shape is supported by the ring GEMM")
        return
    df = pd.DataFrame(rows)
    assert (df["ring err"] == 0).all(), "ring GEMM mismatched the fp32 reference"
    aiter.logger.info("gemm_a16w16_ring summary:\n%s", df.to_markdown(index=False))


if __name__ == "__main__":
    main()
