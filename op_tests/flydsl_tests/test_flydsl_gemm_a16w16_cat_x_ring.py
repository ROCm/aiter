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
    gemm_a16w16_ring_splitk,
    ring_splitk_config,
    ring_splitk_name_tiling,
    ring_splitk_tilings,
    warmup_cat_x_ring,
)
from aiter.ops.flydsl.gemm_a16w16_ring import shuffle_weight_ring
from aiter.ops.flydsl.gemm_kernels import (
    flydsl_hgemm,
    flydsl_hgemm_kernel_name,
    get_flydsl_hgemm_kernel_params,
)
from aiter.ops.shuffle import shuffle_weight
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
    # i32 byte offsets: a weight of 2 GiB or more is refused, not wrapped
    assert not cat_x_ring_supported(8, 2**20, SEG_K)


@_SKIP
def test_cat_x_ring_rejects_non_2d_weight():
    """A weight with extra dimensions (e.g. a trailing size-0 axis) must not reach the raw-pointer launch:
    w.shape[0]/[1] alone do not catch it."""
    xs, w = _inputs(8)
    ws = shuffle_weight_ring(w)
    with pytest.raises(ValueError, match="2-D"):
        gemm_a16w16_cat_x_ring(xs, ws.unsqueeze(-1))
    with pytest.raises(ValueError, match="2-D"):
        gemm_a16w16_ring_splitk(xs[0].repeat(1, NSEG), ws.unsqueeze(-1))


@_SKIP
@pytest.mark.parametrize("m_max", (8, 17))
def test_warmup_cat_x_ring_compiles_the_final_partial_tier(m_max):
    """``warmup_cat_x_ring`` must compile every tier up to and including a non-multiple-of-16 ``m_max`` (m_max=8:
    the loop over range(16, m_max+1, 16) is empty; m_max=17: the loop stops at mt=1 and never reaches mt=2),
    so a later call at that exact M does not trigger a fresh (capture-unsafe) compile.
    """
    from aiter.ops.flydsl.kernels.gemm_a16w16_cat_x_ring_gfx950 import compile_cat_ring

    xs, w = _inputs(m_max)
    ws = shuffle_weight_ring(w)
    compile_cat_ring.cache_clear()
    warmup_cat_x_ring(m_max, N, SEG_K, ws)
    misses_after_warmup = compile_cat_ring.cache_info().misses
    gemm_a16w16_cat_x_ring(xs, ws)
    assert compile_cat_ring.cache_info().misses == misses_after_warmup


@_SKIP
def test_cat_x_ring_expanded_taps_and_output_rows():
    """A tap row stride of 0 (one row broadcast over M) and a padded output are served; overlapping output rows
    (a ``[1, N]`` buffer expanded to ``[M, N]``) are refused, not written over each other.
    """
    xs, w = _inputs(8)
    ws = shuffle_weight_ring(w)
    one = [x[:1].expand(8, SEG_K) for x in xs]
    out = gemm_a16w16_cat_x_ring(one, ws)
    torch.testing.assert_close(
        out.float(), run_torch(one, w).float(), atol=0.05, rtol=1e-2
    )
    assert torch.equal(out[0], out[7])
    padded = torch.full((8, N + 64), float("nan"), dtype=dtypes.bf16, device="cuda")
    got = gemm_a16w16_cat_x_ring(xs, ws, out=padded[:, :N])
    assert got.data_ptr() == padded.data_ptr()
    assert torch.equal(got, gemm_a16w16_cat_x_ring(xs, ws))
    assert padded[:, N:].isnan().all()  # the pad columns are not written
    overlap = torch.empty(1, N, dtype=dtypes.bf16, device="cuda").expand(8, N)
    with pytest.raises(ValueError):
        gemm_a16w16_cat_x_ring(xs, ws, out=overlap)
    # a row stride whose span passes the reduce kernel's 32-bit dword count is refused before any launch
    stride = (
        2**28
    )  # 7 * 2**28 + N bf16 = 3.5 GiB of address space, never written (refused before any launch)
    big = torch.empty(7 * stride + N, dtype=dtypes.bf16, device="cuda")
    huge = torch.as_strided(big, (8, N), (stride, 1))
    with pytest.raises(ValueError, match="32-bit"):
        gemm_a16w16_cat_x_ring(xs, ws, out=huge)


@_SKIP
@pytest.mark.parametrize("m, mt", ((40, 3), (40, 4), (100, 7), (100, 8)))
def test_ring_splitk_tile_larger_than_rows(m, mt):
    """A padded-M lookup runs a row tuned for a larger M: M = 33..48 on the M = 64 tile (``mt = 4``), 65..112 on the
    M = 128 tile (``mt = 8``). Rows past M load the last row and are not stored."""
    xs, w = _inputs(m)
    x = torch.cat(xs, dim=-1)
    out = gemm_a16w16_ring_splitk(x, w, mt=mt)
    torch.testing.assert_close(
        out.float(), run_torch(xs, w).float(), atol=0.05, rtol=1e-2
    )
    # a row's result does not depend on the tile that holds it
    assert torch.equal(out, gemm_a16w16_ring_splitk(x, w))


@_SKIP
@pytest.mark.parametrize("m", (40, 100))
def test_ring_splitk_tuned_row_of_a_larger_m(m, monkeypatch):
    """``tgemm.mm`` at M reaches an injected ``rk`` row tuned at the padded M (a larger tile than M needs)."""
    import aiter.tuned_gemm as tuned
    from aiter.jit.utils.chip_info import get_cu_num
    from aiter.ops.gemm_op_common import get_padded_m

    k = NSEG * SEG_K
    padded = get_padded_m(
        m, N, k, 1
    )  # the coarse level: M = 33..48 -> 64, 65..112 -> 128
    assert (padded + 15) // 16 > (m + 15) // 16
    xs, w = _inputs(m)
    x = torch.cat(xs, dim=-1)
    name = _tuned_name(padded, ((padded + 15) // 16, 10, 4))
    key = (
        "gfx950",
        get_cu_num(),
        padded,
        N,
        k,
        False,
        str(dtypes.bf16),
        str(dtypes.bf16),
        False,
        False,
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
        got = tgemm.mm(x, w)
    finally:
        tuned.get_GEMM_A16W16_config.cache_clear()
    assert calls, "tgemm.mm did not reach the injected rk row"
    torch.testing.assert_close(
        got.float(), run_torch(xs, w).float(), atol=0.05, rtol=1e-2
    )


@_SKIP
@pytest.mark.parametrize("layout", ["row", "aiter", "ring"])
@pytest.mark.parametrize("m", (1, 17, 64, 128))
def test_ring_splitk_plain_matches_torch(m, layout):
    """The K-concatenated form: ``x [M, 5 * seg_k]`` read as five chunks, any of the three weight layouts."""
    xs, w = _inputs(m)
    x = torch.cat(xs, dim=-1)
    ref = run_torch(xs, w).float()
    wl, shuffled = {
        "row": (w, False),
        "aiter": (shuffle_weight(w, layout=(16, 16)), "aiter"),
        "ring": (shuffle_weight_ring(w), True),
    }[layout]
    out = gemm_a16w16_ring_splitk(x, wl, shuffled=shuffled)
    torch.testing.assert_close(out.float(), ref, atol=0.05, rtol=1e-2)
    # the same split order as the five-pointer op, so the two are bit-identical
    assert torch.equal(out, gemm_a16w16_cat_x_ring(xs, shuffle_weight_ring(w), ks=10))


def _tuned_name(m, mt_ks_r):
    return flydsl_hgemm_kernel_name(
        dtype=dtypes.bf16,
        out_dtype=dtypes.bf16,
        config=ring_splitk_config(*mt_ks_r),
        has_bias=False,
    )


@_SKIP
def test_ring_splitk_name_round_trip():
    tilings = ring_splitk_tilings(32, NSEG * SEG_K, N)
    assert (2, 10, 4) in tilings and all(t[0] == 2 for t in tilings)
    for t in tilings:
        params = get_flydsl_hgemm_kernel_params(_tuned_name(32, t))
        assert params is not None and params["policy"] == "rk"
        assert ring_splitk_name_tiling(params) == t
    # a bias, a non-bf16 output or a stray field is not a split-K ring name
    bad = _tuned_name(32, (2, 10, 4)).replace("_gm0_", "_gm1_")
    assert get_flydsl_hgemm_kernel_params(bad) is None


@_SKIP
@pytest.mark.parametrize("bpreshuffle", [False, True])
def test_ring_splitk_through_flydsl_hgemm(bpreshuffle):
    """The tuned-row path: ``flydsl_hgemm(policy="rk")`` on the plain weight or AITER's preshuffled one."""
    xs, w = _inputs(32)
    x = torch.cat(xs, dim=-1)
    cfg = ring_splitk_config(2, 10, 4)
    wl = shuffle_weight(w, layout=(16, 16)) if bpreshuffle else w
    out = flydsl_hgemm(
        x,
        wl,
        block_m=cfg["block_m"],
        block_n=cfg["block_n"],
        block_k=cfg["block_k"],
        split_k=cfg["split_k"],
        m_waves=cfg["m_waves"],
        n_waves=cfg["n_waves"],
        k_waves=cfg["k_waves"],
        stages=cfg["stages"],
        policy="rk",
        bpreshuffle=bpreshuffle,
    )
    torch.testing.assert_close(
        out.float(), run_torch(xs, w).float(), atol=0.05, rtol=1e-2
    )


@_SKIP
def test_ring_splitk_plain_graph_replay():
    """Through the workspace cache of ``flydsl_hgemm``: capture, then replay."""
    xs, w = _inputs(16)
    x = torch.cat(xs, dim=-1)
    cfg = ring_splitk_config(1, 10, 4)

    def call():
        return flydsl_hgemm(
            x,
            w,
            block_m=cfg["block_m"],
            block_n=cfg["block_n"],
            block_k=cfg["block_k"],
            split_k=cfg["split_k"],
            m_waves=1,
            n_waves=4,
            k_waves=1,
            stages=cfg["stages"],
            policy="rk",
        )

    eager = call()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = call()
    g.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, eager)


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
        # checkAllclose reports the mismatch fraction, it does not raise: a wrong candidate must fail the run
        assert ret[f"{name} err"] == 0, f"{name} mismatches the reference at M={m}"
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
