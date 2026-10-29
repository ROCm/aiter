# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Correctness + perf for FlyDSL A16W16 HGEMM (gfx950).

The model path (``tuned_gemm.flydsl_gemm``) is BF16 x BF16:

    a    : [M, K]  contiguous
    w    : [N, K]  contiguous
    out  : [M, N]  preallocated
    y    = flydsl_hgemm(a, w, out=out, ...)   # kernel sees w.t() as NT

Run:
    python op_tests/test_flydsl_hgemm.py
    python op_tests/test_flydsl_hgemm.py -s 128,4096,4096
    python op_tests/test_flydsl_hgemm.py -s 32,384,7168
"""

import argparse

import pandas as pd
import torch

import aiter
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl import flydsl_hgemm
from aiter.ops.flydsl.gemm_kernels import (
    flydsl_hgemm_supports_shape,
    get_flydsl_hgemm_kernel_params,
)
from aiter.ops.flydsl.kernels.gemm_a16w16_gfx950 import gemm_a16w16
from aiter.test_common import (
    benchmark,
    checkAllclose,
    run_perftest,
)

torch.set_default_device("cuda")

# Public wrapper currently supports gfx950 only. Positive allow-list: an unknown
# new card must not silently run an unbuilt kernel.
SUPPORTED_GFX = ["gfx950"]
SEED = 0

# (m, n, k, block_m, block_n, block_k, stages, split_k, m_waves, n_waves, k_waves,
#  group_m, hti)
_TUNED = [
    (8, 4096, 4096, 16, 16, 256, 4, 1, 1, 1, 2, 4, False),
    (16, 4096, 4096, 16, 16, 256, 4, 1, 1, 1, 2, 4, False),
    (32, 4096, 4096, 16, 32, 256, 4, 1, 1, 2, 2, 4, False),
    (64, 4096, 4096, 32, 32, 128, 6, 1, 2, 2, 1, 4, False),
    (128, 4096, 4096, 32, 64, 128, 4, 1, 1, 4, 1, 0, False),
    (256, 4096, 4096, 64, 64, 128, 4, 1, 4, 2, 1, 0, False),
    (512, 4096, 4096, 64, 128, 64, 5, 1, 2, 4, 1, 4, False),
    (1024, 4096, 4096, 128, 128, 64, 4, 1, 2, 4, 1, 4, False),
    (2048, 4096, 4096, 256, 128, 64, 3, 1, 4, 2, 2, 0, False),
    (1024, 1024, 1024, 64, 64, 64, 4, 1, 1, 4, 1, 4, False),
    (2048, 2048, 2048, 128, 128, 64, 4, 1, 2, 4, 1, 0, False),
    (4096, 4096, 4096, 256, 256, 64, 2, 1, 2, 4, 1, 4, True),
    (4096, 4096, 8192, 256, 256, 64, 2, 1, 2, 4, 1, 4, True),
    (8192, 8192, 8192, 256, 256, 64, 2, 1, 2, 4, 1, 4, True),
    (8, 7168, 2048, 16, 16, 64, 8, 1, 1, 1, 1, 4, False),
    (32, 384, 7168, 16, 16, 256, 4, 1, 1, 1, 2, 0, False),
    (32, 14336, 4096, 32, 64, 128, 5, 1, 2, 2, 1, 4, False),
    (16, 28672, 4096, 16, 64, 256, 2, 1, 1, 2, 2, 4, False),
    (4096, 256, 4096, 64, 64, 128, 4, 1, 4, 2, 1, 4, False),
    (1, 5120, 2880, 16, 64, 64, 7, 3, 1, 2, 1, 0, False),
    (2, 5120, 2880, 16, 64, 64, 7, 3, 1, 2, 1, 0, False),
    (4, 5120, 2880, 16, 64, 64, 7, 3, 1, 2, 1, 0, False),
    (8, 5120, 2880, 16, 64, 64, 8, 3, 1, 2, 1, 0, False),
    (16, 5120, 2880, 16, 64, 64, 8, 3, 1, 2, 1, 0, False),
    (32, 5120, 2880, 16, 32, 64, 8, 1, 1, 2, 1, 0, False),
    (48, 5120, 2880, 16, 64, 64, 8, 1, 1, 2, 1, 0, False),
    (32, 384, 7168, 32, 32, 64, 8, 8, 2, 1, 1, 0, False),
    (32, 384, 16384, 32, 32, 64, 8, 8, 1, 2, 1, 0, False),
    (800, 384, 7168, 64, 96, 64, 4, 4, 2, 2, 1, 0, False),
    (32, 7168, 2048, 32, 32, 64, 8, 1, 2, 1, 1, 0, False),
    (32, 2880, 2048, 32, 32, 64, 8, 2, 1, 2, 1, 0, False),
    # M * N > 2^31
    (360448, 6144, 1280, 256, 256, 64, 2, 1, 2, 4, 1, 0, True),
]
_CONCURRENT_COPY_MNK = [(8192, 8192, 128)]


def _policy(hti):
    return "hti" if hti else "ft"


def _unpack(row):
    (
        m,
        n,
        k,
        block_m,
        block_n,
        block_k,
        stages,
        split_k,
        m_waves,
        n_waves,
        k_waves,
        group_m,
        hti,
    ) = row
    tile = {
        "block_m": block_m,
        "block_n": block_n,
        "block_k": block_k,
        "stages": stages,
        "split_k": split_k,
        "m_waves": m_waves,
        "n_waves": n_waves,
        "k_waves": k_waves,
        "group_m": group_m,
        "use_half_tile_interleaved": hti,
    }
    return m, n, k, _policy(hti), split_k, tile


def _default_mnk():
    seen = []
    got = set()
    for row in _TUNED:
        m, n, k, *_rest = _unpack(row)
        key = (m, n, k)
        if key not in got:
            got.add(key)
            seen.append(key)
    return seen


def _tile_for(m, n, k, policy, split_k):
    for row in _TUNED:
        rm, rn, rk, rpolicy, rsplit_k, tile = _unpack(row)
        if (rm, rn, rk, rpolicy, rsplit_k) == (m, n, k, policy, split_k):
            return tile
    return None


def run_torch(a, w, dtype):
    # Reference only: fp32 math, cast back. Not timed, not in the table.
    return torch.mm(a.to(dtypes.fp32), w.t().to(dtypes.fp32)).to(dtype)


def _atol_rtol(k, split_k, k_waves, dtype):
    k_scale = (k / 8192) ** 0.5 * split_k * k_waves
    if dtype is dtypes.bf16:
        return 2e-1 * k_scale, 2e-1
    return 5e-2 * k_scale, 5e-2


@benchmark()
def test_hgemm(m, n, k, dtype, policy, split_k):
    tile = _tile_for(m, n, k, policy, split_k)
    assert tile is not None, f"no tuned tile for {(m, n, k, policy, split_k)}"

    torch.manual_seed(SEED)
    a = torch.empty((m, k), dtype=dtype, device="cuda").uniform_(-1, 1)
    w = torch.empty((n, k), dtype=dtype, device="cuda").uniform_(-1, 1)
    # Dirty preallocated C, as the model passes out= a buffer it owns.
    out = torch.randn((m, n), dtype=dtype, device="cuda")
    ref = run_torch(a, w, dtype)

    def _run_flydsl():
        return flydsl_hgemm(
            a,
            w,
            out=out,
            block_m=tile["block_m"],
            block_n=tile["block_n"],
            block_k=tile["block_k"],
            stages=tile["stages"],
            split_k=tile["split_k"],
            m_waves=tile["m_waves"],
            n_waves=tile["n_waves"],
            k_waves=tile["k_waves"],
            group_m=tile["group_m"],
            policy=policy,
        )

    out_mm = torch.empty_like(out)

    def _run_torch_mm():
        torch.mm(a, w.t(), out=out_mm)
        return out_mm

    candidates = {"flydsl": _run_flydsl, "torch_mm": _run_torch_mm}

    flops = 2 * m * n * k
    nbytes = (
        m * k * a.element_size() + n * k * w.element_size() + m * n * out.element_size()
    )
    atol, rtol = _atol_rtol(k, split_k, tile["k_waves"], dtype)
    ret = {
        "gfx": get_gfx(),
        "tile": (
            f"{tile['block_m']}x{tile['block_n']}x{tile['block_k']}"
            f"s{tile['stages']}"
        ),
        "waves": f"{tile['m_waves']}x{tile['n_waves']}x{tile['k_waves']}",
    }
    for name, fn in candidates.items():
        # Not `out`: `_run_flydsl` closes over the preallocated buffer.
        y, us = run_perftest(fn)
        err = checkAllclose(
            ref.to(dtypes.fp32),
            y.to(dtypes.fp32),
            rtol=rtol,
            atol=atol,
            msg=f"{name}: flydsl hgemm",
        )
        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = flops / us / 1e6
        ret[f"{name} TB/s"] = nbytes / us / 1e6
        ret[f"{name} err"] = err
    return ret


def check_shape_limits():
    params = get_flydsl_hgemm_kernel_params(
        "flydsl_hgemm_abf16_wbf16_bf16_t256x256x64x2_ks1_w2x4x1_bias1_ktail0_gm0_pht_gfx950"
    )
    # A must stay below the 4 GiB flat-buffer limit: 1677721 * 1280 * 2 < 2^32.
    assert flydsl_hgemm_supports_shape(1677721, 6144, 1280, params)
    assert not flydsl_hgemm_supports_shape(1677722, 6144, 1280, params)


def check_input_span():
    # A strided A spans 4 GiB while M * K * 2 stays below it; no memory needed.
    a = torch.empty_strided((600000, 1280), (4096, 1), dtype=dtypes.bf16, device="meta")
    b = torch.empty((256, 1280), dtype=dtypes.bf16, device="meta").t()
    try:
        gemm_a16w16(a, b)
    except ValueError as e:
        assert "A must span less than 4 GiB" in str(e), e
    else:
        raise AssertionError("expected ValueError for a 4 GiB A span")


def check_large_a():
    # A just under 4 GiB: its byte offsets pass 2^31. Checks the first and last rows.
    m, n, k = 1677721, 256, 1280
    torch.manual_seed(SEED)
    a = torch.empty((m, k), dtype=dtypes.bf16).uniform_(-1, 1)
    w = torch.empty((n, k), dtype=dtypes.bf16).uniform_(-1, 1)
    y = flydsl_hgemm(
        a,
        w,
        block_m=256,
        block_n=256,
        block_k=64,
        stages=2,
        split_k=1,
        m_waves=2,
        n_waves=4,
        k_waves=1,
        group_m=0,
        policy="hti",
    )
    rows = torch.cat([torch.arange(1024), torch.arange(m - 1024, m)])
    atol, rtol = _atol_rtol(k, 1, 1, dtypes.bf16)
    err = checkAllclose(
        run_torch(a[rows], w, dtypes.bf16).to(dtypes.fp32),
        y[rows].to(dtypes.fp32),
        rtol=rtol,
        atol=atol,
        msg=f"{m}x{n}x{k} first and last rows",
    )
    assert err == 0


def check_strided_out():
    # A row-strided `out` (a view into a larger buffer) is written in place.
    for row in _TUNED:
        m, n, k, policy, split_k, tile = _unpack(row)
        if (m, n, k, policy, split_k) not in (
            (32, 14336, 4096, "ft", 1),
            (4096, 4096, 4096, "hti", 1),
            (16, 5120, 2880, "ft", 3),
        ):
            continue
        torch.manual_seed(SEED)
        a = torch.empty((m, k), dtype=dtypes.bf16).uniform_(-1, 1)
        w = torch.empty((n, k), dtype=dtypes.bf16).uniform_(-1, 1)
        parent = torch.full(
            (m + tile["block_m"], n + tile["block_n"]), 4096.0, dtype=dtypes.bf16
        )
        out = parent[:m, :n]
        y = flydsl_hgemm(
            a,
            w,
            out=out,
            block_m=tile["block_m"],
            block_n=tile["block_n"],
            block_k=tile["block_k"],
            stages=tile["stages"],
            split_k=tile["split_k"],
            m_waves=tile["m_waves"],
            n_waves=tile["n_waves"],
            k_waves=tile["k_waves"],
            group_m=tile["group_m"],
            policy=policy,
        )
        assert y is out
        assert torch.all(parent[m:] == 4096) and torch.all(parent[:, n:] == 4096)
        atol, rtol = _atol_rtol(k, split_k, tile["k_waves"], dtypes.bf16)
        err = checkAllclose(
            run_torch(a, w, dtypes.bf16).to(dtypes.fp32),
            y.to(dtypes.fp32),
            rtol=rtol,
            atol=atol,
            msg=f"strided out {m}x{n}x{k} {policy} ks{split_k}",
        )
        assert err == 0


@benchmark()
def test_hgemm_concurrent_copy(m, n, k, dtype):
    # The HTI tail LDS race shows only with exactly two K tiles (no main loop) and
    # memory traffic from another stream. Without split-K the output is
    # deterministic, so every launch must match the idle-GPU output bitwise.
    tile = {
        "block_m": 256,
        "block_n": 256,
        "block_k": k // 2,
        "stages": 2,
        "split_k": 1,
        "m_waves": 2,
        "n_waves": 4,
        "k_waves": 1,
        "group_m": 0,
    }
    torch.manual_seed(SEED)
    a = torch.empty((m, k), dtype=dtype, device="cuda").uniform_(-1, 1)
    w = torch.empty((n, k), dtype=dtype, device="cuda").uniform_(-1, 1)
    golden = flydsl_hgemm(a, w, **tile, policy="hti")
    atol, rtol = _atol_rtol(k, 1, 1, dtype)
    err = checkAllclose(
        run_torch(a, w, dtype).to(dtypes.fp32),
        golden.to(dtypes.fp32),
        rtol=rtol,
        atol=atol,
        msg="flydsl: idle hgemm",
    )

    side_stream = torch.cuda.Stream()
    src = torch.empty(256 << 20, dtype=torch.uint8, device="cuda")
    dst = torch.empty_like(src)
    out = torch.empty_like(golden)
    launches = 1000
    bad_launches = torch.zeros((), dtype=torch.int64, device="cuda")
    for _ in range(launches):
        with torch.cuda.stream(side_stream):
            dst.copy_(src)
        out.fill_(float("nan"))
        flydsl_hgemm(a, w, out=out, **tile, policy="hti")
        bad_launches += (out != golden).any()
    torch.cuda.synchronize()
    bad = bad_launches.item()
    assert bad == 0, f"{bad} of {launches} launches differ from the idle-GPU output"
    return {
        "gfx": get_gfx(),
        "tile": (
            f"{tile['block_m']}x{tile['block_n']}x{tile['block_k']}"
            f"s{tile['stages']}"
        ),
        "flydsl err": err,
        "launches": launches,
        "bad launches": bad,
    }


def summarize(title, rows):
    df = pd.DataFrame(rows)
    if df.empty:
        return
    aiter.logger.info("%s summary (markdown):\n%s", title, df.to_markdown(index=False))


def main():
    if get_gfx() not in SUPPORTED_GFX:
        aiter.logger.warning("flydsl hgemm unsupported on %s; skipping", get_gfx())
        return

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="config input of test",
    )
    parser.add_argument(
        "-d",
        "--dtype",
        type=dtypes.str2Dtype,
        choices=[dtypes.d_dtypes["bf16"]],
        nargs="*",
        default="bf16,",
        metavar="{bf16}",
        help="""Data type.
        e.g.: -d bf16""",
    )
    parser.add_argument(
        "-s",
        "--mnk",
        type=dtypes.str2tuple,
        nargs="*",
        default=_default_mnk() + _CONCURRENT_COPY_MNK,
        help="""Shape of mnk.
        e.g.:   -s 128,4096,4096
                --mnk 32,384,7168""",
    )
    args = parser.parse_args()
    wanted = set(args.mnk)
    if wanted == set(parser.get_default("mnk")):
        check_shape_limits()
        check_input_span()
        check_strided_out()
        check_large_a()

    for dtype in args.dtype:
        rows = []
        for row in _TUNED:
            m, n, k, policy, split_k, _tile = _unpack(row)
            if (m, n, k) not in wanted:
                continue
            rows.append(test_hgemm(m, n, k, dtype, policy, split_k))
        summarize(f"flydsl_hgemm {dtype}", rows)
        rows = [
            test_hgemm_concurrent_copy(m, n, k, dtype)
            for m, n, k in _CONCURRENT_COPY_MNK
            if (m, n, k) in wanted
        ]
        summarize(f"flydsl_hgemm {dtype} with a concurrent copy", rows)


if __name__ == "__main__":
    main()
