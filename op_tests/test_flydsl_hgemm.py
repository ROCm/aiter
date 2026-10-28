# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Correctness + perf for FlyDSL A16W16 HGEMM (gfx942 and gfx950).

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
import itertools
from pathlib import Path

import pandas as pd
import torch

import aiter
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl import flydsl_hgemm
from aiter.test_common import (
    benchmark,
    checkAllclose,
    run_perftest,
)

torch.set_default_device("cuda")

# Positive allow-list: an unknown new card must not silently run an unbuilt kernel.
SUPPORTED_GFX = ["gfx942", "gfx950"]
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
]


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


@benchmark()
def test_hgemm_gfx942(m, n, k, dtype):
    """Tuned public dispatch and direct gfx942 kernel against an fp32 oracle."""
    from flydsl_tests.gfx942_gemm_oracle import error_ratio

    from aiter.ops.flydsl.kernels.gemm_a16w16_gfx942 import gemm_a16w16

    torch.manual_seed(SEED)
    a = (torch.randn(m, k, device="cuda") * k**-0.5).to(dtype)
    w = (torch.randn(n, k, device="cuda") * k**-0.5).to(dtype)
    ref = a.float() @ w.float().t()
    public_out = torch.randn((m, n), dtype=dtype, device="cuda")
    direct_out = torch.randn_like(public_out)

    candidates = {
        "flydsl": lambda: flydsl_hgemm(a, w, out=public_out),
        "direct": lambda: gemm_a16w16(a, w.t(), out=direct_out, layout="nt"),
    }
    flops = 2 * m * n * k
    nbytes = (
        m * k * a.element_size()
        + n * k * w.element_size()
        + m * n * public_out.element_size()
    )
    ret = {"gfx": get_gfx()}
    for name, fn in candidates.items():
        y, us = run_perftest(fn)
        err = checkAllclose(
            ref,
            y.float(),
            msg=f"{name}: gfx942 flydsl hgemm M{m} N{n} K{k}",
            catastrophic_check=True,
        )
        strict_err = error_ratio(y, ref)
        assert strict_err <= 1.0, (
            f"{name}: M{m} N{n} K{k} error is "
            f"{strict_err:.2f}x the {y.dtype} tolerance"
        )
        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = flops / us / 1e6
        ret[f"{name} TB/s"] = nbytes / us / 1e6
        ret[f"{name} err"] = err
        ret[f"{name} strict err"] = strict_err
    return ret


@benchmark()
def test_hgemm_gfx942_mainloop(pipeline, ks_begin):
    """Reusable gfx942 main loop in the fused scaled-A test kernel."""
    from flydsl_tests.gfx942_gemm_oracle import error_ratio
    from flydsl_tests.test_gemm_a16w16_gfx942_mainloop import (
        PIPELINES,
        _scaled_a_gemm,
        make_gemm_a16w16_gfx942_param,
    )

    m, n, k = 200, 128, 1024
    torch.manual_seed(SEED)
    a = (torch.randn(m, k, device="cuda") * k**-0.5).to(torch.bfloat16)
    w = (torch.randn(n, k, device="cuda") * k**-0.5).to(torch.bfloat16)
    idx = torch.arange(m, device="cuda")[:, None] + torch.arange(k, device="cuda")
    scale = torch.where(idx % 3 == 0, 2.0, 1.0)
    ref = (a.float() * scale)[:, ks_begin:] @ w.float()[:, ks_begin:].t()
    out = torch.empty(m, n, dtype=torch.bfloat16, device="cuda")
    param = make_gemm_a16w16_gfx942_param(
        block_m=64, block_n=64, direct_store=True, **PIPELINES[pipeline]
    )

    def run_mainloop():
        _scaled_a_gemm(out, a, w.t(), ks_begin, param)
        return out

    y, us = run_perftest(run_mainloop)
    err = checkAllclose(
        ref,
        y.float(),
        msg=f"{pipeline}: gfx942 mainloop ks_begin={ks_begin}",
        catastrophic_check=True,
    )
    strict_err = error_ratio(y, ref)
    assert strict_err <= 1.0, (
        f"{pipeline}: ks_begin={ks_begin} error is "
        f"{strict_err:.2f}x the bf16 tolerance"
    )
    flops = 2 * m * n * (k - ks_begin)
    nbytes = (
        m * (k - ks_begin) * a.element_size()
        + n * (k - ks_begin) * w.element_size()
        + m * n * out.element_size()
    )
    return {
        "gfx": get_gfx(),
        "M": m,
        "N": n,
        "K": k,
        "mainloop us": us,
        "mainloop TFLOPS": flops / us / 1e6,
        "mainloop TB/s": nbytes / us / 1e6,
        "mainloop err": err,
        "mainloop strict err": strict_err,
    }


def summarize(title, rows):
    df = pd.DataFrame(rows)
    if df.empty:
        return
    aiter.logger.info("%s summary (markdown):\n%s", title, df.to_markdown(index=False))


def run_gfx942_regressions():
    """Run the detailed standalone and reusable-mainloop pytest suites in CI."""
    import pytest

    test_dir = Path(__file__).resolve().parent / "flydsl_tests"
    result = pytest.main(
        [
            str(test_dir / "test_gemm_a16w16_gfx942.py"),
            str(test_dir / "test_gemm_a16w16_gfx942_mainloop.py"),
            "-q",
            "-p",
            "no:cacheprovider",
        ]
    )
    if result != pytest.ExitCode.OK:
        raise SystemExit(int(result))


def main():
    gfx = get_gfx()
    if gfx not in SUPPORTED_GFX:
        aiter.logger.warning("flydsl hgemm unsupported on %s; skipping", gfx)
        return

    if gfx == "gfx942":
        from flydsl_tests.test_gemm_a16w16_gfx942 import SHAPES

        default_mnk = [(m, n, k) for m, n, k, _regime in SHAPES]
    else:
        default_mnk = _default_mnk()

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
        default=default_mnk,
        help="""Shape of mnk.
        e.g.:   -s 128,4096,4096
                --mnk 32,384,7168""",
    )
    args = parser.parse_args()

    if gfx == "gfx942":
        rows = [
            test_hgemm_gfx942(m, n, k, dtype)
            for dtype, (m, n, k) in itertools.product(args.dtype, args.mnk)
        ]
        summarize("flydsl_hgemm gfx942", rows)

        from flydsl_tests.test_gemm_a16w16_gfx942_mainloop import PIPELINES

        mainloop_rows = [
            test_hgemm_gfx942_mainloop(pipeline, ks_begin)
            for pipeline, ks_begin in itertools.product(PIPELINES, (0, 512))
        ]
        summarize("flydsl_hgemm gfx942 reusable mainloop", mainloop_rows)
        run_gfx942_regressions()
        return

    wanted = set(args.mnk)

    for dtype in args.dtype:
        rows = []
        for row in _TUNED:
            m, n, k, policy, split_k, _tile = _unpack(row)
            if (m, n, k) not in wanted:
                continue
            rows.append(test_hgemm(m, n, k, dtype, policy, split_k))
        summarize(f"flydsl_hgemm {dtype}", rows)


if __name__ == "__main__":
    main()
