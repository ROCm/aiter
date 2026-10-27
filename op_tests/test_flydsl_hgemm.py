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
    (8, 1536, 7168, 16, 32, 128, 8, 4, 1, 2, 1, 0, False),
    (8, 3584, 7168, 16, 16, 256, 4, 1, 1, 1, 2, 0, False),
    (4, 4480, 7168, 16, 128, 128, 3, 7, 1, 2, 2, 0, False),
    (256, 8192, 7168, 128, 128, 128, 2, 2, 2, 4, 1, 0, True),
]

# Rows whose partial sums cross waves or workgroups, also run with bias and with
# fp32 output: (m, n, k, policy, split_k, has_bias, out_dtype).
_PARTIAL_SUM_VARIANTS = [
    (8, 1536, 7168, "ft", 4, True, dtypes.bf16),
    (8, 1536, 7168, "ft", 4, False, dtypes.fp32),
    (8, 3584, 7168, "ft", 1, True, dtypes.bf16),
    (4, 4480, 7168, "ft", 7, True, dtypes.bf16),
    (256, 8192, 7168, "hti", 2, True, dtypes.bf16),
    (256, 8192, 7168, "hti", 2, False, dtypes.fp32),
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


def test_hgemm_partial_sums(m, n, k, policy, split_k, has_bias, out_dtype):
    """Split-K and slice-K partial sums must be added in fp32, deterministically.

    Small-integer inputs keep every fp32 partial sum exact in any order, so the
    output must be the exact product rounded once, bit for bit. Repeated runs on
    Gaussian inputs must be bitwise identical.
    """
    tile = _tile_for(m, n, k, policy, split_k)
    assert tile is not None, f"no tuned tile for {(m, n, k, policy, split_k)}"
    gen = torch.Generator(device="cuda").manual_seed(SEED)

    def ints(shape, bound):
        return torch.randint(-bound, bound + 1, shape, generator=gen, device="cuda").to(
            dtypes.bf16
        )

    kwargs = {key: v for key, v in tile.items() if key != "use_half_tile_interleaved"}

    def run(a, w, bias):
        out = torch.randn((m, n), dtype=out_dtype, device="cuda")
        return flydsl_hgemm(
            a, w, out=out, bias=bias, policy=policy, out_dtype=out_dtype, **kwargs
        )

    a, w = ints((m, k), 16), ints((n, k), 16)
    bias = ints((n,), 256) if has_bias else None
    exact = a.double() @ w.double().t()
    if has_bias:
        exact += bias.double()
    assert exact.abs().max() < 2**24
    y = run(a, w, bias)
    bad = int((y != exact.to(out_dtype)).sum())
    tag = f"{m}x{n}x{k} {policy} ks{split_k} kw{tile['k_waves']} bias{int(has_bias)} {out_dtype}"
    assert bad == 0, f"{tag}: {bad}/{y.numel()} outputs differ from the exact product"

    a = torch.randn((m, k), generator=gen, device="cuda").to(dtypes.bf16)
    w = torch.randn((n, k), generator=gen, device="cuda").to(dtypes.bf16)
    bias = (
        torch.randn((n,), generator=gen, device="cuda").to(dtypes.bf16)
        if has_bias
        else None
    )
    first = run(a, w, bias).clone()
    for _ in range(9):
        assert torch.equal(run(a, w, bias), first), f"{tag}: run-to-run mismatch"
    aiter.logger.info("partial sums exact and deterministic: %s", tag)


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
        default=_default_mnk(),
        help="""Shape of mnk.
        e.g.:   -s 128,4096,4096
                --mnk 32,384,7168""",
    )
    args = parser.parse_args()
    wanted = set(args.mnk)

    for dtype in args.dtype:
        rows = []
        for row in _TUNED:
            m, n, k, policy, split_k, _tile = _unpack(row)
            if (m, n, k) not in wanted:
                continue
            rows.append(test_hgemm(m, n, k, dtype, policy, split_k))
        summarize(f"flydsl_hgemm {dtype}", rows)

    for row in _TUNED:
        m, n, k, policy, split_k, tile = _unpack(row)
        if (m, n, k) in wanted and (split_k > 1 or tile["k_waves"] > 1):
            test_hgemm_partial_sums(m, n, k, policy, split_k, False, dtypes.bf16)
    for m, n, k, policy, split_k, has_bias, out_dtype in _PARTIAL_SUM_VARIANTS:
        if (m, n, k) in wanted:
            test_hgemm_partial_sums(m, n, k, policy, split_k, has_bias, out_dtype)


if __name__ == "__main__":
    main()
