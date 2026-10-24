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
import functools

import pandas as pd
import torch

import aiter
from aiter import dtypes
from aiter.bench import BenchResult, MeasureConfig, run_bench
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
# checkAllclose default: fraction of mismatching elements that still passes.
TOL_ERR_RATIO = 0.05
# Graph replay is the PR-verdict number in aiter/bench and complements
# run_perftest's device-time mean. Warm cache matches graph-based serving.
BENCH_MEASURE = MeasureConfig(mode="graph", cache="warm")

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


def _atol_rtol(k, split_k, k_waves, dtype):
    k_scale = (k / 8192) ** 0.5 * split_k * k_waves
    if dtype is dtypes.bf16:
        return 2e-1 * k_scale, 2e-1
    return 5e-2 * k_scale, 5e-2


def run_torch_mm(a, b, out, *_tile, **_launch):
    """``torch.mm`` with ``flydsl_hgemm``'s signature; tile/launch args are ignored."""
    torch.mm(a, b.t(), out=out)
    return out


def run_perftest_bench(fn, *args, name=None, flops=None, nbytes=None, **kwargs):
    """``run_perftest`` with the ``run_bench`` contract: ``(data, BenchResult)``."""
    data, us = run_perftest(fn, *args, **kwargs)
    result = BenchResult(
        name=name or getattr(fn, "__name__", None),
        backend="run_perftest",
        metric="mean",
        us=us,
        mean_us=us,
        flops=flops,
        nbytes=nbytes,
        tflops=flops / us / 1e6 if flops else None,
        gb_s=nbytes / us / 1e3 if nbytes else None,
    )
    return data, result


def run_candidates(candidates, _make_inputs, runners, *, reference, check):
    """Benchmark and verify every candidate with every runner on fresh inputs.

    Each runner follows the ``run_bench`` contract:
    ``runner(fn, *inputs, name=name)`` returns ``(data, BenchResult)``, where
    ``data`` is the output of an untimed call on ``inputs``. ``data`` is
    verified with ``check(name, ref, data) -> (passed, err)`` against
    ``ref = reference(*inputs)``, computed once from the first generated
    inputs, and the verdict is recorded on the ``BenchResult``.

    Returns ``(columns, results)``: table columns named
    ``"{name} {runner_name} {column}"``, and ``{(name, runner_name): BenchResult}``.
    """
    ref = None
    columns = {}
    results = {}
    for name, fn in candidates.items():
        for runner_name, runner in runners.items():
            inputs = _make_inputs()
            if ref is None:
                ref = reference(*inputs)
            data, result = runner(fn, *inputs, name=name)
            passed, result.err = check(name, ref, data)
            result.verified = "pass" if passed else "fail"
            results[(name, runner_name)] = result
            prefix = f"{name} {runner_name}"
            columns[f"{prefix} us"] = result.us
            columns[f"{prefix} TFLOPS"] = result.tflops
            columns[f"{prefix} TB/s"] = (
                result.gb_s / 1e3 if result.gb_s is not None else None
            )
            columns[f"{prefix} err"] = result.err
    return columns, results


@benchmark()
def test_hgemm(m, n, k, dtype, policy, split_k):
    tile = _tile_for(m, n, k, policy, split_k)
    assert tile is not None, f"no tuned tile for {(m, n, k, policy, split_k)}"

    def _make_inputs():
        """Positional arguments of ``flydsl_hgemm``, in its parameter order."""
        torch.manual_seed(SEED)
        a = torch.empty((m, k), dtype=dtype, device="cuda").uniform_(-1, 1)
        w = torch.empty((n, k), dtype=dtype, device="cuda").uniform_(-1, 1)
        # Dirty preallocated C, as the model passes out= a buffer it owns.
        out = torch.randn((m, n), dtype=dtype, device="cuda")
        bias = None
        return (
            a,
            w,
            out,
            bias,
            tile["block_m"],
            tile["block_n"],
            tile["block_k"],
            tile["stages"],
            tile["split_k"],
            tile["m_waves"],
            tile["n_waves"],
            tile["k_waves"],
            tile["group_m"],
            policy,
        )

    candidates = {"flydsl": flydsl_hgemm, "torch_mm": run_torch_mm}

    elem_size = torch.tensor([], dtype=dtype).element_size()
    flops = 2 * m * n * k
    nbytes = (m * k + n * k + m * n) * elem_size
    atol, rtol = _atol_rtol(k, split_k, tile["k_waves"], dtype)

    def _check(name, ref, y):
        err = checkAllclose(
            ref.to(dtypes.fp32),
            y.to(dtypes.fp32),
            rtol=rtol,
            atol=atol,
            tol_err_ratio=TOL_ERR_RATIO,
            msg=f"{name}: flydsl hgemm",
        )
        return err <= TOL_ERR_RATIO, err

    runners = {
        "perftest": functools.partial(run_perftest_bench, flops=flops, nbytes=nbytes),
        "bench": functools.partial(
            run_bench, measure=BENCH_MEASURE, flops=flops, nbytes=nbytes
        ),
    }

    ret = {
        "gfx": get_gfx(),
        "tile": (
            f"{tile['block_m']}x{tile['block_n']}x{tile['block_k']}"
            f"s{tile['stages']}"
        ),
        "waves": f"{tile['m_waves']}x{tile['n_waves']}x{tile['k_waves']}",
    }
    columns, _results = run_candidates(
        candidates,
        _make_inputs,
        runners,
        reference=run_torch_mm,
        check=_check,
    )
    ret.update(columns)
    return ret


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


if __name__ == "__main__":
    main()
