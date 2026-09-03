# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Runtime correctness tests for the public BF16 small-M HGEMM operation.

Pytest collects the validity cases below (no timing). ``python3`` this file
to run a small ``@benchmark`` / ``run_perftest`` sweep with a markdown table.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import pandas as pd
import pytest
import torch

import aiter
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.test_common import benchmark, checkAllclose, run_perftest

pytest.importorskip("flydsl")

from aiter.ops.flydsl.gemm_kernels import flydsl_small_m_hgemm

ARCH = get_gfx_runtime()
SUPPORTED_ARCHS = ("gfx942", "gfx950")
ATOL = 0.125
RTOL = 0.01
SPLIT_K_ATOL = 0.5
SPLIT_K_RTOL = 0.02

pytestmark = pytest.mark.skipif(
    ARCH not in SUPPORTED_ARCHS,
    reason="BF16 small-M HGEMM requires gfx942 or gfx950",
)


@dataclass(frozen=True)
class Case:
    name: str
    m: int
    n: int
    k: int
    tile_n: int
    tile_k: int
    split_k: int = 1
    block_n_warps: int = 2
    n_tile_repeat: int = 1
    persistent_n_tiles: int = 1
    b_to_lds: bool = False
    with_bias: bool = False


CASES = [
    Case(
        "lower-boundary",
        m=1,
        n=128,
        k=32,
        tile_n=128,
        tile_k=32,
        b_to_lds=True,
    ),
    Case(
        "n-repeat",
        m=7,
        n=192,
        k=128,
        tile_n=64,
        tile_k=64,
        block_n_warps=1,
        n_tile_repeat=2,
    ),
    Case(
        "split-k",
        m=7,
        n=128,
        k=256,
        tile_n=128,
        tile_k=64,
        split_k=2,
    ),
    Case(
        "persistent-bias",
        m=16,
        n=384,
        k=256,
        tile_n=128,
        tile_k=64,
        split_k=2,
        persistent_n_tiles=2,
        b_to_lds=True,
        with_bias=True,
    ),
]
if ARCH == "gfx942":
    CASES.append(
        Case(
            "vgpr-b",
            m=8,
            n=128,
            k=64,
            tile_n=16,
            tile_k=64,
            block_n_warps=1,
            b_to_lds=False,
        )
    )


def _ab(m: int, n: int, k: int, dtype) -> tuple[torch.Tensor, torch.Tensor]:
    row = torch.arange(m, device="cuda", dtype=torch.int32)[:, None]
    col = torch.arange(n, device="cuda", dtype=torch.int32)[:, None]
    red = torch.arange(k, device="cuda", dtype=torch.int32)[None, :]
    a = (((row * 11 + red * 5) % 31) - 15).to(torch.float32).div_(16).to(dtype)
    b = (((col * 7 + red * 3) % 37) - 18).to(torch.float32).div_(16).to(dtype)
    return a, b


def _inputs(
    case: Case,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    a, b = _ab(case.m, case.n, case.k, torch.bfloat16)
    bias = None
    if case.with_bias:
        bias = (
            (((torch.arange(case.n, device="cuda", dtype=torch.int32) * 5) % 19) - 9)
            .to(torch.float32)
            .div_(16)
            .bfloat16()
        )
    return a, b, bias


def _reference(
    a: torch.Tensor, b: torch.Tensor, bias: torch.Tensor | None
) -> torch.Tensor:
    output = a.float() @ b.float().T
    if bias is not None:
        output.add_(bias.float())
    return output.bfloat16()


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
def test_small_m_hgemm_matches_fp32_oracle(case: Case) -> None:
    a, b, bias = _inputs(case)
    output = torch.full(
        (case.m, case.n),
        torch.nan,
        device="cuda",
        dtype=torch.bfloat16,
    )
    returned = flydsl_small_m_hgemm(
        a,
        b,
        out=output,
        bias=bias,
        tile_n=case.tile_n,
        tile_k=case.tile_k,
        split_k=case.split_k,
        block_n_warps=case.block_n_warps,
        n_tile_repeat=case.n_tile_repeat,
        persistent_n_tiles=case.persistent_n_tiles,
        b_to_lds=case.b_to_lds,
    )
    torch.cuda.synchronize()

    assert returned is output
    assert torch.isfinite(output).all()
    atol, rtol = (SPLIT_K_ATOL, SPLIT_K_RTOL) if case.split_k > 1 else (ATOL, RTOL)
    torch.testing.assert_close(
        output,
        _reference(a, b, bias),
        atol=atol,
        rtol=rtol,
    )


def _small_m_launch_kwargs(n: int, k: int) -> dict | None:
    """Overrides of public-API defaults so tile_n / tile_k divide N / K.

    Returns None when N is not a multiple of 16 or K is not a multiple of 32.
    """
    kwargs: dict = {}
    if n % 128 == 0:
        kwargs["b_to_lds"] = True
    elif n % 64 == 0:
        kwargs["tile_n"] = 64
        kwargs["block_n_warps"] = 1
    elif n % 16 == 0:
        kwargs["tile_n"] = 16
        kwargs["block_n_warps"] = 1
    else:
        return None
    if k % 64 != 0:
        if k % 32 != 0:
            return None
        kwargs["tile_k"] = 32
    return kwargs


@benchmark()
def test_small_m_hgemm(m, n, k, dtype):
    a, b = _ab(m, n, k, dtype)
    output = torch.full((m, n), torch.nan, device="cuda", dtype=dtype)
    kwargs = _small_m_launch_kwargs(n, k)
    assert kwargs is not None
    ref = (a.float() @ b.float().T).to(dtypes.fp32)

    candidates = {
        "flydsl": lambda: flydsl_small_m_hgemm(a, b, out=output, **kwargs),
        "torch_mm": lambda: torch.mm(a, b.T),
    }
    flops = 2 * m * n * k
    nbytes = (m * k + n * k + m * n) * a.element_size()

    ret = {"gfx": ARCH}
    for name, fn in candidates.items():
        out, us = run_perftest(fn)
        err = checkAllclose(
            ref,
            out.to(dtypes.fp32),
            rtol=RTOL,
            atol=ATOL,
            msg=f"{name}: flydsl_small_m_hgemm {m}x{n}x{k} {dtype}",
        )
        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = flops / us / 1e6 if us else 0
        ret[f"{name} TB/s"] = nbytes / us / 1e6 if us else 0
        ret[f"{name} err"] = err
    return ret


# Binding starts with test_*; pytest must not collect or time this sweep.
test_small_m_hgemm.__test__ = False


def main():
    if ARCH not in SUPPORTED_ARCHS:
        aiter.logger.warning("flydsl_small_m_hgemm unsupported on %s; skipping", ARCH)
        return

    torch.set_default_device("cuda")
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
        default=[dtypes.d_dtypes["bf16"]],
        metavar="{bf16}",
        help="""Data type.
    e.g.: -d bf16""",
    )
    parser.add_argument(
        "-s",
        "--mnk",
        type=dtypes.str2tuple,
        nargs="*",
        default=[
            (1, 128, 32),
            (7, 256, 128),
            (1, 896, 7168),
        ],
        help="""Shape of mnk. Tiny-K plus a large-K cell.
    e.g.:   -s 1,128,32
            --mnk 1,896,7168""",
    )
    args = parser.parse_args()

    for dtype in args.dtype:
        df = []
        for m, n, k in args.mnk:
            if not 1 <= m <= 16:
                aiter.logger.warning(
                    "flydsl_small_m_hgemm supports M in [1, 16]; skipping m=%s", m
                )
                continue
            if _small_m_launch_kwargs(n, k) is None:
                aiter.logger.warning(
                    "flydsl_small_m_hgemm cannot tile N=%s K=%s; skipping", n, k
                )
                continue
            df.append(test_small_m_hgemm(m, n, k, dtype))
        if df:
            df = pd.DataFrame(df)
            aiter.logger.info(
                "flydsl_small_m_hgemm summary (markdown):\n%s",
                df.to_markdown(index=False),
            )


if __name__ == "__main__":
    main()
