# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Correctness and performance test for FlyDSL split-K HGEMM regressions.

Usage:
    python op_tests/flydsl_tests/test_flydsl_splitk_hgemm.py
    python op_tests/flydsl_tests/test_flydsl_splitk_hgemm.py --case \
        splitk8_tile32_m104_n384_k7168
"""

from __future__ import annotations

import argparse
import itertools

import pandas as pd
import torch

import aiter
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl.gemm_kernels import flydsl_hgemm
from aiter.ops.shuffle import shuffle_weight
from aiter.test_common import benchmark, checkAllclose, run_perftest

torch.set_default_device("cuda")

SUPPORTED_GFX = ["gfx942", "gfx950"]

DEFAULT_ATOL = 1e-2
DEFAULT_RTOL = 1e-2
DEFAULT_TOL_ERR_RATIO = 1e-3
DEFAULT_INPUT_SEED = 20260401

SPLITK_PRECISION_CASES = {
    "splitk8_tile32_m104_n384_k7168": {
        "m": 104,
        "n": 384,
        "k": 7168,
        "tile_k": 128,
        "tile_m": 32,
        "tile_n": 64,
        "pack_n": 1,
        "split_k": 8,
        "block_m_warps": 1,
        "block_n_warps": 4,
        "b_to_lds": False,
        "b_preshuffle": False,
        "atol": DEFAULT_ATOL,
        "rtol": DEFAULT_RTOL,
        "tol_err_ratio": DEFAULT_TOL_ERR_RATIO,
        "max_abs_delta": None,
    },
    "splitk4_tile16_m1_n7168_k512": {
        "m": 1,
        "n": 7168,
        "k": 512,
        "tile_k": 128,
        "tile_m": 16,
        "tile_n": 128,
        "pack_n": 1,
        "split_k": 2,
        "block_m_warps": 1,
        "block_n_warps": 4,
        "b_to_lds": False,
        "b_preshuffle": False,
        "atol": DEFAULT_ATOL,
        "rtol": DEFAULT_RTOL,
        "tol_err_ratio": DEFAULT_TOL_ERR_RATIO,
        "max_abs_delta": None,
    },
    "splitk16_tile32_m1_n2112_k7168_warp2x2_blds": {
        "m": 1,
        "n": 2112,
        "k": 7168,
        "tile_k": 64,
        "tile_m": 32,
        "tile_n": 64,
        "pack_n": 1,
        "split_k": 16,
        "block_m_warps": 2,
        "block_n_warps": 2,
        "b_to_lds": True,
        "b_preshuffle": False,
        "atol": DEFAULT_ATOL,
        "rtol": DEFAULT_RTOL,
        "tol_err_ratio": 1e-2,
        "max_abs_delta": 32.0,
    },
    "splitk8_tile32_m1_n3072_k1536_warp2x2_blds": {
        "m": 1,
        "n": 3072,
        "k": 1536,
        "tile_k": 64,
        "tile_m": 32,
        "tile_n": 64,
        "pack_n": 1,
        "split_k": 8,
        "block_m_warps": 2,
        "block_n_warps": 2,
        "b_to_lds": True,
        "b_preshuffle": False,
        "atol": DEFAULT_ATOL,
        "rtol": DEFAULT_RTOL,
        "tol_err_ratio": 1e-2,
        "max_abs_delta": 8.0,
    },
}


def run_torch(
    a: torch.Tensor, b: torch.Tensor, dtype: torch.dtype = dtypes.bf16
) -> torch.Tensor:
    """Compute the untimed reference with fp32 matmul."""
    return torch.mm(a.to(dtypes.fp32), b.to(dtypes.fp32).t()).to(dtype)


def make_inputs(
    m: int,
    n: int,
    k: int,
    torch_dtype: torch.dtype,
    *,
    seed: int = DEFAULT_INPUT_SEED,
) -> tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator(device="cuda")
    gen.manual_seed(seed)
    a = torch.rand((m, k), generator=gen, device="cuda", dtype=torch_dtype)
    b = torch.rand((n, k), generator=gen, device="cuda", dtype=torch_dtype)
    return a, b


@benchmark()
def test_flydsl_splitk_hgemm(
    case,
    dtype,
    m,
    n,
    k,
    tile_k,
    tile_m,
    tile_n,
    pack_n,
    split_k,
    block_m_warps,
    block_n_warps,
    b_to_lds,
    b_preshuffle,
    atol,
    rtol,
    tol_err_ratio,
    max_abs_delta,
):
    a, b = make_inputs(m, n, k, dtype)
    ref = run_torch(a, b, dtype)
    b_shuf = shuffle_weight(b, layout=(16 * pack_n, 16)) if b_preshuffle else b

    candidates = {
        "flydsl": lambda: flydsl_hgemm(
            a,
            b_shuf,
            tile_k=tile_k,
            tile_m=tile_m,
            tile_n=tile_n,
            pack_n=pack_n,
            split_k=split_k,
            block_m_warps=block_m_warps,
            block_n_warps=block_n_warps,
            b_to_lds=b_to_lds,
            b_preshuffle=b_preshuffle,
        )
    }

    # HGEMM counts one multiply and one add per output/K pair. Traffic includes
    # the two inputs and output; implementation-specific workspace is additional.
    flops = 2 * m * n * k
    nbytes = (a.numel() + b.numel() + m * n) * a.element_size()

    ret = {"gfx": get_gfx()}
    ref_fp32 = ref.to(dtypes.fp32)
    for name, candidate in candidates.items():
        out, us = run_perftest(candidate)
        out_fp32 = out.to(dtypes.fp32)
        err = checkAllclose(
            ref_fp32,
            out_fp32,
            rtol=rtol,
            atol=atol,
            tol_err_ratio=tol_err_ratio,
            max_abs_delta=max_abs_delta,
            msg=f"{name}: split-K HGEMM {case}",
        )
        if err > tol_err_ratio:
            raise AssertionError(
                f"{name}: mismatch ratio {err:.4%} exceeds {tol_err_ratio:.4%}"
            )
        if max_abs_delta is not None:
            actual_max_abs_delta = (ref_fp32 - out_fp32).abs().max().item()
            if actual_max_abs_delta > max_abs_delta:
                raise AssertionError(
                    f"{name}: max abs delta {actual_max_abs_delta:.4f} exceeds "
                    f"{max_abs_delta:.4f}"
                )

        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = flops / us / 1e6
        ret[f"{name} TB/s"] = nbytes / us / 1e6
        ret[f"{name} err"] = err

    return ret


test_flydsl_splitk_hgemm.__test__ = False


def main():
    gfx = get_gfx()
    if gfx not in SUPPORTED_GFX:
        aiter.logger.warning("FlyDSL split-K HGEMM unsupported on %s; skipping", gfx)
        return

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="FlyDSL split-K HGEMM precision/performance cases",
    )
    parser.add_argument(
        "-d",
        "--dtype",
        type=dtypes.str2Dtype,
        choices=[dtypes.bf16],
        nargs="*",
        default=[dtypes.bf16],
        help="Input/output dtype (bf16 only).",
    )
    parser.add_argument(
        "-c",
        "--case",
        type=str,
        choices=list(SPLITK_PRECISION_CASES),
        nargs="*",
        default=list(SPLITK_PRECISION_CASES),
        help="""Regression cases to sweep.
        e.g.: -c splitk8_tile32_m104_n384_k7168""",
    )
    args = parser.parse_args()

    rows = []
    for dtype, case_name in itertools.product(args.dtype, args.case):
        rows.append(
            test_flydsl_splitk_hgemm(
                case=case_name,
                dtype=dtype,
                **SPLITK_PRECISION_CASES[case_name],
            )
        )

    df = pd.DataFrame(rows)
    aiter.logger.info(
        "FlyDSL split-K HGEMM summary (markdown):\n%s",
        df.to_markdown(index=False),
    )


if __name__ == "__main__":
    main()
