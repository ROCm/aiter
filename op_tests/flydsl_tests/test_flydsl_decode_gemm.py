# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Runtime correctness tests for the public BF16 decode GEMM operation.

Pytest collects the validity cases below (no timing). ``python3`` this file
to run a small ``@benchmark`` / ``run_perftest`` sweep with a markdown table.
"""

from __future__ import annotations

import argparse

import pandas as pd
import pytest
import torch

import aiter
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.test_common import benchmark, checkAllclose, run_perftest

pytest.importorskip("flydsl")

from aiter.ops.flydsl.gemm_kernels import (
    ActivationSource,
    BlockMfmaDecodeConfig,
    ContractionMode,
    ReductionMode,
    WaveDecodeConfig,
    gemm_decode_bf16,
)

ARCH = get_gfx_runtime()
SUPPORTED_ARCHS = ("gfx942", "gfx950")
ATOL = 0.125
RTOL = 0.01

pytestmark = pytest.mark.skipif(
    ARCH not in SUPPORTED_ARCHS,
    reason="BF16 decode GEMM requires gfx942 or gfx950",
)


def _inputs(m: int, n: int, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    row = torch.arange(m, device="cuda", dtype=torch.int32)[:, None]
    col = torch.arange(n, device="cuda", dtype=torch.int32)[:, None]
    red = torch.arange(k, device="cuda", dtype=torch.int32)[None, :]
    a = (((row * 7 + red * 3) % 23) - 11).to(torch.float32).div_(16).bfloat16()
    b = (((col * 5 + red * 7) % 29) - 14).to(torch.float32).div_(16).bfloat16()
    return a, b


def _bias(n: int) -> torch.Tensor:
    return (
        (((torch.arange(n, device="cuda", dtype=torch.int32) * 3) % 17) - 8)
        .to(torch.float32)
        .div_(16)
        .bfloat16()
    )


def _reference(
    a: torch.Tensor,
    b: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    product = a.float() @ b.float().T
    if bias is not None:
        product = product + bias.float()
    return product.bfloat16()


def _output(m: int, n: int) -> torch.Tensor:
    return torch.full((m, n), torch.nan, device="cuda", dtype=torch.bfloat16)


def _assert_output(
    output: torch.Tensor,
    reference: torch.Tensor,
) -> None:
    assert torch.isfinite(output).all()
    torch.testing.assert_close(output, reference, atol=ATOL, rtol=RTOL)


def _block_config(
    source: ActivationSource,
    *,
    columns_per_wave: int,
    persistent_n: bool = False,
) -> BlockMfmaDecodeConfig:
    return BlockMfmaDecodeConfig(
        waves_per_workgroup=4,
        columns_per_wave=columns_per_wave,
        activation_source=source,
        b_load_width=8,
        k_unroll=2,
        prefetch_stages=2 if ARCH == "gfx950" else 1,
        persistent_n=persistent_n,
        workgroups_per_cu=1,
        waves_per_eu=2 if ARCH == "gfx950" else 0,
    )


def _wave_config(m: int, k: int) -> WaveDecodeConfig:
    return WaveDecodeConfig(
        m_per_wave=m,
        n_per_wave=1,
        kvec=2,
        prefetch_depth=0,
        waves_per_eu=4,
        reduction=ReductionMode.DPP if k % 2 == 0 else ReductionMode.BPERMUTE,
        contraction=(
            ContractionMode.DOT2_BF16
            if ARCH == "gfx950"
            else ContractionMode.SCALAR_F32
        ),
    )


def _run_config(
    m: int,
    n: int,
    k: int,
    config: BlockMfmaDecodeConfig,
    *,
    with_bias: bool = False,
) -> None:
    a, b = _inputs(m, n, k)
    bias = _bias(n) if with_bias else None
    output = _output(m, n)
    returned = gemm_decode_bf16(
        a,
        b,
        output,
        config,
        bias=bias,
    )
    torch.cuda.synchronize()
    assert returned is output
    _assert_output(output, _reference(a, b, bias))


def test_wave_public_path_no_bias() -> None:
    m, n, k = 1, 64, 128
    a, b = _inputs(m, n, k)
    output = _output(m, n)
    returned = gemm_decode_bf16(a, b, output, _wave_config(m, k))
    torch.cuda.synchronize()
    assert returned is output
    _assert_output(output, _reference(a, b))


def test_wave_public_path_bias_and_odd_n_and_k_tails() -> None:
    m, n, k = 5, 65, 257
    a, b = _inputs(m, n, k)
    bias = _bias(n)
    output = _output(m, n)
    returned = gemm_decode_bf16(
        a,
        b,
        output,
        _wave_config(m, k),
        bias=bias,
    )
    torch.cuda.synchronize()
    assert returned is output
    _assert_output(output, _reference(a, b, bias))


def test_block_mfma_global() -> None:
    _run_config(
        3,
        65,
        257,
        _block_config(ActivationSource.GLOBAL, columns_per_wave=2),
        with_bias=True,
    )


def test_block_mfma_full_lds_k_padding_and_n_boundary() -> None:
    _run_config(
        5,
        17,
        129,
        _block_config(ActivationSource.FULL_LDS, columns_per_wave=1),
    )


def test_block_mfma_persistent_n_multiple_turns_and_partial_group() -> None:
    _run_config(
        3,
        5001,
        257,
        _block_config(
            ActivationSource.FULL_LDS,
            columns_per_wave=1,
            persistent_n=True,
        ),
        with_bias=True,
    )


@benchmark()
def test_gemm_decode(m, n, k, dtype):
    a, b = _inputs(m, n, k)
    output = _output(m, n)
    config = _wave_config(m, k)
    ref = a.float() @ b.float().T

    candidates = {
        "flydsl": lambda: gemm_decode_bf16(a, b, output, config),
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
            msg=f"{name}: gemm_decode_bf16 {m}x{n}x{k} {dtype}",
        )
        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = flops / us / 1e6 if us else 0
        ret[f"{name} TB/s"] = nbytes / us / 1e6 if us else 0
        ret[f"{name} err"] = err
    return ret


# Binding starts with test_*; pytest must not collect or time this sweep.
test_gemm_decode.__test__ = False


def main():
    if ARCH not in SUPPORTED_ARCHS:
        aiter.logger.warning("gemm_decode_bf16 unsupported on %s; skipping", ARCH)
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
        default=[
            (1, 64, 128),
            (3, 1536, 128),
            (1, 896, 7168),
        ],
        help="""Shape of mnk. Tiny-K plus a large-K decode cell.
    e.g.:   -s 1,64,128
            --mnk 1,896,7168""",
    )
    args = parser.parse_args()

    for dtype in args.dtype:
        df = []
        for m, n, k in args.mnk:
            if not 1 <= m <= 5:
                aiter.logger.warning(
                    "gemm_decode_bf16 supports M in [1, 5]; skipping m=%s", m
                )
                continue
            try:
                _wave_config(m, k).validate(m=m, n=n, k=k, arch=ARCH)
            except ValueError as err:
                aiter.logger.warning(
                    "gemm_decode_bf16 skipping %sx%sx%s: %s", m, n, k, err
                )
                continue
            df.append(test_gemm_decode(m, n, k, dtype))
        if df:
            df = pd.DataFrame(df)
            aiter.logger.info(
                "gemm_decode_bf16 summary (markdown):\n%s",
                df.to_markdown(index=False),
            )


if __name__ == "__main__":
    main()
