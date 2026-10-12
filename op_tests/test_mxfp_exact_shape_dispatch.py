# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Numerical coverage for exact-shape MXFP assembly tuning rows."""

import csv
import gc
from pathlib import Path

import pytest
import torch

from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.ops.gemm_op_a4w6 import (
    _default_gemm_a4w6_kernel,
    _select_gemm_a4w6_kernel,
    gemm_a4w6_asm,
)
from aiter.ops.gemm_op_a6w4 import (
    _default_gemm_a6w4_kernel,
    _select_gemm_a6w4_kernel,
    gemm_a6w4_asm,
    quant_mxfp4_gemm,
)
from aiter.ops.gemm_op_a6w6 import (
    _default_gemm_a6w6_kernel,
    _select_gemm_a6w6_kernel,
    gemm_a6w6_asm,
    quant_mxfp6_gemm,
)

ROOT = Path(__file__).resolve().parents[1]
SPECS = {
    "a6w6": (
        ROOT / "hsa/gfx950/f6gemm/f6gemm_bf16_per1x32Fp6.csv",
        ROOT / "aiter/configs/a6w6_blockscale_tuned_gemm.csv",
        27,
    ),
    "a6w4": (
        ROOT / "hsa/gfx950/f6f4gemm/f6f4gemm_bf16_per1x32Fp6Fp4.csv",
        ROOT / "aiter/configs/a6w4_asm_tuned_gemm.csv",
        22,
    ),
    "a4w6": (
        ROOT / "hsa/gfx950/f4f6gemm/f4f6gemm_bf16_per1x32Fp4Fp6.csv",
        ROOT / "aiter/configs/a4w6_asm_tuned_gemm.csv",
        22,
    ),
}


def _is_gfx950():
    try:
        return torch.cuda.is_available() and get_gfx_runtime() == "gfx950"
    except (KeyError, RuntimeError):
        return False


requires_gfx950 = pytest.mark.skipif(
    not _is_gfx950(), reason="shape-specialized MXFP assembly requires gfx950"
)


def _load_cases():
    cases = []
    for family, (manifest_path, config_path, expected_count) in SPECS.items():
        with manifest_path.open(newline="") as stream:
            manifest_rows = list(csv.DictReader(stream))
        exact_kernels = {
            row["knl_name"]
            for row in manifest_rows
            if min(int(row[name]) for name in ("exact_M", "exact_N", "exact_K")) > 0
        }
        with config_path.open(newline="") as stream:
            config_rows = list(csv.DictReader(stream))
        family_cases = [
            (
                family,
                (int(row["M"]), int(row["N"]), int(row["K"])),
                row["kernelName"],
            )
            for row in config_rows
            if row["kernelName"] in exact_kernels and int(row["M"]) != 9450
        ]
        if len(family_cases) != expected_count:
            raise RuntimeError(
                f"{family}: expected {expected_count} exact-shape tuning rows, "
                f"found {len(family_cases)}"
            )
        cases.extend(
            pytest.param(
                family,
                shape,
                kernel,
                id=f"{family}-{shape[0]}x{shape[1]}x{shape[2]}",
            )
            for family, shape, kernel in family_cases
        )
    return cases


@pytest.mark.parametrize("family,shape,candidate_kernel", _load_cases())
@torch.no_grad()
@requires_gfx950
def test_exact_shape_dispatch_matches_generic_kernel(family, shape, candidate_kernel):
    M, N, K = shape
    torch.manual_seed(20261012 + M + N + K)
    x = torch.randn((M, K), dtype=torch.bfloat16, device="cuda")
    weight = torch.randn((N, K), dtype=torch.bfloat16, device="cuda")
    out_ref = torch.empty((M, N), dtype=torch.bfloat16, device="cuda")
    out = torch.empty_like(out_ref)

    if family == "a6w6":
        A, A_scale = quant_mxfp6_gemm(x)
        B, B_scale = quant_mxfp6_gemm(weight)
        selected = _select_gemm_a6w6_kernel(M, N, K, None)
        baseline = _default_gemm_a6w6_kernel(M, N, K)
        launch = gemm_a6w6_asm
    elif family == "a6w4":
        A, A_scale = quant_mxfp6_gemm(x)
        B, B_scale = quant_mxfp4_gemm(weight)
        selected = _select_gemm_a6w4_kernel(M, N, K, None, device=x.device)
        baseline = _default_gemm_a6w4_kernel(M, N, K)
        launch = gemm_a6w4_asm
    else:
        A, A_scale = quant_mxfp4_gemm(x)
        B, B_scale = quant_mxfp6_gemm(weight)
        selected = _select_gemm_a4w6_kernel(M, N, K, None, device=x.device)
        baseline = _default_gemm_a4w6_kernel(M, N, K)
        launch = gemm_a4w6_asm

    assert selected == candidate_kernel
    launch(A, B, A_scale, B_scale, out_ref, K, baseline)
    launch(A, B, A_scale, B_scale, out, K, selected)
    torch.cuda.synchronize()
    assert torch.equal(out.view(torch.int16), out_ref.view(torch.int16))

    del x, weight, A, A_scale, B, B_scale, out_ref, out
    gc.collect()
    torch.cuda.empty_cache()
