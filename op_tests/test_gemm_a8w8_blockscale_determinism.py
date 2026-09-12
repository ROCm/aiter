# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import csv
from pathlib import Path

import pytest
import torch

_HISTORICAL_ROWS = [
    pytest.param(
        "gfx950",
        256,
        1,
        2048,
        3072,
        "ck",
        8,
        2,
        "a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1",
        id="gfx950-ck-splitk2",
    ),
    pytest.param(
        "gfx950",
        256,
        4,
        7168,
        4096,
        "ck",
        6,
        3,
        "a8w8_blockscale_1x128x128_256x16x64x128_8x16_16x16_1x1_16x16x1_8x32x1_1x16x1x16_4_1x1_intrawave_v1",
        id="gfx950-ck-splitk3",
    ),
    pytest.param(
        "gfx950",
        256,
        8,
        2048,
        3072,
        "cktile",
        20,
        3,
        "a8w8_blockscale_cktile_32x128x128_1x4x1_16x16x64_intrawave_0x1x0_4",
        id="gfx950-cktile-splitk3",
    ),
    pytest.param(
        "gfx942",
        304,
        1,
        2048,
        2048,
        "ck",
        8,
        2,
        "a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1",
        id="gfx942-ck-splitk2",
    ),
    pytest.param(
        "gfx942",
        304,
        1,
        512,
        6144,
        "ck",
        8,
        3,
        "a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1",
        id="gfx942-ck-splitk3",
    ),
]


def test_tuned_blockscale_configs_use_deterministic_splitk():
    config_root = Path(__file__).resolve().parents[1] / "aiter" / "configs"
    config_files = [config_root / "a8w8_blockscale_tuned_gemm.csv"]
    config_files.extend(
        sorted((config_root / "model_configs").glob("*a8w8_blockscale_tuned_gemm*.csv"))
    )

    offenders = []
    for path in config_files:
        with path.open(newline="", encoding="utf-8") as handle:
            for line, row in enumerate(csv.DictReader(handle), start=2):
                if row["libtype"] in {"ck", "cktile"} and int(row["splitK"]) > 1:
                    offenders.append(
                        f"{path.relative_to(config_root)}:{line} "
                        f"M={row['M']} N={row['N']} K={row['K']} "
                        f"splitK={row['splitK']}"
                    )

    assert not offenders, "nondeterministic tuned split-K rows:\n" + "\n".join(
        offenders
    )


def _make_inputs(m, n, k):
    from aiter import dtypes

    torch.manual_seed(0)
    x = (torch.rand((m, k), dtype=torch.float16, device="cuda") / 10).to(dtypes.fp8)
    weight = (torch.rand((n, k), dtype=torch.float16, device="cuda") / 10).to(
        dtypes.fp8
    )
    x_scale = torch.rand((m, k // 128), dtype=torch.float32, device="cuda")
    w_scale = torch.rand((n // 128, k // 128), dtype=torch.float32, device="cuda")
    return x, weight, x_scale, w_scale


def _reference(inputs, m, n, k):
    x, weight, x_scale, w_scale = inputs
    x = (x.float().view(m, -1, 128) * x_scale.unsqueeze(-1)).view(m, k)
    w_scale = w_scale.repeat_interleave(128, dim=0).repeat_interleave(128, dim=1)
    weight = weight.float() * w_scale[:n, :k]
    return torch.nn.functional.linear(x, weight).to(torch.bfloat16)


@pytest.mark.parametrize(
    "gfx,cu_num,m,n,k,backend,kernel_id,split_k,kernel_name", _HISTORICAL_ROWS
)
def test_gemm_a8w8_blockscale_splitk_is_deterministic(
    gfx, cu_num, m, n, k, backend, kernel_id, split_k, kernel_name
):
    if not torch.cuda.is_available():
        pytest.skip("ROCm GPU is required")

    import aiter
    from aiter.jit.utils.chip_info import get_cu_num, get_gfx_runtime
    from aiter.ops.gemm_op_a8w8 import (
        gemm_a8w8_blockscale_ck,
        gemm_a8w8_blockscale_cktile,
    )
    from aiter.test_common import checkAllclose

    if get_gfx_runtime() != gfx or get_cu_num() != cu_num:
        pytest.skip(f"historical row targets {gfx}/{cu_num} CUs")

    inputs = _make_inputs(m, n, k)
    x, weight, x_scale, w_scale = inputs
    reference = _reference(inputs, m, n, k)

    def run_production(requested_split_k):
        out = torch.empty((m, n), dtype=torch.bfloat16, device="cuda")
        if backend == "cktile":
            return gemm_a8w8_blockscale_cktile(
                x,
                weight,
                x_scale,
                w_scale,
                out,
                isBpreshuffled=False,
                splitK=requested_split_k,
                kernelName=kernel_name,
            )
        return gemm_a8w8_blockscale_ck(
            x,
            weight,
            x_scale,
            w_scale,
            out,
            splitK=requested_split_k,
            kernelName=kernel_name,
        )

    raw_split1 = torch.empty((m, n), dtype=torch.bfloat16, device="cuda")
    if backend == "cktile":
        aiter.gemm_a8w8_blockscale_cktile_tune(
            x, weight, x_scale, w_scale, raw_split1, kernel_id, 1, False
        )
    else:
        aiter.gemm_a8w8_blockscale_tune(
            x, weight, x_scale, w_scale, raw_split1, kernel_id, 1
        )

    outputs = [run_production(split_k) for _ in range(20)]
    torch.cuda.synchronize()

    max_diff = max(
        (raw_split1.float() - output.float()).abs().max().item() for output in outputs
    )
    assert all(torch.equal(raw_split1, output) for output in outputs), (
        f"historical {backend} splitK={split_k} row did not use its exact "
        f"kernel at deterministic splitK=1, "
        f"max absolute difference={max_diff}"
    )
    err_ratio = checkAllclose(
        reference,
        outputs[0],
        rtol=1e-2,
        atol=1e-2,
        tol_err_ratio=0.01,
        catastrophic_check=True,
        msg=f"{gfx} {backend} splitK={split_k}",
    )
    assert err_ratio <= 0.01


@pytest.mark.parametrize("backend", ["ck", "cktile"])
@pytest.mark.parametrize("split_k", [-1, 31])
def test_gemm_a8w8_blockscale_splitk_validates_before_clamping(backend, split_k):
    if not torch.cuda.is_available():
        pytest.skip("ROCm GPU is required")

    from aiter.ops.gemm_op_a8w8 import (
        gemm_a8w8_blockscale_ck,
        gemm_a8w8_blockscale_cktile,
    )

    m, n, k = 1, 128, 128
    x, weight, x_scale, w_scale = _make_inputs(m, n, k)
    out = torch.empty((m, n), dtype=torch.bfloat16, device="cuda")
    with pytest.raises(RuntimeError, match=r"splitK must be in the range \[0, 30\]"):
        if backend == "cktile":
            gemm_a8w8_blockscale_cktile(
                x,
                weight,
                x_scale,
                w_scale,
                out,
                isBpreshuffled=False,
                splitK=split_k,
            )
        else:
            gemm_a8w8_blockscale_ck(x, weight, x_scale, w_scale, out, splitK=split_k)
