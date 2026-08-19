# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""CSV overlay identity for FlyDSL HGEMM AOT jobs whose names omit n/k."""

from __future__ import annotations

import csv
import tempfile
from pathlib import Path

from aiter.aot.flydsl.common import job_identity
from aiter.aot.flydsl.gemm import parse_csv

_CSV_FIELDS = (
    "gfx",
    "cu_num",
    "M",
    "N",
    "K",
    "bias",
    "dtype",
    "outdtype",
    "scaleAB",
    "bpreshuffle",
    "libtype",
    "solidx",
    "splitK",
    "us",
    "kernelName",
    "err_ratio",
    "tflops",
    "bw",
)

# Legacy names omit n/k; the HGEMM parse helper then returns n=None, k=None.
_KERNEL_NAME = (
    "flydsl_gemm5_abf16_wbf16_bf16_"
    "t16x64x128_split_k2_"
    "block_m_warp1_block_n_warp2_block_k_warp1_"
    "async_copyTrue_b_to_ldsTrue_b_preshuffleFalse_c_to_ldsFalse_gfx950"
)


def _row(*, m: int, n: int, k: int) -> dict[str, str]:
    return {
        "gfx": "gfx950",
        "cu_num": "256",
        "M": str(m),
        "N": str(n),
        "K": str(k),
        "bias": "False",
        "dtype": "torch.bfloat16",
        "outdtype": "torch.bfloat16",
        "scaleAB": "False",
        "bpreshuffle": "False",
        "libtype": "flydsl",
        "solidx": "0",
        "splitK": "2",
        "us": "0",
        "kernelName": _KERNEL_NAME,
        "err_ratio": "0",
        "tflops": "0",
        "bw": "0",
    }


def test_parse_csv_keeps_distinct_hgemm_jobs_when_name_omits_nk():
    rows = (
        _row(m=4, n=6144, k=2048),
        _row(m=4, n=6144, k=3072),
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = Path(tmpdir) / "tuned_gemm.csv"
        with csv_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=_CSV_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        jobs = parse_csv(str(csv_path))

    assert len(jobs) == 2
    assert [(job["n"], job["k"]) for job in jobs] == [(6144, 2048), (6144, 3072)]
    assert job_identity(jobs[0]) != job_identity(jobs[1])
