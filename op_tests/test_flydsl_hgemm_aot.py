# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The runtime finds the gfx950 FlyDSL hgemm kernels built by aiter.aot.flydsl.gemm."""

import csv
import os
import subprocess
import sys
from pathlib import Path

import pytest

from aiter.jit.utils.chip_info import get_gfx

# A split-K kernel, so the run also uses the split-K semaphore.
KERNEL = (
    "flydsl_hgemm_abf16_wbf16_bf16_t16x32x128x7_ks7_w1x2x1_bias0_ktail0_gm0_pft_gfx950"
)


def _hgemm_row(table: Path) -> tuple[list[str], dict]:
    with table.open(newline="") as stream:
        reader = csv.DictReader(stream)
        assert reader.fieldnames is not None
        rows = [row for row in reader if row["kernelName"].strip() == KERNEL]
        assert rows
        return list(reader.fieldnames), rows[0]


def test_hgemm_aot_cache_is_retrievable_in_run_only_mode(tmp_path: Path):
    if get_gfx() != "gfx950":
        pytest.skip("FlyDSL hgemm AOT cache retrieval requires gfx950")
    root = Path(__file__).resolve().parents[1]
    fields, row = _hgemm_row(
        root / "aiter/configs/model_configs/dsv3_bf16_tuned_gemm.csv"
    )
    table = tmp_path / "bf16_tuned_gemm.csv"
    with table.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerow(row)
    env = os.environ.copy()
    env["FLYDSL_RUNTIME_CACHE_DIR"] = str(tmp_path / "cache")
    compiled = subprocess.run(
        [sys.executable, "-m", "aiter.aot.flydsl.gemm", "--csv", str(table)],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert compiled.returncode == 0, compiled.stdout + compiled.stderr

    run_env = dict(env, FLYDSL_RUNTIME_RUN_ONLY="1")
    m, n, k, solidx = row["M"], row["N"], row["K"], row["solidx"]
    retrieve_code = f"""
import torch
from aiter.tuned_gemm import flydsl_gemm
a = torch.randn({m}, {k}, dtype=torch.bfloat16, device="cuda")
w = torch.randn({n}, {k}, dtype=torch.bfloat16, device="cuda")
flydsl_gemm(a, w, solidx={solidx}, config={{"kernelName": "{KERNEL}"}})
torch.cuda.synchronize()
"""
    retrieved = subprocess.run(
        [sys.executable, "-c", retrieve_code],
        cwd=root,
        env=run_env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert retrieved.returncode == 0, retrieved.stderr


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
