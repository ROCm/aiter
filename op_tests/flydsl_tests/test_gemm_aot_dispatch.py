# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""CPU-side regressions for mixed-family FlyDSL GEMM AOT dispatch."""

import csv
import os
import sys

import pytest
import torch
from torch._subclasses.fake_tensor import is_fake

from aiter.aot.flydsl import gemm
from aiter.ops.flydsl.gemm_mxfp8 import DEFAULT_CONFIG, flydsl_mxfp8_kernel_name


@pytest.fixture
def mixed_csv(tmp_path):
    rows = [
        {
            "libtype": "flydsl",
            "kernelName": flydsl_mxfp8_kernel_name(
                DEFAULT_CONFIG, out_dtype=torch.bfloat16, bpreshuffle=False
            ),
            "M": 17,
            "N": 128,
            "K": 7168,
            "gfx": "gfx950",
            "cu_num": 256,
            "outdtype": "torch.bfloat16",
            "bias": False,
            "bpreshuffle": False,
            "splitK": 1,
        }
    ]
    for gfx in ("gfx950", "gfx1250"):
        rows.append(
            dict(
                rows[0],
                gfx=gfx,
                kernelName=(
                    "flydsl_hgemm_abf16_wbf16_bf16_"
                    f"t128x128x64x2_ks1_w2x2x1_bias0_ktail0_gm0_pft_{gfx}"
                ),
            )
        )
    path = tmp_path / "mixed.csv"
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows + rows[:1])  # Duplicate jobs must still be removed.
    return path


def test_parse_mixed_kernel_families(mixed_csv):
    jobs = gemm.parse_csv(mixed_csv)
    assert [(j["kind"], j["target_gfx"]) for j in jobs] == [
        ("mxfp8", "gfx950"),
        ("hgemm", "gfx950"),
        ("a16w16_gfx1250", "gfx1250"),
    ]


@pytest.mark.parametrize(
    "kind,arch,fake",
    [
        ("hgemm", "gfx950", False),
        ("mxfp8", "gfx950", False),
        ("a16w16_gfx1250", "gfx1250", False),
        ("preshuffle", "gfx950", True),
        ("8wave", "gfx950", True),
        ("mxfp8_128_wmma", "gfx1250", True),
        ("ptpc_wmma", "gfx1250", True),
    ],
)
def test_compile_dispatch_context(monkeypatch, kind, arch, fake):
    calls = []

    def compile_stub(**kwargs):
        assert os.environ["FLYDSL_GPU_ARCH"] == arch
        assert is_fake(torch.empty(1)) == fake
        calls.append(kwargs)

    monkeypatch.setenv("FLYDSL_GPU_ARCH", "original")
    monkeypatch.setattr(gemm, f"_compile_{kind}_to_cache", compile_stub)
    result = gemm.compile_one_config(
        kernel_name="test_kernel", kind=kind, m=17, n=128, k=7168, gfx=arch
    )
    assert result["compile_time"] is not None
    assert result["compile_arch"] == arch
    assert len(calls) == 1
    assert {key: calls[0][key] for key in ("m", "n", "k")} == {
        "m": 17,
        "n": 128,
        "k": 7168,
    }
    if not fake:
        assert calls[0]["target_gfx"] == arch
    assert os.environ["FLYDSL_GPU_ARCH"] == "original"


@pytest.mark.parametrize(
    "arch,kinds",
    [
        (None, ["mxfp8", "hgemm", "a16w16_gfx1250"]),
        ("gfx950", ["mxfp8", "hgemm"]),
        ("gfx1250", ["a16w16_gfx1250"]),
        ("gfx950;gfx1250", ["mxfp8", "hgemm", "a16w16_gfx1250"]),
        ("gfx950,gfx1250", ["mxfp8", "hgemm", "a16w16_gfx1250"]),
    ],
)
def test_main_submits_all_filtered_jobs(monkeypatch, mixed_csv, arch, kinds):
    monkeypatch.setattr(sys, "argv", ["gemm", "--csv", str(mixed_csv)])
    monkeypatch.delenv("ARCH", raising=False)
    monkeypatch.delenv("GPU_ARCHS", raising=False)
    if arch is not None:
        monkeypatch.setenv("GPU_ARCHS", arch)
    submitted = []

    def run_stub(compile_fn, jobs):
        assert compile_fn is gemm.compile_one_config
        submitted.extend(jobs)
        return [{"compile_time": 0.0} for _ in jobs]

    monkeypatch.setattr(gemm, "run_jobs_parallel", run_stub)
    with pytest.raises(SystemExit) as exc:
        gemm.main()
    assert exc.value.code == 0
    assert [job["kind"] for job in submitted] == kinds
