# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""A gfx942;gfx950 CK MoE codegen keeps every arch's tuned instance names.

No GPU and no hipcc. gen_instances.py only writes the lookup and the
instance sources. CI prebuilds with GPU_ARCHS=gfx942;gfx950, and get_gfx()
returns the last entry, so this has to hold for both orders.
"""

import csv
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
GEN = REPO / "csrc/ck_gemm_moe_2stages_codegen/gen_instances.py"
LIMIT_CSV = (
    REPO / "aiter/configs/model_configs/a8w8_plain_tuned_swiglu_limit_fmoe_gfx942.csv"
)

# The name MI300X shard 5 failed on, and one gfx950-list name the heuristic uses.
_GFX942_TILE = (
    "moe_ck2stages_gemm1_256x16x64x256_1x4_MulABScale_v1_"
    "Nswizzle0_Quant2_MulRoutedWeight0_silu_F8_F8_B16"
)
_GFX950_TILE = (
    "moe_ck2stages_gemm1_256x32x64x256_1x4_TypeCast_v1_"
    "Nswizzle0_Quant2_MulRoutedWeight0_silu_F8_F8_B16"
)


def _generate(gpu_archs: str) -> Path:
    out = Path(tempfile.mkdtemp())
    env = os.environ.copy()
    env["GPU_ARCHS"] = gpu_archs
    subprocess.check_call(
        [
            sys.executable,
            str(GEN),
            "-a",
            "f8",
            "-b",
            "f8",
            "-c",
            "b16",
            "-q",
            "per_token",
            "-act",
            "silu",
            "-m",
            "2",
            "--runtime-swiglu-limit",
            "-w",
            str(out),
        ],
        env=env,
    )
    return out


def _csv_kernel_names() -> set[str]:
    names = set()
    with LIMIT_CSV.open() as handle:
        for row in csv.DictReader(handle):
            names.add(row["kernelName1"])
            names.add(row["kernelName2"])
    return names


def test_dual_arch_lookup_keeps_both_lists():
    tuned = _csv_kernel_names()
    assert _GFX942_TILE in tuned
    for gpu_archs in ("gfx942;gfx950", "gfx950;gfx942"):
        out = _generate(gpu_archs)
        lookup = (out / "gemm_moe_ck2stages_lookup.h").read_text()
        missing = sorted(name for name in tuned if name not in lookup)
        assert missing == [], (gpu_archs, missing)
        assert _GFX950_TILE in lookup
        # The heuristic follows the last GPU_ARCHS entry and must not switch
        # element op just because the other arch's tiles were added.
        stage1 = (out / "ck2stages_moe_stage1_heuristic_dispatch.hpp").read_text()
        if gpu_archs.endswith("gfx950"):
            assert "TypeCast" in stage1 and "MulABScale" not in stage1
        else:
            assert "MulABScale" in stage1
        guarded = (out / "instances" / f"{_GFX942_TILE}.cu").read_text()
        assert "__gfx942__" in guarded
        plain = (out / "instances" / f"{_GFX950_TILE}.cu").read_text()
        assert "__HIP_DEVICE_COMPILE__" not in plain


def test_single_arch_codegen_is_unchanged():
    out = _generate("gfx942")
    lookup = (out / "gemm_moe_ck2stages_lookup.h").read_text()
    assert _GFX942_TILE in lookup
    assert _GFX950_TILE not in lookup
    source = (out / "instances" / f"{_GFX942_TILE}.cu").read_text()
    assert "__HIP_DEVICE_COMPILE__" not in source
