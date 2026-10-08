# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import csv
import re
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_MANIFESTS = (
    (
        "f6f4gemm",
        "f6f4gemm_bf16_per1x32Fp6Fp4.csv",
        "mxfp6_mxfp4_c0_256_padk2",
        "a6w4_asm_tuned_gemm.csv",
    ),
    (
        "f4f6gemm",
        "f4f6gemm_bf16_per1x32Fp4Fp6.csv",
        "mxfp4_c0_mxfp6_256_padk2",
        "a4w6_asm_tuned_gemm.csv",
    ),
)


@pytest.mark.parametrize(
    ("module", "manifest_name", "pack_layout", "tuned_name"), _MANIFESTS
)
def test_mixed_mxfp_manifest_objects(module, manifest_name, pack_layout, tuned_name):
    manifest_dir = _REPO_ROOT / "hsa" / "gfx950" / module
    with (manifest_dir / manifest_name).open(newline="") as manifest_file:
        rows = list(csv.DictReader(manifest_file))

    kernel_names = {row["knl_name"] for row in rows}
    assert len(rows) >= 4
    assert len(kernel_names) == len(rows)
    for kernel_name in kernel_names:
        match = re.fullmatch(r"_ZN5aiter(\d+)(.+)E", kernel_name)
        assert match is not None
        declared_length, entrypoint = match.groups()
        assert int(declared_length) == len(entrypoint)
        assert entrypoint.startswith(f"{module}_")
    assert len({row["co_name"] for row in rows}) == len(rows)
    assert {row["co_name"] for row in rows} == {
        path.name for path in manifest_dir.glob("*.co")
    }
    for row in rows:
        assert row["pack_layout"] == pack_layout
        assert row["tile_M"] == row["tile_N"] == row["block_size"] == "256"
        assert row["splitK"] == "0"
        assert row["bpreshuffle"] == "0"
        swizzle_bounds = tuple(
            int(row[name])
            for name in ("swizzle_max_M", "swizzle_max_N", "swizzle_max_K")
        )
        assert min(swizzle_bounds) >= 0
        if swizzle_bounds[2] > 0:
            assert swizzle_bounds[0] > 0 and swizzle_bounds[1] > 0

        code_object = manifest_dir / row["co_name"]
        code_object_bytes = code_object.read_bytes()
        assert code_object_bytes.startswith(b"\x7fELF")
        assert row["knl_name"].encode() in code_object_bytes

    tuned_path = _REPO_ROOT / "aiter" / "configs" / tuned_name
    with tuned_path.open(newline="") as tuned_file:
        tuned_rows = list(csv.DictReader(tuned_file))
    assert {row["kernelName"] for row in tuned_rows} <= kernel_names
    assert all(row["splitK"] == "0" for row in tuned_rows)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
