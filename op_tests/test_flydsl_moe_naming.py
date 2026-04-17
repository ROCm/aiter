# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Unit tests for FlyDSL MOE kernel naming and suffix parsing.

These are pure-Python tests (no GPU / torch required) that verify the kernel
name construction, suffix regex, and the FQ-detection logic used in
get_2stage_cfgs.

The regex and helpers are copied here to avoid importing aiter (which
transitively requires torch).  If the source definitions change, these
tests should be updated to match.
"""

import re
import pytest

# ── Mirror of _SUFFIX_RE from aiter/ops/flydsl/moe_kernels.py ──
_SUFFIX_RE = re.compile(r"(?P<fp4>_fp4|_fq)?(?P<fp8>_fp8)?(?:_sbm(?P<sbm>\d+))?$")


def flydsl_kernel_name(
    stage: int,
    a_dtype: str,
    b_dtype: str,
    out_dtype: str,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    mode: str = "",
    sort_block_m: int = 0,
) -> str:
    """Mirror of aiter.ops.flydsl.moe_kernels.flydsl_kernel_name."""
    name = f"flydsl_moe{stage}_a{a_dtype}_w{b_dtype}_{out_dtype}_t{tile_m}x{tile_n}x{tile_k}"
    if mode:
        name += f"_{mode}"
    if sort_block_m > 0 and sort_block_m != tile_m:
        name += f"_sbm{sort_block_m}"
    return name


def _detect_fq(kernel_name: str) -> bool:
    """Mirror of the _s1_fq logic in aiter.fused_moe.get_2stage_cfgs."""
    is_flydsl = bool(kernel_name) and kernel_name.startswith("flydsl_")
    suffix = kernel_name.split("_t")[-1] if is_flydsl else ""
    return is_flydsl and ("_fp4" in suffix or "_fq" in suffix)


# ── Tests ────────────────────────────────────────────────────────


class TestSuffixRegex:
    """Verify _SUFFIX_RE recognises _fp4, _fq, _fp8, and _sbm suffixes."""

    @pytest.mark.parametrize(
        "suffix, expected_fp4",
        [
            ("_fp4", "_fp4"),
            ("_fq", "_fq"),
        ],
    )
    def test_fp4_group_variants(self, suffix, expected_fp4):
        name = f"flydsl_moe1_abf16_wbf16_bf16_t64x128x64{suffix}"
        m = _SUFFIX_RE.search(name)
        assert m is not None
        assert m.group("fp4") == expected_fp4

    def test_fp8_suffix(self):
        m = _SUFFIX_RE.search("flydsl_moe1_abf16_wbf16_bf16_t64x128x64_fp8")
        assert m is not None
        assert m.group("fp8") == "_fp8"
        assert m.group("fp4") is None

    def test_fq_fp8_combined(self):
        m = _SUFFIX_RE.search("flydsl_moe1_abf16_wbf16_bf16_t64x128x64_fq_fp8")
        assert m is not None
        assert m.group("fp4") == "_fq"
        assert m.group("fp8") == "_fp8"

    def test_fp4_with_sbm(self):
        m = _SUFFIX_RE.search("flydsl_moe1_abf16_wbf16_bf16_t64x128x64_fp4_sbm64")
        assert m is not None
        assert m.group("fp4") == "_fp4"
        assert m.group("sbm") == "64"

    def test_fq_with_sbm(self):
        m = _SUFFIX_RE.search("flydsl_moe1_abf16_wbf16_bf16_t64x128x64_fq_sbm128")
        assert m is not None
        assert m.group("fp4") == "_fq"
        assert m.group("sbm") == "128"

    def test_no_suffix(self):
        m = _SUFFIX_RE.search("flydsl_moe1_abf16_wbf16_bf16_t64x128x64")
        assert m is not None
        assert m.group("fp4") is None
        assert m.group("fp8") is None
        assert m.group("sbm") is None


class TestKernelNameConstruction:
    """Verify flydsl_kernel_name builds names correctly."""

    def test_basic_name(self):
        name = flydsl_kernel_name(1, "bf16", "bf16", "bf16", 64, 128, 64)
        assert name == "flydsl_moe1_abf16_wbf16_bf16_t64x128x64"

    def test_name_with_fp4_mode(self):
        name = flydsl_kernel_name(1, "bf16", "bf16", "bf16", 64, 128, 64, mode="fp4")
        assert name == "flydsl_moe1_abf16_wbf16_bf16_t64x128x64_fp4"

    def test_name_with_fq_mode(self):
        name = flydsl_kernel_name(1, "bf16", "bf16", "bf16", 64, 128, 64, mode="fq")
        assert name == "flydsl_moe1_abf16_wbf16_bf16_t64x128x64_fq"

    def test_name_with_sbm(self):
        name = flydsl_kernel_name(
            1, "bf16", "bf16", "bf16", 64, 128, 64, sort_block_m=128
        )
        assert name == "flydsl_moe1_abf16_wbf16_bf16_t64x128x64_sbm128"

    def test_sbm_equal_to_tile_m_is_omitted(self):
        name = flydsl_kernel_name(
            1, "bf16", "bf16", "bf16", 64, 128, 64, sort_block_m=64
        )
        assert name == "flydsl_moe1_abf16_wbf16_bf16_t64x128x64"


class TestStage1FqDetection:
    """Verify the _s1_fq detection logic from get_2stage_cfgs."""

    @pytest.mark.parametrize(
        "name, expected",
        [
            ("flydsl_moe1_abf16_wbf16_bf16_t64x128x64_fp4", True),
            ("flydsl_moe1_abf16_wbf16_bf16_t64x128x64_fq", True),
            ("flydsl_moe1_abf16_wbf16_bf16_t64x128x64_fq_sbm64", True),
            ("flydsl_moe1_abf16_wbf16_bf16_t64x128x64_fp4_fp8", True),
            ("flydsl_moe1_abf16_wbf16_bf16_t64x128x64", False),
            ("flydsl_moe1_abf16_wbf16_bf16_t64x128x64_fp8", False),
            ("ck_moe_stage1_kernel", False),
            ("", False),
        ],
    )
    def test_fq_detection(self, name, expected):
        assert _detect_fq(name) == expected


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
