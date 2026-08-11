# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Coverage for the dedicated small-M BF16 HGEMM family, including its gfx942 port.

The gfx942 port uses the native `16x16x16` BF16 MFMA atom twice per logical K32
step and offers two global-to-LDS staging forms as an explicit compile-time
axis. Hardware tests run on the attached architecture; architecture legality is
additionally checked statically with `llvm-mc` so the gfx950-only forms are
covered without a gfx950 device.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile

import pytest
import torch

pytest.importorskip("flydsl")

from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl.gemm_kernels import flydsl_hgemm
from aiter.ops.flydsl.kernels.small_m_hgemm import (
    LDS_STAGING_DIRECT,
    LDS_STAGING_OPTIONS,
    LDS_STAGING_VGPR,
    SMALL_M_KERNEL_MAX,
    compile_small_m_hgemm_kernel,
    iter_small_m_registry_configs,
    parse_small_m_kernel_name,
    small_m_arch_params,
    small_m_lds_bytes,
    small_m_max_lds_bytes,
    small_m_kernel_name,
    small_m_tile_k_is_swizzle_safe,
)
from aiter.ops.flydsl.kernels.splitk_hgemm import (
    SPLIT_K_SEMAPHORE_MAX_LEN,
    WmmaHalf_m16n16k16,
    WmmaHalf_m16n16k32,
)

ATOL = 0.125
RTOL = 0.01

ARCH = get_gfx()
IS_GFX942 = ARCH == "gfx942"

requires_gfx942 = pytest.mark.skipif(
    not IS_GFX942, reason="the small-M gfx942 port requires a gfx942 device"
)
requires_gfx950 = pytest.mark.skipif(
    ARCH != "gfx950", reason="requires a gfx950 device"
)


def _llvm_mc() -> str | None:
    for candidate in (
        os.path.join(os.environ.get("ROCM_PATH", "/opt/rocm"), "llvm/bin/llvm-mc"),
        shutil.which("llvm-mc"),
    ):
        if candidate and os.path.exists(candidate):
            return candidate
    return None


def assembles(mcpu: str, instruction: str) -> bool:
    """True when `instruction` assembles for `mcpu`."""
    tool = _llvm_mc()
    assert tool is not None
    result = subprocess.run(
        [tool, "-arch=amdgcn", f"-mcpu={mcpu}", "-show-encoding"],
        input=instruction + "\n",
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def reference(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a.float() @ b.float().T).bfloat16()


def _csv_kernel_name(m=5, n=128, k=128):
    return small_m_kernel_name(
        ARCH,
        "bf16",
        m,
        n,
        k,
        tile_n=128,
        tile_k=64,
        split_k=1,
        block_n_warps=2,
        n_tile_repeat=1,
        persistent_n_tiles=1,
        waves_per_eu=0,
        b_to_lds_unroll=0,
        b_to_lds=False,
        has_bias=False,
        lds_staging=LDS_STAGING_DIRECT,
    )


def run_small_m(
    m: int,
    n: int,
    k: int,
    *,
    tile_n: int,
    tile_k: int = 64,
    block_n_warps: int,
    b_to_lds: bool,
    lds_staging: str,
    split_k: int = 1,
    seed: int = 20260806,
    **extra,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    a = torch.randn((m, k), generator=generator, device="cuda", dtype=torch.bfloat16)
    b = torch.randn((n, k), generator=generator, device="cuda", dtype=torch.bfloat16)
    # A poisoned output makes any silently skipped output column an obvious
    # failure instead of a coincidentally-correct zero.
    out = torch.full((m, n), float("nan"), device="cuda", dtype=torch.bfloat16)
    flydsl_hgemm(
        a,
        b,
        out=out,
        kernel_family="small_m",
        tile_m=16,
        tile_n=tile_n,
        tile_k=tile_k,
        split_k=split_k,
        block_m_warps=1,
        block_n_warps=block_n_warps,
        b_to_lds=b_to_lds,
        lds_staging=lds_staging,
        **extra,
    )
    torch.cuda.synchronize()
    return out, reference(a, b)


# ---------------------------------------------------------------------------
# Architecture contracts (host-only, run on any architecture)
# ---------------------------------------------------------------------------


def test_gfx942_uses_the_native_k16_bf16_atom():
    params = small_m_arch_params("gfx942")
    assert params["wmma_cls"] is WmmaHalf_m16n16k16
    assert params["mfma_per_warp_k"] == 2
    assert params["direct_dma_bytes"] == 4
    # The two K16 MFMAs must cover exactly the K32 step the gfx950 atom does.
    assert (
        params["wmma_cls"].WMMA_K * params["mfma_per_warp_k"]
        == WmmaHalf_m16n16k32.WMMA_K
    )


def test_gfx950_arch_params_are_unchanged():
    params = small_m_arch_params("gfx950")
    assert params["wmma_cls"] is WmmaHalf_m16n16k32
    assert params["mfma_per_warp_k"] == 1
    assert params["direct_dma_bytes"] == 16


def test_lds_capacity_is_architecture_aware():
    assert small_m_max_lds_bytes("gfx942") == 65536
    assert small_m_max_lds_bytes("gfx950") == 163840


def test_lds_model_matches_the_kernel_allocation():
    # A staging is double-buffered and aliases the C reshape buffer; B staging
    # is double-buffered on top of it.
    assert small_m_lds_bytes(tile_n=128, tile_k=64, b_to_lds=False) == 4096
    assert small_m_lds_bytes(tile_n=128, tile_k=64, b_to_lds=True) == 36864
    assert small_m_lds_bytes(tile_n=256, tile_k=64, b_to_lds=True) > 65536


@pytest.mark.parametrize(
    "instruction",
    [
        # Variant A: 4-byte direct global-to-LDS is the widest gfx942 supports.
        "buffer_load_dword v1, s[8:11], 0 offen sc0 lds",
        # Variant B: wide load into VGPRs plus a 128-bit LDS write.
        "buffer_load_dwordx4 v[4:7], v1, s[8:11], 0 offen",
        "ds_write_b128 v0, v[4:7]",
        # Shared: the native gfx942 BF16 MFMA atom and its LDS reads.
        "v_mfma_f32_16x16x16_bf16 v[0:3], v[4:5], v[6:7], v[0:3]",
        "ds_read_b128 v[4:7], v0",
    ],
)
def test_gfx942_supports_every_instruction_the_port_emits(instruction):
    if _llvm_mc() is None:
        pytest.skip("llvm-mc is not available")
    assert assembles("gfx942", instruction)


@pytest.mark.parametrize(
    "instruction",
    [
        # 16-byte direct-to-LDS: the gfx950 staging form.
        "buffer_load_dwordx4 v1, s[8:11], 0 offen sc0 lds",
        # The K32 BF16 MFMA atom the gfx950 path uses.
        "v_mfma_f32_16x16x32_bf16 v[0:3], v[4:7], v[8:11], v[0:3]",
    ],
)
def test_gfx950_only_forms_are_rejected_by_gfx942(instruction):
    if _llvm_mc() is None:
        pytest.skip("llvm-mc is not available")
    assert assembles("gfx950", instruction)
    assert not assembles("gfx942", instruction)


def test_non_gfx942_rejects_the_vgpr_staging_variant():
    if IS_GFX942:
        pytest.skip("this device is gfx942, where both staging forms exist")
    with pytest.raises(ValueError, match="only implemented for gfx942"):
        compile_small_m_hgemm_kernel(
            "bf16", 128, 128, TILE_N=128, BLOCK_N_WARPS=2, LDS_STAGING=LDS_STAGING_VGPR
        )


# ---------------------------------------------------------------------------
# Registry safety
# ---------------------------------------------------------------------------


def test_registry_configs_are_resource_safe():
    configs = list(
        iter_small_m_registry_configs("bf16", "bf16", m=8, n=7168, k=7168)
    )
    assert configs, "the registry yields no config for a supported shape"
    limit = small_m_max_lds_bytes(ARCH)
    for config in configs:
        assert config["lds_staging"] in LDS_STAGING_OPTIONS
        assert (
            small_m_lds_bytes(
                tile_n=config["tile_n"],
                tile_k=config["tile_k"],
                b_to_lds=config["b_to_lds"],
            )
            <= limit
        )
        assert 7168 % config["tile_n"] == 0
        assert (7168 // config["split_k"]) % config["tile_k"] == 0
        if config["split_k"] > 1:
            counters = 1 * (7168 // config["tile_n"])
            assert counters <= SPLIT_K_SEMAPHORE_MAX_LEN


@requires_gfx942
def test_registry_enumerates_both_staging_variants_on_gfx942():
    stagings = {
        config["lds_staging"]
        for config in iter_small_m_registry_configs("bf16", "bf16", m=8, n=7168, k=7168)
    }
    assert stagings == set(LDS_STAGING_OPTIONS)


@requires_gfx950
def test_registry_keeps_gfx950_on_the_direct_form_only():
    stagings = {
        config["lds_staging"]
        for config in iter_small_m_registry_configs("bf16", "bf16", m=8, n=7168, k=7168)
    }
    assert stagings == {LDS_STAGING_DIRECT}


@requires_gfx942
def test_registry_offers_a_legal_tile_for_a_16_aligned_output_width():
    configs = list(
        iter_small_m_registry_configs("bf16", "bf16", m=8, n=6288, k=7168)
    )
    assert configs, "N=6288 must have at least one legal gfx942 config"
    assert all(6288 % config["tile_n"] == 0 for config in configs)


def test_registry_rejects_shapes_outside_the_supported_m_range():
    assert not list(
        iter_small_m_registry_configs(
            "bf16", "bf16", m=SMALL_M_KERNEL_MAX, n=7168, k=7168
        )
    )


@pytest.mark.parametrize("m", [1, 5, 8, 16])
@pytest.mark.parametrize("lds_staging", LDS_STAGING_OPTIONS)
def test_versioned_kernel_name_round_trips_every_axis(m, lds_staging):
    name = small_m_kernel_name(
        "gfx942",
        "bf16",
        m,
        128,
        128,
        tile_n=128,
        tile_k=64,
        split_k=1,
        block_n_warps=2,
        n_tile_repeat=1,
        persistent_n_tiles=1,
        waves_per_eu=4,
        b_to_lds_unroll=8,
        b_to_lds=True,
        has_bias=False,
        lds_staging=lds_staging,
    )
    arch, name_m, name_n, name_k, config = parse_small_m_kernel_name(name)
    assert (arch, name_m, name_n, name_k) == ("gfx942", m, 128, 128)
    assert config["lds_staging"] == lds_staging
    assert config["tile_n"] == 128
    assert config["tile_k"] == 64
    assert config["split_k"] == 1
    assert config["block_n_warps"] == 2
    assert config["n_tile_repeat"] == 1
    assert config["persistent_n_tiles"] == 1
    assert config["waves_per_eu"] == 4
    assert config["b_to_lds"] is True
    assert config["b_to_lds_unroll"] == 8


def test_staging_variants_have_distinct_names_and_stale_names_are_rejected():
    common = dict(
        tile_n=128,
        tile_k=64,
        split_k=1,
        block_n_warps=2,
        n_tile_repeat=1,
        persistent_n_tiles=1,
        waves_per_eu=0,
        b_to_lds_unroll=0,
        b_to_lds=False,
        has_bias=False,
    )
    direct = small_m_kernel_name(
        "gfx942", "bf16", 8, 128, 128, lds_staging="direct", **common
    )
    vgpr = small_m_kernel_name(
        "gfx942", "bf16", 8, 128, 128, lds_staging="vgpr", **common
    )
    assert direct != vgpr
    with pytest.raises(ValueError, match="unrecognized"):
        parse_small_m_kernel_name("smallm_hgemm_bf16_16x128x64_S2TN_AS_BNW2")
    with pytest.raises(ValueError, match="not supported"):
        parse_small_m_kernel_name(vgpr.replace("agfx942", "agfx950"))


def test_aot_csv_parser_recognizes_small_m_and_rejects_stale_rows(tmp_path):
    from aiter.aot.flydsl.gemm import parse_csv

    cu_num = 304 if ARCH == "gfx942" else 256
    name = _csv_kernel_name()
    csv_path = tmp_path / "small_m.csv"
    csv_path.write_text(
        "gfx,cu_num,M,N,K,bias,libtype,kernelName\n"
        f"{ARCH},{cu_num},5,128,128,False,flydsl_small_m,{name}\n"
    )
    jobs = parse_csv(str(csv_path))
    assert len(jobs) == 1
    assert jobs[0]["kind"] == "small_m"
    assert jobs[0]["arch"] == ARCH
    assert jobs[0]["lds_staging"] == LDS_STAGING_DIRECT

    csv_path.write_text(
        "gfx,cu_num,M,N,K,bias,libtype,kernelName\n"
        f"{ARCH},{cu_num},5,128,128,False,flydsl_small_m,old_small_m_name\n"
    )
    with pytest.raises(ValueError, match="unrecognized"):
        parse_csv(str(csv_path))


def test_synthetic_tuned_csv_dispatches_exact_small_m(tmp_path, monkeypatch):
    import aiter.tuned_gemm as tuned_gemm
    from aiter.jit.core import AITER_CONFIGS

    m, n, k = 5, 128, 128
    cu_num = torch.cuda.get_device_properties(0).multi_processor_count
    name = _csv_kernel_name(m, n, k)
    csv_path = tmp_path / "bf16_tuned.csv"
    csv_path.write_text(
        "gfx,cu_num,M,N,K,bias,dtype,outdtype,scaleAB,bpreshuffle,"
        "libtype,solidx,splitK,us,kernelName,err_ratio,tflops,bw\n"
        f"{ARCH},{cu_num},{m},{n},{k},False,torch.bfloat16,torch.bfloat16,"
        f"False,False,flydsl_small_m,0,1,1.0,{name},0.0,0.0,0.0\n"
    )
    try:
        with monkeypatch.context() as isolated:
            isolated.setenv("AITER_CONFIG_GEMM_BF16", str(csv_path))
            AITER_CONFIGS.get_config_file.cache_clear()
            tuned_gemm.get_GEMM_A16W16_config_.cache_clear()
            tuned_gemm.get_GEMM_A16W16_config.cache_clear()
            selected = tuned_gemm.get_GEMM_A16W16_config(
                m,
                n,
                k,
                False,
                str(torch.bfloat16),
                str(torch.bfloat16),
            )
            assert selected["libtype"] == "flydsl_small_m"
            assert selected["kernelName"] == name
            a = torch.randn((m, k), dtype=torch.bfloat16, device="cuda")
            b = torch.randn((n, k), dtype=torch.bfloat16, device="cuda")
            output = tuned_gemm.gemm_a16w16(a, b)
            torch.cuda.synchronize()
            torch.testing.assert_close(output, reference(a, b), atol=ATOL, rtol=RTOL)
    finally:
        AITER_CONFIGS.get_config_file.cache_clear()
        tuned_gemm.get_GEMM_A16W16_config_.cache_clear()
        tuned_gemm.get_GEMM_A16W16_config.cache_clear()


def test_small_m_runtime_reloads_from_disk_cache(tmp_path, monkeypatch):
    import aiter.tuned_gemm as tuned_gemm
    from aiter.ops.flydsl.gemm_kernels import _compile_flydsl_hgemm

    m, n, k = 5, 128, 128
    name = _csv_kernel_name(m, n, k)
    config = {"kernelName": name}
    a = torch.randn((m, k), dtype=torch.bfloat16, device="cuda")
    b = torch.randn((n, k), dtype=torch.bfloat16, device="cuda")
    cache_dir = tmp_path / "cache"
    try:
        with monkeypatch.context() as isolated:
            isolated.setenv("FLYDSL_RUNTIME_CACHE_DIR", str(cache_dir))
            isolated.setenv("FLYDSL_RUNTIME_ENABLE_CACHE", "1")
            isolated.delenv("FLYDSL_RUNTIME_RUN_ONLY", raising=False)
            _compile_flydsl_hgemm.cache_clear()
            compile_small_m_hgemm_kernel.cache_clear()
            first = tuned_gemm.flydsl_small_m_gemm(
                a, b, 0, otype=torch.bfloat16, config=config
            )
            torch.cuda.synchronize()
            assert any(path.is_file() for path in cache_dir.rglob("*"))

            isolated.setenv("FLYDSL_RUNTIME_ENABLE_CACHE", "0")
            isolated.setenv("FLYDSL_RUNTIME_RUN_ONLY", "1")
            _compile_flydsl_hgemm.cache_clear()
            compile_small_m_hgemm_kernel.cache_clear()
            second = tuned_gemm.flydsl_small_m_gemm(
                a, b, 0, otype=torch.bfloat16, config=config
            )
            torch.cuda.synchronize()
            assert torch.equal(first, second)
    finally:
        _compile_flydsl_hgemm.cache_clear()
        compile_small_m_hgemm_kernel.cache_clear()


# ---------------------------------------------------------------------------
# Explicit rejection instead of silent truncation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("k", [257, 769])
def test_indivisible_k_is_rejected_not_truncated(k):
    with pytest.raises(ValueError, match="no legal small-M K schedule"):
        compile_small_m_hgemm_kernel(
            "bf16", 128, k, TILE_N=128, TILE_K=64, BLOCK_N_WARPS=2
        )


@pytest.mark.parametrize("tile_n,block_n_warps", [(32, 1), (128, 2)])
def test_indivisible_n_is_rejected_not_truncated(tile_n, block_n_warps):
    with pytest.raises(ValueError, match="not a positive multiple of the block-N"):
        compile_small_m_hgemm_kernel(
            "bf16",
            6288,
            768,
            TILE_N=tile_n,
            TILE_K=64,
            BLOCK_N_WARPS=block_n_warps,
        )


@requires_gfx942
def test_configs_over_the_gfx942_lds_budget_are_rejected():
    with pytest.raises(ValueError, match="LDS"):
        compile_small_m_hgemm_kernel(
            "bf16",
            7168,
            7168,
            TILE_N=256,
            TILE_K=64,
            BLOCK_N_WARPS=4,
            B_TO_LDS=True,
        )


def test_unknown_staging_is_rejected():
    with pytest.raises(ValueError, match="LDS_STAGING"):
        compile_small_m_hgemm_kernel(
            "bf16", 128, 128, TILE_N=128, BLOCK_N_WARPS=2, LDS_STAGING="pretend"
        )


# ---------------------------------------------------------------------------
# gfx942 correctness
# ---------------------------------------------------------------------------


@requires_gfx942
@pytest.mark.parametrize("m", list(range(1, SMALL_M_KERNEL_MAX)))
@pytest.mark.parametrize("lds_staging", LDS_STAGING_OPTIONS)
def test_gfx942_all_supported_m_rows(m, lds_staging):
    out, expected = run_small_m(
        m,
        256,
        128,
        tile_n=128,
        block_n_warps=2,
        b_to_lds=True,
        lds_staging=lds_staging,
    )
    torch.testing.assert_close(out, expected, atol=ATOL, rtol=RTOL)


@requires_gfx942
@pytest.mark.parametrize("k", [128, 768, 7168])
@pytest.mark.parametrize("b_to_lds", [False, True])
@pytest.mark.parametrize("lds_staging", LDS_STAGING_OPTIONS)
def test_gfx942_supported_k_values(k, b_to_lds, lds_staging):
    out, expected = run_small_m(
        8,
        1536,
        k,
        tile_n=128,
        block_n_warps=2,
        b_to_lds=b_to_lds,
        lds_staging=lds_staging,
    )
    torch.testing.assert_close(out, expected, atol=ATOL, rtol=RTOL)


@requires_gfx942
@pytest.mark.parametrize("m", [7, 8])
@pytest.mark.parametrize("lds_staging", LDS_STAGING_OPTIONS)
def test_gfx942_16_wide_tile_covers_a_16_aligned_output_width(m, lds_staging):
    # 6288 % 32 == 16, so only the 16-wide tile can express it exactly.
    out, expected = run_small_m(
        m,
        6288,
        768,
        tile_n=16,
        block_n_warps=1,
        b_to_lds=True,
        lds_staging=lds_staging,
    )
    torch.testing.assert_close(out, expected, atol=ATOL, rtol=RTOL)
    assert not out.isnan().any()


@pytest.mark.parametrize("tile_k", [96, 160, 192, 224])
def test_swizzle_unsafe_tile_k_is_rejected(tile_k):
    # `TILE_K * 2 / 16` is not a power of two for these widths, so the XOR-16
    # LDS swizzle would map columns outside the staged row.
    assert not small_m_tile_k_is_swizzle_safe(tile_k)
    with pytest.raises(ValueError, match="LDS swizzle"):
        compile_small_m_hgemm_kernel(
            "bf16", 512, 768, TILE_N=64, TILE_K=tile_k, BLOCK_N_WARPS=1
        )


@requires_gfx942
@pytest.mark.parametrize("lds_staging", LDS_STAGING_OPTIONS)
@pytest.mark.parametrize(
    "tile_k,b_to_lds",
    # TILE_K=256 with B staged in LDS does not fit the 64 KiB gfx942 budget.
    [(32, True), (64, True), (128, True), (256, False)],
)
def test_gfx942_tile_k_variants(tile_k, b_to_lds, lds_staging):
    out, expected = run_small_m(
        8,
        512,
        768,
        tile_n=64,
        tile_k=tile_k,
        block_n_warps=1,
        b_to_lds=b_to_lds,
        lds_staging=lds_staging,
    )
    torch.testing.assert_close(out, expected, atol=ATOL, rtol=RTOL)


@requires_gfx942
@pytest.mark.parametrize("lds_staging", LDS_STAGING_OPTIONS)
def test_gfx942_wide_n_repeat_path(lds_staging):
    out, expected = run_small_m(
        8,
        2048,
        768,
        tile_n=64,
        block_n_warps=1,
        b_to_lds=False,
        lds_staging=lds_staging,
        n_tile_repeat=2,
    )
    torch.testing.assert_close(out, expected, atol=ATOL, rtol=RTOL)


@requires_gfx942
@pytest.mark.parametrize("lds_staging", LDS_STAGING_OPTIONS)
def test_gfx942_persistent_n_path(lds_staging):
    out, expected = run_small_m(
        8,
        2048,
        768,
        tile_n=128,
        block_n_warps=2,
        b_to_lds=True,
        lds_staging=lds_staging,
        persistent_n_tiles=2,
    )
    torch.testing.assert_close(out, expected, atol=ATOL, rtol=RTOL)


@requires_gfx942
@pytest.mark.parametrize("lds_staging", LDS_STAGING_OPTIONS)
def test_gfx942_split_k_accumulation_is_bounded(lds_staging):
    # Split-K accumulates through BF16 device atomics, so it cannot meet the
    # single-pass tolerance; it must still be race-free and bounded.
    out, expected = run_small_m(
        8,
        2048,
        768,
        tile_n=128,
        block_n_warps=2,
        b_to_lds=True,
        lds_staging=lds_staging,
        split_k=4,
    )
    assert not out.isnan().any()
    assert (out.float() - expected.float()).abs().max().item() < 2.0


@requires_gfx942
@pytest.mark.parametrize("b_to_lds", [False, True])
def test_gfx942_staging_variants_agree(b_to_lds):
    direct, expected = run_small_m(
        8,
        1024,
        768,
        tile_n=128,
        block_n_warps=2,
        b_to_lds=b_to_lds,
        lds_staging=LDS_STAGING_DIRECT,
    )
    staged, _ = run_small_m(
        8,
        1024,
        768,
        tile_n=128,
        block_n_warps=2,
        b_to_lds=b_to_lds,
        lds_staging=LDS_STAGING_VGPR,
    )
    torch.testing.assert_close(direct, expected, atol=ATOL, rtol=RTOL)
    # Both variants feed the same MFMA sequence, so they must agree exactly.
    torch.testing.assert_close(direct, staged, atol=0.0, rtol=0.0)


# ---------------------------------------------------------------------------
# Generated code inspection
# ---------------------------------------------------------------------------


def _compile_and_read_isa(lds_staging: str) -> str:
    directory = tempfile.mkdtemp(prefix="small_m_isa_")
    previous = {
        key: os.environ.get(key)
        for key in ("FLYDSL_DUMP_IR", "FLYDSL_DUMP_DIR", "FLYDSL_RUNTIME_ENABLE_CACHE")
    }
    os.environ["FLYDSL_DUMP_IR"] = "1"
    os.environ["FLYDSL_DUMP_DIR"] = directory
    os.environ["FLYDSL_RUNTIME_ENABLE_CACHE"] = "0"
    try:
        run_small_m(
            8,
            1024,
            768,
            tile_n=128,
            block_n_warps=2,
            b_to_lds=True,
            lds_staging=lds_staging,
        )
        dumps = sorted(
            path
            for path in os.listdir(directory)
            if os.path.isdir(os.path.join(directory, path))
        )
        for name in dumps:
            candidate = os.path.join(directory, name)
            for entry in os.listdir(candidate):
                if entry.endswith("final_isa.s"):
                    with open(os.path.join(candidate, entry)) as handle:
                        return handle.read()
        pytest.skip("FlyDSL produced no ISA dump in this environment")
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        shutil.rmtree(directory, ignore_errors=True)


@requires_gfx942
def test_gfx942_isa_uses_the_expected_staging_instructions():
    direct = _compile_and_read_isa(LDS_STAGING_DIRECT)
    staged = _compile_and_read_isa(LDS_STAGING_VGPR)

    # Both variants must use the native gfx942 atom and no gfx950-only atom.
    for isa in (direct, staged):
        assert "v_mfma_f32_16x16x16_bf16" in isa
        assert "v_mfma_f32_16x16x32_bf16" not in isa
        # gfx942 has no wide LDS DMA; a `buffer_load_dwordx4 ... lds` would
        # not even assemble for this target.
        assert not any(
            "buffer_load_dwordx4" in line and line.rstrip().endswith("lds")
            for line in isa.splitlines()
        )
        assert "scratch_store" not in isa and "scratch_load" not in isa

    assert direct.count("buffer_load_dword ") > 0
    assert "ds_write_b128" not in direct

    assert staged.count("buffer_load_dwordx4") > 0
    assert staged.count("ds_write_b128") > 0
    assert "buffer_load_dword " not in staged


@requires_gfx942
def test_gfx942_static_lds_stays_within_the_hardware_budget():
    for lds_staging in LDS_STAGING_OPTIONS:
        isa = _compile_and_read_isa(lds_staging)
        sizes = [
            int(line.split()[-1])
            for line in isa.splitlines()
            if ".amdhsa_group_segment_fixed_size" in line
        ]
        assert sizes, "no LDS size found in the generated ISA"
        assert max(sizes) <= small_m_max_lds_bytes("gfx942")
