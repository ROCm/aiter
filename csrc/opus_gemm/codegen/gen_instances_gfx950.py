# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
"""gfx950 codegen -- emit launchers for gfx950-targeted kid families.

Free functions taking the parent opus_gemm_codegen instance as first arg.
Self-registers each emit into codegen.common.EMIT_REGISTRY at import time.
"""

import os
from pathlib import Path

from opus_gemm_common import OpusGemmInstance

from codegen.common import (
    WARP_SIZE,
    register_arch_map,
    register_emit,
)
from codegen.template_env import render as _render

# ---------------- gfx950 arch-override maps ----------------

PIPELINE_HEADER_MAP = {
    "a8w8_scale": "gfx950/opus_gemm_pipeline_a8w8_scale_gfx950.cuh",
    "a8w8": "gfx950/opus_gemm_pipeline_a8w8_noscale_gfx950.cuh",
    "a16w16": "gfx950/opus_gemm_pipeline_a16w16_gfx950.cuh",
    "a16w16_flatmm": "gfx950/opus_gemm_pipeline_a16w16_flatmm_gfx950.cuh",
    "a16w16_flatmm_splitk": "gfx950/opus_gemm_pipeline_a16w16_flatmm_splitk_gfx950.cuh",
    "a16w16_persistent": "gfx950/opus_gemm_pipeline_a16w16_persistent_gfx950.cuh",
    "a16w16_mono_tile": "gfx950/opus_gemm_pipeline_a16w16_mono_tile_gfx950.cuh",
}

# 4g_safe sibling pipelines: only defined for the a16w16-family tags that have
# matching *_4g_safe_gfx950.cuh files. Kids with is_4g_safe=True route to these
# headers/kernel symbols instead of the legacy maps above.
PIPELINE_HEADER_MAP_4G_SAFE = {
    "a16w16": "gfx950/opus_gemm_pipeline_a16w16_4g_safe_gfx950.cuh",
    "a16w16_persistent": "gfx950/opus_gemm_pipeline_a16w16_persistent_4g_safe_gfx950.cuh",
    "a16w16_mono_tile": "gfx950/opus_gemm_pipeline_a16w16_mono_tile_4g_safe_gfx950.cuh",
}

TRAITS_HEADER_MAP = {
    "a8w8_scale": "gfx950/opus_gemm_traits_a8w8_scale_gfx950.cuh",
    "a8w8": "gfx950/opus_gemm_traits_a8w8_noscale_gfx950.cuh",
    "a16w16": "gfx950/opus_gemm_traits_a16w16_gfx950.cuh",
    "a16w16_flatmm": "gfx950/opus_gemm_traits_a16w16_gfx950.cuh",
    "a16w16_flatmm_splitk": "gfx950/opus_gemm_traits_a16w16_gfx950.cuh",
    "a16w16_persistent": "gfx950/opus_gemm_traits_a16w16_gfx950.cuh",
    "a16w16_mono_tile": "gfx950/opus_gemm_traits_a16w16_gfx950.cuh",
}

KERNEL_FUNC_MAP = {
    "a8w8_scale": "gemm_a8w8_scale_kernel",
    "a8w8": "gemm_a8w8_noscale_kernel",
    "a16w16": "gemm_a16w16_kernel",
    "a16w16_flatmm": "gemm_a16w16_flatmm_kernel",
    "a16w16_flatmm_splitk": "gemm_a16w16_flatmm_splitk_kernel",
    "a16w16_persistent": "gemm_a16w16_persistent_kernel",
    "a16w16_mono_tile": "gemm_a16w16_mono_tile_kernel_gfx950",
}

KERNEL_FUNC_MAP_4G_SAFE = {
    "a16w16": "gemm_a16w16_4g_safe_kernel",
    "a16w16_persistent": "gemm_a16w16_persistent_4g_safe_kernel",
    "a16w16_mono_tile": "gemm_a16w16_mono_tile_4g_safe_kernel_gfx950",
}

TRAITS_NAME_MAP = {
    "a8w8_scale": "opus_gemm_a8w8_scale_traits_gfx950",
    "a8w8": "opus_gemm_a8w8_noscale_traits_gfx950",
    "a16w16": "opus_gemm_a16w16_traits_gfx950",
    "a16w16_flatmm": "opus_gemm_a16w16_flatmm_traits_gfx950",
    "a16w16_flatmm_splitk": "opus_flatmm_splitk_traits_gfx950",
    "a16w16_persistent": "opus_gemm_a16w16_persistent_traits_gfx950",
    "a16w16_mono_tile": "opus_gemm_a16w16_mono_tile_traits_gfx950",
}

KARGS_NAME_MAP = {
    "a8w8_scale": "opus_gemm_scale_kargs_gfx950",
    "a8w8": "opus_gemm_noscale_kargs_gfx950",
    "a16w16": "opus_gemm_noscale_kargs_gfx950",
    "a16w16_flatmm": "opus_gemm_flatmm_kargs_gfx950",
    "a16w16_flatmm_splitk": "opus_gemm_flatmm_splitk_kargs_gfx950",
    "a16w16_persistent": "opus_gemm_persistent_kargs_gfx950",
    "a16w16_mono_tile": "opus_gemm_mono_tile_kargs_gfx950",
}

register_arch_map("gfx950", "pipeline_header", PIPELINE_HEADER_MAP)
register_arch_map("gfx950", "traits_header", TRAITS_HEADER_MAP)
register_arch_map("gfx950", "kernel_func", KERNEL_FUNC_MAP)
register_arch_map("gfx950", "traits_name", TRAITS_NAME_MAP)
register_arch_map("gfx950", "kargs_name", KARGS_NAME_MAP)


# ---------------- gfx950 validators ----------------

VALID_BF16_MFMA = {(16, 16, 32), (32, 32, 16)}
# Flatmm pipeline currently only supports W_M < 32 (ra layout relies on
# LOAD_GROUP_M_LANE == 1). W_M == 32 (LGML == 4) path not rewritten.
VALID_FLATMM_MFMA = {(16, 16, 32)}
VALID_FLATMM_SPLITK_MFMA = {(16, 16, 32)}
VALID_PERSISTENT_MFMA = {(16, 16, 32)}
VALID_MONO_TILE_MFMA = {(16, 16, 32)}


def _validate_a16w16(k: OpusGemmInstance):
    """Validate a gfx950 split-barrier a16w16 instance at codegen time."""
    errors = []
    sizeof_da = 2  # bf16

    T_K = 1
    HALF_B_M = k.B_M // 2
    HALF_B_N = k.B_N // 2
    num_waves = k.T_M * k.T_N * T_K
    smem_linear_wave = WARP_SIZE * 16 // sizeof_da  # 512

    if k.BLOCK_SIZE > 512:
        errors.append(f"BLOCK_SIZE={k.BLOCK_SIZE} exceeds 512")

    if k.T_M != 2:
        errors.append(f"T_M={k.T_M} must be 2")

    if k.BLOCK_SIZE != num_waves * WARP_SIZE:
        errors.append(
            f"BLOCK_SIZE={k.BLOCK_SIZE} != "
            f"{k.T_M}*{k.T_N}*{T_K}*{WARP_SIZE}={num_waves * WARP_SIZE}"
        )

    if k.T_N % k.T_M != 0:
        errors.append(f"T_N={k.T_N} not divisible by T_M={k.T_M}")

    if (k.W_M, k.W_N, k.W_K) not in VALID_BF16_MFMA:
        errors.append(f"WAVE=({k.W_M},{k.W_N},{k.W_K}) not in {VALID_BF16_MFMA}")
    if WARP_SIZE % k.W_M != 0:
        errors.append(f"WARP_SIZE not divisible by W_M={k.W_M}")
    if WARP_SIZE % k.W_N != 0:
        errors.append(f"WARP_SIZE not divisible by W_N={k.W_N}")
    if k.W_M % k.T_N != 0:
        errors.append(f"W_M={k.W_M} not divisible by T_N={k.T_N}")
    if k.W_N % k.T_N != 0:
        errors.append(f"W_N={k.W_N} not divisible by T_N={k.T_N}")

    expected_vec = 16 // sizeof_da
    if k.VEC_A != expected_vec:
        errors.append(f"VEC_A={k.VEC_A} must be {expected_vec}")

    if k.B_M % 2 != 0 or k.B_N % 2 != 0:
        errors.append(f"B_M={k.B_M}, B_N={k.B_N} must be even")
    if HALF_B_M % (k.W_M * k.T_M) != 0:
        errors.append(f"HALF_B_M={HALF_B_M} not div by W_M*T_M={k.W_M * k.T_M}")
    if HALF_B_N % (k.W_N * k.T_N) != 0:
        errors.append(f"HALF_B_N={HALF_B_N} not div by W_N*T_N={k.W_N * k.T_N}")
    if k.B_K % k.W_K != 0:
        errors.append(f"B_K={k.B_K} not div by W_K={k.W_K}")

    E_M = HALF_B_M // (k.W_M * k.T_M) if (k.W_M * k.T_M) else 0
    E_N = HALF_B_N // (k.W_N * k.T_N) if (k.W_N * k.T_N) else 0
    E_K = k.B_K // k.W_K if k.W_K else 0

    if smem_linear_wave % k.B_K != 0:
        errors.append(f"smem_linear_wave={smem_linear_wave} not div by B_K={k.B_K}")
    else:
        smem_sub = smem_linear_wave // k.B_K
        if HALF_B_M % smem_sub != 0:
            errors.append(f"HALF_B_M={HALF_B_M} not div by smem_sub={smem_sub}")
        if HALF_B_N % smem_sub != 0:
            errors.append(f"HALF_B_N={HALF_B_N} not div by smem_sub={smem_sub}")

    for name, num, den in [
        ("a_buffer_load_insts", HALF_B_M * k.B_K, k.BLOCK_SIZE * k.VEC_A),
        ("b_buffer_load_insts", HALF_B_N * k.B_K, k.BLOCK_SIZE * k.VEC_B),
        ("a_ds_read_insts", E_M * E_K * k.W_M * k.W_K, WARP_SIZE * k.VEC_A),
        ("b_ds_read_insts", E_N * E_K * k.W_N * k.W_K, WARP_SIZE * k.VEC_B),
    ]:
        if den == 0 or num % den != 0 or num // den < 1:
            errors.append(f"{name}={num}/{den} invalid")

    for tag, ww, vec in [
        ("ra", k.W_M * k.W_K, k.VEC_A),
        ("rb", k.W_N * k.W_K, k.VEC_B),
    ]:
        denom = WARP_SIZE * vec
        if ww < denom or ww % denom != 0:
            errors.append(f"{tag}: W*W_K={ww} must be >= and div by {denom}")

    if k.VEC_B and k.B_K % k.VEC_B == 0:
        threads_k_b = k.B_K // k.VEC_B
        if k.BLOCK_SIZE % threads_k_b == 0:
            thr_n = k.BLOCK_SIZE // threads_k_b
            if HALF_B_N % thr_n != 0:
                errors.append(f"gb: HALF_B_N={HALF_B_N} not div by {thr_n}")

    if smem_linear_wave % k.B_K == 0:
        smem_sub = smem_linear_wave // k.B_K
        if smem_sub and HALF_B_N % smem_sub == 0:
            smem_n_rep = HALF_B_N // smem_sub
            if smem_n_rep % num_waves != 0:
                errors.append(f"sb: smem_n_rep={smem_n_rep} not div by {num_waves}")

    for tag, vec in [("ga", k.VEC_A), ("gb", k.VEC_B)]:
        if vec and k.B_K // vec > WARP_SIZE:
            errors.append(f"{tag}: B_K/VEC={k.B_K // vec} > WARP_SIZE")

    agpr_per_mfma = (k.W_M * k.W_N) // WARP_SIZE
    total_agprs = 4 * E_M * E_N * agpr_per_mfma
    if total_agprs >= 256:
        errors.append(f"AGPR={total_agprs} must be < 256")

    if smem_linear_wave % k.B_K == 0:
        smem_sub = smem_linear_wave // k.B_K
        smem_m_rep = (
            HALF_B_M // smem_sub if smem_sub and HALF_B_M % smem_sub == 0 else 0
        )
        smem_n_rep = (
            HALF_B_N // smem_sub if smem_sub and HALF_B_N % smem_sub == 0 else 0
        )
        smem_padding = 2 * 16 // sizeof_da
        smem_a = smem_m_rep * (smem_linear_wave + smem_padding) * sizeof_da
        smem_b = smem_n_rep * (smem_linear_wave + smem_padding) * sizeof_da
        total_lds = (smem_a + smem_b) * 4
        if total_lds > 160 * 1024:
            errors.append(f"LDS={total_lds // 1024}KiB exceeds 160KiB")

    vgpr_ops = 4 * E_K * (E_M + 2 * E_N)
    vgpr_est = vgpr_ops + 80
    if vgpr_est > 256:
        errors.append(f"VGPR_est={vgpr_est} exceeds 256")
    if vgpr_est + total_agprs > 512:
        errors.append(f"VGPR+AGPR={vgpr_est + total_agprs} exceeds 512")

    required_bk = k.T_N * k.W_K // 2
    if k.B_K != required_bk:
        errors.append(
            f"B_K={k.B_K} must equal T_N*W_K/2={required_bk} "
            f"(ra/rb layout E_K/T_N coupling)"
        )

    if errors:
        msg = f"Invalid a16w16 instance '{k.name}':\n" + "\n".join(
            f"  - {e}" for e in errors
        )
        raise ValueError(msg)

    return {
        "E_M": E_M,
        "E_N": E_N,
        "E_K": E_K,
        "agprs": total_agprs,
        "vgpr_est": vgpr_est,
        "lds_bytes": total_lds if smem_linear_wave % k.B_K == 0 else -1,
        "min_k": 2 * k.B_K,
    }


def _validate_a16w16_flatmm(k: OpusGemmInstance):
    """gfx950 a16w16_flatmm validator. See historical opus_gemm_codegen._validate_a16w16_flatmm."""
    errors = []
    sizeof_da = 2

    if k.BLOCK_SIZE != 256:
        errors.append(f"BLOCK_SIZE={k.BLOCK_SIZE} must be 256 (4-wave warp-spec)")
    if k.T_M != 2:
        errors.append(f"T_M={k.T_M} must be 2")
    if k.T_N != 1:
        errors.append(f"T_N={k.T_N} must be 1")

    if (k.W_M, k.W_N, k.W_K) not in VALID_FLATMM_MFMA:
        errors.append(
            f"WAVE=({k.W_M},{k.W_N},{k.W_K}) not in {VALID_FLATMM_MFMA} "
            f"(flatmm ra layout requires W_M<32)"
        )
    if k.W_M >= 32:
        errors.append(f"W_M={k.W_M}: flatmm LGML=4 path not implemented")

    expected_vec = 16 // sizeof_da
    if k.VEC_A != expected_vec or k.VEC_B != expected_vec:
        errors.append(f"VEC_A={k.VEC_A}, VEC_B={k.VEC_B} must be {expected_vec}")
    if k.VEC_C != 4:
        errors.append(f"VEC_C={k.VEC_C} must be 4")

    LOAD_GROUP_M = 64 if k.W_M >= 32 else 32
    LOAD_GROUP_N = 64 if k.W_N >= 32 else 32
    LOAD_GROUP_K = k.W_K * 2
    if k.B_M % LOAD_GROUP_M != 0:
        errors.append(f"B_M={k.B_M} not div by LOAD_GROUP_M={LOAD_GROUP_M}")
    if k.B_N % LOAD_GROUP_N != 0:
        errors.append(f"B_N={k.B_N} not div by LOAD_GROUP_N={LOAD_GROUP_N}")
    if k.B_K % LOAD_GROUP_K != 0:
        errors.append(f"B_K={k.B_K} not div by LOAD_GROUP_K={LOAD_GROUP_K}")

    num_load_groups_per_bm = k.B_M // LOAD_GROUP_M
    num_load_groups_per_bn = k.B_N // LOAD_GROUP_N
    num_load_groups_per_bk = k.B_K // LOAD_GROUP_K

    smem_linear_wave = WARP_SIZE * 16 // sizeof_da
    smem_sub = smem_linear_wave // LOAD_GROUP_K
    slots = LOAD_GROUP_M // smem_sub
    smem_padding = 16 // sizeof_da if k.W_M >= 32 else 2 * 16 // sizeof_da
    smem_per_group_load_size = slots * (smem_linear_wave + smem_padding) * sizeof_da

    if k.WG_PER_CU not in (1, 2):
        errors.append(f"WG_PER_CU={k.WG_PER_CU} must be 1 or 2")

    lds_total = 163840
    max_lds_per_wg = lds_total // max(k.WG_PER_CU, 1)
    per_block_iter = (
        (num_load_groups_per_bm + num_load_groups_per_bn)
        * num_load_groups_per_bk
        * smem_per_group_load_size
    )
    pfk = max_lds_per_wg // per_block_iter if per_block_iter > 0 else 0
    if pfk < 3:
        errors.append(
            f"prefetch_k_iter={pfk} < 3 "
            f"(LDS budget {max_lds_per_wg} / per-iter {per_block_iter})"
        )

    min_k = pfk * k.B_K
    lds_footprint = pfk * per_block_iter

    if errors:
        msg = f"Invalid a16w16_flatmm instance '{k.name}':\n" + "\n".join(
            f"  - {e}" for e in errors
        )
        raise ValueError(msg)

    return {
        "pfk": pfk,
        "min_k": min_k,
        "lds_bytes": lds_footprint,
        "slots": slots,
        "groups_bm": num_load_groups_per_bm,
        "groups_bn": num_load_groups_per_bn,
        "groups_bk": num_load_groups_per_bk,
    }


def _validate_a16w16_flatmm_splitk(k: OpusGemmInstance):
    """gfx950 a16w16_flatmm_splitk validator."""
    errors = []
    sizeof_da = 2

    if k.BLOCK_SIZE != 256:
        errors.append(f"BLOCK_SIZE={k.BLOCK_SIZE} must be 256 (4-wave warp-spec)")
    if k.T_M != 2:
        errors.append(f"T_M={k.T_M} must be 2")
    if k.T_N != 1:
        errors.append(f"T_N={k.T_N} must be 1")

    if (k.W_M, k.W_N, k.W_K) not in VALID_FLATMM_SPLITK_MFMA:
        errors.append(
            f"WAVE=({k.W_M},{k.W_N},{k.W_K}) not in {VALID_FLATMM_SPLITK_MFMA} "
            f"(flatmm_splitk ra layout requires W_M<32)"
        )
    if k.W_M >= 32:
        errors.append(f"W_M={k.W_M}: flatmm_splitk LGML=4 path not implemented")

    expected_vec = 16 // sizeof_da
    if k.VEC_A != expected_vec or k.VEC_B != expected_vec:
        errors.append(f"VEC_A={k.VEC_A}, VEC_B={k.VEC_B} must be {expected_vec}")
    if k.VEC_C != 4:
        errors.append(f"VEC_C={k.VEC_C} must be 4")

    LOAD_GROUP_M = 64 if k.W_M >= 32 else 32
    LOAD_GROUP_N = 64 if k.W_N >= 32 else 32
    LOAD_GROUP_K = k.W_K * 2
    if k.B_M % LOAD_GROUP_M != 0:
        errors.append(f"B_M={k.B_M} not div by LOAD_GROUP_M={LOAD_GROUP_M}")
    if k.B_N % LOAD_GROUP_N != 0:
        errors.append(f"B_N={k.B_N} not div by LOAD_GROUP_N={LOAD_GROUP_N}")
    if k.B_K % LOAD_GROUP_K != 0:
        errors.append(f"B_K={k.B_K} not div by LOAD_GROUP_K={LOAD_GROUP_K}")

    num_load_groups_per_bm = k.B_M // LOAD_GROUP_M
    num_load_groups_per_bn = k.B_N // LOAD_GROUP_N
    num_load_groups_per_bk = k.B_K // LOAD_GROUP_K

    smem_linear_wave = WARP_SIZE * 16 // sizeof_da
    smem_sub = smem_linear_wave // LOAD_GROUP_K
    slots = LOAD_GROUP_M // smem_sub
    smem_padding = 16 // sizeof_da if k.W_M >= 32 else 2 * 16 // sizeof_da
    smem_per_group_load_size = slots * (smem_linear_wave + smem_padding) * sizeof_da

    if k.WG_PER_CU not in (1, 2):
        errors.append(f"WG_PER_CU={k.WG_PER_CU} must be 1 or 2")

    lds_total = 163840
    max_lds_per_wg = lds_total // max(k.WG_PER_CU, 1)
    per_block_iter = (
        (num_load_groups_per_bm + num_load_groups_per_bn)
        * num_load_groups_per_bk
        * smem_per_group_load_size
    )
    pfk = max_lds_per_wg // per_block_iter if per_block_iter > 0 else 0
    if pfk < 3:
        errors.append(
            f"prefetch_k_iter={pfk} < 3 "
            f"(LDS budget {max_lds_per_wg} / per-iter {per_block_iter})"
        )

    com_rep_m = k.B_M // (k.W_M * 2)
    com_rep_n = k.B_N // k.W_N
    if k.WG_PER_CU == 1 and com_rep_m * com_rep_n > 16:
        errors.append(
            f"WG_PER_CU=1 requires COM_REP_M*COM_REP_N<=16 "
            f"(got {com_rep_m * com_rep_n}={com_rep_m}*{com_rep_n}); "
            f"larger WG=1 tiles spill VGPR to scratch, ~1000x slower"
        )

    min_k = pfk * k.B_K
    lds_footprint = pfk * per_block_iter

    if errors:
        msg = f"Invalid a16w16_flatmm_splitk instance '{k.name}':\n" + "\n".join(
            f"  - {e}" for e in errors
        )
        raise ValueError(msg)

    return {
        "pfk": pfk,
        "min_k": min_k,
        "lds_bytes": lds_footprint,
        "slots": slots,
        "com_rep_m": com_rep_m,
        "com_rep_n": com_rep_n,
    }


def _validate_a16w16_persistent(k: OpusGemmInstance):
    """gfx950 a16w16_persistent validator. Delegates to the shared split-barrier
    validator (which itself is arch-aware on ra/rb stride checks).
    """
    if (k.W_M, k.W_N, k.W_K) not in VALID_PERSISTENT_MFMA:
        raise ValueError(
            f"Invalid a16w16_persistent instance '{k.name}':\n"
            f"  - WAVE=({k.W_M},{k.W_N},{k.W_K}) not in {VALID_PERSISTENT_MFMA}"
        )
    if k.BLOCK_SIZE != 512:
        raise ValueError(
            f"Invalid a16w16_persistent instance '{k.name}':\n"
            f"  - BLOCK_SIZE={k.BLOCK_SIZE} must be 512 (mouter 8-wave WG)"
        )
    return _validate_a16w16(k)


def _validate_a16w16_mono_tile(k: OpusGemmInstance):
    """gfx950 a16w16_mono_tile validator."""
    errors = []
    sizeof_da = 2

    if k.BLOCK_SIZE != 512:
        errors.append(f"BLOCK_SIZE={k.BLOCK_SIZE} must be 512 (mono-tile 8-wave WG)")
    if k.T_M != 2:
        errors.append(f"T_M={k.T_M} must be 2 (mono-tile locked)")
    if k.T_N != 4:
        errors.append(f"T_N={k.T_N} must be 4 (mono-tile locked)")
    if (k.W_M, k.W_N, k.W_K) not in VALID_MONO_TILE_MFMA:
        errors.append(f"WAVE=({k.W_M},{k.W_N},{k.W_K}) not in {VALID_MONO_TILE_MFMA}")

    expected_vec = 16 // sizeof_da
    if k.VEC_A != expected_vec or k.VEC_B != expected_vec or k.VEC_C != expected_vec:
        errors.append(f"VEC=({k.VEC_A},{k.VEC_B},{k.VEC_C}) must all be {expected_vec}")

    if k.B_M > 192:
        errors.append(f"B_M={k.B_M} exceeds mono-tile cap of 192")

    if k.has_oob:
        errors.append("mono-tile is intrinsically non-OOB; has_oob must be False")

    if k.B_M % (k.W_M * k.T_M) != 0:
        errors.append(f"B_M={k.B_M} not div by W_M*T_M={k.W_M * k.T_M}")
    if k.B_N % (k.W_N * k.T_N) != 0:
        errors.append(f"B_N={k.B_N} not div by W_N*T_N={k.W_N * k.T_N}")
    if k.B_K % (k.W_K * 1) != 0:
        errors.append(f"B_K={k.B_K} not div by W_K*T_K={k.W_K}")

    E_M = k.B_M // (k.W_M * k.T_M) if (k.W_M * k.T_M) else 0
    E_N = k.B_N // (k.W_N * k.T_N) if (k.W_N * k.T_N) else 0
    E_K = k.B_K // k.W_K if k.W_K else 0

    if k.T_M and (E_N * k.T_M) % k.T_N != 0:
        errors.append(
            f"E_N={E_N} not div by T_N/T_M={k.T_N // k.T_M} "
            f"(mono-tile rb layout grouping; needs B_N % 128 == 0)"
        )

    smem_linear_wave = WARP_SIZE * 16 // sizeof_da
    if k.B_K and smem_linear_wave % k.B_K != 0:
        errors.append(
            f"B_K={k.B_K} does not divide smem_linear_wave={smem_linear_wave}"
        )
        total_lds = -1
    elif k.B_K:
        smem_sub = smem_linear_wave // k.B_K
        num_waves = k.BLOCK_SIZE // WARP_SIZE
        if k.B_M % smem_sub != 0:
            errors.append(f"B_M={k.B_M} not div by smem_sub={smem_sub}")
        if k.B_N % smem_sub != 0:
            errors.append(f"B_N={k.B_N} not div by smem_sub={smem_sub}")
        smem_m_rep = k.B_M // smem_sub if smem_sub else 0
        smem_n_rep = k.B_N // smem_sub if smem_sub else 0
        if smem_m_rep < num_waves or (smem_m_rep % num_waves) != 0:
            errors.append(
                f"smem_m_rep={smem_m_rep} must be >= {num_waves} "
                f"and divisible by {num_waves}"
            )
        if smem_n_rep < num_waves or (smem_n_rep % num_waves) != 0:
            errors.append(
                f"smem_n_rep={smem_n_rep} must be >= {num_waves} "
                f"and divisible by {num_waves}"
            )
        if k.T_N and (k.W_M % k.T_N) != 0:
            errors.append(f"W_M={k.W_M} not div by T_N={k.T_N} (mono-tile ra layout)")
        else:
            ratio = k.W_M // k.T_N
            if ratio and smem_sub % ratio != 0:
                errors.append(
                    f"smem_sub={smem_sub} not div by W_M/T_N={ratio} (ra layout)"
                )
            else:
                smem_sub_e_m = smem_sub // ratio if ratio else 0
                if smem_sub_e_m == 0 or (E_M % smem_sub_e_m) != 0:
                    errors.append(
                        f"E_M={E_M} not div by smem_sub_e_m={smem_sub_e_m} "
                        f"(ra layout)"
                    )

        smem_padding = 2 * 16 // sizeof_da
        smem_a_one = smem_m_rep * (smem_linear_wave + smem_padding) * sizeof_da
        smem_b_one = smem_n_rep * (smem_linear_wave + smem_padding) * sizeof_da
        total_lds = smem_a_one * 2 + smem_b_one * 3
        if total_lds > 160 * 1024:
            errors.append(f"LDS={total_lds // 1024}KiB exceeds 160KiB")
    else:
        total_lds = -1

    if errors:
        msg = f"Invalid a16w16_mono_tile instance '{k.name}':\n" + "\n".join(
            f"  - {e}" for e in errors
        )
        raise ValueError(msg)

    return {
        "E_M": E_M,
        "E_N": E_N,
        "E_K": E_K,
        "lds_bytes": total_lds,
        "min_k": 2 * k.B_K,
    }


def gen_persistent_instance(
    cg,
    k,
    pipeline_header,
    traits_header,
    kernel_func,
    da,
    db,
    traits_name,
    kargs_name,
    kargs_template_vars,
    record_one_instantiation,
    A16W16_TUNE_HOST_EXTRA,
    **_unused,
):
    """gfx950 a16w16_persistent launcher emit. See gen_instances.opus_gemm_codegen._gen_persistent_instance."""
    kargs_explicit_param, fwd_decl_kargs_tpl, fwd_decl_kargs_fnarg = (
        kargs_template_vars(k.kernel_tag, kargs_name)
    )

    INSTANCE_IMPL = _render(
        "impl_persistent_gfx950.cuh.j2",
        traits_header=traits_header,
        pipeline_header=pipeline_header,
        fwd_decl_kargs_tpl=fwd_decl_kargs_tpl,
        fwd_decl_kargs_fnarg=fwd_decl_kargs_fnarg,
        k=k,
        traits_name=traits_name,
        da=da,
        db=db,
        kargs_name=kargs_name,
        kernel_func=kernel_func,
    )
    Path(os.path.join(cg.impl_path, f"{k.name}.cuh")).write_text(INSTANCE_IMPL)
    record_one_instantiation(cg, k, kernel_func, kargs_name, A16W16_TUNE_HOST_EXTRA)


def gen_scale_instance(
    cg,
    k,
    pipeline_header,
    traits_header,
    kernel_func,
    da,
    db,
    traits_name,
    kargs_name,
    kargs_template_vars,
    record_one_instantiation,
    A8W8_SCALE_HOST_EXTRA,
    **_unused,
):
    """gfx950 a8w8_scale launcher emit."""
    kargs_explicit_param, fwd_decl_kargs_tpl, fwd_decl_kargs_fnarg = (
        kargs_template_vars(k.kernel_tag, kargs_name)
    )
    INSTANCE_IMPL = _render(
        "impl_scale_gfx950.cuh.j2",
        traits_header=traits_header,
        pipeline_header=pipeline_header,
        fwd_decl_kargs_tpl=fwd_decl_kargs_tpl,
        fwd_decl_kargs_fnarg=fwd_decl_kargs_fnarg,
        k=k,
        traits_name=traits_name,
        da=da,
        db=db,
        kargs_name=kargs_name,
        kernel_func=kernel_func,
    )
    Path(os.path.join(cg.impl_path, f"{k.name}.cuh")).write_text(INSTANCE_IMPL)
    record_one_instantiation(cg, k, kernel_func, kargs_name, A8W8_SCALE_HOST_EXTRA)


def gen_noscale_instance_gfx950(
    cg,
    k,
    pipeline_header,
    traits_header,
    kernel_func,
    da,
    db,
    traits_name,
    kargs_name,
    kargs_template_vars,
    BIAS_HOST_VALIDATE,
    A16W16_TUNE_TAGS,
    **_unused,
):
    """gfx950 noscale launcher emit: a16w16 split-barrier (bias-aware double-traits)
    and a8w8 noscale (single traits). a8w8 falls through the else branch."""
    kargs_explicit_param, fwd_decl_kargs_tpl, fwd_decl_kargs_fnarg = (
        kargs_template_vars(k.kernel_tag, kargs_name)
    )
    is_a16w16_split_barrier = k.kernel_tag == "a16w16"
    has_tune_tags = k.kernel_tag in A16W16_TUNE_TAGS

    if is_a16w16_split_barrier:
        cachectl_extra = ""
        if hasattr(k, "cachectl_a") and k.cachectl_a >= 0:
            cachectl_extra = f",\n    {k.cachectl_a}, {k.cachectl_b}"
        INSTANCE_IMPL = _render(
            "impl_noscale_a16w16_gfx950.cuh.j2",
            traits_header=traits_header,
            pipeline_header=pipeline_header,
            fwd_decl_kargs_tpl=fwd_decl_kargs_tpl,
            fwd_decl_kargs_fnarg=fwd_decl_kargs_fnarg,
            kernel_func=kernel_func,
            k=k,
            traits_name=traits_name,
            kargs_name=kargs_name,
            has_tune_tags=has_tune_tags,
            is_a16w16=True,
            cachectl_extra=cachectl_extra,
            da=da,
            db=db,
        )
    else:
        # a8w8 noscale: single-traits, no bias.
        INSTANCE_IMPL = _render(
            "impl_noscale_a8w8_gfx950.cuh.j2",
            traits_header=traits_header,
            pipeline_header=pipeline_header,
            fwd_decl_kargs_tpl=fwd_decl_kargs_tpl,
            fwd_decl_kargs_fnarg=fwd_decl_kargs_fnarg,
            kernel_func=kernel_func,
            k=k,
            traits_name=traits_name,
            kargs_name=kargs_name,
            has_tune_tags=has_tune_tags,
            da=da,
            db=db,
        )
    Path(os.path.join(cg.impl_path, f"{k.name}.cuh")).write_text(INSTANCE_IMPL)

    if k.kernel_tag in A16W16_TUNE_TAGS:
        inst_extra_param = ",\n    std::optional<aiter_tensor_t>,\n    int"
    else:
        inst_extra_param = ""

    for CDtype in k.output_dtypes:
        cg._host_instantiations.append(
            {"kid_name": k.name, "dtype": CDtype, "host_extra_params": inst_extra_param}
        )
        if is_a16w16_split_barrier:
            # split-barrier: NoBias + Bias trait variants share the same kid/dtype TU.
            cg._device_instantiations.append(
                {
                    "kid_name": k.name,
                    "dtype": CDtype,
                    "kernel_func": kernel_func,
                    "kargs_name": kargs_name,
                    "kargs_explicit_param": "",
                    "traits_name_override": f"{k.name}_TraitsNoBias",
                    "extra_device_decls": [
                        {"traits_name_override": f"{k.name}_TraitsBias"},
                    ],
                }
            )
        else:
            cg._device_instantiations.append(
                {
                    "kid_name": k.name,
                    "dtype": CDtype,
                    "kernel_func": kernel_func,
                    "kargs_name": kargs_name,
                    "kargs_explicit_param": kargs_explicit_param,
                }
            )


def gen_mono_tile_instance(
    cg,
    k,
    pipeline_header,
    traits_header,
    kernel_func,
    da,
    db,
    traits_name,
    kargs_name,
    **_unused,
):
    """gfx950 a16w16_mono_tile launcher emit."""
    INSTANCE_IMPL = _render(
        "impl_mono_tile_gfx950.cuh.j2",
        traits_header=traits_header,
        pipeline_header=pipeline_header,
        fwd_decl_kargs_tpl="",
        fwd_decl_kargs_fnarg=kargs_name,
        kernel_func=kernel_func,
        kargs_name=kargs_name,
        k=k,
        traits_name=traits_name,
        da=da,
        db=db,
    )
    Path(os.path.join(cg.impl_path, f"{k.name}.cuh")).write_text(INSTANCE_IMPL)

    for CDtype in k.output_dtypes:
        cg._host_instantiations.append(
            {
                "kid_name": k.name,
                "dtype": CDtype,
                "host_extra_params": ",\n    std::optional<aiter_tensor_t>,\n    int",
            }
        )
        cg._device_instantiations.append(
            {
                "kid_name": k.name,
                "dtype": CDtype,
                "kernel_func": kernel_func,
                "kargs_name": kargs_name,
                "kargs_explicit_param": "",
            }
        )


def gen_flatmm_instance(
    cg,
    k,
    pipeline_header,
    traits_header,
    kernel_func,
    da,
    db,
    traits_name,
    kargs_name,
    kargs_template_vars,
    record_one_instantiation,
    A16W16_TUNE_HOST_EXTRA,
    **_unused,
):
    """gfx950 a16w16_flatmm launcher emit."""
    kargs_explicit_param, fwd_decl_kargs_tpl, fwd_decl_kargs_fnarg = (
        kargs_template_vars(k.kernel_tag, kargs_name)
    )
    INSTANCE_IMPL = _render(
        "impl_flatmm_gfx950.cuh.j2",
        traits_header=traits_header,
        pipeline_header=pipeline_header,
        fwd_decl_kargs_tpl=fwd_decl_kargs_tpl,
        fwd_decl_kargs_fnarg=fwd_decl_kargs_fnarg,
        k=k,
        traits_name=traits_name,
        kargs_name=kargs_name,
        da=da,
        db=db,
        kernel_func=kernel_func,
    )
    Path(os.path.join(cg.impl_path, f"{k.name}.cuh")).write_text(INSTANCE_IMPL)
    record_one_instantiation(cg, k, kernel_func, kargs_name, A16W16_TUNE_HOST_EXTRA)


def gen_flatmm_splitk_instance(
    cg,
    k,
    pipeline_header,
    traits_header,
    kernel_func,
    da,
    db,
    traits_name,
    kargs_name,
    kargs_template_vars,
    record_one_instantiation,
    A16W16_TUNE_HOST_EXTRA,
    **_unused,
):
    """gfx950 a16w16_flatmm_splitk launcher emit (uses ws_handle + reduce kernel call)."""
    kargs_explicit_param, fwd_decl_kargs_tpl, fwd_decl_kargs_fnarg = (
        kargs_template_vars(k.kernel_tag, kargs_name)
    )
    INSTANCE_IMPL = _render(
        "impl_flatmm_splitk_gfx950.cuh.j2",
        traits_header=traits_header,
        pipeline_header=pipeline_header,
        fwd_decl_kargs_tpl=fwd_decl_kargs_tpl,
        fwd_decl_kargs_fnarg=fwd_decl_kargs_fnarg,
        k=k,
        traits_name=traits_name,
        kargs_name=kargs_name,
        da=da,
        db=db,
        kernel_func=kernel_func,
    )
    Path(os.path.join(cg.impl_path, f"{k.name}.cuh")).write_text(INSTANCE_IMPL)
    record_one_instantiation(cg, k, kernel_func, kargs_name, A16W16_TUNE_HOST_EXTRA)


# ---------- Self-register at import time ----------
register_emit("gfx950", "a16w16_persistent", gen_persistent_instance)
register_emit("gfx950", "a8w8_scale", gen_scale_instance)
register_emit("gfx950", "a16w16", gen_noscale_instance_gfx950)
register_emit("gfx950", "a8w8", gen_noscale_instance_gfx950)
register_emit("gfx950", "a16w16_mono_tile", gen_mono_tile_instance)
register_emit("gfx950", "a16w16_flatmm", gen_flatmm_instance)
register_emit("gfx950", "a16w16_flatmm_splitk", gen_flatmm_splitk_instance)
