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
    instance_impl_preamble,
    instance_impl_host_tu_split,
    record_one_instantiation,
    A16W16_TUNE_HOST_EXTRA,
    **_unused,
):
    """gfx950 a16w16_persistent launcher emit. See gen_instances.opus_gemm_codegen._gen_persistent_instance."""
    kargs_explicit_param, fwd_decl_kargs_tpl, fwd_decl_kargs_fnarg = (
        kargs_template_vars(k.kernel_tag, kargs_name)
    )
    has_oob_str = "true" if k.has_oob else "false"

    traits_aliases = f"""
template <typename D_C>
using {k.name}_Traits = {traits_name}<{k.BLOCK_SIZE},
    opus::seq<{k.B_M}, {k.B_N}, {k.B_K}>,
    opus::tuple<{da}, {db}, D_C, fp32_t>,
    opus::seq<{k.VEC_A}, {k.VEC_B}, {k.VEC_C}>,
    opus::seq<{k.T_M}, {k.T_N}, 1>,
    opus::seq<{k.W_M}, {k.W_N}, {k.W_K}>,
    {has_oob_str},
    {k.cachectl_a},
    {k.cachectl_b}>;
"""

    min_k = 2 * k.B_K
    k_check = f"""
    int loops_ = (K + {k.B_K} - 1) / {k.B_K};
    AITER_CHECK(loops_ >= 2,
        "K=", K, " too small for B_K={k.B_K}, need K >= {min_k}");
    AITER_CHECK(loops_ % 2 == 0,
        "ceil_div(K, {k.B_K})=", loops_, " must be even (prefetch constraint)");
    AITER_CHECK(K % 2 == 0,
        "K=", K, " must be even (a16w16 family rejects odd K)");
    AITER_CHECK(M >= 1 && N >= 1, "M and N must be >= 1");
    AITER_CHECK(batch >= 1, "batch must be >= 1");
"""

    grid_setup = f"""
    constexpr int NUM_CU = 256;
    constexpr int NUM_XCD = 8;
    const int num_tiles_m = (M + {k.B_M} - 1) / {k.B_M};
    const int num_tiles_n = (N + {k.B_N} - 1) / {k.B_N};
    int split_m = std::max(1, (NUM_CU + num_tiles_n - 1) / num_tiles_n);
    while (split_m < num_tiles_m && (num_tiles_m % split_m) != 0) split_m++;
    if (split_m > num_tiles_m) split_m = num_tiles_m;
    const int m_per_wg = num_tiles_m / split_m;
    AITER_CHECK(num_tiles_m % split_m == 0,
        "persistent: num_tiles_m=", num_tiles_m,
        " must be divisible by split_m=", split_m);

    // Pad grid.y so the XCD-local swizzle math stays bijective. See the
    // long comment in opus_gemm_pipeline_a16w16_persistent_gfx950.cuh
    // for why this is needed and why it is free on the large-M shapes
    // the swizzle is tuned for (split_m is already a multiple of
    // NUM_XCD there, so the pad is a no-op). When split_m < NUM_XCD
    // (small-M shapes like M=8192 N=8192 K=256), the pad multiplies
    // grid.y by NUM_XCD/split_m and the kernel's wave-uniform
    // early-return guard drops the over-shoot WGs.
    const int m_grp_per_xcd = (split_m + NUM_XCD - 1) / NUM_XCD;
    const int grid_y_padded = m_grp_per_xcd * NUM_XCD;

    kargs.m_per_wg = m_per_wg;
    kargs.num_tiles_n = num_tiles_n;
    kargs.split_m = split_m;          // un-padded; kernel uses for early-return
    kargs.m_grp_per_xcd = m_grp_per_xcd;

    dim3 grid(num_tiles_n, grid_y_padded, batch);
    dim3 block({k.BLOCK_SIZE});
"""

    preamble = instance_impl_preamble("\n#include <algorithm>")
    host_tu_split = instance_impl_host_tu_split(
        traits_header,
        pipeline_header,
        fwd_decl_kargs_tpl,
        kernel_func,
        fwd_decl_kargs_fnarg,
    )
    INSTANCE_IMPL = f"""{preamble}
{host_tu_split}
{traits_aliases}
#if !defined(__HIP_DEVICE_COMPILE__) && !defined(__HIPCC_RTC__)
template <typename D_C>
void
{k.name}(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &Y,
    std::optional<aiter_tensor_t> bias,
    int /*splitK*/)   // persistent ignores splitK; shares tune-lookup slot signature
{{{{
    int batch = XQ.size(0);
    int M = XQ.size(1);
    int N = WQ.size(1);
    int K = XQ.size(2);
{k_check}
    AITER_CHECK(!bias.has_value(),
        "bias is not supported on a16w16_persistent kid; use a16w16 "
        "split-barrier (kid 4..9) or a16w16_flatmm_splitk (kid 200..299)");

    {kargs_name} kargs{{{{}}}};
    kargs.ptr_a = XQ.data_ptr();
    kargs.ptr_b = WQ.data_ptr();
    kargs.ptr_c = Y.data_ptr();
    kargs.m = M;
    kargs.n = N;
    kargs.k = K;
    kargs.batch = batch;
    kargs.stride_a = XQ.stride(1);
    kargs.stride_b = WQ.stride(1);
    kargs.stride_c = N;
    kargs.stride_a_batch = XQ.stride(0);
    kargs.stride_b_batch = WQ.stride(0);
    kargs.stride_c_batch = M * N;
{grid_setup}
    auto stream = aiter::getCurrentHIPStream();
    {kernel_func}<{k.name}_Traits<D_C>><<<grid, block, 0, stream>>>(kargs);

}}}}
#endif // launcher only on regular host pass
"""
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
    instance_impl_preamble,
    instance_impl_host_tu_split,
    record_one_instantiation,
    A8W8_SCALE_HOST_EXTRA,
    **_unused,
):
    """gfx950 a8w8_scale launcher emit."""
    kargs_explicit_param, fwd_decl_kargs_tpl, fwd_decl_kargs_fnarg = (
        kargs_template_vars(k.kernel_tag, kargs_name)
    )
    traits_aliases = f"""
template <typename D_C>
using {k.name}_Traits = {traits_name}<{k.BLOCK_SIZE},
    opus::seq<{k.B_M}, {k.B_N}, {k.B_K}>,
    opus::tuple<{da}, {db}, D_C, fp32_t, fp32_t>,
    opus::seq<{k.VEC_A}, {k.VEC_B}, {k.VEC_C}>,
    opus::seq<{k.GROUP_M}, {k.GROUP_N}, {k.GROUP_K}>>;
"""

    preamble = instance_impl_preamble()
    host_tu_split = instance_impl_host_tu_split(
        traits_header,
        pipeline_header,
        fwd_decl_kargs_tpl,
        kernel_func,
        fwd_decl_kargs_fnarg,
    )
    INSTANCE_IMPL = f"""{preamble}
{host_tu_split}
{traits_aliases}
#if !defined(__HIP_DEVICE_COMPILE__) && !defined(__HIPCC_RTC__)
template <typename D_C>
void
{k.name}(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &Y,
    std::optional<aiter_tensor_t> x_scale,
    std::optional<aiter_tensor_t> w_scale)
{{{{
    int batch = XQ.size(0);
    int M = XQ.size(1);
    int N = WQ.size(1);
    int K = XQ.size(2);

    using Traits = {k.name}_Traits<D_C>;

    int GROUP_M = {k.GROUP_M};
    int GROUP_N = {k.GROUP_N};
    int GROUP_K = {k.GROUP_K};
    int num_groups_m = M / GROUP_M;
    int num_groups_n = N / GROUP_N;
    int num_groups_k = K / GROUP_K;

    {kargs_name} kargs{{}};
    kargs.ptr_a = XQ.data_ptr();
    kargs.ptr_b = WQ.data_ptr();
    kargs.ptr_c = Y.data_ptr();
    kargs.m = M;
    kargs.n = N;
    kargs.k = K;
    kargs.batch = batch;
    kargs.stride_a = K;
    kargs.stride_b = K;
    kargs.stride_c = N;
    kargs.stride_a_batch = M * K;
    kargs.stride_b_batch = N * K;
    kargs.stride_c_batch = M * N;

    kargs.ptr_sfa = x_scale.value().data_ptr();
    kargs.ptr_sfb = w_scale.value().data_ptr();
    kargs.stride_sfa = num_groups_k;
    kargs.stride_sfb = num_groups_k;
    kargs.stride_sfa_batch = num_groups_m * num_groups_k;
    kargs.stride_sfb_batch = num_groups_n * num_groups_k;

    int num_tiles_m = (M + {k.B_M} - 1) / {k.B_M};
    int num_tiles_n = (N + {k.B_N} - 1) / {k.B_N};
    dim3 grid(num_tiles_m * num_tiles_n, 1, batch);
    dim3 block({k.BLOCK_SIZE});

    auto stream = aiter::getCurrentHIPStream();
    {kernel_func}<{k.name}_Traits<D_C>><<<grid, block, 0, stream>>>(kargs);

}}}}
#endif // launcher only on regular host pass
"""
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
    instance_impl_preamble,
    instance_impl_host_tu_split,
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
    is_a16w16_traits_with_tile_wave = (
        is_a16w16_split_barrier  # gfx950 noscale only a16w16 SB
    )
    traits_extra = ""
    if is_a16w16_traits_with_tile_wave:
        traits_extra = (
            f",\n        opus::seq<{k.T_M}, {k.T_N}, 1>,"
            f"\n        opus::seq<{k.W_M}, {k.W_N}, {k.W_K}>"
        )

    min_k = 2 * k.B_K
    k_check = f"""
    int loops_ = (K + {k.B_K} - 1) / {k.B_K};
    AITER_CHECK(loops_ >= 2,
        "K=", K, " too small for B_K={k.B_K}, need K >= {min_k}");
    AITER_CHECK(loops_ % 2 == 0,
        "ceil_div(K, {k.B_K})=", loops_, " must be even (prefetch constraint)");
    AITER_CHECK(K % 2 == 0,
        "K=", K, " must be even (a16w16 family rejects odd K due to a "
        "latent K-tail accumulation bug; pass an even K)");
    AITER_CHECK(M >= 1 && N >= 1, "M and N must be >= 1");
"""

    if k.kernel_tag in A16W16_TUNE_TAGS:
        extra_param = (
            ",\n    std::optional<aiter_tensor_t> bias," "\n    int /*splitK*/"
        )
    else:
        extra_param = ""

    has_oob_str = "true" if k.has_oob else "false"

    if is_a16w16_split_barrier:
        bias_kargs_block = (
            BIAS_HOST_VALIDATE
            + "    kargs.ptr_bias = ptr_bias_;\n"
            + "    kargs.stride_bias_batch = stride_bias_batch_;\n"
        )
    elif k.kernel_tag in A16W16_TUNE_TAGS:
        bias_kargs_block = (
            "    AITER_CHECK(!bias.has_value(),\n"
            '        "bias not supported on this a16w16 kid");\n'
        )
    else:
        bias_kargs_block = ""

    kargs_init_extra = ""

    cachectl_extra = ""
    if is_a16w16_split_barrier and hasattr(k, "cachectl_a") and k.cachectl_a >= 0:
        cachectl_extra = f",\n    {k.cachectl_a}, {k.cachectl_b}"
    traits_alias_tail = f",\n    {has_oob_str}"
    if is_a16w16_split_barrier:
        traits_aliases = f"""
template <typename D_C>
using {k.name}_TraitsNoBias = {traits_name}<{k.BLOCK_SIZE},
    opus::seq<{k.B_M}, {k.B_N}, {k.B_K}>,
    opus::tuple<{da}, {db}, D_C, fp32_t>,
    opus::seq<{k.VEC_A}, {k.VEC_B}, {k.VEC_C}>{traits_extra},
    false,
    D_C{traits_alias_tail}{cachectl_extra}>;
template <typename D_C>
using {k.name}_TraitsBias = {traits_name}<{k.BLOCK_SIZE},
    opus::seq<{k.B_M}, {k.B_N}, {k.B_K}>,
    opus::tuple<{da}, {db}, D_C, fp32_t>,
    opus::seq<{k.VEC_A}, {k.VEC_B}, {k.VEC_C}>{traits_extra},
    true,
    D_C{traits_alias_tail}{cachectl_extra}>;
"""
    else:
        traits_aliases = f"""
template <typename D_C>
using {k.name}_Traits = {traits_name}<{k.BLOCK_SIZE},
    opus::seq<{k.B_M}, {k.B_N}, {k.B_K}>,
    opus::tuple<{da}, {db}, D_C, fp32_t>,
    opus::seq<{k.VEC_A}, {k.VEC_B}, {k.VEC_C}>{traits_extra}>;
"""

    if is_a16w16_split_barrier:
        launch_block = f"""
    auto stream = aiter::getCurrentHIPStream();
    if (bias.has_value()) {{{{
        {kernel_func}<{k.name}_TraitsBias<D_C>><<<grid, block, 0, stream>>>(kargs);
    }}}} else {{{{
        {kernel_func}<{k.name}_TraitsNoBias<D_C>><<<grid, block, 0, stream>>>(kargs);
    }}}}"""
    else:
        launch_block = f"""
    auto stream = aiter::getCurrentHIPStream();
    {kernel_func}<{k.name}_Traits<D_C>><<<grid, block, 0, stream>>>(kargs);"""

    preamble = instance_impl_preamble()
    host_tu_split = instance_impl_host_tu_split(
        traits_header,
        pipeline_header,
        fwd_decl_kargs_tpl,
        kernel_func,
        fwd_decl_kargs_fnarg,
    )
    INSTANCE_IMPL = f"""{preamble}
{host_tu_split}
{traits_aliases}
#if !defined(__HIP_DEVICE_COMPILE__) && !defined(__HIPCC_RTC__)
template <typename D_C>
void
{k.name}(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &Y{extra_param})
{{{{
    int batch = XQ.size(0);
    int M = XQ.size(1);
    int N = WQ.size(1);
    int K = XQ.size(2);
{k_check}
    {kargs_name} kargs{{}};
    kargs.ptr_a = XQ.data_ptr();
    kargs.ptr_b = WQ.data_ptr();
    kargs.ptr_c = Y.data_ptr();
    kargs.m = M;
    kargs.n = N;
    kargs.k = K;
    kargs.batch = batch;
    kargs.stride_a = XQ.stride(1);
    kargs.stride_b = WQ.stride(1);
    kargs.stride_c = N;
    kargs.stride_a_batch = XQ.stride(0);
    kargs.stride_b_batch = WQ.stride(0);
    kargs.stride_c_batch = M * N;
{kargs_init_extra}{bias_kargs_block}
    int num_tiles_m = (M + {k.B_M} - 1) / {k.B_M};
    int num_tiles_n = (N + {k.B_N} - 1) / {k.B_N};
    dim3 grid(num_tiles_m * num_tiles_n, 1, batch);
    dim3 block({k.BLOCK_SIZE});
{launch_block}

}}}}
#endif // launcher only on regular host pass
"""
    Path(os.path.join(cg.impl_path, f"{k.name}.cuh")).write_text(INSTANCE_IMPL)

    if k.kernel_tag in A16W16_TUNE_TAGS:
        inst_extra_param = ",\n    std::optional<aiter_tensor_t>,\n    int"
    else:
        inst_extra_param = ""

    if is_a16w16_split_barrier:

        def _device_decl(dtype):
            return (
                f"template __global__ void {kernel_func}<\n"
                f"    {k.name}_TraitsNoBias<{dtype}>>({kargs_name});\n"
                f"template __global__ void {kernel_func}<\n"
                f"    {k.name}_TraitsBias<{dtype}>>({kargs_name});\n"
            )

    else:

        def _device_decl(dtype):
            return (
                f"template __global__ void {kernel_func}<\n"
                f"    {k.name}_Traits<{dtype}>{kargs_explicit_param}>({kargs_name});\n"
            )

    for CDtype in k.output_dtypes:
        host_decl = (
            f"template void\n"
            f"{k.name}<{CDtype}>(\n"
            f"    aiter_tensor_t &XQ,\n"
            f"    aiter_tensor_t &WQ,\n"
            f"    aiter_tensor_t &Y{inst_extra_param});\n"
        )
        cg._host_instantiations.append(
            {"kid_name": k.name, "dtype": CDtype, "host_decl": host_decl}
        )
        cg._device_instantiations.append(
            {"kid_name": k.name, "dtype": CDtype, "device_decl": _device_decl(CDtype)}
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
    traits_aliases = f"""
template <typename D_C>
using {k.name}_Traits = {traits_name}<{k.BLOCK_SIZE},
    opus::seq<{k.B_M}, {k.B_N}, {k.B_K}>,
    opus::tuple<{da}, {db}, D_C, fp32_t>,
    opus::seq<{k.VEC_A}, {k.VEC_B}, {k.VEC_C}>>;
"""
    min_k = 2 * k.B_K
    k_check = f"""
    int loops_ = K / {k.B_K};
    AITER_CHECK(K % {k.B_K} == 0,
        "mono-tile requires K divisible by B_K={k.B_K}; got K=", K);
    AITER_CHECK(loops_ >= 2,
        "K=", K, " too small for B_K={k.B_K}, need K >= {min_k}");
    AITER_CHECK(K % 2 == 0,
        "K=", K, " must be even (a16w16 family rejects odd K)");
    AITER_CHECK(M >= 1 && N >= 1, "M and N must be >= 1");
    AITER_CHECK(batch >= 1, "batch must be >= 1");
    AITER_CHECK(N % {k.B_N} == 0,
        "mono-tile requires N divisible by B_N={k.B_N}; got N=", N);
"""
    INSTANCE_IMPL = f"""// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
#pragma once
#if !defined(__HIP_DEVICE_COMPILE__) && !defined(__HIPCC_RTC__)
#include "aiter_tensor.h"
#include "aiter_stream.h"
#include <optional>
#endif
// See _gen_noscale_instance for the rationale of the host/device pass split.
#ifdef OPUS_FUSED_HOST_TU
#include "{traits_header}"
template<typename Traits>
__global__ void {kernel_func}({kargs_name} kargs);
#else
#include "{pipeline_header}"
#endif
{traits_aliases}
#if !defined(__HIP_DEVICE_COMPILE__) && !defined(__HIPCC_RTC__)
template <typename D_C>
void
{k.name}(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &Y,
    std::optional<aiter_tensor_t> bias,
    int /*splitK*/)
{{{{
    int batch = XQ.size(0);
    int M = XQ.size(1);
    int N = WQ.size(1);
    int K = XQ.size(2);
{k_check}
    AITER_CHECK(!bias.has_value(),
        "bias is not supported on a16w16_mono_tile kid; use a16w16 "
        "split-barrier (kid 4..9) or a16w16_flatmm_splitk (kid 200..299)");

    {kargs_name} kargs{{{{}}}};
    kargs.ptr_a = XQ.data_ptr();
    kargs.ptr_b = WQ.data_ptr();
    kargs.ptr_c = Y.data_ptr();
    kargs.m = M;
    kargs.n = N;
    kargs.k = K;
    kargs.batch = batch;
    kargs.stride_a = XQ.stride(1);
    kargs.stride_b = WQ.stride(1);
    kargs.stride_c = N;
    kargs.stride_a_batch = XQ.stride(0);
    kargs.stride_b_batch = WQ.stride(0);
    kargs.stride_c_batch = M * N;

    int num_tiles_m = (M + {k.B_M} - 1) / {k.B_M};
    int num_tiles_n = (N + {k.B_N} - 1) / {k.B_N};
    dim3 grid(num_tiles_m * num_tiles_n, 1, batch);
    dim3 block({k.BLOCK_SIZE});

    auto stream = aiter::getCurrentHIPStream();
    {kernel_func}<{k.name}_Traits<D_C>><<<grid, block, 0, stream>>>(kargs);

}}}}
#endif // launcher only on regular host pass
"""
    Path(os.path.join(cg.impl_path, f"{k.name}.cuh")).write_text(INSTANCE_IMPL)

    for CDtype in k.output_dtypes:
        host_decl = (
            f"template void\n"
            f"{k.name}<{CDtype}>(\n"
            f"    aiter_tensor_t &XQ,\n"
            f"    aiter_tensor_t &WQ,\n"
            f"    aiter_tensor_t &Y,\n"
            f"    std::optional<aiter_tensor_t>,\n"
            f"    int);\n"
        )
        device_decl = (
            f"template __global__ void {kernel_func}<\n"
            f"    {k.name}_Traits<{CDtype}>>({kargs_name});\n"
        )
        cg._host_instantiations.append(
            {"kid_name": k.name, "dtype": CDtype, "host_decl": host_decl}
        )
        cg._device_instantiations.append(
            {"kid_name": k.name, "dtype": CDtype, "device_decl": device_decl}
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
    instance_impl_preamble,
    instance_impl_host_tu_split,
    record_one_instantiation,
    A16W16_TUNE_HOST_EXTRA,
    **_unused,
):
    """gfx950 a16w16_flatmm launcher emit."""
    kargs_explicit_param, fwd_decl_kargs_tpl, fwd_decl_kargs_fnarg = (
        kargs_template_vars(k.kernel_tag, kargs_name)
    )
    has_bias_str = "false"

    k_check = f"""
    int loops_ = (K + {k.B_K} - 1) / {k.B_K};
    AITER_CHECK(loops_ >= Traits::prefetch_k_iter,
        "K=", K, " too small for flatmm B_K={k.B_K}, need K >= pfk*B_K = ",
        Traits::prefetch_k_iter * {k.B_K}, " (pfk=", Traits::prefetch_k_iter, ")");
    AITER_CHECK(M >= 1 && N >= 1 && K >= 1, "M, N, K must be >= 1");
    AITER_CHECK(batch >= 1, "batch must be >= 1");
    AITER_CHECK(K % 2 == 0,
        "K=", K, " must be even (a16w16 family rejects odd K due to a "
        "latent K-tail accumulation bug; pass an even K)");
"""

    traits_aliases = f"""
template <typename D_C>
using {k.name}_Traits = {traits_name}<{k.BLOCK_SIZE},
    opus::seq<{k.B_M}, {k.B_N}, {k.B_K}>,
    opus::tuple<{da}, {db}, D_C, fp32_t, D_C>,
    opus::seq<{k.VEC_A}, {k.VEC_B}, {k.VEC_C}>,
    opus::seq<{k.W_M}, {k.W_N}, {k.W_K}>,
    {k.WG_PER_CU},
    {has_bias_str}>;
"""

    preamble = instance_impl_preamble()
    host_tu_split = instance_impl_host_tu_split(
        traits_header,
        pipeline_header,
        fwd_decl_kargs_tpl,
        kernel_func,
        fwd_decl_kargs_fnarg,
    )
    INSTANCE_IMPL = f"""{preamble}
{host_tu_split}
{traits_aliases}
#if !defined(__HIP_DEVICE_COMPILE__) && !defined(__HIPCC_RTC__)
template <typename D_C>
void
{k.name}(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &Y,
    std::optional<aiter_tensor_t> bias,
    int /*splitK*/)
{{{{
    int batch = XQ.size(0);
    int M = XQ.size(1);
    int N = WQ.size(1);
    int K = XQ.size(2);

    AITER_CHECK(!bias.has_value(),
        "bias is not yet supported on a16w16_flatmm kid; use a16w16 "
        "split-barrier (kid 4..9) or a16w16_flatmm_splitk (kid 200..299)");

    using Traits = {k.name}_Traits<D_C>;
{k_check}
    {kargs_name} kargs{{{{}}}};
    kargs.ptr_a = XQ.data_ptr();
    kargs.ptr_b = WQ.data_ptr();
    kargs.ptr_c = Y.data_ptr();
    kargs.ptr_bias = nullptr;
    kargs.m = M;
    kargs.n = N;
    kargs.k = K;
    kargs.batch = batch;
    kargs.stride_a = XQ.stride(1);
    kargs.stride_b = WQ.stride(1);
    kargs.stride_c = N;
    kargs.stride_a_batch = XQ.stride(0);
    kargs.stride_b_batch = WQ.stride(0);
    kargs.stride_c_batch = M * N;

    int num_tiles_m = (M + {k.B_M} - 1) / {k.B_M};
    int num_tiles_n = (N + {k.B_N} - 1) / {k.B_N};
    dim3 grid(num_tiles_m * num_tiles_n, 1, batch);
    dim3 block({k.BLOCK_SIZE});

    auto stream = aiter::getCurrentHIPStream();
    {kernel_func}<{k.name}_Traits<D_C>><<<grid, block, 0, stream>>>(kargs);

}}}}
#endif // launcher only on regular host pass
"""
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
    instance_impl_preamble,
    instance_impl_host_tu_split,
    record_one_instantiation,
    A16W16_TUNE_HOST_EXTRA,
    BIAS_HOST_VALIDATE,
    **_unused,
):
    """gfx950 a16w16_flatmm_splitk launcher emit (uses ws_handle + reduce kernel call)."""
    kargs_explicit_param, fwd_decl_kargs_tpl, fwd_decl_kargs_fnarg = (
        kargs_template_vars(k.kernel_tag, kargs_name)
    )
    has_oob_str = "true" if k.has_oob else "false"
    traits_aliases = f"""
template <typename D_C>
using {k.name}_Traits = {traits_name}<{k.BLOCK_SIZE},
    opus::seq<{k.B_M}, {k.B_N}, {k.B_K}>,
    opus::tuple<{da}, {db}, fp32_t, fp32_t, {da}>,
    opus::seq<{k.VEC_A}, {k.VEC_B}, {k.VEC_C}>,
    opus::seq<{k.W_M}, {k.W_N}, {k.W_K}>,
    {k.WG_PER_CU},
    false,
    {has_oob_str}>;
"""

    preamble = instance_impl_preamble()
    host_tu_split = instance_impl_host_tu_split(
        traits_header,
        pipeline_header,
        fwd_decl_kargs_tpl,
        kernel_func,
        fwd_decl_kargs_fnarg,
    )
    INSTANCE_IMPL = f"""{preamble}
{host_tu_split}
{traits_aliases}
#if !defined(__HIP_DEVICE_COMPILE__) && !defined(__HIPCC_RTC__)
template <typename D_C>
void
{k.name}(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &Y,
    std::optional<aiter_tensor_t> bias,
    int splitK)
{{{{
    static_assert(std::is_same<D_C, fp32_t>::value,
        "splitk main kernel uses fp32 workspace; D_C template param must be fp32_t "
        "(Y can be bf16 or fp32; reduce kernel handles the cast / passthrough)");

    int batch = XQ.size(0);
    int M = XQ.size(1);
    int N = WQ.size(1);
    int K = XQ.size(2);

    AITER_CHECK(Y.dtype() == AITER_DTYPE_bf16
                || Y.dtype() == AITER_DTYPE_fp32,
        "flatmm_splitk requires Y dtype bf16 or fp32 "
        "(reduce kernel casts fp32 workspace to D_OUT)");
    AITER_CHECK(M >= 1 && N >= 1 && K >= 1 && batch >= 1,
        "M, N, K, batch must be >= 1");
    AITER_CHECK(K % 2 == 0,
        "K=", K, " must be even (a16w16 family rejects odd K due to a "
        "latent K-tail accumulation bug; pass an even K)");
{BIAS_HOST_VALIDATE}
    using Traits = {k.name}_Traits<D_C>;

    int split_k = (splitK <= 1) ? 1 : splitK;

    int total_iters = (K + {k.B_K} - 1) / {k.B_K};
    constexpr int pfk = Traits::prefetch_k_iter;
    while (split_k > 1) {{{{
        int iters_full = (total_iters + split_k - 1) / split_k;
        int last_loops = total_iters - (split_k - 1) * iters_full;
        if (iters_full >= pfk && last_loops >= pfk) break;
        split_k--;
    }}}}
    AITER_CHECK(total_iters >= pfk,
        "K=", K, " too small for flatmm_splitk B_K={k.B_K}: "
        "need total_iters >= pfk*B_K = ", pfk * {k.B_K},
        " (pfk=", pfk, ")");

    int num_tiles_m = (M + {k.B_M} - 1) / {k.B_M};
    int num_tiles_n = (N + {k.B_N} - 1) / {k.B_N};
    int padded_M    = num_tiles_m * {k.B_M};
    int padded_N    = num_tiles_n * {k.B_N};

    // Per-stream workspace handle (process-global registry, mutex-protected
    // in opus_gemm.cu). Replaces the prior `static thread_local` cache --
    // under TBO two CPU threads drive two streams concurrently, and each
    // captured graph must bake in its own buffer pointer. Eager: lazy-
    // create. Capture: must be pre-warmed via
    // aiter.opus_gemm_workspace_init() on the capture stream.
    // (opus_splitk_ws_handle is already a complete type at this point via
    // the traits header included at the top of this launcher .cuh.)
    extern opus_splitk_ws_handle* opus_splitk_ws_get(hipStream_t, bool);

    auto stream = aiter::getCurrentHIPStream();
    hipStreamCaptureStatus capture_status = hipStreamCaptureStatusNone;
    HIP_CALL(hipStreamIsCapturing(stream, &capture_status));
    const bool capturing = (capture_status != hipStreamCaptureStatusNone);
    auto* ws_handle_ = opus_splitk_ws_get(stream, /*allow_create=*/!capturing);

    size_t ws_bytes = (size_t)split_k * (size_t)batch
                    * (size_t)padded_M * (size_t)padded_N * sizeof(float);
    if (ws_handle_->ptr == nullptr || ws_bytes > ws_handle_->bytes)
    {{
        AITER_CHECK(!capturing,
            "splitk workspace grow inside HIP graph capture is not "
            "supported (hipMalloc / hipFree are stream-capture-illegal). "
            "Warm the cache once eagerly with the largest workspace before "
            "capturing. Call aiter.opus_gemm_workspace_init() on the capture "
            "stream first.");

        void* new_ptr = nullptr;
        const size_t kGrowAlign = (size_t)4 * 1024 * 1024;
        size_t grow_bytes = ((ws_bytes + kGrowAlign - 1) / kGrowAlign) * kGrowAlign;
        HIP_CALL(hipMalloc(&new_ptr, grow_bytes));
        if (ws_handle_->ptr != nullptr)
        {{
            HIP_CALL(hipDeviceSynchronize());
            HIP_CALL(hipFree(ws_handle_->ptr));
        }}
        ws_handle_->ptr = new_ptr;
        ws_handle_->bytes = grow_bytes;
    }}

    {kargs_name} kargs{{{{}}}};
    kargs.ptr_a         = XQ.data_ptr();
    kargs.ptr_b         = WQ.data_ptr();
    kargs.ws_handle     = ws_handle_;
    kargs.ptr_c         = Y.data_ptr();
    kargs.ptr_bias      = ptr_bias_;
    kargs.m = M; kargs.n = N; kargs.k = K; kargs.batch = batch;
    kargs.split_k = split_k;
    kargs.stride_a        = XQ.stride(1);
    kargs.stride_b        = WQ.stride(1);
    kargs.stride_ws       = padded_N;
    kargs.stride_c        = N;
    kargs.stride_a_batch  = XQ.stride(0);
    kargs.stride_b_batch  = WQ.stride(0);
    kargs.stride_ws_batch = padded_M * padded_N;
    kargs.stride_c_batch  = M * N;
    kargs.stride_bias_batch = stride_bias_batch_;

    dim3 grid_main(num_tiles_m * num_tiles_n * split_k, 1, batch);
    dim3 block_main({k.BLOCK_SIZE});

    constexpr int REDUCE_VEC = 16;
    constexpr int REDUCE_BS  = 64;
    dim3 grid_reduce((N + REDUCE_VEC * REDUCE_BS - 1) / (REDUCE_VEC * REDUCE_BS),
                      batch * M, 1);
    dim3 block_reduce(REDUCE_BS);

    {kernel_func}<{k.name}_Traits<D_C>><<<grid_main, block_main, 0, stream>>>(kargs);
    if (Y.dtype() == AITER_DTYPE_bf16) {{{{
        if (bias.has_value()) {{{{
            splitk_reduce_kernel<REDUCE_VEC, REDUCE_BS, __bf16, true, __bf16, {has_oob_str}>
                <<<grid_reduce, block_reduce, 0, stream>>>(
                    ws_handle_,
                    reinterpret_cast<__bf16*>(Y.data_ptr()),
                    split_k, M, N, batch, padded_M, padded_N,
                    reinterpret_cast<const __bf16*>(ptr_bias_),
                    stride_bias_batch_);
        }}}} else {{{{
            splitk_reduce_kernel<REDUCE_VEC, REDUCE_BS, __bf16, false, __bf16, {has_oob_str}>
                <<<grid_reduce, block_reduce, 0, stream>>>(
                    ws_handle_,
                    reinterpret_cast<__bf16*>(Y.data_ptr()),
                    split_k, M, N, batch, padded_M, padded_N,
                    nullptr, 0);
        }}}}
    }}}} else {{{{
        if (bias.has_value()) {{{{
            splitk_reduce_kernel<REDUCE_VEC, REDUCE_BS, float, true, float, {has_oob_str}>
                <<<grid_reduce, block_reduce, 0, stream>>>(
                    ws_handle_,
                    reinterpret_cast<float*>(Y.data_ptr()),
                    split_k, M, N, batch, padded_M, padded_N,
                    reinterpret_cast<const float*>(ptr_bias_),
                    stride_bias_batch_);
        }}}} else {{{{
            splitk_reduce_kernel<REDUCE_VEC, REDUCE_BS, float, false, float, {has_oob_str}>
                <<<grid_reduce, block_reduce, 0, stream>>>(
                    ws_handle_,
                    reinterpret_cast<float*>(Y.data_ptr()),
                    split_k, M, N, batch, padded_M, padded_N,
                    nullptr, 0);
        }}}}
    }}}}

}}}}
#endif // launcher only on regular host pass
"""
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
