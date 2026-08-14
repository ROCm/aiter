# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Generate gfx950 OPUS launchers."""

import os
from pathlib import Path

from opus_gemm_common import OpusGemmInstance

from codegen.common import (
    WARP_SIZE,
    register_arch_map,
    register_emit,
    splitk_workspace_type,
)

# ---------------- gfx950 arch-override maps ----------------

PIPELINE_HEADER_MAP = {
    "a8w8_scale": "gfx950/opus_bmm_pipeline_a8w8_mxscale_gfx950.cuh",
    "a8w8_mxscale": "gfx950/opus_bmm_pipeline_a8w8_mxscale_gfx950.cuh",
    "a8w8": "gfx950/opus_gemm_pipeline_a8w8_noscale_gfx950.cuh",
    "a16w16": "gfx950/opus_gemm_pipeline_a16w16_gfx950.cuh",
    "a16w16_flatmm": "gfx950/opus_gemm_pipeline_a16w16_flatmm_gfx950.cuh",
    "a16w16_flatmm_splitk": "gfx950/opus_gemm_pipeline_a16w16_flatmm_splitk_gfx950.cuh",
    "a16w16_persistent": "gfx950/opus_gemm_pipeline_a16w16_persistent_gfx950.cuh",
    "a16w16_mono_tile": "gfx950/opus_gemm_pipeline_a16w16_mono_tile_gfx950.cuh",
    "a8w8_mxscale_bmm_flatmm_splitk": "gfx950/opus_gemm_pipeline_a8w8_mxscale_flatmm_splitk_gfx950.cuh",
    "a8w8_mxscale_bmm_minterleave": "gfx950/opus_gemm_pipeline_a8w8_mxscale_flatmm_splitk_gfx950.cuh",
    "a8w8_mxscale_bmm_fused": "gfx950/opus_gemm_pipeline_a8w8_mxscale_flatmm_splitk_gfx950.cuh",
    "a8w8_mxscale_bmm_pipeline": "gfx950/opus_bmm_pipeline_a8w8_mxscale_gfx950.cuh",
    "a8w8_mxscale_bmm_mouter": "gfx950/opus_gemm_pipeline_a8w8_mxscale_flatmm_splitk_gfx950.cuh",
    "a8w8_mxscale_bmm_mouter_tunable": "gfx950/opus_gemm_pipeline_a8w8_mxscale_flatmm_splitk_gfx950.cuh",
    "a8w8_mxscale_bmm_wave8n2": "gfx950/opus_gemm_pipeline_a8w8_mxscale_flatmm_splitk_gfx950.cuh",
    "a8w8_mxscale_bmm_wave4m2_selfload": "gfx950/opus_gemm_pipeline_a8w8_mxscale_flatmm_splitk_gfx950.cuh",
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
    "a8w8_mxscale": "gfx950/opus_gemm_traits_a8w8_scale_gfx950.cuh",
    "a8w8": "gfx950/opus_gemm_traits_a8w8_noscale_gfx950.cuh",
    "a16w16": "gfx950/opus_gemm_traits_a16w16_gfx950.cuh",
    "a16w16_flatmm": "gfx950/opus_gemm_traits_a16w16_gfx950.cuh",
    "a16w16_flatmm_splitk": "gfx950/opus_gemm_traits_a16w16_gfx950.cuh",
    "a16w16_persistent": "gfx950/opus_gemm_traits_a16w16_gfx950.cuh",
    "a16w16_mono_tile": "gfx950/opus_gemm_traits_a16w16_gfx950.cuh",
    "a8w8_mxscale_bmm_flatmm_splitk": "gfx950/opus_gemm_traits_a8w8_scale_gfx950.cuh",
    "a8w8_mxscale_bmm_minterleave": "gfx950/opus_gemm_traits_a8w8_scale_gfx950.cuh",
    "a8w8_mxscale_bmm_fused": "gfx950/opus_gemm_traits_a8w8_scale_gfx950.cuh",
    "a8w8_mxscale_bmm_pipeline": "gfx950/opus_gemm_traits_a8w8_scale_gfx950.cuh",
    "a8w8_mxscale_bmm_mouter": "gfx950/opus_gemm_traits_a8w8_scale_gfx950.cuh",
    "a8w8_mxscale_bmm_mouter_tunable": "gfx950/opus_gemm_traits_a8w8_scale_gfx950.cuh",
    "a8w8_mxscale_bmm_wave8n2": "gfx950/opus_gemm_traits_a8w8_scale_gfx950.cuh",
    "a8w8_mxscale_bmm_wave4m2_selfload": "gfx950/opus_gemm_traits_a8w8_scale_gfx950.cuh",
}

KERNEL_FUNC_MAP = {
    "a8w8_scale": "gemm_a8w8_scale_kernel",
    "a8w8_mxscale": "gemm_a8w8_scale_kernel",
    "a8w8": "gemm_a8w8_noscale_kernel",
    "a16w16": "gemm_a16w16_kernel",
    "a16w16_flatmm": "gemm_a16w16_flatmm_kernel",
    "a16w16_flatmm_splitk": "gemm_a16w16_flatmm_splitk_kernel",
    "a16w16_persistent": "gemm_a16w16_persistent_kernel",
    "a16w16_mono_tile": "gemm_a16w16_mono_tile_kernel_gfx950",
    "a8w8_mxscale_bmm_flatmm_splitk": "gemm_a8w8_mxscale_flatmm_splitk_kernel",
    "a8w8_mxscale_bmm_minterleave": "gemm_a8w8_mxscale_flatmm_minterleave_kernel",
    "a8w8_mxscale_bmm_fused": "gemm_a8w8_mxscale_flatmm_splitk_kernel",
    # pipeline: default; the emit fn selects the real kernel per-kid from flags.
    "a8w8_mxscale_bmm_pipeline": "gemm_a8w8_scale_kernel",
    "a8w8_mxscale_bmm_mouter": "gemm_a8w8_mxscale_flatmm_splitk_mouter_kernel",
    "a8w8_mxscale_bmm_mouter_tunable": "gemm_a8w8_mxscale_flatmm_splitk_mouter_kernel",
    "a8w8_mxscale_bmm_wave8n2": "gemm_a8w8_mxscale_flatmm_splitk_wave8n2_kernel",
    "a8w8_mxscale_bmm_wave4m2_selfload": "gemm_a8w8_mxscale_flatmm_splitk_wave4m2_selfload_kernel",
}

KERNEL_FUNC_MAP_4G_SAFE = {
    "a16w16": "gemm_a16w16_4g_safe_kernel",
    "a16w16_persistent": "gemm_a16w16_persistent_4g_safe_kernel",
    "a16w16_mono_tile": "gemm_a16w16_mono_tile_4g_safe_kernel_gfx950",
}

TRAITS_NAME_MAP = {
    "a8w8_scale": "opus_gemm_a8w8_scale_traits_gfx950",
    "a8w8_mxscale": "opus_gemm_a8w8_scale_traits_gfx950",
    "a8w8": "opus_gemm_a8w8_noscale_traits_gfx950",
    "a16w16": "opus_gemm_a16w16_traits_gfx950",
    "a16w16_flatmm": "opus_gemm_a16w16_flatmm_traits_gfx950",
    "a16w16_flatmm_splitk": "opus_flatmm_splitk_traits_gfx950",
    "a16w16_persistent": "opus_gemm_a16w16_persistent_traits_gfx950",
    "a16w16_mono_tile": "opus_gemm_a16w16_mono_tile_traits_gfx950",
    "a8w8_mxscale_bmm_flatmm_splitk": "opus_gemm_a8w8_mxscale_flatmm_splitk_traits_gfx950",
    "a8w8_mxscale_bmm_minterleave": "opus_gemm_a8w8_mxscale_flatmm_splitk_traits_gfx950",
    "a8w8_mxscale_bmm_fused": "opus_gemm_a8w8_mxscale_flatmm_splitk_traits_gfx950",
    "a8w8_mxscale_bmm_pipeline": "opus_gemm_a8w8_scale_traits_gfx950",
    "a8w8_mxscale_bmm_mouter": "opus_gemm_a8w8_mxscale_flatmm_splitk_traits_gfx950",
    "a8w8_mxscale_bmm_mouter_tunable": "opus_gemm_a8w8_mxscale_flatmm_splitk_traits_gfx950",
    "a8w8_mxscale_bmm_wave8n2": "opus_gemm_a8w8_mxscale_flatmm_splitk_traits_gfx950",
    "a8w8_mxscale_bmm_wave4m2_selfload": "opus_gemm_a8w8_mxscale_flatmm_splitk_traits_gfx950",
}

KARGS_NAME_MAP = {
    "a8w8_scale": "opus_gemm_scale_kargs_gfx950",
    "a8w8_mxscale": "opus_gemm_scale_kargs_gfx950",
    "a8w8": "opus_gemm_noscale_kargs_gfx950",
    "a16w16": "opus_gemm_noscale_kargs_gfx950",
    "a16w16_flatmm": "opus_gemm_flatmm_kargs_gfx950",
    "a16w16_flatmm_splitk": "opus_gemm_flatmm_splitk_kargs_gfx950",
    "a16w16_persistent": "opus_gemm_persistent_kargs_gfx950",
    "a16w16_mono_tile": "opus_gemm_mono_tile_kargs_gfx950",
    "a8w8_mxscale_bmm_flatmm_splitk": "opus_gemm_scale_splitk_kargs_gfx950",
    "a8w8_mxscale_bmm_minterleave": "opus_gemm_scale_splitk_kargs_gfx950",
    "a8w8_mxscale_bmm_fused": "opus_gemm_scale_splitk_kargs_gfx950",
    "a8w8_mxscale_bmm_pipeline": "opus_gemm_scale_kargs_gfx950",
    "a8w8_mxscale_bmm_mouter": "opus_gemm_scale_splitk_kargs_gfx950",
    "a8w8_mxscale_bmm_mouter_tunable": "opus_gemm_scale_splitk_kargs_gfx950",
    "a8w8_mxscale_bmm_wave8n2": "opus_gemm_scale_splitk_kargs_gfx950",
    "a8w8_mxscale_bmm_wave4m2_selfload": "opus_gemm_scale_splitk_kargs_gfx950",
}


def splitk_reduce_extra_device_instantiations():
    # gfx950 carries a second reduce kernel: the mmajor BMM reduce used by the
    # a8w8_mxscale BMM split-K launchers (VEC=8/BLOCK=128, explicit C strides,
    # no bias fold). Those launchers <<<>>> it from their fused host TU and only
    # see a forward decl, so exactly one TU must own the device kernel plus its
    # host stub. It lives in the same splitk_reduce_gfx950.cuh as the baseline
    # reduce, so it rides along in this TU; that keeps opus_bmm.cu out of the
    # device pass entirely, matching opus_gemm.cu.
    return (
        "// mmajor BMM reduce (a8w8_mxscale split-K launchers)\n"
        "template __global__ void opus_bmm_splitk_reduce_kernel<__bf16, 8, 128>(\n"
        "    const void*, __bf16*,\n"
        "    int, int, int, int, int, int, int, int);\n"
        "template __global__ void opus_bmm_splitk_reduce_kernel<float, 8, 128>(\n"
        "    const void*, float*,\n"
        "    int, int, int, int, int, int, int, int);\n"
    )


SPLITK_REDUCE_EXTRA_MAP = {
    "device_instantiations": splitk_reduce_extra_device_instantiations,
}

register_arch_map("gfx950", "pipeline_header", PIPELINE_HEADER_MAP)
register_arch_map("gfx950", "traits_header", TRAITS_HEADER_MAP)
register_arch_map("gfx950", "kernel_func", KERNEL_FUNC_MAP)
register_arch_map("gfx950", "traits_name", TRAITS_NAME_MAP)
register_arch_map("gfx950", "kargs_name", KARGS_NAME_MAP)
register_arch_map("gfx950", "splitk_reduce_extra", SPLITK_REDUCE_EXTRA_MAP)


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
    A16W16_LAUNCH_HOST_EXTRA,
    **_unused,
):
    """gfx950 a16w16_persistent launcher emit. See gen_instances.opus_gemm_codegen._gen_persistent_instance."""
    _kargs_explicit_param, fwd_decl_kargs_tpl, fwd_decl_kargs_fnarg = (
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
    int /*splitK*/)   // persistent ignores splitK; shares launch-table signature
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
    record_one_instantiation(cg, k, kernel_func, kargs_name, A16W16_LAUNCH_HOST_EXTRA)


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
    A8W8_BLOCKSCALE_HOST_EXTRA,
    **_unused,
):
    """Emit the checked gfx950 A8W8 blockscale launcher."""
    _kargs_explicit_param, fwd_decl_kargs_tpl, fwd_decl_kargs_fnarg = (
        kargs_template_vars(k.kernel_tag, kargs_name)
    )
    traits_aliases = f"""
template <typename D_C>
using {k.name}_Traits = {traits_name}<{k.BLOCK_SIZE},
    opus::seq<{k.B_M}, {k.B_N}, {k.B_K}>,
    opus::tuple<{da}, {db}, D_C, fp32_t, {"unsigned char" if k.kernel_tag == "a8w8_mxscale" else "fp32_t"}>,
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
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale)
{{{{
    AITER_CHECK(XQ.dim() == 3 && WQ.dim() == 3 && Y.dim() == 3,
        "opus_gemm_a8w8_blockscale_launch: XQ/WQ/Y must be 3D");
    AITER_CHECK(XQ.dtype() == AITER_DTYPE_fp8 && WQ.dtype() == AITER_DTYPE_fp8,
        "opus_gemm_a8w8_blockscale_launch: expected fp8 XQ/WQ");
    AITER_CHECK(Y.dtype() == AITER_DTYPE_fp32,
        "opus_gemm_a8w8_blockscale_launch: expected fp32 Y");
    AITER_CHECK(XQ.is_contiguous() && WQ.is_contiguous() && Y.is_contiguous(),
        "opus_gemm_a8w8_blockscale_launch: XQ/WQ must be K-contiguous "
        "and Y must be contiguous");
    int batch = XQ.size(0);
    int M = XQ.size(1);
    int N = WQ.size(1);
    int K = XQ.size(2);
    AITER_CHECK(batch >= 1 && WQ.size(0) == batch && Y.size(0) == batch,
        "opus_gemm_a8w8_blockscale_launch: batch dimensions must match");
    AITER_CHECK(WQ.size(2) == K && Y.size(1) == M && Y.size(2) == N,
        "opus_gemm_a8w8_blockscale_launch: tensor shapes must be "
        "[B,M,K], [B,N,K], [B,M,N]");
    AITER_CHECK(M >= 1 && N >= 1 && K >= {k.B_K},
        "opus_gemm_a8w8_blockscale_launch: requires positive M/N and "
        "K >= {k.B_K}");
    AITER_CHECK(M % {k.GROUP_M} == 0 && N % {k.GROUP_N} == 0 &&
                    K % {k.GROUP_K} == 0,
        "opus_gemm_a8w8_blockscale_launch: M/N/K must be divisible by "
        "scale groups ",
        {k.GROUP_M}, "/", {k.GROUP_N}, "/", {k.GROUP_K});

    // The pipeline primes two K tiles, then advances in pairs. Rejecting a
    // one-tile or odd-tile launch here prevents a negative final tile and an
    // out-of-range prefetch in device code.
    int loops_ = (K + {k.B_K} - 1) / {k.B_K};
    AITER_CHECK(loops_ >= 2,
        "opus_gemm_a8w8_blockscale_launch: ceil_div(K, B_K)=", loops_,
        " must be >= 2 (K=", K, ", B_K=", {k.B_K}, ")");
    AITER_CHECK(loops_ % 2 == 0,
        "opus_gemm_a8w8_blockscale_launch: ceil_div(K, B_K)=", loops_,
        " must be even (prefetch constraint)");
    AITER_CHECK(K % 2 == 0,
        "opus_gemm_a8w8_blockscale_launch: K must be even; got K=", K);

    using Traits = {k.name}_Traits<D_C>;

    int GROUP_M = {k.GROUP_M};
    int GROUP_N = {k.GROUP_N};
    int GROUP_K = {k.GROUP_K};
    int num_groups_m = M / GROUP_M;
    int num_groups_n = N / GROUP_N;
    int num_groups_k = K / GROUP_K;

    AITER_CHECK(x_scale.dtype() == AITER_DTYPE_fp32 &&
                    w_scale.dtype() == AITER_DTYPE_fp32,
        "opus_gemm_a8w8_blockscale_launch: expects fp32 scales");
    AITER_CHECK(x_scale.is_contiguous() && w_scale.is_contiguous(),
        "opus_gemm_a8w8_blockscale_launch: expects contiguous scales");
    AITER_CHECK(x_scale.device_id == XQ.device_id &&
                    w_scale.device_id == XQ.device_id,
        "opus_gemm_a8w8_blockscale_launch: scales must be on the XQ device");
    const bool x_scale_2d = x_scale.dim() == 2 && batch == 1 &&
        x_scale.size(0) == M && x_scale.size(1) == num_groups_k;
    const bool x_scale_3d = x_scale.dim() == 3 &&
        x_scale.size(0) == batch && x_scale.size(1) == M &&
        x_scale.size(2) == num_groups_k;
    const bool w_scale_2d = w_scale.dim() == 2 && batch == 1 &&
        w_scale.size(0) == num_groups_n && w_scale.size(1) == num_groups_k;
    const bool w_scale_3d = w_scale.dim() == 3 &&
        w_scale.size(0) == batch && w_scale.size(1) == num_groups_n &&
        w_scale.size(2) == num_groups_k;
    AITER_CHECK(x_scale_2d || x_scale_3d,
        "opus_gemm_a8w8_blockscale_launch: x_scale must be "
        "[B,M,K/128], or [M,K/128] when B=1");
    AITER_CHECK(w_scale_2d || w_scale_3d,
        "opus_gemm_a8w8_blockscale_launch: w_scale must be "
        "[B,N/128,K/128], or [N/128,K/128] when B=1");

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

    kargs.ptr_sfa = x_scale.data_ptr();
    kargs.ptr_sfb = w_scale.data_ptr();
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
    record_one_instantiation(
        cg, k, kernel_func, kargs_name, A8W8_BLOCKSCALE_HOST_EXTRA
    )

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
    A16W16_KID_DISPATCH_TAGS,
    **_unused,
):
    """Emit a gfx950 A16W16 or A8W8 no-scale launcher."""
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
    shape_preamble = ""
    if is_a16w16_split_barrier:
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
    else:
        shape_preamble = """
    AITER_CHECK(XQ.dim() == 3 && WQ.dim() == 3 && Y.dim() == 3,
        "opus_gemm_a8w8_launch: XQ/WQ/Y must be 3D");
    AITER_CHECK(XQ.dtype() == AITER_DTYPE_fp8 && WQ.dtype() == AITER_DTYPE_fp8,
        "opus_gemm_a8w8_launch: expected fp8 XQ/WQ");
    AITER_CHECK(Y.dtype() == AITER_DTYPE_fp32,
        "opus_gemm_a8w8_launch: expected fp32 Y");
    AITER_CHECK(XQ.is_contiguous() && WQ.is_contiguous() && Y.is_contiguous(),
        "opus_gemm_a8w8_launch: XQ/WQ must be K-contiguous and Y must be "
        "contiguous");
"""
        k_check = f"""
    AITER_CHECK(batch >= 1 && WQ.size(0) == batch && Y.size(0) == batch,
        "opus_gemm_a8w8_launch: batch dimensions must match");
    AITER_CHECK(WQ.size(2) == K && Y.size(1) == M && Y.size(2) == N,
        "opus_gemm_a8w8_launch: tensor shapes must be "
        "[B,M,K], [B,N,K], [B,M,N]");
    int loops_ = (K + {k.B_K} - 1) / {k.B_K};
    AITER_CHECK(loops_ >= 2,
        "K=", K, " too small for B_K={k.B_K}, need K >= {min_k}");
    AITER_CHECK(loops_ % 2 == 0,
        "ceil_div(K, {k.B_K})=", loops_, " must be even (prefetch constraint)");
    AITER_CHECK(K % 2 == 0,
        "opus_gemm_a8w8_launch: K must be even; got K=", K);
    AITER_CHECK(M >= 1 && N >= 1, "M and N must be >= 1");
"""

    if k.kernel_tag in A16W16_KID_DISPATCH_TAGS:
        extra_param = (
            ",\n    std::optional<aiter_tensor_t> bias," "\n    int /*split_k*/"
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
    elif k.kernel_tag in A16W16_KID_DISPATCH_TAGS:
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
{shape_preamble}
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

    if k.kernel_tag in A16W16_KID_DISPATCH_TAGS:
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
    A16W16_LAUNCH_HOST_EXTRA,
    **_unused,
):
    """gfx950 a16w16_flatmm launcher emit."""
    _kargs_explicit_param, fwd_decl_kargs_tpl, fwd_decl_kargs_fnarg = (
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
    record_one_instantiation(cg, k, kernel_func, kargs_name, A16W16_LAUNCH_HOST_EXTRA)


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
    A16W16_WORKSPACE_LAUNCH_HOST_EXTRA,
    BIAS_HOST_VALIDATE,
    **_unused,
):
    """Emit a gfx950 split-K launcher using a caller-owned typed workspace."""
    workspace_dtype, _workspace_ptr_type, workspace_aiter_dtype = (
        splitk_workspace_type(k)
    )
    if workspace_dtype != "fp32_t":
        raise ValueError(
            f"gfx950 kid {k.name} declares {workspace_dtype} workspace, but "
            "the current gfx950 main/reduce kernels support fp32_t only"
        )
    _kargs_explicit_param, fwd_decl_kargs_tpl, fwd_decl_kargs_fnarg = (
        kargs_template_vars(k.kernel_tag, kargs_name)
    )
    has_oob_str = "true" if k.has_oob else "false"
    traits_aliases = f"""
template <typename D_C>
using {k.name}_Traits = {traits_name}<{k.BLOCK_SIZE},
    opus::seq<{k.B_M}, {k.B_N}, {k.B_K}>,
    opus::tuple<{da}, {db}, {workspace_dtype}, fp32_t, {da}>,
    opus::seq<{k.VEC_A}, {k.VEC_B}, {k.VEC_C}>,
    opus::seq<{k.W_M}, {k.W_N}, {k.W_K}>,
    {k.WG_PER_CU},
    false,
    {has_oob_str}>;
"""

    preamble = instance_impl_preamble('\n#include "opus_gemm_common.cuh"')
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
    aiter_tensor_t &workspace,
    std::optional<aiter_tensor_t> bias,
    int splitK)
{{{{
    static_assert(std::is_same<D_C, fp32_t>::value,
        "split_k launcher uses the fp32 launch specialization");

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

    int num_tiles_m = 1 + (M - 1) / {k.B_M};
    int num_tiles_n = 1 + (N - 1) / {k.B_N};
    const size_t padded_M_size = opus_checked_extent_product(
        {{static_cast<size_t>(num_tiles_m), static_cast<size_t>({k.B_M})}},
        "{k.name}");
    const size_t padded_N_size = opus_checked_extent_product(
        {{static_cast<size_t>(num_tiles_n), static_cast<size_t>({k.B_N})}},
        "{k.name}");
    const size_t workspace_slice_numel = opus_checked_extent_product(
        {{padded_M_size, padded_N_size}}, "{k.name}");
    AITER_CHECK(padded_M_size <= static_cast<size_t>(std::numeric_limits<int>::max())
                    && padded_N_size <= static_cast<size_t>(std::numeric_limits<int>::max())
                    && workspace_slice_numel <= static_cast<size_t>(std::numeric_limits<int>::max()),
        "{k.name}: padded workspace extents exceed 32-bit kernel stride limits");
    int padded_M = static_cast<int>(padded_M_size);
    int padded_N = static_cast<int>(padded_N_size);

    const size_t required_numel = opus_checked_extent_product(
        {{static_cast<size_t>(split_k), static_cast<size_t>(batch),
          workspace_slice_numel}},
        "{k.name}");
    void* workspace_ptr_ = opus_validate_workspace(
        workspace, XQ, {workspace_aiter_dtype}, required_numel, 16, "{k.name}");
    auto stream = aiter::getCurrentHIPStream();

    {kargs_name} kargs{{{{}}}};
    kargs.ptr_a         = XQ.data_ptr();
    kargs.ptr_b         = WQ.data_ptr();
    kargs.ptr_ws        = workspace_ptr_;
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
    kargs.stride_ws_batch = static_cast<int>(workspace_slice_numel);
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
                    workspace_ptr_,
                    reinterpret_cast<__bf16*>(Y.data_ptr()),
                    split_k, M, N, batch, padded_M, padded_N,
                    reinterpret_cast<const __bf16*>(ptr_bias_),
                    stride_bias_batch_);
        }}}} else {{{{
            splitk_reduce_kernel<REDUCE_VEC, REDUCE_BS, __bf16, false, __bf16, {has_oob_str}>
                <<<grid_reduce, block_reduce, 0, stream>>>(
                    workspace_ptr_,
                    reinterpret_cast<__bf16*>(Y.data_ptr()),
                    split_k, M, N, batch, padded_M, padded_N,
                    nullptr, 0);
        }}}}
    }}}} else {{{{
        if (bias.has_value()) {{{{
            splitk_reduce_kernel<REDUCE_VEC, REDUCE_BS, float, true, float, {has_oob_str}>
                <<<grid_reduce, block_reduce, 0, stream>>>(
                    workspace_ptr_,
                    reinterpret_cast<float*>(Y.data_ptr()),
                    split_k, M, N, batch, padded_M, padded_N,
                    reinterpret_cast<const float*>(ptr_bias_),
                    stride_bias_batch_);
        }}}} else {{{{
            splitk_reduce_kernel<REDUCE_VEC, REDUCE_BS, float, false, float, {has_oob_str}>
                <<<grid_reduce, block_reduce, 0, stream>>>(
                    workspace_ptr_,
                    reinterpret_cast<float*>(Y.data_ptr()),
                    split_k, M, N, batch, padded_M, padded_N,
                    nullptr, 0);
        }}}}
    }}}}

}}}}
#endif // launcher only on regular host pass
"""
    Path(os.path.join(cg.impl_path, f"{k.name}.cuh")).write_text(INSTANCE_IMPL)
    record_one_instantiation(
        cg,
        k,
        kernel_func,
        kargs_name,
        A16W16_WORKSPACE_LAUNCH_HOST_EXTRA,
    )


def _assert_m_align(k, tile_mult):
    """Tie the declared m_align to the M guard the launcher body actually emits.

    `tile_mult` is the B_M multiple the body below hardcodes in its AITER_CHECK,
    or 0 when it emits no M check because the kernel masks the partial tile.
    OpusGemmInstance.m_align is what the tuner's candidate filter and the
    runtime's padded-M lookup read, so a guard edit that forgets to update
    _BMM_M_ALIGN_TILES must fail the build rather than silently teach the two
    consumers a wrong alignment.
    """
    expect = k.B_M * tile_mult if tile_mult else 1
    assert k.m_align == expect, (
        f"{k.name}: launcher guards M % {expect} == 0 but m_align says "
        f"{k.m_align}; fix _BMM_M_ALIGN_TILES in opus_gemm_common.py"
    )


# Body of the a8w8_mxscale BMM flatmm split-K launcher (mmajor layout), a
# faithful port of opus_bmm_a8w8_mxscale_flatmm_splitk_impl() in
# opus_bmm.cu. Written with @@TOKEN@@ placeholders + .replace() (NOT an
# f-string) so the C++ body keeps plain single braces and stays trivially
# reviewable against the hand-written original.
#
# Templated on D_C only to satisfy the codegen host-decl machinery
# (one <fp32_t> instantiation); the body ignores D_C and branches on Y.dtype()
# at runtime with native __bf16/float, exactly like the original. The fused
# reduce path (splitK==2 counter variant) is intentionally NOT ported -- the
# fused-reduce kid stays monolithic in opus_bmm.cu.
_BMM_MXSCALE_SPLITK_LAUNCHER_BODY = r"""
#if !defined(__HIP_DEVICE_COMPILE__) && !defined(__HIPCC_RTC__)
// mmajor: O/Y are [M, batch, *] (dim0=M, dim1=batch); wo_a stays batch-major
// [batch, N, K]. Caller (opus_bmm.cu switch) does dtype/arch/common checks.
template <typename D_C>
void
@@NAME@@(
    aiter_tensor_t &O,
    aiter_tensor_t &wo_a,
    aiter_tensor_t &Y,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    std::optional<aiter_tensor_t> workspace,
    int splitK)
{
  using Traits = @@NAME@@_Traits;
  constexpr bool DIRECT_ONLY = @@DIRECT@@;
  constexpr bool PREFETCH_SCALE = @@PREFETCH@@;
  constexpr bool PRELOAD_SF_LDS = @@PRELOAD@@;

  AITER_CHECK(splitK >= 1, "splitK must be >= 1");
  if constexpr (DIRECT_ONLY) {
    AITER_CHECK(splitK == 1, "@@NAME@@ consumer-self-load kernel requires splitK == 1");
  }

  const int M = O.size(0);
  const int batch = O.size(1);
  const int N = wo_a.size(1);
  const int K = O.size(2);
  // No M alignment at any tile size: A and SFA are bounded to the tile's valid row
  // count, the split_k==1 store bounds C the same way, split_k>1 partials go to a
  // workspace sized for padded_M, and both reducers touch only rows < M.
  AITER_CHECK(N % Traits::B_N == 0,
              "@@NAME@@ requires N % ", Traits::B_N, " == 0, got ", N);
  AITER_CHECK(K % Traits::B_K == 0,
              "@@NAME@@ requires K % ", Traits::B_K, " == 0, got ", K);

  const int split_k = splitK;
  const bool no_split_k = (split_k == 1);
  const int total_iters = K / Traits::B_K;
  const int iters_full = (total_iters + split_k - 1) / split_k;
  const int last_loops = total_iters - (split_k - 1) * iters_full;
  AITER_CHECK(last_loops >= Traits::prefetch_k_iter,
              "@@NAME@@ requires every split to have at least ",
              Traits::prefetch_k_iter, " K-tiles; K=", K,
              " gives total_iters=", total_iters, ", splitK=", split_k,
              ", last split loops=", last_loops);

  const int num_tiles_m = (M + Traits::B_M - 1) / Traits::B_M;
  const int num_tiles_n = (N + Traits::B_N - 1) / Traits::B_N;
  const int padded_M = num_tiles_m * Traits::B_M;
  const int padded_N = num_tiles_n * Traits::B_N;
  const size_t required_numel = (size_t)split_k * (size_t)batch
                              * (size_t)padded_M * (size_t)padded_N;

  auto stream = aiter::getCurrentHIPStream();

  opus_gemm_scale_splitk_kargs_gfx950 kargs{};
  kargs.ptr_a = O.data_ptr();
  kargs.ptr_b = wo_a.data_ptr();
  kargs.ptr_ws = nullptr;
  kargs.m = M; kargs.n = N; kargs.k = K; kargs.batch = batch;
  kargs.split_k = split_k;
  kargs.stride_a = (int)O.stride(0);
  kargs.stride_b = (int)wo_a.stride(1);
  kargs.stride_ws = padded_N;
  kargs.stride_a_batch = (int)O.stride(1);
  kargs.stride_b_batch = (int)wo_a.stride(0);
  kargs.stride_ws_batch = padded_M * padded_N;
  kargs.ptr_sfa = x_scale.data_ptr();
  kargs.ptr_sfb = w_scale.data_ptr();
  kargs.stride_sfa = (int)x_scale.stride(0);
  kargs.stride_sfa_batch = (int)x_scale.stride(1);
  kargs.stride_sfb = (int)w_scale.stride(1);
  kargs.stride_sfb_batch = (int)w_scale.stride(0);

  dim3 grid_main(num_tiles_m * num_tiles_n * split_k, 1, batch);
  dim3 block_main(Traits::BLOCK_SIZE);
  if (no_split_k) {
    AITER_CHECK(!workspace.has_value(),
                "@@NAME@@ splitK == 1 does not use workspace");
    kargs.ptr_c = Y.data_ptr();
    kargs.stride_c = (int)Y.stride(0);
    kargs.stride_c_batch = (int)Y.stride(1);
    if (Y.dtype() == AITER_DTYPE_bf16) {
      @@KERNEL@@<Traits, __bf16, DIRECT_ONLY, PREFETCH_SCALE, PRELOAD_SF_LDS>
          <<<grid_main, block_main, 0, stream>>>(kargs);
    } else {
      @@KERNEL@@<Traits, float, DIRECT_ONLY, PREFETCH_SCALE, PRELOAD_SF_LDS>
          <<<grid_main, block_main, 0, stream>>>(kargs);
    }
    return;
  }

  if constexpr (!DIRECT_ONLY) {
    AITER_CHECK(workspace.has_value(),
                "@@NAME@@ splitK > 1 requires workspace");
    void* workspace_ptr = opus_validate_workspace(
        workspace.value(), O, AITER_DTYPE_fp32, required_numel, 16, "@@NAME@@");
    kargs.ptr_ws = workspace_ptr;

    // Pass all 4 template args explicitly (D_OUT=void: the split-K main kernel
    // writes an fp32 workspace, so its output dtype is irrelevant; the reduce
    // kernel casts to the runtime Y dtype). The fused host TU only sees a
    // no-default forward decl of @@KERNEL@@, so relying on the template's
    // default args here would fail overload resolution ("no matching function").
    @@KERNEL@@<Traits, void, DIRECT_ONLY, PREFETCH_SCALE, PRELOAD_SF_LDS>
        <<<grid_main, block_main, 0, stream>>>(kargs);

    constexpr int REDUCE_VEC = 8;
    constexpr int REDUCE_BS = 128;
    dim3 grid_reduce((N + REDUCE_VEC * REDUCE_BS - 1) / (REDUCE_VEC * REDUCE_BS),
                     batch * M, 1);
    dim3 block_reduce(REDUCE_BS);
    const int y_stride_c = (int)Y.stride(0);
    const int y_stride_c_batch = (int)Y.stride(1);
    if (Y.dtype() == AITER_DTYPE_bf16) {
      opus_bmm_splitk_reduce_kernel<__bf16, REDUCE_VEC, REDUCE_BS>
          <<<grid_reduce, block_reduce, 0, stream>>>(
              workspace_ptr, reinterpret_cast<__bf16*>(Y.data_ptr()),
              split_k, M, N, batch, padded_M, padded_N,
              y_stride_c, y_stride_c_batch);
    } else {
      opus_bmm_splitk_reduce_kernel<float, REDUCE_VEC, REDUCE_BS>
          <<<grid_reduce, block_reduce, 0, stream>>>(
              workspace_ptr, reinterpret_cast<float*>(Y.data_ptr()),
              split_k, M, N, batch, padded_M, padded_N,
              y_stride_c, y_stride_c_batch);
    }
  }
}
#endif // launcher only on regular host pass
"""


def gen_bmm_mxscale_flatmm_splitk_instance(
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
    **_unused,
):
    """gfx950 a8w8_mxscale BMM flatmm split-K launcher emit.

    Differs from the GEMM emitters:
      * traits alias is NOT templated on D_C (fp32 workspace is fixed); the
        launcher is templated on D_C only for the host-decl machinery.
      * launcher signature is (O, wo_a, Y, x_scale, w_scale, int splitK) with
        the mmajor layout, matching opus_bmm.cu's _impl.
      * custom device-instantiation matrix over the kernel's (D_OUT, DIRECT_ONLY,
        PREFETCH_SCALE) template params (the standard record_one_instantiation
        assumes a single-template-arg <Traits<dtype>> kernel).
      * the split-K reduce kernel (opus_bmm_splitk_reduce_kernel) is declared in
        the a8w8_scale traits header and instantiated once in opus_bmm.cu, so it
        is NOT re-instantiated here.
    """
    _, fwd_decl_kargs_tpl, fwd_decl_kargs_fnarg = kargs_template_vars(
        k.kernel_tag, kargs_name
    )

    # Non-templated traits alias: fp32 split-K workspace is fixed; the workspace
    # tuple slot 4 (scale) is `unsigned char` for the e8m0 mxscale path.
    traits_aliases = f"""
using {k.name}_Traits = {traits_name}<{k.BLOCK_SIZE},
    opus::seq<{k.B_M}, {k.B_N}, {k.B_K}>,
    opus::tuple<{da}, {db}, fp32_t, fp32_t, unsigned char>,
    opus::seq<{k.VEC_A}, {k.VEC_B}, {k.VEC_C}>,
    opus::seq<{k.GROUP_M}, {k.GROUP_N}, {k.GROUP_K}>,
    {k.WG_PER_CU}>;
"""

    preamble = instance_impl_preamble('\n#include "opus_gemm_common.cuh"')
    host_tu_split = instance_impl_host_tu_split(
        traits_header,
        pipeline_header,
        fwd_decl_kargs_tpl,
        kernel_func,
        fwd_decl_kargs_fnarg,
    )

    # Forward-declare the split-K reduce kernel. On the fused host TU pass
    # host_tu_split only pulls in the (light) traits header -- not the pipeline
    # header that defines this kernel -- so the launcher body's <<<...>>> call
    # needs a visible declaration. On the non-fused device pass the pipeline
    # header (via splitk_reduce_gfx950.cuh) provides a compatible definition, so
    # this is just a harmless redeclaration there.
    reduce_fwd_decl = """
template <typename D_OUT, int VEC, int BLOCK>
__global__ void opus_bmm_splitk_reduce_kernel(
    const void* __restrict__ workspace,
    D_OUT* __restrict__ out,
    int split_k, int M, int N, int batch,
    int padded_M, int padded_N,
    int stride_c, int stride_c_batch);
"""

    launcher = (
        _BMM_MXSCALE_SPLITK_LAUNCHER_BODY.replace("@@NAME@@", k.name)
        .replace("@@KERNEL@@", kernel_func)
        .replace("@@DIRECT@@", "true" if k.direct_only else "false")
        .replace("@@PREFETCH@@", "true" if k.prefetch_scale else "false")
        .replace("@@PRELOAD@@", "true" if k.preload_sf else "false")
    )

    INSTANCE_IMPL = (
        f"{preamble}\n{host_tu_split}\n{reduce_fwd_decl}\n{traits_aliases}\n{launcher}"
    )
    Path(os.path.join(cg.impl_path, f"{k.name}.cuh")).write_text(INSTANCE_IMPL)

    # Host instantiation(s): launcher templated on D_C; a single <fp32_t> stub.
    # (XQ/WQ/Y positional names in _make_host_decl map to O/wo_a/Y by type.)
    host_extra = (
        ",\n    aiter_tensor_t &x_scale,"
        "\n    aiter_tensor_t &w_scale,"
        "\n    std::optional<aiter_tensor_t> workspace,"
        "\n    int splitK"
    )
    for dtype in k.output_dtypes:
        host_decl = (
            f"template void\n"
            f"{k.name}<{dtype}>(\n"
            f"    aiter_tensor_t &O,\n"
            f"    aiter_tensor_t &wo_a,\n"
            f"    aiter_tensor_t &Y{host_extra});\n"
        )
        cg._host_instantiations.append(
            {"kid_name": k.name, "dtype": dtype, "host_decl": host_decl}
        )

    # Device instantiation matrix: split-1 direct-store variants for both Y
    # dtypes, plus (non-direct kids only) the fp32-workspace variant used by the
    # split-K > 1 path (kernel default D_OUT=void).
    direct = "true" if k.direct_only else "false"
    prefetch = "true" if k.prefetch_scale else "false"
    preload = "true" if k.preload_sf else "false"

    def _dev(dtype_tag, d_out, dir_flag, pfk_flag):
        decl = (
            f"template __global__ void {kernel_func}<\n"
            f"    {k.name}_Traits, {d_out}, {dir_flag}, {pfk_flag}, {preload}>({kargs_name});\n"
        )
        cg._device_instantiations.append(
            {"kid_name": k.name, "dtype": dtype_tag, "device_decl": decl}
        )

    _dev("bf16", "__bf16", direct, prefetch)
    _dev("fp32", "float", direct, prefetch)
    if not k.direct_only:
        # Split-K > 1 workspace path: host launches <Traits, void, DIRECT_ONLY,
        # PREFETCH_SCALE, PRELOAD_SF_LDS>. DIRECT_ONLY is false here (direct kids
        # never take the workspace path), but PREFETCH_SCALE / PRELOAD_SF_LDS must
        # match the kid, else the <void, false, ...> instantiation is missing for
        # prefetch/preload kids -> undefined symbol at load.
        _dev("void", "void", "false", prefetch)


_BMM_MXSCALE_MINTERLEAVE_LAUNCHER_BODY = r"""
#if !defined(__HIP_DEVICE_COMPILE__) && !defined(__HIPCC_RTC__)
// M-tile interleaved launcher: MI=2 consecutive M tiles per WG share the B
// stream (requires M % (MI*B_M) == 0). splitK arg is unused (must be 1). mmajor:
// O/Y are [M, batch, *] (dim0=M, dim1=batch); wo_a stays batch-major [batch,N,K].
// Caller (opus_bmm.cu dispatch) does dtype/arch/common checks.
template <typename D_C>
void
@@NAME@@(
    aiter_tensor_t &O,
    aiter_tensor_t &wo_a,
    aiter_tensor_t &Y,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    std::optional<aiter_tensor_t> workspace,
    int splitK)
{
  using Traits = @@NAME@@_Traits;
  constexpr bool SKIP_SCALE_WAIT = @@SKIP@@;
  constexpr int MI = 2;
  AITER_CHECK(splitK == 1, "@@NAME@@ requires splitK == 1");
  AITER_CHECK(!workspace.has_value(), "@@NAME@@ does not use workspace");

  const int M = O.size(0);
  const int batch = O.size(1);
  const int N = wo_a.size(1);
  const int K = O.size(2);
  AITER_CHECK(M % (MI * Traits::B_M) == 0,
              "@@NAME@@ requires M % ", (MI * Traits::B_M), " == 0, got ", M);
  AITER_CHECK(N % Traits::B_N == 0,
              "@@NAME@@ requires N % ", Traits::B_N, " == 0, got ", N);
  AITER_CHECK(K % Traits::B_K == 0,
              "@@NAME@@ requires K % ", Traits::B_K, " == 0, got ", K);
  const int total_iters = K / Traits::B_K;
  AITER_CHECK(total_iters >= Traits::prefetch_k_iter,
              "@@NAME@@ requires at least ", Traits::prefetch_k_iter,
              " K-tiles, got ", total_iters);

  auto stream = aiter::getCurrentHIPStream();

  opus_gemm_scale_splitk_kargs_gfx950 kargs{};
  kargs.ptr_a = O.data_ptr();
  kargs.ptr_b = wo_a.data_ptr();
  kargs.ptr_ws = nullptr;
  kargs.m = M; kargs.n = N; kargs.k = K; kargs.batch = batch;
  const int num_tiles_m = M / Traits::B_M;
  const int num_tiles_n = N / Traits::B_N;
  kargs.split_k = MI;
  kargs.stride_a = (int)O.stride(0);
  kargs.stride_b = (int)wo_a.stride(1);
  kargs.stride_a_batch = (int)O.stride(1);
  kargs.stride_b_batch = (int)wo_a.stride(0);
  kargs.ptr_sfa = x_scale.data_ptr();
  kargs.ptr_sfb = w_scale.data_ptr();
  kargs.stride_sfa = (int)x_scale.stride(0);
  kargs.stride_sfa_batch = (int)x_scale.stride(1);
  kargs.stride_sfb = (int)w_scale.stride(1);
  kargs.stride_sfb_batch = (int)w_scale.stride(0);
  kargs.ptr_c = Y.data_ptr();
  kargs.stride_c = (int)Y.stride(0);
  kargs.stride_c_batch = (int)Y.stride(1);

  const int split_m = num_tiles_m / MI;          // M-tile groups (WGs along M)
  constexpr int NUM_XCD = 8;
  const int m_grp_per_xcd = (split_m + NUM_XCD - 1) / NUM_XCD;
  kargs.stride_ws = split_m;
  kargs.stride_ws_batch = m_grp_per_xcd;
  dim3 grid_main(NUM_XCD * m_grp_per_xcd * num_tiles_n, 1, batch);
  dim3 block_main(Traits::BLOCK_SIZE);
  if (Y.dtype() == AITER_DTYPE_bf16) {
    @@KERNEL@@<Traits, __bf16, SKIP_SCALE_WAIT>
        <<<grid_main, block_main, 0, stream>>>(kargs);
  } else {
    @@KERNEL@@<Traits, float, SKIP_SCALE_WAIT>
        <<<grid_main, block_main, 0, stream>>>(kargs);
  }
}
#endif // launcher only on regular host pass
"""


def gen_bmm_mxscale_minterleave_instance(
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
    **_unused,
):
    """gfx950 a8w8_mxscale BMM M-tile-interleaved launcher emit (kids 162/163).

    Sibling of gen_bmm_mxscale_flatmm_splitk_instance:
      * kernel template is <Traits, D_OUT, bool SKIP_SCALE_WAIT> (no DIRECT_ONLY/
        PREFETCH_SCALE/PRELOAD_SF_LDS axes, no split-K workspace/reduce path).
      * MI=2 is baked in the launcher; splitK is ignored (must be 1).
      * device instantiation matrix is just (D_OUT in {bf16, float}) x the kid's
        fixed SKIP_SCALE_WAIT flag.
    """
    _, fwd_decl_kargs_tpl, fwd_decl_kargs_fnarg = kargs_template_vars(
        k.kernel_tag, kargs_name
    )

    # Non-templated traits alias: identical geometry/tuple to the flatmm split-K
    # family (fp32 workspace slot, unsigned char e8m0 scale slot).
    traits_aliases = f"""
using {k.name}_Traits = {traits_name}<{k.BLOCK_SIZE},
    opus::seq<{k.B_M}, {k.B_N}, {k.B_K}>,
    opus::tuple<{da}, {db}, fp32_t, fp32_t, unsigned char>,
    opus::seq<{k.VEC_A}, {k.VEC_B}, {k.VEC_C}>,
    opus::seq<{k.GROUP_M}, {k.GROUP_N}, {k.GROUP_K}>,
    {k.WG_PER_CU}>;
"""

    preamble = instance_impl_preamble()
    host_tu_split = instance_impl_host_tu_split(
        traits_header,
        pipeline_header,
        fwd_decl_kargs_tpl,
        kernel_func,
        fwd_decl_kargs_fnarg,
    )

    launcher = (
        _BMM_MXSCALE_MINTERLEAVE_LAUNCHER_BODY.replace("@@NAME@@", k.name)
        .replace("@@KERNEL@@", kernel_func)
        .replace("@@SKIP@@", "true" if k.skip_scale_wait else "false")
    )

    INSTANCE_IMPL = f"{preamble}\n{host_tu_split}\n{traits_aliases}\n{launcher}"
    Path(os.path.join(cg.impl_path, f"{k.name}.cuh")).write_text(INSTANCE_IMPL)

    # Host instantiation: launcher templated on D_C; single <fp32_t> stub.
    host_extra = (
        ",\n    aiter_tensor_t &x_scale,"
        "\n    aiter_tensor_t &w_scale,"
        "\n    std::optional<aiter_tensor_t> workspace,"
        "\n    int splitK"
    )
    for dtype in k.output_dtypes:
        host_decl = (
            f"template void\n"
            f"{k.name}<{dtype}>(\n"
            f"    aiter_tensor_t &O,\n"
            f"    aiter_tensor_t &wo_a,\n"
            f"    aiter_tensor_t &Y{host_extra});\n"
        )
        cg._host_instantiations.append(
            {"kid_name": k.name, "dtype": dtype, "host_decl": host_decl}
        )

    # Device instantiation matrix: <Traits, D_OUT, SKIP_SCALE_WAIT> for both Y
    # dtypes (the kid's SKIP_SCALE_WAIT is fixed).
    skip = "true" if k.skip_scale_wait else "false"

    def _dev(dtype_tag, d_out):
        decl = (
            f"template __global__ void {kernel_func}<\n"
            f"    {k.name}_Traits, {d_out}, {skip}>({kargs_name});\n"
        )
        cg._device_instantiations.append(
            {"kid_name": k.name, "dtype": dtype_tag, "device_decl": decl}
        )

    _dev("bf16", "__bf16")
    _dev("fp32", "float")


def _bmm_specialized_traits_alias(k, traits_name, da, db):
    """Non-templated traits alias shared by all a8w8_mxscale BMM specialized
    pipelines (fp32 workspace slot, unsigned char e8m0 scale slot)."""
    return f"""
using {k.name}_Traits = {traits_name}<{k.BLOCK_SIZE},
    opus::seq<{k.B_M}, {k.B_N}, {k.B_K}>,
    opus::tuple<{da}, {db}, fp32_t, fp32_t, unsigned char>,
    opus::seq<{k.VEC_A}, {k.VEC_B}, {k.VEC_C}>,
    opus::seq<{k.GROUP_M}, {k.GROUP_N}, {k.GROUP_K}>,
    {k.WG_PER_CU}>;
"""


def _emit_bmm_specialized(
    cg,
    k,
    kernel_func,
    traits_name,
    kargs_name,
    da,
    db,
    preamble,
    host_tu_split,
    launcher,
    dev_flag_suffix,
    emit_device=True,
):
    """Shared tail for BMM specialized-pipeline emits: write impl/{name}.cuh
    (preamble + host-TU split + traits alias + inlined launcher), then register
    one <fp32_t> host stub and the (bf16, fp32) device instantiation pair.

    dev_flag_suffix is the comma-prefixed template-arg tail after D_OUT in the
    kernel instantiation (e.g. ", true, false, ..." for the wave families,
    "" for wave8n2).

    emit_device=False emits only the host launcher (used by mouter_tunable,
    which reuses the identical gemm_..._mouter_kernel<wg1, D_OUT, SKIP>
    specializations already emitted by the mouter family -- emitting them again
    under a different alias name would be a duplicate-symbol ODR violation).
    """
    traits_aliases = _bmm_specialized_traits_alias(k, traits_name, da, db)
    INSTANCE_IMPL = f"{preamble}\n{host_tu_split}\n{traits_aliases}\n{launcher}"
    Path(os.path.join(cg.impl_path, f"{k.name}.cuh")).write_text(INSTANCE_IMPL)

    host_extra = (
        ",\n    aiter_tensor_t &x_scale,"
        "\n    aiter_tensor_t &w_scale,"
        "\n    std::optional<aiter_tensor_t> workspace,"
        "\n    int splitK"
    )
    for dtype in k.output_dtypes:
        host_decl = (
            f"template void\n"
            f"{k.name}<{dtype}>(\n"
            f"    aiter_tensor_t &O,\n"
            f"    aiter_tensor_t &wo_a,\n"
            f"    aiter_tensor_t &Y{host_extra});\n"
        )
        cg._host_instantiations.append(
            {"kid_name": k.name, "dtype": dtype, "host_decl": host_decl}
        )

    if not emit_device:
        return
    for dtype_tag, d_out in (("bf16", "__bf16"), ("fp32", "float")):
        decl = (
            f"template __global__ void {kernel_func}<\n"
            f"    {k.name}_Traits, {d_out}{dev_flag_suffix}>({kargs_name});\n"
        )
        cg._device_instantiations.append(
            {"kid_name": k.name, "dtype": dtype_tag, "device_decl": decl}
        )


# Common launcher signature + shared checks/kargs preamble. mmajor: O/Y are
# [M, batch, *] (dim0=M, dim1=batch); wo_a stays batch-major. Caller does the
# dtype/arch/common checks (see opus_bmm.cu dispatch).
_BMM_SPEC_SIG = r"""
#if !defined(__HIP_DEVICE_COMPILE__) && !defined(__HIPCC_RTC__)
template <typename D_C>
void
@@NAME@@(
    aiter_tensor_t &O,
    aiter_tensor_t &wo_a,
    aiter_tensor_t &Y,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    std::optional<aiter_tensor_t> workspace,
    int @@SPLITK_ARG@@)
{
  using Traits = @@NAME@@_Traits;
  AITER_CHECK(!workspace.has_value(), "@@NAME@@ does not use workspace");
"""

_BMM_SPEC_KARGS = r"""
  auto stream = aiter::getCurrentHIPStream();

  opus_gemm_scale_splitk_kargs_gfx950 kargs{};
  kargs.ptr_a = O.data_ptr();
  kargs.ptr_b = wo_a.data_ptr();
  kargs.ptr_ws = nullptr;
  kargs.m = M; kargs.n = N; kargs.k = K; kargs.batch = batch;
  kargs.stride_a = (int)O.stride(0);
  kargs.stride_b = (int)wo_a.stride(1);
  kargs.stride_ws = N;
  kargs.stride_a_batch = (int)O.stride(1);
  kargs.stride_b_batch = (int)wo_a.stride(0);
  kargs.stride_ws_batch = M * N;
  kargs.ptr_sfa = x_scale.data_ptr();
  kargs.ptr_sfb = w_scale.data_ptr();
  kargs.stride_sfa = (int)x_scale.stride(0);
  kargs.stride_sfa_batch = (int)x_scale.stride(1);
  kargs.stride_sfb = (int)w_scale.stride(1);
  kargs.stride_sfb_batch = (int)w_scale.stride(0);
  kargs.ptr_c = Y.data_ptr();
  kargs.stride_c = (int)Y.stride(0);
  kargs.stride_c_batch = (int)Y.stride(1);
"""

# ---- wave8n2 (kid 132) ----
_BMM_WAVE8N2_LAUNCHER_BODY = (
    _BMM_SPEC_SIG.replace("@@SPLITK_ARG@@", "splitK")
    + r"""  AITER_CHECK(splitK == 1, "@@NAME@@ requires splitK == 1");
  const int M = O.size(0);
  const int batch = O.size(1);
  const int N = wo_a.size(1);
  const int K = O.size(2);
  constexpr int LOGICAL_B_N = Traits::B_N * 2;
  AITER_CHECK(M % Traits::B_M == 0,
              "@@NAME@@ requires M % ", Traits::B_M, " == 0, got ", M);
  AITER_CHECK(N % LOGICAL_B_N == 0,
              "@@NAME@@ requires N % ", LOGICAL_B_N, " == 0, got ", N);
  AITER_CHECK(K % Traits::B_K == 0,
              "@@NAME@@ requires K % ", Traits::B_K, " == 0, got ", K);
"""
    + _BMM_SPEC_KARGS.replace(
        "kargs.m = M; kargs.n = N; kargs.k = K; kargs.batch = batch;",
        "kargs.m = M; kargs.n = N; kargs.k = K; kargs.batch = batch;\n  kargs.split_k = 1;",
    )
    + r"""
  const int num_tiles_m = M / Traits::B_M;
  const int num_tiles_n = N / LOGICAL_B_N;
  dim3 grid_main(num_tiles_m * num_tiles_n, 1, batch);
  dim3 block_main(512);
  if (Y.dtype() == AITER_DTYPE_bf16) {
    @@KERNEL@@<Traits, __bf16>
        <<<grid_main, block_main, 0, stream>>>(kargs);
  } else {
    @@KERNEL@@<Traits, float>
        <<<grid_main, block_main, 0, stream>>>(kargs);
  }
}
#endif // launcher only on regular host pass
"""
)


def gen_bmm_mxscale_wave8n2_instance(
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
    **_unused,
):
    _, tpl, fn = kargs_template_vars(k.kernel_tag, kargs_name)
    launcher = _BMM_WAVE8N2_LAUNCHER_BODY.replace("@@NAME@@", k.name).replace(
        "@@KERNEL@@", kernel_func
    )
    _emit_bmm_specialized(
        cg,
        k,
        kernel_func,
        traits_name,
        kargs_name,
        da,
        db,
        instance_impl_preamble(),
        instance_impl_host_tu_split(
            traits_header, pipeline_header, tpl, kernel_func, fn
        ),
        launcher,
        "",
    )


def _cppbool(v):
    return "true" if v else "false"


# ---- wave4m2_selfload (kids 134/142/148) ----
_BMM_WAVE4M2_LAUNCHER_BODY = (
    _BMM_SPEC_SIG.replace("@@SPLITK_ARG@@", "splitK")
    + r"""  AITER_CHECK(splitK == 1, "@@NAME@@ requires splitK == 1");
  constexpr bool SKIP_SCALE_WAIT = @@SSW@@;
  constexpr bool PACK_SCALE_ON_DEMAND = @@PSOD@@;
  const int M = O.size(0);
  const int batch = O.size(1);
  const int N = wo_a.size(1);
  const int K = O.size(2);
  constexpr int LOGICAL_B_M = Traits::B_M * 2;
  AITER_CHECK(M % LOGICAL_B_M == 0,
              "@@NAME@@ requires M % ", LOGICAL_B_M, " == 0, got ", M);
  AITER_CHECK(N % Traits::B_N == 0,
              "@@NAME@@ requires N % ", Traits::B_N, " == 0, got ", N);
  AITER_CHECK(K % Traits::B_K == 0,
              "@@NAME@@ requires K % ", Traits::B_K, " == 0, got ", K);
"""
    + _BMM_SPEC_KARGS.replace(
        "kargs.m = M; kargs.n = N; kargs.k = K; kargs.batch = batch;",
        "kargs.m = M; kargs.n = N; kargs.k = K; kargs.batch = batch;\n  kargs.split_k = 1;",
    )
    + r"""
  const int num_tiles_m = M / LOGICAL_B_M;
  const int num_tiles_n = N / Traits::B_N;
  dim3 grid_main(num_tiles_m * num_tiles_n, 1, batch);
  dim3 block_main(256);
  if (Y.dtype() == AITER_DTYPE_bf16) {
    @@KERNEL@@<
        Traits, __bf16, SKIP_SCALE_WAIT, PACK_SCALE_ON_DEMAND>
        <<<grid_main, block_main, 0, stream>>>(kargs);
  } else {
    @@KERNEL@@<
        Traits, float, SKIP_SCALE_WAIT, PACK_SCALE_ON_DEMAND>
        <<<grid_main, block_main, 0, stream>>>(kargs);
  }
}
#endif // launcher only on regular host pass
"""
)


def gen_bmm_mxscale_wave4m2_selfload_instance(
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
    **_unused,
):
    _, tpl, fn = kargs_template_vars(k.kernel_tag, kargs_name)
    launcher = (
        _BMM_WAVE4M2_LAUNCHER_BODY.replace("@@NAME@@", k.name)
        .replace("@@KERNEL@@", kernel_func)
        .replace("@@SSW@@", _cppbool(k.skip_scale_wait))
        .replace("@@PSOD@@", _cppbool(k.pack_scale_on_demand))
    )
    suffix = f", {_cppbool(k.skip_scale_wait)}, {_cppbool(k.pack_scale_on_demand)}"
    _emit_bmm_specialized(
        cg,
        k,
        kernel_func,
        traits_name,
        kargs_name,
        da,
        db,
        instance_impl_preamble(),
        instance_impl_host_tu_split(
            traits_header, pipeline_header, tpl, kernel_func, fn
        ),
        launcher,
        suffix,
    )


# ---- mouter (kids 131/144) + mouter_tunable (kids 160/161) ----
# Shared persistent-mouter kernel; the two families differ only in how m_per_wg
# is derived (heuristic vs API-splitK sweep). XCD-aware grid remap is identical.
_BMM_MOUTER_CHECKS = r"""  constexpr bool SKIP_SCALE_WAIT = @@SSW@@;
  const int M = O.size(0);
  const int batch = O.size(1);
  const int N = wo_a.size(1);
  const int K = O.size(2);
  AITER_CHECK(M % Traits::B_M == 0,
              "@@NAME@@ requires M % ", Traits::B_M, " == 0, got ", M);
  AITER_CHECK(N % Traits::B_N == 0,
              "@@NAME@@ requires N % ", Traits::B_N, " == 0, got ", N);
  AITER_CHECK(K % Traits::B_K == 0,
              "@@NAME@@ requires K % ", Traits::B_K, " == 0, got ", K);
  const int total_iters = K / Traits::B_K;
  AITER_CHECK(total_iters >= Traits::prefetch_k_iter,
              "@@NAME@@ requires at least ", Traits::prefetch_k_iter,
              " K-tiles, got ", total_iters);
"""

_BMM_MOUTER_KARGS = r"""
  auto stream = aiter::getCurrentHIPStream();

  opus_gemm_scale_splitk_kargs_gfx950 kargs{};
  kargs.ptr_a = O.data_ptr();
  kargs.ptr_b = wo_a.data_ptr();
  kargs.ptr_ws = nullptr;
  kargs.m = M; kargs.n = N; kargs.k = K; kargs.batch = batch;
  const int num_tiles_m = M / Traits::B_M;
  const int num_tiles_n = N / Traits::B_N;
"""

_BMM_MOUTER_TAIL = r"""  kargs.split_k = m_per_wg;
  kargs.stride_a = (int)O.stride(0);
  kargs.stride_b = (int)wo_a.stride(1);
  kargs.stride_ws = N;
  kargs.stride_a_batch = (int)O.stride(1);
  kargs.stride_b_batch = (int)wo_a.stride(0);
  kargs.stride_ws_batch = M * N;
  kargs.ptr_sfa = x_scale.data_ptr();
  kargs.ptr_sfb = w_scale.data_ptr();
  kargs.stride_sfa = (int)x_scale.stride(0);
  kargs.stride_sfa_batch = (int)x_scale.stride(1);
  kargs.stride_sfb = (int)w_scale.stride(1);
  kargs.stride_sfb_batch = (int)w_scale.stride(0);
  kargs.ptr_c = Y.data_ptr();
  kargs.stride_c = (int)Y.stride(0);
  kargs.stride_c_batch = (int)Y.stride(1);

  const int split_m = (num_tiles_m + m_per_wg - 1) / m_per_wg;
  constexpr int NUM_XCD = 8;
  const int m_grp_per_xcd = (split_m + NUM_XCD - 1) / NUM_XCD;
  kargs.stride_ws = split_m;
  kargs.stride_ws_batch = m_grp_per_xcd;
  dim3 grid_main(NUM_XCD * m_grp_per_xcd * num_tiles_n, 1, batch);
  dim3 block_main(Traits::BLOCK_SIZE);
  if (Y.dtype() == AITER_DTYPE_bf16) {
    @@KERNEL@@<Traits, __bf16, SKIP_SCALE_WAIT>
        <<<grid_main, block_main, 0, stream>>>(kargs);
  } else {
    @@KERNEL@@<Traits, float, SKIP_SCALE_WAIT>
        <<<grid_main, block_main, 0, stream>>>(kargs);
  }
}
#endif // launcher only on regular host pass
"""

_BMM_MOUTER_LAUNCHER_BODY = (
    _BMM_SPEC_SIG.replace("@@SPLITK_ARG@@", "splitK")
    + "  AITER_CHECK(splitK == 1, \"@@NAME@@ requires splitK == 1\");\n"
    + _BMM_MOUTER_CHECKS
    + _BMM_MOUTER_KARGS
    + "  const int m_per_wg = (num_tiles_m >= 16) ? 2 : 1;\n"
    + _BMM_MOUTER_TAIL
)

# Tunable variant: API splitK is repurposed as m_per_wg (clamped to [1,
# num_tiles_m]); reuses the same mouter kernel.
_BMM_MOUTER_TUNABLE_LAUNCHER_BODY = (
    _BMM_SPEC_SIG.replace("@@SPLITK_ARG@@", "splitK")
    + "  AITER_CHECK(splitK >= 1, \"@@NAME@@ requires splitK >= 1\");\n"
    + _BMM_MOUTER_CHECKS
    + _BMM_MOUTER_KARGS
    + "  int m_per_wg = splitK;\n"
    "  if (m_per_wg > num_tiles_m) m_per_wg = num_tiles_m;\n"
    "  if (m_per_wg < 1) m_per_wg = 1;\n" + _BMM_MOUTER_TAIL
)


def gen_bmm_mxscale_mouter_instance(
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
    **_unused,
):
    _, tpl, fn = kargs_template_vars(k.kernel_tag, kargs_name)
    launcher = (
        _BMM_MOUTER_LAUNCHER_BODY.replace("@@NAME@@", k.name)
        .replace("@@KERNEL@@", kernel_func)
        .replace("@@SSW@@", _cppbool(k.skip_scale_wait))
    )
    _emit_bmm_specialized(
        cg,
        k,
        kernel_func,
        traits_name,
        kargs_name,
        da,
        db,
        instance_impl_preamble(),
        instance_impl_host_tu_split(
            traits_header, pipeline_header, tpl, kernel_func, fn
        ),
        launcher,
        f", {_cppbool(k.skip_scale_wait)}",
    )


def gen_bmm_mxscale_mouter_tunable_instance(
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
    **_unused,
):
    _, tpl, fn = kargs_template_vars(k.kernel_tag, kargs_name)
    launcher = (
        _BMM_MOUTER_TUNABLE_LAUNCHER_BODY.replace("@@NAME@@", k.name)
        .replace("@@KERNEL@@", kernel_func)
        .replace("@@SSW@@", _cppbool(k.skip_scale_wait))
    )
    # host-only: device instantiations are shared with the mouter family.
    _emit_bmm_specialized(
        cg,
        k,
        kernel_func,
        traits_name,
        kargs_name,
        da,
        db,
        instance_impl_preamble(),
        instance_impl_host_tu_split(
            traits_header, pipeline_header, tpl, kernel_func, fn
        ),
        launcher,
        f", {_cppbool(k.skip_scale_wait)}",
        emit_device=False,
    )


# ---- pipeline (kids 150/158/151/152) ----
# Dual bf16/fp32 traits (output dtype baked into the traits tuple slot 3),
# non-splitk scale kargs, BLOCK_SIZE 512. Flags pick one of four scale kernels.
_BMM_PIPELINE_LAUNCHER_BODY = r"""
#if !defined(__HIP_DEVICE_COMPILE__) && !defined(__HIPCC_RTC__)
// mmajor: O/Y are [M, batch, *] (dim0=M, dim1=batch); wo_a stays batch-major.
// splitK must be 1 (checked by caller). Caller does dtype/arch/common checks.
template <typename D_C>
void
@@NAME@@(
    aiter_tensor_t &O,
    aiter_tensor_t &wo_a,
    aiter_tensor_t &Y,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    std::optional<aiter_tensor_t> workspace,
    int splitK)
{
  using Bf16Traits = @@NAME@@_Bf16Traits;
  using Fp32Traits = @@NAME@@_Fp32Traits;
  AITER_CHECK(splitK == 1, "@@NAME@@ requires splitK == 1");
  AITER_CHECK(!workspace.has_value(), "@@NAME@@ does not use workspace");
  const int M = O.size(0);
  const int batch = O.size(1);
  const int N = wo_a.size(1);
  const int K = O.size(2);
  // No M alignment requirement: the kernel bounds its A / sfa / C buffers to the
  // tile's valid row window, so a partial trailing M tile is masked by buffer OOB.
  AITER_CHECK(N % Bf16Traits::B_N == 0,
              "@@NAME@@ requires N % ", Bf16Traits::B_N, " == 0, got ", N);
  AITER_CHECK(K % Bf16Traits::B_K == 0,
              "@@NAME@@ requires K % ", Bf16Traits::B_K, " == 0, got ", K);
@@K1024_CHECK@@
  opus_gemm_scale_kargs_gfx950 kargs{};
  kargs.ptr_a = O.data_ptr();
  kargs.ptr_b = wo_a.data_ptr();
  kargs.ptr_c = Y.data_ptr();
  kargs.m = M;
  kargs.n = N;
  kargs.k = K;
  kargs.batch = batch;
  kargs.stride_a = (int)O.stride(0);
  kargs.stride_b = (int)wo_a.stride(1);
  kargs.stride_c = (int)Y.stride(0);
  kargs.stride_a_batch = (int)O.stride(1);
  kargs.stride_b_batch = (int)wo_a.stride(0);
  kargs.stride_c_batch = (int)Y.stride(1);
  kargs.ptr_sfa = x_scale.data_ptr();
  kargs.ptr_sfb = w_scale.data_ptr();
  kargs.stride_sfa = (int)x_scale.stride(0);
  kargs.stride_sfa_batch = (int)x_scale.stride(1);
  kargs.stride_sfb = (int)w_scale.stride(1);
  kargs.stride_sfb_batch = (int)w_scale.stride(0);

  const int num_tiles_m = (M + Bf16Traits::B_M - 1) / Bf16Traits::B_M;
  const int num_tiles_n = N / Bf16Traits::B_N;
  dim3 grid_main(num_tiles_m * num_tiles_n, 1, batch);
  dim3 block_main(Bf16Traits::BLOCK_SIZE);
  auto stream = aiter::getCurrentHIPStream();
  if (Y.dtype() == AITER_DTYPE_bf16) {
    @@KERNEL@@<Bf16Traits><<<grid_main, block_main, 0, stream>>>(kargs);
  } else {
    @@KERNEL@@<Fp32Traits><<<grid_main, block_main, 0, stream>>>(kargs);
  }
}
#endif // launcher only on regular host pass
"""


def _bmm_pipeline_dual_traits_alias(k, traits_name):
    def one(suffix, out_dtype):
        return (
            f"using {k.name}_{suffix} = {traits_name}<{k.BLOCK_SIZE},\n"
            f"    opus::seq<{k.B_M}, {k.B_N}, {k.B_K}>,\n"
            f"    opus::tuple<fp8_t, fp8_t, {out_dtype}, fp32_t, unsigned char>,\n"
            f"    opus::seq<{k.VEC_A}, {k.VEC_B}, {k.VEC_C}>,\n"
            f"    opus::seq<{k.GROUP_M}, {k.GROUP_N}, {k.GROUP_K}>>;\n"
        )

    return "\n" + one("Bf16Traits", "bf16_t") + one("Fp32Traits", "fp32_t")


def gen_bmm_mxscale_pipeline_instance(
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
    **_unused,
):
    if k.preload_sf_lds:
        real_kernel = "gemm_a8w8_scale_preload_sf_kernel"
    elif k.k1024_lb1:
        real_kernel = "gemm_a8w8_scale_k1024_lb1_kernel"
    elif k.k1024_only:
        real_kernel = "gemm_a8w8_scale_k1024_kernel"
    else:
        real_kernel = "gemm_a8w8_scale_kernel"

    _, tpl, fn = kargs_template_vars(k.kernel_tag, kargs_name)
    k1024_check = ""
    if k.k1024_only or k.k1024_lb1:
        k1024_check = (
            f'  AITER_CHECK(K == 1024, "{k.name} requires K == 1024, got ", K);\n'
        )
    launcher = (
        _BMM_PIPELINE_LAUNCHER_BODY.replace("@@NAME@@", k.name)
        .replace("@@KERNEL@@", real_kernel)
        .replace("@@K1024_CHECK@@", k1024_check)
    )

    traits_aliases = _bmm_pipeline_dual_traits_alias(k, traits_name)
    host_tu = instance_impl_host_tu_split(
        traits_header, pipeline_header, tpl, real_kernel, fn
    )
    INSTANCE_IMPL = (
        f"{instance_impl_preamble()}\n{host_tu}\n{traits_aliases}\n{launcher}"
    )
    Path(os.path.join(cg.impl_path, f"{k.name}.cuh")).write_text(INSTANCE_IMPL)

    host_extra = (
        ",\n    aiter_tensor_t &x_scale,"
        "\n    aiter_tensor_t &w_scale,"
        "\n    std::optional<aiter_tensor_t> workspace,"
        "\n    int splitK"
    )
    for dtype in k.output_dtypes:
        host_decl = (
            f"template void\n{k.name}<{dtype}>(\n"
            f"    aiter_tensor_t &O,\n    aiter_tensor_t &wo_a,\n"
            f"    aiter_tensor_t &Y{host_extra});\n"
        )
        cg._host_instantiations.append(
            {"kid_name": k.name, "dtype": dtype, "host_decl": host_decl}
        )

    for dtype_tag, traits_suffix in (("bf16", "Bf16Traits"), ("fp32", "Fp32Traits")):
        decl = (
            f"template __global__ void {real_kernel}<\n"
            f"    {k.name}_{traits_suffix}>({kargs_name});\n"
        )
        cg._device_instantiations.append(
            {"kid_name": k.name, "dtype": dtype_tag, "device_decl": decl}
        )


# ---- fused (kid 100) ----
# Fused-reduce split-K path (counter/atomic variant). Same 256x32x128x128 wg2
# traits + gemm_a8w8_mxscale_flatmm_splitk_kernel<Traits, D_OUT, false, false,
# false> device symbols as standard kid 0/32 -> host-only emit.
_BMM_FUSED_LAUNCHER_BODY = r"""
#if !defined(__HIP_DEVICE_COMPILE__) && !defined(__HIPCC_RTC__)
// mmajor fused-reduce launcher: the main kernel accumulates partials into the
// Y buffer directly via an atomic tile counter (no separate reduce kernel).
// Caller (opus_bmm.cu dispatch) does dtype/arch/common checks.
template <typename D_C>
void
@@NAME@@(
    aiter_tensor_t &O,
    aiter_tensor_t &wo_a,
    aiter_tensor_t &Y,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    std::optional<aiter_tensor_t> workspace,
    int splitK)
{
  using Traits = @@NAME@@_Traits;
  AITER_CHECK(splitK >= 1, "splitK must be >= 1");

  const int M = O.size(0);
  const int batch = O.size(1);
  const int N = wo_a.size(1);
  const int K = O.size(2);
  // No M alignment: the partial tile is masked in-kernel (see the launcher above).
  AITER_CHECK(N % Traits::B_N == 0,
              "@@NAME@@ requires N % ", Traits::B_N, " == 0, got ", N);
  AITER_CHECK(K % Traits::B_K == 0,
              "@@NAME@@ requires K % ", Traits::B_K, " == 0, got ", K);

  const int split_k = splitK;
  const bool no_split_k = (split_k == 1);
  const int total_iters = K / Traits::B_K;
  const int iters_full = (total_iters + split_k - 1) / split_k;
  const int last_loops = total_iters - (split_k - 1) * iters_full;
  AITER_CHECK(last_loops >= Traits::prefetch_k_iter,
              "@@NAME@@ requires every split to have at least ",
              Traits::prefetch_k_iter, " K-tiles; K=", K,
              " gives total_iters=", total_iters, ", splitK=", split_k,
              ", last split loops=", last_loops);

  const int num_tiles_m = (M + Traits::B_M - 1) / Traits::B_M;
  const int num_tiles_n = (N + Traits::B_N - 1) / Traits::B_N;
  const int padded_M = num_tiles_m * Traits::B_M;
  const int padded_N = num_tiles_n * Traits::B_N;
  const size_t partial_bytes = (size_t)split_k * (size_t)batch
                             * (size_t)padded_M * (size_t)padded_N * sizeof(float);
  const size_t counter_offset = (partial_bytes + 255) & ~((size_t)255);
  const size_t counter_bytes = (size_t)batch * (size_t)num_tiles_m
                             * (size_t)num_tiles_n * sizeof(int);

  auto stream = aiter::getCurrentHIPStream();

  opus_gemm_scale_splitk_kargs_gfx950 kargs{};
  kargs.ptr_a = O.data_ptr();
  kargs.ptr_b = wo_a.data_ptr();
  kargs.ptr_ws = nullptr;
  kargs.m = M; kargs.n = N; kargs.k = K; kargs.batch = batch;
  kargs.split_k = split_k;
  kargs.stride_a = (int)O.stride(0);
  kargs.stride_b = (int)wo_a.stride(1);
  kargs.stride_ws = padded_N;
  kargs.stride_a_batch = (int)O.stride(1);
  kargs.stride_b_batch = (int)wo_a.stride(0);
  kargs.stride_ws_batch = padded_M * padded_N;
  kargs.ptr_sfa = x_scale.data_ptr();
  kargs.ptr_sfb = w_scale.data_ptr();
  kargs.stride_sfa = (int)x_scale.stride(0);
  kargs.stride_sfa_batch = (int)x_scale.stride(1);
  kargs.stride_sfb = (int)w_scale.stride(1);
  kargs.stride_sfb_batch = (int)w_scale.stride(0);

  dim3 grid_main(num_tiles_m * num_tiles_n * split_k, 1, batch);
  dim3 block_main(Traits::BLOCK_SIZE);
  if (no_split_k) {
    AITER_CHECK(!workspace.has_value(),
                "@@NAME@@ splitK == 1 does not use workspace");
    kargs.ptr_c = Y.data_ptr();
    kargs.stride_c = (int)Y.stride(0);
    kargs.stride_c_batch = (int)Y.stride(1);
    if (Y.dtype() == AITER_DTYPE_bf16) {
      @@KERNEL@@<Traits, __bf16, false, false, false>
          <<<grid_main, block_main, 0, stream>>>(kargs);
    } else {
      @@KERNEL@@<Traits, float, false, false, false>
          <<<grid_main, block_main, 0, stream>>>(kargs);
    }
    return;
  }

  const size_t ws_bytes = counter_offset + counter_bytes;
  const size_t required_numel = (ws_bytes + sizeof(float) - 1) / sizeof(float);
  AITER_CHECK(workspace.has_value(),
              "@@NAME@@ splitK > 1 requires workspace");
  void* workspace_ptr = opus_validate_workspace(
      workspace.value(), O, AITER_DTYPE_fp32, required_numel, 16, "@@NAME@@");
  kargs.ptr_ws = workspace_ptr;

  kargs.ptr_c = Y.data_ptr();
  kargs.stride_c = (int)Y.stride(0);
  kargs.stride_c_batch = (int)Y.stride(1);
  kargs.counter_offset_bytes = counter_offset;
  HIP_CALL(hipMemsetAsync(static_cast<char*>(workspace_ptr) + counter_offset,
                          0, counter_bytes, stream));
  if (Y.dtype() == AITER_DTYPE_bf16) {
    @@KERNEL@@<Traits, __bf16, false, false, false>
        <<<grid_main, block_main, 0, stream>>>(kargs);
  } else {
    @@KERNEL@@<Traits, float, false, false, false>
        <<<grid_main, block_main, 0, stream>>>(kargs);
  }
}
#endif // launcher only on regular host pass
"""


def gen_bmm_mxscale_fused_instance(
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
    **_unused,
):
    _, tpl, fn = kargs_template_vars(k.kernel_tag, kargs_name)
    launcher = _BMM_FUSED_LAUNCHER_BODY.replace("@@NAME@@", k.name).replace(
        "@@KERNEL@@", kernel_func
    )
    # host-only: device symbols <Traits, D_OUT, false, false, false> are shared
    # with the standard flatmm split-K kid 0/32 (same traits) and emitted there.
    _emit_bmm_specialized(
        cg,
        k,
        kernel_func,
        traits_name,
        kargs_name,
        da,
        db,
        instance_impl_preamble('\n#include "opus_gemm_common.cuh"'),
        instance_impl_host_tu_split(
            traits_header, pipeline_header, tpl, kernel_func, fn
        ),
        launcher,
        "",
        emit_device=False,
    )


# ---------- Self-register at import time ----------
register_emit("gfx950", "a16w16_persistent", gen_persistent_instance)
register_emit("gfx950", "a8w8_scale", gen_scale_instance)
register_emit("gfx950", "a8w8_mxscale", gen_scale_instance)
register_emit("gfx950", "a16w16", gen_noscale_instance_gfx950)
register_emit("gfx950", "a8w8", gen_noscale_instance_gfx950)
register_emit("gfx950", "a16w16_mono_tile", gen_mono_tile_instance)
register_emit("gfx950", "a16w16_flatmm", gen_flatmm_instance)
register_emit("gfx950", "a16w16_flatmm_splitk", gen_flatmm_splitk_instance)


def _register_bmm_emit(kernel_tag, fn, launcher_tile_mult):
    """register_emit + the m_align cross-check for the BMM families.

    `launcher_tile_mult` is the B_M multiple this family's launcher body
    hardcodes in its M guard, or 0 for the bodies that mask a partial M tile and
    emit no M check. Checking it at emit time is what stops the guard and
    OpusGemmInstance.m_align -- which the tuner and the runtime dispatch both
    read -- from drifting apart.
    """

    def emit(cg, k, **kwargs):
        _assert_m_align(k, launcher_tile_mult)
        return fn(cg, k, **kwargs)

    emit.__name__ = fn.__name__
    register_emit("gfx950", kernel_tag, emit)


# _BMM_MXSCALE_SPLITK_LAUNCHER_BODY / _BMM_PIPELINE_LAUNCHER_BODY /
# _BMM_FUSED_LAUNCHER_BODY emit no M check ("No M alignment ..."); minterleave
# guards MI(=2)*B_M, wave4m2 guards LOGICAL_B_M(=2*B_M), the rest guard B_M.
_register_bmm_emit(
    "a8w8_mxscale_bmm_flatmm_splitk", gen_bmm_mxscale_flatmm_splitk_instance, 0
)
_register_bmm_emit(
    "a8w8_mxscale_bmm_minterleave", gen_bmm_mxscale_minterleave_instance, 2
)
_register_bmm_emit("a8w8_mxscale_bmm_fused", gen_bmm_mxscale_fused_instance, 0)
_register_bmm_emit("a8w8_mxscale_bmm_pipeline", gen_bmm_mxscale_pipeline_instance, 0)
_register_bmm_emit("a8w8_mxscale_bmm_mouter", gen_bmm_mxscale_mouter_instance, 1)
_register_bmm_emit(
    "a8w8_mxscale_bmm_mouter_tunable", gen_bmm_mxscale_mouter_tunable_instance, 1
)
_register_bmm_emit("a8w8_mxscale_bmm_wave8n2", gen_bmm_mxscale_wave8n2_instance, 1)
_register_bmm_emit(
    "a8w8_mxscale_bmm_wave4m2_selfload", gen_bmm_mxscale_wave4m2_selfload_instance, 2
)
