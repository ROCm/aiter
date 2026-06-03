# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
import argparse
import glob
import json
import os
import shutil
from pathlib import Path

import pandas as pd
import torch
from codegen.common import kid_arch as _kid_arch_common

# Import for side-effect: each arch module self-registers into EMIT_REGISTRY.
from codegen import gen_instances_gfx950 as _gfx950  # noqa: F401
from codegen import gen_instances_gfx942 as _gfx942  # noqa: F401
from opus_gemm_common import (
    HEURISTIC_DEFAULT_KIDS,
    OpusGemmInstance,
    heuristic_kids_for_arch,
    a8w8_kernels_list,
    a8w8_scale_kernels_list,
    a16w16_flatmm_kernels_list,
    a16w16_flatmm_splitk_kernels_list,
    a16w16_kernels_list,
    a16w16_mono_tile_kernels_list,
    default_kernels_dict,
    gfx942_nosplit_kernels_list,
    gfx942_splitk_kernels_list,
    kernels_list,
)

# Paired W3 kernels (nosplit_tag -> splitk_tag) share one <Traits, Kargs> template.
W3_KERNEL_PAIRS = {
    "a16w16_kbuf3": "a16w16_kbuf3_sk",
    "a16w16_kbuf2v": "a16w16_kbuf2v_sk",
    "a16w16_kbuf2v_bk128": "a16w16_kbuf2v_bk128_sk",
    "a16w16_kbuf1": "a16w16_kbuf1_sk",
}
_NOSPLIT = tuple(W3_KERNEL_PAIRS.keys())
_SPLITK = tuple(W3_KERNEL_PAIRS.values())
_GFX942_A16W16_TAGS = (
    _SPLITK + ("a16w16_fused_reduce", "a16w16_kbuf1_large_tile") + _NOSPLIT
)
_A16W16_TAGS = (
    "a16w16",
    "a16w16_flatmm",
    "a16w16_flatmm_splitk",
    "a16w16_persistent",
    "a16w16_mono_tile",
) + _GFX942_A16W16_TAGS


# gfx942 pipeline header derived from W3_KERNEL_PAIRS: splitk_X reuses
# nosplit_X's .cuh (paired template); splitk_fused has its own.
def _gfx942_pipeline(tag):
    return f"gfx942/opus_gemm_pipeline_{tag}.cuh"


PIPELINE_HEADER_MAP = {
    "a8w8_scale": "gfx950/opus_gemm_pipeline_a8w8_scale_gfx950.cuh",
    "a8w8": "gfx950/opus_gemm_pipeline_a8w8_noscale_gfx950.cuh",
    "a16w16": "gfx950/opus_gemm_pipeline_a16w16_gfx950.cuh",
    "a16w16_flatmm": "gfx950/opus_gemm_pipeline_a16w16_flatmm_gfx950.cuh",
    "a16w16_flatmm_splitk": "gfx950/opus_gemm_pipeline_a16w16_flatmm_splitk_gfx950.cuh",
    "a16w16_persistent": "gfx950/opus_gemm_pipeline_a16w16_persistent_gfx950.cuh",
    "a16w16_mono_tile": "gfx950/opus_gemm_pipeline_a16w16_mono_tile_gfx950.cuh",
    "a16w16_fused_reduce": _gfx942_pipeline("a16w16_fused_reduce"),
    "a16w16_kbuf1_large_tile": _gfx942_pipeline("a16w16_kbuf1_large_tile"),
    **{nosplit: _gfx942_pipeline(nosplit) for nosplit in _NOSPLIT},
    **{
        splitk: _gfx942_pipeline(nosplit) for nosplit, splitk in W3_KERNEL_PAIRS.items()
    },
}
GFX942_PIPELINE_HEADER_MAP = {
    "a16w16_kbuf1_large_tile": _gfx942_pipeline("a16w16_kbuf1_large_tile")
}

# Traits header carries the traits struct + kargs struct definitions for a given pipeline tag.
GFX942_TRAITS_HEADER = "gfx942/opus_gemm_traits_a16w16.cuh"

TRAITS_HEADER_MAP = {
    "a8w8_scale": "gfx950/opus_gemm_traits_a8w8_scale_gfx950.cuh",
    "a8w8": "gfx950/opus_gemm_traits_a8w8_noscale_gfx950.cuh",
    "a16w16": "gfx950/opus_gemm_traits_a16w16_gfx950.cuh",
    "a16w16_flatmm": "gfx950/opus_gemm_traits_a16w16_gfx950.cuh",
    "a16w16_flatmm_splitk": "gfx950/opus_gemm_traits_a16w16_gfx950.cuh",
    "a16w16_persistent": "gfx950/opus_gemm_traits_a16w16_gfx950.cuh",
    "a16w16_mono_tile": "gfx950/opus_gemm_traits_a16w16_gfx950.cuh",
    **{tag: GFX942_TRAITS_HEADER for tag in _GFX942_A16W16_TAGS},
}
GFX942_TRAITS_HEADER_MAP = {"a16w16_kbuf1_large_tile": GFX942_TRAITS_HEADER}

# Per-tag splitk reduce header (splitk_fused omitted: in-kernel reduce).
SPLITK_REDUCE_HEADER_MAP = {
    "a16w16_flatmm_splitk": "gfx950/splitk_reduce_gfx950.cuh",
    "a16w16_kbuf3_sk": "gfx942/splitk_reduce_gfx942.cuh",
    "a16w16_kbuf1_sk": "gfx942/splitk_reduce_gfx942.cuh",
}

# Arches that expose the V2/V3 fast-path reduce kernels.
SPLITK_REDUCE_FAST_ARCHES = {"gfx942"}

# split_k values explicitly instantiated for V2 fast-path (covers tuner range 2..10).
V2_SUPPORTED_SPLITKS = (2, 3, 4, 5, 6, 7, 8, 10)

# V3 reduce: (N_VEC, ROWS_PER_BLOCK), BLOCK = N_VEC * ROWS_PER_BLOCK = 64 (1 wave).
V3_NVEC_ROWS = (
    (8, 8),  # N=64,  8 rows/wg
    (16, 4),  # N=128, 4 rows/wg
    (32, 2),  # N=256, 2 rows/wg
    (64, 1),  # N=512, 1 row/wg
)
# V4 (8, 32) DEAD 2026-05-30: BLOCK=256 4-wave, 0 wall-time benefit (vmcnt contention).

KERNEL_FUNC_MAP = {
    "a8w8_scale": "gemm_a8w8_scale_kernel",
    "a8w8": "gemm_a8w8_noscale_kernel",
    "a16w16": "gemm_a16w16_kernel",
    "a16w16_flatmm": "gemm_a16w16_flatmm_kernel",
    "a16w16_flatmm_splitk": "gemm_a16w16_flatmm_splitk_kernel",
    "a16w16_persistent": "gemm_a16w16_persistent_kernel",
    "a16w16_mono_tile": "gemm_a16w16_mono_tile_kernel_gfx950",
    "a16w16_fused_reduce": "gemm_a16w16_fused_reduce_kernel",
    "a16w16_kbuf1_large_tile": "gemm_a16w16_kbuf1_large_tile_kernel",
    # gfx942 paired tags: nosplit_tag's kernel symbol; splitk_tag reuses it.
    **{nosplit: f"gemm_{nosplit}_kernel" for nosplit in W3_KERNEL_PAIRS.keys()},
    **{splitk: f"gemm_{nosplit}_kernel" for nosplit, splitk in W3_KERNEL_PAIRS.items()},
}

# 4g_safe sibling pipelines: only defined for the a16w16-family tags that have
# matching *_4g_safe_gfx950.cuh files. Kids with is_4g_safe=True route to these
# headers/kernel symbols instead of the legacy maps above.
PIPELINE_HEADER_MAP_4G_SAFE = {
    "a16w16": "gfx950/opus_gemm_pipeline_a16w16_4g_safe_gfx950.cuh",
    "a16w16_persistent": "gfx950/opus_gemm_pipeline_a16w16_persistent_4g_safe_gfx950.cuh",
    "a16w16_mono_tile": "gfx950/opus_gemm_pipeline_a16w16_mono_tile_4g_safe_gfx950.cuh",
}

KERNEL_FUNC_MAP_4G_SAFE = {
    "a16w16": "gemm_a16w16_4g_safe_kernel",
    "a16w16_persistent": "gemm_a16w16_persistent_4g_safe_kernel",
    "a16w16_mono_tile": "gemm_a16w16_mono_tile_4g_safe_kernel_gfx950",
}


def _pipeline_header_for(k):
    if getattr(k, "is_4g_safe", False):
        return PIPELINE_HEADER_MAP_4G_SAFE[k.kernel_tag]
    return PIPELINE_HEADER_MAP[k.kernel_tag]


def _kernel_func_for(k):
    if getattr(k, "is_4g_safe", False):
        return KERNEL_FUNC_MAP_4G_SAFE[k.kernel_tag]
    return KERNEL_FUNC_MAP[k.kernel_tag]


INPUT_DTYPE_MAP = {
    "a8w8_scale": ("fp8_t", "fp8_t"),
    "a8w8": ("fp8_t", "fp8_t"),
    **{tag: ("bf16_t", "bf16_t") for tag in _A16W16_TAGS},
}

# All a16w16 tags share the 4-arg (XQ, WQ, Y, int splitK) lookup-table slot.
A16W16_TUNE_TAGS = set(_A16W16_TAGS)
# NOSCALE: 3-arg launchers (a16w16 family + a8w8 non-scale).
NOSCALE_TAGS = A16W16_TUNE_TAGS | {"a8w8"}

# Splitk tags forced to <fp32_t> in lookup (main kernel writes fp32 workspace).
SPLITK_TAGS = {
    "a16w16_flatmm_splitk",
    "a16w16_fused_reduce",
    *_SPLITK,
}

# gfx942 a16w16 tags all share one traits class name (no arch suffix).
GFX942_TRAITS_NAME = "opus_gemm_a16w16_traits"

TRAITS_NAME_MAP = {
    "a8w8_scale": "opus_gemm_a8w8_scale_traits_gfx950",
    "a8w8": "opus_gemm_a8w8_noscale_traits_gfx950",
    "a16w16": "opus_gemm_a16w16_traits_gfx950",
    "a16w16_flatmm": "opus_gemm_a16w16_flatmm_traits_gfx950",
    "a16w16_flatmm_splitk": "opus_flatmm_splitk_traits_gfx950",
    "a16w16_persistent": "opus_gemm_a16w16_persistent_traits_gfx950",
    "a16w16_mono_tile": "opus_gemm_a16w16_mono_tile_traits_gfx950",
    **{tag: GFX942_TRAITS_NAME for tag in _GFX942_A16W16_TAGS},
}
GFX942_TRAITS_NAME_MAP = {"a16w16_kbuf1_large_tile": GFX942_TRAITS_NAME}

KARGS_NAME_MAP = {
    "a8w8_scale": "opus_gemm_scale_kargs_gfx950",
    "a8w8": "opus_gemm_noscale_kargs_gfx950",
    "a16w16": "opus_gemm_noscale_kargs_gfx950",
    "a16w16_flatmm": "opus_gemm_flatmm_kargs_gfx950",
    "a16w16_flatmm_splitk": "opus_gemm_flatmm_splitk_kargs_gfx950",
    "a16w16_persistent": "opus_gemm_persistent_kargs_gfx950",
    "a16w16_mono_tile": "opus_gemm_mono_tile_kargs_gfx950",
    "a16w16_fused_reduce": "opus_gemm_splitk_fused_kargs",
    **{tag: "opus_gemm_splitk_kargs" for tag in _SPLITK},
    **{tag: "opus_gemm_noscale_kargs" for tag in _NOSPLIT},
}
GFX942_KARGS_NAME_MAP = {"a16w16_kbuf1_large_tile": "opus_gemm_noscale_kargs"}


def _lookup(k, default_map, arch_map):
    """Pick the gfx942 override when k.arch_prefix=='gfx942', else default."""
    if getattr(k, "arch_prefix", "") == "gfx942" and k.kernel_tag in arch_map:
        return arch_map[k.kernel_tag]
    return default_map[k.kernel_tag]


def _kargs_template_vars(kernel_tag, kargs_name):
    # Paired W3 kernels: fn arg 'Kargs' so deduction keeps host/device mangling.
    if kernel_tag in _NOSPLIT or kernel_tag in _SPLITK:
        return f", {kargs_name}", ", typename Kargs", "Kargs"
    return "", "", kargs_name


# INSTANCE_IMPL building blocks. Host pass needs torch/optional; RTC/device passes skip them.
_INSTANCE_IMPL_PREAMBLE_TEMPLATE = """// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
#pragma once
#if !defined(__HIP_DEVICE_COMPILE__) && !defined(__HIPCC_RTC__)
#include "aiter_tensor.h"
#include "aiter_stream.h"{extra_host_includes}
#include <optional>
#endif"""


def instance_impl_preamble(extra_host_includes=""):
    return _INSTANCE_IMPL_PREAMBLE_TEMPLATE.format(
        extra_host_includes=extra_host_includes
    )


# Fused host TU sees only traits header + fwd decl; avoids layout-helper ODR clash.
_INSTANCE_IMPL_HOST_TU_SPLIT_TEMPLATE = """#ifdef OPUS_FUSED_HOST_TU
#include "{traits_header}"
template<typename Traits{fwd_decl_kargs_tpl}>
__global__ void {kernel_func}({fwd_decl_kargs_fnarg} kargs);
#else
#include "{pipeline_header}"
#endif"""


def instance_impl_host_tu_split(
    traits_header,
    pipeline_header,
    fwd_decl_kargs_tpl,
    kernel_func,
    fwd_decl_kargs_fnarg,
):
    return _INSTANCE_IMPL_HOST_TU_SPLIT_TEMPLATE.format(
        traits_header=traits_header,
        pipeline_header=pipeline_header,
        fwd_decl_kargs_tpl=fwd_decl_kargs_tpl,
        kernel_func=kernel_func,
        fwd_decl_kargs_fnarg=fwd_decl_kargs_fnarg,
    )


# Launcher signature tails after Y.
A16W16_TUNE_HOST_EXTRA = ",\n    std::optional<aiter_tensor_t>,\n    int"
A8W8_SCALE_HOST_EXTRA = (
    ",\n    std::optional<aiter_tensor_t> x_scale,"
    "\n    std::optional<aiter_tensor_t> w_scale"
)


def _make_host_decl(kid_name, dtype, host_extra_params):
    return (
        f"template void\n"
        f"{kid_name}<{dtype}>(\n"
        f"    aiter_tensor_t &XQ,\n"
        f"    aiter_tensor_t &WQ,\n"
        f"    aiter_tensor_t &Y{host_extra_params});\n"
    )


def _make_device_decl(
    kid_name, dtype, kernel_func, kargs_name, kargs_explicit_param=""
):
    return (
        f"template __global__ void {kernel_func}<\n"
        f"    {kid_name}_Traits<{dtype}>{kargs_explicit_param}>({kargs_name});\n"
    )


def _record_one_instantiation(
    self_obj, k, kernel_func, kargs_name, host_extra, kargs_explicit_param=""
):
    """Record (host_decl, device_decl) for every (kid, dtype) in k.output_dtypes."""
    for CDtype in k.output_dtypes:
        self_obj._host_instantiations.append(
            {
                "kid_name": k.name,
                "dtype": CDtype,
                "host_decl": _make_host_decl(k.name, CDtype, host_extra),
            }
        )
        self_obj._device_instantiations.append(
            {
                "kid_name": k.name,
                "dtype": CDtype,
                "device_decl": _make_device_decl(
                    k.name, CDtype, kernel_func, kargs_name, kargs_explicit_param
                ),
            }
        )


WARP_SIZE = 64
VALID_BF16_MFMA = {(16, 16, 32), (32, 32, 16)}
# gfx942 a16w16 family supports only the 16x16x16 BF16 MFMA shape.
VALID_GFX942_BF16_MFMA = {(16, 16, 16)}
# Flatmm pipeline currently only supports W_M < 32 (ra layout relies on
# LOAD_GROUP_M_LANE == 1). W_M == 32 (LGML == 4) path not rewritten.
VALID_FLATMM_MFMA = {(16, 16, 32)}
VALID_FLATMM_SPLITK_MFMA = {(16, 16, 32)}
# Persistent pipeline ports the mouter reference which only validated
# 16x16x32 BF16 MFMA. Add 32x32x16 later if needed.
VALID_PERSISTENT_MFMA = {(16, 16, 32)}
# Mono-tile pipeline: same MFMA lock as persistent (16x16x32 BF16) -- the
# kernel template hard-codes T_M=2, T_N=4, T_K=1, W_M=W_N=16, W_K=32.
VALID_MONO_TILE_MFMA = {(16, 16, 32)}


class opus_gemm_codegen:
    def __init__(self, working_path, istune=False):
        self.working_path = working_path
        self.impl_path = os.path.join(working_path, "impl")
        self.instances_path = os.path.join(working_path, "instances")
        self.istune = istune
        # Compile-time split: Build layout: * One fused HOST TU (instances/all_instances_host.cu)
        # instantiates every launcher's `template...
        self._host_instantiations = []
        self._device_instantiations = []
        self._kid_records = []
        # Pipeline headers for each kernel_tag (used by the per-kid
        # device TU only).
        self._kid_pipeline_header = {}

    # -- a16w16 compile-time + VGPR spill validator --

    @staticmethod
    def _validate_a16w16(k: OpusGemmInstance):
        """Validate an a16w16 instance at codegen time. Raises ValueError if invalid."""
        errors = []
        sizeof_da = 2  # bf16

        T_K = 1
        HALF_B_M = k.B_M // 2
        HALF_B_N = k.B_N // 2
        num_waves = k.T_M * k.T_N * T_K
        smem_linear_wave = WARP_SIZE * 16 // sizeof_da  # 512

        # -- Hardware --
        if k.BLOCK_SIZE > 512:
            errors.append(f"BLOCK_SIZE={k.BLOCK_SIZE} exceeds 512")

        # -- Pipeline: T_M must be 2 (split-barrier) --
        if k.T_M != 2:
            errors.append(f"T_M={k.T_M} must be 2")

        # -- Traits: BLOCK_SIZE = T_M * T_N * T_K * WARP_SIZE --
        if k.BLOCK_SIZE != num_waves * WARP_SIZE:
            errors.append(
                f"BLOCK_SIZE={k.BLOCK_SIZE} != "
                f"{k.T_M}*{k.T_N}*{T_K}*{WARP_SIZE}={num_waves * WARP_SIZE}"
            )

        # -- Layout: T_N % T_M == 0 (rb: T_N/T_M) --
        if k.T_N % k.T_M != 0:
            errors.append(f"T_N={k.T_N} not divisible by T_M={k.T_M}")

        # -- MFMA validity --
        valid_mfma = (
            VALID_GFX942_BF16_MFMA
            if getattr(k, "arch_prefix", "") == "gfx942"
            else VALID_BF16_MFMA
        )
        if (k.W_M, k.W_N, k.W_K) not in valid_mfma:
            errors.append(f"WAVE=({k.W_M},{k.W_N},{k.W_K}) not in {valid_mfma}")
        if WARP_SIZE % k.W_M != 0:
            errors.append(f"WARP_SIZE not divisible by W_M={k.W_M}")
        if WARP_SIZE % k.W_N != 0:
            errors.append(f"WARP_SIZE not divisible by W_N={k.W_N}")
        if k.W_M % k.T_N != 0:
            errors.append(f"W_M={k.W_M} not divisible by T_N={k.T_N}")
        if k.W_N % k.T_N != 0:
            errors.append(f"W_N={k.W_N} not divisible by T_N={k.T_N}")

        # -- VEC --
        expected_vec = 16 // sizeof_da
        if k.VEC_A != expected_vec:
            errors.append(f"VEC_A={k.VEC_A} must be {expected_vec}")

        # -- Block tile divisibility --
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

        # -- smem layout --
        if smem_linear_wave % k.B_K != 0:
            errors.append(f"smem_linear_wave={smem_linear_wave} not div by B_K={k.B_K}")
        else:
            smem_sub = smem_linear_wave // k.B_K
            if HALF_B_M % smem_sub != 0:
                errors.append(f"HALF_B_M={HALF_B_M} not div by smem_sub={smem_sub}")
            if HALF_B_N % smem_sub != 0:
                errors.append(f"HALF_B_N={HALF_B_N} not div by smem_sub={smem_sub}")

        # -- buffer/ds instruction counts >= 1 and integer --
        for name, num, den in [
            ("a_buffer_load_insts", HALF_B_M * k.B_K, k.BLOCK_SIZE * k.VEC_A),
            ("b_buffer_load_insts", HALF_B_N * k.B_K, k.BLOCK_SIZE * k.VEC_B),
            ("a_ds_read_insts", E_M * E_K * k.W_M * k.W_K, WARP_SIZE * k.VEC_A),
            ("b_ds_read_insts", E_N * E_K * k.W_N * k.W_K, WARP_SIZE * k.VEC_B),
        ]:
            if den == 0 or num % den != 0 or num // den < 1:
                errors.append(f"{name}={num}/{den} invalid")

        # -- ra/rb: W_M*W_K / (WARP_SIZE*VEC_A) >= 1 (gfx942 ra/rb uses different stride; skip). --
        if getattr(k, "arch_prefix", "") != "gfx942":
            for tag, ww, vec in [
                ("ra", k.W_M * k.W_K, k.VEC_A),
                ("rb", k.W_N * k.W_K, k.VEC_B),
            ]:
                denom = WARP_SIZE * vec
                if ww < denom or ww % denom != 0:
                    errors.append(f"{tag}: W*W_K={ww} must be >= and div by {denom}")

        # -- gb: exact division (not ceil_div) --
        if k.VEC_B and k.B_K % k.VEC_B == 0:
            threads_k_b = k.B_K // k.VEC_B
            if k.BLOCK_SIZE % threads_k_b == 0:
                thr_n = k.BLOCK_SIZE // threads_k_b
                if HALF_B_N % thr_n != 0:
                    errors.append(f"gb: HALF_B_N={HALF_B_N} not div by {thr_n}")

        # -- sb: exact division --
        if smem_linear_wave % k.B_K == 0:
            smem_sub = smem_linear_wave // k.B_K
            if smem_sub and HALF_B_N % smem_sub == 0:
                smem_n_rep = HALF_B_N // smem_sub
                if smem_n_rep % num_waves != 0:
                    errors.append(f"sb: smem_n_rep={smem_n_rep} not div by {num_waves}")

        # -- threads_k <= WARP_SIZE --
        for tag, vec in [("ga", k.VEC_A), ("gb", k.VEC_B)]:
            if vec and k.B_K // vec > WARP_SIZE:
                errors.append(f"{tag}: B_K/VEC={k.B_K // vec} > WARP_SIZE")

        # -- AGPR < 256 --
        agpr_per_mfma = (k.W_M * k.W_N) // WARP_SIZE
        total_agprs = 4 * E_M * E_N * agpr_per_mfma
        if total_agprs >= 256:
            errors.append(f"AGPR={total_agprs} must be < 256")

        # -- LDS <= 160 KiB --
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

        # -- VGPR spill estimate --
        vgpr_ops = 4 * E_K * (E_M + 2 * E_N)
        vgpr_est = vgpr_ops + 80
        if vgpr_est > 256:
            errors.append(f"VGPR_est={vgpr_est} exceeds 256")
        if vgpr_est + total_agprs > 512:
            errors.append(f"VGPR+AGPR={vgpr_est + total_agprs} exceeds 512")

        # -- ra/rb layout constraint: B_K must equal T_N * W_K / 2 -- The ra/rb LDS read layouts couple
        # E_K with T_N through the T_M part...
        if getattr(k, "arch_prefix", "") != "gfx942":
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

    # -- a16w16_flatmm validator --

    @staticmethod
    def _validate_a16w16_flatmm(k: OpusGemmInstance):
        """Validate an a16w16_flatmm instance at codegen time.

        Mirrors the static_asserts in opus_gemm_a16w16_flatmm_traits_gfx950: derives
        pfk from LDS budget / WG_PER_CU and requires pfk >= 3 (depth-1 pipeline
        entry point). Raises ValueError if invalid.
        """
        errors = []
        sizeof_da = 2  # bf16 locked

        # -- Locked config (traits enforces these via templates) --
        if k.BLOCK_SIZE != 256:
            errors.append(f"BLOCK_SIZE={k.BLOCK_SIZE} must be 256 (4-wave warp-spec)")
        if k.T_M != 2:
            errors.append(f"T_M={k.T_M} must be 2")
        if k.T_N != 1:
            errors.append(f"T_N={k.T_N} must be 1")

        # -- MFMA: only W_M<32 path supported (LOAD_GROUP_M_LANE=1) --
        if (k.W_M, k.W_N, k.W_K) not in VALID_FLATMM_MFMA:
            errors.append(
                f"WAVE=({k.W_M},{k.W_N},{k.W_K}) not in {VALID_FLATMM_MFMA} "
                f"(flatmm ra layout requires W_M<32)"
            )
        if k.W_M >= 32:
            errors.append(f"W_M={k.W_M}: flatmm LGML=4 path not implemented")

        # -- VEC --
        expected_vec = 16 // sizeof_da
        if k.VEC_A != expected_vec or k.VEC_B != expected_vec:
            errors.append(f"VEC_A={k.VEC_A}, VEC_B={k.VEC_B} must be {expected_vec}")
        if k.VEC_C != 4:
            errors.append(f"VEC_C={k.VEC_C} must be 4")

        # -- Tile geometry (LOAD_GROUP_K = W_K * 2 = 64 for W_K=32) --
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

        # -- LDS per-group-load size --
        smem_linear_wave = WARP_SIZE * 16 // sizeof_da  # 512 for bf16
        smem_sub = smem_linear_wave // LOAD_GROUP_K
        slots = LOAD_GROUP_M // smem_sub
        smem_padding = 16 // sizeof_da if k.W_M >= 32 else 2 * 16 // sizeof_da
        smem_per_group_load_size = slots * (smem_linear_wave + smem_padding) * sizeof_da

        # -- WG_PER_CU --
        if k.WG_PER_CU not in (1, 2):
            errors.append(f"WG_PER_CU={k.WG_PER_CU} must be 1 or 2")

        # -- pfk derivation (match traits formula) --
        lds_total = 163840  # gfx950 budget; host-side constant for validation only
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

    # -- a16w16_flatmm_splitk validator --

    @staticmethod
    def _validate_a16w16_flatmm_splitk(k: OpusGemmInstance):
        """Validate an a16w16_flatmm_splitk instance at codegen time.

        Mirrors _validate_a16w16_flatmm's checks (LDS budget, pfk>=3, MFMA,
        VEC, tile divisibility) and adds a VGPR-spill guard: WG_PER_CU=1 with
        COM_REP_M*COM_REP_N > 16 causes 100+ VGPR spill to scratch and ~1000x
        slowdown (cc lines 1143-1150 hand-picked only 3 WG=1 tiles for this
        reason). Raises ValueError if invalid.
        """
        errors = []
        sizeof_da = 2  # bf16 locked

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

        lds_total = 163840  # gfx950
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

        # VGPR-spill guard: cc hand-picked only 3 WG=1 tiles because larger tiles (COM_REP_M*COM_REP_N >
        # 16) spill v_c to scratch and run...
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

    @staticmethod
    def _validate_a16w16_persistent(k: OpusGemmInstance):
        """Validate an a16w16 persistent instance.

        Persistent uses the same per-tile layout as the split-barrier pipeline
        (TILE/WAVE traits, E_M/E_N/E_K derivation, smem footprint), so its
        constraints are a superset of _validate_a16w16's. Additionally:
          * MFMA restricted to VALID_PERSISTENT_MFMA (mouter reference only).
          * BLOCK_SIZE locked to 512 and T_M*T_N == 8 (matches mouter
            8-wave WG; smaller WGs not yet ported).
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
        # All other shape/divisibility constraints fall through to the split-barrier validator.
        return opus_gemm_codegen._validate_a16w16(k)

    @staticmethod
    def _validate_a16w16_mono_tile(k: OpusGemmInstance):
        """Validate an a16w16 mono-tile instance.

        Mirrors the static_asserts in opus_gemm_a16w16_mono_tile_traits_gfx950
        and the kernel-internal constraints in the mono-tile pipeline header.
        Mono-tile locks T_M=2, T_N=4, T_K=1, W_M=W_N=16, W_K=32 (MFMA
        16x16x32 BF16), VEC=8, BLOCK_SIZE=512 (8 waves * 64 lanes); the
        tile must satisfy:
          * B_M divisible by W_M*T_M = 32
          * B_N divisible by W_N*T_N = 64
          * B_K divisible by W_K*T_K = 32
          * B_K divides smem_linear_wave = 512 (bf16)
          * smem_m_rep = B_M / smem_sub >= 8 and divisible by 8 (num_waves)
          * smem_n_rep = B_N / smem_sub >= 8 and divisible by 8
          * E_N = B_N / (W_N*T_N) divisible by (T_N/T_M) = 2  ->  B_N % 128 == 0
          * E_M divisible by smem_sub / (W_M/T_N)
        Plus a user-imposed B_M <= 192 cap.
        """
        errors = []
        sizeof_da = 2  # bf16 locked

        # -- Locked config --
        if k.BLOCK_SIZE != 512:
            errors.append(
                f"BLOCK_SIZE={k.BLOCK_SIZE} must be 512 (mono-tile 8-wave WG)"
            )
        if k.T_M != 2:
            errors.append(f"T_M={k.T_M} must be 2 (mono-tile locked)")
        if k.T_N != 4:
            errors.append(f"T_N={k.T_N} must be 4 (mono-tile locked)")
        if (k.W_M, k.W_N, k.W_K) not in VALID_MONO_TILE_MFMA:
            errors.append(
                f"WAVE=({k.W_M},{k.W_N},{k.W_K}) not in {VALID_MONO_TILE_MFMA}"
            )

        # -- VEC --
        expected_vec = 16 // sizeof_da  # 8 for bf16
        if (
            k.VEC_A != expected_vec
            or k.VEC_B != expected_vec
            or k.VEC_C != expected_vec
        ):
            errors.append(
                f"VEC=({k.VEC_A},{k.VEC_B},{k.VEC_C}) must all be {expected_vec}"
            )

        # -- User cap: B_M <= 192 --
        if k.B_M > 192:
            errors.append(f"B_M={k.B_M} exceeds mono-tile cap of 192")

        # -- Mono-tile must be non-OOB (intrinsic; launcher rejects unaligned) --
        if k.has_oob:
            errors.append("mono-tile is intrinsically non-OOB; has_oob must be False")

        # -- Block tile divisibility --
        if k.B_M % (k.W_M * k.T_M) != 0:
            errors.append(f"B_M={k.B_M} not div by W_M*T_M={k.W_M * k.T_M}")
        if k.B_N % (k.W_N * k.T_N) != 0:
            errors.append(f"B_N={k.B_N} not div by W_N*T_N={k.W_N * k.T_N}")
        if k.B_K % (k.W_K * 1) != 0:
            errors.append(f"B_K={k.B_K} not div by W_K*T_K={k.W_K}")

        E_M = k.B_M // (k.W_M * k.T_M) if (k.W_M * k.T_M) else 0
        E_N = k.B_N // (k.W_N * k.T_N) if (k.W_N * k.T_N) else 0
        E_K = k.B_K // k.W_K if k.W_K else 0

        # -- E_N divisibility (rb layout grouping by T_N/T_M = 2) --
        if k.T_M and (E_N * k.T_M) % k.T_N != 0:
            errors.append(
                f"E_N={E_N} not div by T_N/T_M={k.T_N // k.T_M} "
                f"(mono-tile rb layout grouping; needs B_N % 128 == 0)"
            )

        # -- LDS layout --
        smem_linear_wave = WARP_SIZE * 16 // sizeof_da  # 512 for bf16
        if k.B_K and smem_linear_wave % k.B_K != 0:
            errors.append(
                f"B_K={k.B_K} does not divide smem_linear_wave={smem_linear_wave}"
            )
        elif k.B_K:
            smem_sub = smem_linear_wave // k.B_K
            num_waves = k.BLOCK_SIZE // WARP_SIZE  # 8
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
            # ra layout: smem_sub_e_m = smem_sub / (W_M / T_N); E_M must
            # divide cleanly.
            if k.T_N and (k.W_M % k.T_N) != 0:
                errors.append(
                    f"W_M={k.W_M} not div by T_N={k.T_N} (mono-tile ra layout)"
                )
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

            # -- LDS footprint --
            # Mono-tile pipeline allocates `smem_a[2]` (double-buffered:
            # one compute slot + one fetch slot) and `smem_b[3]` (two
            # read slots sb_r0/sb_r1 plus a write slot sb_w; B is
            # consumed twice per MMA pair under the T_N/T_M = 2 grouping).
            # See the pipeline header (smem_a / smem_b allocation).
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

    # -- Instance generation --

    def gen_instance(self, k: OpusGemmInstance):
        if k.kernel_tag in (
            "a16w16",
            "a16w16_kbuf1_large_tile",
            "a16w16_kbuf2v",
            "a16w16_kbuf2v_bk128",
            "a16w16_kbuf3",
            "a16w16_kbuf1",
        ):
            info = self._validate_a16w16(k)
            print(
                f"  {k.name}: E=({info['E_M']},{info['E_N']},{info['E_K']})"
                f"  VGPR~{info['vgpr_est']}  AGPR={info['agprs']}"
                f"  LDS={info['lds_bytes'] // 1024}KiB"
                f"  K>={info['min_k']}"
            )
        elif k.kernel_tag == "a16w16_persistent":
            info = self._validate_a16w16_persistent(k)
            print(
                f"  {k.name}: E=({info['E_M']},{info['E_N']},{info['E_K']})"
                f"  VGPR~{info['vgpr_est']}  AGPR={info['agprs']}"
                f"  LDS={info['lds_bytes'] // 1024}KiB"
                f"  K>={info['min_k']}"
            )
        elif k.kernel_tag == "a16w16_mono_tile":
            info = self._validate_a16w16_mono_tile(k)
            print(
                f"  {k.name}: E=({info['E_M']},{info['E_N']},{info['E_K']})"
                f"  LDS={info['lds_bytes'] // 1024}KiB"
                f"  K>={info['min_k']}"
            )
        elif k.kernel_tag == "a16w16_flatmm":
            info = self._validate_a16w16_flatmm(k)
            print(
                f"  {k.name}: pfk={info['pfk']} "
                f"slots={info['slots']} "
                f"groups=({info['groups_bm']},{info['groups_bn']},{info['groups_bk']}) "
                f"LDS={info['lds_bytes'] // 1024}KiB K>={info['min_k']}"
            )
        elif k.kernel_tag == "a16w16_flatmm_splitk":
            info = self._validate_a16w16_flatmm_splitk(k)
            print(
                f"  {k.name}: pfk={info['pfk']} "
                f"slots={info['slots']} "
                f"comrep=({info['com_rep_m']},{info['com_rep_n']}) "
                f"LDS={info['lds_bytes'] // 1024}KiB K>={info['min_k']} WG={k.WG_PER_CU}"
            )
        elif k.kernel_tag in (
            "a16w16_kbuf3_sk",
            "a16w16_kbuf1_sk",
            "a16w16_fused_reduce",
        ):
            # gfx942 splitk: reuse split-barrier per-tile validator.
            info = self._validate_a16w16(k)
            print(
                f"  {k.name}: E=({info['E_M']},{info['E_N']},{info['E_K']})"
                f"  VGPR~{info['vgpr_est']}  AGPR={info['agprs']}"
                f"  LDS={info['lds_bytes'] // 1024}KiB"
            )

        pipeline_header = _pipeline_header_for(k)
        traits_header = _lookup(k, TRAITS_HEADER_MAP, GFX942_TRAITS_HEADER_MAP)
        kernel_func = _kernel_func_for(k)
        da, db = INPUT_DTYPE_MAP[k.kernel_tag]
        traits_name = _lookup(k, TRAITS_NAME_MAP, GFX942_TRAITS_NAME_MAP)
        kargs_name = _lookup(k, KARGS_NAME_MAP, GFX942_KARGS_NAME_MAP)

        # Track per-kid pipeline header so the per-kid device.cu can include
        # exactly the right one without re-running the full logic.
        self._kid_pipeline_header[k.name] = pipeline_header

        # Dispatch via registry (codegen/common.py EMIT_REGISTRY). Each arch
        # module under codegen/ self-registers (arch, kernel_tag) -> emit fn.
        # Adding a new arch (e.g. gfx1250) = create codegen/gen_instances_gfx1250.py
        # with register_emit("gfx1250", ...) calls + one import in this file.
        from codegen.common import dispatch_emit

        emit_kwargs = dict(
            pipeline_header=pipeline_header,
            traits_header=traits_header,
            kernel_func=kernel_func,
            da=da,
            db=db,
            traits_name=traits_name,
            kargs_name=kargs_name,
            kargs_template_vars=_kargs_template_vars,
            instance_impl_preamble=instance_impl_preamble,
            instance_impl_host_tu_split=instance_impl_host_tu_split,
            record_one_instantiation=_record_one_instantiation,
            make_host_decl=_make_host_decl,
            make_device_decl=_make_device_decl,
            A16W16_TUNE_HOST_EXTRA=A16W16_TUNE_HOST_EXTRA,
            A8W8_SCALE_HOST_EXTRA=A8W8_SCALE_HOST_EXTRA,
            A16W16_TUNE_TAGS=A16W16_TUNE_TAGS,
            BIAS_HOST_VALIDATE=self.BIAS_HOST_VALIDATE,
            SPLITK_REDUCE_FAST_ARCHES=SPLITK_REDUCE_FAST_ARCHES,
            V3_NVEC_ROWS=V3_NVEC_ROWS,
            V2_SUPPORTED_SPLITKS=V2_SUPPORTED_SPLITKS,
            # fused flag -- only consumed by gfx942 splitk emit. Forced True
            # for fused_reduce, False otherwise.
            fused=(k.kernel_tag == "a16w16_fused_reduce"),
        )
        dispatch_emit(self, k, **emit_kwargs)

    # Shared host-side bias validation + kargs population. Consumed by gfx950
    # noscale + gfx950 flatmm_splitk + gfx942 splitk emit modules.
    BIAS_HOST_VALIDATE = """
    const void* ptr_bias_ = nullptr;
    int stride_bias_batch_ = 0;
    if (bias.has_value()) {{
        const auto& bt = bias.value();
        AITER_CHECK(bt.is_contiguous(),
            "bias must be contiguous (got non-contiguous tensor)");
        AITER_CHECK(bt.dtype() == Y.dtype(),
            "bias dtype must match Y dtype (got bias=",
            AiterDtype_to_str(bt.dtype()),
            " Y=", AiterDtype_to_str(Y.dtype()), ")");
        if (bt.dim() == 1) {{
            AITER_CHECK(bt.size(0) == N,
                "bias 1D length must equal N (got bias.size(0)=", bt.size(0),
                " N=", N, ")");
            stride_bias_batch_ = 0;
        }} else if (bt.dim() == 2) {{
            AITER_CHECK(bt.size(0) == batch && bt.size(1) == N,
                "bias 2D shape must equal [batch, N] (got [", bt.size(0), ", ",
                bt.size(1), "] vs batch=", batch, " N=", N, ")");
            stride_bias_batch_ = N;
        }} else {{
            AITER_CHECK(false, "bias must be 1D [N] or 2D [batch, N]; got dim=",
                bt.dim());
        }}
        ptr_bias_ = bt.data_ptr();
    }}
"""

    def gen_lookup_dict(self, kernels_dict):
        """Emit opus_gemm_lookup.h with two (M,N,K)->kernel macros.

        Tuned-CSV driven lookup consumed by opus_gemm.cu's runtime
        `opus_dispatch_a16w16<CDataType>`. Two macros (BF16 / FP32)
        mirror `gen_a16w16_tune_lookup` and exist because splitk kids
        (200..210) are only emitted as `<fp32_t>` (their traits
        static_assert D_C==float, so referencing `splitk<bf16_t>`
        produces a linker error).

        Outdtype-aware bucketing
        ------------------------
        kernels_dict tuple keys carry the outdtype string in slot 3
        ((M, N, K, outdtype_str), produced by get_tune_dict). The BF16
        macro picks up rows whose outdtype is "torch.bfloat16" and the
        FP32 macro picks up rows whose outdtype is "torch.float32";
        same-(M,N,K) rows with different outdtypes therefore land in
        different macros and the two C++ maps can resolve to different
        kernels for the same shape. Legacy CSVs without an outdtype
        column are normalized to bf16 by get_tune_dict, so they only
        populate the BF16 map -- matching pre-outdtype-split behavior.

        Per-kid template argument rule:

          * a16w16 kid 4..9         -> `<CTYPE>` (both bf16/fp32 exist).
          * a16w16_flatmm 100..115  -> `<CTYPE>` (both exist).
          * a16w16_flatmm_splitk    -> always `<fp32_t>`. Splitk rows
            with outdtype=bf16 land in the BF16 map (with forced
            <fp32_t> template arg) and rows with outdtype=fp32 land in
            the FP32 map (also with <fp32_t>). Both work because the
            splitk reduce kernel handles the cast / passthrough at
            launch time based on the actual Y dtype.
        """
        # Sorted flat-array layout (was: {(M,N,K), kernel<CTYPE>} initializer list for std::unordered_map).
        HEADER = """#pragma once
// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Auto-generated. Do not edit. See gen_instances.py:gen_lookup_dict.
//
// Per-CTYPE sorted flat arrays for (M,N,K)->kernel runtime dispatch.
// Same (M,N,K) can resolve to different kernels in the BF16 vs FP32
// tables because get_tune_dict keys winners on (M, N, K, outdtype_str)
// and gen_lookup_dict buckets the rows into per-CTYPE macros below.
// splitk kids appear in either table with their main-kernel template
// forced to <fp32_t> (the reduce kernel handles the final Y cast at
// launch time).
//
// Lookup is std::lower_bound on the lex-ordered (M, N, K) key. See
// opus_gemm_arch_gfx950.cuh for the dispatch wrapper.
"""

        ENTRY_MATCH_CTYPE = """\
    {{ {{{M}, {N}, {K}}}, &{kernel_name}<CTYPE> }},  \\
"""
        ENTRY_FORCE_FP32 = """\
    {{ {{{M}, {N}, {K}}}, &{kernel_name}<fp32_t> }}, \\
"""

        # Map ctype short name -> CSV outdtype string emitted by the
        # tuner's result_to_df.
        ctype_to_outdtype = {
            "bf16_t": "torch.bfloat16",
            "fp32_t": "torch.float32",
        }

        def _emit_map(f, macro_name: str, ctype: str):
            # No body line break between `\` and the first entry; macro continuation requires every line
            # that participates in the definition ...
            f.write(f"#define {macro_name}(CTYPE) \\\n")
            target_outdtype = ctype_to_outdtype.get(ctype)
            # Collect all (M, N, K, kernel_name, is_splitk) rows for this
            # CTYPE first, so we can sort lex on (M, N, K) before emitting.
            rows = []
            for mnk, k in kernels_dict.items():
                if self.istune and isinstance(mnk, int):
                    # tune mode shouldn't reach here (gen_lookup_dict is
                    # for the runtime (M,N,K) map). Skip defensively.
                    continue
                if not (isinstance(mnk, tuple) and mnk[0] > 0):
                    continue
                if len(mnk) >= 4:
                    row_outdtype = str(mnk[3])
                    if target_outdtype is not None and row_outdtype != target_outdtype:
                        continue
                is_splitk = k.kernel_tag in SPLITK_TAGS
                if not is_splitk and ctype not in k.output_dtypes:
                    continue
                rows.append((int(mnk[0]), int(mnk[1]), int(mnk[2]), k.name, is_splitk))

            rows.sort(key=lambda r: (r[0], r[1], r[2]))
            n = len(rows)
            for i, (M, N, K, name, is_splitk) in enumerate(rows):
                entry = ENTRY_FORCE_FP32 if is_splitk else ENTRY_MATCH_CTYPE
                line = entry.format(M=M, N=N, K=K, kernel_name=name)
                if i == n - 1:
                    # Last entry: drop the trailing `\` so the macro
                    # ends cleanly. Strip the line's continuation.
                    line = line.rstrip().rstrip("\\").rstrip() + "\n"
                f.write(line)
            f.write("\n")

        with open(os.path.join(self.working_path, "opus_gemm_lookup.h"), "w") as f:
            f.write(HEADER)
            _emit_map(f, "GENERATE_OPUS_LOOKUP_TABLE_BF16", "bf16_t")
            _emit_map(f, "GENERATE_OPUS_LOOKUP_TABLE_FP32", "fp32_t")

    def gen_a16w16_tune_lookup(self, kernels_dict):
        """Emit opus_gemm_a16w16_tune_lookup.h with int-ID-to-kernel maps for tuning.

        Three a16w16-family tags share the 4-arg launcher signature
        (XQ, WQ, Y, int splitK):
          * a16w16 (split-barrier)      - output_dtypes=["fp32_t", "bf16_t"]
          * a16w16_flatmm (warp-spec)   - output_dtypes=["bf16_t", "fp32_t"]
          * a16w16_flatmm_splitk        - output_dtypes=["fp32_t"] ONLY
            (main kernel writes fp32 workspace; Y=bf16 via reduce kernel.
            Traits static_assert D_C=float, so no <bf16_t> instantiation
            exists for these kids.)

        The bf16 lookup map therefore must NOT reference splitk kids (their
        <bf16_t> specialization is never instantiated -> linker error). The
        dispatcher in opus_gemm.cu forces kid>=200 to the <fp32_t> branch
        anyway, so having them absent from the bf16 map is correct.

        Emit two macros side by side, gated on each kid's output_dtypes set.
        """
        # Same flat-array design as gen_lookup_dict, keyed on int kid instead of (M,N,K).
        HEADER = """#pragma once
// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Auto-generated. Do not edit. See gen_instances.py:gen_a16w16_tune_lookup.
//
// Per-CTYPE sorted flat arrays for kid->kernel tune dispatch. Kids whose
// output_dtypes doesn't include CTYPE are omitted from that CTYPE's table
// (splitk kids only live in the fp32 table). See
// opus_gemm_arch_gfx950.cuh for the dispatch wrapper.
"""
        ENTRY = """\
    {{ {kid}, &{kernel_name}<CTYPE> }},  \\
"""

        def _emit_map(f, macro_name, ctype):
            f.write(f"#define {macro_name}(CTYPE) \\\n")
            rows = []
            for kid, k in kernels_dict.items():
                if not (isinstance(kid, int) and k.kernel_tag in A16W16_TUNE_TAGS):
                    continue
                if ctype not in k.output_dtypes:
                    continue
                rows.append((kid, k.name))
            rows.sort(key=lambda r: r[0])
            n = len(rows)
            for i, (kid, name) in enumerate(rows):
                line = ENTRY.format(kid=kid, kernel_name=name)
                if i == n - 1:
                    line = line.rstrip().rstrip("\\").rstrip() + "\n"
                f.write(line)
            f.write("\n")

        with open(
            os.path.join(self.working_path, "opus_gemm_a16w16_tune_lookup.h"), "w"
        ) as f:
            f.write(HEADER)
            # Use explicit per-CTYPE macro names; the dispatcher in opus_gemm.cu calls the right one from
            # each opus_a16w16_tune_dispatch<CDat...
            _emit_map(f, "GENERATE_A16W16_TUNE_LOOKUP_BF16", "bf16_t")
            _emit_map(f, "GENERATE_A16W16_TUNE_LOOKUP_FP32", "fp32_t")

    def gen_manifest_head(self, kernels_dict):
        # Forward declarations for every launcher symbol the dispatcher references.
        MANIFEST_HEAD = """#pragma once
// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
#include "aiter_tensor.h"
#include <cstdlib>
#include <optional>
"""
        MANIFEST_SCALE = """
template <typename D_C>
void
{kernel_name}(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &Y,
    std::optional<aiter_tensor_t> x_scale,
    std::optional<aiter_tensor_t> w_scale);
"""
        # a8w8 noscale (3 args, no splitK): stays compatible with
        # opus_gemm_lookup.h where a8w8 kids live.
        MANIFEST_NOSCALE_3ARG = """
template <typename D_C>
void
{kernel_name}(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &Y);
"""
        # a16w16 family (5 args with optional bias + splitK): shared signature for tune lookup.
        MANIFEST_NOSCALE_4ARG = """
template <typename D_C>
void
{kernel_name}(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &Y,
    std::optional<aiter_tensor_t> bias,
    int splitK);
"""
        with open(os.path.join(self.working_path, "opus_gemm_manifest.h"), "w") as f:
            f.write(MANIFEST_HEAD)
            for mnk, k in kernels_dict.items():
                if k.kernel_tag in A16W16_TUNE_TAGS:
                    f.write(MANIFEST_NOSCALE_4ARG.format(kernel_name=k.name))
                elif k.kernel_tag in NOSCALE_TAGS:
                    f.write(MANIFEST_NOSCALE_3ARG.format(kernel_name=k.name))
                else:
                    f.write(MANIFEST_SCALE.format(kernel_name=k.name))

    # -- Per-pass TU emission -- Replaces the old "one .cpp per (kid, dtype)" scheme.

    def _emit_fused_host_tu(self):
        """Emit per-arch HOST translation units (one .cu per arch).

        Splitting by arch lets each TU's reduce-kernel forward decl match
        its arch's launcher emit signature: gfx950 launchers pass
        ws_handle* to splitk_reduce_kernel, gfx942 launchers pass float*.
        In mixed-arch builds (GPU_ARCHS=gfx942;gfx950) a single host TU
        would force one signature for both arches -> no matching function
        for the other arch's launcher -> link / compile fail.

        Per-arch buckets also keep impl-include sets disjoint: gfx950 TU
        only #includes gfx950 kid impl .cuh, etc. ODR clashes between
        same-named layout helpers in different pipeline headers are
        naturally avoided.
        """

        # Bucket host/device instantiations by arch. We classify by the
        # kid_name prefix `opus_gemm_<arch>_*`; legacy kid names without
        # explicit arch prefix default to gfx950 (matches kid_arch).
        def _kid_name_arch(kid_name):
            for ap in ("gfx942", "gfx950"):
                if kid_name.startswith(f"opus_gemm_{ap}_"):
                    return ap
            return "gfx950"

        host_by_arch = {}
        for row in self._host_instantiations:
            arch = _kid_name_arch(row["kid_name"])
            host_by_arch.setdefault(arch, []).append(row)

        for arch, rows in host_by_arch.items():
            impl_includes = sorted({row["kid_name"] for row in rows})
            host_body = "".join(row["host_decl"] for row in rows)
            # gfx950 splitk launcher passes ws_handle*; gfx942 passes raw float*.
            if arch == "gfx950":
                _fwd_ws_decl = '#include "gfx950/opus_gemm_traits_a16w16_gfx950.cuh"\n'
                _fwd_ws_arg = "const opus_splitk_ws_handle* ws_handle"
            else:
                _fwd_ws_decl = ""
                _fwd_ws_arg = "const float* workspace"
            forward_decls = (
                "// Forward declaration only. Specialisations live in per-arch device TUs.\n"
                f"{_fwd_ws_decl}"
                "template<int VEC_, int BLOCK_, typename D_OUT,\n"
                "         bool HAS_BIAS_, typename D_BIAS_,\n"
                "         bool HAS_OOB_>\n"
                "__global__ void splitk_reduce_kernel(\n"
                f"    {_fwd_ws_arg}, D_OUT* c_out,\n"
                "    int split_k, int M, int N, int batch,\n"
                "    int padded_M, int padded_N,\n"
                "    const D_BIAS_* bias, int stride_bias_batch);\n"
                "template<int SPLIT_K, int VEC_, int BLOCK_, typename D_OUT,\n"
                "         bool HAS_BIAS_, typename D_BIAS_>\n"
                "__global__ void splitk_reduce_kernel_v2(\n"
                "    const float* workspace, D_OUT* c_out,\n"
                "    int M, int N, int batch,\n"
                "    int padded_M, int padded_N,\n"
                "    const D_BIAS_* bias, int stride_bias_batch);\n"
                "template<int SPLIT_K, int N_VEC, int ROWS_PER_BLOCK, int VEC_,\n"
                "         typename D_OUT, bool HAS_BIAS_, typename D_BIAS_>\n"
                "__global__ void splitk_reduce_kernel_v3(\n"
                "    const float* workspace, D_OUT* c_out,\n"
                "    int M, int N, int batch,\n"
                "    int padded_M, int padded_N,\n"
                "    const D_BIAS_* bias, int stride_bias_batch);\n"
            )
            contents = (
                "// SPDX-License-Identifier: MIT\n"
                "// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.\n"
                "//\n"
                f"// Auto-generated per-arch host TU ({arch}). See gen_instances.py:_emit_fused_host_tu.\n"
                "#ifndef __HIP_DEVICE_COMPILE__\n"
                "#define OPUS_FUSED_HOST_TU 1\n"
                '#include "aiter_tensor.h"\n'
                '#include "aiter_stream.h"\n'
                "#include <optional>\n"
                + forward_decls
                + "".join(f'#include "impl/{name}.cuh"\n' for name in impl_includes)
                + host_body
                + "#endif // host pass only\n"
            )
            Path(
                os.path.join(self.instances_path, f"all_instances_host_{arch}.cu")
            ).write_text(contents)

    def _emit_device_tus(self):
        """Emit one device-only .device.cu per (kid, dtype).

        Each .cu includes the kid's pipeline header (so the kernel
        template body is visible) and explicitly instantiates the
        kernel template. The companion fused host TU's <<<...>>> calls
        end up referencing host stubs that the linker resolves to the
        instantiations here.

        This TU does not include torch -- it doesn't need to, because
        the host pass only sees `template __global__ void k<...>(...)`
        which doesn't depend on any libtorch type. Skipping the torch
        parse on host pass drops each device TU's compile to ~1.5s
        (down from ~13s when torch was forced in).
        """
        for row in self._device_instantiations:
            name = row["kid_name"]
            dtype = row["dtype"]
            # Include the kid's .cuh -- it transitively pulls in the full pipeline header (because
            # OPUS_FUSED_HOST_TU is NOT defined here) an...
            contents = (
                "// SPDX-License-Identifier: MIT\n"
                "// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.\n"
                "//\n"
                "// Auto-generated. Do not edit. See gen_instances.py:_emit_device_tus.\n"
                "//\n"
                "// Device-only translation unit for one (kid, dtype) pair.\n"
                "// Compiled with -D__HIPCC_RTC__ (per-source flag in\n"
                "// optCompilerConfig.json) so the host pass takes the\n"
                "// minimal branch -- no torch, no full HIP runtime.\n"
                f'#include "impl/{name}.cuh"\n' + row["device_decl"]
            )
            Path(
                os.path.join(self.instances_path, f"{name}_C{dtype}.device.cu")
            ).write_text(contents)

    def _emit_splitk_reduce_tu(self):
        """Emit a single splitk_reduce.device.cu carrying the 4 reduce
        kernel specialisations (D_OUT bf16/fp32 x HAS_BIAS true/false).

        Why a dedicated TU: each splitk kid's fused-host launcher body
        does <<<...>>> on all 4 reduce specialisations to handle every
        Y dtype / bias combination at runtime. That used to inline the
        4 `template __global__` instantiations into every splitk kid's
        device.cu (see _gen_flatmm_splitk_instance comment). The linker
        deduped the resulting weak symbols, but each splitk TU still
        paid the full RA + ISA-emit cost on its own compile -- ~0.4s
        wall per TU x 23 splitk TUs = ~9s of duplicated CPU work that
        also lengthened each TU's individual wall and tightened the
        ninja schedule on the slowest splitk kid.

        Centralising them here means:
          * each splitk device.cu only carries its own main-kernel
            instantiation (~50% smaller .o, ~0.3-0.5s less wall each),
          * one new tiny TU compiles the 4 reduces in ~1s wall total,
          * link still works because the reduce symbols are __global__
            (the host stubs the fused TU emits are linked against this
            single TU's GPU code, not against per-splitk-TU copies).

        The reduce kernel template lives in splitk_reduce_{arch}.cuh,
        with one header per arch. Both headers define the same
        `splitk_reduce_kernel` template (arch-guarded internally), so the
        TU only ever includes one. We pick a kid-driven arch when at
        least one kid of that arch has an indep-reduce splitk kid in the
        build; otherwise we fall back to gfx950 (legacy default).
        """
        # Bucket present archs from splitk kids.
        present_archs = set()
        for row in self._device_instantiations:
            name = row["kid_name"]
            if "splitk_fused" in name or "splitk_atomic" in name:
                continue
            for arch_prefix in ("gfx942", "gfx950"):
                if f"opus_gemm_{arch_prefix}_splitk_" in name:
                    present_archs.add(arch_prefix)
                    break
            else:
                if "splitk" in name:
                    present_archs.add("gfx950")

        # Emit one reduce device TU per arch (each arch's reduce kernel
        # ABI is different: gfx942 = float*, gfx950 = ws_handle*).
        for reduce_arch in sorted(present_archs):
            reduce_header = f"{reduce_arch}/splitk_reduce_{reduce_arch}.cuh"
            ws_ptr_type = (
                "const opus_splitk_ws_handle*"
                if reduce_arch == "gfx950"
                else "const float*"
            )
            contents = (
                "// SPDX-License-Identifier: MIT\n"
                "// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.\n"
                "//\n"
                f"// Auto-generated per-arch reduce TU ({reduce_arch}). See gen_instances.py:_emit_splitk_reduce_tu.\n"
                f'#include "{reduce_header}"\n'
                "// HAS_OOB=true variants\n"
                f"template __global__ void splitk_reduce_kernel<16, 64, __bf16, true,  __bf16, true>(\n"
                f"    {ws_ptr_type}, __bf16*, int, int, int, int, int, int,\n"
                f"    const __bf16*, int);\n"
                f"template __global__ void splitk_reduce_kernel<16, 64, __bf16, false, __bf16, true>(\n"
                f"    {ws_ptr_type}, __bf16*, int, int, int, int, int, int,\n"
                f"    const __bf16*, int);\n"
                f"template __global__ void splitk_reduce_kernel<16, 64, float,  true,  float,  true>(\n"
                f"    {ws_ptr_type}, float*,  int, int, int, int, int, int,\n"
                f"    const float*,  int);\n"
                f"template __global__ void splitk_reduce_kernel<16, 64, float,  false, float,  true>(\n"
                f"    {ws_ptr_type}, float*,  int, int, int, int, int, int,\n"
                f"    const float*,  int);\n"
                "// HAS_OOB=false variants\n"
                f"template __global__ void splitk_reduce_kernel<16, 64, __bf16, true,  __bf16, false>(\n"
                f"    {ws_ptr_type}, __bf16*, int, int, int, int, int, int,\n"
                f"    const __bf16*, int);\n"
                f"template __global__ void splitk_reduce_kernel<16, 64, __bf16, false, __bf16, false>(\n"
                f"    {ws_ptr_type}, __bf16*, int, int, int, int, int, int,\n"
                f"    const __bf16*, int);\n"
                f"template __global__ void splitk_reduce_kernel<16, 64, float,  true,  float,  false>(\n"
                f"    {ws_ptr_type}, float*,  int, int, int, int, int, int,\n"
                f"    const float*,  int);\n"
                f"template __global__ void splitk_reduce_kernel<16, 64, float,  false, float,  false>(\n"
                f"    {ws_ptr_type}, float*,  int, int, int, int, int, int,\n"
                f"    const float*,  int);\n"
            )
            if reduce_arch in SPLITK_REDUCE_FAST_ARCHES:
                contents += "// V2 (split_k static-unroll, no OOB) instantiations -- gfx942 only\n"
                for sk in V2_SUPPORTED_SPLITKS:
                    contents += (
                        f"template __global__ void splitk_reduce_kernel_v2<{sk}, 8, 8, __bf16, true,  __bf16>(\n"
                        "    const float*, __bf16*, int, int, int, int, int,\n"
                        "    const __bf16*, int);\n"
                        f"template __global__ void splitk_reduce_kernel_v2<{sk}, 8, 8, __bf16, false, __bf16>(\n"
                        "    const float*, __bf16*, int, int, int, int, int,\n"
                        "    const __bf16*, int);\n"
                    )
                contents += "// V3 (multi-row per wg, BLOCK=64=1 wave) instantiations\n"
                for nvec, rows in V3_NVEC_ROWS:
                    for sk in V2_SUPPORTED_SPLITKS:
                        contents += (
                            f"template __global__ void splitk_reduce_kernel_v3<{sk}, {nvec}, {rows}, 8, __bf16, true,  __bf16>(\n"
                            "    const float*, __bf16*, int, int, int, int, int,\n"
                            "    const __bf16*, int);\n"
                            f"template __global__ void splitk_reduce_kernel_v3<{sk}, {nvec}, {rows}, 8, __bf16, false, __bf16>(\n"
                            "    const float*, __bf16*, int, int, int, int, int,\n"
                            "    const __bf16*, int);\n"
                        )
            Path(
                os.path.join(
                    self.instances_path, f"splitk_reduce_{reduce_arch}.device.cu"
                )
            ).write_text(contents)

    def gen_instances(self, kernels_dict):
        if os.path.exists(self.impl_path):
            shutil.rmtree(self.impl_path)
        os.mkdir(self.impl_path)
        if os.path.exists(self.instances_path):
            shutil.rmtree(self.instances_path)
        os.mkdir(self.instances_path)

        # Reset the instantiation accumulators so reruns under the same
        # codegen object don't double-emit.
        self._host_instantiations = []
        self._device_instantiations = []

        for mnk, k in kernels_dict.items():
            self.gen_instance(k)

        # Emit one fused HOST TU + N device TUs (one per kid, dtype) + one dedicated splitk_reduce.device.cu.
        self._emit_fused_host_tu()
        self._emit_device_tus()
        # Only emit the standalone reduce TU if the build actually has a splitk kid (otherwise the fused
        # host TU will never reference any...
        needs_reduce_tu = any(
            ("flatmm_splitk" in row["kid_name"])
            or (
                "_splitk_" in row["kid_name"]
                and "splitk_fused" not in row["kid_name"]
                and "splitk_atomic" not in row["kid_name"]
            )
            for row in self._device_instantiations
        )
        if needs_reduce_tu:
            self._emit_splitk_reduce_tu()

        self.gen_lookup_dict(kernels_dict)
        self.gen_manifest_head(kernels_dict)
        self.gen_a16w16_tune_lookup(kernels_dict)


def get_tune_dict(tune_dict_csv):
    """Load a tuned CSV into the lookup-dict shape consumed by gen_lookup_dict.

    Key layout
    ----------
    Tuple keys: (M, N, K, outdtype_str). Promoting outdtype into the key
    is what lets a single (M, N, K) shape carry distinct winners for bf16
    vs fp32 output (the underlying main kernel hardware rules differ
    enough that the best kid is not always the same; e.g. fp32 output
    biases reduce-bound shapes toward larger split-K). gen_lookup_dict
    then writes outdtype="torch.bfloat16" rows only into the BF16 (M,N,K)
    map and outdtype="torch.float32" rows only into the FP32 (M,N,K) map.

    Backwards compat
    ----------------
    Legacy CSVs without an `outdtype` column are interpreted as
    bf16-output (matches what the tuner used to write). int keys from
    default_kernels_dict are passed through untouched -- gen_lookup_dict
    skips them via the `isinstance(mnk, tuple) and mnk[0] > 0` guard.
    """
    tune_dict = default_kernels_dict
    if os.path.exists(tune_dict_csv):
        tune_df = pd.read_csv(tune_dict_csv)
        if torch.cuda.is_available():
            gpu = torch.cuda.current_device()
            device_properties = torch.cuda.get_device_properties(gpu)
            cu_num = device_properties.multi_processor_count
            tune_df = tune_df[tune_df["cu_num"] == cu_num].reset_index()
        # Accept either the legacy "kernelId" column or the new "solidx" column (matches
        # aiter/configs/model_configs/gptoss_bf16_tuned_ge...
        kid_col = "solidx" if "solidx" in tune_df.columns else "kernelId"
        has_outdtype = "outdtype" in tune_df.columns
        for i in range(len(tune_df)):
            M = tune_df.loc[i, "M"]
            N = tune_df.loc[i, "N"]
            K = tune_df.loc[i, "K"]
            outdtype = (
                str(tune_df.loc[i, "outdtype"]) if has_outdtype else "torch.bfloat16"
            )
            kid = int(tune_df.loc[i, kid_col])
            if kid in kernels_list:
                tune_dict[(M, N, K, outdtype)] = kernels_list[kid]
    return tune_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="generate",
        description="gen API for opus GEMM kernel instances",
    )

    parser.add_argument(
        "-w",
        "--working_path",
        default="./",
        required=False,
        help="the path where all the blobs are going to be generated",
    )

    parser.add_argument(
        "--tune",
        action="store_true",
        default=False,
        help="generate all kernel instances for tuning (id-based lookup)",
    )

    parser.add_argument(
        "--kernel_tag",
        default=None,
        required=False,
        help="filter kernels by tag (e.g. a16w16, a16w16_flatmm, a16w16_flatmm_splitk, a8w8, a8w8_scale)",
    )

    parser.add_argument(
        "--tune_files",
        default=None,
        required=False,
        help=(
            "Colon-separated list of glob patterns pointing at tuned BF16 "
            "GEMM CSVs (e.g. aiter/configs/bf16_tuned_gemm.csv and "
            "aiter/configs/model_configs/*_bf16_tuned_gemm.csv). Each "
            "file is filtered by `libtype == 'opus'`; surviving rows "
            "contribute their `solidx` to the subset-compile set S and "
            "are also baked into opus_gemm_lookup.h via "
            "GENERATE_OPUS_LOOKUP_TABLE_*. Without this flag we still "
            "generate a working module (only HEURISTIC_DEFAULT_KIDS + "
            "sidecar contents), the lookup table stays empty, and the "
            "C++ dispatch falls through to the heuristic for every "
            "untuned shape."
        ),
    )

    parser.add_argument(
        "--compiled_kids_sidecar",
        default=None,
        required=False,
        help=(
            "Path to the subset-compile sidecar (JSON list of int kids). "
            "Defaults to {working_path}/compiled_kids.json. The sidecar "
            "captures the union of CSV opus rows + previous sidecar "
            "contents + HEURISTIC_DEFAULT_KIDS so subsequent rebuilds "
            "are idempotent (no rebuild if every required kid is already "
            "in the .so). gradlib's GemmTuner and opus_gemm_tune.py "
            "expand this sidecar in tuner-startup to add new kids before "
            "triggering an AITER_REBUILD."
        ),
    )

    # Legacy --tune_file alias kept for backward compat with any existing
    # invocations / scripts. Treated as `--tune_files <path>`.
    parser.add_argument(
        "--tune_file",
        default=None,
        required=False,
        help="[DEPRECATED] alias for --tune_files (single path). Use --tune_files instead.",
    )

    args = parser.parse_args()
    if args.tune_files is None and args.tune_file is not None:
        args.tune_files = args.tune_file
    TAG_TO_LIST = {
        "a8w8_scale": a8w8_scale_kernels_list,
        "a8w8": a8w8_kernels_list,
        "a16w16": a16w16_kernels_list,
        "a16w16_flatmm": a16w16_flatmm_kernels_list,
        "a16w16_flatmm_splitk": a16w16_flatmm_splitk_kernels_list,
        "a16w16_mono_tile": a16w16_mono_tile_kernels_list,
        # gfx942 kid range (50000+); two-bucket registry: nosplit + splitk.
        "gfx942_nosplit": gfx942_nosplit_kernels_list,
        "gfx942_splitk": gfx942_splitk_kernels_list,
    }

    # --- Compute the subset-compile set S ------------------------------------ S = (CSV opus rows'
    # kids) ?

    def _expand_tune_paths(spec):
        out = []
        seen = set()
        if not spec:
            return out
        for pat in str(spec).split(os.pathsep):
            pat = pat.strip()
            if not pat:
                continue
            for path in sorted(glob.glob(pat)):
                if path in seen:
                    continue
                seen.add(path)
                out.append(path)
        return out

    csv_kids: set[int] = set()
    csv_paths = _expand_tune_paths(args.tune_files)
    for path in csv_paths:
        try:
            df = pd.read_csv(path)
        except (pd.errors.EmptyDataError, FileNotFoundError):
            continue
        if "libtype" not in df.columns:
            continue
        df = df[df["libtype"] == "opus"]
        if df.empty:
            continue
        kid_col = (
            "solidx"
            if "solidx" in df.columns
            else ("kernelId" if "kernelId" in df.columns else None)
        )
        if kid_col is None:
            continue
        for v in df[kid_col].dropna().tolist():
            try:
                csv_kids.add(int(v))
            except (TypeError, ValueError):
                continue

    sidecar_path = args.compiled_kids_sidecar or os.path.join(
        args.working_path, "compiled_kids.json"
    )
    sidecar_kids: set[int] = set()
    if os.path.exists(sidecar_path):
        try:
            with open(sidecar_path) as f:
                sidecar_kids = set(int(x) for x in json.load(f))
        except (OSError, ValueError):
            sidecar_kids = set()

    # The compile set: union, intersected with valid kernels_list entries.
    valid_kids = set(kernels_list.keys())
    S = (csv_kids | sidecar_kids | set(HEURISTIC_DEFAULT_KIDS)) & valid_kids

    # Per-arch filter: drop kids whose arch_prefix is not in the target build set.
    _kid_arch = _kid_arch_common

    target_arches = None
    gpu_archs_env = os.getenv("GPU_ARCHS", "native").strip()
    explicit = [
        a.strip().lower()
        for a in gpu_archs_env.split(";")
        if a.strip() and a.strip().lower() != "native"
    ]
    if explicit:
        target_arches = set(explicit)
    else:
        # GPU_ARCHS=native: probe live GPU; skip filter if rocminfo unavailable.
        try:
            from aiter.jit.utils.chip_info import get_gfx_runtime

            target_arches = {get_gfx_runtime().lower()}
        except Exception:
            target_arches = None

    if target_arches is not None:
        before = len(S)
        S = {kid for kid in S if _kid_arch(kernels_list[kid]) in target_arches}
        dropped = before - len(S)
        print(
            f"[opus gen_instances] arch filter: target={sorted(target_arches)} "
            f"dropped {dropped} off-arch kids from |S|"
        )

    # Emit OPUS_BUILD_HAS_* macros so opus_gemm.cu can gate per-arch dispatch
    # tables: a single-arch build (GPU_ARCHS=gfx950) must not link gfx942
    # launcher symbols and vice versa.
    archs_for_header = (
        sorted(target_arches) if target_arches is not None else ["gfx942", "gfx950"]
    )
    with open(os.path.join(args.working_path, "opus_build_archs.h"), "w") as f:
        f.write(
            "// SPDX-License-Identifier: MIT\n"
            "// Auto-generated. See gen_instances.py.\n"
            "#pragma once\n"
        )
        for a in archs_for_header:
            f.write(f"#define OPUS_BUILD_HAS_{a.upper()} 1\n")

    # a8w8 (kid 1, 2) referenced unconditionally by dispatcher; symbols must exist on every arch.
    S |= set(a8w8_scale_kernels_list.keys())
    S |= set(a8w8_kernels_list.keys())

    # Honor --kernel_tag as a developer override that *further restricts* the set (within the a16w16
    # / a8w8 families).
    if args.kernel_tag:
        tag_keys = set(TAG_TO_LIST.get(args.kernel_tag, {}).keys())
        if tag_keys:
            # Restrict to the requested family + heuristic defaults + a8w8 dispatch.
            S = (S & tag_keys) | set(HEURISTIC_DEFAULT_KIDS)
            S |= set(a8w8_scale_kernels_list.keys())
            S |= set(a8w8_kernels_list.keys())

    # Heuristic-fallback invariant (single source of truth: opus_gemm_common.py).
    required_heuristic = set(heuristic_kids_for_arch(target_arches))
    missing_heuristic = required_heuristic - S
    assert not missing_heuristic, (
        f"Subset-compile error: heuristic-fallback kids "
        f"{sorted(missing_heuristic)} are missing from the compile set S; "
        f"opus_a16w16_heuristic_kid_gfx950() would return an unbakeable "
        f"kid. Add them to the compile set or update HEURISTIC_DEFAULT_KIDS "
        f"in csrc/opus_gemm/opus_gemm_common.py."
    )

    # Build the per-kid dict that drives codegen.
    kdict = {kid: kernels_list[kid] for kid in sorted(S)}

    print(
        f"[opus gen_instances] subset compile: |S|={len(S)} kids "
        f"(CSV={len(csv_kids)}, sidecar={len(sidecar_kids)}, heuristic={len(HEURISTIC_DEFAULT_KIDS)})"
    )

    codegen = opus_gemm_codegen(args.working_path, args.tune)
    codegen.gen_instances(kdict)

    # Bake the (M, N, K) -> kernel runtime lookup.
    if csv_paths:
        # Concatenate all opus rows from all matched CSV files (filtered by libtype).
        combined_frames = []
        for path in csv_paths:
            try:
                df = pd.read_csv(path)
            except (pd.errors.EmptyDataError, FileNotFoundError):
                continue
            if "libtype" not in df.columns:
                continue
            df = df[df["libtype"] == "opus"]
            if df.empty:
                continue
            # Drop off-arch kids: lookup must only reference symbols S actually emitted.
            if "solidx" in df.columns:
                df = df[df["solidx"].astype(int).isin(S)]
                if df.empty:
                    continue
            combined_frames.append(df)

        if combined_frames:
            combined = pd.concat(combined_frames, ignore_index=True).drop_duplicates()
            tmp_csv = os.path.join(args.working_path, "_combined_opus_tuned.csv")
            combined.to_csv(tmp_csv, index=False)
            tune_dict = get_tune_dict(tmp_csv)
            try:
                os.remove(tmp_csv)
            except OSError:
                pass
            # Filter tune_dict entries to those whose kid is in S (defense
            # in depth -- valid_kids should have already caught everything).
            filtered = {}
            for k, v in tune_dict.items():
                if isinstance(k, tuple) and k[0] > 0:
                    # Find the kid for this entry by reverse-lookup against S.
                    filtered[k] = v
                else:
                    filtered[k] = v  # default_kernels_dict negative-int entries
            codegen.gen_lookup_dict(filtered)
            n_real = sum(1 for k in filtered if isinstance(k, tuple) and k[0] > 0)
            print(
                f"[opus gen_instances] baked {n_real} tuned entries from "
                f"{len(csv_paths)} CSV file(s) into opus_gemm_lookup.h"
            )
        else:
            print(
                f"[opus gen_instances] no `libtype=='opus'` rows found in "
                f"{len(csv_paths)} CSV file(s); using empty lookup"
            )
    elif args.tune_files:
        print(
            f"[opus gen_instances] --tune_files {args.tune_files} matched no "
            f"existing files; using empty lookup"
        )

    # Persist the expanded compile set so subsequent rebuilds reuse it.
    try:
        os.makedirs(os.path.dirname(sidecar_path) or ".", exist_ok=True)
    except OSError:
        pass
    with open(sidecar_path, "w") as f:
        json.dump(sorted(S), f)
    print(f"[opus gen_instances] wrote sidecar with {len(S)} kids: {sidecar_path}")
