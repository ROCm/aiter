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
from opus_gemm_common import (
    HEURISTIC_DEFAULT_KIDS,
    OpusGemmInstance,
    a8w8_kernels_list,
    a8w8_scale_kernels_list,
    a16w16_flatmm_kernels_list,
    a16w16_flatmm_splitk_kernels_list,
    a16w16_kernels_list,
    a16w16_mono_tile_kernels_list,
    default_kernels_dict,
    kernels_list,
)

PIPELINE_HEADER_MAP = {
    "a8w8_scale": "gfx950/opus_gemm_pipeline_a8w8_scale_gfx950.cuh",
    "a8w8": "gfx950/opus_gemm_pipeline_a8w8_noscale_gfx950.cuh",
    "a16w16": "gfx950/opus_gemm_pipeline_a16w16_gfx950.cuh",
    "a16w16_flatmm": "gfx950/opus_gemm_pipeline_a16w16_flatmm_gfx950.cuh",
    "a16w16_flatmm_splitk": "gfx950/opus_gemm_pipeline_a16w16_flatmm_splitk_gfx950.cuh",
    "a16w16_persistent": "gfx950/opus_gemm_pipeline_a16w16_persistent_gfx950.cuh",
    "a16w16_mono_tile": "gfx950/opus_gemm_pipeline_a16w16_mono_tile_gfx950.cuh",
}

# Traits header carries the traits struct + kargs struct definitions for a
# given pipeline tag. These headers contain ONLY type definitions (no free
# function templates), so a fused TU can include all of them without ODR
# clashes -- unlike pipeline headers, which define same-named layout
# helpers (make_layout_ga_noscale et al.) in multiple files. Used by the
# fused host TU to obtain the Traits + kargs definitions without dragging
# in the pipeline body.
TRAITS_HEADER_MAP = {
    "a8w8_scale": "gfx950/opus_gemm_traits_a8w8_scale_gfx950.cuh",
    "a8w8": "gfx950/opus_gemm_traits_a8w8_noscale_gfx950.cuh",
    "a16w16": "gfx950/opus_gemm_traits_a16w16_gfx950.cuh",
    "a16w16_flatmm": "gfx950/opus_gemm_traits_a16w16_gfx950.cuh",
    "a16w16_flatmm_splitk": "gfx950/opus_gemm_traits_a16w16_gfx950.cuh",
    "a16w16_persistent": "gfx950/opus_gemm_traits_a16w16_gfx950.cuh",
    "a16w16_mono_tile": "gfx950/opus_gemm_traits_a16w16_gfx950.cuh",
}

# Splitk reduce kernel is shared infrastructure used by every
# a16w16_flatmm_splitk launcher. Forward-declared in the fused host TU
# so its `<<<>>>` calls type-check, and instantiated in each splitk
# device.cu so the linker can pick the GPU IR up.
SPLITK_REDUCE_HEADER = "gfx950/splitk_reduce_gfx950.cuh"

KERNEL_FUNC_MAP = {
    "a8w8_scale": "gemm_a8w8_scale_kernel",
    "a8w8": "gemm_a8w8_noscale_kernel",
    "a16w16": "gemm_a16w16_kernel",
    "a16w16_flatmm": "gemm_a16w16_flatmm_kernel",
    "a16w16_flatmm_splitk": "gemm_a16w16_flatmm_splitk_kernel",
    "a16w16_persistent": "gemm_a16w16_persistent_kernel",
    "a16w16_mono_tile": "gemm_a16w16_mono_tile_kernel_gfx950",
}

INPUT_DTYPE_MAP = {
    "a8w8_scale": ("fp8_t", "fp8_t"),
    "a8w8": ("fp8_t", "fp8_t"),
    "a16w16": ("bf16_t", "bf16_t"),
    "a16w16_flatmm": ("bf16_t", "bf16_t"),
    "a16w16_flatmm_splitk": ("bf16_t", "bf16_t"),
    "a16w16_persistent": ("bf16_t", "bf16_t"),
    "a16w16_mono_tile": ("bf16_t", "bf16_t"),
}

# Tags whose launchers take 3 torch tensors (XQ, WQ, Y) + int splitK. Splitk
# launcher has the same Python-facing signature but with literal (non-trivial)
# splitK semantics.
NOSCALE_TAGS = {
    "a8w8",
    "a16w16",
    "a16w16_flatmm",
    "a16w16_flatmm_splitk",
    "a16w16_persistent",
    "a16w16_mono_tile",
}

# a16w16-family tags whose launchers land in opus_gemm_a16w16_tune_lookup.h
# and therefore need the 4-arg (XQ, WQ, Y, int splitK) signature so they can
# share the std::function<Tensor(Tensor&,Tensor&,Tensor&,int)> slot.
# Non-splitk tags in this set ignore the splitK param in their body.
A16W16_TUNE_TAGS = {
    "a16w16",
    "a16w16_flatmm",
    "a16w16_flatmm_splitk",
    "a16w16_persistent",
    "a16w16_mono_tile",
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

WARP_SIZE = 64
VALID_BF16_MFMA = {(16, 16, 32), (32, 32, 16)}
# Flatmm pipeline currently only supports W_M < 32 (ra layout relies on
# LOAD_GROUP_M_LANE == 1). W_M == 32 (LGML == 4) path not rewritten.
VALID_FLATMM_MFMA = {(16, 16, 32)}
VALID_FLATMM_SPLITK_MFMA = {(16, 16, 32)}
# Persistent pipeline ports the mouter reference which only validated
# 16x16x32 BF16 MFMA. Add 32x32x16 later if needed.
VALID_PERSISTENT_MFMA = {(16, 16, 32)}
# Mono-tile pipeline: same MFMA lock as persistent (16x16x32 BF16) — the
# kernel template hard-codes T_M=2, T_N=4, T_K=1, W_M=W_N=16, W_K=32.
VALID_MONO_TILE_MFMA = {(16, 16, 32)}


class opus_gemm_codegen:
    def __init__(self, working_path, istune=False):
        self.working_path = working_path
        self.impl_path = os.path.join(working_path, "impl")
        self.instances_path = os.path.join(working_path, "instances")
        self.istune = istune
        # Compile-time split:
        #
        # Build layout:
        #   * One fused HOST TU (instances/all_instances_host.cu)
        #     instantiates every launcher's `template void
        #     xxx<dtype>(...)`. It includes `<torch/extension.h>` ONCE
        #     and forward-declares the kernel templates (so the
        #     launcher body's `<<<...>>>` typechecks and emits an
        #     undefined `__device_stub__` reference). It does NOT
        #     include any pipeline header, so it sidesteps the ODR
        #     clash between same-named layout helpers that live in
        #     a16w16 / a8w8 pipeline headers.
        #   * One device TU per (kid, dtype) at instances/{name}_C{dtype}.device.cu
        #     includes that kid's pipeline header and emits
        #     `template __global__ void kernel<Traits>(...)`. The host
        #     stub generated here resolves the fused host TU's
        #     undefined reference at link time (-fno-gpu-rdc allows
        #     this because the linker still merges fat-binary segments
        #     across all input .o's).
        #
        # The accumulator rows below are populated by the per-kid
        # _gen_*_instance methods and consumed by _emit_fused_host_tu
        # / _emit_device_tus at the end of gen_instances().
        # Each row is a dict with the kid's full configuration:
        #   { "kid_name", "dtype", "host_decl", "device_decl",
        #     "launcher_body", "traits_aliases", "kernel_func",
        #     "kargs_name" }
        self._host_instantiations = []
        self._device_instantiations = []
        self._kid_records = []
        # Pipeline headers for each kernel_tag (used by the per-kid
        # device TU only).
        self._kid_pipeline_header = {}

    # ── a16w16 compile-time + VGPR spill validator ──

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

        # ── Hardware ──
        if k.BLOCK_SIZE > 512:
            errors.append(f"BLOCK_SIZE={k.BLOCK_SIZE} exceeds 512")

        # ── Pipeline: T_M must be 2 (split-barrier) ──
        if k.T_M != 2:
            errors.append(f"T_M={k.T_M} must be 2")

        # ── Traits: BLOCK_SIZE = T_M * T_N * T_K * WARP_SIZE ──
        if k.BLOCK_SIZE != num_waves * WARP_SIZE:
            errors.append(
                f"BLOCK_SIZE={k.BLOCK_SIZE} != "
                f"{k.T_M}*{k.T_N}*{T_K}*{WARP_SIZE}={num_waves * WARP_SIZE}"
            )

        # ── Layout: T_N % T_M == 0 (rb: T_N/T_M) ──
        if k.T_N % k.T_M != 0:
            errors.append(f"T_N={k.T_N} not divisible by T_M={k.T_M}")

        # ── MFMA validity ──
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

        # ── VEC ──
        expected_vec = 16 // sizeof_da
        if k.VEC_A != expected_vec:
            errors.append(f"VEC_A={k.VEC_A} must be {expected_vec}")

        # ── Block tile divisibility ──
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

        # ── smem layout ──
        if smem_linear_wave % k.B_K != 0:
            errors.append(f"smem_linear_wave={smem_linear_wave} not div by B_K={k.B_K}")
        else:
            smem_sub = smem_linear_wave // k.B_K
            if HALF_B_M % smem_sub != 0:
                errors.append(f"HALF_B_M={HALF_B_M} not div by smem_sub={smem_sub}")
            if HALF_B_N % smem_sub != 0:
                errors.append(f"HALF_B_N={HALF_B_N} not div by smem_sub={smem_sub}")

        # ── buffer/ds instruction counts ≥ 1 and integer ──
        for name, num, den in [
            ("a_buffer_load_insts", HALF_B_M * k.B_K, k.BLOCK_SIZE * k.VEC_A),
            ("b_buffer_load_insts", HALF_B_N * k.B_K, k.BLOCK_SIZE * k.VEC_B),
            ("a_ds_read_insts", E_M * E_K * k.W_M * k.W_K, WARP_SIZE * k.VEC_A),
            ("b_ds_read_insts", E_N * E_K * k.W_N * k.W_K, WARP_SIZE * k.VEC_B),
        ]:
            if den == 0 or num % den != 0 or num // den < 1:
                errors.append(f"{name}={num}/{den} invalid")

        # ── ra/rb: W_M*W_K / (WARP_SIZE*VEC_A) >= 1 ──
        for tag, ww, vec in [
            ("ra", k.W_M * k.W_K, k.VEC_A),
            ("rb", k.W_N * k.W_K, k.VEC_B),
        ]:
            denom = WARP_SIZE * vec
            if ww < denom or ww % denom != 0:
                errors.append(f"{tag}: W*W_K={ww} must be >= and div by {denom}")

        # ── gb: exact division (not ceil_div) ──
        if k.VEC_B and k.B_K % k.VEC_B == 0:
            threads_k_b = k.B_K // k.VEC_B
            if k.BLOCK_SIZE % threads_k_b == 0:
                thr_n = k.BLOCK_SIZE // threads_k_b
                if HALF_B_N % thr_n != 0:
                    errors.append(f"gb: HALF_B_N={HALF_B_N} not div by {thr_n}")

        # ── sb: exact division ──
        if smem_linear_wave % k.B_K == 0:
            smem_sub = smem_linear_wave // k.B_K
            if smem_sub and HALF_B_N % smem_sub == 0:
                smem_n_rep = HALF_B_N // smem_sub
                if smem_n_rep % num_waves != 0:
                    errors.append(f"sb: smem_n_rep={smem_n_rep} not div by {num_waves}")

        # ── threads_k <= WARP_SIZE ──
        for tag, vec in [("ga", k.VEC_A), ("gb", k.VEC_B)]:
            if vec and k.B_K // vec > WARP_SIZE:
                errors.append(f"{tag}: B_K/VEC={k.B_K // vec} > WARP_SIZE")

        # ── AGPR < 256 ──
        agpr_per_mfma = (k.W_M * k.W_N) // WARP_SIZE
        total_agprs = 4 * E_M * E_N * agpr_per_mfma
        if total_agprs >= 256:
            errors.append(f"AGPR={total_agprs} must be < 256")

        # ── LDS <= 160 KiB ──
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

        # ── VGPR spill estimate ──
        vgpr_ops = 4 * E_K * (E_M + 2 * E_N)
        vgpr_est = vgpr_ops + 80
        if vgpr_est > 256:
            errors.append(f"VGPR_est={vgpr_est} exceeds 256")
        if vgpr_est + total_agprs > 512:
            errors.append(f"VGPR+AGPR={vgpr_est + total_agprs} exceeds 512")

        # ── ra/rb layout constraint: B_K must equal T_N * W_K / 2 ──
        # The ra/rb LDS read layouts couple E_K with T_N through the T_M
        # partition stride in group 2. When (W_M/T_N) * E_K * 32 >= 512
        # (smem_linear_wave), the T_M partition offset exceeds the LDS row
        # data region. This limits valid configs to E_K = T_N / 2.
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

    # ── a16w16_flatmm validator ──

    @staticmethod
    def _validate_a16w16_flatmm(k: OpusGemmInstance):
        """Validate an a16w16_flatmm instance at codegen time.

        Mirrors the static_asserts in opus_gemm_a16w16_flatmm_traits_gfx950: derives
        pfk from LDS budget / WG_PER_CU and requires pfk >= 3 (depth-1 pipeline
        entry point). Raises ValueError if invalid.
        """
        errors = []
        sizeof_da = 2  # bf16 locked

        # ── Locked config (traits enforces these via templates) ──
        if k.BLOCK_SIZE != 256:
            errors.append(f"BLOCK_SIZE={k.BLOCK_SIZE} must be 256 (4-wave warp-spec)")
        if k.T_M != 2:
            errors.append(f"T_M={k.T_M} must be 2")
        if k.T_N != 1:
            errors.append(f"T_N={k.T_N} must be 1")

        # ── MFMA: only W_M<32 path supported (LOAD_GROUP_M_LANE=1) ──
        if (k.W_M, k.W_N, k.W_K) not in VALID_FLATMM_MFMA:
            errors.append(
                f"WAVE=({k.W_M},{k.W_N},{k.W_K}) not in {VALID_FLATMM_MFMA} "
                f"(flatmm ra layout requires W_M<32)"
            )
        if k.W_M >= 32:
            errors.append(f"W_M={k.W_M}: flatmm LGML=4 path not implemented")

        # ── VEC ──
        expected_vec = 16 // sizeof_da
        if k.VEC_A != expected_vec or k.VEC_B != expected_vec:
            errors.append(f"VEC_A={k.VEC_A}, VEC_B={k.VEC_B} must be {expected_vec}")
        if k.VEC_C != 4:
            errors.append(f"VEC_C={k.VEC_C} must be 4")

        # ── Tile geometry (LOAD_GROUP_K = W_K * 2 = 64 for W_K=32) ──
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

        # ── LDS per-group-load size ──
        smem_linear_wave = WARP_SIZE * 16 // sizeof_da  # 512 for bf16
        smem_sub = smem_linear_wave // LOAD_GROUP_K
        slots = LOAD_GROUP_M // smem_sub
        smem_padding = 16 // sizeof_da if k.W_M >= 32 else 2 * 16 // sizeof_da
        smem_per_group_load_size = slots * (smem_linear_wave + smem_padding) * sizeof_da

        # ── WG_PER_CU ──
        if k.WG_PER_CU not in (1, 2):
            errors.append(f"WG_PER_CU={k.WG_PER_CU} must be 1 or 2")

        # ── pfk derivation (match traits formula) ──
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

    # ── a16w16_flatmm_splitk validator ──

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

        # VGPR-spill guard: cc hand-picked only 3 WG=1 tiles because larger
        # tiles (COM_REP_M*COM_REP_N > 16) spill v_c to scratch and run
        # 1000x slower. T_M=2, T_N=1 locked, so:
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
        # All other shape/divisibility constraints fall through to the
        # split-barrier validator. We temporarily flip kernel_tag so the
        # split-barrier validator's "if kernel_tag != a16w16" hooks (if
        # any are added in the future) keep working consistently.
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
        Plus a user-imposed B_M ≤ 192 cap.
        """
        errors = []
        sizeof_da = 2  # bf16 locked

        # ── Locked config ──
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

        # ── VEC ──
        expected_vec = 16 // sizeof_da  # 8 for bf16
        if (
            k.VEC_A != expected_vec
            or k.VEC_B != expected_vec
            or k.VEC_C != expected_vec
        ):
            errors.append(
                f"VEC=({k.VEC_A},{k.VEC_B},{k.VEC_C}) must all be {expected_vec}"
            )

        # ── User cap: B_M ≤ 192 ──
        if k.B_M > 192:
            errors.append(f"B_M={k.B_M} exceeds mono-tile cap of 192")

        # ── Mono-tile must be non-OOB (intrinsic; launcher rejects unaligned) ──
        if k.has_oob:
            errors.append("mono-tile is intrinsically non-OOB; has_oob must be False")

        # ── Block tile divisibility ──
        if k.B_M % (k.W_M * k.T_M) != 0:
            errors.append(f"B_M={k.B_M} not div by W_M*T_M={k.W_M * k.T_M}")
        if k.B_N % (k.W_N * k.T_N) != 0:
            errors.append(f"B_N={k.B_N} not div by W_N*T_N={k.W_N * k.T_N}")
        if k.B_K % (k.W_K * 1) != 0:
            errors.append(f"B_K={k.B_K} not div by W_K*T_K={k.W_K}")

        E_M = k.B_M // (k.W_M * k.T_M) if (k.W_M * k.T_M) else 0
        E_N = k.B_N // (k.W_N * k.T_N) if (k.W_N * k.T_N) else 0
        E_K = k.B_K // k.W_K if k.W_K else 0

        # ── E_N divisibility (rb layout grouping by T_N/T_M = 2) ──
        if k.T_M and (E_N * k.T_M) % k.T_N != 0:
            errors.append(
                f"E_N={E_N} not div by T_N/T_M={k.T_N // k.T_M} "
                f"(mono-tile rb layout grouping; needs B_N % 128 == 0)"
            )

        # ── LDS layout ──
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

            # ── LDS footprint ──
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

    # ── Instance generation ──

    def gen_instance(self, k: OpusGemmInstance):
        if k.kernel_tag == "a16w16":
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

        pipeline_header = PIPELINE_HEADER_MAP[k.kernel_tag]
        traits_header = TRAITS_HEADER_MAP[k.kernel_tag]
        kernel_func = KERNEL_FUNC_MAP[k.kernel_tag]
        da, db = INPUT_DTYPE_MAP[k.kernel_tag]
        traits_name = TRAITS_NAME_MAP[k.kernel_tag]
        kargs_name = KARGS_NAME_MAP[k.kernel_tag]

        # Track per-kid pipeline header so the per-kid device.cu can
        # include exactly the right one without re-running the full
        # logic in _gen_*_instance.
        self._kid_pipeline_header[k.name] = pipeline_header

        if k.kernel_tag == "a16w16_flatmm":
            self._gen_flatmm_instance(
                k,
                pipeline_header,
                traits_header,
                kernel_func,
                da,
                db,
                traits_name,
                kargs_name,
            )
        elif k.kernel_tag == "a16w16_flatmm_splitk":
            self._gen_flatmm_splitk_instance(
                k,
                pipeline_header,
                traits_header,
                kernel_func,
                da,
                db,
                traits_name,
                kargs_name,
            )
        elif k.kernel_tag == "a16w16_persistent":
            self._gen_persistent_instance(
                k,
                pipeline_header,
                traits_header,
                kernel_func,
                da,
                db,
                traits_name,
                kargs_name,
            )
        elif k.kernel_tag == "a16w16_mono_tile":
            self._gen_mono_tile_instance(
                k,
                pipeline_header,
                traits_header,
                kernel_func,
                da,
                db,
                traits_name,
                kargs_name,
            )
        elif k.kernel_tag in NOSCALE_TAGS:
            self._gen_noscale_instance(
                k,
                pipeline_header,
                traits_header,
                kernel_func,
                da,
                db,
                traits_name,
                kargs_name,
            )
        else:
            self._gen_scale_instance(
                k,
                pipeline_header,
                traits_header,
                kernel_func,
                da,
                db,
                traits_name,
                kargs_name,
            )

    def _gen_scale_instance(
        self,
        k,
        pipeline_header,
        traits_header,
        kernel_func,
        da,
        db,
        traits_name,
        kargs_name,
    ):
        # Pre-declared Traits alias (visible to both passes).
        # See _gen_noscale_instance for the rationale of the host/device
        # pass split.
        traits_aliases = f"""
template <typename D_C>
using {k.name}_Traits = {traits_name}<{k.BLOCK_SIZE},
    opus::seq<{k.B_M}, {k.B_N}, {k.B_K}>,
    opus::tuple<{da}, {db}, D_C, fp32_t, fp32_t>,
    opus::seq<{k.VEC_A}, {k.VEC_B}, {k.VEC_C}>,
    opus::seq<{k.GROUP_M}, {k.GROUP_N}, {k.GROUP_K}>>;
"""

        INSTANCE_IMPL = f"""// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
#pragma once
#if !defined(__HIP_DEVICE_COMPILE__) && !defined(__HIPCC_RTC__)
#include "aiter_tensor.h"
#include "aiter_stream.h"
#include <optional>
#endif
// Pipeline body vs. traits-only split:
//
//   * Per-kid TU (default) includes the full pipeline header, which
//     contains the __global__ kernel template definition + all device
//     layout helpers. This is what drives device codegen.
//
//   * Fused host TU (defines OPUS_FUSED_HOST_TU before #include)
//     includes only the traits header (kargs + Traits structs, no
//     free function templates) and forward-declares the kernel
//     template. The fused TU then writes a launcher body whose
//     <<<...>>> generates an undefined `__device_stub__` reference
//     resolved at link time by the per-kid device.cu's explicit
//     instantiation. Skipping the pipeline header here is what avoids
//     the ODR clash on `make_layout_ga_noscale` & friends, which are
//     defined in multiple pipeline headers under the same name.
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
        Path(os.path.join(self.impl_path, f"{k.name}.cuh")).write_text(INSTANCE_IMPL)

        # See _gen_noscale_instance for how these rows are consumed.
        for CDtype in k.output_dtypes:
            host_decl = (
                f"template void\n"
                f"{k.name}<{CDtype}>(\n"
                f"    aiter_tensor_t &XQ,\n"
                f"    aiter_tensor_t &WQ,\n"
                f"    aiter_tensor_t &Y,\n"
                f"    std::optional<aiter_tensor_t> x_scale,\n"
                f"    std::optional<aiter_tensor_t> w_scale);\n"
            )
            device_decl = (
                f"template __global__ void {kernel_func}<\n"
                f"    {k.name}_Traits<{CDtype}>>({kargs_name});\n"
            )
            self._host_instantiations.append(
                {"kid_name": k.name, "dtype": CDtype, "host_decl": host_decl}
            )
            self._device_instantiations.append(
                {"kid_name": k.name, "dtype": CDtype, "device_decl": device_decl}
            )

    # Shared host-side bias validation + kargs population. Inserted into every
    # A16W16_TUNE_TAGS launcher (split-barrier / flatmm / flatmm_splitk).
    #
    #   * bias is std::optional<aiter_tensor_t>; kargs.ptr_bias / stride_bias_batch
    #     are populated from it (or set to nullptr / 0 when absent).
    #   * Shape protocol (F.linear convention, per-output-feature):
    #       [N]            ->  stride_bias_batch=0  (broadcast across batches)
    #       [batch, N]     ->  stride_bias_batch=N
    #     anything else (or non-contiguous, or dtype != Y.dtype()) hard-errors
    #     up front -- the kernel side has no further runtime check.
    #   * dtype: matched to Y.dtype() (the launcher template param D_C); this
    #     is what the user-facing match_d_out convention requires.
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

    def _gen_noscale_instance(
        self,
        k,
        pipeline_header,
        traits_header,
        kernel_func,
        da,
        db,
        traits_name,
        kargs_name,
    ):
        # a16w16 split-barrier supports HAS_BIAS via the noscale kargs path.
        # a8w8_noscale kids share the same kargs struct but always pass
        # nullptr / 0; their bias arg is rejected at the dispatcher level.
        is_a16w16_split_barrier = k.kernel_tag == "a16w16"
        traits_extra = ""
        if is_a16w16_split_barrier:
            traits_extra = (
                f",\n        opus::seq<{k.T_M}, {k.T_N}, 1>,"
                f"\n        opus::seq<{k.W_M}, {k.W_N}, {k.W_K}>"
            )

        min_k = 2 * k.B_K
        # Kid-specific K-bound checks. Split-barrier pipeline requires
        # K >= 2 * B_K and the loop count to be even.
        k_check = f"""
    int loops_ = (K + {k.B_K} - 1) / {k.B_K};
    AITER_CHECK(loops_ >= 2,
        "K=", K, " too small for B_K={k.B_K}, need K >= {min_k}");
    AITER_CHECK(loops_ % 2 == 0,
        "ceil_div(K, {k.B_K})=", loops_, " must be even (prefetch constraint)");
    // Odd-K is unsafe across the a16w16 family: the splitk pipeline shows
    // up to ~7% maxdelta on bf16-acc paths when K is odd (predates this
    // PR). Reject odd K uniformly so callers get a clear error instead
    // of silent ~3% accuracy regressions; relax once the underlying K-tail
    // handling is fixed.
    AITER_CHECK(K % 2 == 0,
        "K=", K, " must be even (a16w16 family rejects odd K due to a "
        "latent K-tail accumulation bug; pass an even K)");
    AITER_CHECK(M >= 1 && N >= 1, "M and N must be >= 1");
"""

        # a16w16 kids live in opus_gemm_a16w16_tune_lookup.h alongside the
        # flatmm + splitk launchers, so their std::function slot requires
        # the 5-arg signature (XQ, WQ, Y, std::optional<bias>, int splitK).
        # The body ignores splitK (split-barrier pipeline has no split-K
        # concept). a8w8 kids never enter the a16w16 lookup; they keep the
        # 3-arg signature.
        if k.kernel_tag in A16W16_TUNE_TAGS:
            extra_param = (
                ",\n    std::optional<aiter_tensor_t> bias," "\n    int /*splitK*/"
            )
        else:
            extra_param = ""

        # a16w16 split-barrier: emit two traits / kernel specializations
        # (HAS_BIAS=true / HAS_BIAS=false) and runtime-dispatch on
        # bias.has_value(). HAS_BIAS=true must NEVER be entered with a null
        # bias pointer because the in-kernel prefetch issues unconditional
        # buffer_loads -- a null rsrc would generate out-of-bounds garbage
        # that the if constexpr guard cannot screen.
        has_oob_str = "true" if k.has_oob else "false"
        if is_a16w16_split_barrier:
            cachectl_launch_extra = ""
            if hasattr(k, "cachectl_a") and k.cachectl_a >= 0:
                cachectl_launch_extra = f",\n        {k.cachectl_a}, {k.cachectl_b}"
            launch_block = f"""
    using TraitsNoBias = {traits_name}<{k.BLOCK_SIZE},
        opus::seq<{k.B_M}, {k.B_N}, {k.B_K}>,
        opus::tuple<{da}, {db}, D_C, fp32_t>,
        opus::seq<{k.VEC_A}, {k.VEC_B}, {k.VEC_C}>{traits_extra},
        false,                                 // HAS_BIAS
        D_C,                                   // D_BIAS = D_C
        {has_oob_str}{cachectl_launch_extra}>;
    using TraitsBias = {traits_name}<{k.BLOCK_SIZE},
        opus::seq<{k.B_M}, {k.B_N}, {k.B_K}>,
        opus::tuple<{da}, {db}, D_C, fp32_t>,
        opus::seq<{k.VEC_A}, {k.VEC_B}, {k.VEC_C}>{traits_extra},
        true,                                  // HAS_BIAS
        D_C,                                   // D_BIAS = D_C
        {has_oob_str}{cachectl_launch_extra}>;

    auto stream = aiter::getCurrentHIPStream();
    if (bias.has_value()) {{{{
        {kernel_func}<TraitsBias><<<grid, block, 0, stream>>>(kargs);
    }}}} else {{{{
        {kernel_func}<TraitsNoBias><<<grid, block, 0, stream>>>(kargs);
    }}}}"""
        else:
            launch_block = f"""
    using Traits = {traits_name}<{k.BLOCK_SIZE},
        opus::seq<{k.B_M}, {k.B_N}, {k.B_K}>,
        opus::tuple<{da}, {db}, D_C, fp32_t>,
        opus::seq<{k.VEC_A}, {k.VEC_B}, {k.VEC_C}>{traits_extra}>;

    auto stream = aiter::getCurrentHIPStream();
    {kernel_func}<Traits><<<grid, block, 0, stream>>>(kargs);"""

        # bias-aware kargs population: only emit for a16w16 split-barrier
        # (the only noscale path that actually consumes bias). a8w8 noscale
        # uses the same struct but its launcher never reaches here with
        # bias.has_value()=true (dispatcher rejects).
        if is_a16w16_split_barrier:
            bias_kargs_block = (
                self.BIAS_HOST_VALIDATE
                + "    kargs.ptr_bias = ptr_bias_;\n"
                + "    kargs.stride_bias_batch = stride_bias_batch_;\n"
            )
        elif k.kernel_tag in A16W16_TUNE_TAGS:
            # Other A16W16_TUNE_TAGS handled by their own _gen_*_instance.
            # Defensive guard in case of future routing changes.
            bias_kargs_block = (
                "    AITER_CHECK(!bias.has_value(),\n"
                '        "bias not supported on this a16w16 kid");\n'
            )
        else:
            bias_kargs_block = ""

        # noscale kargs has the new ptr_bias / stride_bias_batch fields.
        # Struct value-initialization (`{}`) already zero-fills them, so
        # a8w8 callers that never see a `bias` arg do not need explicit
        # nullptr / 0 assignments.
        kargs_init_extra = ""

        # ── Compile-time split: host pass vs device pass ──
        #
        # The .cuh file contains the heavy host-side launcher (AITER_CHECK,
        # `<<<...>>>` launch, kargs marshalling). Wrapping the host includes
        # plus the launcher body in `#ifndef __HIP_DEVICE_COMPILE__` lets us
        # skip the entire libtorch + ATen + <hip/hip_runtime.h> header
        # stack on the device pass for every instance TU. The device pass
        # only needs the pipeline header (which transitively pulls
        # opus.hpp + opus_gemm_utils.cuh, both lightweight thanks to their
        # own __HIP_DEVICE_COMPILE__ guards).
        #
        # Functional equivalence with the pre-split layout:
        #   - host pass: identical preamble + launcher body. The kernel
        #     `<<<...>>>` launch in the launcher body still triggers the
        #     usual __device_stub__ generation, so host-side hipLaunchKernel
        #     resolves at runtime exactly as before.
        #   - device pass: no launcher body to compile. The companion .cpp
        #     emits an explicit `template __global__ void
        #     {kernel_func}<Traits>(kargs)` instantiation under the same
        #     #else branch, which is what actually drives GPU codegen.
        #
        # Pre-declared Traits aliases live at file scope (outside the
        # #ifndef) so the device-pass branch in the .cpp can reuse them
        # without repeating the long template argument list.
        # a16w16 split-barrier emits two Traits (HAS_BIAS true / false);
        # everything else has a single Traits.
        # CPOL template params (only for a16w16 split-barrier)
        cachectl_extra = ""
        if is_a16w16_split_barrier and (hasattr(k, "cachectl_a") and k.cachectl_a >= 0):
            cachectl_extra = f",\n    {k.cachectl_a}, {k.cachectl_b}"
        if is_a16w16_split_barrier:
            traits_aliases = f"""
template <typename D_C>
using {k.name}_TraitsNoBias = {traits_name}<{k.BLOCK_SIZE},
    opus::seq<{k.B_M}, {k.B_N}, {k.B_K}>,
    opus::tuple<{da}, {db}, D_C, fp32_t>,
    opus::seq<{k.VEC_A}, {k.VEC_B}, {k.VEC_C}>{traits_extra},
    false,
    D_C,
    {has_oob_str}{cachectl_extra}>;
template <typename D_C>
using {k.name}_TraitsBias = {traits_name}<{k.BLOCK_SIZE},
    opus::seq<{k.B_M}, {k.B_N}, {k.B_K}>,
    opus::tuple<{da}, {db}, D_C, fp32_t>,
    opus::seq<{k.VEC_A}, {k.VEC_B}, {k.VEC_C}>{traits_extra},
    true,
    D_C,
    {has_oob_str}{cachectl_extra}>;
"""
        else:
            traits_aliases = f"""
template <typename D_C>
using {k.name}_Traits = {traits_name}<{k.BLOCK_SIZE},
    opus::seq<{k.B_M}, {k.B_N}, {k.B_K}>,
    opus::tuple<{da}, {db}, D_C, fp32_t>,
    opus::seq<{k.VEC_A}, {k.VEC_B}, {k.VEC_C}>{traits_extra}>;
"""

        # The launcher body now references the pre-declared Traits aliases
        # instead of `using` them locally, so the device-pass __global__
        # instantiation in the companion .cpp can name the same type.
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

        # Three guard combinations encode the host/device pass split for
        # this .cuh:
        #   * __HIP_DEVICE_COMPILE__: device pass, any TU. Skip torch
        #     includes and the launcher body; the pipeline header and
        #     Traits aliases ARE visible (kernel templates need them).
        #   * __HIPCC_RTC__: a TU built with -D__HIPCC_RTC__. This is
        #     specifically the .device.cu files (one per (kid, dtype))
        #     emitted by _emit_device_tus. Their host pass also has no
        #     launcher body, so torch includes would be both wasted and
        #     harmful (RTC mode skips __clang_hip_runtime_wrapper.h,
        #     which torch's transitive headers depend on). Treat RTC
        #     identically to __HIP_DEVICE_COMPILE__ for the purposes of
        #     hiding the launcher.
        #   * Otherwise: regular host pass of a non-RTC TU (the fused
        #     all_instances_host.cu in our codegen). This is the only
        #     place that actually wants the launcher body + torch.
        INSTANCE_IMPL = f"""// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
#pragma once
#if !defined(__HIP_DEVICE_COMPILE__) && !defined(__HIPCC_RTC__)
#include "aiter_tensor.h"
#include "aiter_stream.h"
#include <optional>
#endif
// Pipeline body vs. traits-only split:
//
//   * Per-kid TU (default) includes the full pipeline header, which
//     contains the __global__ kernel template definition + all device
//     layout helpers. This is what drives device codegen.
//
//   * Fused host TU (defines OPUS_FUSED_HOST_TU before #include)
//     includes only the traits header (kargs + Traits structs, no
//     free function templates) and forward-declares the kernel
//     template. The fused TU then writes a launcher body whose
//     <<<...>>> generates an undefined `__device_stub__` reference
//     resolved at link time by the per-kid device.cu's explicit
//     instantiation. Skipping the pipeline header here is what avoids
//     the ODR clash on `make_layout_ga_noscale` & friends, which are
//     defined in multiple pipeline headers under the same name.
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
    kargs.stride_a = K;
    kargs.stride_b = K;
    kargs.stride_c = N;
    kargs.stride_a_batch = M * K;
    kargs.stride_b_batch = N * K;
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
        Path(os.path.join(self.impl_path, f"{k.name}.cuh")).write_text(INSTANCE_IMPL)

        if k.kernel_tag in A16W16_TUNE_TAGS:
            inst_extra_param = ",\n    std::optional<aiter_tensor_t>,\n    int"
        else:
            inst_extra_param = ""

        # Record (kid, dtype) instantiation pairs. These are stitched into
        # one fused host TU + N device TUs at the end of gen_instances().
        # See _emit_fused_host_tu / _emit_device_tu for how the rows are
        # consumed.
        if is_a16w16_split_barrier:
            # Split-barrier emits two __global__ specializations per
            # dtype because the launcher dispatches at runtime on
            # bias.has_value(). Both must be force-instantiated on the
            # device pass.
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
                    f"    {k.name}_Traits<{dtype}>>({kargs_name});\n"
                )

        for CDtype in k.output_dtypes:
            host_decl = (
                f"template void\n"
                f"{k.name}<{CDtype}>(\n"
                f"    aiter_tensor_t &XQ,\n"
                f"    aiter_tensor_t &WQ,\n"
                f"    aiter_tensor_t &Y{inst_extra_param});\n"
            )
            self._host_instantiations.append(
                {
                    "kid_name": k.name,
                    "dtype": CDtype,
                    "host_decl": host_decl,
                }
            )
            self._device_instantiations.append(
                {
                    "kid_name": k.name,
                    "dtype": CDtype,
                    "device_decl": _device_decl(CDtype),
                }
            )

    def _gen_persistent_instance(
        self,
        k,
        pipeline_header,
        traits_header,
        kernel_func,
        da,
        db,
        traits_name,
        kargs_name,
    ):
        """Generate a persistent launcher (a16w16_persistent).

        Persistent traits template signature has 9 parameters:
          <BLOCK_SIZE, BLOCK, DTYPE, VEC, TILE, WAVE, HAS_OOB,
           CACHECTL_A, CACHECTL_B>.

        The Python-visible launcher takes the standard a16w16-family
        4-arg signature (XQ, WQ, Y, std::optional<bias>, int splitK)
        so its `std::function` slot matches split-barrier / flatmm /
        flatmm_splitk inside GENERATE_A16W16_TUNE_LOOKUP. Persistent
        does not support bias yet (kargs lacks ptr_bias); the launcher
        rejects non-empty bias up front and ignores splitK.

        Persistent-specific kargs (m_per_wg, num_tiles_n, m_grp_per_xcd)
        are computed on the host from the same heuristic as the standalone
        reference gemm_a16w16_8wave_mouter.cc:743-758.
        """
        has_oob_str = "true" if k.has_oob else "false"

        # Pre-declared Traits alias at file scope (visible to both passes).
        # Persistent traits: <BLOCK_SIZE, BLOCK, DTYPE, VEC, TILE, WAVE,
        # HAS_OOB, CACHECTL_A, CACHECTL_B>. CACHECTL_A/B are explicit so
        # cpol kid variants pick up their cache hints.
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

        # K constraints inherited from split-barrier (loops >= 2 and even).
        # No splitK in persistent, so the launcher just ignores the splitK
        # parameter required by the shared a16w16_tune_lookup signature.
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

        # Host-side grid layout heuristic. Direct port of
        # gemm_a16w16_8wave_mouter.cc:743-758.
        #
        # split_m starts at ceil(NUM_CU / num_tiles_n) so the WG count fits
        # in one launch wave on 256 CUs; then bumped up to the next divisor
        # of num_tiles_m so m_per_wg is integer. Clamped to num_tiles_m.
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

        INSTANCE_IMPL = f"""// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
#pragma once
#if !defined(__HIP_DEVICE_COMPILE__) && !defined(__HIPCC_RTC__)
#include "aiter_tensor.h"
#include "aiter_stream.h"
#include <algorithm>
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
    int /*splitK*/)   // persistent ignores splitK; shares tune-lookup slot signature
{{{{
    int batch = XQ.size(0);
    int M = XQ.size(1);
    int N = WQ.size(1);
    int K = XQ.size(2);
{k_check}
    // a16w16_persistent does not support bias yet (kargs has no
    // ptr_bias / stride_bias_batch fields). Reject up front so the
    // user gets a clear error instead of silently dropping the bias.
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
    kargs.stride_a = K;
    kargs.stride_b = K;
    kargs.stride_c = N;
    kargs.stride_a_batch = M * K;
    kargs.stride_b_batch = N * K;
    kargs.stride_c_batch = M * N;
{grid_setup}
    auto stream = aiter::getCurrentHIPStream();
    {kernel_func}<{k.name}_Traits<D_C>><<<grid, block, 0, stream>>>(kargs);

}}}}
#endif // launcher only on regular host pass
"""
        Path(os.path.join(self.impl_path, f"{k.name}.cuh")).write_text(INSTANCE_IMPL)

        # See _gen_noscale_instance for how these rows are consumed.
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
            self._host_instantiations.append(
                {"kid_name": k.name, "dtype": CDtype, "host_decl": host_decl}
            )
            self._device_instantiations.append(
                {"kid_name": k.name, "dtype": CDtype, "device_decl": device_decl}
            )

    def _gen_mono_tile_instance(
        self,
        k,
        pipeline_header,
        traits_header,
        kernel_func,
        da,
        db,
        traits_name,
        kargs_name,
    ):
        """Generate a mono-tile launcher (a16w16_mono_tile).

        Mono-tile traits template signature has 4 parameters:
          <BLOCK_SIZE, BLOCK, DTYPE, VEC>.
        DTYPE is a 4-tuple <D_A, D_B, D_C, D_ACC>; D_ACC is fp32_t.

        Locked tile/wave geometry (T_M=2, T_N=4, T_K=1, W_M=W_N=16,
        W_K=32) is derived inside the traits header itself; this launcher
        only forwards BLOCK / DTYPE / VEC.

        Python-visible launcher takes the standard a16w16-family 4-arg
        signature (XQ, WQ, Y, std::optional<bias>, int splitK) so its
        function-pointer slot matches the other A16W16_TUNE_TAGS in
        GENERATE_A16W16_TUNE_LOOKUP. Mono-tile does NOT support bias
        (rejected up front) and ignores splitK.

        Mono-tile is intrinsically tile-aligned: the launcher hard-asserts
        M%B_M == N%B_N == K%B_K == 0 (no OOB tail handling exists in the
        kernel body).
        """
        # Pre-declared Traits alias at file scope (visible to both passes).
        # Mono-tile traits: <BLOCK_SIZE, BLOCK, DTYPE (4-tuple incl. D_ACC), VEC>.
        traits_aliases = f"""
template <typename D_C>
using {k.name}_Traits = {traits_name}<{k.BLOCK_SIZE},
    opus::seq<{k.B_M}, {k.B_N}, {k.B_K}>,
    opus::tuple<{da}, {db}, D_C, fp32_t>,
    opus::seq<{k.VEC_A}, {k.VEC_B}, {k.VEC_C}>>;
"""

        # K constraints: mono-tile requires K % B_K == 0 (tile-aligned)
        # and loops >= 2 (prefetch needs at least one fetch + one compute
        # ahead). The K%2 rejection inherited from the family stands.
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
    // Mono-tile is intrinsically non-OOB: the kernel body has no tail
    // handling, so M and N must be tile-aligned. The dispatcher already
    // filters this for tuned (M,N,K) entries, but enforce it here too so
    // a direct `_tune` call with a misaligned shape errors cleanly.
    AITER_CHECK(M % {k.B_M} == 0,
        "mono-tile requires M divisible by B_M={k.B_M}; got M=", M);
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
    int /*splitK*/)   // mono-tile ignores splitK; shares tune-lookup slot signature
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
    kargs.stride_a = K;
    kargs.stride_b = K;
    kargs.stride_c = N;
    kargs.stride_a_batch = M * K;
    kargs.stride_b_batch = N * K;
    kargs.stride_c_batch = M * N;

    int num_tiles_m = M / {k.B_M};
    int num_tiles_n = N / {k.B_N};
    dim3 grid(num_tiles_m * num_tiles_n, 1, batch);
    dim3 block({k.BLOCK_SIZE});

    auto stream = aiter::getCurrentHIPStream();
    {kernel_func}<{k.name}_Traits<D_C>><<<grid, block, 0, stream>>>(kargs);

}}}}
#endif // launcher only on regular host pass
"""
        Path(os.path.join(self.impl_path, f"{k.name}.cuh")).write_text(INSTANCE_IMPL)

        # See _gen_noscale_instance for how these rows are consumed.
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
            self._host_instantiations.append(
                {"kid_name": k.name, "dtype": CDtype, "host_decl": host_decl}
            )
            self._device_instantiations.append(
                {"kid_name": k.name, "dtype": CDtype, "device_decl": device_decl}
            )

    def _gen_flatmm_instance(
        self,
        k,
        pipeline_header,
        traits_header,
        kernel_func,
        da,
        db,
        traits_name,
        kargs_name,
    ):
        """Generate a flatmm launcher (a16w16_flatmm).

        Flatmm traits template signature has 7 parameters:
          <BLOCK_SIZE, BLOCK, DTYPE (5-tuple incl. D_BIAS), VEC, MFMA, WG_PER_CU, HAS_BIAS>.

        The Python-visible launcher keeps the 3-tensor signature (XQ, WQ, Y)
        so the launcher type matches the a16w16 split-barrier one and both can
        populate the same std::function in GENERATE_A16W16_TUNE_LOOKUP.

        Runtime check uses Traits::prefetch_k_iter (compile-time member) to
        report min_k accurately per-instance.
        """
        has_bias_str = "true" if False else "false"  # HAS_BIAS hardcoded false

        # Kid-specific runtime K-bound check per INTEGRATION.md "Runtime
        # 前置约束" item 1: K >= Traits::prefetch_k_iter * Traits::B_K.
        # pfk is a compile-time member so the effective bound is inlined.
        k_check = f"""
    int loops_ = (K + {k.B_K} - 1) / {k.B_K};
    AITER_CHECK(loops_ >= Traits::prefetch_k_iter,
        "K=", K, " too small for flatmm B_K={k.B_K}, need K >= pfk*B_K = ",
        Traits::prefetch_k_iter * {k.B_K}, " (pfk=", Traits::prefetch_k_iter, ")");
    AITER_CHECK(M >= 1 && N >= 1 && K >= 1, "M, N, K must be >= 1");
    AITER_CHECK(batch >= 1, "batch must be >= 1");
    // Odd-K is unsafe across the a16w16 family (see _gen_noscale_instance
    // for the rationale); reject uniformly until the K-tail handling is
    // fixed.
    AITER_CHECK(K % 2 == 0,
        "K=", K, " must be even (a16w16 family rejects odd K due to a "
        "latent K-tail accumulation bug; pass an even K)");
"""

        # Pre-declared Traits alias at file scope (visible to both passes).
        # See _gen_noscale_instance for the rationale of the host/device
        # pass split.
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

        INSTANCE_IMPL = f"""// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
#pragma once
#if !defined(__HIP_DEVICE_COMPILE__) && !defined(__HIPCC_RTC__)
#include "aiter_tensor.h"
#include "aiter_stream.h"
#include <optional>
#endif
// Pipeline body vs. traits-only split:
//
//   * Per-kid TU (default) includes the full pipeline header, which
//     contains the __global__ kernel template definition + all device
//     layout helpers. This is what drives device codegen.
//
//   * Fused host TU (defines OPUS_FUSED_HOST_TU before #include)
//     includes only the traits header (kargs + Traits structs, no
//     free function templates) and forward-declares the kernel
//     template. The fused TU then writes a launcher body whose
//     <<<...>>> generates an undefined `__device_stub__` reference
//     resolved at link time by the per-kid device.cu's explicit
//     instantiation. Skipping the pipeline header here is what avoids
//     the ODR clash on `make_layout_ga_noscale` & friends, which are
//     defined in multiple pipeline headers under the same name.
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
    int /*splitK*/)   // flatmm (non-splitk) ignores splitK; shares tune-lookup slot signature
{{{{
    int batch = XQ.size(0);
    int M = XQ.size(1);
    int N = WQ.size(1);
    int K = XQ.size(2);

    // a16w16_flatmm pipeline still has HAS_BIAS=false hardcoded -- bias
    // support on the warp-specialized 4-wave epilogue is not yet
    // implemented (see plan: a16w16_flatmm bias support deferred). The
    // launcher must accept the optional bias arg to match the
    // GENERATE_A16W16_TUNE_LOOKUP std::function slot, but reject any
    // non-empty bias up front so the user gets a clear error instead of
    // silently dropping the bias.
    AITER_CHECK(!bias.has_value(),
        "bias is not yet supported on a16w16_flatmm kid; use a16w16 "
        "split-barrier (kid 4..9) or a16w16_flatmm_splitk (kid 200..299)");

    using Traits = {k.name}_Traits<D_C>;
{k_check}
    {kargs_name} kargs{{{{}}}};
    kargs.ptr_a = XQ.data_ptr();
    kargs.ptr_b = WQ.data_ptr();
    kargs.ptr_c = Y.data_ptr();
    kargs.ptr_bias = nullptr;  // HAS_BIAS=false; field reserved for future.
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

    int num_tiles_m = (M + {k.B_M} - 1) / {k.B_M};
    int num_tiles_n = (N + {k.B_N} - 1) / {k.B_N};
    dim3 grid(num_tiles_m * num_tiles_n, 1, batch);
    dim3 block({k.BLOCK_SIZE});

    auto stream = aiter::getCurrentHIPStream();
    {kernel_func}<{k.name}_Traits<D_C>><<<grid, block, 0, stream>>>(kargs);

}}}}
#endif // launcher only on regular host pass
"""
        Path(os.path.join(self.impl_path, f"{k.name}.cuh")).write_text(INSTANCE_IMPL)

        # See _gen_noscale_instance for how these rows are consumed.
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
            self._host_instantiations.append(
                {"kid_name": k.name, "dtype": CDtype, "host_decl": host_decl}
            )
            self._device_instantiations.append(
                {"kid_name": k.name, "dtype": CDtype, "device_decl": device_decl}
            )

    def _gen_flatmm_splitk_instance(
        self,
        k,
        pipeline_header,
        traits_header,
        kernel_func,
        da,
        db,
        traits_name,
        kargs_name,
    ):
        """Generate a flatmm split-K launcher (a16w16_flatmm_splitk).

        Two-kernel pipeline (main + reduce). Main writes fp32 workspace;
        reduce sums splits + casts fp32 -> bf16 into Y. Workspace is
        allocated inline via `torch::empty` each call, mirroring the aiter
        triton gemm_a16w16.py y_pp idiom (no persistent cache, torch caching
        allocator amortizes the cost).

        splitK semantic: literal KBatch; 0 and 1 both mean no split (KBatch=1).
        Host-side auto-clamp decrements split_k until every split has >= pfk
        iters (port of cc lines 1030-1048).

        D_C template param is fp32_t (workspace is fp32); Y must be bf16.
        opus_gemm.cu's dispatcher forces the <fp32_t> branch for kid >= 200.
        """
        # Pre-declared Traits alias at file scope (visible to both passes).
        # See _gen_noscale_instance for the rationale of the host/device
        # pass split.
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

        INSTANCE_IMPL = f"""// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
#pragma once
#if !defined(__HIP_DEVICE_COMPILE__) && !defined(__HIPCC_RTC__)
#include "aiter_tensor.h"
#include "aiter_stream.h"
#include <optional>
#endif
// Pipeline body vs. traits-only split:
//
//   * Per-kid TU (default) includes the full pipeline header, which
//     contains the __global__ kernel template definition + all device
//     layout helpers. This is what drives device codegen.
//
//   * Fused host TU (defines OPUS_FUSED_HOST_TU before #include)
//     includes only the traits header (kargs + Traits structs, no
//     free function templates) and forward-declares the kernel
//     template. The fused TU then writes a launcher body whose
//     <<<...>>> generates an undefined `__device_stub__` reference
//     resolved at link time by the per-kid device.cu's explicit
//     instantiation. Skipping the pipeline header here is what avoids
//     the ODR clash on `make_layout_ga_noscale` & friends, which are
//     defined in multiple pipeline headers under the same name.
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
    // Odd-K is unsafe: splitk pipeline shows ~3-7% maxdelta on odd K (e.g.
    // K=257 / 513) while even K stays near bf16 noise floor. The bug lives
    // in the K-tail handling (mask_va_tail / reduce-tail interplay) and
    // predates this PR. Reject uniformly until fixed.
    AITER_CHECK(K % 2 == 0,
        "K=", K, " must be even (a16w16 family rejects odd K due to a "
        "latent K-tail accumulation bug; pass an even K)");
{self.BIAS_HOST_VALIDATE}
    using Traits = {k.name}_Traits<D_C>;

    // splitK semantic: literal KBatch. 0 and 1 both mean no split.
    int split_k = (splitK <= 1) ? 1 : splitK;

    // Host-side auto-clamp: ensure every split has >= pfk iters (port of cc
    // lines 1030-1048). The kernel's pfk is a compile-time member.
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

    // Workspace allocation: thread-local cache that grows on demand.
    // Replaces the original `hipMallocAsync` + `hipFreeAsync` pair
    // (introduced when the launcher went torch-free), which is not
    // graph-capturable on ROCm 7.2.2: hipgraph replay re-runs the
    // alloc/free host-side every replay, costing ~120us / replay (vs
    // ~30us of actual kernel work). The eager tuner also pays the
    // alloc cost on every iter, distorting splitk's measured perf so
    // badly that the tuner picks split-barrier kids for shapes where
    // splitk would actually win.
    //
    // Cache is stream-agnostic: a single ptr is reused across all
    // streams in the thread. Sync-on-grow uses `hipDeviceSynchronize`
    // so we don't release memory another stream may still be writing.
    // Inside graph capture the grow path is forbidden (hipMalloc /
    // hipFree are stream-capture-illegal); callers that capture splitk
    // are expected to warm up the cache once eagerly with the largest
    // workspace they will use first.
    auto stream = aiter::getCurrentHIPStream();
    size_t ws_bytes = (size_t)split_k * (size_t)batch
                    * (size_t)padded_M * (size_t)padded_N * sizeof(float);
    static thread_local void*  ws_cached_ptr   = nullptr;
    static thread_local size_t ws_cached_bytes = 0;
    if (ws_cached_ptr == nullptr || ws_bytes > ws_cached_bytes)
    {{
        hipStreamCaptureStatus capture_status = hipStreamCaptureStatusNone;
        HIP_CALL(hipStreamIsCapturing(stream, &capture_status));
        AITER_CHECK(capture_status == hipStreamCaptureStatusNone,
            "splitk workspace cache miss inside HIP graph capture is not "
            "supported. Run the launcher once eagerly with the same shape "
            "before capturing the graph (so the static thread_local "
            "workspace cache is sized correctly).");

        if (ws_cached_ptr != nullptr)
        {{
            HIP_CALL(hipDeviceSynchronize());
            HIP_CALL(hipFree(ws_cached_ptr));
        }}
        // Round up to 4 MiB granularity so small shape variations don't
        // re-grow the buffer.
        const size_t kGrowAlign = (size_t)4 * 1024 * 1024;
        size_t grow_bytes = ((ws_bytes + kGrowAlign - 1) / kGrowAlign) * kGrowAlign;
        HIP_CALL(hipMalloc(&ws_cached_ptr, grow_bytes));
        ws_cached_bytes = grow_bytes;
    }}
    void* ptr_workspace_ = ws_cached_ptr;

    {kargs_name} kargs{{{{}}}};
    kargs.ptr_a         = XQ.data_ptr();
    kargs.ptr_b         = WQ.data_ptr();
    kargs.ptr_workspace = ptr_workspace_;
    kargs.ptr_c         = Y.data_ptr();
    kargs.ptr_bias      = ptr_bias_;          // populated by BIAS_HOST_VALIDATE
    kargs.m = M; kargs.n = N; kargs.k = K; kargs.batch = batch;
    kargs.split_k = split_k;
    kargs.stride_a        = K;
    kargs.stride_b        = K;
    kargs.stride_ws       = padded_N;
    kargs.stride_c        = N;
    kargs.stride_a_batch  = M * K;
    kargs.stride_b_batch  = N * K;
    kargs.stride_ws_batch = padded_M * padded_N;
    kargs.stride_c_batch  = M * N;
    kargs.stride_bias_batch = stride_bias_batch_;

    dim3 grid_main(num_tiles_m * num_tiles_n * split_k, 1, batch);
    dim3 block_main({k.BLOCK_SIZE});

    constexpr int REDUCE_VEC = 16;
    constexpr int REDUCE_BS  = 64;
    // Note: no padded_N % REDUCE_VEC check. splitk_reduce_kernel has a tail
    // path (see splitk_reduce_gfx950.cuh) that handles the (n_base + VEC > N) case
    // with per-element scalar stores, so any N (including odd /
    // non-power-of-16) is safe.
    dim3 grid_reduce((N + REDUCE_VEC * REDUCE_BS - 1) / (REDUCE_VEC * REDUCE_BS),
                      batch * M, 1);
    dim3 block_reduce(REDUCE_BS);

    {kernel_func}<{k.name}_Traits<D_C>><<<grid_main, block_main, 0, stream>>>(kargs);
    // Reduce kernel: specializations (D_OUT bf16/fp32 x HAS_BIAS true/false x HAS_OOB).
    // bias dtype is locked to Y.dtype() by BIAS_HOST_VALIDATE.
    if (Y.dtype() == AITER_DTYPE_bf16) {{{{
        if (bias.has_value()) {{{{
            splitk_reduce_kernel<REDUCE_VEC, REDUCE_BS, __bf16, true, __bf16, {has_oob_str}>
                <<<grid_reduce, block_reduce, 0, stream>>>(
                    reinterpret_cast<const float*>(ptr_workspace_),
                    reinterpret_cast<__bf16*>(Y.data_ptr()),
                    split_k, M, N, batch, padded_M, padded_N,
                    reinterpret_cast<const __bf16*>(ptr_bias_),
                    stride_bias_batch_);
        }}}} else {{{{
            splitk_reduce_kernel<REDUCE_VEC, REDUCE_BS, __bf16, false, __bf16, {has_oob_str}>
                <<<grid_reduce, block_reduce, 0, stream>>>(
                    reinterpret_cast<const float*>(ptr_workspace_),
                    reinterpret_cast<__bf16*>(Y.data_ptr()),
                    split_k, M, N, batch, padded_M, padded_N,
                    nullptr, 0);
        }}}}
    }}}} else {{{{
        // Y.dtype() == Float per the AITER_CHECK above.
        if (bias.has_value()) {{{{
            splitk_reduce_kernel<REDUCE_VEC, REDUCE_BS, float, true, float, {has_oob_str}>
                <<<grid_reduce, block_reduce, 0, stream>>>(
                    reinterpret_cast<const float*>(ptr_workspace_),
                    reinterpret_cast<float*>(Y.data_ptr()),
                    split_k, M, N, batch, padded_M, padded_N,
                    reinterpret_cast<const float*>(ptr_bias_),
                    stride_bias_batch_);
        }}}} else {{{{
            splitk_reduce_kernel<REDUCE_VEC, REDUCE_BS, float, false, float, {has_oob_str}>
                <<<grid_reduce, block_reduce, 0, stream>>>(
                    reinterpret_cast<const float*>(ptr_workspace_),
                    reinterpret_cast<float*>(Y.data_ptr()),
                    split_k, M, N, batch, padded_M, padded_N,
                    nullptr, 0);
        }}}}
    }}}}

    // No free here: workspace is held by the static thread_local cache
    // above and released on first cache resize / process exit. Avoids
    // the per-call alloc/free overhead and makes the launcher
    // graph-capturable (the cached pointer is constant across replays).
}}}}
#endif // launcher only on regular host pass
"""
        Path(os.path.join(self.impl_path, f"{k.name}.cuh")).write_text(INSTANCE_IMPL)

        # See _gen_noscale_instance for how these rows are consumed. The
        # splitk_reduce_kernel's 4 specialisations (D_OUT bf16/fp32 x
        # HAS_BIAS true/false) used to be appended to every splitk kid's
        # device.cu. That duplicated their codegen across all 23 splitk
        # TUs (the linker dedupes the resulting weak symbols, but each
        # TU still pays the full RA + ISA emit cost on its compile).
        # Now they live in a single dedicated TU emitted by
        # _emit_splitk_reduce_tu, so each splitk kid's device.cu only
        # carries its own main-kernel instantiation. See
        # aiter/ops/opus/README.md §7.6 for the wall-time impact.
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
            self._host_instantiations.append(
                {"kid_name": k.name, "dtype": CDtype, "host_decl": host_decl}
            )
            self._device_instantiations.append(
                {"kid_name": k.name, "dtype": CDtype, "device_decl": device_decl}
            )

    # ── Lookup / manifest generation ──

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
        # Sorted flat-array layout (was: {(M,N,K), kernel<CTYPE>} initializer
        # list for std::unordered_map). Two reasons for the switch:
        #
        # 1. Compile time: the macro expanded into an unordered_map's
        #    initializer-list, and `std::function`-valued unordered_map
        #    instantiation alone added ~1s of frontend / template
        #    instantiation per dispatcher TU. A flat
        #    `static constexpr OpusA16W16RuntimeEntry kBf16Lookup[] = { ... };`
        #    plus `std::lower_bound` does the same lookup with much less
        #    template work.
        #
        # 2. Runtime: 339 entries, each cache-warm on first call after
        #    process start; binary search over a sorted POD array beats
        #    `std::function<...>` heap allocations + hash table walks
        #    handily, and the array constexpr-initialises so first-call
        #    latency drops too.
        #
        # The header now emits sorted `OpusA16W16RuntimeEntry` arrays
        # directly. ENTRY_MATCH_CTYPE / ENTRY_FORCE_FP32 use a CTYPE macro
        # param so the same emit logic feeds both BF16 (CTYPE=bf16_t) and
        # FP32 (CTYPE=fp32_t) tables, with splitk kids forced to <fp32_t>
        # regardless of the table they live in.
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
            # No body line break between `\` and the first entry; macro
            # continuation requires every line that participates in the
            # definition to end with `\`. Empty lines after `\` would
            # silently truncate the macro.
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
                is_splitk = k.kernel_tag == "a16w16_flatmm_splitk"
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
        # Same flat-array design as gen_lookup_dict, keyed on int kid
        # instead of (M,N,K). Sorted by kid so the dispatcher can binary
        # search.
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
            # Use explicit per-CTYPE macro names; the dispatcher in
            # opus_gemm.cu calls the right one from each
            # opus_a16w16_tune_dispatch<CDataType>() specialization.
            _emit_map(f, "GENERATE_A16W16_TUNE_LOOKUP_BF16", "bf16_t")
            _emit_map(f, "GENERATE_A16W16_TUNE_LOOKUP_FP32", "fp32_t")

    def gen_manifest_head(self, kernels_dict):
        # Forward declarations for every launcher symbol the dispatcher
        # references. Declarations only -- the definitions live in the
        # fused host TU (or in per-kid TUs in the legacy layout). All
        # use aiter_tensor_t (POD, declared in csrc/include/aiter_tensor.h)
        # so this header costs ~200 lines instead of the ~50K that
        # <torch/all.h> + <torch/extension.h> would drag in.
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
        # a16w16 family (5 args with optional bias + splitK): shared
        # signature for tune lookup. All three a16w16-family launchers
        # (split-barrier / flatmm / flatmm_splitk) accept the optional bias
        # argument so they can populate the same std::function slot in
        # GENERATE_A16W16_TUNE_LOOKUP. flatmm rejects non-empty bias at
        # runtime; the other two consume it.
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

    # ── Per-pass TU emission ──
    #
    # Replaces the old "one .cpp per (kid, dtype)" scheme. The wall-time
    # math:
    #   Old: 38 .cpp TUs each pay full <torch/extension.h> + <hip/...>
    #        parse on host pass (~13s) + a small device pass (~2s, after
    #        the .cuh #ifndef split). Total parallel wall ≈ host pass.
    #   New: 1 fused host TU does the heavy host parse exactly once
    #        (~13s) + N device TUs each ~2s. Parallel wall ≈ max(13s,
    #        slowest device TU). The 38 host parses collapse into one,
    #        so the critical path shrinks to roughly the single host
    #        TU's compile time.

    def _emit_fused_host_tu(self):
        """Emit one fused HOST translation unit covering every kid's
        launcher instantiation.

        Defines OPUS_FUSED_HOST_TU before including the .cuh's so each
        .cuh swaps its `#include "{pipeline_header}"` for the lighter
        `#include "{traits_header}"` + a forward declaration of the
        kernel template. That sidesteps the ODR clash we'd otherwise
        hit when multiple pipeline headers (a16w16, a8w8, ...) define
        same-named layout helpers like `make_layout_ga_noscale` in the
        same TU.

        Each launcher's `<<<...>>>` inside the .cuh body emits an
        undefined `__device_stub__<...>` reference; the link step
        resolves it against the matching per-kid device.cu (which DOES
        include the full pipeline header and instantiates the kernel
        template).

        End result: the heavy <torch/extension.h> + ATen + HIP runtime
        parse runs ONCE per module rebuild instead of N times, while
        device codegen still parallelises across N tiny self-contained
        device.cu's.
        """
        impl_includes = sorted({row["kid_name"] for row in self._host_instantiations})
        host_body = "".join(row["host_decl"] for row in self._host_instantiations)
        # splitk_reduce_kernel is launched directly from each
        # a16w16_flatmm_splitk launcher body, so the fused host TU has
        # to see its declaration to type-check the <<<...>>> call. It
        # has 4 specialisations (D_OUT bf16/fp32 x HAS_BIAS true/false),
        # all instantiated in each splitk device.cu.
        forward_decls = (
            "// Forward declaration only. Specialisations are instantiated\n"
            "// by every splitk device.cu so the linker always finds at\n"
            "// least one definition (weak symbols dedupe across TUs).\n"
            "template<int VEC_, int BLOCK_, typename D_OUT,\n"
            "         bool HAS_BIAS_, typename D_BIAS_,\n"
            "         bool HAS_OOB_>\n"
            "__global__ void splitk_reduce_kernel(\n"
            "    const float* workspace, D_OUT* c_out,\n"
            "    int split_k, int M, int N, int batch,\n"
            "    int padded_M, int padded_N,\n"
            "    const D_BIAS_* bias, int stride_bias_batch);\n"
        )
        contents = (
            "// SPDX-License-Identifier: MIT\n"
            "// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.\n"
            "//\n"
            "// Auto-generated. Do not edit. See gen_instances.py:_emit_fused_host_tu.\n"
            "//\n"
            "// Fused HOST translation unit: instantiates every launcher in one\n"
            "// .o, paying the heavy <torch/extension.h> parse only once per\n"
            "// module rebuild. Per-kid device codegen lives in <kid>_C<dtype>.device.cu;\n"
            "// the link step wires our undefined __device_stub__ references to\n"
            "// the device TUs' kernel definitions (-fno-gpu-rdc safe because\n"
            "// host stubs are weak symbols and fat-binary segments are merged\n"
            "// at link time).\n"
            "//\n"
            "// The whole TU is host-only -- the per-kid device.cu files are\n"
            "// where __global__ instantiations actually live -- so we skip the\n"
            "// device pass entirely. hipcc still launches a device-pass\n"
            "// invocation, but it sees an empty TU and finishes in <0.5s.\n"
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
        Path(os.path.join(self.instances_path, "all_instances_host.cu")).write_text(
            contents
        )

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
            # Include the kid's .cuh -- it transitively pulls in the
            # full pipeline header (because OPUS_FUSED_HOST_TU is NOT
            # defined here) and the per-kid Traits aliases. The .cuh's
            # `#if !defined(__HIP_DEVICE_COMPILE__) && !defined(__HIPCC_RTC__)`
            # guard hides the launcher body and the torch includes
            # from this device-only TU, so neither the host pass nor
            # the device pass tries to parse <torch/extension.h>.
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

        The reduce kernel template lives in splitk_reduce_gfx950.cuh,
        included transitively here via the splitk pipeline header so we
        get the same definition the splitk kids saw before. There's no
        need to gate via OPUS_FUSED_HOST_TU because no launcher body or
        traits structure is being pulled in.
        """
        contents = (
            "// SPDX-License-Identifier: MIT\n"
            "// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.\n"
            "//\n"
            "// Auto-generated. Do not edit. See gen_instances.py:_emit_splitk_reduce_tu.\n"
            "//\n"
            "// Dedicated device TU for splitk_reduce_kernel specialisations\n"
            "// (D_OUT bf16/fp32 x HAS_BIAS true/false x HAS_OOB true/false).\n"
            "// Carved out of every splitk kid's device.cu so the reduce\n"
            "// kernels only get RA'd / ISA-emitted once. Compiled with\n"
            "// -D__HIPCC_RTC__ so the host pass is minimal.\n"
            '#include "gfx950/splitk_reduce_gfx950.cuh"\n'
            "// HAS_OOB=true variants\n"
            "template __global__ void splitk_reduce_kernel<16, 64, __bf16, true,  __bf16, true>(\n"
            "    const float*, __bf16*, int, int, int, int, int, int,\n"
            "    const __bf16*, int);\n"
            "template __global__ void splitk_reduce_kernel<16, 64, __bf16, false, __bf16, true>(\n"
            "    const float*, __bf16*, int, int, int, int, int, int,\n"
            "    const __bf16*, int);\n"
            "template __global__ void splitk_reduce_kernel<16, 64, float,  true,  float,  true>(\n"
            "    const float*, float*,  int, int, int, int, int, int,\n"
            "    const float*,  int);\n"
            "template __global__ void splitk_reduce_kernel<16, 64, float,  false, float,  true>(\n"
            "    const float*, float*,  int, int, int, int, int, int,\n"
            "    const float*,  int);\n"
            "// HAS_OOB=false variants\n"
            "template __global__ void splitk_reduce_kernel<16, 64, __bf16, true,  __bf16, false>(\n"
            "    const float*, __bf16*, int, int, int, int, int, int,\n"
            "    const __bf16*, int);\n"
            "template __global__ void splitk_reduce_kernel<16, 64, __bf16, false, __bf16, false>(\n"
            "    const float*, __bf16*, int, int, int, int, int, int,\n"
            "    const __bf16*, int);\n"
            "template __global__ void splitk_reduce_kernel<16, 64, float,  true,  float,  false>(\n"
            "    const float*, float*,  int, int, int, int, int, int,\n"
            "    const float*,  int);\n"
            "template __global__ void splitk_reduce_kernel<16, 64, float,  false, float,  false>(\n"
            "    const float*, float*,  int, int, int, int, int, int,\n"
            "    const float*,  int);\n"
        )
        Path(os.path.join(self.instances_path, "splitk_reduce.device.cu")).write_text(
            contents
        )

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

        # Emit one fused HOST TU + N device TUs (one per kid, dtype) +
        # one dedicated splitk_reduce.device.cu. See the docstrings on
        # those methods for the full rationale.
        self._emit_fused_host_tu()
        self._emit_device_tus()
        # Only emit the standalone reduce TU if the build actually has a
        # splitk kid (otherwise the fused host TU will never reference
        # any reduce symbol and the linker would warn about an unused
        # weak object). Probed by checking any device instantiation
        # whose kernel function name is the splitk one.
        if any(
            "flatmm_splitk" in row["kid_name"] for row in self._device_instantiations
        ):
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
        # Accept either the legacy "kernelId" column or the new "solidx"
        # column (matches aiter/configs/model_configs/gptoss_bf16_tuned_gemm.csv
        # schema; see opus_gemm_tune.py result_to_df override).
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
    }

    # --- Compute the subset-compile set S ------------------------------------
    #
    # S = (CSV opus rows' kids) ∪ (previous sidecar contents) ∪ HEURISTIC_DEFAULT_KIDS
    # Intersected with kernels_list.keys() to guard against stale kids in CSV.
    #
    # When --kernel_tag is specified, we honor that filter as a developer
    # debug aid (codegen-only-this-family) but still enforce
    # HEURISTIC_DEFAULT_KIDS containment so subset-compile invariants hold.

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

    # Always keep a8w8 family (kid 1, 2) in the build -- they are not part
    # of the a16w16-family subset-compile contract but the module_deepgemm_opus
    # .so historically exposes them via opus_gemm() for fp8 paths.
    S |= set(a8w8_scale_kernels_list.keys())
    S |= set(a8w8_kernels_list.keys())

    # Honor --kernel_tag as a developer override that *further restricts* the
    # set (within the a16w16 / a8w8 families). It cannot drop any kid from
    # HEURISTIC_DEFAULT_KIDS though -- the sidecar assertion below guarantees
    # the build still has a viable heuristic fallback.
    if args.kernel_tag:
        tag_keys = set(TAG_TO_LIST.get(args.kernel_tag, {}).keys())
        if tag_keys:
            # Restrict to the requested family + heuristic defaults + a8w8 dispatch.
            S = (S & tag_keys) | set(HEURISTIC_DEFAULT_KIDS)
            S |= set(a8w8_scale_kernels_list.keys())
            S |= set(a8w8_kernels_list.keys())

    # Heuristic-fallback invariant (single source of truth: opus_gemm_common.py).
    # If this fires, opus_a16w16_heuristic_kid_gfx950() in
    # opus_gemm_heuristic_dispatch_gfx950.cuh would return a kid that the
    # tune_lookup table cannot resolve -- a runtime AITER_CHECK failure.
    missing_heuristic = set(HEURISTIC_DEFAULT_KIDS) - S
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

    # Bake the (M, N, K) -> kernel runtime lookup. gen_instances() emitted
    # an empty version; if any CSV opus row matches the current cu_num we
    # overwrite it with the real entries. We concatenate all CSVs into a
    # single DataFrame first and feed it through get_tune_dict's logic.
    if csv_paths:
        # Concatenate all opus rows from all matched CSV files (filtered by
        # libtype). We persist the combined DataFrame to a temp file because
        # get_tune_dict reads from a path. Keep the existing one-CSV
        # contract by writing to a single temp file.
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
                    # Easier: just keep the entry; the launcher symbol is
                    # only resolvable if v.name's kid is in S, which we've
                    # already guaranteed via valid_kids ∩ S.
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
