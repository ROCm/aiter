# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
import os
from pathlib import Path
import pandas as pd
import argparse
import shutil
import torch
from opus_gemm_common import (
    OpusGemmInstance,
    kernels_list,
    default_kernels_dict,
    a8w8_scale_kernels_list,
    a8w8_kernels_list,
    a16w16_kernels_list,
    a16w16_flatmm_kernels_list,
    a16w16_flatmm_splitk_kernels_list,
)


PIPELINE_HEADER_MAP = {
    "a8w8_scale":           "pipeline/opus_gemm_pipeline_a8w8_scale.cuh",
    "a8w8":                 "pipeline/opus_gemm_pipeline_a8w8_noscale.cuh",
    "a16w16":               "pipeline/opus_gemm_pipeline_a16w16.cuh",
    "a16w16_flatmm":        "pipeline/opus_gemm_pipeline_a16w16_flatmm.cuh",
    "a16w16_flatmm_splitk": "pipeline/opus_gemm_pipeline_a16w16_flatmm_splitk.cuh",
}

KERNEL_FUNC_MAP = {
    "a8w8_scale":           "gemm_a8w8_scale_kernel",
    "a8w8":                 "gemm_a8w8_noscale_kernel",
    "a16w16":               "gemm_a16w16_kernel",
    "a16w16_flatmm":        "gemm_a16w16_flatmm_kernel",
    "a16w16_flatmm_splitk": "gemm_a16w16_flatmm_splitk_kernel",
}

INPUT_DTYPE_MAP = {
    "a8w8_scale":           ("fp8_t", "fp8_t"),
    "a8w8":                 ("fp8_t", "fp8_t"),
    "a16w16":               ("bf16_t", "bf16_t"),
    "a16w16_flatmm":        ("bf16_t", "bf16_t"),
    "a16w16_flatmm_splitk": ("bf16_t", "bf16_t"),
}

# Tags whose launchers take 3 torch tensors (XQ, WQ, Y) + int splitK. Splitk
# launcher has the same Python-facing signature but with literal (non-trivial)
# splitK semantics.
NOSCALE_TAGS = {"a8w8", "a16w16", "a16w16_flatmm", "a16w16_flatmm_splitk"}

# a16w16-family tags whose launchers land in opus_gemm_a16w16_tune_lookup.h
# and therefore need the 4-arg (XQ, WQ, Y, int splitK) signature so they can
# share the std::function<Tensor(Tensor&,Tensor&,Tensor&,int)> slot.
# Non-splitk tags in this set ignore the splitK param in their body.
A16W16_TUNE_TAGS = {"a16w16", "a16w16_flatmm", "a16w16_flatmm_splitk"}

TRAITS_NAME_MAP = {
    "a8w8_scale":           "opus_gemm_a8w8_scale_traits",
    "a8w8":                 "opus_gemm_a8w8_noscale_traits",
    "a16w16":               "opus_gemm_a16w16_traits",
    "a16w16_flatmm":        "opus_gemm_a16w16_flatmm_traits",
    "a16w16_flatmm_splitk": "opus_flatmm_splitk_traits",
}

KARGS_NAME_MAP = {
    "a8w8_scale":           "opus_gemm_scale_kargs",
    "a8w8":                 "opus_gemm_noscale_kargs",
    "a16w16":                "opus_gemm_noscale_kargs",
    "a16w16_flatmm":        "opus_gemm_flatmm_kargs",
    "a16w16_flatmm_splitk": "opus_gemm_flatmm_splitk_kargs",
}

WARP_SIZE = 64
VALID_BF16_MFMA = {(16, 16, 32), (32, 32, 16)}
# Flatmm pipeline currently only supports W_M < 32 (ra layout relies on
# LOAD_GROUP_M_LANE == 1). W_M == 32 (LGML == 4) path not rewritten.
VALID_FLATMM_MFMA = {(16, 16, 32)}
VALID_FLATMM_SPLITK_MFMA = {(16, 16, 32)}


class opus_gemm_codegen:
    def __init__(self, working_path, istune=False):
        self.working_path = working_path
        self.impl_path = os.path.join(working_path, "impl")
        self.instances_path = os.path.join(working_path, "instances")
        self.istune = istune

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
        for tag, ww, vec in [("ra", k.W_M * k.W_K, k.VEC_A), ("rb", k.W_N * k.W_K, k.VEC_B)]:
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
            smem_m_rep = HALF_B_M // smem_sub if smem_sub and HALF_B_M % smem_sub == 0 else 0
            smem_n_rep = HALF_B_N // smem_sub if smem_sub and HALF_B_N % smem_sub == 0 else 0
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
            msg = f"Invalid a16w16 instance '{k.name}':\n" + "\n".join(f"  - {e}" for e in errors)
            raise ValueError(msg)

        return {
            "E_M": E_M, "E_N": E_N, "E_K": E_K,
            "agprs": total_agprs, "vgpr_est": vgpr_est,
            "lds_bytes": total_lds if smem_linear_wave % k.B_K == 0 else -1,
            "min_k": 2 * k.B_K,
        }

    # ── a16w16_flatmm validator ──

    @staticmethod
    def _validate_a16w16_flatmm(k: OpusGemmInstance):
        """Validate an a16w16_flatmm instance at codegen time.

        Mirrors the static_asserts in opus_gemm_a16w16_flatmm_traits: derives
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
        per_block_iter = (num_load_groups_per_bm + num_load_groups_per_bn) * \
                          num_load_groups_per_bk * smem_per_group_load_size
        pfk = max_lds_per_wg // per_block_iter if per_block_iter > 0 else 0
        if pfk < 3:
            errors.append(
                f"prefetch_k_iter={pfk} < 3 "
                f"(LDS budget {max_lds_per_wg} / per-iter {per_block_iter})"
            )

        min_k = pfk * k.B_K
        lds_footprint = pfk * per_block_iter

        if errors:
            msg = f"Invalid a16w16_flatmm instance '{k.name}':\n" + "\n".join(f"  - {e}" for e in errors)
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
        per_block_iter = (num_load_groups_per_bm + num_load_groups_per_bn) * \
                          num_load_groups_per_bk * smem_per_group_load_size
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
            msg = (
                f"Invalid a16w16_flatmm_splitk instance '{k.name}':\n"
                + "\n".join(f"  - {e}" for e in errors)
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
        kernel_func = KERNEL_FUNC_MAP[k.kernel_tag]
        da, db = INPUT_DTYPE_MAP[k.kernel_tag]
        traits_name = TRAITS_NAME_MAP[k.kernel_tag]
        kargs_name = KARGS_NAME_MAP[k.kernel_tag]

        if k.kernel_tag == "a16w16_flatmm":
            self._gen_flatmm_instance(k, pipeline_header, kernel_func, da, db, traits_name, kargs_name)
        elif k.kernel_tag == "a16w16_flatmm_splitk":
            self._gen_flatmm_splitk_instance(k, pipeline_header, kernel_func, da, db, traits_name, kargs_name)
        elif k.kernel_tag in NOSCALE_TAGS:
            self._gen_noscale_instance(k, pipeline_header, kernel_func, da, db, traits_name, kargs_name)
        else:
            self._gen_scale_instance(k, pipeline_header, kernel_func, da, db, traits_name, kargs_name)

    def _gen_scale_instance(self, k, pipeline_header, kernel_func, da, db, traits_name, kargs_name):
        INSTANCE_IMPL = f"""// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
#include <torch/all.h>
#include <torch/extension.h>
#include <ATen/hip/HIPContext.h>
#include <ATen/hip/impl/HIPStreamMasqueradingAsCUDA.h>
#include "{pipeline_header}"

template <typename D_C>
torch::Tensor
{k.name}(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &Y,
    std::optional<torch::Tensor> x_scale,
    std::optional<torch::Tensor> w_scale)
{{{{
    int batch = XQ.size(0);
    int M = XQ.size(1);
    int N = WQ.size(1);
    int K = XQ.size(2);

    using Traits = {traits_name}<{k.BLOCK_SIZE},
        opus::seq<{k.B_M}, {k.B_N}, {k.B_K}>,
        opus::tuple<{da}, {db}, D_C, fp32_t, fp32_t>,
        opus::seq<{k.VEC_A}, {k.VEC_B}, {k.VEC_C}>,
        opus::seq<{k.GROUP_M}, {k.GROUP_N}, {k.GROUP_K}>>;

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

    auto stream = at::hip::getCurrentHIPStreamMasqueradingAsCUDA().stream();
    {kernel_func}<Traits><<<grid, block, 0, stream>>>(kargs);

    return Y;
}}}}
"""
        Path(os.path.join(self.impl_path, f"{k.name}.cuh")).write_text(INSTANCE_IMPL)

        INSTANCE_TEMPLATE = """// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
#include "impl/{name}.cuh"
template torch::Tensor
{name}<{dtype}>(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &Y,
    std::optional<torch::Tensor> x_scale,
    std::optional<torch::Tensor> w_scale);
"""
        for CDtype in k.output_dtypes:
            instance = INSTANCE_TEMPLATE.format(name=k.name, dtype=CDtype)
            Path(
                os.path.join(self.instances_path, f"{k.name}_C{CDtype}.cpp")
            ).write_text(instance)

    def _gen_noscale_instance(self, k, pipeline_header, kernel_func, da, db, traits_name, kargs_name):
        traits_extra = ""
        if k.kernel_tag == "a16w16":
            traits_extra = (f",\n        opus::seq<{k.T_M}, {k.T_N}, 1>,"
                           f"\n        opus::seq<{k.W_M}, {k.W_N}, {k.W_K}>")

        min_k = 2 * k.B_K
        # Kid-specific K-bound checks. Split-barrier pipeline requires
        # K >= 2 * B_K and the loop count to be even.
        k_check = f"""
    int loops_ = (K + {k.B_K} - 1) / {k.B_K};
    TORCH_CHECK(loops_ >= 2,
        "K=", K, " too small for B_K={k.B_K}, need K >= {min_k}");
    TORCH_CHECK(loops_ % 2 == 0,
        "ceil_div(K, {k.B_K})=", loops_, " must be even (prefetch constraint)");
    TORCH_CHECK(M >= 1 && N >= 1, "M and N must be >= 1");
"""

        # a16w16 kids live in opus_gemm_a16w16_tune_lookup.h alongside the
        # flatmm + splitk launchers, so their std::function slot requires the
        # 4-arg signature (XQ, WQ, Y, int splitK). The body ignores splitK
        # (split-barrier pipeline has no split-K concept). a8w8 kids never
        # enter the a16w16 lookup; they keep the 3-arg signature.
        extra_param  = ",\n    int /*splitK*/"           if k.kernel_tag in A16W16_TUNE_TAGS else ""
        extra_tmpl_param = ",\n    int" if k.kernel_tag in A16W16_TUNE_TAGS else ""

        INSTANCE_IMPL = f"""// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
#include <torch/all.h>
#include <torch/extension.h>
#include <ATen/hip/HIPContext.h>
#include <ATen/hip/impl/HIPStreamMasqueradingAsCUDA.h>
#include "{pipeline_header}"

template <typename D_C>
torch::Tensor
{k.name}(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &Y{extra_param})
{{{{
    int batch = XQ.size(0);
    int M = XQ.size(1);
    int N = WQ.size(1);
    int K = XQ.size(2);
{k_check}
    using Traits = {traits_name}<{k.BLOCK_SIZE},
        opus::seq<{k.B_M}, {k.B_N}, {k.B_K}>,
        opus::tuple<{da}, {db}, D_C, fp32_t>,
        opus::seq<{k.VEC_A}, {k.VEC_B}, {k.VEC_C}>{traits_extra}>;

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

    int num_tiles_m = (M + {k.B_M} - 1) / {k.B_M};
    int num_tiles_n = (N + {k.B_N} - 1) / {k.B_N};
    dim3 grid(num_tiles_m * num_tiles_n, 1, batch);
    dim3 block({k.BLOCK_SIZE});

    auto stream = at::hip::getCurrentHIPStreamMasqueradingAsCUDA().stream();
    {kernel_func}<Traits><<<grid, block, 0, stream>>>(kargs);

    return Y;
}}}}
"""
        Path(os.path.join(self.impl_path, f"{k.name}.cuh")).write_text(INSTANCE_IMPL)

        inst_extra_param = ",\n    int" if k.kernel_tag in A16W16_TUNE_TAGS else ""
        INSTANCE_TEMPLATE = f"""// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
#include "impl/{{name}}.cuh"
template torch::Tensor
{{name}}<{{dtype}}>(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &Y{inst_extra_param});
"""
        for CDtype in k.output_dtypes:
            instance = INSTANCE_TEMPLATE.format(name=k.name, dtype=CDtype)
            Path(
                os.path.join(self.instances_path, f"{k.name}_C{CDtype}.cpp")
            ).write_text(instance)

    def _gen_flatmm_instance(self, k, pipeline_header, kernel_func, da, db, traits_name, kargs_name):
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
    TORCH_CHECK(loops_ >= Traits::prefetch_k_iter,
        "K=", K, " too small for flatmm B_K={k.B_K}, need K >= pfk*B_K = ",
        Traits::prefetch_k_iter * {k.B_K}, " (pfk=", Traits::prefetch_k_iter, ")");
    TORCH_CHECK(M >= 1 && N >= 1 && K >= 1, "M, N, K must be >= 1");
    TORCH_CHECK(batch >= 1, "batch must be >= 1");
"""

        INSTANCE_IMPL = f"""// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
#include <torch/all.h>
#include <torch/extension.h>
#include <ATen/hip/HIPContext.h>
#include <ATen/hip/impl/HIPStreamMasqueradingAsCUDA.h>
#include "{pipeline_header}"

template <typename D_C>
torch::Tensor
{k.name}(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &Y,
    int /*splitK*/)   // flatmm (non-splitk) ignores splitK; shares tune-lookup slot signature
{{{{
    int batch = XQ.size(0);
    int M = XQ.size(1);
    int N = WQ.size(1);
    int K = XQ.size(2);

    using Traits = {traits_name}<{k.BLOCK_SIZE},
        opus::seq<{k.B_M}, {k.B_N}, {k.B_K}>,
        opus::tuple<{da}, {db}, D_C, fp32_t, D_C>,
        opus::seq<{k.VEC_A}, {k.VEC_B}, {k.VEC_C}>,
        opus::seq<{k.W_M}, {k.W_N}, {k.W_K}>,
        {k.WG_PER_CU},
        {has_bias_str}>;
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

    auto stream = at::hip::getCurrentHIPStreamMasqueradingAsCUDA().stream();
    {kernel_func}<Traits><<<grid, block, 0, stream>>>(kargs);

    return Y;
}}}}
"""
        Path(os.path.join(self.impl_path, f"{k.name}.cuh")).write_text(INSTANCE_IMPL)

        INSTANCE_TEMPLATE = """// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
#include "impl/{name}.cuh"
template torch::Tensor
{name}<{dtype}>(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &Y,
    int);
"""
        for CDtype in k.output_dtypes:
            instance = INSTANCE_TEMPLATE.format(name=k.name, dtype=CDtype)
            Path(
                os.path.join(self.instances_path, f"{k.name}_C{CDtype}.cpp")
            ).write_text(instance)

    def _gen_flatmm_splitk_instance(self, k, pipeline_header, kernel_func, da, db, traits_name, kargs_name):
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
        INSTANCE_IMPL = f"""// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
#include <torch/all.h>
#include <torch/extension.h>
#include <ATen/hip/HIPContext.h>
#include <ATen/hip/impl/HIPStreamMasqueradingAsCUDA.h>
#include "{pipeline_header}"

template <typename D_C>
torch::Tensor
{k.name}(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &Y,
    int splitK)
{{{{
    static_assert(std::is_same<D_C, fp32_t>::value,
        "splitk main kernel uses fp32 workspace; D_C template param must be fp32_t "
        "(Y is still bf16; reduce kernel does the fp32->bf16 cast)");

    int batch = XQ.size(0);
    int M = XQ.size(1);
    int N = WQ.size(1);
    int K = XQ.size(2);

    TORCH_CHECK(Y.dtype() == at::ScalarType::BFloat16,
        "flatmm_splitk requires Y dtype bf16 (reduce kernel casts fp32 workspace to bf16)");
    TORCH_CHECK(M >= 1 && N >= 1 && K >= 1 && batch >= 1,
        "M, N, K, batch must be >= 1");

    using Traits = {traits_name}<{k.BLOCK_SIZE},
        opus::seq<{k.B_M}, {k.B_N}, {k.B_K}>,
        opus::tuple<{da}, {db}, fp32_t, fp32_t, {da}>,   // D_C=fp32 forced
        opus::seq<{k.VEC_A}, {k.VEC_B}, {k.VEC_C}>,
        opus::seq<{k.W_M}, {k.W_N}, {k.W_K}>,
        {k.WG_PER_CU},
        false>;                                           // HAS_BIAS=false

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
    TORCH_CHECK(total_iters >= pfk,
        "K=", K, " too small for flatmm_splitk B_K={k.B_K}: "
        "need total_iters >= pfk*B_K = ", pfk * {k.B_K},
        " (pfk=", pfk, ")");

    int num_tiles_m = (M + {k.B_M} - 1) / {k.B_M};
    int num_tiles_n = (N + {k.B_N} - 1) / {k.B_N};
    int padded_M    = num_tiles_m * {k.B_M};
    int padded_N    = num_tiles_n * {k.B_N};

    // Allocate fp32 workspace via torch caching allocator (same idiom as
    // aiter/ops/triton/gemm/basic/gemm_a16w16.py y_pp). No persistent cache.
    auto workspace = torch::empty(
        {{{{(int64_t)split_k, (int64_t)batch, (int64_t)padded_M, (int64_t)padded_N}}}},
        torch::TensorOptions().dtype(torch::kFloat32).device(Y.device()));

    {kargs_name} kargs{{{{}}}};
    kargs.ptr_a         = XQ.data_ptr();
    kargs.ptr_b         = WQ.data_ptr();
    kargs.ptr_workspace = workspace.data_ptr();
    kargs.ptr_c         = Y.data_ptr();
    kargs.ptr_bias      = nullptr;  // HAS_BIAS=false; field reserved for future.
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

    dim3 grid_main(num_tiles_m * num_tiles_n * split_k, 1, batch);
    dim3 block_main({k.BLOCK_SIZE});

    constexpr int REDUCE_VEC = 16;
    constexpr int REDUCE_BS  = 64;
    // Note: no padded_N % REDUCE_VEC check. The reduce kernel has a tail
    // path (see opus_gemm_pipeline_a16w16_flatmm_splitk.cuh) that handles
    // the (n_base + VEC > N) case with per-element scalar stores, so any
    // N (including odd / non-power-of-16) is safe.
    dim3 grid_reduce((N + REDUCE_VEC * REDUCE_BS - 1) / (REDUCE_VEC * REDUCE_BS),
                      batch * M, 1);
    dim3 block_reduce(REDUCE_BS);

    auto stream = at::hip::getCurrentHIPStreamMasqueradingAsCUDA().stream();
    {kernel_func}<Traits><<<grid_main, block_main, 0, stream>>>(kargs);
    gemm_a16w16_flatmm_splitk_reduce_kernel<REDUCE_VEC, REDUCE_BS>
        <<<grid_reduce, block_reduce, 0, stream>>>(
            reinterpret_cast<const float*>(workspace.data_ptr()),
            reinterpret_cast<__bf16*>(Y.data_ptr()),
            split_k, M, N, batch, padded_M, padded_N);

    return Y;
}}}}
"""
        Path(os.path.join(self.impl_path, f"{k.name}.cuh")).write_text(INSTANCE_IMPL)

        INSTANCE_TEMPLATE = """// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
#include "impl/{name}.cuh"
template torch::Tensor
{name}<{dtype}>(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &Y,
    int);
"""
        for CDtype in k.output_dtypes:
            instance = INSTANCE_TEMPLATE.format(name=k.name, dtype=CDtype)
            Path(
                os.path.join(self.instances_path, f"{k.name}_C{CDtype}.cpp")
            ).write_text(instance)

    # ── Lookup / manifest generation ──

    def gen_lookup_dict(self, kernels_dict):
        """Emit opus_gemm_lookup.h with two (M,N,K)->kernel macros.

        Tuned-CSV driven lookup consumed by opus_gemm.cu's runtime
        `opus_dispatch_a16w16<CDataType>`. Two macros (BF16 / FP32)
        mirror `gen_a16w16_tune_lookup` and exist because splitk kids
        (200..210) are only emitted as `<fp32_t>` (their traits
        static_assert D_C==float, so referencing `splitk<bf16_t>`
        produces a linker error).

        Per-kid template argument rule (the interesting bit): the map
        *key* is (M, N, K) regardless of user Y dtype, but the *value*
        template argument depends on the kid's `output_dtypes`:

          * a16w16 kid 4..9         -> `<CTYPE>` (both bf16/fp32 exist).
          * a16w16_flatmm 100..115  -> `<CTYPE>` (both exist).
          * a16w16_flatmm_splitk    -> always `<fp32_t>`. The BF16 map
            therefore contains splitk entries too, but those entries
            instantiate `splitk<fp32_t>` -- same trick opus_gemm.cu's
            heuristic uses. Y is still bf16 because the reduce kernel
            casts at the end.

          * FP32 map excludes splitk kids entirely (their launcher
            TORCH_CHECKs Y.dtype()==BFloat16).
        """
        HEADER = """#pragma once
// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
// Per-CTYPE (M,N,K)->kernel tables. splitk kids only live in the FP32
// specialization of each map (the reduce kernel casts fp32->bf16 Y at
// runtime); their BF16-map entries still reference splitk<fp32_t>.
"""

        # ENTRY_MATCH_CTYPE uses the CTYPE macro parameter so it expands
        # to `kernel<bf16_t>` inside the BF16 macro and `kernel<fp32_t>`
        # inside the FP32 macro. ENTRY_FORCE_FP32 hardcodes the template
        # parameter for splitk kids that only have the fp32 instance.
        ENTRY_MATCH_CTYPE = """
       {{{MNK},                                                        \\
        {kernel_name}<CTYPE>}},                                        \\"""
        ENTRY_FORCE_FP32 = """
       {{{MNK},                                                        \\
        {kernel_name}<fp32_t>}},                                       \\"""

        def _emit_map(f, macro_name: str, ctype: str):
            f.write(f"#define {macro_name}(CTYPE)                         \\\n")
            f.write("   {                                                                   \\")
            for mnk, k in kernels_dict.items():
                if self.istune and isinstance(mnk, int):
                    mnk_lit = str(mnk)
                elif (not self.istune) and isinstance(mnk, tuple) and mnk[0] > 0:
                    mnk_lit = "{" + ", ".join(map(str, list(mnk))) + "}"
                else:
                    continue

                is_splitk = k.kernel_tag == "a16w16_flatmm_splitk"
                if is_splitk:
                    # splitk only emits <fp32_t>; skip from FP32-output map
                    # (launcher requires bf16 Y), include in BF16-output
                    # map with forced <fp32_t> template argument.
                    if ctype == "fp32_t":
                        continue
                    entry = ENTRY_FORCE_FP32
                else:
                    if ctype not in k.output_dtypes:
                        continue
                    entry = ENTRY_MATCH_CTYPE
                f.write(entry.format(MNK=mnk_lit, kernel_name=k.name))
            f.write("\n   }\n\n")

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
        HEADER = """#pragma once
// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
// A macro selector picks the right map based on CTYPE. Kids whose
// output_dtypes doesn't include CTYPE are omitted from that CTYPE's map
// (splitk kids only live in the fp32 map).
"""
        ENTRY = """
       {{{kid},                                                        \\
        {kernel_name}<CTYPE>}},                                        \\"""

        def _emit_map(f, macro_name, ctype):
            f.write(f"#define {macro_name}(CTYPE)                         \\\n")
            f.write("   {                                                                   \\")
            for kid, k in kernels_dict.items():
                if not (isinstance(kid, int) and k.kernel_tag in A16W16_TUNE_TAGS):
                    continue
                if ctype not in k.output_dtypes:
                    continue
                f.write(ENTRY.format(kid=kid, kernel_name=k.name))
            f.write("\n   }\n\n")

        with open(os.path.join(self.working_path, "opus_gemm_a16w16_tune_lookup.h"), "w") as f:
            f.write(HEADER)
            # Use explicit per-CTYPE macro names; the dispatcher in
            # opus_gemm.cu calls the right one from each
            # opus_a16w16_tune_dispatch<CDataType>() specialization.
            _emit_map(f, "GENERATE_A16W16_TUNE_LOOKUP_BF16", "bf16_t")
            _emit_map(f, "GENERATE_A16W16_TUNE_LOOKUP_FP32", "fp32_t")

    def gen_manifest_head(self, kernels_dict):
        MANIFEST_HEAD = """#pragma once
// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
#include <cstdlib>
#include <torch/extension.h>
"""
        MANIFEST_SCALE = """
template <typename D_C>
torch::Tensor
{kernel_name}(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &Y,
    std::optional<torch::Tensor> x_scale,
    std::optional<torch::Tensor> w_scale);
"""
        # a8w8 noscale (3 args, no splitK): stays compatible with
        # opus_gemm_lookup.h where a8w8 kids live.
        MANIFEST_NOSCALE_3ARG = """
template <typename D_C>
torch::Tensor
{kernel_name}(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &Y);
"""
        # a16w16 family (4 args with splitK): shared signature for tune lookup.
        MANIFEST_NOSCALE_4ARG = """
template <typename D_C>
torch::Tensor
{kernel_name}(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &Y,
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

    def gen_instances(self, kernels_dict):
        if os.path.exists(self.impl_path):
            shutil.rmtree(self.impl_path)
        os.mkdir(self.impl_path)
        if os.path.exists(self.instances_path):
            shutil.rmtree(self.instances_path)
        os.mkdir(self.instances_path)

        for mnk, k in kernels_dict.items():
            self.gen_instance(k)

        self.gen_lookup_dict(kernels_dict)
        self.gen_manifest_head(kernels_dict)
        self.gen_a16w16_tune_lookup(kernels_dict)


def get_tune_dict(tune_dict_csv):
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
        for i in range(len(tune_df)):
            M = tune_df.loc[i, "M"]
            N = tune_df.loc[i, "N"]
            K = tune_df.loc[i, "K"]
            kid = int(tune_df.loc[i, kid_col])
            if kid in kernels_list:
                tune_dict[(M, N, K)] = kernels_list[kid]
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
        "--tune_file",
        default=None,
        required=False,
        help=(
            "Optional path to the opus-private tuned CSV "
            "(e.g. aiter/ops/opus/configs/opus_gemm_a16w16_tuned.csv). "
            "When given and the file exists, winners matching the current "
            "device's cu_num are baked into opus_gemm_lookup.h via "
            "GENERATE_OPUS_LOOKUP_TABLE so opus_gemm()'s bf16 dispatch can "
            "hit them at runtime without the Python wrapper being in the "
            "loop. Without this flag the lookup table stays empty and the "
            "C++ dispatch falls straight through to the heuristic."
        ),
    )

    args = parser.parse_args()
    TAG_TO_LIST = {
        "a8w8_scale":           a8w8_scale_kernels_list,
        "a8w8":                 a8w8_kernels_list,
        "a16w16":                a16w16_kernels_list,
        "a16w16_flatmm":        a16w16_flatmm_kernels_list,
        "a16w16_flatmm_splitk": a16w16_flatmm_splitk_kernels_list,
    }
    kdict = TAG_TO_LIST.get(args.kernel_tag, kernels_list) if args.kernel_tag else kernels_list
    codegen = opus_gemm_codegen(args.working_path, args.tune)
    codegen.gen_instances(kdict)

    # If a tuned CSV is provided and present on disk, rewrite
    # opus_gemm_lookup.h so the C++ side can resolve tuned winners by
    # (M, N, K). gen_instances() has already emitted an empty version;
    # overwrite it with the CSV-driven entries. The kernel impls /
    # instance .cpp files were already produced above so every kernel
    # referenced by get_tune_dict is guaranteed to be buildable.
    if args.tune_file and os.path.exists(args.tune_file):
        tune_dict = get_tune_dict(args.tune_file)
        # get_tune_dict seeds with default_kernels_dict (negative int keys
        # used only in --tune runtime lookup); gen_lookup_dict already
        # skips those via the `isinstance(mnk, tuple) and mnk[0] > 0`
        # guard, so passing the full dict straight through is safe.
        codegen.gen_lookup_dict(tune_dict)
        print(f"[opus gen_instances] baked {sum(1 for k in tune_dict if isinstance(k, tuple) and k[0] > 0)} tuned entries from {args.tune_file} into opus_gemm_lookup.h")
    else:
        if args.tune_file:
            print(f"[opus gen_instances] --tune_file {args.tune_file} not found, using empty lookup")
