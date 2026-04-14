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
)


PIPELINE_HEADER_MAP = {
    "a8w8_scale": "pipeline/opus_gemm_pipeline_a8w8_scale.cuh",
    "a8w8":       "pipeline/opus_gemm_pipeline_a8w8_noscale.cuh",
    "a16w16":     "pipeline/opus_gemm_pipeline_a16w16.cuh",
}

KERNEL_FUNC_MAP = {
    "a8w8_scale": "gemm_a8w8_scale_kernel",
    "a8w8":       "gemm_a8w8_noscale_kernel",
    "a16w16":     "gemm_a16w16_kernel",
}

INPUT_DTYPE_MAP = {
    "a8w8_scale": ("fp8_t", "fp8_t"),
    "a8w8":       ("fp8_t", "fp8_t"),
    "a16w16":     ("bf16_t", "bf16_t"),
}

NOSCALE_TAGS = {"a8w8", "a16w16"}

TRAITS_NAME_MAP = {
    "a8w8_scale": "opus_gemm_a8w8_scale_traits",
    "a8w8":       "opus_gemm_a8w8_noscale_traits",
    "a16w16":     "opus_gemm_a16w16_traits",
}

KARGS_NAME_MAP = {
    "a8w8_scale": "opus_gemm_scale_kargs",
    "a8w8":       "opus_gemm_noscale_kargs",
    "a16w16":     "opus_gemm_noscale_kargs",
}

WARP_SIZE = 64
VALID_BF16_MFMA = {(16, 16, 32), (32, 32, 16)}


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

        pipeline_header = PIPELINE_HEADER_MAP[k.kernel_tag]
        kernel_func = KERNEL_FUNC_MAP[k.kernel_tag]
        da, db = INPUT_DTYPE_MAP[k.kernel_tag]
        traits_name = TRAITS_NAME_MAP[k.kernel_tag]
        kargs_name = KARGS_NAME_MAP[k.kernel_tag]

        if k.kernel_tag in NOSCALE_TAGS:
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
        k_check = f"""
    int loops_ = (K + {k.B_K} - 1) / {k.B_K};
    TORCH_CHECK(loops_ >= 2,
        "K=", K, " too small for B_K={k.B_K}, need K >= {min_k}");
    TORCH_CHECK(loops_ % 2 == 0,
        "ceil_div(K, {k.B_K})=", loops_, " must be even (prefetch constraint)");
    TORCH_CHECK(M >= 1 && N >= 1, "M and N must be >= 1");
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
    torch::Tensor &Y)
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

        INSTANCE_TEMPLATE = """// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
#include "impl/{name}.cuh"
template torch::Tensor
{name}<{dtype}>(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &Y);
"""
        for CDtype in k.output_dtypes:
            instance = INSTANCE_TEMPLATE.format(name=k.name, dtype=CDtype)
            Path(
                os.path.join(self.instances_path, f"{k.name}_C{CDtype}.cpp")
            ).write_text(instance)

    # ── Lookup / manifest generation ──

    def gen_lookup_dict(self, kernels_dict):
        LOOKUP_HEAD = """#pragma once
// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
#define GENERATE_OPUS_LOOKUP_TABLE(CTYPE)                              \\
   {                                                                   \\"""

        LOOKUP_TEMPLATE = """
       {{{MNK},                                                        \\
        {kernel_name}<CTYPE>}},                                        \\"""

        LOOKUP_END = """
   }
"""
        with open(os.path.join(self.working_path, "opus_gemm_lookup.h"), "w") as f:
            f.write(LOOKUP_HEAD)
            for mnk, k in kernels_dict.items():
                if not self.istune and (isinstance(mnk, tuple) and mnk[0] > 0):
                    f.write(
                        LOOKUP_TEMPLATE.format(
                            MNK="{"
                            + ", ".join(map(str, list(mnk)))
                            + "}",
                            kernel_name=k.name,
                        )
                    )
                elif self.istune and isinstance(mnk, int):
                    f.write(LOOKUP_TEMPLATE.format(MNK=mnk, kernel_name=k.name))
            f.write(LOOKUP_END)

    def gen_a16w16_tune_lookup(self, kernels_dict):
        """Emit opus_gemm_a16w16_tune_lookup.h with int-ID-to-kernel map for tuning."""
        HEADER = """#pragma once
// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
#define GENERATE_A16W16_TUNE_LOOKUP(CTYPE)                             \\
   {                                                                   \\"""
        ENTRY = """
       {{{kid},                                                        \\
        {kernel_name}<CTYPE>}},                                        \\"""
        FOOTER = """
   }
"""
        with open(os.path.join(self.working_path, "opus_gemm_a16w16_tune_lookup.h"), "w") as f:
            f.write(HEADER)
            for kid, k in kernels_dict.items():
                if isinstance(kid, int) and k.kernel_tag == "a16w16":
                    f.write(ENTRY.format(kid=kid, kernel_name=k.name))
            f.write(FOOTER)

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
        MANIFEST_NOSCALE = """
template <typename D_C>
torch::Tensor
{kernel_name}(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &Y);
"""
        with open(os.path.join(self.working_path, "opus_gemm_manifest.h"), "w") as f:
            f.write(MANIFEST_HEAD)
            for mnk, k in kernels_dict.items():
                if k.kernel_tag in NOSCALE_TAGS:
                    f.write(MANIFEST_NOSCALE.format(kernel_name=k.name))
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
        for i in range(len(tune_df)):
            M = tune_df.loc[i, "M"]
            N = tune_df.loc[i, "N"]
            K = tune_df.loc[i, "K"]
            kid = tune_df.loc[i, "kernelId"]
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
        help="filter kernels by tag (e.g. a16w16, a8w8, a8w8_scale)",
    )

    args = parser.parse_args()
    TAG_TO_LIST = {
        "a8w8_scale": a8w8_scale_kernels_list,
        "a8w8": a8w8_kernels_list,
        "a16w16": a16w16_kernels_list,
    }
    kdict = TAG_TO_LIST.get(args.kernel_tag, kernels_list) if args.kernel_tag else kernels_list
    codegen = opus_gemm_codegen(args.working_path, args.tune)
    codegen.gen_instances(kdict)
