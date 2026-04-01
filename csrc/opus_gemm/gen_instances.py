# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
import os
from pathlib import Path
import pandas as pd
import argparse
import shutil
import torch
from opus_gemm_common import OpusGemmInstance, kernels_list, default_kernels_dict


PIPELINE_HEADER_MAP = {
    "a8w8_scale": "pipeline/opus_gemm_pipeline_a8w8_scale.cuh",
    "a8w8":       "pipeline/opus_gemm_pipeline_noscale.cuh",
    "a16w16":     "pipeline/opus_gemm_pipeline_noscale.cuh",
}

KERNEL_FUNC_MAP = {
    "a8w8_scale": "gemm_a8w8_scale_kernel",
    "a8w8":       "gemm_noscale_kernel",
    "a16w16":     "gemm_noscale_kernel",
}

INPUT_DTYPE_MAP = {
    "a8w8_scale": ("fp8_t", "fp8_t"),
    "a8w8":       ("fp8_t", "fp8_t"),
    "a16w16":     ("bf16_t", "bf16_t"),
}

NOSCALE_TAGS = {"a8w8", "a16w16"}


class opus_gemm_codegen:
    def __init__(self, working_path, istune=False):
        self.working_path = working_path
        self.impl_path = os.path.join(working_path, "impl")
        self.instances_path = os.path.join(working_path, "instances")
        self.istune = istune
        assert istune is False, "tuning not yet supported for opus_gemm"

    def gen_instance(self, k: OpusGemmInstance):
        pipeline_header = PIPELINE_HEADER_MAP[k.kernel_tag]
        kernel_func = KERNEL_FUNC_MAP[k.kernel_tag]
        da, db = INPUT_DTYPE_MAP[k.kernel_tag]

        if k.kernel_tag in NOSCALE_TAGS:
            self._gen_noscale_instance(k, pipeline_header, kernel_func, da, db)
        else:
            self._gen_scale_instance(k, pipeline_header, kernel_func, da, db)

    def _gen_scale_instance(self, k, pipeline_header, kernel_func, da, db):
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

    using Traits = opus_gemm_traits<{k.BLOCK_SIZE},
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

    opus_gemm_kargs kargs{{}};
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

    def _gen_noscale_instance(self, k, pipeline_header, kernel_func, da, db):
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

    using Traits = opus_gemm_noscale_traits<{k.BLOCK_SIZE},
        opus::seq<{k.B_M}, {k.B_N}, {k.B_K}>,
        opus::tuple<{da}, {db}, D_C, fp32_t>,
        opus::seq<{k.VEC_A}, {k.VEC_B}, {k.VEC_C}>>;

    opus_gemm_noscale_kargs kargs{{}};
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

    args = parser.parse_args()
    codegen = opus_gemm_codegen(args.working_path, False)
    codegen.gen_instances(kernels_list)
