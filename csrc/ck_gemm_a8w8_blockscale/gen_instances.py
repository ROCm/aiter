# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
import argparse
import os
import shutil
from pathlib import Path

import pandas as pd
import torch

from gemm_a8w8_blockscale_instance import (
    default_kernels_dict,
    KernelInstance,
    candidate_kernels_dict,
)

"""
a8w8_blockscale_gemm instance gen for CK Legacy
"""


class gemm_a8w8_blockscale_codegen:
    def __init__(self, working_path: str, istune=False, tune_file=None):
        self.working_path = working_path
        if not os.path.exists(working_path):
            os.makedirs(working_path)

        self.impl_path = os.path.join(working_path, "impl")
        self.instances_path = os.path.join(working_path, "instances")
        self.istune = istune
        self.tune_file = tune_file

    def get_tune_dict(self, tune_dict_csv: str):
        """
        Get tune dict from csv file
        """

        tune_dict = default_kernels_dict

        if os.path.exists(tune_dict_csv):
            tune_df = pd.read_csv(tune_dict_csv)
            if torch.cuda.is_available():
                gpu = torch.cuda.current_device()
                device_properties = torch.cuda.get_device_properties(gpu)
                cu_num = device_properties.multi_processor_count
                tune_df = tune_df[
                    (tune_df["cu_num"] == cu_num) & (tune_df["libtype"] == "ck")
                ].reset_index()

            for i in range(len(tune_df)):
                M = int(tune_df.loc[i, "M"])
                N = int(tune_df.loc[i, "N"])
                K = int(tune_df.loc[i, "K"])
                kid = int(tune_df.loc[i, "kernelId"])

                if kid in candidate_kernels_dict:
                    tune_dict[(M, N, K)] = candidate_kernels_dict[kid]
                else:
                    print(
                        f"Warning: kernelId {kid} not found in candidate_kernels_dict for shape ({M}, {N}, {K})"
                    )

        return tune_dict

    def gen_legacy_instance(self, k: KernelInstance):
        """
        Generate kernel instance code for legacy gemm a8w8 blockscale
        """

        LEGACY_INSTANCE_IMPL = f"""// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#include "gemm_a8w8_blockscale_common.cuh"

enum class GemmSpecialization {{
    Default    = 0,
    MPadding   = 1,
    NPadding   = 2,
    KPadding   = 3,
    MNPadding  = 4,
    MKPadding  = 5,
    NKPadding  = 6,
    MNKPadding = 7
}};

static const std::unordered_map<std::string, GemmSpecialization> g_gemm_spec_names{{
    {{"", GemmSpecialization::Default}},
    {{"M", GemmSpecialization::MPadding}},
    {{"N", GemmSpecialization::NPadding}},
    {{"K", GemmSpecialization::KPadding}},
    {{"MN", GemmSpecialization::MNPadding}},
    {{"MK", GemmSpecialization::MKPadding}},
    {{"NK", GemmSpecialization::NKPadding}},
    {{"MNK", GemmSpecialization::MNKPadding}}
}};

static GemmSpecialization GetGemmSpec(const int64_t m,
                               const int64_t n,
                               const int64_t k,
                               const int64_t m_per_block,
                               const int64_t n_per_block,
                               const int64_t k_per_block)
{{
    auto IntegerDivideCeil = [](int x, int y) {{
        return (x + y - size_t{{1}}) / y;
    }};

    std::string spec = "";
    if (IntegerDivideCeil(m, m_per_block) * m_per_block - m != 0)
        spec += "M";
    if (IntegerDivideCeil(n, n_per_block) * n_per_block - n != 0)
        spec += "N";
    if (IntegerDivideCeil(k, k_per_block) * k_per_block - k != 0)
        spec += "K";

    return g_gemm_spec_names.at(spec);
}}

template <typename DDataType, typename EDataType>
torch::Tensor
{k.name}(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y
    )
{{
    // Get M, N, K from input tensors.
    int M = XQ.numel() / XQ.size(-1);
    int N = WQ.size(0);
    int K = WQ.size(1);

    // Get whether this input needs to be padded.
    auto gemm_spec = GetGemmSpec(M, N, K, {k.MPerBLOCK}, {k.NPerBLOCK}, {k.KPerBLOCK});


    if(gemm_spec == GemmSpecialization::Default)
    {{
        // Default
        __INSTANCE_CONTENT_DEFAULT__
    }} else if(gemm_spec == GemmSpecialization::MPadding)
    {{
        // MNK Padding
        __INSTANCE_CONTENT_MPAD__
    }} else if(gemm_spec == GemmSpecialization::NPadding)
    {{
        // N Padding
        __INSTANCE_CONTENT_NPAD__
    }} else if(gemm_spec == GemmSpecialization::KPadding)
    {{
        // K Padding
        __INSTANCE_CONTENT_KPAD__
    }} else if(gemm_spec == GemmSpecialization::MNPadding)
    {{
        // MN Padding
        __INSTANCE_CONTENT_MNPAD__
    }} else if(gemm_spec == GemmSpecialization::MKPadding)
    {{
        // MK Padding
        __INSTANCE_CONTENT_MKPAD__
    }} else if(gemm_spec == GemmSpecialization::NKPadding)
    {{
        // NK Padding
        __INSTANCE_CONTENT_NKPAD__
    }} else if(gemm_spec == GemmSpecialization::MNKPadding)
    {{
        // MNK Padding
        __INSTANCE_CONTENT_MNKPAD__
    }} else
    {{
        throw std::runtime_error("Unsupported GemmSpecialization!");
    }}
}}

"""

        LEGACY_INSTANCE = f"""using LegacyGemmInstance = DeviceLegacyGemmHelperF8BlockScale<
            DDataType, EDataType,
            {k.BLOCK_SIZE},
            {k.ScaleBlockM}, {k.ScaleBlockN}, {k.ScaleBlockK},
            {k.MPerBLOCK}, {k.NPerBLOCK}, {k.KPerBLOCK},
            {k.AK1}, {k.BK1},
            {k.MPerXDL}, {k.NPerXDL},
            {k.WAVE_MAP_M}, {k.WAVE_MAP_N},
            S<{(", ").join(map(lambda x:str(x),k.ABLOCK_TRANSFER))}>,
            S<{(", ").join(map(lambda x:str(x),k.BBLOCK_TRANSFER))}>,
            {k.CSHUFFLE_MX_PER_WAVE_PERSHUFFLE},
            {k.CSHUFFLE_NX_PER_WAVE_PERSHUFFLE},
            S<{(", ").join(map(lambda x:str(x),k.CBLOCK_TRANSFER))}>,
            S<{(", ").join(map(lambda x:str(x),k.CBLOCK_SPV))}>,
            ck::BlockGemmPipelineScheduler::{k.PIPELINE_Sched},
            ck::BlockGemmPipelineVersion::v{k.PIPELINE_VERSION},
            ck::tensor_operation::device::GemmSpecialization::{{GemmSpec}}>;

        // Run kernel instance.
        return gemm_a8w8_blockscale_impl<DDataType, EDataType, LegacyGemmInstance>(XQ, WQ, x_scale, w_scale, Y);
"""
        INSTANCE_IMPL_str = (
            LEGACY_INSTANCE_IMPL.replace(
                "__INSTANCE_CONTENT_DEFAULT__",
                LEGACY_INSTANCE.replace("{GemmSpec}", "Default"),
            )
            .replace(
                "__INSTANCE_CONTENT_MPAD__",
                LEGACY_INSTANCE.replace("{GemmSpec}", "MPadding"),
            )
            .replace(
                "__INSTANCE_CONTENT_NPAD__",
                LEGACY_INSTANCE.replace("{GemmSpec}", "NPadding"),
            )
            .replace(
                "__INSTANCE_CONTENT_KPAD__",
                LEGACY_INSTANCE.replace("{GemmSpec}", "KPadding"),
            )
            .replace(
                "__INSTANCE_CONTENT_MNPAD__",
                LEGACY_INSTANCE.replace("{GemmSpec}", "MNPadding"),
            )
            .replace(
                "__INSTANCE_CONTENT_MKPAD__",
                LEGACY_INSTANCE.replace("{GemmSpec}", "MKPadding"),
            )
            .replace(
                "__INSTANCE_CONTENT_NKPAD__",
                LEGACY_INSTANCE.replace("{GemmSpec}", "NKPadding"),
            )
            .replace(
                "__INSTANCE_CONTENT_MNKPAD__",
                LEGACY_INSTANCE.replace("{GemmSpec}", "MNKPadding"),
            )
        )

        Path(os.path.join(self.impl_path, f"{k.name}.cuh")).write_text(
            INSTANCE_IMPL_str
        )

        INSTANCE_template = """// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#include "impl/{name}.cuh"

template torch::Tensor
{name}<{dtypes}>(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y
    );

"""
        INSTANCE_dFP32_eBF16 = INSTANCE_template.format(
            name=k.name, dtypes="FP32, BF16"
        )
        INSTANCE_dFP32_eFP16 = INSTANCE_template.format(
            name=k.name, dtypes="FP32, FP16"
        )
        # TODO: dFP8_eFP8

        Path(os.path.join(self.instances_path, f"{k.name}_dFP32_eBF16.cpp")).write_text(
            INSTANCE_dFP32_eBF16
        )
        Path(os.path.join(self.instances_path, f"{k.name}_dFP32_eFP16.cpp")).write_text(
            INSTANCE_dFP32_eFP16
        )

    def gen_lookup_dict(self, kernels_dict):
        """
        Generate lookup dictionary for kernel instances
        """

        LOOKUP_head = """#pragma once
// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#ifdef USE_ROCM

#define GENERATE_LOOKUP_TABLE(DTYPE, ETYPE)                                                                                      \\
   {                                                                                                                             \\"""

        LOOKUP_template = """
       {{{MNK},                                                                                                       \\
        {kernel_name}<DTYPE, ETYPE>}},                       \\"""

        LOOKUP_end = """
   }

#endif // USE_ROCM
"""
        with open(
            os.path.join(self.working_path, "gemm_a8w8_blockscale_lookup.h"), "w"
        ) as f:
            f.write(LOOKUP_head)
            for mnk, k in kernels_dict.items():
                # print((", ").join(map(lambda x: str(x), list(mnk))), ":", k.name)
                if not self.istune and (isinstance(mnk, tuple) and mnk[0] > 0):
                    f.write(
                        LOOKUP_template.format(
                            MNK="{"
                            + (", ").join(map(lambda x: str(x), list(mnk)))
                            + "}",
                            kernel_name=k.name,
                        )
                    )
                elif self.istune and isinstance(mnk, int):
                    f.write(LOOKUP_template.format(MNK=mnk, kernel_name=k.name))
            f.write(LOOKUP_end)

    def gen_manifest_head(self, kernels_dict):
        """
        Generate manifest header for kernel instances, declaring all the kernel APIs
        """

        MAINFEST_head = """#pragma once
// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#ifdef USE_ROCM

#include <cstdlib>

#include <torch/extension.h>
"""
        MAINFEST_template = """
template <typename DDataType, typename EDataType>
torch::Tensor
{kernel_name}(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);
"""
        MAINFEST_end = """

#endif // USE_ROCM
"""

        with open(
            os.path.join(self.working_path, "gemm_a8w8_blockscale_manifest.h"),
            "w",
        ) as f:
            f.write(MAINFEST_head)
            for _, k in kernels_dict.items():
                f.write(MAINFEST_template.format(kernel_name=k.name))
            f.write(MAINFEST_end)

    def gen_code(self, kernels_dict: dict):
        """
        Codegen for legacy gemm a8w8 blockscale
        """

        # generate instances code
        for _, k in kernels_dict.items():
            self.gen_legacy_instance(k)

        # generate lookup dict for kernel instances
        self.gen_lookup_dict(kernels_dict)

        # generate manifest header for kernel instances
        self.gen_manifest_head(kernels_dict)

    def run(self):
        """
        Run codegen and generate all the files together
        """

        # clean impl and instances path
        if os.path.exists(self.impl_path):
            shutil.rmtree(self.impl_path)
        os.mkdir(self.impl_path)
        if os.path.exists(self.instances_path):
            shutil.rmtree(self.instances_path)
        os.mkdir(self.instances_path)

        # generate code for legacy and tile
        if self.istune:
            # generate code for default kernels
            self.gen_code(candidate_kernels_dict)
        else:
            # generate code for tuned kernels from tune_file
            self.gen_code(self.get_tune_dict(self.tune_file))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="generate",
        description="gen API for CK gemm a8w8 kernel",
    )

    # the directory for list_blobs/gen_blobs to write files into
    parser.add_argument(
        "-w",
        "--working_path",
        default="./",
        required=False,
        help="the path where all the blobs are going to be generated",
    )

    # the tune file including the best kernel instance
    parser.add_argument(
        "-f",
        "--tune_file",
        default="aiter/configs/a8w8_blockscale_tuned_gemm.csv",
        required=False,
        help="tune_file include the result after run gemm_a8w8_tune.py",
    )

    # whether to generate tune instances
    parser.add_argument(
        "--tune", action="store_true", required=False, help="generated tune instances"
    )

    args = parser.parse_args()
    codegen = gemm_a8w8_blockscale_codegen(args.working_path, args.tune, args.tune_file)
    codegen.run()
