# SPDX-License-Identifier: MIT
# Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
import os
import sys
from dataclasses import dataclass
import copy
from pathlib import Path
import pandas as pd
import argparse
import shutil
from batched_gemm_a8w8_common import kernelInstance, kernels_list, default_kernels_dict



class batched_gemm_a8w8_fwd_codegen:
    def __init__(self, working_path, istune=False):
        self.working_path = working_path
        self.impl_path = os.path.join(working_path, "impl")
        self.instances_path = os.path.join(working_path, "instances")
        self.istune = istune

    def gen_instance(self, k: kernelInstance):
        INSTANCE_IMPL = f"""// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#include "batched_gemm_a8w8_common.cuh"

template <typename DDataType, typename EDataType = DDataType>
torch::Tensor
{k.name}(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y,
    std::optional<torch::Tensor> bias,
    int KBatch)
{{{{
    // The smallest kernel we have available. Works well for memory bound shapes.

    // Check if this input needs to be padded.
    int B = XQ.size(0);
    int M = XQ.size(1);
    int N = WQ.size(1);
    int K = WQ.size(2);
    bool pad = (M % {k.MPerBLOCK} != 0) || (N % {k.NPerBLOCK} != 0) || (K % ({k.KPerBLOCK} * KBatch) != 0);
    if (pad)
    {{{{
        // pad
        {{INSTANCE_CONTENT_pad}}
        // pad
    }}}}
    else
    {{{{
        // no pad
        {{INSTANCE_CONTENT_nopad}}
        // no pad
    }}}}
}}}}

"""
        INSTANCE_CONTENT_bias = f"""
        {{{{
           using DeviceGemmInstance = DeviceGemmHelper<
                DDataType, EDataType,
                {k.BLOCK_SIZE},
                {k.MPerBLOCK},
                {k.NPerBLOCK},
                {k.KPerBLOCK},
                {k.WAVE_TILE_M},
                {k.WAVE_TILE_N},
                {k.WAVE_MAP_M},
                {k.WAVE_MAP_N},
                S<{(", ").join(map(lambda x:str(x),k.ABLOCK_TRANSFER))}>,
                S<{(", ").join(map(lambda x:str(x),k.BBLOCK_TRANSFER))}>,
                S<{(", ").join(map(lambda x:str(x),k.CBLOCK_TRANSFER))}>,
                S<{(", ").join(map(lambda x:str(x),k.CBLOCK_SPV))}>,
                {k.CSHUFFLE_MX_PER_WAVE_PERSHUFFLE},
                {k.CSHUFFLE_NX_PER_WAVE_PERSHUFFLE},
                ck::BlockGemmPipelineScheduler::{k.LOOP_SCHED},
                ck::BlockGemmPipelineVersion::v{k.PIPELINE_VERSION},
                ck::tensor_operation::device::GemmSpecialization::{{GemmSpec}}>;
            // Run kernel instance.
            return batched_gemm_a8w8_rowwise_impl<DDataType, EDataType, DeviceGemmInstance>(XQ, WQ, x_scale, w_scale, Y, bias, KBatch);
        }}}}
"""
        INSTANCE_CONTENT_nobias = f"""using DeviceGemmInstance = DeviceGemmHelper<
            DDataType, EDataType,
            {k.BLOCK_SIZE},
            {k.MPerBLOCK},
            {k.NPerBLOCK},
            {k.KPerBLOCK},
            {k.WAVE_TILE_M},
            {k.WAVE_TILE_N},
            {k.WAVE_MAP_M},
            {k.WAVE_MAP_N},
            S<{(", ").join(map(lambda x:str(x),k.ABLOCK_TRANSFER))}>,
            S<{(", ").join(map(lambda x:str(x),k.BBLOCK_TRANSFER))}>,
            S<{(", ").join(map(lambda x:str(x),k.CBLOCK_TRANSFER))}>,
            S<{(", ").join(map(lambda x:str(x),k.CBLOCK_SPV))}>,
            {k.CSHUFFLE_MX_PER_WAVE_PERSHUFFLE},
            {k.CSHUFFLE_NX_PER_WAVE_PERSHUFFLE},
            ck::BlockGemmPipelineScheduler::{k.LOOP_SCHED},
            ck::BlockGemmPipelineVersion::v{k.PIPELINE_VERSION},
            ck::tensor_operation::device::GemmSpecialization::{{GemmSpec}}>;
        // Run kernel instance.
        return batched_gemm_a8w8_rowwise_impl<DDataType, EDataType, DeviceGemmInstance>(XQ, WQ, x_scale, w_scale, Y, bias, KBatch);
"""
        if self.istune:
            INSTANCE_IMPL_str = INSTANCE_IMPL.format(INSTANCE_CONTENT_pad=(INSTANCE_CONTENT_nobias.format(GemmSpec="MNKPadding")),
                                                     INSTANCE_CONTENT_nopad=(INSTANCE_CONTENT_nobias.format(GemmSpec="Default")))
        else:
            INSTANCE_IMPL_str = INSTANCE_IMPL.format(INSTANCE_CONTENT_pad=INSTANCE_CONTENT_bias.format(GemmSpec="MNKPadding"),
                                                     INSTANCE_CONTENT_nopad=INSTANCE_CONTENT_bias.format(GemmSpec="Default"))

        Path(os.path.join(self.impl_path, f"{k.name}.cuh")).write_text(
                INSTANCE_IMPL_str)

        INSTANCE_template = """// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#include "{name}.cuh"

template torch::Tensor
{name}<{dtypes}>(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y,
    std::optional<torch::Tensor> bias,
    int KBatch);

"""
        INSTANCE_dBF16_eBF16 = INSTANCE_template.format(
            name=k.name, dtypes="B16")
        INSTANCE_dFP32_eBF16 = INSTANCE_template.format(
            name=k.name, dtypes="F32, B16")
        INSTANCE_dFP16_eFP16 = INSTANCE_template.format(
            name=k.name, dtypes="F16")
        INSTANCE_dFP32_eFP16 = INSTANCE_template.format(
            name=k.name, dtypes="F32, F16")

        if self.istune:
            Path(os.path.join(self.instances_path, f"{k.name}_dBF16_eBF16.cpp")).write_text(
                INSTANCE_dBF16_eBF16)
        else:
            Path(os.path.join(self.instances_path, f"{k.name}_dBF16_eBF16.cpp")).write_text(
                INSTANCE_dBF16_eBF16)
            Path(os.path.join(self.instances_path, f"{k.name}_dFP32_eBF16.cpp")).write_text(
                INSTANCE_dFP32_eBF16)
            Path(os.path.join(self.instances_path, f"{k.name}_dFP16_eFP16.cpp")).write_text(
                INSTANCE_dFP16_eFP16)
            Path(os.path.join(self.instances_path, f"{k.name}_dFP32_eFP16.cpp")).write_text(
                INSTANCE_dFP32_eFP16)

    def gen_lookup_dict(self, kernels_dict):
        LOOKUP_head = """#pragma once
// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#ifdef USE_ROCM

#define GENERATE_LOOKUP_TABLE(DTYPE, ETYPE)                                                                                      \\
   {                                                                                                                             \\"""

        LOOKUP_template = """
       {{{bmnk},                                                                                                       \\
        {kernel_name}<DTYPE, ETYPE>}},                       \\"""

        LOOKUP_end = """
   }

#endif // USE_ROCM
"""
        with open(os.path.join(self.working_path, "batched_gemm_a8w8_lookup.h"), "w") as f:
            f.write(LOOKUP_head)
            for bmnk, k in kernels_dict.items():
                # print((", ").join(map(lambda x: str(x), list(bmnk))), ":", k.name)
                if not self.istune and (isinstance(bmnk, tuple) and bmnk[0] > 0):
                    f.write(LOOKUP_template.format(bmnk="{"+(", ").join(
                        map(lambda x: str(x), list(bmnk))) + "}", kernel_name=k.name))
                elif self.istune and isinstance(bmnk, int):
                    f.write(LOOKUP_template.format(bmnk=bmnk, kernel_name=k.name))
            f.write(LOOKUP_end)

    def gen_manifest_head(self, kernels_dict):
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
    torch::Tensor &Y,
    std::optional<torch::Tensor> bias,
    int KBatch);
"""
        MAINFEST_end = """

#endif // USE_ROCM
"""

        with open(os.path.join(self.working_path, "batched_gemm_a8w8_manifest.h"), "w") as f:
            f.write(MAINFEST_head)
            for bmnk, k in kernels_dict.items():
                f.write(MAINFEST_template.format(kernel_name=k.name))
            f.write(MAINFEST_end)

    def gen_instances(self, kernels_dict):
        if os.path.exists(self.impl_path):
            shutil.rmtree(self.impl_path)
        os.mkdir(self.impl_path)
        if os.path.exists(self.instances_path):
            shutil.rmtree(self.instances_path)
        os.mkdir(self.instances_path)

        for bmnk, k in kernels_dict.items():
            self.gen_instance(k)

        self.gen_lookup_dict(kernels_dict)
        self.gen_manifest_head(kernels_dict)


def get_tune_dict(tune_dict_csv):
    tune_dict = default_kernels_dict
    if os.path.exists(tune_dict_csv):
        tune_df = pd.read_csv(tune_dict_csv)
        for i in range(len(tune_df)):
            B = tune_df.loc[i, "B"]
            M = tune_df.loc[i, "M"]
            N = tune_df.loc[i, "N"]
            K = tune_df.loc[i, "K"]
            kid = tune_df.loc[i, "kernelId"]
            tune_dict[(B, M, N, K)] = kernels_list[kid]
    return tune_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="generate",
        description="gen API for CK batched gemm a8w8 kernel",
    )

    # the directory for list_blobs/gen_blobs to write files into
    parser.add_argument(
        "-w",
        "--working_path",
        default="./",
        required=False,
        help="the path where all the blobs are going to be generated"
    )

    parser.add_argument(
        "-f",
        "--tune_file",
        default="aiter/configs/a8w8_tuned_batched_gemm.csv",
        required=False,
        help="tune_file include the result after run batched_gemm_a8w8_tune.py"
    )

    parser.add_argument(
        "--tune",
        action='store_true',
        required=False,
        help="generated tune instanses"
    )

    # parser.add_argument(
    #     "--out_type",
    #     default="all",
    #     required=False,
    #     help="Specifie the type of scale\n \
    #         all: [bf16, fp16] \n  \
    #         bf16, fp16"
    # )

    # parser.add_argument(
    #     "--scale_type",
    #     default="all",
    #     required=False,
    #     help="Specifie the type of scale\n \
    #         all: [fp32, same as out] \n  \
    #         same: [same as out]"
    # )


    args = parser.parse_args()
    codegen = batched_gemm_a8w8_fwd_codegen(args.working_path, args.tune)

    if args.tune:
        codegen.gen_instances(kernels_list)
    else:
        codegen.gen_instances(get_tune_dict(args.tune_file))
