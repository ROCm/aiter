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
from batched_gemm_bf16_common import kernelInstance, kernels_list, default_kernels_dict



class batched_gemm_bf16_fwd_codegen:
    def __init__(self, working_path, istune=False):
        self.working_path = working_path
        self.impl_path = os.path.join(working_path, "impl")
        self.instances_path = os.path.join(working_path, "instances")
        self.istune = istune

    def gen_instance(self, k: kernelInstance):
        INSTANCE_IMPL = f"""// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#include "batched_gemm_bf16_common.cuh"

torch::Tensor
{k.name}(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &Y,
    std::optional<torch::Tensor> bias,
    int KBatch)
{{{{
    // The smallest kernel we have available. Works well for memory bound shapes.

#if 0
    // Check if this input needs to be padded.
    int B = XQ.size(0);
    int M = XQ.size(1);
    int N = WQ.size(1);
    int K = WQ.size(2);
    bool pad = (M % {k.MPerBLOCK} != 0) || (N % {k.NPerBLOCK} != 0) || (K % ({k.KPerBLOCK} * KBatch) != 0);
#else
    // Disable padding for packed tensor
    bool pad = false;
#endif
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
            return batched_gemm_bf16_impl<DeviceGemmInstance>(XQ, WQ, Y, bias, KBatch);
        }}}}
"""
        INSTANCE_CONTENT_nobias = f"""using DeviceGemmInstance = DeviceGemmHelper<
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
        return batched_gemm_bf16_impl<DeviceGemmInstance>(XQ, WQ, Y, bias, KBatch);
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

torch::Tensor
{name}(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &Y,
    std::optional<torch::Tensor> bias,
    int KBatch);

"""
        INSTANCE = INSTANCE_template.format(name=k.name)

        if self.istune:
            Path(os.path.join(self.instances_path, f"{k.name}.cpp")).write_text(INSTANCE)
        else:
            Path(os.path.join(self.instances_path, f"{k.name}.cpp")).write_text(INSTANCE)

    def gen_lookup_dict(self, kernels_dict):
        LOOKUP_head = """#pragma once
// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#ifdef USE_ROCM

#define GENERATE_LOOKUP_TABLE()                                                                                      \\
   {                                                                                                                             \\"""

        LOOKUP_template = """
       {{{mnk},                                                                                                       \\
        {kernel_name}}},                       \\"""

        LOOKUP_end = """
   }

#endif // USE_ROCM
"""
        with open(os.path.join(self.working_path, "batched_gemm_bf16_lookup.h"), "w") as f:
            f.write(LOOKUP_head)
            for mnk, k in kernels_dict.items():
                #print((", ").join(map(lambda x: str(x), list(mnk))), ":", k.name)
                if not self.istune and (isinstance(mnk, tuple) and mnk[0] > 0):
                    f.write(LOOKUP_template.format(mnk="{"+(", ").join(
                        map(lambda x: str(x), list(mnk))) + "}", kernel_name=k.name))
                elif self.istune and isinstance(mnk, int):
                    f.write(LOOKUP_template.format(mnk=mnk, kernel_name=k.name))
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
torch::Tensor
{kernel_name}(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &Y,
    std::optional<torch::Tensor> bias,
    int KBatch);
"""
        MAINFEST_end = """

#endif // USE_ROCM
"""

        with open(os.path.join(self.working_path, "batched_gemm_bf16_manifest.h"), "w") as f:
            f.write(MAINFEST_head)
            for mnk, k in kernels_dict.items():
                f.write(MAINFEST_template.format(kernel_name=k.name))
            f.write(MAINFEST_end)

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
        description="gen API for CK batched gemm bf16 kernel",
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
        default="aiter/configs/bf16_tuned_batched_gemm.csv",
        required=False,
        help="tune_file include the result after run batched_gemm_bf16_tune.py"
    )

    parser.add_argument(
        "--tune",
        action='store_true',
        required=False,
        help="generated tune instanses"
    )

    args = parser.parse_args()
    codegen = batched_gemm_bf16_fwd_codegen(args.working_path, args.tune)

    if args.tune:
        codegen.gen_instances(kernels_list)
    else:
        codegen.gen_instances(get_tune_dict(args.tune_file))
