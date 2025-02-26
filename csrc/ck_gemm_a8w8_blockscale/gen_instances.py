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
from gemm_a8w8_blockscale_common import kernelInstance, kernels_list, default_kernels_dict


""" 

a8w8_blockscale_gemm instance gen 

"""


class gemm_a8w8_blockscale_codegen:
    def __init__(self, working_path, istune=False):
        self.working_path = working_path
        self.impl_path = os.path.join(working_path, "impl")
        self.instances_path = os.path.join(working_path, "instances")
        self.istune = istune
    
    def gen_instance(self, k: kernelInstance):
        INSTANCE_IMPL = f"""// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#include "gemm_a8w8_blockscale_common.cuh"

template <tyename DDataType, typename EDataType=DDataType>
torch::Tensor
{k.name}(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y
    )
{{{{
    // The smallest kernel we have available. Works well for memory bound shapes.

    // Check if this input needs to be padded.
    int M = size_to_dim_(XQ.dim() - 1, XQ.sizes());
    int N = WQ.size(0);
    int K = WQ.size(1);
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

        INSTANCE_CONTENT_nobias = f"""using DeviceGemmInstance = DeviceGemmHelperF8BlockScale<        
            DDataType, EDataType,
            {k.BLOCK_SIZE},
            {k.MPerBLOCK}, {k.NPerBLOCK}, {k.KPerBLOCK},
            {k.AK1}, {k.BK1},
            {k.MPerXDL}, {k.NPerXDL}, 
            {k.MXdlPerWave}, {k.NXdlPerWave}, 
            S<{(", ").join(map(lambda x:str(x),k.ABlockTransferThreadClusterLengths_AK0_M_AK1))}>, 
            S<{(", ").join(map(lambda x:str(x),k.BBlockTransferThreadClusterLengths_BK0_N_BK1))}>,
            S<{(", ").join(map(lambda x:str(x),k.CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock))}>,
            S<{(", ").join(map(lambda x:str(x),k.CDEShuffleBlockTransferScalarPerVectors))}>,
            ck::BlockGemmPipelineScheduler::{k.BlkGemmPipeSched},
            ck::BlockGemmPipelineVersion::v{k.BlkGemmPipelineVer}>;
        // Run kernel instance.
        return gemm_a8w8_blockscale_impl<DDataType, EDataType, DeviceGemmInstance>(XQ, WQ, x_scale, w_scale, Y);
""" 
        if self.istune:
            INSTANCE_IMPL_str = INSTANCE_IMPL.format(INSTANCE_CONTENT_pad=(INSTANCE_CONTENT_nobias.format(GemmSpec="MNKPadding")),
                                                     INSTANCE_CONTENT_nopad=(INSTANCE_CONTENT_nobias.format(GemmSpec="Default")))
        else:
            INSTANCE_IMPL_str = INSTANCE_IMPL.format(INSTANCE_CONTENT_pad=INSTANCE_CONTENT_nobias.format(GemmSpec="MNKPadding"),
                                                     INSTANCE_CONTENT_nopad=INSTANCE_CONTENT_nobias.format(GemmSpec="Default"))

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
    torch::Tensor &Y
    );

"""
        INSTANCE_dBF16_eBF16 = INSTANCE_template.format(
            name=k.name, dtypes="B16")
        INSTANCE_dFP32_eBF16 = INSTANCE_template.format(
            name=k.name, dtypes="F32, B16")
        INSTANCE_dFP16_eFP16 = INSTANCE_template.format(
            name=k.name, dtypes="F16")
        INSTANCE_dFP32_eFP16 = INSTANCE_template.format(
            name=k.name, dtypes="F32, F16")
        # TODO: dFP8_eFP8 

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
       {{{MNK},                                                                                                       \\
        {kernel_name}<DTYPE, ETYPE>}},                       \\"""

        LOOKUP_end = """
   }

#endif // USE_ROCM
"""
        with open(os.path.join(self.working_path, "gemm_a8w8_blockscale_lookup.h"), "w") as f:
            f.write(LOOKUP_head)
            for mnk, k in kernels_dict.items():
                # print((", ").join(map(lambda x: str(x), list(mnk))), ":", k.name)
                if not self.istune and (isinstance(mnk, tuple) and mnk[0] > 0):
                    f.write(LOOKUP_template.format(MNK="{"+(", ").join(
                        map(lambda x: str(x), list(mnk))) + "}", kernel_name=k.name))
                elif self.istune and isinstance(mnk, int):
                    f.write(LOOKUP_template.format(MNK=mnk, kernel_name=k.name))
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
    torch::Tensor &Y);
"""
        MAINFEST_end = """

#endif // USE_ROCM
"""

        with open(os.path.join(self.working_path, "gemm_a8w8_blockscale_manifest.h"), "w") as f:
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
            M = tune_df.loc[i, "M"]
            N = tune_df.loc[i, "N"]
            K = tune_df.loc[i, "K"]
            kid = tune_df.loc[i, "kernelId"]
            tune_dict[(M, N, K)] = kernels_list[kid] 
    return tune_dict

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
        help="the path where all the blobs are going to be generated"
    )

    parser.add_argument(
        "-f",
        "--tune_file",
        default="aiter/configs/a8w8_tuned_gemm.csv",
        required=False,
        help="tune_file include the result after run gemm_a8w8_tune.py"
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
    codegen = gemm_a8w8_blockscale_codegen(args.working_path, args.tune)
    
    if args.tune:
        codegen.gen_instances(kernels_list)
    else:
        codegen.gen_instances(get_tune_dict(args.tune_file))
