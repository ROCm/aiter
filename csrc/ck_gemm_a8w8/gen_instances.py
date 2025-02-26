# SPDX-License-Identifier: MIT
# Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
import os
from pathlib import Path
import pandas as pd
import argparse
from gemm_a8w8_common import KernelParameters, kernels_params_dict, default_kernels_dict

TEMPLATE_PARAMS = "template_params"
INSTANCE_FILE_SUFFIX = "instance_file_suffix"
LOOK_UP_TABLE_HEADER_PATH = "gemm_a8w8_lookup.h"
MANIFEST_HEADER_PATH = "gemm_a8w8_manifest.h"

PARAM_INSTANCE_SUFFIX_LIST= [
    {
        TEMPLATE_PARAMS: "I8, I8, I32, I8, I32, B16, B16",
        INSTANCE_FILE_SUFFIX: "I8_I8_I32_I8_I32_BF16_BF16.cpp"
    },
    {
        TEMPLATE_PARAMS: "I8, I8, I32, I8, I32, F32, B16",
        INSTANCE_FILE_SUFFIX: "I8_I8_I32_I8_I32_F32_BF16.cpp"
    },
    {
        TEMPLATE_PARAMS: "I8, I8, I32, I8, I32, F16, F16",
        INSTANCE_FILE_SUFFIX: "I8_I8_I32_I8_I32_F16_F16.cpp"
    },
    {
        TEMPLATE_PARAMS: "I8, I8, I32, I8, I32, F32, F16",
        INSTANCE_FILE_SUFFIX: "I8_I8_I32_I8_I32_F32_F16.cpp"
    },
    {
        TEMPLATE_PARAMS: "FP8, FP8, F32, FP8, F32, B16, B16",
        INSTANCE_FILE_SUFFIX: "FP8_FP8_F32_FP8_F32_B16_B16.cpp"
    },
    {
        TEMPLATE_PARAMS: "FP8, FP8, F32, FP8, F32, F32, B16",
        INSTANCE_FILE_SUFFIX: "FP8_FP8_F32_FP8_F32_F32_B16.cpp"
    },
    {
        TEMPLATE_PARAMS: "FP8, FP8, F32, FP8, F32, F16, F16",
        INSTANCE_FILE_SUFFIX: "FP8_FP8_F32_FP8_F32_F16_F16.cpp"
    },
    {
        TEMPLATE_PARAMS: "FP8, FP8, F32, FP8, F32, F32, F16",
        INSTANCE_FILE_SUFFIX: "FP8_FP8_F32_FP8_F32_F32_F16.cpp"
    }
]

TUNING_PARAM_INSTANCE_SUFFIX_LIST = [
    {
        TEMPLATE_PARAMS: "I8, I8, I32, I8, I32, B16, B16",
        INSTANCE_FILE_SUFFIX: "I8_I8_I32_I8_I32_BF16_BF16.cpp"
    },
    {
        TEMPLATE_PARAMS: "FP8, FP8, F32, FP8, F32, B16, B16",
        INSTANCE_FILE_SUFFIX: "FP8_FP8_F32_FP8_F32_B16_B16.cpp"
    },
]

def gen_a8w8_device_gemm_call_skip_bias_branch(k: KernelParameters, gemm_specialization: str) -> str:
    return f"""using DeviceGemmInstance = DeviceGemmHelper<
        ADataType,
        BDataType,
        AccDataType,
        CShuffleDataType,
        ComputeDataType,
        DDataType,
        EDataType,
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
        ck::tensor_operation::device::GemmSpecialization::{gemm_specialization}>;
        
        return gemm_a8w8_rowwise_impl<
                ADataType,
                BDataType,
                AccDataType,
                DDataType,
                EDataType,
                DeviceGemmInstance
            >(XQ, WQ, x_scale, w_scale, Y, bias, KBatch);
"""

def gen_a8w8_device_gemm_call(k: KernelParameters, gemm_specialization: str):
    return f"""if (bias != std::nullopt)
    {{
        using DeviceGemmInstance = DeviceGemmHelperMMA<
        ADataType,
        BDataType,
        AccDataType,
        CShuffleDataType,
        ComputeDataType,
        DDataType,
        EDataType,
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
        S<{(", ").join(map(lambda x:str(x),k.CBLOCK_SPV))}, {k.CBLOCK_SPV[0]}>,
        {k.CSHUFFLE_MX_PER_WAVE_PERSHUFFLE},
        {k.CSHUFFLE_NX_PER_WAVE_PERSHUFFLE},
        ck::BlockGemmPipelineScheduler::{k.LOOP_SCHED},
        ck::BlockGemmPipelineVersion::v{k.PIPELINE_VERSION},
        ck::tensor_operation::device::GemmSpecialization::{gemm_specialization}>;
        // Run kernel instance.
        
        return gemm_a8w8_mma_impl<
                ADataType,
                BDataType,
                DDataType,
                EDataType,
                DeviceGemmInstance
            >(XQ, WQ, x_scale, w_scale, Y, bias, KBatch);
    }}
    else
    {{
        using DeviceGemmInstance = DeviceGemmHelper<
        ADataType,
        BDataType,
        AccDataType,
        CShuffleDataType,
        ComputeDataType,
        DDataType,
        EDataType,
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
        ck::tensor_operation::device::GemmSpecialization::{gemm_specialization}>;
        
        return gemm_a8w8_rowwise_impl<
                ADataType,
                BDataType,
                AccDataType,
                DDataType,
                EDataType,
                DeviceGemmInstance
            >(XQ, WQ, x_scale, w_scale, Y, bias, KBatch);
    }}

"""

def gen_a8w8_implementation(k: KernelParameters, skip_bias_branch: bool) -> str:
    if skip_bias_branch:
        gemm_a8w8_device_gemm_instance_generator = gen_a8w8_device_gemm_call_skip_bias_branch
    else:
        gemm_a8w8_device_gemm_instance_generator = gen_a8w8_device_gemm_call

    padding = "MNKPadding"
    no_padding = "Default"
    return f"""
// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#include "gemm_a8w8_common.cuh"

template <
    typename ADataType,
    typename BDataType,
    typename CShuffleDataType,
    typename ComputeDataType,
    typename AccDataType,
    typename DDataType,
    typename EDataType
    >
torch::Tensor
{k.name}(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y,
    std::optional<torch::Tensor> bias,
    int KBatch)
{{
    // The smallest kernel we have available. Works well for memory bound shapes.

    // Check if this input needs to be padded.
    int M = size_to_dim_(XQ.dim() - 1, XQ.sizes());
    int N = WQ.size(0);
    int K = WQ.size(1);

    bool pad = (M % {k.MPerBLOCK} != 0) || (N % {k.NPerBLOCK} != 0) || (K % ({k.KPerBLOCK} * KBatch) != 0);
    if (pad) {{
        {gemm_a8w8_device_gemm_instance_generator(k, padding)}
    }}
    else{{
        {gemm_a8w8_device_gemm_instance_generator(k, no_padding)}
    }}
}}
"""

def gen_a8w8_instance(k: KernelParameters, template_params: str) -> str:
    return f"""// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#include "{k.name}.cuh"

template torch::Tensor
{k.name}<{template_params}>(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y,
    std::optional<torch::Tensor> bias,
    int KBatch);
"""

def gen_kernel_dict_item_as_str(mnk: tuple | int, k: KernelParameters) -> str:
    if isinstance(mnk, tuple):
        mnk_formatted = ', '.join(str(item) for item in mnk)
    else:
        mnk_formatted = f"{str(mnk)}"
    return f"{{ {{{mnk_formatted}}}, {k.name}<A_TYPE, B_TYPE, C_SHUFFLE_TYPE, COMPUTE_TYPE, ACC_TYPE, D_TYPE, E_TYPE>}}"

def gen_lookup_dict(kernel_dict: dict, is_tune: bool) -> str:
    # Do not include default kernels in the lookup table for non-tuning calls.
    filter_mnk = lambda mnk: True if is_tune else isinstance(mnk, tuple)
    kernel_dict_items = [
        gen_kernel_dict_item_as_str(mnk, k)
        for mnk, k in kernel_dict.items()
        if filter_mnk(mnk)
    ]
    
    lookup_table = ",\\\n    ".join(kernel_dict_items)
    
    return f"""#pragma once
// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#ifdef USE_ROCM
#define GENERATE_LOOKUP_TABLE(A_TYPE, B_TYPE, C_SHUFFLE_TYPE, COMPUTE_TYPE, ACC_TYPE, D_TYPE, E_TYPE)      \\
{{                                                                                                         \\
    {lookup_table} \\
}}
#endif
"""

def gen_kernel_definition(kernel_name: str) -> str:
    return f"""
template <
    typename ADataType,
    typename BDataType,
    typename CShuffleDataType,
    typename ComputeDataType,
    typename AccDataType,
    typename DDataType,
    typename EDataType
>
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

def gen_manifest(kernels_dict: dict) -> str:
    kernel_definition_list = [
        gen_kernel_definition(k.name) for k in kernels_dict.values()   
    ]
    kernel_definitions = "\n".join(kernel_definition_list)
    
    return f"""#pragma once
// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#ifdef USE_ROCM

#include <cstdlib>

#include <torch/extension.h>
{kernel_definitions}
#endif
"""

def gemm_a8w8_fwd_codegen(working_path: Path, kernel_parameters_dict: dict, is_tune: bool):
    impl_directory = Path(os.path.join(working_path, "impl"))
    instance_directory = Path(os.path.join(working_path, "instances"))
    impl_directory.mkdir(exist_ok=True)
    instance_directory.mkdir(exist_ok=True)
    # generate and write the implementation files
    for _, kernel_parameters in kernel_parameters_dict.items():
        impl_path =  Path(os.path.join(impl_directory, f"{kernel_parameters.name}.cuh"))
        kernels_impl_str = gen_a8w8_implementation(kernel_parameters, is_tune)
        impl_path.write_text(kernels_impl_str)


    # generate and write the implementation files for each supported specialization
    for _, kernel_parameters in kernel_parameters_dict.items():
        param_instance_list = TUNING_PARAM_INSTANCE_SUFFIX_LIST if is_tune else PARAM_INSTANCE_SUFFIX_LIST
        for param_instance in param_instance_list:
            template_params = param_instance[TEMPLATE_PARAMS]
            instance_file_suffix = param_instance[INSTANCE_FILE_SUFFIX]
            instance_file_name = f"{kernel_parameters.name}_{instance_file_suffix.lower()}"
            instance_path = Path(os.path.join(instance_directory, instance_file_name))
            kernel_instance_str = gen_a8w8_instance(kernel_parameters, template_params)
            instance_path.write_text(kernel_instance_str)
    
    # generate and write the lookup table
    look_up_dict_str = gen_lookup_dict(kernel_parameters_dict, is_tune)
    look_up_table_header_path = Path(os.path.join(working_path, LOOK_UP_TABLE_HEADER_PATH))
    look_up_table_header_path.write_text(look_up_dict_str)
    
    # generate and write the manifest
    manifest_str = gen_manifest(kernel_parameters_dict)
    manifest_header_path = Path(os.path.join(working_path, MANIFEST_HEADER_PATH))
    manifest_header_path.write_text(manifest_str)

def get_tune_dict(tune_dict_csv: Path) -> dict:
    if not os.path.exists(tune_dict_csv):
        return default_kernels_dict
    
    tune_dict = default_kernels_dict
    tune_df = pd.read_csv(tune_dict_csv)

    for i in range(len(tune_df)):
        M = tune_df.loc[i, "M"]
        N = tune_df.loc[i, "N"]
        K = tune_df.loc[i, "K"]
        kid = tune_df.loc[i, "kernelId"]
        tune_dict[(M, N, K)] = kernels_params_dict[kid]

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

    args = parser.parse_args()


    if args.tune:
        gemm_a8w8_fwd_codegen(args.working_path, kernels_params_dict, args.tune)
    else:
        gemm_a8w8_fwd_codegen(args.working_path, get_tune_dict(args.tune_file), args.tune)
