import os
import sys
from dataclasses import dataclass
import copy
from pathlib import Path
import argparse
import shutil


@dataclass
class kernelInstance:
    BLOCK_SIZE: int
    MPerBLOCK: int
    NPerBLOCK: int
    KPerBLOCK: int
    WAVE_TILE_M: int
    WAVE_TILE_N: int
    WAVE_MAP_M: int
    WAVE_MAP_N: int
    ABLOCK_TRANSFER: list[int]
    BBLOCK_TRANSFER: list[int]
    CBLOCK_TRANSFER: list[int]
    CBLOCK_SPV: list[int]
    CSHUFFLE_MX_PER_WAVE_PERSHUFFLE: int
    CSHUFFLE_NX_PER_WAVE_PERSHUFFLE: int
    LOOP_SCHED: str
    PIPELINE_VERSION: int

    @property
    def name(self) -> str:
        return ("_").join([
            "a8w8_rowwise",
            ("x").join(map(lambda x: str(x), [
                self.BLOCK_SIZE, self.MPerBLOCK, self.NPerBLOCK, self.KPerBLOCK])),
            ("x").join(map(lambda x: str(x), [
                self.WAVE_TILE_M, self.WAVE_TILE_N])),
            ("x").join(map(lambda x: str(x), [
                self.WAVE_MAP_M, self.WAVE_MAP_N])),
            ("x").join(map(lambda x: str(x), self.ABLOCK_TRANSFER)),
            ("x").join(map(lambda x: str(x), self.BBLOCK_TRANSFER)),
            ("x").join(map(lambda x: str(x), self.CBLOCK_TRANSFER)),
            ("x").join(map(lambda x: str(x), self.CBLOCK_SPV)),
            ("x").join(map(lambda x: str(x), [
                self.CSHUFFLE_MX_PER_WAVE_PERSHUFFLE, self.CSHUFFLE_NX_PER_WAVE_PERSHUFFLE])),
            self.LOOP_SCHED.lower(),
            f"v{self.PIPELINE_VERSION}"
        ])


class gemm_a8w8_fwd_codegen:
    def __init__(self, working_path):
        self.working_path = working_path
        self.impl_path = os.path.join(working_path, "impl")
        self.instances_path = os.path.join(working_path, "instances")

    def gen_instance(self, k: kernelInstance):
        INSTANCE_HEAD = f"""// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#include "gemm_a8w8_common.cuh"

template <typename DDataType, typename EDataType = DDataType>
torch::Tensor
{k.name}(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y,
    std::optional<torch::Tensor> bias)
{{
    // The smallest kernel we have available. Works well for memory bound shapes.

    // Check if this input needs to be padded.
    int M = size_to_dim_(XQ.dim() - 1, XQ.sizes());
    int N = WQ.size(0);
    int K = WQ.size(1);
    bool pad = (M % {k.MPerBLOCK} != 0) || (N % {k.NPerBLOCK} != 0) || (K % {k.KPerBLOCK} != 0);
    if (pad)
    {{
        if (bias != std::nullopt)
        {{
            using DeviceGemmInstance = DeviceGemmHelperMMA<
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
                S<{(", ").join(map(lambda x:str(x),k.CBLOCK_SPV))}, {k.CBLOCK_SPV[0]}>,
                {k.CSHUFFLE_MX_PER_WAVE_PERSHUFFLE},
                {k.CSHUFFLE_NX_PER_WAVE_PERSHUFFLE},
                ck::BlockGemmPipelineScheduler::{k.LOOP_SCHED},
                ck::BlockGemmPipelineVersion::v{k.PIPELINE_VERSION},
                ck::tensor_operation::device::GemmSpecialization::MNKPadding>;
            // Run kernel instance.
            return gemm_a8w8_mma_impl<DDataType, EDataType, DeviceGemmInstance>(XQ, WQ, x_scale, w_scale, Y, bias);
        }}
        else
        {{
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
                ck::tensor_operation::device::GemmSpecialization::MNKPadding>;
            // Run kernel instance.
            return gemm_a8w8_rowwise_impl<DDataType, EDataType, DeviceGemmInstance>(XQ, WQ, x_scale, w_scale, Y, bias);
        }}
    }}
    else
    {{
        if (bias != std::nullopt)
        {{
            using DeviceGemmInstance = DeviceGemmHelperMMA<
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
                S<{(", ").join(map(lambda x:str(x),k.CBLOCK_SPV))}, {k.CBLOCK_SPV[0]}>,
                {k.CSHUFFLE_MX_PER_WAVE_PERSHUFFLE},
                {k.CSHUFFLE_NX_PER_WAVE_PERSHUFFLE},
                ck::BlockGemmPipelineScheduler::{k.LOOP_SCHED},
                ck::BlockGemmPipelineVersion::v{k.PIPELINE_VERSION},
                ck::tensor_operation::device::GemmSpecialization::Default>;
            // Run kernel instance.
            return gemm_a8w8_mma_impl<DDataType, EDataType, DeviceGemmInstance>(XQ, WQ, x_scale, w_scale, Y, bias);
        }}
        else
        {{
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
                ck::tensor_operation::device::GemmSpecialization::Default>;
            // Run kernel instance.
            return gemm_a8w8_rowwise_impl<DDataType, EDataType, DeviceGemmInstance>(XQ, WQ, x_scale, w_scale, Y, bias);
        }}
    }}
}}

"""

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
    std::optional<torch::Tensor> bias);

"""
        INSTANCE_dBF16_eBF16 = INSTANCE_template.format(
            name=k.name, dtypes="B16")
        INSTANCE_dFP32_eBF16 = INSTANCE_template.format(
            name=k.name, dtypes="F32, B16")
        INSTANCE_dFP16_eFP16 = INSTANCE_template.format(
            name=k.name, dtypes="F16")
        INSTANCE_dFP32_eFP16 = INSTANCE_template.format(
            name=k.name, dtypes="F32, F16")

        Path(os.path.join(self.impl_path, f"{k.name}.cuh")).write_text(
            INSTANCE_HEAD)
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
       {{{{{MNK}}},                                                                                                       \\
        {kernel_name}<DTYPE, ETYPE>}},                       \\"""

        LOOKUP_end = """
   }

#endif // USE_ROCM
"""
        with open(os.path.join(self.working_path, "gemm_a8w8_lookup.h"), "w") as f:
            f.write(LOOKUP_head)
            for mnk, k in kernels_dict.items():
                if not isinstance(mnk, tuple) or mnk[0] < 0:
                    continue
                print((", ").join(map(lambda x: str(x), list(mnk))), ":", k.name)
                f.write(LOOKUP_template.format(MNK=(", ").join(
                    map(lambda x: str(x), list(mnk))), kernel_name=k.name))
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
    std::optional<torch::Tensor> bias);
"""
        MAINFEST_end = """

#endif // USE_ROCM
"""

        with open(os.path.join(self.working_path, "gemm_a8w8_manifest.h"), "w") as f:
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


def gen_instances(working_path, kernels_dict):
    gemm_a8w8_fwd_codegen(working_path).gen_instances(kernels_dict)


kernels_dict = {
#   (    M,     N,     K): kernel:        BLOCK_SIZE| MPerBLOCK| NPerBLOCK| KPerBLOCK| WAVE_TILE_M| WAVE_TILE_N| WAVE_MAP_M| WAVE_MAP_N| ABLOCK_TRANSFER| BBLOCK_TRANSFER| CBLOCK_TRANSFER| CBLOCK_SPV| CSHUFFLE_MX| CSHUFFLE_NX|  LOOP_SCHED|PIPELINE_VERSION
    (    1,  1280,  8192): kernelInstance(       256,        16,        64,       512,          16,          16,          1,          1,      [32, 8, 1],      [32, 8, 1],   [1, 16, 1,16],  [4, 4, 1],           1,           1, "Intrawave",  3),
    (   32,  1280,  8192): kernelInstance(       256,        16,        64,       512,          16,          16,          1,          1,      [32, 8, 1],      [32, 8, 1],   [1, 16, 1,16],  [4, 4, 1],           1,           1, "Intrawave",  3),
    (   64,  1280,  8192): kernelInstance(       256,        16,        64,       512,          16,          16,          1,          1,      [32, 8, 1],      [32, 8, 1],   [1, 16, 1,16],  [4, 4, 1],           1,           1, "Intrawave",  3),
    (  128,  1280,  8192): kernelInstance(       256,        32,        64,       512,          16,          16,          1,          2,      [32, 8, 1],      [32, 8, 1],   [1, 32, 1, 8],  [8, 8, 1],           1,           2, "Intrawave",  3),
    (  192,  1280,  8192): kernelInstance(       256,        64,        64,       512,          32,          32,          1,          1,      [32, 8, 1],      [32, 8, 1],   [1, 32, 1, 8],  [8, 8, 1],           1,           1, "Intrawave",  3),
    (  256,  1280,  8192): kernelInstance(       256,        64,        64,       512,          32,          32,          1,          1,      [32, 8, 1],      [32, 8, 1],   [1, 32, 1, 8],  [8, 8, 1],           1,           1, "Intrawave",  3),
    (  320,  1280,  8192): kernelInstance(       256,       128,        64,       256,          32,          32,          2,          1,      [16, 16,1],      [16, 16,1],   [1, 32, 1, 8],  [8, 8, 1],           1,           1, "Intrawave",  3),
    (  512,  1280,  8192): kernelInstance(       256,       128,        64,       256,          32,          32,          2,          1,      [16, 16,1],      [16, 16,1],   [1, 32, 1, 8],  [8, 8, 1],           1,           1, "Intrawave",  3),
    ( 1024,  1280,  8192): kernelInstance(       256,       128,       128,       128,          32,          32,          2,          2,      [8, 32, 1],      [8, 32, 1],   [1, 32, 1, 8],  [8, 8, 1],           1,           1, "Intrawave",  4),
    ( 2048,  1280,  8192): kernelInstance(       256,       128,       128,       128,          32,          32,          2,          2,      [8, 32, 1],      [8, 32, 1],   [1, 32, 1, 8],  [8, 8, 1],           1,           1, "Intrawave",  3),
    ( 4096,  1280,  8192): kernelInstance(       256,       128,       128,       128,          32,          32,          2,          2,      [8, 32, 1],      [8, 32, 1],   [1, 32, 1, 8],  [8, 8, 1],           1,           1, "Intrawave",  3),
    ( 8192,  1280,  8192): kernelInstance(       256,       128,       128,       128,          32,          32,          2,          2,      [8, 32, 1],      [8, 32, 1],   [1, 32, 1, 8],  [8, 8, 1],           1,           1, "Intrawave",  3),
    (16384,  1280,  8192): kernelInstance(       256,       128,       128,       128,          32,          32,          2,          2,      [8, 32, 1],      [8, 32, 1],   [1, 32, 1, 8],  [8, 8, 1],           1,           1, "Intrawave",  3),
    (32768,  1280,  8192): kernelInstance(       256,       128,       128,       128,          32,          32,          2,          2,      [8, 32, 1],      [8, 32, 1],   [1, 32, 1, 8],  [8, 8, 1],           1,           1, "Intrawave",  3),
#   (    M,     N,     K): kernel:        BLOCK_SIZE| MPerBLOCK| NPerBLOCK| KPerBLOCK| WAVE_TILE_M| WAVE_TILE_N| WAVE_MAP_M| WAVE_MAP_N| ABLOCK_TRANSFER| BBLOCK_TRANSFER| CBLOCK_TRANSFER| CBLOCK_SPV| CSHUFFLE_MX| CSHUFFLE_NX|  LOOP_SCHED|PIPELINE_VERSION
    (    1,  8192,  1024): kernelInstance(       128,        16,        32,       128,           16,         16,          1,          1,      [8, 16, 1],      [8, 16, 1],   [1, 16, 1, 8],  [4, 4, 1],           1,           1, "Intrawave",  2),
    (   32,  8192,  1024): kernelInstance(       256,        32,       128,       256,           32,         32,          1,          1,      [16, 16,1],      [16, 16,1],   [1, 32, 1, 8],  [8, 8, 1],           1,           1, "Intrawave",  3),
    (   64,  8192,  1024): kernelInstance(       256,        64,       128,       256,           32,         32,          1,          2,      [16, 16,1],      [16, 16,1],   [1, 32, 1, 8],  [8, 8, 1],           1,           1, "Intrawave",  3),
    (  128,  8192,  1024): kernelInstance(       256,       128,       128,       128,           32,         32,          2,          2,      [8, 32, 1],      [8, 32, 1],   [1, 32, 1, 8],  [8, 8, 1],           1,           1, "Intrawave",  3),
    (  192,  8192,  1024): kernelInstance(       256,        64,        64,       128,           32,         32,          1,          1,      [8, 32, 1],      [8, 32, 1],   [1, 32, 1, 8],  [8, 8, 1],           1,           1, "Intrawave",  3),
    (  256,  8192,  1024): kernelInstance(       256,       128,       128,       128,           32,         32,          2,          2,      [8, 32, 1],      [8, 32, 1],   [1, 32, 1, 8],  [8, 8, 1],           1,           1, "Intrawave",  3),
    (  320,  8192,  1024): kernelInstance(       256,        64,       128,       128,           32,         32,          1,          2,      [8, 32, 1],      [8, 32, 1],   [1, 32, 1, 8],  [8, 8, 1],           1,           1, "Intrawave",  3),
    (  512,  8192,  1024): kernelInstance(       256,       256,       224,       128,           32,         32,          2,          7,      [8, 32, 1],      [8, 32, 1],   [1, 64, 1, 4],  [8, 8, 1],           2,           1, "Intrawave",  3),
    ( 1024,  8192,  1024): kernelInstance(       256,       128,       128,       128,           32,         32,          2,          2,      [8, 32, 1],      [8, 32, 1],   [1, 32, 1, 8],  [8, 8, 1],           1,           1, "Intrawave",  3),
    ( 2048,  8192,  1024): kernelInstance(       256,       128,       128,       128,           32,         32,          2,          2,      [8, 32, 1],      [8, 32, 1],   [1, 32, 1, 8],  [8, 8, 1],           1,           1, "Intrawave",  3),
    ( 4096,  8192,  1024): kernelInstance(       256,       128,       128,       128,           32,         32,          2,          2,      [8, 32, 1],      [8, 32, 1],   [1, 32, 1, 8],  [8, 8, 1],           1,           1, "Intrawave",  3),
    ( 8192,  8192,  1024): kernelInstance(       256,       128,       128,        64,           32,         32,          2,          2,      [4, 64, 1],      [4, 64, 1],   [1, 32, 1, 8],  [8, 8, 1],           1,           1, "Intrawave",  4),
    (16384,  8192,  1024): kernelInstance(       256,       128,       128,        64,           32,         32,          2,          2,      [4, 64, 1],      [4, 64, 1],   [1, 32, 1, 8],  [8, 8, 1],           1,           1, "Intrawave",  4),
# other kernel, not in lookup.h           
    (-1):                  kernelInstance(        64,        16,        16,       128,           16,         16,          1,          1,      [8, 8,  1],      [8, 8,  1],   [1, 16, 1, 4],  [4, 4, 1],           1,           1, "Interwave",  2),
    (-3):                  kernelInstance(       128,        32,        16,       128,           16,         16,          1,          1,      [8, 16, 1],      [8, 16, 1],   [1, 16, 1, 8],  [2, 2, 1],           1,           1, "Interwave",  2),
    (-4):                  kernelInstance(        64,        16,        16,       256,           16,         16,          1,          1,      [16, 4, 1],      [16, 4, 1],   [1, 16, 1, 4],  [4, 4, 1],           1,           1, "Intrawave",  1),
    (-5):                  kernelInstance(       128,        16,        32,       128,           16,         16,          1,          1,      [8, 16, 1],      [8, 16, 1],   [1, 16, 1, 8],  [2, 2, 1],           1,           1, "Intrawave",  2),
    (-6):                  kernelInstance(       256,       128,       128,       128,           32,         32,          2,          2,      [8, 32, 1],      [8, 32, 1],   [1, 32, 1, 8],  [8, 8, 1],           1,           1, "Interwave",  1),
    (-7):                  kernelInstance(       256,       128,       128,       128,           32,         32,          2,          2,      [8, 32, 1],      [8, 32, 1],   [1, 32, 1, 8],  [8, 8, 1],           1,           1, "Intrawave",  3),
    (-8):                  kernelInstance(       256,       256,       128,        64,           32,         32,          4,          2,      [4, 64, 1],      [4, 64, 1],   [1, 32, 1, 8],  [8, 8, 1],           1,           1, "Interwave",  1),
    (-9):                  kernelInstance(       256,       224,       256,       128,           16,         16,          7,          8,      [8, 32, 1],      [8, 32, 1],   [1, 32, 1, 8],  [8, 8, 1],           1,           2, "Intrawave",  3),
}

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

    args = parser.parse_args()

    gen_instances(args.working_path, kernels_dict)
