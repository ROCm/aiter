# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
import os
from pathlib import Path
import pandas as pd
import argparse
import shutil
import torch
from deepgemm_common import kernelInstance, kernels_list, default_kernels_dict


class deepgemm_codegen:
    def __init__(self, working_path, istune=False):
        self.working_path = working_path
        self.impl_path = os.path.join(working_path, "impl")
        self.instances_path = os.path.join(working_path, "instances")
        self.istune = istune
        # self.a_dtype = a_dtype.upper()
        # self.b_dtype = b_dtype.upper()
        # self.c_dtype = c_dtype.upper()
        # self.quant_type = quant_type
        assert istune is False, "not support tuning!"

    def gen_instance(self, k: kernelInstance):
        INSTANCE_IMPL = f"""// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
#include "deepgemm_common.cuh"
template <typename ABDataType, typename AccDataType, typename CDataType>
torch::Tensor
{k.name}(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &Y,
    torch::Tensor &group_layout,
    std::optional<torch::Tensor> x_scale,
    std::optional<torch::Tensor> w_scale)
{{{{
    // The smallest kernel we have available. Works well for memory bound shapes.
    int group_count = Y.size(0);
    int M = XQ.size(1);
    int N = Y.size(2);
    int K = XQ.size(2);
    int Stride_A = K;
    int Stride_B = K;
    int Stride_C = N;
    {{INSTANCE_CONTENT}}
    return Y;
}}}}
"""
        INSTANCE_CONTENT = f"""if (x_scale != std::nullopt && w_scale != std::nullopt )
        {{{{
            auto per_a_scale_dev_ptr = ck_tile::FlatmmScalePointer<1>{{static_cast<float*>(x_scale.value().data_ptr())}};
            auto per_b_scale_dev_ptr = ck_tile::FlatmmScalePointer<1>{{static_cast<float*>(w_scale.value().data_ptr())}};
            ck_tile::MaskedGroupedFlatmmHostArgs<decltype(per_a_scale_dev_ptr), decltype(per_b_scale_dev_ptr)> kernel_args{{
                reinterpret_cast<ck_tile::index_t *>(group_layout.data_ptr()),
                group_count,
                M,
                N,
                K,
                reinterpret_cast<const void*>(XQ.data_ptr()),
                Stride_A,
                reinterpret_cast<const void*>(WQ.data_ptr()),
                Stride_B,
                {{}},{{}},
                reinterpret_cast<void*>(Y.data_ptr()),
                Stride_C,
                1, //KBatch
                per_a_scale_dev_ptr,
                per_b_scale_dev_ptr
            }};
            using TileConfig = MGroupedFlatmmConfig<ABDataType,
                {k.MPerBLOCK},
                {k.NPerBLOCK},
                {k.KPerBLOCK},
                {k.WAVE_MAP_M},
                {k.WAVE_MAP_N},
                {k.WAVE_TILE_M},
                {k.WAVE_TILE_N},
                64>;
            // Run kernel instance.
            auto stream_config = ck_stream_config{{at::hip::getCurrentHIPStreamMasqueradingAsCUDA().stream()}};
            grouped_flatmm<TileConfig, ABDataType, ABDataType, ck_tile::tuple<>, AccDataType, CDataType, row_major, col_major, ck_tile::tuple<>, row_major, false, ck_tile::element_wise::PassThrough>(kernel_args, stream_config);
        }}}}
        else
        {{{{
            auto per_a_scale_dev_ptr = ck_tile::FlatmmScalePointer<-1>{{nullptr}};
            auto per_b_scale_dev_ptr = ck_tile::FlatmmScalePointer<-1>{{nullptr}};
            ck_tile::MaskedGroupedFlatmmHostArgs<decltype(per_a_scale_dev_ptr), decltype(per_b_scale_dev_ptr)> kernel_args{{
                reinterpret_cast<ck_tile::index_t *>(group_layout.data_ptr()),
                group_count,
                M,
                N,
                K,
                reinterpret_cast<const void*>(XQ.data_ptr()),
                Stride_A,
                reinterpret_cast<const void*>(WQ.data_ptr()),
                Stride_B,
                {{}},{{}},
                reinterpret_cast<void*>(Y.data_ptr()),
                Stride_C,
                1, //KBatch
                per_a_scale_dev_ptr,
                per_b_scale_dev_ptr
            }};
            using TileConfig = MGroupedFlatmmConfig<ABDataType,
                {k.MPerBLOCK},
                {k.NPerBLOCK},
                {k.KPerBLOCK},
                {k.WAVE_MAP_M},
                {k.WAVE_MAP_N},
                {k.WAVE_TILE_M},
                {k.WAVE_TILE_N},
                32>;
            // Run kernel instance.
            auto stream_config = ck_stream_config{{at::hip::getCurrentHIPStreamMasqueradingAsCUDA().stream()}};
            grouped_flatmm<TileConfig, ABDataType, ABDataType, ck_tile::tuple<>, AccDataType, CDataType, row_major, col_major, ck_tile::tuple<>, row_major, false, ck_tile::element_wise::PassThrough>(kernel_args, stream_config);
        }}}}
"""

        INSTANCE_IMPL_str = INSTANCE_IMPL.format(INSTANCE_CONTENT=(INSTANCE_CONTENT))

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
    torch::Tensor &Y,
    torch::Tensor &grouped_layout,
    std::optional<torch::Tensor> x_scale,
    std::optional<torch::Tensor> w_scale);
"""
        # if self.istune:
        #     INSTANCE_abI8_dBF16_eBF16 = INSTANCE_template.format(
        #         name=k.name, dtypes="I8, B16"
        #     )
        #     Path(
        #         os.path.join(self.instances_path, f"{k.name}_abI8_dB16_eB16.cpp")
        #     ).write_text(INSTANCE_abI8_dBF16_eBF16)
        # else:
        for CDtype in ["bf16", "fp16"]:
            for ABDtype in ["bf16", "fp16", "fp8"]:
                for AccDtype in ["float"]:
                    intsance = INSTANCE_template.format(
                        name=k.name, dtypes=f"{ABDtype}, {AccDtype}, {CDtype}"
                    )
                    Path(
                        os.path.join(
                            self.instances_path,
                            f"{k.name}_ab{ABDtype}_acc{AccDtype}_C{CDtype}.cpp",
                        )
                    ).write_text(intsance)

    def gen_lookup_dict(self, kernels_dict):
        LOOKUP_head = """#pragma once
// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
// #ifdef USE_ROCM
#define GENERATE_LOOKUP_TABLE(ABTYPE, ACCTYPE, CTYPE)                                                                                      \\
   {                                                                                                                             \\"""

        LOOKUP_template = """
       {{{MNK},                                                                                                       \\
        {kernel_name}<ABTYPE, ACCTYPE, CTYPE>}},                       \\"""

        LOOKUP_end = """
   }
// #endif // USE_ROCM
"""
        with open(os.path.join(self.working_path, "deepgemm_lookup.h"), "w") as f:
            f.write(LOOKUP_head)
            for mnk, k in kernels_dict.items():
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
        MAINFEST_head = """#pragma once
// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
// #ifdef USE_ROCM
#include <cstdlib>
#include <torch/extension.h>
"""
        MAINFEST_template = """
template <typename ABDataType, typename DDataType, typename EDataType>
torch::Tensor
{kernel_name}(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &Y,
    torch::Tensor &grouped_layout,
    std::optional<torch::Tensor> x_scale,
    std::optional<torch::Tensor> w_scale);
"""
        MAINFEST_end = """
// endif // USE_ROCM
"""

        with open(os.path.join(self.working_path, "deepgemm_manifest.h"), "w") as f:
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

    # parser.add_argument(
    #     "-f",
    #     "--tune_file",
    #     default="aiter/configs/a8w8_tuned_gemm.csv",
    #     required=False,
    #     help="tune_file include the result after run gemm_a8w8_tune.py",
    # )

    # parser.add_argument(
    #     "--tune", action="store_true", required=False, help="generated tune instances"
    # )

    parser.add_argument(
        "--out_type",
        default="all",
        required=False,
        help="Specifie the type of scale\n \
            all: [bf16, fp16] \n  \
            bf16, fp16",
    )

    # parser.add_argument(
    #     "--scale_type",
    #     default="all",
    #     required=False,
    #     help="Specifie the type of scale\n \
    #         all: [fp32, same as out] \n  \
    #         same: [same as out]"
    # )

    args = parser.parse_args()
    # TODO: use tune flag.
    codegen = deepgemm_codegen(args.working_path, False)

    # if args.tune:
    codegen.gen_instances(kernels_list)
    # else:
    # codegen.gen_instances(get_tune_dict(args.tune_file))
