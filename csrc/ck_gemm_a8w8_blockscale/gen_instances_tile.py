# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
import argparse
import os
import shutil
from pathlib import Path

import pandas as pd
import torch

from aiter.jit.core import AITER_LOG_TUNED_CONFIG
from gemm_a8w8_blockscale_instance_tile import (
    default_kernels_dict_tile,
    TileKernelInstance,
    candidate_kernels_dict_tile,
)


"""
a8w8_blockscale_gemm instance gen for tile CK 
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
        
        tune_dict = default_kernels_dict_tile
        
        if os.path.exists(tune_dict_csv):
            tune_df = pd.read_csv(tune_dict_csv)
            if torch.cuda.is_available():
                gpu = torch.cuda.current_device()
                device_properties = torch.cuda.get_device_properties(gpu)
                cu_num = device_properties.multi_processor_count
                tune_df = tune_df[
                    (tune_df["cu_num"] == cu_num) & (tune_df["libtype"] == "ck_tile")
                ].reset_index()
            for i in range(len(tune_df)):
                M = int(tune_df.loc[i, "M"])
                N = int(tune_df.loc[i, "N"])
                K = int(tune_df.loc[i, "K"])
                kid = int(tune_df.loc[i, "kernelId"])
                
                if kid in candidate_kernels_dict_tile:
                    tune_dict[(M, N, K)] = candidate_kernels_dict_tile[kid]
                else:
                    print(f"Warning: kernelId {kid} not found in candidate_kernels_dict_tile for shape ({M}, {N}, {K})")
                
        return tune_dict
    
    def gen_tile_instance(self, k: TileKernelInstance):
        """
        Generate kernel instance code for tile gemm a8w8 blockscale
        """
        
        TILE_INSTANCE_IMPL = f"""// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#include "gemm_a8w8_blockscale_common_tile.cuh"

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
    
    // Instantiate tile gemm instance.
    __TILE_INSTANCE_PLACEHOLDER__
    
}}

"""    
        TILE_INSTANCE = f"""using TileGemmInstance = TileGemmTileHelperF8BlockScale<
            DDataType, EDataType,
            {k.M_Tile}, {k.N_Tile}, {k.K_Tile},
            {k.M_Warp}, {k.N_Warp}, {k.K_Warp},
            {k.M_Warp_Tile}, {k.N_Warp_Tile}, {k.K_Warp_Tile},
            {str(k.TransposeC).lower()},
            {str(k.DoubleSmemBuffer).lower()},
            {str(k.UsePersistentKernel).lower()},
            {str(k.kPadM).lower()}, {str(k.kPadN).lower()}, {str(k.kPadK).lower()},
            ck_tile::GemmPipelineScheduler::{k.Scheduler}>;
            
        // Run kernel instance.
        return tile_gemm_a8w8_blockscale_impl<DDataType, EDataType, TileGemmInstance>(XQ, WQ, x_scale, w_scale, Y);
"""

        TILE_INSTANCE_IMPL_str = TILE_INSTANCE_IMPL.replace(
            "__TILE_INSTANCE_PLACEHOLDER__", TILE_INSTANCE
        )

        Path(os.path.join(self.impl_path, f"{k.name}.cuh")).write_text(
            TILE_INSTANCE_IMPL_str
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
        INSTANCE_dFP32_eBF16 = INSTANCE_template.format(name=k.name, dtypes="TILE_FP32, TILE_BF16")
        INSTANCE_dFP32_eFP16 = INSTANCE_template.format(name=k.name, dtypes="TILE_FP32, TILE_FP16")
        # TODO: dFP8_eFP8
        
        Path(
            os.path.join(self.instances_path, f"{k.name}_dFP32_eBF16.cpp")
        ).write_text(INSTANCE_dFP32_eBF16)
        Path(
            os.path.join(self.instances_path, f"{k.name}_dFP32_eFP16.cpp")
        ).write_text(INSTANCE_dFP32_eFP16)

    def gen_lookup_dict(self, kernels_dict: dict):
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
            os.path.join(self.working_path, "gemm_a8w8_blockscale_lookup_tile.h"), "w"
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
            os.path.join(self.working_path, "gemm_a8w8_blockscale_manifest_tile.h"), "w"
        ) as f:
            f.write(MAINFEST_head)
            for _, k in kernels_dict.items():
                f.write(MAINFEST_template.format(kernel_name=k.name))
            f.write(MAINFEST_end)

    def gen_code(self, kernels_dict: dict):
        """
        Codegen for tile gemm a8w8 blockscale
        """
        
        # generate instances code
        if AITER_LOG_TUNED_CONFIG:
            logger.info(
                f"Generating {len(kernels_dict)} instances for tile gemm a8w8 blockscale...")
        for _, k in kernels_dict.items():
            if AITER_LOG_TUNED_CONFIG:
                logger.info(f"Generating tile instance: {k.name}")
            self.gen_tile_instance(k)

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
            self.gen_code(candidate_kernels_dict_tile)
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
    
    

