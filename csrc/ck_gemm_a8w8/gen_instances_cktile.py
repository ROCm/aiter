# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
import argparse
import os
import sys
import shutil
from pathlib import Path

import pandas as pd

this_dir = os.path.dirname(os.path.abspath(__file__))
AITER_CORE_DIR = (
    os.path.join(os.path.abspath(f"{this_dir}/../../../"), "aiter/jit/utils")
    if os.path.exists(
        os.path.join(os.path.abspath(f"{this_dir}/../../../"), "aiter_meta")
    )
    else os.path.abspath(f"{this_dir}/../../aiter/jit/utils")
)
sys.path.insert(0, AITER_CORE_DIR)
from chip_info import build_tune_dict, write_lookup_header  # noqa: E402

from gemm_a8w8_common import (  # noqa: E402
    default_kernels_dict_cktile,
    tileKernelInstance,
    kernels_list_cktile,
)


"""
a8w8_gemm instance gen for cktile
"""


class gemm_a8w8_fwd_codegen:
    def __init__(self, working_path, istune=False, tune_file=None):
        self.working_path = working_path
        if not os.path.exists(working_path):
            os.makedirs(working_path)

        self.impl_path = os.path.join(working_path, "impl")
        self.instances_path = os.path.join(working_path, "instances")
        self.istune = istune
        self.tune_file = tune_file


    def get_tune_dict(self):
        if os.path.exists(self.tune_file):
            return build_tune_dict(
                pd.read_csv(self.tune_file), default_kernels_dict_cktile, kernels_list_cktile
            )
        return default_kernels_dict_cktile


    def gen_code(self, kernels_dict: dict):
        """
        Codegen for cktile gemm a8w8
        """
        # generate instances code
        for _, k in kernels_dict.items():
            self.gen_instance(k)

        # generate lookup dict for kernel instances
        self.gen_lookup_dict(kernels_dict)

        # generate manifest header for kernel instances
        self.gen_manifest_head(kernels_dict)


    def run(self):
        """
        Run codegen and generate all the files together
        """

        if os.path.exists(self.impl_path):
            shutil.rmtree(self.impl_path)
        os.mkdir(self.impl_path)
        if os.path.exists(self.instances_path):
            shutil.rmtree(self.instances_path)
        os.mkdir(self.instances_path)

        # generate code for cktile
        if self.istune:
            # generate code for default kernels
            self.gen_code(kernels_list_cktile)
        else:
            # generate code for tuned kernels from tune_file
            self.gen_code(self.get_tune_dict())       


    def gen_instance(self, k: tileKernelInstance):
        TILE_INSTANCE_IMPL = f"""// SPDX-License-Identifier: MIT
// Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.

#include "gemm_a8w8_cktile_common.cuh"

template <typename ABDataType, typename DDataType, typename EDataType, bool HasBias>
torch::Tensor
{k.name}(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y,
    std::optional<torch::Tensor> bias,
    int k_batch
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

        TILE_INSTANCE = f"""using TileGemmInstance = TileGemmConfig<
            {k.M_Tile}, {k.N_Tile}, {k.K_Tile},
            {k.M_Warp}, {k.N_Warp}, {k.K_Warp},
            {k.M_Warp_Tile}, {k.N_Warp_Tile}, {k.K_Warp_Tile},
            {str(k.TiledMMAPermuteN).lower()},
            {str(k.TransposeC).lower()},
            {str(k.UsePersistentKernel).lower()},
            ck_tile::GemmPipelineScheduler::{k.Scheduler},
            {k.BlockPerCu},
            {str(k.AQRowMajor).lower()}>;

    // Run kernel instance.
    return gemm_a8w8_cktile_impl<ABDataType, DDataType, EDataType, HasBias, TileGemmInstance>(XQ, WQ, x_scale, w_scale, Y, bias, k_batch);
"""

        TILE_INSTANCE_IMPL_str = TILE_INSTANCE_IMPL.replace(
            "__TILE_INSTANCE_PLACEHOLDER__", TILE_INSTANCE
        )

        Path(os.path.join(self.impl_path, f"{k.name}.cuh")).write_text(
            TILE_INSTANCE_IMPL_str
        )

        INSTANCE_template = """// SPDX-License-Identifier: MIT
// Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.

#include "impl/{name}.cuh"

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
        if self.istune:
            # Generate both I8 and F8 instances for tuning
            # I8 instances
            for EDtype in ["TILE_BF16"]:
                INSTANCE_abI8 = INSTANCE_template.format(
                    name=k.name, dtypes=f"TILE_I8, TILE_BF16, {EDtype}, false"
                )
                Path(
                    os.path.join(
                        self.instances_path, f"{k.name}_abI8_dB16_e{EDtype}.cpp"
                    )
                ).write_text(INSTANCE_abI8)

            # F8 instances
            for EDtype in ["TILE_BF16"]:
                INSTANCE_abF8 = INSTANCE_template.format(
                    name=k.name, dtypes=f"TILE_FP8, TILE_FP32, {EDtype}, false"
                )
                Path(
                    os.path.join(
                        self.instances_path, f"{k.name}_abF8_dF32_e{EDtype}.cpp"
                    )
                ).write_text(INSTANCE_abF8)
        else:
            for EDtype in ["TILE_BF16", "TILE_FP16"]:
                for ABDtype in ["TILE_FP8", "TILE_I8"]:
                    for DDtype in ["TILE_FP32", EDtype]:
                        intsance = INSTANCE_template.format(
                            name=k.name, dtypes=f"{ABDtype}, {DDtype}, {EDtype}, true"
                        )
                        Path(
                            os.path.join(
                                self.instances_path,
                                f"{k.name}_ab{ABDtype}_d{DDtype}_e{EDtype}.cpp",
                            )
                        ).write_text(intsance)


    def gen_lookup_dict(self, kernels_dict: dict):
        LOOKUP_head = """#pragma once
// SPDX-License-Identifier: MIT
// Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.

#ifdef USE_ROCM

#define GENERATE_LOOKUP_TABLE(ABTYPE, DTYPE, ETYPE, HasBias)                                                                                      \\
   {                                                                                                                             \\"""

        LOOKUP_template = """
       {{{MNK},                                                                                                       \\
        {kernel_name}<ABTYPE, DTYPE, ETYPE, HasBias>}},                       \\"""

        LOOKUP_end = """
   }

#endif // USE_ROCM
"""
        write_lookup_header(
            os.path.join(self.working_path, "gemm_a8w8_cktile_lookup.h"),
            kernels_dict,
            LOOKUP_head,
            LOOKUP_template,
            LOOKUP_end,
            self.istune,
        )

    def gen_manifest_head(self, kernels_dict):
        """
        Generate manifest header for kernel instances, declaring all the kernel APIs
        """

        MAINFEST_head = """#pragma once
// SPDX-License-Identifier: MIT
// Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.

#ifdef USE_ROCM

#include <cstdlib>

#include <torch/extension.h>
"""
        MAINFEST_template = """
template <typename ABDataType, typename DDataType, typename EDataType, bool HasBias>
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

        with open(os.path.join(self.working_path, "gemm_a8w8_cktile_manifest.h"), "w") as f:
            f.write(MAINFEST_head)
            for mnk, k in kernels_dict.items():
                f.write(MAINFEST_template.format(kernel_name=k.name))
            f.write(MAINFEST_end)


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

    parser.add_argument(
        "-f",
        "--tune_file",
        default="aiter/configs/a8w8_tuned_gemm.csv",
        required=False,
        help="tune_file include the result after run gemm_a8w8_tune.py",
    )

    parser.add_argument(
        "--tune", action="store_true", required=False, help="generated tune instances"
    )

    args = parser.parse_args()
    codegen = gemm_a8w8_fwd_codegen(args.working_path, args.tune, args.tune_file)
    codegen.run()
