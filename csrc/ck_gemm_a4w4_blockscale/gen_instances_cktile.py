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
from chip_info import (  # noqa: E402
    build_tune_dict,
    write_lookup_header,
    write_name_keyed_lookup_header,
)

from gemm_a4w4_blockscale_common import (  # noqa: E402
    default_kernels_dict_cktile,
    tileKernelInstance,
    kernels_list_cktile,
    kernels_by_name,
)

"""

a4w4_blockscale_gemm instance gen

"""


class gemm_a4w4_blockscale_codegen:
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
            df = pd.read_csv(self.tune_file)
            # The a4w4 tuned CSV mixes CK kernels and ASM kernels in one file with
            # no libtype column.  The Python dispatcher in
            # aiter/ops/gemm_op_a4w4.py routes ASM rows by
            # kernelName.startswith("_ZN") (mangled C++ symbol of the ASM kernel);
            # apply the same filter here so the CK codegen sees only CK rows and
            # build_tune_dict's strict validation stays effective on genuine CK
            # references.
            df = df[~df["kernelName"].astype(str).str.startswith("_ZN")]
            return build_tune_dict(
                df,
                default_kernels_dict_cktile,
                kernels_list,
                kernels_by_name=kernels_by_name,
            )
        return default_kernels_dict_cktile


    def gen_code(self, kernels_dict: dict):
        """
        Codegen for cktile gemm a4w4
        """
        # generate instances code
        for _, k in kernels_dict.items():
            self.gen_instance(k)

        # generate lookup dict for kernel instances
        self.gen_lookup_dict(kernels_dict)

        # generate manifest header for kernel instances
        self.gen_manifest_head(kernels_dict)


    def run(self):
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

#include "gemm_a4w4_blockscale_cktile_common.cuh"

template <typename CDataType, bool HasBias>
torch::Tensor
{k.name}(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y,
    int splitK
    )
{{
    // The smallest kernel we have available. Works well for memory bound shapes.

    // Check if this input needs to be padded.
    int M = size_to_dim_(XQ.dim() - 1, XQ.sizes());
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
        return gemm_a4w4_blockscale_cktile_impl<CDataType, HasBias, TileGemmInstance>(XQ, WQ, x_scale, w_scale, Y, splitK);
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
    int splitK
    );

"""
        if self.istune:
            Path(
                os.path.join(self.instances_path, f"{k.name}_dBF16_eBF16.cpp")
            ).write_text(INSTANCE_template.format(name=k.name, dtypes="TILE_BF16, false"))
            Path(
                os.path.join(self.instances_path, f"{k.name}_dFP32_eFP16.cpp")
            ).write_text(INSTANCE_template.format(name=k.name, dtypes="TILE_FP16, false"))
        else:
            Path(
                os.path.join(self.instances_path, f"{k.name}_dFP32_eBF16.cpp")
            ).write_text(INSTANCE_template.format(name=k.name, dtypes="TILE_BF16, true"))
            Path(
                os.path.join(self.instances_path, f"{k.name}_dFP32_eFP16.cpp")
            ).write_text(INSTANCE_template.format(name=k.name, dtypes="TILE_FP16, true"))

    def gen_lookup_dict(self, kernels_dict):
        """Generate the dispatch lookup header.

        - Tune mode: kernelId-keyed table for *_tune.cu (unchanged).
        - Non-tune mode: name-keyed registry for the Python-driven dispatch
          in gemm_a4w4_blockscale_cktile.cu.
        """
        output_path = os.path.join(self.working_path, "gemm_a4w4_blockscale_cktile_lookup.h")

        if self.istune:
            LOOKUP_head = """#pragma once

// SPDX-License-Identifier: MIT
// Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.

#ifdef USE_ROCM

#define GENERATE_LOOKUP_TABLE(CTYPE, HasBias)                                                                                      \\
   {                                                                                                                             \\"""

            LOOKUP_template = """
       {{{MNK},                                                                                                       \\
        {kernel_name}<CTYPE, HasBias>}},                       \\"""

            LOOKUP_end = """
   }

#endif // USE_ROCM
"""
            write_lookup_header(
                output_path,
                kernels_dict,
                LOOKUP_head,
                LOOKUP_template,
                LOOKUP_end,
                self.istune,
            )
        else:
            LOOKUP_head = """#pragma once
// SPDX-License-Identifier: MIT
// Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.

#ifdef USE_ROCM

#define GENERATE_LOOKUP_TABLE(CTYPE, HasBias)                                                                                      \\
   {                                                                                                                             \\"""

            LOOKUP_template = """
       {{"{kernel_name}",                                                                                                       \\
        {kernel_name}<CTYPE, HasBias>}},                       \\"""

            LOOKUP_end = """
   }

#endif // USE_ROCM
"""
            write_name_keyed_lookup_header(
                output_path,
                kernels_dict,
                LOOKUP_head,
                LOOKUP_template,
                LOOKUP_end,
            )

    def gen_manifest_head(self, kernels_dict):
        MAINFEST_head = """#pragma once
// SPDX-License-Identifier: MIT
// Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.

#ifdef USE_ROCM

#include <cstdlib>

#include <torch/extension.h>
"""
        MAINFEST_template = """
template <typename CDataType, bool HasBias>
torch::Tensor
{kernel_name}(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y,
    int splitK);
"""
        MAINFEST_end = """

#endif // USE_ROCM
"""

        with open(
            os.path.join(self.working_path, "gemm_a4w4_blockscale_cktile_manifest.h"), "w"
        ) as f:
            f.write(MAINFEST_head)
            for mnk, k in kernels_dict.items():
                f.write(MAINFEST_template.format(kernel_name=k.name))
            f.write(MAINFEST_end)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="generate",
        description="gen API for CK gemm a4w4 kernel",
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
        default="aiter/configs/a4w4_blockscale_tuned_gemm.csv",
        required=False,
        help="tune_file include the result after run gemm_a4w4_tune.py",
    )

    parser.add_argument(
        "--tune", action="store_true", required=False, help="generated tune instances"
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
    codegen = gemm_a4w4_blockscale_codegen(args.working_path, args.tune, args.tune_file)
    codegen.run()