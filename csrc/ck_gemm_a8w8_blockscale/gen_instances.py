# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
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

from gemm_a8w8_blockscale_instance import (  # noqa: E402
    default_kernels_dict,
    KernelInstance,
    candidate_kernels_dict,
    candidate_kernels_by_name,
)

"""
a8w8_blockscale_gemm instance gen for ck
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
        if os.path.exists(tune_dict_csv):
            return build_tune_dict(
                pd.read_csv(tune_dict_csv),
                default_kernels_dict,
                candidate_kernels_dict,
                libtype="ck",
                kernels_by_name=candidate_kernels_by_name,
            )
        return default_kernels_dict

    def gen_ck_instance(self, k: KernelInstance):
        """
        Generate kernel instance code for ck gemm a8w8 blockscale
        """

        LEGACY_INSTANCE_IMPL = f"""// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#include "gemm_a8w8_blockscale_common.cuh"

template <typename DDataType, typename EDataType>
torch::Tensor
{k.name}(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y,
    int KBatch
    )
{{
    // Get M, N, K from input tensors.
    int M = XQ.numel() / XQ.size(-1);
    int N = WQ.size(0);
    int K = WQ.size(1);

    // Get whether this input needs to be padded.
    auto gemm_spec = GetGemmSpec(M, N, K, {k.MPerBLOCK}, {k.NPerBLOCK}, {k.KPerBLOCK});

    using CKGemmSpec = ck::tensor_operation::device::GemmSpecialization;

    // Run kernel instance.
    auto run_kernel = [&]<CKGemmSpec spec> [[clang::always_inline]] (std::integral_constant<CKGemmSpec, spec>) {{
        using LegacyGemmInstance = DeviceLegacyGemmHelperF8BlockScale<
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
                    spec>;

        return gemm_a8w8_blockscale_impl<DDataType, EDataType, LegacyGemmInstance>(XQ, WQ, x_scale, w_scale, Y, KBatch);
    }};

    if(gemm_spec == GemmSpecialization::Default)
    {{
        return run_kernel(std::integral_constant<CKGemmSpec, CKGemmSpec::Default>{{}});
    }} else if(gemm_spec == GemmSpecialization::MPadding)
    {{
        return run_kernel(std::integral_constant<CKGemmSpec, CKGemmSpec::MPadding>{{}});
    }} else if(gemm_spec == GemmSpecialization::NPadding)
    {{
        return run_kernel(std::integral_constant<CKGemmSpec, CKGemmSpec::NPadding>{{}});
    }} else if(gemm_spec == GemmSpecialization::KPadding)
    {{
        return run_kernel(std::integral_constant<CKGemmSpec, CKGemmSpec::KPadding>{{}});
    }} else if(gemm_spec == GemmSpecialization::MNPadding)
    {{
        return run_kernel(std::integral_constant<CKGemmSpec, CKGemmSpec::MNPadding>{{}});
    }} else if(gemm_spec == GemmSpecialization::MKPadding)
    {{
        return run_kernel(std::integral_constant<CKGemmSpec, CKGemmSpec::MKPadding>{{}});
    }} else if(gemm_spec == GemmSpecialization::NKPadding)
    {{
        return run_kernel(std::integral_constant<CKGemmSpec, CKGemmSpec::NKPadding>{{}});
    }} else if(gemm_spec == GemmSpecialization::MNKPadding)
    {{
        return run_kernel(std::integral_constant<CKGemmSpec, CKGemmSpec::MNKPadding>{{}});
    }} else
    {{
        throw std::runtime_error("Unsupported GemmSpecialization!");
    }}
}}
"""

        Path(os.path.join(self.impl_path, f"{k.name}.cuh")).write_text(
            LEGACY_INSTANCE_IMPL
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
    torch::Tensor &Y,
    int KBatch
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
        Generate lookup dictionary for kernel instances.

        - Tune mode (istune=True): emits a kernelId-keyed table for the
          tuner's *_tune.cu, unchanged from before.
        - Non-tune mode (istune=False): emits a name-keyed registry consumed
          by gemm_a8w8_blockscale.cu's Python-driven dispatch.  The Python
          frontend (aiter/ops/gemm_op_a8w8.py) reads the tuned CSV and passes
          the resolved kernelName string in; C++ looks it up directly here.
        """

        output_path = os.path.join(self.working_path, "gemm_a8w8_blockscale_lookup.h")

        if self.istune:
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
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#ifdef USE_ROCM

#define GENERATE_LOOKUP_TABLE(DTYPE, ETYPE)                                                                                      \\
   {                                                                                                                             \\"""

            LOOKUP_template = """
       {{"{kernel_name}",                                                                                                       \\
        {kernel_name}<DTYPE, ETYPE>}},                       \\"""

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
    torch::Tensor &Y,
    int KBatch);
"""
        MAINFEST_end = """

#endif // USE_ROCM
"""

        with open(
            os.path.join(self.working_path, "gemm_a8w8_blockscale_manifest.h"),
            "w",
        ) as f:
            f.write(MAINFEST_head)
            seen_kernel_names = set()
            for _, k in kernels_dict.items():
                if k.name not in seen_kernel_names:
                    seen_kernel_names.add(k.name)
                    f.write(MAINFEST_template.format(kernel_name=k.name))
            f.write(MAINFEST_end)

    def gen_code(self, kernels_dict: dict):
        """
        Codegen for ck gemm a8w8 blockscale
        """

        # generate instances code
        for _, k in kernels_dict.items():
            self.gen_ck_instance(k)

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

        # generate code for ck
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
