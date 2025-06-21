# SPDX-License-Identifier: MIT
# Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
import os
import argparse
from gemm_moe_ck2stages_common import get_gemm1_kernels_list, get_gemm2_kernels_list

STG_INSTANCE_IMPL = """// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
#include "gemm_moe_ck2stages_common{quanttype}.cuh"

using A0DataType = {A0DataType};
using B0DataType = {B0DataType};
using AccDataType = {AccDataType};
using EDataType = {EDataType};
using CDEElementOp = {CDEElementOp};
const bool Nswizzle = {Nswizzle};
const bool PerTensorQuant = {PerTensorQuant};
const bool MulRoutedWeight = {MulRoutedWeight};
const int ActOP = {ActOP};
CK_MOE_STAGE{Stage}_GEMM_DEFINE({BlockSize}, {MPerBlock}, {NPerBlock}, {KPerBlock}, {MWaves}, {NWaves}, V{PipelineVer})
"""


LOOKUP_head = """#pragma once
// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
#include "gemm_moe_ck2stages.h"

#define GENERATE_LOOKUP_TABLE()                                                                                      \\
   {                                                                                                                             \\"""

LOOKUP_template = """
       {{"{kernel_tag}",                                                                                                       \\
        ck_moe_stage{Stage}_gemm<{A0DataType}, {B0DataType}, {AccDataType}, {EDataType}, {CDEElementOp}, V{PipelineVer}, {BlockSize}, {MPerBlock}, {NPerBlock}, {KPerBlock}, {MWaves}, {NWaves}, {Nswizzle}, {PerTensorQuant}, {MulRoutedWeight}, {ActOP}>}},                       \\"""

LOOKUP_end = """
   }

"""

A16W16_A8W8_gemm1_heuristic_dispatch = """#pragma once
// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
#include "gemm_moe_ck2stages.h"

MoeKernel moe_stage1_heuristic_dispatch(int block_m)
{{
    if (block_m == 32)
    {{
        return ck_moe_stage1_gemm<{A0DataType}, {B0DataType}, {AccDataType}, {EDataType}, {CDEElementOp}, V1, 256, 32, 64, 128/sizeof({A0DataType}), 1, 4, {Nswizzle}, {PerTensorQuant}, {MulRoutedWeight}, {ActOP}>;
    }}
    else if (block_m == 64)
    {{
        return ck_moe_stage1_gemm<{A0DataType}, {B0DataType}, {AccDataType}, {EDataType}, {CDEElementOp}, V1, 256, 64, 64, 128/sizeof({A0DataType}), 1, 4, {Nswizzle}, {PerTensorQuant}, {MulRoutedWeight}, {ActOP}>;
    }}
    else if (block_m == 128)
    {{
        return ck_moe_stage1_gemm<{A0DataType}, {B0DataType}, {AccDataType}, {EDataType}, {CDEElementOp}, V1, 256, 128, 64, 128/sizeof({A0DataType}), 1, 4, {Nswizzle}, {PerTensorQuant}, {MulRoutedWeight}, {ActOP}>;
    }}
    else
    {{
        TORCH_CHECK(
            false,
            "Unsupported block_m value for moe heuristic dispatch: ",
            block_m);
    }}
}}

"""

A8W4_gemm1_heuristic_dispatch = """#pragma once
// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
#include "gemm_moe_ck2stages.h"

MoeKernel moe_stage1_heuristic_dispatch(int block_m)
{{
    if (block_m == 32)
    {{
        return ck_moe_stage1_gemm<{A0DataType}, {B0DataType}, {AccDataType}, {EDataType}, {CDEElementOp}, V1, 256, 32, 64, 128/sizeof({A0DataType}), 1, 4, {Nswizzle}, {PerTensorQuant}, {MulRoutedWeight}, {ActOP}>;
    }}
    else if (block_m == 64)
    {{
        return ck_moe_stage1_gemm<{A0DataType}, {B0DataType}, {AccDataType}, {EDataType}, {CDEElementOp}, V1, 256, 64, 64, 128/sizeof({A0DataType}), 1, 4, {Nswizzle}, {PerTensorQuant}, {MulRoutedWeight}, {ActOP}>;
    }}
    else if (block_m == 128)
    {{
        return ck_moe_stage1_gemm<{A0DataType}, {B0DataType}, {AccDataType}, {EDataType}, {CDEElementOp}, V1, 256, 128, 64, 128/sizeof({A0DataType}), 1, 4, {Nswizzle}, {PerTensorQuant}, {MulRoutedWeight}, {ActOP}>;
    }}
    else
    {{
        TORCH_CHECK(
            false,
            "Unsupported block_m value for moe heuristic dispatch: ",
            block_m);
    }}
}}

"""

A4W4_gemm1_heuristic_dispatch = """#pragma once
// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
#include "gemm_moe_ck2stages.h"

MoeKernel moe_stage1_heuristic_dispatch(int block_m)
{{
    if (block_m == 32)
    {{
        return ck_moe_stage1_gemm<{A0DataType}, {B0DataType}, {AccDataType}, {EDataType}, {CDEElementOp}, V3, 256, 32, 128, 128/sizeof({A0DataType}), 1, 4, {Nswizzle}, {PerTensorQuant}, {MulRoutedWeight}, {ActOP}>;
    }}
    else
    {{
        TORCH_CHECK(
            false,
            "Unsupported block_m value for moe heuristic dispatch: ",
            block_m);
    }}
}}

"""


A8W8_blockscale_gemm1_heuristic_dispatch = """#pragma once
// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
#include "gemm_moe_ck2stages.h"

MoeKernel moe_stage1_heuristic_dispatch(int block_m)
{{
    if (block_m == 64)
    {{
        return ck_moe_stage1_gemm<{A0DataType}, {B0DataType}, {AccDataType}, {EDataType}, {CDEElementOp}, V3, 256, 64, 128, 128/sizeof({A0DataType}), 1, 4, {Nswizzle}, {PerTensorQuant}, {MulRoutedWeight}, {ActOP}>;
    }}
    else
    {{
        TORCH_CHECK(
            false,
            "Unsupported block_m value for moe heuristic dispatch: ",
            block_m);
    }}
}}

"""


A16W16_A8W8_gemm2_heuristic_dispatch = """
MoeKernel moe_stage2_heuristic_dispatch(int block_m)
{{
    if (block_m == 32)
    {{
        return ck_moe_stage2_gemm<{A0DataType}, {B0DataType}, {AccDataType}, {EDataType}, {CDEElementOp}, V1, 256, 32, 128, 256/sizeof({A0DataType}), 1, 4, {Nswizzle}, {PerTensorQuant}, {MulRoutedWeight}, {ActOP}>;
    }}
    else if (block_m == 64)
    {{
        return ck_moe_stage2_gemm<{A0DataType}, {B0DataType}, {AccDataType}, {EDataType}, {CDEElementOp}, V1, 256, 64, 128, 256/sizeof({A0DataType}), 1, 4, {Nswizzle}, {PerTensorQuant}, {MulRoutedWeight}, {ActOP}>;
    }}
    else if (block_m == 128)
    {{
        return ck_moe_stage2_gemm<{A0DataType}, {B0DataType}, {AccDataType}, {EDataType}, {CDEElementOp}, V1, 256, 128, 128, 128/sizeof({A0DataType}), 1, 4, {Nswizzle}, {PerTensorQuant}, {MulRoutedWeight}, {ActOP}>;
    }}
    else
    {{
        TORCH_CHECK(
            false,
            "Unsupported block_m value for moe heuristic dispatch: ",
            block_m);
    }}
}}

"""

A8W4_gemm2_heuristic_dispatch = """
MoeKernel moe_stage2_heuristic_dispatch(int block_m)
{{
    if (block_m == 32)
    {{
        return ck_moe_stage2_gemm<{A0DataType}, {B0DataType}, {AccDataType}, {EDataType}, {CDEElementOp}, V1, 256, 32, 128, 128/sizeof({A0DataType}), 1, 4, {Nswizzle}, {PerTensorQuant}, {MulRoutedWeight}, {ActOP}>;
    }}
    else if (block_m == 64)
    {{
        return ck_moe_stage2_gemm<{A0DataType}, {B0DataType}, {AccDataType}, {EDataType}, {CDEElementOp}, V1, 256, 64, 128, 128/sizeof({A0DataType}), 1, 4, {Nswizzle}, {PerTensorQuant}, {MulRoutedWeight}, {ActOP}>;
    }}
    else if (block_m == 128)
    {{
        return ck_moe_stage2_gemm<{A0DataType}, {B0DataType}, {AccDataType}, {EDataType}, {CDEElementOp}, V1, 256, 128, 128, 128/sizeof({A0DataType}), 1, 4, {Nswizzle}, {PerTensorQuant}, {MulRoutedWeight}, {ActOP}>;
    }}
    else
    {{
        TORCH_CHECK(
            false,
            "Unsupported block_m value for moe heuristic dispatch: ",
            block_m);
    }}
}}

"""

A4W4_gemm2_heuristic_dispatch = """
MoeKernel moe_stage2_heuristic_dispatch(int block_m)
{{
    if (block_m == 32)
    {{
        return ck_moe_stage2_gemm<{A0DataType}, {B0DataType}, {AccDataType}, {EDataType}, {CDEElementOp}, V1, 64, 32, 32, 128/sizeof({A0DataType}), 1, 1, {Nswizzle}, {PerTensorQuant}, {MulRoutedWeight}, {ActOP}>;
    }}
    else
    {{
        TORCH_CHECK(
            false,
            "Unsupported block_m value for moe heuristic dispatch: ",
            block_m);
    }}
}}

"""


A8W8_blockscale_gemm2_heuristic_dispatch = """
MoeKernel moe_stage2_heuristic_dispatch(int block_m)
{{

    if (block_m == 64)
    {{
        return ck_moe_stage2_gemm<{A0DataType}, {B0DataType}, {AccDataType}, {EDataType}, {CDEElementOp}, V3, 256, 64, 128, 128/sizeof({A0DataType}), 1, 4, {Nswizzle}, {PerTensorQuant}, {MulRoutedWeight}, {ActOP}>;
    }}
    else
    {{
        TORCH_CHECK(
            false,
            "Unsupported block_m value for moe heuristic dispatch: ",
            block_m);
    }}
}}

"""


heuristic_dispatch_dict = {
    "a8w8": [
        A16W16_A8W8_gemm1_heuristic_dispatch,
        A16W16_A8W8_gemm2_heuristic_dispatch,
    ],
    "a8w8blkscale": [
        A8W8_blockscale_gemm1_heuristic_dispatch,
        A8W8_blockscale_gemm2_heuristic_dispatch,
    ],
    "a16w16": [
        A16W16_A8W8_gemm1_heuristic_dispatch,
        A16W16_A8W8_gemm2_heuristic_dispatch,
    ],
    "a8w4": [
        A8W4_gemm1_heuristic_dispatch,
        A8W4_gemm2_heuristic_dispatch,
    ],
    "a4w4": [
        A4W4_gemm1_heuristic_dispatch,
        A4W4_gemm2_heuristic_dispatch,
    ],
}


class ck_moe_2stage_gemm_codegen:
    def __init__(
        self,
        working_path,
        a_dtype,
        b_dtype,
        c_dtype,
        quant_type,
        activation,
        mul_routed_weight_stage,
    ):
        self.working_path = working_path
        self.a_dtype = a_dtype.upper()
        self.b_dtype = b_dtype.upper()
        self.c_dtype = c_dtype.upper()
        self.quant_type = quant_type
        self.activation = activation
        self.mul_routed_weight_stage = mul_routed_weight_stage
        self.nswizzle = False

    def generate_instance_and_lookUpTable(self):
        _, gemm1_kernel_list = get_gemm1_kernels_list(
            self.a_dtype,
            self.b_dtype,
            self.nswizzle,
            self.quant_type,
            self.activation,
            self.mul_routed_weight_stage == 1,
        )
        tag, gemm2_kernel_list = get_gemm2_kernels_list(
            self.a_dtype,
            self.b_dtype,
            self.nswizzle,
            self.quant_type,
            self.mul_routed_weight_stage == 2,
        )
        kernel_list = list(gemm1_kernel_list.values()) + list(
            gemm2_kernel_list.values()
        )
        f_lookUpTable = os.path.join(self.working_path, "gemm_moe_ck2stages_lookup.h")
        if os.path.exists(f_lookUpTable):
            os.remove(f_lookUpTable)
        with open(f_lookUpTable, "w") as f_lookup:
            f_lookup.write(LOOKUP_head)
            for kernel in kernel_list:
                ## generate instance
                os.makedirs(os.path.join(self.working_path, "instances"), exist_ok=True)
                f_instance = os.path.join(
                    self.working_path, "instances", f"{kernel.name}.cu"
                )
                if os.path.exists(f_instance):
                    os.remove(f_instance)
                if "per_128x128" in self.quant_type:
                    quanttype = "_blockscale"
                elif "FP4" in self.a_dtype:
                    quanttype = "_mxfp4"
                else: 
                    quanttype = ""
                with open(f_instance, "w") as f_ins:
                    stage_instance = STG_INSTANCE_IMPL.format(
                        quanttype=quanttype,
                        A0DataType=self.a_dtype,
                        B0DataType=self.b_dtype,
                        AccDataType="F32" if self.a_dtype != "I8" else "I32",
                        EDataType=self.c_dtype,
                        CDEElementOp=kernel.CDEElementOp,
                        Nswizzle=str(self.nswizzle).lower(),
                        PerTensorQuant=str(self.quant_type != "per_token").lower(),
                        ActOP=int(self.activation == "silu"),
                        Stage=kernel.stage,
                        BlockSize=kernel.BLOCK_SIZE,
                        MPerBlock=kernel.MPerBlock,
                        NPerBlock=kernel.NPerBlock,
                        KPerBlock=kernel.KPerBlock,
                        MWaves=kernel.MWaves,
                        NWaves=kernel.NWaves,
                        PipelineVer=kernel.GemmPipelineVersion,
                        MulRoutedWeight=str(
                            self.mul_routed_weight_stage == kernel.stage
                        ).lower(),
                    )
                    f_ins.write(stage_instance)

                ## generate lookUpTable
                lookup_ele = LOOKUP_template.format(
                    kernel_tag=kernel.name,
                    A0DataType=self.a_dtype,
                    B0DataType=self.b_dtype,
                    AccDataType="F32" if self.a_dtype != "I8" else "I32",
                    EDataType=self.c_dtype,
                    CDEElementOp=kernel.CDEElementOp,
                    Nswizzle=str(self.nswizzle).lower(),
                    PerTensorQuant=str(self.quant_type != "per_token").lower(),
                    ActOP=int(self.activation == "silu"),
                    Stage=kernel.stage,
                    BlockSize=kernel.BLOCK_SIZE,
                    MPerBlock=kernel.MPerBlock,
                    NPerBlock=kernel.NPerBlock,
                    KPerBlock=kernel.KPerBlock,
                    MWaves=kernel.MWaves,
                    NWaves=kernel.NWaves,
                    PipelineVer=kernel.GemmPipelineVersion,
                    MulRoutedWeight=str(
                        self.mul_routed_weight_stage == kernel.stage
                    ).lower(),
                )
                f_lookup.write(lookup_ele)
            f_lookup.write(LOOKUP_end)
        f_heuristic_dispatch = os.path.join(
            self.working_path, "gemm_moe_ck2stages_heuristic_dispatch.hpp"
        )
        if os.path.exists(f_heuristic_dispatch):
            os.remove(f_heuristic_dispatch)
        gemm1_heuristic_dispatch, gemm2_heuristic_dispatch = heuristic_dispatch_dict[
            tag
        ]
        with open(f_heuristic_dispatch, "w") as f_h:
            gemm1_heuristic_dispatch_str = gemm1_heuristic_dispatch.format(
                A0DataType=self.a_dtype,
                B0DataType=self.b_dtype,
                AccDataType="F32" if self.a_dtype != "I8" else "I32",
                EDataType=self.c_dtype,
                CDEElementOp=kernel_list[0].CDEElementOp,
                Nswizzle=str(self.nswizzle).lower(),
                PerTensorQuant=str(self.quant_type != "per_token").lower(),
                ActOP=str(int(self.activation == "silu")),
                MulRoutedWeight=str(self.mul_routed_weight_stage == 1).lower(),
            )
            f_h.write(gemm1_heuristic_dispatch_str)

            gemm2_heuristic_dispatch_str = gemm2_heuristic_dispatch.format(
                A0DataType=self.a_dtype,
                B0DataType=self.b_dtype,
                AccDataType="F32" if self.a_dtype != "I8" else "I32",
                EDataType=self.c_dtype,
                CDEElementOp=kernel_list[-1].CDEElementOp,
                Nswizzle=str(self.nswizzle).lower(),
                PerTensorQuant=str(self.quant_type != "per_token").lower(),
                ActOP=int(self.activation == "silu"),
                MulRoutedWeight=str(self.mul_routed_weight_stage == 2).lower(),
            )
            f_h.write(gemm2_heuristic_dispatch_str)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate ck 2stage gemm instance.")

    # Add arguments
    parser.add_argument(
        "-a",
        "--a_dtype",
        default="f8",
        required=False,
        type=str,
        choices=["f8", "i8", "f16", "b16", "fp4x2"],
        help="select input dtype",
    )

    parser.add_argument(
        "-b",
        "--b_dtype",
        default="f8",
        required=False,
        type=str,
        choices=["f8", "i8", "f16", "b16", "i4", "fp4x2"],
        help="select weight dtype",
    )

    parser.add_argument(
        "-c",
        "--c_dtype",
        default="b16",
        required=False,
        type=str,
        choices=["f16", "b16"],
        help="select out dtype",
    )

    parser.add_argument(
        "-q",
        "--quant_type",
        default="per_tensor",
        required=False,
        type=str,
        choices=["per_tensor", "per_token", "per_128x128", "per_1x32", "no"],
        help="select quant_type",
    )

    parser.add_argument(
        "-act",
        "--activation",
        default="silu",
        required=False,
        type=str,
        choices=["silu", "gelu"],
        help="select activation",
    )

    parser.add_argument(
        "-m",
        "--mul_routed_weight_stage",
        default=2,
        required=False,
        type=int,
        choices=[1, 2],
        help="select quant_type",
    )

    parser.add_argument(
        "-w",
        "--working_path",
        default="./",
        required=False,
        help="the path where all the blobs are going to be generated",
    )

    args = parser.parse_args()

    codegen = ck_moe_2stage_gemm_codegen(
        args.working_path,
        args.a_dtype,
        args.b_dtype,
        args.c_dtype,
        args.quant_type,
        args.activation,
        args.mul_routed_weight_stage,
    )
    codegen.generate_instance_and_lookUpTable()
