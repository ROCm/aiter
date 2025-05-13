# SPDX-License-Identifier: MIT
# Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
import os
import sys
import argparse
import pandas as pd
import re

STG_INSTANCE_IMPL = """// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
#include "gemm_moe_2stages_common.cuh"

using A0DataType = {A0DataType};
using B0DataType = {B0DataType};
using AccDataType = {AccDataType};
using EDataType = {EDataType};
using CDEElementOp = {CDEElementOp};
const bool Nswizzle = {Nswizzle};
const bool PerTensorQuant = {PerTensorQuant};
CK_MOE_STAGE{Stage}_GEMM_DEFINE({BlockSize}, {MPerBlock}, {NPerBlock}, {KPerBlock}, {MWaves}, {NWaves}, {MNPerXDL})
"""


LOOKUP_head = """#pragma once
// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#ifdef USE_ROCM

#define GENERATE_LOOKUP_TABLE()                                                                                      \\
   {                                                                                                                             \\"""

LOOKUP_template = """
       {{"{kernel_tag}",                                                                                                       \\
        moe_stage{stage}_gemm<{A0DataType}, {B0DataType}, {AccDataType}, {EDataType}, {CDEElementOp}, {BLOCKSIZE}, {MPerBlock}, {NPerBlock}, {KPerBlock}, {MWaves}, {NWaves}, {MNPerXDL}, {Nswizzle}, {PerTensorQuant}>}},                       \\"""

LOOKUP_end = """
   }

#endif // USE_ROCM
"""


class ck_moe_2stage_gemm_codegen:
    def __init__(self, working_path, tune_file):
        self.working_path = working_path
        self.tune_file = tune_file
        
    def generate_instance_and_lookUpTable(self):
        df_tuned = pd.read_csv(self.tune_file, dtype=str)
        f_lookUpTable = os.path.join(self.working_path, "gemm_moe_2stages_lookup.h")
        if os.path.exists(f_lookUpTable):
            os.remove(f_lookUpTable)
        with open(f_lookUpTable, "w") as f_lookup:
            f_lookup.write(LOOKUP_head)
            for idx, row in df_tuned.iterrows():
                ## generate instance
                kernel_tag = row['tag']
                os.makedirs(os.path.join(self.working_path, "instances"), exist_ok=True)
                f_instance = os.path.join(self.working_path, "instances" ,f"{kernel_tag}.cu")
                if os.path.exists(f_instance):
                    os.remove(f_instance)
                with open(f_instance, "w") as f_ins:
                    # ck_moe_stage1   ## stage
                    # _B16    # EDataType
                    # _F8    # A0DataType
                    # _I4    # B0DataType
                    # _PerToken
                    # _256x128x128x128 # BLOCKSIZE, MPerfBlock, NPerBlock, KPerBlock
                    # _1x4     # MWaves, NWaves
                    # _32     # MNPerXDL
                    # _MulABScaleWint4    # CDEElementOp
                    # _Nswizzle0
                    # _interwave
                    # _v1

                    args =  r"stage([12])_([a-zA-Z0-9]+)_([a-zA-Z0-9]+)_([a-zA-Z0-9]+)_([a-zA-Z]+)_(\d+)x(\d+)x(\d+)x(\d+)_(\d+)x(\d+)_(\d+)_([a-zA-Z0-9]+)_Nswizzle([01])"
                    match = re.search(args, kernel_tag)
                    if not match:
                        print(f"Kernel tag {kernel_tag} does not match the expected pattern.")
                        continue
                    # Extract the parameters from the kernel name
                    print()
                    stage_instance = STG_INSTANCE_IMPL.format(
                        A0DataType=match.group(3),
                        B0DataType=match.group(4),
                        AccDataType="F32",
                        EDataType=match.group(2),
                        CDEElementOp=match.group(13),
                        Nswizzle=str(match.group(14) == "1").lower(),
                        PerTensorQuant= str(match.group(5)=="PerTensor").lower(),
                        Stage=match.group(1),
                        BlockSize=match.group(6),
                        MPerBlock=match.group(7),
                        NPerBlock=match.group(8),
                        KPerBlock=match.group(9),
                        MWaves=match.group(10),
                        NWaves=match.group(11),
                        MNPerXDL=match.group(12))
                    f_ins.write(stage_instance)

                ## generate lookUpTable
                lookup_ele = LOOKUP_template.format(
                    kernel_tag=kernel_tag,
                    stage=match.group(1),
                    A0DataType=match.group(3),
                    B0DataType=match.group(4),
                    AccDataType="F32",
                    EDataType=match.group(2),
                    CDEElementOp=match.group(13),
                    BLOCKSIZE=match.group(6),
                    MPerBlock=match.group(7),
                    NPerBlock=match.group(8),
                    KPerBlock=match.group(9),
                    MWaves=match.group(10),
                    NWaves=match.group(11),
                    MNPerXDL=match.group(12),
                    Nswizzle=str(match.group(14) == "1").lower(),
                    PerTensorQuant= str(match.group(5)=="PerTensor").lower())
                f_lookup.write(lookup_ele)
            f_lookup.write(LOOKUP_end)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate ck 2stage gemm instance.")
    
    # Add arguments
    parser.add_argument(
        "-f", 
        "--tune_file", 
        default="./tuned_config.csv",
        required=False,
        type=str,
        help="select knl based on tuned config file")
    
    parser.add_argument(
        "-w",
        "--working_path",
        default="./",
        required=False,
        help="the path where all the blobs are going to be generated"
    )

    parser.add_argument(
        "-t",
        "--tune",
        action='store_true',
        required=False,
        help="generated tune(all) instanses"
    )

    args = parser.parse_args()

    # generate all instances for tune.
    if args.tune:
        pass
    # generate tuned instances.
    else:
        codegen = ck_moe_2stage_gemm_codegen(args.working_path, args.tune_file)
        codegen.generate_instance_and_lookUpTable()