# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
from dataclasses import dataclass
from aiter.jit.utils.chip_info import get_gfx


@dataclass
class kernelInstance:
    stage: int
    BLOCK_SIZE: int
    MPerBlock: int
    NPerBlock: int
    KPerBlock: int
    WAVE_TILE_M: int
    WAVE_TILE_N: int
    WAVE_TILE_K: int
    WAVE_MAP_M: int
    WAVE_MAP_N: int
    Block_Per_CU: int = 1
    MulRoutedWeight: bool = False
    ActOP: str = "silu"
    QuantType: str = "per_tensor"

    @property
    def name(self) -> str:
        return ("_").join(element for element in 
            [
                f"moe_cktile2stages_gemm{self.stage}",
                ("x").join(
                    map(
                        lambda x: str(x),
                        [
                            self.BLOCK_SIZE,
                            self.MPerBlock,
                            self.NPerBlock,
                            self.KPerBlock,
                        ],
                    )
                ),
                ("x").join(map(lambda x: str(x), [self.WAVE_MAP_M, self.WAVE_MAP_N])),
                ("x").join(map(lambda x: str(x), [self.WAVE_TILE_M, self.WAVE_TILE_N, self.WAVE_TILE_K])),
                str(self.Block_Per_CU) + "perCU",
                self.QuantType,
                "MulRoutedWeight" if self.MulRoutedWeight else "",
                "" if (self.stage == 2) else self.ActOP,
            ] if element != ""
        )

# fmt: off
# gemm1 out:bf16/fp16 AB:fp8/i8
a8w8_gemm1_kernels_list_gfx950= {
    #  kernel:           stage| BLOCK_SIZE|MPerBLOCK|  NPerBLOCK| KPerBLOCK| WAVE_TILE_M| WAVE_TILE_N| WAVE_TILE_K| WAVE_MAP_M| WAVE_MAP_N|
    # 0: kernelInstance(       1,        256,       32,         64,       256,           16,         16,         128,          1,        4,),
    1: kernelInstance(       1,        256,       32,        128,       128,           16,         16,         128,          1,        4,),
    2: kernelInstance(       1,        256,       64,        128,       128,           16,         16,         128,          1,        4,),
    4: kernelInstance(       1,        256,       64,        128,       256,           16,         16,         128,          1,        4,),
    4: kernelInstance(       1,        256,      128,        128,       128,           16,         16,         128,          1,        4,),
    5: kernelInstance(       1,        256,      128,        128,       128,           16,         16,         128,          1,        4,),
    6: kernelInstance(       1,        256,      256,        128,       128,           16,         16,         128,          1,        4,),
}

# gemm2 out:bf16/fp16 AB:fp8/i8
a8w8_gemm2_kernels_list_gfx950= {
    #  kernel:           stage| BLOCK_SIZE|MPerBLOCK|  NPerBLOCK| KPerBLOCK| WAVE_TILE_M| WAVE_TILE_N| WAVE_TILE_K| WAVE_MAP_M| WAVE_MAP_N|
    0: kernelInstance(       2,        256,       32,        128,       256,           16,         16,         128,          1,        4,),
    1: kernelInstance(       2,        256,       64,        128,       256,           16,         16,         128,          1,        4,),
    2: kernelInstance(       2,        256,      128,        128,       128,           16,         16,         128,          1,        4,),
    3: kernelInstance(       2,        256,      256,        128,       128,           16,         16,         128,          1,        4,),
    4: kernelInstance(       2,        256,      256,        128,       128,           16,         16,         128,          1,        4,),
}


#a8w8
a8w8_gemm1_kernels_list= {
    #  kernel:           stage| BLOCK_SIZE|MPerBLOCK|  NPerBLOCK| KPerBLOCK| WAVE_TILE_M| WAVE_TILE_N| WAVE_TILE_K| WAVE_MAP_M| WAVE_MAP_N|
    # 0: kernelInstance(       1,        256,       32,         64,       256,           16,         16,          64,          1,        4,),
    # 1: kernelInstance(       1,        256,       32,         64,       128,           16,         16,          64,          1,        4,),
    # 2: kernelInstance(       1,        256,       64,         64,       256,           16,         16,          64,          2,        2,),
    # 3: kernelInstance(       1,        256,       64,         64,       128,           16,         16,          64,          1,        4,),
    3: kernelInstance(       1,        256,       64,         128,       128,           16,         16,          64,          1,        4),
    # 4: kernelInstance(       1,        256,      128,         64,       128,           16,         16,          64,          1,        4,),
    # 5: kernelInstance(       1,        256,      128,        128,       128,           16,         16,          64,          1,        4,),
    # 6: kernelInstance(       1,        256,      256,        128,       128,           16,         16,          64,          1,        4,),
}
a8w8_gemm2_kernels_list= {
    #  kernel:           stage| BLOCK_SIZE|MPerBLOCK|  NPerBLOCK| KPerBLOCK| WAVE_TILE_M| WAVE_TILE_N| WAVE_TILE_K| WAVE_MAP_M| WAVE_MAP_N|
    # 0: kernelInstance(       2,        256,       32,         64,       256,           16,         16,          64,          1,        4,),
    # 1: kernelInstance(       2,        256,       64,         64,       256,           16,         16,          64,          1,        4,),
    # 2: kernelInstance(       2,        256,      128,         64,       128,           16,         16,          64,          1,        4,),
    # 3: kernelInstance(       2,        256,      256,         64,       128,           16,         16,          64,          1,        4,),
    # 4: kernelInstance(       2,        256,       64,        128,       256,           16,         16,         128,          1,        4,),
    # 5: kernelInstance(       2,        256,      128,        128,       128,           16,         16,          64,          1,        4,),
    # 6: kernelInstance(       2,        256,      256,        128,       128,           16,         16,          64,          1,        4,),
    # 7: kernelInstance(       2,        256,       32,         64,       128,           16,         16,          64,          1,        4,),
    8: kernelInstance(       2,        256,       64,        128,       128,           16,         16,          64,          1,        4,),
}


# gemm1 out:bf16/fp16 AB:bf16/fp4
a16w4_gemm1_kernels_list_gfx950= {
    #   kernel:           stage| BLOCK_SIZE|MPerBLOCK|  NPerBLOCK| KPerBLOCK| WAVE_TILE_M| WAVE_TILE_N| WAVE_TILE_K| WAVE_MAP_M| WAVE_MAP_N|| BlockPerCU|
    #default kernel
    1:  kernelInstance(       1,        256,       16,        128,       256,           16,         16,          32,          1,           4,          2,),
    13: kernelInstance(       1,        256,       32,        256,       256,           16,         16,          32,          1,           4,          2,),
    22: kernelInstance(       1,        256,       64,        256,       256,           16,         16,          32,          1,           4,          1,),
    32: kernelInstance(       1,        256,      128,        256,       256,           16,         16,          32,          1,           4,          1,),
}
# gemm2 out:bf16/fp16 AB:bf16/fp4
a16w4_gemm2_kernels_list_gfx950= {
    #   kernel:           stage| BLOCK_SIZE|MPerBLOCK|  NPerBLOCK| KPerBLOCK| WAVE_TILE_M| WAVE_TILE_N| WAVE_TILE_K| WAVE_MAP_M| WAVE_MAP_N| BlockPerCU|
    1:  kernelInstance(       2,        256,       16,        128,       256,           16,         16,          32,          1,        4,            2,),
    13: kernelInstance(       2,        256,       32,        256,       256,           16,         16,          32,          1,        4,            2,),
    22: kernelInstance(       2,        256,       64,        256,       256,           16,         16,          32,          1,        4,            1,),
    32: kernelInstance(       2,        256,      128,        256,       256,           16,         16,          32,          1,        4,            1,),
}


tune_a16w4_gemm1_kernels_list_gfx950= {
    #   kernel:           stage| BLOCK_SIZE|MPerBLOCK|  NPerBLOCK| KPerBLOCK| WAVE_TILE_M| WAVE_TILE_N| WAVE_TILE_K| WAVE_MAP_M| WAVE_MAP_N|| BlockPerCU|
    #default kernel
    0:  kernelInstance(       1,        256,       16,        128,       256,           16,         16,          32,          1,           4,          1,),
    1:  kernelInstance(       1,        256,       16,        128,       256,           16,         16,          32,          1,           4,          2,),
    # 2:  kernelInstance(       1,        256,       16,        128,       256,           16,         16,          32,          1,           4,          4,),
    # 3:  kernelInstance(       1,        256,       16,        256,       256,           16,         16,          32,          1,           4,          1,),
    # 4:  kernelInstance(       1,        256,       16,        256,       256,           16,         16,          32,          1,           4,          2,),
    # 5:  kernelInstance(       1,        256,       16,        512,       256,           16,         16,          32,          1,           4,          1,),
    # 6:  kernelInstance(       1,        256,       16,        512,       256,           16,         16,          32,          1,           4,          2,),
    # 10: kernelInstance(       1,        256,       32,        128,       256,           16,         16,          32,          1,           4,          1,),
    # 11: kernelInstance(       1,        256,       32,        128,       256,           16,         16,          32,          1,           4,          2,),
    # 12: kernelInstance(       1,        256,       32,        256,       256,           16,         16,          32,          1,           4,          1,),
    13: kernelInstance(       1,        256,       32,        256,       256,           16,         16,          32,          1,           4,          2,),
    # 14: kernelInstance(       1,        256,       32,        512,       256,           16,         16,          32,          1,           4,          1,),
    # 15: kernelInstance(       1,        256,       32,        512,       256,           16,         16,          32,          1,           4,          2,),
    # 20: kernelInstance(       1,        256,       64,        128,       256,           16,         16,          32,          1,           4,          1,),
    # 21: kernelInstance(       1,        256,       64,        128,       256,           16,         16,          32,          1,           4,          2,),
    22: kernelInstance(       1,        256,       64,        256,       256,           16,         16,          32,          1,           4,          1,),
    # 23: kernelInstance(       1,        256,       64,        256,       256,           16,         16,          32,          1,           4,          2,),
    # 24: kernelInstance(       1,        256,       64,        512,       256,           16,         16,          32,          1,           4,          1,),
    # 25: kernelInstance(       1,        256,       64,        512,       256,           16,         16,          32,          1,           4,          2,),
    # 30: kernelInstance(       1,        256,      128,        128,       256,           16,         16,          32,          1,           4,          1,),
    # 31: kernelInstance(       1,        256,      128,        128,       256,           16,         16,          32,          1,           4,          2,),
    32: kernelInstance(       1,        256,      128,        256,       256,           16,         16,          32,          1,           4,          1,),
    # 33: kernelInstance(       1,        256,      128,        256,       256,           16,         16,          32,          1,           4,          2,), #LDS exceed
    # 40: kernelInstance(       1,        256,      256,        128,       256,           16,         16,          32,          1,           4,          1,), #LDS exceed
    # 41: kernelInstance(       1,        256,      256,        256,       256,           16,         16,          32,          1,           4,          1,), #LDS exceed
}

tune_a16w4_gemm2_kernels_list_gfx950= {
    #   kernel:           stage| BLOCK_SIZE|MPerBLOCK|  NPerBLOCK| KPerBLOCK| WAVE_TILE_M| WAVE_TILE_N| WAVE_TILE_K| WAVE_MAP_M| WAVE_MAP_N|| BlockPerCU|
    #default kernel
    0:  kernelInstance(       2,        256,       16,        128,       256,           16,         16,          32,          1,           4,          1,),
    1:  kernelInstance(       2,        256,       16,        128,       256,           16,         16,          32,          1,           4,          2,),
    # 2:  kernelInstance(       2,        256,       16,        128,       256,           16,         16,          32,          1,           4,          4,),
    # 3:  kernelInstance(       2,        256,       16,        256,       256,           16,         16,          32,          1,           4,          1,),
    # 4:  kernelInstance(       2,        256,       16,        256,       256,           16,         16,          32,          1,           4,          2,),
    # 5:  kernelInstance(       2,        256,       16,        512,       256,           16,         16,          32,          1,           4,          1,),
    # 6:  kernelInstance(       2,        256,       16,        512,       256,           16,         16,          32,          1,           4,          2,),
    # 10: kernelInstance(       2,        256,       32,        128,       256,           16,         16,          32,          1,           4,          1,),
    # 11: kernelInstance(       2,        256,       32,        128,       256,           16,         16,          32,          1,           4,          2,),
    # 12: kernelInstance(       2,        256,       32,        256,       256,           16,         16,          32,          1,           4,          1,),
    13: kernelInstance(       2,        256,       32,        256,       256,           16,         16,          32,          1,           4,          2,),
    # 14: kernelInstance(       2,        256,       32,        512,       256,           16,         16,          32,          1,           4,          1,),
    # 15: kernelInstance(       2,        256,       32,        512,       256,           16,         16,          32,          1,           4,          2,),
    # 20: kernelInstance(       2,        256,       64,        128,       256,           16,         16,          32,          1,           4,          1,),
    # 21: kernelInstance(       2,        256,       64,        128,       256,           16,         16,          32,          1,           4,          2,),
    22: kernelInstance(       2,        256,       64,        256,       256,           16,         16,          32,          1,           4,          1,),
    # 23: kernelInstance(       2,        256,       64,        256,       256,           16,         16,          32,          1,           4,          2,),
    # 24: kernelInstance(       2,        256,       64,        512,       256,           16,         16,          32,          1,           4,          1,),
    # 25: kernelInstance(       2,        256,       64,        512,       256,           16,         16,          32,          1,           4,          2,),
    # 30: kernelInstance(       2,        256,      128,        128,       256,           16,         16,          32,          1,           4,          1,),
    # 31: kernelInstance(       2,        256,      128,        128,       256,           16,         16,          32,          1,           4,          2,),
    32: kernelInstance(       2,        256,      128,        256,       256,           16,         16,          32,          1,           4,          1,),
    # 33: kernelInstance(       2,        256,      128,        256,       256,           16,         16,          32,          1,           4,          2,), #LDS exceed
    # 40: kernelInstance(       2,        256,      256,        128,       256,           16,         16,          32,          1,           4,          1,), #LDS exceed
    # 41: kernelInstance(       2,        256,      256,        256,       256,           16,         16,          32,          1,           4,          1,), #LDS exceed
}


# fmt: on
gemm1_kernels_dict = {
    "a8w8_gfx950": a8w8_gemm1_kernels_list_gfx950,
    "a8w8": a8w8_gemm1_kernels_list,
    "a16w4_gfx950": a16w4_gemm1_kernels_list_gfx950,
    "tune_a16w4_gfx950": tune_a16w4_gemm1_kernels_list_gfx950,
}

gemm2_kernels_dict = {
    "a8w8_gfx950": a8w8_gemm2_kernels_list_gfx950,
    "a8w8": a8w8_gemm2_kernels_list,
    "a16w4_gfx950": a16w4_gemm2_kernels_list_gfx950,
    "tune_a16w4_gfx950": tune_a16w4_gemm2_kernels_list_gfx950,
}



a8w8_gfx950_heuristic_dispatch= ("""
    // Apply shape heuristics to find a suitable kernel implementation.
    if (block_m == 32)
    {{
        return {(1, 1)}<ADataType, BDataType, AccDataType, CDataType>;
    }}
    else if (block_m == 64)
    {{
        return {(1, 2)}<ADataType, BDataType, AccDataType, CDataType>;
    }}
    else if (block_m == 128)
    {{
        return {(1, 4)}<ADataType, BDataType, AccDataType, CDataType>;
    }}
    //else if (block_m == 256)
    //{{
    //    return {(1, 6)}<ADataType, BDataType, AccDataType, CDataType>;
    //}}
""", """
    // Apply shape heuristics to find a suitable kernel implementation.
    if (block_m == 32)
    {{
        return {(2, 0)}<ADataType, BDataType, AccDataType, CDataType>;
    }}
    else if (block_m == 64)
    {{
        return {(2, 1)}<ADataType, BDataType, AccDataType, CDataType>;
    }}
    else if (block_m == 128)
    {{
        return {(2, 2)}<ADataType, BDataType, AccDataType, CDataType>;
    }}
    //else if (block_m == 256)
    //{{
    //    return {(2, 3)}<ADataType, BDataType, AccDataType, CDataType>;
    //}}

""")

a16w4_gfx950_heuristic_dispatch= ("""
    // Apply shape heuristics to find a suitable kernel implementation.
    if (block_m == 16)
    {{
        return {(1, 1)}<ADataType, BDataType, AccDataType, CDataType>;
    }}
    else if (block_m == 32)
    {{
        return {(1, 13)}<ADataType, BDataType, AccDataType, CDataType>;
    }}
    else if (block_m == 64)
    {{
        return {(1, 22)}<ADataType, BDataType, AccDataType, CDataType>;
    }}
    else if (block_m == 128)
    {{
        return {(1, 32)}<ADataType, BDataType, AccDataType, CDataType>;
    }}
    """, """
    // Apply shape heuristics to find a suitable kernel implementation.
    if (block_m == 16)
    {{
        return {(2, 1)}<ADataType, BDataType, AccDataType, CDataType>;
    }}
    else if (block_m == 32)
    {{
        return {(2, 13)}<ADataType, BDataType, AccDataType, CDataType>;
    }}
    else if (block_m == 64)
    {{
        return {(2, 22)}<ADataType, BDataType, AccDataType, CDataType>;
    }}
    else if (block_m == 128)
    {{
        return {(2, 32)}<ADataType, BDataType, AccDataType, CDataType>;
    }}
""")

heuristic_dispatch_dict = {
    "a8w8_gfx950": a8w8_gfx950_heuristic_dispatch,
    # "a8w8": a8w8_gemm2_kernels_list,
    "a16w4_gfx950": a16w4_gfx950_heuristic_dispatch
}




bit8_list = ["f8", "i8", "fp8"]
bit16_list = ["b16", "f16", "bf16", "fp16"]
bit4_list = ["i4", "fp4x2", "fp4"]
QuantType_list = ["no", "per_tensor", "per_token", "per_1x128", "per_1x32"]


def get_gemm1_kernels_list(
    Adtype: str,
    Bdtype: str,
    QuantType: str = "none",
    ActOP: str = "silu",
    MulRoutedWeight: bool = False,
    istune: bool = False,
) -> list:
    arch = get_gfx()
    if Adtype.lower() in bit8_list and Bdtype.lower() in bit8_list and Adtype == Bdtype:
        if arch == "gfx950":
            tag = "a8w8_gfx950"
        else:
            tag = "a8w8"
    elif Adtype in bit16_list and Bdtype in bit4_list:
        tag = "a16w4_gfx950"
    else:
        raise ValueError(f"Unsupported data type combination: {Adtype}, {Bdtype}")
    if istune:
        tag = "tune_" + tag
    kernels_list = gemm1_kernels_dict[tag]
    for id, kernel in kernels_list.items():
        kernel.MulRoutedWeight = MulRoutedWeight
        kernel.ActOP = ActOP
        kernel.QuantType = QuantType
        # if tag == "a8w4":
            # kernel.CDEElementOp = "MulABScaleWint4"
        # elif tag == "a8w8blkscale":
            # kernel.CDEElementOp = "MulABScaleExpertWeightA8W8blkscale"
        # elif tag == "a8w8" or tag == "a4w4":
            # kernel.CDEElementOp = "MulABScale"
        # elif tag == "a16w16":
            # if MulRoutedWeight:
                # kernel.CDEElementOp = "TypeCastExpertWeight"
            # else:
                # kernel.CDEElementOp = "TypeCast"
    return tag, kernels_list


def get_gemm2_kernels_list(    
    Adtype: str,
    Bdtype: str,
    QuantType: str = "",
    ActOP: str = "",
    MulRoutedWeight: bool = True,
    istune: bool = False,
) -> list:
    arch = get_gfx()
    if Adtype in bit8_list and Bdtype in bit8_list and Adtype == Bdtype:
        if arch == "gfx950":
            tag = "a8w8_gfx950"
        else:
            tag = "a8w8"
    elif Adtype in bit16_list and Bdtype in bit4_list:
        tag = "a16w4_gfx950"
    else:
        raise ValueError(f"Unsupported data type combination: {Adtype}, {Bdtype}")
    if istune:
        tag = "tune_" + tag
    kernels_list = gemm2_kernels_dict[tag]
    for id, kernel in kernels_list.items():
        kernel.MulRoutedWeight = MulRoutedWeight
        kernel.ActOP = ""
        kernel.QuantType = QuantType
        # if tag == "a8w4":
        #     kernel.CDEElementOp = "MulABScaleExpertWeightWin4"
        # elif tag == "a8w8blkscale":
        #     kernel.CDEElementOp = "MulABScaleExpertWeightA8W8blkscale"
        # elif tag == "a8w8" or tag == "a4w4":
        #     kernel.CDEElementOp = "MulABScaleExpertWeight"
        # elif tag == "a16w16":
        #     if MulRoutedWeight:
        #         kernel.CDEElementOp = "TypeCastExpertWeight"
        #     else:
        #         kernel.CDEElementOp = "TypeCast"
    return tag, kernels_list

def get_heuristic_dispatch_template(tag):
    if tag.startswith("tune"):
        return ("", "")
    if (tag not in heuristic_dispatch_dict.keys()):
        raise ValueError(f"Unsupported type for heuristic_dispatch: {tag}")
    return heuristic_dispatch_dict[tag]