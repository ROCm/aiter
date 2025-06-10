# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
from dataclasses import dataclass

@dataclass
class kernelInstanceGEMM1:
    BLOCK_SIZE: int
    MPerBlock: int
    NPerBlock: int
    KPerBlock: int
    MWaves: int
    NWaves: int
    GemmPipelineVersion: int
    Nswizzle: bool = False
    MulRoutedWeight: bool = False
    ActOP: bool = False
    CDEElementOp: str = "TypeCast"
    QuantType: str = "per_tensor"
    stage:int = 1

    @property
    def name(self) -> str:
        return ("_").join([
            f"moe_ck2stages_gemm{self.stage}",
            ("x").join(map(lambda x: str(x), [
                self.BLOCK_SIZE, self.MPerBlock, self.NPerBlock, self.KPerBlock])),
            ("x").join(map(lambda x: str(x), [
                self.MWaves, self.NWaves])),
            self.CDEElementOp,
            f"v{self.GemmPipelineVersion}",
            "Nswizzle" + str(int(self.Nswizzle)),
            self.QuantType,
            "MulRoutedWeight" + str(int( self.MulRoutedWeight)),
            "silu" if  self.ActOP else "gelu",
        ])

@dataclass
class kernelInstanceGEMM2:
    BLOCK_SIZE: int
    MPerBlock: int
    NPerBlock: int
    KPerBlock: int
    MWaves: int
    NWaves: int
    GemmPipelineVersion: int
    Nswizzle: bool = False
    MulRoutedWeight: bool = True
    CDEElementOp: str = "TypeCast"
    QuantType: str = "per_tensor"
    stage:int = 2

    @property
    def name(self) -> str:
        return ("_").join([
            f"moe_ck2stages_gemm{self.stage}",
            ("x").join(map(lambda x: str(x), [
                self.BLOCK_SIZE, self.MPerBlock, self.NPerBlock, self.KPerBlock])),
            ("x").join(map(lambda x: str(x), [
                self.MWaves, self.NWaves])),
            self.CDEElementOp,
            f"v{self.GemmPipelineVersion}",
            "Nswizzle" + str(int(self.Nswizzle)),
            self.QuantType,
            "MulRoutedWeight" + str(int( self.MulRoutedWeight)),
        ])

# gemm1 out&AB:bf16/fp16
a16w16_gemm1_kernels_list= {
#   id: kernel:             BLOCK_SIZE| MPerBLOCK| NPerBLOCK| KPerBLOCK| MWaves| NWaves|GemmPipelineVersion
     0: kernelInstanceGEMM1(       256,        32,        64,       128,     1,       4,        1,),
     1: kernelInstanceGEMM1(       256,        32,        64,        64,     1,       4,        1,),
     2: kernelInstanceGEMM1(       256,        64,        64,       128,     1,       4,        1,),
     3: kernelInstanceGEMM1(       256,        64,        64,        64,     1,       4,        1,),
     4: kernelInstanceGEMM1(       256,       128,        64,        64,     2,       2,        1,),
    # 5: kernelInstanceGEMM1(       256,       128,       128,        64,     1,       4,        3,),
}
# gemm1 out:bf16/fp16 AB:fp8/i8
a8w8_gemm1_kernels_list= {
     0: kernelInstanceGEMM1(       256,       32,         64,       256,     1,       4,        1,),
     1: kernelInstanceGEMM1(       256,       32,         64,       128,     1,       4,        1,),
     2: kernelInstanceGEMM1(       256,       64,         64,       256,     1,       4,        1,),
     3: kernelInstanceGEMM1(       256,       64,         64,       128,     1,       4,        1,),
     4: kernelInstanceGEMM1(       256,      128,         64,       128,     2,       2,        1,),
    # 5: kernelInstanceGEMM1(       256,      128,        128,       128,     1,       4,        3,),
}
# gemm1 blockscale out:bf16/fp16 AB:fp8/i8
a8w8_gemm1_blockscale_kernels_list= {
     #0: kernelInstanceGEMM1(       256,       32,        128,       128,     1,       4,        1,),
     0: kernelInstanceGEMM1(       256,       64,        128,       128,     1,       4,        3,),
     #2: kernelInstanceGEMM1(       256,      128,        128,       128,     1,       4,        3,),
}

# gemm1 out:bf16/fp16 A:fp8 B:win4
a8w4_gemm1_kernels_list= {
     0: kernelInstanceGEMM1(       256,       32,         64,       128,     1,       4,        1,),
     1: kernelInstanceGEMM1(       256,       64,         64,       128,     1,       4,        1,),
     2: kernelInstanceGEMM1(       256,      128,         64,       128,     1,       4,        1,),
    #3: kernelInstanceGEMM1(       256,      128,        256,       128,     1,       4,        3,),
}

gemm1_kernels_dict = {
    "a16w16": a16w16_gemm1_kernels_list,
    "a8w8": a8w8_gemm1_kernels_list,
    "a8w8blkscale": a8w8_gemm1_blockscale_kernels_list,
    "a8w4": a8w4_gemm1_kernels_list,
}


a16w16_gemm2_kernels_list= {
#   id: kernel:             BLOCK_SIZE| MPerBLOCK| NPerBLOCK| KPerBLOCK| MWaves| NWaves|GemmPipelineVersion
# gemm2 out&AB:bf16/fp16
     0: kernelInstanceGEMM2(       256,        32,       128,       128,     1,       4,         1,),
     1: kernelInstanceGEMM2(       256,        64,       128,       128,     1,       4,         1,),
     2: kernelInstanceGEMM2(       256,       128,       128,        64,     2,       2,         1,),
    # 3: kernelInstanceGEMM2(       256,       128,       256,        64,     1,       4,         3,),
}
# gemm2 out:bf16/fp16 AB:fp8/i8
a8w8_gemm2_kernels_list= {
     0: kernelInstanceGEMM2(       256,       32,        128,       256,     1,       4,         1,),
     1: kernelInstanceGEMM2(       256,       64,        128,       256,     1,       4,         1,),
     2: kernelInstanceGEMM2(       256,      128,        128,       128,     2,       2,         1,),
    # 3: kernelInstanceGEMM2(       256,      128,        256,       128,     1,       4,         3,),
}

# gemm2 MXDLPerWave out:bf16/fp16 AB:fp8/i8
a8w8_gemm2_blockscale_kernels_list= {
     #0: kernelInstanceGEMM2(       256,       32,        128,       128,     1,       4,        1,),
     1: kernelInstanceGEMM2(       256,       64,        128,       128,     1,       4,        3,),
     #2: kernelInstanceGEMM2(       256,      128,        128,       128,     2,       2,        3,),
}

# gemm2 out:bf16/fp16 A:fp8 B:in4
a8w4_gemm2_kernels_list= {
     0: kernelInstanceGEMM2(       256,       32,        128,       128,     1,       4,         1,),
     1: kernelInstanceGEMM2(       256,       64,        128,       128,     1,       4,         1,),
     2: kernelInstanceGEMM2(       256,      128,        128,       128,     1,       4,         1,),
    #3: kernelInstanceGEMM2(       256,       64,        256,       128,     1,       4,         3,),
}

gemm2_kernels_dict = {
    "a16w16": a16w16_gemm2_kernels_list,
    "a8w8": a8w8_gemm2_kernels_list,
    "a8w8blkscale": a8w8_gemm2_blockscale_kernels_list,
    "a8w4": a8w4_gemm2_kernels_list,
}


bit8_list = ["F8", "I8", "f8", "i8"]
bit16_list = ["B16", "F16", "b16", "f16"]
bit4_list = ["I4", "i4"]
QuantType_list = ["per_128x128"]

def get_gemm1_kernels_list(Adtype: str, Bdtype: str, Nswizzle: bool, QuantType: str, ActOP: bool, MulRoutedWeight: bool) -> list:

    if Adtype in bit16_list and Bdtype in bit16_list and Adtype == Adtype:
        tag = "a16w16"
    elif Adtype in bit8_list and Bdtype in bit8_list and Adtype == Adtype and QuantType in QuantType_list:
        tag = "a8w8blkscale"
    elif Adtype in bit8_list and Bdtype in bit8_list and Adtype == Adtype:
        tag = "a8w8"
    elif Adtype in bit8_list and Bdtype in bit4_list and Adtype == "F8":
        tag = "a8w4"
    else:
        raise ValueError(f"Unsupported data type combination: {Adtype}, {Bdtype}")
    kernels_list = gemm1_kernels_dict[tag]
    for id, kernel in kernels_list.items():
        kernel.MulRoutedWeight = MulRoutedWeight
        kernel.ActOP = ActOP
        kernel.Nswizzle = Nswizzle
        kernel.QuantType = QuantType
        if tag == "a8w4":
            kernel.CDEElementOp = "MulABScaleWint4"
        elif tag == "a8w8blkscale":
            kernel.CDEElementOp = "MulABScaleExpertWeight"
        elif tag == "a8w8":
            kernel.CDEElementOp = "MulABScale"
        elif tag == "a16w16":
            if MulRoutedWeight:
                kernel.CDEElementOp = "TypeCastExpertWeight"
            else:
                kernel.CDEElementOp = "TypeCast"
    return tag, kernels_list


def get_gemm2_kernels_list(Adtype: str, Bdtype: str, Nswizzle: bool, QuantType: str, MulRoutedWeight: bool) -> list:
    if Adtype in bit16_list and Bdtype in bit16_list and Adtype == Adtype:
        tag = "a16w16"
    elif Adtype in bit8_list and Bdtype in bit8_list and Adtype == Adtype and QuantType in QuantType_list:
        tag = "a8w8blkscale"
    elif Adtype in bit8_list and Bdtype in bit8_list and Adtype == Adtype:
        tag = "a8w8"
    elif Adtype in bit8_list and Bdtype in bit4_list and Adtype == "F8":
        tag = "a8w4"
    else:
        raise ValueError(f"Unsupported data type combination: {Adtype}, {Bdtype}")
    kernels_list = gemm2_kernels_dict[tag]
    for id, kernel in kernels_list.items():
        kernel.MulRoutedWeight = MulRoutedWeight
        kernel.Nswizzle = Nswizzle
        kernel.QuantType = QuantType
        if tag == "a8w4":
            kernel.CDEElementOp = "MulABScaleExpertWeightWin4"
        elif tag == "a8w8blkscale":
            kernel.CDEElementOp = "MulABScaleExpertWeight"
        elif tag == "a8w8":
            kernel.CDEElementOp = "MulABScaleExpertWeight"
        elif tag == "a16w16":
            if MulRoutedWeight:
                kernel.CDEElementOp = "TypeCastExpertWeight"
            else:
                kernel.CDEElementOp = "TypeCast"
    return tag, kernels_list
