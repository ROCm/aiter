# SPDX-License-Identifier: MIT
# Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
from dataclasses import dataclass

@dataclass
class kernelInstanceGEMM1:
    BLOCK_SIZE: int
    MPerBLOCK: int
    NPerBLOCK: int
    KPerBLOCK: int
    WAVE_MAP_M: int
    WAVE_MAP_N: int
    BlockGemmPipelineVersion: int
    MulRoutedWeight: bool
    ActOP: bool
    Nswizzle: bool
    CDEElementOp: str
    
    @property
    def name(self) -> str:
        return ("_").join([
            "moe_ck2stages_gemm1",
            ("x").join(map(lambda x: str(x), [
                self.BLOCK_SIZE, self.MPerBLOCK, self.NPerBLOCK, self.KPerBLOCK])),
            ("x").join(map(lambda x: str(x), [
                self.WAVE_MAP_M, self.WAVE_MAP_N])),
            ("x").join(map(lambda x: str(x), self.MulRoutedWeight)),
            ("x").join(map(lambda x: str(x), self.ActOP)),
            ("x").join(map(lambda x: str(x), self.Nswizzle)),
            ("x").join(map(lambda x: str(x), [
                self.CDEElementOp])),
            f"v{self.BlockGemmPipelineVersion}"
        ])

@dataclass
class kernelInstanceGEMM2:
    BLOCK_SIZE: int
    MPerBLOCK: int
    NPerBLOCK: int
    KPerBLOCK: int
    WAVE_MAP_M: int
    WAVE_MAP_N: int
    BlockGemmPipelineVersion: int
    MulRoutedWeight: bool
    Nswizzle: bool
    CDEElementOp: str
    
    @property
    def name(self) -> str:
        return ("_").join([
            "moe_ck2stages_gemm2",
            ("x").join(map(lambda x: str(x), [
                self.BLOCK_SIZE, self.MPerBLOCK, self.NPerBLOCK, self.KPerBLOCK])),
            ("x").join(map(lambda x: str(x), [
                self.WAVE_MAP_M, self.WAVE_MAP_N])),
            ("x").join(map(lambda x: str(x), self.MulRoutedWeight)),
            ("x").join(map(lambda x: str(x), self.Nswizzle)),
            ("x").join(map(lambda x: str(x), [
                self.CDEElementOp])),
            f"v{self.BlockGemmPipelineVersion}"
        ])
    
kernels_list_gemm1= {
#   id: kernel:             BLOCK_SIZE| MPerBLOCK| NPerBLOCK| KPerBLOCK| WAVE_MAP_M| WAVE_MAP_N|BlockGemmPipelineVersion | MulRoutedWeight| ActOP| Nswizzle  |  CDEElementOp|   
# gemm1 out&AB:bf16/fp16 
     0: kernelInstanceGEMM1(       256,        32,        64,       128,         1,         4,                        1,           False,     1,    False,      "TypeCast",),
     1: kernelInstanceGEMM1(       256,        32,        64,        64,         1,         4,                        1,           False,     1,    False,      "TypeCast",),
     2: kernelInstanceGEMM1(       256,        64,        64,       128,         1,         4,                        1,           False,     1,    False,      "TypeCast",),
     3: kernelInstanceGEMM1(       256,        64,        64,        64,         1,         4,                        1,           False,     1,    False,      "TypeCast",),
     4: kernelInstanceGEMM1(       256,       128,        64,        64,         1,         4,                        1,           False,     1,    False,      "TypeCast",),
     5: kernelInstanceGEMM1(       256,        32,        64,       128,         1,         4,                        1,           False,     0,    False,      "TypeCast",),
     6: kernelInstanceGEMM1(       256,        32,        64,        64,         1,         4,                        1,           False,     0,    False,      "TypeCast",),
     7: kernelInstanceGEMM1(       256,        64,        64,       128,         1,         4,                        1,           False,     0,    False,      "TypeCast",),
     8: kernelInstanceGEMM1(       256,        64,        64,        64,         1,         4,                        1,           False,     0,    False,      "TypeCast",),
     9: kernelInstanceGEMM1(       256,       128,        64,        64,         1,         4,                        1,           False,     0,    False,      "TypeCast",),
    10: kernelInstanceGEMM1(       256,       128,       128,        64,         1,         4,                        3,           False,     1,    False,      "TypeCast",),
    11: kernelInstanceGEMM1(       256,       128,       128,       128,         1,         4,                        3,           False,     0,    False,      "TypeCast",),
# gemm1 out:bf16/fp16 AB:fp8/i8 
    12: kernelInstanceGEMM1(       256,       32,         64,       256,         1,         4,                        1,            True,     1,    False,      "MulABScale",),
    13: kernelInstanceGEMM1(       256,       32,         64,       128,         1,         4,                        1,            True,     1,    False,      "MulABScale",),
    14: kernelInstanceGEMM1(       256,       64,         64,       256,         1,         4,                        1,            True,     1,    False,      "MulABScale",),
    15: kernelInstanceGEMM1(       256,       64,         64,       128,         1,         4,                        1,            True,     1,    False,      "MulABScale",),
    16: kernelInstanceGEMM1(       256,      128,         64,       128,         1,         4,                        1,            True,     1,    False,      "MulABScale",),
    17: kernelInstanceGEMM1(       256,       32,         64,       256,         1,         4,                        1,            True,     0,    False,      "MulABScale",),
    18: kernelInstanceGEMM1(       256,       32,         64,       128,         1,         4,                        1,            True,     0,    False,      "MulABScale",),
    19: kernelInstanceGEMM1(       256,       64,         64,       256,         1,         4,                        1,            True,     0,    False,      "MulABScale",),
    20: kernelInstanceGEMM1(       256,       64,         64,       128,         1,         4,                        1,            True,     0,    False,      "MulABScale",),
    21: kernelInstanceGEMM1(       256,      128,         64,       128,         1,         4,                        1,            True,     0,    False,      "MulABScale",),
    22: kernelInstanceGEMM1(       256,       32,         64,       256,         1,         4,                        1,           False,     1,    False,      "MulABScale",),
    23: kernelInstanceGEMM1(       256,       32,         64,       128,         1,         4,                        1,           False,     1,    False,      "MulABScale",),
    24: kernelInstanceGEMM1(       256,       64,         64,       256,         1,         4,                        1,           False,     1,    False,      "MulABScale",),
    25: kernelInstanceGEMM1(       256,       64,         64,       128,         1,         4,                        1,           False,     1,    False,      "MulABScale",),
    26: kernelInstanceGEMM1(       256,      128,         64,       128,         1,         4,                        1,           False,     1,    False,      "MulABScale",),
    27: kernelInstanceGEMM1(       256,       32,         64,       256,         1,         4,                        1,           False,     0,    False,      "MulABScale",),
    28: kernelInstanceGEMM1(       256,       32,         64,       128,         1,         4,                        1,           False,     0,    False,      "MulABScale",),
    29: kernelInstanceGEMM1(       256,       64,         64,       256,         1,         4,                        1,           False,     0,    False,      "MulABScale",),
    30: kernelInstanceGEMM1(       256,       64,         64,       128,         1,         4,                        1,           False,     0,    False,      "MulABScale",),
    31: kernelInstanceGEMM1(       256,      128,         64,       128,         1,         4,                        1,           False,     0,    False,      "MulABScale",),
    32: kernelInstanceGEMM1(       256,      128,        128,       128,         1,         4,                        3,            True,     1,    False,      "MulABScale",),
    33: kernelInstanceGEMM1(       256,      128,        128,       128,         1,         4,                        3,            True,     0,    False,      "MulABScale",),
    34: kernelInstanceGEMM1(       256,      128,        128,       128,         1,         4,                        3,           False,     1,    False,      "MulABScale",),
    35: kernelInstanceGEMM1(       256,      128,        128,       128,         1,         4,                        3,           False,     0,    False,      "MulABScale",),
# gemm1 out:bf16/fp16 A:fp8 B:win4 
    36: kernelInstanceGEMM1(       256,       32,         64,       128,         1,         4,                        1,            True,     0,    False,      "MulABScaleWint4",),
    37: kernelInstanceGEMM1(       256,       64,         64,       128,         1,         4,                        1,            True,     0,    False,      "MulABScaleWint4",),
    38: kernelInstanceGEMM1(       256,      128,         64,       128,         1,         4,                        1,            True,     0,    False,      "MulABScaleWint4",),
    39: kernelInstanceGEMM1(       256,       32,         64,       128,         1,         4,                        1,           False,     0,    False,      "MulABScaleWint4",),
    40: kernelInstanceGEMM1(       256,       64,         64,       128,         1,         4,                        1,           False,     0,    False,      "MulABScaleWint4",),
    41: kernelInstanceGEMM1(       256,      128,         64,       128,         1,         4,                        1,           False,     0,    False,      "MulABScaleWint4",),
    42: kernelInstanceGEMM1(       256,       32,         64,       128,         1,         4,                        1,            True,     1,    False,      "MulABScaleWint4",),
    43: kernelInstanceGEMM1(       256,       64,         64,       128,         1,         4,                        1,            True,     1,    False,      "MulABScaleWint4",),
    44: kernelInstanceGEMM1(       256,      128,         64,       128,         1,         4,                        1,            True,     1,    False,      "MulABScaleWint4",),
    45: kernelInstanceGEMM1(       256,       32,         64,       128,         1,         4,                        1,           False,     1,    False,      "MulABScaleWint4",),
    46: kernelInstanceGEMM1(       256,       64,        128,       128,         1,         4,                        3,           False,     1,    False,      "MulABScaleWint4",),
    47: kernelInstanceGEMM1(       256,      128,        128,       128,         1,         4,                        3,           False,     1,    False,      "MulABScaleWint4",),
# gemm1 out&AB:bf16/fp16 mulweight
    48: kernelInstanceGEMM1(       256,        32,        64,       128,         1,         4,                        1,            True,     1,    False,      "TypeCastExpertWeight",),
    49: kernelInstanceGEMM1(       256,        32,        64,        64,         1,         4,                        1,            True,     1,    False,      "TypeCastExpertWeight",),
    50: kernelInstanceGEMM1(       256,        64,        64,       128,         1,         4,                        1,            True,     1,    False,      "TypeCastExpertWeight",),
    51: kernelInstanceGEMM1(       256,        64,        64,        64,         1,         4,                        1,            True,     1,    False,      "TypeCastExpertWeight",),
    52: kernelInstanceGEMM1(       256,       128,        64,        64,         1,         4,                        1,            True,     1,    False,      "TypeCastExpertWeight",),
    53: kernelInstanceGEMM1(       256,        32,        64,       128,         1,         4,                        1,            True,     0,    False,      "TypeCastExpertWeight",),
    54: kernelInstanceGEMM1(       256,        32,        64,        64,         1,         4,                        1,            True,     0,    False,      "TypeCastExpertWeight",),
    55: kernelInstanceGEMM1(       256,        64,        64,       128,         1,         4,                        1,            True,     0,    False,      "TypeCastExpertWeight",),
    56: kernelInstanceGEMM1(       256,        64,        64,        64,         1,         4,                        1,            True,     0,    False,      "TypeCastExpertWeight",),
    57: kernelInstanceGEMM1(       256,       128,        64,        64,         1,         4,                        1,            True,     0,    False,      "TypeCastExpertWeight",),
    58: kernelInstanceGEMM1(       256,       128,       128,        64,         1,         4,                        3,            True,     1,    False,      "TypeCastExpertWeight",),
    59: kernelInstanceGEMM1(       256,       128,       128,       128,         1,         4,                        3,            True,     0,    False,      "TypeCastExpertWeight",),
}    

kernels_list_gemm2= {
#   id: kernel:             BLOCK_SIZE| MPerBLOCK| NPerBLOCK| KPerBLOCK| WAVE_MAP_M| WAVE_MAP_N|BlockGemmPipelineVersion | MulRoutedWeight| Nswizzle  |  CDEElementOp|   
# gemm2 out&AB:bf16/fp16 
     0: kernelInstanceGEMM2(       256,        32,       128,       128,         1,          4,                        1,           False,    False,      "TypeCast",),
     1: kernelInstanceGEMM2(       256,        64,       128,       128,         1,          4,                        1,           False,    False,      "TypeCast",),
     2: kernelInstanceGEMM2(       256,       128,       128,        64,         1,          4,                        1,           False,    False,      "TypeCast",),
     3: kernelInstanceGEMM2(       256,       128,       256,        64,         1,          4,                        3,           False,    False,      "TypeCast",),
# gemm2 out:bf16/fp16 AB:fp8/i8    
     4: kernelInstanceGEMM2(       256,       32,        128,       256,         1,          4,                        1,            True,    False,      "MulABScaleExpertWeight",),
     5: kernelInstanceGEMM2(       256,       64,        128,       256,         1,          4,                        1,            True,    False,      "MulABScaleExpertWeight",),
     6: kernelInstanceGEMM2(       256,      128,        128,       128,         1,          4,                        1,            True,    False,      "MulABScaleExpertWeight",),
     7: kernelInstanceGEMM2(       256,       32,        128,       256,         1,          4,                        1,           False,    False,      "MulABScaleExpertWeight",),
     8: kernelInstanceGEMM2(       256,       64,        128,       256,         1,          4,                        1,           False,    False,      "MulABScaleExpertWeight",),
     9: kernelInstanceGEMM2(       256,      128,        128,       128,         1,          4,                        1,           False,    False,      "MulABScaleExpertWeight",),
    10: kernelInstanceGEMM2(       256,      128,        256,       128,         1,          4,                        3,            True,    False,      "MulABScaleExpertWeight",),
    11: kernelInstanceGEMM2(       256,      128,        256,       128,         1,          4,                        3,           False,    False,      "MulABScaleExpertWeight",),
# gemm2 out:bf16/fp16 A:fp8 B:in4    
    12: kernelInstanceGEMM2(       256,       32,        128,       128,         1,          4,                        1,            True,    False,      "MulABScaleExpertWeightWin4",),
    13: kernelInstanceGEMM2(       256,       64,        128,       128,         1,          4,                        1,            True,    False,      "MulABScaleExpertWeightWin4",),
    14: kernelInstanceGEMM2(       256,      128,        128,       128,         1,          4,                        1,            True,    False,      "MulABScaleExpertWeightWin4",),
    15: kernelInstanceGEMM2(       256,       32,        128,       128,         1,          4,                        1,           False,    False,      "MulABScaleExpertWeightWin4",),
    16: kernelInstanceGEMM2(       256,       64,        128,       128,         1,          4,                        1,           False,    False,      "MulABScaleExpertWeightWin4",),
    17: kernelInstanceGEMM2(       256,      128,        128,       128,         1,          4,                        1,           False,    False,      "MulABScaleExpertWeightWin4",),
    18: kernelInstanceGEMM2(       256,       64,        256,       128,         1,          4,                        3,            True,    False,      "MulABScaleExpertWeightWin4",),
    19: kernelInstanceGEMM2(       256,      128,        256,       128,         1,          4,                        3,           False,    False,      "MulABScaleExpertWeightWin4",),
# gemm2 out&AB:bf16/fp16 mulweight  
    20: kernelInstanceGEMM2(       256,        32,       128,       128,         1,          4,                        1,            True,    False,      "TypeCastExpertWeight",),
    21: kernelInstanceGEMM2(       256,        64,       128,       128,         1,          4,                        1,            True,    False,      "TypeCastExpertWeight",),
    22: kernelInstanceGEMM2(       256,       128,       128,        64,         1,          4,                        1,            True,    False,      "TypeCastExpertWeight",),
    23: kernelInstanceGEMM2(       256,       128,       256,        64,         1,          4,                        3,            True,    False,      "TypeCastExpertWeight",),
}       