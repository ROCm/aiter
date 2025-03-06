# SPDX-License-Identifier: MIT
# Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
from dataclasses import dataclass

@dataclass
class kernelInstance:
    BLOCK_SIZE: int
    ScaleBlockM: int
    ScaleBlockN: int
    ScaleBlockK: int
    MPerBLOCK: int
    NPerBLOCK: int
    KPerBLOCK: int
    AK1: int
    BK1: int
    MPerXDL: int
    NPerXDL: int
    WAVE_MAP_M: int
    WAVE_MAP_N: int
    ABLOCK_TRANSFER: list[int]
    BBLOCK_TRANSFER: list[int]
    CSHUFFLE_MX_PER_WAVE_PERSHUFFLE: int
    CSHUFFLE_NX_PER_WAVE_PERSHUFFLE: int
    CBLOCK_TRANSFER: list[int]
    CBLOCK_SPV: list[int]
    PIPELINE_Sched: str
    PIPELINE_VERSION: int

    @property
    def name(self) -> str:
        return ("_").join([
            "a8w8_blockscale",
            ("x").join(map(lambda x: str(x), [
                self.ScaleBlockM, self.ScaleBlockN, self.ScaleBlockK])),
            ("x").join(map(lambda x: str(x), [
                self.BLOCK_SIZE, self.MPerBLOCK, self.NPerBLOCK, self.KPerBLOCK])),
            ("x").join(map(lambda x: str(x), [
                self.AK1, self.BK1])),
            ("x").join(map(lambda x: str(x), [
                self.MPerXDL, self.NPerXDL])),
            ("x").join(map(lambda x: str(x), self.ABLOCK_TRANSFER)),
            ("x").join(map(lambda x: str(x), self.BBLOCK_TRANSFER)),
            ("x").join(map(lambda x: str(x), self.CBLOCK_TRANSFER)),
            ("x").join(map(lambda x: str(x), self.CBLOCK_SPV)),
            ("x").join(map(lambda x: str(x), [self.CSHUFFLE_MX_PER_WAVE_PERSHUFFLE, self.CSHUFFLE_NX_PER_WAVE_PERSHUFFLE])),
            self.PIPELINE_Sched.lower(),
            f"v{self.PIPELINE_VERSION}"
        ])

kernels_list = {
    # clang-format off
        ##############| Block| Scale| Scale| Scale|  MPer|  NPer|  KPer| AK1| BK1|MPer| NPer| MXdl| NXdl|  ABlockTransfer|  BBlockTransfer|    CShuffle|    CShuffle|     CBlockTransferClusterLengths|  CBlockTransfer|  Block-wiseGemm|     Block-wiseGemm|
        ###############| Size| Block| Block| Block| Block| Block| Block|    |    | XDL|  XDL|  Per|  Per|   ThreadCluster|   ThreadCluster| MXdlPerWave| NXdlPerWave| _MBlock_MXdlPerWave_MWaveMPerXdl| ScalarPerVector|    Pipeline    |           Pipeline|
        ###############|     |     M|     N|     K|      |      |      |    |    |    |     | Wave| Wave| Lengths_K0_M_K1| Lengths_K0_N_K1|  PerShuffle|  PerShuffle| _NBlock_NXdlPerWave_NWaveNPerXdl|   _NWaveNPerXdl|    Scheduler   |           Verision|
        ###############|     |      |      |      |      |      |      |    |    |    |     |     |     |                |                |            |            |                                 |                |                |                   |

        # Compute friendly
    0: kernelInstance(  256,   1,   128,   128,   128,   128,   128,   16,  16,   32,   32,    2,    2,     [8, 32, 1],        [8, 32, 1],            1,           1,                 [1, 32, 1, 8],            [8],        "Intrawave",         3 ),
    1: kernelInstance(  256,   1,   128,   128,   128,    64,   128,   16,  16,   32,   32,    2,    1,     [8, 32, 1],        [8, 32, 1],            1,           1,                 [1, 32, 1, 8],            [8],        "Intrawave",         3 ),
    2: kernelInstance(  256,   1,   128,   128,    64,   128,   128,   16,  16,   32,   32,    1,    2,     [8, 32, 1],        [8, 32, 1],            1,           1,                 [1, 32, 1, 8],            [8],        "Intrawave",         3 ),
    3: kernelInstance(  256,   1,   128,   128,    64,    64,   128,   16,  16,   32,   32,    1,    1,     [8, 32, 1],        [8, 32, 1],            1,           1,                 [1, 32, 1, 8],            [8],        "Intrawave",         3 ),
    
        # Memory friendly                    
    4:  kernelInstance( 256,   1,   128,   128,    16,   256,   128,  8,  16,     16,   16,    1,    4,     [16, 16, 1],       [8, 32, 1],            1,           2,                 [1, 16, 1, 16],           [8],        "Intrawave",         1, ),
    5:  kernelInstance( 256,   1,   128,   128,    16,   128,   128,  8,  16,     16,   16,    1,    2,     [16, 16, 1],       [8, 32, 1],            1,           2,                 [1, 16, 1, 16],           [8],        "Intrawave",         1, ),
    6:  kernelInstance( 256,   1,   128,   128,    16,    64,   128,  8,  16,     16,   16,    1,    1,     [16, 16, 1],       [8, 32, 1],            1,           1,                 [1, 16, 1, 16],           [4],        "Intrawave",         1, ),
    7:  kernelInstance( 256,   1,   128,   128,    16,   128,   256, 16,  16,     16,   16,    1,    2,     [16, 16, 1],       [16, 16, 1],           1,           2,                 [1, 16, 1, 16],           [8],        "Intrawave",         1, ),
    8:  kernelInstance( 256,   1,   128,   128,    16,    64,   256, 16,  16,     16,   16,    1,    1,     [16, 16, 1],       [16, 16, 1],           1,           1,                 [1, 16, 1, 16],           [4],        "Intrawave",         1, ),
    
    9:  kernelInstance( 256,   1,   128,   128,    32,   256,   128, 16,  16,     32,   32,    1,    2,     [8, 32, 1],        [8, 32, 1],            1,           1,                 [1, 32, 1, 8],            [8],        "Intrawave",         1, ),
    10: kernelInstance( 256,   1,   128,   128,    32,   128,   128, 16,  16,     32,   32,    1,    1,     [8, 32, 1],        [8, 32, 1],            1,           1,                 [1, 32, 1, 8],            [8],        "Intrawave",         1, ),
    11: kernelInstance( 256,   1,   128,   128,    32,    64,   128, 16,  16,     16,   16,    2,    1,     [8, 32, 1],        [8, 32, 1],            2,           1,                 [1, 32, 1, 8],            [8],        "Intrawave",         1, ),
    
    12: kernelInstance( 256,   1,   128,   128,    32,   128,   256, 16,  16,     32,   32,    1,    1,     [16, 16, 1],       [16, 16, 1],           1,           1,                 [1, 32, 1, 8],            [8],        "Intrawave",         1, ),
    13: kernelInstance( 256,   1,   128,   128,    32,    64,   256, 16,  16,     16,   16,    2,    1,     [16, 16, 1],       [16, 16, 1],           2,           1,                 [1, 32, 1, 8],            [8],        "Intrawave",         1, ),
    
    14: kernelInstance( 256,   1,   128,   128,    64,   256,   128, 16,  16,     32,   32,    2,    2,     [8, 32, 1],        [8, 32, 1],            1,           1,                 [1, 32, 1, 8],            [8],        "Intrawave",         1, ),
    15: kernelInstance( 256,   1,   128,   128,    64,   128,   128, 16,  16,     32,   32,    2,    1,     [8, 32, 1],        [8, 32, 1],            1,           1,                 [1, 32, 1, 8],            [8],        "Intrawave",         1, ),
    16: kernelInstance( 256,   1,   128,   128,    64,    64,   128, 16,  16,     32,   32,    1,    1,     [8, 32, 1],        [8, 32, 1],            1,           1,                 [1, 32, 1, 8],            [8],        "Intrawave",         1, ),
    
    17: kernelInstance( 256,   1,   128,   128,    64,   128,   256, 16,  16,     32,   32,    2,    1,     [16, 16, 1],       [16, 16, 1],           1,           1,                 [1, 32, 1, 8],            [8],        "Intrawave",         1, ),
    18: kernelInstance( 256,   1,   128,   128,    64,    64,   256, 16,  16,     32,   32,    1,    1,     [16, 16, 1],       [16, 16, 1],           1,           1,                 [1, 32, 1, 8],            [8],        "Intrawave",         1, )
    # clang-format on
}




default_kernels_dict = {
    # clang-format off
        ##############| Block| Scale| Scale| Scale|  MPer|  NPer|  KPer| AK1| BK1|MPer| NPer| MXdl| NXdl|  ABlockTransfer|  BBlockTransfer|    CShuffle|    CShuffle|     CBlockTransferClusterLengths|  CBlockTransfer|  Block-wiseGemm|     Block-wiseGemm|
        ###############| Size| Block| Block| Block| Block| Block| Block|    |    | XDL|  XDL|  Per|  Per|   ThreadCluster|   ThreadCluster| MXdlPerWave| NXdlPerWave| _MBlock_MXdlPerWave_MWaveMPerXdl| ScalarPerVector|    Pipeline    |           Pipeline|
        ###############|     |     M|     N|     K|      |      |      |    |    |    |     | Wave| Wave| Lengths_K0_M_K1| Lengths_K0_N_K1|  PerShuffle|  PerShuffle| _NBlock_NXdlPerWave_NWaveNPerXdl|   _NWaveNPerXdl|    Scheduler   |           Verision|
        ###############|     |      |      |      |      |      |      |    |    |    |     |     |     |                |                |            |            |                                 |                |                |                   |

        # Compute friendly
    (-1): kernelInstance( 256,   1,   128,   128,   16,   128,   256,   16,  16,   16,   16,    1,    2,     [16, 16, 1],   [16, 16, 1],          1,           2,                     [1, 16, 1, 16],             [8],      "Intrawave",           1 ),
    
}
